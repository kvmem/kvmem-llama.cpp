// Int8 tensor-core small-T kernel for the C8 cohort-decode regime, the counterpart of
// q4_ksplit_mma.cuh. Weight storage stays Q4G64 (4 bits/code, the same artifact bytes);
// activations are quantised to s8 with one scale per (token, 64-k group), matching the weight's own
// group size, and both operands feed mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32. The same
// instruction and rescale are proven at the 128-token prefill tile in
// ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h; this file narrows them to small T.
//
// Result on the 27B gate_up, cold, median of 15, each width paired against the bf16 small-T kernel
// inside one sitting -- the card drifts several percent between sittings, so only the pairing is
// meaningful. Two sittings, i8 against bf16:
//
//   T        8      12      16      20      24      28      32
//   run 1  +10.7%  -5.8%   -3.4%   -4.8%   -4.8%   -5.0%   -4.5%
//   run 2   +7.5%  +4.3%   -0.7%   -5.9%   -6.1%   -4.0%   -6.4%
//
// So it wins from sixteen columns up, by 4-6%, and loses at eight: a 64-k group of eight columns has
// too little MMA work to pay for quantising the activations, which costs a fixed ~10us pre-pass
// (measured by skipping it) plus a scale plane the A16 route does not read. Twelve is inside the
// noise. A route table would want it from 16 upward and not below.
//
// Relative L2 against the fp64 oracle is 0.0080-0.0371 across T=2..32, against the prefill route's
// 0.023-0.031 at its own widths, and the cost is the activation quantisation error the A16 route
// does not have.
//
// What ncu says this kernel is, which is not what the roofline predicted. docs/performance.md put
// gate_up at T=32 at ~68% of the card's bf16 MMA peak and called C8 tensor-rate bound, which is why
// int8 was the lever. It is not what binds here: at 64 registers and no spills this kernel runs at
// 74% L1/TEX throughput against 42% DRAM and 47% SM, so operand movement through L1 and shared
// memory is the limit, not the MMA rate. The int8 MMA's arithmetic advantage is therefore mostly
// unspendable at this shape -- the win above is what is left after the traffic was cut, not the
// 4.7x the instruction rate alone would suggest. A wider tile that spends the spare tensor rate is
// where the rest would have to come from.
//
// Measured and rejected on the way, so nobody re-runs them (all at T=32, same harness):
//
//   staging each tile group's own copy of the activation slab   311.6us ncu / 280.6 wall
//     -- kTileGroups-fold redundant reads and shared writes of identical bytes, 77.9% L1/TEX
//   16-byte weight staging copies (lane -> row, 16B chunk)      273.4us
//     -- halves the transaction count and loses anyway; the (gid, tig) mapping the compute loop
//        reads by is also the one that stages fastest
//   two row tiles per warp (each B fragment feeds both)         275.5us
//     -- same verdict the bf16 kernel records for 32 columns
//   KWarps 2 with a 2- or 3-deep cp.async ring                  249.9 / 260.1us
//   KWarps 4 with a 2-deep ring                                 263.2us
//   a runtime column count instead of padding to the tile width  222.2us at T=28 against 196.6
//     -- the staging bounds test cannot fold away and sits beside the tile-group test
//
// KWarps 4 with a single stage wins at every width from sixteen columns up, and eight columns wants
// KWarps 8: at 20 outer steps rather than 40 it splits K harder, and the ring it does without was
// never hiding DRAM latency in the first place -- see the L1 reading above. Swept per band:
//
//   TileCols   KWarps 8   KWarps 4   KWarps 2
//   8            137.2      139.3      237.6
//   16           176.1      154.6      214.0
//   24             --       179.2      220.2      (two stages: 198.7 / 233.5)
//   32             --       220.2      255.0      (two stages: 262.1 / 255.0)
//
// Columns are padded to the tile width by the launcher rather than masked at runtime, for the
// reason the rejected list gives.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/small_t_layout.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

// Eight packed Q4 codes -> two s8-packed words of four codes each, in natural physical k order
// (verified against the fp64 oracle via q4a8_linear_swiglu.cu's identical formula). w0 holds codes
// 0..3, w1 holds codes 4..7.
__device__ __forceinline__ void q4_unpack_s8(unsigned word, unsigned& w0, unsigned& w1) {
    word ^= 0x88888888u;
    const unsigned mask = 0x0f0f0f0fu;
    const unsigned even = __vsub4(word & mask, 0x08080808u);
    const unsigned odd  = __vsub4((word >> 4) & mask, 0x08080808u);
    w0                  = __byte_perm(even, odd, 0x5140);
    w1                  = __byte_perm(even, odd, 0x7362);
}

struct Q4SmallTMmaI8StoreEpilogue {};

// Shared storage for a ring of Stages staged K slabs, one 64*KWarps-wide group each. Weight rows
// stay nibble-packed (32 B/row); activations are already s8 (64 B/col); one scale per (row,
// 64-group) and per (col, 64-group). Padded so tig's 8-or-16-byte loads never tie the same bank
// across the four tig values sharing a row.
template <int KWarps, int TileCols, int Stages, int TilesPerWarp = 1>
struct Q4SmallTI8Storage {
    using Layout                     = SmallTLayout<KWarps, TilesPerWarp>;
    static constexpr int kGroupK     = 64; // one quantisation group
    static constexpr int kPackedRow  = kGroupK / 2;      // 32: packed nibble bytes per row
    static constexpr int kSRowPacked = kPackedRow + 16;  // padded
    static constexpr int kSRowAct    = kGroupK + 16;     // padded
    static constexpr int kRows       = Layout::kRowsPerCta;

    struct Stage {
        std::uint8_t weight[KWarps][kRows][kSRowPacked];
        std::int8_t activation[KWarps][TileCols][kSRowAct];
        __half w_scale[KWarps][kRows];
        __half x_scale[KWarps][TileCols];
    };
    union Shared {
        Stage stages[Stages];
        float partial[Layout::kWarps * TilesPerWarp * (TileCols / 8) * 32 * 4];
    };
    static constexpr int kBytes = sizeof(Shared);
};

// Geometry/TileCols/ActiveCols/Epilogue/RowPolicy/KWarps/Stages/TilesPerWarp mirror
// q4_ksplit_mma_kernel's parameter list exactly. x_codes/x_scales are the pre-quantised
// activations (q4_small_t_quantize_activations below); codes/scales are the artifact's native
// Q4G64 planes, unchanged.
template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4SmallTMmaI8StoreEpilogue,
          class RowPolicy = Q4KSplitIdentityRows, bool MaskedColumns = false, int KWarps = 2,
          int Stages = 2, int TilesPerWarp = 1>
__launch_bounds__(256) __global__
    void q4_small_t_mma_i8_kernel(const std::int8_t* __restrict__ x_codes,
                                  const __half* __restrict__ x_scales,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, Epilogue epilogue = {},
                                  RowPolicy row_policy = {}, int columns = ActiveCols) {
    using Layout                = SmallTLayout<KWarps, TilesPerWarp>;
    constexpr int kTpw          = TilesPerWarp;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kKWarps       = Layout::kKWarps;
    constexpr int kRowTiles     = Layout::kRowTiles;
    constexpr int kGroupK       = 64;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    constexpr int kTileGroups   = Layout::kWarps / kKWarps;
    static_assert(kTileCols >= 8 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(kGroups % kKWarps == 0);
    static_assert(RowPolicy::kOutputRowsPerTile <= 16);
    static_assert(Stages >= 1 && Stages <= 4);

    using Storage = Q4SmallTI8Storage<KWarps, TileCols, Stages, TilesPerWarp>;
    using Stage   = typename Storage::Stage;
    __shared__ __align__(16) typename Storage::Shared shared;

    constexpr int kOuterSteps = kGroups / kKWarps;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2; // 0..7: row/col selector within a group of 8
    const int tig          = lane & 3;  // 0..3: 8-or-16-byte k quarter
    const int k_split      = warp % kKWarps;
    const int tile_group   = warp / kKWarps; // 0..kTileGroups-1, one per set of KWarps warps
    const int first_tile   = tile_group * kTpw;
    const int tile0        = static_cast<int>(blockIdx.x) * kRowTiles;
    const int live_columns = MaskedColumns ? columns : ActiveCols;

    // Stage one 64-wide group of packed weight rows for this K-warp via cp.async; decode happens in
    // the compute loop once the copy lands. Each 16-row MMA tile needs both its top eight rows
    // (local_row = gid) and bottom eight (gid + 8).
    // Wider 16-byte staging copies were tried here and lost: mapping lanes to (row, 16-byte chunk)
    // halves the transaction count but measured 273.4us against this mapping's 249.9us at T=32, so
    // the (gid, tig) mapping the compute loop also reads by is kept.
    const auto stage_weight = [&](int group_index, Stage& stage) {
        const int group_k0 = (group_index * kKWarps + k_split) * kGroupK;
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int local_row = gid + half * 8;
                const int tile_row  = (first_tile + t) * 16 + local_row;
                const int weight_row = row_policy.weight_row(
                    (tile0 + first_tile + t) * RowPolicy::kOutputRowsPerTile, local_row);
                const std::uint8_t* src =
                    codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                    tig * 8;
                cp_async<8>(stage.weight[k_split][tile_row] + tig * 8, src);
                if (tig == 0) {
                    stage.w_scale[k_split][tile_row] = *reinterpret_cast<const __half*>(
                        scales +
                        (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                         group_k0 / kGroupK) *
                            2);
                }
            }
        }
    };

    // Stage this K-warp's slice of the pre-quantised s8 activations via cp.async (already s8, no
    // decode needed). The B operand's N = 8 span is exactly the 8 gid values, so unlike the A
    // operand above no top/bottom split is needed.
    //
    // Every row tile reads the same staged slab (ops/common/small_t_layout.cuh), so the eight-column
    // blocks are dealt out across the tile groups rather than staged by each of them: staging all of
    // them from every group costs kTileGroups times the global reads and shared writes for identical
    // bytes, which ncu put at 77.9% L1/TEX throughput against the bf16 kernel's 34.2%.
    const auto stage_activation = [&](int group_index, Stage& stage) {
        const int group_k0 = (group_index * kKWarps + k_split) * kGroupK;
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            if (nt % kTileGroups != tile_group) { continue; }
            const int col = nt * 8 + gid;
            if (col >= live_columns) {
                // Masked column. The compute loop zeroes its B fragment, but the rescale still
                // multiplies by this scale, so zero it rather than leave the previous group's --
                // a stale scale times a zero product is harmless, an uninitialised one need not be.
                if (tig == 0) { stage.x_scale[k_split][col] = __float2half(0.0F); }
                continue;
            }
            const std::int8_t* src =
                x_codes + static_cast<std::int64_t>(col) * kHidden + group_k0 + tig * 16;
            cp_async<16>(stage.activation[k_split][col] + tig * 16, src);
            if (tig == 0) {
                stage.x_scale[k_split][col] =
                    x_scales[static_cast<std::int64_t>(col) * kGroups + group_k0 / kGroupK];
            }
        }
    };

    const auto issue_group = [&](int group_index) {
        if (group_index < kOuterSteps) {
            Stage& stage = shared.stages[group_index % Stages];
            stage_weight(group_index, stage);
            stage_activation(group_index, stage);
        }
        cp_commit(); // empty commits keep cp_wait<Stages - 1> exact through the tail
    };

    float acc[kTpw][kNt][4] = {};

#pragma unroll
    for (int prefetch = 0; prefetch < Stages - 1; ++prefetch) { issue_group(prefetch); }

#pragma unroll 1
    for (int group_index = 0; group_index < kOuterSteps; ++group_index) {
        issue_group(group_index + Stages - 1);
        cp_wait<Stages - 1>();
        __syncthreads();

        const Stage& stage             = shared.stages[group_index % Stages];
        float group_acc[kTpw][kNt][4] = {};
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row = (first_tile + t) * 16 + gid;
            const uint2 lo_packed =
                *reinterpret_cast<const uint2*>(stage.weight[k_split][tile_row] + tig * 8);
            const uint2 hi_packed =
                *reinterpret_cast<const uint2*>(stage.weight[k_split][tile_row + 8] + tig * 8);
            unsigned lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;
            q4_unpack_s8(lo_packed.x, lo0, lo1);
            q4_unpack_s8(lo_packed.y, lo2, lo3);
            q4_unpack_s8(hi_packed.x, hi0, hi1);
            q4_unpack_s8(hi_packed.y, hi2, hi3);
            const unsigned af[2][4] = {{lo0, hi0, lo1, hi1}, {lo2, hi2, lo3, hi3}};
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = nt * 8 + gid;
                uint4 b       = make_uint4(0u, 0u, 0u, 0u);
                if (col < live_columns) {
                    b = *reinterpret_cast<const uint4*>(stage.activation[k_split][col] + tig * 16);
                }
                const unsigned bf[2][2] = {{b.x, b.y}, {b.z, b.w}};
                int s[4]                = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[ks][0], af[ks][1], af[ks][2], af[ks][3],
                           bf[ks][0], bf[ks][1]);
                }
                float(&g)[4] = group_acc[t][nt];
                g[0] += static_cast<float>(s[0]);
                g[1] += static_cast<float>(s[1]);
                g[2] += static_cast<float>(s[2]);
                g[3] += static_cast<float>(s[3]);
            }
        }

#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row    = (first_tile + t) * 16 + gid;
            const float top_scale = __half2float(stage.w_scale[k_split][tile_row]);
            const float bot_scale = __half2float(stage.w_scale[k_split][tile_row + 8]);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                // The MMA's fixed output-fragment layout gives this lane accumulator values for
                // columns {2*tig, 2*tig+1} of this nt's 8-column block -- not the gid-indexed
                // column this lane happened to *load* for the B operand (that indexing is the
                // input-side collaborative-load pattern; the instruction redistributes results
                // across lanes independently of it). Same convention as q4a8_linear_swiglu.cu.
                const int col0  = nt * 8 + 2 * tig;
                const float xa0 = __half2float(stage.x_scale[k_split][col0]);
                const float xa1 = __half2float(stage.x_scale[k_split][col0 + 1]);
                acc[t][nt][0] = fmaf(group_acc[t][nt][0], top_scale * xa0, acc[t][nt][0]);
                acc[t][nt][1] = fmaf(group_acc[t][nt][1], top_scale * xa1, acc[t][nt][1]);
                acc[t][nt][2] = fmaf(group_acc[t][nt][2], bot_scale * xa0, acc[t][nt][2]);
                acc[t][nt][3] = fmaf(group_acc[t][nt][3], bot_scale * xa1, acc[t][nt][3]);
            }
        }
        // The next iteration's issue_group writes the stage this one just read.
        __syncthreads();
    }
    cp_wait<0>();

    // K-warp reduction: odd warps publish, even ones fold, warp 0 sums the rest. Identical to
    // q4_ksplit_mma.cuh's tree; operates on plain floats so it is dtype-agnostic.
    __syncthreads();
    auto* partial   = shared.partial;
    const auto slot = [&](int w, int t, int nt) {
        return partial + (((w * kTpw + t) * kNt + nt) * 32 + lane) * 4;
    };
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const float(&a)[4] = acc[t][nt];
                *reinterpret_cast<float4*>(slot(warp, t, nt)) = make_float4(a[0], a[1], a[2], a[3]);
            }
        }
    }
    __syncthreads();
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float(&a)[4]         = acc[t][nt];
                const float4 partner = *reinterpret_cast<const float4*>(slot(warp + 1, t, nt));
                a[0] += partner.x;
                a[1] += partner.y;
                a[2] += partner.z;
                a[3] += partner.w;
                if (k_split != 0) {
                    *reinterpret_cast<float4*>(slot(warp, t, nt)) =
                        make_float4(a[0], a[1], a[2], a[3]);
                }
            }
        }
    }
    if constexpr (kKWarps > 2) { __syncthreads(); }

    if (k_split == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int row0 = (tile0 + first_tile + t) * RowPolicy::kOutputRowsPerTile;
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float4 sum = make_float4(acc[t][nt][0], acc[t][nt][1], acc[t][nt][2], acc[t][nt][3]);
#pragma unroll
                for (int split = 2; split < kKWarps; split += 2) {
                    const float4 value = *reinterpret_cast<const float4*>(slot(warp + split, t, nt));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int col0 = nt * 8 + 2 * tig;
                epilogue.template store<ActiveCols>(row0 + gid, col0, sum);
            }
        }
    }
}

// One scale per (token, 64-k group), matching the weight's own group size. bf16 activations in
// [rows, tokens]; codes/scales sized rows*tokens and tokens*(rows/64) respectively.
__global__ void q4_small_t_quantize_activations_kernel(const __nv_bfloat16* __restrict__ x,
                                                        std::int32_t rows, std::int32_t tokens,
                                                        std::int8_t* __restrict__ codes,
                                                        __half* __restrict__ scales) {
    const int token  = static_cast<int>(blockIdx.x);
    const int groups = rows / 64;
    if (token >= tokens) { return; }
    for (int g = static_cast<int>(threadIdx.x); g < groups; g += static_cast<int>(blockDim.x)) {
        const __nv_bfloat16* src = x + static_cast<std::int64_t>(token) * rows + g * 64;
        float amax               = 0.0F;
#pragma unroll
        for (int j = 0; j < 64; ++j) { amax = fmaxf(amax, fabsf(__bfloat162float(src[j]))); }
        amax                                                = fmaxf(amax, 1.0e-20F);
        scales[static_cast<std::int64_t>(token) * groups + g] = __float2half(amax / 127.0F);
        const float inv                                       = 127.0F / amax;
        std::int8_t* dst = codes + static_cast<std::int64_t>(token) * rows + g * 64;
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            const float v = __bfloat162float(src[j]) * inv;
            dst[j]        = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v))));
        }
    }
}

inline void q4_small_t_quantize_activations(const __nv_bfloat16* x, std::int32_t rows,
                                            std::int32_t tokens, std::int8_t* codes,
                                            __half* scales, cudaStream_t stream) {
    q4_small_t_quantize_activations_kernel<<<tokens, 128, 0, stream>>>(x, rows, tokens, codes,
                                                                        scales);
}

template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4SmallTMmaI8StoreEpilogue,
          class RowPolicy = Q4KSplitIdentityRows, bool MaskedColumns = false, int KWarps = 2,
          int Stages = 2, int TilesPerWarp = 1>
void q4_small_t_mma_i8_launch(int blocks, cudaStream_t stream, const std::int8_t* x_codes,
                              const __half* x_scales, const std::uint8_t* codes,
                              const std::uint8_t* scales, __nv_bfloat16* out,
                              Epilogue epilogue = {}, RowPolicy row_policy = {},
                              int columns = ActiveCols) {
    constexpr int kBytes = Q4SmallTI8Storage<KWarps, TileCols, Stages, TilesPerWarp>::kBytes;
    constexpr auto kernel =
        q4_small_t_mma_i8_kernel<Geometry, TileCols, ActiveCols, Epilogue, RowPolicy,
                                 MaskedColumns, KWarps, Stages, TilesPerWarp>;
    static_assert(kBytes <= 48 * 1024, "small-T i8 MMA ring must fit static shared memory");
    kernel<<<blocks, SmallTLayout<KWarps>::kThreads, 0, stream>>>(
        x_codes, x_scales, codes, scales, out, epilogue, row_policy, columns);
}

} // namespace ninfer::ops::detail
