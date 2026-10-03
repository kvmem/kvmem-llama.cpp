#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ops/linear/common/route_table.h"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_small_t_mma_i8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                 = 34816;
constexpr int kK                 = 5120;
constexpr int kIntermediate      = kN / 2;
constexpr int kGroupK            = 64;
constexpr int kGroups            = kK / kGroupK;
constexpr int kBytesPerGroup     = 32;
constexpr int kVecBytes          = 16;
constexpr int kGroupsPerWarpTile = 16;
constexpr int kVecsPerWarpTile   = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
constexpr int kWarpsPerBlock     = 4;
constexpr int kBlockThreads      = kWarpsPerBlock * 32;
constexpr int kPairsPerBlock     = kWarpsPerBlock;
constexpr int kXVecs             = kK / 8; // x as uint4 (8 bf16 each)
constexpr int kTiles             = kGroups / kGroupsPerWarpTile;
static_assert(kIntermediate % kPairsPerBlock == 0);
static_assert(kBytesPerGroup == 2 * kVecBytes);
static_assert(kGroups % kGroupsPerWarpTile == 0);
static_assert(kVecsPerWarpTile == 32);

struct Q4SwiGluSmallTGeometry {
    static constexpr int kInputRows    = kK;
    static constexpr int kGroupsPerRow = kK / kGroupK;
};

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerTile = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluSmallTEpilogue {
    __nv_bfloat16* out;
    int columns;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 projected) const {
        if (col0 < columns) {
            out[static_cast<std::int64_t>(col0) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < columns) {
            out[static_cast<std::int64_t>(col0 + 1) * kIntermediate + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Past eight columns the activation slab, not the weights, dominates what a CTA pulls through L2:
// at 32 columns and one sixteen-row tile per CTA that is ~713 MB per call against 89 MB of
// weights. Wider tiles share each staged slab between two or four row tiles
// (ops/common/small_t_layout.cuh). One kernel, cold L2, median of 21, us:
//
//   T    kwarps 8           kwarps 4           kwarps 2
//   8    121.9              128.0 / 120.8 (s2) 186.4
//   16   293.9 / 244.7 (s2) 138.2              194.6
//   24   291.8              189.4 / 186.4 (s2) 209.9 / 193.4 (s2)
//   32   429.1              269.3 / 231.6 (s2) 244.7 / 204.8 (s2)
//
// Two tiles per warp (each B fragment feeds both) only pays at 24 columns: kwarps 4, two stages,
// 163.8 us against 189.4; at 16 and 32 it is 139.3 / 230.4-278.4, no better or worse.
template <int TileCols>
struct Q4SwiGluSmallTTile {
    static constexpr int kKWarps       = TileCols <= 8 ? 8 : (TileCols <= 24 ? 4 : 2);
    static constexpr int kStages       = TileCols <= 16 ? 1 : 2;
    static constexpr int kTilesPerWarp = TileCols == 24 ? 2 : 1;
    // Occupancy ceiling, and a measured dead end. ncu puts this kernel at 31% occupancy at T=32,
    // register-limited to two blocks per SM, with tensor, L1 and DRAM all near 38% -- nothing
    // saturated, which reads as an invitation to buy warps. It is not one: raising the ceiling caps
    // registers, and at 96 registers this kernel needs them. Cold medians, us, and the number of
    // instantiations ptxas reports spilling:
    //
    //   min blocks   T=16    T=24    T=32   spilling
    //   2 (this)    161.8   175.1   205.8      0
    //   3           162.8   321.5   227.3      4
    //   4           161.8   543.7   281.6      4
    //
    // More warps in flight is still what the counters ask for; it has to come from a kernel that
    // needs fewer registers, not from squeezing this one.
    static constexpr int kMinBlocks = kKWarps < 8 ? 2 : (TileCols <= 16 ? 6 : 4);
};

// Masked carries the column count at runtime, which the staging loop's trip count and the inner
// loop's per-fragment bounds test both depend on. A cohort that exactly fills its tile -- eight
// lanes verifying four MTP columns is thirty-two -- does not need either, and an unmasked
// instantiation lets both fold away.
template <int ActiveCols, bool Masked>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Tile            = Q4SwiGluSmallTTile<TileCols>;
    using Layout          = SmallTLayout<Tile::kKWarps, Tile::kTilesPerWarp>;
    constexpr int kBlocks = kIntermediate / (Q4SwiGluSmallTRows::kOutputRowsPerTile *
                                             Layout::kRowTiles);
    const Q4SwiGluSmallTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), x.ne[1]};
    q4_ksplit_mma_launch<Q4SwiGluSmallTGeometry, TileCols, ActiveCols, Q4SwiGluSmallTEpilogue,
                         Q4SwiGluSmallTRows, Masked, Tile::kKWarps, Tile::kStages,
                         Tile::kTilesPerWarp, Tile::kMinBlocks>(
        kBlocks, stream, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), epilogue, Q4SwiGluSmallTRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked, std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<8 * (1 + static_cast<int>(Offsets)), Masked>...};
}

// The eight-column slot of the exact table is deliberately the masked launcher: the dispatch below
// never selects an exact eight-column width (it spills -- see the note there), and instantiating one
// anyway would compile a spilling kernel nothing can call.
template <std::size_t... Offsets>
constexpr auto make_small_t_exact_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<8 * (1 + static_cast<int>(Offsets)), Offsets == 0>...};
}

constexpr auto kSmallTLaunchers = make_small_t_launchers<true>(std::make_index_sequence<4>{});
constexpr auto kSmallTExactLaunchers =
    make_small_t_exact_launchers(std::make_index_sequence<4>{});

using SmallTI8Launcher = void (*)(const Tensor&, const Weight&, Tensor&, const std::int8_t*,
                                  const __half*, cudaStream_t);

// Per-width schedule for the int8 small-T kernel (ops/linear/q4/q4_small_t_mma_i8.cuh). Swept the
// same way the bf16 table above was: one kernel, cold L2, paired against the bf16 kernel in the
// same sitting. Numbers live beside the sweep in the kernel header.
template <int TileCols>
struct Q4SwiGluSmallTI8Tile {
    static constexpr int kKWarps =
        TileCols <= 8 ? 8 : (TileCols <= 16 ? 4 : (TileCols <= 24 ? 4 : 4));
    static constexpr int kStages =
        TileCols <= 8 ? 1 : (TileCols <= 16 ? 1 : (TileCols <= 24 ? 1 : 1));
    static constexpr int kTilesPerWarp = 1;
};

// Masked is a runtime column count, so the staging loop's bounds test cannot fold away. That costs
// this kernel more than it costs the bf16 one -- its staging carries a tile-group test as well --
// and measured 225.3us against 206.8 at T=32 where an exact width runs 220.2 against 236.5. Widths
// that fill their tile exactly therefore get their own unmasked instantiation.
template <int ActiveCols, bool Masked>
void launch_small_t_i8_active(const Tensor& x, const Weight& w, Tensor& out,
                              const std::int8_t* x_codes, const __half* x_scales,
                              cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Tile            = Q4SwiGluSmallTI8Tile<TileCols>;
    using Layout          = SmallTLayout<Tile::kKWarps, Tile::kTilesPerWarp>;
    constexpr int kBlocks =
        kIntermediate / (Q4SwiGluSmallTRows::kOutputRowsPerTile * Layout::kRowTiles);
    const Q4SwiGluSmallTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), x.ne[1]};
    q4_small_t_mma_i8_launch<Q4SwiGluSmallTGeometry, TileCols, ActiveCols,
                             Q4SwiGluSmallTEpilogue, Q4SwiGluSmallTRows, Masked, Tile::kKWarps,
                             Tile::kStages, Tile::kTilesPerWarp>(
        kBlocks, stream, x_codes, x_scales, static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        epilogue, Q4SwiGluSmallTRows{}, x.ne[1]);
}

template <bool Masked, std::size_t... Offsets>
constexpr auto make_small_t_i8_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTI8Launcher, sizeof...(Offsets)>{
        &launch_small_t_i8_active<8 * (1 + static_cast<int>(Offsets)), Masked>...};
}

constexpr auto kSmallTI8Launchers = make_small_t_i8_launchers<false>(std::make_index_sequence<4>{});

__device__ __forceinline__ void q4_issue_pair_tile(uint4 (*__restrict__ s_code)[kVecsPerWarpTile],
                                                   uint4 (*__restrict__ s_scale)[2],
                                                   const std::uint8_t* __restrict__ gate_code_row,
                                                   const std::uint8_t* __restrict__ gate_scale_row,
                                                   const std::uint8_t* __restrict__ up_code_row,
                                                   const std::uint8_t* __restrict__ up_scale_row,
                                                   int tile, int lane) {
    const int g0 = tile * kGroupsPerWarpTile;
    pipe_copy<16>(&s_code[0][lane],
                  reinterpret_cast<const uint4*>(gate_code_row + g0 * kBytesPerGroup) + lane);
    pipe_copy<16>(&s_code[1][lane],
                  reinterpret_cast<const uint4*>(up_code_row + g0 * kBytesPerGroup) + lane);
    if (lane < 2) {
        pipe_copy<16>(&s_scale[0][lane],
                      reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        pipe_copy<16>(&s_scale[1][lane],
                      reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[kK];
    __shared__ uint4 code_tile[kWarpsPerBlock][kStages][2][kVecsPerWarpTile];
    __shared__ uint4 scale_tile[kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * kGroups * kBytesPerGroup;
    const std::uint8_t* gate_scale_row = scales + static_cast<std::int64_t>(out_row) * kGroups * 2;
    const std::uint8_t* up_code_row =
        codes + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales + static_cast<std::int64_t>(out_row + kIntermediate) * kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q4_issue_pair_tile(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                               gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
        // Eight weights per lane per step, not two. Same change, same reason, and the same 4x as
        // q5_rowsplit_gemv's consume loop -- see the long note there for the measurement that
        // motivated it (DRAM at 50% while Mem Pipes Busy sat at 81%).
        //
        // This kernel was the worse of the two: one byte of gate codes, one byte of up codes and
        // two 2-byte scale broadcasts produced four weights, so five memory-pipe instructions per
        // group. Now each lane takes a 4-byte word from each of gate and up -- 8 weights each --
        // so eight lanes cover a group's 32 code bytes and 32 lanes cover FOUR groups per step.
        // Five instructions per four groups against twenty.
        //
        // Q4 has no high plane, so this is simpler than the Q5 case: code byte b holds the weights
        // at 2b (low nibble) and 2b+1 (high), which makes weight k0+j nibble (j & 1) of byte
        // (j >> 1). Verified on the host that the (sub, pos) split covers all 1,024 weights of a
        // tile exactly once, and the code index comes out as step * 32 + lane, so the shared reads
        // stay conflict-free.
        const auto* gate_codes32 = reinterpret_cast<const std::uint32_t*>(gate_codes);
        const auto* up_codes32   = reinterpret_cast<const std::uint32_t*>(up_codes);
        constexpr int kWeightsPerLane = 8;
        constexpr int kLanesPerGroup  = kGroupK / kWeightsPerLane; // 8
        constexpr int kGroupsPerStep  = 32 / kLanesPerGroup;       // 4
        constexpr int kWordsPerGroup  = kBytesPerGroup / 4;        // 8
        const int sub                 = lane / kLanesPerGroup;
        const int pos                 = lane % kLanesPerGroup;
#pragma unroll
        for (int step = 0; step < kGroupsPerWarpTile / kGroupsPerStep; ++step) {
            const int tile_group = step * kGroupsPerStep + sub;
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const std::uint32_t gate_word = gate_codes32[tile_group * kWordsPerGroup + pos];
            const std::uint32_t up_word    = up_codes32[tile_group * kWordsPerGroup + pos];

            const int k0 = (tile * kGroupsPerWarpTile + tile_group) * kGroupK +
                           pos * kWeightsPerLane;
            const uint4 xv  = load_vec<uint4>(reinterpret_cast<const uint4*>(x2 + (k0 >> 1)));
            const float2 f0 = bf16x2_bits_to_float2(xv.x);
            const float2 f1 = bf16x2_bits_to_float2(xv.y);
            const float2 f2 = bf16x2_bits_to_float2(xv.z);
            const float2 f3 = bf16x2_bits_to_float2(xv.w);
            const float xs[kWeightsPerLane]{f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};
#pragma unroll
            for (int j = 0; j < kWeightsPerLane; ++j) {
                const int shift = 8 * (j >> 1) + 4 * (j & 1);
                const int gate_q = sign_extend<4>(static_cast<int>((gate_word >> shift) & 0x0fu));
                const int up_q   = sign_extend<4>(static_cast<int>((up_word >> shift) & 0x0fu));
                gate_acc = fmaf(static_cast<float>(gate_q) * gate_scale, xs[j], gate_acc);
                up_acc   = fmaf(static_cast<float>(up_q) * up_scale, xs[j], up_acc);
            }
        }
        __syncwarp();
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(silu(gate_acc) * up_acc); }
}

} // namespace

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu GEMV requires weight [34816,5120]");
    }
    const int grid = kIntermediate / kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<<<grid, kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    if (fused_route_table("unified/q4_linear_swiglu", x.ne[1]) == LinearRouteTable::Unified) {
        q4_linear_swiglu_small_t_unified_launch(x, w, out, stream);
        return;
    }
    // A width that fills its tile needs no runtime column count, and dropping it is worth 2-4% at
    // sixteen and twenty-four columns (bench/ops/q4_linear_swiglu_schedule_bench.cu prices it
    // against small_t_masked; + is the exact instantiation winning):
    //
    //   T        8       16      24      32
    //   run 1  -27.3%   +1.3%   +3.7%   +2.0%
    //   run 2  -16.5%   +2.6%   +3.6%   +0.0%
    //   run 3  -15.7%   +3.3%   +3.7%   +0.5%
    //
    // Eight columns is excluded because there the exact instantiation *spills*: ptxas reports 16
    // bytes of cumulative stack against the masked build's none. A compile-time column count lets
    // the staging loop unroll fully, and that clears the 42-register ceiling this schedule's
    // __launch_bounds__(256, 6) imposes at KWarps 8. Every wider tile runs KWarps < 8, so it is
    // bounded to 2 blocks per SM and unrolls with registers to spare.
    const bool exact  = x.ne[1] % 8 == 0 && x.ne[1] > 8;
    const auto& table = exact ? kSmallTExactLaunchers : kSmallTLaunchers;
    table[static_cast<std::size_t>((x.ne[1] - 1) / 8)](x, w, out, stream);
}

// Always the runtime-column-count instantiation, so a bench can price masking against the exact
// one above inside a single process. Nothing on the inference path should call this.
void q4_linear_swiglu_small_t_tiled_masked_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                  cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    kSmallTLaunchers[static_cast<std::size_t>((x.ne[1] - 1) / 8)](x, w, out, stream);
}

bool q4_linear_swiglu_small_t_i8_supported(std::int32_t tokens) noexcept {
    return tokens >= 16 && tokens <= 32;
}

std::size_t q4_linear_swiglu_small_t_tiled_i8_workspace_bytes(std::int32_t tokens) {
    // s8 codes plus one FP16 scale per (token, group), 256-aligned like q4a8_swiglu's workspace,
    // over the padded tile width rather than the request's: see the launch below for why the pad
    // columns exist.
    const std::size_t t = static_cast<std::size_t>((tokens + 7) / 8 * 8);
    return ((t * kK + 255) / 256) * 256 + ((t * kGroups * sizeof(__half) + 255) / 256) * 256;
}

void q4_linear_swiglu_small_t_tiled_i8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens < 2 || tokens > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU small-T i8 requires T=2..32");
    }
    if (w.n != kN || w.k != kK || w.padded_shape[1] != kK) {
        throw std::invalid_argument("q4 linear_swiglu small-T i8 requires weight [34816,5120]");
    }

    // Columns are padded up to the tile width and the pad zeroed, so every width runs the
    // compile-time-width kernel rather than one carrying a runtime column count. Masking is not
    // free here: the staging loop's bounds test cannot fold away and it sits beside a tile-group
    // test, which measured T=28 at 222.2us against T=32's 193.5 on the same 32-wide tile. The
    // being zero -- its MMAs were issued by the masked path too, only against a zeroed fragment.
    // epilogue already discards columns past the request, so the pad needs no handling beyond
    const std::int32_t padded = (tokens + 7) / 8 * 8;
    auto scope                = workspace.scope();
    const DeviceSpan codes    = workspace.alloc_bytes(static_cast<std::size_t>(padded) * kK);
    const DeviceSpan xscale =
        workspace.alloc_bytes(static_cast<std::size_t>(padded) * kGroups * sizeof(__half));
    if (padded != tokens) {
        const std::size_t pad_codes = static_cast<std::size_t>(padded - tokens) * kK;
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(codes.data) +
                                       static_cast<std::size_t>(tokens) * kK,
                                   0, pad_codes, stream));
        const std::size_t pad_scales =
            static_cast<std::size_t>(padded - tokens) * kGroups * sizeof(__half);
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint8_t*>(xscale.data) +
                                       static_cast<std::size_t>(tokens) * kGroups * sizeof(__half),
                                   0, pad_scales, stream));
    }
    q4_small_t_quantize_activations(static_cast<const __nv_bfloat16*>(x.data), kK, tokens,
                                    reinterpret_cast<std::int8_t*>(codes.data),
                                    reinterpret_cast<__half*>(xscale.data), stream);

    kSmallTI8Launchers[static_cast<std::size_t>(padded / 8 - 1)](
        x, w, out, reinterpret_cast<const std::int8_t*>(codes.data),
        reinterpret_cast<const __half*>(xscale.data), stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
