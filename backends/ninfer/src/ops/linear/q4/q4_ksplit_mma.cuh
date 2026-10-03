#pragma once

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/small_t_layout.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q4KSplitStoreEpilogue {};

struct Q4KSplitIdentityRows {
    static constexpr int kOutputRowsPerTile = 16;
    static constexpr int kOutputRowsPerCta  = kOutputRowsPerTile;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

template <int OutputRows, int InputRows>
struct Q4LinearGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

struct Q4KSplitMmaSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 6;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

// Shared memory of q4_ksplit_mma_kernel: a ring of Stages K groups, reused for the final K
// reduction; static, as rings deeper than two lose more occupancy than they hide. Padding: a half-warp of 64-bit code loads spans four rows, so code rows are staggered
// by 32 B (eight banks; a row is 32 * KWarps bytes, so the stride is 32 or 96 mod 128); unpadded,
// rows 256 B apart all share a bank. Activation columns are 64 bf16 (128 B) per K warp and 16 B of
// padding staggers neighbours by four banks for the quarter-warp 128-bit loads.
template <int KWarps, int TileCols, int Stages, int TilesPerWarp = 1>
struct Q4KSplitStorage {
    using Layout                     = SmallTLayout<KWarps, TilesPerWarp>;
    static constexpr int kCodeRowPad = 32;
    static constexpr int kXColStride = 64 + 8;

    struct Stage {
        std::uint8_t codes[Layout::kRowsPerCta][Layout::kGroupK / 2 + kCodeRowPad];
        __nv_bfloat16 activations[KWarps][TileCols][kXColStride];
        std::uint16_t scales[Layout::kRowsPerCta][KWarps];
    };
    union Shared {
        Stage stages[Stages];
        float partial[Layout::kWarps * TilesPerWarp * (TileCols / 8) * 32 * 4];
    };
    static constexpr int kBytes = sizeof(Shared);
};

// KWarps / TilesPerWarp / Stages: the warp layout (ops/common/small_t_layout.cuh) and the depth
// of the cp.async ring. RowPolicy maps one sixteen-row MMA tile to weight rows and names how many
// output rows the tile produces; a CTA covers SmallTLayout::kRowTiles consecutive tiles, so the
// grid is output rows / (RowPolicy::kOutputRowsPerTile * kRowTiles).
template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4KSplitStoreEpilogue,
          class RowPolicy = Q4KSplitIdentityRows, bool MaskedColumns = false, int KWarps = 8,
          int Stages = 1, int TilesPerWarp = 1,
          int MinBlocks = (KWarps < 8 ? 2 : (TileCols <= 16 ? 6 : 4))>
__launch_bounds__(256, MinBlocks) __global__
    void q4_ksplit_mma_kernel(const __nv_bfloat16* __restrict__ x,
                               const std::uint8_t* __restrict__ codes,
                               const std::uint8_t* __restrict__ scales,
                               __nv_bfloat16* __restrict__ out, Epilogue epilogue = {},
                               RowPolicy row_policy = {}, int columns = ActiveCols) {
    using Layout                = SmallTLayout<KWarps, TilesPerWarp>;
    constexpr int kTpw          = TilesPerWarp;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = 64;
    constexpr int kKWarps       = Layout::kKWarps;
    constexpr int kRowTiles     = Layout::kRowTiles;
    constexpr int kRowsPerCta   = Layout::kRowsPerCta;
    constexpr int kGroupK       = Layout::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(RowPolicy::kOutputRowsPerTile <= 16);
    static_assert(Stages >= 1 && Stages <= 4);

    // K order. The MMA's k slots may be any permutation of a group's 64 k, provided A and B agree.
    // Lane lid is given k 16*lid .. 16*lid + 15 of its warp's group: for step ks, slots (2*lid,
    // 2*lid + 1) carry k 16*lid + 4*ks + {0, 1} and slots (2*lid + 8, 2*lid + 9) carry
    // 16*lid + 4*ks + {2, 3}. The lane's A operand over a group is then eight contiguous code bytes
    // per row -- one 64-bit load, decoded by Q4MmaDecodeAtom::decode_eight without int-to-float
    // conversion -- and its B operand sixteen contiguous activations of one column, two 128-bit
    // loads, where the natural slot order costs a byte load and a two-conversion decode per A pair
    // and an ldmatrix per B fragment. On sm_86 that instruction stream, not DRAM, bounded this
    // kernel: once the shared-memory conflicts were gone it still spent ~27 cycles per MMA against
    // the ~21 its weight bytes allow at full bandwidth.
    using Storage = Q4KSplitStorage<KWarps, TileCols, Stages, TilesPerWarp>;
    using Stage   = typename Storage::Stage;
    __shared__ __align__(16) typename Storage::Shared shared;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int k_split      = warp % kKWarps;
    const int first_tile   = warp / kKWarps * kTpw; // this warp's tiles: first_tile + [0, kTpw)
    const int tile0        = static_cast<int>(blockIdx.x) * kRowTiles;
    const int live_columns = MaskedColumns ? columns : ActiveCols;

    // Only live columns are staged; the B fragments of the others are zero.
    const auto stage_x = [&](int group_k0, Stage& stage) {
        constexpr int kChunksPerCol = kGroupK / 8;
        const int items             = live_columns * kChunksPerCol;
        for (int item = tid; item < items; item += Layout::kThreads) {
            const int col   = item / kChunksPerCol;
            const int chunk = item - col * kChunksPerCol;
            cp_async<16>(&stage.activations[chunk / 8][col][(chunk % 8) * 8],
                         &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + chunk * 8]);
        }
    };

    // kTpw 16-byte code chunks per thread: rows * (kGroupK / 32) = 256 * kTpw.
    const auto stage_weight = [&](int group_k0, Stage& stage) {
        constexpr int kChunksPerRow = kGroupK / 32;
        static_assert(kRowsPerCta * kChunksPerRow == Layout::kThreads * kTpw);
        // A constant trip count keeps this straight-line: a tid-strided loop here costs the
        // narrow tiles, which run at 42 registers, five spills and ~15% at T=4.
#pragma unroll
        for (int i = 0; i < kTpw; ++i) {
            const int item       = tid + i * Layout::kThreads;
            const int row        = item / kChunksPerRow;
            const int chunk      = item % kChunksPerRow;
            const int weight_row = row_policy.weight_row(
                (tile0 + row / 16) * RowPolicy::kOutputRowsPerTile, row % 16);
            cp_async<16, Q4KSplitMmaSchedule::kCodeCache>(
                &stage.codes[row][chunk * 16],
                codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                    chunk * 16);
        }
        if (tid < kRowsPerCta) {
            const int row        = tid;
            const int weight_row = row_policy.weight_row(
                (tile0 + row / 16) * RowPolicy::kOutputRowsPerTile, row % 16);
            cp_async<2 * kKWarps>(&stage.scales[row][0],
                                  scales + (static_cast<std::int64_t>(weight_row) *
                                                Geometry::kGroupsPerRow +
                                            group_k0 / 64) *
                                               2);
        }
    };

    // The weights are immutable: with a row map that reads no memory, group 0's stream before the
    // dependency wait, and its activations and every later group follow it.
    constexpr bool kPrefetchWeights = std::is_empty_v<RowPolicy>;
    const auto issue_group          = [&](int group_index) {
        if (group_index < kGroups) {
            Stage& stage = shared.stages[group_index % Stages];
            if (!kPrefetchWeights || group_index != 0) {
                stage_weight(group_index * kGroupK, stage);
            }
            stage_x(group_index * kGroupK, stage);
        }
        cp_commit(); // empty commits keep cp_wait<Stages - 1> exact through the tail
    };

    const int code_byte     = k_split * (kTileK / 2) + 8 * lid;
    float acc[kTpw][kNt][4] = {};

    if constexpr (kPrefetchWeights) { stage_weight(0, shared.stages[0]); }
    pdl::enter_streaming();
#pragma unroll
    for (int prefetch = 0; prefetch < Stages - 1; ++prefetch) { issue_group(prefetch); }

#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        issue_group(group_index + Stages - 1);
        cp_wait<Stages - 1>();
        __syncthreads();

        const Stage& stage            = shared.stages[group_index % Stages];
        float group_acc[kTpw][kNt][4] = {};
        uint2 top[kTpw], bot[kTpw];
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row = (first_tile + t) * 16 + gid;
            top[t]             = load_vec<uint2>(&stage.codes[tile_row][code_byte]);
            bot[t]             = load_vec<uint2>(&stage.codes[tile_row + 8][code_byte]);
        }

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            // Four code bytes per row cover steps ks = 2*half and 2*half + 1.
            unsigned a_top[kTpw][4], a_bot[kTpw][4];
#pragma unroll
            for (int t = 0; t < kTpw; ++t) {
                Q4MmaDecodeAtom::decode_eight(half == 0 ? top[t].x : top[t].y, a_top[t]);
                Q4MmaDecodeAtom::decode_eight(half == 0 ? bot[t].x : bot[t].y, a_bot[t]);
            }
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = nt * 8 + gid;
                uint4 b       = make_uint4(0u, 0u, 0u, 0u);
                if (col < live_columns) {
                    b = load_vec<uint4>(&stage.activations[k_split][col][16 * lid + 8 * half]);
                }
#pragma unroll
                for (int t = 0; t < kTpw; ++t) {
                    float(&g)[4] = group_acc[t][nt];
                    mma_bf16(g[0], g[1], g[2], g[3], a_top[t][0], a_bot[t][0], a_top[t][1],
                             a_bot[t][1], b.x, b.y);
                    mma_bf16(g[0], g[1], g[2], g[3], a_top[t][2], a_bot[t][2], a_top[t][3],
                             a_bot[t][3], b.z, b.w);
                }
            }
        }

#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row    = (first_tile + t) * 16 + gid;
            const float top_scale = __half2float(__ushort_as_half(stage.scales[tile_row][k_split]));
            const float bot_scale =
                __half2float(__ushort_as_half(stage.scales[tile_row + 8][k_split]));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[t][nt][0] = fmaf(group_acc[t][nt][0], top_scale, acc[t][nt][0]);
                acc[t][nt][1] = fmaf(group_acc[t][nt][1], top_scale, acc[t][nt][1]);
                acc[t][nt][2] = fmaf(group_acc[t][nt][2], bot_scale, acc[t][nt][2]);
                acc[t][nt][3] = fmaf(group_acc[t][nt][3], bot_scale, acc[t][nt][3]);
            }
        }
        // The next iteration's issue_group writes the stage this one just read.
        __syncthreads();
    }
    pdl::trigger_dependents();
    cp_wait<0>();

    // Reduce each tile's K partials: odd K warps publish, even ones fold their neighbour, then K
    // warp 0 sums the even ones.
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
                store_vec(slot(warp, t, nt), make_float4(a[0], a[1], a[2], a[3]));
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
                const float4 partner = load_vec<float4>(slot(warp + 1, t, nt));
                a[0] += partner.x;
                a[1] += partner.y;
                a[2] += partner.z;
                a[3] += partner.w;
                if (k_split != 0) { store_vec(slot(warp, t, nt), make_float4(a[0], a[1], a[2], a[3])); }
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
                    const float4 value = load_vec<float4>(slot(warp + split, t, nt));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int col0 = nt * 8 + 2 * lid;
                if constexpr (std::is_same_v<Epilogue, Q4KSplitStoreEpilogue>) {
                    constexpr int kRows = Geometry::kOutputRows;
                    if (col0 < live_columns) {
                        out[static_cast<std::int64_t>(col0) * kRows + row0 + gid] =
                            __float2bfloat16_rn(sum.x);
                        out[static_cast<std::int64_t>(col0) * kRows + row0 + gid + 8] =
                            __float2bfloat16_rn(sum.z);
                    }
                    if (col0 + 1 < live_columns) {
                        out[static_cast<std::int64_t>(col0 + 1) * kRows + row0 + gid] =
                            __float2bfloat16_rn(sum.y);
                        out[static_cast<std::int64_t>(col0 + 1) * kRows + row0 + gid + 8] =
                            __float2bfloat16_rn(sum.w);
                    }
                } else {
                    epilogue.template store<ActiveCols>(row0 + gid, col0, sum);
                }
            }
        }
    }
}

// Launches q4_ksplit_mma_kernel over `blocks` CTAs.
template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4KSplitStoreEpilogue,
          class RowPolicy = Q4KSplitIdentityRows, bool MaskedColumns = false, int KWarps = 8,
          int Stages = 1, int TilesPerWarp = 1,
          int MinBlocks = (KWarps < 8 ? 2 : (TileCols <= 16 ? 6 : 4))>
void q4_ksplit_mma_launch(int blocks, cudaStream_t stream, const __nv_bfloat16* x,
                           const std::uint8_t* codes, const std::uint8_t* scales,
                           __nv_bfloat16* out, Epilogue epilogue = {}, RowPolicy row_policy = {},
                           int columns = ActiveCols) {
    constexpr int kBytes  = Q4KSplitStorage<KWarps, TileCols, Stages, TilesPerWarp>::kBytes;
    constexpr auto kernel = q4_ksplit_mma_kernel<Geometry, TileCols, ActiveCols, Epilogue,
                                                  RowPolicy, MaskedColumns, KWarps, Stages,
                                                  TilesPerWarp, MinBlocks>;
    static_assert(kBytes <= 48 * 1024, "small-T MMA stages must fit static shared memory");
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned>(blocks)), dim3(SmallTLayout<KWarps>::kThreads), 0, stream},
        kernel, x, codes, scales, out, epilogue, row_policy, columns));
}

} // namespace ninfer::ops::detail
