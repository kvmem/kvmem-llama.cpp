#pragma once

// T2G128 RowSplit x BF16 tensor-core GEMM for prefill widths.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// The Q4 MMA schedule (ops/linear/q4/q4_rowsplit_gemm_mma.cuh) with the weight tile loader
// rewritten for T2: a K tile of 64 is half of one 128-wide group, so every row stages 16 code
// bytes (one cp.async vector) and the group's binary16 scale per tile; lane i decodes codes
// 2i and 2i + 1 of that half group into one bf16 pair of the swizzled A tile. Activations,
// ldmatrix fragments, the MMA loop and the epilogue are the Q4 kernel's.

#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q4/q4_rowsplit_gemm_mma.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int BlockRows_, int BlockCols_, int WarpRows_, int WarpCols_, int PipelineStages_,
          int LaunchBoundsMinBlocks_, Q4FragmentPipeline FragmentPipeline_, Cache QuantCache_,
          Cache ActivationCache_>
struct T2RowSplitMmaGemmSchedule {
    static constexpr int kBlockRows = BlockRows_;
    static constexpr int kBlockCols = BlockCols_;
    static constexpr int kBlockK    = T2RowSplitStorage::kGroupK / 2;
    static constexpr int kWarpRows  = WarpRows_;
    static constexpr int kWarpCols  = WarpCols_;

    static constexpr int kPipelineStages                  = PipelineStages_;
    static constexpr int kLaunchBoundsMinBlocks           = LaunchBoundsMinBlocks_;
    static constexpr Q4FragmentPipeline kFragmentPipeline = FragmentPipeline_;
    static constexpr Cache kQuantCache                    = QuantCache_;
    static constexpr Cache kActivationCache               = ActivationCache_;

    static constexpr int kWarpGridRows = kBlockRows / kWarpRows;
    static constexpr int kWarpGridCols = kBlockCols / kWarpCols;
    static constexpr int kWarps        = kWarpGridRows * kWarpGridCols;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMmaRows      = kWarpRows / 16;
    static constexpr int kMmaCols      = kWarpCols / 8;
    static constexpr int kMmaKSteps    = kBlockK / 16;
    // Half a group per K tile: 16 code bytes and one scale word per row.
    static constexpr int kCodeBytesPerTileRow  = T2RowSplitStorage::kCodeBytesPerGroup / 2;
    static constexpr int kScaleBytesPerTileRow = T2RowSplitStorage::kScaleBytesPerGroup;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockCols * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockRows * kCodeBytesPerTileRow +
        kPipelineStages * kBlockRows * kScaleBytesPerTileRow;

    static_assert(kBlockRows > 0 && kBlockCols > 0);
    static_assert(kBlockK == 64, "T2 MMA stages half of one quant group per K tile");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0,
                  "T2 MMA block tile must divide into warp tiles");
    static_assert(kWarpRows % 16 == 0 && kWarpCols % 8 == 0,
                  "T2 MMA warp tile must be composed of m16n8 MMA tiles");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "T2 MMA cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "T2 MMA staged shared memory exceeds the static 48 KiB budget");
};

// clang-format off
template <class Schedule_, bool Full>
__global__ __launch_bounds__(Schedule_::kThreads, Schedule_::kLaunchBoundsMinBlocks)
void t2_rowsplit_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k) {
    // clang-format on
    using Schedule       = Schedule_;
    constexpr bool kFull = Full;
    constexpr int BM     = Schedule::kBlockRows;
    constexpr int BN     = Schedule::kBlockCols;
    constexpr int BK     = Schedule::kBlockK;
    constexpr int WM     = Schedule::kWarpRows;
    constexpr int WN     = Schedule::kWarpCols;
    constexpr int MT     = Schedule::kMmaRows;
    constexpr int NT     = Schedule::kMmaCols;
    constexpr int KSUB   = Schedule::kMmaKSteps;
    constexpr int S      = Schedule::kPipelineStages;
    constexpr int CB     = Schedule::kCodeBytesPerTileRow;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * CB];
    __shared__ __align__(4) std::uint16_t Sr[S][BM];

    const int groups_per_row = padded_k / T2RowSplitStorage::kGroupK;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int warp_row       = warp / Schedule::kWarpGridCols;
    const int warp_col       = warp % Schedule::kWarpGridCols;
    const int mma_row        = lane >> 2;
    const int mma_col        = lane & 3;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    float accum[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            accum[mi][ni][0] = 0.0f;
            accum[mi][ni][1] = 0.0f;
            accum[mi][ni][2] = 0.0f;
            accum[mi][ni][3] = 0.0f;
        }
    }

    const int k_tiles = padded_k / BK;

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Schedule::kThreads) {
            const int local_col = item / (BK / 8);
            const int k8        = item - local_col * (BK / 8);
            const int kk        = k0 + k8 * 8;
            const int col       = col0 + local_col;
            auto* dst = &Bs[stage][local_col * BK + q4_mma_swizzle_k64(local_col, k8 * 8)];
            if constexpr (kFull) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                if (col < cols && kk + 8 <= k) {
                    cp_async<16, Schedule::kActivationCache>(
                        dst, &x[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int group = (k_tile * BK) / T2RowSplitStorage::kGroupK;
        const int half  = k_tile & 1;
#pragma unroll 1
        for (int local_row = tid; local_row < BM; local_row += Schedule::kThreads) {
            const int row = row0 + local_row;
            auto* dst     = &Cr[stage][local_row * CB];
            auto* sdst    = &Sr[stage][local_row];
            if (kFull || row < rows) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group;
                cp_async<16, Schedule::kQuantCache>(
                    dst, &codes[group_index * T2RowSplitStorage::kCodeBytesPerGroup + half * CB]);
                *sdst = *reinterpret_cast<const std::uint16_t*>(
                    &scales[group_index * T2RowSplitStorage::kScaleBytesPerGroup]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
                *sdst = 0;
            }
        }
    };

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_quant(stage, k_tile);
    };

    // Lane i owns codes 2i and 2i + 1 of the half group: byte i / 2, fields (i & 1) * 2 and + 1.
    auto decode_weight = [&](int stage) {
        for (int local_row = warp; local_row < BM; local_row += Schedule::kWarps) {
            const std::uint32_t byte = Cr[stage][local_row * CB + (lane >> 1)];
            const std::uint32_t pair = (byte >> ((lane & 1) * 4)) & 0xFu;
            const float scale        = __half2float(__ushort_as_half(Sr[stage][local_row]));
            const auto decode        = [scale](std::uint32_t field) -> float {
                return (field & 0x1u) ? ((field & 0x2u) ? -scale : scale) : 0.0f;
            };
            const __nv_bfloat162 weights =
                __floats2bfloat162_rn(decode(pair & 0x3u), decode((pair >> 2) & 0x3u));
            auto* dst            = &As[local_row * BK];
            const int shared_col = q4_mma_swizzle_k64(local_row, 2 * lane);
            store_vec(&dst[shared_col], weights);
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        cp_wait<S - 1>();
        __syncthreads();

        decode_weight(stage);
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned (&a_frag)[MT][4],
                                  unsigned (&b_frag)[NT][2]) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_row * WM + mi * 16 + a_row_offset;
                const int col = k_step + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&As[row * BK + q4_mma_swizzle_k64(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_col * WN + ni * 8 + b_inner_row;
                const int col = k_step + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&Bs[stage][row * BK + q4_mma_swizzle_k64(row, col)]));
            }
        };

        if constexpr (Schedule::kFragmentPipeline == Q4FragmentPipeline::PingPong) {
            unsigned a_frag[2][MT][4];
            unsigned b_frag[2][NT][2];
            load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                const int current = ki & 1;
                const int next    = (ki + 1) & 1;
                if (ki + 1 < KSUB) { load_fragments((ki + 1) * 16, a_frag[next], b_frag[next]); }
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                                 accum[mi][ni][3], a_frag[current][mi][0], a_frag[current][mi][1],
                                 a_frag[current][mi][2], a_frag[current][mi][3],
                                 b_frag[current][ni][0], b_frag[current][ni][1]);
                    }
                }
            }
        } else {
            unsigned a_frag[MT][4];
            unsigned b_frag[NT][2];
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                load_fragments(ki * 16, a_frag, b_frag);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                                 accum[mi][ni][3], a_frag[mi][0], a_frag[mi][1], a_frag[mi][2],
                                 a_frag[mi][3], b_frag[ni][0], b_frag[ni][1]);
                    }
                }
            }
        }

        __syncthreads();
        const int prefetch_tile = k_tile + S;
        if (prefetch_tile < k_tiles) { stage_inputs(stage, prefetch_tile); }
        cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int output_row0 = row0 + warp_row * WM + mi * 16 + mma_row;
        const int output_row1 = output_row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int output_col0 = col0 + warp_col * WN + ni * 8 + 2 * mma_col;
            const int output_col1 = output_col0 + 1;
            const float* values   = accum[mi][ni];
            if constexpr (kFull) {
                out[static_cast<std::int64_t>(output_col0) * rows + output_row0] =
                    __float2bfloat16_rn(values[0]);
                out[static_cast<std::int64_t>(output_col1) * rows + output_row0] =
                    __float2bfloat16_rn(values[1]);
                out[static_cast<std::int64_t>(output_col0) * rows + output_row1] =
                    __float2bfloat16_rn(values[2]);
                out[static_cast<std::int64_t>(output_col1) * rows + output_row1] =
                    __float2bfloat16_rn(values[3]);
            } else {
                if (output_row0 < rows) {
                    if (output_col0 < cols) {
                        out[static_cast<std::int64_t>(output_col0) * rows + output_row0] =
                            __float2bfloat16_rn(values[0]);
                    }
                    if (output_col1 < cols) {
                        out[static_cast<std::int64_t>(output_col1) * rows + output_row0] =
                            __float2bfloat16_rn(values[1]);
                    }
                }
                if (output_row1 < rows) {
                    if (output_col0 < cols) {
                        out[static_cast<std::int64_t>(output_col0) * rows + output_row1] =
                            __float2bfloat16_rn(values[2]);
                    }
                    if (output_col1 < cols) {
                        out[static_cast<std::int64_t>(output_col1) * rows + output_row1] =
                            __float2bfloat16_rn(values[3]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
