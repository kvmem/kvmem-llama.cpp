#pragma once

// T2G128 RowSplit x BF16 SIMT GEMM for small T (the MTP verify and cohort widths).
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One warp owns one output row across ColsPerTile columns. Raw T2 code bytes and adjacent
// binary16 scale pairs are staged per row with cp.async in whole groups; each lane owns one
// 32-bit word (sixteen codes) of one group per phase, and a phase covers four groups (512 K).
// The group partial is a signed activation sum (codes are {-1, 0, +1}) scaled once per group.
// Derived from ops/linear/q4/q4_rowsplit_gemm_simt.cuh.

#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/t2/t2_rowsplit_gemv.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int RowsPerCta_, int ColsPerTile_, int GroupsPerStage_, int PipelineStages_,
          Cache CodeCache_, int LaunchBoundsMinBlocks_>
struct T2RowSplitSimtGemmSchedule {
    static constexpr int kRowsPerCta            = RowsPerCta_;
    static constexpr int kColsPerTile           = ColsPerTile_;
    static constexpr int kGroupsPerStage        = GroupsPerStage_;
    static constexpr int kPipelineStages        = PipelineStages_;
    static constexpr Cache kCodeCache           = CodeCache_;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks_;

    static constexpr int kCtaWarps = kRowsPerCta;
    static constexpr int kThreads  = kCtaWarps * 32;
    static constexpr int kStageK   = kGroupsPerStage * T2RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage =
        kGroupsPerStage * T2RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
    static constexpr int kScalePairsPerStage = kGroupsPerStage / 2;
    static constexpr int kCodePhases         = kGroupsPerStage / 4;
    static constexpr int kSharedBytes =
        kRowsPerCta * kPipelineStages *
        (kGroupsPerStage * T2RowSplitStorage::kCodeBytesPerGroup +
         kScalePairsPerStage * static_cast<int>(sizeof(std::uint32_t)));

    static_assert(kRowsPerCta > 0 && kRowsPerCta <= 32);
    static_assert(kColsPerTile > 0 && kColsPerTile <= 8);
    static_assert(kGroupsPerStage > 0 && kGroupsPerStage % 4 == 0,
                  "T2 SIMT phases consume four groups of aligned scale pairs");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "T2 SIMT cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "T2 SIMT staged shared memory exceeds the static 48 KiB budget");
};

template <class Schedule, bool FullStage>
__device__ __forceinline__ void t2_simt_issue_stage(uint4* __restrict__ shared_codes,
                                                    std::uint32_t* __restrict__ shared_scales,
                                                    const std::uint8_t* __restrict__ code_row,
                                                    const std::uint8_t* __restrict__ scale_row,
                                                    int stage, int active_groups, int lane) {
    constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairs = Schedule::kScalePairsPerStage;

    const std::int64_t group0        = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerStage;
    const std::uint8_t* stage_codes  = code_row + group0 * T2RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* stage_scales = scale_row + group0 * T2RowSplitStorage::kScaleBytesPerGroup;

    const int active_code_vecs = FullStage ? kCodeVecs
                                           : active_groups * T2RowSplitStorage::kCodeBytesPerGroup /
                                                 static_cast<int>(sizeof(uint4));
    for (int vec = lane; vec < kCodeVecs; vec += 32) {
        if (FullStage || vec < active_code_vecs) {
            cp_async<16, Schedule::kCodeCache>(&shared_codes[vec],
                                               stage_codes + static_cast<std::int64_t>(vec) * 16);
        } else {
            shared_codes[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    // A partial stage still copies whole aligned scale pairs; the odd trailing group's pair
    // partner lies inside the row's padded scale plane.
    const int active_scale_pairs = FullStage ? kScalePairs : (active_groups + 1) / 2;
    for (int pair = lane; pair < kScalePairs; pair += 32) {
        if (FullStage || pair < active_scale_pairs) {
            cp_async<4>(&shared_scales[pair], stage_scales + static_cast<std::int64_t>(pair) * 4);
        } else {
            shared_scales[pair] = 0u;
        }
    }
    cp_commit();
}

template <class Schedule, bool FullStage, bool FullCols>
__device__ __forceinline__ void
t2_simt_consume_stage(const __nv_bfloat16* __restrict__ x, std::int32_t k, int col0,
                      int active_cols, int stage, int active_groups,
                      const uint4* __restrict__ shared_codes,
                      const std::uint32_t* __restrict__ shared_scales, int lane,
                      float (&acc)[Schedule::kColsPerTile]) {
    constexpr int kCols       = Schedule::kColsPerTile;
    constexpr int kCodePhases = Schedule::kCodePhases;

#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group        = phase * 4 + (lane >> 3);
        const int stage_groups = FullStage ? Schedule::kGroupsPerStage : active_groups;
        if (group < stage_groups) {
            const std::uint32_t packed =
                reinterpret_cast<const std::uint32_t*>(shared_codes)[phase * 32 + lane];
            const std::uint32_t scale_pair = shared_scales[group >> 1];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(scale_pair >> ((group & 1) * 16));
            const float scale = __half2float(__ushort_as_half(scale_bits));

            // Word `lane` of the phase is word (lane & 7) of group (phase * 4 + (lane >> 3)):
            // sixteen consecutive codes starting at k = group * 128 + (lane & 7) * 16.
            const std::int64_t xk = static_cast<std::int64_t>(stage) * Schedule::kStageK +
                                    static_cast<std::int64_t>(phase) * 512 +
                                    lane * T2RowSplitStorage::kCodesPerWord;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                if (FullCols || col < active_cols) {
                    const __nv_bfloat16* column =
                        x + static_cast<std::int64_t>(col0 + col) * k + xk;
                    const uint4 xa = load_vec<uint4>(column);
                    const uint4 xb = load_vec<uint4>(column + 8);
                    acc[col]       = fmaf(t2_gemv_word_partial(packed, xa, xb), scale, acc[col]);
                }
            }
        }
    }
}

template <class Schedule, bool Full>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kLaunchBoundsMinBlocks) void t2_rowsplit_gemm_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                                  const std::
                                                                      uint8_t* __restrict__ codes,
                                                                  const std::
                                                                      uint8_t* __restrict__ scales,
                                                                  __nv_bfloat16* __restrict__ out,
                                                                  std::int32_t out_ld,
                                                                  std::int32_t rows, std::int32_t k,
                                                                  std::int32_t cols,
                                                                  std::int32_t padded_k) {
    constexpr bool kFull              = Full;
    constexpr int kRowsPerCta         = Schedule::kRowsPerCta;
    constexpr int kColsPerTile        = Schedule::kColsPerTile;
    constexpr int kGroupsPerStage     = Schedule::kGroupsPerStage;
    constexpr int kPipelineStages     = Schedule::kPipelineStages;
    constexpr int kPipelinePrefetch   = kPipelineStages - 1;
    constexpr int kCodeVecsPerStage   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairsPerStage = Schedule::kScalePairsPerStage;

    __shared__ __align__(16) uint4 shared_codes[kRowsPerCta][kPipelineStages][kCodeVecsPerStage];
    __shared__ __align__(16)
        std::uint32_t shared_scales[kRowsPerCta][kPipelineStages][kScalePairsPerStage];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
    if constexpr (!kFull) {
        if (row >= rows) { return; }
    }

    const int col0        = static_cast<int>(blockIdx.y) * kColsPerTile;
    const int active_cols = kFull ? kColsPerTile : min(kColsPerTile, cols - col0);

    const int padded_groups = padded_k / T2RowSplitStorage::kGroupK;
    const int groups        = k / T2RowSplitStorage::kGroupK;
    const int stages =
        kFull ? groups / kGroupsPerStage : (groups + kGroupsPerStage - 1) / kGroupsPerStage;

    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * padded_groups *
                                                T2RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * padded_groups *
                                                 T2RowSplitStorage::kScaleBytesPerGroup;

    float acc[kColsPerTile];
#pragma unroll
    for (int col = 0; col < kColsPerTile; ++col) { acc[col] = 0.0f; }

#pragma unroll
    for (int prefetch = 0; prefetch < kPipelinePrefetch; ++prefetch) {
        if (prefetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - prefetch * kGroupsPerStage);
            t2_simt_issue_stage<Schedule, kFull>(shared_codes[warp][prefetch],
                                                 shared_scales[warp][prefetch], code_row, scale_row,
                                                 prefetch, active_groups, lane);
        } else {
            cp_commit();
        }
    }

#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + kPipelinePrefetch;
        if (fetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - fetch * kGroupsPerStage);
            const int buffer = fetch % kPipelineStages;
            t2_simt_issue_stage<Schedule, kFull>(shared_codes[warp][buffer],
                                                 shared_scales[warp][buffer], code_row, scale_row,
                                                 fetch, active_groups, lane);
        } else {
            cp_commit();
        }

        cp_wait<kPipelinePrefetch>();
        __syncwarp();

        const int active_groups =
            kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - stage * kGroupsPerStage);
        const int buffer = stage % kPipelineStages;
        t2_simt_consume_stage<Schedule, kFull, kFull>(x, k, col0, active_cols, stage, active_groups,
                                                      shared_codes[warp][buffer],
                                                      shared_scales[warp][buffer], lane, acc);
        __syncwarp();
    }

#pragma unroll
    for (int col = 0; col < kColsPerTile; ++col) {
        if (kFull || col < active_cols) {
            const float sum = warp_reduce_sum(acc[col]);
            if (lane == 0) {
                out[static_cast<std::int64_t>(col0 + col) * out_ld + row] = __float2bfloat16(sum);
            }
        }
    }
}

} // namespace ninfer::ops::detail
