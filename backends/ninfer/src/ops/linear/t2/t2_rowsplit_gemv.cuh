#pragma once

// T2G128_F16S RowSplit x BF16 GEMV (T = 1).
//
// out[row] = sum_k w[row, k] * x[k], w = scale_group * code, code in {-1, 0, +1}.
//
// Two lane mappings, both derived from the Q4 GEMV schedules:
//   PackedByte4  -- one warp per row segment; lane i owns byte i of every 32-byte group, i.e.
//                   codes 4i..4i+3 of that group. Static ownership: every warp of a row keeps
//                   StaticGroupsPerRow / WarpsPerRow whole groups, so K is a compile-time fact.
//   PackedWord16 -- lane i owns word i % 8 of group (tile_base + i / 8): sixteen codes, two
//                   16-byte activation vectors. Dynamic ownership over any K that is a multiple
//                   of 512 (four groups per phase).
// A weight of {-1, 0, +1} needs no multiply: the group partial adds or subtracts activations
// and the binary16 scale is applied once per group.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class T2GemvLaneMapping {
    PackedByte4,
    PackedWord16,
};

template <int RowsPerCta, int WarpsPerRow, int GroupsPerWarpTile, T2GemvLaneMapping LaneMapping,
          int StaticGroupsPerRow, int LaunchBoundsMinBlocks>
struct T2RowSplitGemvSchedule {
    static_assert(RowsPerCta > 0, "T2 GEMV requires at least one row per CTA");
    static_assert(WarpsPerRow > 0, "T2 GEMV requires at least one warp per row");
    static_assert(GroupsPerWarpTile > 0 && GroupsPerWarpTile <= 256,
                  "T2 GEMV tiles are bounded by the register scale slots");
    static_assert(LaneMapping == T2GemvLaneMapping::PackedByte4 || GroupsPerWarpTile <= 32,
                  "T2 GEMV word tiles keep one scale per lane");
    static_assert(StaticGroupsPerRow >= 0 && (StaticGroupsPerRow % WarpsPerRow) == 0,
                  "T2 GEMV static ownership must divide groups across row warps");
    static_assert(LaneMapping == T2GemvLaneMapping::PackedByte4 || (GroupsPerWarpTile % 4) == 0,
                  "T2 GEMV word tiles consume four groups per phase");
    static_assert(LaneMapping == T2GemvLaneMapping::PackedWord16 || StaticGroupsPerRow > 0,
                  "T2 GEMV byte mapping is the static-ownership schedule");
    static_assert(LaunchBoundsMinBlocks >= 1, "T2 GEMV launch-bounds occupancy must be positive");

    static constexpr int kRowsPerCta            = RowsPerCta;
    static constexpr int kWarpsPerRow           = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile     = GroupsPerWarpTile;
    static constexpr auto kLaneMapping          = LaneMapping;
    static constexpr int kStaticGroupsPerRow    = StaticGroupsPerRow;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks;

    static constexpr int kCtaWarps = kRowsPerCta * kWarpsPerRow;
    static constexpr int kThreads  = kCtaWarps * 32;
    static constexpr int kGroupsPerWarp =
        kStaticGroupsPerRow > 0 ? kStaticGroupsPerRow / kWarpsPerRow : kGroupsPerWarpTile;
    static constexpr int kCodeVectorsPerTile =
        kGroupsPerWarp * T2RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));

    // Static ownership keeps the warp's scales in registers, one 32-group word per lane slot.
    static constexpr int kScaleSlots = (kGroupsPerWarp + 31) / 32;

    static_assert(kCtaWarps <= 32, "T2 GEMV cannot exceed the CUDA CTA warp limit");
    static_assert(kStaticGroupsPerRow == 0 || kGroupsPerWarp <= 256,
                  "T2 GEMV static ownership is bounded by the register scale slots");
    static_assert(kCtaWarps * kCodeVectorsPerTile * 16 <= 48 * 1024,
                  "T2 GEMV staged codes exceed the static shared-memory budget");
};

// K = 5120, 6144 and 17408: eight rows per CTA, one warp per row owning every group of its row.
// One row per CTA with eight warps (five groups each) ran at a third of the DRAM bandwidth: a
// warp's five dependent group iterations end long before enough bytes are in flight, so the
// kernel was bound by per-CTA latency, not by traffic. A warp per row keeps 40..136 independent
// group loads in the unrolled loop and cuts the CTA count eightfold.
using T2GemvR8W1K5120Schedule =
    T2RowSplitGemvSchedule<8, 1, 40, T2GemvLaneMapping::PackedByte4, 40, 1>;
using T2GemvR8W1K6144Schedule =
    T2RowSplitGemvSchedule<8, 1, 48, T2GemvLaneMapping::PackedByte4, 48, 1>;
using T2GemvR8W1K17408Schedule =
    T2RowSplitGemvSchedule<8, 1, 136, T2GemvLaneMapping::PackedByte4, 136, 1>;
// The vocabulary and draft heads: four rows per CTA, one warp per row, eight-group tiles.
using T2GemvR4W1WordSchedule =
    T2RowSplitGemvSchedule<4, 1, 8, T2GemvLaneMapping::PackedWord16, 0, 1>;

template <class Schedule>
struct T2GemvTileStorage {
    __align__(16) uint4 codes[Schedule::kCtaWarps][Schedule::kCodeVectorsPerTile];
};

// Signed sum of the four activations selected by one code byte: field j of `packed` picks
// activation j (bit 0) and its sign (bit 1).
__device__ __forceinline__ float t2_gemv_byte_partial(std::uint32_t packed, const uint2& xb) {
    const float2 x01  = bf16x2_bits_to_float2(xb.x);
    const float2 x23  = bf16x2_bits_to_float2(xb.y);
    const float xs[4] = {x01.x, x01.y, x23.x, x23.y};
    float partial     = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint32_t field = (packed >> (2 * j)) & 0x3u;
        const float selected      = (field & 0x1u) ? xs[j] : 0.0f;
        partial += (field & 0x2u) ? -selected : selected;
    }
    return partial;
}

// Signed sum of the sixteen activations selected by one code word.
__device__ __forceinline__ float t2_gemv_word_partial(std::uint32_t packed, const uint4& xa,
                                                      const uint4& xb) {
    const std::uint32_t halves[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float partial                 = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const float2 x         = bf16x2_bits_to_float2(halves[pair]);
        const std::uint32_t f0 = (packed >> (4 * pair)) & 0x3u;
        const std::uint32_t f1 = (packed >> (4 * pair + 2)) & 0x3u;
        const float s0         = (f0 & 0x1u) ? x.x : 0.0f;
        const float s1         = (f1 & 0x1u) ? x.y : 0.0f;
        partial += (f0 & 0x2u) ? -s0 : s0;
        partial += (f1 & 0x2u) ? -s1 : s1;
    }
    return partial;
}

template <class Schedule>
__device__ __forceinline__ float
t2_gemv_dot_byte_static(T2GemvTileStorage<Schedule>& tiles, int cta_warp,
                        const __nv_bfloat16* __restrict__ activation,
                        const std::uint8_t* __restrict__ code_row,
                        const std::uint8_t* __restrict__ scale_row, int group_begin, int lane) {
    static_assert(Schedule::kLaneMapping == T2GemvLaneMapping::PackedByte4);
    constexpr int kGroupsPerWarp = Schedule::kGroupsPerWarp;
    constexpr int kVectors       = Schedule::kCodeVectorsPerTile;

    uint4* shared_codes      = tiles.codes[cta_warp];
    const auto* global_codes = reinterpret_cast<const uint4*>(
        code_row + static_cast<std::int64_t>(group_begin) * T2RowSplitStorage::kCodeBytesPerGroup);
    for (int vector = lane; vector < kVectors; vector += 32) {
        shared_codes[vector] = load_ldg<uint4>(&global_codes[vector]);
    }
    constexpr int kScaleSlots = Schedule::kScaleSlots;
    std::uint16_t lane_scale_bits[kScaleSlots];
#pragma unroll
    for (int slot = 0; slot < kScaleSlots; ++slot) {
        const int group       = slot * 32 + lane;
        lane_scale_bits[slot] = 0;
        if (group < kGroupsPerWarp) {
            lane_scale_bits[slot] =
                load_vec<std::uint16_t>(scale_row + static_cast<std::int64_t>(group_begin + group) *
                                                        T2RowSplitStorage::kScaleBytesPerGroup);
        }
    }
    __syncwarp();

    const auto* tile_codes = reinterpret_cast<const std::uint8_t*>(shared_codes);
    float accumulator      = 0.0f;
#pragma unroll
    for (int local_group = 0; local_group < kGroupsPerWarp; ++local_group) {
        const std::uint16_t scale_bits = static_cast<std::uint16_t>(
            __shfl_sync(kFullWarpMask, lane_scale_bits[local_group / 32], local_group % 32));
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const std::uint32_t packed =
            tile_codes[local_group * T2RowSplitStorage::kCodeBytesPerGroup + lane];
        const int k_begin = (group_begin + local_group) * T2RowSplitStorage::kGroupK +
                            lane * T2RowSplitStorage::kCodesPerByte;
        const uint2 xb    = load_vec<uint2>(activation + k_begin);
        accumulator       = fmaf(t2_gemv_byte_partial(packed, xb), scale, accumulator);
    }
    __syncwarp();
    return accumulator;
}

template <class Schedule>
__device__ __forceinline__ float t2_gemv_dot_word_sync(T2GemvTileStorage<Schedule>& tiles,
                                                       int cta_warp,
                                                       const __nv_bfloat16* __restrict__ activation,
                                                       const std::uint8_t* __restrict__ code_row,
                                                       const std::uint8_t* __restrict__ scale_row,
                                                       int group_begin, int group_end, int lane) {
    static_assert(Schedule::kLaneMapping == T2GemvLaneMapping::PackedWord16);
    constexpr int kGroupsPerTile = Schedule::kGroupsPerWarpTile;

    uint4* shared_codes     = tiles.codes[cta_warp];
    const int lane_group    = lane >> 3;
    const int lane_in_group = lane & 7;
    float accumulator       = 0.0f;

    for (int tile_begin = group_begin; tile_begin < group_end; tile_begin += kGroupsPerTile) {
        const int active_groups = min(kGroupsPerTile, group_end - tile_begin);
        const int active_vectors =
            active_groups * T2RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
        const auto* global_codes =
            reinterpret_cast<const uint4*>(code_row + static_cast<std::int64_t>(tile_begin) *
                                                          T2RowSplitStorage::kCodeBytesPerGroup);
        for (int vector = lane; vector < active_vectors; vector += 32) {
            shared_codes[vector] = load_ldg<uint4>(&global_codes[vector]);
        }
        std::uint16_t lane_scale_bits = 0;
        if (lane < active_groups) {
            lane_scale_bits =
                load_vec<std::uint16_t>(scale_row + static_cast<std::int64_t>(tile_begin + lane) *
                                                        T2RowSplitStorage::kScaleBytesPerGroup);
        }
        __syncwarp();

        const auto* words = reinterpret_cast<const std::uint32_t*>(shared_codes);
#pragma unroll
        for (int base = 0; base < kGroupsPerTile; base += 4) {
            const int local_group = base + lane_group;
            // The shuffle is warp-collective; only the use below is predicated.
            const std::uint16_t scale_bits = static_cast<std::uint16_t>(
                __shfl_sync(kFullWarpMask, lane_scale_bits, min(local_group, 31)));
            if (local_group < active_groups) {
                const std::uint32_t packed = words[base * 8 + lane];
                const float scale          = __half2float(__ushort_as_half(scale_bits));
                const int k_begin = (tile_begin + local_group) * T2RowSplitStorage::kGroupK +
                                    lane_in_group * T2RowSplitStorage::kCodesPerWord;
                const uint4 xa    = load_vec<uint4>(activation + k_begin);
                const uint4 xb    = load_vec<uint4>(activation + k_begin + 8);
                accumulator       = fmaf(t2_gemv_word_partial(packed, xa, xb), scale, accumulator);
            }
        }
        __syncwarp();
    }
    return accumulator;
}

template <class Schedule>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kLaunchBoundsMinBlocks) void t2_rowsplit_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                                             const std::uint8_t* __restrict__ codes,
                                                             const std::
                                                                 uint8_t* __restrict__ scales,
                                                             __nv_bfloat16* __restrict__ out,
                                                             std::int32_t rows, std::int32_t k) {
    constexpr int kRowsPerCta  = Schedule::kRowsPerCta;
    constexpr int kWarpsPerRow = Schedule::kWarpsPerRow;

    __shared__ T2GemvTileStorage<Schedule> tiles;
    __shared__ float row_partials[kRowsPerCta][kWarpsPerRow];

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int cta_warp    = static_cast<int>(threadIdx.x) >> 5;
    const int row_in_cta  = cta_warp / kWarpsPerRow;
    const int warp_in_row = cta_warp % kWarpsPerRow;
    const int row         = static_cast<int>(blockIdx.x) * kRowsPerCta + row_in_cta;

    const int groups_per_row = Schedule::kStaticGroupsPerRow > 0 ? Schedule::kStaticGroupsPerRow
                                                                 : k / T2RowSplitStorage::kGroupK;
    int group_begin;
    int group_end;
    if constexpr (Schedule::kStaticGroupsPerRow > 0) {
        group_begin = warp_in_row * Schedule::kGroupsPerWarp;
        group_end   = group_begin + Schedule::kGroupsPerWarp;
    } else {
        group_begin = groups_per_row * warp_in_row / kWarpsPerRow;
        group_end   = groups_per_row * (warp_in_row + 1) / kWarpsPerRow;
    }

    float accumulator = 0.0f;
    if (row < rows) {
        const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * groups_per_row *
                                                    T2RowSplitStorage::kCodeBytesPerGroup;
        const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * groups_per_row *
                                                     T2RowSplitStorage::kScaleBytesPerGroup;
        if constexpr (Schedule::kLaneMapping == T2GemvLaneMapping::PackedByte4) {
            accumulator = t2_gemv_dot_byte_static<Schedule>(tiles, cta_warp, x, code_row, scale_row,
                                                            group_begin, lane);
        } else {
            if (group_begin < group_end) {
                accumulator = t2_gemv_dot_word_sync<Schedule>(
                    tiles, cta_warp, x, code_row, scale_row, group_begin, group_end, lane);
            }
        }
    }

    accumulator = warp_reduce_sum(accumulator);
    if constexpr (kWarpsPerRow == 1) {
        if (lane == 0 && row < rows) { out[row] = __float2bfloat16(accumulator); }
    } else {
        if (lane == 0) { row_partials[row_in_cta][warp_in_row] = accumulator; }
        __syncthreads();
        if (warp_in_row == 0 && lane == 0 && row < rows) {
            float row_accumulator = 0.0f;
#pragma unroll
            for (int warp = 0; warp < kWarpsPerRow; ++warp) {
                row_accumulator += row_partials[row_in_cta][warp];
            }
            out[row] = __float2bfloat16(row_accumulator);
        }
    }
}

} // namespace ninfer::ops::detail
