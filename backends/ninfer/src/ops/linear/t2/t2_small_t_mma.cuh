#pragma once

// T2G128 RowSplit x BF16 small-T tensor-core GEMM (T = 1..16): single-token decode and the
// MTP/DFlash2 verify widths.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One CTA owns 64 rows (four m16 warps) and every column. A stage covers two groups (256 K): the
// 64 rows' 64 code bytes, the T activation rows (swizzled 16-byte chunks, rows above T zeroed once)
// and the rows' scale pairs, all cp.async, several stages deep. A fragments are decoded straight
// from the code words in shared memory (field -> 0 / +-scale as bf16), B fragments come from
// ldmatrix. Codes are {-1, 0, +1}, so the only arithmetic per weight is the fragment pack and the
// kernel is meant to be bound by the weight stream, like the Q4 small-T kernel it replaces on the
// verify step.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int ColumnTiles_, int Stages_>
struct T2SmallTMmaSchedule {
    static constexpr int kRows           = 64;
    static constexpr int kWarps          = 4;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kColumnTiles    = ColumnTiles_;
    static constexpr int kColumns        = kColumnTiles * 8;
    static constexpr int kGroupsPerStage = 2;
    static constexpr int kStageK         = kGroupsPerStage * T2RowSplitStorage::kGroupK;
    static constexpr int kStages         = Stages_;
    static constexpr int kCodeBytesPerRowStage =
        kGroupsPerStage * T2RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kActivationRowBytes = kStageK * 2;
    static constexpr int kSharedBytes =
        kStages * (kRows * kCodeBytesPerRowStage + kColumns * kActivationRowBytes + kRows * 4);

    static_assert(kColumnTiles == 1 || kColumnTiles == 2, "T2 small-T covers one or two n8 tiles");
    static_assert(kStages >= 2 && kStages <= 4, "T2 small-T pipeline depth must fit cp_wait");
    static_assert(kSharedBytes <= 48 * 1024, "T2 small-T stages exceed the static shared budget");
};

// Swizzled byte offset of 16-byte chunk `chunk` (0..31) of activation row `t` in a stage.
__device__ __forceinline__ int t2_small_t_activation_offset(int t, int chunk) {
    return t * 512 + ((chunk ^ (t & 7)) << 4);
}

// A fragment decode through byte permutes. The four codes a lane needs from a 16-code word for
// one k16 step (fields 2t, 2t+1, 2t+8, 2t+9 for t = lane & 3) become the four selector nibbles of
// prmt; two lookup registers per row hold the low and high bytes of {0, +scale, -scale} as bf16
// (byte index = code: 0 -> 0, 1 -> +s, 3 -> -s; the illegal 2 decodes as 0), and two more
// permutes interleave them into the two bf16x2 fragment registers.
struct T2FragmentLut {
    std::uint32_t low;
    std::uint32_t high;
};

__device__ __forceinline__ T2FragmentLut t2_small_t_lut(std::uint32_t scale_bits) {
    const std::uint32_t lo = scale_bits & 0xFFu;
    const std::uint32_t hi = (scale_bits >> 8) & 0xFFu;
    return {(lo << 8) | (lo << 24), (hi << 8) | ((hi | 0x80u) << 24)};
}

__device__ __forceinline__ void t2_small_t_fragment(std::uint32_t word, int tig,
                                                    const T2FragmentLut& lut, unsigned& a_low,
                                                    unsigned& a_high) {
    const std::uint32_t v = word >> (4 * tig);
    const std::uint32_t sel =
        (v & 0x3u) | ((v << 2) & 0x30u) | ((v >> 8) & 0x300u) | ((v >> 6) & 0x3000u);
    const std::uint32_t low  = __byte_perm(lut.low, 0u, sel);
    const std::uint32_t high = __byte_perm(lut.high, 0u, sel);
    a_low                    = __byte_perm(low, high, 0x5140);
    a_high                   = __byte_perm(low, high, 0x7362);
}

template <class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, 4) void t2_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t rows,
    std::int32_t k, std::int32_t cols) {
    constexpr int S             = Schedule::kStages;
    constexpr int NT            = Schedule::kColumnTiles;
    constexpr int NC            = Schedule::kColumns;
    constexpr int kCodeRowBytes = Schedule::kCodeBytesPerRowStage;

    __shared__ __align__(16) std::uint8_t Cs[S][64 * kCodeRowBytes];
    __shared__ __align__(16) std::uint8_t Bs[S][NC * 512];
    __shared__ __align__(16) std::uint32_t Ss[S][64];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * 64;

    const int groups_per_row = k / T2RowSplitStorage::kGroupK;
    const int pairs          = groups_per_row / 2;

    // Activation rows above `cols` are never written by cp.async; zero them once so the padded
    // n8 tile multiplies by zero.
    if (cols < NC) {
        const int pad_vectors = S * (NC - cols) * 32;
        for (int v = tid; v < pad_vectors; v += Schedule::kThreads) {
            const int stage                                         = v / ((NC - cols) * 32);
            const int rem                                           = v - stage * (NC - cols) * 32;
            const int t                                             = cols + rem / 32;
            const int c                                             = rem & 31;
            *reinterpret_cast<uint4*>(&Bs[stage][t * 512 + c * 16]) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncthreads();

    auto issue = [&](int stage, int pair) {
        const int g0 = pair * 2;
        for (int v = tid; v < 64 * (kCodeRowBytes / 16); v += Schedule::kThreads) {
            const int r             = v / (kCodeRowBytes / 16);
            const int c             = v - r * (kCodeRowBytes / 16);
            const std::uint8_t* src = codes +
                                      (static_cast<std::int64_t>(row0 + r) * groups_per_row + g0) *
                                          T2RowSplitStorage::kCodeBytesPerGroup +
                                      c * 16;
            cp_async<16>(&Cs[stage][r * kCodeRowBytes + c * 16], src);
        }
        if (tid < 64) {
            cp_async<4>(&Ss[stage][tid],
                        scales + (static_cast<std::int64_t>(row0 + tid) * groups_per_row + g0) *
                                     T2RowSplitStorage::kScaleBytesPerGroup);
        }
        for (int v = tid; v < cols * 32; v += Schedule::kThreads) {
            const int t = v >> 5;
            const int c = v & 31;
            cp_async<16>(&Bs[stage][t2_small_t_activation_offset(t, c)],
                         x + static_cast<std::int64_t>(t) * k +
                             static_cast<std::int64_t>(pair) * Schedule::kStageK + c * 8);
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        acc[nt][0] = 0.0f;
        acc[nt][1] = 0.0f;
        acc[nt][2] = 0.0f;
        acc[nt][3] = 0.0f;
    }

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < pairs) { issue(stage, stage); }
        cp_commit();
    }

    const int r0 = warp * 16 + gid;
    const int r1 = r0 + 8;
    for (int pair = 0; pair < pairs; ++pair) {
        const int stage = pair % S;
        cp_wait<S - 1>();
        __syncthreads();

        const std::uint32_t scale_pair0 = Ss[stage][r0];
        const std::uint32_t scale_pair1 = Ss[stage][r1];
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            // The stored scale is binary16; the fragment carries it as bf16.
            const T2FragmentLut lut0 =
                t2_small_t_lut(__bfloat16_as_ushort(__float2bfloat16(__half2float(
                    __ushort_as_half(static_cast<unsigned short>(scale_pair0 >> (16 * g)))))));
            const T2FragmentLut lut1 =
                t2_small_t_lut(__bfloat16_as_ushort(__float2bfloat16(__half2float(
                    __ushort_as_half(static_cast<unsigned short>(scale_pair1 >> (16 * g)))))));
#pragma unroll
            for (int step = 0; step < 8; ++step) {
                const int word_offset     = g * T2RowSplitStorage::kCodeBytesPerGroup + step * 4;
                const std::uint32_t word0 = *reinterpret_cast<const std::uint32_t*>(
                    &Cs[stage][r0 * kCodeRowBytes + word_offset]);
                const std::uint32_t word1 = *reinterpret_cast<const std::uint32_t*>(
                    &Cs[stage][r1 * kCodeRowBytes + word_offset]);
                unsigned a0;
                unsigned a1;
                unsigned a2;
                unsigned a3;
                t2_small_t_fragment(word0, tig, lut0, a0, a2);
                t2_small_t_fragment(word1, tig, lut1, a1, a3);
                const int kstep = g * 8 + step;
                const int chunk = kstep * 2 + ((lane >> 3) & 1);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int t = nt * 8 + (lane & 7);
                    unsigned b0;
                    unsigned b1;
                    ldmatrix_x2(b0, b1,
                                smem_addr(&Bs[stage][t2_small_t_activation_offset(t, chunk)]));
                    mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3, b0,
                             b1);
                }
            }
        }
        __syncthreads();
        const int next = pair + S;
        if (next < pairs) { issue(stage, next); }
        cp_commit();
    }

#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int col = nt * 8 + tig * 2;
        if (col < cols) {
            out[static_cast<std::int64_t>(col) * rows + row0 + r0] =
                __float2bfloat16_rn(acc[nt][0]);
            out[static_cast<std::int64_t>(col) * rows + row0 + r1] =
                __float2bfloat16_rn(acc[nt][2]);
        }
        if (col + 1 < cols) {
            out[static_cast<std::int64_t>(col + 1) * rows + row0 + r0] =
                __float2bfloat16_rn(acc[nt][1]);
            out[static_cast<std::int64_t>(col + 1) * rows + row0 + r1] =
                __float2bfloat16_rn(acc[nt][3]);
        }
    }
}

} // namespace ninfer::ops::detail
