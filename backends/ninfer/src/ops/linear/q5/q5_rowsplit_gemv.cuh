#pragma once

// Shared Q5 row-split decode (T=1) GEMV core.
//
// Memory-bound design for the narrow Qwen3.6 decode shapes (N ~= 5-7K):
//   - A block owns kRowsPerBlock output rows; one warp computes one full row.
//   - Each warp streams its row's 16-group weight tiles through shared memory with
//     a cp.async pipeline kStages buffers deep: tiles j+1 .. j+kStages-1 are in
//     flight while tile j is unpacked/accumulated, so DRAM latency is hidden
//     without relying on very high occupancy (these matrices cannot fill the GPU
//     with one warp per row otherwise).
//   - The activation x (full K vector, reused by every row) is optionally staged
//     into shared once per block (kStageX); disable it when K is so large that x
//     would not fit alongside the weight buffers (e.g. mlp_down, K=17408).
//   - Each warp does a warp-shuffle reduction and writes one bf16 output; there is
//     no block barrier on the hot path.
// A tile is 16 groups: 512 nibble bytes (32 uint4, one per lane), 128 high bytes
// (8 uint4), 32 scale bytes (2 uint4). All *global* weight reads are fully coalesced
// 128-bit loads.
//
// That does NOT make the kernel DRAM-bound, which this comment used to claim. `ncu`
// on the 27B decode measures DRAM throughput at 50.10% against L1/TEX at 86.60% and
// Mem Pipes Busy at 81.04%: the limit is memory-pipe instruction issue, and it lives
// in the shared-memory consume loop rather than in the staging. See the note on
// q5_gemv_consume_tile for what that cost and how it was cut.

#include "core/pdl.cuh"
#include "ops/common/math.cuh" // bf16x2_bits_to_float2, for the widened consume loop
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Issue the cp.async copies for one 16-group tile into a shared buffer slot, then
// commit them as one pipeline group. Every lane commits (even lanes that issue
// fewer copies) so a uniform pipe_wait works warp-wide.
__device__ __forceinline__ void
q5_gemv_issue_tile(uint4* __restrict__ s_nib, uint4* __restrict__ s_hi, uint4* __restrict__ s_sc,
                   const std::uint8_t* __restrict__ code_row,
                   const std::uint8_t* __restrict__ high_row,
                   const std::uint8_t* __restrict__ scale_row, int tile, int lane) {
    constexpr int kGroupsPerTile = 16;
    const int g0                 = tile * kGroupsPerTile;
    pipe_copy<16>(&s_nib[lane], reinterpret_cast<const uint4*>(code_row + g0 * 32) + lane);
    if (lane < 8) {
        pipe_copy<16>(&s_hi[lane], reinterpret_cast<const uint4*>(high_row + g0 * 8) + lane);
    }
    if (lane < 2) {
        pipe_copy<16>(&s_sc[lane], reinterpret_cast<const uint4*>(scale_row + g0 * 2) + lane);
    }
    pipe_commit();
}

// Unpack + accumulate one staged 16-group tile. x is read from x2 (shared or global).
__device__ __forceinline__ float q5_gemv_consume_tile(const __nv_bfloat162* __restrict__ x2,
                                                      const uint4* __restrict__ s_nib,
                                                      const uint4* __restrict__ s_hi,
                                                      const uint4* __restrict__ s_sc, int tile,
                                                      int lane, float acc) {
    constexpr int kGroupK              = 64;
    constexpr int kGroupsPerTile       = 16;
    constexpr int kNibbleBytesPerGroup = 32;
    constexpr int kHighBytesPerGroup   = 8;
    // Eight weights per lane per step, not two, because this loop -- not the staging above -- is
    // what the kernel is actually limited by.
    //
    // `ncu` on the 27B decode: DRAM throughput 50.10%, but **L1/TEX 86.60% and Mem Pipes Busy
    // 81.04%**, with ncu's own note that the workload is over 80% of a pipe and work should be
    // shifted off it. So the kernel is issue-bound on the memory pipe, not DRAM-bound (the file
    // header used to claim the latter; it is wrong and now says so).
    //
    // The old shape read ONE code byte per lane per group and so spent four memory-pipe
    // instructions to produce two weights: a 1-byte `tn`, a 1-byte `th`, a 2-byte broadcast `tsc`,
    // and a 4-byte `x`. Sixteen groups per tile is 64 instructions on the consume side against
    // three on the staging side, so consume outweighed staging roughly 20 to 1.
    //
    // Now each lane takes a 4-byte code word -- 8 nibbles, 8 weights -- so eight lanes cover a
    // group's 32 code bytes and 32 lanes cover FOUR groups per step. Per step: one 4-byte `tn`,
    // one 1-byte `th`, one 2-byte `tsc`, one 16-byte `x`. Four instructions per four groups
    // against sixteen before: **a 4x cut in memory-pipe instructions for identical arithmetic.**
    //
    // The packing convention is unchanged and is this file's own, not another Op's: code byte b
    // holds the weights at 2b (low nibble) and 2b+1 (high nibble), and high-plane byte h bit j
    // carries bit 4 of the weight at 8h+j. Verified on the host that the (sub, pos) split covers
    // all 1,024 weights of a tile exactly once.
    //
    // Bank behaviour is why this is free rather than a trade: the code index works out to
    // `step * 32 + lane`, so lanes read consecutive 32-bit words and the access is
    // conflict-free. `x` is likewise 16 consecutive bytes per lane.
    const auto* tn32                   = reinterpret_cast<const std::uint32_t*>(s_nib);
    const auto* th                     = reinterpret_cast<const std::uint8_t*>(s_hi);
    const auto* tsc                    = reinterpret_cast<const std::uint16_t*>(s_sc);
    const int g0                       = tile * kGroupsPerTile;
    constexpr int kWeightsPerLane      = 8;
    constexpr int kLanesPerGroup       = kGroupK / kWeightsPerLane; // 8
    constexpr int kGroupsPerStep       = 32 / kLanesPerGroup;       // 4
    const int sub                      = lane / kLanesPerGroup;     // group within the step
    const int pos                      = lane % kLanesPerGroup;     // 8-weight span in the group
#pragma unroll
    for (int step = 0; step < kGroupsPerTile / kGroupsPerStep; ++step) {
        const int tg             = step * kGroupsPerStep + sub;
        const float scale        = __half2float(__ushort_as_half(tsc[tg]));
        const std::uint32_t word = tn32[tg * (kNibbleBytesPerGroup / 4) + pos];
        const std::uint32_t high = th[tg * kHighBytesPerGroup + pos];

        const int k0    = (g0 + tg) * kGroupK + pos * kWeightsPerLane;
        const uint4 xv  = load_vec<uint4>(reinterpret_cast<const uint4*>(x2 + (k0 >> 1)));
        const float2 f0 = bf16x2_bits_to_float2(xv.x);
        const float2 f1 = bf16x2_bits_to_float2(xv.y);
        const float2 f2 = bf16x2_bits_to_float2(xv.z);
        const float2 f3 = bf16x2_bits_to_float2(xv.w);
        const float xs[kWeightsPerLane]{f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};

        // Dequantize two weights at a time through the half2 bit-trick, and apply the group scale
        // ONCE for the whole group rather than once per weight.
        //
        // Both halves are about the arithmetic, because after the widening above this kernel is no
        // longer memory-bound: `ncu` puts it at SM throughput 73.31% against DRAM 55.83%, so the
        // per-weight unpacking is what is left.
        //
        // The trick is `Q5SimtDecodeAtom::decode_eight`'s, already used by
        // q5_rowsplit_gemm_simt_split4_kernel, and it is exact rather than approximate. half
        // 0x6400 is 1024.0, whose mantissa LSB at that exponent is exactly 1.0, so OR-ing a nibble
        // into the mantissa *adds* it; inverting the high bit and subtracting 1040.0 then yields
        // `nibble - 16 * high_bit`, which is precisely what `sign_extend<5>(nibble | high << 4)`
        // computes. Verified on the host for all 32 five-bit codes. It replaces a per-weight
        // sign-extend and int-to-float convert with one `hsub2` and one `__half22float2` per pair.
        //
        // Hoisting the scale is exact too -- `sum_j(q_j * x_j) * scale` against
        // `sum_j((q_j * scale) * x_j)` -- and removes seven of every eight multiplies. It rounds
        // *less*, not more, since the scale is applied to one accumulated value instead of eight
        // products.
        //
        // Weight ordering is unchanged: bit-pair `pair` carries weights `pair` and `pair + 4`,
        // which under this file's convention are the same k as before.
        const std::uint32_t high_inv = high ^ 0xffu;
        const __half2 bias           = __half2half2(__ushort_as_half(0x6410)); // 1040.0
        float group_acc              = 0.0f;
#pragma unroll
        for (int pair = 0; pair < kWeightsPerLane / 2; ++pair) {
            std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((high_inv >> pair) & 1u) << 4) | (((high_inv >> (pair + 4)) & 1u) << 20);
            const float2 q = __half22float2(__hsub2(half2_from_bits(bits), bias));
            group_acc      = fmaf(q.x, xs[pair], group_acc);
            group_acc      = fmaf(q.y, xs[pair + 4], group_acc);
        }
        acc = fmaf(group_acc, scale, acc);
    }
    return acc;
}

// kN  : output rows, kK : reduction dim (multiple of 1024).
// kRowsPerBlock : rows (= warps) per block.
// kStages : cp.async pipeline depth (shared buffers; >=2 to overlap).
// kStageX : stage the activation vector into shared (false when x is too large).
struct Q5GemvStoreEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, int row,
                                               float value) const {
        if constexpr (SplitOutput) {
            if (row < SplitRow) {
                out[row] = __float2bfloat16_rn(value);
            } else {
                out_tail[row - SplitRow] = __float2bfloat16_rn(value);
            }
        } else {
            out[row] = __float2bfloat16_rn(value);
        }
    }
};

template <int kN, int kK, int kRowsPerBlock, int kStages, bool kStageX, bool kSplitOutput = false,
          int kSplitRow = 0, class Epilogue = Q5GemvStoreEpilogue, bool TriggerPdl = false,
          bool JoinPdl = false>
__global__ void
q5_rowsplit_gemv_kernel(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                        const std::uint8_t* __restrict__ high_bits,
                        const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
                        __nv_bfloat16* __restrict__ out_tail, Epilogue epilogue = {}) {
    constexpr int kGroupK              = 64;
    constexpr int kGroups              = kK / kGroupK;
    constexpr int kGroupsPerTile       = 16;
    constexpr int kTiles               = kGroups / kGroupsPerTile;
    constexpr int kNibbleBytesPerGroup = 32;
    constexpr int kHighBytesPerGroup   = 8;
    constexpr int kXVecs               = kK / 8; // x as uint4 (8 bf16 each)
    constexpr int kPrefetch            = kStages - 1;
    static_assert(kGroups % kGroupsPerTile == 0, "K must be a multiple of 16 groups (1024)");
    static_assert(kN % kRowsPerBlock == 0, "N must be a multiple of kRowsPerBlock");
    static_assert(kStages >= 2, "need at least double buffering");
    static_assert(!kSplitOutput || (kSplitRow > 0 && kSplitRow < kN),
                  "split-output Q5 GEMV requires an interior compile-time seam");

    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }

    // __align__(16) so the uint4 staging below is well-defined by construction.
    __shared__ __align__(16) __nv_bfloat16 x_sh[kStageX ? kK : 1];
    __shared__ uint4 s_nib[kRowsPerBlock][kStages][32];
    __shared__ uint4 s_hi[kRowsPerBlock][kStages][8];
    __shared__ uint4 s_sc[kRowsPerBlock][kStages][2];

    if constexpr (kStageX) {
        // x must be 16-byte aligned for the uint4 staging (guaranteed: activations
        // come from the 256-byte-aligned workspace arena).
        auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
        const auto* x_g = reinterpret_cast<const uint4*>(x);
        for (int i = static_cast<int>(threadIdx.x); i < kXVecs; i += static_cast<int>(blockDim.x)) {
            x_sh_v[i] = x_g[i];
        }
        __syncthreads();
    }

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

    const std::uint8_t* code_row =
        codes + static_cast<std::int64_t>(row) * kGroups * kNibbleBytesPerGroup;
    const std::uint8_t* high_row =
        high_bits + static_cast<std::int64_t>(row) * kGroups * kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kGroups * 2;
    const auto* x2                = kStageX ? reinterpret_cast<const __nv_bfloat162*>(x_sh)
                                            : reinterpret_cast<const __nv_bfloat162*>(x);

    // Prime up to kPrefetch tiles; empty commits keep the group count uniform so the
    // constant pipe_wait<kPrefetch>() in the loop is always valid.
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < kTiles) {
            q5_gemv_issue_tile(s_nib[warp][p], s_hi[warp][p], s_sc[warp][p], code_row, high_row,
                               scale_row, p, lane);
        } else {
            pipe_commit();
        }
    }

    float acc = 0.0f;
#pragma unroll 1
    for (int tile = 0; tile < kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < kTiles) {
            const int buf = fetch % kStages;
            q5_gemv_issue_tile(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row,
                               high_row, scale_row, fetch, lane);
        } else {
            pipe_commit();
        }
        pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf = tile % kStages;
        acc = q5_gemv_consume_tile(x2, s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], tile,
                                   lane, acc);
        __syncwarp();
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        epilogue.template operator()<kSplitOutput, kSplitRow>(out, out_tail, row, acc);
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

// One block per kRowsPerBlock rows; kRowsPerBlock warps per block.
template <int kN, int kK, int kRowsPerBlock, int kStages = 2, bool kStageX = true>
inline void q5_rowsplit_gemv_launch_kernel(const __nv_bfloat16* x, const std::uint8_t* codes,
                                           const std::uint8_t* high_bits,
                                           const std::uint8_t* scales, __nv_bfloat16* out,
                                           cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const int grid              = kN / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kN, kK, kRowsPerBlock, kStages, kStageX>
        <<<grid, kBlockThreads, 0, stream>>>(x, codes, high_bits, scales, out, nullptr);
}

} // namespace ninfer::ops::detail
