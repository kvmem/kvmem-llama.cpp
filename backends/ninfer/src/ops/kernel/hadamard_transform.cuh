#pragma once

// ninfer::ops - normalized 1024-point Sylvester Walsh-Hadamard transform with a sign vector, one
// warp per (column, 1024-block). Lane l holds elements 8l + j + 256r in v[r][j] (j = 0..7,
// r = 0..3), so every warp-wide load or store is 512 contiguous bytes: index bits 0..2 and 8..9 are
// butterflied within each lane's registers, bits 3..7 across lanes with shuffles. The same
// butterfly and sign convention as ops/kv_cache/hadamard_d256.cuh: the low element of a pair takes
// x + y, the high element x - y.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kHadamardTransformBlock        = 1024;
inline constexpr int kHadamardTransformLaneElements = kHadamardTransformBlock / kWarpSize;
inline constexpr int kHadamardTransformLaneVectors  = kHadamardTransformLaneElements / 8;
inline constexpr float kHadamardTransformNormalizer = 0.03125f; // 2^-5 = 1 / sqrt(1024)

// Byte offset of lane vector r inside a 1024-element block.
__device__ __forceinline__ int hadamard_lane_offset(int lane, int r) {
    return r * (kWarpSize * 8) + lane * 8;
}

__device__ __forceinline__ void hadamard_unpack8(const uint4& bits, float (&values)[8]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&bits);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float2 f    = __bfloat1622float2(pairs[p]);
        values[2 * p]     = f.x;
        values[2 * p + 1] = f.y;
    }
}

__device__ __forceinline__ uint4 hadamard_pack8(const float (&values)[8]) {
    uint4 bits;
    auto* pairs = reinterpret_cast<__nv_bfloat162*>(&bits);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        pairs[p] = __floats2bfloat162_rn(values[2 * p], values[2 * p + 1]);
    }
    return bits;
}

// The ten butterfly stages over one lane's 32 elements of a 1024-block.
__device__ __forceinline__ void
hadamard_1024_butterfly(float (&v)[kHadamardTransformLaneVectors][8], int lane) {
    // Index bits 0..2 live in j.
#pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if ((j & stride) == 0) {
                    const float low  = v[r][j];
                    const float high = v[r][j + stride];
                    v[r][j]          = __fadd_rn(low, high);
                    v[r][j + stride] = __fsub_rn(low, high);
                }
            }
        }
    }
    // Index bits 3..7 live in the lane id: partner lanes differ in one bit.
#pragma unroll
    for (int stride = 1; stride < kWarpSize; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float peer = __shfl_xor_sync(kFullWarpMask, v[r][j], stride);
                v[r][j] =
                    (lane & stride) == 0 ? __fadd_rn(v[r][j], peer) : __fsub_rn(peer, v[r][j]);
            }
        }
    }
    // Index bits 8..9 live in r.
#pragma unroll
    for (int stride = 1; stride < kHadamardTransformLaneVectors; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
            if ((r & stride) == 0) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float low  = v[r][j];
                    const float high = v[r + stride][j];
                    v[r][j]          = __fadd_rn(low, high);
                    v[r + stride][j] = __fsub_rn(low, high);
                }
            }
        }
    }
}

// Four warps per 1024-block. Warp w holds the elements 256w + 8l + j (the r = w vector of the
// one-warp layout) and runs index bits 0..7 in the order of hadamard_1024_butterfly; bits 8..9 then
// cross the warps through shared memory, where warp w takes elements 64w + 2l + k of every r. Each
// element sees the one-warp transform's operations in the same order, so the result is bitwise
// the same, with a quarter of the per-warp work.
inline constexpr int kHadamardQuarterWarps = 4;

struct alignas(16) HadamardQuarterShared {
    float v[kHadamardQuarterWarps][kWarpSize * 8];
};

// Signs, then index bits 0..7 of one warp's 256 elements.
__device__ __forceinline__ void
hadamard_quarter_low_bits(float (&v)[8], const __nv_bfloat16* __restrict__ signs, int lane) {
    float s[8];
    hadamard_unpack8(load_vec<uint4>(signs + lane * 8), s);
#pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] *= s[j]; }
#pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if ((j & stride) == 0) {
                const float low  = v[j];
                const float high = v[j + stride];
                v[j]             = __fadd_rn(low, high);
                v[j + stride]    = __fsub_rn(low, high);
            }
        }
    }
#pragma unroll
    for (int stride = 1; stride < kWarpSize; stride <<= 1) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float peer = __shfl_xor_sync(kFullWarpMask, v[j], stride);
            v[j]             = (lane & stride) == 0 ? __fadd_rn(v[j], peer) : __fsub_rn(peer, v[j]);
        }
    }
}

// The forward transform of one block held by four warps (v = the warp's 256 BF16-rounded inputs,
// block_signs/out_block at the block's start), `barrier` synchronising the four warps: the signs,
// the ten butterfly stages, the normalisation and the store.
template <class Barrier>
__device__ __forceinline__ void
hadamard_quarter_forward_store(float (&v)[8], const __nv_bfloat16* __restrict__ block_signs,
                               __nv_bfloat16* __restrict__ out_block, HadamardQuarterShared& shared,
                               int warp, int lane, Barrier barrier) {
    hadamard_quarter_low_bits(v, block_signs + warp * kWarpSize * 8, lane);
    *reinterpret_cast<float4*>(&shared.v[warp][lane * 8])     = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(&shared.v[warp][lane * 8 + 4]) = make_float4(v[4], v[5], v[6], v[7]);
    barrier();
    const int e = 64 * warp + 2 * lane;
    float t[kHadamardQuarterWarps][2];
#pragma unroll
    for (int r = 0; r < kHadamardQuarterWarps; ++r) {
        const float2 pair = *reinterpret_cast<const float2*>(&shared.v[r][e]);
        t[r][0]           = pair.x;
        t[r][1]           = pair.y;
    }
    barrier();
#pragma unroll
    for (int stride = 1; stride < kHadamardQuarterWarps; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < kHadamardQuarterWarps; ++r) {
            if ((r & stride) == 0) {
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const float low  = t[r][k];
                    const float high = t[r + stride][k];
                    t[r][k]          = __fadd_rn(low, high);
                    t[r + stride][k] = __fsub_rn(low, high);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kHadamardQuarterWarps; ++r) {
        const __nv_bfloat162 packed =
            __floats2bfloat162_rn(__fmul_rn(t[r][0], kHadamardTransformNormalizer),
                                  __fmul_rn(t[r][1], kHadamardTransformNormalizer));
        *reinterpret_cast<__nv_bfloat162*>(out_block + r * kWarpSize * 8 + e) = packed;
    }
}

template <bool SignsAfter, int WarpsPerCta>
__launch_bounds__(WarpsPerCta* kWarpSize) __global__
    void hadamard_transform_1024_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ signs, __nv_bfloat16* out,
                                        std::int64_t items, std::int32_t blocks_per_column) {
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * WarpsPerCta + warp;
    if (item >= items) { return; }
    const std::int64_t column = item / blocks_per_column;
    const int block           = static_cast<int>(item - column * blocks_per_column);
    const std::int64_t base =
        column * static_cast<std::int64_t>(blocks_per_column) * kHadamardTransformBlock +
        static_cast<std::int64_t>(block) * kHadamardTransformBlock;
    const __nv_bfloat16* block_signs =
        signs + static_cast<std::int64_t>(block) * kHadamardTransformBlock;

    float v[kHadamardTransformLaneVectors][8];
    float s[kHadamardTransformLaneVectors][8];
#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
        const int offset = hadamard_lane_offset(lane, r);
        hadamard_unpack8(load_vec<uint4>(x + base + offset), v[r]);
        hadamard_unpack8(load_vec<uint4>(block_signs + offset), s[r]);
        if constexpr (!SignsAfter) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[r][j] *= s[r][j]; }
        }
    }

    hadamard_1024_butterfly(v, lane);

#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float value = __fmul_rn(v[r][j], kHadamardTransformNormalizer);
            if constexpr (SignsAfter) { value *= s[r][j]; }
            v[r][j] = value;
        }
        store_vec(out + base + hadamard_lane_offset(lane, r), hadamard_pack8(v[r]));
    }
}

} // namespace ninfer::ops
