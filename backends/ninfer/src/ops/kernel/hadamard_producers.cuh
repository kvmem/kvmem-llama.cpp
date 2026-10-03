#pragma once

// ninfer::ops - producers of a Hadamard-rotated projection input that apply the forward transform
// on the way out. Each kernel computes its values exactly as the unfused op does, rounds them to
// BF16 exactly as that op stores them, and hands them to hadamard_quarter_forward_store, so the
// fused output is bit-identical to the op followed by hadamard_transform.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/hadamard_transform.cuh"
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kRmsnormHadamardWidth = 5120;

inline constexpr int kRmsnormHadamardThreads = 5 * kHadamardQuarterWarps * kWarpSize;

// rmsnorm_cta_bf16x2_kernel<Epilogue, 256, 10, true, 5120> (the route rmsnorm takes for every
// 5120-wide row) on the first 256 threads, then the forward transform of the row by five groups of
// four warps: the normalised row is staged in shared memory as BF16 and group b transforms block
// b. The other warps enter the reduction with a zero so it runs in the 256-thread op's order.
template <RmsEpilogue Epilogue>
__launch_bounds__(kRmsnormHadamardThreads) __global__
    void rmsnorm_hadamard_5120_kernel(const __nv_bfloat162* __restrict__ x,
                                      const __nv_bfloat162* __restrict__ weight,
                                      const __nv_bfloat16* __restrict__ signs,
                                      __nv_bfloat16* __restrict__ out, std::int64_t rows,
                                      float eps) {
    constexpr int kBlock          = 256;
    constexpr int kWidth          = kRmsnormHadamardWidth;
    constexpr int kPairs          = kWidth / 2;
    constexpr int kPairsPerThread = kPairs / kBlock;
    const std::int64_t row        = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const bool reducer = threadIdx.x < kBlock;

    const std::int64_t row_base = row * static_cast<std::int64_t>(kPairs);
    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    float sum = 0.0f;
    if (reducer) {
#pragma unroll
        for (int k = 0; k < kPairsPerThread; ++k) {
            const int pair  = static_cast<int>(threadIdx.x) + k * kBlock;
            values[k]       = x[row_base + pair];
            weights[k]      = weight[pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[kRmsnormHadamardThreads / kWarpSize];
    __shared__ float inv_shared;
    __shared__ alignas(16) __nv_bfloat162 staged[kPairs];
    __shared__ HadamardQuarterShared transform[kWidth / kHadamardTransformBlock];
    const float block_sum = block_reduce_sum<kBlock>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(kWidth) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    if (reducer) {
#pragma unroll
        for (int k = 0; k < kPairsPerThread; ++k) {
            const int pair  = static_cast<int>(threadIdx.x) + k * kBlock;
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(weights[k]);
            staged[pair] = __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, 0.0f),
                                                 rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, 0.0f));
        }
    }
    __syncthreads();

    const int lane       = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp       = static_cast<int>(threadIdx.x) / kWarpSize;
    const int group      = warp / kHadamardQuarterWarps;
    const int quarter    = warp % kHadamardQuarterWarps;
    const int block_base = group * kHadamardTransformBlock;
    float v[8];
    hadamard_unpack8(
        *reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(staged) +
                                        block_base + quarter * kWarpSize * 8 + lane * 8),
        v);
    hadamard_quarter_forward_store(v, signs + block_base, out + row * kWidth + block_base,
                                   transform[group], quarter, lane, [] { __syncthreads(); });
}

inline constexpr int kGatedRmsnormHadamardHeadDim    = 128;
inline constexpr int kGatedRmsnormHadamardRowsPerCta = 16;

// rmsnorm_warp_bf16x2_kernel<Gated, 512, *> over 128-wide head rows (one warp per row), then the
// forward transform over each token's heads laid end to end. Sixteen rows per CTA are two
// 1024-blocks of one token, so the launcher requires the per-token head count to be a multiple of
// sixteen.
__launch_bounds__(kGatedRmsnormHadamardRowsPerCta* kWarpSize) __global__
    void gated_rmsnorm_hadamard_d128_kernel(const __nv_bfloat162* __restrict__ x,
                                            const __nv_bfloat162* __restrict__ weight,
                                            const __nv_bfloat162* __restrict__ z,
                                            const __nv_bfloat16* __restrict__ signs,
                                            __nv_bfloat16* __restrict__ out,
                                            std::int32_t heads_per_column, std::int64_t rows,
                                            float eps) {
    constexpr int kD               = kGatedRmsnormHadamardHeadDim;
    constexpr int kPairs           = kD / 2;
    constexpr int kMaxPairsPerLane = 4;
    constexpr int kRows            = kGatedRmsnormHadamardRowsPerCta;
    const int lane                 = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t first_row   = static_cast<std::int64_t>(blockIdx.x) * kRows;
    const std::int64_t row         = first_row + warp;

    __shared__ alignas(16) __nv_bfloat162 staged[kRows * kPairs];
    if (row < rows) {
        const std::int64_t row_base = row * kPairs;
        __nv_bfloat162 values[kMaxPairsPerLane];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kMaxPairsPerLane; ++k) {
            const int pair = lane + k * kWarpSize;
            if (pair < kPairs) {
                values[k]       = x[row_base + pair];
                const float2 xf = __bfloat1622float2(values[k]);
                sum += xf.x * xf.x + xf.y * xf.y;
            }
        }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(kD) + eps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int k = 0; k < kMaxPairsPerLane; ++k) {
            const int pair = lane + k * kWarpSize;
            if (pair < kPairs) {
                const float2 xf              = __bfloat1622float2(values[k]);
                const float2 wf              = __bfloat1622float2(weight[pair]);
                const float2 zf              = __bfloat1622float2(z[row_base + pair]);
                staged[warp * kPairs + pair] = __floats2bfloat162_rn(
                    rmsnorm_epilogue<RmsEpilogue::Gated>(xf.x, inv, wf.x, zf.x),
                    rmsnorm_epilogue<RmsEpilogue::Gated>(xf.y, inv, wf.y, zf.y));
            }
        }
    }
    __syncthreads();

    // Two 1024-blocks of one token, each transformed by four of the sixteen warps; every warp
    // enters the quarter transform's barriers.
    constexpr int kBlocksPerCta = kRows * kD / kHadamardTransformBlock;
    __shared__ HadamardQuarterShared transform[kBlocksPerCta];
    const int group   = warp / kHadamardQuarterWarps;
    const int quarter = warp % kHadamardQuarterWarps;
    const bool active =
        group < kBlocksPerCta && first_row + (group + 1) * (kHadamardTransformBlock / kD) <= rows;
    const std::int64_t column = first_row / heads_per_column;
    const int head            = static_cast<int>(first_row - column * heads_per_column);
    const std::int64_t width  = static_cast<std::int64_t>(heads_per_column) * kD;
    const std::int64_t offset =
        static_cast<std::int64_t>(head) * kD + (active ? group : 0) * kHadamardTransformBlock;
    float v[8];
    if (active) {
        hadamard_unpack8(*reinterpret_cast<const uint4*>(
                             reinterpret_cast<const __nv_bfloat16*>(staged) +
                             group * kHadamardTransformBlock + quarter * kWarpSize * 8 + lane * 8),
                         v);
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = 0.0f; }
    }
    if (active) {
        hadamard_quarter_forward_store(v, signs + offset, out + column * width + offset,
                                       transform[group], quarter, lane, [] { __syncthreads(); });
    } else {
        __syncthreads();
        __syncthreads();
    }
}

// out[b*1024+i, t] = 2^-5 * sum_j H[i][j] * signs[b*1024+j] * silu(gate[b*1024+j, t]) *
// up[b*1024+j, t], with gate and up the two halves of one [2*I, T] plane: the down-projection
// input of a Hadamard-rotated checkpoint in one pass over the SwiGLU plane. One CTA of four warps
// per (column, 1024-block) item; warp w computes the SwiGLU values of its quarter (rounded to BF16
// as the two-op composition stores them) and the quarter transform writes the block.
__launch_bounds__(kHadamardQuarterWarps* kWarpSize) __global__
    void silu_mul_hadamard_quarter_kernel(const __nv_bfloat16* __restrict__ plane,
                                          const __nv_bfloat16* __restrict__ signs,
                                          __nv_bfloat16* __restrict__ out, std::int64_t items,
                                          std::int32_t blocks_per_column) {
    __shared__ HadamardQuarterShared shared;
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x);
    if (item >= items) { return; }
    const std::int64_t column = item / blocks_per_column;
    const int block           = static_cast<int>(item - column * blocks_per_column);
    const std::int64_t width =
        static_cast<std::int64_t>(blocks_per_column) * kHadamardTransformBlock;
    const std::int64_t block_base = static_cast<std::int64_t>(block) * kHadamardTransformBlock;
    const int offset              = warp * kWarpSize * 8 + lane * 8;
    const __nv_bfloat16* gate     = plane + column * 2 * width + block_base;

    float g[8];
    float u[8];
    hadamard_unpack8(load_vec<uint4>(gate + offset), g);
    hadamard_unpack8(load_vec<uint4>(gate + width + offset), u);
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(__float2bfloat16_rn(silu(g[j]) * u[j])); }
    hadamard_quarter_forward_store(v, signs + block_base, out + column * width + block_base, shared,
                                   warp, lane, [] { __syncthreads(); });
}

// sigmoid_mul's BF16 product x * sigmoid(gate), then the forward transform: one CTA of four warps
// per (column, 1024-block). out may alias x exactly; every warp reads its quarter before the first
// barrier of the transform, and the block is written after it.
__launch_bounds__(kHadamardQuarterWarps* kWarpSize) __global__
    void sigmoid_mul_hadamard_quarter_kernel(const __nv_bfloat16* __restrict__ gate,
                                             const __nv_bfloat16* x,
                                             const __nv_bfloat16* __restrict__ signs,
                                             __nv_bfloat16* out, std::int64_t items,
                                             std::int32_t blocks_per_column) {
    __shared__ HadamardQuarterShared shared;
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x);
    if (item >= items) { return; }
    const std::int64_t column = item / blocks_per_column;
    const int block           = static_cast<int>(item - column * blocks_per_column);
    const std::int64_t width =
        static_cast<std::int64_t>(blocks_per_column) * kHadamardTransformBlock;
    const std::int64_t base =
        column * width + static_cast<std::int64_t>(block) * kHadamardTransformBlock;
    const int offset = warp * kWarpSize * 8 + lane * 8;

    float g[8];
    float a[8];
    hadamard_unpack8(load_vec<uint4>(gate + base + offset), g);
    hadamard_unpack8(*reinterpret_cast<const uint4*>(x + base + offset), a);
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = __bfloat162float(__float2bfloat16_rn(a[j] * sigmoid(g[j])));
    }
    const std::int64_t block_base = static_cast<std::int64_t>(block) * kHadamardTransformBlock;
    hadamard_quarter_forward_store(v, signs + block_base, out + base, shared, warp, lane,
                                   [] { __syncthreads(); });
}

// Rows of a Hadamard-rotated T2G128 table (a token embedding stored in the rotated basis) restored
// to the primal basis on the way out: out[b*1024+i, t] = 2^-5 * signs[b*1024+i] * sum_j H[i][j] *
// z[b*1024+j] with z = code * scale of row ids[t], one warp per (token, 1024-block). The codes are
// two's complement over two bits, lowest element in the lowest bits, one FP16 scale per 128
// columns; the dequantised values enter the butterfly unrounded.
template <int WarpsPerCta>
__launch_bounds__(WarpsPerCta* kWarpSize) __global__
    void embed_gather_t2_hadamard_kernel(const std::int32_t* __restrict__ ids,
                                         const std::uint8_t* __restrict__ codes,
                                         const __half* __restrict__ scales,
                                         const __nv_bfloat16* __restrict__ signs,
                                         __nv_bfloat16* __restrict__ out, std::int64_t items,
                                         std::int32_t width) {
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * WarpsPerCta + warp;
    if (item >= items) { return; }
    const std::int32_t blocks_per_row = width / kHadamardTransformBlock;
    const std::int64_t token          = item / blocks_per_row;
    const int block                   = static_cast<int>(item - token * blocks_per_row);
    const std::int64_t row            = ids[token];
    const std::uint8_t* row_codes     = codes + row * (width / 4);
    const __half* row_scales          = scales + row * (width / 128);
    const int block_base              = block * kHadamardTransformBlock;

    float v[kHadamardTransformLaneVectors][8];
#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
        const int first     = block_base + hadamard_lane_offset(lane, r);
        const unsigned bits = *reinterpret_cast<const std::uint16_t*>(row_codes + first / 4);
        const float scale   = __half2float(row_scales[first / 128]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned code = (bits >> (2 * j)) & 3U;
            v[r][j]             = code == 1U ? scale : (code == 3U ? -scale : 0.0f);
        }
    }

    hadamard_1024_butterfly(v, lane);

    const __nv_bfloat16* block_signs = signs + block_base;
    __nv_bfloat16* out_block         = out + token * width + block_base;
#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
        float s[8];
        hadamard_unpack8(load_vec<uint4>(block_signs + hadamard_lane_offset(lane, r)), s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[r][j] = __fmul_rn(v[r][j], kHadamardTransformNormalizer) * s[j];
        }
        store_vec(out_block + hadamard_lane_offset(lane, r), hadamard_pack8(v[r]));
    }
}

} // namespace ninfer::ops
