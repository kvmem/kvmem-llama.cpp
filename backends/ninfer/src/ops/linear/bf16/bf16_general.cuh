#pragma once

// General-purpose BF16 x BF16 GEMM fallback for shapes the specialised bf16 linear path does not
// tile (for example a full-precision vocab projection kept as BF16 by a QAT checkpoint). Computes
// out[T, N] = x[T, K] * weight[N, K]^T with an fp32 accumulation and a BF16 store, matching the
// contiguous token-major layout of the specialised kernels. Tiling is fixed and the extents are
// runtime, so any (N, K, T) is handled without a per-shape instantiation.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
constexpr std::int32_t kTileN         = 32;
constexpr std::int32_t kTileT         = 32;
constexpr std::int32_t kTileK         = 32;
constexpr std::int32_t kThreads       = 256;
constexpr std::int32_t kThreadCols    = kThreads / 16;      // 16
constexpr std::int32_t kRowsPerThread = kTileN / 16;        // 2
constexpr std::int32_t kColsPerThread = kTileT / kThreadCols;  // 2

__global__ void bf16_general_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ weight,
                                         __nv_bfloat16* __restrict__ out, std::int32_t N,
                                         std::int32_t K, std::int32_t T) {
    __shared__ __nv_bfloat16 ws[kTileN][kTileK + 1];
    __shared__ __nv_bfloat16 wx[kTileT][kTileK + 1];

    const std::int32_t out_row0 = static_cast<std::int32_t>(blockIdx.x) * kTileN;
    const std::int32_t token0   = static_cast<std::int32_t>(blockIdx.y) * kTileT;
    const std::int32_t tid      = static_cast<std::int32_t>(threadIdx.x);

    const std::int32_t trow = tid / kThreadCols;  // 0..15
    const std::int32_t tcol = tid % kThreadCols;  // 0..15

    float accum[kRowsPerThread][kColsPerThread] = {};

    for (std::int32_t k0 = 0; k0 < K; k0 += kTileK) {
        for (std::int32_t i = tid; i < kTileN * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = out_row0 + r;
            const std::int32_t gc = k0 + c;
            ws[r][c + 1] = (gr < N && gc < K)
                               ? weight[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        for (std::int32_t i = tid; i < kTileT * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = token0 + r;
            const std::int32_t gc = k0 + c;
            wx[r][c + 1] = (gr < T && gc < K)
                               ? x[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
            const std::int32_t row = trow * kRowsPerThread + rr;
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = tcol * kColsPerThread + cc;
                float sum = accum[rr][cc];
#pragma unroll
                for (std::int32_t kk = 0; kk < kTileK; ++kk) {
                    sum = fmaf(static_cast<float>(wx[col][kk + 1]),
                               static_cast<float>(ws[row][kk + 1]), sum);
                }
                accum[rr][cc] = sum;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
        const std::int32_t row = out_row0 + trow * kRowsPerThread + rr;
        if (row < N) {
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = token0 + tcol * kColsPerThread + cc;
                if (col < T) {
                    out[static_cast<std::int64_t>(col) * N + row] =
                        __float2bfloat16_rn(accum[rr][cc]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
