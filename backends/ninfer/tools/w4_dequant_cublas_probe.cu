// Is "dequantise to int8, then call cuBLAS" worth it for prefill?
//
// cuBLAS's int8 GEMM runs this fork's prefill shapes 1.8-2.5x faster than our W4A8 kernels
// (tools/int8_gemm_reference_probe.cu), so the obvious question is whether we can simply hand the
// work to it. cuBLAS cannot read 4-bit codes with a per-group scale, so the weights would have to
// be materialised as int8 first -- and they cannot be materialised once and kept, because int8
// weights for this model are about 24 GB against the card's 24 GB. That leaves materialising one
// layer at a time into scratch, per prefill chunk, which is what this measures:
//
//   dequantise W4 (group-64 FP16 scales) -> int8 with one scale per row, into a workspace
//   cuBLAS int8 GEMM over that workspace
//
// against our own kernel's measured time at the same shape. The dequantised form is also coarser
// than what we compute today -- one scale per row rather than per 64 -- so a win here would still
// have to be paid for in quality; this is only about whether the speed is there at all.
//
// Build:
//   nvcc -O3 -arch=sm_86 -allow-unsupported-compiler w4_dequant_cublas_probe.cu -lcublas \
//        -o w4_dequant_cublas_probe

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

#define CHECK_BLAS(x)                                                                              \
    do {                                                                                           \
        cublasStatus_t st = (x);                                                                   \
        if (st != CUBLAS_STATUS_SUCCESS) {                                                         \
            printf("cuBLAS error %d at line %d\n", static_cast<int>(st), __LINE__);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// One row per block: read the row's 4-bit codes and its group scales, and write int8 codes scaled
// to the row's own maximum. Vectorised at 16 bytes in, 32 bytes out.
__global__ void dequantise_rows_to_int8(const unsigned char* __restrict__ codes,
                                        const __half* __restrict__ scales, int k,
                                        signed char* __restrict__ out, float* __restrict__ row_scale) {
    const int row     = blockIdx.x;
    const int groups  = k / 64;
    const unsigned char* src = codes + static_cast<size_t>(row) * (k / 2);
    const __half* src_scale  = scales + static_cast<size_t>(row) * groups;

    // Pass one: the row's absolute maximum, which sets the single scale int8 has to live with.
    float amax = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        amax = fmaxf(amax, fabsf(__half2float(src_scale[g])) * 8.0f);
    }
    __shared__ float shared_max[32];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int off = 16; off > 0; off >>= 1) { amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off)); }
    if (lane == 0) { shared_max[warp] = amax; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.0f;
        for (int i = 0; i < (blockDim.x >> 5); ++i) { m = fmaxf(m, shared_max[i]); }
        shared_max[0] = fmaxf(m, 1e-20f);
        row_scale[row] = shared_max[0] / 127.0f;
    }
    __syncthreads();
    const float inv = 127.0f / shared_max[0];

    signed char* dst = out + static_cast<size_t>(row) * k;
    // 16 packed bytes in, 32 codes out, and 32 codes never straddle a group of 64.
    for (int i = threadIdx.x; i < k / 32; i += blockDim.x) {
        const uint4 packed = *reinterpret_cast<const uint4*>(src + i * 16);
        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
        const int g             = (i * 32) / 64;
        const float s           = __half2float(src_scale[g]) * inv;
        signed char quad[32];
        for (int w = 0; w < 4; ++w) {
            for (int b = 0; b < 4; ++b) {
                const unsigned byte = (words[w] >> (b * 8)) & 0xffu;
                const int lo        = static_cast<int>(byte & 0xfu) - 8;
                const int hi        = static_cast<int>(byte >> 4) - 8;
                quad[w * 8 + b * 2]     = static_cast<signed char>(max(-127, min(127, __float2int_rn(lo * s))));
                quad[w * 8 + b * 2 + 1] = static_cast<signed char>(max(-127, min(127, __float2int_rn(hi * s))));
            }
        }
        for (int j = 0; j < 4; ++j) {
            *reinterpret_cast<uint2*>(dst + i * 32 + j * 8) =
                *reinterpret_cast<const uint2*>(quad + j * 8);
        }
    }
}

struct Shape {
    const char* name;
    int rows;
    int cols;
    double ours_us; // this fork's W4A8 kernel at T=1024, measured by ninfer_a8_prefill_schedule_bench
};

int main() {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));
    // Q4_G64_FP16 only -- dequantise_rows_to_int8 reads just the low nibble plane, so a Q5 shape
    // (mlp/down and out_proj both carry an extra high-plane bit per element) would silently
    // measure fewer bytes than its real route moves. gate_up is the only registered A8 prefill
    // shape that is genuinely Q4, so it is the only one this dequantise-then-cuBLAS question
    // applies to as-is.
    const Shape shapes[] = {
        {"mlp/gate_up 34816x5120", 34816, 5120, 3824.6},
    };
#ifndef T_TOKENS
#define T_TOKENS 1024
#endif
    // The dequantise pass is weight-sized and the GEMM is token-sized, so the overhead this route
    // carries falls as the prefill chunk grows. That ratio is the whole question, hence the sweep.
    const int t = T_TOKENS;

    cublasHandle_t handle;
    CHECK_BLAS(cublasCreate(&handle));
    void* flush = nullptr;
    CHECK(cudaMalloc(&flush, 256u << 20));

    printf("GPU: %s  sm_%d%d   W4 -> int8 materialisation plus cuBLAS, T=%d, median of 9\n", p.name,
           p.major, p.minor, t);
    printf("%-26s %10s %10s %10s %12s %10s\n", "shape", "dequant", "cublas", "total", "ours", "vs ours");

    for (const Shape& shape : shapes) {
        const size_t code_bytes = static_cast<size_t>(shape.rows) * shape.cols / 2;
        const size_t int8_bytes = static_cast<size_t>(shape.rows) * shape.cols;
        void *codes = nullptr, *scales = nullptr, *w8 = nullptr, *rs = nullptr;
        void *b = nullptr, *c = nullptr;
        CHECK(cudaMalloc(&codes, code_bytes));
        CHECK(cudaMalloc(&scales, static_cast<size_t>(shape.rows) * (shape.cols / 64) * sizeof(__half)));
        CHECK(cudaMalloc(&w8, int8_bytes));
        CHECK(cudaMalloc(&rs, static_cast<size_t>(shape.rows) * sizeof(float)));
        CHECK(cudaMalloc(&b, static_cast<size_t>(shape.cols) * t));
        CHECK(cudaMalloc(&c, static_cast<size_t>(shape.rows) * t * sizeof(int)));
        CHECK(cudaMemset(codes, 0x73, code_bytes));
        CHECK(cudaMemset(scales, 0x3c, static_cast<size_t>(shape.rows) * (shape.cols / 64) * sizeof(__half)));
        CHECK(cudaMemset(b, 5, static_cast<size_t>(shape.cols) * t));

        const auto run_dequant = [&] {
            dequantise_rows_to_int8<<<shape.rows, 256>>>(
                static_cast<const unsigned char*>(codes), static_cast<const __half*>(scales),
                shape.cols, static_cast<signed char*>(w8), static_cast<float*>(rs));
        };
        const int alpha = 1, beta = 0;
        const auto run_gemm = [&] {
            CHECK_BLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.rows, t, shape.cols,
                                    &alpha, w8, CUDA_R_8I, shape.cols, b, CUDA_R_8I, shape.cols,
                                    &beta, c, CUDA_R_32I, shape.rows, CUBLAS_COMPUTE_32I,
                                    CUBLAS_GEMM_DEFAULT));
        };

        run_dequant();
        run_gemm();
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t s0, s1, s2;
        CHECK(cudaEventCreate(&s0));
        CHECK(cudaEventCreate(&s1));
        CHECK(cudaEventCreate(&s2));
        std::vector<float> dq(9), gm(9);
        for (int i = 0; i < 9; ++i) {
            CHECK(cudaMemsetAsync(flush, i & 0xff, 256u << 20));
            CHECK(cudaEventRecord(s0));
            run_dequant();
            CHECK(cudaEventRecord(s1));
            run_gemm();
            CHECK(cudaEventRecord(s2));
            CHECK(cudaEventSynchronize(s2));
            CHECK(cudaEventElapsedTime(&dq[i], s0, s1));
            CHECK(cudaEventElapsedTime(&gm[i], s1, s2));
        }
        std::sort(dq.begin(), dq.end());
        std::sort(gm.begin(), gm.end());
        const double dq_us = dq[4] * 1000.0, gm_us = gm[4] * 1000.0;
        const double total = dq_us + gm_us;
        printf("%-26s %10.1f %10.1f %10.1f %12.1f %9.2fx\n", shape.name, dq_us, gm_us, total,
               shape.ours_us, shape.ours_us / total);

        CHECK(cudaEventDestroy(s0));
        CHECK(cudaEventDestroy(s1));
        CHECK(cudaEventDestroy(s2));
        for (void* ptr : {codes, scales, w8, rs, b, c}) { CHECK(cudaFree(ptr)); }
    }
    CHECK(cudaFree(flush));
    CHECK_BLAS(cublasDestroy(handle));
    return 0;
}
