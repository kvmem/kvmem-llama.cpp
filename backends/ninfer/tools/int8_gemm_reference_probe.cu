// What a tuned int8 GEMM achieves on this card at the shapes prefill actually runs.
//
// Our W4A8 kernels measure 95-102 TOP/s against a 314.8 TOP/s pure-MMA microbenchmark, and the gap
// has been attributed in turn to occupancy, operand layout and streaming. None of those arguments
// is worth much without knowing what a well-tuned library kernel gets on the same silicon at the
// same shape, because the microbenchmark reads no memory at all and is therefore not a ceiling any
// real GEMM can approach.
//
// So this runs cuBLAS's own int8 GEMM (IMMA, TN, s32 accumulate) over the four prefill shapes, with
// both operands already int8 in the same k-major layout our kernels read. It is a strictly easier
// problem than ours -- 8-bit weights rather than 4-bit codes with a per-group scale and an unpack,
// no rescale, no fused epilogue -- so whatever it reaches is an upper bound on what our schedule
// could reach, and the distance to it is the honest measure of how much is left.
//
// Build:
//   nvcc -O3 -arch=sm_86 -allow-unsupported-compiler int8_gemm_reference_probe.cu -lcublas \
//        -o int8_gemm_reference_probe

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
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
        cublasStatus_t s = (x);                                                                    \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                          \
            printf("cuBLAS error %d at line %d\n", static_cast<int>(s), __LINE__);                  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

struct Shape {
    const char* name;
    int rows;   // M, the weight's output rows
    int cols;   // K, the shared dimension
};

int main() {
    cudaDeviceProp p{};
    CHECK(cudaGetDeviceProperties(&p, 0));

    // The registered prefill profiles, plus the token counts a 1,024-token chunk runs at.
    const Shape shapes[] = {
        {"mlp/gate_up  34816x5120", 34816, 5120},
        {"mlp/down      5120x17408", 5120, 17408},
        {"out_proj      5120x6144", 5120, 6144},
        {"gdn value_z  12288x5120", 12288, 5120},
    };
    const int token_counts[] = {512, 1024};

    cublasHandle_t handle;
    CHECK_BLAS(cublasCreate(&handle));

    void* flush = nullptr;
    CHECK(cudaMalloc(&flush, 256u << 20));

    printf("GPU: %s  sm_%d%d   cuBLAS int8 TN GEMM, s32 accumulate, cold, median of 9\n", p.name,
           p.major, p.minor);
    printf("%-26s %7s %12s %12s\n", "shape", "T", "us", "TOP/s");

    for (const Shape& shape : shapes) {
        for (const int t : token_counts) {
            void *a = nullptr, *b = nullptr, *c = nullptr;
            CHECK(cudaMalloc(&a, static_cast<size_t>(shape.cols) * shape.rows));
            CHECK(cudaMalloc(&b, static_cast<size_t>(shape.cols) * t));
            CHECK(cudaMalloc(&c, static_cast<size_t>(shape.rows) * t * sizeof(int)));
            CHECK(cudaMemset(a, 3, static_cast<size_t>(shape.cols) * shape.rows));
            CHECK(cudaMemset(b, 5, static_cast<size_t>(shape.cols) * t));

            const int alpha = 1, beta = 0;
            // C[M,N] = A[K,M]^T * B[K,N], everything column-major and k-major, which is the layout
            // our own kernels read.
            const auto run = [&] {
                CHECK_BLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.rows, t, shape.cols, &alpha,
                                        a, CUDA_R_8I, shape.cols, b, CUDA_R_8I, shape.cols, &beta, c,
                                        CUDA_R_32I, shape.rows, CUBLAS_COMPUTE_32I,
                                        CUBLAS_GEMM_DEFAULT));
            };

            run();
            CHECK(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CHECK(cudaEventCreate(&start));
            CHECK(cudaEventCreate(&stop));
            std::vector<float> ms(9);
            for (int i = 0; i < 9; ++i) {
                CHECK(cudaMemsetAsync(flush, i & 0xff, 256u << 20));
                CHECK(cudaEventRecord(start));
                run();
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                CHECK(cudaEventElapsedTime(&ms[i], start, stop));
            }
            std::sort(ms.begin(), ms.end());
            const double us   = ms[4] * 1000.0;
            const double work = 2.0 * shape.rows * shape.cols * t;
            printf("%-26s %7d %12.1f %12.2f\n", shape.name, t, us, work / (us * 1e-6) / 1e12);

            // Same shape in bf16, for scale: this fork's A16 route runs at ~88% of the card's
            // bf16 ceiling, so if cuBLAS bf16 lands near it the gap is specific to the integer
            // path rather than to these GEMMs in general.
            void *ab = nullptr, *bb = nullptr, *cb = nullptr;
            CHECK(cudaMalloc(&ab, static_cast<size_t>(shape.cols) * shape.rows * 2));
            CHECK(cudaMalloc(&bb, static_cast<size_t>(shape.cols) * t * 2));
            CHECK(cudaMalloc(&cb, static_cast<size_t>(shape.rows) * t * 2));
            CHECK(cudaMemset(ab, 0x3c, static_cast<size_t>(shape.cols) * shape.rows * 2));
            CHECK(cudaMemset(bb, 0x3c, static_cast<size_t>(shape.cols) * t * 2));
            const float alpha_f = 1.0F, beta_f = 0.0F;
            const auto run_bf16 = [&] {
                CHECK_BLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, shape.rows, t, shape.cols,
                                        &alpha_f, ab, CUDA_R_16BF, shape.cols, bb, CUDA_R_16BF,
                                        shape.cols, &beta_f, cb, CUDA_R_16BF, shape.rows,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            };
            run_bf16();
            CHECK(cudaDeviceSynchronize());
            for (int i = 0; i < 9; ++i) {
                CHECK(cudaMemsetAsync(flush, i & 0xff, 256u << 20));
                CHECK(cudaEventRecord(start));
                run_bf16();
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                CHECK(cudaEventElapsedTime(&ms[i], start, stop));
            }
            std::sort(ms.begin(), ms.end());
            const double us_bf16 = ms[4] * 1000.0;
            printf("%-26s %7d %12.1f %12.2f   (bf16)\n", shape.name, t, us_bf16,
                   work / (us_bf16 * 1e-6) / 1e12);
            CHECK(cudaFree(ab));
            CHECK(cudaFree(bb));
            CHECK(cudaFree(cb));

            CHECK(cudaEventDestroy(start));
            CHECK(cudaEventDestroy(stop));
            CHECK(cudaFree(a));
            CHECK(cudaFree(b));
            CHECK(cudaFree(c));
        }
    }
    CHECK(cudaFree(flush));
    CHECK_BLAS(cublasDestroy(handle));
    return 0;
}
