// Is "materialise the ternary weight, then call cuBLAS" worth it for T2G128 prefill on sm_120?
//
// tools/w4_dequant_cublas_probe.cu asked this for Q4 on sm_86 and the answer was yes, about 2x at
// 4096 tokens. The ternary integer route runs the same shared int8 mainloop (ops/linear/t2/t2_a8.h),
// so the same gap may be there -- but on a GeForce Blackwell card the rates differ by format, and
// materialising costs a weight-sized write per call. This measures every part separately, per
// shape and prefill chunk, against both of this fork's ternary routes:
//
//   ours A16      ops::linear, LinearPolicy::A16Only (the MMA route of t2_dispatch)
//   ours int8     ops::linear, LinearPolicy::AllowA8Int (t2_a8_linear)
//   cuBLAS int8   T2 -> int8 with one scale per row, activations to int8 per token,
//                 int32 GEMM, int32 -> BF16 rescale
//   cuBLAS fp8    the same through E4M3 and cublasLt, BF16 out, then the row/token rescale
//   cuBLAS bf16   T2 -> BF16 exactly (code * group scale), BF16 GEMM with FP32 accumulation
//   cuBLAS fp16   T2 -> FP16 exactly, FP16 GEMM with FP16 accumulation, FP16 -> BF16
//
// "err" is the relative L2 distance to the A16 route's output. The weights and activations are
// synthetic, so it checks that the routes compute the same product; it is not a quality figure for
// a real model, whose group scales spread further within a row than these do.
//
// Build (inside the build container, after the sm120a tree is built):
//   nvcc -O3 -std=c++20 -arch=sm_120a -ccbin g++-13 -DNINFER_SM8X_COMPAT=1 -Iinclude -Isrc \
//        tools/t2_cublas_probe.cu build-sm120a/src/ops/libninfer_ops.a \
//        build-sm120a/src/core/libninfer_core.a -lcublas -lcublasLt \
//        -o build-sm120a/tools/ninfer-t2-cublas-probe
// Run: ninfer-t2-cublas-probe [tokens...]   (default 1024 2048 4096)

#include "core/arena.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t status_ = (x);                                                                 \
        if (status_ != cudaSuccess) {                                                              \
            std::printf("CUDA error %s at line %d\n", cudaGetErrorString(status_), __LINE__);      \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

#define CHECK_BLAS(x)                                                                              \
    do {                                                                                           \
        cublasStatus_t st = (x);                                                                   \
        if (st != CUBLAS_STATUS_SUCCESS) {                                                         \
            std::printf("cuBLAS error %d at line %d\n", static_cast<int>(st), __LINE__);           \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

using namespace ninfer;

enum Target : int { kInt8 = 0, kFp8 = 1, kBf16 = 2, kFp16 = 3 };

__device__ float block_max(float value) {
    __shared__ float partial[32];
    for (int off = 16; off > 0; off >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, off));
    }
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = value; }
    __syncthreads();
    if (threadIdx.x < 32) {
        float m = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        }
        if (threadIdx.x == 0) { partial[0] = m; }
    }
    __syncthreads();
    return partial[0];
}

// Element j of a 32-bit word of codes sits at bits 2j; 01 is +1, 11 is -1, 00 is zero.
__device__ __forceinline__ float t2_value(std::uint32_t word, int field) {
    const int u = static_cast<int>((word >> (2 * field)) & 3u);
    return static_cast<float>((u & 2) != 0 ? u - 4 : u);
}

// One row per block. A word holds 16 codes and a group of 128 spans 8 words, so a word never
// straddles a group. The integer and FP8 targets fold the group scales into one per row, which is
// the whole of the weight-side quality trade; BF16 and FP16 keep code * group scale exactly.
template <int kTarget>
__global__ void materialise_t2(const std::uint32_t* __restrict__ codes,
                               const __half* __restrict__ scales, int k, void* __restrict__ out,
                               float* __restrict__ row_scale) {
    const int row            = blockIdx.x;
    const int groups         = k / 128;
    const std::uint32_t* src = codes + static_cast<std::size_t>(row) * (k / 16);
    const __half* scale      = scales + static_cast<std::size_t>(row) * groups;
    float inv                = 1.0f;
    if constexpr (kTarget == kInt8 || kTarget == kFp8) {
        float amax = 0.0f;
        for (int g = threadIdx.x; g < groups; g += blockDim.x) {
            amax = fmaxf(amax, fabsf(__half2float(scale[g])));
        }
        amax                 = fmaxf(block_max(amax), 1e-20f);
        constexpr float qmax = kTarget == kInt8 ? 127.0f : 448.0f;
        if (threadIdx.x == 0) { row_scale[row] = amax / qmax; }
        inv = qmax / amax;
    }
    for (int i = threadIdx.x; i < k / 16; i += blockDim.x) {
        const std::uint32_t word = src[i];
        const float s            = __half2float(scale[i / 8]) * inv;
        const std::size_t base   = static_cast<std::size_t>(row) * k + static_cast<std::size_t>(i) * 16;
        if constexpr (kTarget == kInt8) {
            __align__(16) signed char v[16];
#pragma unroll
            for (int f = 0; f < 16; ++f) {
                v[f] = static_cast<signed char>(
                    max(-127, min(127, __float2int_rn(t2_value(word, f) * s))));
            }
            *reinterpret_cast<uint4*>(static_cast<signed char*>(out) + base) =
                *reinterpret_cast<const uint4*>(v);
        } else if constexpr (kTarget == kFp8) {
            __align__(16) __nv_fp8_e4m3 v[16];
#pragma unroll
            for (int f = 0; f < 16; ++f) { v[f] = __nv_fp8_e4m3(t2_value(word, f) * s); }
            *reinterpret_cast<uint4*>(static_cast<__nv_fp8_e4m3*>(out) + base) =
                *reinterpret_cast<const uint4*>(v);
        } else if constexpr (kTarget == kBf16) {
            __align__(16) __nv_bfloat16 v[16];
#pragma unroll
            for (int f = 0; f < 16; ++f) { v[f] = __float2bfloat16(t2_value(word, f) * s); }
            auto* dst = reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(out) + base);
            dst[0]    = reinterpret_cast<const uint4*>(v)[0];
            dst[1]    = reinterpret_cast<const uint4*>(v)[1];
        } else {
            __align__(16) __half v[16];
#pragma unroll
            for (int f = 0; f < 16; ++f) { v[f] = __float2half(t2_value(word, f) * s); }
            auto* dst = reinterpret_cast<uint4*>(static_cast<__half*>(out) + base);
            dst[0]    = reinterpret_cast<const uint4*>(v)[0];
            dst[1]    = reinterpret_cast<const uint4*>(v)[1];
        }
    }
}

// One token per block: the int8 and FP8 targets take one scale per token, FP16 is a plain cast.
template <int kTarget>
__global__ void convert_tokens(const __nv_bfloat16* __restrict__ x, int k, void* __restrict__ out,
                               float* __restrict__ token_scale) {
    const int t              = blockIdx.x;
    const __nv_bfloat16* src = x + static_cast<std::size_t>(t) * k;
    float inv                = 1.0f;
    if constexpr (kTarget == kInt8 || kTarget == kFp8) {
        float amax = 0.0f;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
        }
        amax                 = fmaxf(block_max(amax), 1e-20f);
        constexpr float qmax = kTarget == kInt8 ? 127.0f : 448.0f;
        if (threadIdx.x == 0) { token_scale[t] = amax / qmax; }
        inv = qmax / amax;
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const float v        = __bfloat162float(src[i]) * inv;
        const std::size_t at = static_cast<std::size_t>(t) * k + i;
        if constexpr (kTarget == kInt8) {
            static_cast<signed char*>(out)[at] =
                static_cast<signed char>(max(-127, min(127, __float2int_rn(v))));
        } else if constexpr (kTarget == kFp8) {
            static_cast<__nv_fp8_e4m3*>(out)[at] = __nv_fp8_e4m3(v);
        } else {
            static_cast<__half*>(out)[at] = __float2half(v);
        }
    }
}

// Rows are a multiple of 8 for every shape here, so a vector never crosses a token column.
__global__ void rescale_int32(const int4* __restrict__ c, const float* __restrict__ row_scale,
                              const float* __restrict__ token_scale, int n, std::size_t quads,
                              uint2* __restrict__ out) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; q < quads;
         q += stride) {
        const std::size_t idx = q * 4;
        const int row         = static_cast<int>(idx % n);
        const float ts        = token_scale[idx / n];
        const int4 v          = c[q];
        __align__(8) __nv_bfloat16 r[4] = {
            __float2bfloat16(static_cast<float>(v.x) * row_scale[row] * ts),
            __float2bfloat16(static_cast<float>(v.y) * row_scale[row + 1] * ts),
            __float2bfloat16(static_cast<float>(v.z) * row_scale[row + 2] * ts),
            __float2bfloat16(static_cast<float>(v.w) * row_scale[row + 3] * ts)};
        out[q] = *reinterpret_cast<const uint2*>(r);
    }
}

__global__ void rescale_bf16(uint4* __restrict__ data, const float* __restrict__ row_scale,
                             const float* __restrict__ token_scale, int n, std::size_t octets) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; q < octets;
         q += stride) {
        const std::size_t idx = q * 8;
        const int row         = static_cast<int>(idx % n);
        const float ts        = token_scale[idx / n];
        uint4 v               = data[q];
        auto* h               = reinterpret_cast<__nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            h[j] = __float2bfloat16(__bfloat162float(h[j]) * row_scale[row + j] * ts);
        }
        data[q] = v;
    }
}

__global__ void half_to_bf16(const uint4* __restrict__ in, uint4* __restrict__ out,
                             std::size_t octets) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; q < octets;
         q += stride) {
        const uint4 v  = in[q];
        const auto* h  = reinterpret_cast<const __half*>(&v);
        __align__(16) __nv_bfloat16 r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { r[j] = __float2bfloat16(__half2float(h[j])); }
        out[q] = *reinterpret_cast<const uint4*>(r);
    }
}

struct Shape {
    const char* name;
    std::int32_t rows;
    std::int32_t cols;
};

// One arm's parts in microseconds; a part the arm does not have stays at zero.
struct Parts {
    double weight   = 0.0;
    double activate = 0.0;
    double gemm     = 0.0;
    double epilogue = 0.0;
    double total() const { return weight + activate + gemm + epilogue; }
};

class Timer {
public:
    explicit Timer(void* flush, std::size_t flush_bytes) : flush_(flush), flush_bytes_(flush_bytes) {
        for (auto& e : events_) { CHECK(cudaEventCreate(&e)); }
    }
    ~Timer() {
        for (auto& e : events_) { cudaEventDestroy(e); }
    }

    // Median of 9 over up to four stages run back to back, the L2 flushed before each repeat.
    template <class A, class B, class C, class D>
    Parts measure(A&& weight, B&& activate, C&& gemm, D&& epilogue) {
        std::vector<float> parts[4];
        for (int i = 0; i < 11; ++i) {
            CHECK(cudaMemsetAsync(flush_, i & 0xff, flush_bytes_));
            CHECK(cudaEventRecord(events_[0]));
            weight();
            CHECK(cudaEventRecord(events_[1]));
            activate();
            CHECK(cudaEventRecord(events_[2]));
            gemm();
            CHECK(cudaEventRecord(events_[3]));
            epilogue();
            CHECK(cudaEventRecord(events_[4]));
            CHECK(cudaEventSynchronize(events_[4]));
            if (i < 2) { continue; }
            for (int p = 0; p < 4; ++p) {
                float ms = 0.0f;
                CHECK(cudaEventElapsedTime(&ms, events_[p], events_[p + 1]));
                parts[p].push_back(ms);
            }
        }
        double median[4];
        for (int p = 0; p < 4; ++p) {
            std::sort(parts[p].begin(), parts[p].end());
            median[p] = parts[p][parts[p].size() / 2] * 1000.0;
        }
        return {median[0], median[1], median[2], median[3]};
    }

private:
    void* flush_;
    std::size_t flush_bytes_;
    cudaEvent_t events_[5]{};
};

double relative_l2(const __nv_bfloat16* device, const std::vector<__nv_bfloat16>& reference) {
    std::vector<__nv_bfloat16> host(reference.size());
    CHECK(cudaMemcpy(host.data(), device, host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < host.size(); ++i) {
        const double r = __bfloat162float(reference[i]);
        const double d = __bfloat162float(host[i]) - r;
        diff += d * d;
        norm += r * r;
    }
    return std::sqrt(diff / std::max(norm, 1e-30));
}

void print_row(const char* arm, const Parts& p, double ops, double baseline_us, double err) {
    const double busy = p.gemm > 0.0 ? p.gemm : p.total();
    std::printf("  %-12s %8.1f %8.1f %9.1f %8.1f %9.1f %7.1f %8.2fx %9.2e\n", arm, p.weight,
                p.activate, p.gemm, p.epilogue, p.total(), ops / busy / 1e6, baseline_us / p.total(),
                err);
}

void print_skip(const char* arm, const char* why) { std::printf("  %-12s %s\n", arm, why); }

} // namespace

int main(int argc, char** argv) {
    std::vector<int> token_counts;
    for (int i = 1; i < argc; ++i) { token_counts.push_back(std::atoi(argv[i])); }
    if (token_counts.empty()) { token_counts = {1024, 2048, 4096}; }

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const int epilogue_blocks = prop.multiProcessorCount * 16;

    cublasHandle_t blas;
    CHECK_BLAS(cublasCreate(&blas));
    cublasLtHandle_t lt;
    CHECK_BLAS(cublasLtCreate(&lt));
    constexpr std::size_t kLtWorkspaceBytes = 64u << 20;
    void* lt_workspace                      = nullptr;
    CHECK(cudaMalloc(&lt_workspace, kLtWorkspaceBytes));
    constexpr std::size_t kFlushBytes = 128u << 20;
    void* flush                       = nullptr;
    CHECK(cudaMalloc(&flush, kFlushBytes));
    Timer timer(flush, kFlushBytes);

    std::printf("GPU: %s  sm_%d%d, %d SMs, L2 %d MiB; microseconds, median of 9, L2 flushed\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.l2CacheSize >> 20);

    const Shape shapes[] = {
        {"mlp/gate_up", 34816, 5120},
        {"mlp/down", 5120, 17408},
        {"gdn/value_z", 12288, 5120},
    };

    std::mt19937_64 rng(20260923);
    for (const Shape& shape : shapes) {
        const std::int32_t n        = shape.rows;
        const std::int32_t k        = shape.cols;
        const std::size_t words     = static_cast<std::size_t>(n) * (k / 16);
        const std::size_t groups    = static_cast<std::size_t>(n) * (k / 128);
        const std::size_t code_size = words * sizeof(std::uint32_t);

        // Legal ternary codes only (0b10 is not a T2 field); group scales spread 4x within a row.
        std::vector<std::uint32_t> host_codes(words);
        for (auto& word : host_codes) {
            std::uint64_t bits = rng();
            std::uint32_t out  = 0;
            for (int f = 0; f < 16; ++f) {
                std::uint32_t code = static_cast<std::uint32_t>(bits >> (2 * f)) & 3u;
                if (code == 2u) { code = 3u; }
                out |= code << (2 * f);
            }
            word = out;
        }
        std::uniform_real_distribution<float> spread(0.005f, 0.02f);
        std::vector<__half> host_scales(groups);
        for (auto& s : host_scales) { s = __float2half(spread(rng)); }

        void* codes  = nullptr;
        void* scales = nullptr;
        CHECK(cudaMalloc(&codes, code_size));
        CHECK(cudaMalloc(&scales, groups * sizeof(__half)));
        CHECK(cudaMemcpy(codes, host_codes.data(), code_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(scales, host_scales.data(), groups * sizeof(__half),
                         cudaMemcpyHostToDevice));

        Weight w;
        w.payload         = codes;
        w.payload_bytes   = code_size + groups * sizeof(__half);
        w.qtype           = QType::T2_G128_FP16;
        w.group_size      = 128;
        w.shape[0]        = n;
        w.shape[1]        = k;
        w.padded_shape[0] = n;
        w.padded_shape[1] = k;
        w.ndim            = 2;
        w.qdata           = codes;
        w.scales          = scales;
        w.n               = n;
        w.k               = k;
        w.group           = 128;
        w.layout          = QuantLayout::RowSplit;
        w.scale_dtype     = DType::FP16;

        void* materialised = nullptr; // the widest target is two bytes per element
        void* row_scale    = nullptr;
        CHECK(cudaMalloc(&materialised, static_cast<std::size_t>(n) * k * 2));
        CHECK(cudaMalloc(&row_scale, static_cast<std::size_t>(n) * sizeof(float)));

        for (const int t : token_counts) {
            const double ops       = 2.0 * n * k * t;
            const std::size_t outs = static_cast<std::size_t>(n) * t;
            std::printf("\n%s %dx%d, T=%d\n", shape.name, n, k, t);
            std::printf("  %-12s %8s %8s %9s %8s %9s %7s %9s %9s\n", "route", "weight", "activ",
                        "gemm", "epilog", "total", "TOP/s", "vs int8", "err");

            DeviceArena io(static_cast<std::size_t>(k) * t * 2 * 2 + outs * 2 * 3 + (64u << 20));
            Tensor x         = io.alloc(DType::BF16, {k, t});
            Tensor reference = io.alloc(DType::BF16, {n, t});
            Tensor result    = io.alloc(DType::BF16, {n, t});
            Tensor staging   = io.alloc(DType::BF16, {n, t});
            void* x_low      = io.alloc_bytes(static_cast<std::size_t>(k) * t * 2).data;
            void* token_scale = io.alloc_bytes(static_cast<std::size_t>(t) * sizeof(float)).data;
            {
                std::normal_distribution<float> normal(0.0f, 1.0f);
                std::vector<__nv_bfloat16> host_x(static_cast<std::size_t>(k) * t);
                for (auto& v : host_x) { v = __float2bfloat16(normal(rng)); }
                CHECK(cudaMemcpy(x.data, host_x.data(), host_x.size() * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice));
            }
            DeviceArena scratch(512u << 20);
            const auto none = [] {};

            // Reference and the two routes this fork ships.
            Parts a16, a8;
            try {
                a16 = timer.measure(none, none, [&] { ops::linear(x, w, reference, nullptr); },
                                    none);
                a8 = timer.measure(none, none,
                                   [&] {
                                       auto scope = scratch.scope();
                                       ops::linear(x, w, result, ops::LinearPolicy::AllowA8Int,
                                                   scratch, nullptr);
                                   },
                                   none);
            } catch (const std::exception& error) {
                std::printf("  ours: %s\n", error.what());
                continue;
            }
            CHECK(cudaDeviceSynchronize());
            std::vector<__nv_bfloat16> host_reference(outs);
            CHECK(cudaMemcpy(host_reference.data(), reference.data, outs * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost));
            a16.epilogue = a16.gemm; // report the whole call as the total, not as GEMM rate
            a16.gemm     = 0.0;
            a8.epilogue  = a8.gemm;
            a8.gemm      = 0.0;
            const double baseline = a8.total();
            print_row("ours A16", a16, ops, baseline, 0.0);
            print_row("ours int8", a8, ops, baseline, relative_l2(
                                                           static_cast<__nv_bfloat16*>(result.data),
                                                           host_reference));

            // cuBLAS int8: the int32 product for the whole chunk, as the probe for Q4 did.
            {
                void* c32 = nullptr;
                if (cudaMalloc(&c32, outs * sizeof(int)) != cudaSuccess) {
                    cudaGetLastError();
                    print_skip("cuBLAS int8", "no memory for the int32 product");
                } else {
                    const int alpha = 1, beta = 0;
                    const Parts p = timer.measure(
                        [&] {
                            materialise_t2<kInt8><<<n, 256>>>(
                                static_cast<const std::uint32_t*>(codes),
                                static_cast<const __half*>(scales), k, materialised,
                                static_cast<float*>(row_scale));
                        },
                        [&] {
                            convert_tokens<kInt8><<<t, 256>>>(
                                static_cast<const __nv_bfloat16*>(x.data), k, x_low,
                                static_cast<float*>(token_scale));
                        },
                        [&] {
                            CHECK_BLAS(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &alpha,
                                                    materialised, CUDA_R_8I, k, x_low, CUDA_R_8I, k,
                                                    &beta, c32, CUDA_R_32I, n, CUBLAS_COMPUTE_32I,
                                                    CUBLAS_GEMM_DEFAULT));
                        },
                        [&] {
                            rescale_int32<<<epilogue_blocks, 256>>>(
                                static_cast<const int4*>(c32), static_cast<const float*>(row_scale),
                                static_cast<const float*>(token_scale), n, outs / 4,
                                static_cast<uint2*>(result.data));
                        });
                    CHECK(cudaGetLastError());
                    print_row("cuBLAS int8", p, ops, baseline,
                              relative_l2(static_cast<__nv_bfloat16*>(result.data), host_reference));
                    CHECK(cudaFree(c32));
                }
            }

            // cuBLAS fp8 through cublasLt, BF16 out; the row and token scales follow in one pass.
            {
                cublasLtMatmulDesc_t desc = nullptr;
                cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
                cublasLtMatmulPreference_t pref = nullptr;
                CHECK_BLAS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
                const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
                CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                                          sizeof(ta)));
                CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                                          sizeof(tb)));
                CHECK_BLAS(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, k, n, k));
                CHECK_BLAS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, t, k));
                CHECK_BLAS(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, n, t, n));
                CHECK_BLAS(cublasLtMatmulPreferenceCreate(&pref));
                const std::size_t ws_bytes = kLtWorkspaceBytes;
                CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(
                    pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
                cublasLtMatmulHeuristicResult_t heuristic{};
                int found = 0;
                const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
                    lt, desc, la, lb, lc, lc, pref, 1, &heuristic, &found);
                if (status != CUBLAS_STATUS_SUCCESS || found == 0) {
                    print_skip("cuBLAS fp8", "cublasLt offers no algorithm for this shape");
                } else {
                    const float alpha = 1.0f, beta = 0.0f;
                    const Parts p = timer.measure(
                        [&] {
                            materialise_t2<kFp8><<<n, 256>>>(
                                static_cast<const std::uint32_t*>(codes),
                                static_cast<const __half*>(scales), k, materialised,
                                static_cast<float*>(row_scale));
                        },
                        [&] {
                            convert_tokens<kFp8><<<t, 256>>>(
                                static_cast<const __nv_bfloat16*>(x.data), k, x_low,
                                static_cast<float*>(token_scale));
                        },
                        [&] {
                            CHECK_BLAS(cublasLtMatmul(lt, desc, &alpha, materialised, la, x_low, lb,
                                                      &beta, result.data, lc, result.data, lc,
                                                      &heuristic.algo, lt_workspace, ws_bytes,
                                                      nullptr));
                        },
                        [&] {
                            rescale_bf16<<<epilogue_blocks, 256>>>(
                                static_cast<uint4*>(result.data),
                                static_cast<const float*>(row_scale),
                                static_cast<const float*>(token_scale), n, outs / 8);
                        });
                    CHECK(cudaGetLastError());
                    print_row("cuBLAS fp8", p, ops, baseline,
                              relative_l2(static_cast<__nv_bfloat16*>(result.data), host_reference));
                }
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(lc);
                cublasLtMatrixLayoutDestroy(lb);
                cublasLtMatrixLayoutDestroy(la);
                cublasLtMatmulDescDestroy(desc);
            }

            // cuBLAS bf16: exact weights, the activations as they are, FP32 accumulation.
            {
                const float alpha = 1.0f, beta = 0.0f;
                const Parts p = timer.measure(
                    [&] {
                        materialise_t2<kBf16><<<n, 256>>>(
                            static_cast<const std::uint32_t*>(codes),
                            static_cast<const __half*>(scales), k, materialised, nullptr);
                    },
                    none,
                    [&] {
                        CHECK_BLAS(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &alpha,
                                                materialised, CUDA_R_16BF, k, x.data, CUDA_R_16BF,
                                                k, &beta, result.data, CUDA_R_16BF, n,
                                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                    },
                    none);
                CHECK(cudaGetLastError());
                print_row("cuBLAS bf16", p, ops, baseline,
                          relative_l2(static_cast<__nv_bfloat16*>(result.data), host_reference));
            }

            // cuBLAS fp16 with FP16 accumulation: exact weights, the rate GeForce parts run fastest
            // among the 16-bit forms, at the price of accumulating in half precision.
            {
                const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
                const Parts p = timer.measure(
                    [&] {
                        materialise_t2<kFp16><<<n, 256>>>(
                            static_cast<const std::uint32_t*>(codes),
                            static_cast<const __half*>(scales), k, materialised, nullptr);
                    },
                    [&] {
                        convert_tokens<kFp16><<<t, 256>>>(
                            static_cast<const __nv_bfloat16*>(x.data), k, x_low, nullptr);
                    },
                    [&] {
                        CHECK_BLAS(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &alpha,
                                                materialised, CUDA_R_16F, k, x_low, CUDA_R_16F, k,
                                                &beta, staging.data, CUDA_R_16F, n,
                                                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                    },
                    [&] {
                        half_to_bf16<<<epilogue_blocks, 256>>>(
                            static_cast<const uint4*>(staging.data),
                            static_cast<uint4*>(result.data), outs / 8);
                    });
                CHECK(cudaGetLastError());
                print_row("cuBLAS fp16", p, ops, baseline,
                          relative_l2(static_cast<__nv_bfloat16*>(result.data), host_reference));
            }
        }
        CHECK(cudaFree(row_scale));
        CHECK(cudaFree(materialised));
        CHECK(cudaFree(scales));
        CHECK(cudaFree(codes));
    }

    CHECK(cudaFree(flush));
    CHECK(cudaFree(lt_workspace));
    cublasLtDestroy(lt);
    CHECK_BLAS(cublasDestroy(blas));
    return 0;
}
