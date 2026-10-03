// Entry points into the vendored ggml CUDA kernels and the vector kernel (ggml_bridge_vec.cuh). The
// activation quantizers follow ggml's q8_1, reading BF16 and optionally gathering the input columns.

#include "ggml_bridge_internal.cuh"

#include "ggml-cuda/dequantize.cuh"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// ggml's GGML_ABORT/GGML_ASSERT land here; only the vendored headers' own invariants reach it.
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml abort at %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

namespace ninfer::ops::gguf {
namespace detail {

const DeviceFacts& device_facts() {
    static std::array<DeviceFacts, 16> facts{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    if (device < 0 || device >= int(facts.size())) {
        throw std::invalid_argument("gguf: device ordinal out of range");
    }
    std::call_once(once[device], [&] {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        facts[device].cc                     = 100 * properties.major + 10 * properties.minor;
        facts[device].sm_count               = properties.multiProcessorCount;
        facts[device].shared_per_block_optin = properties.sharedMemPerBlockOptin;
    });
    return facts[device];
}

namespace {

// x BF16 [columns][k] -> the vector kernel's planar q8 activation: ggml's q8_1 numbers, one warp per
// 32 values, element i of column c read from x[c][input_columns[i]] when a gather is given.
constexpr int kQuantizeThreads = 256;

__launch_bounds__(kQuantizeThreads) __global__
    void quantize_vector_kernel(const __nv_bfloat16* __restrict__ x,
                                const std::int32_t* __restrict__ input_columns,
                                std::int8_t* __restrict__ qs, half2* __restrict__ ds, int k) {
    const int i      = blockDim.x * blockIdx.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    const int source = input_columns != nullptr ? input_columns[i] : i;
    const float xi   = __bfloat162float(x[std::int64_t(column) * k + source]);
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float sum  = warp_reduce_sum<QK8_1>(xi);
    const float d    = amax / 127.0f;
    qs[std::int64_t(column) * k + i] = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
    if (i % QK8_1 == 0) { ds[std::int64_t(column) * (k / QK8_1) + i / QK8_1] = make_half2(d, sum); }
}

using VectorLaunch = void (*)(const VecArgs&, int, bool, cudaStream_t);

VectorLaunch vector_launch(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return vec_launch<GGML_TYPE_Q8_0>;
    case GgmlType::Q2_K: return vec_launch<GGML_TYPE_Q2_K>;
    case GgmlType::Q3_K: return vec_launch<GGML_TYPE_Q3_K>;
    case GgmlType::Q4_K: return vec_launch<GGML_TYPE_Q4_K>;
    case GgmlType::Q5_K: return vec_launch<GGML_TYPE_Q5_K>;
    case GgmlType::Q6_K: return vec_launch<GGML_TYPE_Q6_K>;
    case GgmlType::IQ2_XXS: return vec_launch<GGML_TYPE_IQ2_XXS>;
    case GgmlType::IQ2_XS: return vec_launch<GGML_TYPE_IQ2_XS>;
    case GgmlType::IQ2_S: return vec_launch<GGML_TYPE_IQ2_S>;
    case GgmlType::IQ3_XXS: return vec_launch<GGML_TYPE_IQ3_XXS>;
    case GgmlType::IQ3_S: return vec_launch<GGML_TYPE_IQ3_S>;
    case GgmlType::IQ1_S: return vec_launch<GGML_TYPE_IQ1_S>;
    case GgmlType::IQ1_M: return vec_launch<GGML_TYPE_IQ1_M>;
    case GgmlType::IQ4_NL: return vec_launch<GGML_TYPE_IQ4_NL>;
    case GgmlType::IQ4_XS: return vec_launch<GGML_TYPE_IQ4_XS>;
    }
    throw std::invalid_argument("gguf vector product: unsupported block type");
}

VecArgs vector_args(const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VecStore& out) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    return VecArgs{static_cast<const std::uint8_t*>(weight),
                   row_bytes,
                   rows,
                   k,
                   reinterpret_cast<const std::int8_t*>(base),
                   reinterpret_cast<const half2*>(base + vec_scales_offset(k, columns)),
                   out};
}

constexpr int kMatrixQuantizeThreads = 128;

template <mmq_q8_1_ds_layout ds_layout>
__global__ void quantize_matrix_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::int32_t* __restrict__ input_columns,
                                       block_q8_1_mmq* __restrict__ y, int k, int columns) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const std::int64_t i0 = (std::int64_t(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= k) { return; }
    const int column             = blockIdx.x;
    const __nv_bfloat16* source  = x + std::int64_t(column) * k;
    float4 xi;
    if (input_columns != nullptr) {
        xi = make_float4(__bfloat162float(source[input_columns[i0 + 0]]),
                         __bfloat162float(source[input_columns[i0 + 1]]),
                         __bfloat162float(source[input_columns[i0 + 2]]),
                         __bfloat162float(source[input_columns[i0 + 3]]));
    } else {
        const uint2 words = *reinterpret_cast<const uint2*>(source + i0);
        const float2 lo   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.x));
        const float2 hi   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.y));
        xi                = make_float4(lo.x, lo.y, hi.x, hi.y);
    }

    const std::int64_t k_block = i0 / QK8_1_MMQ;
    const std::int64_t iqs     = i0 % QK8_1_MMQ;

    float amax = fabsf(xi.x);
    amax       = fmaxf(amax, fabsf(xi.y));
    amax       = fmaxf(amax, fabsf(xi.z));
    amax       = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x           = roundf(xi.x * d_inv);
    q.y           = roundf(xi.y * d_inv);
    q.z           = roundf(xi.z * d_inv);
    q.w           = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;

    const std::int64_t ib = k_block * columns + column;
    reinterpret_cast<char4*>(y[ib].qs)[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) { y[ib].d2s6[iqs / 64] = d; }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

__global__ void store_plane_kernel(const float* __restrict__ in, std::int64_t in_column_stride,
                                   int rows, VecStore out) {
    const int row    = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (row >= rows) { return; }
    vec_store(out, row, column, in[column * in_column_stride + row]);
}

__global__ void gather_columns_kernel(const __nv_bfloat16* __restrict__ x, int k,
                                      const std::int32_t* __restrict__ input_columns,
                                      __nv_bfloat16* __restrict__ out) {
    const int i      = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    out[std::int64_t(column) * k + i] = x[std::int64_t(column) * k + input_columns[i]];
}

cublasHandle_t blas_handle() {
    static std::array<cublasHandle_t, 16> handles{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    if (device < 0 || device >= int(handles.size())) {
        throw std::invalid_argument("gguf: device ordinal out of range");
    }
    std::call_once(once[device], [&] {
        if (cublasCreate(&handles[device]) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("gguf: cublasCreate failed");
        }
    });
    return handles[device];
}

template <ggml_type type, int threads, typename dst_t>
__global__ void dequantize_rows_kernel(const char* __restrict__ weight, std::int64_t row_bytes,
                                       const std::int32_t* __restrict__ row_ids,
                                       dst_t* __restrict__ out, std::int64_t out_row_stride) {
    const int row          = blockIdx.y;
    const std::int64_t src = row_ids != nullptr ? row_ids[row] : row;
    const void* x          = weight + src * row_bytes;
    dst_t* y               = out + std::int64_t(row) * out_row_stride + std::int64_t(blockIdx.x) * QK_K;
    if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0* blocks = static_cast<const block_q8_0*>(x) + blockIdx.x * (QK_K / QK8_0);
        for (int i = threadIdx.x; i < QK_K; i += threads) {
            const block_q8_0& b = blocks[i / QK8_0];
            y[i] = ggml_cuda_cast<dst_t>(__half2float(b.d) * b.qs[i % QK8_0]);
        }
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        dequantize_q2_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        dequantize_q3_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        dequantize_q4_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        dequantize_q5_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        dequantize_q6_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        dequantize_iq2_xxs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        dequantize_iq2_xs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        dequantize_iq2_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        dequantize_iq3_xxs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        dequantize_iq3_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ1_S) {
        dequantize_iq1_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ1_M) {
        dequantize_iq1_m(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        dequantize_iq4_nl(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        dequantize_iq4_xs(x, blockIdx.x, y, threadIdx.x);
    }
}

template <ggml_type type, int threads, typename dst_t>
void launch_dequantize(const void* weight, std::int64_t row_bytes, int k,
                       const std::int32_t* row_ids, int rows, dst_t* out,
                       std::int64_t out_row_stride, cudaStream_t stream) {
    if (k % QK_K != 0 || rows <= 0) {
        throw std::invalid_argument("gguf dequantize: K must be whole 256-value blocks");
    }
    dequantize_rows_kernel<type, threads, dst_t><<<dim3(k / QK_K, rows, 1), threads, 0, stream>>>(
        static_cast<const char*>(weight), row_bytes, row_ids, out, out_row_stride);
    check(cudaGetLastError(), "dequantize launch");
}

template <typename dst_t>
void dispatch_dequantize(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                         const std::int32_t* row_ids, int rows, dst_t* out,
                         std::int64_t out_row_stride, cudaStream_t stream) {
    switch (type) {
    case GgmlType::Q8_0: return launch_dequantize<GGML_TYPE_Q8_0, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q2_K: return launch_dequantize<GGML_TYPE_Q2_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q3_K: return launch_dequantize<GGML_TYPE_Q3_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q4_K: return launch_dequantize<GGML_TYPE_Q4_K, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q5_K: return launch_dequantize<GGML_TYPE_Q5_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q6_K: return launch_dequantize<GGML_TYPE_Q6_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_XXS: return launch_dequantize<GGML_TYPE_IQ2_XXS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_XS: return launch_dequantize<GGML_TYPE_IQ2_XS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_S: return launch_dequantize<GGML_TYPE_IQ2_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ3_XXS: return launch_dequantize<GGML_TYPE_IQ3_XXS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ3_S: return launch_dequantize<GGML_TYPE_IQ3_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ1_S: return launch_dequantize<GGML_TYPE_IQ1_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ1_M: return launch_dequantize<GGML_TYPE_IQ1_M, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ4_NL: return launch_dequantize<GGML_TYPE_IQ4_NL, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ4_XS: return launch_dequantize<GGML_TYPE_IQ4_XS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    }
    throw std::invalid_argument("gguf dequantize: unsupported block type");
}

} // namespace
} // namespace detail

BlockShape block_shape(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return {32, 34};
    case GgmlType::Q2_K: return {256, 84};
    case GgmlType::Q3_K: return {256, 110};
    case GgmlType::Q4_K: return {256, 144};
    case GgmlType::Q5_K: return {256, 176};
    case GgmlType::Q6_K: return {256, 210};
    case GgmlType::IQ2_XXS: return {256, 66};
    case GgmlType::IQ2_XS: return {256, 74};
    case GgmlType::IQ3_XXS: return {256, 98};
    case GgmlType::IQ1_S: return {256, 50};
    case GgmlType::IQ4_NL: return {32, 18};
    case GgmlType::IQ3_S: return {256, 110};
    case GgmlType::IQ2_S: return {256, 82};
    case GgmlType::IQ4_XS: return {256, 136};
    case GgmlType::IQ1_M: return {256, 56};
    }
    throw std::invalid_argument("gguf: unsupported block type");
}

bool has_matrix_kernel(GgmlType type) { return type != GgmlType::IQ1_M; }

int multiprocessor_count() { return detail::device_facts().sm_count; }

std::size_t vector_activation_bytes(int k, int columns) {
    return detail::vec_scales_offset(k, columns) + std::size_t(columns) * (k / QK8_1) * sizeof(half2);
}

void quantize_vector_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out,
                                cudaStream_t stream) {
    if (k <= 0 || k % detail::kQuantizeThreads != 0 || columns <= 0) {
        throw std::invalid_argument("gguf vector activation: K must be whole 256-value groups");
    }
    auto* base = static_cast<std::uint8_t*>(out);
    const dim3 blocks(k / detail::kQuantizeThreads, columns, 1);
    detail::quantize_vector_kernel<<<blocks, detail::kQuantizeThreads, 0, stream>>>(
        x, input_columns, reinterpret_cast<std::int8_t*>(base),
        reinterpret_cast<half2*>(base + detail::vec_scales_offset(k, columns)), k);
    detail::check(cudaGetLastError(), "vector activation launch");
}

void vector_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr)) {
        throw std::invalid_argument("gguf vector product: invalid output");
    }
    const detail::VecStore store{out.bf16,     out.f32,  out.column_stride,
                                 out.epilogue, out.gate, out.gate_column_stride};
    detail::vector_launch(type)(
        detail::vector_args(weight, row_bytes, rows, k, activation, columns, store), columns,
        false, stream);
}

void vector_swiglu(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                   const void* activation, int columns, __nv_bfloat16* out,
                   std::int64_t out_column_stride, cudaStream_t stream) {
    const detail::VecStore store{out, nullptr, out_column_stride, Epilogue::Store, nullptr, 0};
    detail::vector_launch(type)(
        detail::vector_args(weight, row_bytes, rows, k, activation, columns, store), columns,
        true, stream);
}

int matrix_activation_layout(GgmlType type) {
    return static_cast<int>(mmq_get_q8_1_ds_layout(detail::to_ggml(type)));
}

std::size_t matrix_activation_bytes(int k, int columns) {
    return std::size_t(columns) * (k / QK8_1_MMQ) * sizeof(block_q8_1_mmq) +
           detail::kMatrixActivationSlack;
}

void quantize_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out,
                                cudaStream_t stream) {
    if (k <= 0 || k % (4 * detail::kMatrixQuantizeThreads) != 0 || columns <= 0) {
        throw std::invalid_argument("gguf matrix activation: K must be whole 512-value groups");
    }
    const dim3 blocks(columns, (k + 4 * detail::kMatrixQuantizeThreads - 1) /
                                   (4 * detail::kMatrixQuantizeThreads),
                      1);
    auto* y = static_cast<block_q8_1_mmq*>(out);
    switch (mmq_get_q8_1_ds_layout(detail::to_ggml(type))) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(x, input_columns, y, k, columns);
        break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(x, input_columns, y, k, columns);
        break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(x, input_columns, y, k, columns);
        break;
    }
    detail::check(cudaGetLastError(), "matrix activation launch");
}

std::size_t matrix_fixup_bytes(GgmlType type, int rows, int columns) {
    switch (type) {
    case GgmlType::Q8_0: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q8_0>(rows, columns);
    case GgmlType::Q2_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q2_K>(rows, columns);
    case GgmlType::Q3_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q3_K>(rows, columns);
    case GgmlType::Q4_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q4_K>(rows, columns);
    case GgmlType::Q5_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q5_K>(rows, columns);
    case GgmlType::Q6_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q6_K>(rows, columns);
    case GgmlType::IQ2_XXS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_XXS>(rows, columns);
    case GgmlType::IQ2_XS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_XS>(rows, columns);
    case GgmlType::IQ2_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_S>(rows, columns);
    case GgmlType::IQ3_XXS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ3_XXS>(rows, columns);
    case GgmlType::IQ3_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ3_S>(rows, columns);
    case GgmlType::IQ1_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ1_S>(rows, columns);
    case GgmlType::IQ4_NL: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ4_NL>(rows, columns);
    case GgmlType::IQ4_XS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ4_XS>(rows, columns);
    case GgmlType::IQ1_M: break;
    }
    throw std::invalid_argument("gguf matrix product: no integer kernel for this block type");
}

void matrix_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, float* out,
                    std::int64_t out_column_stride, void* fixup, cudaStream_t stream) {
#define NINFER_GGUF_MATRIX_CASE(NAME)                                                              \
    case GgmlType::NAME:                                                                           \
        detail::matrix_product_impl<GGML_TYPE_##NAME>(weight, row_bytes, rows, k, activation,      \
                                                      columns, out, out_column_stride, fixup,      \
                                                      stream);                                     \
        return
    switch (type) {
        NINFER_GGUF_MATRIX_CASE(Q8_0);
        NINFER_GGUF_MATRIX_CASE(Q2_K);
        NINFER_GGUF_MATRIX_CASE(Q3_K);
        NINFER_GGUF_MATRIX_CASE(Q4_K);
        NINFER_GGUF_MATRIX_CASE(Q5_K);
        NINFER_GGUF_MATRIX_CASE(Q6_K);
        NINFER_GGUF_MATRIX_CASE(IQ2_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ2_XS);
        NINFER_GGUF_MATRIX_CASE(IQ2_S);
        NINFER_GGUF_MATRIX_CASE(IQ3_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ3_S);
        NINFER_GGUF_MATRIX_CASE(IQ1_S);
        NINFER_GGUF_MATRIX_CASE(IQ4_NL);
        NINFER_GGUF_MATRIX_CASE(IQ4_XS);
    case GgmlType::IQ1_M:
        break;
    }
#undef NINFER_GGUF_MATRIX_CASE
    throw std::invalid_argument("gguf matrix product: no integer kernel for this block type");
}

void store_plane(const float* in, std::int64_t in_column_stride, int rows, int columns,
                 const VectorOutput& out, cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr) || rows <= 0 ||
        columns <= 0) {
        throw std::invalid_argument("gguf store plane: invalid output");
    }
    const detail::VecStore store{out.bf16,     out.f32,  out.column_stride,
                                 out.epilogue, out.gate, out.gate_column_stride};
    detail::store_plane_kernel<<<dim3((rows + 255) / 256, columns, 1), 256, 0, stream>>>(
        in, in_column_stride, rows, store);
    detail::check(cudaGetLastError(), "store plane launch");
}

void gather_columns(const __nv_bfloat16* x, int k, int columns, const std::int32_t* input_columns,
                    __nv_bfloat16* out, cudaStream_t stream) {
    detail::gather_columns_kernel<<<dim3((k + 255) / 256, columns, 1), 256, 0, stream>>>(
        x, k, input_columns, out);
    detail::check(cudaGetLastError(), "gather columns launch");
}

int dequantized_rows_per_pass(int k, std::size_t scratch_bytes) {
    const std::size_t row = std::size_t(k) * sizeof(__nv_bfloat16);
    return static_cast<int>(std::min<std::size_t>(scratch_bytes / row, 1U << 20) / 128 * 128);
}

void dequantized_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                         const __nv_bfloat16* x, int columns, float* out,
                         std::int64_t out_column_stride, void* scratch, std::size_t scratch_bytes,
                         cudaStream_t stream) {
    const int pass = dequantized_rows_per_pass(k, scratch_bytes);
    if (pass <= 0) { throw std::invalid_argument("gguf dequantized product: scratch too small"); }
    cublasHandle_t blas = detail::blas_handle();
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("gguf: cublasSetStream failed");
    }
    auto* rows_bf16    = static_cast<__nv_bfloat16*>(scratch);
    const float alpha  = 1.0f;
    const float beta   = 0.0f;
    for (int first = 0; first < rows; first += pass) {
        const int count = std::min(pass, rows - first);
        dequantize_rows(type, static_cast<const char*>(weight) + first * row_bytes, row_bytes, k,
                        nullptr, count, rows_bf16, k, stream);
        const cublasStatus_t status = cublasGemmEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, count, columns, k, &alpha, rows_bf16, CUDA_R_16BF, k,
            x, CUDA_R_16BF, k, &beta, out + first, CUDA_R_32F,
            static_cast<int>(out_column_stride), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("gguf dequantized product: cublasGemmEx failed");
        }
    }
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __nv_bfloat16* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, float* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

} // namespace ninfer::ops::gguf
