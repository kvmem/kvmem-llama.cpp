#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"

#include "core/device.h"
#include "core/weight_view.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroup   = 64;
// How much of the channel-magnitude spread moves from the activations into the weights. 0.5 is the
// usual choice: it equalises the two sides rather than fully flattening either. 0 disables it and
// makes every kernel below compute exactly what it computed before this existed.
constexpr float kChannelEqualisationAlpha = 0.5F;
constexpr int kThreads = 256;

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS failure in ") + what + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// One handle per device, created on first use and never destroyed: cuBLAS handles are not cheap to
// build, the op is called once per linear per chunk, and the process owns the device for its
// lifetime. The stream is set per call because the handle is shared.
cublasHandle_t handle_for_current_device() {
    static constexpr int kMaxDevices = 8;
    static cublasHandle_t handles[kMaxDevices]{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("cuBLAS prefill: device index out of range");
    }
    if (!handles[device]) { check_blas(cublasCreate(&handles[device]), "cublasCreate"); }
    return handles[device];
}

// One block per weight row. Two things keep this off the critical path, which the first cut was
// squarely on -- a scalar two-pass decode cost more than the GEMM it was feeding for the narrow
// shapes.
//
// First, the row scale is derived from the group scales alone rather than from a pass over the
// values: a Q4 code is in [-8, 7] and a Q5 code in [-16, 15], so `max_g(scale[g]) * half` bounds the
// row's absmax without reading a single code. The bound is nearly tight, because the group scale
// was itself chosen from the group's absmax -- it can only be loose when a group's largest
// magnitude is below its own limit, which costs a fraction of a bit.
//
// Second, the decode is vectorised: one thread takes the eight codes packed in a uint32 and writes
// them as eight bytes, so the pass is a straight stream rather than per-nibble addressing.
template <bool kHasHigh>
__global__ void dequantise_row_to_int8(const std::uint8_t* __restrict__ codes,
                                       const std::uint8_t* __restrict__ high,
                                       const __half* __restrict__ scales, int groups,
                                       const float* __restrict__ channel_scale,
                                       const float* __restrict__ group_peak,
                                       std::int8_t* __restrict__ out,
                                       float* __restrict__ row_scale) {
    const int row             = blockIdx.x;
    const int k               = groups * kGroup;
    const auto* c_row         = reinterpret_cast<const std::uint32_t*>(
        codes + static_cast<std::size_t>(row) * (k / 2));
    const std::uint8_t* h_row =
        kHasHigh ? high + static_cast<std::size_t>(row) * groups * 8 : nullptr;
    const __half* s_row = scales + static_cast<std::size_t>(row) * groups;
    std::int8_t* o_row  = out + static_cast<std::size_t>(row) * k;

    __shared__ float s_max[kThreads / 32];
    float local = 0.0F;
    for (int g = threadIdx.x; g < groups; g += kThreads) {
        local = fmaxf(local, fabsf(__half2float(s_row[g])) * group_peak[g]);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if ((threadIdx.x & 31) == 0) { s_max[threadIdx.x >> 5] = local; }
    __syncthreads();
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float peak = 0.0F;
        for (int w = 0; w < kThreads / 32; ++w) { peak = fmaxf(peak, s_max[w]); }
        const float bound = peak * (kHasHigh ? 16.0F : 8.0F);
        s_scale           = bound > 0.0F ? bound / 127.0F : 1.0F;
        row_scale[row]    = s_scale;
    }
    __syncthreads();
    const float inv = 1.0F / s_scale;

    // Eight codes per iteration: one uint32 of nibbles, one byte of high bits when present.
    const int words = k / 8;
    for (int w = threadIdx.x; w < words; w += kThreads) {
        const std::uint32_t packed = c_row[w];
        const int base             = w * 8;
        const float group          = __half2float(s_row[base / kGroup]) * inv;
        const std::uint32_t hbits  = kHasHigh ? h_row[(base / kGroup) * 8 + (base % kGroup) / 8] : 0u;
        std::int8_t bytes[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t nibble = (packed >> (j * 4)) & 0xfu;
            int code                   = static_cast<int>(nibble);
            if (kHasHigh) { code |= static_cast<int>((hbits >> j) & 1u) << 4; }
            const int half = kHasHigh ? 16 : 8;
            bytes[j]       = static_cast<std::int8_t>(
                fminf(fmaxf(rintf(static_cast<float>((code ^ half) - half) * group *
                                  channel_scale[base + j]),
                            -127.0F),
                      127.0F));
        }
        reinterpret_cast<std::uint32_t*>(o_row + base)[0] =
            *reinterpret_cast<const std::uint32_t*>(bytes);
        reinterpret_cast<std::uint32_t*>(o_row + base)[1] =
            *reinterpret_cast<const std::uint32_t*>(bytes + 4);
    }
}

// --- activation channel equalisation --------------------------------------------------------
//
// One scale per token is what cuBLAS forces, and it is the expensive half of this route's quality
// cost: activation outliers concentrate in a few input channels, so a token's absmax is set by
// those channels and every other channel is quantised against a step far larger than it needs.
// Measured at 1.2e-2 typically but 1.29e-1 on outlier-heavy inputs.
//
// The fix is exact rather than approximate. For a per-channel s, X[j,t]/s[j] paired with
// W[i,j]*s[j] leaves X@W unchanged, so choosing s to flatten the channel magnitudes costs nothing
// in the product while making the per-token quantisation of X far more accurate. The weight side is
// free here because this route already rewrites every weight byte on its way to int8; it is only
// affordable at all because of that.
//
// s[j] = (a[j] / geomean(a))^alpha with a[j] the channel's absmax over the chunk, so s is centred
// on 1 and alpha picks how much of the spread moves. alpha = 0 disables it exactly.
__global__ void fill_ones(float* __restrict__ out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = 1.0F; }
}

// One thread per channel walking every token leaves k/256 blocks -- twenty of them for this model,
// on an eighty-two SM card -- so the grid is two-dimensional and the token axis is split across
// blocks that combine with an atomic. For non-negative floats the IEEE bit pattern is monotonic, so
// an integer atomicMax is an atomicMax on the value.
__global__ void channel_absmax(const __nv_bfloat16* __restrict__ x, int k, int tokens,
                               float* __restrict__ absmax) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) { return; }
    float peak = 0.0F;
    for (int t = blockIdx.y; t < tokens; t += gridDim.y) {
        peak = fmaxf(peak, fabsf(__bfloat162float(x[static_cast<std::size_t>(t) * k + j])));
    }
    atomicMax(reinterpret_cast<unsigned*>(absmax) + j, __float_as_uint(peak));
}

// A sum of logs rather than a product: k is thousands of terms and the product underflows.
//
// One block, and a tree rather than an atomic, because this has to be reproducible. Float atomicAdd
// commits in whatever order the blocks finish, so the geometric mean below -- and through it every
// channel scale, and through those every rounding decision in the quantiser -- came out slightly
// different on each run. That moved perplexity by ~0.04% between identical runs, which is a third
// of what the whole route costs, and made greedy output non-reproducible.
__global__ void channel_log_sum(const float* __restrict__ absmax, int k,
                                float* __restrict__ log_sum) {
    __shared__ float partial[kThreads];
    float sum = 0.0F;
    for (int j = static_cast<int>(threadIdx.x); j < k; j += kThreads) {
        sum += logf(fmaxf(absmax[j], 1e-20F));
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(stride)) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { *log_sum = partial[0]; }
}

__global__ void channel_scales(const float* __restrict__ absmax, const float* __restrict__ log_sum,
                               int k, float alpha, float* __restrict__ scale) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) { return; }
    const float geo = __expf(*log_sum / static_cast<float>(k));
    const float a   = fmaxf(absmax[j], 1e-20F);
    scale[j]        = __powf(a / fmaxf(geo, 1e-20F), alpha);
}

// The weight's row bound has to see the channel scales, since W is multiplied by them: the tight
// bound per group is its own fp16 scale times the largest channel scale inside it.
__global__ void group_scale_peaks(const float* __restrict__ channel_scale, int groups,
                                  float* __restrict__ peak) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) { return; }
    float best = 0.0F;
    for (int j = 0; j < kGroup; ++j) { best = fmaxf(best, channel_scale[g * kGroup + j]); }
    peak[g] = best;
}

// One block per token. x is [k, t] with k contiguous, so a token is one contiguous column.
__global__ void quantise_tokens_to_int8(const __nv_bfloat16* __restrict__ x, int k,
                                        const float* __restrict__ channel_scale,
                                        std::int8_t* __restrict__ out,
                                        float* __restrict__ token_scale) {
    const int token             = blockIdx.x;
    const __nv_bfloat16* x_col  = x + static_cast<std::size_t>(token) * k;
    std::int8_t* o_col          = out + static_cast<std::size_t>(token) * k;

    __shared__ float s_absmax[kThreads / 32];
    float local = 0.0F;
    for (int i = threadIdx.x; i < k; i += kThreads) {
        local = fmaxf(local, fabsf(__bfloat162float(x_col[i]) / channel_scale[i]));
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if ((threadIdx.x & 31) == 0) { s_absmax[threadIdx.x >> 5] = local; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float absmax = 0.0F;
        for (int w = 0; w < kThreads / 32; ++w) { absmax = fmaxf(absmax, s_absmax[w]); }
        s_absmax[0]        = absmax;
        token_scale[token] = absmax > 0.0F ? absmax / 127.0F : 1.0F;
    }
    __syncthreads();
    const float inv = s_absmax[0] > 0.0F ? 127.0F / s_absmax[0] : 0.0F;
    for (int i = threadIdx.x; i < k; i += kThreads) {
        o_col[i] = static_cast<std::int8_t>(fminf(
            fmaxf(rintf(__bfloat162float(x_col[i]) / channel_scale[i] * inv), -127.0F), 127.0F));
    }
}

__device__ __forceinline__ float silu(float v) { return v / (1.0F + __expf(-v)); }

// C is [n, t] int32 column-major with ld = n; gate row i pairs with up row i + out_rows.
__global__ void swiglu_epilogue(const int* __restrict__ c, const float* __restrict__ row_scale,
                                const float* __restrict__ token_scale, int n, int out_rows,
                                int tokens, int token_base, __nv_bfloat16* __restrict__ out,
                                int out_ld) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= out_rows * tokens) { return; }
    const int row   = index % out_rows;
    const int token = index / out_rows;
    const float ts  = token_scale[token_base + token];
    const float gate =
        static_cast<float>(c[static_cast<std::size_t>(token) * n + row]) * row_scale[row] * ts;
    const float up = static_cast<float>(c[static_cast<std::size_t>(token) * n + row + out_rows]) *
                     row_scale[row + out_rows] * ts;
    out[static_cast<std::size_t>(token_base + token) * out_ld + row] =
        __float2bfloat16(silu(gate) * up);
}

// dst[r, t] = C[row_begin + r, t] rescaled. One destination of a split parent.
__global__ void store_epilogue(const int* __restrict__ c, const float* __restrict__ row_scale,
                               const float* __restrict__ token_scale, int n, int row_begin,
                               int rows, int tokens, int token_base, int leading, int dst_begin,
                               __nv_bfloat16* __restrict__ out) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * tokens) { return; }
    const int row   = index % rows;
    const int token = index / rows;
    const float value =
        static_cast<float>(c[static_cast<std::size_t>(token) * n + row_begin + row]) *
        row_scale[row_begin + row] * token_scale[token_base + token];
    // The GDN input projection packs two parents' outputs into one tensor, so a destination is a
    // row range of something wider rather than a tensor of its own.
    out[static_cast<std::size_t>(token_base + token) * leading + dst_begin + row] =
        __float2bfloat16(value);
}

__global__ void add_epilogue(const int* __restrict__ c, const float* __restrict__ row_scale,
                             const float* __restrict__ token_scale, int n, int tokens,
                             int token_base, __nv_bfloat16* __restrict__ residual, int out_ld) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * tokens) { return; }
    const int row    = index % n;
    const int token  = index / n;
    const float value = static_cast<float>(c[static_cast<std::size_t>(token) * n + row]) *
                        row_scale[row] * token_scale[token_base + token];
    __nv_bfloat16* slot = residual + static_cast<std::size_t>(token_base + token) * out_ld + row;
    *slot               = __float2bfloat16(__bfloat162float(*slot) + value);
}

} // namespace

std::int32_t cublas_token_tile(std::int32_t rows, std::int32_t tokens) {
    const auto per_token = static_cast<std::size_t>(rows) * sizeof(int);
    auto tile            = static_cast<std::int32_t>(kCublasTileBudgetBytes / per_token);
    tile                 = (tile / 128) * 128;              // keep tiles a whole number of warps' work
    tile                 = std::max(tile, 128);
    return std::min(tile, tokens);
}

namespace {

struct Scratch {
    std::int8_t* w8      = nullptr;
    float* row_scale     = nullptr;
    std::int8_t* x8      = nullptr;
    float* token_scale   = nullptr;
    int* c32             = nullptr;
    float* channel_scale = nullptr;
    float* group_peak    = nullptr;
    float* reduction     = nullptr; // one float: the log-sum behind the geometric mean
};

std::size_t aligned_256(std::size_t bytes) { return ((bytes + 255) / 256) * 256; }

Scratch take_scratch(WorkspaceArena& ws, std::int32_t n, std::int32_t k, std::int32_t tokens) {
    const auto tile = static_cast<std::size_t>(cublas_token_tile(n, tokens));
    Scratch s;
    s.w8 = static_cast<std::int8_t*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(n) * k)).data);
    s.row_scale = static_cast<float*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(n) * sizeof(float))).data);
    s.x8 = static_cast<std::int8_t*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(k) * tokens)).data);
    s.token_scale = static_cast<float*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(tokens) * sizeof(float))).data);
    s.c32 = static_cast<int*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(n) * tile * sizeof(int))).data);
    s.channel_scale = static_cast<float*>(
        ws.alloc_bytes(aligned_256(static_cast<std::size_t>(k) * 2 * sizeof(float))).data);
    s.group_peak = s.channel_scale + k; // the absmax pass borrows this before the peaks need it
    s.reduction  = static_cast<float*>(ws.alloc_bytes(aligned_256(sizeof(float))).data);
    return s;
}

// Materialise the weight once, quantise the activations once, then walk the tokens in tiles.
// The activation half: channel equalisation and the per-token quantisation. Depends only on x and
// k, so parents sharing an input share this.
void prepare_activations(const Tensor& x, std::int32_t tokens, std::int32_t k, const Scratch& s,
                         cudaStream_t stream) {
    const int groups = k / kGroup;
    // Channel equalisation first: the weight dequantise and the token quantise both consume it.
    const float alpha = kChannelEqualisationAlpha;
    const int chan_blocks = (k + 255) / 256;
    if (alpha > 0.0F) {
        CUDA_CHECK(cudaMemsetAsync(s.group_peak, 0, static_cast<std::size_t>(k) * sizeof(float),
                                   stream));
        // Enough token slices to fill the card, but never more than there are tokens.
        const int token_slices = std::min(tokens, 64);
        const dim3 absmax_grid(static_cast<unsigned>(chan_blocks),
                               static_cast<unsigned>(token_slices));
        channel_absmax<<<absmax_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), k, tokens, s.group_peak);
        channel_log_sum<<<1, kThreads, 0, stream>>>(s.group_peak, k, s.reduction);
        channel_scales<<<chan_blocks, 256, 0, stream>>>(s.group_peak, s.reduction, k, alpha,
                                                        s.channel_scale);
    } else {
        // A scale of one everywhere is the identity, so the kernels below need no second form.
        fill_ones<<<chan_blocks, 256, 0, stream>>>(s.channel_scale, k);
    }
    group_scale_peaks<<<(groups + 255) / 256, 256, 0, stream>>>(s.channel_scale, groups,
                                                                s.group_peak);
    CUDA_CHECK(cudaGetLastError());

    quantise_tokens_to_int8<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), k, s.channel_scale, s.x8, s.token_scale);
    CUDA_CHECK(cudaGetLastError());
}

// The weight half. Separate because a split parent is materialised once and read by several
// destinations, and because parents sharing an input share prepare_activations above.
void materialise_weight(const Weight& weight, const Scratch& s, cudaStream_t stream) {
    const auto n     = weight.n;
    const auto k     = weight.k;
    const int groups = k / kGroup;
    auto* codes      = static_cast<const std::uint8_t*>(weight.qdata);
    auto* high       = static_cast<const std::uint8_t*>(weight.qhigh);
    auto* scales     = static_cast<const __half*>(weight.scales);
    if (weight.qtype == QType::Q5_G64_FP16) {
        dequantise_row_to_int8<true><<<n, kThreads, 0, stream>>>(
            codes, high, scales, groups, s.channel_scale, s.group_peak, s.w8, s.row_scale);
    } else {
        dequantise_row_to_int8<false><<<n, kThreads, 0, stream>>>(
            codes, nullptr, scales, groups, s.channel_scale, s.group_peak, s.w8, s.row_scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

void prepare(const Weight& weight, const Tensor& x, std::int32_t tokens, const Scratch& s,
             cudaStream_t stream) {
    prepare_activations(x, tokens, weight.k, s, stream);
    materialise_weight(weight, s, stream);
}

void gemm_tile(cublasHandle_t blas, const Scratch& s, std::int32_t n, std::int32_t k,
               std::int32_t token_base, std::int32_t tile) {
    const int alpha = 1;
    const int beta  = 0;
    check_blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, tile, k, &alpha, s.w8, CUDA_R_8I, k,
                            s.x8 + static_cast<std::size_t>(token_base) * k, CUDA_R_8I, k, &beta,
                            s.c32, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
}

} // namespace

bool w4_cublas_prefill_supported(const Weight& weight, std::int32_t tokens) {
    return tokens > 0 && is_row_split(weight.layout) && weight.group == kGroup &&
           weight.k % kGroup == 0 && weight.qdata != nullptr && weight.scales != nullptr &&
           weight.scale_dtype == DType::FP16 &&
           (weight.qtype == QType::Q4_G64_FP16 ||
            (weight.qtype == QType::Q5_G64_FP16 && weight.qhigh != nullptr));
}

std::size_t w4_cublas_prefill_workspace_capacity_bytes(std::int32_t rows, std::int32_t cols,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (rows <= 0 || cols <= 0 || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("cuBLAS prefill workspace: invalid interval");
    }
    const auto n    = static_cast<std::size_t>(rows);
    const auto k    = static_cast<std::size_t>(cols);
    const auto t    = static_cast<std::size_t>(max_tokens);
    const auto tile = static_cast<std::size_t>(cublas_token_tile(rows, max_tokens));
    return aligned_256(n * k) + aligned_256(n * sizeof(float)) + aligned_256(k * t) +
           aligned_256(t * sizeof(float)) + aligned_256(n * tile * sizeof(int)) +
           aligned_256(k * 2 * sizeof(float)) + aligned_256(sizeof(float));
}

void w4_cublas_swiglu_launch(const Tensor& x, const Weight& gate_up, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t n      = gate_up.n;
    const std::int32_t k      = gate_up.k;
    const std::int32_t out_rows = out.ne[0];
    if (!w4_cublas_prefill_supported(gate_up, tokens) || out_rows * 2 != n) {
        throw std::invalid_argument("cuBLAS prefill swiglu: unsupported shape");
    }
    const auto scope = workspace.scope();
    const Scratch s  = take_scratch(workspace, n, k, tokens);
    prepare(gate_up, x, tokens, s, stream);

    cublasHandle_t blas = handle_for_current_device();
    check_blas(cublasSetStream(blas, stream), "cublasSetStream");
    const std::int32_t step = cublas_token_tile(n, tokens);
    for (std::int32_t base = 0; base < tokens; base += step) {
        const std::int32_t tile = std::min(step, tokens - base);
        gemm_tile(blas, s, n, k, base, tile);
        const int total  = out_rows * tile;
        const int blocks = (total + 255) / 256;
        swiglu_epilogue<<<blocks, 256, 0, stream>>>(
            s.c32, s.row_scale, s.token_scale, n, out_rows, tile, base,
            static_cast<__nv_bfloat16*>(out.data), out_rows);
        CUDA_CHECK(cudaGetLastError());
    }
}

void w4_cublas_add_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    const std::int32_t n      = weight.n;
    const std::int32_t k      = weight.k;
    if (!w4_cublas_prefill_supported(weight, tokens) || residual.ne[0] != n) {
        throw std::invalid_argument("cuBLAS prefill add: unsupported shape");
    }
    const auto scope = workspace.scope();
    const Scratch s  = take_scratch(workspace, n, k, tokens);
    prepare(weight, x, tokens, s, stream);

    cublasHandle_t blas = handle_for_current_device();
    check_blas(cublasSetStream(blas, stream), "cublasSetStream");
    const std::int32_t step = cublas_token_tile(n, tokens);
    for (std::int32_t base = 0; base < tokens; base += step) {
        const std::int32_t tile = std::min(step, tokens - base);
        gemm_tile(blas, s, n, k, base, tile);
        const int total  = n * tile;
        const int blocks = (total + 255) / 256;
        add_epilogue<<<blocks, 256, 0, stream>>>(s.c32, s.row_scale, s.token_scale, n, tile, base,
                                                 static_cast<__nv_bfloat16*>(residual.data), n);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool w4_cublas_projection_supported(const CublasProjection* parents, int parent_count,
                                   std::int32_t tokens) {
    if (parents == nullptr || parent_count <= 0) { return false; }
    const std::int32_t k = parents[0].weight != nullptr ? parents[0].weight->k : 0;
    for (int p = 0; p < parent_count; ++p) {
        const auto& parent = parents[p];
        if (parent.weight == nullptr || parent.destination_count <= 0 ||
            parent.destinations == nullptr) {
            return false;
        }
        // One activation quantisation serves every parent, so they must share an input width.
        if (parent.weight->k != k) { return false; }
        if (!w4_cublas_prefill_supported(*parent.weight, tokens)) { return false; }
        for (int d = 0; d < parent.destination_count; ++d) {
            const auto& dst = parent.destinations[d];
            if (dst.data == nullptr || dst.rows <= 0 || dst.row_begin < 0 ||
                dst.row_begin + dst.rows > parent.weight->n || dst.begin < 0 ||
                dst.begin + dst.rows > dst.leading) {
                return false;
            }
        }
    }
    return true;
}

std::size_t w4_cublas_projection_workspace_capacity_bytes(std::int32_t max_rows, std::int32_t cols,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    return w4_cublas_prefill_workspace_capacity_bytes(max_rows, cols, min_tokens, max_tokens);
}

void w4_cublas_projection_launch(const Tensor& x, const CublasProjection* parents, int parent_count,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!w4_cublas_projection_supported(parents, parent_count, tokens)) {
        throw std::invalid_argument("cuBLAS projection: unsupported profile");
    }
    const std::int32_t k = parents[0].weight->k;
    std::int32_t max_rows = 0;
    for (int p = 0; p < parent_count; ++p) {
        max_rows = std::max(max_rows, parents[p].weight->n);
    }

    const auto scope = workspace.scope();
    const Scratch s  = take_scratch(workspace, max_rows, k, tokens);
    prepare_activations(x, tokens, k, s, stream);

    cublasHandle_t blas = handle_for_current_device();
    check_blas(cublasSetStream(blas, stream), "cublasSetStream");
    // One tile width for every parent, taken from the widest: the int32 output buffer is shared, and
    // a narrower parent would otherwise be handed a larger tile than that buffer was sized for.
    const std::int32_t step = cublas_token_tile(max_rows, tokens);

    for (int p = 0; p < parent_count; ++p) {
        const auto& parent      = parents[p];
        const std::int32_t n    = parent.weight->n;
        materialise_weight(*parent.weight, s, stream);
        for (std::int32_t base = 0; base < tokens; base += step) {
            const std::int32_t tile = std::min(step, tokens - base);
            gemm_tile(blas, s, n, k, base, tile);
            for (int d = 0; d < parent.destination_count; ++d) {
                const auto& dst  = parent.destinations[d];
                const int total  = dst.rows * tile;
                const int blocks = (total + 255) / 256;
                store_epilogue<<<blocks, 256, 0, stream>>>(
                    s.c32, s.row_scale, s.token_scale, n, dst.row_begin, dst.rows, tile, base,
                    dst.leading, dst.begin, static_cast<__nv_bfloat16*>(dst.data));
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

} // namespace ninfer::ops::detail
