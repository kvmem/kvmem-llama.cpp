#include "llama-kvmem-stagein.h"
#include "llama-kvmem-transfer.h"
#include "llama-kvmem-gpu.h"


#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int QK8_0 = 32;
constexpr int QK4_0 = 32;

struct __align__(2) block_q8_0 {
    half    d;
    int8_t  qs[QK8_0];
};

struct __align__(2) block_q4_0 {
    half    d;
    uint8_t qs[QK4_0 / 2];
};

static_assert(sizeof(block_q8_0) == 34, "q8_0 block");
static_assert(sizeof(block_q4_0) == 18, "q4_0 block");

constexpr size_t SLAB = (size_t) 32 * 1024 * 1024;
constexpr int GATHER_MAX = 8192;

struct CopyOp {
    const uint8_t * src;
    uint8_t *       dst;
    uint32_t        nbytes;
    uint32_t        pad;
};
static_assert(sizeof(CopyOp) == 24, "CopyOp");

enum ItemKind { ITEM_K = 0, ITEM_V = 1 };

struct Item {
    size_t     off    = 0;
    size_t     nbytes = 0;
    uint8_t *  dst    = nullptr;
    ItemKind   kind   = ITEM_K;
    ggml_type  ty     = GGML_TYPE_Q8_0;
    int64_t    nt     = 0;
    int64_t    n_embd = 0;
    int        nrot   = 0;
    int        n_head = 0;
    int        n_embd_head = 0;
    int        n_rot_rope = 0;
    int32_t    pos0   = 0;
};

struct Stage {
    float *     dev_f32   = nullptr;
    size_t      n_f32     = 0;
    uint8_t *   dev_q     = nullptr;
    size_t      n_q       = 0;
    uint8_t *   pin[2]    = {nullptr, nullptr};
    size_t      pin_n     = 0;
    size_t      used      = 0;
    int         out_cur   = 0;
    cudaEvent_t out_ev[2] = {nullptr, nullptr};
    std::vector<Item> items;
    struct OutItem {
        size_t          off      = 0;
        size_t          nbytes   = 0;
        const uint8_t * gpu_src  = nullptr;
    };
    std::vector<OutItem> outs;
    CopyOp *    dev_ops   = nullptr;
    size_t      n_ops     = 0;
    std::vector<CopyOp> host_ops;
    std::vector<float> theta;
    float *     dev_theta = nullptr;
    size_t      n_theta   = 0;
    cudaEvent_t h2d_ev    = nullptr;
    int64_t *   copy_us     = nullptr;
    int64_t *   rope_us     = nullptr;
    int64_t *   hadamard_us = nullptr;
    int64_t *   set_us      = nullptr;
};

Stage g_st;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "KVMEM stagein %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

cudaStream_t stream() {
    return cudaStreamPerThread;
}

template <int N>
__global__ void fwht_kernel(float * x, int64_t n_rows, float scale) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) {
        return;
    }
    float * row = x + r * (int64_t) N;
    float tmp[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
        tmp[i] = row[i] * scale;
    }
    for (int h = 1; h < N; h *= 2) {
        for (int i = 0; i < N; i += 2 * h) {
            for (int j = 0; j < h; ++j) {
                const float a = tmp[i + j];
                const float b = tmp[i + j + h];
                tmp[i + j]       = a + b;
                tmp[i + j + h]   = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
        row[i] = tmp[i];
    }
}

__global__ void dequant_q8_0(const block_q8_0 * x, float * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float d = __half2float(x[i].d);
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        y[i * QK8_0 + j] = (float) x[i].qs[j] * d;
    }
}

__global__ void dequant_q4_0(const block_q4_0 * x, float * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float d = __half2float(x[i].d);
#pragma unroll
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const int x0 = (x[i].qs[j] & 0x0F) - 8;
        const int x1 = (x[i].qs[j] >> 4) - 8;
        y[i * QK4_0 + j]              = (float) x0 * d;
        y[i * QK4_0 + j + QK4_0 / 2]  = (float) x1 * d;
    }
}

__global__ void rope_neox_kernel(float * x, int64_t n_tokens, int n_head, int n_embd_head,
                                 int n_rot, int32_t pos0, const float * theta) {
    const int t = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    const int h = (int) blockIdx.y;
    if (t >= n_tokens || h >= n_head) {
        return;
    }
    float * row = x + ((int64_t) t * n_head + h) * n_embd_head;
    const int half = n_rot / 2;
    const float pos = (float) (pos0 + t);
    for (int i = 0; i < half; ++i) {
        float s;
        float c;
        sincosf(pos * theta[i], &s, &c);
        const float x0 = row[i];
        const float x1 = row[i + half];
        row[i]        = x0 * c - x1 * s;
        row[i + half] = x0 * s + x1 * c;
    }
}

__global__ void quant_q8_0(const float * x, block_q8_0 * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float * src = x + i * QK8_0;
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        amax = fmaxf(amax, fabsf(src[j]));
    }
    const float d  = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = __float2half(d);
#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        y[i].qs[j] = (int8_t) roundf(src[j] * id);
    }
}

__global__ void quant_q4_0(const float * x, block_q4_0 * y, int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const float * src = x + i * QK4_0;
    float amax = 0.0f;
    float vmax = 0.0f;
#pragma unroll
    for (int j = 0; j < QK4_0; ++j) {
        const float v = src[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }
    const float d  = vmax / -8.0f;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = __float2half(d);
#pragma unroll
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = src[j] * id;
        const float x1 = src[j + QK4_0 / 2] * id;
        int xi0 = (int) (x0 + 8.5f);
        int xi1 = (int) (x1 + 8.5f);
        if (xi0 > 15) {
            xi0 = 15;
        }
        if (xi1 > 15) {
            xi1 = 15;
        }
        y[i].qs[j] = (uint8_t) ((xi0 & 15) | ((xi1 & 15) << 4));
    }
}

__global__ void copy_bytes(const CopyOp * ops, int n) {
    const int i = (int) blockIdx.x;
    if (i >= n) {
        return;
    }
    const CopyOp op = ops[i];
    const uint8_t * s = op.src;
    uint8_t * d = op.dst;
    const uint32_t nbytes = op.nbytes;
    if (!s || !d || nbytes == 0) {
        return;
    }
    const uintptr_t a = (uintptr_t) s | (uintptr_t) d;
    if ((a & 15u) == 0 && (nbytes & 15u) == 0) {
        const uint32_t n4 = nbytes >> 4;
        const uint4 * s4 = (const uint4 *) s;
        uint4 * d4 = (uint4 *) d;
        for (uint32_t j = threadIdx.x; j < n4; j += blockDim.x) {
            d4[j] = s4[j];
        }
        return;
    }
    for (uint32_t j = threadIdx.x; j < nbytes; j += blockDim.x) {
        d[j] = s[j];
    }
}

struct MeanKAcc {
    float * acc = nullptr;
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
};
MeanKAcc g_mk;

__global__ void meank_add_f32(const uint8_t * k, float * acc, int tok0, int n_keep,
                              int n_embd, int ne0, size_t nb0, size_t nb1, size_t nb2) {
    const int d = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (d >= n_embd || ne0 <= 0) {
        return;
    }
    const int head = d / ne0;
    const int dim = d % ne0;
    float s = acc[d];
    for (int t = 0; t < n_keep; ++t) {
        const size_t off = (size_t) (tok0 + t) * nb2 + (size_t) head * nb1 + (size_t) dim * nb0;
        s += *reinterpret_cast<const float *>(k + off);
    }
    acc[d] = s;
}

__global__ void meank_add_f16(const uint8_t * k, float * acc, int tok0, int n_keep,
                              int n_embd, int ne0, size_t nb0, size_t nb1, size_t nb2) {
    const int d = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (d >= n_embd || ne0 <= 0) {
        return;
    }
    const int head = d / ne0;
    const int dim = d % ne0;
    float s = acc[d];
    for (int t = 0; t < n_keep; ++t) {
        const size_t off = (size_t) (tok0 + t) * nb2 + (size_t) head * nb1 + (size_t) dim * nb0;
        s += __half2float(*reinterpret_cast<const half *>(k + off));
    }
    acc[d] = s;
}

__global__ void meank_add_bf16(const uint8_t * k, float * acc, int tok0, int n_keep,
                               int n_embd, int ne0, size_t nb0, size_t nb1, size_t nb2) {
    const int d = (int) blockIdx.x * (int) blockDim.x + (int) threadIdx.x;
    if (d >= n_embd || ne0 <= 0) {
        return;
    }
    const int head = d / ne0;
    const int dim = d % ne0;
    float s = acc[d];
    for (int t = 0; t < n_keep; ++t) {
        const size_t off = (size_t) (tok0 + t) * nb2 + (size_t) head * nb1 + (size_t) dim * nb0;
#if defined(GGML_USE_HIP)
        s += __bfloat162float(*reinterpret_cast<const nv_bfloat16 *>(k + off));
#else
        s += __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(k + off));
#endif
    }
    acc[d] = s;
}

}  // namespace

bool kvmem_stagein_gpu_ready(size_t n_f32, size_t n_packed) {
    if (n_f32 == 0) {
        return false;
    }
    if (!g_st.h2d_ev) {
        if (!cuda_ok(cudaEventCreateWithFlags(&g_st.h2d_ev, cudaEventDisableTiming), "event")) {
            return false;
        }
    }
    if (!g_st.dev_f32 || g_st.n_f32 < n_f32) {
        if (g_st.dev_f32) {
            cudaFree(g_st.dev_f32);
            g_st.dev_f32 = nullptr;
            g_st.n_f32 = 0;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_f32), n_f32 * sizeof(float)),
                     "f32 scratch")) {
            g_st.dev_f32 = nullptr;
            return false;
        }
        g_st.n_f32 = n_f32;
    }
    const size_t want_q = n_packed > SLAB ? n_packed : (n_packed > 0 ? SLAB : 0);
    if (want_q > 0 && (!g_st.dev_q || g_st.n_q < want_q)) {
        if (g_st.dev_q) {
            cudaFree(g_st.dev_q);
            g_st.dev_q = nullptr;
            g_st.n_q = 0;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_q), want_q), "q scratch")) {
            g_st.dev_q = nullptr;
            return false;
        }
        g_st.n_q = want_q;
    }
    if (want_q >= SLAB && (!g_st.pin[0] || !g_st.pin[1] || g_st.pin_n < SLAB)) {
        bool ok = true;
        for (int i = 0; i < 2; ++i) {
            if (g_st.pin[i]) {
                continue;
            }
            if (!cuda_ok(cudaMallocHost(reinterpret_cast<void **>(&g_st.pin[i]), SLAB),
                         "slab pin")) {
                g_st.pin[i] = nullptr;
                ok = false;
                break;
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (g_st.out_ev[i]) {
                continue;
            }
            if (!cuda_ok(cudaEventCreateWithFlags(&g_st.out_ev[i], cudaEventDisableTiming),
                         "stageout event")) {
                ok = false;
                break;
            }
        }
        g_st.pin_n = (ok && g_st.pin[0] && g_st.pin[1]) ? SLAB : 0;
        g_st.used = 0;
        g_st.out_cur = 0;
        g_st.items.clear();
        g_st.outs.clear();
    }
    if (want_q >= SLAB && !g_st.dev_ops) {
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_ops),
                                (size_t) GATHER_MAX * sizeof(CopyOp)),
                     "gather ops")) {
            g_st.dev_ops = nullptr;
        } else {
            g_st.n_ops = (size_t) GATHER_MAX;
        }
    }
    return g_st.dev_f32 != nullptr;
}

void kvmem_stagein_gpu_free() {
    if (g_st.dev_f32) {
        cudaFree(g_st.dev_f32);
        g_st.dev_f32 = nullptr;
    }
    g_st.n_f32 = 0;
    if (g_st.dev_q) {
        cudaFree(g_st.dev_q);
        g_st.dev_q = nullptr;
    }
    g_st.n_q = 0;
    for (int i = 0; i < 2; ++i) {
        if (g_st.pin[i]) {
            cudaFreeHost(g_st.pin[i]);
            g_st.pin[i] = nullptr;
        }
        if (g_st.out_ev[i]) {
            cudaEventDestroy(g_st.out_ev[i]);
            g_st.out_ev[i] = nullptr;
        }
    }
    g_st.pin_n = 0;
    g_st.used = 0;
    g_st.out_cur = 0;
    g_st.items.clear();
    g_st.outs.clear();
    if (g_st.dev_ops) {
        cudaFree(g_st.dev_ops);
        g_st.dev_ops = nullptr;
    }
    g_st.n_ops = 0;
    g_st.host_ops.clear();
    kvmem_meank_free();
    g_st.theta.clear();
    if (g_st.dev_theta) {
        cudaFree(g_st.dev_theta);
        g_st.dev_theta = nullptr;
    }
    g_st.n_theta = 0;
    if (g_st.h2d_ev) {
        cudaEventDestroy(g_st.h2d_ev);
        g_st.h2d_ev = nullptr;
    }
}

bool kvmem_stagein_fwht_ok(int nrot) {
    return nrot == 64 || nrot == 128 || nrot == 256 || nrot == 512;
}

bool kvmem_stagein_quant_ok(ggml_type ty) {
    return ty == GGML_TYPE_Q8_0 || ty == GGML_TYPE_Q4_0;
}

bool kvmem_stagein_h2d_packed(const void * host, size_t n) {
    if (!host || n == 0 || !g_st.dev_q || n > g_st.n_q) {
        return false;
    }
    if (!cuda_ok(kvmem_copy_async(g_st.dev_q, host, n, cudaMemcpyHostToDevice, stream()),
                 "H2D packed")) {
        return false;
    }
    if (!cuda_ok(cudaEventRecord(g_st.h2d_ev, stream()), "H2D packed record")) {
        return false;
    }
    return cuda_ok(cudaEventSynchronize(g_st.h2d_ev), "H2D packed wait");
}

static bool dequant_from(ggml_type ty, const void * src, int64_t n_rows, int64_t n_embd) {
    if (!src || !g_st.dev_f32 || n_rows <= 0 || n_embd <= 0) {
        return false;
    }
    if ((size_t) n_rows * (size_t) n_embd > g_st.n_f32) {
        return false;
    }
    const int threads = 256;
    if (ty == GGML_TYPE_Q8_0) {
        if (n_embd % QK8_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK8_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        dequant_q8_0<<<blocks, threads, 0, stream()>>>(
                static_cast<const block_q8_0 *>(src), g_st.dev_f32, n_blocks);
        return cuda_ok(cudaGetLastError(), "dequant q8_0");
    }
    if (ty == GGML_TYPE_Q4_0) {
        if (n_embd % QK4_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK4_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        dequant_q4_0<<<blocks, threads, 0, stream()>>>(
                static_cast<const block_q4_0 *>(src), g_st.dev_f32, n_blocks);
        return cuda_ok(cudaGetLastError(), "dequant q4_0");
    }
    return false;
}

bool kvmem_stagein_dequant(ggml_type ty, int64_t n_rows, int64_t n_embd) {
    return dequant_from(ty, g_st.dev_q, n_rows, n_embd);
}

bool kvmem_stagein_rope_neox(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                             int32_t pos0, const float * theta, int n_theta) {
    if (!g_st.dev_f32 || !theta || n_tokens <= 0 || n_head <= 0 || n_embd_head <= 0) {
        return false;
    }
    if (n_rot < 2 || (n_rot % 2) != 0 || n_rot > n_embd_head || n_theta != n_rot / 2) {
        return false;
    }
    if ((size_t) n_tokens * (size_t) n_head * (size_t) n_embd_head > g_st.n_f32) {
        return false;
    }
    if (!g_st.dev_theta || g_st.n_theta < (size_t) n_theta) {
        if (g_st.dev_theta) {
            cudaFree(g_st.dev_theta);
            g_st.dev_theta = nullptr;
            g_st.n_theta = 0;
        }
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_theta),
                                (size_t) n_theta * sizeof(float)),
                     "theta")) {
            return false;
        }
        g_st.n_theta = (size_t) n_theta;
    }
    if (!cuda_ok(kvmem_copy_async(g_st.dev_theta, theta, (size_t) n_theta * sizeof(float),
                                 cudaMemcpyHostToDevice, stream()),
                 "H2D theta")) {
        return false;
    }
    const int threads = 64;
    const int blocks_t = (int) ((n_tokens + threads - 1) / threads);
    dim3 grid(blocks_t, n_head, 1);
    rope_neox_kernel<<<grid, threads, 0, stream()>>>(
            g_st.dev_f32, n_tokens, n_head, n_embd_head, n_rot, pos0, g_st.dev_theta);
    return cuda_ok(cudaGetLastError(), "rope launch");
}

bool kvmem_stagein_h2d_f32(const float * host, int64_t n) {
    if (!host || n <= 0 || !g_st.dev_f32 || (size_t) n > g_st.n_f32) {
        return false;
    }
    if (!cuda_ok(kvmem_copy_async(g_st.dev_f32, host, (size_t) n * sizeof(float),
                                 cudaMemcpyHostToDevice, stream()),
                 "H2D f32")) {
        return false;
    }
    if (!cuda_ok(cudaEventRecord(g_st.h2d_ev, stream()), "H2D record")) {
        return false;
    }
    return cuda_ok(cudaEventSynchronize(g_st.h2d_ev), "H2D wait");
}

bool kvmem_stagein_fwht(int64_t n_rows, int64_t n_embd, int nrot) {
    if (!g_st.dev_f32 || n_rows <= 0 || n_embd <= 0 || nrot <= 0) {
        return false;
    }
    if (n_embd % nrot != 0 || !kvmem_stagein_fwht_ok(nrot)) {
        return false;
    }
    const int64_t nrows = n_rows * (n_embd / nrot);
    const float scale = 1.0f / sqrtf((float) nrot);
    const int threads = 128;
    const int blocks = (int) ((nrows + threads - 1) / threads);
    switch (nrot) {
        case 64:
            fwht_kernel<64><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        case 128:
            fwht_kernel<128><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        case 256:
            fwht_kernel<256><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        case 512:
            fwht_kernel<512><<<blocks, threads, 0, stream()>>>(g_st.dev_f32, nrows, scale);
            break;
        default:
            return false;
    }
    return cuda_ok(cudaGetLastError(), "fwht launch");
}

bool kvmem_stagein_quantize(ggml_type ty, void * gpu_dst, int64_t n_rows, int64_t n_embd) {
    if (!g_st.dev_f32 || !gpu_dst || n_rows <= 0 || n_embd <= 0) {
        return false;
    }
    const int threads = 256;
    if (ty == GGML_TYPE_Q8_0) {
        if (n_embd % QK8_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK8_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        quant_q8_0<<<blocks, threads, 0, stream()>>>(
                g_st.dev_f32, static_cast<block_q8_0 *>(gpu_dst), n_blocks);
        return cuda_ok(cudaGetLastError(), "q8_0 launch");
    }
    if (ty == GGML_TYPE_Q4_0) {
        if (n_embd % QK4_0 != 0) {
            return false;
        }
        const int64_t n_blocks = n_rows * (n_embd / QK4_0);
        const int blocks = (int) ((n_blocks + threads - 1) / threads);
        quant_q4_0<<<blocks, threads, 0, stream()>>>(
                g_st.dev_f32, static_cast<block_q4_0 *>(gpu_dst), n_blocks);
        return cuda_ok(cudaGetLastError(), "q4_0 launch");
    }
    return false;
}

bool kvmem_stagein_h2d_bytes(void * gpu_dst, const void * host, size_t n) {
    if (!gpu_dst || !host || n == 0) {
        return n == 0;
    }
    return cuda_ok(kvmem_copy_async(gpu_dst, host, n, cudaMemcpyHostToDevice, stream()),
                   "H2D bytes");
}

void kvmem_stagein_sync() {
    cuda_ok(cudaStreamSynchronize(stream()), "sync");
}

static bool rope_from_dev(int64_t n_tokens, int n_head, int n_embd_head,
                          int n_rot, int32_t pos0) {
    if (!g_st.dev_f32 || !g_st.dev_theta || n_tokens <= 0) {
        return false;
    }
    const int threads = 64;
    const int blocks_t = (int) ((n_tokens + threads - 1) / threads);
    dim3 grid(blocks_t, n_head, 1);
    rope_neox_kernel<<<grid, threads, 0, stream()>>>(
            g_st.dev_f32, n_tokens, n_head, n_embd_head, n_rot, pos0, g_st.dev_theta);
    return cuda_ok(cudaGetLastError(), "rope launch");
}

bool kvmem_stagein_flush(int64_t * copy_us, int64_t * rope_us,
                         int64_t * hadamard_us, int64_t * set_us) {
    if (g_st.used == 0 || g_st.items.empty()) {
        return true;
    }
    if (!g_st.pin[0] || !g_st.dev_q || g_st.used > g_st.n_q) {
        g_st.used = 0;
        g_st.items.clear();
        return false;
    }
    {
        const int64_t t0 = ggml_time_us();
        if (!cuda_ok(kvmem_copy_async(g_st.dev_q, g_st.pin[0], g_st.used,
                                     cudaMemcpyHostToDevice, stream()),
                     "slab H2D")) {
            g_st.used = 0;
            g_st.items.clear();
            return false;
        }
        if (!cuda_ok(cudaEventRecord(g_st.h2d_ev, stream()), "slab H2D record") ||
            !cuda_ok(cudaEventSynchronize(g_st.h2d_ev), "slab H2D wait")) {
            g_st.used = 0;
            g_st.items.clear();
            return false;
        }
        if (set_us) {
            *set_us += ggml_time_us() - t0;
        }
    }
    bool have_k = false;
    for (const Item & it : g_st.items) {
        if (it.kind == ITEM_K) {
            have_k = true;
            break;
        }
    }
    if (have_k && !g_st.theta.empty()) {
        if (!g_st.dev_theta || g_st.n_theta < g_st.theta.size()) {
            if (g_st.dev_theta) {
                cudaFree(g_st.dev_theta);
                g_st.dev_theta = nullptr;
                g_st.n_theta = 0;
            }
            if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_st.dev_theta),
                                    g_st.theta.size() * sizeof(float)),
                         "theta")) {
                g_st.used = 0;
                g_st.items.clear();
                return false;
            }
            g_st.n_theta = g_st.theta.size();
        }
        if (!cuda_ok(kvmem_copy_async(g_st.dev_theta, g_st.theta.data(),
                                     g_st.theta.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream()),
                     "H2D theta")) {
            g_st.used = 0;
            g_st.items.clear();
            return false;
        }
    }
    bool ok = true;
    for (const Item & it : g_st.items) {
        uint8_t * src = g_st.dev_q + it.off;
        if (it.kind == ITEM_V) {
            const int64_t t0 = ggml_time_us();
            ok = cuda_ok(kvmem_copy_async(it.dst, src, it.nbytes,
                                         cudaMemcpyDeviceToDevice, stream()),
                         "slab V D2D");
            if (set_us) {
                *set_us += ggml_time_us() - t0;
            }
            if (!ok) {
                break;
            }
            continue;
        }
        {
            const int64_t t0 = ggml_time_us();
            ok = dequant_from(it.ty, src, it.nt, it.n_embd);
            if (copy_us) {
                *copy_us += ggml_time_us() - t0;
            }
            if (!ok) {
                break;
            }
        }
        {
            const int64_t t0 = ggml_time_us();
            ok = rope_from_dev(it.nt, it.n_head, it.n_embd_head, it.n_rot_rope, it.pos0);
            if (rope_us) {
                *rope_us += ggml_time_us() - t0;
            }
            if (!ok) {
                break;
            }
        }
        if (it.nrot > 0) {
            const int64_t t0 = ggml_time_us();
            ok = kvmem_stagein_fwht(it.nt, it.n_embd, it.nrot);
            if (hadamard_us) {
                *hadamard_us += ggml_time_us() - t0;
            }
            if (!ok) {
                break;
            }
        }
        {
            const int64_t t0 = ggml_time_us();
            ok = kvmem_stagein_quantize(it.ty, it.dst, it.nt, it.n_embd);
            if (set_us) {
                *set_us += ggml_time_us() - t0;
            }
            if (!ok) {
                break;
            }
        }
    }
    g_st.used = 0;
    g_st.items.clear();
    return ok;
}

static bool slab_ready(size_t nbytes) {
    return g_st.pin[0] && g_st.pin_n >= SLAB && g_st.dev_q && g_st.n_q >= SLAB &&
           nbytes > 0 && nbytes <= SLAB;
}

static bool slab_reserve(size_t nbytes, int64_t * set_us) {
    if (!g_st.outs.empty()) {
        return false;
    }
    if (!slab_ready(nbytes)) {
        return false;
    }
    if (g_st.used + nbytes > g_st.pin_n) {
        if (!kvmem_stagein_flush(g_st.copy_us, g_st.rope_us, g_st.hadamard_us,
                                 set_us ? set_us : g_st.set_us)) {
            return false;
        }
    }
    return g_st.used + nbytes <= g_st.pin_n;
}

bool kvmem_stagein_enqueue_k(
        ggml_type ty, const void * packed, size_t nbytes, uint8_t * dst,
        int64_t nt, int64_t n_embd, int nrot,
        int n_head, int n_embd_head, int n_rot_rope, int32_t pos0,
        const float * theta, int n_theta,
        int64_t * copy_us, int64_t * rope_us, int64_t * hadamard_us, int64_t * set_us) {
    if (!packed || !dst || !kvmem_stagein_quant_ok(ty)) {
        return false;
    }
    if (nrot > 0 && !kvmem_stagein_fwht_ok(nrot)) {
        return false;
    }
    if ((int64_t) n_head * n_embd_head != n_embd) {
        return false;
    }
    g_st.copy_us = copy_us;
    g_st.rope_us = rope_us;
    g_st.hadamard_us = hadamard_us;
    g_st.set_us = set_us;
    if (theta && n_theta > 0 && g_st.theta.size() != (size_t) n_theta) {
        g_st.theta.assign(theta, theta + n_theta);
    }
    if (!slab_reserve(nbytes, set_us)) {
        return false;
    }
    std::memcpy(g_st.pin[0] + g_st.used, packed, nbytes);
    Item it;
    it.off = g_st.used;
    it.nbytes = nbytes;
    it.dst = dst;
    it.kind = ITEM_K;
    it.ty = ty;
    it.nt = nt;
    it.n_embd = n_embd;
    it.nrot = nrot;
    it.n_head = n_head;
    it.n_embd_head = n_embd_head;
    it.n_rot_rope = n_rot_rope;
    it.pos0 = pos0;
    g_st.items.push_back(it);
    g_st.used += nbytes;
    return true;
}

bool kvmem_stagein_enqueue_v(const void * packed, size_t nbytes, uint8_t * dst,
                             int64_t * set_us) {
    if (!packed || !dst || nbytes == 0) {
        return false;
    }
    if (set_us) {
        g_st.set_us = set_us;
    }
    if (!slab_reserve(nbytes, set_us)) {
        return false;
    }
    std::memcpy(g_st.pin[0] + g_st.used, packed, nbytes);
    Item it;
    it.off = g_st.used;
    it.nbytes = nbytes;
    it.dst = dst;
    it.kind = ITEM_V;
    g_st.items.push_back(it);
    g_st.used += nbytes;
    return true;
}

bool kvmem_stageout_enqueue(const void * gpu_src, size_t nbytes) {
    if (!gpu_src || nbytes == 0 || !g_st.items.empty()) {
        return false;
    }
    if (!g_st.pin[0] || !g_st.pin[1] || g_st.pin_n < SLAB || !g_st.dev_q ||
        g_st.n_q < SLAB || nbytes > SLAB) {
        return false;
    }
    if (g_st.used + nbytes > g_st.pin_n) {
        return false;
    }
    Stage::OutItem o;
    o.off = g_st.used;
    o.nbytes = nbytes;
    o.gpu_src = static_cast<const uint8_t *>(gpu_src);
    g_st.outs.push_back(o);
    g_st.used += nbytes;
    return true;
}

size_t kvmem_stageout_used() {
    return g_st.used;
}

int kvmem_stageout_submit(int64_t * copy_us) {
    if (g_st.outs.empty()) {
        return -1;
    }
    const int slot = g_st.out_cur;
    uint8_t * dst = (slot == 0 || slot == 1) ? g_st.pin[slot] : nullptr;
    if (!dst || !g_st.dev_q || g_st.used == 0 || g_st.used > g_st.n_q ||
        g_st.used > g_st.pin_n || !g_st.out_ev[slot]) {
        return -1;
    }
    const int64_t t0 = ggml_time_us();
    bool packed = false;
    const int nitem = (int) g_st.outs.size();
    if (nitem > 0 && nitem <= GATHER_MAX && g_st.dev_ops &&
        g_st.n_ops >= (size_t) nitem) {
        g_st.host_ops.resize((size_t) nitem);
        bool ops_ok = true;
        for (int i = 0; i < nitem; ++i) {
            if (g_st.outs[i].nbytes > 0xffffffffu || g_st.outs[i].off > 0xffffffffu) {
                ops_ok = false;
                break;
            }
            g_st.host_ops[i].src = g_st.outs[i].gpu_src;
            g_st.host_ops[i].dst = g_st.dev_q + g_st.outs[i].off;
            g_st.host_ops[i].nbytes = (uint32_t) g_st.outs[i].nbytes;
            g_st.host_ops[i].pad = 0;
        }
        if (ops_ok &&
            cuda_ok(kvmem_copy_async(g_st.dev_ops, g_st.host_ops.data(),
                                    (size_t) nitem * sizeof(CopyOp),
                                    cudaMemcpyHostToDevice, stream()),
                    "gather ops H2D")) {
            copy_bytes<<<nitem, 256, 0, stream()>>>(g_st.dev_ops, nitem);
            packed = cuda_ok(cudaGetLastError(), "gather kernel");
            if (packed) {
                uint64_t bytes = 0;
                for (const auto & item : g_st.outs) bytes += item.nbytes;
                kvmem_record_transfer(cudaMemcpyDeviceToDevice, bytes);
            }
        }
    }
    if (packed) {
        packed = cuda_ok(kvmem_copy_async(dst, g_st.dev_q, g_st.used,
                                         cudaMemcpyDeviceToHost, stream()),
                         "stageout D2H");
    }
    if (!packed) {
        cudaStreamSynchronize(stream());
        for (const Stage::OutItem & o : g_st.outs) {
            if (!cuda_ok(kvmem_copy_async(dst + o.off, o.gpu_src, o.nbytes,
                                         cudaMemcpyDeviceToHost, stream()),
                         "stageout D2H item")) {
                return -1;
            }
        }
    }
    if (!cuda_ok(cudaEventRecord(g_st.out_ev[slot], stream()), "stageout record")) {
        return -1;
    }
    if (copy_us) {
        *copy_us += ggml_time_us() - t0;
    }
    g_st.outs.clear();
    g_st.used = 0;
    g_st.out_cur = 1 - slot;
    return slot;
}

bool kvmem_stageout_wait(int slot, int64_t * copy_us) {
    if (slot < 0) {
        return true;
    }
    if (slot > 1 || !g_st.out_ev[slot]) {
        return false;
    }
    const int64_t t0 = ggml_time_us();
    if (!cuda_ok(cudaEventSynchronize(g_st.out_ev[slot]), "stageout wait")) {
        return false;
    }
    if (copy_us) {
        *copy_us += ggml_time_us() - t0;
    }
    return true;
}

const uint8_t * kvmem_stageout_slot_base(int slot) {
    if (slot < 0 || slot > 1) {
        return nullptr;
    }
    return g_st.pin[slot];
}

void kvmem_stageout_clear() {
    g_st.outs.clear();
    if (g_st.items.empty()) {
        g_st.used = 0;
    }
}

bool kvmem_d2d_batched(const void * const * src, void * const * dst,
                       const size_t * nbytes, int n) {
    if (n <= 0) {
        return true;
    }
    if (!src || !dst || !nbytes || n > GATHER_MAX || !g_st.dev_ops ||
        g_st.n_ops < (size_t) n) {
        return false;
    }
    g_st.host_ops.resize((size_t) n);
    for (int i = 0; i < n; ++i) {
        if (!src[i] || !dst[i] || nbytes[i] == 0 || nbytes[i] > 0xffffffffu) {
            return false;
        }
        g_st.host_ops[i].src = static_cast<const uint8_t *>(src[i]);
        g_st.host_ops[i].dst = static_cast<uint8_t *>(dst[i]);
        g_st.host_ops[i].nbytes = (uint32_t) nbytes[i];
        g_st.host_ops[i].pad = 0;
    }
    if (!cuda_ok(kvmem_copy_async(g_st.dev_ops, g_st.host_ops.data(),
                                 (size_t) n * sizeof(CopyOp),
                                 cudaMemcpyHostToDevice, stream()),
                 "layout ops H2D")) {
        return false;
    }
    copy_bytes<<<n, 256, 0, stream()>>>(g_st.dev_ops, n);
    const bool ok = cuda_ok(cudaGetLastError(), "layout copy kernel");
    if (ok) {
        uint64_t bytes = 0;
        for (int i = 0; i < n; ++i) bytes += nbytes[i];
        kvmem_record_transfer(cudaMemcpyDeviceToDevice, bytes);
    }
    return ok;
}

bool kvmem_meank_ready(uint32_t n_layer, uint32_t n_embd) {
    if (n_layer == 0 || n_embd == 0) {
        return false;
    }
    const size_t need = (size_t) n_layer * n_embd * sizeof(float);
    if (g_mk.acc && g_mk.n_layer == n_layer && g_mk.n_embd == n_embd) {
        return true;
    }
    kvmem_meank_free();
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&g_mk.acc), need), "meank acc")) {
        g_mk.acc = nullptr;
        return false;
    }
    g_mk.n_layer = n_layer;
    g_mk.n_embd = n_embd;
    if (!cuda_ok(cudaMemsetAsync(g_mk.acc, 0, need, stream()), "meank zero")) {
        kvmem_meank_free();
        return false;
    }
    return true;
}

void kvmem_meank_free() {
    if (g_mk.acc) {
        cudaFree(g_mk.acc);
        g_mk.acc = nullptr;
    }
    g_mk.n_layer = 0;
    g_mk.n_embd = 0;
}

void kvmem_meank_zero(uint32_t il) {
    if (!g_mk.acc || il >= g_mk.n_layer) {
        return;
    }
    cudaMemsetAsync(g_mk.acc + (size_t) il * g_mk.n_embd, 0,
                    (size_t) g_mk.n_embd * sizeof(float), stream());
}

bool kvmem_meank_add(uint32_t il, ggml_type ty, const void * gpu_k,
                     uint32_t tok0, uint32_t n_keep, uint32_t n_embd,
                     int64_t ne0, size_t nb0, size_t nb1, size_t nb2) {
    if (!g_mk.acc || !gpu_k || n_keep == 0 || n_embd == 0 || ne0 <= 0 ||
        il >= g_mk.n_layer || n_embd != g_mk.n_embd) {
        return false;
    }
    float * acc = g_mk.acc + (size_t) il * g_mk.n_embd;
    const uint8_t * k = static_cast<const uint8_t *>(gpu_k);
    const int threads = 64;
    const int blocks = ((int) n_embd + threads - 1) / threads;
    if (ty == GGML_TYPE_F32) {
        meank_add_f32<<<blocks, threads, 0, stream()>>>(
                k, acc, (int) tok0, (int) n_keep, (int) n_embd, (int) ne0, nb0, nb1, nb2);
    } else if (ty == GGML_TYPE_F16) {
        meank_add_f16<<<blocks, threads, 0, stream()>>>(
                k, acc, (int) tok0, (int) n_keep, (int) n_embd, (int) ne0, nb0, nb1, nb2);
    } else if (ty == GGML_TYPE_BF16) {
        meank_add_bf16<<<blocks, threads, 0, stream()>>>(
                k, acc, (int) tok0, (int) n_keep, (int) n_embd, (int) ne0, nb0, nb1, nb2);
    } else {
        return false;
    }
    return cuda_ok(cudaGetLastError(), "meank add");
}

bool kvmem_meank_d2h(uint32_t il, float * host, uint32_t n_embd) {
    if (!g_mk.acc || !host || il >= g_mk.n_layer || n_embd != g_mk.n_embd) {
        return false;
    }
    return cuda_ok(kvmem_copy_async(host, g_mk.acc + (size_t) il * g_mk.n_embd,
                                   (size_t) n_embd * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream()),
                   "meank D2H");
}
