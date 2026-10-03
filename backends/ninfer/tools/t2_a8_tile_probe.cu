// A candidate replacement for the ternary prefill mainloop, measured against the shipped one.
//
// tools/t2_cublas_probe.cu put the shipped T2 integer GEMM (rowsplit_a8_mma.cuh with T2Codec) at
// 74-94 TOP/s on the RTX 5060 Ti, where cuBLAS runs the same int8 product at 170-184. Reading the
// shipped kernel against the ternary shapes gives four suspects, each a knob here:
//
//   scale      the shipped contract scales activations per (token, 64 k), so every two MMAs the
//              int32 partial is converted and multiplied by two scales. Here the partial runs the
//              whole 128-k weight group, and the activation scale is either per (token, 128 k) or
//              per token, the latter folded into the epilogue. The conversion is a float add: the
//              accumulator starts at the bits of 1.5 * 2^23, so its bits read as that plus the dot.
//   tile       the shipped T2 tile is 64 rows, and every such block re-streams its activation band.
//              Ternary rows are a quarter of an int8 one, so the band, not the weight, is the
//              traffic that matters, and taller tiles cut it.
//   stages     the shipped ring is two stages of 64 k with two barriers each; here 128 k per stage
//              and one barrier.
//   decode     thread tig of an MMA quad owns k in [32 tig, 32 tig + 32) of each 128-k group rather
//              than the interleaved quarters of the fragment. The activations then sit as a plain
//              [T][K] int8 plane (the quantiser owns that layout), and a thread's codes for one row
//              are one 8-byte shared load, decoded two bytes at a time through PRMT.
//
// Build (inside the build container, after the sm120a tree is built):
//   nvcc -O3 -std=c++20 -arch=sm_120a -DNINFER_SM8X_COMPAT=1 -Iinclude -Isrc -Xptxas -v \
//        tools/t2_a8_tile_probe.cu build-sm120a/src/ops/libninfer_ops.a \
//        build-sm120a/src/core/libninfer_core.a -lcublas -L/usr/local/cuda/lib64/stubs -lcuda \
//        -o build-sm120a/t2_a8_tile_probe
// Run: t2_a8_tile_probe [tokens...]   (default 1024 2048 4096)

#include "core/arena.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t status_ = (x);                                                                 \
        if (status_ != cudaSuccess) {                                                              \
            std::printf("CUDA error %s at line %d\n", cudaGetErrorString(status_), __LINE__);      \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

using namespace ninfer;

constexpr int kGroupK   = 128; // one weight scale group, one pipeline stage
constexpr int kXStride  = 144; // a column's 128 bytes padded so an MMA quad's rows miss each other
constexpr int kRing     = 8;   // weight scale groups per 16-byte copy
constexpr float kMagic  = 12582912.0f; // 1.5 * 2^23
constexpr int kMagicBits = 0x4B400000;

enum Scale : int { kPerToken = 0, kPerGroup = 1 };

__device__ __forceinline__ void mma_s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                       unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(addr));
    return r;
}

__device__ __forceinline__ uint2 lds64(const void* p) {
    uint2 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "r"(addr));
    return r;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr), "l"(gmem));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void wait_group() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// Bytes q and q+1 of a code word (four 2-bit codes each, lowest k lowest) to two registers of four
// s8 values. PRMT picks each output byte from the table {0, +1, -, -1} by a nibble selector, so the
// work is spreading the eight 2-bit fields to eight nibbles.
__device__ __forceinline__ void decode_pair(unsigned word, int q, unsigned& lo, unsigned& hi) {
    constexpr unsigned kTable = 0xFF000100u;
    const unsigned sel = static_cast<unsigned>(q) | (4u << 4) | (static_cast<unsigned>(q + 1) << 8) |
                         (4u << 12);
    unsigned p = __byte_perm(word, 0u, sel);        // byte q -> bits 0-7, byte q+1 -> bits 16-23
    p          = (p | (p << 4)) & 0x0F0F0F0Fu;      // fields 0,1 | 2,3 into separate bytes
    p          = (p | (p << 2)) & 0x33333333u;      // one field per nibble
    lo         = __byte_perm(kTable, 0u, p & 0xFFFFu);
    hi         = __byte_perm(kTable, 0u, p >> 16);
}

__device__ __forceinline__ void decode_word_spread(unsigned word, unsigned (&d)[4]) {
    decode_pair(word, 0, d[0], d[1]);
    decode_pair(word, 2, d[2], d[3]);
}

// The same word in nine instructions instead of eighteen, by leaving the codes where they are.
// Masked with 0x3333, every nibble of the word holds one even code with zeros above it, which is a
// PRMT selector as it stands; shifted by two, the odd codes. PRMT reads only the low 16 bits of the
// selector, so each masked word serves two registers. The registers come out as k {0,2,4,6},
// {1,3,5,7}, {8,10,12,14}, {9,11,13,15} of the word, and the quantiser stores activations in that
// order (kPermute).
__device__ __forceinline__ void decode_word(unsigned word, unsigned (&d)[4]) {
    constexpr unsigned kTable = 0xFF000100u;
    const unsigned even = word & 0x33333333u;
    const unsigned odd  = (word >> 2) & 0x33333333u;
    d[0]                = __byte_perm(kTable, 0u, even);
    d[1]                = __byte_perm(kTable, 0u, odd);
    d[2]                = __byte_perm(kTable, 0u, even >> 16);
    d[3]                = __byte_perm(kTable, 0u, odd >> 16);
}

// Where k (mod 16) of a code word lands in the activation row, for decode_word.
__host__ __device__ constexpr int permute16(int k) {
    return (k / 8) * 8 + (k % 2) * 4 + (k % 8) / 2;
}

template <int WM, int WN, int MT, int NT, int STAGES, int SCALE>
struct Config {
    static constexpr int kWarps   = WM * WN;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int BM       = WM * MT * 16;
    static constexpr int BN       = WN * NT * 8;
    static constexpr int kWBytes  = BM * 32;
    static constexpr int kXBytes  = BN * kXStride;
    static constexpr int kSBytes  = SCALE == kPerGroup ? (BN * 2 + 15) / 16 * 16 : 0;
    static constexpr int kStage   = kWBytes + kXBytes + kSBytes;
    static constexpr int kRingBytes = BM * kRing * 2;
    static constexpr int kSmem      = STAGES * kStage + 2 * kRingBytes;
};

// D[row, t] = sum_k W[row, k] X[k, t]. W: T2 row-split codes [N][K/4] and FP16 scales [N][K/128];
// X: s8 [Tpad][K]; xs: per-group FP16 [K/128][Tpad] or per-token FP32 [Tpad]. out: BF16 [T][N].
// The grid runs column blocks fastest, so one weight row block is read once from DRAM while every
// column block that needs it passes through.
template <int WM, int WN, int MT, int NT, int STAGES, int SCALE, int MINB, int ABL>
__global__ __launch_bounds__(WM * WN * 32, MINB) void t2_tile_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const void* __restrict__ xs, int k, int t_pad, int tokens,
    int n_rows, __nv_bfloat16* __restrict__ out) {
    using C = Config<WM, WN, MT, NT, STAGES, SCALE>;
    extern __shared__ __align__(16) char smem[];
    char* const s_ring = smem + STAGES * C::kStage;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int warp_m = warp / WN;
    const int warp_n = warp % WN;
    const int col0   = blockIdx.x * C::BN;
    const int row0   = blockIdx.y * C::BM;
    const int groups = k / kGroupK;
    const int row_bytes = k / 4;

    auto issue = [&](int g, int buf) {
        char* const s_w = smem + buf * C::kStage;
        char* const s_x = s_w + C::kWBytes;
        char* const s_s = s_x + C::kXBytes;
        for (int c = tid; c < C::BM * 2; c += C::kThreads) {
            const int r = c >> 1;
            cp_async16(s_w + r * 32 + (c & 1) * 16,
                       w_codes + static_cast<std::size_t>(row0 + r) * row_bytes + g * 32 +
                           (c & 1) * 16);
        }
        for (int c = tid; c < C::BN * 8; c += C::kThreads) {
            const int col = c >> 3;
            cp_async16(s_x + col * kXStride + (c & 7) * 16,
                       x + static_cast<std::size_t>(col0 + col) * k + g * kGroupK + (c & 7) * 16);
        }
        if constexpr (SCALE == kPerGroup) {
            const __half* src = static_cast<const __half*>(xs) + static_cast<std::size_t>(g) * t_pad;
            for (int c = tid; c < C::BN / 8; c += C::kThreads) {
                cp_async16(s_s + c * 16, src + col0 + c * 8);
            }
        }
        if (g % kRing == 0) {
            char* const ring = s_ring + ((g / kRing) & 1) * C::kRingBytes;
            for (int r = tid; r < C::BM; r += C::kThreads) {
                cp_async16(ring + r * 16, w_scales + static_cast<std::size_t>(row0 + r) * groups + g);
            }
        }
        commit();
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0f;
    int iacc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) iacc[m][n][j] = 0;

#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        if (i < groups) {
            issue(i, i);
        } else {
            commit();
        }
    }

    for (int g = 0; g < groups; ++g) {
        wait_group<STAGES - 2>();
        __syncthreads();
        const int next = g + STAGES - 1;
        if (next < groups) {
            issue(next, next % STAGES);
        } else {
            commit();
        }

        const char* const s_w = smem + (g % STAGES) * C::kStage;
        const char* const s_x = s_w + C::kWBytes;
        const __half* const s_s = reinterpret_cast<const __half*>(s_x + C::kXBytes);
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / kRing) & 1) * C::kRingBytes);
        const int slot = g % kRing;

        unsigned b[NT][8];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const char* p = s_x + ((warp_n * NT + n) * 8 + gid) * kXStride + tig * 32;
            const uint4 v0 = lds128(p);
            const uint4 v1 = lds128(p + 16);
            b[n][0] = v0.x;
            b[n][1] = v0.y;
            b[n][2] = v0.z;
            b[n][3] = v0.w;
            b[n][4] = v1.x;
            b[n][5] = v1.y;
            b[n][6] = v1.z;
            b[n][7] = v1.w;
        }
        float xa[NT][2];
        if constexpr (SCALE == kPerGroup) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int c = (warp_n * NT + n) * 8 + tig * 2;
                xa[n][0]    = __half2float(s_s[c]);
                xa[n][1]    = __half2float(s_s[c + 1]);
            }
        }

#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0   = (warp_m * MT + m) * 16 + gid;
            const uint2 w0 = lds64(s_w + r0 * 32 + tig * 8);
            const uint2 w1 = lds64(s_w + (r0 + 8) * 32 + tig * 8);
            unsigned d0[8], d1[8];
            if constexpr ((ABL & 1) != 0) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    d0[j] = (j & 1) != 0 ? w0.y : w0.x;
                    d1[j] = (j & 1) != 0 ? w1.y : w1.x;
                }
            } else if constexpr ((ABL & 8) != 0) {
                decode_word_spread(w0.x, *reinterpret_cast<unsigned(*)[4]>(&d0[0]));
                decode_word_spread(w0.y, *reinterpret_cast<unsigned(*)[4]>(&d0[4]));
                decode_word_spread(w1.x, *reinterpret_cast<unsigned(*)[4]>(&d1[0]));
                decode_word_spread(w1.y, *reinterpret_cast<unsigned(*)[4]>(&d1[4]));
            } else {
                decode_word(w0.x, *reinterpret_cast<unsigned(*)[4]>(&d0[0]));
                decode_word(w0.y, *reinterpret_cast<unsigned(*)[4]>(&d0[4]));
                decode_word(w1.x, *reinterpret_cast<unsigned(*)[4]>(&d1[0]));
                decode_word(w1.y, *reinterpret_cast<unsigned(*)[4]>(&d1[4]));
            }
            if constexpr ((ABL & 2) != 0) {
#pragma unroll
                for (int n = 0; n < NT; ++n) {
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        mma_s8(iacc[m][n], d0[2 * s], d1[2 * s], d0[2 * s + 1], d1[2 * s + 1],
                               b[n][2 * s], b[n][2 * s + 1]);
                    }
                }
                continue;
            }
            const float ws0 = __half2float(ring[r0 * kRing + slot]);
            const float ws1 = __half2float(ring[(r0 + 8) * kRing + slot]);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int c[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    mma_s8(c, d0[2 * s], d1[2 * s], d0[2 * s + 1], d1[2 * s + 1], b[n][2 * s],
                           b[n][2 * s + 1]);
                }
                const float f0 = __int_as_float(c[0]) - kMagic;
                const float f1 = __int_as_float(c[1]) - kMagic;
                const float f2 = __int_as_float(c[2]) - kMagic;
                const float f3 = __int_as_float(c[3]) - kMagic;
                if constexpr (SCALE == kPerGroup) {
                    acc[m][n][0] = fmaf(f0, ws0 * xa[n][0], acc[m][n][0]);
                    acc[m][n][1] = fmaf(f1, ws0 * xa[n][1], acc[m][n][1]);
                    acc[m][n][2] = fmaf(f2, ws1 * xa[n][0], acc[m][n][2]);
                    acc[m][n][3] = fmaf(f3, ws1 * xa[n][1], acc[m][n][3]);
                } else {
                    acc[m][n][0] = fmaf(f0, ws0, acc[m][n][0]);
                    acc[m][n][1] = fmaf(f1, ws0, acc[m][n][1]);
                    acc[m][n][2] = fmaf(f2, ws1, acc[m][n][2]);
                    acc[m][n][3] = fmaf(f3, ws1, acc[m][n][3]);
                }
            }
        }
    }

    if constexpr ((ABL & 2) != 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[m][n][j] = static_cast<float>(iacc[m][n][j]);
    }
    // Bit 2 of ABL keeps the stores behind a test that never passes.
    const int limit = (ABL & 4) != 0 ? (acc[0][0][0] == 1.2345e30f ? tokens : 0) : tokens;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int c0 = col0 + (warp_n * NT + n) * 8 + tig * 2;
        float t0 = 1.0f, t1 = 1.0f;
        if constexpr (SCALE == kPerToken) {
            t0 = static_cast<const float*>(xs)[c0];
            t1 = static_cast<const float*>(xs)[c0 + 1];
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0 = row0 + (warp_m * MT + m) * 16 + gid;
            if (c0 < limit) {
                out[static_cast<std::size_t>(c0) * n_rows + r0]     = __float2bfloat16(acc[m][n][0] * t0);
                out[static_cast<std::size_t>(c0) * n_rows + r0 + 8] = __float2bfloat16(acc[m][n][2] * t0);
            }
            if (c0 + 1 < limit) {
                out[static_cast<std::size_t>(c0 + 1) * n_rows + r0] =
                    __float2bfloat16(acc[m][n][1] * t1);
                out[static_cast<std::size_t>(c0 + 1) * n_rows + r0 + 8] =
                    __float2bfloat16(acc[m][n][3] * t1);
            }
        }
    }
}

// One block per padded token; padded tokens get zero codes and a zero scale.
template <int SCALE, bool PERMUTE>
__global__ void quantise(const __nv_bfloat16* __restrict__ x, int k, int tokens, int t_pad,
                         std::int8_t* __restrict__ codes, void* __restrict__ xs) {
    const int t      = blockIdx.x;
    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int warps  = blockDim.x >> 5;
    const int groups = k / kGroupK;
    std::int8_t* dst = codes + static_cast<std::size_t>(t) * k;
    if (t >= tokens) {
        for (int i = threadIdx.x; i < k / 4; i += blockDim.x) {
            reinterpret_cast<int*>(dst)[i] = 0;
        }
        if constexpr (SCALE == kPerGroup) {
            for (int g = threadIdx.x; g < groups; g += blockDim.x) {
                static_cast<__half*>(xs)[static_cast<std::size_t>(g) * t_pad + t] = __float2half(0.0f);
            }
        } else if (threadIdx.x == 0) {
            static_cast<float*>(xs)[t] = 0.0f;
        }
        return;
    }
    const __nv_bfloat16* src = x + static_cast<std::size_t>(t) * k;
    float inv_token          = 0.0f;
    if constexpr (SCALE == kPerToken) {
        __shared__ float partial[32];
        float amax = 0.0f;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
        }
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        if (lane == 0) { partial[warp] = amax; }
        __syncthreads();
        amax = lane < warps ? partial[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        const float scale = fmaxf(amax, 1e-20f) / 127.0f;
        if (threadIdx.x == 0) { static_cast<float*>(xs)[t] = scale; }
        inv_token = 1.0f / scale;
    }
    for (int g = warp; g < groups; g += warps) {
        const uint2 raw = *reinterpret_cast<const uint2*>(src + g * kGroupK + lane * 4);
        const auto* v   = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float f[4];
        for (int j = 0; j < 4; ++j) { f[j] = __bfloat162float(v[j]); }
        float inv = inv_token;
        if constexpr (SCALE == kPerGroup) {
            float amax = fmaxf(fmaxf(fabsf(f[0]), fabsf(f[1])), fmaxf(fabsf(f[2]), fabsf(f[3])));
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            }
            const __half scale = __float2half(fmaxf(amax, 1e-20f) / 127.0f);
            if (lane == 0) {
                static_cast<__half*>(xs)[static_cast<std::size_t>(g) * t_pad + t] = scale;
            }
            inv = 1.0f / __half2float(scale);
        }
        signed char q[4];
        for (int j = 0; j < 4; ++j) {
            q[j] = static_cast<signed char>(max(-127, min(127, __float2int_rn(f[j] * inv))));
        }
        std::int8_t* const row = dst + g * kGroupK + lane * 4;
        if constexpr (PERMUTE) {
            std::int8_t* const word = dst + g * kGroupK + (lane / 4) * 16;
            for (int j = 0; j < 4; ++j) { word[permute16((lane % 4) * 4 + j)] = q[j]; }
        } else {
            *reinterpret_cast<char4*>(row) = make_char4(q[0], q[1], q[2], q[3]);
        }
    }
}

struct Timing {
    double quantise = 0.0;
    double gemm     = 0.0;
};

class Timer {
public:
    Timer(void* flush, std::size_t bytes) : flush_(flush), bytes_(bytes) {
        for (auto& e : events_) { CHECK(cudaEventCreate(&e)); }
    }
    ~Timer() {
        for (auto& e : events_) { cudaEventDestroy(e); }
    }
    template <class A, class B>
    Timing measure(A&& first, B&& second) {
        std::vector<float> a, b;
        for (int i = 0; i < 11; ++i) {
            CHECK(cudaMemsetAsync(flush_, i & 0xff, bytes_));
            CHECK(cudaEventRecord(events_[0]));
            first();
            CHECK(cudaEventRecord(events_[1]));
            second();
            CHECK(cudaEventRecord(events_[2]));
            CHECK(cudaEventSynchronize(events_[2]));
            if (i < 2) { continue; }
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, events_[0], events_[1]));
            a.push_back(ms);
            CHECK(cudaEventElapsedTime(&ms, events_[1], events_[2]));
            b.push_back(ms);
        }
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        return {a[a.size() / 2] * 1000.0, b[b.size() / 2] * 1000.0};
    }

private:
    void* flush_;
    std::size_t bytes_;
    cudaEvent_t events_[3]{};
};

double relative_l2(const void* device, const std::vector<__nv_bfloat16>& reference) {
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

struct Problem {
    int n;
    int k;
    int tokens;
    const void* codes;
    const void* scales;
    const Tensor* x;
    std::int8_t* x_codes;
    void* x_scales;
    void* out;
    const std::vector<__nv_bfloat16>* reference;
    double baseline_us;
    Timer* timer;
};

// ABL removes work to see what it costs, and the result is then wrong: bit 0 feeds the raw code
// words to the MMA instead of decoding them, bit 1 accumulates int32 over the whole K instead of
// converting per weight group.
template <int WM, int WN, int MT, int NT, int STAGES, int SCALE, int MINB, int ABL = 0>
void run_variant(const Problem& p) {
    using C = Config<WM, WN, MT, NT, STAGES, SCALE>;
    char name[64];
    std::snprintf(name, sizeof(name), "%s %dx%d s%d w%dx%d b%d%s", SCALE == kPerGroup ? "grp" : "tok",
                  C::BM, C::BN, STAGES, WM, WN, MINB,
                  ABL == 1 ? " -dec" : ABL == 2 ? " -cvt" : ABL == 3 ? " -dec-cvt" :
                  ABL == 4 ? " -st" : ABL == 7 ? " -all" : ABL == 8 ? " spread" : "");
    if (p.n % C::BM != 0) {
        std::printf("  %-26s rows not a multiple of %d\n", name, C::BM);
        return;
    }
    auto* kernel = t2_tile_kernel<WM, WN, MT, NT, STAGES, SCALE, MINB, ABL>;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem) !=
        cudaSuccess) {
        cudaGetLastError();
        std::printf("  %-26s %d bytes of shared memory refused\n", name, C::kSmem);
        return;
    }
    const int t_pad = (p.tokens + C::BN - 1) / C::BN * C::BN;
    const dim3 grid(t_pad / C::BN, p.n / C::BM);
    const auto x_data = static_cast<const __nv_bfloat16*>(p.x->data);
    const Timing t    = p.timer->measure(
        [&] {
            quantise<SCALE, (ABL & 8) == 0><<<t_pad, 256>>>(x_data, p.k, p.tokens, t_pad, p.x_codes, p.x_scales);
        },
        [&] {
            kernel<<<grid, C::kThreads, C::kSmem>>>(
                static_cast<const std::uint8_t*>(p.codes), static_cast<const __half*>(p.scales),
                p.x_codes, p.x_scales, p.k, t_pad, p.tokens, p.n,
                static_cast<__nv_bfloat16*>(p.out));
        });
    CHECK(cudaGetLastError());
    const double ops   = 2.0 * p.n * p.k * p.tokens;
    const double total = t.quantise + t.gemm;
    std::printf("  %-26s %8.1f %9.1f %9.1f %7.1f %8.2fx %9.2e\n", name, t.quantise, t.gemm, total,
                ops / t.gemm / 1e6, p.baseline_us / total, relative_l2(p.out, *p.reference));
}

void sweep(const Problem& p) {
    //          WM WN MT NT ST  scale     MINB ABL
    run_variant<2, 2, 4, 4, 2, kPerGroup, 3, 8>(p); // 128 x  64, 4 warps, three blocks
    run_variant<2, 2, 4, 4, 2, kPerGroup, 3>(p);
    run_variant<2, 2, 4, 4, 2, kPerToken, 3, 8>(p);
    run_variant<2, 2, 4, 4, 2, kPerToken, 3>(p);
    run_variant<2, 2, 4, 4, 2, kPerToken, 3, 1>(p);
    run_variant<2, 2, 4, 4, 2, kPerToken, 3, 2>(p);
    run_variant<2, 2, 4, 4, 2, kPerToken, 3, 7>(p);
    run_variant<2, 2, 4, 4, 3, kPerGroup, 2>(p);    // two blocks
    run_variant<2, 2, 4, 4, 3, kPerToken, 2>(p);
    run_variant<4, 2, 2, 4, 2, kPerGroup, 3>(p);    // 128 x  64, 8 warps, three blocks
    run_variant<4, 2, 2, 4, 2, kPerToken, 3>(p);
    run_variant<2, 4, 4, 2, 2, kPerGroup, 3>(p);    // 128 x  64, 8 warps of 64 x 16
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> token_counts;
    bool shipped_only = false; // --shipped: time the installed routes, skip the sweep
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--shipped") {
            shipped_only = true;
        } else {
            token_counts.push_back(std::atoi(argv[i]));
        }
    }
    if (token_counts.empty()) { token_counts = {1024, 2048, 4096}; }

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    constexpr std::size_t kFlushBytes = 128u << 20;
    void* flush                       = nullptr;
    CHECK(cudaMalloc(&flush, kFlushBytes));
    Timer timer(flush, kFlushBytes);
    std::printf("GPU: %s  sm_%d%d, %d SMs; microseconds, median of 9, L2 flushed\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount);

    struct Shape {
        const char* name;
        int n;
        int k;
    };
    const Shape shapes[] = {{"mlp/gate_up", 34816, 5120},
                            {"mlp/down", 5120, 17408},
                            {"gdn/value_z", 12288, 5120}};

    std::mt19937_64 rng(20260923);
    for (const Shape& shape : shapes) {
        const std::size_t code_bytes = static_cast<std::size_t>(shape.n) * shape.k / 4;
        const std::size_t groups     = static_cast<std::size_t>(shape.n) * (shape.k / kGroupK);
        std::vector<std::uint32_t> host_codes(code_bytes / 4);
        for (auto& word : host_codes) {
            const std::uint64_t bits = rng();
            std::uint32_t v          = 0;
            for (int f = 0; f < 16; ++f) {
                std::uint32_t code = static_cast<std::uint32_t>(bits >> (2 * f)) & 3u;
                if (code == 2u) { code = 3u; }
                v |= code << (2 * f);
            }
            word = v;
        }
        std::uniform_real_distribution<float> spread(0.005f, 0.02f);
        std::vector<__half> host_scales(groups);
        for (auto& s : host_scales) { s = __float2half(spread(rng)); }
        void* codes  = nullptr;
        void* scales = nullptr;
        CHECK(cudaMalloc(&codes, code_bytes));
        CHECK(cudaMalloc(&scales, groups * sizeof(__half)));
        CHECK(cudaMemcpy(codes, host_codes.data(), code_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(scales, host_scales.data(), groups * sizeof(__half),
                         cudaMemcpyHostToDevice));

        Weight w;
        w.payload         = codes;
        w.payload_bytes   = code_bytes + groups * sizeof(__half);
        w.qtype           = QType::T2_G128_FP16;
        w.group_size      = 128;
        w.shape[0]        = shape.n;
        w.shape[1]        = shape.k;
        w.padded_shape[0] = shape.n;
        w.padded_shape[1] = shape.k;
        w.ndim            = 2;
        w.qdata           = codes;
        w.scales          = scales;
        w.n               = shape.n;
        w.k               = shape.k;
        w.group           = 128;
        w.layout          = QuantLayout::RowSplit;
        w.scale_dtype     = DType::FP16;

        for (const int tokens : token_counts) {
            const std::size_t outs  = static_cast<std::size_t>(shape.n) * tokens;
            const int t_max         = (tokens + 511) / 512 * 512;
            DeviceArena io(static_cast<std::size_t>(shape.k) * tokens * 2 + outs * 2 * 2 +
                           static_cast<std::size_t>(shape.k) * t_max * 2 + (16u << 20));
            Tensor x         = io.alloc(DType::BF16, {shape.k, tokens});
            Tensor reference = io.alloc(DType::BF16, {shape.n, tokens});
            Tensor result    = io.alloc(DType::BF16, {shape.n, tokens});
            auto* x_codes =
                static_cast<std::int8_t*>(io.alloc_bytes(static_cast<std::size_t>(shape.k) * t_max).data);
            void* x_scales =
                io.alloc_bytes(static_cast<std::size_t>(shape.k / kGroupK) * t_max * 2 + 4096).data;
            {
                std::normal_distribution<float> normal(0.0f, 1.0f);
                std::vector<__nv_bfloat16> host_x(static_cast<std::size_t>(shape.k) * tokens);
                for (auto& v : host_x) { v = __float2bfloat16(normal(rng)); }
                CHECK(cudaMemcpy(x.data, host_x.data(), host_x.size() * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice));
            }
            DeviceArena scratch(512u << 20);
            const auto none = [] {};
            Timing a16{}, a8{};
            try {
                a16 = timer.measure(none, [&] { ops::linear(x, w, reference, nullptr); });
                a8  = timer.measure(none, [&] {
                    auto scope = scratch.scope();
                    ops::linear(x, w, result, ops::LinearPolicy::AllowA8Int, scratch, nullptr);
                });
            } catch (const std::exception& error) {
                std::printf("ours: %s\n", error.what());
                continue;
            }
            CHECK(cudaDeviceSynchronize());
            std::vector<__nv_bfloat16> host_reference(outs);
            CHECK(cudaMemcpy(host_reference.data(), reference.data, outs * sizeof(__nv_bfloat16),
                             cudaMemcpyDeviceToHost));
            const double ops = 2.0 * shape.n * shape.k * tokens;
            std::printf("\n%s %dx%d, T=%d\n", shape.name, shape.n, shape.k, tokens);
            std::printf("  %-26s %8s %9s %9s %7s %9s %9s\n", "variant", "quant", "gemm", "total",
                        "TOP/s", "vs int8", "err");
            std::printf("  %-26s %8s %9s %9.1f %7.1f %8.2fx %9s\n", "ours A16", "", "", a16.gemm,
                        ops / a16.gemm / 1e6, a8.gemm / a16.gemm, "ref");
            std::printf("  %-26s %8s %9s %9.1f %7.1f %8.2fx %9.2e\n", "ours int8", "", "", a8.gemm,
                        ops / a8.gemm / 1e6, 1.0,
                        relative_l2(result.data, host_reference));

            const Problem p{shape.n, shape.k,  tokens,  codes,          scales,  &x,
                            x_codes, x_scales, result.data, &host_reference, a8.gemm, &timer};
            if (!shipped_only) { sweep(p); }
        }
        CHECK(cudaFree(scales));
        CHECK(cudaFree(codes));
    }
    CHECK(cudaFree(flush));
    return 0;
}
