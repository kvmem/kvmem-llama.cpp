#pragma once

// The vector kernel: products of a GGUF matrix with one to eight q8 activation columns. A warp owns
// two rows; lane l takes the 32-value slices l, l + 32, ... of each. Every slice of a row is decoded
// once into eight int8x4 words in value order (word i holds values 4i..4i+3) plus the factors of its
// type, then dotted with that slice of every column. The activation is planar: int8 values
// [columns][K] and one (d, sum of x) half2 per 32 values [columns][K / 32], ggml's q8_1 numbers.
//
// The decoders follow ggml-quants.c's dequantize_row_* exactly, with float slice factors where
// llama.cpp's vector kernels round integer scales (the 2- and 3-bit i-quants).

#include "ggml_bridge_internal.cuh"

#include <algorithm>

namespace ninfer::ops::gguf::detail {

// Types whose vector kernels are also built for the static K of the 27B text matrices.
template <ggml_type type>
inline constexpr bool kVecStaticK =
    type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS ||
    type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS ||
    type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_Q2_K;

inline constexpr int kVecWarps = 4;

namespace vec {

struct Slice {
    int w[8];
    float f[6];
};

__device__ __forceinline__ std::uint32_t u32_a2(const std::uint8_t* p) {
    const auto* h = reinterpret_cast<const std::uint16_t*>(p);
    return std::uint32_t(h[0]) | (std::uint32_t(h[1]) << 16);
}

__device__ __forceinline__ std::uint32_t u32_a4(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ float half_at(const std::uint8_t* p) {
    return __half2float(*reinterpret_cast<const half*>(p));
}

// Negates the bytes of `g` whose bit is set in the low nibble of `bits`. Every grid this serves has
// no zero byte, so the per-byte two's-complement increment cannot carry.
__device__ __forceinline__ int negate_bytes(std::uint32_t g, std::uint32_t bits) {
    const std::uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
    return int((g ^ (ones * 0xFFu)) + ones);
}

// ksigns_iq2xs: seven stored sign bits and an eighth that makes the parity even.
__device__ __forceinline__ std::uint32_t ksigns(std::uint32_t v) {
    return v ^ ((__popc(v) & 1) << 7);
}

__device__ __forceinline__ int dot4(const int* w, const int* a, int acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc = ggml_cuda_dp4a(w[i], a[i], acc); }
    return acc;
}

__device__ __forceinline__ int sum4(const int* a) {
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc = ggml_cuda_dp4a(a[i], 0x01010101, acc); }
    return acc;
}

__device__ __forceinline__ int sum2(const int* a) {
    return ggml_cuda_dp4a(a[1], 0x01010101, ggml_cuda_dp4a(a[0], 0x01010101, 0));
}

// The 6-bit scale and min of sub-block j of a Q4_K / Q5_K block (ggml's get_scale_min_k4).
__device__ __forceinline__ void scale_min_k4(const std::uint8_t* s, int j, int& sc, int& m) {
    if (j < 4) {
        sc = s[j] & 63;
        m  = s[j + 4] & 63;
    } else {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m  = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// dot() is the contribution of one slice to one column: d and sum are the column's slice scale and
// the sum of its unquantized values.
template <ggml_type type>
struct Decoder;

template <>
struct Decoder<GGML_TYPE_Q8_0> {
    static constexpr int kBlockElems = 32, kBlockBytes = 34, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int, const std::uint32_t*,
                                                  Slice& s) {
        s.f[0] = half_at(b);
#pragma unroll
        for (int i = 0; i < 8; ++i) { s.w[i] = int(u32_a2(b + 2 + 4 * i)); }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ4_NL> {
    static constexpr int kBlockElems = 32, kBlockBytes = 18, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int, const std::uint32_t*,
                                                  Slice& s) {
        s.f[0] = half_at(b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16(int(u32_a2(b + 2 + 4 * j)), kvalues_iq4nl);
            s.w[j]       = v.x;
            s.w[j + 4]   = v.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ4_XS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 136, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const std::uint32_t sh = *reinterpret_cast<const std::uint16_t*>(b + 2);
        const int ls           = ((b[4 + u / 2] >> (4 * (u & 1))) & 0xF) | (((sh >> (2 * u)) & 3) << 4);
        s.f[0]                 = half_at(b) * float(ls - 32);
        const uint2 lo         = *reinterpret_cast<const uint2*>(b + 8 + 16 * u);
        const uint2 hi         = *reinterpret_cast<const uint2*>(b + 16 + 16 * u);
        const std::uint32_t q[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16(int(q[j]), kvalues_iq4nl);
            s.w[j]       = v.x;
            s.w[j + 4]   = v.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q4_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 144, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
        int sc, m;
        scale_min_k4(b + 4, u, sc, m);
        s.f[0]              = dm.x * float(sc);
        s.f[1]              = dm.y * float(m);
        const uint4* qs     = reinterpret_cast<const uint4*>(b + 16 + 32 * (u / 2));
        const uint4 a0      = qs[0];
        const uint4 a1      = qs[1];
        const int shift     = 4 * (u & 1);
        const std::uint32_t v[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { s.w[i] = int((v[i] >> shift) & 0x0F0F0F0Fu); }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) - s.f[1] * sum;
    }
};

template <>
struct Decoder<GGML_TYPE_Q5_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 176, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
        int sc, m;
        scale_min_k4(b + 4, u, sc, m);
        s.f[0]          = dm.x * float(sc);
        s.f[1]          = dm.y * float(m);
        const uint4* qh = reinterpret_cast<const uint4*>(b + 16);
        const uint4* ql = reinterpret_cast<const uint4*>(b + 48 + 32 * (u / 2));
        const uint4 h0 = qh[0], h1 = qh[1], l0 = ql[0], l1 = ql[1];
        const std::uint32_t hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const std::uint32_t lv[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        const int shift = 4 * (u & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s.w[i] = int(((lv[i] >> shift) & 0x0F0F0F0Fu) | (((hv[i] >> u) << 4) & 0x10101010u));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) - s.f[1] * sum;
    }
};

template <>
struct Decoder<GGML_TYPE_Q6_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 210, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, q = u % 4;
        const float d        = half_at(b + 208);
        const auto* sc       = reinterpret_cast<const std::int8_t*>(b + 192 + 8 * h);
        s.f[0]               = d * float(sc[2 * q]);
        s.f[1]               = d * float(sc[2 * q + 1]);
        const std::uint8_t* ql = b + 64 * h + 32 * (q & 1);
        const std::uint8_t* qh = b + 128 + 32 * h;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t v = ((u32_a2(ql + 4 * i) >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                    (((u32_a2(qh + 4 * i) >> (2 * q)) << 4) & 0x30303030u);
            // v - 32 per byte, v in [0, 64): flip the high bits by whether v is below 32.
            s.w[i] = int(v ^ 0xE0E0E0E0u ^ ((v & 0x20202020u) * 6u));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q2_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 84, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, j = u % 4;
        const float2 dm      = __half22float2(*reinterpret_cast<const half2*>(b + 80));
        const std::uint32_t lo = b[8 * h + 2 * j], hi = b[8 * h + 2 * j + 1];
        s.f[0]               = dm.x * float(lo & 0xF);
        s.f[1]               = dm.x * float(hi & 0xF);
        s.f[2]               = dm.y * float(lo >> 4);
        s.f[3]               = dm.y * float(hi >> 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s.w[i] = int((u32_a4(b + 16 + 32 * h + 4 * i) >> (2 * j)) & 0x03030303u);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)) -
                    s.f[2] * float(sum4(a)) - s.f[3] * float(sum4(a + 4)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q3_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 110, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static int scale(const std::uint8_t* b, int k) {
        return (((b[96 + k % 8] >> (4 * (k / 8))) & 0xF) | (((b[104 + k % 4] >> (2 * (k / 4))) & 3) << 4)) - 32;
    }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, j = u % 4;
        const float d = half_at(b + 108);
        s.f[0]        = d * float(scale(b, 8 * h + 2 * j));
        s.f[1]        = d * float(scale(b, 8 * h + 2 * j + 1));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t v = ((u32_a2(b + 32 + 32 * h + 4 * i) >> (2 * j)) & 0x03030303u) |
                                    (((u32_a2(b + 4 * i) >> (4 * h + j)) & 0x01010101u) << 2);
            // v - 4 per byte, v in [0, 8).
            s.w[i] = int(v ^ 0xFCFCFCFCu ^ ((v & 0x04040404u) * 0x3Eu));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XXS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 66, kTableWords = 512;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2xxs_grid);
    }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* tab, Slice& s) {
        const std::uint32_t idx = u32_a2(b + 2 + 8 * u);
        const std::uint32_t aux = u32_a2(b + 6 + 8 * u);
        s.f[0]                  = half_at(b) * (0.5f + float(aux >> 28)) * 0.25f;
        const auto* grid        = reinterpret_cast<const uint2*>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g          = grid[(idx >> (8 * l)) & 0xFF];
            const std::uint32_t sg = ksigns((aux >> (7 * l)) & 0x7F);
            s.w[2 * l]             = negate_bytes(g.x, sg);
            s.w[2 * l + 1]         = negate_bytes(g.y, sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 74, kTableWords = 1024;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2xs_grid);
    }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* tab, Slice& s) {
        const float d          = half_at(b);
        const std::uint32_t sc = b[66 + u];
        s.f[0]                 = d * (0.5f + float(sc & 0xF)) * 0.25f;
        s.f[1]                 = d * (0.5f + float(sc >> 4)) * 0.25f;
        const std::uint32_t q01 = u32_a2(b + 2 + 8 * u), q23 = u32_a2(b + 6 + 8 * u);
        const auto* grid        = reinterpret_cast<const uint2*>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t q  = ((l < 2 ? q01 : q23) >> (16 * (l & 1))) & 0xFFFF;
            const uint2 g          = grid[q & 511];
            const std::uint32_t sg = ksigns(q >> 9);
            s.w[2 * l]             = negate_bytes(g.x, sg);
            s.w[2 * l + 1]         = negate_bytes(g.y, sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_S> {
    static constexpr int kBlockElems = 256, kBlockBytes = 82, kTableWords = 2048;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2s_grid);
    }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* tab, Slice& s) {
        const float d          = half_at(b);
        const std::uint32_t sc = b[74 + u];
        s.f[0]                 = d * (0.5f + float(sc & 0xF)) * 0.25f;
        s.f[1]                 = d * (0.5f + float(sc >> 4)) * 0.25f;
        const std::uint32_t qs = u32_a2(b + 2 + 4 * u);
        const std::uint32_t sg = u32_a2(b + 34 + 4 * u);
        const std::uint32_t qh = b[66 + u];
        const auto* grid       = reinterpret_cast<const uint2*>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = grid[((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)];
            s.w[2 * l]     = negate_bytes(g.x, sg >> (8 * l));
            s.w[2 * l + 1] = negate_bytes(g.y, sg >> (8 * l + 4));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ3_XXS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 98, kTableWords = 256;
    __device__ static const std::uint32_t* table() { return iq3xxs_grid; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint32_t q0  = u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1  = u32_a2(b + 6 + 8 * u);
        const std::uint32_t aux = u32_a2(b + 66 + 4 * u);
        s.f[0]                  = half_at(b) * (0.5f + float(aux >> 28)) * 0.5f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const std::uint32_t q  = p < 2 ? q0 : q1;
            const std::uint32_t sg = ksigns((aux >> (7 * p)) & 0x7F);
            s.w[2 * p]     = negate_bytes(grid[(q >> (16 * (p & 1))) & 0xFF], sg);
            s.w[2 * p + 1] = negate_bytes(grid[(q >> (16 * (p & 1) + 8)) & 0xFF], sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ3_S> {
    static constexpr int kBlockElems = 256, kBlockBytes = 110, kTableWords = 512;
    __device__ static const std::uint32_t* table() { return iq3s_grid; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint32_t q0 = u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1 = u32_a2(b + 6 + 8 * u);
        const std::uint32_t qh = b[66 + u];
        const std::uint32_t sg = u32_a2(b + 74 + 4 * u);
        const int sc           = (b[106 + u / 2] >> (4 * (u & 1))) & 0xF;
        s.f[0]                 = half_at(b) * float(1 + 2 * sc);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const std::uint32_t idx = (((e < 4 ? q0 : q1) >> (8 * (e & 3))) & 0xFF) | ((qh << (8 - e)) & 0x100);
            s.w[e]                  = negate_bytes(grid[idx], sg >> (4 * e));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return s.f[0] * d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ1_S> {
    static constexpr int kBlockElems = 256, kBlockBytes = 50, kTableWords = 2048;
    __device__ static const std::uint32_t* table() { return iq1s_grid_gpu; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint32_t qs = u32_a2(b + 2 + 4 * u);
        const std::uint32_t qh = *reinterpret_cast<const std::uint16_t*>(b + 34 + 2 * u);
        s.f[0] = half_at(b) * float(((qh >> 11) & 0x0E) + 1);
        s.f[1] = -1.0f + IQ1S_DELTA - float(qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t g = grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
            s.w[2 * l]            = int(g & 0x0F0F0F0Fu);
            s.w[2 * l + 1]        = int((g >> 4) & 0x0F0F0F0Fu);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return s.f[0] * (d * float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))) + s.f[1] * sum);
    }
};

template <>
struct Decoder<GGML_TYPE_IQ1_M> {
    static constexpr int kBlockElems = 256, kBlockBytes = 56, kTableWords = 2048;
    __device__ static const std::uint32_t* table() { return iq1s_grid_gpu; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const auto* sc = reinterpret_cast<const std::uint16_t*>(b + 48);
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d          = __half2float(scale.f16);
        const int tmp          = sc[u / 2] >> (6 * (u % 2));
        s.f[0]                 = d * float(2 * (tmp & 7) + 1);
        s.f[1]                 = d * float(2 * ((tmp >> 3) & 7) + 1);
        const std::uint32_t qs = u32_a4(b + 4 * u);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t qhl = b[32 + 2 * u + l / 2] >> (4 * (l % 2));
            const std::uint32_t g   = grid[((qs >> (8 * l)) & 0xFF) | ((qhl & 7) << 8)];
            s.w[2 * l]              = int(g & 0x0F0F0F0Fu);
            s.w[2 * l + 1]          = int((g >> 4) & 0x0F0F0F0Fu);
            s.f[2 + l] = -1.0f + IQ1M_DELTA - float(qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        const float lo = float(dot4(s.w, a, 0)) + s.f[2] * float(sum2(a)) + s.f[3] * float(sum2(a + 2));
        const float hi =
            float(dot4(s.w + 4, a + 4, 0)) + s.f[4] * float(sum2(a + 4)) + s.f[5] * float(sum2(a + 6));
        return d * (s.f[0] * lo + s.f[1] * hi);
    }
};

// KS: the static K the slice loop unrolls over, or 0 for the runtime K. Fused: the weight is one
// [gate; up] parent and each output row is silu(gate row) * up row.
template <ggml_type type, int T, bool Fused, int KS>
__launch_bounds__(kVecWarps * 32) __global__ void kernel(VecArgs p) {
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kRows     = 2;
    __shared__ __align__(16) std::uint32_t table[kTable];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kVecWarps * 32) { table[i] = src[i]; }
        __syncthreads();
    }
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int k      = KS != 0 ? KS : p.k;
    const int slices = k / 32;
    const int step   = Fused ? 1 : kRows;
    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * D::kBlockBytes;
    const int sub                  = lane % kPerBlock;

    for (int row0 = (blockIdx.x * kVecWarps + warp) * step; row0 < p.rows;
         row0 += gridDim.x * kVecWarps * step) {
        const std::uint8_t* rowp[kRows];
        if constexpr (Fused) {
            rowp[0] = p.weight + std::int64_t(row0) * p.row_bytes + lane_offset;
            rowp[1] = p.weight + std::int64_t(p.rows + row0) * p.row_bytes + lane_offset;
        } else {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                rowp[r] = p.weight + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes + lane_offset;
            }
        }
        float acc[kRows][T];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        const auto slice = [&](int it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);
                d[j]            = ds.x;
                sum[j]          = ds.y;
                const auto* q   = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                Slice sl;
                D::decode(rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes, sub, table, sl);
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[r][j] += D::dot(sl, a[j], d[j], sum[j]); }
            }
        };
        if constexpr (KS != 0) {
            static_assert(KS % 1024 == 0);
#pragma unroll
            for (int it = 0; it < KS / 1024; ++it) { slice(it); }
        } else {
            for (int it = 0; lane + 32 * it < slices; ++it) { slice(it); }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                }
            }
        }
        if (lane == 0) {
            if constexpr (Fused) {
#pragma unroll
                for (int j = 0; j < T; ++j) { vec_store(p.out, row0, j, vec_silu(acc[0][j]) * acc[1][j]); }
            } else {
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if (row0 + r < p.rows) {
#pragma unroll
                        for (int j = 0; j < T; ++j) { vec_store(p.out, row0 + r, j, acc[r][j]); }
                    }
                }
            }
        }
    }
}

template <ggml_type type, int T, bool Fused, int KS>
void launch(const VecArgs& args, cudaStream_t stream) {
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel<type, T, Fused, KS>,
                                                            kVecWarps * 32, 0),
              "vector occupancy");
        return std::max(blocks, 1);
    }();
    const int per_block = kVecWarps * (Fused ? 1 : 2);
    const int groups    = (args.rows + per_block - 1) / per_block;
    const int blocks    = std::min(groups, resident * device_facts().sm_count);
    kernel<type, T, Fused, KS><<<blocks, kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "vector product launch");
}

template <ggml_type type, int T, bool Fused>
void launch_k(const VecArgs& args, cudaStream_t stream) {
    if constexpr (kVecStaticK<type>) {
        switch (args.k) {
        case 5120: return launch<type, T, Fused, 5120>(args, stream);
        case 6144: return launch<type, T, Fused, 6144>(args, stream);
        case 10240: return launch<type, T, Fused, 10240>(args, stream);
        case 17408: return launch<type, T, Fused, 17408>(args, stream);
        default: break;
        }
    }
    launch<type, T, Fused, 0>(args, stream);
}

template <ggml_type type, bool Fused>
void launch_columns(const VecArgs& args, int columns, cudaStream_t stream) {
    switch (columns) {
    case 1: return launch_k<type, 1, Fused>(args, stream);
    case 2: return launch_k<type, 2, Fused>(args, stream);
    case 3: return launch_k<type, 3, Fused>(args, stream);
    case 4: return launch_k<type, 4, Fused>(args, stream);
    case 5: return launch_k<type, 5, Fused>(args, stream);
    case 6: return launch_k<type, 6, Fused>(args, stream);
    case 7: return launch_k<type, 7, Fused>(args, stream);
    case 8: return launch_k<type, 8, Fused>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf vector product: columns must be 1..8");
}

} // namespace vec

template <ggml_type type>
void vec_launch(const VecArgs& args, int columns, bool fused, cudaStream_t stream) {
    using D = vec::Decoder<type>;
    if (args.k % D::kBlockElems != 0 || args.row_bytes % D::kBlockBytes != 0 ||
        args.row_bytes < std::int64_t(args.k / D::kBlockElems) * D::kBlockBytes || args.rows <= 0) {
        throw std::invalid_argument("gguf vector product: unsupported geometry");
    }
    if (fused) {
        vec::launch_columns<type, true>(args, columns, stream);
    } else {
        vec::launch_columns<type, false>(args, columns, stream);
    }
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_VECTOR_INSTANCE(TYPE)                                                          \
    template void ninfer::ops::gguf::detail::vec_launch<TYPE>(                                     \
        const ninfer::ops::gguf::detail::VecArgs&, int, bool, cudaStream_t)
