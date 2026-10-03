#pragma once

// T2G128 RowSplit x int8 prefill GEMM with its own activation contract, in place of the shared
// rowsplit_a8_mma.cuh mainloop with the ternary codec. On the RTX 5060 Ti (tools/t2_a8_tile_probe.cu,
// T = 1024, L2 flushed) that one ran the text-layer shapes at 74-94 TOP/s, cuBLAS int8 at 170-184;
// this runs at 134-138 and beats cuBLAS with its conversions on every shape. Where the time went:
//
//   scale    the shared contract scales activations per (token, 64 k), so every two MMAs the int32
//            partial is converted and multiplied by two scales. Here the partial runs the whole
//            128-k weight group and the activation scale is per (token, 128 k). The conversion is a
//            float subtract: the accumulator starts at the bits of 1.5 * 2^23, so as a float it
//            reads that plus the dot.
//   tile     the shared T2 tile is 64 rows and every such block re-streams its activation band; a
//            ternary row is a quarter of an int8 one, so the band is the traffic. 128 x 64 tiles
//            with three blocks per SM won over 64 x 128 and 256 x 64.
//   stages   two stages of 128 k with one barrier each; three or four stages measure the same.
//   decode   thread tig of an MMA quad owns k in [32 tig, 32 tig + 32) of each group, so its codes
//            for a row are one 8-byte shared load and every code word decodes in nine instructions:
//            masked with 0x3333 its nibbles are PRMT selectors of the even codes as they stand,
//            shifted by two those of the odd ones. The quantiser stores activations in the k order
//            that produces (t2_prefill_permute16).
//
// Activations: codes [Tpad][K] s8, scales [K / 128][Tpad] binary16, Tpad a multiple of kColumns;
// padded columns are zero. Output rows must be a multiple of kRows.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct T2PrefillI8 {
    static constexpr int kGroupK    = 128; // one weight scale group and one pipeline stage
    static constexpr int kWarpsM    = 2;
    static constexpr int kWarpsN    = 2;
    static constexpr int kMTiles    = 4;
    static constexpr int kNTiles    = 4;
    static constexpr int kStages    = 2;
    static constexpr int kMinBlocks = 3;
    static constexpr int kThreads   = kWarpsM * kWarpsN * 32;
    static constexpr int kRows      = kWarpsM * kMTiles * 16;
    static constexpr int kColumns   = kWarpsN * kNTiles * 8;
    // A column's 128 bytes padded to 144 so the eight columns an MMA quad reads miss each other.
    static constexpr int kXStride    = 144;
    static constexpr int kCodeBytes  = kRows * kGroupK / 4;
    static constexpr int kXBytes     = kColumns * kXStride;
    static constexpr int kScaleBytes = kColumns * 2;
    static constexpr int kStage      = kCodeBytes + kXBytes + kScaleBytes;
    // Weight scales arrive eight groups (16 bytes) per row at a time, double-buffered.
    static constexpr int kRing       = 8;
    static constexpr int kRingBytes  = kRows * kRing * 2;
    static constexpr int kSmem       = kStages * kStage + 2 * kRingBytes;
    static constexpr int kQuantThreads = 256;

    static_assert(kSmem <= 48 * 1024, "the T2 prefill tile must fit static-size shared memory");
    static_assert(kScaleBytes % 16 == 0, "activation scales are staged 16 bytes at a time");
};

// Where k (mod 16) of a code word lands among the sixteen activation bytes the MMA pairs it with:
// the decode yields k {0,2,4,6}, {1,3,5,7}, {8,10,12,14}, {9,11,13,15} per register.
__host__ __device__ constexpr int t2_prefill_permute16(int k) {
    return (k / 8) * 8 + (k % 2) * 4 + (k % 8) / 2;
}

// One code word (sixteen 2-bit two's-complement codes, lowest k lowest) to four registers of s8.
__device__ __forceinline__ void t2_prefill_decode(unsigned word, unsigned* d) {
    constexpr unsigned kTable = 0xFFFE0100u; // 0, +1, -2, -1
    const unsigned even       = word & 0x33333333u;
    const unsigned odd        = (word >> 2) & 0x33333333u;
    d[0]                      = __byte_perm(kTable, 0u, even);
    d[1]                      = __byte_perm(kTable, 0u, odd);
    d[2]                      = __byte_perm(kTable, 0u, even >> 16);
    d[3]                      = __byte_perm(kTable, 0u, odd >> 16);
}

// out[row, t] = sum_k W[row, k] X[k, t] for t < tokens, through epilogue(row, token, value). With
// rows_fast false the grid is (column blocks, row blocks) and runs column blocks fastest, so a
// weight row block is read once from DRAM while the column blocks that need it pass through; with
// rows_fast it is (row blocks, column blocks), for activation bands too wide to stay in L2.
template <std::int32_t kCols, class Epilogue>
__global__ __launch_bounds__(T2PrefillI8::kThreads, T2PrefillI8::kMinBlocks) void t2_prefill_i8_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const __half* __restrict__ xs, std::int32_t t_pad,
    std::int32_t tokens, bool rows_fast, Epilogue epilogue) {
    using C                     = T2PrefillI8;
    constexpr int kGroups       = kCols / C::kGroupK;
    constexpr int kRowBytes     = kCols / 4;
    static_assert(kCols % (C::kGroupK * C::kRing) == 0, "K must be whole scale rings");
    __shared__ __align__(16) char smem[C::kSmem];
    char* const s_ring = smem + C::kStages * C::kStage;

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int warp_m = warp / C::kWarpsN;
    const int warp_n = warp % C::kWarpsN;
    const int col0   = static_cast<int>(rows_fast ? blockIdx.y : blockIdx.x) * C::kColumns;
    const int row0   = static_cast<int>(rows_fast ? blockIdx.x : blockIdx.y) * C::kRows;

    const auto issue = [&](int g, int buf) {
        char* const s_w = smem + buf * C::kStage;
        char* const s_x = s_w + C::kCodeBytes;
        char* const s_s = s_x + C::kXBytes;
        for (int c = tid; c < C::kRows * 2; c += C::kThreads) {
            const int r = c >> 1;
            cp_async<16, Cache::cg>(s_w + r * 32 + (c & 1) * 16,
                                    w_codes + static_cast<std::size_t>(row0 + r) * kRowBytes +
                                        g * 32 + (c & 1) * 16);
        }
        for (int c = tid; c < C::kColumns * 8; c += C::kThreads) {
            const int col = c >> 3;
            cp_async<16, Cache::cg>(s_x + col * C::kXStride + (c & 7) * 16,
                                    x + static_cast<std::size_t>(col0 + col) * kCols +
                                        g * C::kGroupK + (c & 7) * 16);
        }
        for (int c = tid; c < C::kScaleBytes / 16; c += C::kThreads) {
            cp_async<16, Cache::cg>(s_s + c * 16,
                                    xs + static_cast<std::size_t>(g) * t_pad + col0 + c * 8);
        }
        if (g % C::kRing == 0) {
            char* const ring = s_ring + ((g / C::kRing) & 1) * C::kRingBytes;
            for (int r = tid; r < C::kRows; r += C::kThreads) {
                cp_async<16, Cache::cg>(ring + r * 16,
                                        w_scales + static_cast<std::size_t>(row0 + r) * kGroups + g);
            }
        }
        cp_commit();
    };

    float acc[C::kMTiles][C::kNTiles][4];
#pragma unroll
    for (int m = 0; m < C::kMTiles; ++m) {
#pragma unroll
        for (int n = 0; n < C::kNTiles; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[m][n][j] = 0.0F; }
        }
    }

#pragma unroll
    for (int i = 0; i < C::kStages - 1; ++i) { issue(i, i); }

    for (int g = 0; g < kGroups; ++g) {
        cp_wait<C::kStages - 2>();
        __syncthreads();
        const int next = g + C::kStages - 1;
        if (next < kGroups) {
            issue(next, next % C::kStages);
        } else {
            cp_commit();
        }

        const char* const s_w   = smem + (g % C::kStages) * C::kStage;
        const char* const s_x   = s_w + C::kCodeBytes;
        const __half* const s_s = reinterpret_cast<const __half*>(s_x + C::kXBytes);
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / C::kRing) & 1) * C::kRingBytes);
        const int slot = g % C::kRing;

        unsigned b[C::kNTiles][8];
        float xa[C::kNTiles][2];
#pragma unroll
        for (int n = 0; n < C::kNTiles; ++n) {
            const char* p  = s_x + ((warp_n * C::kNTiles + n) * 8 + gid) * C::kXStride + tig * 32;
            const uint4 v0 = load_vec<uint4>(p);
            const uint4 v1 = load_vec<uint4>(p + 16);
            b[n][0]        = v0.x;
            b[n][1]        = v0.y;
            b[n][2]        = v0.z;
            b[n][3]        = v0.w;
            b[n][4]        = v1.x;
            b[n][5]        = v1.y;
            b[n][6]        = v1.z;
            b[n][7]        = v1.w;
            const int c    = (warp_n * C::kNTiles + n) * 8 + tig * 2;
            xa[n][0]       = __half2float(s_s[c]);
            xa[n][1]       = __half2float(s_s[c + 1]);
        }

#pragma unroll
        for (int m = 0; m < C::kMTiles; ++m) {
            const int r0   = (warp_m * C::kMTiles + m) * 16 + gid;
            const uint2 w0 = load_vec<uint2>(s_w + r0 * 32 + tig * 8);
            const uint2 w1 = load_vec<uint2>(s_w + (r0 + 8) * 32 + tig * 8);
            unsigned d0[8];
            unsigned d1[8];
            t2_prefill_decode(w0.x, d0);
            t2_prefill_decode(w0.y, d0 + 4);
            t2_prefill_decode(w1.x, d1);
            t2_prefill_decode(w1.y, d1 + 4);
            const float ws0 = __half2float(ring[r0 * C::kRing + slot]);
            const float ws1 = __half2float(ring[(r0 + 8) * C::kRing + slot]);
#pragma unroll
            for (int n = 0; n < C::kNTiles; ++n) {
                constexpr int kMagicBits = 0x4B400000; // 1.5 * 2^23
                int c[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    mma_s8(c[0], c[1], c[2], c[3], d0[2 * s], d1[2 * s], d0[2 * s + 1],
                           d1[2 * s + 1], b[n][2 * s], b[n][2 * s + 1]);
                }
                constexpr float kMagic = 12582912.0F;
                acc[m][n][0] = fmaf(__int_as_float(c[0]) - kMagic, ws0 * xa[n][0], acc[m][n][0]);
                acc[m][n][1] = fmaf(__int_as_float(c[1]) - kMagic, ws0 * xa[n][1], acc[m][n][1]);
                acc[m][n][2] = fmaf(__int_as_float(c[2]) - kMagic, ws1 * xa[n][0], acc[m][n][2]);
                acc[m][n][3] = fmaf(__int_as_float(c[3]) - kMagic, ws1 * xa[n][1], acc[m][n][3]);
            }
        }
    }

#pragma unroll
    for (int n = 0; n < C::kNTiles; ++n) {
        const int c0 = col0 + (warp_n * C::kNTiles + n) * 8 + tig * 2;
#pragma unroll
        for (int m = 0; m < C::kMTiles; ++m) {
            const int r0 = row0 + (warp_m * C::kMTiles + m) * 16 + gid;
            if (c0 < tokens) {
                epilogue(r0, c0, acc[m][n][0]);
                epilogue(r0 + 8, c0, acc[m][n][2]);
            }
            if (c0 + 1 < tokens) {
                epilogue(r0, c0 + 1, acc[m][n][1]);
                epilogue(r0 + 8, c0 + 1, acc[m][n][3]);
            }
        }
    }
}

// One block per padded column, a warp per 128-k group, a lane per four k. Padded columns get zero
// codes and scales. A zero group keeps a zero scale and zero codes.
template <std::int32_t kCols>
__global__ __launch_bounds__(T2PrefillI8::kQuantThreads) void t2_prefill_i8_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t tokens, std::int32_t t_pad,
    std::int8_t* __restrict__ codes, __half* __restrict__ scales) {
    using C               = T2PrefillI8;
    constexpr int kGroups = kCols / C::kGroupK;
    constexpr int kWarps  = C::kQuantThreads / 32;
    const int t           = static_cast<int>(blockIdx.x);
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    std::int8_t* const dst = codes + static_cast<std::size_t>(t) * kCols;
    if (t >= tokens) {
        for (int i = static_cast<int>(threadIdx.x); i < kCols / 16; i += C::kQuantThreads) {
            store_vec(dst + i * 16, make_uint4(0u, 0u, 0u, 0u));
        }
        for (int g = static_cast<int>(threadIdx.x); g < kGroups; g += C::kQuantThreads) {
            scales[static_cast<std::size_t>(g) * t_pad + t] = __float2half(0.0F);
        }
        return;
    }
    const __nv_bfloat16* const src = x + static_cast<std::size_t>(t) * kCols;
    // Lane l writes bytes [4 l, 4 l + 4) of the group, which t2_prefill_permute16 fills from the
    // even (lanes 4q, 4q+2) or odd (4q+1, 4q+3) codes of lanes 4q + (l & 2) and the one after.
    const int source = lane & ~1;
    const int shift  = (lane & 1) * 16;
    for (int g = warp; g < kGroups; g += kWarps) {
        const uint2 raw = load_vec<uint2>(src + g * C::kGroupK + lane * 4);
        const auto* v   = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float f[4];
        float amax = 0.0F;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            f[j] = __bfloat162float(v[j]);
            amax = fmaxf(amax, fabsf(f[j]));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        const __half scale = __float2half(amax / 127.0F);
        if (lane == 0) { scales[static_cast<std::size_t>(g) * t_pad + t] = scale; }
        const float s   = __half2float(scale);
        const float inv = s > 0.0F ? 1.0F / s : 0.0F;
        unsigned q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = static_cast<unsigned>(max(-127, min(127, __float2int_rn(f[j] * inv)))) & 0xffu;
        }
        const unsigned packed = q[0] | (q[2] << 8) | (q[1] << 16) | (q[3] << 24);
        const unsigned a      = __shfl_sync(0xffffffffu, packed, source);
        const unsigned b      = __shfl_sync(0xffffffffu, packed, source + 1);
        const unsigned word   = ((a >> shift) & 0xffffu) | (((b >> shift) & 0xffffu) << 16);
        store_vec(dst + g * C::kGroupK + lane * 4, word);
    }
}

} // namespace ninfer::ops::detail
