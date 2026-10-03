#pragma once

// E8 root key codec of the rk2v4-e8 KV storage, after UDPSendToFailed/ninfer-4090's cylinder
// factorization. A key is rotated as for rk8v4 and scaled per G64 group by the packed int4 scale,
// FP16-RNE(absmax/7). Each block of eight consecutive dimensions is then stored in two bytes: the
// nearest of the 240 E8 roots to the block's direction, and a byte holding a 4-bit log-radius over
// a 4-bit residual axis. A key row is 64 bytes, against 128 for rk4v4-e8 and 256 for INT8. The read
// side turns a block back into eight signed int8 codes in the INT8 layout, so QK stays the same s8
// MMA and the group scale dequantizes as before.
//
// The encoder takes one coordinate per lane with each block held by an aligned lane octet, and
// must be called by the whole warp. It uses only correctly rounded arithmetic in a fixed order (no
// transcendental, no contraction), so the host oracle in the tests reproduces its bytes exactly.

#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheE8RootRowBytes = 64;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_e8_root_code_index(int physical_page, int kv_head,
                                                                    int byte, int page_offset) {
    return paged_kv_element_offset<kKVCacheE8RootRowBytes, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, byte);
}

// Root code r < 112 is the pair r/4 of (i<j) in lexicographic order with the signs of i and j in
// bits 1 and 0 (set = +); 112 <= r < 240 is 1/2(+-1)^8 with the signs of dimensions 0..6 in the
// bits of r-112 and dimension 7's sign making the count of minus signs even. The table holds each
// root times four as eight signed bytes, dimension 0 lowest: both kinds then have norm 4*sqrt(2).
static __device__ const std::uint64_t kKVCacheE8Roots[240] = {
    0x000000000000fcfcULL, 0x00000000000004fcULL, 0x000000000000fc04ULL, 0x0000000000000404ULL,
    0x0000000000fc00fcULL, 0x00000000000400fcULL, 0x0000000000fc0004ULL, 0x0000000000040004ULL,
    0x00000000fc0000fcULL, 0x00000000040000fcULL, 0x00000000fc000004ULL, 0x0000000004000004ULL,
    0x000000fc000000fcULL, 0x00000004000000fcULL, 0x000000fc00000004ULL, 0x0000000400000004ULL,
    0x0000fc00000000fcULL, 0x00000400000000fcULL, 0x0000fc0000000004ULL, 0x0000040000000004ULL,
    0x00fc0000000000fcULL, 0x00040000000000fcULL, 0x00fc000000000004ULL, 0x0004000000000004ULL,
    0xfc000000000000fcULL, 0x04000000000000fcULL, 0xfc00000000000004ULL, 0x0400000000000004ULL,
    0x0000000000fcfc00ULL, 0x000000000004fc00ULL, 0x0000000000fc0400ULL, 0x0000000000040400ULL,
    0x00000000fc00fc00ULL, 0x000000000400fc00ULL, 0x00000000fc000400ULL, 0x0000000004000400ULL,
    0x000000fc0000fc00ULL, 0x000000040000fc00ULL, 0x000000fc00000400ULL, 0x0000000400000400ULL,
    0x0000fc000000fc00ULL, 0x000004000000fc00ULL, 0x0000fc0000000400ULL, 0x0000040000000400ULL,
    0x00fc00000000fc00ULL, 0x000400000000fc00ULL, 0x00fc000000000400ULL, 0x0004000000000400ULL,
    0xfc0000000000fc00ULL, 0x040000000000fc00ULL, 0xfc00000000000400ULL, 0x0400000000000400ULL,
    0x00000000fcfc0000ULL, 0x0000000004fc0000ULL, 0x00000000fc040000ULL, 0x0000000004040000ULL,
    0x000000fc00fc0000ULL, 0x0000000400fc0000ULL, 0x000000fc00040000ULL, 0x0000000400040000ULL,
    0x0000fc0000fc0000ULL, 0x0000040000fc0000ULL, 0x0000fc0000040000ULL, 0x0000040000040000ULL,
    0x00fc000000fc0000ULL, 0x0004000000fc0000ULL, 0x00fc000000040000ULL, 0x0004000000040000ULL,
    0xfc00000000fc0000ULL, 0x0400000000fc0000ULL, 0xfc00000000040000ULL, 0x0400000000040000ULL,
    0x000000fcfc000000ULL, 0x00000004fc000000ULL, 0x000000fc04000000ULL, 0x0000000404000000ULL,
    0x0000fc00fc000000ULL, 0x00000400fc000000ULL, 0x0000fc0004000000ULL, 0x0000040004000000ULL,
    0x00fc0000fc000000ULL, 0x00040000fc000000ULL, 0x00fc000004000000ULL, 0x0004000004000000ULL,
    0xfc000000fc000000ULL, 0x04000000fc000000ULL, 0xfc00000004000000ULL, 0x0400000004000000ULL,
    0x0000fcfc00000000ULL, 0x000004fc00000000ULL, 0x0000fc0400000000ULL, 0x0000040400000000ULL,
    0x00fc00fc00000000ULL, 0x000400fc00000000ULL, 0x00fc000400000000ULL, 0x0004000400000000ULL,
    0xfc0000fc00000000ULL, 0x040000fc00000000ULL, 0xfc00000400000000ULL, 0x0400000400000000ULL,
    0x00fcfc0000000000ULL, 0x0004fc0000000000ULL, 0x00fc040000000000ULL, 0x0004040000000000ULL,
    0xfc00fc0000000000ULL, 0x0400fc0000000000ULL, 0xfc00040000000000ULL, 0x0400040000000000ULL,
    0xfcfc000000000000ULL, 0x04fc000000000000ULL, 0xfc04000000000000ULL, 0x0404000000000000ULL,
    0xfefefefefefefefeULL, 0x02fefefefefefe02ULL, 0x02fefefefefe02feULL, 0xfefefefefefe0202ULL,
    0x02fefefefe02fefeULL, 0xfefefefefe02fe02ULL, 0xfefefefefe0202feULL, 0x02fefefefe020202ULL,
    0x02fefefe02fefefeULL, 0xfefefefe02fefe02ULL, 0xfefefefe02fe02feULL, 0x02fefefe02fe0202ULL,
    0xfefefefe0202fefeULL, 0x02fefefe0202fe02ULL, 0x02fefefe020202feULL, 0xfefefefe02020202ULL,
    0x02fefe02fefefefeULL, 0xfefefe02fefefe02ULL, 0xfefefe02fefe02feULL, 0x02fefe02fefe0202ULL,
    0xfefefe02fe02fefeULL, 0x02fefe02fe02fe02ULL, 0x02fefe02fe0202feULL, 0xfefefe02fe020202ULL,
    0xfefefe0202fefefeULL, 0x02fefe0202fefe02ULL, 0x02fefe0202fe02feULL, 0xfefefe0202fe0202ULL,
    0x02fefe020202fefeULL, 0xfefefe020202fe02ULL, 0xfefefe02020202feULL, 0x02fefe0202020202ULL,
    0x02fe02fefefefefeULL, 0xfefe02fefefefe02ULL, 0xfefe02fefefe02feULL, 0x02fe02fefefe0202ULL,
    0xfefe02fefe02fefeULL, 0x02fe02fefe02fe02ULL, 0x02fe02fefe0202feULL, 0xfefe02fefe020202ULL,
    0xfefe02fe02fefefeULL, 0x02fe02fe02fefe02ULL, 0x02fe02fe02fe02feULL, 0xfefe02fe02fe0202ULL,
    0x02fe02fe0202fefeULL, 0xfefe02fe0202fe02ULL, 0xfefe02fe020202feULL, 0x02fe02fe02020202ULL,
    0xfefe0202fefefefeULL, 0x02fe0202fefefe02ULL, 0x02fe0202fefe02feULL, 0xfefe0202fefe0202ULL,
    0x02fe0202fe02fefeULL, 0xfefe0202fe02fe02ULL, 0xfefe0202fe0202feULL, 0x02fe0202fe020202ULL,
    0x02fe020202fefefeULL, 0xfefe020202fefe02ULL, 0xfefe020202fe02feULL, 0x02fe020202fe0202ULL,
    0xfefe02020202fefeULL, 0x02fe02020202fe02ULL, 0x02fe0202020202feULL, 0xfefe020202020202ULL,
    0x0202fefefefefefeULL, 0xfe02fefefefefe02ULL, 0xfe02fefefefe02feULL, 0x0202fefefefe0202ULL,
    0xfe02fefefe02fefeULL, 0x0202fefefe02fe02ULL, 0x0202fefefe0202feULL, 0xfe02fefefe020202ULL,
    0xfe02fefe02fefefeULL, 0x0202fefe02fefe02ULL, 0x0202fefe02fe02feULL, 0xfe02fefe02fe0202ULL,
    0x0202fefe0202fefeULL, 0xfe02fefe0202fe02ULL, 0xfe02fefe020202feULL, 0x0202fefe02020202ULL,
    0xfe02fe02fefefefeULL, 0x0202fe02fefefe02ULL, 0x0202fe02fefe02feULL, 0xfe02fe02fefe0202ULL,
    0x0202fe02fe02fefeULL, 0xfe02fe02fe02fe02ULL, 0xfe02fe02fe0202feULL, 0x0202fe02fe020202ULL,
    0x0202fe0202fefefeULL, 0xfe02fe0202fefe02ULL, 0xfe02fe0202fe02feULL, 0x0202fe0202fe0202ULL,
    0xfe02fe020202fefeULL, 0x0202fe020202fe02ULL, 0x0202fe02020202feULL, 0xfe02fe0202020202ULL,
    0xfe0202fefefefefeULL, 0x020202fefefefe02ULL, 0x020202fefefe02feULL, 0xfe0202fefefe0202ULL,
    0x020202fefe02fefeULL, 0xfe0202fefe02fe02ULL, 0xfe0202fefe0202feULL, 0x020202fefe020202ULL,
    0x020202fe02fefefeULL, 0xfe0202fe02fefe02ULL, 0xfe0202fe02fe02feULL, 0x020202fe02fe0202ULL,
    0xfe0202fe0202fefeULL, 0x020202fe0202fe02ULL, 0x020202fe020202feULL, 0xfe0202fe02020202ULL,
    0x02020202fefefefeULL, 0xfe020202fefefe02ULL, 0xfe020202fefe02feULL, 0x02020202fefe0202ULL,
    0xfe020202fe02fefeULL, 0x02020202fe02fe02ULL, 0x02020202fe0202feULL, 0xfe020202fe020202ULL,
    0xfe02020202fefefeULL, 0x0202020202fefe02ULL, 0x0202020202fe02feULL, 0xfe02020202fe0202ULL,
    0x020202020202fefeULL, 0xfe0202020202fe02ULL, 0xfe020202020202feULL, 0x0202020202020202ULL,
};

// Radius index k in 1..15 stands for 1/2 * 2^((k-8)/3) of the block norm over its group scale
// times sqrt(8); zero is the empty block. The values are the reference codec's. Global rather than
// constant memory: the lanes of a warp decode blocks of different radii, which the constant cache
// would serialize.
static __device__ const float kKVCacheE8RadiusScale[16] = {
    0.0000f, 0.0992f, 0.1250f, 0.1575f, 0.1984f, 0.2500f, 0.3150f, 0.3969f,
    0.5000f, 0.6300f, 0.7937f, 1.0000f, 1.2599f, 1.5874f, 2.0000f, 2.5198f,
};

// Table-driven decode. A decoded lane is rint(v * scale) over eleven arguments: a root lane v is
// one of {-4, -2, 0, 2, 4} and the residual axis adds +-1 to exactly one lane. kKVCacheE8RootLanes
// holds each root's lane indices v/2 + 2 as eight nibbles, dimension 0 lowest, so one byte permute
// per four lanes gathers the codes from the radius's row of kKVCacheE8EvenCodes; the axis lane is
// patched from kKVCacheE8OddCodes, v + 1 at index k + 1 and v - 1 at index k. The codes are the
// products of the reference arithmetic below, and tests/ops/test_e8_root_decode.cu checks every
// code pair against it.
static __device__ const std::uint32_t kKVCacheE8RootLanes[240] = {
    0x22222200u, 0x22222240u, 0x22222204u, 0x22222244u, 0x22222020u, 0x22222420u, 0x22222024u,
    0x22222424u, 0x22220220u, 0x22224220u, 0x22220224u, 0x22224224u, 0x22202220u, 0x22242220u,
    0x22202224u, 0x22242224u, 0x22022220u, 0x22422220u, 0x22022224u, 0x22422224u, 0x20222220u,
    0x24222220u, 0x20222224u, 0x24222224u, 0x02222220u, 0x42222220u, 0x02222224u, 0x42222224u,
    0x22222002u, 0x22222402u, 0x22222042u, 0x22222442u, 0x22220202u, 0x22224202u, 0x22220242u,
    0x22224242u, 0x22202202u, 0x22242202u, 0x22202242u, 0x22242242u, 0x22022202u, 0x22422202u,
    0x22022242u, 0x22422242u, 0x20222202u, 0x24222202u, 0x20222242u, 0x24222242u, 0x02222202u,
    0x42222202u, 0x02222242u, 0x42222242u, 0x22220022u, 0x22224022u, 0x22220422u, 0x22224422u,
    0x22202022u, 0x22242022u, 0x22202422u, 0x22242422u, 0x22022022u, 0x22422022u, 0x22022422u,
    0x22422422u, 0x20222022u, 0x24222022u, 0x20222422u, 0x24222422u, 0x02222022u, 0x42222022u,
    0x02222422u, 0x42222422u, 0x22200222u, 0x22240222u, 0x22204222u, 0x22244222u, 0x22020222u,
    0x22420222u, 0x22024222u, 0x22424222u, 0x20220222u, 0x24220222u, 0x20224222u, 0x24224222u,
    0x02220222u, 0x42220222u, 0x02224222u, 0x42224222u, 0x22002222u, 0x22402222u, 0x22042222u,
    0x22442222u, 0x20202222u, 0x24202222u, 0x20242222u, 0x24242222u, 0x02202222u, 0x42202222u,
    0x02242222u, 0x42242222u, 0x20022222u, 0x24022222u, 0x20422222u, 0x24422222u, 0x02022222u,
    0x42022222u, 0x02422222u, 0x42422222u, 0x00222222u, 0x40222222u, 0x04222222u, 0x44222222u,
    0x11111111u, 0x31111113u, 0x31111131u, 0x11111133u, 0x31111311u, 0x11111313u, 0x11111331u,
    0x31111333u, 0x31113111u, 0x11113113u, 0x11113131u, 0x31113133u, 0x11113311u, 0x31113313u,
    0x31113331u, 0x11113333u, 0x31131111u, 0x11131113u, 0x11131131u, 0x31131133u, 0x11131311u,
    0x31131313u, 0x31131331u, 0x11131333u, 0x11133111u, 0x31133113u, 0x31133131u, 0x11133133u,
    0x31133311u, 0x11133313u, 0x11133331u, 0x31133333u, 0x31311111u, 0x11311113u, 0x11311131u,
    0x31311133u, 0x11311311u, 0x31311313u, 0x31311331u, 0x11311333u, 0x11313111u, 0x31313113u,
    0x31313131u, 0x11313133u, 0x31313311u, 0x11313313u, 0x11313331u, 0x31313333u, 0x11331111u,
    0x31331113u, 0x31331131u, 0x11331133u, 0x31331311u, 0x11331313u, 0x11331331u, 0x31331333u,
    0x31333111u, 0x11333113u, 0x11333131u, 0x31333133u, 0x11333311u, 0x31333313u, 0x31333331u,
    0x11333333u, 0x33111111u, 0x13111113u, 0x13111131u, 0x33111133u, 0x13111311u, 0x33111313u,
    0x33111331u, 0x13111333u, 0x13113111u, 0x33113113u, 0x33113131u, 0x13113133u, 0x33113311u,
    0x13113313u, 0x13113331u, 0x33113333u, 0x13131111u, 0x33131113u, 0x33131131u, 0x13131133u,
    0x33131311u, 0x13131313u, 0x13131331u, 0x33131333u, 0x33133111u, 0x13133113u, 0x13133131u,
    0x33133133u, 0x13133311u, 0x33133313u, 0x33133331u, 0x13133333u, 0x13311111u, 0x33311113u,
    0x33311131u, 0x13311133u, 0x33311311u, 0x13311313u, 0x13311331u, 0x33311333u, 0x33313111u,
    0x13313113u, 0x13313131u, 0x33313133u, 0x13313311u, 0x33313313u, 0x33313331u, 0x13313333u,
    0x33331111u, 0x13331113u, 0x13331131u, 0x33331133u, 0x13331311u, 0x33331313u, 0x33331331u,
    0x13331333u, 0x13333111u, 0x33333113u, 0x33333131u, 0x13333133u, 0x33333311u, 0x13333313u,
    0x13333331u, 0x33333333u,
};

// rint(v * scale[radius]) for v = -4, -2, 0, 2, 4 in bytes 0..4.
static __device__ const std::uint64_t kKVCacheE8EvenCodes[16] = {
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x00000001000000ffULL,
    0x00000001000000ffULL, 0x00000001000000ffULL, 0x000000010100ffffULL, 0x000000020100fffeULL,
    0x000000020100fffeULL, 0x000000030100fffdULL, 0x000000030200fefdULL, 0x000000040200fefcULL,
    0x000000050300fdfbULL, 0x000000060300fdfaULL, 0x000000080400fcf8ULL, 0x0000000a0500fbf6ULL,
};

// rint(v * scale[radius]) for v = -5, -3, -1, 1, 3, 5 in bytes 0..5.
static __device__ const std::uint64_t kKVCacheE8OddCodes[16] = {
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x00000100000000ffULL, 0x00000100000000ffULL,
    0x000001010000ffffULL, 0x000001010000ffffULL, 0x000002010000fffeULL, 0x000002010000fffeULL,
    0x000002020000fefeULL, 0x0000030201fffefdULL, 0x0000040201fffefcULL, 0x0000050301fffdfbULL,
    0x0000060401fffcfaULL, 0x0000080502fefbf8ULL, 0x00000a0602fefaf6ULL, 0x00000d0803fdf8f3ULL,
};

// The radius index is the nearest integer to 3*log2(r)+8 clamped to [1,15], for r the block norm
// over its group scale times sqrt(8): one plus the count of thresholds 2^((k-8.5)/3), k=2..15, that
// r reaches, so no logarithm is evaluated. Relative norms below 0.08 are the empty block.
inline constexpr float kKVCacheE8RadiusFloor               = 0.08f;
static __constant__ const float kKVCacheE8RadiusBounds[14] = {
    0.2227247f, 0.2806155f, 0.3535534f, 0.4454494f, 0.5612310f, 0.7071068f, 0.8908987f,
    1.1224620f, 1.4142136f, 1.7817974f, 2.2449241f, 2.8284271f, 3.5635949f, 4.4898482f,
};

struct KVCacheE8RootBlock {
    std::uint8_t root;
    std::uint8_t radius_axis;
};

// Octet reductions in butterfly order: every lane of an octet ends with the same bits, because each
// step adds the same two operands, only commuted.
__device__ __forceinline__ float kv_cache_e8_octet_sum(float x) {
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) {
        x = __fadd_rn(x, __shfl_xor_sync(FullMask, x, step));
    }
    return x;
}

__device__ __forceinline__ int kv_cache_e8_octet_count(int x) {
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) { x += __shfl_xor_sync(FullMask, x, step); }
    return x;
}

// The largest (|value|, lowest index on ties) of the octet, carried with its index.
__device__ __forceinline__ void kv_cache_e8_octet_argmax(float& value, int& index) {
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) {
        const float other_value = __shfl_xor_sync(FullMask, value, step);
        const int other_index   = __shfl_xor_sync(FullMask, index, step);
        if (other_value > value || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ KVCacheE8RootBlock kv_cache_e8_root_encode(float x, float scale,
                                                                      int lane) {
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr float kSqrt8      = 2.82842712474619f;
    constexpr float kInvSqrt2   = 0.7071067811865475f;
    const int sub               = lane & 7;
    const int octet_base        = lane & ~7;

    const float norm     = __fsqrt_rn(kv_cache_e8_octet_sum(__fmul_rn(x, x)));
    const float relative = __fdiv_rn(norm, __fadd_rn(__fmul_rn(scale, kSqrt8), 1e-8f));
    int radius           = 0;
    if (relative >= kKVCacheE8RadiusFloor) {
        radius = 1;
#pragma unroll
        for (int k = 0; k < 14; ++k) { radius += relative >= kKVCacheE8RadiusBounds[k] ? 1 : 0; }
    }

    // The unit direction. An empty block still runs every octet shuffle below: they are full-mask,
    // and the other octets of the warp may be encoding.
    const float u     = __fmul_rn(x, __frcp_rn(__fadd_rn(norm, 1e-8f)));
    const float abs_u = fabsf(u);

    // Pair roots: the two largest |u|, lowest index first on ties.
    float top1 = abs_u;
    int top1_i = sub;
    float top2 = -1.0f;
    int top2_i = 8;
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) {
        const float o1         = __shfl_xor_sync(FullMask, top1, step);
        const int o1_i         = __shfl_xor_sync(FullMask, top1_i, step);
        const float o2         = __shfl_xor_sync(FullMask, top2, step);
        const int o2_i         = __shfl_xor_sync(FullMask, top2_i, step);
        const bool other_first = o1 > top1 || (o1 == top1 && o1_i < top1_i);
        if (other_first) {
            const bool mine_second = top1 > o2 || (top1 == o2 && top1_i < o2_i);
            top2                   = mine_second ? top1 : o2;
            top2_i                 = mine_second ? top1_i : o2_i;
            top1                   = o1;
            top1_i                 = o1_i;
        } else if (o1 > top2 || (o1 == top2 && o1_i < top2_i)) {
            top2   = o1;
            top2_i = o1_i;
        }
    }
    const int pair_i       = min(top1_i, top2_i);
    const int pair_j       = max(top1_i, top2_i);
    const int positive     = u >= 0.0f ? 1 : 0;
    const int positive_i   = __shfl_sync(FullMask, positive, octet_base + pair_i);
    const int positive_j   = __shfl_sync(FullMask, positive, octet_base + pair_j);
    const int pair_index   = pair_i * (15 - pair_i) / 2 + (pair_j - pair_i - 1);
    const float pair_score = __fadd_rn(top1, top2);
    const int pair_code    = (pair_index << 2) | (positive_i << 1) | positive_j;

    // Half roots: every sign of u, then the smallest |u| flipped when the count of minus signs is
    // odd.
    float smallest = abs_u;
    int smallest_i = sub;
#pragma unroll
    for (int step = 1; step < 8; step <<= 1) {
        const float other = __shfl_xor_sync(FullMask, smallest, step);
        const int other_i = __shfl_xor_sync(FullMask, smallest_i, step);
        if (other < smallest || (other == smallest && other_i < smallest_i)) {
            smallest   = other;
            smallest_i = other_i;
        }
    }
    const int minus  = kv_cache_e8_octet_count(1 - positive);
    const bool odd   = (minus & 1) != 0;
    float half_score = __fmul_rn(0.5f, kv_cache_e8_octet_sum(abs_u));
    if (odd) { half_score = __fsub_rn(half_score, smallest); }
    const int half_positive = (odd && sub == smallest_i) ? 1 - positive : positive;
    const int half_bits     = kv_cache_e8_octet_count(sub < 7 ? (half_positive << sub) : 0);

    const bool pair_root        = pair_score >= half_score;
    const float root_coordinate = pair_root ? (sub == pair_i   ? (positive_i != 0 ? 1.0f : -1.0f)
                                               : sub == pair_j ? (positive_j != 0 ? 1.0f : -1.0f)
                                                               : 0.0f)
                                            : (half_positive != 0 ? 0.5f : -0.5f);

    // Residual axis: the largest coordinate of u minus its projection on the root direction.
    const float direction  = __fmul_rn(root_coordinate, kInvSqrt2);
    const float projection = kv_cache_e8_octet_sum(__fmul_rn(u, direction));
    const float residual   = __fsub_rn(u, __fmul_rn(projection, direction));
    float largest          = fabsf(residual);
    int largest_i          = sub;
    kv_cache_e8_octet_argmax(largest, largest_i);
    const float residual_at = __shfl_sync(FullMask, residual, octet_base + largest_i);
    const int axis          = (largest_i << 1) | (residual_at >= 0.0f ? 0 : 1);

    if (radius == 0) { return {0, 0}; }
    return {static_cast<std::uint8_t>(pair_root ? pair_code : 112 + half_bits),
            static_cast<std::uint8_t>((radius << 4) | axis)};
}

// Store one G64 group of rk2v4-e8 key codes. This lane owns d0 = 64g+lane and d1 = d0+32; the
// octets of d0 are blocks 8g..8g+3 of the row and those of d1 blocks 8g+4..8g+7. row_base
// addresses byte 0 of the token's key row. Must be called by the full warp.
__device__ __forceinline__ void kv_cache_e8_root_store_key_group(std::uint8_t* plane,
                                                                 std::int64_t row_base, int group,
                                                                 float k0, float k1, float scale,
                                                                 int lane) {
    const KVCacheE8RootBlock first  = kv_cache_e8_root_encode(k0, scale, lane);
    const KVCacheE8RootBlock second = kv_cache_e8_root_encode(k1, scale, lane);
    if ((lane & 7) == 0) {
        const int block                       = group * 8 + (lane >> 3);
        plane[row_base + 2 * block]           = first.root;
        plane[row_base + 2 * block + 1]       = first.radius_axis;
        plane[row_base + 2 * (block + 4)]     = second.root;
        plane[row_base + 2 * (block + 4) + 1] = second.radius_axis;
    }
}

// One block back to eight signed int8 codes, dimension 0 in the lowest byte: the root and the unit
// residual axis, added and scaled by the radius. This is the arithmetic the tables encode.
__device__ __forceinline__ uint2
kv_cache_e8_root_decode_block_reference(std::uint32_t root, std::uint32_t radius_axis) {
    const std::uint32_t radius = radius_axis >> 4;
    if (radius == 0 || root >= 240) { return make_uint2(0, 0); }
    const std::uint64_t direction = __ldg(&kKVCacheE8Roots[root]);
    const int axis_dim            = static_cast<int>((radius_axis >> 1) & 7);
    const int axis_sign           = (radius_axis & 1) != 0 ? -1 : 1;
    const float scale             = __ldg(&kKVCacheE8RadiusScale[radius]);
    std::uint32_t words[2]        = {0, 0};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
        int value = static_cast<int>(static_cast<std::int8_t>((direction >> (8 * d)) & 0xff));
        if (d == axis_dim) { value += axis_sign; }
        const int code = __float2int_rn(__fmul_rn(static_cast<float>(value), scale));
        words[d >> 2] |= (static_cast<std::uint32_t>(code) & 0xffu) << (8 * (d & 3));
    }
    return make_uint2(words[0], words[1]);
}

__device__ __forceinline__ uint2 kv_cache_e8_root_decode_block(std::uint32_t root,
                                                               std::uint32_t radius_axis) {
    if (root >= 240) { return make_uint2(0, 0); }
    const std::uint32_t radius   = radius_axis >> 4;
    const std::uint32_t lanes    = __ldg(&kKVCacheE8RootLanes[root]);
    const std::uint64_t even     = __ldg(&kKVCacheE8EvenCodes[radius]);
    const auto even_low          = static_cast<std::uint32_t>(even);
    const auto even_high         = static_cast<std::uint32_t>(even >> 32);
    std::uint32_t words[2]       = {__byte_perm(even_low, even_high, lanes & 0xffffu),
                                    __byte_perm(even_low, even_high, lanes >> 16)};
    const std::uint32_t axis_dim = (radius_axis >> 1) & 7u;
    const std::uint32_t index    = ((lanes >> (4 * axis_dim)) & 0xfu) + ((radius_axis & 1u) ^ 1u);
    const std::uint32_t axis_code =
        static_cast<std::uint32_t>(__ldg(&kKVCacheE8OddCodes[radius]) >> (8 * index)) & 0xffu;
    const std::uint32_t shift = 8 * (axis_dim & 3u);
    std::uint32_t& word       = words[axis_dim >> 2];
    word                      = (word & ~(0xffu << shift)) | (axis_code << shift);
    return make_uint2(words[0], words[1]);
}

// Sixteen dimensions [d, d+16) from the four code bytes at row byte d/4: two blocks, expanded to
// sixteen int8 codes in dimension order for the INT8 key layout.
__device__ __forceinline__ int4 kv_cache_e8_root_unpack_i8x16(std::uint32_t codes) {
    const uint2 low  = kv_cache_e8_root_decode_block(codes & 0xffu, (codes >> 8) & 0xffu);
    const uint2 high = kv_cache_e8_root_decode_block((codes >> 16) & 0xffu, codes >> 24);
    return make_int4(static_cast<int>(low.x), static_cast<int>(low.y), static_cast<int>(high.x),
                     static_cast<int>(high.y));
}

} // namespace ninfer::ops
