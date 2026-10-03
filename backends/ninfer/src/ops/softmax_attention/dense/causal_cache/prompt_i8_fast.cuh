#pragma once

// Fast INT8-cache causal prompt kernel for the registered head geometries, selected per launch by
// CausalAttentionExecutionEnvelope::fast_prompt_kernel (ninfer-serve --fast-prefill-kernel); the
// default route keeps the kernel in prompt_i8.cuh.
//
//   * Each warp owns 16 query rows of one head for the whole key sweep (eight warps per CTA, or
//     four for launches too narrow to occupy every SM with 128-row CTAs), so
//     scores, probabilities and the D256 output accumulator never leave registers and no warp
//     waits for another between QK and PV. The only CTA barrier is the one that publishes a key
//     tile.
//   * Q and cached K use the same fixed register-only D256 rotation before their private G64
//     encoders; QK stays INT8 through m16n8k32.s8 Tensor Cores with the per-group scales applied
//     in FP32.
//   * K and V codes are double-buffered as raw INT8 pages (one 64-key page per tile). V is decoded
//     in registers: ldmatrix.trans on byte pairs yields, per lane, two adjacent keys for two
//     adjacent dimensions, which is exactly one FP16 B fragment for an even-dimension and an
//     odd-dimension n8 tile. Codes widen exactly to FP16 and receive their represented FP16 group
//     scale once, the same value the retired FP16 staging tile held.
//   * PV runs FP16 Tensor Cores with FP16 accumulation over the 64 keys of a tile, promoted into
//     the FP32 output accumulator once per tile. Every probability is at most one, so a partial is
//     bounded by 64 * 127 * max_scale; a tile whose largest V scale could exceed the FP16 range
//     decodes V with its scales divided by an exact power of two and multiplies the partial back.
//   * CTAs are issued longest-first so a causal prompt's heaviest row blocks do not form the tail.
//   * PackedValues serves rk8v4's value plane: two signed int4 codes per byte over a G32 scale. A
//     byte-pair ldmatrix.trans lane then holds four dimensions of two keys, which decode into the
//     B fragments of four n8 tiles (dimensions 4c+r of each 32-dimension group, r = 0..3); keys
//     are the INT8 G64 codes the int8 cache stores, so QK is shared.
//   * Keys selects a packed key plane (rk4v4, rk4v4-e8, rk2v4-e8). Its codes for the next tile are
//     loaded into registers where the INT8 path issues its copy and expanded into that stage's
//     INT8 K tile after this tile's PV, so QK is unchanged.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptI8FastBc     = 64;
inline constexpr int kCausalPromptI8FastGroups = kCausalPromptHeadDim / kKVCacheInt8Group;

inline constexpr int kCausalPromptI8FastTileBytes = kCausalPromptI8FastBc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptI8FastScaleBytes =
    kCausalPromptI8FastBc * kCausalPromptI8FastGroups * static_cast<int>(sizeof(__half));

// One pipeline stage: the K codes, the V codes (half as many bytes when packed), then the G64 key
// scales and the value scales (G64, or G32 when packed).
template <bool PackedValues>
struct CausalPromptI8FastStage {
    static constexpr int VGroups =
        PackedValues ? kCausalPromptHeadDim / kKVCacheInt4ValueGroup : kCausalPromptI8FastGroups;
    static constexpr int VRowBytes    = kCausalPromptHeadDim / (PackedValues ? 2 : 1);
    static constexpr int VTileBytes   = kCausalPromptI8FastBc * VRowBytes;
    static constexpr int VScaleBytes  = kCausalPromptI8FastBc * VGroups * static_cast<int>(sizeof(__half));
    static constexpr int KScaleOffset = kCausalPromptI8FastTileBytes + VTileBytes;
    static constexpr int Bytes        = KScaleOffset + kCausalPromptI8FastScaleBytes + VScaleBytes;
};
inline constexpr int kCausalPromptI8FastStageBytes = CausalPromptI8FastStage<false>::Bytes;

// Each warp owns 16 query rows. Eight warps fill an SM's register file; four warps serve
// launches too narrow to occupy every SM with 128-row CTAs.
template <int Warps, bool PackedValues = false>
struct CausalPromptI8FastShape {
    static_assert(Warps == 4 || Warps == 8);
    static constexpr int Threads   = Warps * 32;
    static constexpr int Br        = Warps * 16;
    static constexpr int QBytes    = Br * kCausalPromptHeadDim;
    static constexpr int SmemBytes = QBytes + 2 * CausalPromptI8FastStage<PackedValues>::Bytes;
};

// A 64-key FP16 partial is bounded by 64 * 127 * max_scale (every probability is at most one).
// Keeping max_scale at or below 8 leaves that bound, with FP16 rounding slack, under 65504. An
// int4 code is at most 8 in magnitude, so the packed coding allows scales up to 64.
inline constexpr float kCausalPromptI8FastF16PartialScaleLimit       = 8.0f;
inline constexpr float kCausalPromptI8FastPackedF16PartialScaleLimit = 64.0f;

static_assert(kCausalPromptI8FastBc == kPagedKVPageSize);
static_assert(kCausalPromptI8FastGroups == 4);
static_assert(CausalPromptI8FastShape<8>::SmemBytes == 100352);

__device__ __forceinline__ void causal_prompt_i8_fast_mma_f16_acc(unsigned& c0, unsigned& c1,
                                                                  unsigned a0, unsigned a1,
                                                                  unsigned a2, unsigned a3,
                                                                  unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One ldmatrix.trans b16 lane of an INT8 [key][d] tile holds the codes
// {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}. Returns the FP16 B-fragment halves
// {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code widened exactly and multiplied
// once by its key's represented group scale.
__device__ __forceinline__ void causal_prompt_i8_fast_decode_v_pair(unsigned codes, unsigned scales,
                                                                    unsigned& even, unsigned& odd) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte.
    const unsigned biased = codes ^ 0x80808080u;
    unsigned e            = __byte_perm(biased, 0x64646464u, 0x4240);
    unsigned o            = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), s2);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), s2);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

// One ldmatrix.trans b16 lane of a packed int4 [key][d/2] tile holds dimensions 4c..4c+3 of keys k
// and k+1, one nibble each (low nibble first, byte order: k lo pair, k hi pair, k+1 lo, k+1 hi).
// Returns, for r = 0..3, the FP16 B-fragment half {V[k][4c+r], V[k+1][4c+r]}, each code widened
// exactly and multiplied once by its key's represented group scale.
__device__ __forceinline__ void causal_prompt_i8_fast_decode_v_quad(unsigned codes, unsigned scales,
                                                                    unsigned (&out)[4]) {
    // Signed nibble ^ 8 is code + 8; 0x6400 | nibble is the FP16 value 1024 + nibble.
    const unsigned biased = codes ^ 0x88888888u;
    const __half2 offset  = __float2half2_rn(1032.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const unsigned widened = ((biased >> (4 * r)) & 0x000F000Fu) | 0x64006400u;
        const __half2 value    = __hmul2(__hsub2(load_vec<__half2>(&widened), offset), s2);
        out[r]                 = load_vec<unsigned>(&value);
    }
}

template <typename Geometry, typename Metadata, int Warps, bool PackedValues = false,
          KvKeyCoding Keys = KvKeyCoding::Int8>
__global__ __launch_bounds__(
    CausalPromptI8FastShape<Warps, PackedValues>::Threads,
    1) void causal_attention_prompt_i8_fast_kernel(const __nv_bfloat16* __restrict__ q,
                                                   const std::int8_t* __restrict__ cache_k,
                                                   const std::int8_t* __restrict__ cache_v,
                                                   const __half* __restrict__ cache_k_scale,
                                                   const __half* __restrict__ cache_v_scale,
                                                   Metadata metadata,
                                                   const std::int32_t* __restrict__ positions,
                                                   float scale, __nv_bfloat16* __restrict__ out,
                                                   std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int DB16          = D / 2;
    using Shape                 = CausalPromptI8FastShape<Warps, PackedValues>;
    using Stage                 = CausalPromptI8FastStage<PackedValues>;
    constexpr int VGroups       = Stage::VGroups;
    constexpr int VRowBytes     = Stage::VRowBytes;
    constexpr bool PackedKeys   = Keys != KvKeyCoding::Int8;
    static_assert(!PackedKeys || PackedValues, "packed keys pair with packed values");
    constexpr int Threads       = Shape::Threads;
    constexpr int Br            = Shape::Br;
    constexpr int Bc            = kCausalPromptI8FastBc;
    constexpr int Groups        = kCausalPromptI8FastGroups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int DBlocks       = D / 16;
    constexpr int GroupDBlocks  = kKVCacheInt8Group / 16;
    constexpr int PassDBlocks   = 2;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(GroupDBlocks == 4);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Longest-first issue order: the first QHeads CTAs take the last row block of every head.
    const int row_blocks = static_cast<int>(gridDim.x);
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % Geometry::QHeads;
    const int q0         = (row_blocks - 1 - linear / Geometry::QHeads) * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    // Rows past the valid token count are inactive columns and publish zeros.
    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (token >= width) { return; }
        const bool valid   = token < tokens;
        const uint2 packed = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                        pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        store_vec(&out[causal_prompt_q_index<Geometry>(q_head, d0, token)], packed);
    };

    if (q0 >= tokens) {
        for (int element = tid; element < Br * (D / 4); element += Threads) {
            const int row = element / (D / 4);
            store_row(row, (element - row * (D / 4)) * 4, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows                  = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;

    // Each warp rotates and encodes its own 16 rows, keeping the two rows each lane owns in the
    // MMA C layout as register scales.
    float q_scale_r[2][Groups];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row    = warp * 16 + r;
        const int token  = q0 + row;
        const bool valid = row < rows;
        float q_values[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q_values[k] =
                valid ? __bfloat162float(
                            q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * k, token)])
                      : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            const float mx  = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
            const float qs  = mx > 0.0f ? mx / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_prompt_store_byte_swizzled(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_prompt_store_byte_swizzled(q_i8, row, d0 + 32,
                                              kv_cache_int8_quant_code(x1, inv));
            if (gid == (r & 7)) { q_scale_r[r >> 3][grp] = qs; }
        }
    }

    const auto stage_base = [&](int stage) {
        return smem_raw + Shape::QBytes + stage * Stage::Bytes;
    };

    constexpr int KChunks          = Bc * (D / 16);
    constexpr int KChunksPerThread = KChunks / Threads;
    static_assert(KChunks % Threads == 0);
    using KeyChunk = KvPackedKeyChunk<PackedKeys ? Keys : KvKeyCoding::Lloyd4>;
    KeyChunk k_packed[PackedKeys ? KChunksPerThread : 1];
    // Packed keys past the last visible key keep their placeholder's expansion: their scales are
    // zero-filled and their scores masked.
    const auto commit_keys = [&](int kb) {
        if constexpr (PackedKeys) {
            std::int8_t* k_s = reinterpret_cast<std::int8_t*>(stage_base(kb & 1));
#pragma unroll
            for (int i = 0; i < KChunksPerThread; ++i) {
                const int chunk = tid + i * Threads;
                const int key   = chunk >> 4;
                const int c     = chunk & 15;
                store_vec(k_s + key * D + ((c ^ (key & 7)) << 4),
                          kv_cache_packed_key_expand16<Keys>(k_packed[i]));
            }
        }
    };

    // One tile is one physical page of this KV head: 16 KiB of K codes, 16 KiB of V codes (8 KiB
    // packed) and their scales, all contiguous in the cache. Keys past the CTA's last visible key
    // are zero-filled so masked columns stay finite.
    const auto issue_tile = [&](int kb) {
        unsigned char* base = stage_base(kb & 1);
        std::int8_t* k_s    = reinterpret_cast<std::int8_t*>(base);
        std::int8_t* v_s    = k_s + kCausalPromptI8FastTileBytes;
        __half* ks_s        = reinterpret_cast<__half*>(base + Stage::KScaleOffset);
        __half* vs_s        = ks_s + Bc * Groups;
        const int page      = block_table[kb];
        const int valid     = min(Bc, max_query_abs + 1 - kb * Bc);
        if constexpr (PackedKeys) {
#pragma unroll
            for (int i = 0; i < KChunksPerThread; ++i) {
                const int chunk = tid + i * Threads;
                const int key   = chunk >> 4;
                k_packed[i]     = KeyChunk{};
                if (key < valid) {
                    k_packed[i] = load_vec<KeyChunk>(
                        reinterpret_cast<const std::uint8_t*>(cache_k) +
                        kv_cache_packed_key_chunk_index<Geometry, Keys>(page, kv_head,
                                                                        (chunk & 15) * 16, key));
                }
            }
        } else {
            const std::int64_t code_base =
                kv_cache_int8_quant_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < kCausalPromptI8FastTileBytes / 16 / Threads; ++i) {
                const int chunk        = tid + i * Threads;
                const int key          = chunk >> 4;
                const int c            = chunk & 15;
                const int dst          = key * D + ((c ^ (key & 7)) << 4);
                const int bytes        = key < valid ? 16 : 0;
                const std::int64_t src = code_base + key * D + c * 16;
                cp_async_zfill<16, Cache::cg>(k_s + dst, cache_k + src, bytes);
                if constexpr (!PackedValues) {
                    cp_async_zfill<16, Cache::cg>(v_s + dst, cache_v + src, bytes);
                }
            }
        }
        if constexpr (PackedValues) {
            const std::int64_t value_base =
                kv_cache_int4_value_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < Stage::VTileBytes / 16 / Threads; ++i) {
                const int chunk = tid + i * Threads;
                const int key   = chunk >> 3;
                const int c     = chunk & 7;
                cp_async_zfill<16, Cache::cg>(v_s + key * VRowBytes + ((c ^ (key & 7)) << 4),
                                              cache_v + value_base + key * VRowBytes + c * 16,
                                              key < valid ? 16 : 0);
            }
        }
        if (tid < Bc) {
            cp_async_zfill<8>(ks_s + tid * Groups,
                              cache_k_scale +
                                  kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, tid),
                              tid < valid ? 8 : 0);
        } else if (tid < 2 * Bc) {
            const int key = tid - Bc;
            if constexpr (PackedValues) {
                cp_async_zfill<16>(vs_s + key * VGroups,
                                   cache_v_scale + kv_cache_int4_value_scale_index<Geometry>(
                                                       page, kv_head, 0, key),
                                   key < valid ? 16 : 0);
            } else {
                cp_async_zfill<8>(vs_s + key * VGroups,
                                  cache_v_scale +
                                      kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, key),
                                  key < valid ? 8 : 0);
            }
        }
        cp_commit();
    };

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;

    // Rows this warp owns, for warp-uniform tile skipping and mask elision.
    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    // INT8 values: [16-dimension block][even/odd n8 tile]; packed values: [G32 group][r] where
    // tile r of a group holds its dimensions 4c+r. Both are 128 floats per lane.
    constexpr int AccOuter = PackedValues ? VGroups : DBlocks;
    constexpr int AccInner = PackedValues ? 4 : 2;
    float acc[AccOuter][AccInner][4];
#pragma unroll
    for (int b = 0; b < AccOuter; ++b) {
#pragma unroll
        for (int p = 0; p < AccInner; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    // What QK hands to PV for one tile: the probability A fragments, the row rescale factors, and
    // the exact power of two that keeps the tile's FP16 partials representable.
    unsigned pa[PVKs][4];
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    int tile_shift    = 0;
    bool tile_live    = false;

    const auto qk_softmax = [&](int kb) {
        tile_live    = false;
        const int k0 = kb * Bc;
        if (!warp_active || k0 > warp_max_qabs) { return; }
        const unsigned char* base  = stage_base(kb & 1);
        const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(base);
        const __half* ks_s = reinterpret_cast<const __half*>(base + Stage::KScaleOffset);
        const __half* vs_s = ks_s + Bc * Groups;

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            unsigned af[GroupKc][4];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
                const int keya  = nt * 8 + 2 * lid;
                const float ks0 = __half2float(ks_s[keya * Groups + grp]);
                const float ks1 = __half2float(ks_s[(keya + 1) * Groups + grp]);
                const float qs0 = q_scale_r[0][grp];
                const float qs1 = q_scale_r[1][grp];
                score[nt][0]    = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                score[nt][1]    = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                score[nt][2]    = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                score[nt][3]    = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
            }
        }

        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        float bm0            = -CUDART_INF_F;
        float bm1            = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        // Probabilities become the PV A fragments directly: k-step j covers keys 16j..16j+15,
        // which are score tiles 2j (keys 2t, 2t+1) and 2j+1 (keys 8+2t, 9+2t).
        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
        }
        // Row sums stay lane-partial; the quad is reduced once after the sweep.
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);

        // Every probability is at most one, so a 64-key FP16 partial is bounded by
        // 64 * 127 * max_scale. A tile whose largest V scale exceeds the limit decodes V with its
        // scales divided by an exact power of two and multiplies the partial back at promotion.
        static_assert(Bc * VGroups % (32 * 8) == 0);
        __half2 vmax2 = __float2half2_rn(0.0f);
#pragma unroll
        for (int part = 0; part < Bc * VGroups / (32 * 8); ++part) {
            const uint4 v8 = load_vec<uint4>(&vs_s[8 * (lane + 32 * part)]);
            vmax2          = __hmax2(
                vmax2, __hmax2(__hmax2(__habs2(load_vec<__half2>(&v8.x)),
                                       __habs2(load_vec<__half2>(&v8.y))),
                               __hmax2(__habs2(load_vec<__half2>(&v8.z)),
                                       __habs2(load_vec<__half2>(&v8.w)))));
        }
        const float vmax = warp_max(fmaxf(__low2float(vmax2), __high2float(vmax2)), FullMask);
        constexpr float limit = PackedValues ? kCausalPromptI8FastPackedF16PartialScaleLimit
                                             : kCausalPromptI8FastF16PartialScaleLimit;
        tile_shift = 0;
        if (vmax > limit) {
            while (ldexpf(vmax, -tile_shift) > limit) { ++tile_shift; }
        }
        tile_live = true;
    };

    const auto pv_packed = [&](int kb) {
        const unsigned char* base = stage_base(kb & 1);
        const std::int8_t* v_s =
            reinterpret_cast<const std::int8_t*>(base) + kCausalPromptI8FastTileBytes;
        const __half* vs_s =
            reinterpret_cast<const __half*>(base + Stage::KScaleOffset) + Bc * Groups;
        const __half2 mul   = __float2half2_rn(ldexpf(1.0f, -tile_shift));
        const float unscale = ldexpf(1.0f, tile_shift);
#pragma unroll
        for (int grp = 0; grp < VGroups; ++grp) {
            unsigned h[4][2];
#pragma unroll
            for (int r = 0; r < 4; ++r) { h[r][0] = h[r][1] = 0u; }
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                // Group scales of the keys each lane supplies: (2t, 2t+1) and (8+2t, 9+2t).
                const int key = j * 16 + 2 * lid;
                __half2 lo =
                    __halves2half2(vs_s[key * VGroups + grp], vs_s[(key + 1) * VGroups + grp]);
                __half2 hi = __halves2half2(vs_s[(key + 8) * VGroups + grp],
                                            vs_s[(key + 9) * VGroups + grp]);
                if (tile_shift != 0) {
                    lo = __hmul2(lo, mul);
                    hi = __hmul2(hi, mul);
                }
                // Matrices: keys 16j+0..7, then 16j+8..15, of this group's 16-byte chunk.
                const int row = j * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                unsigned r0, r1;
                ldmatrix_x2_t(r0, r1, smem_addr(&v_s[row * VRowBytes + ((grp ^ (row & 7)) << 4)]));
                unsigned b_lo[4], b_hi[4];
                causal_prompt_i8_fast_decode_v_quad(r0, load_vec<unsigned>(&lo), b_lo);
                causal_prompt_i8_fast_decode_v_quad(r1, load_vec<unsigned>(&hi), b_hi);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    causal_prompt_i8_fast_mma_f16_acc(h[r][0], h[r][1], pa[j][0], pa[j][1],
                                                      pa[j][2], pa[j][3], b_lo[r], b_hi[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                float (&a)[4] = acc[grp][r];
                float2 r0     = __half22float2(load_vec<__half2>(&h[r][0]));
                float2 r1     = __half22float2(load_vec<__half2>(&h[r][1]));
                if (tile_shift != 0) {
                    r0.x *= unscale;
                    r0.y *= unscale;
                    r1.x *= unscale;
                    r1.y *= unscale;
                }
                a[0] = __fmaf_rn(a[0], tile_alpha0, r0.x);
                a[1] = __fmaf_rn(a[1], tile_alpha0, r0.y);
                a[2] = __fmaf_rn(a[2], tile_alpha1, r1.x);
                a[3] = __fmaf_rn(a[3], tile_alpha1, r1.y);
            }
        }
    };

    const auto pv = [&](int kb) {
        if (!tile_live) { return; }
        if constexpr (PackedValues) {
            pv_packed(kb);
            return;
        }
        const std::int8_t* v_s =
            reinterpret_cast<const std::int8_t*>(stage_base(kb & 1)) + kCausalPromptI8FastTileBytes;
        const __half* vs_s =
            reinterpret_cast<const __half*>(v_s + kCausalPromptI8FastTileBytes) + Bc * Groups;
        const __half2 mul   = __float2half2_rn(ldexpf(1.0f, -tile_shift));
        const float unscale = ldexpf(1.0f, tile_shift);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            // Group scales of the keys each lane supplies: (2t, 2t+1) and (8+2t, 9+2t) per k-step.
            unsigned vsc[PVKs][2];
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                const int key = j * 16 + 2 * lid;
                __half2 lo =
                    __halves2half2(vs_s[key * Groups + grp], vs_s[(key + 1) * Groups + grp]);
                __half2 hi =
                    __halves2half2(vs_s[(key + 8) * Groups + grp], vs_s[(key + 9) * Groups + grp]);
                if (tile_shift != 0) {
                    lo = __hmul2(lo, mul);
                    hi = __hmul2(hi, mul);
                }
                vsc[j][0] = load_vec<unsigned>(&lo);
                vsc[j][1] = load_vec<unsigned>(&hi);
            }
#pragma unroll
            for (int pass = 0; pass < GroupDBlocks / PassDBlocks; ++pass) {
                const int db0 = grp * GroupDBlocks + pass * PassDBlocks;
                unsigned h[PassDBlocks][2][2];
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) { h[b][p][0] = h[b][p][1] = 0u; }
                }
#pragma unroll
                for (int j = 0; j < PVKs; ++j) {
#pragma unroll
                    for (int q2 = 0; q2 < PassDBlocks / 2; ++q2) {
                        // Matrices: keys 16j+0..7 and 16j+8..15 of d-block db, then of db + 1.
                        const int db    = db0 + 2 * q2;
                        const int key   = j * 16 + ((a_mat & 1) << 3) + a_rin;
                        const int chunk = db + (a_mat >> 1);
                        unsigned r[4];
                        ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                      smem_addr(&v_s[key * D + ((chunk ^ (key & 7)) << 4)]));
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            unsigned even_lo, odd_lo, even_hi, odd_hi;
                            causal_prompt_i8_fast_decode_v_pair(r[2 * b], vsc[j][0], even_lo,
                                                                odd_lo);
                            causal_prompt_i8_fast_decode_v_pair(r[2 * b + 1], vsc[j][1], even_hi,
                                                                odd_hi);
                            unsigned (&he)[2] = h[2 * q2 + b][0];
                            unsigned (&ho)[2] = h[2 * q2 + b][1];
                            causal_prompt_i8_fast_mma_f16_acc(he[0], he[1], pa[j][0], pa[j][1],
                                                              pa[j][2], pa[j][3], even_lo, even_hi);
                            causal_prompt_i8_fast_mma_f16_acc(ho[0], ho[1], pa[j][0], pa[j][1],
                                                              pa[j][2], pa[j][3], odd_lo, odd_hi);
                        }
                    }
                }
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        float (&a)[4] = acc[db0 + b][p];
                        float2 r0     = __half22float2(load_vec<__half2>(&h[b][p][0]));
                        float2 r1     = __half22float2(load_vec<__half2>(&h[b][p][1]));
                        if (tile_shift != 0) {
                            r0.x *= unscale;
                            r0.y *= unscale;
                            r1.x *= unscale;
                            r1.y *= unscale;
                        }
                        a[0] = __fmaf_rn(a[0], tile_alpha0, r0.x);
                        a[1] = __fmaf_rn(a[1], tile_alpha0, r0.y);
                        a[2] = __fmaf_rn(a[2], tile_alpha1, r1.x);
                        a[3] = __fmaf_rn(a[3], tile_alpha1, r1.y);
                    }
                }
            }
        }
    };

    // Every warp publishes one tile per barrier; the next tile's copy overlaps this tile's math.
    issue_tile(0);
    commit_keys(0);
#pragma unroll 1
    for (int kb = 0; kb < key_blocks; ++kb) {
        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) { issue_tile(kb + 1); }
        qk_softmax(kb);
        pv(kb);
        if (kb + 1 < key_blocks) { commit_keys(kb + 1); }
    }

    running_l0         = warp_sum<4>(running_l0, FullMask);
    running_l1         = warp_sum<4>(running_l1, FullMask);
    const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
    const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;
    if constexpr (PackedValues) {
        // Tile r of group g holds dimensions 32g + 4n + r, so lane (g, t) owns the eight
        // contiguous dimensions 32g + 8t .. 32g + 8t + 7 of its two rows: columns 2t (first four)
        // and 2t + 1 (last four) of the four tiles.
#pragma unroll
        for (int grp = 0; grp < AccOuter; ++grp) {
            const int d0 = grp * 32 + 8 * lid;
            store_row(row0, d0, acc[grp][0][0] * inv_l0, acc[grp][1][0] * inv_l0,
                      acc[grp][2][0] * inv_l0, acc[grp][3][0] * inv_l0);
            store_row(row0, d0 + 4, acc[grp][0][1] * inv_l0, acc[grp][1][1] * inv_l0,
                      acc[grp][2][1] * inv_l0, acc[grp][3][1] * inv_l0);
            store_row(row1, d0, acc[grp][0][2] * inv_l1, acc[grp][1][2] * inv_l1,
                      acc[grp][2][2] * inv_l1, acc[grp][3][2] * inv_l1);
            store_row(row1, d0 + 4, acc[grp][0][3] * inv_l1, acc[grp][1][3] * inv_l1,
                      acc[grp][2][3] * inv_l1, acc[grp][3][3] * inv_l1);
        }
    } else {
        // Even n8 tiles hold dimensions 16b + 2n and odd tiles 16b + 2n + 1, so lane (g, t) owns
        // the four contiguous dimensions 16b + 4t .. 16b + 4t + 3 of its two rows.
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            const int d0 = b * 16 + 4 * lid;
            store_row(row0, d0, acc[b][0][0] * inv_l0, acc[b][1][0] * inv_l0,
                      acc[b][0][1] * inv_l0, acc[b][1][1] * inv_l0);
            store_row(row1, d0, acc[b][0][2] * inv_l1, acc[b][1][2] * inv_l1,
                      acc[b][0][3] * inv_l1, acc[b][1][3] * inv_l1);
        }
    }
}

} // namespace ninfer::ops
