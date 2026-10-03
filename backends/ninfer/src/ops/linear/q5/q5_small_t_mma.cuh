#pragma once

// Q5 row-split small-T MMA core, the Q5 counterpart of q4_ksplit_mma.cuh. For narrow decode
// extents -- an MTP verify is T = K+1 = 4 -- the SIMT split2/split4 kernels grow with T because
// every column is its own FMA stream, while an MMA over an eight-column tile costs the same from
// T=1 to T=8.
//
// A CTA owns 16 weight rows (one m16 tile) and splits each 512-wide K slab over eight warps, one
// quantization group per warp -- or, with KWarps < 8, 128 / KWarps rows sharing one staged slab
// of 64 * KWarps k (ops/common/small_t_layout.cuh). Nibbles, high bits and scales are staged with cp.async; each lane
// decodes its eight contiguous code bytes per row straight into A fragments (exactly: a Q5 code is
// an integer in [-16, 15]) and multiplies them against the staged activations with bf16 mma.sync
// in a permuted k order (see the kernel); each warp folds its group's fp16 scale into an fp32
// accumulator, and the eight K partials are reduced through shared memory at the end.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/small_t_layout.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Per-row staging of one K slab for the layout SmallTLayout<KWarps> (ops/common/small_t_layout.cuh).
template <int KWarps, int TilesPerWarp = 1>
struct Q5SmallTSchedule : SmallTLayout<KWarps, TilesPerWarp> {
    static constexpr int kCodeRowBytes = KWarps * 32;
    static constexpr int kHighRowBytes = KWarps * 8;
    // A lane reads eight contiguous code bytes per row (LDS.64) and two high bytes (LDS.16); a
    // half-warp of 64-bit loads spans four rows, so code rows are staggered by 32 B (8 banks) and
    // the eight rows of a warp's high-bit loads by 16 or 48 B, which puts every row a warp touches
    // in its own banks.
    static constexpr int kCodeRowPad = 32;
    static constexpr int kHighRowPad = KWarps == 2 ? 32 : 16;
    // Activation columns are 64 bf16 (128 B) per K warp; 16 B of padding staggers adjacent
    // columns by four banks for the quarter-warp 128-bit B loads.
    static constexpr int kXColPad = 8;
    static constexpr Cache kWeightCache = Cache::cg;
};

// Shared memory of q5_small_t_mma_kernel: a ring of Stages slabs, reused for the final K
// reduction. Rings deeper than two lose more occupancy than they hide latency (measured to four
// stages, past the 48 KB static limit), so it stays static.
template <int KWarps, int XCols, int Stages, int TilesPerWarp = 1>
struct Q5SmallTStorage {
    using Schedule = Q5SmallTSchedule<KWarps, TilesPerWarp>;
    struct Stage {
        std::uint8_t codes[Schedule::kRowsPerCta][Schedule::kCodeRowBytes + Schedule::kCodeRowPad];
        std::uint8_t high[Schedule::kRowsPerCta][Schedule::kHighRowBytes + Schedule::kHighRowPad];
        __nv_bfloat16 activations[KWarps][XCols][64 + Schedule::kXColPad];
        std::uint16_t scales[Schedule::kRowsPerCta][KWarps];
    };
    union Shared {
        Stage stages[Stages];
        float partial[Schedule::kWarps * TilesPerWarp * (XCols < 8 ? 1 : XCols / 8) * 32 * 4];
    };
    static constexpr int kBytes = sizeof(Shared);
};

// Eight Q5 codes -> four bf16 pairs, exactly, without int-to-float conversion. `word` holds four
// code bytes (weight 2j in byte j's low nibble, 2j + 1 in its high nibble) and `high` their eight
// high bits (bit i for weight i). A code is the five-bit two's-complement value nib - 16 * h; with
// the high bit inverted into bit 4 it becomes v = nib + 16 * (1 - h) in [0, 31], which placed in
// the mantissa of bf16 128.0 reads 128 + v, and subtracting 144 leaves the code. out[j] holds
// weights (2j, 2j + 1).
__device__ __forceinline__ void q5_small_t_decode_eight(unsigned word, unsigned high,
                                                        unsigned (&out)[4]) {
    const unsigned kMagic = 0x43004300u; // bf16 128.0 in both halves
    const unsigned kBias  = 0x43104310u; // bf16 144.0 in both halves
    const unsigned lo     = word & 0x0f0f0f0fu;
    const unsigned hi     = (word >> 4) & 0x0f0f0f0fu;
    unsigned even         = __byte_perm(lo, hi, 0x5140); // weights 0..3, one per byte
    unsigned odd          = __byte_perm(lo, hi, 0x7362); // weights 4..7
    const unsigned inv    = ~high;
    // (b & 0xf) * 0x00204081 lays bits 0..3 at 0, 8, 16, 24 with no carries between the copies.
    even |= (((inv & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
    odd |= ((((inv >> 4) & 0xfu) * 0x00204081u) & 0x01010101u) << 4;
    const unsigned biased[4] = {
        __byte_perm(even, kMagic, 0x7150), __byte_perm(even, kMagic, 0x7372),
        __byte_perm(odd, kMagic, 0x7150), __byte_perm(odd, kMagic, 0x7372)};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_bfloat162 value = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&biased[j]),
                                             *reinterpret_cast<const __nv_bfloat162*>(&kBias));
        out[j]                     = *reinterpret_cast<const unsigned*>(&value);
    }
}

// Rows: output rows. K: reduction length, a multiple of the slab (64 * KWarps). XCols: activation
// columns staged (4, 8, 16 or 32), covered by max(1, XCols / 8) eight-column MMA tiles; the B
// fragments of columns at or past `columns` are zero. Stages: depth of the cp.async ring (1..3).
// KWarps: the warp layout, SmallTLayout<KWarps>; the grid is Rows / (128 / KWarps).
// Epilogue::store(row, col0, v) receives, for the lane's rows row and row + 8, v.x = (row, col0),
// v.y = (row, col0 + 1), v.z = (row + 8, col0), v.w = (row + 8, col0 + 1); columns at or past
// `columns` must be ignored by the epilogue.
//
// K order. The MMA's k slots may be any permutation of a group's 64 k, provided A and B agree.
// Lane lid is given k 16*lid .. 16*lid + 15 of its warp's group: for step ks, slots (2*lid,
// 2*lid + 1) carry k 16*lid + 4*ks + {0, 1} and slots (2*lid + 8, 2*lid + 9) carry
// 16*lid + 4*ks + {2, 3}. The lane's A operand over the four steps is then eight contiguous code
// bytes per row -- one 64-bit load, decoded by q5_small_t_decode_eight -- and its B operand
// sixteen contiguous activations of one column -- two 128-bit loads, no ldmatrix -- where the
// natural slot order needs a byte load, a decode and an ldmatrix per fragment.
template <int Rows, int K, int XCols, int Stages, class Epilogue, int KWarps = 8,
          int TilesPerWarp = 1>
__launch_bounds__(256, (KWarps < 8 || XCols >= 32 ? 2 : (XCols >= 16 || Stages == 2 ? 4 : 6)))
    __global__ void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                          const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ high_bits,
                                          const std::uint8_t* __restrict__ scales,
                                          Epilogue epilogue, int columns) {
    using Schedule              = Q5SmallTSchedule<KWarps, TilesPerWarp>;
    constexpr int kTpw          = TilesPerWarp;
    constexpr int kTileK        = 64;
    constexpr int kKWarps       = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kSlabs        = K / kGroupK;
    constexpr int kGroupsPerRow = K / 64;
    constexpr int kNt           = XCols < 8 ? 1 : XCols / 8;
    static_assert(XCols == 4 || XCols == 8 || XCols == 16 || XCols == 32);
    static_assert(Stages >= 1 && Stages <= 4);
    static_assert(K % kGroupK == 0, "K must be a whole number of slabs");
    static_assert(Rows % kRowsPerCta == 0);

    using Storage = Q5SmallTStorage<KWarps, XCols, Stages, TilesPerWarp>;
    using Stage   = typename Storage::Stage;
    __shared__ __align__(16) typename Storage::Shared shared;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int k_split  = warp % kKWarps;
    const int first_tile = warp / kKWarps * kTpw; // this warp's tiles: first_tile + [0, kTpw)
    const int cta_row0   = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int slab_k0, Stage& stage) {
        constexpr int kChunksPerCol = kGroupK / 8;
        const int items             = columns * kChunksPerCol;
        for (int item = tid; item < items; item += Schedule::kThreads) {
            const int col   = item / kChunksPerCol;
            const int chunk = item - col * kChunksPerCol;
            cp_async<16>(&stage.activations[chunk / 8][col][(chunk % 8) * 8],
                         &x[static_cast<std::int64_t>(col) * K + slab_k0 + chunk * 8]);
        }
    };

    // Per row: the code chunks, then the high-bit chunks, then one scale copy.
    const auto stage_weight = [&](int slab_k0, Stage& stage) {
        constexpr int kCodeChunks  = Schedule::kCodeRowBytes / 16;
        constexpr int kHighChunks  = (Schedule::kHighRowBytes + 15) / 16;
        constexpr int kItemsPerRow = kCodeChunks + kHighChunks + 1;
        static_assert(Schedule::kHighRowBytes % 16 == 0);
        for (int item = tid; item < kRowsPerCta * kItemsPerRow; item += Schedule::kThreads) {
            const int row        = item / kItemsPerRow;
            const int j          = item - row * kItemsPerRow;
            const std::int64_t r = cta_row0 + row;
            if (j < kCodeChunks) {
                cp_async<16, Schedule::kWeightCache>(&stage.codes[row][j * 16],
                                        codes + r * (K / 2) + slab_k0 / 2 + j * 16);
            } else if (j < kCodeChunks + kHighChunks) {
                const int chunk = j - kCodeChunks;
                cp_async<16, Schedule::kWeightCache>(&stage.high[row][chunk * 16],
                                        high_bits + r * (K / 8) + slab_k0 / 8 + chunk * 16);
            } else {
                cp_async<2 * kKWarps>(&stage.scales[row][0],
                                      scales + (r * kGroupsPerRow + slab_k0 / 64) * 2);
            }
        }
    };

    const auto issue_slab = [&](int slab) {
        if (slab < kSlabs) {
            Stage& stage = shared.stages[slab % Stages];
            stage_weight(slab * kGroupK, stage);
            stage_x(slab * kGroupK, stage);
        }
        cp_commit(); // empty commits keep cp_wait<Stages - 1> exact through the tail
    };

    const int code_byte     = k_split * (kTileK / 2) + 8 * lid;
    const int high_byte     = k_split * (kTileK / 8) + 2 * lid;
    float acc[kTpw][kNt][4] = {};

#pragma unroll
    for (int prefetch = 0; prefetch < Stages - 1; ++prefetch) { issue_slab(prefetch); }

#pragma unroll 1
    for (int slab = 0; slab < kSlabs; ++slab) {
        issue_slab(slab + Stages - 1);
        cp_wait<Stages - 1>();
        __syncthreads();

        const Stage& stage = shared.stages[slab % Stages];
        uint2 top[kTpw], bot[kTpw];
        unsigned top_high[kTpw], bot_high[kTpw];
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row = (first_tile + t) * 16 + gid;
            top[t]             = load_vec<uint2>(&stage.codes[tile_row][code_byte]);
            bot[t]             = load_vec<uint2>(&stage.codes[tile_row + 8][code_byte]);
            top_high[t] = *reinterpret_cast<const std::uint16_t*>(&stage.high[tile_row][high_byte]);
            bot_high[t] =
                *reinterpret_cast<const std::uint16_t*>(&stage.high[tile_row + 8][high_byte]);
        }

        float group_acc[kTpw][kNt][4] = {};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            // Four code bytes and one high byte per row cover steps ks = 2*half and 2*half + 1.
            unsigned a_top[kTpw][4], a_bot[kTpw][4];
#pragma unroll
            for (int t = 0; t < kTpw; ++t) {
                q5_small_t_decode_eight(half == 0 ? top[t].x : top[t].y,
                                        (top_high[t] >> (8 * half)) & 0xffu, a_top[t]);
                q5_small_t_decode_eight(half == 0 ? bot[t].x : bot[t].y,
                                        (bot_high[t] >> (8 * half)) & 0xffu, a_bot[t]);
            }
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = nt * 8 + gid;
                uint4 b       = make_uint4(0u, 0u, 0u, 0u);
                if (col < columns) {
                    b = load_vec<uint4>(&stage.activations[k_split][col][16 * lid + 8 * half]);
                }
#pragma unroll
                for (int t = 0; t < kTpw; ++t) {
                    float(&g)[4] = group_acc[t][nt];
                    mma_bf16(g[0], g[1], g[2], g[3], a_top[t][0], a_bot[t][0], a_top[t][1],
                             a_bot[t][1], b.x, b.y);
                    mma_bf16(g[0], g[1], g[2], g[3], a_top[t][2], a_bot[t][2], a_top[t][3],
                             a_bot[t][3], b.z, b.w);
                }
            }
        }

#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int tile_row    = (first_tile + t) * 16 + gid;
            const float top_scale = __half2float(__ushort_as_half(stage.scales[tile_row][k_split]));
            const float bot_scale =
                __half2float(__ushort_as_half(stage.scales[tile_row + 8][k_split]));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[t][nt][0] = fmaf(group_acc[t][nt][0], top_scale, acc[t][nt][0]);
                acc[t][nt][1] = fmaf(group_acc[t][nt][1], top_scale, acc[t][nt][1]);
                acc[t][nt][2] = fmaf(group_acc[t][nt][2], bot_scale, acc[t][nt][2]);
                acc[t][nt][3] = fmaf(group_acc[t][nt][3], bot_scale, acc[t][nt][3]);
            }
        }
        // The next iteration's issue_slab writes the stage this one just read.
        __syncthreads();
    }
    cp_wait<0>();

    // Reduce each tile's K partials: odd K warps publish, even ones fold their neighbour, then K
    // warp 0 sums the even ones -- the same order as the Q4 kernel.
    __syncthreads();
    auto* partial   = shared.partial;
    const auto slot = [&](int w, int t, int nt) {
        return partial + (((w * kTpw + t) * kNt + nt) * 32 + lane) * 4;
    };
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const float(&a)[4] = acc[t][nt];
                store_vec(slot(warp, t, nt), make_float4(a[0], a[1], a[2], a[3]));
            }
        }
    }
    __syncthreads();
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float(&a)[4]         = acc[t][nt];
                const float4 partner = load_vec<float4>(slot(warp + 1, t, nt));
                a[0] += partner.x;
                a[1] += partner.y;
                a[2] += partner.z;
                a[3] += partner.w;
                if (k_split != 0) { store_vec(slot(warp, t, nt), make_float4(a[0], a[1], a[2], a[3])); }
            }
        }
    }
    if constexpr (kKWarps > 2) { __syncthreads(); }
    if (k_split == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float4 sum = make_float4(acc[t][nt][0], acc[t][nt][1], acc[t][nt][2], acc[t][nt][3]);
#pragma unroll
                for (int split = 2; split < kKWarps; split += 2) {
                    const float4 value = load_vec<float4>(slot(warp + split, t, nt));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                epilogue.store(cta_row0 + (first_tile + t) * 16 + gid, nt * 8 + 2 * lid, sum);
            }
        }
    }
}

// Launches q5_small_t_mma_kernel over Rows / SmallTLayout::kRowsPerCta CTAs.
template <int Rows, int K, int XCols, int Stages, class Epilogue, int KWarps = 8,
          int TilesPerWarp = 1>
void q5_small_t_mma_launch(cudaStream_t stream, const __nv_bfloat16* x, const std::uint8_t* codes,
                           const std::uint8_t* high_bits, const std::uint8_t* scales,
                           Epilogue epilogue, int columns) {
    using Schedule        = Q5SmallTSchedule<KWarps, TilesPerWarp>;
    constexpr int kBytes  = Q5SmallTStorage<KWarps, XCols, Stages, TilesPerWarp>::kBytes;
    constexpr auto kernel =
        q5_small_t_mma_kernel<Rows, K, XCols, Stages, Epilogue, KWarps, TilesPerWarp>;
    static_assert(kBytes <= 48 * 1024, "small-T MMA stages must fit static shared memory");
    kernel<<<Rows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
        x, codes, high_bits, scales, epilogue, columns);
}

} // namespace ninfer::ops::detail
