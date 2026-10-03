#pragma once

// T2G128 RowSplit x int8 small-T tensor-core GEMM (T = 1..192), the integer counterpart of
// t2_small_t_v2.cuh. On GeForce Ampere m16n8k16 with an FP32 accumulator runs at a quarter of the
// s8 m16n8k32 rate, and an n8 tile costs the same tensor time at T = 1 as at T = 8: the bf16 kernel
// spends about three quarters of the weight stream's time in the tensor pipe at every T <= 8 and
// holds the text layers near 550-630 GB/s. Here the same tile takes a quarter of that time and the
// code unpack is four prmt per sixteen weights instead of sixteen.
//
// Layout follows v2: eight warps per CTA, KWarps of them split every K slab (64 k per code word a
// lane takes), the CTA covers 8 / KWarps sets of TilesPerWarp sixteen-row tiles; codes and scales
// are staged by cp.async; activations are read straight from global memory through L1. They arrive
// quantised to s8 with one binary16 scale per (token, 64-k group), as in the integer prefill route:
// codes [T, K] token-major, scales [K / 64, 192] group-major, so a lane's two output columns share
// one 32-bit scale load. One launch covers up to 32 columns from column col0; wider T takes
// several.
//
// Lane lid owns k = 16 * lid + j (j = 0..15) of each 64-k group of its warp's slice. The m16n8k32
// k slots may be any permutation shared by A and B, so the two k32 steps take j {0, 2, 4, 6 |
// 1, 3, 5, 7} and {8, 10, 12, 14 | 9, 11, 13, 15}: on the A side the even and odd code masks of the
// lane's word looked up in a {0, +1, -2, -1} byte table, on the B side byte permutes of its sixteen
// activation bytes. The int32 sums of each group are rescaled once per row and column.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kT2I8MaxColumns    = 192;
inline constexpr int kT2I8LaunchColumns = 32;
inline constexpr int kT2I8ActivationK   = 64;

// WordsPerLane code words of a row per lane and slab: a warp's slice of a slab is 64 * WordsPerLane
// k, taken as that many 64-k groups one after another.
template <int KWarps_, int TilesPerWarp_, int ColumnTiles_, int Stages_, int MinBlocks_,
          int WordsPerLane_ = 1>
struct T2SmallTI8Schedule {
    static constexpr int kWarps               = 8;
    static constexpr int kThreads             = kWarps * 32;
    static constexpr int kKWarps              = KWarps_;
    static constexpr int kTilesPerWarp        = TilesPerWarp_;
    static constexpr int kRowTiles            = kWarps / kKWarps * kTilesPerWarp;
    static constexpr int kRows                = 16 * kRowTiles;
    static constexpr int kColumnTiles         = ColumnTiles_;
    static constexpr int kColumns             = 8 * kColumnTiles;
    static constexpr int kWords               = WordsPerLane_;
    static constexpr int kWarpK               = 64 * kWords;
    static constexpr int kSlabK               = kWarpK * kKWarps;
    static constexpr int kGroupsPerSlab       = kSlabK / T2RowSplitStorage::kGroupK;
    static constexpr int kCodeBytesPerRowSlab = kSlabK / T2RowSplitStorage::kCodesPerByte;
    // A 16-byte pad staggers the rows' code words across the banks for the k_split x lid reads.
    static constexpr int kCodeRowStride = kCodeBytesPerRowSlab + 16;
    static constexpr int kScaleBytesPerRowSlab =
        kGroupsPerSlab * T2RowSplitStorage::kScaleBytesPerGroup;
    static constexpr int kStages    = Stages_;
    static constexpr int kMinBlocks = MinBlocks_;

    static_assert(kKWarps == 4 || kKWarps == 8, "T2 small-T i8 splits K over four or eight warps");
    static_assert(kTilesPerWarp == 1 || kTilesPerWarp == 2 || kTilesPerWarp == 4,
                  "T2 small-T i8 gives a warp one, two or four tiles");
    static_assert(kColumnTiles == 1 || kColumnTiles == 2 || kColumnTiles == 4,
                  "T2 small-T i8 covers one, two or four n8 tiles");
    static_assert(kStages >= 1 && kStages <= 4, "T2 small-T i8 pipeline depth must fit cp_wait");
    static_assert(kMinBlocks >= 1, "T2 small-T i8 launch-bounds occupancy must be positive");
    static_assert(kWords == 1 || kWords == 2 || kWords == 4,
                  "T2 small-T i8 takes one, two or four code words per lane");
    static_assert(kScaleBytesPerRowSlab == 4 || kScaleBytesPerRowSlab == 8 ||
                      kScaleBytesPerRowSlab == 16,
                  "T2 small-T i8 stages a row's slab scales with one cp.async");

    struct Stage {
        __align__(16) std::uint8_t codes[kRows][kCodeRowStride];
        __align__(16) std::uint8_t scales[kRows][kScaleBytesPerRowSlab];
    };

    static constexpr int kPartialFloats = kWarps * kTilesPerWarp * kColumnTiles * 32 * 4;

    union Shared {
        Stage stages[kStages];
        float partial[kPartialFloats];
    };

    static_assert(sizeof(Shared) <= 48 * 1024,
                  "T2 small-T i8 stages exceed the static shared budget");
};

// The four codes in the low selector nibbles -> four s8 lanes: 00 -> 0, 01 -> +1, 11 -> -1 (10 is
// outside the artifact language and decodes as its two's-complement -2).
constexpr std::uint32_t kT2I8Lut = 0xFFFE0100u;

__device__ __forceinline__ unsigned t2_i8_codes(std::uint32_t selector) {
    return __byte_perm(kT2I8Lut, 0u, selector);
}

// One launch over up to two weights of the same input width that read the same activations: the
// first `first_blocks` CTAs take the first weight and its epilogue, the rest the second.
template <class Epilogue>
struct T2I8Parents {
    const std::uint8_t* codes[2];
    const std::uint8_t* scales[2];
    Epilogue epilogue[2];
    std::int32_t first_blocks;
};

template <class Schedule, class Epilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) void t2_small_t_i8_kernel(
    const std::int8_t* __restrict__ x_codes, const __half* __restrict__ x_scales,
    const __grid_constant__ T2I8Parents<Epilogue> parents, std::int32_t k, std::int32_t col0,
    std::int32_t cols) {
    constexpr int kTpw    = Schedule::kTilesPerWarp;
    constexpr int kNt     = Schedule::kColumnTiles;
    constexpr int kKW     = Schedule::kKWarps;
    constexpr int kStages = Schedule::kStages;
    using Stage           = typename Schedule::Stage;

    __shared__ __align__(16) typename Schedule::Shared shared;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int k_split    = warp % kKW;
    const int first_tile = warp / kKW * kTpw;
    const bool second    = static_cast<int>(blockIdx.x) >= parents.first_blocks;
    const int row0 =
        (static_cast<int>(blockIdx.x) - (second ? parents.first_blocks : 0)) * Schedule::kRows;
    const std::uint8_t* __restrict__ codes  = parents.codes[second ? 1 : 0];
    const std::uint8_t* __restrict__ scales = parents.scales[second ? 1 : 0];
    const Epilogue& epilogue                = parents.epilogue[second ? 1 : 0];

    const int groups_per_row = k / T2RowSplitStorage::kGroupK;
    const int slabs          = k / Schedule::kSlabK;
    const std::int64_t code_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * T2RowSplitStorage::kCodeBytesPerGroup;
    const std::int64_t scale_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * T2RowSplitStorage::kScaleBytesPerGroup;
    // Word w of the lane: code byte k_split * 16W + 16w + 4 lid of the row's slab, k from
    // k_split * 64W + 64w + 16 lid.
    constexpr int kWords = Schedule::kWords;
    const int code_byte  = k_split * 16 * kWords + 4 * lid;
    const int slice_k    = k_split * Schedule::kWarpK + 16 * lid;

    const auto issue = [&](int stage_index, int slab) {
        Stage& stage                   = shared.stages[stage_index];
        constexpr int kChunksPerRow    = Schedule::kCodeBytesPerRowSlab / 16;
        constexpr int kChunks          = Schedule::kRows * kChunksPerRow;
        constexpr int kChunksPerThread = (kChunks + Schedule::kThreads - 1) / Schedule::kThreads;
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int item = tid + i * Schedule::kThreads;
            if (item < kChunks) {
                const int r = item / kChunksPerRow;
                const int c = item - r * kChunksPerRow;
                cp_async<16, Cache::cg>(
                    &stage.codes[r][c * 16],
                    codes + static_cast<std::int64_t>(row0 + r) * code_row_bytes +
                        static_cast<std::int64_t>(slab) * Schedule::kCodeBytesPerRowSlab + c * 16);
            }
        }
        if (tid < Schedule::kRows) {
            cp_async<Schedule::kScaleBytesPerRowSlab>(
                &stage.scales[tid][0],
                scales + static_cast<std::int64_t>(row0 + tid) * scale_row_bytes +
                    static_cast<std::int64_t>(slab) * Schedule::kScaleBytesPerRowSlab);
        }
    };

    float acc[kTpw][kNt][4] = {};

#pragma unroll
    for (int prefetch = 0; prefetch < kStages - 1; ++prefetch) {
        if (prefetch < slabs) { issue(prefetch, prefetch); }
        cp_commit(); // empty commits keep cp_wait<kStages - 1> exact through the tail
    }

#pragma unroll 1
    for (int slab = 0; slab < slabs; ++slab) {
        {
            const int next = slab + kStages - 1;
            if (next < slabs) { issue(next % kStages, next); }
            cp_commit();
        }

        // B: the lane's sixteen activation bytes of each word and live column and the scales of the
        // two output columns its accumulators hold. Both are independent of the staged codes, so
        // they are in flight while the ring waits.
        const int group64 = (slab * kKW + k_split) * kWords;
        uint4 xv[kWords][kNt];
        __half2 xs[kWords][kNt];
#pragma unroll
        for (int w = 0; w < kWords; ++w) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int col = nt * 8 + gid;
                xv[w][nt]     = make_uint4(0u, 0u, 0u, 0u);
                if (col < cols) {
                    xv[w][nt] = load_ldg<uint4>(
                        x_codes + static_cast<std::int64_t>(col0 + col) * k +
                        static_cast<std::int64_t>(slab) * Schedule::kSlabK + slice_k + 64 * w);
                }
                xs[w][nt] = load_ldg<__half2>(x_scales + (group64 + w) * kT2I8MaxColumns + col0 +
                                              nt * 8 + 2 * lid);
            }
        }

        cp_wait<kStages - 1>();
        __syncthreads();
        const Stage& stage = shared.stages[slab % kStages];

#pragma unroll
        for (int w = 0; w < kWords; ++w) {
            int sums[kTpw][kNt][4] = {};
#pragma unroll
            for (int t = 0; t < kTpw; ++t) {
                const int tile_row      = (first_tile + t) * 16 + gid;
                const std::uint32_t top = *reinterpret_cast<const std::uint32_t*>(
                    &stage.codes[tile_row][code_byte + 16 * w]);
                const std::uint32_t bottom = *reinterpret_cast<const std::uint32_t*>(
                    &stage.codes[tile_row + 8][code_byte + 16 * w]);
                const std::uint32_t top_even    = top & 0x33333333u;
                const std::uint32_t top_odd     = (top >> 2) & 0x33333333u;
                const std::uint32_t bottom_even = bottom & 0x33333333u;
                const std::uint32_t bottom_odd  = (bottom >> 2) & 0x33333333u;
#pragma unroll
                for (int step = 0; step < 2; ++step) {
                    const int shift   = 16 * step;
                    const unsigned a0 = t2_i8_codes(top_even >> shift);
                    const unsigned a1 = t2_i8_codes(bottom_even >> shift);
                    const unsigned a2 = t2_i8_codes(top_odd >> shift);
                    const unsigned a3 = t2_i8_codes(bottom_odd >> shift);
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt) {
                        const unsigned lo = step == 0 ? xv[w][nt].x : xv[w][nt].z;
                        const unsigned hi = step == 0 ? xv[w][nt].y : xv[w][nt].w;
                        int (&s)[4]       = sums[t][nt];
                        mma_s8(s[0], s[1], s[2], s[3], a0, a1, a2, a3, __byte_perm(lo, hi, 0x6420),
                               __byte_perm(lo, hi, 0x7531));
                    }
                }
            }

            const int group_in_slab = (k_split * kWords + w) >> 1;
#pragma unroll
            for (int t = 0; t < kTpw; ++t) {
                const int tile_row = (first_tile + t) * 16 + gid;
                const float top_scale =
                    __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                        &stage.scales[tile_row][2 * group_in_slab])));
                const float bottom_scale =
                    __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                        &stage.scales[tile_row + 8][2 * group_in_slab])));
#pragma unroll
                for (int nt = 0; nt < kNt; ++nt) {
                    const float2 column = __half22float2(xs[w][nt]);
                    const int (&s)[4]   = sums[t][nt];
                    float (&c)[4]       = acc[t][nt];
                    c[0] = fmaf(static_cast<float>(s[0]), top_scale * column.x, c[0]);
                    c[1] = fmaf(static_cast<float>(s[1]), top_scale * column.y, c[1]);
                    c[2] = fmaf(static_cast<float>(s[2]), bottom_scale * column.x, c[2]);
                    c[3] = fmaf(static_cast<float>(s[3]), bottom_scale * column.y, c[3]);
                }
            }
        }
        // The next iteration's issue writes the stage this one just read.
        __syncthreads();
    }
    cp_wait<0>();
    __syncthreads();

    // Reduce the K partials of each tile in a fixed order: odd K warps publish, even ones fold
    // their neighbour, then K warp 0 sums the even ones.
    float* partial  = shared.partial;
    const auto slot = [&](int w, int t, int nt) {
        return partial + (((w * kTpw + t) * kNt + nt) * 32 + lane) * 4;
    };
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const float (&c)[4] = acc[t][nt];
                store_vec(slot(warp, t, nt), make_float4(c[0], c[1], c[2], c[3]));
            }
        }
    }
    __syncthreads();
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float (&c)[4]        = acc[t][nt];
                const float4 partner = load_vec<float4>(slot(warp + 1, t, nt));
                c[0] += partner.x;
                c[1] += partner.y;
                c[2] += partner.z;
                c[3] += partner.w;
                if (k_split != 0) {
                    store_vec(slot(warp, t, nt), make_float4(c[0], c[1], c[2], c[3]));
                }
            }
        }
    }
    if constexpr (kKW > 2) { __syncthreads(); }

    if (k_split == 0) {
#pragma unroll
        for (int t = 0; t < kTpw; ++t) {
            const int row = row0 + (first_tile + t) * 16 + gid;
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float4 sum =
                    make_float4(acc[t][nt][0], acc[t][nt][1], acc[t][nt][2], acc[t][nt][3]);
#pragma unroll
                for (int split = 2; split < kKW; split += 2) {
                    const float4 value = load_vec<float4>(slot(warp + split, t, nt));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int col = nt * 8 + 2 * lid;
                if (col < cols) {
                    epilogue(row, col0 + col, sum.x);
                    epilogue(row + 8, col0 + col, sum.z);
                }
                if (col + 1 < cols) {
                    epilogue(row, col0 + col + 1, sum.y);
                    epilogue(row + 8, col0 + col + 1, sum.w);
                }
            }
        }
    }
}

// One warp per (token, 64-k group) of a BF16 [K, T] activation: s8 codes at [T, K] and the binary16
// scale absmax / 127 at [K / 64, 192]. Warps of the padding columns up to the next multiple of 8
// write a zero scale, so the GEMM's masked accumulators (exact zeros) never meet an uninitialised
// one.
__global__ void t2_small_t_i8_quantize_kernel(const __nv_bfloat16* __restrict__ x, std::int32_t k,
                                              std::int32_t tokens, std::int8_t* __restrict__ codes,
                                              __half* __restrict__ scales) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int group  = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) +
                       (static_cast<int>(threadIdx.x) >> 5);
    const int token  = static_cast<int>(blockIdx.y);
    const int groups = k / kT2I8ActivationK;
    if (group >= groups) { return; }
    __half* scale = scales + group * kT2I8MaxColumns + token;
    if (token >= tokens) {
        if (lane == 0) { *scale = __float2half(0.0F); }
        return;
    }
    const std::int64_t offset = static_cast<std::int64_t>(token) * k +
                                static_cast<std::int64_t>(group) * kT2I8ActivationK + 2 * lane;
    const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + offset));
    float amax         = fmaxf(fabsf(value.x), fabsf(value.y));
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
    amax            = fmaxf(amax, 1.0e-20F);
    const float inv = 127.0F / amax;
    const int low   = max(-127, min(127, __float2int_rn(value.x * inv)));
    const int high  = max(-127, min(127, __float2int_rn(value.y * inv)));
    *reinterpret_cast<std::uint16_t*>(codes + offset) =
        static_cast<std::uint16_t>((low & 0xFF) | ((high & 0xFF) << 8));
    if (lane == 0) { *scale = __float2half(amax / 127.0F); }
}

} // namespace ninfer::ops::detail
