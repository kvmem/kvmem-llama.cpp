#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"
#if !defined(NINFER_SM8X_COMPAT)
#include "ops/linear/q8/q8_ksplit_grouped_mma.cuh"
#endif

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N2048K16384;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
// Four K warps on sm_8x (q8_ksplit_config.h): the sixteen- and eight-warp tiles from C16 and C48
// up do not fit its static shared memory.
template <int KWarps, int TileTokens, int MinBlocks, Access A, Cache AC, Cache WC, Stage S>
using Tile = Q8KSplitSm8xFourWarpSchedule<KWarps, TileTokens, MinBlocks, A, AC, WC, S>;
using C4   = Tile<16, 8, 1, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C8   = Tile<16, 8, 1, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C16  = Tile<16, 16, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C24  = Tile<16, 24, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C32  = Tile<16, 32, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40  = Tile<16, 40, 1, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C48  = Tile<8, 48, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C56  = Tile<8, 56, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C64  = Tile<8, 64, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;

#if defined(NINFER_SM8X_COMPAT)
// sm_86 ladder, measured 2026-09-17 with bench/ops/linear_schedule_bench.cu (`q8:2048x16384`),
// cold, median of 11. Three things separate it from the sm_120 ladder above, and all three recur
// on every Q8 shape this bench swept:
//
//   * `cg` activations lose to `ca`. That is upstream's 028eb61e applied to this card, and it is
//     backwards here: T=24 79.9 vs 91.1 us (+14%), T=32 87.0 vs 103.4 (+19%), T=48 129.0 vs 143.4
//     (+11%). The L1 that `cg` bypasses is where the staged activation slab wants to live.
//   * Eight K warps *do* fit 49,152 bytes up to a 32-column tile (codes 8 KiB + 1 KiB per column
//     tile), so the blanket four-warp fallback gives away the narrow rungs: T=8 52.2 vs 60.4
//     (+16%), T=12 56.3 vs 68.6 (+22%), T=16 58.4 vs 71.7 (+23%).
//   * The 49..128 composite -- 32-column capacity tiles plus a tail, this fork's own sm_8x stand-in
//     for the grouped medium kernels -- is beaten by a plain 32x64 MMA tile from 65 up and by the
//     capacity ladder below that: T=56 141.3 vs 193.5 (+37%), T=64 173.1 vs 204.8 (+18%),
//     T=96 263.2 vs 304.1 (+16%), T=112 276.5 vs 372.7 (+35%), T=128 258.0 vs 404.5 (+57%).
//     The composite is therefore gone; 65..128 simply joins the 129..384 band.
using S8   = Q8KSplitSchedule<8, 8, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S16  = Q8KSplitSchedule<8, 16, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S24  = Q8KSplitSchedule<4, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S32  = Q8KSplitSchedule<4, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S40  = Q8KSplitSchedule<4, 40, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S48  = Q8KSplitSchedule<4, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S56  = Q8KSplitSchedule<4, 56, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using S64  = Q8KSplitSchedule<4, 64, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
#else
template <int Capacity, int KWarps, int TokenGroups>
void launch_grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.padded_shape[1] != Geometry::kInputRows) {
        throw std::invalid_argument(
            "q8 grouped K-split: padded K differs from registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    q8_ksplit_grouped_mma_kernel<Geometry::kInputRows, Capacity, KWarps, TokenGroups, 1>
        <<<Geometry::kOutputRows / 16, KWarps * TokenGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

Q8Launch select_small(std::int32_t tokens) {
    if (tokens == 1) return launch_q8_gemv_n2048_k16384;
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40>;
    return launch_q8_ksplit<Geometry, 48, C48>;
}
#endif
} // namespace

Q8Launch select_q8_n2048_k16384(std::int32_t tokens) {
#if defined(NINFER_SM8X_COMPAT)
    if (tokens == 1) return launch_q8_gemv_n2048_k16384;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, S8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, S16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, S24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, S32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, S40>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, S48>;
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, S56>;
    if (tokens <= 64) return launch_q8_ksplit<Geometry, 64, S64>;
#else
    if (tokens <= 48) return select_small(tokens);
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_ksplit<Geometry, 64, C64>;
    if (tokens <= 80) return launch_grouped<80, 8, 2>;
    if (tokens <= 96) return launch_grouped<96, 4, 6>;
    if (tokens <= 128) return launch_grouped<128, 4, 8>;
#endif

    // Broad throughput regions; each selected MMA handles its own complete and partial tiles.
    // On sm_86 the 32x64 tile also carries 65..128 (see the ladder note above); T=144..192 was
    // re-measured there and keeps this bound (r32_c96 is 4% ahead at 144/160 and behind at 192,
    // inside this bench's spread).
    if (tokens <= 384) return launch_q8_mma_r32_c64;
    if (tokens <= 480) return launch_q8_mma_r32_c96;
    if (tokens <= 640) return launch_q8_mma_r32_c128;
    if (tokens <= 704) return launch_q8_mma_r48_c64;
    if (tokens <= 960) return launch_q8_mma_r64_c96;
    if (tokens <= 1344) return launch_q8_mma_r128_c64;
    if (tokens <= 1680) return launch_q8_mma_r128_c80;
    if (tokens <= 2016) return launch_q8_mma_r64_c96;
    if (tokens <= 2112) return launch_q8_mma_r96_c96;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
