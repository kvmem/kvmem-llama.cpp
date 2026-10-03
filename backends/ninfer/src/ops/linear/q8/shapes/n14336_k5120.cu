#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N14336K5120;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
using C4 = Q8KSplitSchedule<8, 8, 3, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8 = Q8KSplitSchedule<4, 8, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
// Bound registers to allow six resident 128-thread CTAs per SM.
using C16 = Q8KSplitSchedule<4, 16, 6, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C24 = Q8KSplitSchedule<8, 24, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32 = Q8KSplitSchedule<4, 32, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C40 = Q8KSplitSchedule<4, 40, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C48 = Q8KSplitSchedule<4, 48, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C56 = Q8KSplitSchedule<4, 56, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using W16 =
    Q8KSplitSchedule<8, 16, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S32 = Q8KSplitSchedule<4, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

} // namespace

Q8Launch select_q8_n14336_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    // sm_86, measured 2026-09-17 (`q8:14336x5120`, cold, median of 11; us, new vs shipped):
    //   T=12  W16 104.4 vs 130.0 (+25%)      T=16 106.5 vs 134.1 (+26%)
    //   T=32  S32 173.1 vs 241.7 (+40%)
    //   T=40  r32_c64 236.5 vs 322.6 (+36%)  T=48 239.6 vs 342.0 (+43%)  T=56 248.8 vs 316.4 (+27%)
    //   T=64  r128_c64 235.5 vs 298.0 (+27%)
    //   T=80  r96_c96 265.2 vs 360.4 (+36%)  T=96 265.2 vs 359.4 (+36%)
    //   T=160 r128_c80 463.9 vs 666.6 (+44%) T=192 r96_c96 515.1 vs 671.7 (+30%)
    // T=17..24, 97..128 and T>=256 keep upstream's routes and measure best here.
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, W16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, S32>;
    if (tokens <= 56) return launch_q8_mma_r32_c64;
    if (tokens <= 64) return launch_q8_mma_r128_c64;
    if (tokens <= 96) return launch_q8_mma_r96_c96;
    if (tokens <= 128) return launch_q8_mma_r64_c128;
    if (tokens <= 160) return launch_q8_mma_r128_c80;
    if (tokens <= 192) return launch_q8_mma_r96_c96;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
