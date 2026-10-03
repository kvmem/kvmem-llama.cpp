#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N6144K5120;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
using C4 = Q8KSplitSchedule<16, 8, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8 = Q8KSplitSchedule<8, 8, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
// Bound registers to allow three resident 256-thread CTAs per SM.
using C16 = Q8KSplitSchedule<8, 16, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<4, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C32 = Q8KSplitSchedule<4, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8KSplitSchedule<4, 40, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C48 = Q8KSplitSchedule<4, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C56 = Q8KSplitSchedule<4, 56, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
} // namespace

Q8Launch select_q8_n6144_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    // sm_86, measured 2026-09-17 (`q8:6144x5120`, cold, median of 11; us, new vs shipped):
    //   T=40  r32_c64 111.6 vs 133.1 (+19%)  T=48 112.6 vs 130.0 (+16%)
    //   T=56  r32_c64 128.0 vs 236.5 (+85%)  -- the 56-column K-split rung is the worst entry here
    //   T=80  r32_c96 154.6 vs 207.9 (+34%)  T=96 158.7 vs 212.0 (+34%)
    //   T=128 r48_c64 214.0 vs 254.0 (+19%)
    //   T=192 r128_c64 229.4 vs 287.7 (+25%)
    // T<=32, 57..64 and 97..112 keep upstream's routes; each measured within 1% of the best
    // candidate. T>=256 is unchanged.
    if (tokens <= 64) return launch_q8_mma_r32_c64;
    if (tokens <= 96) return launch_q8_mma_r32_c96;
    if (tokens <= 112) return launch_q8_mma_r32_c64;
    if (tokens <= 128) return launch_q8_mma_r48_c64;
    if (tokens <= 192) return launch_q8_mma_r128_c64;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
