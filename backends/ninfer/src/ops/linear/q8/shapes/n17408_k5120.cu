#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N17408K5120;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
using C4 = Q8KSplitSchedule<8, 8, 3, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using C8 = Q8KSplitSchedule<4, 8, 3, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<8, 24, 3, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using W16 =
    Q8KSplitSchedule<8, 16, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
using S32 = Q8KSplitSchedule<4, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

} // namespace

// A split Q8 MTP gate or up projection. The routes follow n14336_k5120's, measured on sm_86 at
// the same K.
Q8Launch select_q8_n17408_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
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
