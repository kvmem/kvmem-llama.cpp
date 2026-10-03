#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C32Wide = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 8, 4, 2, Q4FragmentPipeline::Serial,
                                                Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C64 = Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

// The DFlash2 adapter's hidden-width output over a longer input (a Q4 drafter of a ternary
// target). The ladder is n5120_k6144.cu's: the adapter runs it at the draft block and the accepted
// columns (T <= 16) and at prefill chunks, where the same row count sees the same tiles.
Q4Launch select_q4_n5120_k4096(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<5120, 4096, 8>;
    if (tokens <= 24) return launch_q4_ksplit<5120, 4096, 24>;
    if (tokens <= 32) return launch_q4_mma<MmaR32C32Wide>;
    if (tokens <= 64) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 80) return launch_q4_mma_r64_c80;
    if (tokens <= 96) return launch_q4_mma_r64_c96;
    if (tokens <= 128) return launch_q4_mma<MmaR64C64>;
    if (tokens <= 160) return launch_q4_mma_r64_c80;
    if (tokens <= 192) return launch_q4_mma_r64_c96;
    if (tokens <= 224) return launch_q4_mma_r64_c112;
    if (tokens <= 256) return launch_q4_mma_r64_c128;
    if (tokens <= 320) return launch_q4_mma_r64_c80;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
