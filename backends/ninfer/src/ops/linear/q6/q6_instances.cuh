#pragma once

#include "ops/linear/q6/q6_schedule.cuh"

namespace ninfer::ops::detail::q6_instances {

using SimtR8T4 = Q6A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>;

// Legacy-table SIMT routes stream the weight planes through L2 only.
using SimtR8T4Cg = Q6A16SimtSchedule<8, 4, 1, 16, 2, Cache::cg, 1>;
using SimtR8T5Cg = Q6A16SimtSchedule<8, 5, 1, 16, 2, Cache::cg, 1>;
using SimtR8T6Cg = Q6A16SimtSchedule<8, 6, 1, 16, 2, Cache::cg, 1>;
using SimtR8T7Cg = Q6A16SimtSchedule<8, 7, 1, 16, 2, Cache::cg, 1>;

using GemvR4W2G16      = Q6A16GemvSchedule<4, 2, 16, 2, Cache::ca, 1>;
using SlicedR16T8W4S2  = Q6A16SlicedKMmaSchedule<16, 8, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T16W4S2 = Q6A16SlicedKMmaSchedule<32, 16, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T32W4S1 = Q6A16SlicedKMmaSchedule<32, 32, 4, 1, Cache::cg, Cache::ca, 2>;
using SlicedR32T64W2S1 = Q6A16SlicedKMmaSchedule<32, 64, 2, 1, Cache::cg, Cache::ca, 2>;
using SlicedR16T24W4S2 = Q6A16SlicedKMmaSchedule<16, 24, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR16T32W4S2 = Q6A16SlicedKMmaSchedule<16, 32, 4, 2, Cache::cg, Cache::ca, 2>;
using SlicedR32T32W4S2 = Q6A16SlicedKMmaSchedule<32, 32, 4, 2, Cache::cg, Cache::ca, 2>;

// Legacy-table tiles: the 16..32 column band and the vision mid band before sliced-K.
using MmaR64T16K128 = Q6A16MmaSchedule<64, 16, 128, 16, 8, 2, 3, Q6MmaFragmentPipeline::Serial,
                                       Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T24K128 = Q6A16MmaSchedule<64, 24, 128, 16, 8, 2, 2, Q6MmaFragmentPipeline::Serial,
                                       Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T32K128 = Q6A16MmaSchedule<64, 32, 128, 16, 8, 2, 2, Q6MmaFragmentPipeline::Serial,
                                       Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T64PingPong =
    Q6A16MmaSchedule<64, 64, 64, 32, 32, 2, 3, Q6MmaFragmentPipeline::PingPong, Cache::ca,
                     Cache::ca, Q6ScaleLoad::Scalar16>;
using MmaR32T16K256     = Q6A16MmaSchedule<32, 16, 256, 16, 8, 2, 3, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR32T24K256     = Q6A16MmaSchedule<32, 24, 256, 16, 8, 2, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR32T32K256     = Q6A16MmaSchedule<32, 32, 256, 16, 8, 2, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T40K128     = Q6A16MmaSchedule<64, 40, 128, 32, 8, 2, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T48K128     = Q6A16MmaSchedule<64, 48, 128, 16, 16, 2, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T56K128     = Q6A16MmaSchedule<64, 56, 128, 32, 8, 2, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T64K128     = Q6A16MmaSchedule<64, 64, 128, 16, 16, 2, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T72K128     = Q6A16MmaSchedule<64, 72, 128, 32, 24, 2, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T80         = Q6A16MmaSchedule<64, 80, 128, 16, 40, 1, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T96         = Q6A16MmaSchedule<64, 96, 128, 16, 48, 1, 2, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32, 1>;
using MmaR64T112Partial = Q6A16MmaSchedule<64, 112, 64, 64, 16, 2, 1, Q6MmaFragmentPipeline::Serial,
                                           Cache::cg, Cache::cg, Q6ScaleLoad::Pair32>;
using MmaR64T112 = Q6A16MmaSchedule<64, 112, 64, 16, 112, 2, 1, Q6MmaFragmentPipeline::Serial,
                                    Cache::cg, Cache::cg, Q6ScaleLoad::Pair32>;
using MmaR64T128 = Q6A16MmaSchedule<64, 128, 64, 64, 32, 2, 1, Q6MmaFragmentPipeline::Serial,
                                    Cache::cg, Cache::cg, Q6ScaleLoad::Pair32>;

} // namespace ninfer::ops::detail::q6_instances
