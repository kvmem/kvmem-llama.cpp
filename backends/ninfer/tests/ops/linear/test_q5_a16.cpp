#include "ops/linear/common/route_table.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

// Same public Op, replayed through a captured CUDA graph and verified twice against the FP64
// oracle (the second replay negates the activation), so the route is checked in the mode the
// engine actually runs it in. Used here for the routes this batch introduced or re-bounded.
constexpr Invocation graph(std::int32_t t) {
    return {t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true};
}

int q5_a16_conformance() {
    int failures = 0;

    // The lists cover both sides of every route boundary the shape tables declare: the
    // narrow-column tiles end at 48/160/512 here, at 11/112 for the k5120 row geometries and at
    // 15/112/256 for the 5120-row ones. The 1280/1281 pair is the restored wide route after the
    // r32c128 one-wave bound.
    constexpr std::array kN1024K5120{
        convenience(1),
        graph(1),
        a16(4),
        a16(5),
        a16(16),
        graph(17),
        a16(48),
        a16(49),
        a16(160),
        a16(161),
        graph(512),
        a16(513),
        a16(640),
        a16(128),
        a16(1280),
        graph(1281),
        a16(1282),
        // Route boundaries introduced by the 2026-09-17 sm_86 retune of this shape
        // table (src/ops/linear/...); each pair straddles one of them.
        a16(2),
        a16(3),
        a16(6),
        a16(7),
        a16(8),
        a16(32),
        a16(33),
        a16(80),
        a16(81),
        a16(448),
        a16(449),
        a16(704),
        a16(705),
        // Route boundaries of the unified-template table.
        a16(9),
        a16(15),
        a16(17),
        a16(31),
        a16(63),
        a16(64),
        a16(65),
        a16(79),
        a16(111),
        a16(112),
        a16(113),
        a16(159),
        a16(479),
        a16(480),
        a16(481),
        a16(512),
        a16(639),
        a16(641),
        a16(767),
        a16(768),
        a16(769),
        a16(1024),
        a16(1279),
        a16(1281),
        a16(1343),
        a16(1344),
        a16(1345),
        a16(2048),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {1024, 5120, 151U, Comparison::Full, true, kN1024K5120});

    constexpr std::array kN6144K5120{
        graph(1),
        a16(2),
        a16(3),
        a16(4),
        a16(5),
        a16(6),
        Invocation{4, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        Invocation{6, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        a16(7),
        a16(11),
        graph(12),
        a16(24),
        a16(25),
        a16(64),
        a16(65),
        a16(112),
        a16(113),
        a16(128),
        graph(1024),
        // Route boundaries introduced by the 2026-09-17 sm_86 retune of this shape
        // table (src/ops/linear/...); each pair straddles one of them.
        a16(8),
        a16(9),
        a16(160),
        a16(161),
        // Route boundaries of the unified-template table.
        a16(15),
        a16(16),
        a16(17),
        a16(23),
        a16(31),
        a16(32),
        a16(33),
        a16(159),
        a16(191),
        a16(192),
        a16(383),
        a16(384),
        a16(385),
        a16(447),
        a16(448),
        a16(449),
        a16(512),
        a16(671),
        a16(672),
        a16(673),
        a16(1024),
        a16(2048),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {6144, 5120, 157U, Comparison::SampledRows, false, kN6144K5120});

    constexpr std::array kN7168K5120{
        graph(1),
        a16(2),
        a16(3),
        a16(4),
        a16(5),
        a16(6),
        Invocation{4, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        Invocation{6, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        a16(7),
        a16(11),
        graph(12),
        a16(16),
        a16(17),
        a16(112),
        a16(113),
        a16(128),
        graph(1024),
        // Route boundaries introduced by the 2026-09-17 sm_86 retune of this shape
        // table (src/ops/linear/...); each pair straddles one of them.
        a16(8),
        a16(9),
        // Route boundaries of the unified-template table.
        a16(15),
        a16(23),
        a16(24),
        a16(25),
        a16(31),
        a16(32),
        a16(33),
        a16(95),
        a16(96),
        a16(97),
        a16(127),
        a16(129),
        a16(191),
        a16(192),
        a16(383),
        a16(384),
        a16(385),
        a16(512),
        a16(575),
        a16(576),
        a16(577),
        a16(1024),
        a16(2048),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {7168, 5120, 163U, Comparison::SampledRows, false, kN7168K5120});

    constexpr std::array kN5120K6144{
        graph(1),
        a16(2),
        a16(3),
        a16(4),
        a16(5),
        a16(6),
        Invocation{4, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        Invocation{6, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        a16(7),
        a16(15),
        graph(16),
        a16(24),
        a16(25),
        a16(112),
        a16(113),
        graph(128),
        a16(256),
        a16(257),
        a16(1024),
        // Route boundaries introduced by the 2026-09-17 sm_86 retune of this shape
        // table (src/ops/linear/...); each pair straddles one of them.
        a16(8),
        a16(9),
        a16(17),
        a16(96),
        a16(97),
        a16(176),
        a16(177),
        // Route boundaries of the unified-template table.
        a16(16),
        a16(31),
        a16(32),
        a16(33),
        a16(47),
        a16(48),
        a16(49),
        a16(63),
        a16(64),
        a16(65),
        a16(127),
        a16(128),
        a16(129),
        a16(159),
        a16(160),
        a16(161),
        a16(255),
        a16(512),
        a16(2048),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {5120, 6144, 167U, Comparison::SampledRows, false, kN5120K6144});

    constexpr std::array kN5120K17408{
        graph(1),
        a16(2),
        a16(3),
        a16(4),
        a16(5),
        a16(6),
        Invocation{4, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        Invocation{6, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true},
        a16(7),
        a16(15),
        graph(16),
        a16(24),
        a16(25),
        a16(112),
        a16(113),
        graph(128),
        a16(256),
        a16(257),
        a16(1024),
        // Route boundaries introduced by the 2026-09-17 sm_86 retune of this shape
        // table (src/ops/linear/...); each pair straddles one of them.
        a16(8),
        a16(9),
        a16(17),
        a16(96),
        a16(97),
        a16(129),
        // Route boundaries of the unified-template table.
        a16(16),
        a16(23),
        a16(31),
        a16(32),
        a16(33),
        a16(47),
        a16(48),
        a16(49),
        a16(127),
        a16(128),
        a16(159),
        a16(160),
        a16(161),
        a16(255),
        a16(512),
        a16(2048),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {5120, 17408, 173U, Comparison::SampledRows, false, kN5120K17408});

    constexpr std::array kN1152K1152{
        a16(4),
        a16(76),
        a16(80),
        a16(636),
        a16(640),
        a16(700),
        a16(704),
        a16(708),
        a16(828),
        a16(832),
        a16(836),
        a16(896),
        a16(900),
        a16(960),
        a16(964),
        a16(1024),
        a16(1028),
        a16(1088),
        a16(1092),
        a16(131072),
        // Route boundaries of the unified-template table.
        a16(8),
        a16(60),
        a16(64),
        a16(68),
        a16(124),
        a16(128),
        a16(132),
        a16(512),
        a16(572),
        a16(576),
        a16(580),
        a16(1148),
        a16(1152),
        a16(2048),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {1152, 1152, 179U, Comparison::SampledRows, false, kN1152K1152});

    constexpr std::array kN1152K4304{
        a16(4),
        a16(120),
        a16(124),
        a16(1148),
        a16(1152),
        a16(131072),
        // Route boundaries of the unified-template table.
        a16(8),
        a16(28),
        a16(32),
        a16(36),
        a16(44),
        a16(48),
        a16(52),
        a16(60),
        a16(64),
        a16(68),
        a16(92),
        a16(96),
        a16(100),
        a16(284),
        a16(288),
        a16(292),
        a16(444),
        a16(448),
        a16(452),
        a16(572),
        a16(576),
        a16(580),
        a16(668),
        a16(672),
        a16(676),
        a16(1024),
        a16(1156),
        a16(1532),
        a16(1536),
        a16(1540),
        a16(2048),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {1152, 4304, 181U, Comparison::SampledRows, false, kN1152K4304});

    // Full-output oracle over the unified-template mechanisms, with CUDA Graph replays.
    constexpr std::array kN1024K5120Full{convenience(1), graph(1),   graph(2),  graph(7),
                                         graph(13),      graph(23),  graph(40), graph(75),
                                         graph(100),     graph(128), graph(176)};
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {1024, 5120, 151U, Comparison::Full, true, kN1024K5120Full});

    constexpr std::array kN6144K5120Full{graph(1),   graph(4),   graph(7),  graph(13),
                                         graph(19),  graph(27),  graph(48), graph(96),
                                         graph(161), graph(193), graph(480)};
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {6144, 5120, 157U, Comparison::Full, true, kN6144K5120Full});

    constexpr std::array kN7168K5120Full{graph(1),   graph(3),   graph(7),  graph(13),
                                         graph(19),  graph(27),  graph(48), graph(113),
                                         graph(144), graph(193), graph(480)};
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {7168, 5120, 163U, Comparison::Full, true, kN7168K5120Full});

    constexpr std::array kN5120K6144Full{graph(1),  graph(4),  graph(7),  graph(13),  graph(24),
                                         graph(40), graph(56), graph(80), graph(144), graph(193)};
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {5120, 6144, 169U, Comparison::Full, true, kN5120K6144Full});

    constexpr std::array kN5120K17408Full{graph(1),  graph(4),  graph(7),  graph(13),  graph(20),
                                          graph(28), graph(40), graph(80), graph(144), graph(193)};
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {5120, 17408, 175U, Comparison::Full, true, kN5120K17408Full});

    constexpr std::array kN1152K1152Full{graph(4),   graph(12),  graph(40),  graph(100),
                                         graph(160), graph(640), graph(1156)};
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {1152, 1152, 181U, Comparison::Full, true, kN1152K1152Full});

    constexpr std::array kN1152K4304Full{graph(4),   graph(12),  graph(40),  graph(56),
                                         graph(80),  graph(116), graph(320), graph(512),
                                         graph(704), graph(1280)};
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5_g64_fp16_weight,
                          {1152, 4304, 187U, Comparison::Full, true, kN1152K4304Full});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        using ninfer::ops::detail::LinearRouteTable;
        int failures = 0;
        for (const LinearRouteTable table : {LinearRouteTable::Legacy, LinearRouteTable::Unified}) {
            ninfer::ops::detail::force_linear_route_table(table);
            const int table_failures = q5_a16_conformance();
            std::cout << (table_failures == 0 ? "OK" : "FAIL") << " Q5_A16 Linear ("
                      << (table == LinearRouteTable::Legacy ? "legacy" : "unified") << " routes)\n";
            failures += table_failures;
        }
        ninfer::ops::detail::force_linear_route_table(std::nullopt);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
