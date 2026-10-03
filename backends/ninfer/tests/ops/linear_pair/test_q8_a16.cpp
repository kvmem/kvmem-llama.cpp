#include "ops/linear_pair/linear_pair_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_pair::ShapeCase;

int q8_a16_conformance() {
    int failures = 0;

    // These track kK5120Routes in q8_pair_plan.cpp, which was re-measured on sm_86; the previous
    // {86, 961} were upstream's sm_120 boundaries.
    constexpr std::array<std::int32_t, 2> kK5120RouteStarts{49, 449};
    // 85 is kept as an explicit interior although it is no longer a boundary. It is the width that
    // exposed how tightly the gross-error limit was calibrated (see kLinearPairA16Tolerance in
    // linear_pair_test_common.cpp), and dropping it once it stopped being a route start would have
    // quietly retired the only case that covers that edge.
    constexpr std::array<std::int32_t, 4> kK5120RouteInteriors{24, 85, 512, 1024};
    failures += ninfer::test::linear_pair::run_q8_a16_shape(
        "Q8_A16 LinearPair", ShapeCase{5120, 431U, kK5120RouteStarts, kK5120RouteInteriors});

    // These track kK2048Routes. The eight DualSplitKMedium starts that used to sit between 81 and
    // 193 are gone: on sm_86 that whole family is one kernel, and 65..384 is now a single
    // ConcatMmaR32C64 band. The widths themselves stay below as interiors, so the coverage that
    // mattered -- every one of those column counts -- is unchanged.
    constexpr std::array<std::int32_t, 28> kK2048RouteStarts{
        2,    33,   49,   65,   385,  481,  641,  642,  673,  681,
        785,  897,  961,  977,  1281, 1317, 1345, 1346, 1441, 1467,
        1681, 1709, 1921, 1923, 2017, 2019, 2209, 2271,
    };
    constexpr std::array<std::int32_t, 53> kK2048RouteInteriors{
        1,    16,   40,   56,   72,   80,   81,   84,   88,   89,   92,   96,   97,   100,
        104,  105,  108,  112,  113,  120,  128,  129,  144,  160,  161,  176,  192,  193,
        288,  432,  560,  641,  656,  676,  736,  840,  928,  968,  1120, 1296, 1332, 1345,
        1392, 1456, 1576, 1696, 1816, 1921, 1968, 2017, 2112, 2240, 4096,
    };
    failures += ninfer::test::linear_pair::run_q8_a16_shape(
        "Q8_A16 LinearPair", ShapeCase{2048, 433U, kK2048RouteStarts, kK2048RouteInteriors});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_pair::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 LinearPair\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 LinearPair: " << error.what() << '\n';
        return 1;
    }
}
