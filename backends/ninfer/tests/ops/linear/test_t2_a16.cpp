#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

// The ternary profile's problems across the decode, small-T and
// prefill widths.
int t2_a16_conformance() {
    int failures = 0;

    constexpr std::array kDecode{convenience(1), a16(1),   a16(2),   a16(3),   a16(4),
                                 a16(5),         a16(8),   a16(9),   a16(12),  a16(16),
                                 a16(17),        a16(18),  a16(32),  a16(33),  a16(64),
                                 a16(65),        a16(128), a16(129), a16(1024)};

    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {1024, 5120, 201U, Comparison::Full, true, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {7168, 5120, 203U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {34816, 5120, 205U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {5120, 6144, 207U, Comparison::Full, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {5120, 17408, 209U, Comparison::Full, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {131072, 5120, 211U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("T2_A16", ActivationCompute::A16, make_t2_g128_fp16_weight,
                          {248320, 5120, 213U, Comparison::SampledRows, false, kDecode});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = t2_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
