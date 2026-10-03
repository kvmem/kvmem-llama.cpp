#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a8(std::int32_t t) {
    return {t, CallForm::Policy, ops::LinearPolicy::AllowA8Int};
}

// The ternary integer-activation routes through one policy: the small-T kernel (1..192 tokens, in
// launches of one, two and four column tiles) and above it the prefill GEMM, which pads T to its
// cheapest tile width inside.
int t2_a8_conformance() {
    int failures = 0;

    constexpr std::array kWidths{a8(1),   a8(2),   a8(3),    a8(4),   a8(5),   a8(7),   a8(8),
                                 a8(9),   a8(12),  a8(15),   a8(16),  a8(17),  a8(24),  a8(32),
                                 a8(33),  a8(40),  a8(63),   a8(64),  a8(65),  a8(95),  a8(100),
                                 a8(127), a8(128), a8(129),  a8(192), a8(193), a8(256), a8(300),
                                 a8(385), a8(600), a8(1007), a8(1024)};

    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {1024, 5120, 301U, Comparison::Full, true, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {4096, 5120, 303U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {7168, 5120, 305U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {34816, 5120, 307U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {5120, 6144, 309U, Comparison::Full, false, kWidths});
    failures += run_shape("T2_A8", ActivationCompute::A8, make_t2_g128_fp16_weight,
                          {5120, 17408, 311U, Comparison::Full, false, kWidths});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = t2_a8_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2_A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2_A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
