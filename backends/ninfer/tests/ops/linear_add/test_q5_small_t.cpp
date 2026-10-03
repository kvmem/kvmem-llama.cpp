// The Q5 small-T MMA residual kernel against an fp64 oracle, at T=1..32 on both registered K, with
// a nonzero residual and dense two-signed activations. The oracle is computed once for 32 columns
// from the decoded weights; columns are independent, so each T checks a prefix of it.
//
// Tolerance: the kernel reads exact bf16 codes and activations, accumulates in fp32 and rounds the
// residual sum to bf16 once, so it must land within one bf16 rounding step (2^-8 relative, plus a
// small absolute floor for outputs near zero after cancellation). Each case also runs three times
// and must give identical bytes every time.
//
// Comparing against the routed MMA tiles instead would not work at this tolerance: their
// CtaCollectiveResidual epilogue rounds the product to bf16 before adding the residual
// (q5_rowsplit_gemm_mma.cuh), so where the residual nearly cancels the product they sit a bf16
// step of the product away from the oracle. This kernel rounds once.

#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;

constexpr std::int32_t kRows      = 5120;
constexpr std::int32_t kMaxTokens = 32;

std::uint16_t f32_to_bf16_rne(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f                  = 0.0F;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::vector<std::uint16_t> random_bf16(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<std::uint16_t> bits(n);
    std::uint64_t state = seed;
    for (auto& value : bits) {
        state                 = ninfer::test::quantized_weight::detail::mix64(state);
        const float numerator = static_cast<float>(static_cast<int>(state % 255U) - 127);
        value                 = f32_to_bf16_rne(numerator * scale);
    }
    return bits;
}

bool within(double oracle, float value) {
    return std::isfinite(value) &&
           std::fabs(static_cast<double>(value) - oracle) <= std::fabs(oracle) / 256.0 + 1.0e-5;
}

// Count the elements of `out` (rows x tokens) that miss the oracle; `first` is the first miss.
std::size_t score(const std::vector<std::uint16_t>& out, const std::vector<double>& oracle,
                  std::int32_t tokens, std::size_t& first) {
    std::size_t misses = 0;
    first              = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(kRows) * tokens; ++i) {
        if (!within(oracle[i], bf16_to_f32(out[i]))) {
            if (misses == 0) first = i;
            ++misses;
        }
    }
    return misses;
}

int run_k(std::int32_t k) {
    namespace qw = ninfer::test::quantized_weight;
    qw::PatternedWeightOptions options;
    options.row_split_codes = qw::RowSplitCodePattern::Hashed;
    options.row_split_scale = qw::RowSplitScalePattern::Small;
    qw::PackedWeight host_weight =
        qw::make_patterned_weight(QType::Q5_G64_FP16, kRows, k, 0x5157U + k, options);
    ninfer::test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const ninfer::Weight weight = host_weight.device_weight(device_weight.data());

    const auto activation =
        random_bf16(static_cast<std::size_t>(k) * kMaxTokens, 0x1234, 1.0e-3F);
    const auto residual =
        random_bf16(static_cast<std::size_t>(kRows) * kMaxTokens, 0x9876, 1.0e-2F);

    // The patterned fixture carries no dequantized matrix; decode the payload for the oracle.
    const std::vector<float> dequant =
        qw::decode_row_split_lowbit(host_weight.payload, kRows, k, k, QType::Q5_G64_FP16);
    std::vector<double> oracle(static_cast<std::size_t>(kRows) * kMaxTokens);
    for (std::int32_t col = 0; col < kMaxTokens; ++col) {
        const std::uint16_t* xcol = activation.data() + static_cast<std::size_t>(col) * k;
        for (std::int32_t row = 0; row < kRows; ++row) {
            const float* wrow = dequant.data() + static_cast<std::size_t>(row) * k;
            double sum        = 0.0;
            for (std::int32_t kk = 0; kk < k; ++kk) {
                sum += static_cast<double>(wrow[kk]) * bf16_to_f32(xcol[kk]);
            }
            const std::size_t i = static_cast<std::size_t>(col) * kRows + row;
            oracle[i]           = sum + bf16_to_f32(residual[i]);
        }
    }

    ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
    device_x.copy_from_host(activation.data(), activation.size() * 2);
    ninfer::test::GuardedDeviceBuffer device_out(residual.size() * 2);

    int failures = 0;
    std::vector<std::uint16_t> out(residual.size());
    std::vector<std::uint16_t> first_run;
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 20, 24, 31, 32}) {
        const std::size_t elements = static_cast<std::size_t>(kRows) * tokens;
        Tensor x(device_x.data(), DType::BF16, {k, tokens});
        Tensor y(device_out.data(), DType::BF16, {kRows, tokens});
        for (int repeat = 0; repeat < 3; ++repeat) {
            device_out.copy_from_host(residual.data(), elements * 2);
            ninfer::ops::detail::q5_linear_add_small_t_mma_launch(x, weight, y, nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "q5 linear_add small-T MMA");
            device_out.copy_to_host(out.data(), elements * 2);
            if (repeat == 0) {
                first_run.assign(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(elements));
            } else if (!std::equal(first_run.begin(), first_run.end(), out.begin())) {
                ++failures;
                std::cerr << "K=" << k << " T=" << tokens << ": repeat " << repeat
                          << " differs from the first run\n";
            }
        }
        std::size_t first = 0;
        const auto misses = score(out, oracle, tokens, first);
        if (misses != 0) {
            ++failures;
            std::cerr << "K=" << k << " T=" << tokens << ": " << misses
                      << " outputs miss the fp64 oracle, first at " << first << " ("
                      << bf16_to_f32(out[first]) << " vs " << oracle[first] << ")\n";
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        const int failures = run_k(6144) + run_k(17408);
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " Q5 LinearAdd small-T MMA matches the fp64 oracle at T=1..32, "
                     "K=6144/17408\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5 LinearAdd small-T MMA test failed: " << error.what() << '\n';
        return 1;
    }
}
