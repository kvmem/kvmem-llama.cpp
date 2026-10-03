#include "core/token_logprobs.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::uint16_t bf16(float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
}

} // namespace

int main() {
    int failures = 0;

    // Logits log(1), log(2), log(3), log(2) (exact in BF16 only approximately, so the reference
    // uses the rounded values): probabilities 1/8, 2/8, 3/8, 2/8.
    std::vector<float> raw{0.0F, 0.6931472F, 1.0986123F, 0.6931472F};
    std::vector<std::uint16_t> logits;
    std::vector<double> rounded;
    for (const float value : raw) {
        logits.push_back(bf16(value));
        rounded.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(logits.back()) << 16U));
    }
    double sum = 0.0;
    for (const double value : rounded) { sum += std::exp(value); }
    const auto reference = [&](std::size_t token) { return rounded[token] - std::log(sum); };

    const ninfer::FirstTokenLogprobs out = ninfer::token_logprobs_from_bf16(logits, 3, 3);
    failures +=
        check(out.selected.token == 3 && std::abs(out.selected.logprob - reference(3)) < 1e-6,
              "the selected token's log probability is wrong");
    failures += check(out.top.size() == 3, "top did not return three tokens");
    if (out.top.size() == 3) {
        failures += check(out.top[0].token == 2 && out.top[1].token == 1 && out.top[2].token == 3,
                          "top is not ordered by probability with the lower id first on ties");
        for (const ninfer::TokenLogprob& entry : out.top) {
            failures += check(
                std::abs(entry.logprob - reference(static_cast<std::size_t>(entry.token))) < 1e-6,
                "a top log probability is wrong");
        }
    }

    const ninfer::FirstTokenLogprobs wide = ninfer::token_logprobs_from_bf16(logits, 0, 20);
    failures +=
        check(wide.top.size() == logits.size(), "top larger than the domain was not clamped");

    std::vector<std::uint16_t> with_nan  = logits;
    with_nan[1]                          = bf16(std::numeric_limits<float>::quiet_NaN());
    const ninfer::FirstTokenLogprobs nan = ninfer::token_logprobs_from_bf16(with_nan, 1, 4);
    failures += check(std::isinf(nan.selected.logprob) && nan.selected.logprob < 0.0F &&
                          nan.top.back().token == 1,
                      "a NaN logit was not treated as impossible");

    bool rejected = false;
    try {
        (void)ninfer::token_logprobs_from_bf16(logits, 4, 1);
    } catch (const std::invalid_argument&) { rejected = true; }
    failures += check(rejected, "a selected token outside the domain was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
