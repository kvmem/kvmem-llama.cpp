#include "core/token_logprobs.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ninfer {
namespace {

float bf16_to_float(std::uint16_t bits) noexcept {
    const float value = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    return std::isnan(value) ? -std::numeric_limits<float>::infinity() : value;
}

} // namespace

FirstTokenLogprobs token_logprobs_from_bf16(std::span<const std::uint16_t> logits, TokenId selected,
                                            std::uint32_t top) {
    if (logits.empty() || selected < 0 || static_cast<std::size_t>(selected) >= logits.size()) {
        throw std::invalid_argument("token log probabilities need the selected token's logit");
    }
    std::vector<float> values(logits.size());
    std::transform(logits.begin(), logits.end(), values.begin(), bf16_to_float);
    const float maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum)) {
        throw std::invalid_argument("token log probabilities need a finite logit");
    }
    double sum = 0.0;
    for (const float value : values) { sum += std::exp(static_cast<double>(value - maximum)); }
    const double normalizer = static_cast<double>(maximum) + std::log(sum);
    const auto logprob      = [&](std::size_t token) {
        return static_cast<float>(static_cast<double>(values[token]) - normalizer);
    };

    FirstTokenLogprobs out;
    out.selected = {.token = selected, .logprob = logprob(static_cast<std::size_t>(selected))};
    const std::size_t count = std::min<std::size_t>(top, values.size());
    std::vector<std::uint32_t> order(values.size());
    std::iota(order.begin(), order.end(), 0U);
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count),
                      order.end(), [&](std::uint32_t left, std::uint32_t right) {
                          return values[left] != values[right] ? values[left] > values[right]
                                                               : left < right;
                      });
    out.top.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        out.top.push_back(
            {.token = static_cast<TokenId>(order[index]), .logprob = logprob(order[index])});
    }
    return out;
}

} // namespace ninfer
