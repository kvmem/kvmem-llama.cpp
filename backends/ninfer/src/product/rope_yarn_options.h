#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace ninfer::product {

inline float parse_rope_yarn_factor(std::string_view text) {
    double value = 0.0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 1.0 || value > 4.0) {
        throw std::invalid_argument("--rope-yarn-factor must be finite and in [1,4]");
    }
    return static_cast<float>(value);
}

inline float parse_rope_scaling_factor(std::string_view text) {
    double value            = 0.0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 1.0 || value > 32.0) {
        throw std::invalid_argument("--rope-scaling-factor must be finite and in [1,32]");
    }
    return static_cast<float>(value);
}

inline std::uint32_t parse_rope_scaling_original_context(std::string_view text) {
    std::uint32_t value     = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::invalid_argument("--rope-scaling-original-context must be a positive integer");
    }
    return value;
}

} // namespace ninfer::product
