#pragma once

#include "ninfer/types.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

// Sets one post-thinking sampling field by its --post-thinking-sampler key: temp, top_p, top_k,
// min_p, presence or frequency.
inline void set_post_thinking_field(SamplingOverrides& out, std::string_view key,
                                    std::string_view text) {
    const auto invalid = [&](const char* range) {
        return std::invalid_argument("post-thinking " + std::string(key) + " must be in " + range +
                                     ": " + std::string(text));
    };
    if (key == "top_k") {
        std::int32_t value      = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (text.empty() || error != std::errc{} || end != text.data() + text.size() || value < 0 ||
            value > 20) {
            throw invalid("[0,20]");
        }
        out.top_k = value;
        return;
    }

    struct Field {
        std::string_view key;
        double low, high;
        const char* range;
        std::optional<float> SamplingOverrides::* member;
    };

    static constexpr Field kFields[] = {
        {"temp", 0.0, 2.0, "[0,2]", &SamplingOverrides::temperature},
        {"top_p", 0.0, 1.0, "[0,1]", &SamplingOverrides::top_p},
        {"min_p", 0.0, 1.0, "[0,1]", &SamplingOverrides::min_p},
        {"presence", -2.0, 2.0, "[-2,2]", &SamplingOverrides::presence_penalty},
        {"frequency", -2.0, 2.0, "[-2,2]", &SamplingOverrides::frequency_penalty},
    };
    for (const Field& field : kFields) {
        if (field.key != key) { continue; }
        double value            = 0.0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (text.empty() || error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(value) || value < field.low || value > field.high) {
            throw invalid(field.range);
        }
        out.*field.member = static_cast<float>(value);
        return;
    }
    throw std::invalid_argument("unknown --post-thinking-sampler key: " + std::string(key));
}

// --post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F].
inline void apply_post_thinking_sampler(std::string_view spec, SamplingOverrides& out) {
    if (spec.empty()) { throw std::invalid_argument("--post-thinking-sampler must not be empty"); }
    std::size_t begin = 0;
    while (begin <= spec.size()) {
        const std::size_t comma      = spec.find(',', begin);
        const std::string_view field = spec.substr(
            begin, comma == std::string_view::npos ? std::string_view::npos : comma - begin);
        const std::size_t equals = field.find('=');
        if (equals == std::string_view::npos) {
            throw std::invalid_argument("--post-thinking-sampler field missing '=': " +
                                        std::string(field));
        }
        set_post_thinking_field(out, field.substr(0, equals), field.substr(equals + 1));
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
}

} // namespace ninfer::product
