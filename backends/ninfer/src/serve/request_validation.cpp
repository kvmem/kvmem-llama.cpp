#include "serve/request_validation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

const char* request_json_type_name(const RequestJson& value) noexcept {
    switch (value.type()) {
    case RequestJson::value_t::null: return "null";
    case RequestJson::value_t::object: return "object";
    case RequestJson::value_t::array: return "array";
    case RequestJson::value_t::string: return "string";
    case RequestJson::value_t::boolean: return "boolean";
    case RequestJson::value_t::number_integer:
    case RequestJson::value_t::number_unsigned:
    case RequestJson::value_t::number_float: return "number";
    case RequestJson::value_t::binary: return "binary";
    case RequestJson::value_t::discarded: return "discarded";
    }
    return "value";
}

std::string ascii_preview(std::string_view value, std::size_t limit) {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() < limit ? value.size() : limit);
    std::size_t shown = 0;
    for (const unsigned char character : value) {
        if (shown == limit) {
            out += "...";
            break;
        }
        if (character >= 0x20 && character <= 0x7e) {
            out.push_back(static_cast<char>(character));
        } else {
            out += "\\x";
            out.push_back(kHexDigits[character >> 4]);
            out.push_back(kHexDigits[character & 0x0f]);
        }
        ++shown;
    }
    return out;
}

std::optional<SamplingParams> parse_post_thinking(const RequestJson& body, double max_temperature) {
    if (!body.contains("post_thinking") || body.at("post_thinking").is_null()) {
        return std::nullopt;
    }
    const RequestJson& object = body.at("post_thinking");
    if (!object.is_object()) { bad_request("post_thinking must be an object", "post_thinking"); }
    static constexpr std::string_view kFields[] = {
        "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed"};
    for (const auto& [key, value] : object.items()) {
        (void)value;
        if (std::find(std::begin(kFields), std::end(kFields), key) == std::end(kFields)) {
            bad_request("post_thinking." + key + " is not a sampling field", "post_thinking");
        }
    }
    const auto ranged = [&](const char* key, double low, double high) {
        const std::optional<double> value = optional_number(object, key);
        if (value && !(*value >= low && *value <= high)) {
            bad_request(std::string("post_thinking.") + key + " is out of range",
                        std::string("post_thinking.") + key);
        }
        return value;
    };
    SamplingParams out;
    out.temperature       = ranged("temperature", 0.0, max_temperature);
    out.top_p             = ranged("top_p", 0.0, 1.0);
    out.min_p             = ranged("min_p", 0.0, 1.0);
    out.presence_penalty  = ranged("presence_penalty", -2.0, 2.0);
    out.frequency_penalty = ranged("frequency_penalty", -2.0, 2.0);
    out.top_k             = optional_int(object, "top_k");
    if (out.top_k && (*out.top_k < 0 || *out.top_k > 20)) {
        bad_request("post_thinking.top_k must be in [0,20]", "post_thinking.top_k");
    }
    if (object.contains("seed") && !object.at("seed").is_null()) {
        const RequestJson& seed = object.at("seed");
        if (!seed.is_number_integer()) {
            bad_request("post_thinking.seed must be an integer", "post_thinking.seed");
        }
        out.seed = seed.is_number_unsigned() ? seed.get<std::uint64_t>()
                                             : static_cast<std::uint64_t>(seed.get<std::int64_t>());
    }
    return out;
}

} // namespace ninfer::serve
