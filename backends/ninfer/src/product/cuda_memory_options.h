#pragma once

#include <ninfer/types.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

struct CudaMemoryOptions {
    CudaMemoryPolicy policy = CudaMemoryPolicy::DriverDefault;
    std::size_t reserve_bytes = 0;
    std::size_t probe_step_bytes = 0;
};

inline CudaMemoryOptions parse_cuda_memory_options(std::string_view value) {
    if (value == "default") { return {}; }
    if (value == "mixed") { return {CudaMemoryPolicy::Mixed, 0, 0}; }
    if (value == "strict") { return {CudaMemoryPolicy::StrictVram, 64ULL << 20, 128ULL << 20}; }
    const auto invalid = [] {
        return std::invalid_argument(
            "--cuda-memory-policy expects default, mixed, strict or strict-RESERVE-STEP; "
            "RESERVE is nonnegative MiB and STEP is 1..16384 MiB, using decimal digits only");
    };
    if (!value.starts_with("strict-")) { throw invalid(); }
    const auto parts = value.substr(7);
    const auto separator = parts.find('-');
    if (separator == std::string_view::npos) { throw invalid(); }
    const auto integer = [&](std::string_view text, std::size_t maximum) {
        if (text.empty()) { throw invalid(); }
        std::size_t result = 0;
        for (const char ch : text) {
            if (ch < '0' || ch > '9') { throw invalid(); }
            const auto digit = static_cast<std::size_t>(ch - '0');
            if (result > maximum / 10 ||
                (result == maximum / 10 && digit > maximum % 10)) { throw invalid(); }
            result = result * 10 + digit;
        }
        return result;
    };
    constexpr auto maximum_mib = std::numeric_limits<std::size_t>::max() >> 20;
    const auto reserve = integer(parts.substr(0, separator), maximum_mib);
    const auto step = integer(parts.substr(separator + 1), std::min<std::size_t>(16384, maximum_mib));
    if (step == 0) { throw invalid(); }
    return {CudaMemoryPolicy::StrictVram, reserve << 20, step << 20};
}

inline std::string format_cuda_memory_policy(CudaMemoryPolicy policy,
                                            std::size_t reserve_bytes,
                                            std::size_t probe_step_bytes) {
    switch (policy) {
    case CudaMemoryPolicy::DriverDefault: return "default";
    case CudaMemoryPolicy::Mixed: return "mixed";
    case CudaMemoryPolicy::StrictVram:
        if (reserve_bytes == (64ULL << 20) && probe_step_bytes == (128ULL << 20)) {
            return "strict";
        }
        return "strict-" + std::to_string(reserve_bytes >> 20) + '-' +
               std::to_string(probe_step_bytes >> 20);
    }
    throw std::invalid_argument("unknown CUDA memory policy");
}

} // namespace ninfer::product
