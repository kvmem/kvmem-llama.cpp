#pragma once

// GET /metrics in the Prometheus text format. The llamacpp:-prefixed series keep llama.cpp's
// --metrics names and meanings, so dashboards and autoscalers built against it read this server
// unchanged; the ninfer:-prefixed series report what llama.cpp has no name for.

#include "serve/generation_service.h"
#include "serve/load_report.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

class ServeMetrics {
public:
    // Accumulates one completed request, from the same funnel as the request-done record.
    void record(const GenerationOutcome& outcome);

    [[nodiscard]] std::string render(const LoadCapacity& capacity, const LoadSample& sample) const;

private:
    mutable std::mutex mutex_;
    std::uint64_t requests_total_                = 0;
    double prompt_seconds_total_                 = 0.0;
    std::uint64_t tokens_predicted_total_        = 0;
    double tokens_predicted_seconds_total_       = 0.0;
    std::uint64_t prefix_cache_hit_tokens_total_ = 0;
    std::uint64_t draft_tokens_total_            = 0;
    std::uint64_t draft_accepted_tokens_total_   = 0;
};

} // namespace ninfer::serve
