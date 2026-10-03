#include "serve/serve_metrics.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::map<std::string, double> samples(const std::string& body) {
    std::map<std::string, double> out;
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        const auto space = line.find(' ');
        out[line.substr(0, space)] = std::stod(line.substr(space + 1));
    }
    return out;
}

} // namespace

int main() {
    int failures = 0;
    LoadCapacity capacity;
    capacity.max_concurrency   = 4;
    capacity.kv_capacity_tokens = 131072;
    capacity.kv_capacity_pages  = 2048;

    LoadSample sample;
    sample.uptime_seconds                      = 30.0;
    sample.admitted_requests                   = 5;
    sample.stats.computed_prefill_tokens       = 9000;
    sample.stats.decode_rounds                 = 10;
    sample.stats.decode_row_rounds             = 25;
    sample.stats.device_main_kv_occupied_pages = 512;
    sample.stats.running_requests              = 3;
    sample.stats.waiting_requests              = 2;
    sample.stats.reused_prompt_tokens          = 700;

    ServeMetrics metrics;
    GenerationOutcome outcome;
    outcome.completion_tokens                        = 40;
    outcome.metrics.prefill_seconds                  = 1.5;
    outcome.metrics.decode_seconds                   = 2.0;
    outcome.metrics.prefix_cache_hit_tokens          = 300;
    outcome.metrics.speculative_draft_tokens         = 30;
    outcome.metrics.speculative_accepted_tokens      = 21;
    metrics.record(outcome);
    metrics.record(outcome);

    const std::string body = metrics.render(capacity, sample);
    const auto values      = samples(body);
    failures += check(body.find("# TYPE llamacpp:prompt_tokens_total counter") != std::string::npos,
                      "series carry their Prometheus type");
    failures += check(values.at("llamacpp:prompt_tokens_total") == 9000.0,
                      "prompt tokens come from the Engine's computed prefill counter");
    failures += check(values.at("llamacpp:prompt_seconds_total") == 3.0 &&
                          values.at("llamacpp:tokens_predicted_total") == 80.0 &&
                          values.at("llamacpp:tokens_predicted_seconds_total") == 4.0,
                      "completed requests accumulate prefill and decode time and tokens");
    failures += check(values.at("llamacpp:predicted_tokens_seconds") == 20.0 &&
                          values.at("llamacpp:prompt_tokens_seconds") == 3000.0,
                      "average throughputs divide the matching counters");
    failures += check(values.at("llamacpp:n_decode_total") == 10.0 &&
                          values.at("llamacpp:n_busy_slots_per_decode") == 2.5,
                      "decode rounds and their average batch are reported");
    failures += check(values.at("llamacpp:kv_cache_usage_ratio") == 0.25 &&
                          values.at("llamacpp:kv_cache_tokens") == 32768.0,
                      "KV occupancy is reported in pages and tokens");
    failures += check(values.at("llamacpp:requests_processing") == 3.0 &&
                          values.at("llamacpp:requests_deferred") == 2.0 &&
                          values.at("ninfer:requests_admitted") == 5.0,
                      "request gauges follow the Engine snapshot and ingress");
    failures += check(values.at("ninfer:requests_total") == 2.0 &&
                          values.at("ninfer:prefix_cache_hit_tokens_total") == 600.0 &&
                          values.at("ninfer:draft_tokens_total") == 60.0 &&
                          values.at("ninfer:draft_accepted_tokens_total") == 42.0 &&
                          values.at("ninfer:reused_prompt_tokens_total") == 700.0,
                      "ninfer series accumulate reuse and speculation");

    const auto empty = samples(ServeMetrics{}.render(LoadCapacity{}, LoadSample{}));
    failures += check(empty.at("llamacpp:predicted_tokens_seconds") == 0.0 &&
                          empty.at("llamacpp:kv_cache_usage_ratio") == 0.0,
                      "an idle server reports zero rather than a division by zero");
    if (failures != 0) { return 1; }
    std::cout << "serve metrics render llama.cpp-compatible Prometheus series\n";
    return 0;
}
