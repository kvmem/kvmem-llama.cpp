#include "serve/load_report.h"
#include "serve/cuda_residency_json.h"
#include "serve/kvmem_status_json.h"

#include "serve/request_events.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <utility>

namespace ninfer::serve {

LoadCapacity make_load_capacity(std::string model_id, const ninfer::EngineOptions& engine,
                                const ninfer::MemorySummary& memory) {
    LoadCapacity capacity;
    capacity.model_id             = std::move(model_id);
    capacity.max_concurrency      = engine.max_concurrency;
    capacity.max_pending_requests = engine.max_pending_requests;
    capacity.max_context          = memory.max_context;
    capacity.kv_capacity_tokens   = memory.kv_capacity;
    capacity.kv_capacity_pages    = memory.kv_capacity_page_groups;
    // Engine::options() carries resolved values: C lane slots plus H extra checkpoint slots.
    capacity.device_state_slots =
        engine.max_concurrency + engine.context_cache.device_state_slots.value_or(0);
    capacity.host_state_slots       = memory.host_state_capacity_slots;
    capacity.host_kv_capacity_bytes = memory.host_kv_capacity_bytes;
    return capacity;
}

namespace {

using Json = nlohmann::json;

Json load_report(const LoadCapacity& capacity, const LoadSample& sample) {
    const auto& stats = sample.stats;
    // kv_capacity is page_groups * page_tokens exactly (SequenceCapacityCurve::resolved_tokens).
    const std::uint32_t page_tokens =
        capacity.kv_capacity_pages == 0 ? 0
                                        : capacity.kv_capacity_tokens / capacity.kv_capacity_pages;
    const std::uint64_t occupied_tokens =
        static_cast<std::uint64_t>(stats.device_main_kv_occupied_pages) * page_tokens;

    Json report = {
        {"object", "ninfer.load"},
        {"model", capacity.model_id},
        {"uptime_seconds", std::isfinite(sample.uptime_seconds) ? sample.uptime_seconds : 0.0},
        {"capacity",
         Json{{"max_concurrency", capacity.max_concurrency},
              {"max_pending_requests", capacity.max_pending_requests},
              {"max_admitted_requests", static_cast<std::uint64_t>(capacity.max_concurrency) +
                                            capacity.max_pending_requests},
              {"max_context", capacity.max_context},
              {"kv_capacity_tokens", capacity.kv_capacity_tokens},
              {"kv_capacity_pages", capacity.kv_capacity_pages},
              {"kv_page_tokens", page_tokens},
              {"device_state_slots", capacity.device_state_slots},
              {"host_state_slots", capacity.host_state_slots},
              {"host_kv_bytes", capacity.host_kv_capacity_bytes}}},
        {"requests", Json{{"admitted", sample.admitted_requests},
                          {"running", stats.running_requests},
                          {"prefilling", stats.prefilling_requests},
                          {"decode_ready", stats.decode_ready_requests},
                          {"waiting", stats.waiting_requests},
                          {"materializing", stats.materializing_requests}}},
        {"occupancy", Json{{"device_main_kv_pages", stats.device_main_kv_occupied_pages},
                           {"device_main_kv_tokens", occupied_tokens},
                           {"device_state_slots", stats.device_state_occupied_slots},
                           {"host_state_slots", stats.host_state_occupied_slots},
                           {"host_kv_bytes", stats.host_kv_occupied_bytes}}},
        {"counters", Json{{"computed_prefill_tokens", stats.computed_prefill_tokens},
                          {"committed_decode_tokens", stats.committed_decode_tokens},
                          {"reused_prompt_tokens", stats.reused_prompt_tokens},
                          {"decode_rounds", stats.decode_rounds},
                          {"decode_row_rounds", stats.decode_row_rounds}}},
    };
    if (stats.kvmem.enabled) report["kvmem"] = kvmem_status_json(stats.kvmem);
    return report;
}

} // namespace

std::string make_load_report(const LoadCapacity& capacity, const LoadSample& sample) {
    return load_report(capacity, sample).dump();
}

std::string make_stats_report(const LoadCapacity& capacity, const LoadSample& sample) {
    Json report                         = load_report(capacity, sample);
    report["object"]                    = "ninfer.stats";
    report["requests"]["peak_admitted"] = sample.peak_admitted_requests;
    const ninfer::RuntimeStats& stats   = sample.stats;
    report["memory"] = Json{{"cuda_residency", cuda_residency_json(stats.cuda_residency)}};
    const ThroughputReport since_startup{
        .interval_seconds = std::isfinite(sample.uptime_seconds) ? sample.uptime_seconds : 0.0,
        .computed_prefill_tokens = stats.computed_prefill_tokens,
        .committed_decode_tokens = stats.committed_decode_tokens,
        .decode_rounds           = stats.decode_rounds,
        .decode_row_rounds       = stats.decode_row_rounds,
        .previous                = {},
        .current                 = stats,
    };
    Json cumulative = Json::parse(format_throughput_json({}, 0, since_startup));
    for (const char* block : {"tokens", "throughput_tokens_per_second", "scheduler", "decode_batch",
                              "host_work", "context_cache"}) {
        report[block] = std::move(cumulative.at(block));
    }
    Json entries = Json::array();
    for (std::uint32_t i = 0; i < stats.queue_entries; ++i) {
        entries.push_back(Json{{"request_id", stats.queue[i].request_id},
                               {"position", i + 1},
                               {"wait_seconds", stats.queue[i].wait_seconds}});
    }
    report["queue"] = Json{{"depth", stats.waiting_requests}, {"entries", std::move(entries)}};
    return report.dump();
}

} // namespace ninfer::serve
