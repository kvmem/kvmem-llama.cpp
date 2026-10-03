#include "serve/load_report.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <limits>
#include <string>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    ninfer::EngineOptions engine;
    engine.max_concurrency                  = 4;
    engine.max_pending_requests             = 32;
    engine.context_cache.device_state_slots = 4;

    ninfer::MemorySummary memory;
    memory.max_context               = 65536;
    memory.kv_capacity               = 131072; // 2048 page groups of 64 tokens
    memory.kv_capacity_page_groups   = 2048;
    memory.host_state_capacity_slots = 24;
    memory.host_kv_capacity_bytes    = 16ULL << 30;

    const LoadCapacity capacity = make_load_capacity("qwen3.8-27b", engine, memory);
    failures += check(capacity.device_state_slots == 8,
                      "device StateImage capacity is lane slots plus extra checkpoint slots");

    LoadSample sample;
    sample.uptime_seconds                      = 12.5;
    sample.admitted_requests                   = 6;
    sample.stats.running_requests              = 4;
    sample.stats.prefilling_requests           = 1;
    sample.stats.decode_ready_requests         = 3;
    sample.stats.waiting_requests              = 2;
    sample.stats.materializing_requests        = 0;
    sample.stats.device_main_kv_occupied_pages = 100;
    sample.stats.device_state_occupied_slots   = 5;
    sample.stats.host_state_occupied_slots     = 7;
    sample.stats.host_kv_occupied_bytes        = 1024;
    sample.stats.computed_prefill_tokens       = std::numeric_limits<std::uint64_t>::max() - 1;
    sample.stats.committed_decode_tokens       = 987654321;
    sample.stats.reused_prompt_tokens          = 5555;
    sample.stats.decode_rounds                 = 42;
    sample.stats.decode_row_rounds             = 160;

    const Json report = Json::parse(make_load_report(capacity, sample));
    failures += check(report.at("object") == "ninfer.load", "object tag");
    failures += check(report.at("model") == "qwen3.8-27b", "public model id");
    failures += check(report.at("uptime_seconds") == 12.5, "uptime");

    const Json& cap = report.at("capacity");
    failures += check(cap.at("max_concurrency") == 4 && cap.at("max_pending_requests") == 32,
                      "concurrency limits");
    failures += check(cap.at("max_admitted_requests") == 36,
                      "admission limit is max_concurrency + max_pending_requests");
    failures += check(cap.at("max_context") == 65536, "max_context");
    failures += check(cap.at("kv_capacity_tokens") == 131072 && cap.at("kv_capacity_pages") == 2048,
                      "KV capacity");
    failures += check(cap.at("kv_page_tokens") == 64, "KV page size derives from capacity");
    failures += check(cap.at("device_state_slots") == 8 && cap.at("host_state_slots") == 24 &&
                          cap.at("host_kv_bytes") == (16ULL << 30),
                      "checkpoint capacities");

    const Json& requests = report.at("requests");
    failures += check(requests.at("admitted") == 6 && requests.at("running") == 4 &&
                          requests.at("prefilling") == 1 && requests.at("decode_ready") == 3 &&
                          requests.at("waiting") == 2 && requests.at("materializing") == 0,
                      "request gauges");

    const Json& occupancy = report.at("occupancy");
    failures += check(occupancy.at("device_main_kv_pages") == 100 &&
                          occupancy.at("device_main_kv_tokens") == 6400,
                      "KV occupancy in pages and tokens");
    failures +=
        check(occupancy.at("device_state_slots") == 5 && occupancy.at("host_state_slots") == 7 &&
                  occupancy.at("host_kv_bytes") == 1024,
              "checkpoint occupancy");

    const Json& counters = report.at("counters");
    failures += check(counters.at("computed_prefill_tokens").get<std::uint64_t>() ==
                          std::numeric_limits<std::uint64_t>::max() - 1,
                      "64-bit counters survive JSON rendering");
    failures +=
        check(counters.at("committed_decode_tokens") == 987654321 &&
                  counters.at("reused_prompt_tokens") == 5555 &&
                  counters.at("decode_rounds") == 42 && counters.at("decode_row_rounds") == 160,
              "decode counters");

    // A capacity without resolved pages (never expected after attach) must not divide by zero.
    LoadCapacity unresolved      = capacity;
    unresolved.kv_capacity_pages = 0;
    const Json degenerate        = Json::parse(make_load_report(unresolved, sample));
    failures += check(degenerate.at("capacity").at("kv_page_tokens") == 0 &&
                          degenerate.at("occupancy").at("device_main_kv_tokens") == 0,
                      "zero page groups renders zero page size");

    // /stats: the same snapshot with the ingress peak, and the Engine counters since startup in
    // the throughput record's shape, averaged over the uptime.
    sample.peak_admitted_requests                 = 9;
    sample.stats.computed_prefill_tokens          = 5000;
    sample.stats.committed_decode_tokens          = 250;
    sample.stats.active_captures_completed        = 3;
    sample.stats.pressure_private_owners_demoted  = 2;
    sample.stats.pressure_private_owners_degraded = 2;
    sample.stats.cuda_residency = {.enabled = true,
                                  .verified = true,
                                  .cuda_free_bytes = 0,
                                  .cuda_total_bytes = 16ULL << 30,
                                  .dedicated_bytes = 15ULL << 30,
                                  .shared_bytes = 74ULL << 20,
                                  .shared_baseline_bytes = 74ULL << 20,
                                  .device_allocated_bytes = 14ULL << 30,
                                  .host_pool_bytes = 4ULL << 30,
                                  .host_used_bytes = 2ULL << 30,
                                  .verified_reserve_bytes = 64ULL << 20};
    const Json stats = Json::parse(make_stats_report(capacity, sample));
    const Json& residency = stats.at("memory").at("cuda_residency");
    failures += check(residency.at("enabled") == true && residency.at("verified") == true &&
                          residency.at("cuda_free_bytes") == 0 &&
                          residency.at("cuda_total_bytes") == (16ULL << 30) &&
                          residency.at("dedicated_bytes") == (15ULL << 30) &&
                          residency.at("shared_bytes") == (74ULL << 20) &&
                          residency.at("shared_baseline_bytes") == (74ULL << 20) &&
                          residency.at("device_allocated_bytes") == (14ULL << 30) &&
                          residency.at("host_pool_bytes") == (4ULL << 30) &&
                          residency.at("host_used_bytes") == (2ULL << 30) &&
                          residency.at("verified_reserve_bytes") == (64ULL << 20),
                      "stats report lost cached residency verification or byte counters");
    const Json disabled_residency = Json::parse(make_stats_report(capacity, LoadSample{}))
                                        .at("memory").at("cuda_residency");
    failures += check(disabled_residency.at("enabled") == false &&
                          disabled_residency.at("verified") == false,
                      "disabled residency monitoring must not claim verification");
    failures += check(stats.at("object") == "ninfer.stats" &&
                          stats.at("capacity") == report.at("capacity") &&
                          stats.at("requests").at("admitted") == 6 &&
                          stats.at("requests").at("peak_admitted") == 9,
                      "stats report does not extend the load report with the ingress peak");
    failures += check(stats.at("tokens").at("computed_prefill") == 5000 &&
                          stats.at("tokens").at("committed_decode") == 250 &&
                          stats.at("throughput_tokens_per_second").at("prefill") == 400.0 &&
                          stats.at("throughput_tokens_per_second").at("decode") == 20.0,
                      "stats report does not average the counters over the uptime");
    failures +=
        check(stats.at("context_cache").at("captures").at("completed") == 3 &&
                  stats.at("context_cache").at("pressure").at("private_owners_demoted") == 2 &&
                  stats.at("scheduler").at("running") == 4 && stats.contains("host_work") &&
                  stats.contains("decode_batch"),
              "stats report does not carry the cumulative context-cache counters");
    failures += check(
        !Json::parse(make_load_report(capacity, sample)).at("requests").contains("peak_admitted"),
        "the load report grew the stats-only ingress peak");
    failures +=
        check(stats.at("queue").at("entries").empty(), "an empty queue reported waiting entries");
    sample.stats.queue[0]      = {.request_id = 41, .wait_seconds = 2.5};
    sample.stats.queue[1]      = {.request_id = 43, .wait_seconds = 0.5};
    sample.stats.queue_entries = 2;
    const Json queued          = Json::parse(make_stats_report(capacity, sample)).at("queue");
    failures += check(queued.at("depth") == sample.stats.waiting_requests &&
                          queued.at("entries").size() == 2 &&
                          queued.at("entries")[0].at("request_id") == 41 &&
                          queued.at("entries")[0].at("position") == 1 &&
                          queued.at("entries")[0].at("wait_seconds") == 2.5 &&
                          queued.at("entries")[1].at("position") == 2,
                      "the stats report did not list the waiting requests in order");

    if (failures != 0) { std::cerr << failures << " load report check(s) failed\n"; }
    return failures == 0 ? 0 : 1;
}
