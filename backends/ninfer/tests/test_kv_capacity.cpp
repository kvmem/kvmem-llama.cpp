#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // A model split across devices: each holds its own layers' KV, so a page group costs each a
    // different amount, and the capacity is the smallest any one allows.
    ninfer::runtime::SequenceCapacityCurve split{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 20,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 100,
        .extra_ranks                          = {{.minimum_device_reservation_bytes     = 500,
                                                  .bytes_per_additional_main_page_group = 300}},
    };
    const std::size_t split_available[] = {1500};
    // Device 0 alone would allow 11 pages; device 1 allows 2 + (1500 - 50 - 500) / 300 = 5.
    const auto both = ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50),
                                                           split, 2000, split_available);
    failures += check(both.main_page_groups == 5 && both.binding_rank == 1 &&
                          both.extra_rank_reservation_bytes.size() == 1 &&
                          both.extra_rank_reservation_bytes[0] == 500 + 3 * 300,
                      "the capacity was not limited by the tighter device");
    failures += check(split.extra_rank_reservation_bytes(0, 5) == 1400,
                      "a further device's reservation does not follow its own curve");

    // A device holding no KV does not limit the count, so the primary device decides.
    split.extra_ranks[0].bytes_per_additional_main_page_group = 0;
    const auto free_rank = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), split, 2000, split_available);
    failures += check(free_rank.main_page_groups == 11 && free_rank.binding_rank == 0,
                      "a device without KV constrained the capacity");
    split.extra_ranks[0].bytes_per_additional_main_page_group = 300;

    // A failure names the device that could not hold the plan.
    const auto names_device = [&](auto&& call, const char* device) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what()).find(device) != std::string::npos;
        }
        return false;
    };
    const std::size_t starved[] = {520};
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000, starved);
                          },
                          "device 1"),
                      "a device below its minimum reservation was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::explicit_capacity(64 * 19), split, 100000,
                                  split_available);
                          },
                          "device 1"),
                      "an explicit capacity too large for one device was not named");
    failures += check(names_device(
                          [&] {
                              (void)ninfer::runtime::resolve_kv_capacity(
                                  ninfer::KvCapacityPolicy::automatic(50), split, 2000);
                          },
                          "every device"),
                      "missing free memory for a device was accepted");

    // Mixed allocation is demand-bounded, not free-memory-bounded. Two 129-token windows
    // each need three 64-token pages: six pages and 1512 runtime bytes, even with zero free.
    const auto mixed_auto = ninfer::runtime::resolve_mixed_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(0), curve, 129, 2, 0);
    failures += check(mixed_auto.main_page_groups == 6 && mixed_auto.resolved_tokens == 384 &&
                          mixed_auto.runtime_reservation_bytes == 1512 &&
                          mixed_auto.mode == ninfer::KvCapacityMode::Automatic &&
                          mixed_auto.available_after_weights_bytes == 0 &&
                          mixed_auto.planned_slack_bytes == 0,
                      "mixed auto did not allocate one page-aligned context window per lane");
    // A large Hybrid curve must not turn a driver-paging policy into an unbounded RAM probe.
    auto hybrid = curve;
    hybrid.maximum_main_page_groups = 1000000;
    const auto mixed_room = ninfer::runtime::resolve_mixed_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(0), hybrid, 129, 2, 1ULL << 40);
    failures += check(mixed_room.main_page_groups == 6 && mixed_room.runtime_reservation_bytes == 1512 &&
                          mixed_room.available_after_weights_bytes == (1ULL << 40) &&
                          mixed_room.planned_slack_bytes == (1ULL << 40) - 1512,
                      "mixed auto expanded idle cache beyond its finite workload bound");
    const auto mixed_capped = ninfer::runtime::resolve_mixed_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(0), curve, 129, 8, 0);
    failures += check(mixed_capped.main_page_groups == 6,
                      "mixed auto ignored the target's maximum usable page count");
    const auto mixed_fixed = ninfer::runtime::resolve_mixed_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(257), curve, 129, 1, 10);
    failures += check(mixed_fixed.main_page_groups == 5 && mixed_fixed.resolved_tokens == 320 &&
                          mixed_fixed.runtime_reservation_bytes == 1384 &&
                          mixed_fixed.available_after_weights_bytes == 10 &&
                          mixed_fixed.planned_slack_bytes == 0,
                      "mixed explicit capacity was reduced or rejected by reported free bytes");
    const auto invalid = [](auto&& call) {
        try { call(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(invalid([&] {
        (void)ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(257), curve, 10);
    }), "default fixed capacity stopped checking reported free memory");
    failures += check(invalid([&] {
        (void)ninfer::runtime::resolve_mixed_kv_capacity(
            ninfer::KvCapacityPolicy::automatic(1), curve, 129, 2, 0);
    }), "unnormalized mixed automatic headroom was silently ignored");
    failures += check(invalid([&] {
        (void)ninfer::runtime::resolve_mixed_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(64), curve, 129, 1, 999999);
    }), "mixed fixed capacity below max_context was accepted");
    failures += check(invalid([&] {
        (void)ninfer::runtime::resolve_mixed_kv_capacity(
            ninfer::KvCapacityPolicy::automatic(0), curve, 1024, 1, 999999);
    }), "mixed accepted a capacity curve unable to hold one full context");
    failures += check(invalid([&] {
        (void)ninfer::runtime::resolve_mixed_kv_capacity(
            ninfer::KvCapacityPolicy::automatic(0), split, 129, 2, 999999);
    }), "mixed accepted a multi-device capacity plan");
    for (const auto concurrency : {0U, 9U}) {
        failures += check(invalid([&] {
            (void)ninfer::runtime::resolve_mixed_kv_capacity(
                ninfer::KvCapacityPolicy::automatic(0), curve, 129, concurrency, 999999);
        }), "mixed accepted an invalid concurrency");
    }
    const auto overflows = [](auto&& call) {
        try { call(); } catch (const std::overflow_error&) { return true; }
        return false;
    };
    auto byte_overflow = curve;
    byte_overflow.minimum_device_reservation_bytes = std::numeric_limits<std::size_t>::max() - 1;
    failures += check(overflows([&] {
        (void)ninfer::runtime::resolve_mixed_kv_capacity(
            ninfer::KvCapacityPolicy::automatic(0), byte_overflow, 129, 2, 0);
    }), "mixed runtime byte reservation overflow was not detected");
    const ninfer::runtime::SequenceCapacityCurve token_overflow{
        .main_page_tokens = std::numeric_limits<std::uint32_t>::max(),
        .minimum_main_page_groups = 1,
        .maximum_main_page_groups = 3,
        .minimum_device_reservation_bytes = 1000,
        .bytes_per_additional_main_page_group = 128,
    };
    failures += check(overflows([&] {
        (void)ninfer::runtime::resolve_mixed_kv_capacity(ninfer::KvCapacityPolicy::automatic(0),
            token_overflow, std::numeric_limits<std::uint32_t>::max(), 2, 0);
    }), "mixed per-lane token capacity overflow was not detected");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
