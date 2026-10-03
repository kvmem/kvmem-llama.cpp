#include "core/disk_kv_bridge.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using ninfer::DiskKVBridge;
using ninfer::DiskKVIdentity;
using ninfer::DiskKVKind;
using ninfer::SpillStatus;
using ninfer::SpillTicket;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<std::byte> page(std::size_t bytes, std::uint32_t seed) {
    std::vector<std::byte> out(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<std::byte>((i * 31U + seed * 7U) & 0xFFU);
    }
    return out;
}

DiskKVIdentity id(std::uint64_t key, std::uint32_t frontier) {
    return DiskKVIdentity{.lo = key, .hi = key * 3U + 1U, .tag = 0x10001, .frontier = frontier};
}

SpillStatus settle(const SpillTicket& ticket) {
    while (DiskKVBridge::poll(ticket) == SpillStatus::Pending) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return DiskKVBridge::poll(ticket);
}

template <typename Make>
SpillTicket retry(Make make) {
    for (;;) {
        if (SpillTicket ticket = make()) { return ticket; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

constexpr std::size_t kMain    = 32 * 1024;
constexpr std::size_t kBackend = 8 * 1024;
constexpr std::size_t kState   = 96 * 1024;

DiskKVBridge::Options options(const std::filesystem::path& root) {
    return DiskKVBridge::Options{.base_path           = root.string(),
                                 .main_page_stride    = kMain,
                                 .backend_page_stride = kBackend,
                                 .state_page_stride   = kState,
                                 .capacity_bytes      = 64ULL << 20U};
}

} // namespace

int main() {
    try {
        const auto root = std::filesystem::temp_directory_path() /
                          ("ninfer_disk_kv_bridge_" + std::to_string(std::random_device{}()));
        std::filesystem::remove_all(root);
        const auto main_page  = page(kMain, 1);
        const auto state_page = page(kState, 2);
        const auto tail_page  = page(kMain, 3);
        {
            DiskKVBridge bridge(options(root));
            check(bridge.slot_count(DiskKVKind::MainKV) != 0 &&
                      bridge.slot_count(DiskKVKind::BackendKV) != 0 &&
                      bridge.slot_count(DiskKVKind::StateImage) != 0,
                  "every family with a stride gets a store");
            check(bridge.spill_page_wait(id(1, 64), DiskKVKind::MainKV, main_page,
                                         std::chrono::milliseconds(1000)),
                  "a queued spill is accepted");
            check(!bridge.spill_page_wait(id(2, 64), DiskKVKind::MainKV, page(kBackend, 1),
                                          std::chrono::milliseconds(10)),
                  "a page of the wrong stride is refused");
            check(bridge.spill_page_sync(id(9, 100), DiskKVKind::StateImage, state_page),
                  "a state image is written synchronously");
            bridge.wait_idle();
            std::vector<std::byte> out(kMain);
            check(bridge.restore_page(id(1, 64), DiskKVKind::MainKV, out) && out == main_page,
                  "a spilled page restores byte for byte");
            check(!bridge.restore_page(id(1, 64), DiskKVKind::BackendKV, out),
                  "families do not share pages");

            const SpillTicket missing =
                retry([&] { return bridge.try_probe(id(5, 128), DiskKVKind::MainKV); });
            check(settle(missing) == SpillStatus::Missing, "a probe reports an absent page");
            const SpillTicket written =
                retry([&] { return bridge.try_submit(id(5, 128), DiskKVKind::MainKV, tail_page); });
            check(settle(written) == SpillStatus::Written, "a submitted page is written");
            const SpillTicket present =
                retry([&] { return bridge.try_probe(id(5, 128), DiskKVKind::MainKV); });
            check(settle(present) == SpillStatus::Present, "a probe reports a stored page");
            std::vector<std::byte> pair(2 * kMain);
            check(bridge.restore_pages(std::vector<DiskKVIdentity>{id(1, 64), id(5, 128)},
                                       DiskKVKind::MainKV, pair) &&
                      std::equal(main_page.begin(), main_page.end(), pair.begin()) &&
                      std::equal(tail_page.begin(), tail_page.end(), pair.begin() + kMain),
                  "several pages restore in parallel in their order");
            check(!bridge.restore_pages(std::vector<DiskKVIdentity>{id(1, 64), id(6, 192)},
                                        DiskKVKind::MainKV, pair),
                  "a missing page fails the whole range");
            const auto frontiers = bridge.live_state_frontiers();
            check(frontiers.size() == 1 && frontiers.front() == 100,
                  "state frontiers list the stored state images");
        }
        {
            DiskKVBridge bridge(options(root));
            std::vector<std::byte> out(kMain);
            check(bridge.restore_page(id(5, 128), DiskKVKind::MainKV, out) && out == tail_page,
                  "pages survive a restart of the bridge");
            std::vector<std::byte> state(kState);
            check(bridge.restore_page(id(9, 100), DiskKVKind::StateImage, state) &&
                      state == state_page,
                  "state images survive a restart");
            check(bridge.drop_page(id(1, 64), DiskKVKind::MainKV) &&
                      !bridge.contains(id(1, 64), DiskKVKind::MainKV),
                  "a dropped page is gone");
            const auto stats = bridge.stats();
            check(stats.restores == 2, "restore statistics count");
        }
        if (!ninfer::DirectStorageReader::available_in_build()) {
            DiskKVBridge::Options direct = options(root);
            direct.direct_storage        = true;
            bool refused                 = false;
            try {
                DiskKVBridge bridge(direct);
            } catch (const std::runtime_error&) { refused = true; }
            check(refused, "a build without DirectStorage refuses to open a DirectStorage bridge");
        }
        std::filesystem::remove_all(root);
    } catch (const std::exception& error) {
        std::cerr << "disk KV bridge test: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "disk KV bridge: queued, synchronous and ticketed spills, restores, restart ok\n";
    return 0;
}
