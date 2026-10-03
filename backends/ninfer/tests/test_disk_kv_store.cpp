#include "core/disk_kv_store.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using ninfer::DiskKVIdentity;
using ninfer::DiskKVStore;

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
        out[i] = static_cast<std::byte>((i * 131U + seed) & 0xFFU);
    }
    return out;
}

DiskKVIdentity id(std::uint64_t key, std::uint32_t frontier, std::uint32_t tag = 0) {
    return DiskKVIdentity{.lo = key, .hi = ~key, .tag = tag, .frontier = frontier};
}

bool reads_back(DiskKVStore& store, const DiskKVIdentity& key,
                const std::vector<std::byte>& expected) {
    std::vector<std::byte> actual(expected.size());
    return store.read_page(key, actual) &&
           std::memcmp(actual.data(), expected.data(), expected.size()) == 0;
}

std::filesystem::path fresh_directory() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("ninfer_disk_kv_store_" + std::to_string(std::random_device{}()));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}

constexpr std::size_t kSlot        = 64 * 1024;
constexpr std::size_t kSlotPitch   = (DiskKVStore::kSlotHeaderSize + kSlot + 4095) / 4096 * 4096;
constexpr std::uint32_t kSlotCount = 16;

void lru_and_persistence(const std::filesystem::path& root) {
    const DiskKVStore::Options options{.path       = (root / "main").string(),
                                       .slot_size  = kSlot,
                                       .max_slots  = kSlotCount,
                                       .verify_crc = true};
    const auto first  = page(kSlot, 7);
    const auto second = page(kSlot, 9);
    const auto late   = page(kSlot, 77);
    {
        DiskKVStore store(options);
        check(store.slot_count() == kSlotCount && store.live_slots() == 0, "fresh store geometry");
        check(store.upsert_page(id(1, 64), first) && store.upsert_page(id(2, 128), second),
              "two pages stored");
        check(store.upsert_page(id(1, 64), first) && store.live_slots() == 2,
              "a repeated identity is deduplicated");
        check(reads_back(store, id(1, 64), first), "a page reads back byte for byte");
        check(!store.contains(id(1, 64, 1)), "the tag is part of the identity");
        for (std::uint32_t i = 0; store.live_slots() < kSlotCount; ++i) {
            check(store.upsert_page(id(100 + i, 64 * (i + 3)), page(kSlot, 200 + i)), "fill");
        }
        store.touch(id(1, 64));
        std::vector<DiskKVIdentity> evicted;
        check(store.upsert_page(id(900, 4096), late, &evicted), "a full store accepts a page");
        check(evicted.size() == 1 && evicted.front() == id(2, 128),
              "the least recently used page is the one evicted");
        check(store.contains(id(1, 64)) && store.live_slots() == kSlotCount,
              "a touched page survives eviction");
        store.flush_index();
    }
    {
        DiskKVStore store(options);
        check(!store.index_rebuilt_from_scan(), "a published index is loaded directly");
        check(store.live_slots() == kSlotCount, "every page survives a reopen");
        check(reads_back(store, id(1, 64), first) && reads_back(store, id(900, 4096), late),
              "pages read back after a reopen");
        store.touch(id(1, 64));
        check(store.upsert_page(id(901, 8192), page(kSlot, 3)) && store.contains(id(1, 64)),
              "the LRU clock continues across a reopen");
        store.flush_index();
    }
    std::filesystem::remove((root / "main").string() + ".idx");
    {
        DiskKVStore store(options);
        check(store.index_rebuilt_from_scan() && store.live_slots() == kSlotCount,
              "a lost index is rebuilt from the slot headers");
    }
}

// The LRU order is rebuilt from the slot headers on open, from the published index or from a
// scan: after a restart the page used least recently before it is still the first one evicted.
void lru_order_survives_reopen(const std::filesystem::path& root) {
    for (const bool lose_index : {false, true}) {
        const DiskKVStore::Options options{
            .path      = (root / (lose_index ? "order-scan" : "order-index")).string(),
            .slot_size = kSlot,
            .max_slots = 3};
        {
            DiskKVStore store(options);
            check(store.upsert_page(id(1, 64), page(kSlot, 1)) &&
                      store.upsert_page(id(2, 128), page(kSlot, 2)) &&
                      store.upsert_page(id(3, 192), page(kSlot, 3)),
                  "order seeds stored");
            store.touch(id(1, 64));
            std::vector<std::byte> out(kSlot);
            check(store.read_page(id(2, 128), out), "a read refreshes the order");
            store.flush_index();
        }
        if (lose_index) { std::filesystem::remove(options.path + ".idx"); }
        DiskKVStore store(options);
        check(store.index_rebuilt_from_scan() == lose_index, "the reopen path is the one intended");
        std::vector<DiskKVIdentity> evicted;
        check(store.upsert_page(id(4, 256), page(kSlot, 4), &evicted) && evicted.size() == 1 &&
                  evicted.front() == id(3, 192),
              "after a reopen a full store evicts the page used least recently");
        evicted.clear();
        check(store.upsert_page(id(5, 320), page(kSlot, 5), &evicted) && evicted.size() == 1 &&
                  evicted.front() == id(1, 64),
              "eviction continues in the order the pages were last used");
    }
}

void corruption_is_a_miss(const std::filesystem::path& root) {
    const DiskKVStore::Options options{
        .path = (root / "one").string(), .slot_size = kSlot, .max_slots = 1};
    const auto bytes = page(kSlot, 5);
    {
        DiskKVStore store(options);
        check(store.upsert_page(id(7, 64), bytes), "single page stored");
        store.flush_index();
    }
    if (std::FILE* file = std::fopen(options.path.c_str(), "r+b")) {
        std::fseek(file, static_cast<long>(DiskKVStore::kSlotHeaderSize + 123), SEEK_SET);
        std::fputc(0x5A ^ static_cast<int>(bytes[123]), file);
        std::fclose(file);
    }
    {
        DiskKVStore store(options);
        std::vector<std::byte> out(kSlot);
        check(store.contains(id(7, 64)) && !store.read_page(id(7, 64), out),
              "a torn payload fails its CRC instead of being returned");
        check(store.evict(id(7, 64)) && !store.contains(id(7, 64)),
              "an unreadable page is dropped");
    }
    std::filesystem::remove(options.path + ".idx");
    {
        DiskKVStore store(options);
        check(store.live_slots() == 0, "the scan rebuild skips a torn slot");
    }
}

void capacity_sizing(const std::filesystem::path& root) {
    const DiskKVStore::Options options{
        .path = (root / "sized").string(), .slot_size = kSlot, .capacity_bytes = 5 * kSlotPitch};
    DiskKVStore store(options);
    check(store.slot_count() == 5, "the capacity in bytes sets the slot count");
}

// A reader copies outside the store lock, so eviction must never recycle a slot mid-read: every
// successful read returns the exact bytes of its identity.
void concurrent_reads_survive_eviction(const std::filesystem::path& root) {
    const DiskKVStore::Options options{
        .path = (root / "stress").string(), .slot_size = kSlot, .max_slots = 4};
    DiskKVStore store(options);
    const auto pinned = page(kSlot, 11);
    check(store.upsert_page(id(1, 64), pinned), "stress seed stored");
    std::atomic<bool> stop{false};
    std::atomic<std::uint32_t> mismatches{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&] {
            std::vector<std::byte> out(kSlot);
            while (!stop.load(std::memory_order_relaxed)) {
                if (store.read_page(id(1, 64), out) &&
                    std::memcmp(out.data(), pinned.data(), kSlot) != 0) {
                    mismatches.fetch_add(1);
                }
                store.touch(id(1, 64));
            }
        });
    }
    for (std::uint32_t i = 0; i < 400; ++i) {
        (void)store.upsert_page(id(1000 + i, 64 * (i + 2)), page(kSlot, i));
    }
    stop = true;
    for (std::thread& reader : readers) { reader.join(); }
    check(mismatches.load() == 0, "no read returns bytes of a recycled slot");
}

// A claimed read (the DirectStorage path) reads the payload from the file at the claimed offset,
// keeps the slot from eviction until released, and checks the bytes against the stored CRC.
void claimed_reads(const std::filesystem::path& root) {
    const DiskKVStore::Options options{
        .path = (root / "claimed").string(), .slot_size = kSlot, .max_slots = 2};
    DiskKVStore store(options);
    const auto first = page(kSlot, 21);
    check(store.upsert_page(id(1, 64), first) && store.upsert_page(id(2, 128), page(kSlot, 22)),
          "claimed-read seeds stored");
    const std::optional<DiskKVStore::ReadClaim> claim = store.claim_read(id(1, 64));
    check(claim.has_value() && !store.claim_read(id(3, 192)).has_value(),
          "a claim exists exactly for a stored page");
    if (!claim) { return; }

    std::vector<std::byte> bytes(kSlot);
    std::FILE* file = std::fopen(options.path.c_str(), "rb");
    const bool read = file != nullptr &&
                      std::fseek(file, static_cast<long>(claim->offset), SEEK_SET) == 0 &&
                      std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
    if (file != nullptr) { std::fclose(file); }
    check(read && bytes == first && store.claim_intact(*claim, bytes),
          "the claimed offset holds the page and its CRC matches");
    std::vector<std::byte> torn = bytes;
    torn[kSlot / 2] ^= std::byte{0x40};
    check(!store.claim_intact(*claim, torn), "a torn claimed read fails its CRC");
    check(!store.evict(id(1, 64)), "a claimed page is not evicted");
    (void)store.upsert_page(id(4, 256), page(kSlot, 24));
    check(store.contains(id(1, 64)) && !store.contains(id(2, 128)),
          "LRU eviction skips the claimed page");
    store.release_read(*claim, true);
    check(store.evict(id(1, 64)), "a released page can be evicted");
}

} // namespace

int main() {
    try {
        const std::filesystem::path root = fresh_directory();
        lru_and_persistence(root);
        lru_order_survives_reopen(root);
        corruption_is_a_miss(root);
        capacity_sizing(root);
        concurrent_reads_survive_eviction(root);
        claimed_reads(root);
        std::filesystem::remove_all(root);
    } catch (const std::exception& error) {
        std::cerr << "disk KV store test: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "disk KV store: LRU, persistence, corruption, sizing, concurrent and claimed "
                 "reads ok\n";
    return 0;
}
