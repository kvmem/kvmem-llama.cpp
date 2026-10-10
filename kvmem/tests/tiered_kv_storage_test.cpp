#include "memory_test_support.hpp"
#include "kvmem/host_kv_storage.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/snapshot.hpp"
#include "kvmem/spill_io.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <future>
#include <random>
#include <thread>
#ifdef __linux__
#include <csignal>
#include <sys/resource.h>
#endif

static void pressure(const std::filesystem::path & directory) {
    auto file = std::make_shared<SpillFile>(directory, 512);
    HostKvStorage storage(48, file);
    std::vector<HostKvRecord> history;
    std::vector<std::array<uint8_t, 8>> expected(20);
    for (size_t i = 0; i < expected.size(); ++i) {
        for (size_t j = 0; j < 8; ++j) expected[i][j] = uint8_t(i * 9 + j);
        history.push_back(storage.allocate(8, expected[i].data()));
        history.back().seal();
        CHECK(storage.charged_bytes() <= 48);
        CHECK(storage.charged_bytes() + storage.disk_bytes() == (i + 1) * 8);
    }
    CHECK(storage.disk_bytes() == 112 && history[0].resident_bytes() == 0);
    for (size_t i = 0; i < history.size(); ++i) {
        std::array<uint8_t, 8> got{};
        history[i].read(0, got.data(), 8);
        CHECK(got == expected[i]);
    }
    CHECK(storage.charged_bytes() == 48); // Cold reads don't refill the whole RAM tier.
    const std::array<uint8_t, 3> replacement{{51, 73, 91}};
    history[0].write(2, replacement.data(), 3);
    std::memcpy(expected[0].data() + 2, replacement.data(), 3);
    CHECK(history[0].resident_bytes() == 8);
    storage.trim(0);
    CHECK(storage.charged_bytes() == 8); // The writable tail is not evictable.
    history[0].seal();
    storage.trim(0);
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == 160);
    for (size_t i = 0; i < history.size(); ++i) {
        std::array<uint8_t, 8> got{};
        history[i].read(0, got.data(), 8); CHECK(got == expected[i]);
    }
    rejects(MemoryErrorCode::InvalidPlan, [&] { history[0].data(); });

    auto lease = history[0].lease();
    CHECK(lease.size() == 8 && !lease.data());
    rejects(MemoryErrorCode::Pending, [&] { history[0].write(0, replacement.data(), 3); });
    history.clear();
    CHECK(storage.disk_bytes() == 8); // Last cold lease pins the extent/quota.
    std::array<uint8_t, 8> got{};
    lease.read(0, got.data(), 8); CHECK(got == expected[0]);
    auto copy = lease;
    lease = {};
    CHECK(storage.disk_bytes() == 8);
    copy = {};
    CHECK(storage.disk_bytes() == 0 && storage.charged_bytes() == 0);
}

static void limits(const std::filesystem::path & directory) {
    const std::array<uint8_t, 8> value{{0, 1, 2, 3, 4, 5, 6, 7}};
    HostKvStorage storage(8, std::make_shared<SpillFile>(directory, 8));
    auto first = storage.allocate(8, value.data());
    rejects(MemoryErrorCode::BudgetExceeded, [&] { storage.allocate(1); });
    first.seal();
    {
        auto lease = first.lease();
        auto copied = lease;
        HostKvLease moved = std::move(lease);
        copied = {};
        rejects(MemoryErrorCode::BudgetExceeded, [&] { storage.allocate(1); });
        storage.trim(0);
        CHECK(first.resident_bytes() == 8 && moved.data());
    }
    auto second = storage.allocate(8, value.data()); second.seal();
    CHECK(storage.charged_bytes() == 8 && storage.disk_bytes() == 8);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { storage.allocate(1); });
    std::array<uint8_t, 8> got{};
    first.read(0, got.data(), 8); CHECK(got == value);
    second.read(0, got.data(), 8); CHECK(got == value);
    // Capacity refusal did not discard the only copy or mark it falsely cold.
    CHECK(second.resident_bytes() == 8 && first.resident_bytes() == 0);
    second.reset();
    first.write(0, value.data(), 8);
    CHECK(storage.charged_bytes() == 8 && storage.disk_bytes() == 0);
    first.reset();
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == 0);

    // A larger eviction candidate must not hide a later candidate that fits D.
    HostKvStorage mixed(12, std::make_shared<SpillFile>(directory, 4));
    auto large = mixed.allocate(8); large.seal();
    auto small = mixed.allocate(4); small.seal();
    auto tail = mixed.allocate(4);
    CHECK(large.resident_bytes() == 8 && small.resident_bytes() == 0);
}

static void queued_read(const std::filesystem::path & directory) {
    HostKvStorage storage(8, std::make_shared<SpillFile>(directory, 8));
    const std::array<uint8_t, 8> value{{20, 21, 22, 23, 24, 25, 26, 27}};
    auto source = storage.allocate(8, value.data()); source.seal(); storage.trim(0);
    HostKvStorage scratch(8);
    auto destination = std::make_shared<HostKvRecord>(scratch.allocate(8));
    StorageIoQueue queue(1, 8);
    std::promise<void> release, entered;
    auto gate = release.get_future().share();
    auto ticket = queue.submit(8, [lease = source.lease(), destination, gate, &entered] {
        entered.set_value(); gate.wait();
        lease.read(0, destination->writable_data(), destination->size());
    });
    entered.get_future().wait();
    source.reset();
    CHECK(storage.disk_bytes() == 8 && !ticket.ready());
    release.set_value(); ticket.wait();
    CHECK(storage.disk_bytes() == 0);
    CHECK(std::memcmp(destination->data(), value.data(), 8) == 0);
    destination.reset(); CHECK(scratch.charged_bytes() == 0);
}

static void shared_owners(const std::filesystem::path & directory) {
    HostKvStorage storage(4 * 16, std::make_shared<SpillFile>(directory, 4 * 16 * 30));
    std::vector<std::thread> threads;
    for (unsigned owner = 0; owner < 4; ++owner) threads.emplace_back([&, owner] {
        std::vector<HostKvRecord> records;
        std::array<uint8_t, 16> value{}, got{};
        for (unsigned i = 0; i < 24; ++i) {
            value.fill(uint8_t(owner * 32 + i));
            records.push_back(storage.allocate(16, value.data()));
            records.back().seal();
        }
        for (unsigned i = 0; i < records.size(); ++i) {
            value.fill(uint8_t(owner * 32 + i));
            records[i].read(0, got.data(), got.size()); CHECK(got == value);
        }
    });
    for (auto & thread : threads) thread.join();
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == 0);
}

static void write_failure(const std::filesystem::path & directory) {
#ifdef __linux__
    HostKvStorage storage(8, std::make_shared<SpillFile>(directory, 8));
    const std::array<uint8_t, 8> value{{110, 111, 112, 113, 114, 115, 116, 117}};
    auto source = storage.allocate(8, value.data()); source.seal();
    struct rlimit previous{}; CHECK(getrlimit(RLIMIT_FSIZE, &previous) == 0);
    auto handler = std::signal(SIGXFSZ, SIG_IGN);
    auto limit = previous; limit.rlim_cur = 0;
    CHECK(setrlimit(RLIMIT_FSIZE, &limit) == 0);
    bool failed = false;
    try { storage.trim(0); }
    catch (const MemoryError & error) { failed = error.code() == MemoryErrorCode::TransferFailed; }
    CHECK(setrlimit(RLIMIT_FSIZE, &previous) == 0);
    std::signal(SIGXFSZ, handler);
    CHECK(failed && storage.charged_bytes() == 8 && storage.disk_bytes() == 0);
    std::array<uint8_t, 8> got{}; source.read(0, got.data(), 8); CHECK(got == value);
    storage.trim(0);
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == 8);
    source.read(0, got.data(), 8); CHECK(got == value);
#else
    (void)directory; // OS write-limit injection is Linux-only, not a Windows pass.
#endif
}
static void packed_store(const std::filesystem::path & directory) {
    auto storage = std::make_shared<HostKvStorage>(128, std::make_shared<SpillFile>(directory, 4096));
    RawKvStoreConfig cfg;
    cfg.n_layer = 1; cfg.n_embd_k = cfg.n_embd_v = 4; cfg.block_tokens = 4;
    cfg.k_gpu_row_bytes = cfg.v_gpu_row_bytes = 8; cfg.payload_storage = storage;
    {
        RawKvStore raw(cfg);
        std::array<uint8_t, 32> k{}, v{}, got{};
        std::array<float, 16> mean{}; mean.fill(2.0f);
        for (uint32_t block = 0; block < 20; ++block) {
            k.fill(uint8_t(block)); v.fill(uint8_t(block + 80));
            raw.write_layer_k_gpu(block * 4, 4, 0, k.data());
            raw.write_layer_v_gpu(block * 4, 4, 0, v.data());
            raw.write_layer_mean_k(block * 4, 4, 0, mean.data());
            CHECK(storage->charged_bytes() <= 128);
        }
        CHECK(storage->disk_bytes() > 0);
        for (uint32_t block = 0; block < 20; ++block) {
            k.fill(uint8_t(block)); v.fill(uint8_t(block + 80));
            CHECK(raw.copy_k_gpu(block, 0, got.data(), 4) && got == k);
            CHECK(raw.copy_v_gpu(block, 0, got.data(), 4) && got == v);
            std::array<float, 4> avg{}; raw.mean_k(block, 0, avg.data());
            CHECK(std::all_of(avg.begin(), avg.end(), [](float x) { return x == 2.0f; }));
        }
        // Rewrite only one row of a cold packed block; the remaining native bytes survive.
        k.fill(199); raw.write_layer_k_gpu(1, 1, 0, k.data());
        CHECK(raw.copy_k_gpu(0, 0, got.data(), 4));
        for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == (i >= 8 && i < 16 ? 199 : 0));
        std::vector<uint8_t> snapshot;
        SnapshotWriter writer([&](const void * data, size_t bytes) {
            const auto * p = static_cast<const uint8_t *>(data);
            snapshot.insert(snapshot.end(), p, p + bytes);
        });
        raw.snapshot_write(writer);
        size_t offset = 0;
        SnapshotReader reader([&](void * data, size_t bytes) {
            std::memcpy(data, snapshot.data() + offset, bytes); offset += bytes;
        }, snapshot.size());
        RawKvStore restored(cfg);
        restored.snapshot_read(reader, 20);
        CHECK(reader.remaining() == 0);
        for (uint32_t block = 0; block < 20; ++block) {
            std::array<uint8_t, 32> expected{};
            CHECK(raw.copy_k_gpu(block, 0, expected.data(), 4));
            CHECK(restored.copy_k_gpu(block, 0, got.data(), 4) && got == expected);
            CHECK(raw.copy_v_gpu(block, 0, expected.data(), 4));
            CHECK(restored.copy_v_gpu(block, 0, got.data(), 4) && got == expected);
        }
    }
    CHECK(storage->charged_bytes() == 0 && storage->disk_bytes() == 0);
    cfg.payload_storage = std::make_shared<HostKvStorage>(64, std::make_shared<SpillFile>(directory, 64));
    RawKvStore limited(cfg);
    std::array<uint8_t, 32> value{}, got{}; value.fill(77);
    for (uint32_t block = 0; block < 2; ++block) {
        limited.write_layer_k_gpu(block * 4, 4, 0, value.data());
        limited.write_layer_v_gpu(block * 4, 4, 0, value.data());
    }
    rejects(MemoryErrorCode::BudgetExceeded, [&] { limited.write_layer_k_gpu(8, 4, 0, value.data()); });
    for (uint32_t block = 0; block < 2; ++block) {
        CHECK(limited.copy_k_gpu(block, 0, got.data(), 4) && got == value);
        CHECK(limited.copy_v_gpu(block, 0, got.data(), 4) && got == value);
    }
}
static void deferred_source(const std::filesystem::path & directory) {
    HostKvStorage storage(64, std::make_shared<SpillFile>(directory, 256));
    std::array<uint8_t, 64> expected{}, got{}; expected.fill(91);
    auto owner = storage.allocate(expected.size(), expected.data());
    rejects(MemoryErrorCode::Pending, [&] { (void) owner.source(); });
    owner.seal();
    auto source = owner.source();
    auto copy = source;
    auto other = storage.allocate(64); other.seal();
    CHECK(owner.resident_bytes() == 0 && storage.disk_bytes() == 64);
    source.read(0, got.data(), got.size()); CHECK(got == expected);
    // A deferred reader does not pin H, but it forbids changing its version.
    rejects(MemoryErrorCode::Pending, [&] { owner.write(0, expected.data(), 1); });
    CHECK(storage.charged_bytes() == 64 && storage.disk_bytes() == 64);
    source = {}; owner.reset(); other.reset();
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == 64);
    StorageIoQueue queue(1, 64);
    auto ticket = queue.submit(64, [held = copy, &got] { held.read(0, got.data(), got.size()); });
    copy = {};
    ticket.wait();
    CHECK(got == expected && storage.disk_bytes() == 0);
}

static void batched_pressure(const std::filesystem::path & directory) {
    constexpr size_t bytes = 17 * 1024, count = 96;
    HostKvStorage storage(1024 * 1024, std::make_shared<SpillFile>(directory, 4 * 1024 * 1024));
    std::vector<HostKvRecord> records;
    std::vector<uint8_t> value(bytes), got(bytes);
    for (size_t i = 0; i < count; ++i) {
        std::fill(value.begin(), value.end(), uint8_t(i));
        records.push_back(storage.allocate(bytes, value.data()));
        records.back().seal();
        CHECK(storage.charged_bytes() <= storage.capacity_bytes());
        CHECK(storage.charged_bytes() + storage.disk_bytes() == (i + 1) * bytes);
    }
    CHECK(storage.disk_bytes() > 0);
    storage.trim(0);
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == count * bytes);
    CHECK(storage.io_stats().writes < count / 2);
    for (size_t i = 0; i < count; ++i) {
        records[i].read(0, got.data(), bytes);
        CHECK(std::all_of(got.begin(), got.end(), [i](uint8_t v) { return v == i; }));
    }
    auto held = records[1].lease();
    records.clear();
    CHECK(storage.disk_bytes() == bytes); // Independent quota after batched write.
    held.read(0, got.data(), bytes); CHECK(got.front() == 1 && got.back() == 1);
    held = {}; CHECK(storage.disk_bytes() == 0);

    // Optional headroom must not reject an otherwise legal full-H allocation,
    // even if D is too small to relocate the existing resident record.
    HostKvStorage tight(512 * 1024, std::make_shared<SpillFile>(directory, 1));
    auto first = tight.allocate(256 * 1024); first.seal();
    auto second = tight.allocate(256 * 1024);
    CHECK(tight.charged_bytes() == tight.capacity_bytes() && tight.disk_bytes() == 0);

    // Readers and mutable tails remain excluded from batch selection.
    HostKvStorage pinned(1024 * 1024, std::make_shared<SpillFile>(directory, 1024 * 1024));
    auto hot = pinned.allocate(bytes); hot.seal(); auto pin = hot.lease();
    auto tail = pinned.allocate(bytes);
    std::vector<HostKvRecord> cold;
    for (size_t i = 0; i < 20; ++i) { cold.push_back(pinned.allocate(bytes)); cold.back().seal(); }
    pinned.trim(0);
    CHECK(hot.resident_bytes() == bytes && tail.resident_bytes() == bytes);
    CHECK(pinned.charged_bytes() == 2 * bytes && pinned.disk_bytes() == 20 * bytes);
}

static void batch_write_failure(const std::filesystem::path & directory) {
#ifdef __linux__
    constexpr size_t bytes = 17 * 1024, count = 15;
    HostKvStorage storage(1024 * 1024, std::make_shared<SpillFile>(directory, 1024 * 1024));
    std::vector<HostKvRecord> records;
    std::vector<uint8_t> value(bytes, 73), got(bytes);
    for (size_t i = 0; i < count; ++i) { records.push_back(storage.allocate(bytes, value.data())); records.back().seal(); }
    struct rlimit previous{}; CHECK(getrlimit(RLIMIT_FSIZE, &previous) == 0);
    auto handler = std::signal(SIGXFSZ, SIG_IGN);
    auto limit = previous; limit.rlim_cur = 4096; // Partial physical batch write, then failure.
    CHECK(setrlimit(RLIMIT_FSIZE, &limit) == 0);
    bool failed = false;
    try { storage.trim(0); }
    catch (const MemoryError & error) { failed = error.code() == MemoryErrorCode::TransferFailed; }
    CHECK(setrlimit(RLIMIT_FSIZE, &previous) == 0); std::signal(SIGXFSZ, handler);
    CHECK(failed && storage.charged_bytes() == count * bytes && storage.disk_bytes() == 0);
    for (auto & record : records) { record.read(0, got.data(), bytes); CHECK(got == value); }
    storage.trim(0);
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == count * bytes);
    CHECK(storage.io_stats().writes == 1 && storage.io_stats().errors == 1);
    for (auto & record : records) { record.read(0, got.data(), bytes); CHECK(got == value); }
#else
    (void)directory;
#endif
}

static void batched_shared_owners(const std::filesystem::path & directory) {
    constexpr size_t bytes = 17 * 1024;
    HostKvStorage storage(1024 * 1024, std::make_shared<SpillFile>(directory, 8 * 1024 * 1024));
    std::vector<std::thread> threads;
    for (unsigned owner = 0; owner < 4; ++owner) threads.emplace_back([&, owner] {
        std::vector<HostKvRecord> records;
        std::vector<uint8_t> value(bytes), got(bytes);
        for (unsigned i = 0; i < 40; ++i) {
            std::fill(value.begin(), value.end(), uint8_t(owner * 40 + i));
            records.push_back(storage.allocate(bytes, value.data()));
            records.back().seal();
            CHECK(storage.charged_bytes() <= storage.capacity_bytes());
        }
        for (unsigned i = 0; i < records.size(); ++i) {
            records[i].read(0, got.data(), bytes);
            CHECK(std::all_of(got.begin(), got.end(), [=](uint8_t v) { return v == owner * 40 + i; }));
        }
    });
    for (auto & thread : threads) thread.join();
    CHECK(storage.charged_bytes() == 0 && storage.disk_bytes() == 0);
}

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("kvmem-tiered-test-" + std::to_string(std::random_device{}()));
    pressure(directory); limits(directory); queued_read(directory);
    shared_owners(directory); write_failure(directory);
    packed_store(directory);
    deferred_source(directory);
    batched_pressure(directory);
    batch_write_failure(directory);
    batched_shared_owners(directory);
    CHECK(std::filesystem::is_empty(directory));
    CHECK(std::filesystem::remove(directory));
    std::puts("tiered KV: bounded RAM, full payload spill, leased cold reads, shared owners and failure retention passed");
}
