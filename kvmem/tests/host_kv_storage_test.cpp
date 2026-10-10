#include "memory_test_support.hpp"
#include "kvmem/host_kv_storage.hpp"
#include <array>
#include <cstring>
#include <thread>

int main() {
    HostKvStorage storage(16);
    std::array<uint8_t, 8> bytes{{1, 2, 3, 4, 5, 6, 7, 8}}, copy{};
    {
        auto reserved = storage.reserve(16);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { storage.reserve(1); });
        auto first = reserved.allocate(8, bytes.data());
        CHECK(reserved.remaining() == 8 && storage.charged_bytes() == 16);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { reserved.allocate(9); });
        CHECK(reserved.remaining() == 8);
        auto second = reserved.allocate(8);
        second.read(0, copy.data(), copy.size());
        CHECK((copy == std::array<uint8_t, 8>{}));
        {
            auto lease = first.lease();
            rejects(MemoryErrorCode::Pending, [&] { first.write(0, bytes.data(), 1); });
            auto moved = std::move(first);
            CHECK(first.empty());
            moved.reset();
            CHECK(storage.charged_bytes() == 16);
            CHECK(lease.size() == 8 && std::memcmp(lease.data(), bytes.data(), 8) == 0);
        }
        CHECK(storage.charged_bytes() == 8);
        second.write(2, bytes.data(), 4);
        second.read(0, copy.data(), 8);
        CHECK((copy == std::array<uint8_t, 8>{{0, 0, 1, 2, 3, 4, 0, 0}}));
        rejects(MemoryErrorCode::InvalidPlan, [&] { second.read(SIZE_MAX, copy.data(), 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { second.write(7, bytes.data(), 2); });
    }
    CHECK(storage.charged_bytes() == 0);
    {
        auto first = storage.reserve(8), second = storage.reserve(8);
        first = std::move(second);
        CHECK(storage.charged_bytes() == 8 && second.remaining() == 0);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { second.allocate(1); });
    }
    CHECK(storage.charged_bytes() == 0);
    // Separate engine owners share one hard bound, while record contents are private.
    HostKvStorage shared(8 * 4);
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) workers.emplace_back([&] {
        for (int j = 0; j < 1000; ++j) {
            auto record = shared.allocate(8, bytes.data());
            CHECK(record.lease().size() == 8);
        }
    });
    for (auto & worker : workers) worker.join();
    CHECK(shared.charged_bytes() == 0);
    // Leases also outlive the storage facade, as required by cancellation cleanup.
    auto survivor = [] {
        HostKvStorage temporary(8);
        auto record = temporary.allocate(8);
        return record.lease();
    }();
    CHECK(survivor.size() == 8 && survivor.data()[0] == 0);
    std::puts("host KV storage: byte preservation, reservations, leases and shared limits passed");
}
