#include "memory_test_support.hpp"
#include "kvmem/spill_io.hpp"
#include "kvmem/host_kv_storage.hpp"

#include <array>
#include <atomic>
#include <cstring>
#include <future>
#include <new>
#include <random>
#include <thread>
#ifdef __linux__
#include <csignal>
#include <sys/resource.h>
#endif

// Limit injection to this test's calling thread and the allocator operation.
// In particular, release must still succeed with every allocation disabled.
static thread_local int allocation_failure_after = -1;
void * operator new(std::size_t bytes) {
    if (allocation_failure_after == 0) throw std::bad_alloc();
    if (allocation_failure_after > 0) --allocation_failure_after;
    if (auto * memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void * operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void * memory) noexcept { std::free(memory); }
void operator delete[](void * memory) noexcept { std::free(memory); }
void operator delete(void * memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void * memory, std::size_t) noexcept { std::free(memory); }

template<class F> static void fails(F && f) {
    bool failed = false;
    try { f(); } catch (const std::exception &) { failed = true; }
    CHECK(failed);
}

static void file_tests(const std::filesystem::path & directory) {
    const std::array<uint8_t, 8> input{{12, 34, 56, 78, 90, 123, 234, 255}};
    std::array<uint8_t, 8> output{};
    SpillFile file(directory, 32);
    auto a = file.reserve(8), b = file.reserve(8), c = file.reserve(16);
    CHECK(file.charged_bytes() == 32);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { file.reserve(1); });
    fails([&] { a.read(0, output.data(), 8); }); // real short read, no initialized bytes
    a.write(0, input.data(), 8);
    a.read(0, output.data(), 8);
    CHECK(output == input);
    b.write(3, input.data(), 5);
    std::array<uint8_t, 5> slice{};
    b.read(3, slice.data(), 5);
    CHECK(std::memcmp(slice.data(), input.data(), 5) == 0);
    rejects(MemoryErrorCode::InvalidPlan, [&] { a.read(UINT64_MAX, output.data(), 1); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { a.write(7, input.data(), 2); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { a.write(0, nullptr, 1); });
    const auto stats = file.stats();
    CHECK(stats.read_bytes == 13 && stats.written_bytes == 13);
    CHECK(stats.reads == 2 && stats.writes == 2 && stats.errors == 1);
    auto pin = b;
    b.reset();
    CHECK(file.charged_bytes() == 32);
    pin.reset();
    CHECK(file.charged_bytes() == 24);
    a.reset(); c.reset();
    CHECK(file.charged_bytes() == 0);
    auto merged = file.reserve(32);
    CHECK(merged.offset() == 0 && merged.size() == 32);
    merged.reset();

    // Separate instances in one directory never share/truncate a backing file.
    SpillFile sibling(directory, 32);
    a = file.reserve(8); b = sibling.reserve(8);
    a.write(0, input.data(), 8);
    const std::array<uint8_t, 8> other{{99, 98, 97, 96, 95, 94, 93, 92}};
    b.write(0, other.data(), 8);
    a.read(0, output.data(), 8); CHECK(output == input);
    b.read(0, output.data(), 8); CHECK(output == other);
    auto survivor = [&] {
        SpillFile temporary(directory, 8);
        auto record = temporary.reserve(8);
        record.write(0, input.data(), 8);
        return record;
    }();
    survivor.read(0, output.data(), 8); CHECK(output == input);

    // A sparse high offset catches Windows CRT/32-bit-offset regressions without
    // allocating or streaming multiple GiB of process memory.
    SpillFile large(directory, (uint64_t(1) << 32) + 8);
    auto gap = large.reserve(uint64_t(1) << 32);
    auto high = large.reserve(8);
    CHECK(high.offset() == (uint64_t(1) << 32));
    high.write(0, input.data(), 8);
    high.read(0, output.data(), 8); CHECK(output == input);

    // Different file offsets are independent even with simultaneous operations.
    SpillFile parallel(directory, 8 * 4096);
    std::vector<SpillExtent> extents;
    for (unsigned i = 0; i < 8; ++i) extents.push_back(parallel.reserve(4096));
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < 8; ++i) threads.emplace_back([&, i] {
        std::array<uint8_t, 4096> bytes{}, copy{};
        for (unsigned round = 0; round < 20; ++round) {
            bytes.fill(static_cast<uint8_t>(i * 23 + round));
            extents[i].write(0, bytes.data(), bytes.size());
            extents[i].read(0, copy.data(), copy.size());
            CHECK(bytes == copy);
        }
    });
    for (auto & thread : threads) thread.join();
    CHECK(parallel.stats().read_bytes == 8 * 20 * 4096);
    CHECK(parallel.stats().written_bytes == 8 * 20 * 4096);
    CHECK(parallel.stats().reads == 160 && parallel.stats().writes == 160);
    CHECK(parallel.stats().errors == 0);

#ifdef __linux__
    // An actual OS write error, after a valid source is established. The low-level
    // file reports it; the owning transaction (tested separately) retains authority.
    struct rlimit previous{};
    CHECK(getrlimit(RLIMIT_FSIZE, &previous) == 0);
    auto handler = std::signal(SIGXFSZ, SIG_IGN);
    auto limit = previous; limit.rlim_cur = 0;
    CHECK(setrlimit(RLIMIT_FSIZE, &limit) == 0);
    bool failed = false;
    try { a.write(0, other.data(), 8); }
    catch (const MemoryError & error) { failed = error.code() == MemoryErrorCode::TransferFailed; }
    CHECK(setrlimit(RLIMIT_FSIZE, &previous) == 0);
    std::signal(SIGXFSZ, handler);
    CHECK(failed);
    a.read(0, output.data(), 8); CHECK(output == input);
#endif
}

static void batch_file_tests(const std::filesystem::path & directory) {
    SpillFile file(directory, 32), other(directory, 8);
    auto a = file.reserve(8), gap = file.reserve(8), b = file.reserve(8), c = file.reserve(8);
    std::array<uint8_t, 24> packed{};
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = uint8_t(i + 30);
    std::array<uint8_t, 8> sentinel{}, got{}; sentinel.fill(199);
    gap.write(0, sentinel.data(), sentinel.size());
    // A fragmented batch coalesces only b+c, never overwriting the live gap.
    file.write_batch({a, b, c}, packed.data(), packed.size());
    CHECK(file.stats().writes == 3 && file.stats().written_bytes == 32);
    a.read(0, got.data(), 8); CHECK(std::memcmp(got.data(), packed.data(), 8) == 0);
    b.read(0, got.data(), 8); CHECK(std::memcmp(got.data(), packed.data() + 8, 8) == 0);
    c.read(0, got.data(), 8); CHECK(std::memcmp(got.data(), packed.data() + 16, 8) == 0);
    gap.read(0, got.data(), 8); CHECK(got == sentinel);
    auto foreign = other.reserve(8);
    rejects(MemoryErrorCode::InvalidPlan, [&] { file.write_batch({a, foreign}, packed.data(), 16); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { file.write_batch({b, c}, packed.data(), 15); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { file.write_batch({b, c}, nullptr, 16); });
    CHECK(file.stats().writes == 3);
    b.reset(); CHECK(file.charged_bytes() == 24); // A batch does not pin siblings.
    auto reused = file.reserve(8); CHECK(reused.offset() == 16);
    reused.write(0, sentinel.data(), 8);
    c.read(0, got.data(), 8); CHECK(std::memcmp(got.data(), packed.data() + 16, 8) == 0);
}

static void free_index_tests(const std::filesystem::path & directory) {
    // Best fit, equal-size tie by address, splitting and fragmented failure.
    SpillFile file(directory, 64);
    auto a = file.reserve(12), guard1 = file.reserve(4), b = file.reserve(8);
    auto guard2 = file.reserve(4), c = file.reserve(8), guard3 = file.reserve(28);
    a.reset(); b.reset(); c.reset();
    CHECK(file.charged_bytes() == 36);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { file.reserve(13); });
    CHECK(file.charged_bytes() == 36);
    auto best1 = file.reserve(7), best2 = file.reserve(7), larger = file.reserve(9);
    CHECK(best1.offset() == 16 && best2.offset() == 28 && larger.offset() == 0);
    // Free adjacent ranges in an order exercising right, left and both merges.
    allocation_failure_after = 0;
    guard1.reset(); guard2.reset(); guard3.reset();
    best1.reset(); best2.reset(); larger.reset();
    allocation_failure_after = -1;
    CHECK(file.charged_bytes() == 0);
    auto whole = file.reserve(64); CHECK(whole.offset() == 0);
    whole.reset();

    for (bool split : {false, true}) {
        bool reached_success = false;
        unsigned failures = 0;
        for (int fail_at = 0; fail_at < 16 && !reached_success; ++fail_at) {
            SpillFile fault(directory, 64);
            auto left = fault.reserve(8), hole = fault.reserve(16);
            auto right = fault.reserve(8), tail = fault.reserve(32);
            std::array<uint8_t, 8> sentinel{}, readback{}; sentinel.fill(123);
            left.write(0, sentinel.data(), 8); right.write(0, sentinel.data(), 8);
            hole.reset();
            SpillExtent allocated;
            bool failed = false;
            allocation_failure_after = fail_at;
            try { allocated = fault.reserve(split ? 8 : 16); }
            catch (const std::bad_alloc &) { failed = true; }
            allocation_failure_after = -1;
            if (failed) {
                ++failures;
                CHECK(fault.charged_bytes() == 48);
                allocated = fault.reserve(16);
            } else reached_success = true;
            CHECK(allocated.offset() == 8);
            CHECK(fault.charged_bytes() == 48 + allocated.size());
            left.read(0, readback.data(), 8); CHECK(readback == sentinel);
            right.read(0, readback.data(), 8); CHECK(readback == sentinel);
            allocation_failure_after = 0;
            left.reset(); tail.reset(); allocated.reset(); right.reset();
            allocation_failure_after = -1;
            CHECK(fault.charged_bytes() == 0);
            auto recovered = fault.reserve(64); CHECK(recovered.offset() == 0);
        }
        CHECK(reached_success && failures > 0);
    }
}

static void queue_tests(const std::filesystem::path & directory) {
    HostKvStorage ram(16);
    StorageIoQueue queue(2, 16);
    std::promise<void> release, entered;
    auto gate = release.get_future().share();
    auto first = queue.submit(8, [&] { entered.set_value(); gate.wait(); });
    entered.get_future().wait();
    CHECK(!first.cancel()); // running I/O cannot free its source early
    auto record = ram.allocate(8);
    std::atomic<bool> called{false};
    auto cancelled = queue.submit(8, [lease = record.lease(), &called] { called = true; });
    record.reset();
    CHECK(ram.charged_bytes() == 8 && queue.pinned_bytes() == 16);
    CHECK(cancelled.cancel() && !cancelled.ready());
    rejects(MemoryErrorCode::Pending, [&] { queue.submit(1, [] {}); });
    release.set_value();
    first.wait();
    fails([&] { cancelled.wait(); });
    CHECK(cancelled.status() == StorageTaskStatus::Cancelled && !called);
    CHECK(ram.charged_bytes() == 0 && queue.pending_jobs() == 0 && queue.pinned_bytes() == 0);

    auto failure = queue.submit(16, [] { throw std::runtime_error("injected I/O failure"); });
    fails([&] { failure.wait(); });
    CHECK(failure.status() == StorageTaskStatus::Failed && queue.pinned_bytes() == 0);
    const auto epoch = queue.completion_epoch();
    auto recovery = queue.submit(1, [] {});
    queue.wait_for_completion(epoch, std::chrono::seconds(5));
    recovery.wait();
    CHECK(queue.completion_epoch() != epoch);

    // Destruction drains jobs and their file/RAM leases. A ticket outlives its queue.
    SpillFile file(directory, 16);
    std::array<uint8_t, 16> input{}; input.fill(71);
    std::array<uint8_t, 16> output{};
    auto extent = file.reserve(16);
    StorageTicket completion;
    {
        StorageIoQueue temporary(1, 16);
        auto bytes = ram.allocate(16, input.data());
        completion = temporary.submit(16, [destination = extent, source = bytes.lease()] {
            destination.write(0, source.data(), source.size());
        });
        bytes.reset();
    }
    completion.wait();
    CHECK(ram.charged_bytes() == 0);
    extent.read(0, output.data(), 16); CHECK(output == input);
    extent.reset(); CHECK(file.charged_bytes() == 0);
}

int main() {
    auto directory = std::filesystem::temp_directory_path() /
        std::filesystem::u8path("kvmem-spill-\xE5\xAD\x98\xE5\x82\xA8-" + std::to_string(std::random_device{}()));
    file_tests(directory);
    batch_file_tests(directory);
    free_index_tests(directory);
    queue_tests(directory);
    // Only an empty directory created by this test is removed. Live files must
    // have disappeared automatically when the final file/extent handle closed.
    CHECK(std::filesystem::is_empty(directory));
    CHECK(std::filesystem::remove(directory));
    std::puts("spill I/O: exact bytes, 64-bit offsets, quota reuse, parallel ranges, cancellation and drain passed");
}
