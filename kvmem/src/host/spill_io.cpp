#include "kvmem/spill_io.hpp"
#include "kvmem/memory_contract.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace kvmem {
namespace detail {
struct SpillFileState {
    using FreeKey = std::pair<uint64_t, uint64_t>; // (bytes, offset), best fit
    struct FreeLess {
        bool operator()(const FreeKey & a, const FreeKey & b) const noexcept {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        }
    };
    using FreeRanges = std::set<FreeKey, FreeLess>;
    struct Range {
        uint64_t bytes;
        bool used;
        // An occupied range owns its extracted index node. Returning it to the
        // free index never allocates, including during noexcept destruction.
        FreeRanges::node_type spare;
    };
    using Ranges = std::map<uint64_t, Range>;
    std::mutex mutex;
    const uint64_t capacity;
    uint64_t charged = 0;
    std::atomic<uint64_t> read_bytes{0}, written_bytes{0}, reads{0}, writes{0}, errors{0};
    Ranges ranges;
    FreeRanges free_ranges;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    explicit SpillFileState(uint64_t capacity) : capacity(capacity) {
        ranges.emplace(0, Range{capacity, false, {}});
        free_ranges.emplace(capacity, 0);
    }
    ~SpillFileState() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (fd >= 0) ::close(fd);
#endif
    }
    void release(Ranges::iterator range) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        charged -= range->second.bytes;
        range->second.used = false;
        auto node = std::move(range->second.spare);
        assert(!node.empty());
        // Remove free neighbors from both indexes. The released range's node
        // supplies the merged entry; no allocation or throwing comparison.
        // Never merge an outstanding extent (its address iterator stays valid).
        auto next = std::next(range);
        if (next != ranges.end() && !next->second.used) {
            free_ranges.erase(FreeKey{next->second.bytes, next->first});
            range->second.bytes += next->second.bytes;
            ranges.erase(next);
        }
        if (range != ranges.begin()) {
            auto previous = std::prev(range);
            if (!previous->second.used) {
                free_ranges.erase(FreeKey{previous->second.bytes, previous->first});
                previous->second.bytes += range->second.bytes;
                ranges.erase(range);
                range = previous;
            }
        }
        node.value() = FreeKey{range->second.bytes, range->first};
        const auto inserted = free_ranges.insert(std::move(node));
        assert(inserted.inserted);
        (void) inserted;
    }
    void transfer(bool writing, uint64_t offset, void * bytes, size_t length) const {
        auto * cursor = static_cast<uint8_t *>(bytes);
#ifdef _WIN32
        struct Event {
            HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            ~Event() { if (handle) CloseHandle(handle); }
        } event;
        if (!event.handle) throw std::system_error(GetLastError(), std::system_category(), "KV I/O event");
#endif
        while (length) {
            const size_t count = std::min<size_t>(length, 1024 * 1024);
#ifdef _WIN32
            OVERLAPPED operation{};
            operation.Offset = static_cast<DWORD>(offset);
            operation.OffsetHigh = static_cast<DWORD>(offset >> 32);
            operation.hEvent = event.handle;
            ResetEvent(event.handle);
            DWORD transferred = 0;
            const BOOL started = writing
                ? WriteFile(handle, cursor, static_cast<DWORD>(count), &transferred, &operation)
                : ReadFile(handle, cursor, static_cast<DWORD>(count), &transferred, &operation);
            if (!started) {
                const auto error = GetLastError();
                if (error != ERROR_IO_PENDING)
                    throw std::system_error(error, std::system_category(), writing ? "KV write" : "KV read");
                if (!GetOverlappedResult(handle, &operation, &transferred, TRUE))
                    throw std::system_error(GetLastError(), std::system_category(), writing ? "KV write" : "KV read");
            }
            const size_t done = transferred;
#else
            static_assert(sizeof(off_t) >= 8, "KV spill requires 64-bit file offsets");
            const auto transferred = writing
                ? ::pwrite(fd, cursor, count, static_cast<off_t>(offset))
                : ::pread(fd, cursor, count, static_cast<off_t>(offset));
            if (transferred < 0) {
                if (errno == EINTR) continue;
                throw std::system_error(errno, std::generic_category(), writing ? "KV write" : "KV read");
            }
            const size_t done = static_cast<size_t>(transferred);
#endif
            if (!done) throw MemoryError(MemoryErrorCode::TransferFailed,
                writing ? "KV write made no progress" : "truncated KV spill extent");
            cursor += done; offset += done; length -= done;
        }
    }
};
struct SpillExtentState {
    std::shared_ptr<SpillFileState> file;
    SpillFileState::Ranges::iterator range;
    uint64_t offset = 0, size = 0;
    ~SpillExtentState() { if (size) file->release(range); }
};
struct StorageTaskState {
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<StorageTaskStatus> status{StorageTaskStatus::Pending};
    bool running = false, cancel = false;
    std::exception_ptr error;
};
} // namespace detail

SpillExtent::SpillExtent(std::shared_ptr<detail::SpillExtentState> state) : state_(std::move(state)) {}
uint64_t SpillExtent::size() const noexcept { return state_ ? state_->size : 0; }
uint64_t SpillExtent::offset() const noexcept { return state_ ? state_->offset : 0; }
namespace {
void check_extent(uint64_t size, uint64_t offset, const void * data, size_t bytes) {
    if (offset > size || bytes > size - offset || (bytes && !data))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "KV spill byte range is invalid");
}
void write_range(const std::shared_ptr<detail::SpillFileState> & file,
                 uint64_t offset, const void * source, size_t bytes) {
    try { file->transfer(true, offset, const_cast<void *>(source), bytes); }
    catch (const std::system_error & error) {
        file->errors.fetch_add(1, std::memory_order_relaxed);
        throw MemoryError(MemoryErrorCode::TransferFailed, error.what());
    }
    catch (...) { file->errors.fetch_add(1, std::memory_order_relaxed); throw; }
    file->written_bytes.fetch_add(bytes, std::memory_order_relaxed);
    file->writes.fetch_add(1, std::memory_order_relaxed);
}
} // namespace
void SpillExtent::read(uint64_t offset, void * destination, size_t bytes) const {
    check_extent(size(), offset, destination, bytes);
    if (!bytes) return;
    try { state_->file->transfer(false, state_->offset + offset, destination, bytes); }
    catch (const std::system_error & error) {
        state_->file->errors.fetch_add(1, std::memory_order_relaxed);
        throw MemoryError(MemoryErrorCode::TransferFailed, error.what());
    }
    catch (...) { state_->file->errors.fetch_add(1, std::memory_order_relaxed); throw; }
    state_->file->read_bytes.fetch_add(bytes, std::memory_order_relaxed);
    state_->file->reads.fetch_add(1, std::memory_order_relaxed);
}
void SpillExtent::write(uint64_t offset, const void * source, size_t bytes) const {
    check_extent(size(), offset, source, bytes);
    if (!bytes) return;
    write_range(state_->file, state_->offset + offset, source, bytes);
}
void SpillFile::write_batch(const std::vector<SpillExtent> & extents, const void * source, size_t bytes) {
    size_t total = 0;
    // Validate every destination before issuing any write, including file identity.
    for (const auto & extent : extents) {
        if (!extent.state_ || extent.state_->file != state_ || extent.size() > bytes - total)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid KV spill write batch");
        total += static_cast<size_t>(extent.size());
    }
    if (total != bytes || (bytes && !source))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid KV spill batch source");
    const auto * cursor = static_cast<const uint8_t *>(source);
    for (size_t first = 0; first < extents.size();) {
        size_t end = first + 1;
        size_t length = static_cast<size_t>(extents[first].size());
        while (end < extents.size() && extents[end].offset() == extents[first].offset() + length) {
            length += static_cast<size_t>(extents[end++].size());
        }
        write_range(state_, extents[first].offset(), cursor, length);
        cursor += length;
        first = end;
    }
}
SpillFile::SpillFile(const std::filesystem::path & directory, uint64_t capacity_bytes)
    : state_(std::make_shared<detail::SpillFileState>(capacity_bytes)) {
    if (!capacity_bytes || capacity_bytes > uint64_t(INT64_MAX) || directory.empty())
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid KV spill file capacity/directory");
    std::filesystem::create_directories(directory);
    std::random_device random;
    for (unsigned attempt = 0; attempt < 32; ++attempt) {
        const auto path = directory / ("kvmem-spill-" + std::to_string(random()) + "-" +
                                      std::to_string(random()) + ".tmp");
#ifdef _WIN32
        state_->handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED |
            FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (state_->handle != INVALID_HANDLE_VALUE) {
            // Recycled/gapped ranges need not consume physical space until written.
            // Sparse support is an optimization, not a different read/write contract.
            OVERLAPPED operation{};
            operation.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (operation.hEvent) {
                DWORD count = 0;
                if (!DeviceIoControl(state_->handle, FSCTL_SET_SPARSE, nullptr, 0,
                        nullptr, 0, &count, &operation) && GetLastError() == ERROR_IO_PENDING)
                    GetOverlappedResult(state_->handle, &operation, &count, TRUE);
                CloseHandle(operation.hEvent);
            }
            return;
        }
        const auto error = GetLastError();
        if (error != ERROR_FILE_EXISTS && error != ERROR_ALREADY_EXISTS)
            throw std::system_error(error, std::system_category(), "create KV spill file");
#else
        state_->fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (state_->fd >= 0) {
            if (::unlink(path.c_str()))
                throw std::system_error(errno, std::generic_category(), "unlink KV spill file");
            return;
        }
        if (errno != EEXIST) throw std::system_error(errno, std::generic_category(), "create KV spill file");
#endif
    }
    throw MemoryError(MemoryErrorCode::TransferFailed, "cannot create a unique KV spill file");
}
SpillExtent SpillFile::reserve(uint64_t bytes) {
    if (!bytes) return {};
    // Allocate the handle before changing the range map (strong failure guarantee).
    auto extent = std::make_shared<detail::SpillExtentState>();
    extent->file = state_;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (bytes > state_->capacity - state_->charged)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "KV SSD capacity exhausted");
    const auto free = state_->free_ranges.lower_bound({bytes, 0});
    if (free == state_->free_ranges.end())
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "KV SSD has no contiguous free extent");
    const auto it = state_->ranges.find(free->second);
    assert(it != state_->ranges.end() && !it->second.used && it->second.bytes == free->first);
    if (it->second.bytes > bytes) {
        const auto remaining = it->second.bytes - bytes;
        const auto tail = state_->ranges.emplace(it->first + bytes,
            detail::SpillFileState::Range{remaining, false, {}}).first;
        try { state_->free_ranges.emplace(remaining, tail->first); }
        catch (...) { state_->ranges.erase(tail); throw; }
    }
    // All allocations succeeded. Extract, publish and charge without throwing;
    // a split allocation failure above leaves the original free range intact.
    it->second.spare = state_->free_ranges.extract(free);
    it->second.bytes = bytes;
    it->second.used = true;
    state_->charged += bytes;
    extent->range = it; extent->offset = it->first; extent->size = bytes;
    return SpillExtent(std::move(extent));
}
uint64_t SpillFile::charged_bytes() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->charged;
}
uint64_t SpillFile::capacity_bytes() const noexcept { return state_->capacity; }
SpillIoStats SpillFile::stats() const noexcept {
    return {state_->read_bytes.load(std::memory_order_relaxed), state_->written_bytes.load(std::memory_order_relaxed),
        state_->reads.load(std::memory_order_relaxed), state_->writes.load(std::memory_order_relaxed),
        state_->errors.load(std::memory_order_relaxed)};
}

StorageTicket::StorageTicket(std::shared_ptr<detail::StorageTaskState> state) : state_(std::move(state)) {}
StorageTaskStatus StorageTicket::status() const noexcept {
    return state_ ? state_->status.load(std::memory_order_acquire) : StorageTaskStatus::Complete;
}
bool StorageTicket::cancel() const {
    if (!state_) return false;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->running || ready()) return false;
    state_->cancel = true;
    return true;
}
void StorageTicket::wait() const {
    if (!state_) return;
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->changed.wait(lock, [&] { return ready(); });
    if (state_->error) std::rethrow_exception(state_->error);
    if (status() == StorageTaskStatus::Cancelled)
        throw MemoryError(MemoryErrorCode::TransferFailed, "KV storage operation cancelled");
}
struct StorageIoQueue::Impl {
    struct Job {
        std::shared_ptr<detail::StorageTaskState> state;
        std::function<void()> work;
        uint64_t bytes;
    };
    mutable std::mutex mutex;
    mutable std::condition_variable completed;
    std::condition_variable available;
    std::deque<Job> jobs;
    std::vector<std::thread> workers;
    size_t outstanding = 0;
    uint64_t bytes = 0, epoch = 0;
    const size_t max_jobs;
    const uint64_t max_bytes;
    bool stopping = false;
    Impl(size_t count, uint64_t bytes) : max_jobs(count), max_bytes(bytes) {}
    ~Impl() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        available.notify_all();
        for (auto & worker : workers) if (worker.joinable()) worker.join();
    }
    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                available.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (jobs.empty()) return;
                job = std::move(jobs.front()); jobs.pop_front();
            }
            bool cancelled;
            {
                std::lock_guard<std::mutex> lock(job.state->mutex);
                cancelled = job.state->cancel;
                job.state->running = true;
            }
            std::exception_ptr error;
            if (!cancelled) {
                try { job.work(); } catch (...) { error = std::current_exception(); }
            }
            job.work = {}; // Release captured leases before signalling completion.
            {
                std::lock_guard<std::mutex> lock(mutex);
                --outstanding; bytes -= job.bytes;
            }
            {
                std::lock_guard<std::mutex> lock(job.state->mutex);
                job.state->error = error;
                job.state->status.store(cancelled ? StorageTaskStatus::Cancelled :
                    error ? StorageTaskStatus::Failed : StorageTaskStatus::Complete, std::memory_order_release);
            }
            job.state->changed.notify_all();
            { std::lock_guard<std::mutex> lock(mutex); ++epoch; }
            completed.notify_all();
        }
    }
};
StorageIoQueue::StorageIoQueue(size_t max_jobs, uint64_t max_pinned_bytes, size_t workers) {
    if (!max_jobs || !max_pinned_bytes || !workers || workers > 8)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid KV I/O queue limits");
    impl_ = std::make_unique<Impl>(max_jobs, max_pinned_bytes);
    impl_->workers.reserve(workers);
    for (size_t i = 0; i < workers; ++i) impl_->workers.emplace_back([this] { impl_->run(); });
}
StorageIoQueue::~StorageIoQueue() = default;
StorageTicket StorageIoQueue::submit(uint64_t pinned_bytes, std::function<void()> work) {
    if (!work) throw MemoryError(MemoryErrorCode::InvalidPlan, "empty KV I/O job");
    auto state = std::make_shared<detail::StorageTaskState>();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping || impl_->outstanding >= impl_->max_jobs ||
        pinned_bytes > impl_->max_bytes - impl_->bytes)
        throw MemoryError(MemoryErrorCode::Pending, "KV I/O queue is full");
    impl_->jobs.push_back({state, std::move(work), pinned_bytes});
    ++impl_->outstanding; impl_->bytes += pinned_bytes;
    impl_->available.notify_one();
    return StorageTicket(std::move(state));
}
uint64_t StorageIoQueue::completion_epoch() const {
    std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->epoch;
}
void StorageIoQueue::wait_for_completion(uint64_t epoch, std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->completed.wait_for(lock, timeout, [&] { return impl_->epoch != epoch; });
}
size_t StorageIoQueue::pending_jobs() const {
    std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->outstanding;
}
uint64_t StorageIoQueue::pinned_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->bytes;
}
} // namespace kvmem
