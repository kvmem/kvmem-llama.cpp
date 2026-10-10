#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace kvmem {
namespace detail { struct SpillFileState; struct SpillExtentState; struct StorageTaskState; }
struct SpillIoStats {
    uint64_t read_bytes = 0, written_bytes = 0;
    uint64_t reads = 0, writes = 0, errors = 0;
};

// Physical storage only: reserving/writing an extent does not publish a valid KV
// version. Its owner must retain the source and publish only after the entire
// write succeeds. Copies of this handle pin both the file range and its quota.
class SpillExtent {
public:
    SpillExtent() = default;
    uint64_t size() const noexcept;
    uint64_t offset() const noexcept;
    explicit operator bool() const noexcept { return bool(state_); }
    void read(uint64_t offset, void * destination, size_t bytes) const;
    void write(uint64_t offset, const void * source, size_t bytes) const;
    void reset() noexcept { state_.reset(); }
private:
    friend class SpillFile;
    explicit SpillExtent(std::shared_ptr<detail::SpillExtentState> state);
    std::shared_ptr<detail::SpillExtentState> state_;
};

// One ephemeral file per instance; never opens/truncates another instance's file.
// Buffered positional I/O supports arbitrary native layouts on Windows/Linux.
// The quota covers reserved file ranges, not OS page cache or metadata. This is
// process-local backing storage, with no crash/restart durability guarantee.
class SpillFile {
public:
    SpillFile(const std::filesystem::path & directory, uint64_t capacity_bytes);
    SpillExtent reserve(uint64_t bytes);
    // Source concatenates the complete extents in order. Adjacent file ranges
    // share a write; gaps remain untouched. Owners publish only after success.
    // Every extent retains its independent lifetime and quota.
    void write_batch(const std::vector<SpillExtent> & extents, const void * source, size_t bytes);
    uint64_t charged_bytes() const;
    uint64_t capacity_bytes() const noexcept;
    SpillIoStats stats() const noexcept;
private:
    std::shared_ptr<detail::SpillFileState> state_;
};

enum class StorageTaskStatus { Pending, Complete, Failed, Cancelled };
class StorageTicket {
public:
    StorageTicket() = default;
    explicit operator bool() const noexcept { return bool(state_); }
    StorageTaskStatus status() const noexcept;
    bool ready() const noexcept { return status() != StorageTaskStatus::Pending; }
    // Only queued work can be cancelled. Completion includes releasing captured
    // buffers, even for cancellation/failure. Started I/O must finish before drain.
    bool cancel() const;
    void wait() const; // rethrows the task failure/cancellation
private:
    friend class StorageIoQueue;
    explicit StorageTicket(std::shared_ptr<detail::StorageTaskState> state);
    std::shared_ptr<detail::StorageTaskState> state_;
};

// Jobs own their leases and perform CPU/file operations only, never GPU work or
// engine publication. Running jobs count toward both limits. Queue-full reports
// Pending so engine adapters can yield instead of spinning or exceeding budgets.
// Jobs must not submit back into or own this queue. Destruction drains and joins.
class StorageIoQueue {
public:
    StorageIoQueue(size_t max_jobs, uint64_t max_pinned_bytes, size_t workers = 1);
    ~StorageIoQueue();
    StorageIoQueue(const StorageIoQueue &) = delete;
    StorageIoQueue & operator=(const StorageIoQueue &) = delete;
    StorageTicket submit(uint64_t pinned_bytes, std::function<void()> work);
    uint64_t completion_epoch() const;
    // For a scheduler with no runnable requests. The worker never calls it back.
    void wait_for_completion(uint64_t epoch, std::chrono::milliseconds timeout) const;
    size_t pending_jobs() const;
    uint64_t pinned_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kvmem
