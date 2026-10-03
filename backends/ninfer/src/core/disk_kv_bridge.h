#pragma once

// Engine-facing facade over the disk (L3) page stores: one store per KV family, a bounded queue of
// spill jobs drained by writer threads, and synchronous reads for restores.
//
// Identity model: the engine keeps a rolling 128-bit digest per token frontier, a pure function of
// the first F tokens. A KV page covering [p * 64, F) is keyed by digest(F), which the live sequence
// can compute when it spills and an incoming request computes from its own prompt, with no other
// persistent metadata; the same content in two sessions dedupes to one page. The tag separates
// engine configurations (speculative backend, proposal head, KV dtype) whose bytes differ for the
// same tokens. A miss of any kind (never spilled, evicted, corrupt) makes the engine recompute.

#include "core/direct_storage_reader.h"
#include "core/disk_kv_store.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ninfer {

enum class DiskKVKind : std::uint8_t {
    MainKV     = 0,
    BackendKV  = 1,
    StateImage = 2,
};

// Completion of one asynchronous probe or write. Written means readable, not durably indexed.
enum class SpillStatus : std::uint8_t { Pending, Present, Missing, Written, Failed };

struct SpillCompletion {
    std::atomic<SpillStatus> status{SpillStatus::Pending};
};

using SpillTicket = std::shared_ptr<SpillCompletion>;

struct DiskKVBridgeStats {
    std::uint64_t spills         = 0;
    std::uint64_t spill_bytes    = 0;
    std::uint64_t spill_dups     = 0;
    std::uint64_t restores       = 0;
    std::uint64_t restore_bytes  = 0;
    std::uint64_t restore_misses = 0;
    std::uint64_t evicted_pages  = 0;
    std::uint64_t queue_drops    = 0;
};

class DiskKVBridge {
public:
    struct Options {
        std::string base_path;
        std::size_t main_page_stride    = 0;
        std::size_t backend_page_stride = 0; // zero disables the family
        std::size_t state_page_stride   = 0; // zero disables the family
        // Total budget, split 65/25/10 across main/backend/state (85/15 without a backend).
        std::size_t capacity_bytes = 0;
        bool verify_crc            = true;
        // Restore consecutive pages through DirectStorage (Windows builds with
        // NINFER_DIRECTSTORAGE); a failed batch falls back to mapped reads.
        bool direct_storage = false;
    };

    explicit DiskKVBridge(Options options);
    ~DiskKVBridge();

    DiskKVBridge(const DiskKVBridge&)            = delete;
    DiskKVBridge& operator=(const DiskKVBridge&) = delete;

    // Copies the page into the queue, waiting up to `timeout` for room: a caller writing a prefix
    // chain must not leave a hole, since a restore stops at the first missing page. True when
    // queued or already stored.
    bool spill_page_wait(const DiskKVIdentity& id, DiskKVKind kind,
                         std::span<const std::byte> bytes, std::chrono::milliseconds timeout);

    // Writes on the calling thread without a queued copy, for large single blobs (state images).
    bool spill_page_sync(const DiskKVIdentity& id, DiskKVKind kind,
                         std::span<const std::byte> bytes);

    // Non-blocking probe and write for the resumable owner spill. A null ticket means the queue is
    // busy; retry later. Submitted bytes are borrowed and must stay alive and unchanged until the
    // ticket leaves Pending, cancellation included.
    [[nodiscard]] SpillTicket try_probe(const DiskKVIdentity& id, DiskKVKind kind);
    [[nodiscard]] SpillTicket try_submit(const DiskKVIdentity& id, DiskKVKind kind,
                                         std::span<const std::byte> bytes);

    [[nodiscard]] static SpillStatus poll(const SpillTicket& ticket) noexcept {
        return ticket->status.load(std::memory_order_acquire);
    }

    // Waits until every queued job has been applied.
    void wait_idle();

    // Reads one page into `destination` (the family stride) on the calling thread.
    [[nodiscard]] bool restore_page(const DiskKVIdentity& id, DiskKVKind kind,
                                    std::span<std::byte> destination) const;

    // Reads consecutive pages into `destination` (ids.size() strides) with several readers: cold
    // reads scale with the number in flight. False as soon as any page is missing or corrupt.
    [[nodiscard]] bool restore_pages(std::span<const DiskKVIdentity> ids, DiskKVKind kind,
                                     std::span<std::byte> destination) const;

    // Drops a page that is indexed but unreadable, so later probes stop selecting it.
    bool drop_page(const DiskKVIdentity& id, DiskKVKind kind);

    [[nodiscard]] bool contains(const DiskKVIdentity& id, DiskKVKind kind) const;

    // Frontiers of every stored state image: the candidates a restore can resume from.
    [[nodiscard]] std::vector<std::uint32_t> live_state_frontiers() const;

    [[nodiscard]] DiskKVBridgeStats stats() const;
    [[nodiscard]] std::uint32_t slot_count(DiskKVKind kind) const;

private:
    struct Family {
        std::unique_ptr<DiskKVStore> store;
        std::size_t stride = 0;
    };

    struct Job {
        DiskKVIdentity id;
        DiskKVKind kind = DiskKVKind::MainKV;
        std::vector<std::byte> bytes;
        std::span<const std::byte> borrowed;
        SpillTicket completion;
        bool probe = false;
    };

    void worker_loop();
    void apply(Job& job);
    void flush_indexes();

    [[nodiscard]] const Family& family(DiskKVKind kind) const {
        return families_[static_cast<std::size_t>(kind)];
    }

    [[nodiscard]] Family& family(DiskKVKind kind) {
        return families_[static_cast<std::size_t>(kind)];
    }

    // The writers keep this many jobs in flight; a full owner spill waits or retries instead.
    static constexpr std::size_t kQueueCapacity = 512;
    // Several streams are needed to reach NVMe bandwidth; the store serializes only bookkeeping.
    static constexpr std::size_t kWorkerCount = 8;
    // Index publication is batched: slots are self-describing and a lost index is rebuilt.
    static constexpr std::uint32_t kFlushInterval = 64;

    // Reads a batch through DirectStorage; false leaves the caller to the mapped path.
    [[nodiscard]] bool restore_pages_direct(const Family& target,
                                            std::span<const DiskKVIdentity> ids,
                                            std::span<std::byte> destination) const;

    Options options_;
    Family families_[3];
    std::unique_ptr<DirectStorageReader> direct_reader_;
    mutable std::mutex stats_mutex_;
    mutable DiskKVBridgeStats stats_;
    std::atomic<std::uint32_t> flush_pending_{0};
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<Job> queue_;
    std::size_t in_flight_ = 0;
    bool quit_             = false;
    std::vector<std::thread> workers_;
};

} // namespace ninfer
