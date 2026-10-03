#include "core/disk_kv_bridge.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer {
namespace {

const char* family_name(DiskKVKind kind) noexcept {
    switch (kind) {
    case DiskKVKind::MainKV:
        return "main";
    case DiskKVKind::BackendKV:
        return "backend";
    case DiskKVKind::StateImage:
        return "state";
    }
    return "unknown";
}

} // namespace

DiskKVBridge::DiskKVBridge(Options options) : options_(std::move(options)) {
    if (options_.base_path.empty() || options_.capacity_bytes == 0 ||
        options_.main_page_stride == 0) {
        throw std::invalid_argument("disk KV tier needs a path, a budget, and a main page stride");
    }
    std::error_code error;
    std::filesystem::create_directories(options_.base_path, error);
    if (error) {
        throw std::runtime_error(options_.base_path +
                                 ": cannot create the disk KV directory: " + error.message());
    }
    const bool backend              = options_.backend_page_stride != 0;
    const std::size_t main_share    = backend ? 65 : 85;
    const std::size_t backend_share = backend ? 25 : 0;
    const std::size_t state_share   = 100 - main_share - backend_share;
    const auto open                 = [&](DiskKVKind kind, std::size_t stride, std::size_t share) {
        Family& target           = family(kind);
        target.stride            = stride;
        const std::size_t budget = options_.capacity_bytes / 100 * share;
        if (stride == 0 || budget < stride) { return; }
        DiskKVStore::Options store;
        store.path                = options_.base_path + "/diskkv_" + family_name(kind);
        store.slot_size           = stride;
        store.capacity_bytes      = budget;
        store.verify_crc          = options_.verify_crc;
        store.defer_index_updates = true;
        target.store              = std::make_unique<DiskKVStore>(std::move(store));
    };
    open(DiskKVKind::MainKV, options_.main_page_stride, main_share);
    open(DiskKVKind::BackendKV, options_.backend_page_stride, backend_share);
    open(DiskKVKind::StateImage, options_.state_page_stride, state_share);
    if (family(DiskKVKind::MainKV).store == nullptr) {
        throw std::invalid_argument("disk KV budget does not hold one main KV page");
    }
    if (options_.direct_storage) { direct_reader_ = DirectStorageReader::open(); }
    workers_.reserve(kWorkerCount);
    for (std::size_t i = 0; i < kWorkerCount; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

DiskKVBridge::~DiskKVBridge() {
    {
        std::lock_guard lock(queue_mutex_);
        quit_ = true;
    }
    queue_cv_.notify_all();
    for (std::thread& worker : workers_) { worker.join(); }
    flush_indexes();
}

void DiskKVBridge::flush_indexes() {
    flush_pending_.store(0, std::memory_order_relaxed);
    for (Family& target : families_) {
        if (target.store != nullptr) { target.store->flush_index(); }
    }
}

void DiskKVBridge::wait_idle() {
    std::unique_lock lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return queue_.empty() && in_flight_ == 0; });
}

void DiskKVBridge::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return quit_ || !queue_.empty(); });
            // Pending jobs are drained before exit: an accepted write is never dropped.
            if (queue_.empty()) { return; }
            job = std::move(queue_.front());
            queue_.pop_front();
            ++in_flight_;
        }
        queue_cv_.notify_all();
        apply(job);
        bool flush = false;
        {
            std::unique_lock lock(queue_mutex_);
            // Coalesce the index publication of a burst that pauses briefly between tickets.
            if (queue_.empty() && flush_pending_.load(std::memory_order_relaxed) != 0 && !quit_) {
                queue_cv_.wait_for(lock, std::chrono::milliseconds(10),
                                   [this] { return quit_ || !queue_.empty(); });
            }
            flush = queue_.empty() && flush_pending_.load(std::memory_order_relaxed) != 0;
        }
        if (flush) {
            try {
                flush_indexes();
            } catch (...) {
                // A failed index publication only costs a slot scan on the next open.
            }
        }
        {
            std::lock_guard lock(queue_mutex_);
            --in_flight_;
        }
        queue_cv_.notify_all();
    }
}

void DiskKVBridge::apply(Job& job) {
    const auto finish = [&job](SpillStatus status) {
        if (job.completion) { job.completion->status.store(status, std::memory_order_release); }
    };
    try {
        Family& target = family(job.kind);
        if (job.probe) {
            const bool present = target.store != nullptr && target.store->contains(job.id);
            if (present) {
                std::lock_guard lock(stats_mutex_);
                ++stats_.spill_dups;
            }
            finish(present ? SpillStatus::Present : SpillStatus::Missing);
            return;
        }
        const std::span<const std::byte> bytes =
            job.borrowed.empty() ? std::span<const std::byte>(job.bytes) : job.borrowed;
        std::vector<DiskKVIdentity> evicted;
        if (target.store == nullptr || bytes.size() != target.stride ||
            !target.store->upsert_page(job.id, bytes, &evicted)) {
            finish(SpillStatus::Failed);
            return;
        }
        if (flush_pending_.fetch_add(1, std::memory_order_relaxed) + 1 >= kFlushInterval) {
            flush_indexes();
        }
        {
            std::lock_guard lock(stats_mutex_);
            ++stats_.spills;
            stats_.spill_bytes += bytes.size();
            stats_.evicted_pages += evicted.size();
        }
        finish(SpillStatus::Written);
    } catch (...) { finish(SpillStatus::Failed); }
}

bool DiskKVBridge::spill_page_wait(const DiskKVIdentity& id, DiskKVKind kind,
                                   std::span<const std::byte> bytes,
                                   std::chrono::milliseconds timeout) {
    const Family& target = family(kind);
    if (target.store == nullptr || bytes.size() != target.stride) { return false; }
    if (target.store->contains(id)) {
        std::lock_guard lock(stats_mutex_);
        ++stats_.spill_dups;
        return true;
    }
    std::unique_lock lock(queue_mutex_);
    if (!queue_cv_.wait_for(lock, timeout,
                            [this] { return quit_ || queue_.size() < kQueueCapacity; }) ||
        quit_) {
        std::lock_guard stats_lock(stats_mutex_);
        ++stats_.queue_drops;
        return false;
    }
    queue_.push_back(Job{.id = id, .kind = kind, .bytes = {bytes.begin(), bytes.end()}});
    lock.unlock();
    queue_cv_.notify_all();
    return true;
}

bool DiskKVBridge::spill_page_sync(const DiskKVIdentity& id, DiskKVKind kind,
                                   std::span<const std::byte> bytes) {
    Family& target = family(kind);
    if (target.store == nullptr || bytes.size() != target.stride) { return false; }
    if (target.store->contains(id)) {
        std::lock_guard lock(stats_mutex_);
        ++stats_.spill_dups;
        return true;
    }
    std::vector<DiskKVIdentity> evicted;
    if (!target.store->upsert_page(id, bytes, &evicted)) { return false; }
    target.store->flush_index();
    std::lock_guard lock(stats_mutex_);
    ++stats_.spills;
    stats_.spill_bytes += bytes.size();
    stats_.evicted_pages += evicted.size();
    return true;
}

SpillTicket DiskKVBridge::try_probe(const DiskKVIdentity& id, DiskKVKind kind) {
    std::unique_lock lock(queue_mutex_, std::try_to_lock);
    if (!lock || quit_ || queue_.size() + in_flight_ >= kQueueCapacity) { return {}; }
    auto ticket = std::make_shared<SpillCompletion>();
    queue_.push_back(Job{.id = id, .kind = kind, .completion = ticket, .probe = true});
    lock.unlock();
    queue_cv_.notify_one();
    return ticket;
}

SpillTicket DiskKVBridge::try_submit(const DiskKVIdentity& id, DiskKVKind kind,
                                     std::span<const std::byte> bytes) {
    std::unique_lock lock(queue_mutex_, std::try_to_lock);
    if (!lock || quit_ || queue_.size() + in_flight_ >= kQueueCapacity) { return {}; }
    auto ticket = std::make_shared<SpillCompletion>();
    queue_.push_back(Job{.id = id, .kind = kind, .borrowed = bytes, .completion = ticket});
    lock.unlock();
    queue_cv_.notify_one();
    return ticket;
}

bool DiskKVBridge::restore_page(const DiskKVIdentity& id, DiskKVKind kind,
                                std::span<std::byte> destination) const {
    const Family& target = family(kind);
    const bool read      = target.store != nullptr && destination.size() == target.stride &&
                           target.store->read_page(id, destination);
    std::lock_guard lock(stats_mutex_);
    if (read) {
        ++stats_.restores;
        stats_.restore_bytes += destination.size();
    } else {
        ++stats_.restore_misses;
    }
    return read;
}

bool DiskKVBridge::restore_pages_direct(const Family& target, std::span<const DiskKVIdentity> ids,
                                        std::span<std::byte> destination) const {
    std::vector<DiskKVStore::ReadClaim> claims;
    claims.reserve(ids.size());
    std::vector<FileRangeRead> reads;
    reads.reserve(ids.size());
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const std::optional<DiskKVStore::ReadClaim> claim = target.store->claim_read(ids[index]);
        if (!claim) { break; }
        claims.push_back(*claim);
        reads.push_back(FileRangeRead{
            .offset      = claim->offset,
            .destination = destination.subspan(index * target.stride, target.stride)});
    }
    bool intact = claims.size() == ids.size() && direct_reader_->read(target.store->path(), reads);
    for (std::size_t index = 0; index < claims.size(); ++index) {
        const bool page_intact =
            intact && target.store->claim_intact(claims[index], reads[index].destination);
        intact = intact && page_intact;
        target.store->release_read(claims[index], page_intact);
    }
    return intact;
}

bool DiskKVBridge::restore_pages(std::span<const DiskKVIdentity> ids, DiskKVKind kind,
                                 std::span<std::byte> destination) const {
    const Family& target = family(kind);
    if (target.store == nullptr || destination.size() != ids.size() * target.stride) {
        return false;
    }
    if (direct_reader_ != nullptr && !ids.empty() &&
        restore_pages_direct(target, ids, destination)) {
        std::lock_guard lock(stats_mutex_);
        stats_.restores += ids.size();
        stats_.restore_bytes += destination.size();
        return true;
    }
    constexpr std::size_t kReaders = 8;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> intact{true};
    const auto read = [&] {
        for (std::size_t index = next.fetch_add(1); index < ids.size() && intact.load();
             index = next.fetch_add(1)) {
            if (!target.store->read_page(ids[index],
                                         destination.subspan(index * target.stride, target.stride))) {
                intact.store(false);
            }
        }
    };
    const std::size_t helpers = std::min(kReaders, ids.size()) - (ids.empty() ? 0 : 1);
    std::vector<std::thread> readers;
    readers.reserve(helpers);
    for (std::size_t i = 0; i < helpers; ++i) { readers.emplace_back(read); }
    read();
    for (std::thread& reader : readers) { reader.join(); }
    std::lock_guard lock(stats_mutex_);
    if (intact.load()) {
        stats_.restores += ids.size();
        stats_.restore_bytes += destination.size();
    } else {
        ++stats_.restore_misses;
    }
    return intact.load();
}

bool DiskKVBridge::drop_page(const DiskKVIdentity& id, DiskKVKind kind) {
    Family& target = family(kind);
    return target.store != nullptr && target.store->evict(id);
}

bool DiskKVBridge::contains(const DiskKVIdentity& id, DiskKVKind kind) const {
    const Family& target = family(kind);
    return target.store != nullptr && target.store->contains(id);
}

std::vector<std::uint32_t> DiskKVBridge::live_state_frontiers() const {
    std::vector<std::uint32_t> frontiers;
    const Family& target = family(DiskKVKind::StateImage);
    if (target.store == nullptr) { return frontiers; }
    for (const DiskKVIdentity& id : target.store->live_identities()) {
        if (id.frontier != 0) { frontiers.push_back(id.frontier); }
    }
    return frontiers;
}

DiskKVBridgeStats DiskKVBridge::stats() const {
    std::lock_guard lock(stats_mutex_);
    return stats_;
}

std::uint32_t DiskKVBridge::slot_count(DiskKVKind kind) const {
    const Family& target = family(kind);
    return target.store != nullptr ? target.store->slot_count() : 0U;
}

} // namespace ninfer
