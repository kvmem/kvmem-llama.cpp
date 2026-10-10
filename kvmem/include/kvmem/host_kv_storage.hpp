#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kvmem {
namespace detail { struct HostKvAccount; struct HostKvAllocation; }
class HostKvReservation;
class HostKvSource;
class SpillFile;
struct SpillIoStats;

// A read lease keeps the valid bytes and their RAM/disk capacity charge alive
// after owner deletion. Adapters complete their transfer before releasing it.
class HostKvLease {
public:
    HostKvLease() = default;
    HostKvLease(const HostKvLease &);
    HostKvLease & operator=(const HostKvLease &);
    HostKvLease(HostKvLease &&) noexcept;
    HostKvLease & operator=(HostKvLease &&) noexcept;
    ~HostKvLease();
    // A cold lease pins its disk extent/content; data() is null in that case.
    // read() works for either tier and never promotes the entire record to RAM.
    const uint8_t * data() const noexcept;
    size_t size() const noexcept;
    void read(size_t offset, void * destination, size_t bytes) const;
    explicit operator bool() const noexcept { return bool(allocation_); }
private:
    friend class HostKvRecord;
    friend class HostKvSource;
    explicit HostKvLease(std::shared_ptr<detail::HostKvAllocation> allocation);
    std::shared_ptr<detail::HostKvAllocation> allocation_;
};

// A deferred read owns immutable content without pinning its current RAM tier.
// Workers acquire a short read lease only when copying bytes. This prevents a
// queued cold request from consuming all RAM needed by another transfer.
class HostKvSource {
public:
    HostKvSource() = default;
    HostKvSource(const HostKvSource &);
    HostKvSource & operator=(const HostKvSource &);
    HostKvSource(HostKvSource &&) noexcept;
    HostKvSource & operator=(HostKvSource &&) noexcept;
    ~HostKvSource();
    size_t size() const noexcept;
    explicit operator bool() const noexcept { return bool(allocation_); }
    HostKvLease lease() const;
    void read(size_t offset, void * destination, size_t bytes) const;
private:
    friend class HostKvRecord;
    explicit HostKvSource(std::shared_ptr<detail::HostKvAllocation> allocation);
    std::shared_ptr<detail::HostKvAllocation> allocation_;
};

// Native opaque bytes. Logical identity, valid tokens and content versions remain
// in MemoryBackend's catalog. Adapters serialize mutation at engine-safe boundaries.
class HostKvRecord {
public:
    HostKvRecord() = default;
    HostKvRecord(HostKvRecord &&) noexcept = default;
    HostKvRecord & operator=(HostKvRecord &&) noexcept = default;
    HostKvRecord(const HostKvRecord &) = delete;
    HostKvRecord & operator=(const HostKvRecord &) = delete;
    size_t size() const noexcept;
    bool empty() const noexcept { return size() == 0; }
    explicit operator bool() const noexcept { return !empty(); }
    HostKvLease lease() const;
    HostKvSource source() const;
    // Owner-only query: a mutable buffer cannot be relocated. The owner must
    // serialize this check with sealing and writable borrows.
    bool mutable_resident() const noexcept;
    // data() is a RAM-only synchronous borrow; tiered stores require read()/lease().
    // Writable borrows exclude read leases and remain ineligible for eviction
    // until seal(). The owner serializes mutation and remains alive while borrowed.
    const uint8_t * data() const;
    uint8_t * writable_data();
    void read(size_t offset, void * destination, size_t bytes) const;
    void write(size_t offset, const void * source, size_t bytes);
    // Complete, immutable contents may be evicted under RAM pressure. Writes
    // make the record mutable again; adapters seal only after capture completes.
    void seal();
    uint64_t resident_bytes() const;
    void reset() noexcept { allocation_.reset(); }
private:
    friend class HostKvReservation;
    explicit HostKvRecord(std::shared_ptr<detail::HostKvAllocation> allocation);
    std::shared_ptr<detail::HostKvAllocation> allocation_;
};

// Reservations count against the same limit as live and leased payloads. A
// failed allocation retains its reservation; destruction returns unused capacity.
class HostKvReservation {
public:
    HostKvReservation(HostKvReservation &&) noexcept;
    HostKvReservation & operator=(HostKvReservation &&) noexcept;
    HostKvReservation(const HostKvReservation &) = delete;
    HostKvReservation & operator=(const HostKvReservation &) = delete;
    ~HostKvReservation();
    uint64_t remaining() const noexcept { return remaining_; }
    HostKvRecord allocate(size_t bytes, const void * source = nullptr);
private:
    friend class HostKvStorage;
    HostKvReservation(std::shared_ptr<detail::HostKvAccount>, uint64_t bytes);
    void release() noexcept;
    std::shared_ptr<detail::HostKvAccount> account_;
    uint64_t remaining_ = 0;
};

class HostKvStorage {
public:
    explicit HostKvStorage(uint64_t capacity_bytes = UINT64_MAX);
    HostKvStorage(uint64_t capacity_bytes, std::shared_ptr<SpillFile> spill);
    HostKvReservation reserve(uint64_t bytes);
    HostKvRecord allocate(size_t bytes, const void * source = nullptr);
    uint64_t charged_bytes() const;
    uint64_t capacity_bytes() const noexcept;
    uint64_t disk_bytes() const;
    uint64_t disk_capacity_bytes() const noexcept;
    // Idle state/statistics attachments share the same D account as native KV.
    std::shared_ptr<SpillFile> spill_file() const noexcept;
    SpillIoStats io_stats() const noexcept;
    // Explicit idle reclamation uses the same records/quota as active pressure.
    // Leased/mutable records remain resident. Failure preserves their valid bytes.
    void trim(uint64_t resident_target);
private:
    std::shared_ptr<detail::HostKvAccount> account_;
};
} // namespace kvmem
