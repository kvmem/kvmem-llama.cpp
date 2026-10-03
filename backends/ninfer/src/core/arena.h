#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

namespace ninfer {

namespace core { class ResidentMemorySession; }

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Owning device allocation for long-lived buffers. DeviceArena remains the
// suballocation primitive for workspaces; this type owns exactly one cudaMalloc.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void fill(int byte_value = 0);
    // Completes the upload before returning. Callers must first order any prior device
    // accesses to the destination range.
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
    std::shared_ptr<core::ResidentMemorySession> resident_session_;
};

namespace core {
// WDDM residency (Windows builds with NINFER_D3D12_RESIDENCY, --wddm-evictable-budget): owning
// DeviceArenas come from a D3D12 heap held resident at maximum priority, so WDDM evicts other
// allocations on the adapter rather than paging the arena out. Elsewhere enabling it throws.
void set_wddm_residency_lock_enabled(bool enabled);
[[nodiscard]] bool wddm_residency_lock_enabled() noexcept;
} // namespace core

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class DeviceArena;

        explicit Scope(DeviceArena& arena) noexcept;

        DeviceArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
        // Which rank's offset `saved_offset_` refers to. A scope opened on one rank can outlive a
        // switch to another -- the pipeline split walks ranks inside an enclosing scope -- and
        // restoring the wrong rank's bump pointer would hand out overlapping workspace.
        std::size_t saved_rank_ = 0;
    };

    explicit DeviceArena(std::size_t capacity_bytes);
    // Non-owning arena over an already allocated device region.
    explicit DeviceArena(DeviceSpan storage);
    ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;

    void* base() const noexcept;
    std::size_t used() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t peak_used() const noexcept;
    void reset_peak() noexcept;

    // --- pipeline split support -------------------------------------------------------------
    //
    // A layer split runs each layer on the device that holds its weights, so scratch has to come
    // from that device too. Rather than thread a different arena through every call site, one
    // arena carries a backing per rank and switches between them: the bump pointer, capacity and
    // peak are saved and restored per rank, so callers keep using the same arena object and only
    // the execution loop knows about ranks.
    //
    // Rank 0 is this arena's original storage. Storage attached here is non-owning, exactly like
    // the DeviceSpan constructor, and must live on the device belonging to that rank.
    void attach_rank_storage(DeviceSpan storage);
    void activate_rank(std::size_t rank);
    [[nodiscard]] std::size_t active_rank() const noexcept;
    [[nodiscard]] std::size_t rank_count() const noexcept;
    // Peak for a specific rank, for reporting each card's workspace high-water mark.
    [[nodiscard]] std::size_t peak_used_for_rank(std::size_t rank) const;

private:
    struct RankBacking {
        void* base       = nullptr;
        std::size_t cap  = 0;
        std::size_t off  = 0;
        std::size_t peak = 0;
    };

    void store_active_rank() noexcept;

    void* base_       = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
    std::size_t peak_ = 0;
    bool owns_        = true;
    // The allocation this arena itself owns (rank 0's), independent of which rank's storage
    // `base_` currently aliases. `activate_rank` only ever swaps `base_`/`cap_`/`off_`/`peak_`, so
    // cleanup must free this instead of `base_` -- otherwise destroying or move-assigning the
    // arena while a borrowed rank is active frees storage this arena does not own and leaks the
    // rank-0 allocation it does.
    void* owned_base_ = nullptr;
    std::shared_ptr<core::ResidentMemorySession> resident_session_;

    // Empty until a second rank is attached, so the single-device path carries no extra state and
    // no extra work.
    std::vector<RankBacking> ranks_;
    std::size_t active_rank_ = 0;
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
    std::shared_ptr<core::ResidentMemorySession> resident_session_;
};

using WorkspaceArena = DeviceArena;

// Binds a rank for the duration of a scope and restores the previous one, so an exception thrown
// while the arena is switched away from its caller's rank cannot leave it there -- unlike
// DeviceContext, DeviceArena has no device to fall back on, so an unrestored rank silently hands
// out the wrong device's scratch to the next allocation.
class ScopedArenaRank {
public:
    ScopedArenaRank(DeviceArena& arena, std::size_t rank);
    ~ScopedArenaRank() noexcept;

    ScopedArenaRank(const ScopedArenaRank&)            = delete;
    ScopedArenaRank& operator=(const ScopedArenaRank&) = delete;

private:
    DeviceArena& arena_;
    std::size_t previous_rank_ = 0;
};

} // namespace ninfer
