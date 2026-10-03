#pragma once

// Overlay Vision residency: the tower stays in the pinned Host weight block and each media item is
// encoded inside one window of borrowed device memory. Tier 1 borrows free KV granules and runs on
// the Vision stream beside other lanes' decode; tier 2 borrows the evict-ranked weight tail and is
// exclusive, because the text weights it borrows are unmapped until the window closes.

#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_kv_pool.h"
#include "core/evictable_weight_pool.h"
#include "core/paged_kv_cache.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_control.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using detail::VisionWorkspacePlan;

struct VisionOverlayWindowStats {
    double window_seconds     = 0.0;
    double evict_seconds      = 0.0;
    double restore_seconds    = 0.0;
    std::size_t evicted_bytes = 0;
    std::size_t staged_bytes  = 0;
    std::uint32_t windows     = 0;
    // Windows that had to borrow the weight tail, which stalls every other lane.
    std::uint32_t exclusive_windows = 0;
};

class VisionResidencyBroker;

// One open window, served either from free KV granules or from the evict-ranked weight tail.
class VisionWindow {
public:
    enum class Tier : std::uint8_t { KvGranules, WeightTail };

    VisionWindow() noexcept = default;
    ~VisionWindow();

    VisionWindow(const VisionWindow&)            = delete;
    VisionWindow& operator=(const VisionWindow&) = delete;
    VisionWindow(VisionWindow&& other) noexcept;
    VisionWindow& operator=(VisionWindow&& other) noexcept;

    [[nodiscard]] bool open() const noexcept { return broker_ != nullptr; }

    [[nodiscard]] Tier tier() const noexcept { return tier_; }

    [[nodiscard]] DeviceSpan memory() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.leased() : weights_.leased();
    }

    [[nodiscard]] double open_seconds() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.stats().lease_seconds
                                         : weights_.stats().evict_seconds;
    }

    [[nodiscard]] double close_seconds() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.stats().return_seconds
                                         : weights_.stats().restore_seconds;
    }

    [[nodiscard]] std::size_t borrowed_bytes() const noexcept {
        return tier_ == Tier::KvGranules ? kv_.stats().mapped_bytes
                                         : weights_.stats().mapped_bytes;
    }

    // Returns the borrowed memory. Never throws: a failure poisons the owning pool.
    void close() noexcept;

private:
    friend class VisionResidencyBroker;

    VisionResidencyBroker* broker_ = nullptr;
    Tier tier_                     = Tier::WeightTail;
    EvictableWeightPool::Transaction weights_;
    EvictableKVPool::Transaction kv_;
    std::vector<KVPageRun> runs_;
};

// Program-owned arbiter of the single window a device can hold. Tier 1 is offered only while the
// KV tier is enabled, the Program can afford a capacity change and free pages cover the request.
class VisionResidencyBroker {
public:
    VisionResidencyBroker(DeviceContext& device, EvictableWeightPool& pool) noexcept
        : device_(device), pool_(pool) {}

    // `can_lend` refuses a loan while a sealed plan depends on the current capacity;
    // `on_change` advances the Program resource revision when the capacity moves.
    void enable_kv_tier(EvictableKVPool& arena, DeviceKVPagePool& pages,
                        std::function<bool()> can_lend, std::function<void()> on_change);

    // Free KV granules only; empty when they cannot fund the window.
    [[nodiscard]] std::optional<VisionWindow> try_acquire_kv(std::size_t bytes);
    // Free KV granules when they cover the window, the evict-ranked weight tail otherwise.
    [[nodiscard]] VisionWindow acquire(std::size_t bytes);

    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] bool window_open() const noexcept;

    [[nodiscard]] std::size_t window_capacity_bytes() const noexcept {
        return pool_.window_capacity_bytes();
    }

private:
    friend class VisionWindow;

    void require_closed() const;
    void return_loan(std::vector<KVPageRun>& runs) noexcept;

    DeviceContext& device_;
    EvictableWeightPool& pool_;
    EvictableKVPool* kv_arena_ = nullptr;
    DeviceKVPagePool* kv_pages_ = nullptr;
    std::function<bool()> can_lend_;
    std::function<void()> on_change_;
};

// Pinned slots receiving one item's merged embeddings each. A Vision prefill session holds one
// slot for its lifetime, so max_concurrency slots suffice and no window allocates Host memory.
class PinnedResultPool {
public:
    class Handle {
    public:
        Handle() noexcept = default;
        ~Handle() { release(); }

        Handle(const Handle&)            = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& other) noexcept;
        Handle& operator=(Handle&& other) noexcept;

        [[nodiscard]] std::span<std::byte> bytes() const noexcept { return bytes_; }

    private:
        friend class PinnedResultPool;

        Handle(PinnedResultPool& pool, std::size_t index, std::span<std::byte> bytes) noexcept
            : pool_(&pool), index_(index), bytes_(bytes) {}

        void release() noexcept;

        PinnedResultPool* pool_ = nullptr;
        std::size_t index_      = 0;
        std::span<std::byte> bytes_;
    };

    PinnedResultPool(std::size_t slots, std::size_t slot_bytes);

    [[nodiscard]] Handle acquire();

    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }

    [[nodiscard]] std::size_t pinned_bytes() const noexcept {
        return slot_bytes_ * buffers_.size();
    }

private:
    std::vector<std::unique_ptr<PinnedHostBuffer>> buffers_;
    std::vector<std::size_t> free_;
    std::size_t slot_bytes_ = 0;
};

// Encode workspace of one window for `max_merged_tokens`: the encode scratch followed directly by
// the item handoff, both inside the borrowed extent.
[[nodiscard]] VisionWorkspacePlan plan_vision_window_workspace(const Parameters& parameters,
                                                               std::uint32_t max_merged_tokens);
// Device bytes one window borrows: the streamed weight staging plus the window workspace.
[[nodiscard]] std::size_t vision_window_bytes(const VisionOverlayLayout& layout,
                                              const VisionWorkspacePlan& window_plan);

// Streams the tower from the pinned block through borrowed staging: a prelude region, a merger
// region and two layer slots refilled on the transfer stream one layer ahead of compute. Ordering
// is event-based; the Host never blocks between layers.
class VisionWeightStream {
public:
    VisionWeightStream(DeviceContext& device, const VisionOverlayLayout& layout,
                       std::span<const std::byte> pinned_block, std::byte* staging);
    ~VisionWeightStream();

    VisionWeightStream(const VisionWeightStream&)            = delete;
    VisionWeightStream& operator=(const VisionWeightStream&) = delete;

    // Host-prepared parameters rebased so each group addresses the staging that holds it.
    [[nodiscard]] VisionParameters window_parameters(const VisionParameters& host) const;

    // Issues the uploads of layers 0 and 1 after everything already submitted on `compute`.
    void reset(cudaStream_t compute);
    void prelude_ready(cudaStream_t compute);
    void merger_ready(cudaStream_t compute);
    // Top of the encoder loop for `layer`: gates compute on the slot upload, then refills the slot
    // the previous layer vacated, fenced after every op already issued.
    void arrive(std::uint32_t layer, cudaStream_t compute);

    [[nodiscard]] std::size_t uploaded_bytes() const noexcept { return upload_bytes_; }

private:
    void upload_next_layer();
    // Releases every event this object created, whether or not construction finished.
    void destroy_events() noexcept;

    DeviceContext& device_;
    const VisionOverlayLayout& layout_;
    std::span<const std::byte> block_;
    std::byte* prelude_        = nullptr;
    std::byte* merger_         = nullptr;
    std::byte* slot_[2]        = {nullptr, nullptr};
    cudaEvent_t uploaded_[2]   = {nullptr, nullptr};
    cudaEvent_t prelude_event_ = nullptr;
    cudaEvent_t merger_event_  = nullptr;
    cudaEvent_t compute_fence_ = nullptr;
    std::uint32_t next_upload_ = 0;
    std::size_t upload_bytes_  = 0;
};

// Encodes one media item per window: borrow the extent, stream the tower through it, land the
// merged embeddings in the session's pinned slot, return the memory. A KV-funded window may stay
// open across unit boundaries while the encode runs on the Vision stream; a weight-tail window is
// exclusive and lives inside one prefill unit. Only one lane prefills at a time, so at most one
// window exists even when a submitted one outlives its unit.
class VisionOverlaySession {
public:
    VisionOverlaySession(DeviceContext& device, VisionResidencyBroker& broker,
                         const Parameters& parameters, const VisionWorkspacePlan& window_plan,
                         PinnedResultPool::Handle result);
    ~VisionOverlaySession();

    VisionOverlaySession(const VisionOverlaySession&)            = delete;
    VisionOverlaySession& operator=(const VisionOverlaySession&) = delete;

    // Opens a KV-funded window and enqueues the item's encode on the Vision stream. Returns false,
    // leaving nothing open, when free KV cannot fund the window.
    [[nodiscard]] bool submit_item(std::span<const std::uint16_t> patches,
                                   const qwen3_5::VisionItemControl& control);

    [[nodiscard]] bool pending() const noexcept { return pending_; }

    [[nodiscard]] bool item_ready() const { return pending_ && completion_.ready(); }

    // Waits for the submitted item, closes its window and returns the pinned BF16
    // [output_hidden, merged] embeddings; valid until the next item.
    [[nodiscard]] std::span<const std::byte> complete_item();
    // Synchronous form inside the caller's prefill unit; falls back to the weight tail.
    [[nodiscard]] std::span<const std::byte> encode_item(std::span<const std::uint16_t> patches,
                                                         const qwen3_5::VisionItemControl& control);

    [[nodiscard]] const VisionOverlayWindowStats& stats() const noexcept { return stats_; }

private:
    using Clock = std::chrono::steady_clock;

    void begin(VisionWindow&& window, std::span<const std::uint16_t> patches,
               const qwen3_5::VisionItemControl& control, const VisionWorkspacePlan& item_plan);
    [[nodiscard]] VisionWorkspacePlan item_plan(const qwen3_5::VisionItemControl& control) const;

    DeviceContext& device_;
    VisionResidencyBroker& broker_;
    const Parameters& parameters_;
    const VisionOverlayLayout& layout_;
    VisionWorkspacePlan window_plan_;
    PinnedResultPool::Handle result_;
    VisionOverlayWindowStats stats_;
    VisionWindow window_;
    std::optional<VisionWeightStream> weights_;
    std::optional<VisionParameters> window_parameters_;
    CudaCompletionEvent completion_;
    cudaStream_t encode_stream_ = nullptr;
    std::size_t result_bytes_   = 0;
    bool pending_               = false;
    Clock::time_point window_start_{};
};

} // namespace ninfer::models::qwen3_5::execution
