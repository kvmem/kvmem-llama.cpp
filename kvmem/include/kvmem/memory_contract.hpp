#pragma once

// C++17 contract for native KV working sets. No engine or device runtime types.
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace kvmem {
class KvMemStore;

enum class MemoryErrorCode {
    InvalidPlan, Unsupported, BudgetExceeded, Stale, Pending,
    TransferFailed, Disconnected, InvalidSession
};
class MemoryError : public std::runtime_error {
public:
    MemoryError(MemoryErrorCode code, const std::string & message)
        : std::runtime_error(message), code_(code) {}
    MemoryErrorCode code() const noexcept { return code_; }
private:
    MemoryErrorCode code_;
};

struct BackendIdentity {
    std::string backend;
    std::string revision;
    std::string model;
    std::string state_schema;
    bool operator==(const BackendIdentity & other) const;
};
struct SessionIdentity {
    uint64_t id = 0;
    uint64_t incarnation = 0; // changes on reset/replacement; edits advance content/history versions
    bool operator==(const SessionIdentity & other) const;
};
enum class PayloadPlaneKind { Key, Value, KeyScale, ValueScale };
struct PayloadPlane {
    uint32_t layer = 0; // model layer identity, not a dense attention-layer index
    PayloadPlaneKind kind = PayloadPlaneKind::Key;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t alignment = 1;
    // Zero means opaque page encoding. Nonzero describes contiguous token rows.
    uint64_t token_stride = 0;
    bool operator==(const PayloadPlane & other) const;
};
struct PayloadLayout {
    std::string format; // backend-owned encoding identity; includes representation version
    uint32_t version = 0;
    uint32_t block_tokens = 0;
    uint64_t record_bytes = 0;
    uint64_t alignment = 1;
    std::vector<PayloadPlane> planes;
    void validate() const;
    bool operator==(const PayloadLayout & other) const;
};
struct MemoryCapabilities {
    uint32_t page_tokens = 0; // allocation quantum: native page or legacy slot size
    uint64_t max_position = 0; // maximum logical frontier; token positions are below this limit
    bool native_payload = false;
    bool sparse_view = false;
    bool original_rope_positions = false;
    bool pre_rope_statistics = false;
    bool state_checkpoint = false;
    bool asynchronous_transfer = false;
};
struct MemoryDescriptor {
    BackendIdentity identity;
    PayloadLayout layout;
    MemoryCapabilities capabilities;
    void validate() const;
};

struct BlockDescriptor {
    SessionIdentity session;
    uint64_t id = 0;
    uint64_t original_start = 0;
    uint32_t valid_tokens = 0;
    uint64_t content_version = 0;
    uint64_t statistics_version = 0; // distinct from KV payload version
    bool operator==(const BlockDescriptor & other) const;
};
struct BlockCopies {
    BlockDescriptor block;
    uint64_t host_version = 0; // zero: no host copy; smaller: stale copy
};
struct MemoryStamp {
    bool valid = false;
    SessionIdentity session;
    uint64_t history_revision = 0; // advances on logical content/statistics changes
    uint64_t view_version = 0;
    uint64_t frontier = 0; // committed logical KV frontier, never compacted
    uint64_t execution_history = 0; // distinguishes full/sparse/replayed state histories
    bool operator==(const MemoryStamp & other) const;
};
struct MemorySnapshot {
    MemoryStamp stamp;
    std::vector<BlockCopies> blocks; // complete logical catalog, chronological order
    std::vector<uint64_t> resident; // complete CURRENT versions owned by the device view
};
struct ViewBlock {
    BlockDescriptor block;
    uint64_t compact_start = 0; // attention address only; never a RoPE rewrite
};
struct ExecutionView {
    MemoryStamp stamp;
    std::vector<ViewBlock> blocks;
    uint64_t visible_tokens = 0;
};
struct MemoryBudget {
    uint64_t selected_tokens = 0;
    uint64_t reserve_tokens = 0; // new prompt/decode space, rounded independently to pages
    uint64_t device_tokens = 0;
    uint64_t host_payload_bytes = 0; // retained payload only; state/workspace are engine-owned
};
struct WorkingSetPlan {
    MemoryStamp expected;
    MemoryBudget budget;
    ExecutionView next;
    std::vector<uint64_t> mandatory;
    std::vector<BlockDescriptor> evict;
    std::vector<BlockDescriptor> spill; // evicted copies without a completed current host copy
    std::vector<BlockDescriptor> restore;
    uint64_t device_tokens_required = 0;
    uint64_t host_payload_bytes_required = 0;
};

// Set arithmetic shared by versioned transactions and the llama slot bridge.
// This does not validate payloads, reserve storage, or publish an attention view.
// Returned IDs are sorted; duplicate input IDs are rejected. force_reload is
// the existing transfer-reuse ablation, not a change to RoPE positions.
struct ResidencyDelta {
    std::vector<uint64_t> retain;
    std::vector<uint64_t> evict;
    std::vector<uint64_t> restore;
};
ResidencyDelta plan_residency(const std::vector<uint64_t> & resident,
    const std::vector<uint64_t> & selected, bool force_reload = false);

// Pure planning: no KvMemRemap, no mutation of the policy store or residency.
WorkingSetPlan plan_working_set(const MemoryDescriptor &, const MemorySnapshot &,
    const std::vector<uint64_t> & selected, const std::vector<uint64_t> & mandatory,
    const MemoryBudget &);

struct CheckpointIdentity {
    BackendIdentity backend;
    PayloadLayout layout;
    MemoryStamp stamp;
    // Exact compatibility. Sharing token text alone never authorizes state reuse.
    bool compatible_with(const CheckpointIdentity & other) const;
};

enum class TransferStatus { Pending, Complete, Failed };
struct TransferPoll {
    TransferStatus status = TransferStatus::Pending;
    std::string error;
};
enum class AbortResult { PreviousViewPreserved, SessionInvalidated };

// Every operation is required. This is not the legacy metadata-only KvMemBackend.
// prepare reserves ALL destinations/scratch; start_spill cannot reclaim source pages.
// start_restore is called only after spill completes, and may reclaim evicted pages.
// publish atomically exposes the prepared view and metadata at an engine-safe boundary.
class MemoryTransfer {
public:
    virtual ~MemoryTransfer() = default;
    virtual void start_spill() = 0;
    virtual TransferPoll poll() = 0;
    virtual void start_restore() = 0;
    virtual void publish(const ExecutionView &) = 0;
    // Must drain/cancel outstanding work BEFORE releasing any buffers/reservations.
    // Preserved means bytes AND the previous view remain usable, not just metadata.
    virtual AbortResult abort() noexcept = 0;
};
class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;
    virtual MemoryDescriptor descriptor() const = 0;
    virtual MemorySnapshot snapshot() const = 0;
    virtual MemoryStamp stamp() const noexcept = 0;
    // Strong guarantee on throw/null: no transfers started and previous view intact.
    virtual std::unique_ptr<MemoryTransfer> prepare(const WorkingSetPlan &) = 0;
    virtual void invalidate() noexcept = 0;
};

enum class MemoryPhase { Idle, Prepared, Spilling, Restoring, Ready, Published, Invalid, Detached };
// Single owner, called at an engine-safe boundary. The shared backend and transaction
// outlive pending transfers. This coordinator owns no GPU allocator or checkpoint bytes.
class MemorySession {
public:
    explicit MemorySession(std::shared_ptr<MemoryBackend> backend);
    ~MemorySession();
    MemorySession(const MemorySession &) = delete;
    MemorySession & operator=(const MemorySession &) = delete;
    const WorkingSetPlan & prepare(const MemoryStamp & expected,
        const std::vector<uint64_t> & selected, const std::vector<uint64_t> & mandatory,
        const MemoryBudget & budget);
    const WorkingSetPlan & prepare_reselect(const MemoryStamp & expected,
        const KvMemStore & policy, const std::vector<uint32_t> & mandatory,
        const MemoryBudget & budget);
    MemoryPhase advance(); // nonblocking polling; never publishes implicitly
    ExecutionView publish();
    AbortResult abort() noexcept;
    void disconnect() noexcept;
    MemoryPhase phase() const noexcept { return phase_; }
    const WorkingSetPlan * pending_plan() const noexcept { return plan_ ? &*plan_ : nullptr; }
private:
    void require_attached() const;
    void require_current() const;
    void failed(bool force_invalidate = false) noexcept;
    const WorkingSetPlan & prepare_snapshot(const MemorySnapshot &,
        const std::vector<uint64_t> &, const std::vector<uint64_t> &, const MemoryBudget &);
    std::shared_ptr<MemoryBackend> backend_;
    MemoryDescriptor descriptor_;
    MemoryPhase phase_ = MemoryPhase::Idle;
    std::optional<WorkingSetPlan> plan_;
    std::unique_ptr<MemoryTransfer> transfer_;
};
} // namespace kvmem
