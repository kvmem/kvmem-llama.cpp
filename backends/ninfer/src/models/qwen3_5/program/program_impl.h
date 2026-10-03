#pragma once
#include "models/qwen3_5/program/speculative/mtp_adaptive.h"
#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "kvmem/memory_contract.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/kvmem_store.hpp"
#include "kvmem/mean_key_retrieval.hpp"
#include "core/disk_kv_bridge.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/program/prefix/hybrid_cache.h"
#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/planning/resource_projection.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"
#include "models/qwen3_5/program/ngram_proposer.h"

#include <algorithm>
#include <string_view>
#include <filesystem>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

using PreparedPromptData    = qwen3_5::PreparedPromptData;
using RewriteCheckpointKind = qwen3_5::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_5::RewriteCheckpointSpec;

// Device KV is leased on demand. An active request holds a bounded window of its remaining
// output rather than the whole client budget, and extends that window at a decode-round
// boundary; a full window is requested first and a step-sized extension is enough when the pool
// cannot spare one.
inline constexpr std::uint32_t kKVLeaseGrowthMarginTokens = 4096;

[[nodiscard]] constexpr std::uint32_t kv_pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

[[nodiscard]] constexpr std::uint32_t kv_tokens_for_pages(std::uint32_t pages) noexcept {
    return pages == 0 ? 0U : (pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize) + 1U;
}

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::PrivateTurnClosure || path == ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::PrivateTurnClosure
                                                      : ReusePath::PrivateResponseReplay;
}

[[nodiscard]] constexpr runtime::CheckpointKind
checkpoint_kind(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? runtime::CheckpointKind::TurnClosure
                                                      : runtime::CheckpointKind::ResponseReplay;
}

enum class RewriteCheckpointDisposition : std::uint8_t {
    RetainExisting,
    ReplaceAtCommittedFrontier,
    DropOptional,
};

struct PreparedCaptureBacking {
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
};

struct PreparedCaptureIdentity {
    std::shared_ptr<const PreparedCaptureBacking> backing;
    qwen3_5::PrefixShortlistKey shortlist_key;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] std::span<const TokenId> ledger() const noexcept {
        if (!backing || shortlist_key.frontier > backing->ledger.size()) { return {}; }
        return std::span<const TokenId>(backing->ledger).first(shortlist_key.frontier);
    }

    [[nodiscard]] const qwen3_5::detail::ResidentPrefixIdentity* prefix_identity() const noexcept {
        return backing ? &backing->prefix_identity : nullptr;
    }

    [[nodiscard]] bool prefix_equals(const PreparedCaptureIdentity& other) const {
        const std::span<const TokenId> left  = ledger();
        const std::span<const TokenId> right = other.ledger();
        const auto* left_identity            = prefix_identity();
        const auto* right_identity           = other.prefix_identity();
        return left_identity != nullptr && right_identity != nullptr &&
               left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin()) &&
               left_identity->prefix_equals(*right_identity, left.size());
    }
};

struct CaptureGroup {
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::optional<RewriteCheckpointKind> rewrite;
    std::uint32_t frontier                  = 0;
    std::uint32_t input_order               = 0;
    bool shared                             = false;
    bool long_anchor                        = false;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

enum class PressureStateDecision : std::uint8_t {
    None,
    DropEndpointDeviceDuplicate,
    DemoteEndpointToHost,
    DropEndpointHostDuplicate,
    DropRewriteDeviceDuplicate,
    DemoteRewriteToHost,
    DropRewriteHostDuplicate,
    DropSharedDeviceDuplicate,
    DemoteSharedToHost,
    DropSharedHostDuplicate,
};

enum class PressureKVDecisionKind : std::uint8_t {
    None,
    DropDeviceDuplicate,
    DemoteToHost,
    DropHostDuplicate,
};

struct PressureKVDecision {
    std::uint32_t begin_page    = 0;
    std::uint32_t page_count    = 0;
    PressureKVDecisionKind kind = PressureKVDecisionKind::None;

    [[nodiscard]] friend constexpr bool operator==(PressureKVDecision,
                                                   PressureKVDecision) noexcept = default;
};

// One Program-private complete outcome for one eligible owner. It may combine State, Main KV,
// Backend KV, and checkpoint changes; common scheduling never observes these physical decisions.
struct PressureDecision {
    std::uint64_t id = 0;
    std::vector<PressureStateDecision> state_changes;
    std::vector<PressureKVDecision> main_kv_changes;
    std::vector<PressureKVDecision> backend_kv_changes;
    std::vector<runtime::CheckpointRef> dropped_checkpoints;
    PhysicalDelta checkpoint_drop_effect;
    PhysicalDelta effect;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::uint32_t checkpoint_drops = 0;
    bool evicts_continuation       = false;
    bool shared_owner              = false;

    [[nodiscard]] friend bool operator==(const PressureDecision&,
                                         const PressureDecision&) noexcept = default;
};

struct PressureCheckpointRecoveryProjection {
    runtime::PlanningOwnerId owner;
    runtime::CheckpointRef checkpoint;
    std::uint32_t alternative_offset = 0;
    std::uint32_t alternative_count  = 0;
    bool survives                    = true;
};

struct CaptureAssessmentImpl {
    PhysicalDemand demand;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources capacity_preparation_removed;
};

struct RequestBasePlanImpl {
    std::uint64_t memory_host_reservation = 0, publication_order = 0;
    runtime::RequestPlanSummary summary;
    detail::PhysicalDemand root_demand;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    std::shared_ptr<text::GrammarState> grammar;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_5::VisionControlPlan> vision_control_plan;
    std::optional<qwen3_5::RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<CaptureGroup> capture_groups;
    std::vector<CaptureGroup> shared_candidates;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;
    bool allow_memory_reuse = false;
    std::uint32_t first_token_top_logprobs = 0;
};

// Program-owned physical planning state shared by request materialization and active capture.
// Request scheduling fields never enter this record, and capture never becomes an admission type.
struct ResourceCandidateState {
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment identity_assessment;
    runtime::ProgramResourceRevision planning_revision;
    // Pressure outcomes are canonicalized against the candidate's identity peak.  Composition
    // rewrites demand to the selected post-pressure peak, so regenerating an outcome from that
    // rewritten demand would compare it against a different problem at seal time.
    detail::PhysicalResources identity_pressure_deficit;
    // A structurally valid pressure target can still be blocked by Host extent geometry even when
    // aggregate free bytes are sufficient. Keep the blocked allocation work explicit so a child
    // target can release Host replicas instead of being mistaken for a structurally invalid node.
    std::size_t blocked_host_allocation_bytes = 0;
    detail::PhysicalDemand demand;
    // Resources released by consuming this private owner alone. Shared aliases are intentionally
    // absent; complete pressure targets settle their joint reference graph separately.
    detail::PhysicalResources source_resources;
    ReusePath reuse                                  = ReusePath::Root;
    std::uint32_t reuse_base                         = 0;
    RewriteCheckpointDisposition rewrite_disposition = RewriteCheckpointDisposition::DropOptional;
    bool has_source                                  = false;
    bool has_shared_source                           = false;
    std::optional<runtime::CheckpointRef> selected_checkpoint;
    std::uint32_t source_index             = 0;
    std::uint64_t source_generation        = 0;
    std::uint32_t shared_source_index      = 0;
    std::uint64_t shared_source_generation = 0;
    runtime::PrefillWork remaining_prefill_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    runtime::PrivateSourceMode source_mode = runtime::PrivateSourceMode::ConsumeToActive;
    detail::PhysicalResources active_optional_resources;
    bool state_fork_required          = false;
    bool text_prefix_fork_required    = false;
    bool backend_prefix_fork_required = false;
    bool needs_transfer               = false;
    std::vector<qwen3_5::detail::PressureDecision> pressure_options;
    std::vector<runtime::PlanningOwnerId> pressure_owner_ids;
    std::vector<std::uint32_t> pressure_indices;
    std::vector<std::uint64_t> pressure_generations;
    std::vector<qwen3_5::detail::PressureDecision> shared_pressure_options;
    std::vector<runtime::PlanningOwnerId> shared_pressure_owner_ids;
    std::vector<std::uint32_t> shared_pressure_indices;
    std::vector<std::uint64_t> shared_pressure_generations;
};

struct AdmissionCandidateImpl : ResourceCandidateState {
    std::uint64_t memory_host_reservation = 0, publication_order = 0;
    MtpBridgeMode mtp_bridge = MtpBridgeMode::None;
    bool prepare_mtp         = false;
    std::optional<VisionPrefillPlan> vision;
    std::vector<CaptureGroup> capture_groups;
    std::vector<CaptureGroup> shared_candidates;
    ops::SamplingConfig sampling;
    std::shared_ptr<text::GrammarState> grammar;
    std::uint32_t first_token_top_logprobs    = 0;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    runtime::LaneId destination{};
    std::uint64_t destination_epoch = 0;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    // A root whose prefix [0, frontier) the disk tier can restore; zero otherwise.
    std::uint32_t disk_restore_frontier   = 0;
    std::uint32_t memory_restore_frontier = 0;
    std::uint64_t memory_restore_generation = 0;
    bool allow_memory_reuse = false;
    bool text_retained_tail_release       = false;
    bool backend_retained_tail_release    = false;
};

struct CapturePressureCandidateImpl : ResourceCandidateState {};

// Hybrid prefix cache admission decision (docs/maintainer/hybrid-prefix-cache-spec.md §6). The
// base-plan facts the start path needs are copied so the quote outlives the Engine's base plan.
struct HybridQuoteImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_5::VisionControlPlan> vision_control_plan;
    runtime::PrefillWork root_rebuild_work;
    std::uint32_t root_rebuild_tail_begin = 0;
    runtime::prefix_cache::SnapshotRef snapshot; // invalid: root
    std::uint32_t reuse_frontier = 0;
    // Longest prompt prefix held as cached full blocks, reusable or not.
    std::uint32_t cached_prefix_tokens = 0;
    std::uint32_t destination          = 0;
    std::uint64_t destination_epoch    = 0;
};

// A prefill tap whose StateImage is captured but whose snapshot waits for the blocks it anchors
// on: a frontier inside a block needs that block complete, and an MTP backend trails the text
// frontier by one token, so even a page-aligned frontier waits for its last block's backend page.
struct HybridPendingTap {
    std::uint32_t frontier = 0;
    StateImageHandle image;
    std::uint32_t slot = 0; // staging device snapshot slot
};

// Per-lane hybrid bookkeeping for the active sequence.
struct HybridLaneState {
    bool active  = false;
    bool publish = false;
    // Root path of full blocks this sequence pins, in prompt order. Blocks past the reuse
    // frontier are appended as the sequence commits them.
    std::vector<runtime::prefix_cache::NodeRef> path;
    std::uint64_t path_hash = runtime::prefix_cache::kRootLookupHash;
    // Extra key of every full prompt block (Vision identity), empty for text-only prompts.
    std::vector<std::uint64_t> prompt_extras;
    // Extra key of blocks after the last full prompt block: every Vision item precedes them.
    std::uint64_t trailing_extra = 0;
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    std::size_t next_tap = 0;
    std::vector<runtime::prefix_cache::TapExclusion> exclusions;
    std::vector<HybridPendingTap> pending;
    // Most recent snapshot frontier this sequence reused or captured.
    std::uint32_t last_capture = 0;
    // Deepest snapshot frontier known on this path (reused or created by this sequence).
    std::uint32_t deepest_snapshot = 0;
    // The Host restore this sequence was admitted from (0 without one). Its first prefill pass
    // queues behind the restore's per-layer events; releasing the lane queues behind the whole
    // restore if it may still be landing.
    std::uint64_t restore_ticket = 0;
    bool restore_layers_pending  = false;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

using AdmissionCandidateImpl       = qwen3_5::detail::AdmissionCandidateImpl;
using CapturePressureCandidateImpl = qwen3_5::detail::CapturePressureCandidateImpl;
using ResourceCandidateState       = qwen3_5::detail::ResourceCandidateState;
using RequestBasePlanImpl          = qwen3_5::detail::RequestBasePlanImpl;
using CapturePressureCandidate     = qwen3_5::CapturePressureCandidate;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
    // Columns the speculative round recorded GDN transitions at; the fold reads them back at it.
    std::uint32_t record_width = 0;
};

// Why every owner is being dropped: a failure discards everything; an orderly shutdown first saves
// the hybrid Host tier when a cache file is attached.
enum class ProgramCleanup : std::uint8_t {
    Failure,
    Shutdown,
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Finishable,
};

enum class ContinuationSlotRole : std::uint8_t {
    Free,
    ReservedMaterialization,
    Active,
    Catalogued,
};

struct ContinuationSlot {
    ContinuationSlotRole role = ContinuationSlotRole::Free;
    std::uint64_t generation  = 1;
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
    runtime::PrefillWork rebuild_work;
};

struct LongAnchorCheckpoint {
    StateImageHandle state;
    std::uint32_t frontier = 0;
    std::uint32_t ordinal  = 0;
    runtime::PrefillWork rebuild_work;
};

struct SequenceKVBundle {
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    // MTP: the drafts the round verifies; zero for the other families.
    std::uint32_t verify_window = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct KvmemPrefixCheckpoint {
    kvmem::CheckpointIdentity identity;
    std::vector<TokenId> prefix;
    ResidentPrefixIdentity prefix_identity;
    std::vector<std::uint64_t> resident_pages;
    std::vector<float> mean_tail;
    std::unique_ptr<PinnedHostBuffer> state;
};

using KvmemHostRecord = std::vector<std::byte>;

struct KvmemWindowState {
    std::optional<PreparedSessionKey> session_key;
    std::uint64_t publication_order = 0;
    bool update_session_index = true;
    // Compact execution slots -> original logical pages. Host records retain native
    // packed data AND scales. They never contain a second allocation of device KV.
    std::vector<std::uint32_t> pages;
    std::vector<std::uint32_t> media_pages;
    std::vector<std::unique_ptr<KvmemHostRecord>> archive;
    std::vector<std::uint64_t> host_versions;
    kvmem::MemoryStamp stamp;
    std::uint64_t host_bytes = 0;
    std::uint64_t spilled_bytes = 0, restored_bytes = 0;
    std::uint32_t removed_tokens = 0;
    std::uint32_t swaps = 0;
    std::unique_ptr<kvmem::RawKvStore> statistics;
    std::unique_ptr<kvmem::KvMemStore> selector;
    std::vector<float> query_sum;
    std::uint32_t statistics_frontier = 0;
    std::uint32_t query_tokens = 0;
    std::uint32_t query_begin = 0, query_end = 0;
    // Long prompts retain the pre-query boundary for selection/replay; short prompts
    // retain the input prefix before its final token for request-to-request reuse.
    std::unique_ptr<KvmemPrefixCheckpoint> prefix_checkpoint;
    bool query_replayed = false;

};

struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    ActiveStateBinding state;
    std::optional<StateImageHandle> rewrite_state;
    std::optional<StateImageHandle> reserved_state;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::int32_t rope_delta               = 0;
    KvmemWindowState window;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool endpoint_valid           = false;
    RewriteCheckpoint rewrite_checkpoint;
    std::vector<LongAnchorCheckpoint> long_anchors;
    std::vector<std::uint32_t> shared_prefix_references;
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;
};

struct SharedPrefixState {
    std::optional<SequenceKVBundle> kv;
    StateImageHandle state;
    std::shared_ptr<const PreparedCaptureIdentity> identity;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    bool tail_hidden_valid         = false;
    runtime::PrefillWork rebuild_work;
    std::uint32_t active_references = 0;
};

enum class SharedPrefixSlotRole : std::uint8_t {
    Free,
    ReservedCapture,
    ReservedReplacement,
    Catalogued,
};

struct SharedPrefixSlot {
    SharedPrefixSlotRole role = SharedPrefixSlotRole::Free;
    std::uint64_t generation  = 1;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    std::unique_ptr<NgramProposer> ngram;
    std::shared_ptr<const NgramSnapshot> ngram_snapshot;
    std::uint64_t ngram_copy_source = 0;
    std::uint32_t ngram_copy_offset = 0;
    std::size_t ngram_copy_ledger   = 0;
    std::size_t ngram_indexed       = 0;
    Lifecycle lifecycle             = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    std::shared_ptr<text::GrammarState> grammar;
    std::uint32_t first_token_top_logprobs = 0;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    // Adaptive MTP: how far this request's drafts survive.
    MtpAdaptiveSignal mtp_signal;
    detail::PhysicalResources active_resources;
    detail::PhysicalResources optional_resources;
    bool publish_continuation = true;
    bool allow_memory_reuse = false;
    std::uint64_t memory_host_reservation = 0;
    std::optional<PreparedSessionKey> memory_session_key;
    // The sequence's own output ceiling: the largest frontier its lease may ever cover, so
    // on-demand growth never leases pages the request cannot reach.
    std::uint32_t lease_ceiling = 0;
    // Set when the Device KV lease cannot be extended any further: the request finishes at the
    // frontier its lease covers with its generation limit reason instead of failing a launch on
    // coverage.
    bool lease_settled = false;
    // A settlement caused by pool space rather than the output ceiling, and the entitlement its
    // smallest growth step asked for: retained cache may give those pages back.
    bool lease_space_limited = false;
    DeviceKVPages lease_minimum_target;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<execution::VisionPrefillSession> vision;
        std::vector<CaptureGroup> capture_groups;
        std::size_t next_capture            = 0;
        std::uint64_t pending_capture_offer = 0;
        std::uint32_t base                  = 0;
        std::uint32_t cursor                = 0;
        // A root admitted on a disk-restorable prefix reports that frontier as reused whether or
        // not the restore succeeds; after a failed restore the prefill recomputes [0, frontier)
        // and keeps those tokens out of its reports, so the admitted suffix stays exact.
        std::uint32_t disk_restore_frontier = 0;
        std::uint32_t memory_restore_frontier = 0;
        std::uint64_t memory_restore_generation = 0;
        std::uint32_t hidden_replay_tokens  = 0;
        std::uint32_t prompt_tokens         = 0;
        std::uint32_t initial_mtp_extent    = 0;
        double elapsed_seconds              = 0.0;
        bool prepare_mtp                    = false;
        ReusePath reuse                     = ReusePath::Root;
        MtpBridgeMode mtp_bridge            = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;

    // Returns the request slot to Empty once its lane's resources have been released.
    void retire() noexcept {
        prefill.reset();
        lifecycle            = Lifecycle::Empty;
        pending              = {};
        active_resources     = {};
        optional_resources   = {};
        publish_continuation = true;
        allow_memory_reuse = false;
        memory_host_reservation = 0;
        memory_session_key.reset();
        lease_settled        = false;
        lease_space_limited  = false;
        lease_minimum_target = {};
        lease_ceiling        = 0;
    }
};

class ProgramImpl {
public:
    struct PressureRecoveryScratch {
        struct StatePlacement {
            StateImageHandle state;
            bool device = false;
            bool host   = false;
        };

        struct OwnerProjection {
            const SequenceState* sequence                     = nullptr;
            const SharedPrefixState* shared                   = nullptr;
            const qwen3_5::detail::PressureDecision* decision = nullptr;
            runtime::PlanningOwnerId owner;
        };

        struct CheckpointProjection {
            qwen3_5::CheckpointSummary checkpoint;
            StateImageHandle state;
            bool survives = true;
        };

        std::vector<StatePlacement> state_placements;
        std::vector<OwnerProjection> owners;
        std::vector<CheckpointProjection> checkpoints;
        std::vector<std::optional<runtime::CheckpointRecoveryAlternativeWork>> direct_work;
        qwen3_5::ContinuationSummary continuation_summary;
    };

    ProgramImpl(const execution::Parameters& parameters, const SequencePlanImpl& plan,
                DeviceContext& device, const StartupObserver& startup_observer);
    ~ProgramImpl() noexcept;

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPromptData& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPromptData&& prompt,
                                                  std::uint32_t first_target);
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPromptData& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] std::optional<AdmissionCandidate> seal_materialization(
        const AdmissionCandidate& admission, const PreparedPromptData& prompt,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] std::unique_ptr<CapturePressureCandidateImpl>
    make_capture_physical_candidate(const CaptureAssessment& assessment) const;
    void select_shared_captures(AdmissionCandidate& candidate, const PreparedPromptData& prompt,
                                std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                      const PreparedPromptData& prompt,
                                      std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::PreflightStatus
    revalidate_materialization(const AdmissionCandidate& plan,
                               const PreparedPromptData& prompt) const;
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_materialization(AdmissionCandidate&& plan, PreparedPromptData&& prompt,
                            runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool
    persistent_backfill_safe(const RequestBasePlan& blocked_head,
                             const AdmissionCandidate& candidate,
                             std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    // True while this sequence waits for a media item submitted to a concurrent overlay window:
    // the lane must not be given a prefill unit, and every other lane keeps running.
    [[nodiscard]] bool vision_pending(SequenceHandle sequence) const noexcept;
    void update_sampling(SequenceHandle sequence, const ResolvedSamplingParameters& sampling);
    [[nodiscard]] bool wait_context_transfer() noexcept;
    [[nodiscard]] bool try_claim_seal_window() noexcept;
    void release_seal_window() noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressureCandidate&& pressure, runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    [[nodiscard]] std::optional<qwen3_5::PhysicalUsageSnapshot>
    fail_all_cleanup(ProgramCleanup cleanup) noexcept;
    [[nodiscard]] bool context_stores_idle() const noexcept;
    [[nodiscard]] bool rebuild_context_stores() noexcept;
    [[nodiscard]] detail::PhysicalResources admission_capacity() const noexcept;
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] bool temporary_request_feasible(const RequestBasePlan& base) const noexcept;

    [[nodiscard]] bool hybrid_prefix_cache() const noexcept { return hybrid_ != nullptr; }

    [[nodiscard]] HybridAdmissionQuote hybrid_quote(const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base,
                                                    runtime::LaneId destination);
    [[nodiscard]] bool hybrid_reservable(const HybridAdmissionQuote& quote,
                                         runtime::CancellationFlagView cancellation) const noexcept;
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    hybrid_reserve_materialization(HybridAdmissionQuote&& quote, const PreparedPromptData& prompt,
                                   const std::function<PreparedPromptData()>& take_prompt,
                                   runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::uint32_t hybrid_reclaim_device_kv(std::uint32_t main_pages,
                                                         std::uint32_t backend_pages);
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
    // Installs the Engine's calibrated machine model for hybrid admission and eviction.
    void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost);

    void set_hybrid_coalesce_wait_limit(double seconds) noexcept {
        hybrid_coalesce_wait_seconds_ = seconds > 0.0 ? seconds : 0.0;
    }

    [[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(const std::filesystem::path& path,
                                                                  std::string fingerprint);

    [[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const {
        return hybrid_shutdown_save_;
    }

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return resource_revision_;
    }

    [[nodiscard]] qwen3_5::PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] std::optional<std::uint32_t>
    device_kv_lease_settlement_tokens(SequenceHandle sequence,
                                      std::uint32_t forced_span_tokens) const noexcept;
    [[nodiscard]] std::optional<DeviceKVLeaseShortfall>
    device_kv_lease_shortfall(SequenceHandle sequence) const noexcept;
    [[nodiscard]] bool resume_device_kv_lease(SequenceHandle sequence) noexcept;
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const ContinuationHandle& continuation) const noexcept;
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const SharedPrefixHandle& shared) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    friend struct qwen3_5::detail::PressurePlanningSessionImpl;
    friend class WindowMemoryBackend;
    friend class WindowMemoryTransfer;

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t kvmem_window_tokens;
    const KvmemOptions kvmem_options;
    const std::uint32_t max_concurrency;
    // Frozen context-cache shape: an engaged host budget has already resolved host_state_slots,
    // host_kv_capacity_bytes and the long-anchor count on the plan this Program was built from.
    const ContextCacheOptions context_cache;
    const std::uint32_t continuation_capacity;
    const std::uint32_t shared_prefix_capacity;
    const std::uint32_t prefill_chunk;
    const bool fast_prefill_kernel;
    const std::uint32_t draft_window;
    const std::uint32_t lookup_ngram;
    const MtpDraftPolicy mtp_policy;
    // Copy proposals verified alongside the neural drafter; zero disables them.
    const std::uint32_t ngram_draft_window;
    const std::uint32_t ngram_min_match;

    // The widest round the decode frame verifies: the draft window, or the copy window when wider.
    [[nodiscard]] std::uint32_t widest_verify_window() const noexcept {
        return std::max(draft_window, ngram_draft_window);
    }
    const SpeculativeBackend speculative_backend;
    const KvCacheStorage kv_storage;
    const ProposalHead proposal_head;
    const ops::RopeYarn rope_yarn;
    const std::uint32_t mtp_attention_window;

    // A checkpoint captured under one execution profile (backend, proposal head, KV coding) must
    // never be replayed under another. Capture and reuse-lookup derive the tag from this one place:
    // a divergent copy is what once made every stored rk8v4 checkpoint carry a tag no lookup could
    // produce, so prefix reuse missed unconditionally under that profile.
    [[nodiscard]] std::uint32_t capture_identity_tag() const noexcept {
        return static_cast<std::uint32_t>(speculative_backend) |
               (static_cast<std::uint32_t>(proposal_head) << 8U) |
               (static_cast<std::uint32_t>(kv_storage) << 16U);
    }
    const bool vision_enabled;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    // Free device memory CUDA Graph preparation consumed at startup (instantiate, upload and one
    // launch of every executable), for comparison with graph_allowance_bytes.
    std::size_t graph_measured_bytes = 0;
    const WorkspacePlan workspace_plan;

    // Overlay Vision residency only: the persistent arena is VMM-backed so free KV granules can be
    // lent to a Vision window. Null otherwise, where `persistent` owns a plain allocation.
    std::unique_ptr<EvictableKVPool> kv_arena;
    DeviceArena persistent;
    // Pipeline stages only: the persistent state of each further device -- its layers' KV planes,
    // block-table copy and recurrent state -- allocated in that device's own memory.
    std::vector<DeviceArena> persistent_by_rank;
    DeviceArena workspace_storage;
    // Pipeline stages only: scratch for the ranks past the primary device, each allocated in its
    // own card's memory. `work` borrows a slice of each and switches between them as the layer
    // loop walks stages, so every existing workspace call site keeps using one arena object.
    std::vector<DeviceArena> workspace_storage_by_rank;
    WorkspaceArena work;
    std::unique_ptr<qwen3_5::DecoderState> decoder;
    std::unique_ptr<HostKVArena> host_kv_arena;
    std::unique_ptr<LogicalKVPageStore> text_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_kv_addresses;
    std::unique_ptr<HostKVExtentStore> host_kv_extents;
    std::size_t text_host_kv_page_stride    = 0;
    std::size_t backend_host_kv_page_stride = 0;
    std::unique_ptr<qwen3_5::StateImageDevicePool> state_images;
    // Only when the model is split over several devices: the state shards and the links between
    // stages that a forward pass crosses.
    std::unique_ptr<execution::StageRuntime> stage_runtime;
    std::unique_ptr<qwen3_5::HostStatePool> host_state_images;
    std::unique_ptr<StateImageStore> state_store;
    // Disk (L3) tier, null unless configured. Its staging lives in its own pinned arena and state
    // pool, so the tier takes nothing from the Host tier's capacity. The bridge is declared after
    // everything its queued writes borrow, so it drains them before that memory is released.
    std::unique_ptr<HostKVArena> disk_staging_arena;
    std::optional<HostKVAllocation> disk_main_staging;
    std::optional<HostKVAllocation> disk_backend_staging;
    std::unique_ptr<qwen3_5::HostStatePool> disk_state_staging_pool;
    std::optional<qwen3_5::HostStateSlotHandle> disk_state_staging;
    std::unique_ptr<DiskKVBridge> disk_kv;
    bool disk_kv_restore = false;
    // ReplaySSM records and their fold, for the first state shard; the rest, on other devices, are
    // in the `extra_` vectors.
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::vector<std::unique_ptr<GdnReplayRecords>> extra_replay_records;
    std::vector<std::unique_ptr<ops::GdnReplayFoldPlan>> extra_replay_fold;
    std::optional<DFlashPersistentState> dflash;
    qwen3_5::RoundState io;
    Tensor prefill_hidden;
    std::optional<execution::MemoryStatistics> memory_statistics;
    std::optional<execution::MemoryStatistics> memory_candidate_statistics;
    std::unique_ptr<PinnedHostBuffer> memory_candidate_host;
    std::unique_ptr<PinnedHostBuffer> memory_statistics_host;
    // Reused transfer staging is bounded by the physical window. Durable history
    // uses ordinary host RAM, so pinned allocation count does not grow with H.
    std::unique_ptr<PinnedHostBuffer> memory_kv_staging;
    std::array<std::int32_t, 4> memory_statistics_ranges{};
    // Inactive physical Host histories. Keys follow frontend ContextCacheHints;
    // exact checkpoint identity still decides whether any bytes can be reused.
    std::array<std::optional<KvmemWindowState>, 16> memory_histories;
    struct MemoryPublication {
        std::optional<PreparedSessionKey> key;
        std::uint64_t order = 0;
    };
    // Bounded high-watermarks also cover newer requests that finish without
    // retaining a checkpoint, preventing an older concurrent result publishing.
    std::array<MemoryPublication, 32> memory_publications;
    std::uint64_t memory_history_hits = 0, memory_history_misses = 0, memory_history_evictions = 0;
    std::uint64_t memory_disk_hits = 0, memory_disk_writes = 0, memory_disk_errors = 0;

    std::optional<Tensor> score_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    std::unique_ptr<qwen3_5::StructuredRound> structured_round;

    std::vector<SequenceState> continuation_states;
    std::vector<ContinuationSlot> continuation_slots;
    std::vector<SharedPrefixState> shared_prefix_states;
    std::vector<SharedPrefixSlot> shared_prefix_slots;
    std::array<std::uint32_t, kMaximumConcurrency> active_continuations{};
    // Overlay Vision residency only: the one-window broker and the per-lane pinned result slots.
    // Declared before `requests`, whose Vision sessions borrow both.
    std::optional<execution::VisionResidencyBroker> vision_broker;
    std::optional<execution::PinnedResultPool> vision_results;
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;
    DecodeGraphFamily ngram_graphs;

    std::optional<PinnedHostBuffer> round_host;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    // The logits behind a request's first token, copied when the request reports log
    // probabilities.
    std::optional<PinnedHostBuffer> first_token_logits_host;
    // Present under adaptive MTP: picks each round's verification width.
    std::optional<MtpAdaptiveBatchController> mtp_controller;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_5::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_5::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_5::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_5::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_5::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_5::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    std::size_t vision_handoff_peak_bytes    = 0;

private:
    void advance_resource_revision() noexcept {
        if (++resource_revision_.value == 0) { ++resource_revision_.value; }
    }

    runtime::ProgramResourceRevision resource_revision_{.value = 1};
    std::uint32_t pressure_planning_generation_ = 0;
    bool pressure_planning_active_              = false;

    struct PressurePageScratchSlot {
        std::uint32_t generation     = 0;
        std::uint32_t selected_index = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t host_group     = 0;
        bool projected               = false;
        bool device                  = false;
        bool host                    = false;
        bool pressure_targeted       = false;
    };

    struct PressureSelectedPage {
        LogicalKVPageHandle page;
        std::uint32_t references = 0;
    };

    struct PressureSelectedState {
        StateImageHandle state;
        bool device = false;
        bool host   = false;
    };

    mutable std::uint32_t pressure_page_scratch_generation_ = 0;
    mutable std::vector<PressurePageScratchSlot> pressure_text_page_scratch_;
    mutable std::vector<PressurePageScratchSlot> pressure_backend_page_scratch_;
    mutable std::vector<PressureSelectedPage> pressure_text_selected_pages_;
    mutable std::vector<PressureSelectedPage> pressure_backend_selected_pages_;
    mutable std::vector<std::uint8_t> pressure_private_owner_scratch_;
    mutable std::vector<std::uint8_t> pressure_shared_owner_scratch_;
    mutable std::vector<std::vector<runtime::CheckpointRef>> pressure_private_drop_scratch_;
    mutable std::vector<PressureSelectedState> pressure_state_scratch_;

    void begin_pressure_page_scratch() const noexcept;
    [[nodiscard]] PressurePageScratchSlot& pressure_page_scratch(const LogicalKVPageStore& store,
                                                                 LogicalKVPageHandle page) const;
    [[nodiscard]] const PressurePageScratchSlot*
    find_pressure_page_scratch(const LogicalKVPageStore& store, LogicalKVPageHandle page) const;

    struct MaterializationSourceProtection {
        struct StateOwnershipCandidate {
            StateImageHandle state;
            std::uint32_t source_checkpoint_references = 0;
        };

        std::optional<std::uint32_t> private_source_index;
        bool consumed_private_source = false;
        std::optional<StateImageHandle> state;
        std::uint32_t consumed_state_references = 0;
        bool state_fork_required                = false;
        std::vector<StateOwnershipCandidate> state_ownership_candidates;
        std::optional<KVAddressSpaceHandle> text;
        std::uint32_t text_pages          = 0;
        std::uint32_t text_transfer_pages = 0;
        bool text_prefix_fork_required    = false;
        std::optional<KVAddressSpaceHandle> backend;
        std::uint32_t backend_pages          = 0;
        std::uint32_t backend_transfer_pages = 0;
        bool backend_prefix_fork_required    = false;
    };

    struct PendingTransaction {
        std::uint64_t id = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::uint64_t, kMaximumConcurrency> epochs{};
        std::size_t size = 0;
    };

    std::optional<PendingTransaction> pending_transaction_;
    std::uint64_t next_transaction_id_ = 1;

    // Serializes the materialization seal window (final assess -> seal) so a concurrent
    // demote cannot steal the incumbent's allocation and bump a victim's slot generation
    // between assess and seal. Claimed by the planner, released after seal (success or
    // failure) or on the planner's early exit.
    std::atomic<bool> seal_window_claimed_ = false;

    enum class PressureTransitionPhase : std::uint8_t {
        HostReleases,
        CopyPreparation,
        CopiesInFlight,
        CopyPublication,
        Committed,
    };

    struct PressureTransition {
        PressureTransitionPhase phase = PressureTransitionPhase::HostReleases;
        std::array<TransferWork, 3> transfer_work{};
        std::array<std::uint32_t, 3> transfer_pages{};
        std::uint64_t state_images = 0;
        std::uint8_t timer_mask    = 0;
    };

    // An evicted owner's disk write, resumable across transaction steps: the KV prefix chains, the
    // endpoint StateImage, and the tail page and StateImage of its rewrite seam and earliest long
    // anchors. Items are staged a batch at a time through the tier's own staging memory.
    struct DiskSpillItem {
        DiskKVKind kind         = DiskKVKind::MainKV;
        std::uint32_t frontier  = 0; // the digest frontier that keys the item
        std::uint32_t page      = 0; // KV row page for KV items
        StateImageHandle state;
    };

    struct DiskOwnerSpill {
        std::vector<DiskSpillItem> items;
        std::size_t next = 0;
        DiskKVKind batch_kind = DiskKVKind::MainKV;
        std::vector<DiskKVIdentity> batch_ids;
        std::vector<std::span<const std::byte>> batch_bytes;
        std::size_t batch_submitted = 0;
        std::vector<SpillTicket> tickets;
        std::uint64_t written      = 0;
        std::uint64_t deduplicated = 0;
        bool stopped               = false;
        std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
    };

    struct MaterializationTransaction {
        struct KVRestorePage {
            LogicalKVPageHandle logical;
            HostKVExtentCapability extent;
            std::uint32_t extent_page = 0;
        };

        struct PressureWork {
            struct StateChangeWork {
                std::optional<StateImageTransfer> transfer;
                bool host_released = false;
            };

            struct KVChangeWork {
                std::vector<LogicalKVPageHandle> pages;
                std::vector<DeviceKVPageHandle> sources;
                std::optional<HostKVExtentReservation> backup;
                bool host_released = false;
            };

            qwen3_5::detail::PressureDecision option;
            std::uint32_t continuation_index      = 0;
            std::uint64_t continuation_generation = 0;
            bool shared_owner                     = false;
            std::vector<StateChangeWork> state_changes;
            std::vector<KVChangeWork> main_kv_changes;
            std::vector<KVChangeWork> backend_kv_changes;
            detail::PhysicalDelta committed_delta;
            bool submitted                 = false;
            bool completed                 = false;
            bool checkpoint_drop_published = false;
            bool mutation_published        = false;
            std::uint64_t spill_pages      = 0;
            std::optional<DiskOwnerSpill> disk_spill;
        };

        std::uint64_t id = 0;
        runtime::LaneId destination;
        bool has_source                        = false;
        bool has_shared_source                 = false;
        runtime::PrivateSourceMode source_mode = runtime::PrivateSourceMode::ConsumeToActive;
        std::uint32_t source_index             = 0;
        std::uint64_t source_generation        = 0;
        std::uint32_t shared_source_index      = 0;
        std::uint64_t shared_source_generation = 0;
        std::optional<MaterializationSourceResult> source_result;
        std::optional<MaterializationSharedSourceResult> shared_source_result;
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<bool> victim_released;
        std::vector<PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::size_t pressure_cursor = 0;
        std::size_t victim_count    = 0;
        // Host releases resume here while an evicted owner is still being written to disk.
        std::size_t host_release_cursor        = 0;
        std::size_t shared_host_release_cursor = 0;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<bool> shared_victim_released;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        std::vector<PressureWork> shared_pressure;
        std::size_t shared_pressure_cursor = 0;
        std::size_t shared_victim_count    = 0;
        PressureTransition pressure_transition;
        std::optional<AdmissionCandidate> plan;
        std::optional<std::uint32_t> root_continuation_index;
        bool root_waiting_for_victim = false;
        std::array<StateImageHandle, 2> reserved_states{};
        std::size_t reserved_state_count = 0;
        std::optional<StateImageHandle> state_fork_destination;
        std::optional<KVAddressSpaceHandle> root_text_address;
        std::optional<KVAddressSpaceHandle> root_backend_address;
        std::optional<KVActivationReservation> text_activation;
        std::optional<KVActivationReservation> backend_activation;
        std::optional<DeviceKVPageReservation> text_source_restore_reservation;
        std::optional<DeviceKVPageReservation> backend_source_restore_reservation;
        std::optional<KVPrefixForkReservation> text_prefix_fork;
        std::optional<KVPrefixForkReservation> backend_prefix_fork;
        std::optional<LogicalKVPageHandle> text_retained_tail;
        std::optional<LogicalKVPageHandle> backend_retained_tail;
        std::optional<HostKVExtentReservation> text_retained_tail_backup;
        std::optional<HostKVExtentReservation> backend_retained_tail_backup;
        std::optional<std::uint32_t> text_activation_frontier;
        std::optional<std::uint32_t> backend_activation_frontier;
        std::optional<StateImageTransfer> state_restore;
        bool split_state_identity = false;
        std::vector<KVRestorePage> text_restores;
        std::vector<DeviceKVPageHandle> text_restore_destinations;
        std::vector<KVRestorePage> backend_restores;
        std::vector<DeviceKVPageHandle> backend_restore_destinations;
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        bool state_restored                 = false;
        bool transfer_submitted             = false;
        std::uint8_t transfer_timer_mask    = 0;
        bool prefix_tail_submitted          = false;
        bool retained_tail_backup_submitted = false;
        bool prefix_forks_ready             = false;
        bool source_prepared                = false;
        bool cancel_pending                 = false;
        bool prepared                       = false;
        bool terminal                       = false;
    };

    std::uint64_t next_materialization_id_ = 1;
    // Every rank's compute and transfer stream. Context transactions fan their copies out across
    // ranks and fence on all of them.
    RankStreams compute_streams;
    RankStreams transfer_streams;
    RankFenceSet context_source_ready_;
    RankFenceSet context_completion_;
    std::vector<TokenId> materialization_ledger_;
    qwen3_5::detail::ResidentPrefixIdentity materialization_identity_;
    qwen3_5::detail::PrefixShortlistDigests materialization_prefix_digests_;

    struct ActiveCaptureTransaction {
        std::uint64_t id         = 0;
        std::uint32_t lane       = 0;
        std::uint64_t lane_epoch = 0;
        CaptureGroup group;
        bool publish_private = false;
        bool publish_shared  = false;
        bool replaces_shared = false;
        std::optional<runtime::CheckpointRef> private_replacement;
        std::optional<std::uint32_t> shared_index;
        std::uint64_t replacement_generation = 0;
        StateImageHandle source_state;
        StateImageHandle destination_state;
        qwen3_5::CaptureStatePlacement state_placement = qwen3_5::CaptureStatePlacement::DeviceFork;
        std::optional<StateImageTransfer> state_snapshot;
        std::optional<KVAddressSpaceHandle> active_text_destination;
        std::optional<KVAddressSpaceHandle> active_backend_destination;
        std::optional<KVActiveSnapshotReservation> text_snapshot;
        std::optional<KVActiveSnapshotReservation> backend_snapshot;
        detail::PhysicalDelta resource_delta;
        detail::PhysicalDelta active_entitlement_delta;
        detail::PhysicalResources capacity_preparation_removed;
        ContinuationSummary active_summary;
        std::vector<runtime::ContextTransferRequirement> transfer_requirements;
        std::vector<runtime::ContextTransferObservation> transfer_observations;
        runtime::ContextOperationCounts operations;
        std::vector<std::uint32_t> victim_indices;
        std::vector<std::uint64_t> victim_generations;
        std::vector<MaterializationTransaction::PressureWork> pressure;
        std::vector<MaterializationVictimResult> pressure_results;
        std::vector<std::uint32_t> shared_victim_indices;
        std::vector<std::uint64_t> shared_victim_generations;
        std::vector<MaterializationTransaction::PressureWork> shared_pressure;
        std::vector<MaterializationSharedVictimResult> shared_pressure_results;
        PressureTransition pressure_transition;
        bool recycles_private_state        = false;
        bool replacement_removed           = false;
        bool prepared                      = false;
        std::uint64_t recycled_state_epoch = 0;
        bool transfer_enqueue_pending      = false;
        bool transfer_submitted            = false;
        std::uint8_t transfer_timer_mask   = 0;
        bool published                     = false;
    };

    std::uint64_t next_capture_offer_id_ = 1;

    struct HybridMaterializationTransaction {
        std::shared_ptr<HybridQuoteImpl> quote;
        PreparedPromptData prompt;
        // Staged by the first progress step: the pinned path and snapshot, the Device pages the
        // restore and the fork consume, the reserved StateImage destination and whether Host
        // slabs filled it. A Host restore in flight keeps the transaction in progress.
        bool staged          = false;
        bool snapshot_pinned = false;
        bool state_restored  = false;
        bool terminal        = false;
        std::vector<runtime::prefix_cache::NodeRef> path;
        std::vector<std::uint64_t> hashes;
        std::vector<std::uint64_t> extras;
        std::optional<StateImageHandle> state;
        std::optional<DeviceKVPageReservation> text_pages;
        std::optional<DeviceKVPageReservation> backend_pages;
        std::uint64_t restore_bytes = 0;
        // Names the landing Host restore whose per-layer events the lane's first prefill pass
        // waits on (0 without one).
        std::uint64_t restore_ticket = 0;
    };

    using ContextTransaction =
        std::variant<std::monostate, MaterializationTransaction, ActiveCaptureTransaction,
                     HybridMaterializationTransaction>;
    ContextTransaction context_transaction_;

    [[nodiscard]] MaterializationResult
    progress_materialization_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] ActiveCaptureResult
    progress_active_capture_transaction(runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture_impl(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                                const SharedPrefixHandle* replacement,
                                std::optional<runtime::CheckpointRef> private_replacement,
                                bool permit_shared_publication,
                                std::optional<CapturePressureCandidate> pressure,
                                runtime::CancellationFlagView cancellation);

    std::array<CudaEventTimer, 3> context_transfer_timers_;

    // Hybrid prefix cache (null in Legacy mode).
    std::unique_ptr<HybridPrefixCache> hybrid_;
    std::array<HybridLaneState, kMaximumConcurrency> hybrid_lanes_;
    runtime::prefix_cache::CacheCostModel hybrid_cost_;
    double hybrid_coalesce_wait_seconds_ = 0.0;
    std::filesystem::path hybrid_file_;
    std::string hybrid_fingerprint_;
    std::optional<HybridCachePersistence> hybrid_shutdown_save_;

    void create_hybrid_prefix_cache(const StartupObserver& observer);
    // The per-layer events the lane's first prefill pass waits on, consumed by this call.
    [[nodiscard]] std::span<const cudaEvent_t> hybrid_take_restore_layers(std::uint32_t lane);
    // True when a sibling lane still prefilling a prompt that shares more with this one than the
    // cache offers will publish a snapshot where they diverge soon enough to wait for. Plans that
    // snapshot as an exact tap of the sibling when none is planned near the divergence.
    [[nodiscard]] bool hybrid_await_sibling(const PreparedPromptData& prompt, std::uint32_t reuse,
                                            std::uint32_t destination);
    // Writes the Host tier to the attached file once every Host write has landed. Called by the
    // shutdown cleanup after the lanes wrote their blocks through.
    void save_hybrid_cache_for_shutdown() noexcept;
    [[nodiscard]] MaterializationResult
    progress_hybrid_materialization(runtime::CancellationFlagView cancellation);
    // Pins the quoted path and snapshot, reserves every Device page the admission needs and
    // submits the Host restores its source requires. Returns false, with nothing staged left
    // behind by the caller's abort, when the quote went stale or the pools cannot supply it.
    [[nodiscard]] bool hybrid_stage(HybridMaterializationTransaction& transaction,
                                    const PreparedPromptData& prompt);
    // Builds the lane from the staged, Device-resident source.
    [[nodiscard]] StartResult hybrid_activate(HybridMaterializationTransaction& transaction);
    void hybrid_abort_materialization(HybridMaterializationTransaction& transaction) noexcept;
    void hybrid_prompt_keys(const PreparedPromptData& prompt, std::vector<std::uint64_t>& hashes,
                            std::vector<std::uint64_t>& extras) const;
    [[nodiscard]] bool hybrid_make_room(std::uint32_t text_pages, std::uint32_t backend_pages);
    // The backend KV frontier restored with a snapshot at `frontier` (MTP trails by one token).
    [[nodiscard]] std::uint32_t hybrid_backend_frontier(std::uint32_t frontier) const noexcept;
    // Inserts every newly committed full block of the lane's sequence into the tree, then
    // publishes the pending taps those blocks complete.
    void hybrid_publish_blocks(SequenceState& sequence);
    // Snapshots the lane's committed state at the prefill frontier `frontier`.
    void hybrid_capture_tap(SequenceState& sequence, std::uint32_t frontier);
    // Realizes the planned taps a completed prefill chunk reached.
    void hybrid_after_prefill_chunk(SequenceState& sequence, std::uint32_t cursor,
                                    std::uint32_t prompt_tokens);
    // Publishes pending taps whose blocks are committed; a finishing lane hands its own last
    // pages to the remaining ones or drops them.
    void hybrid_publish_pending(SequenceState& sequence, bool finishing);
    // Copies a tail bundle into cache-owned pages; absent when no Device page can be freed.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_copy_tail(const HybridBlockPages& source,
                                                                std::uint32_t columns);
    // Terminal publication: committed blocks and, when useful, an endpoint snapshot. Then the
    // lane's sequence is released. Returns false when the lane could not be released strictly.
    [[nodiscard]] bool hybrid_finish_lane(SequenceState& sequence, RequestControl& request,
                                          std::uint32_t lane, bool endpoint) noexcept;
    // Drops the lane's index pins. Safe on any lane state.
    void hybrid_release_lane(std::uint32_t lane) noexcept;

    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_lane(std::uint32_t lane, const PreparedPromptData& prompt, const RequestBasePlan& base,
                 const SequenceState* source, const SharedPrefixState* shared_source,
                 std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] StartResult start_request(MaterializationTransaction& transaction);
    void prepare_materialization(MaterializationTransaction& transaction);
    void enqueue_materialization_transfers(MaterializationTransaction& transaction);
    void record_materialization_transfer_observations(MaterializationTransaction& transaction);
    void publish_materialization_transfers(MaterializationTransaction& transaction);
    void prepare_prefix_forks(MaterializationTransaction& transaction);
    void prepare_consumed_source(MaterializationTransaction& transaction);
    void abort_materialization_transfers(MaterializationTransaction& transaction) noexcept;
    void prepare_pressure_bookkeeping(MaterializationTransaction::PressureWork& work);
    void prepare_pressure_work(MaterializationTransaction::PressureWork& work,
                               runtime::ContextResourceClass resource);
    void publish_pressure_host_releases(MaterializationTransaction::PressureWork& work);
    void publish_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void synchronize_transfer_streams() const;
    void abort_pressure_work(MaterializationTransaction::PressureWork& work) noexcept;
    void start_context_transfer_timer(runtime::ContextResourceClass resource);
    void stop_context_transfer_timer(runtime::ContextResourceClass resource);
    [[nodiscard]] runtime::ContextTransferObservation context_transfer_observation(
        runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
        TransferWork work, std::uint32_t page_count = 0, std::uint64_t state_images = 1) const;

    struct PhysicalReleaseResult {
        runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
        detail::PhysicalDelta delta;
    };

    [[nodiscard]] PhysicalReleaseResult
    release_materialization_victim(MaterializationTransaction& transaction, std::size_t position);
    void start_sequence(std::uint32_t lane, SequenceState& sequence,
                        MaterializationTransaction& transaction);
    void release_materialization_staging(MaterializationTransaction& transaction) noexcept;
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_raw(std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
               runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_prefill_raw(std::uint32_t lane, bool terminal, runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming resolve_pending_raw(
        std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
        std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
        std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
        runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] bool valid_sequence(SequenceHandle handle) const noexcept;
    [[nodiscard]] bool valid_continuation(const ContinuationHandle& handle) const noexcept;
    [[nodiscard]] bool valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept;
    [[nodiscard]] bool valid_capture_offer(const CaptureOffer& offer) const noexcept;
    [[nodiscard]] bool materialization_pins(std::uint32_t index,
                                            std::uint64_t generation) const noexcept;
    [[nodiscard]] bool has_unsettled_state_fork() const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch& pending) const noexcept;
    // Per-owner resources are intentionally distinct from global physical occupancy: an aliased
    // allocation contributes only when removing this owner would release it.
    [[nodiscard]] detail::PhysicalResources
    owner_exclusive_resources(const SequenceState& sequence) const;
    [[nodiscard]] detail::PhysicalResources
    owner_exclusive_resources(const SharedPrefixState& shared) const;
    // What an active snapshot of this sequence moves out of its exclusive ownership: every full
    // page it alone references (Device page and any Host replica).
    [[nodiscard]] detail::PhysicalResources
    active_snapshot_shared_resources(const SequenceState& sequence) const;

    // Releasing an owner can leave pages or checkpoints it shared with an active sequence
    // referenced by that sequence alone. Their ownership then moves into the active lineage, so
    // the active entitlement grows by exactly what became exclusive. A release site takes the
    // baseline first and credits after; lanes whose continuation changed in between are skipped.
    struct ActiveExclusiveEntry {
        std::uint32_t continuation = 0;
        detail::PhysicalResources resources;
    };

    using ActiveExclusiveBaseline =
        std::array<std::optional<ActiveExclusiveEntry>, kMaximumConcurrency>;
    [[nodiscard]] ActiveExclusiveBaseline active_exclusive_baseline() const noexcept;
    void credit_active_ownership_transfers(const ActiveExclusiveBaseline& baseline) noexcept;
    [[nodiscard]] detail::PhysicalResources physical_occupancy() const noexcept;
    [[nodiscard]] bool physical_peak_fits(detail::PhysicalResources peak) const noexcept;
    [[nodiscard]] StateImageHandle
    selected_state(const SequenceState& sequence, ReusePath reuse,
                   std::optional<runtime::CheckpointRef> checkpoint) const;
    [[nodiscard]] std::uint32_t
    selected_state_consumed_references(const SequenceState& sequence, ReusePath reuse,
                                       RewriteCheckpointDisposition rewrite_disposition,
                                       std::optional<runtime::CheckpointRef> checkpoint,
                                       std::uint32_t reuse_base) const;
    [[nodiscard]] bool
    selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                 RewriteCheckpointDisposition rewrite_disposition,
                                 std::optional<runtime::CheckpointRef> checkpoint,
                                 std::uint32_t reuse_base) const;
    [[nodiscard]] bool can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                     const RewriteCheckpointSpec& desired,
                                                     const SequenceState& sequence, ReusePath reuse,
                                                     std::uint32_t reuse_base) const;
    [[nodiscard]] std::uint32_t device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                              KVAddressSpaceHandle address,
                                                              std::uint32_t frontier) const;
    [[nodiscard]] bool partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                                 KVAddressSpaceHandle address,
                                                 std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t
    missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                          KVAddressSpaceHandle address,
                                          std::uint32_t frontier) const;
    [[nodiscard]] std::size_t host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                                   KVAddressSpaceHandle address,
                                                   std::uint32_t frontier) const noexcept;
    [[nodiscard]] qwen3_5::CheckpointSummary
    checkpoint_summary(const SequenceState& sequence, runtime::CheckpointRef checkpoint,
                       StateImageHandle state, runtime::PrefillWork rebuild_work) const;
    [[nodiscard]] qwen3_5::ContinuationSummary
    continuation_summary(const SequenceState& sequence) const;
    void populate_continuation_summary(const SequenceState& sequence,
                                       qwen3_5::ContinuationSummary& summary) const;
    [[nodiscard]] qwen3_5::SharedPrefixSummary
    shared_prefix_summary(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<MaterializationSourceProtection>
    materialization_source_protection(const ResourceCandidateState& candidate) const;
    [[nodiscard]] detail::PhysicalResources
    materialization_deficit(const ResourceCandidateState& candidate) const;
    [[nodiscard]] detail::PhysicalResources
    guided_materialization_deficit(const ResourceCandidateState& candidate,
                                   const detail::PhysicalDelta& pressure) const;
    [[nodiscard]] bool
    protected_materialization_page(const MaterializationSourceProtection* protection,
                                   const KVAddressSpaceStore& addresses, std::uint32_t page_offset,
                                   LogicalKVPageHandle page, bool backend) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision>
    inspect_pressure_option(const SequenceState& sequence, detail::PhysicalResources deficit,
                            const MaterializationSourceProtection* protection           = nullptr,
                            const qwen3_5::TargetKVRequirement* retained_requirement    = nullptr,
                            std::span<const runtime::CheckpointRef> dropped_checkpoints = {},
                            std::span<const StateImageHandle> released_states           = {},
                            const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision>
    inspect_pressure_successors(const SequenceState& sequence, detail::PhysicalResources residual,
                                const MaterializationSourceProtection* protection,
                                const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision> inspect_shared_pressure_successors(
        const SharedPrefixState& shared, detail::PhysicalResources residual,
        const MaterializationSourceProtection* protection,
        const qwen3_5::detail::PressureDecision* current = nullptr) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision> inspect_shared_pressure_option(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_5::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] std::vector<qwen3_5::detail::PressureDecision> inspect_shared_pressure_options(
        const SharedPrefixState& shared, detail::PhysicalResources deficit,
        const MaterializationSourceProtection* protection = nullptr,
        const qwen3_5::detail::PressureDecision* current  = nullptr) const;
    [[nodiscard]] qwen3_5::detail::PressureDecision
    inspect_eviction_option(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_5::detail::PressureDecision
    inspect_shared_eviction_option(const SharedPrefixState& shared) const;
    [[nodiscard]] std::optional<qwen3_5::detail::PressureDecision>
    inspect_checkpoint_drop_option(const SequenceState& sequence,
                                   std::span<const runtime::CheckpointRef> checkpoints) const;
    [[nodiscard]] bool
    pressure_decision_valid(const SequenceState& sequence,
                            const qwen3_5::detail::PressureDecision& decision,
                            const MaterializationSourceProtection* protection) const;
    [[nodiscard]] bool
    shared_pressure_decision_valid(const SharedPrefixState& shared,
                                   const qwen3_5::detail::PressureDecision& decision,
                                   const MaterializationSourceProtection* protection) const;
    [[nodiscard]] std::vector<runtime::ContextTransferRequirement>
    checkpoint_restore_requirements(const SequenceKVBundle& kv,
                                    const qwen3_5::TargetKVRequirement& requirement,
                                    StateImageHandle state) const;
    [[nodiscard]] bool pressure_checkpoint_recovery_impacts(
        const ResourceCandidateState& candidate,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const qwen3_5::detail::PressureDecision* const> private_decisions,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_decisions,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids,
        std::vector<qwen3_5::detail::PressureCheckpointRecoveryProjection>& output,
        std::vector<runtime::CheckpointRecoveryAlternativeWork>& alternatives,
        PressureRecoveryScratch& scratch, std::uint64_t& projection_work) const;
    void publish_checkpoint_drop(SequenceState& sequence, runtime::CheckpointRef checkpoint);
    // Disk tier (storage/disk_tier.cpp).
    void open_disk_tier(const SequencePlanImpl& plan);
    [[nodiscard]] DiskKVIdentity disk_identity(const qwen3_5::detail::PrefixShortlistDigests& digests,
                                               std::uint32_t frontier) const;
    [[nodiscard]] DiskOwnerSpill plan_owner_spill(const SequenceState& sequence) const;
    // One bounded step of an owner's disk write; true once every item is written or skipped.
    [[nodiscard]] bool progress_owner_spill(const SequenceState& sequence, DiskOwnerSpill& spill);
    [[nodiscard]] static bool owner_spill_in_flight(const DiskOwnerSpill& spill) noexcept;
    // Writes an owner released outside a materialization; what is not queued by the deadline stays
    // unwritten and costs a recompute later.
    void spill_owner_to_disk(const SequenceState& sequence,
                             std::chrono::steady_clock::time_point deadline) noexcept;
    void spill_released_owner(const SequenceState& sequence) noexcept;
    void flush_disk_tier() noexcept;
    [[nodiscard]] std::optional<std::uint32_t>
    disk_restorable_frontier(const RequestBasePlanImpl& base, const PreparedPromptData& prompt,
                             std::uint32_t prompt_tokens) const;
    [[nodiscard]] bool restore_prefix_from_disk(SequenceState& sequence,
                                                RequestControl::Prefill& staged,
                                                std::uint32_t frontier);
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t> lanes,
                                            const runtime::BatchedGeneratedRound& round);
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    [[nodiscard]] std::optional<std::uint32_t> allocate_continuation_slot() noexcept;
    [[nodiscard]] bool can_release_continuation_slot_strict(std::uint32_t index) const;
    void release_continuation_slot_strict(std::uint32_t index,
                                          bool written_to_disk = false) noexcept;
    void release_continuation_slot_best_effort(std::uint32_t index) noexcept;
    void retire_continuation_slot(std::uint32_t index) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept;
    [[nodiscard]] bool can_clear_lane_strict(const SequenceState& sequence) const;
    [[nodiscard]] bool clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept;
    void clear_lane_best_effort(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState& sequence) const;
    [[nodiscard]] detail::PhysicalResources
    sequence_exclusive_state_resources(const SequenceState& sequence) const;
    [[nodiscard]] std::uint32_t owned_checkpoint_references(const SequenceState& sequence,
                                                            StateImageHandle state) const noexcept;
    [[nodiscard]] bool state_exclusive_to_sequence(const SequenceState& sequence,
                                                   StateImageHandle state) const noexcept;
    [[nodiscard]] bool compose_pressure_candidate(
        ResourceCandidateState& candidate,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
        std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options);
    [[nodiscard]] std::optional<detail::PressureTargetProjection> evaluate_pressure_target(
        const MaterializationSourceProtection* protection,
        std::span<const ContinuationHandle* const> pressure_owners,
        std::span<const qwen3_5::detail::PressureDecision> pressure_options,
        std::span<const SharedPrefixHandle* const> shared_pressure_owners,
        std::span<const qwen3_5::detail::PressureDecision> shared_pressure_options,
        std::vector<HostKVPageReplicaRelease>* released_host_pages) const;
    void refresh_state_views(SequenceState& sequence);
    void reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots);
    void settle_state_fork(SequenceState& sequence);
    bool publish_active_continuation(SequenceState& state, RequestControl& request,
                                     std::uint32_t lane, std::uint32_t continuation_index,
                                     qwen3_5::ContinuationSummary& summary) noexcept;
    bool salvage_continuation(SequenceState& state, RequestControl& request, std::uint32_t lane,
                              std::uint32_t continuation_index, qwen3_5::AbortResult& out) noexcept;
    [[nodiscard]] detail::PhysicalResources
    release_checkpoint_reference(StateImageHandle checkpoint) noexcept;
    [[nodiscard]] bool can_release_shared_prefix_state(std::uint32_t index,
                                                       SharedPrefixSlotRole expected_role) const;
    [[nodiscard]] detail::PhysicalResources
    release_shared_prefix_state_strict(std::uint32_t index,
                                       SharedPrefixSlotRole expected_role) noexcept;
    [[nodiscard]] detail::PhysicalResources
    install_private_capture(SequenceState& sequence, const CaptureGroup& group,
                            StateImageHandle checkpoint,
                            std::optional<runtime::CheckpointRef> replacement);
    // False when the shared replacement released less than its assessment; the replacement is
    // gone and the capture must be aborted.
    [[nodiscard]] bool prepare_active_capture(ActiveCaptureTransaction& transaction);
    void enqueue_active_capture_transfers(ActiveCaptureTransaction& transaction);
    void abort_active_capture(ActiveCaptureTransaction& transaction) noexcept;
    [[nodiscard]] ActiveCaptureResult publish_active_capture(ActiveCaptureTransaction& transaction);
    void release_active_shared_references_strict(SequenceState& sequence) noexcept;
    void release_active_shared_references(SequenceState& sequence) noexcept;
    void release_active_sequence_state_strict(SequenceState& sequence) noexcept;
    void release_sequence_state_strict(SequenceState& sequence) noexcept;
    void release_sequence_state(SequenceState& sequence) noexcept;
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void
    commit_generated_prefix_identity(SequenceState& sequence, std::uint32_t base_ledger_frontier,
                                     std::span<const TokenId> accepted_tokens,
                                     std::optional<std::uint32_t> prefix_execution_split_after);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                    std::uint32_t accepted_tokens, bool terminal,
                                    std::optional<std::uint32_t> prefix_execution_split_after,
                                    runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill(SequenceState& sequence, RequestControl& request,
                    runtime::ExecutionTiming* failed_timing);
    // Row-0 DFlash frame controls (lane, state slots, backend KV row) read by a prefill's
    // feature sink. The frame is shared with decode rounds, so every prefill step re-uploads them.
    void upload_dflash_prefill_controls(const SequenceState& sequence);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] std::vector<NgramProposer::Match>
    propose_ngram(std::span<const std::uint32_t> lanes,
                  std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] NgramProposer::Match propose_ngram_one(std::uint32_t lane,
                                                         const runtime::RoundBudget& budget);
    static void record_ngram_round(RequestControl& request, const NgramProposer::Match& match,
                                   std::uint32_t drafted, std::uint32_t accepted) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets,
                          runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void prepare_window(SequenceState& sequence, std::uint32_t begin, std::uint32_t end,
                        bool force_selection = false);
    void archive_memory_window(SequenceState& sequence);
    bool memory_query_probe(const SequenceState& sequence, std::uint32_t prompt_tokens) const;
    [[nodiscard]] std::uint32_t memory_checkpoint_frontier(const SequenceState& sequence,
        std::uint32_t prompt_tokens, bool allow_reuse) const;
    void capture_memory_prefix(SequenceState& sequence, std::uint32_t chunk_tokens);
    void rewind_memory_query(SequenceState& sequence);
    void restore_memory_prefix(SequenceState& sequence, const KvmemPrefixCheckpoint& checkpoint,
                               bool preserve_view = false);
    std::optional<std::size_t> memory_restorable_history(const PreparedPromptData& prompt) const;
    void reserve_memory_history(SequenceState& sequence, RequestControl& request,
                                const PreparedPromptData& prompt);
    bool restore_memory_history(SequenceState& sequence, RequestControl::Prefill& staged);
    void retain_memory_history(SequenceState& sequence, const RequestControl& request) noexcept;
    void save_memory_snapshot(KvmemWindowState& window) noexcept;
    void load_memory_snapshot(const PreparedPromptData& prompt) noexcept;
    void erase_memory_snapshot(const PreparedSessionKey& key) noexcept;
    void bind_memory_snapshot(SequenceState& sequence);
    void prepare_memory_statistics(SequenceState& sequence, std::uint32_t query_begin,
                                   std::uint32_t query_end);
    void commit_memory_statistics(SequenceState& sequence, std::uint32_t begin,
                                 std::uint32_t end);
    void commit_memory_candidates(SequenceState& sequence, std::uint32_t begin,
                                  std::uint32_t end, std::uint32_t first_column,
                                  std::uint32_t row_width);
    void read_memory_candidates();
    [[nodiscard]] std::vector<std::uint32_t> memory_image_pages(const PreparedPromptData& prompt) const;

    [[nodiscard]] std::uint32_t compact_position(const SequenceState& sequence,
                                                std::uint32_t logical) const {
        if (logical < sequence.window.removed_tokens) {
            throw std::logic_error("position precedes compact KV append frontier");
        }
        return logical - sequence.window.removed_tokens;
    }
    void ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens = 0);

    [[nodiscard]] std::uint32_t kv_lease_growth_margin_tokens() const noexcept {
        return std::max(prefill_chunk, kKVLeaseGrowthMarginTokens);
    }

    [[nodiscard]] std::uint32_t kv_lease_cushion_pages() const noexcept {
        // One round's Backend requirement can sit a verified extent plus a draft window above the
        // frontier the previous round checked, so the cushion has to absorb that jump before the
        // lease is extended again. A copy round can verify wider than the neural draft window.
        const auto page  = static_cast<std::uint32_t>(kPagedKVPageSize);
        const auto slack = widest_verify_window() + draft_window + 2U;
        return (slack + page - 1U) / page + 1U;
    }

    // The Backend lease also covers the drafts a round may still verify past the sequence's
    // output ceiling, and one forced control span.
    [[nodiscard]] std::uint32_t kv_lease_backend_allowance_tokens() const noexcept {
        return widest_verify_window() + std::min(draft_window, qwen3_5::kMtpDecodeMaximumDrafts) +
               1U;
    }

    // Page groups an entitlement needs to cover `tokens` and still hold a full cushion.
    [[nodiscard]] std::uint32_t kv_lease_pages_for_tokens(std::uint32_t tokens) const noexcept {
        return kv_pages_for_tokens(tokens) + kv_lease_cushion_pages();
    }

    void ensure_sequence_kv_lease(SequenceState& sequence, std::uint32_t main_tokens,
                                  std::uint32_t backend_tokens);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    void release_active_sequence_kv_strict(SequenceState& sequence) noexcept;
    void release_sequence_kv_strict(SequenceState& sequence,
                                    bool written_to_disk = false) noexcept;
    void release_sequence_kv(SequenceState& sequence) noexcept;
    void commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                            std::uint32_t backend_tokens = 0);
    [[nodiscard]] qwen3_5::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_5::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_5::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_5::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

struct PressurePlanningSessionImpl {
    using Core                         = ProgramImpl;
    using AdmissionCandidate           = qwen3_5::AdmissionCandidate;
    using CapturePressureCandidate     = qwen3_5::CapturePressureCandidate;
    using AdmissionCandidateImpl       = qwen3_5::detail::AdmissionCandidateImpl;
    using CapturePressureCandidateImpl = qwen3_5::detail::CapturePressureCandidateImpl;
    using CandidateState               = qwen3_5::detail::ResourceCandidateState;
    using ContinuationHandle           = qwen3_5::ContinuationHandle;
    using SharedPrefixHandle           = qwen3_5::SharedPrefixHandle;

    struct Owner {
        const ContinuationHandle* private_handle = nullptr;
        const SharedPrefixHandle* shared_handle  = nullptr;
        runtime::PlanningOwnerId id;
        bool shared = false;
    };

    struct PhysicalCandidateBinding {
        const CandidateState* state                 = nullptr;
        const AdmissionCandidateImpl* admission     = nullptr;
        const CapturePressureCandidateImpl* capture = nullptr;
    };

    struct TargetNode {
        std::uint32_t candidate_index      = 0;
        std::uint32_t victim_choice_offset = 0;
        std::uint32_t victim_choice_count  = 0;
        std::optional<detail::PhysicalResources> assessed_residual;
        std::uint32_t next_expansion_owner = 0;
        std::uint32_t stable_ordinal       = 0;
        bool root_maximal                  = false;
    };

    struct CandidateVictimOptions {
        std::uint32_t owner_index = 0;
        std::vector<PressureDecision> decisions;
        std::uint16_t eviction_choice = 0;
        // Escape-hatch preserve choice: frees this owner's device KV while keeping its host copy
        // (a demote-to-host outcome). Zero when the owner cannot be preserved this way; the
        // protected maximal target then falls back to eviction for that owner.
        std::uint16_t preserve_choice = 0;
        // With value-aware demotion, this private owner's rank by rebuild cost among the
        // candidate's victims, charged to its eviction; zero for shared owners and otherwise.
        std::uint32_t value_weight = 0;
    };

    struct CandidateOptions {
        std::vector<CandidateVictimOptions> victims;
        bool populated = false;
    };

    struct PreparedOwnerDecision {
        std::uint32_t candidate_index = 0;
        std::uint32_t victim_index    = 0;
        std::uint16_t choice          = 0;
        PressureDecision decision;
    };

    struct AssessmentSlot {
        std::vector<runtime::PressureOwnerOutcome> owner_outcomes;
        std::vector<runtime::PressureCheckpointRecoveryImpact> checkpoint_impacts;
        std::vector<runtime::CheckpointRecoveryAlternativeWork> recovery_alternatives;
        std::uint32_t generation = 1;
        bool leased              = false;
    };

    struct ConstructionOption {
        std::size_t victim = 0;
        PressureDecision decision;
        bool identity = false;
    };

    struct ConstructionSlot {
        std::vector<std::uint16_t> choices;
        std::vector<ConstructionOption> options;
        std::size_t next_owner        = 0;
        std::size_t next_option       = 0;
        std::uint32_t candidate_index = 0;
        std::uint32_t generation      = 0;
        std::uint32_t scan_generation = 1;
        detail::PhysicalResources residual;
        bool restore = false;
        bool leased  = false;
    };

    PressurePlanningSessionImpl(
        Core& owner, std::span<const PhysicalCandidateBinding> physical_candidates,
        std::span<const runtime::PlanningCandidateId> admission_candidate_ids,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids,
        std::span<const runtime::PlanningOwnerId> recency_order);
    ~PressurePlanningSessionImpl() noexcept;

    // Full eviction of an owner is reachable from incremental enumeration only inside the LRU tail
    // the admission planner had to sacrifice: one of the `eviction_licence_count_` oldest ranks.
    // Every other ranked owner, private or shared, is either demoted to Host (leaving its prefix
    // matchable) or kept, so a plan can never trade a more recent prefix's content for an older
    // one's device KV. An owner the caller left out of the recency order (rank -1) is sacrificed
    // by every escape-hatch rung, so it is licensed whenever any sacrifice is licensed.
    [[nodiscard]] bool owner_eviction_licensed(std::uint32_t owner_index) const {
        if (owner_index >= recency_rank_.size() || eviction_licence_count_ == 0) { return false; }
        const std::int32_t rank = recency_rank_[owner_index];
        if (rank < 0) { return true; }
        return static_cast<std::uint32_t>(rank) >= ranked_owner_count_ - eviction_licence_count_ &&
               std::find(licence_spared_ranks_.begin(), licence_spared_ranks_.end(),
                         static_cast<std::uint32_t>(rank)) == licence_spared_ranks_.end();
    }

    [[nodiscard]] qwen3_5::PressureTargetHandle
    identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] qwen3_5::PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] qwen3_5::PressureTargetHandle
    maximal_target(runtime::PlanningCandidateId candidate);
    // Escape-hatch recency-ladder rung. For `sacrifice_oldest` = k, the k oldest ranked owners
    // (private and shared, by recency rank) are fully evicted, and every other owner is kept — a
    // kept owner frees its device resources through a demote-to-host outcome wherever Host can
    // take it (keeping its host copy). Whether the sacrifice frees enough device and host capacity
    // is the rung's adoption check, so a pool without a host tier still expresses "evict the k
    // oldest and keep the rest". The ladder walks k = 0..R-1 (most-preserving first) and, if no
    // rung is adoptable, the caller falls back to `root_maximal_target` (k = R: clear everything),
    // the guaranteed liveness backstop.
    [[nodiscard]] qwen3_5::PressureTargetHandle
    recency_maximal_target(runtime::PlanningCandidateId candidate, std::uint32_t sacrifice_oldest,
                           std::span<const std::uint32_t> spared_ranks = {},
                           bool demote_kept                            = true);
    // Number of owners in the recency order; bounds the escape-hatch ladder (rungs 0..R-1, then
    // terminal).
    [[nodiscard]] std::uint32_t ranked_owner_count() const;
    // Licences incremental eviction of the `oldest_licensed` oldest ranked owners. Set once per
    // admission, from the escape-hatch ladder's smallest feasible sacrifice count (0 when the
    // identity target is feasible, i.e. nothing needs to be evicted).
    void set_eviction_licence(std::uint32_t oldest_licensed,
                              std::span<const std::uint32_t> spared_ranks = {});
    [[nodiscard]] qwen3_5::PressureConstructionCursor
    begin_construction(qwen3_5::PressureTargetHandle target, bool restore = false);
    [[nodiscard]] runtime::PressureConstructionStep
    next_construction_option(qwen3_5::PressureConstructionCursor& cursor);
    void choose_construction(qwen3_5::PressureConstructionCursor& cursor,
                             runtime::PressureConstructionOptionId option);
    [[nodiscard]] std::optional<qwen3_5::PressureTargetHandle>
    construction_target(const qwen3_5::PressureConstructionCursor& cursor);
    // Canonical target slots the arena can still hold. The arena also holds targets a planning
    // layer does not count in its own budget (identity targets, escape-hatch maximal rungs), so
    // layers must bound expansion commits by optional_targets_remaining, not by their own
    // budget, or commit_expansion rejects a commit the layer approved.
    [[nodiscard]] std::size_t target_arena_maximum() const noexcept;
    [[nodiscard]] std::uint32_t optional_targets_remaining() const noexcept;
    [[nodiscard]] ConstructionSlot&
    construction_slot(const qwen3_5::PressureConstructionCursor& cursor);
    static void release_construction(const void*, std::uint32_t, std::uint32_t) noexcept;
    [[nodiscard]] runtime::PressureTargetGuidance
    guidance_choices(std::uint32_t candidate_index, std::span<const std::uint16_t> choices,
                     std::uint32_t ordinal,
                     std::optional<std::size_t> override_owner = std::nullopt,
                     const PressureDecision* override_decision = nullptr);
    [[nodiscard]] detail::PhysicalResources
    construction_residual(std::uint32_t candidate_index,
                          std::span<const std::uint16_t> choices) const;
    [[nodiscard]] runtime::PressureTargetGuidance guidance(qwen3_5::PressureTargetHandle target);
    [[nodiscard]] qwen3_5::AssessedPressureTarget assess(qwen3_5::PressureTargetHandle target);
    [[nodiscard]] qwen3_5::PreparedPressureExpansion
    prepare_expansion(qwen3_5::PressureTargetHandle parent,
                      std::uint32_t maximum_owners = std::numeric_limits<std::uint32_t>::max());
    [[nodiscard]] qwen3_5::PressureExpansionView
    commit_expansion(qwen3_5::PreparedPressureExpansion&& prepared);
    void discard_expansion(qwen3_5::PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const qwen3_5::AssessedPressureTarget& assessed,
                                      const PreparedPromptData& prompt,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<AdmissionCandidate> seal(qwen3_5::AssessedPressureTarget&& assessed,
                                                         const PreparedPromptData& prompt,
                                                         runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<CapturePressureCandidate>
    seal_capture(qwen3_5::AssessedPressureTarget&& assessed);

    [[nodiscard]] bool valid(qwen3_5::PressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] std::span<const std::uint16_t> victim_choices(const TargetNode& target) const;
    [[nodiscard]] TargetNode* find_target(std::uint32_t candidate_index,
                                          std::span<const std::uint16_t> choices) noexcept;
    [[nodiscard]] const TargetNode*
    find_target(std::uint32_t candidate_index,
                std::span<const std::uint16_t> choices) const noexcept;
    [[nodiscard]] std::uint32_t intern_target(std::uint32_t candidate_index,
                                              std::span<const std::uint16_t> choices,
                                              bool root_maximal = false);
    void index_target(std::uint32_t target_index);
    void populate_options(std::uint32_t candidate_index);
    void rank_victims_by_rebuild_cost(CandidateOptions& options) const;
    [[nodiscard]] std::vector<PressureDecision>
    pressure_successors(const CandidateVictimOptions& victim_options,
                        const detail::PhysicalResources& residual,
                        const Core::MaterializationSourceProtection& protection,
                        const PressureDecision* current) const;
    [[nodiscard]] std::uint32_t acquire_assessment_slot();
    static void release_assessment_slot(const void* owner, std::uint32_t slot,
                                        std::uint32_t generation) noexcept;

    Core* program = nullptr;
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t generation         = 1;
    std::uint32_t scratch_generation = 1;
    std::vector<PhysicalCandidateBinding> candidates;
    std::vector<runtime::PlanningCandidateId> candidate_ids;
    std::vector<Owner> owners;
    // Parallel to `owners`; recency rank in the caller's order over private and shared owners
    // (0 = most recently hit or published), or -1 for an owner left out of it. Orders the
    // escape-hatch sacrifice: the oldest ranked owner (highest rank) gives up its host copy first.
    std::vector<std::int32_t> recency_rank_;
    // A caller that passes no recency order keeps the economic search alone: no demote-to-host
    // preserve outcomes, and full eviction stays reachable for every owner.
    bool recency_eviction_ = false;
    // Number of ranked owners; the escape-hatch ladder has this many sacrifice rungs
    // plus the clear-all terminal.
    std::uint32_t ranked_owner_count_ = 0;
    // Ranked owners the escape-hatch ladder had to sacrifice for the current admission, oldest
    // first. Full eviction is reachable from incremental enumeration only inside this LRU tail.
    std::uint32_t eviction_licence_count_ = 0;
    // Ranks inside that tail the ladder proved it did not need to sacrifice.
    std::vector<std::uint32_t> licence_spared_ranks_;
    std::vector<CandidateOptions> candidate_options;
    std::vector<TargetNode> targets;
    std::vector<std::uint16_t> target_choice_arena;
    std::vector<std::uint16_t> choice_scratch;
    std::vector<std::uint32_t> target_hash_table;
    std::vector<TargetNode> expansion_scratch;
    std::vector<PreparedOwnerDecision> prepared_owner_decisions;
    std::vector<qwen3_5::PressureTargetHandle> committed_children;
    std::vector<const ContinuationHandle*> selected_private_owners;
    std::vector<runtime::PlanningOwnerId> selected_private_owner_ids;
    std::vector<const PressureDecision*> selected_private_decisions;
    std::vector<const SharedPrefixHandle*> selected_shared_owners;
    std::vector<runtime::PlanningOwnerId> selected_shared_owner_ids;
    std::vector<const PressureDecision*> selected_shared_decisions;
    std::vector<const ContinuationHandle*> recovery_private_owners;
    std::vector<const PressureDecision*> recovery_private_decisions;
    std::vector<runtime::PlanningOwnerId> recovery_private_owner_ids;
    std::vector<const SharedPrefixHandle*> recovery_shared_owners;
    std::vector<const PressureDecision*> recovery_shared_decisions;
    std::vector<runtime::PlanningOwnerId> recovery_shared_owner_ids;
    std::vector<const PressureDecision*> projected_owner_decisions;
    std::vector<runtime::PressureOwnerOutcome> assessment_outcomes;
    std::vector<PressureCheckpointRecoveryProjection> assessment_impact_projections;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> assessment_recovery_alternatives;
    Core::PressureRecoveryScratch recovery_scratch;
    std::vector<runtime::PressureOwnerOutcome> guidance_outcomes;
    std::vector<runtime::PressureCheckpointOutcome> guidance_checkpoint_changes;
    std::vector<runtime::PressureOwnerRecoveryGuidance> guidance_recovery;
    std::array<ConstructionSlot, 4> construction_slots;
    std::uint32_t construction_generation = 0;
    std::array<AssessmentSlot, 2> assessment_slots;
    std::uint32_t prepared_new_count = 0;
    std::uint32_t prepared_owner_end = 0;
    std::size_t scratch_choice_mark  = 0;
    bool scratch_live                = false;
};



} // namespace ninfer::models::qwen3_5::detail
