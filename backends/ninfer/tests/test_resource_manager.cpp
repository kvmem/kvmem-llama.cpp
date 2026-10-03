#include "models/qwen3_5/program/planning/pressure_value_ranking.h"
#include "runtime/engine/context_cache/resource_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
// Minimal declarations; including <windows.h> here would poison the TU with min/max macros.
extern "C" {
unsigned timeBeginPeriod(unsigned);
unsigned timeEndPeriod(unsigned);
}
#pragma comment(lib, "winmm.lib")
#endif

namespace {

// Windows' default system timer quantum (~15.6ms) stretches the fake's 1ms pressure-assessment
// sleep to ~16ms, exhausting the planner's 5ms economic search budget before the deep reuse
// closure is explored. Raise the timing resolution for the process lifetime so sleep_for
// honors the upstream-intended durations.
struct TimerResolution {
    bool raised = false;
#ifdef _WIN32
    TimerResolution() { raised = timeBeginPeriod(1) == 0; }
    ~TimerResolution() {
        if (raised) { timeEndPeriod(1); }
    }
#endif
};

using ninfer::PrefixReusePath;
using ninfer::RuntimeStats;
using ninfer::runtime::CancellationFlagView;
using ninfer::runtime::CheckpointKind;
using ninfer::runtime::CheckpointRecoveryAlternativeWork;
using ninfer::runtime::CheckpointRef;
using ninfer::runtime::CheckpointScope;
using ninfer::runtime::CommitDisposition;
using ninfer::runtime::ConsumeStatus;
using ninfer::runtime::ContextOperationCounts;
using ninfer::runtime::ContextTransactionInProgress;
using ninfer::runtime::ContextTransactionReserveStatus;
using ninfer::runtime::ContextTransactionStatus;
using ninfer::runtime::ContextTransferObservation;
using ninfer::runtime::ContextTransferRequirement;
using ninfer::runtime::FinishDisposition;
using ninfer::runtime::LaneId;
using ninfer::runtime::MaterializationMachineWork;
using ninfer::runtime::PrefillWork;
using ninfer::runtime::PlanningCandidateId;
using ninfer::runtime::PlanningOwnerId;
using ninfer::runtime::PrivateSourceMode;
using ninfer::runtime::ProgramResourceRevision;
using ninfer::runtime::Readiness;
using ninfer::runtime::RequestPlanSummary;
using ninfer::runtime::RetentionClass;
using ninfer::runtime::VictimDisposition;

int failures = 0;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Test>
void run_test(const char* name, Test&& test) {
    try {
        test();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
}

ninfer::runtime::ContextMachineCostModel test_cost_model() {
    ninfer::runtime::ContextMachineCostModel model;
    for (auto& transfer : model.transfer) {
        transfer.batch_ns        = 0;
        transfer.operation_ns    = 0;
        transfer.ns_per_byte_q32 = ninfer::runtime::kContextCostQ32One;
    }
    model.prefill.token_ns_q32          = 100ULL * ninfer::runtime::kContextCostQ32One;
    model.prefill.attention_pair_ns_q32 = ninfer::runtime::kContextCostQ32One;
    model.prefill.vision_item_ns        = 1;
    model.prefill.vision_patch_ns_q32   = ninfer::runtime::kContextCostQ32One;
    return model;
}

void set_fake_machine_costs(MaterializationMachineWork& work, std::uint64_t optimistic_ns,
                            std::uint64_t immediate_ns) {
    work.optimistic_candidate_transfers                  = {};
    work.candidate_transfers                             = {};
    work.pressure_transfers                              = {};
    work.remaining_prefill_work                          = {};
    work.optimistic_candidate_transfers[2].payload_bytes = optimistic_ns;
    work.candidate_transfers[2].payload_bytes            = immediate_ns;
}

CheckpointRecoveryAlternativeWork fake_recovery_work(std::uint64_t ns) {
    CheckpointRecoveryAlternativeWork work;
    work.prefill.attention_pairs = ns;
    return work;
}

struct FakePreparedPrompt {
    std::uint32_t content_key = 0;
    // The key of the prompt's stable prefix. A new question over a known system prompt shares this
    // while its own content_key differs, which is the shape a shared prefix has to serve.
    std::uint32_t prefix_key = 0;
};

struct FakeCacheSessionKey {
    std::uint32_t value = 0;

    [[nodiscard]] std::string_view view() const noexcept {
        return {reinterpret_cast<const char*>(&value), sizeof(value)};
    }

    friend bool operator==(FakeCacheSessionKey, FakeCacheSessionKey) = default;
};

struct FakeShortlistKey {
    std::uint32_t digest   = 0;
    std::uint32_t frontier = 0;
    // The real PrefixShortlistKey surface the NINFER_MAT_DEBUG trace prints.
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t identity_tag = 0;

    friend bool operator==(FakeShortlistKey, FakeShortlistKey) = default;
};

struct FakeRequiredKV {
    std::uint32_t main_pages    = 1;
    std::uint32_t backend_pages = 0;

    friend bool operator==(FakeRequiredKV, FakeRequiredKV) = default;
};

struct FakeCheckpointSummary {
    CheckpointRef ref;
    CheckpointScope scope = CheckpointScope::Private;
    FakeShortlistKey shortlist_key;
    ninfer::runtime::ReplicaResidency state_residency =
        ninfer::runtime::ReplicaResidency::DeviceOnly;
    FakeRequiredKV required_kv;
    PrefillWork rebuild_work;

    friend bool operator==(const FakeCheckpointSummary&, const FakeCheckpointSummary&) = default;
};

struct FakeContinuationSummary {
    std::optional<FakeCheckpointSummary> endpoint;
    std::optional<FakeCheckpointSummary> rewrite;
    std::vector<FakeCheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;
};

struct FakeSharedPrefixSummary {
    FakeCheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    friend bool operator==(const FakeSharedPrefixSummary&,
                           const FakeSharedPrefixSummary&) = default;
};

FakeCheckpointSummary endpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                       .frontier = frontier,
                                       .ordinal  = 0},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary rewrite_checkpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref =
            CheckpointRef{.kind = CheckpointKind::TurnClosure, .frontier = frontier, .ordinal = 0},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary long_anchor(std::uint32_t digest, std::uint32_t frontier,
                                  std::uint32_t ordinal) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::LongAnchor,
                                       .frontier = frontier,
                                       .ordinal  = ordinal},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary shared_checkpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::SharedStablePrefix,
                                       .frontier = frontier,
                                       .ordinal  = 0},
        .scope         = CheckpointScope::Shared,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

struct FakeContextCache {
    struct Opportunity {
        ninfer::PromptCacheMarkerKind kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix;
        ninfer::SharedCandidateEvidence evidence =
            ninfer::SharedCandidateEvidence::ExplicitBoundary;
        std::uint32_t frontier = 0;
    };

    std::optional<FakeCacheSessionKey> session_key;
    RetentionClass retention  = RetentionClass::RecentPrivate;
    bool update_session_index = true;
    std::vector<Opportunity> opportunities;
};

struct FakeRequestBasePlan {
    RequestPlanSummary value;
    FakeContextCache cache;
    std::uint32_t shortlist_digest = 0;
    bool allow_shortlist           = true;
    bool isolated_feasible         = true;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    [[nodiscard]] const FakeContextCache& context_cache() const noexcept { return cache; }

    [[nodiscard]] std::optional<FakeShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept {
        if (!allow_shortlist || frontier == 0) { return std::nullopt; }
        return FakeShortlistKey{.digest = shortlist_digest, .frontier = frontier};
    }

    [[nodiscard]] std::size_t prefix_shortlist_size() const noexcept {
        return allow_shortlist ? 1 : 0;
    }

    [[nodiscard]] std::optional<PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
        const auto found =
            std::find_if(cache.opportunities.begin(), cache.opportunities.end(),
                         [&](const auto& opportunity) { return opportunity.frontier == frontier; });
        return found == cache.opportunities.end()
                   ? std::nullopt
                   : std::optional<PrefillWork>(PrefillWork{.tokens = frontier});
    }
};

FakeRequestBasePlan make_base(std::uint32_t digest,
                              std::optional<FakeCacheSessionKey> session = std::nullopt,
                              RetentionClass retention  = RetentionClass::RecentPrivate,
                              bool update_session_index = true) {
    FakeRequestBasePlan out;
    out.value.prompt_tokens           = 64;
    out.value.requested_output_tokens = 8;
    out.value.effective_output_tokens = 8;
    out.value.service_work_quanta     = 64;
    out.value.publish_continuation    = true;
    out.cache.session_key             = session;
    out.cache.retention               = retention;
    out.cache.update_session_index    = update_session_index;
    out.shortlist_digest              = digest;
    return out;
}

struct FakeContinuationHandle {
    std::uint32_t id          = 0;
    std::uint32_t content_key = 0;

    FakeContinuationHandle() = default;

    FakeContinuationHandle(std::uint32_t id_value, std::uint32_t key_value)
        : id(id_value), content_key(key_value) {}

    FakeContinuationHandle(FakeContinuationHandle&& other) noexcept
        : id(std::exchange(other.id, 0)), content_key(other.content_key) {}

    FakeContinuationHandle& operator=(FakeContinuationHandle&& other) noexcept {
        id          = std::exchange(other.id, 0);
        content_key = other.content_key;
        return *this;
    }

    FakeContinuationHandle(const FakeContinuationHandle&)            = delete;
    FakeContinuationHandle& operator=(const FakeContinuationHandle&) = delete;
};

struct FakeSharedPrefixHandle {
    std::uint32_t id          = 0;
    std::uint32_t content_key = 0;
    // Published checkpoint frontier; zero prices the owner's pressure impact at finish_frontier.
    std::uint32_t frontier = 0;

    FakeSharedPrefixHandle()                                             = default;
    FakeSharedPrefixHandle(FakeSharedPrefixHandle&&) noexcept            = default;
    FakeSharedPrefixHandle& operator=(FakeSharedPrefixHandle&&) noexcept = default;
    FakeSharedPrefixHandle(const FakeSharedPrefixHandle&)                = delete;
    FakeSharedPrefixHandle& operator=(const FakeSharedPrefixHandle&)     = delete;
};

struct FakeSequenceHandle {
    std::uint32_t id = 0;

    friend bool operator==(FakeSequenceHandle, FakeSequenceHandle) = default;
};

struct FakeCaptureOffer {
    std::uint32_t id = 0;
};

struct FakeAdmissionCandidate {
    RequestPlanSummary value;
    PrefillWork remaining;
    std::vector<ContextTransferRequirement> transfers;
    ninfer::runtime::IdentityMaterializationAssessment identity;
    PrivateSourceMode source_mode           = PrivateSourceMode::ConsumeToActive;
    std::uint32_t private_source_id         = 0;
    std::uint32_t shared_source_id          = 0;
    std::uint32_t shared_source_content_key = 0;
    std::uint32_t shared_source_frontier    = 0;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    [[nodiscard]] const ninfer::runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept {
        return identity;
    }
};

struct FakeTargetDecision {
    std::uint64_t id                  = 0;
    std::uint64_t immediate_ns        = 100'000'000;
    std::uint32_t degradation_units   = 1;
    std::uint32_t dropped_checkpoints = 0;
    bool evicts_continuation          = false;
    bool shared_owner                 = false;

    friend bool operator==(const FakeTargetDecision&, const FakeTargetDecision&) = default;
};

struct FakePressureTargetHandle {
    std::uint32_t generation = 0;
    std::uint32_t index      = 0;

    friend bool operator==(FakePressureTargetHandle, FakePressureTargetHandle) = default;
};

struct FakePreparedPressureExpansion {
    std::uint32_t generation         = 0;
    std::uint32_t scratch_generation = 0;
    std::uint32_t new_count          = 0;

    FakePreparedPressureExpansion(std::uint32_t generation_value,
                                  std::uint32_t scratch_generation_value,
                                  std::uint32_t new_count_value) noexcept
        : generation(generation_value), scratch_generation(scratch_generation_value),
          new_count(new_count_value) {}

    FakePreparedPressureExpansion(FakePreparedPressureExpansion&&) noexcept            = default;
    FakePreparedPressureExpansion& operator=(FakePreparedPressureExpansion&&) noexcept = default;
    FakePreparedPressureExpansion(const FakePreparedPressureExpansion&)                = delete;
    FakePreparedPressureExpansion& operator=(const FakePreparedPressureExpansion&)     = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept { return new_count; }
};

struct FakePressureExpansionView {
    std::span<const FakePressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
    bool complete                     = true;
};

class FakeAssessedPressureTarget {
public:
    FakeAssessedPressureTarget(FakeAssessedPressureTarget&&) noexcept            = default;
    FakeAssessedPressureTarget& operator=(FakeAssessedPressureTarget&&) noexcept = default;
    FakeAssessedPressureTarget(const FakeAssessedPressureTarget&)                = delete;
    FakeAssessedPressureTarget& operator=(const FakeAssessedPressureTarget&)     = delete;

    [[nodiscard]] const ninfer::runtime::PressureTargetAssessment& assessment() const noexcept {
        return assessment_;
    }

private:
    FakeAssessedPressureTarget(FakePressureTargetHandle target,
                               ninfer::runtime::PressureTargetAssessment assessment) noexcept
        : target_(target), assessment_(assessment) {}

    FakePressureTargetHandle target_;
    ninfer::runtime::PressureTargetAssessment assessment_;

    friend class FakePressurePlanningSession;
};

struct FakeResourcePlan {
    FakeAdmissionCandidate admission;
    ProgramResourceRevision revision;
    std::vector<FakeTargetDecision> private_actions;
    std::vector<FakeTargetDecision> shared_actions;
    std::vector<std::uint32_t> private_owner_ids;
    std::vector<std::uint32_t> shared_owner_ids;
    std::vector<PlanningOwnerId> private_planning_ids;
    std::vector<PlanningOwnerId> shared_planning_ids;

    FakeResourcePlan() = default;

    FakeResourcePlan(FakeAdmissionCandidate admission_value, ProgramResourceRevision revision_value)
        : admission(std::move(admission_value)), revision(revision_value) {}

    FakeResourcePlan(FakeResourcePlan&&) noexcept            = default;
    FakeResourcePlan& operator=(FakeResourcePlan&&) noexcept = default;
    FakeResourcePlan(const FakeResourcePlan&)                = delete;
    FakeResourcePlan& operator=(const FakeResourcePlan&)     = delete;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return admission.summary(); }

    [[nodiscard]] bool needs_transfer() const noexcept { return !admission.transfers.empty(); }

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept { return revision; }
};

struct FakePersistentBackfillProof {
    ProgramResourceRevision revision;

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept { return revision; }
};

struct FakeStartResult {
    FakeSequenceHandle sequence;
};

struct FakeMaterializationVictimResult {
    PlanningOwnerId owner;
    VictimDisposition disposition = VictimDisposition::Retained;
    bool pressure_committed       = false;
    std::optional<FakeContinuationSummary> final_summary;
};

struct FakeMaterializationSharedVictimResult {
    PlanningOwnerId owner;
    VictimDisposition disposition = VictimDisposition::Retained;
    bool pressure_committed       = false;
    std::optional<FakeSharedPrefixSummary> final_summary;
};

struct FakeMaterializationSourceResult {
    PrivateSourceMode mode = PrivateSourceMode::Retain;
    std::optional<FakeContinuationSummary> final_summary;
};

struct FakeMaterializationSharedSourceResult {
    std::optional<FakeSharedPrefixSummary> final_summary;
};

struct FakeMaterializationResult {
    ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    std::optional<FakeStartResult> published;
    std::optional<FakeMaterializationSourceResult> source;
    std::optional<FakeMaterializationSharedSourceResult> shared_source;
    std::vector<FakeMaterializationVictimResult> victims;
    std::vector<FakeMaterializationSharedVictimResult> shared_victims;
    std::vector<ContextTransferObservation> transfer_observations;
    ContextOperationCounts operations;
    std::string failure;
};

struct FakeSharedPrefixPublication {
    FakeSharedPrefixHandle handle;
    FakeSharedPrefixSummary summary;
};

struct FakeActiveCaptureResult {
    ContextTransactionStatus status     = ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed = false;
    FakeContinuationSummary active_summary;
    std::optional<FakeSharedPrefixPublication> shared;
    std::vector<FakeMaterializationVictimResult> victims;
    std::vector<FakeMaterializationSharedVictimResult> shared_victims;
    std::vector<ContextTransferObservation> transfer_observations;
    ContextOperationCounts operations;
};

using FakeContextTransactionProgress =
    std::variant<ContextTransactionInProgress, FakeMaterializationResult, FakeActiveCaptureResult>;

struct FakeCaptureAssessment {
    FakeShortlistKey shortlist_key;
    ninfer::SharedCandidateEvidence shared_evidence = ninfer::SharedCandidateEvidence::None;
    PrefillWork protected_rebuild_work;
    std::vector<ContextTransferRequirement> transfer_requirements;
    std::vector<CheckpointRecoveryAlternativeWork> projected_recovery_work{fake_recovery_work(0)};
    std::vector<CheckpointRef> private_replacement_candidates;
    bool publishes_private   = false;
    bool publishes_shared    = false;
    bool needs_transfer      = false;
    bool physically_feasible = true;
};

struct FakeTimings {
    std::uint64_t value = 0;
};

struct FakeSpeculativeStats {
    std::uint64_t value = 0;
};

struct FakeFinishResult {
    ConsumeStatus status          = ConsumeStatus::InvariantMismatch;
    FinishDisposition disposition = FinishDisposition::Released;
    FakeTimings timings;
    FakeSpeculativeStats speculative;
    bool salvaged = false;
    FakeContinuationSummary summary;
    std::optional<FakeContinuationHandle> continuation;
};

struct FakeAbortResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    FakeTimings timings;
    FakeSpeculativeStats speculative;
    bool salvaged = false;
    FakeContinuationSummary summary;
    std::optional<FakeContinuationHandle> continuation;
};

struct FakeReleaseResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
};

struct FakeCommitRowResult {
    CommitDisposition disposition = CommitDisposition::Active;
};

struct FakeCommitResult {
    std::array<FakeCommitRowResult, ninfer::kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
};

struct FakeDiscardResult {
    ConsumeStatus status  = ConsumeStatus::InvariantMismatch;
    std::size_t row_count = 0;
};

struct FakePhysicalUsage {
    std::uint32_t device_state_slots            = 0;
    std::uint32_t host_state_slots              = 0;
    std::uint32_t device_main_kv_pages          = 0;
    std::uint32_t device_backend_kv_pages       = 0;
    std::uint32_t device_main_kv_lease_pages    = 0;
    std::uint32_t device_backend_kv_lease_pages = 0;
    std::size_t host_kv_bytes                   = 0;
};

class FakeProgram;

class FakePressurePlanningSession {
public:
    FakePressurePlanningSession(FakeProgram& program,
                                std::span<const FakeAdmissionCandidate* const> candidates,
                                std::span<const PlanningCandidateId> candidate_ids,
                                std::span<const FakeContinuationHandle* const> private_owners,
                                std::span<const PlanningOwnerId> private_owner_ids,
                                std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                std::span<const PlanningOwnerId> shared_owner_ids,
                                std::span<const PlanningOwnerId> recency_order);

    FakePressurePlanningSession(FakePressurePlanningSession&&) noexcept            = default;
    FakePressurePlanningSession& operator=(FakePressurePlanningSession&&) noexcept = default;
    FakePressurePlanningSession(const FakePressurePlanningSession&)                = delete;
    FakePressurePlanningSession& operator=(const FakePressurePlanningSession&)     = delete;

    [[nodiscard]] FakePressureTargetHandle identity_target(PlanningCandidateId candidate) const;
    [[nodiscard]] FakePressureTargetHandle identity_target() const;

    [[nodiscard]] static constexpr PlanningCandidateId candidate_id() noexcept {
        return PlanningCandidateId{.value = 0};
    }

    [[nodiscard]] FakePressureTargetHandle root_maximal_target(PlanningCandidateId candidate);
    [[nodiscard]] FakePressureTargetHandle
    recency_maximal_target(PlanningCandidateId candidate, std::uint32_t sacrifice_oldest,
                           std::span<const std::uint32_t> spared_ranks = {},
                           bool demote_kept                            = true);
    [[nodiscard]] std::uint32_t ranked_owner_count() const;
    void set_eviction_licence(std::uint32_t oldest_licensed,
                              std::span<const std::uint32_t> spared_ranks = {});
    struct Cursor;
    [[nodiscard]] FakePressureTargetHandle maximal_target(PlanningCandidateId candidate);
    [[nodiscard]] Cursor begin_construction(FakePressureTargetHandle target, bool restore = false);
    [[nodiscard]] ninfer::runtime::PressureConstructionStep
    next_construction_option(Cursor& cursor);
    void choose_construction(Cursor& cursor, ninfer::runtime::PressureConstructionOptionId option);
    [[nodiscard]] std::optional<FakePressureTargetHandle> construction_target(const Cursor& cursor);
    // Mirrors the real session's arena accounting (same formula as the commit_expansion guard).
    [[nodiscard]] std::uint32_t optional_targets_remaining() const noexcept;
    [[nodiscard]] ninfer::runtime::PressureTargetGuidance guidance(FakePressureTargetHandle target);
    [[nodiscard]] FakeAssessedPressureTarget assess(FakePressureTargetHandle target);
    [[nodiscard]] FakePreparedPressureExpansion
    prepare_expansion(FakePressureTargetHandle parent,
                      std::uint32_t maximum_owners = std::numeric_limits<std::uint32_t>::max());
    [[nodiscard]] FakePressureExpansionView
    commit_expansion(FakePreparedPressureExpansion&& prepared);
    void discard_expansion(FakePreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] PrefillWork
    shared_capture_split_prefill_work(const FakeAssessedPressureTarget&, const FakePreparedPrompt&,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<FakeResourcePlan> seal(FakeAssessedPressureTarget&& assessed,
                                                       const FakePreparedPrompt& prompt,
                                                       ninfer::runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<FakeResourcePlan> seal(FakeAssessedPressureTarget&& assessed);
    [[nodiscard]] std::optional<FakeResourcePlan>
    seal_capture(FakeAssessedPressureTarget&& assessed);
    [[nodiscard]] bool try_claim_seal_window() noexcept;
    void release_seal_window() noexcept;

private:
    struct Owner {
        const FakeContinuationHandle* private_handle = nullptr;
        const FakeSharedPrefixHandle* shared_handle  = nullptr;
        PlanningOwnerId id;
        bool shared = false;
    };

    struct Target {
        std::uint32_t candidate_index = 0;
        std::vector<std::uint16_t> choices;
        std::uint32_t stable_ordinal       = 0;
        bool root_maximal                  = false;
        std::uint32_t next_expansion_owner = 0;
        // -1 = not a recency-ladder rung; otherwise the rung's `sacrifice_oldest` (used by the
        // ladder-feasibility test mode to model "host only fits the demotions after this many
        // oldest preserved owners are given up").
        std::int32_t sacrifice_oldest = -1;
    };

public:
    struct Cursor {
        Target target;
        std::vector<Target> options;
        std::size_t next_owner = 0, next_option = 0;
        std::uint32_t generation = 0, scan = 1;
        bool restore = false;
    };
private:
    std::uint32_t construction_generation_ = 0;
    [[nodiscard]] ninfer::runtime::PressureTargetGuidance guidance_for(const Target& target);
    [[nodiscard]] bool valid(FakePressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(PlanningCandidateId candidate) const;
    void populate_options(std::uint32_t candidate_index);
    [[nodiscard]] bool same_target(const Target& left, const Target& right) const noexcept;
    [[nodiscard]] std::vector<FakeTargetDecision> decisions_for(std::uint32_t candidate_index,
                                                                std::size_t owner_index) const;

    FakeProgram* program_ = nullptr;
    ProgramResourceRevision revision_;
    std::uint32_t generation_         = 0;
    std::uint32_t scratch_generation_ = 0;
    bool scratch_live_                = false;
    std::uint32_t prepared_parent_ = 0, prepared_owner_end_ = 0;
    std::vector<const FakeAdmissionCandidate*> candidates_;
    std::vector<PlanningCandidateId> candidate_ids_;
    std::vector<Owner> owners_;
    std::vector<std::int32_t> recency_rank_;
    std::uint32_t ranked_owner_count_     = 0;
    std::uint32_t eviction_licence_count_ = 0;
    bool recency_eviction_                = false;
    std::vector<std::uint32_t> licence_spared_ranks_;
    // Mirrors the real session's LRU gate on incremental eviction.
    [[nodiscard]] bool owner_eviction_licensed(std::uint32_t owner_index) const;
    std::vector<std::vector<std::vector<FakeTargetDecision>>> options_;
    std::vector<std::uint8_t> options_populated_;
    std::vector<Target> targets_;
    std::vector<Target> expansion_scratch_;
    std::vector<FakePressureTargetHandle> committed_children_;
    std::vector<ninfer::runtime::PressureOwnerOutcome> guidance_outcomes_;
    std::vector<ninfer::runtime::PressureOwnerOutcome> assessment_outcomes_;
    std::vector<ninfer::runtime::PressureCheckpointRecoveryImpact> assessment_impacts_;
    std::vector<CheckpointRecoveryAlternativeWork> assessment_recovery_work_;
};

class FakeProgram {
public:
    friend class FakePressurePlanningSession;

    enum class TransactionKind : std::uint8_t {
        None,
        Materialization,
        Capture,
    };

    [[nodiscard]] bool isolated_request_feasible(const FakeRequestBasePlan& base) const noexcept {
        return base.isolated_feasible;
    }
    bool auxiliary_tier_available = true;
    [[nodiscard]] bool temporary_request_feasible(const FakeRequestBasePlan&) const noexcept {
        return auxiliary_tier_available;
    }

    [[nodiscard]] std::optional<FakeAdmissionCandidate>
    inspect_admission(const FakePreparedPrompt& prompt, const FakeRequestBasePlan& base, LaneId,
                      const FakeContinuationHandle* source,
                      const FakeSharedPrefixHandle* shared_source,
                      std::optional<CheckpointRef> checkpoint, bool must_retain_source) {
        ++admission_inspections;
        if (source != nullptr) {
            inspected_private_sources.push_back(source->id);
            if (source->content_key != prompt.content_key || !checkpoint) { return std::nullopt; }
        }
        if (shared_source != nullptr) {
            inspected_shared_sources.push_back(shared_source->id);
            const bool shared_matches = shared_source->content_key == prompt.content_key ||
                                        (prompt.prefix_key != 0 &&
                                         shared_source->content_key == prompt.prefix_key);
            if (!shared_matches || !checkpoint) { return std::nullopt; }
        }

        FakeAdmissionCandidate plan;
        plan.value = base.summary();
        if (checkpoint) {
            plan.value.reusable_prompt_tokens = checkpoint->frontier;
            switch (checkpoint->kind) {
            case CheckpointKind::SessionEndpoint:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateEndpoint;
                break;
            case CheckpointKind::TurnClosure:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateTurnClosure;
                break;
            case CheckpointKind::ResponseReplay:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateResponseReplay;
                break;
            case CheckpointKind::LongAnchor:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateLongAnchor;
                break;
            case CheckpointKind::SharedStablePrefix:
                plan.value.prefix_reuse_path = PrefixReusePath::SharedStablePrefix;
                break;
            }
        } else {
            plan.value.reusable_prompt_tokens = 0;
            plan.value.prefix_reuse_path      = PrefixReusePath::Root;
        }
        plan.remaining.tokens = plan.value.prompt_tokens > plan.value.reusable_prompt_tokens
                                    ? plan.value.prompt_tokens - plan.value.reusable_prompt_tokens
                                    : 0;
        if (source != nullptr) {
            plan.private_source_id = source->id;
            plan.source_mode =
                must_retain_source ? PrivateSourceMode::Retain : PrivateSourceMode::ConsumeToActive;
        } else if (shared_source != nullptr) {
            plan.shared_source_id          = shared_source->id;
            plan.shared_source_content_key = shared_source->content_key;
            plan.shared_source_frontier    = checkpoint->frontier;
            plan.source_mode               = PrivateSourceMode::Retain;
        }
        plan.identity.machine_work.remaining_prefill_work = plan.remaining;
        plan.identity.machine_work.reused_prompt_tokens   = plan.value.reusable_prompt_tokens;
        plan.identity.physical_status =
            target_feasible(std::span<const FakeTargetDecision>{})
                ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                : ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        plan.identity.source_mode = plan.source_mode;
        plan.identity.expandable  = plan.identity.physical_status !=
                                   ninfer::runtime::MaterializationPhysicalStatus::Feasible;
        plan.identity.projection_work = 1;
        plan.identity.assessment_digest =
            (static_cast<std::uint64_t>(plan.value.reusable_prompt_tokens) << 32U) ^
            revision_.value;
        return plan;
    }

    [[nodiscard]] std::optional<FakeResourcePlan>
    seal_identity(const FakeAdmissionCandidate& admission, const FakePreparedPrompt&,
                  ninfer::runtime::FinalScheduleIntent intent) {
        if (admission.identity.physical_status !=
            ninfer::runtime::MaterializationPhysicalStatus::Feasible) {
            return std::nullopt;
        }
        selected_shared_capture_frontiers.assign(intent.shared_capture_frontiers.begin(),
                                                 intent.shared_capture_frontiers.end());
        seal_attempts.emplace_back();
        return FakeResourcePlan(admission, revision_);
    }

    [[nodiscard]] FakePressurePlanningSession
    begin_pressure_planning(std::span<const FakeAdmissionCandidate* const> candidates,
                            std::span<const PlanningCandidateId> candidate_ids,
                            std::span<const FakeContinuationHandle* const> private_owners,
                            std::span<const PlanningOwnerId> private_owner_ids,
                            std::span<const FakeSharedPrefixHandle* const> shared_owners,
                            std::span<const PlanningOwnerId> shared_owner_ids,
                            std::span<const PlanningOwnerId> recency_order);

    [[nodiscard]] PrefillWork
    shared_capture_split_prefill_work(const FakeAdmissionCandidate& candidate,
                                      const FakePreparedPrompt&,
                                      std::span<const std::uint32_t> frontiers) const noexcept {
        PrefillWork work = candidate.identity.machine_work.remaining_prefill_work;
        work.attention_pairs += static_cast<std::uint64_t>(frontiers.size()) * 100U;
        return work;
    }

    [[nodiscard]] bool
    target_feasible(std::span<const FakeTargetDecision> decisions) const noexcept {
        if (pressure_units(decisions) < required_pressure_actions) { return false; }
        const bool has_eviction =
            std::any_of(decisions.begin(), decisions.end(),
                        [](const auto& decision) { return decision.evicts_continuation; });
        if (required_action_id && !has_eviction &&
            std::none_of(decisions.begin(), decisions.end(), [&](const auto& decision) {
                return decision.id == *required_action_id;
            })) {
            return false;
        }
        if (require_evictions &&
            std::any_of(decisions.begin(), decisions.end(),
                        [](const auto& decision) { return !decision.evicts_continuation; })) {
            return false;
        }
        return true;
    }

    [[nodiscard]] std::size_t
    pressure_units(std::span<const FakeTargetDecision> decisions) const noexcept {
        std::size_t units = 0;
        for (const FakeTargetDecision& decision : decisions) {
            units += decision.evicts_continuation ? eviction_pressure_action_units : 1U;
        }
        return units;
    }

    [[nodiscard]] ContextTransactionReserveStatus
    start_resource_transaction(FakeResourcePlan&& plan, FakePreparedPrompt&& prompt,
                               CancellationFlagView cancellation) {
        ++start_calls;
        started_source_id   = plan.admission.private_source_id;
        started_source_mode = plan.admission.source_mode;
        started_action_ids.clear();
        for (const auto& action : plan.private_actions) { started_action_ids.push_back(action.id); }
        for (const auto& action : plan.shared_actions) { started_action_ids.push_back(action.id); }
        if (cancellation.requested() || abort_start || plan.revision != revision_) {
            return ContextTransactionReserveStatus::Aborted;
        }
        pending_prompt_ = prompt;
        pending_plan_.emplace(std::move(plan));
        transaction_kind_ = TransactionKind::Materialization;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] std::optional<FakePersistentBackfillProof>
    prove_persistent_backfill(const FakeRequestBasePlan&, const FakeResourcePlan& candidate,
                              std::span<const FakeSequenceHandle>) const {
        if (candidate.resource_revision() != revision_) { return std::nullopt; }
        return FakePersistentBackfillProof{.revision = revision_};
    }

    [[nodiscard]] FakeContextTransactionProgress
    progress_context_transaction(CancellationFlagView cancellation) {
        require(transaction_kind_ != TransactionKind::None,
                "fake Program has no context transaction");
        if (progress_in_progress_once) {
            progress_in_progress_once = false;
            return ContextTransactionInProgress{};
        }
        if (transaction_kind_ == TransactionKind::Capture) {
            FakeActiveCaptureResult result;
            result.status =
                cancellation.requested() ? ContextTransactionStatus::Aborted : capture_status;
            if (pending_plan_) {
                for (std::size_t index = 0; index < pending_plan_->private_actions.size();
                     ++index) {
                    const FakeTargetDecision& action = pending_plan_->private_actions[index];
                    FakeMaterializationVictimResult victim{
                        .owner       = pending_plan_->private_planning_ids[index],
                        .disposition = action.evicts_continuation ? VictimDisposition::Evicted
                                                                  : VictimDisposition::Retained,
                        .pressure_committed = action.evicts_continuation ||
                                              result.status == ContextTransactionStatus::Published,
                    };
                    if (!action.evicts_continuation) {
                        const std::uint32_t owner = pending_plan_->private_owner_ids[index];
                        victim.final_summary.emplace();
                        const std::uint32_t content = sequence_content_keys_.at(owner);
                        if (action.dropped_checkpoints == 0 ||
                            malform_private_checkpoint_identity) {
                            victim.final_summary->endpoint = endpoint(content, finish_frontier);
                        }
                        if (finish_with_rewrite && (action.dropped_checkpoints == 0 ||
                                                    !malform_private_checkpoint_identity)) {
                            victim.final_summary->rewrite =
                                rewrite_checkpoint(content, finish_frontier - 1U);
                        }
                    }
                    result.victims.push_back(std::move(victim));
                }
                for (std::size_t index = 0; index < pending_plan_->shared_actions.size(); ++index) {
                    const FakeTargetDecision& action = pending_plan_->shared_actions[index];
                    FakeMaterializationSharedVictimResult victim{
                        .owner       = pending_plan_->shared_planning_ids[index],
                        .disposition = action.evicts_continuation ? VictimDisposition::Evicted
                                                                  : VictimDisposition::Retained,
                        .pressure_committed = action.evicts_continuation ||
                                              result.status == ContextTransactionStatus::Published,
                    };
                    if (!action.evicts_continuation) {
                        const std::uint32_t owner = pending_plan_->shared_owner_ids[index];
                        victim.final_summary      = FakeSharedPrefixSummary{
                                 .checkpoint = shared_checkpoint(owner, finish_frontier),
                        };
                    }
                    result.shared_victims.push_back(std::move(victim));
                }
            }
            if (malform_last_capture_private_victim && !result.victims.empty()) {
                FakeMaterializationVictimResult& victim = result.victims.back();
                victim.disposition                      = VictimDisposition::Evicted;
                victim.pressure_committed               = false;
                victim.final_summary.reset();
            }
            if (reverse_pressure_results) {
                std::reverse(result.victims.begin(), result.victims.end());
                std::reverse(result.shared_victims.begin(), result.shared_victims.end());
            }
            if (result.status == ContextTransactionStatus::Published) {
                result.active_summary = capture_summary;
                if (pending_capture_publish_shared_) {
                    FakeSharedPrefixHandle handle;
                    handle.id          = next_shared_id_++;
                    handle.content_key = capture_assessment.shortlist_key.digest;
                    handle.frontier    = capture_assessment.shortlist_key.frontier;
                    result.shared      = FakeSharedPrefixPublication{
                             .handle = std::move(handle),
                             .summary =
                            FakeSharedPrefixSummary{
                                     .checkpoint =
                                    shared_checkpoint(capture_assessment.shortlist_key.digest,
                                                           capture_assessment.shortlist_key.frontier),
                                     .active_references = 1,
                            },
                    };
                }
            }
            return result;
        }

        require(pending_plan_.has_value(), "fake materialization plan disappeared");
        const FakeResourcePlan& plan = *pending_plan_;
        FakeMaterializationResult result;
        result.status = (abort_progress || cancellation.requested())
                            ? ContextTransactionStatus::Aborted
                            : ContextTransactionStatus::Published;
        for (std::size_t index = 0; index < plan.private_actions.size(); ++index) {
            const FakeTargetDecision& action = plan.private_actions[index];
            const bool evicted               = action.evicts_continuation;
            FakeMaterializationVictimResult victim{
                .owner       = plan.private_planning_ids[index],
                .disposition = evicted ? VictimDisposition::Evicted : VictimDisposition::Retained,
                .pressure_committed =
                    evicted || result.status == ContextTransactionStatus::Published,
            };
            if (!evicted) {
                const std::uint32_t owner_id = plan.private_owner_ids.at(index);
                victim.final_summary.emplace();
                const std::uint32_t content = sequence_content_keys_.at(owner_id);
                if (action.dropped_checkpoints == 0 || malform_private_checkpoint_identity) {
                    victim.final_summary->endpoint = endpoint(content, finish_frontier);
                }
                if (finish_with_rewrite &&
                    (action.dropped_checkpoints == 0 || !malform_private_checkpoint_identity)) {
                    victim.final_summary->rewrite =
                        rewrite_checkpoint(content, finish_frontier - 1U);
                }
            }
            result.victims.push_back(std::move(victim));
        }
        for (std::size_t index = 0; index < plan.shared_actions.size(); ++index) {
            const FakeTargetDecision& action = plan.shared_actions[index];
            result.shared_victims.push_back(FakeMaterializationSharedVictimResult{
                .owner              = plan.shared_planning_ids[index],
                .disposition        = action.evicts_continuation ? VictimDisposition::Evicted
                                                                 : VictimDisposition::Retained,
                .pressure_committed = action.evicts_continuation ||
                                      result.status == ContextTransactionStatus::Published,
            });
        }
        if (malform_last_private_victim && !result.victims.empty()) {
            FakeMaterializationVictimResult& victim = result.victims.back();
            victim.disposition                      = VictimDisposition::Evicted;
            victim.pressure_committed               = false;
            victim.final_summary.reset();
        }
        if (reverse_pressure_results) {
            std::reverse(result.victims.begin(), result.victims.end());
            std::reverse(result.shared_victims.begin(), result.shared_victims.end());
        }
        if (plan.admission.private_source_id != 0) {
            result.source = FakeMaterializationSourceResult{
                .mode = result.status == ContextTransactionStatus::Aborted
                            ? PrivateSourceMode::Retain
                            : plan.admission.source_mode};
        }
        if (plan.admission.shared_source_id != 0) {
            result.shared_source = FakeMaterializationSharedSourceResult{};
            if (report_shared_source_summary) {
                const std::uint32_t references =
                    result.status == ContextTransactionStatus::Published
                        ? ++reported_shared_active_references
                        : reported_shared_active_references;
                FakeCheckpointSummary checkpoint =
                    shared_checkpoint(plan.admission.shared_source_content_key,
                                      plan.admission.shared_source_frontier);
                if (change_shared_source_residency_on_second_report && references >= 2) {
                    checkpoint.state_residency = ninfer::runtime::ReplicaResidency::Both;
                }
                result.shared_source->final_summary = FakeSharedPrefixSummary{
                    .checkpoint        = std::move(checkpoint),
                    .active_references = references,
                };
            }
        }
        if (result.status == ContextTransactionStatus::Published) {
            const std::uint32_t sequence_id        = next_sequence_id_++;
            sequence_content_keys_.at(sequence_id) = pending_prompt_.content_key;
            result.published = FakeStartResult{.sequence = FakeSequenceHandle{sequence_id}};
        }
        return result;
    }

    void finalize_context_transaction() noexcept {
        transaction_kind_ = TransactionKind::None;
        pending_plan_.reset();
        pending_capture_publish_shared_ = false;
    }

    [[nodiscard]] bool has_context_transaction() const noexcept {
        return transaction_kind_ != TransactionKind::None;
    }

    [[nodiscard]] FakeCaptureAssessment
    inspect_capture(const FakeCaptureOffer&, const FakeSharedPrefixHandle*,
                    const FakeSharedPrefixHandle*, std::optional<CheckpointRef> private_replacement,
                    bool permit_shared_publication) const {
        FakeCaptureAssessment assessment = capture_assessment;
        if (!permit_shared_publication) { assessment.publishes_shared = false; }
        last_capture_private_replacement = private_replacement;
        return assessment;
    }

    [[nodiscard]] std::vector<CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const FakeContinuationHandle&, CheckpointRef) const {
        return {fake_recovery_work(0)};
    }

    [[nodiscard]] std::vector<CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const FakeSharedPrefixHandle&, CheckpointRef) const {
        return {fake_recovery_work(0)};
    }

    [[nodiscard]] FakePressurePlanningSession
    begin_capture_pressure_planning(const FakeCaptureAssessment& assessment,
                                    std::span<const FakeContinuationHandle* const> private_owners,
                                    std::span<const PlanningOwnerId> private_owner_ids,
                                    std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                    std::span<const PlanningOwnerId> shared_owner_ids,
                                    std::span<const PlanningOwnerId> recency_order) {
        capture_pressure_candidate_       = std::make_unique<FakeAdmissionCandidate>();
        FakeAdmissionCandidate& candidate = *capture_pressure_candidate_;
        candidate.value.prompt_tokens     = assessment.shortlist_key.frontier;
        for (const ContextTransferRequirement& transfer : assessment.transfer_requirements) {
            const std::size_t direction = static_cast<std::size_t>(transfer.direction);
            candidate.identity.machine_work.candidate_transfers[direction].payload_bytes +=
                transfer.work.payload_bytes;
            candidate.identity.machine_work.candidate_transfers[direction].copy_operations +=
                transfer.work.copy_operations;
        }
        candidate.identity.machine_work.optimistic_candidate_transfers =
            candidate.identity.machine_work.candidate_transfers;
        candidate.identity.physical_status =
            target_feasible(std::span<const FakeTargetDecision>{})
                ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                : ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        candidate.identity.expandable = candidate.identity.physical_status !=
                                        ninfer::runtime::MaterializationPhysicalStatus::Feasible;
        candidate.identity.projection_work = 1;
        candidate.identity.assessment_digest =
            (static_cast<std::uint64_t>(assessment.shortlist_key.frontier) << 32U) ^
            revision_.value;
        const FakeAdmissionCandidate* candidate_handle = &candidate;
        const std::array candidate_ids{FakePressurePlanningSession::candidate_id()};
        return begin_pressure_planning(
            std::span<const FakeAdmissionCandidate* const>(&candidate_handle, 1), candidate_ids,
            private_owners, private_owner_ids, shared_owners, shared_owner_ids, recency_order);
    }

    [[nodiscard]] bool shared_capture_matches(const FakeCaptureOffer&,
                                              const FakeSharedPrefixHandle&) const {
        return false;
    }

    void skip_capture(FakeCaptureOffer&&) { ++skipped_captures; }

    [[nodiscard]] ContextTransactionReserveStatus
    reserve_active_capture(FakeCaptureOffer&&, const FakeSharedPrefixHandle*,
                           const FakeSharedPrefixHandle*, std::optional<CheckpointRef>, bool,
                           CancellationFlagView cancellation) {
        if (cancellation.requested() || abort_capture_start) {
            return ContextTransactionReserveStatus::Aborted;
        }
        pending_plan_.reset();
        pending_capture_publish_shared_ = false;
        transaction_kind_               = TransactionKind::Capture;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        FakeCaptureOffer&&, const FakeSharedPrefixHandle*, const FakeSharedPrefixHandle*,
        std::optional<CheckpointRef>, bool publish_shared, FakeResourcePlan&& pressure,
        CancellationFlagView cancellation) {
        if (cancellation.requested() || abort_capture_start || pressure.revision != revision_) {
            return ContextTransactionReserveStatus::Aborted;
        }
        started_action_ids.clear();
        for (const auto& action : pressure.private_actions) {
            started_action_ids.push_back(action.id);
        }
        for (const auto& action : pressure.shared_actions) {
            started_action_ids.push_back(action.id);
        }
        pending_plan_.emplace(std::move(pressure));
        pending_capture_publish_shared_ = publish_shared;
        transaction_kind_               = TransactionKind::Capture;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] FakeFinishResult finish(FakeSequenceHandle sequence) noexcept {
        ++finish_calls;
        if (finish_fail_next) {
            finish_fail_next = false;
            return {};
        }
        advance_revision();
        FakeFinishResult result;
        result.status = ConsumeStatus::Consumed;
        if (finish_release) {
            result.disposition = FinishDisposition::Released;
            return result;
        }
        result.disposition      = FinishDisposition::Catalogued;
        const std::uint32_t key = sequence_content_keys_[sequence.id];
        result.summary.endpoint = endpoint(key, finish_frontier);
        if (finish_with_rewrite) {
            result.summary.rewrite = rewrite_checkpoint(key, finish_frontier - 1U);
        }
        result.continuation.emplace(sequence.id, key);
        return result;
    }

    [[nodiscard]] FakeAbortResult abort(FakeSequenceHandle sequence) noexcept {
        ++abort_calls;
        advance_revision();
        FakeAbortResult result{.status      = ConsumeStatus::Consumed,
                               .timings     = FakeTimings{.value = 7},
                               .speculative = FakeSpeculativeStats{.value = 9}};
        if (abort_salvage_next) {
            abort_salvage_next      = false;
            const std::uint32_t key = sequence_content_keys_[sequence.id];
            result.salvaged         = true;
            result.summary.endpoint = endpoint(key, finish_frontier);
            result.continuation.emplace(sequence.id, key);
        }
        return result;
    }

    struct FakeDeviceKVPages {
        std::uint32_t main    = 0;
        std::uint32_t backend = 0;
    };

    // Device pages a retained owner would free; every owner holds one main page unless a test
    // says otherwise (a Host-only owner frees none).
    [[nodiscard]] FakeDeviceKVPages
    retained_device_kv_pages(const FakeContinuationHandle& continuation) const noexcept {
        const auto found = retained_kv_pages.find(continuation.content_key);
        return found == retained_kv_pages.end() ? FakeDeviceKVPages{.main = 1} : found->second;
    }

    [[nodiscard]] FakeDeviceKVPages
    retained_device_kv_pages(const FakeSharedPrefixHandle& shared) const noexcept {
        const auto found = retained_kv_pages.find(shared.content_key);
        return found == retained_kv_pages.end() ? FakeDeviceKVPages{.main = 1} : found->second;
    }

    [[nodiscard]] FakeReleaseResult
    release_continuation(FakeContinuationHandle&& continuation) noexcept {
        released_continuations.push_back(continuation.id);
        if (capture_feasible_after_release) { capture_assessment.physically_feasible = true; }
        released_continuation_keys.push_back(continuation.content_key);
        advance_revision();
        return FakeReleaseResult{.status = ConsumeStatus::Consumed};
    }

    [[nodiscard]] FakeReleaseResult
    release_shared_prefix(FakeSharedPrefixHandle&& shared) noexcept {
        released_shared_prefix_keys.push_back(shared.content_key);
        advance_revision();
        return FakeReleaseResult{.status = ConsumeStatus::Consumed};
    }

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept { return revision_; }

    [[nodiscard]] FakePhysicalUsage physical_usage() const noexcept { return usage; }

    void invalidate_resources() noexcept { advance_revision(); }

    std::vector<std::pair<std::uint32_t, std::vector<FakeTargetDecision>>> owner_decisions;
    std::size_t required_pressure_actions       = 0;
    std::size_t eviction_pressure_action_units  = 1;
    std::uint32_t private_pressure_alternatives = 1;
    std::optional<std::size_t> pressure_optional_target_capacity;
    std::uint64_t pressure_action_immediate_ns      = 100'000'000;
    std::uint32_t pressure_action_degradation_units = 1;
    bool include_cumulative_private_target          = false;
    bool combined_target_cancels_pressure_copy      = false;
    std::optional<std::uint64_t> pressure_target_immediate_ns_override;
    std::optional<std::uint64_t> required_action_id;
    std::uint32_t pressure_assessment_delay_us           = 0;
    std::uint64_t pressure_checkpoint_recovery_ns        = 100;
    bool require_evictions                               = false;
    // Ladder-feasibility test mode: when true, a recency-ladder rung (sacrifice_oldest >= 0) is
    // feasible iff its sacrifice is >= `min_feasible_sacrifice` (modeling "host only fits the
    // demotions after that many oldest preserved owners are given up"), and the terminal
    // clear-all (root_maximal, sacrifice_oldest < 0) is always feasible. Search-path targets
    // (not a rung, not root_maximal) keep using `target_feasible`.
    bool ladder_feasibility_mode                         = false;
    std::uint32_t min_feasible_sacrifice                 = 0;
    bool abort_start                                     = false;
    bool abort_progress                                  = false;
    bool malform_last_private_victim                     = false;
    bool malform_last_capture_private_victim             = false;
    bool malform_private_checkpoint_identity             = false;
    bool reverse_pressure_results                        = false;
    bool progress_in_progress_once                       = false;
    bool finish_fail_next                                = false;
    // Seal-window and capture-seal failure injection.
    bool deny_seal_window                                = false;
    bool fail_capture_seal                               = false;
    std::uint64_t seal_window_claims                     = 0;
    bool finish_release                                  = false;
    bool finish_with_rewrite                             = false;
    // The private replacement the most recent capture inspection was asked to price.
    mutable std::optional<CheckpointRef> last_capture_private_replacement;
    bool abort_salvage_next                              = false;
    bool abort_capture_start                             = false;
    bool report_shared_source_summary                    = false;
    bool change_shared_source_residency_on_second_report = false;
    std::uint32_t reported_shared_active_references      = 0;
    ContextTransactionStatus capture_status              = ContextTransactionStatus::Published;
    FakeCaptureAssessment capture_assessment;
    FakeContinuationSummary capture_summary;
    FakePhysicalUsage usage;

    std::uint64_t admission_inspections       = 0;
    std::uint64_t pressure_planning_sessions  = 0;
    // Continuation ids of the private owners in the recency order handed to the most recent
    // pressure planning session (most recent first).
    std::vector<std::uint32_t> observed_recency_ids;
    std::uint64_t pressure_target_assessments = 0;
    bool capture_feasible_after_release        = false;
    std::uint64_t start_calls                 = 0;
    std::uint64_t finish_calls                = 0;
    std::uint64_t abort_calls                 = 0;
    std::uint64_t skipped_captures            = 0;
    std::size_t pressure_target_count_peak    = 0;
    std::uint32_t finish_frontier             = 16;
    std::uint32_t started_source_id           = 0;
    PrivateSourceMode started_source_mode     = PrivateSourceMode::ConsumeToActive;
    std::vector<std::uint32_t> inspected_private_sources;
    std::vector<std::uint32_t> inspected_shared_sources;
    std::vector<std::vector<std::uint64_t>> seal_attempts;
    std::vector<std::uint64_t> started_action_ids;
    std::vector<std::uint32_t> selected_shared_capture_frontiers;
    std::vector<std::uint32_t> released_continuations;
    std::vector<std::uint32_t> released_continuation_keys;
    std::map<std::uint32_t, FakeDeviceKVPages> retained_kv_pages;
    std::vector<std::uint32_t> released_shared_prefix_keys;

private:
    void advance_revision() noexcept {
        if (++revision_.value == 0) { ++revision_.value; }
    }

    ProgramResourceRevision revision_{.value = 1};
    std::uint32_t planning_generation_ = 0;
    std::uint32_t next_sequence_id_    = 1;
    std::uint32_t next_shared_id_      = 1;
    std::array<std::uint32_t, 256> sequence_content_keys_{};
    TransactionKind transaction_kind_ = TransactionKind::None;
    FakePreparedPrompt pending_prompt_;
    std::optional<FakeResourcePlan> pending_plan_;
    std::unique_ptr<FakeAdmissionCandidate> capture_pressure_candidate_;
    bool pending_capture_publish_shared_ = false;
};

FakePressurePlanningSession::FakePressurePlanningSession(
    FakeProgram& program, std::span<const FakeAdmissionCandidate* const> candidates,
    std::span<const PlanningCandidateId> candidate_ids,
    std::span<const FakeContinuationHandle* const> private_owners,
    std::span<const PlanningOwnerId> private_owner_ids,
    std::span<const FakeSharedPrefixHandle* const> shared_owners,
    std::span<const PlanningOwnerId> shared_owner_ids,
    std::span<const PlanningOwnerId> recency_order)
    : program_(&program), revision_(program.resource_revision()) {
    require(!candidates.empty() && candidates.size() == candidate_ids.size(),
            "fake pressure session has no candidate identity");
    require(private_owners.size() == private_owner_ids.size() &&
                shared_owners.size() == shared_owner_ids.size(),
            "fake pressure owner arrays are not aligned");
    candidates_.assign(candidates.begin(), candidates.end());
    candidate_ids_.assign(candidate_ids.begin(), candidate_ids.end());
    for (std::size_t index = 0; index < private_owners.size(); ++index) {
        owners_.push_back(Owner{.private_handle = private_owners[index],
                                .id             = private_owner_ids[index],
                                .shared         = false});
    }
    for (std::size_t index = 0; index < shared_owners.size(); ++index) {
        owners_.push_back(Owner{
            .shared_handle = shared_owners[index], .id = shared_owner_ids[index], .shared = true});
    }
    std::sort(owners_.begin(), owners_.end(),
              [](const Owner& left, const Owner& right) { return left.id.value < right.id.value; });
    recency_rank_.assign(owners_.size(), -1);
    // `recency_order` is most-recent-first (like the real
    // rank_owners_by_recency), so its index is the recency rank (0 = most recent) that
    // orders the ladder sacrifice.
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const auto found = std::find(recency_order.begin(), recency_order.end(), owners_[index].id);
        if (found != recency_order.end()) {
            recency_rank_[index] = static_cast<std::int32_t>(found - recency_order.begin());
            ++ranked_owner_count_;
        }
    }
    program.observed_recency_ids.clear();
    for (const PlanningOwnerId ranked : recency_order) {
        for (std::size_t index = 0; index < private_owner_ids.size(); ++index) {
            if (private_owner_ids[index] == ranked) {
                program.observed_recency_ids.push_back(private_owners[index]->id);
            }
        }
    }
    // Mirror the real session default: the whole recency order is evictable until the
    // admission planner narrows it.
    eviction_licence_count_ = ranked_owner_count_;
    recency_eviction_       = !recency_order.empty();
    options_.resize(candidates_.size());
    options_populated_.resize(candidates_.size());
    for (std::size_t index = 0; index < candidates_.size(); ++index) {
        require(candidates_[index] != nullptr, "fake pressure candidate is null");
        targets_.push_back(Target{
            .candidate_index = static_cast<std::uint32_t>(index),
            .choices         = std::vector<std::uint16_t>(owners_.size(), 0),
            .stable_ordinal  = static_cast<std::uint32_t>(index),
        });
    }
    program.pressure_target_count_peak =
        std::max(program.pressure_target_count_peak, targets_.size());
    if (++program.planning_generation_ == 0) { ++program.planning_generation_; }
    ++program.pressure_planning_sessions;
    generation_ = program.planning_generation_;
}

bool FakePressurePlanningSession::valid(FakePressureTargetHandle target) const noexcept {
    return program_ != nullptr && target.generation == generation_ &&
           target.index < targets_.size() && program_->resource_revision() == revision_;
}

std::uint32_t FakePressurePlanningSession::candidate_index(PlanningCandidateId candidate) const {
    const auto found = std::find(candidate_ids_.begin(), candidate_ids_.end(), candidate);
    require(found != candidate_ids_.end(), "fake pressure candidate is foreign");
    return static_cast<std::uint32_t>(found - candidate_ids_.begin());
}

bool FakePressurePlanningSession::same_target(const Target& left,
                                              const Target& right) const noexcept {
    return left.candidate_index == right.candidate_index && left.choices == right.choices;
}

std::vector<FakeTargetDecision>
FakePressurePlanningSession::decisions_for(std::uint32_t selected_candidate,
                                           std::size_t owner_index) const {
    const FakeAdmissionCandidate& candidate = *candidates_.at(selected_candidate);
    const Owner& owner                      = owners_.at(owner_index);
    if ((!owner.shared && candidate.private_source_id != 0 &&
         candidate.private_source_id == owner.private_handle->id) ||
        (owner.shared && candidate.shared_source_id != 0 &&
         candidate.shared_source_id == owner.shared_handle->id)) {
        return {};
    }

    if (!owner.shared) {
        for (const auto& [id, decisions] : program_->owner_decisions) {
            if (id == owner.private_handle->id) { return decisions; }
        }
    }
    std::vector<FakeTargetDecision> decisions;
    if (!owner.shared) {
        for (std::uint32_t index = 0; index < program_->private_pressure_alternatives; ++index) {
            decisions.push_back(FakeTargetDecision{
                .id = 1000U + owner.private_handle->id + 10000U * static_cast<std::uint64_t>(index),
                .immediate_ns      = program_->pressure_action_immediate_ns,
                .degradation_units = program_->pressure_action_degradation_units,
            });
        }
        if (program_->include_cumulative_private_target) {
            decisions.push_back(FakeTargetDecision{
                .id                  = 5000U + owner.private_handle->id,
                .degradation_units   = 2,
                .dropped_checkpoints = 1,
            });
        }
        decisions.push_back(FakeTargetDecision{
            .id                  = 2000U + owner.private_handle->id,
            .degradation_units   = 4,
            .dropped_checkpoints = 1,
            .evicts_continuation = true,
        });
    } else {
        decisions.push_back(FakeTargetDecision{
            .id           = 3000U + owner.shared_handle->id,
            .shared_owner = true,
        });
        decisions.push_back(FakeTargetDecision{
            .id                  = 4000U + owner.shared_handle->id,
            .degradation_units   = 4,
            .dropped_checkpoints = 1,
            .evicts_continuation = true,
            .shared_owner        = true,
        });
    }
    return decisions;
}

void FakePressurePlanningSession::populate_options(std::uint32_t selected_candidate) {
    require(selected_candidate < candidates_.size(), "fake pressure candidate index is invalid");
    if (options_populated_[selected_candidate] != 0) { return; }
    options_[selected_candidate].resize(owners_.size());
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        options_[selected_candidate][index] = decisions_for(selected_candidate, index);
    }
    options_populated_[selected_candidate] = 1;
}

FakePressureTargetHandle
FakePressurePlanningSession::identity_target(PlanningCandidateId candidate) const {
    return FakePressureTargetHandle{.generation = generation_, .index = candidate_index(candidate)};
}

FakePressureTargetHandle FakePressurePlanningSession::identity_target() const {
    require(candidates_.size() == 1, "fake capture pressure domain has multiple candidates");
    return FakePressureTargetHandle{.generation = generation_, .index = 0};
}

FakePressureTargetHandle
FakePressurePlanningSession::root_maximal_target(PlanningCandidateId candidate) {
    const std::uint32_t selected = candidate_index(candidate);
    populate_options(selected);
    Target maximal{
        .candidate_index = selected,
        .choices         = std::vector<std::uint16_t>(owners_.size(), 0),
        .root_maximal    = true,
    };
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        maximal.choices[index] = static_cast<std::uint16_t>(options_[selected][index].size());
    }
    auto found = std::find_if(targets_.begin(), targets_.end(),
                              [&](const Target& target) { return same_target(target, maximal); });
    if (found != targets_.end()) {
        found->root_maximal = true;
        return FakePressureTargetHandle{
            .generation = generation_,
            .index      = static_cast<std::uint32_t>(found - targets_.begin()),
        };
    }
    maximal.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
    targets_.push_back(std::move(maximal));
    return FakePressureTargetHandle{
        .generation = generation_,
        .index      = static_cast<std::uint32_t>(targets_.size() - 1U),
    };
}

FakePressureTargetHandle
FakePressurePlanningSession::maximal_target(PlanningCandidateId candidate) {
    const auto target                   = root_maximal_target(candidate);
    targets_[target.index].root_maximal = false;
    return target;
}

std::uint32_t FakePressurePlanningSession::ranked_owner_count() const {
    return ranked_owner_count_;
}

void FakePressurePlanningSession::set_eviction_licence(
    std::uint32_t oldest_licensed, std::span<const std::uint32_t> spared_ranks) {
    eviction_licence_count_ = std::min(oldest_licensed, ranked_owner_count_);
    licence_spared_ranks_.assign(spared_ranks.begin(), spared_ranks.end());
}

bool FakePressurePlanningSession::owner_eviction_licensed(std::uint32_t owner_index) const {
    if (owner_index >= recency_rank_.size() || eviction_licence_count_ == 0) { return false; }
    const std::int32_t rank = recency_rank_[owner_index];
    if (rank < 0) { return true; }
    return static_cast<std::uint32_t>(rank) >= ranked_owner_count_ - eviction_licence_count_ &&
           std::find(licence_spared_ranks_.begin(), licence_spared_ranks_.end(),
                     static_cast<std::uint32_t>(rank)) == licence_spared_ranks_.end();
}

FakePressureTargetHandle FakePressurePlanningSession::recency_maximal_target(
    PlanningCandidateId candidate, std::uint32_t sacrifice_oldest,
    std::span<const std::uint32_t> spared_ranks, bool demote_kept) {
    const std::uint32_t selected = candidate_index(candidate);
    populate_options(selected);
    const std::uint32_t demote_count =
        sacrifice_oldest < ranked_owner_count_ ? ranked_owner_count_ - sacrifice_oldest : 0;
    // `sacrifice_oldest` on the target records how many ranked owners the rung actually gives up
    // (the requested tail less the spared ranks), which is what ladder-feasibility mode tests.
    const auto spared_in_tail = static_cast<std::uint32_t>(
        std::count_if(spared_ranks.begin(), spared_ranks.end(), [&](std::uint32_t rank) {
            return rank >= demote_count && rank < ranked_owner_count_;
        }));
    Target rung{
        .candidate_index  = selected,
        .choices          = std::vector<std::uint16_t>(owners_.size(), 0),
        .root_maximal     = true,
        .sacrifice_oldest = static_cast<std::int32_t>(
            std::min(sacrifice_oldest, ranked_owner_count_) - spared_in_tail),
    };
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const auto& alternatives = options_[selected][index];
        const std::int32_t rank  = recency_rank_[index];
        // The rung fully evicts the sacrificed (oldest ranked, or unranked) owners, keeps every
        // other owner as it is, and demotes a kept private owner when a non-evict (demote) outcome
        // exists. Keeping is choice 0: whether the sacrifice frees enough is what adoption
        // decides.
        const bool in_tail = rank < 0 || static_cast<std::uint32_t>(rank) >= demote_count;
        const bool spared  = in_tail && rank >= 0 &&
                             std::find(spared_ranks.begin(), spared_ranks.end(),
                                       static_cast<std::uint32_t>(rank)) != spared_ranks.end();
        // A spared owner stays at the baseline here. (The real session demotes it with the other
        // kept owners; this fake's expansion ignores the eviction licence, so a costlier spared
        // owner would let it reach evictions the real search cannot.)
        std::uint16_t choice = static_cast<std::uint16_t>(alternatives.size());
        if (spared) {
            choice = 0U;
        } else if (!in_tail) {
            choice = demote_kept && alternatives.size() > 1U && !alternatives[0].evicts_continuation
                         ? static_cast<std::uint16_t>(1)
                         : 0U;
        }
        rung.choices[index] = choice;
    }
    auto found = std::find_if(targets_.begin(), targets_.end(), [&](const Target& target) {
        return same_target(target, rung) && target.sacrifice_oldest == rung.sacrifice_oldest;
    });
    if (found != targets_.end()) {
        found->root_maximal = true;
        return FakePressureTargetHandle{
            .generation = generation_,
            .index      = static_cast<std::uint32_t>(found - targets_.begin()),
        };
    }
    rung.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
    targets_.push_back(std::move(rung));
    return FakePressureTargetHandle{
        .generation = generation_,
        .index      = static_cast<std::uint32_t>(targets_.size() - 1U),
    };
}

FakePressurePlanningSession::Cursor
FakePressurePlanningSession::begin_construction(FakePressureTargetHandle handle, bool restore) {
    require(valid(handle) && !scratch_live_, "invalid fake construction parent");
    Cursor cursor;
    cursor.target     = targets_[handle.index];
    cursor.restore    = restore;
    cursor.generation = ++construction_generation_;
    populate_options(cursor.target.candidate_index);
    return cursor;
}

ninfer::runtime::PressureConstructionStep
FakePressurePlanningSession::next_construction_option(Cursor& cursor) {
    if (cursor.next_option < cursor.options.size()) {
        auto index = static_cast<std::uint32_t>(cursor.next_option++);
        return {.guidance = guidance_for(cursor.options[index]),
                .option   = {.cursor_generation = cursor.generation,
                             .scan_generation   = cursor.scan,
                             .index             = index}};
    }
    if (cursor.next_owner == owners_.size()) { return {.exhausted = true}; }
    const auto owner         = cursor.next_owner++;
    const auto& alternatives = options_[cursor.target.candidate_index][owner];
    const auto current       = cursor.target.choices[owner];
    for (std::size_t choice = cursor.restore ? 0 : 1; choice <= alternatives.size(); ++choice) {
        if (choice == current) { continue; }
        // choice 0 is the baseline (no alternative), so it has no `alternatives[-1]` to consult;
        // with a recency order only a licensed owner's eviction is reachable.
        if (recency_eviction_ && choice != 0 && alternatives[choice - 1].evicts_continuation &&
            !owner_eviction_licensed(static_cast<std::uint32_t>(owner))) {
            continue;
        }
        if (!cursor.restore && current != 0 &&
            (alternatives[current - 1].evicts_continuation ||
             !alternatives[choice - 1].evicts_continuation)) {
            continue;
        }
        Target child         = cursor.target;
        child.choices[owner] = static_cast<std::uint16_t>(choice);
        child.root_maximal   = false;
        cursor.options.push_back(std::move(child));
    }
    return {};
}

void FakePressurePlanningSession::choose_construction(
    Cursor& cursor, ninfer::runtime::PressureConstructionOptionId id) {
    require(id.cursor_generation == cursor.generation && id.scan_generation == cursor.scan &&
                id.index < cursor.options.size(),
            "stale fake construction option");
    cursor.target = cursor.options[id.index];
    cursor.options.clear();
    cursor.next_owner = cursor.next_option = 0;
    ++cursor.scan;
}

std::optional<FakePressureTargetHandle>
FakePressurePlanningSession::construction_target(const Cursor& cursor) {
    auto found = std::find_if(targets_.begin(), targets_.end(), [&](const Target& other) {
        return same_target(other, cursor.target);
    });
    if (found == targets_.end()) {
        if (targets_.size() >= candidates_.size() + 1U + 4096U) { return std::nullopt; }
        auto target           = cursor.target;
        target.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
        targets_.push_back(std::move(target));
        program_->pressure_target_count_peak =
            std::max(program_->pressure_target_count_peak, targets_.size());
        return FakePressureTargetHandle{.generation = generation_,
                                        .index = static_cast<std::uint32_t>(targets_.size() - 1)};
    }
    return FakePressureTargetHandle{.generation = generation_,
                                    .index = static_cast<std::uint32_t>(found - targets_.begin())};
}

std::uint32_t FakePressurePlanningSession::optional_targets_remaining() const noexcept {
    if (program_ == nullptr) { return 0; }
    const std::size_t maximum =
        program_->pressure_optional_target_capacity
            ? candidates_.size() + 1U + *program_->pressure_optional_target_capacity
            : candidates_.size() + 1U + 4096U;
    return targets_.size() >= maximum ? 0U : static_cast<std::uint32_t>(maximum - targets_.size());
}

ninfer::runtime::PressureTargetGuidance
FakePressurePlanningSession::guidance(FakePressureTargetHandle handle) {
    require(valid(handle) && !scratch_live_, "fake pressure guidance is stale");
    return guidance_for(targets_[handle.index]);
}

ninfer::runtime::PressureTargetGuidance
FakePressurePlanningSession::guidance_for(const Target& target) {
    populate_options(target.candidate_index);
    const FakeAdmissionCandidate& candidate = *candidates_[target.candidate_index];
    guidance_outcomes_.clear();
    std::vector<FakeTargetDecision> selected;
    std::uint32_t degradation_units = 0;
    std::uint32_t dropped           = 0;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        if (choice == 0) { continue; }
        const auto& alternatives = options_[target.candidate_index][index];
        require(choice <= alternatives.size(), "fake pressure guidance choice is invalid");
        const FakeTargetDecision& decision = alternatives[choice - 1U];
        selected.push_back(decision);
        degradation_units += decision.degradation_units;
        dropped += decision.dropped_checkpoints;
        guidance_outcomes_.push_back(ninfer::runtime::PressureOwnerOutcome{
            .owner               = owners_[index].id,
            .disposition         = decision.evicts_continuation ? VictimDisposition::Evicted
                                                                : VictimDisposition::Retained,
            .degradation_units   = decision.degradation_units,
            .dropped_checkpoints = decision.dropped_checkpoints,
        });
    }
    MaterializationMachineWork machine = candidate.identity.machine_work;
    const bool combined_copy_cancelled =
        program_->combined_target_cancels_pressure_copy && selected.size() > 1U &&
        std::none_of(selected.begin(), selected.end(),
                     [](const auto& decision) { return decision.evicts_continuation; });
    if (!selected.empty() && program_->pressure_target_immediate_ns_override) {
        const auto optimistic = machine.optimistic_candidate_transfers;
        set_fake_machine_costs(machine, 0, *program_->pressure_target_immediate_ns_override);
        machine.optimistic_candidate_transfers = optimistic;
    } else if (combined_copy_cancelled) {
        ++machine.pressure_transfers[2].payload_bytes;
    } else {
        for (const FakeTargetDecision& decision : selected) {
            machine.pressure_transfers[2].payload_bytes += decision.immediate_ns;
            ++machine.pressure_transfers[2].copy_operations;
        }
    }
    const std::size_t selected_units = program_->pressure_units(selected);
    const std::size_t remaining      = selected_units >= program_->required_pressure_actions
                                           ? 0
                                           : program_->required_pressure_actions - selected_units;
    return ninfer::runtime::PressureTargetGuidance{
        .physical =
            {
                .unsatisfied_constraints   = remaining == 0 ? 0U : 1U,
                .estimated_remaining_steps = static_cast<std::uint32_t>(remaining),
                .normalized_residual_q20   = static_cast<std::uint64_t>(remaining) << 20U,
            },
        .estimated_machine_work     = machine,
        .owner_outcomes             = guidance_outcomes_,
        .candidate                  = candidate_ids_[target.candidate_index],
        .stable_target_ordinal      = target.stable_ordinal,
        .degradation_units          = degradation_units,
        .dropped_checkpoints        = dropped,
        .source_mode                = candidate.source_mode,
        .recovery_estimate_complete = true,
    };
}

FakeAssessedPressureTarget FakePressurePlanningSession::assess(FakePressureTargetHandle handle) {
    require(valid(handle) && !scratch_live_, "fake pressure assessment is stale");
    if (program_->pressure_assessment_delay_us != 0) {
        // Spin rather than sleep: the delay models assessment cost against the planner's wall
        // budget, and sleep_for rounds up to the OS timer tick (~15.6 ms on Windows), which turns
        // a 1 ms assessment into one that exhausts the 50 ms admission allowance in three steps.
        const auto until = std::chrono::steady_clock::now() +
                           std::chrono::microseconds(program_->pressure_assessment_delay_us);
        while (std::chrono::steady_clock::now() < until) {}
    }
    ++program_->pressure_target_assessments;
    const Target& target = targets_[handle.index];
    populate_options(target.candidate_index);
    const FakeAdmissionCandidate& candidate = *candidates_[target.candidate_index];
    assessment_outcomes_.clear();
    assessment_impacts_.clear();
    assessment_recovery_work_.clear();
    assessment_recovery_work_.reserve(2U * owners_.size());
    std::vector<FakeTargetDecision> selected;
    std::uint32_t degradation_units = 0;
    std::uint32_t dropped           = 0;
    bool expandable                 = false;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        const auto& alternatives   = options_[target.candidate_index][index];
        if (choice == 0) {
            expandable = expandable || !alternatives.empty();
            continue;
        }
        require(choice <= alternatives.size(), "fake pressure target choice is invalid");
        const FakeTargetDecision& decision = alternatives[choice - 1U];
        selected.push_back(decision);
        degradation_units += decision.degradation_units;
        dropped += decision.dropped_checkpoints;
        assessment_outcomes_.push_back(ninfer::runtime::PressureOwnerOutcome{
            .owner               = owners_[index].id,
            .disposition         = decision.evicts_continuation ? VictimDisposition::Evicted
                                                                : VictimDisposition::Retained,
            .degradation_units   = decision.degradation_units,
            .dropped_checkpoints = decision.dropped_checkpoints,
        });
        const auto append_checkpoint_outcome = [&](CheckpointRef checkpoint, bool survives) {
            assessment_recovery_work_.push_back(
                fake_recovery_work(survives ? 0 : program_->pressure_checkpoint_recovery_ns));
            assessment_impacts_.push_back(ninfer::runtime::PressureCheckpointRecoveryImpact{
                .owner                = owners_[index].id,
                .checkpoint           = checkpoint,
                .target_recovery_work = std::span<const CheckpointRecoveryAlternativeWork>(
                    &assessment_recovery_work_.back(), 1),
                .survives = survives,
            });
        };
        const bool endpoint_survives =
            !decision.evicts_continuation && decision.dropped_checkpoints == 0;
        const std::uint32_t owner_frontier =
            owners_[index].shared && owners_[index].shared_handle->frontier != 0
                ? owners_[index].shared_handle->frontier
                : program_->finish_frontier;
        append_checkpoint_outcome(CheckpointRef{.kind     = owners_[index].shared
                                                                ? CheckpointKind::SharedStablePrefix
                                                                : CheckpointKind::SessionEndpoint,
                                                .frontier = owner_frontier,
                                                .ordinal  = 0},
                                  endpoint_survives);
        if (!owners_[index].shared && program_->finish_with_rewrite) {
            append_checkpoint_outcome(CheckpointRef{.kind     = CheckpointKind::TurnClosure,
                                                    .frontier = program_->finish_frontier - 1U,
                                                    .ordinal  = 0},
                                      !decision.evicts_continuation);
        }
        if (!decision.evicts_continuation) { expandable = true; }
    }

    MaterializationMachineWork machine = candidate.identity.machine_work;
    const bool combined_copy_cancelled =
        program_->combined_target_cancels_pressure_copy && selected.size() > 1U &&
        std::none_of(selected.begin(), selected.end(),
                     [](const auto& decision) { return decision.evicts_continuation; });
    if (!selected.empty() && program_->pressure_target_immediate_ns_override) {
        const auto optimistic = machine.optimistic_candidate_transfers;
        set_fake_machine_costs(machine, 0, *program_->pressure_target_immediate_ns_override);
        machine.optimistic_candidate_transfers = optimistic;
    } else if (combined_copy_cancelled) {
        // The complete target removes a transfer required by each partial target.  This models
        // source Move replacing Fork, a later eviction cancelling an earlier D2H, or two physical
        // actions coalescing into one direct stage.  Exact target cost is therefore intentionally
        // non-monotonic along the search edge.
        ++machine.pressure_transfers[2].payload_bytes;
    } else {
        for (const FakeTargetDecision& decision : selected) {
            machine.pressure_transfers[2].payload_bytes += decision.immediate_ns;
            ++machine.pressure_transfers[2].copy_operations;
        }
    }
    const bool identity  = selected.empty();
    // Ladder-feasibility test mode: a recency-ladder rung is feasible once the host can absorb
    // the demotions (sacrifice >= min_feasible_sacrifice); the terminal clear-all always is.
    // Search-path targets (not a rung, not root_maximal) keep using `target_feasible`.
    bool feasible;
    if (program_->ladder_feasibility_mode) {
        if (target.sacrifice_oldest >= 0) {
            feasible = static_cast<std::uint32_t>(target.sacrifice_oldest) >=
                       program_->min_feasible_sacrifice;
        } else if (target.root_maximal) {
            feasible = true;
        } else {
            feasible = program_->target_feasible(selected);
        }
    } else {
        feasible = program_->target_feasible(selected);
    }
    std::uint64_t digest = candidate.identity.assessment_digest;
    if (!identity) {
        digest = 1469598103934665603ULL;
        for (const std::uint16_t choice : target.choices) {
            digest ^= choice;
            digest *= 1099511628211ULL;
        }
        digest ^= target.candidate_index;
    }
    ninfer::runtime::PressureTargetAssessment assessment{
        .physical_status    = feasible ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                                       : ninfer::runtime::MaterializationPhysicalStatus::Infeasible,
        .source_mode        = candidate.source_mode,
        .machine_work       = machine,
        .owner_outcomes     = assessment_outcomes_,
        .checkpoint_impacts = assessment_impacts_,
        .candidate          = candidate_ids_[target.candidate_index],
        .stable_target_ordinal = target.stable_ordinal,
        .degradation_units     = degradation_units,
        .dropped_checkpoints   = dropped,
        .projection_work       = 1U + assessment_outcomes_.size(),
        .assessment_digest     = digest,
        .expandable            = expandable,
        .root_maximal          = target.root_maximal,
    };
    return FakeAssessedPressureTarget(handle, assessment);
}

FakePreparedPressureExpansion
FakePressurePlanningSession::prepare_expansion(FakePressureTargetHandle handle,
                                               std::uint32_t maximum_owners) {
    require(valid(handle) && !scratch_live_, "fake pressure expansion is stale");
    const Target& parent = targets_[handle.index];
    populate_options(parent.candidate_index);
    expansion_scratch_.clear();
    prepared_parent_    = handle.index;
    prepared_owner_end_ = static_cast<std::uint32_t>(std::min<std::size_t>(
        owners_.size(), static_cast<std::size_t>(parent.next_expansion_owner) + maximum_owners));
    for (std::size_t owner = parent.next_expansion_owner; owner < prepared_owner_end_; ++owner) {
        const std::uint16_t current = parent.choices[owner];
        const auto& alternatives    = options_[parent.candidate_index][owner];
        if (current == 0) {
            for (std::size_t choice = 1; choice <= alternatives.size(); ++choice) {
                Target child               = parent;
                child.choices[owner]       = static_cast<std::uint16_t>(choice);
                child.root_maximal         = false;
                child.next_expansion_owner = 0;
                if (std::none_of(expansion_scratch_.begin(), expansion_scratch_.end(),
                                 [&](const Target& prior) { return same_target(prior, child); })) {
                    expansion_scratch_.push_back(std::move(child));
                }
            }
        } else if (current <= alternatives.size() &&
                   !alternatives[current - 1U].evicts_continuation) {
            Target child               = parent;
            child.choices[owner]       = static_cast<std::uint16_t>(alternatives.size());
            child.root_maximal         = false;
            child.next_expansion_owner = 0;
            if (std::none_of(expansion_scratch_.begin(), expansion_scratch_.end(),
                             [&](const Target& prior) { return same_target(prior, child); })) {
                expansion_scratch_.push_back(std::move(child));
            }
        }
    }
    std::uint32_t new_count = 0;
    for (const Target& child : expansion_scratch_) {
        if (std::none_of(targets_.begin(), targets_.end(),
                         [&](const Target& prior) { return same_target(prior, child); })) {
            ++new_count;
        }
    }
    if (++scratch_generation_ == 0) { ++scratch_generation_; }
    scratch_live_ = true;
    return FakePreparedPressureExpansion(generation_, scratch_generation_, new_count);
}

FakePressureExpansionView
FakePressurePlanningSession::commit_expansion(FakePreparedPressureExpansion&& prepared) {
    require(scratch_live_ && prepared.generation == generation_ &&
                prepared.scratch_generation == scratch_generation_,
            "fake prepared expansion is stale");
    if (program_->pressure_optional_target_capacity) {
        const std::size_t maximum =
            candidates_.size() + 1U + *program_->pressure_optional_target_capacity;
        if (prepared.new_count > maximum - std::min(maximum, targets_.size())) {
            throw std::length_error("prepared pressure expansion exceeds the target arena");
        }
    }
    committed_children_.clear();
    std::uint32_t new_count = 0;
    for (Target& child : expansion_scratch_) {
        auto found          = std::find_if(targets_.begin(), targets_.end(),
                                           [&](const Target& prior) { return same_target(prior, child); });
        std::uint32_t index = 0;
        if (found == targets_.end()) {
            child.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
            targets_.push_back(std::move(child));
            index = static_cast<std::uint32_t>(targets_.size() - 1U);
            ++new_count;
        } else {
            index = static_cast<std::uint32_t>(found - targets_.begin());
        }
        committed_children_.push_back(
            FakePressureTargetHandle{.generation = generation_, .index = index});
    }
    program_->pressure_target_count_peak =
        std::max(program_->pressure_target_count_peak, targets_.size());
    require(new_count == prepared.new_count, "fake expansion count changed before commit");
    expansion_scratch_.clear();
    scratch_live_                                   = false;
    targets_[prepared_parent_].next_expansion_owner = prepared_owner_end_;
    return FakePressureExpansionView{
        .children            = committed_children_,
        .new_canonical_count = new_count,
        .complete            = prepared_owner_end_ == owners_.size(),
    };
}

void FakePressurePlanningSession::discard_expansion(
    FakePreparedPressureExpansion&& prepared) noexcept {
    if (scratch_live_ && prepared.generation == generation_ &&
        prepared.scratch_generation == scratch_generation_) {
        expansion_scratch_.clear();
        scratch_live_ = false;
    }
}

PrefillWork FakePressurePlanningSession::shared_capture_split_prefill_work(
    const FakeAssessedPressureTarget&, const FakePreparedPrompt&,
    std::span<const std::uint32_t> frontiers) const {
    return PrefillWork{.attention_pairs = static_cast<std::uint64_t>(frontiers.size()) * 100U};
}

std::optional<FakeResourcePlan>
FakePressurePlanningSession::seal(FakeAssessedPressureTarget&& assessed, const FakePreparedPrompt&,
                                  ninfer::runtime::FinalScheduleIntent intent) {
    const FakePressureTargetHandle handle = assessed.target_;
    require(valid(handle) && !scratch_live_, "fake pressure seal is stale");
    const Target& target   = targets_[handle.index];
    const auto& assessment = assessed.assessment_;
    if (assessment.physical_status != ninfer::runtime::MaterializationPhysicalStatus::Feasible) {
        return std::nullopt;
    }
    assessed.target_.generation = 0;
    FakeResourcePlan plan(*candidates_[target.candidate_index], revision_);
    std::vector<std::uint64_t> action_ids;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        if (choice == 0) { continue; }
        const FakeTargetDecision& decision = options_[target.candidate_index][index][choice - 1U];
        action_ids.push_back(decision.id);
        if (owners_[index].shared) {
            plan.shared_actions.push_back(decision);
            plan.shared_owner_ids.push_back(owners_[index].shared_handle->id);
            plan.shared_planning_ids.push_back(owners_[index].id);
        } else {
            plan.private_actions.push_back(decision);
            plan.private_owner_ids.push_back(owners_[index].private_handle->id);
            plan.private_planning_ids.push_back(owners_[index].id);
        }
    }
    program_->selected_shared_capture_frontiers.assign(intent.shared_capture_frontiers.begin(),
                                                       intent.shared_capture_frontiers.end());
    program_->seal_attempts.push_back(std::move(action_ids));
    return plan;
}

std::optional<FakeResourcePlan>
FakePressurePlanningSession::seal_capture(FakeAssessedPressureTarget&& assessed) {
    if (program_->fail_capture_seal) { return std::nullopt; }
    return seal(std::move(assessed), FakePreparedPrompt{}, {});
}

std::optional<FakeResourcePlan>
FakePressurePlanningSession::seal(FakeAssessedPressureTarget&& assessed) {
    return seal_capture(std::move(assessed));
}

bool FakePressurePlanningSession::try_claim_seal_window() noexcept {
    ++program_->seal_window_claims;
    return !program_->deny_seal_window;
}

void FakePressurePlanningSession::release_seal_window() noexcept {}

FakePressurePlanningSession
FakeProgram::begin_pressure_planning(std::span<const FakeAdmissionCandidate* const> candidates,
                                     std::span<const PlanningCandidateId> candidate_ids,
                                     std::span<const FakeContinuationHandle* const> private_owners,
                                     std::span<const PlanningOwnerId> private_owner_ids,
                                     std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                     std::span<const PlanningOwnerId> shared_owner_ids,
                                     std::span<const PlanningOwnerId> recency_order) {
    return FakePressurePlanningSession(*this, candidates, candidate_ids, private_owners,
                                       private_owner_ids, shared_owners, shared_owner_ids,
                                       recency_order);
}

struct FakeModelContract {
    using Program                    = FakeProgram;
    using PreparedPrompt             = FakePreparedPrompt;
    using RequestBasePlan            = FakeRequestBasePlan;
    using AdmissionCandidate         = FakeAdmissionCandidate;
    using ResourcePlan               = FakeResourcePlan;
    using PersistentBackfillProof    = FakePersistentBackfillProof;
    using SequenceHandle             = FakeSequenceHandle;
    using ContinuationHandle         = FakeContinuationHandle;
    using SharedPrefixHandle         = FakeSharedPrefixHandle;
    using CaptureOffer               = FakeCaptureOffer;
    using ContinuationSummary        = FakeContinuationSummary;
    using SharedPrefixSummary        = FakeSharedPrefixSummary;
    using CaptureAssessment          = FakeCaptureAssessment;
    using CapturePressurePlan        = FakeResourcePlan;
    using ActiveCaptureResult        = FakeActiveCaptureResult;
    using ContextTransactionProgress = FakeContextTransactionProgress;
    using MaterializationResult      = FakeMaterializationResult;
    using StartResult                = FakeStartResult;
    using FinishResult               = FakeFinishResult;
    using AbortResult                = FakeAbortResult;
    using PressureTargetHandle       = FakePressureTargetHandle;
    using AssessedPressureTarget     = FakeAssessedPressureTarget;
    using CommitResult               = FakeCommitResult;
    using DiscardResult              = FakeDiscardResult;
    using CacheSessionKey            = FakeCacheSessionKey;
};

using FakeManager = ninfer::runtime::ResourceManager<FakeModelContract>;

FakeManager make_manager(std::uint32_t lanes = 1, std::uint32_t private_capacity = 4,
                         std::uint32_t shared_capacity = 0, bool cache_enabled = true) {
    return FakeManager(lanes, private_capacity, shared_capacity, cache_enabled, 2,
                       test_cost_model());
}

struct ActiveRequest {
    LaneId lane;
    FakeSequenceHandle sequence;
};

ActiveRequest start_active(FakeManager& manager, FakeProgram& program, std::uint32_t content_key,
                           const FakeRequestBasePlan& base, std::uint64_t publication_order,
                           std::uint32_t prefix_key = 0) {
    const FakePreparedPrompt prompt{content_key, prefix_key};
    auto inspection = manager.inspect(program, prompt, base, publication_order);
    require(inspection.choice.has_value(), "request did not produce an admission choice");
    const LaneId lane   = inspection.choice->destination();
    const auto reserved = manager.reserve_materialization(
        program, std::move(*inspection.choice), FakePreparedPrompt{content_key, prefix_key}, {});
    require(reserved == FakeManager::MaterializationReserveResult::Reserved,
            "request materialization was not reserved");
    auto outcome = [&]() -> FakeManager::MaterializationOutcome {
        auto progress = manager.progress_context_transaction(program, {});
        if (!std::holds_alternative<ContextTransactionInProgress>(progress)) {
            return std::get<FakeManager::MaterializationOutcome>(std::move(progress));
        }
        auto completed = manager.progress_context_transaction(program, {});
        return std::get<FakeManager::MaterializationOutcome>(std::move(completed));
    }();
    require(outcome.status == ContextTransactionStatus::Published && outcome.activation,
            "request materialization did not publish");
    auto activation                   = std::move(*outcome.activation);
    const FakeSequenceHandle sequence = activation.sequence();
    manager.adopt(program, std::move(activation));
    require(manager.lane_state(lane) == ninfer::runtime::LogicalLaneState::Active,
            "published lane was not adopted as active");
    return ActiveRequest{.lane = lane, .sequence = sequence};
}

void test_private_portfolio_loss_keeps_checkpoint_identity_fixed() {
    using ninfer::runtime::ContextPortfolioCheckpointValue;
    using ninfer::runtime::ContextPortfolioOwnerPolicy;
    using ninfer::runtime::ContextPortfolioValue;

    const std::array owners{
        ContextPortfolioOwnerPolicy{.owner                    = PlanningOwnerId{.value = 0},
                                    .private_retention_weight = 4},
    };
    const std::array checkpoints{
        ContextPortfolioCheckpointValue{
            .owner                = PlanningOwnerId{.value = 0},
            .rebuild_ns           = 1000,
            .baseline_recovery_ns = 100,
            .target_recovery_ns   = 100,
        },
        ContextPortfolioCheckpointValue{
            .owner                = PlanningOwnerId{.value = 0},
            .rebuild_ns           = 800,
            .baseline_recovery_ns = 100,
            .target_recovery_ns   = 800,
        },
    };
    ContextPortfolioValue value;
    const auto result = value.fold(owners, checkpoints);
    require(result.baseline_public_value == 0 && result.target_public_value == 0 &&
                result.private_transition_loss == 2800 && !result.saturated,
            "a surviving endpoint masked loss of an earlier private checkpoint");
}

void test_unreachable_checkpoint_carries_no_value() {
    using ninfer::runtime::ContextPortfolioCheckpointValue;
    using ninfer::runtime::ContextPortfolioOwnerPolicy;
    using ninfer::runtime::ContextPortfolioValue;

    const std::array owners{
        ContextPortfolioOwnerPolicy{.owner                    = PlanningOwnerId{.value = 0},
                                    .private_retention_weight = 4},
    };
    for (const bool unreachable : {false, true}) {
        const std::array checkpoints{
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 0},
                .demand_mask          = 1,
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 100,
                .target_recovery_ns   = 800,
                .unreachable          = unreachable,
            },
        };
        ContextPortfolioValue value;
        const auto result = value.fold(owners, checkpoints);
        const bool valued = result.baseline_public_value != 0 || result.target_public_value != 0 ||
                            result.private_transition_loss != 0;
        require(valued != unreachable,
                unreachable ? "a checkpoint its own lineage diverged from still carried value"
                            : "a reachable checkpoint lost its value");
    }
}

void test_portfolio_demand_and_owner_aggregation() {
    using ninfer::runtime::ContextPortfolioCheckpointValue;
    using ninfer::runtime::ContextPortfolioOwnerPolicy;
    using ninfer::runtime::ContextPortfolioValue;

    {
        const std::array owners{ContextPortfolioOwnerPolicy{.owner = PlanningOwnerId{.value = 0}}};
        const std::array checkpoints{
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 0},
                .demand_mask          = 1,
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 200,
                .target_recovery_ns   = 500,
            },
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 0},
                .demand_mask          = 1,
                .rebuild_ns           = 800,
                .baseline_recovery_ns = 200,
                .target_recovery_ns   = 400,
            },
        };
        ContextPortfolioValue value;
        const auto result = value.fold(owners, checkpoints);
        require(result.baseline_public_value == 800 && result.target_public_value == 500 &&
                    result.private_transition_loss == 0,
                "nested checkpoints counted one empirical demand more than once");
    }

    {
        const std::array owners{
            ContextPortfolioOwnerPolicy{.owner                    = PlanningOwnerId{.value = 0},
                                        .private_retention_weight = 1},
            ContextPortfolioOwnerPolicy{.owner                    = PlanningOwnerId{.value = 1},
                                        .private_retention_weight = 4},
        };
        const std::array checkpoints{
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 0},
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 100,
                .target_recovery_ns   = 400,
            },
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 1},
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 100,
                .target_recovery_ns   = 200,
            },
        };
        ContextPortfolioValue value;
        const auto result = value.fold(owners, checkpoints);
        require(result.private_transition_loss == 700,
                "private checkpoint transition losses were not summed across owners");
    }
}

void test_shared_capture_subtracts_private_transition_loss() {
    using Planner = ninfer::runtime::SharedCapturePlanner<FakeModelContract>;

    FakeProgram program;
    program.required_pressure_actions             = 1;
    program.require_evictions                     = true;
    program.pressure_target_immediate_ns_override = 0;
    FakeCaptureAssessment capture{
        .shared_evidence     = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .publishes_shared    = true,
        .physically_feasible = false,
    };
    FakeContinuationHandle owner{7, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<PlanningOwnerId, 1> private_owner_ids{PlanningOwnerId{.value = 0}};
    const std::array<Planner::OwnerPolicy, 1> owner_policies{
        Planner::OwnerPolicy{.owner = PlanningOwnerId{.value = 0}, .private_retention_weight = 4},
    };
    const std::array<Planner::CheckpointPolicy, 1> checkpoint_policies{
        Planner::CheckpointPolicy{
            .owner                = PlanningOwnerId{.value = 0},
            .checkpoint           = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                                  .frontier = 16,
                                                  .ordinal  = 0},
            .rebuild_ns           = 1000,
            .baseline_recovery_ns = 0,
        },
    };

    Planner planner;
    const auto result = planner.plan(program, test_cost_model(),
                                     Planner::Input{
                                         .capture              = &capture,
                                         .private_owners       = private_owners,
                                         .private_owner_ids    = private_owner_ids,
                                         .shared_owners        = {},
                                         .shared_owner_ids     = {},
                                         .owner_policies       = owner_policies,
                                         .checkpoint_policies  = checkpoint_policies,
                                         .candidate_rebuild_ns = 1000,
                                     });
    require(result && result->baseline_value == 0 && result->target_value == 1000 &&
                result->immediate_ns == 0 && result->net_gain == 600,
            "shared capture gain did not subtract the private capability transition loss");
}

void test_shared_capture_budget_bounds_committed_canonical_targets() {
    using Planner = ninfer::runtime::SharedCapturePlanner<FakeModelContract>;

    constexpr std::size_t private_owner_count = 16;
    constexpr std::size_t shared_owner_count  = 4;
    constexpr std::uint32_t target_budget     = 64;

    FakeProgram program;
    program.required_pressure_actions         = private_owner_count + shared_owner_count + 1U;
    program.pressure_optional_target_capacity = target_budget;

    std::array<FakeContinuationHandle, private_owner_count> private_handles;
    std::array<const FakeContinuationHandle*, private_owner_count> private_owners;
    std::array<PlanningOwnerId, private_owner_count> private_owner_ids;
    for (std::size_t index = 0; index < private_owner_count; ++index) {
        private_handles[index] = FakeContinuationHandle(static_cast<std::uint32_t>(index + 1U), 0);
        private_owners[index]  = &private_handles[index];
        private_owner_ids[index] = PlanningOwnerId{.value = static_cast<std::uint32_t>(index)};
    }

    std::array<FakeSharedPrefixHandle, shared_owner_count> shared_handles;
    std::array<const FakeSharedPrefixHandle*, shared_owner_count> shared_owners;
    std::array<PlanningOwnerId, shared_owner_count> shared_owner_ids;
    for (std::size_t index = 0; index < shared_owner_count; ++index) {
        shared_handles[index].id = static_cast<std::uint32_t>(index + 1U);
        shared_owners[index]     = &shared_handles[index];
        shared_owner_ids[index] =
            PlanningOwnerId{.value = static_cast<std::uint32_t>(private_owner_count + index)};
    }

    const FakeCaptureAssessment capture{
        .shortlist_key       = FakeShortlistKey{.digest = 91, .frontier = 64},
        .publishes_shared    = true,
        .physically_feasible = false,
    };
    Planner planner;
    const auto result = planner.plan(program, test_cost_model(),
                                     Planner::Input{
                                         .capture           = &capture,
                                         .private_owners    = private_owners,
                                         .private_owner_ids = private_owner_ids,
                                         .shared_owners     = shared_owners,
                                         .shared_owner_ids  = shared_owner_ids,
                                         .target_budget     = target_budget,
                                     });
    require(!result, "bounded infeasible shared-capture search unexpectedly found a plan");
    require(program.pressure_target_count_peak <= target_budget,
            "shared-capture search committed more canonical targets than its budget");
}

void test_equal_lower_bound_does_not_short_circuit_tie_break() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakeModelContract>;

    FakeProgram program;
    program.required_pressure_actions         = 1;
    program.pressure_action_immediate_ns      = 0;
    program.pressure_action_degradation_units = 0;

    FakeAdmissionCandidate incumbent;
    set_fake_machine_costs(incumbent.identity.machine_work, 1'000'000'000, 1'000'000'000);
    incumbent.identity.machine_work.candidate_transfers[2].copy_operations = 1;
    incumbent.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    incumbent.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    incumbent.identity.assessment_digest = 11;

    FakeAdmissionCandidate tied_pressure;
    set_fake_machine_costs(tied_pressure.identity.machine_work, 1'000'000'000, 1'000'000'000);
    tied_pressure.identity.physical_status =
        ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    tied_pressure.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    tied_pressure.identity.expandable        = true;
    tied_pressure.identity.assessment_digest = 22;

    std::array<Planner::CandidateInput, 2> candidates{
        Planner::CandidateInput{.candidate               = &incumbent,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
        Planner::CandidateInput{.candidate               = &tied_pressure,
                                .id                      = PlanningCandidateId{.value = 1},
                                .stable_ordinal          = 1,
                                .current_session_binding = true},
    };
    FakeContinuationHandle owner{7, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<PlanningOwnerId, 1> private_owner_ids{PlanningOwnerId{.value = 0}};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 1> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 0}},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 1> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 0},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
    };

    Planner planner;
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = private_owners,
            .private_owner_ids = private_owner_ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = owner_policy,
            .checkpoint_policy = checkpoint_policy,
        };
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, Planner::Clock::now());

    require(result && result->candidate == PlanningCandidateId{.value = 1} &&
                program.pressure_planning_sessions == 1,
            "equal lower bound bypassed the pressure target that wins the stable tie-break");
}

void test_machine_cost_changes_selection_without_changing_physical_assessment() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakeModelContract>;

    FakeProgram program;
    FakeAdmissionCandidate prefill_candidate;
    prefill_candidate.identity.machine_work.remaining_prefill_work.tokens = 10;
    prefill_candidate.identity.physical_status =
        ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    prefill_candidate.identity.assessment_digest = 31;

    FakeAdmissionCandidate transfer_candidate;
    transfer_candidate.identity.machine_work.candidate_transfers[1] = {.payload_bytes   = 100,
                                                                       .copy_operations = 1};
    transfer_candidate.identity.machine_work.optimistic_candidate_transfers[1] = {
        .payload_bytes = 100, .copy_operations = 1};
    transfer_candidate.identity.physical_status =
        ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    transfer_candidate.identity.assessment_digest = 32;

    const std::array<Planner::CandidateInput, 2> candidates{
        Planner::CandidateInput{.candidate      = &prefill_candidate,
                                .id             = PlanningCandidateId{.value = 0},
                                .stable_ordinal = 0},
        Planner::CandidateInput{.candidate      = &transfer_candidate,
                                .id             = PlanningCandidateId{.value = 1},
                                .stable_ordinal = 1},
    };
    const auto pressure_inputs = [] { return Planner::PressureInputs{}; };
    const auto logical_goal    = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };

    auto prefill_expensive                 = test_cost_model();
    prefill_expensive.prefill.token_ns_q32 = 100U * ninfer::runtime::kContextCostQ32One;
    Planner first_planner;
    const auto first =
        first_planner.plan(program, FakePreparedPrompt{}, prefill_expensive, candidates, 0,
                           pressure_inputs, logical_goal, Planner::Clock::now());

    auto transfer_expensive                 = test_cost_model();
    transfer_expensive.prefill.token_ns_q32 = ninfer::runtime::kContextCostQ32One;
    for (auto& direction : transfer_expensive.transfer) {
        direction.ns_per_byte_q32 = 100U * ninfer::runtime::kContextCostQ32One;
    }
    Planner second_planner;
    const auto second =
        second_planner.plan(program, FakePreparedPrompt{}, transfer_expensive, candidates, 0,
                            pressure_inputs, logical_goal, Planner::Clock::now());

    require(first && first->candidate == PlanningCandidateId{.value = 1} && second &&
                second->candidate == PlanningCandidateId{.value = 0} &&
                prefill_candidate.identity.physical_status ==
                    ninfer::runtime::MaterializationPhysicalStatus::Feasible &&
                transfer_candidate.identity.physical_status ==
                    ninfer::runtime::MaterializationPhysicalStatus::Feasible,
            "machine cost policy changed physical assessment or failed to change selection");
}

void test_candidate_search_prefers_deep_reuse_without_eviction() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakeModelContract>;

    FakeProgram program;
    program.required_pressure_actions         = 2;
    program.eviction_pressure_action_units    = 2;
    program.pressure_action_immediate_ns      = 1'000'000;
    program.pressure_action_degradation_units = 1;
    program.pressure_assessment_delay_us      = 1'000;
    program.pressure_checkpoint_recovery_ns   = 8'000'000'000ULL;

    FakeAdmissionCandidate root;
    set_fake_machine_costs(root.identity.machine_work, 8'000'000'000ULL, 8'000'000'000ULL);
    root.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    root.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    root.identity.expandable        = true;
    root.identity.assessment_digest = 101;

    FakeAdmissionCandidate reuse;
    reuse.value.reusable_prompt_tokens = 55'048;
    reuse.private_source_id            = 1;
    set_fake_machine_costs(reuse.identity.machine_work, 100'000'000, 100'000'000);
    reuse.identity.machine_work.reused_prompt_tokens = 55'048;
    reuse.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    reuse.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    reuse.identity.expandable        = true;
    reuse.identity.assessment_digest = 202;

    std::array<Planner::CandidateInput, 2> candidates{
        Planner::CandidateInput{.candidate               = &root,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
        Planner::CandidateInput{.candidate               = &reuse,
                                .id                      = PlanningCandidateId{.value = 1},
                                .stable_ordinal          = 1,
                                .current_session_binding = true},
    };
    std::array<FakeContinuationHandle, 3> owner_handles{
        FakeContinuationHandle{1, 0},
        FakeContinuationHandle{2, 0},
        FakeContinuationHandle{3, 0},
    };
    const std::array<const FakeContinuationHandle*, 3> private_owners{
        &owner_handles[0], &owner_handles[1], &owner_handles[2]};
    const std::array<PlanningOwnerId, 3> private_owner_ids{
        PlanningOwnerId{.value = 0}, PlanningOwnerId{.value = 1}, PlanningOwnerId{.value = 2}};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 3> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.owner           = PlanningOwnerId{.value = 0},
                                                    .retention_class = RetentionClass::LiveSession,
                                                    .private_retention_weight = 16},
        ninfer::runtime::MaterializationOwnerPolicy{.owner           = PlanningOwnerId{.value = 1},
                                                    .retention_class = RetentionClass::LiveSession,
                                                    .private_retention_weight = 16},
        ninfer::runtime::MaterializationOwnerPolicy{.owner           = PlanningOwnerId{.value = 2},
                                                    .retention_class = RetentionClass::LiveSession,
                                                    .private_retention_weight = 16},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 3> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 0},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 8'000'000'000ULL},
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 1},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 8'000'000'000ULL},
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 2},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 8'000'000'000ULL},
    };

    Planner planner;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = private_owners,
            .private_owner_ids = private_owner_ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = owner_policy,
            .checkpoint_policy = checkpoint_policy,
        };
    };
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, Planner::Clock::now());

    require(result && result->plan && result->candidate == PlanningCandidateId{.value = 1},
            "shallow Root pressure path starved the cheaper reuse candidate");
    require(result->plan->private_actions.size() == 2 &&
                std::none_of(
                    result->plan->private_actions.begin(), result->plan->private_actions.end(),
                    [](const FakeTargetDecision& action) { return action.evicts_continuation; }),
            "one-step eviction outranked the multi-step preserving reuse closure");
}

void test_feasible_identity_expands_when_pressure_can_remove_copy() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakeModelContract>;

    FakeProgram program;
    program.pressure_action_immediate_ns          = 0;
    program.pressure_action_degradation_units     = 1;
    program.pressure_target_immediate_ns_override = 100'000'000;

    FakeAdmissionCandidate candidate;
    set_fake_machine_costs(candidate.identity.machine_work, 100'000'000, 1'000'000'000);
    candidate.identity.machine_work.candidate_transfers[2].copy_operations = 1;
    candidate.identity.pressure_may_change_machine_work                    = true;
    candidate.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    candidate.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    candidate.identity.assessment_digest = 17;

    const std::array<Planner::CandidateInput, 1> candidates{
        Planner::CandidateInput{.candidate               = &candidate,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
    };
    FakeContinuationHandle owner{9, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<PlanningOwnerId, 1> private_owner_ids{PlanningOwnerId{.value = 0}};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 1> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 0}},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 1> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 0},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
    };

    Planner planner;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = private_owners,
            .private_owner_ids = private_owner_ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = owner_policy,
            .checkpoint_policy = checkpoint_policy,
        };
    };
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, Planner::Clock::now());

    require(result && result->diagnostics.predicted_now_ns == 100'000'000 &&
                result->diagnostics.selected_degradation_units == 1 &&
                program.pressure_planning_sessions == 1 && !program.seal_attempts.empty() &&
                program.seal_attempts.back() == std::vector<std::uint64_t>{1009},
            "feasible identity suppressed a cheaper complete pressure target");
}

void test_dominating_identity_does_not_build_pressure_graph() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakeModelContract>;

    FakeProgram program;
    FakeAdmissionCandidate candidate;
    set_fake_machine_costs(candidate.identity.machine_work, 100'000'000, 100'000'000);
    candidate.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    candidate.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    candidate.identity.assessment_digest = 23;
    const std::array<Planner::CandidateInput, 1> candidates{
        Planner::CandidateInput{.candidate               = &candidate,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
    };

    bool pressure_inputs_built = false;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        pressure_inputs_built = true;
        return {};
    };
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };

    Planner planner;
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, Planner::Clock::now());
    require(result &&
                result->diagnostics.stop_reason == ninfer::MaterializationStopReason::NoPressure &&
                !pressure_inputs_built && program.pressure_planning_sessions == 0,
            "dominating identity eagerly constructed the pressure graph");
}

FakeFinishResult finish_active(FakeManager& manager, FakeProgram& program, ActiveRequest request,
                               std::uint32_t frontier = 16) {
    program.finish_frontier = frontier;
    manager.mark_terminal_pending(request.lane);
    return manager.finish(program, request.lane, request.sequence);
}

void test_root_lifecycle_and_prefix_reuse() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 7, make_base(7), 1);
    require(program.pressure_planning_sessions == 0,
            "no-pressure root admission created a pressure planning session");
    const FakeFinishResult finish = finish_active(manager, program, first);
    require(finish.status == ConsumeStatus::Consumed &&
                finish.disposition == FinishDisposition::Catalogued,
            "terminal continuation was not catalogued");
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Free,
            "terminal lane did not return to Free");

    auto reuse = manager.inspect(program, FakePreparedPrompt{7}, make_base(7), 2);
    require(reuse.readiness == Readiness::Ready && reuse.choice,
            "catalogued endpoint was not reusable");
    require(reuse.choice->summary().reusable_prompt_tokens == 16,
            "endpoint reuse frontier was not selected");
    program.abort_start = true;
    const auto status   = manager.reserve_materialization(program, std::move(*reuse.choice),
                                                          FakePreparedPrompt{7}, {});
    require(status == FakeManager::MaterializationReserveResult::Stale &&
                program.started_source_id == first.sequence.id,
            "selected endpoint did not reach the sealed Program plan");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "failed start did not roll back its logical source claim");
}

void test_stale_revision_is_retryable() {
    FakeManager manager = make_manager();
    FakeProgram program;
    auto inspection = manager.inspect(program, FakePreparedPrompt{1}, make_base(1), 1);
    require(inspection.choice.has_value(), "root choice was not produced");
    const std::uint64_t start_calls = program.start_calls;
    program.invalidate_resources();
    const auto status = manager.reserve_materialization(program, std::move(*inspection.choice),
                                                        FakePreparedPrompt{1}, {});
    require(status == FakeManager::MaterializationReserveResult::Stale,
            "revision mismatch was not reported as retryable stale work");
    require(program.start_calls == start_calls,
            "known-stale plan was incorrectly passed into Program start");
    require(manager.lane_state(LaneId{0}) == ninfer::runtime::LogicalLaneState::Free,
            "known-stale plan changed logical lane state");
}

void test_materialization_abort_preserves_source() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 5, make_base(5), 1);
    (void)finish_active(manager, program, seed);

    auto inspection = manager.inspect(program, FakePreparedPrompt{5}, make_base(5), 2);
    require(inspection.choice && inspection.choice->summary().reusable_prompt_tokens == 16,
            "abort test did not select its private source");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{5}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "abort test could not reserve materialization");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted && !outcome.activation,
            "cancelled materialization did not abort before publication");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "aborted Move did not restore its source visibility");

    program.abort_progress = false;
    program.abort_start    = true;
    auto retry             = manager.inspect(program, FakePreparedPrompt{5}, make_base(5), 3);
    require(retry.choice && retry.choice->summary().reusable_prompt_tokens == 16,
            "restored source was not reusable after abort");
    (void)manager.reserve_materialization(program, std::move(*retry.choice), FakePreparedPrompt{5},
                                          {});
    require(program.started_source_id == seed.sequence.id,
            "abort restored the wrong source capability");
}

void test_committed_victim_survives_transaction_abort() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 10, make_base(10), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 20, make_base(20), 2);
    (void)finish_active(manager, program, second);

    auto inspection = manager.inspect(program, FakePreparedPrompt{30}, make_base(30), 3);
    require(inspection.choice.has_value(), "full catalog did not produce an eviction closure");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{30}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "evicting materialization was not reserved");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted,
            "pressure transaction did not take the abort path");
    const std::uint32_t catalogued =
        (manager.catalog_state(0) == FakeManager::CatalogState::Catalogued ? 1U : 0U) +
        (manager.catalog_state(1) == FakeManager::CatalogState::Catalogued ? 1U : 0U);
    require(catalogued == 1,
            "committed victim eviction was incorrectly rolled back with request-local abort");
}

void test_uncommitted_pressure_acknowledgement_is_not_degradation() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 40, make_base(40), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{41}, make_base(41), 2);
    require(inspection.choice.has_value(),
            "pressure-abort test did not select a preserving pressure target");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{41}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "pressure-abort test could not reserve materialization");
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + seed.sequence.id,
            "pressure-abort test selected an eviction instead of preserving pressure");

    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(outcome.status == ContextTransactionStatus::Aborted &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "uncommitted pressure acknowledgement changed owner availability");
    require(stats.pressure_private_owners_degraded == 0 &&
                stats.pressure_private_owners_evicted == 0 &&
                stats.pressure_checkpoints_dropped == 0,
            "uncommitted pressure acknowledgement was counted as a degradation");
    require(stats.pressure_searches == 1,
            "accepted pressure plan was hidden when its request later aborted");
}

void test_aborted_source_selection_does_not_create_hit_history() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 61, make_base(61), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 62, make_base(62), 2);
    (void)finish_active(manager, program, second);

    auto reuse = manager.inspect(program, FakePreparedPrompt{61}, make_base(61), 3);
    require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
            "cancelled-hit test did not select its exact source");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*reuse.choice),
                                            FakePreparedPrompt{61}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "cancelled-hit test could not reserve exact reuse");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted,
            "cancelled-hit test unexpectedly published its request");

    program.abort_progress            = false;
    program.required_pressure_actions = 1;
    program.require_evictions         = true;
    auto pressure = manager.inspect(program, FakePreparedPrompt{63}, make_base(63), 4);
    require(pressure.choice.has_value(), "cancelled-hit test could not plan pressure");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*pressure.choice),
                                          FakePreparedPrompt{63}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 2000U + first.sequence.id,
            "aborted source selection incorrectly biased later retention policy");
}

void test_auxiliary_host_pressure_waits_without_rejecting_request() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    program.auxiliary_tier_available = false;
    const auto blocked = manager.inspect(program, FakePreparedPrompt{17}, make_base(17), 1);
    require(blocked.readiness == Readiness::TemporarilyBlocked && !blocked.choice,
            "temporary Host pressure rejected or admitted the queued request");
    program.auxiliary_tier_available = true;
    const auto available = manager.inspect(program, FakePreparedPrompt{17}, make_base(17), 1);
    require(available.choice.has_value(), "released Host reservation did not unblock admission");
}

void test_retained_source_is_protected_until_terminal() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const FakeCacheSessionKey first_session{1};
    const FakeCacheSessionKey second_session{2};
    const ActiveRequest seed = start_active(
        manager, program, 9, make_base(9, first_session, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, seed);

    const ActiveRequest fork = start_active(
        manager, program, 9, make_base(9, second_session, RetentionClass::LiveSession), 2);
    require(program.started_source_mode == PrivateSourceMode::Retain,
            "different-session source was destructively moved");

    program.required_pressure_actions = 1;
    auto blocked = manager.inspect(program, FakePreparedPrompt{77}, make_base(77), 3);
    require(blocked.readiness == Readiness::TemporarilyBlocked && !blocked.choice,
            "active retained source was exposed as a pressure victim");

    (void)manager.abort(program, fork.lane, fork.sequence);
    program.abort_start = true;
    auto available      = manager.inspect(program, FakePreparedPrompt{77}, make_base(77), 4);
    require(available.choice.has_value(),
            "terminal release did not return retained source to pressure policy");
    (void)manager.reserve_materialization(program, std::move(*available.choice),
                                          FakePreparedPrompt{77}, {});
    require(!program.started_action_ids.empty(),
            "released source did not participate in the sealed pressure plan");
}

void test_session_publication_order_controls_tied_source() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const FakeCacheSessionKey session{42};
    const FakeRequestBasePlan base = make_base(42, session, RetentionClass::LiveSession, true);

    const ActiveRequest older = start_active(manager, program, 42, base, 10);
    const ActiveRequest newer = start_active(manager, program, 42, base, 20);
    (void)finish_active(manager, program, newer, 16);
    (void)finish_active(manager, program, older, 16);

    auto next = manager.inspect(program, FakePreparedPrompt{42}, base, 30);
    require(next.choice && next.choice->summary().reusable_prompt_tokens == 16,
            "same-session endpoint candidates were not reusable");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*next.choice), FakePreparedPrompt{42},
                                          {});
    require(program.started_source_id == newer.sequence.id,
            "older out-of-order finish displaced the current session binding on a tied cost");

    program.abort_start             = false;
    const ActiveRequest replacement = start_active(
        manager, program, 99, make_base(99, session, RetentionClass::LiveSession, true), 40);
    (void)finish_active(manager, program, replacement, 16);
    require(program.released_continuations.empty(),
            "session publication synchronously released an old physical continuation");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(2) == FakeManager::CatalogState::Catalogued,
            "session replacement did not retain its old binding as anonymous cache");
}

void test_canonical_pressure_starts_with_disposable_owner() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest disposable = start_active(
        manager, program, 1, make_base(1, std::nullopt, RetentionClass::Disposable), 1);
    (void)finish_active(manager, program, disposable);
    const ActiveRequest live = start_active(
        manager, program, 2, make_base(2, FakeCacheSessionKey{2}, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, live);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{3}, make_base(3), 3);
    require(inspection.choice.has_value(), "canonical pressure did not find a feasible prefix");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{3}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + disposable.sequence.id,
            "canonical pressure did not degrade Disposable before LiveSession");
}

void test_pressure_tries_every_preserving_alternative_before_eviction() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 15, make_base(15), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions     = 1;
    program.private_pressure_alternatives = 2;
    program.required_action_id            = 11000U + seed.sequence.id;
    auto inspection = manager.inspect(program, FakePreparedPrompt{25}, make_base(25), 2);
    require(inspection.choice.has_value(), "second preserving pressure alternative was skipped");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{25}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == *program.required_action_id,
            "pressure escalated before trying the feasible preserving alternative");
}

void test_cumulative_owner_target_closes_pressure_without_eviction() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    program.finish_with_rewrite = true;
    const ActiveRequest seed    = start_active(manager, program, 31, make_base(31), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions         = 1;
    program.include_cumulative_private_target = true;
    program.required_action_id                = 5000U + seed.sequence.id;
    auto inspection = manager.inspect(program, FakePreparedPrompt{32}, make_base(32), 2);
    require(inspection.choice.has_value(),
            "cumulative checkpoint-drop and spill owner target was unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{32}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == *program.required_action_id,
            "planner replaced a feasible cumulative owner target with eviction");
}

void test_two_owners_jointly_close_pressure() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 41, make_base(41), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 42, make_base(42), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions = 2;
    auto inspection = manager.inspect(program, FakePreparedPrompt{43}, make_base(43), 3);
    require(inspection.choice.has_value(), "joint two-owner pressure target was unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{43}, {});
    std::sort(program.started_action_ids.begin(), program.started_action_ids.end());
    const std::vector<std::uint64_t> expected{
        1000U + first.sequence.id,
        1000U + second.sequence.id,
    };
    require(program.started_action_ids == expected,
            "planner did not combine preserving targets from two owners");
}

void test_rank_private_owners_by_recency_ranks_every_private_owner() {
    using namespace ninfer::runtime;
    const std::vector<MaterializationOwnerPolicy> policies{
        MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 0}},
        MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 1}, .last_hit_epoch = 5},
        MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 2}, .last_hit_epoch = 3},
    };
    const std::vector<PlanningOwnerId> private_ids{
        PlanningOwnerId{.value = 0}, PlanningOwnerId{.value = 1}, PlanningOwnerId{.value = 2}};
    std::span<const MaterializationOwnerPolicy> policy_span(policies);
    std::span<const PlanningOwnerId> id_span(private_ids);

    const auto ranked = rank_owners_by_recency(policy_span, id_span);
    require(ranked.size() == 3 && ranked[0] == PlanningOwnerId{.value = 1} &&
                ranked[1] == PlanningOwnerId{.value = 2} &&
                ranked[2] == PlanningOwnerId{.value = 0},
            "every private owner must be ranked by last hit epoch, most recent first");
    require(
        rank_owners_by_recency(policy_span, std::span<const PlanningOwnerId>(private_ids).first(0))
            .empty(),
        "no private owner must rank nothing");
}

void test_publication_counts_as_recency_for_the_ladder() {
    // A finished request's continuation has no hit history yet: a publication starts its
    // checkpoint observations at zero, and the normal ConsumeToActive continuation clears the
    // consumed source's history. The ladder must still rank it as the most recently used owner,
    // otherwise the conversation most likely to be continued next is sacrificed first.
    FakeManager manager = make_manager(1, 4);
    FakeProgram program;
    manager.enable_recency_eviction();
    const ActiveRequest a = start_active(manager, program, 1, make_base(1), 1);
    (void)finish_active(manager, program, a);
    const ActiveRequest b = start_active(manager, program, 2, make_base(2), 2);
    (void)finish_active(manager, program, b);
    // Continue `a` (endpoint hit, consumed into the active lane) and publish its next turn.
    const ActiveRequest a_next = start_active(manager, program, 1, make_base(1), 3);
    (void)finish_active(manager, program, a_next, 24);
    const ActiveRequest c = start_active(manager, program, 3, make_base(3), 4);
    (void)finish_active(manager, program, c);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{4}, make_base(4), 5);
    require(inspection.choice.has_value(), "pressure admission found no choice");
    require(program.pressure_planning_sessions != 0, "admission did not plan under pressure");
    const std::vector<std::uint32_t> expected{c.sequence.id, a_next.sequence.id, b.sequence.id};
    require(program.observed_recency_ids == expected,
            "recency order did not rank owners by their latest publication or hit");

    // Without --recency-eviction the planners receive no recency order at all.
    FakeManager economic = make_manager(1, 4);
    FakeProgram economic_program;
    const ActiveRequest d = start_active(economic, economic_program, 1, make_base(1), 1);
    (void)finish_active(economic, economic_program, d);
    const ActiveRequest e = start_active(economic, economic_program, 2, make_base(2), 2);
    (void)finish_active(economic, economic_program, e);
    economic_program.required_pressure_actions = 1;
    auto economic_inspection =
        economic.inspect(economic_program, FakePreparedPrompt{4}, make_base(4), 3);
    require(economic_inspection.choice.has_value() &&
                economic_program.pressure_planning_sessions != 0 &&
                economic_program.observed_recency_ids.empty(),
            "recency order reached the planners without recency eviction");
}

void test_incremental_eviction_is_licensed_by_the_lru_tail() {
    // The admission planner licenses incremental full eviction only over the LRU tail its
    // escape-hatch ladder had to sacrifice. Inside the session, a construction step must not
    // offer a full eviction for a more recent owner, even when that owner has no demote outcome
    // to fall back on.
    FakeProgram program;
    program.private_pressure_alternatives = 1; // one non-evict (demote) outcome per private owner
    std::array<FakeContinuationHandle, 2> handles{FakeContinuationHandle(1, 0),
                                                  FakeContinuationHandle(2, 0)};
    std::array<const FakeContinuationHandle*, 2> private_owners{&handles[0], &handles[1]};
    std::array<PlanningOwnerId, 2> private_owner_ids{PlanningOwnerId{.value = 0},
                                                     PlanningOwnerId{.value = 1}};
    // Owner 0 (handle 1) is the most recently hit prefix, owner 1 is the oldest.
    std::array<PlanningOwnerId, 2> recency_order{PlanningOwnerId{.value = 0},
                                                 PlanningOwnerId{.value = 1}};
    FakeAdmissionCandidate candidate;
    const std::array<const FakeAdmissionCandidate*, 1> candidates{&candidate};
    const std::array<PlanningCandidateId, 1> candidate_ids{PlanningCandidateId{.value = 0}};
    auto session = program.begin_pressure_planning(candidates, candidate_ids, private_owners,
                                                   private_owner_ids, {}, {}, recency_order);

    session.set_eviction_licence(1); // only the oldest owner may be fully evicted
    auto cursor = session.begin_construction(session.identity_target(candidate_ids[0]));
    // With one alternative plus the eviction, the eviction is the last choice the session offers.
    const std::uint16_t eviction_choice = 2;
    std::array<bool, 2> eviction_offered{false, false};
    for (;;) {
        const auto step = session.next_construction_option(cursor);
        if (step.exhausted) { break; }
        if (!step.guidance) { continue; } // this call only appended the next owner's options
        const auto& child = cursor.options[step.option.index];
        for (std::size_t owner = 0; owner < cursor.target.choices.size(); ++owner) {
            if (child.choices[owner] != cursor.target.choices[owner] &&
                child.choices[owner] == eviction_choice) {
                eviction_offered[owner] = true;
            }
        }
    }
    require(eviction_offered[1] && !eviction_offered[0],
            "incremental eviction was reachable outside the licensed LRU tail");

    session.set_eviction_licence(0); // nothing must be evicted
    auto unrestricted = session.begin_construction(session.identity_target(candidate_ids[0]));
    std::array<bool, 2> eviction_offered_without_licence{false, false};
    for (;;) {
        const auto step = session.next_construction_option(unrestricted);
        if (step.exhausted) { break; }
        if (!step.guidance) { continue; }
        const auto& child = unrestricted.options[step.option.index];
        for (std::size_t owner = 0; owner < unrestricted.target.choices.size(); ++owner) {
            if (child.choices[owner] != unrestricted.target.choices[owner] &&
                child.choices[owner] == eviction_choice) {
                eviction_offered_without_licence[owner] = true;
            }
        }
    }
    require(!eviction_offered_without_licence[0] && !eviction_offered_without_licence[1],
            "an unlicensed admission still reached incremental eviction");
}

void test_escape_hatch_sacrifices_oldest_first_and_demotes_the_rest() {
    // Three catalogued private owners published in the order a, b, c, so their recency ranks are
    // c=0 (most recent), b=1, a=2. The pressure search fails, so the escape-hatch recency ladder
    // runs. Host fits only after the oldest owner is given up, so rung 1 demotes c and b (kept on
    // host) and fully evicts a: the ladder sacrifices a suffix of the recency order and keeps
    // every more recent prefix.
    FakeManager manager = make_manager(1, 3, 0, true);
    FakeProgram program;
    manager.enable_recency_eviction();
    const ActiveRequest a = start_active(manager, program, 151, make_base(151), 1);
    (void)finish_active(manager, program, a);
    const ActiveRequest b = start_active(manager, program, 152, make_base(152), 2);
    (void)finish_active(manager, program, b);
    const ActiveRequest c = start_active(manager, program, 153, make_base(153), 3);
    (void)finish_active(manager, program, c);

    program.ladder_feasibility_mode        = true;
    program.min_feasible_sacrifice         = 1; // host fits only after giving up the oldest owner
    program.required_pressure_actions      = 4; // above the search's max pressure -> search fails
    program.eviction_pressure_action_units = 1;
    program.private_pressure_alternatives  = 1; // a non-evict (demote) alternative must exist
    auto inspection = manager.inspect(program, FakePreparedPrompt{154}, make_base(154), 4);
    require(inspection.choice.has_value(), "escape-hatch recency ladder found no rung");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{154}, {});
    const auto evicted = [&program](const ActiveRequest& request) {
        return std::find(program.started_action_ids.begin(), program.started_action_ids.end(),
                         2000U + request.sequence.id) != program.started_action_ids.end();
    };
    require(evicted(a) && !evicted(b) && !evicted(c),
            "escape-hatch rung evicted a more recent owner instead of the oldest");
}

void test_escape_hatch_clears_all_when_nothing_fits() {
    // Host cannot absorb any demotion: every ladder rung is infeasible, so the ladder falls
    // through to the terminal clear-all (evict everything) — the liveness backstop that never
    // blocks the active context from getting its device KV.
    FakeManager manager = make_manager(1, 3, 0, true);
    FakeProgram program;
    manager.enable_recency_eviction();
    const ActiveRequest a = start_active(manager, program, 151, make_base(151), 1);
    (void)finish_active(manager, program, a);
    const ActiveRequest b = start_active(manager, program, 152, make_base(152), 2);
    (void)finish_active(manager, program, b);
    const ActiveRequest c = start_active(manager, program, 153, make_base(153), 3);
    (void)finish_active(manager, program, c);

    program.ladder_feasibility_mode        = true;
    program.min_feasible_sacrifice         = 3; // >= private owner count -> every rung infeasible
    program.required_pressure_actions      = 4;
    program.eviction_pressure_action_units = 1;
    program.private_pressure_alternatives  = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{154}, make_base(154), 4);
    require(inspection.choice.has_value(), "escape-hatch terminal clear-all was not reachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{154}, {});
    const auto evicted = [&program](const ActiveRequest& request) {
        return std::find(program.started_action_ids.begin(), program.started_action_ids.end(),
                         2000U + request.sequence.id) != program.started_action_ids.end();
    };
    require(evicted(a) && evicted(b) && evicted(c), "terminal clear-all did not evict every owner");
}

void test_preserved_prefix_ladder_targets_fit_the_committed_target_budget() {
    // Regression (2026-09-22 production 503 wall): the escape-hatch recency ladder interns one
    // arena target per rung that the planner's optional-target budget does not count. When the
    // pressure search then exhausts its target budget, the final expansion commit crossed the
    // session arena limit and threw std::length_error; the worker treated the consecutive
    // occurrences as fatal (fail-all) and rejected every subsequent request until restart. The
    // planner must count ladder rungs in its budget and bound expansion commits by the
    // session's true remaining capacity, so a budget-exhausting search degrades to an early
    // stop instead of throwing.
    using Planner = ninfer::runtime::MaterializationPlanner<FakeModelContract>;

    constexpr std::size_t arena_capacity = 24; // small arena: the overflow is reachable quickly

    FakeProgram program;
    program.ladder_feasibility_mode           = true;
    program.min_feasible_sacrifice            = 1; // rung 1 (sacrifice the oldest preserved) fits
    program.required_pressure_actions         = 5; // above the search's max pressure -> it fails
    program.eviction_pressure_action_units    = 1;
    program.private_pressure_alternatives     = 4; // wide per-owner target space
    program.pressure_optional_target_capacity = arena_capacity;

    FakeAdmissionCandidate root;
    root.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    root.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    root.identity.expandable        = true;
    root.identity.assessment_digest = 301;

    const std::array<Planner::CandidateInput, 1> candidates{Planner::CandidateInput{
        .candidate = &root, .id = PlanningCandidateId{.value = 0}, .stable_ordinal = 0}};

    // Three ranked private owners (most-recent first: handle ids 1, 2, 3) plus one owner outside
    // the recency order (handle id 4), which every rung evicts.
    std::array<FakeContinuationHandle, 4> handles{};
    std::array<const FakeContinuationHandle*, 4> private_owners{};
    std::array<PlanningOwnerId, 4> private_owner_ids{};
    for (std::size_t index = 0; index < handles.size(); ++index) {
        handles[index]        = FakeContinuationHandle(static_cast<std::uint32_t>(index + 1U), 0);
        private_owners[index] = &handles[index];
        private_owner_ids[index] = PlanningOwnerId{.value = static_cast<std::uint32_t>(index)};
    }
    const std::array<PlanningOwnerId, 3> recency_owner_ids{
        PlanningOwnerId{.value = 0}, PlanningOwnerId{.value = 1}, PlanningOwnerId{.value = 2}};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 4> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 0}},
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 1}},
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 2}},
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 3}},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 4> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 0},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 1},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 2},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 3},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
    };

    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = private_owners,
            .private_owner_ids = private_owner_ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = owner_policy,
            .checkpoint_policy = checkpoint_policy,
            .recency_owner_ids = recency_owner_ids,
        };
    };
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };

    Planner planner;
    const auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates,
                                     0, pressure_inputs, logical_goal, Planner::Clock::now());

    require(result && result->candidate == PlanningCandidateId{.value = 0},
            "ladder rung did not survive the budget-exhausting search");
    require(result->diagnostics.budget_exhausted &&
                result->diagnostics.stop_reason ==
                    ninfer::MaterializationStopReason::ExpansionCapacity,
            "budget-exhausting search must stop on the arena capacity instead of running past it");
    require(program.pressure_target_count_peak <= 1U + 1U + arena_capacity,
            "search committed more canonical targets than the arena can hold");
    // The escape-hatch rung evicts the non-preserved owner (handle id 4) to free device KV and
    // keeps the most-recent preserved prefix (handle id 1) on host; the pressure search must
    // preserve that disposition when it stops on capacity.
    const auto evicted = [&result](std::uint32_t handle_id) {
        return std::any_of(result->plan->private_actions.begin(),
                           result->plan->private_actions.end(),
                           [&](const auto& action) { return action.id == 2000U + handle_id; });
    };
    require(evicted(4) && !evicted(1),
            "capacity stop changed the ladder's preserved/non-preserved disposition");
}

void test_materialization_result_is_validated_before_any_adoption() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 141, make_base(141), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 142, make_base(142), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions   = 2;
    program.malform_last_private_victim = true;
    auto inspection = manager.inspect(program, FakePreparedPrompt{143}, make_base(143), 3);
    require(inspection.choice.has_value(), "malformed-result fixture found no pressure plan");
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{143}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "malformed-result fixture could not reserve materialization");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed-result fixture did not claim both owners");

    bool rejected = false;
    try {
        (void)manager.progress_context_transaction(program, {});
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "malformed materialization result was accepted");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed materialization result partially adopted an earlier victim");
}

void test_materialization_result_binds_exact_checkpoint_identity() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    program.finish_with_rewrite = true;
    const ActiveRequest seed    = start_active(manager, program, 145, make_base(145), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions           = 1;
    program.include_cumulative_private_target   = true;
    program.required_action_id                  = 5000U + seed.sequence.id;
    program.malform_private_checkpoint_identity = true;
    auto inspection = manager.inspect(program, FakePreparedPrompt{146}, make_base(146), 2);
    require(inspection.choice.has_value(), "checkpoint-identity fixture found no pressure plan");
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{146}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "checkpoint-identity fixture could not reserve materialization");

    bool rejected = false;
    try {
        (void)manager.progress_context_transaction(program, {});
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "same-count checkpoint substitution was accepted");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed,
            "checkpoint substitution partially mutated its logical owner");
}

void test_materialization_result_is_adopted_by_owner_identity() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 151, make_base(151), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 152, make_base(152), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions = 2;
    program.reverse_pressure_results  = true;
    const ActiveRequest published     = start_active(manager, program, 153, make_base(153), 3);
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued,
            "reordered materialization results changed victim availability");
    (void)finish_active(manager, program, published);

    program.required_pressure_actions = 0;
    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{151}, make_base(151), 4);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered materialization result attached the first summary to another owner");
    }
    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{152}, make_base(152), 5);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered materialization result attached the second summary to another owner");
    }
}

void test_guided_pressure_reaches_deep_retention_before_maximal_fallback() {
    constexpr std::size_t owner_count = 7;
    FakeManager manager               = make_manager(1, owner_count + 1U);
    FakeProgram program;
    std::array<std::uint32_t, owner_count> owner_ids{};
    for (std::size_t index = 0; index < owner_count; ++index) {
        const std::uint32_t content = static_cast<std::uint32_t>(70U + index);
        const ActiveRequest active =
            start_active(manager, program, content, make_base(content), index + 1U);
        owner_ids[index] = active.sequence.id;
        (void)finish_active(manager, program, active);
    }

    program.required_pressure_actions     = 3;
    program.private_pressure_alternatives = 4;
    program.pressure_assessment_delay_us  = 2'000;
    auto inspection = manager.inspect(program, FakePreparedPrompt{90}, make_base(90), 20);
    require(inspection.choice.has_value(), "guided pressure search found no admission plan");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{90}, {});
    require(program.started_action_ids.size() == program.required_pressure_actions,
            "guided pressure search selected maximal release instead of a retention closure");
    for (const std::uint32_t owner_id : owner_ids) {
        require(std::find(program.started_action_ids.begin(), program.started_action_ids.end(),
                          2000U + owner_id) == program.started_action_ids.end(),
                "guided pressure search evicted a parked owner");
    }
    // The counter is time-sensitive -- the fixture injects a per-assessment delay and the search
    // budget is time-granted -- so it depends on the platform's sleep granularity and on
    // instrumentation such as ASan. The assertion rejects what it guards against, an eager
    // breadth-first sweep over every owner option, without depending on that timing.
    const std::size_t eager_sweep =
        owner_count * (static_cast<std::size_t>(program.private_pressure_alternatives) + 2U);
    require(program.pressure_target_assessments < eager_sweep,
            "guided pressure search returned to eager breadth-first assessment");
}

void test_combined_target_reprices_cancelled_pressure_copy() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 51, make_base(51), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 52, make_base(52), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions             = 2;
    program.combined_target_cancels_pressure_copy = true;
    auto inspection = manager.inspect(program, FakePreparedPrompt{53}, make_base(53), 3);
    require(inspection.choice.has_value(),
            "non-monotonic complete target cost made the feasible combination unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{53}, {});
    std::sort(program.started_action_ids.begin(), program.started_action_ids.end());
    const std::vector<std::uint64_t> expected{
        1000U + first.sequence.id,
        1000U + second.sequence.id,
    };
    require(program.started_action_ids == expected,
            "planner accumulated parent transfer cost instead of repricing the complete target");
}

void test_in_progress_adoption_and_private_capture() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    program.progress_in_progress_once = true;
    const ActiveRequest active        = start_active(manager, program, 12, make_base(12), 1);

    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key     = FakeShortlistKey{.digest = 12, .frontier = 24},
        .publishes_private = true,
    };
    program.capture_summary.endpoint = endpoint(12, 24);
    program.capture_summary.long_anchors.push_back(long_anchor(12, 16, 1));
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 1}, true, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "private capture was not reserved");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                !manager.context_transaction_kind(),
            "private capture was not adopted to a stable logical state");
    require(manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Active,
            "private capture disturbed active lane ownership");
    (void)finish_active(manager, program, active, 24);
}

void test_projected_nested_shared_candidates_use_marginal_value() {
    FakeManager manager = make_manager(1, 2, 2);
    FakeProgram program;
    FakeRequestBasePlan base = make_base(61);
    base.cache.opportunities = {
        FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
            .frontier = 32,
        },
        FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
            .frontier = 64,
        },
    };
    const ActiveRequest active = start_active(manager, program, 61, base, 1);
    require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{64},
            "nested shared candidates were selected independently instead of by marginal value");
    (void)finish_active(manager, program, active);
}

void test_observed_shared_candidate_requires_independent_domains() {
    const auto observed_base = [](std::optional<FakeCacheSessionKey> session) {
        FakeRequestBasePlan base = make_base(71, session);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineObserved,
            .frontier = 64,
        });
        return base;
    };

    {
        FakeManager manager = make_manager(1, 3, 1);
        FakeProgram program;
        const ActiveRequest first = start_active(manager, program, 71, observed_base({}), 1);
        require(program.selected_shared_capture_frontiers.empty(),
                "first stateless observation was treated as independent reuse");
        (void)finish_active(manager, program, first);
        const ActiveRequest second = start_active(manager, program, 71, observed_base({}), 2);
        require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{64},
                "two stateless observations did not establish independent reuse demand");
        (void)finish_active(manager, program, second);
    }
    {
        FakeManager manager = make_manager(1, 3, 1);
        FakeProgram program;
        const FakeCacheSessionKey session{.value = 9};
        const ActiveRequest first = start_active(manager, program, 71, observed_base(session), 1);
        (void)finish_active(manager, program, first);
        const ActiveRequest second = start_active(manager, program, 71, observed_base(session), 2);
        require(program.selected_shared_capture_frontiers.empty(),
                "same-session replay was misclassified as shared fanout demand");
        (void)finish_active(manager, program, second);
    }
}

void test_repeated_private_reuse_selects_zero_prefill_shared_promotion() {
    const auto observed_base = [] {
        FakeRequestBasePlan base = make_base(81);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineObserved,
            .frontier = 16,
        });
        return base;
    };

    FakeManager manager = make_manager(1, 3, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 81, observed_base(), 1);
    require(program.selected_shared_capture_frontiers.empty(),
            "first observation selected an unsupported shared promotion");
    (void)finish_active(manager, program, first, 16);

    const ActiveRequest second = start_active(manager, program, 81, observed_base(), 2);
    require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{16},
            "repeated private base was not selected for zero-prefill shared promotion");
    (void)finish_active(manager, program, second, 16);
}

void test_shared_prefix_serves_a_new_question_after_an_exact_repeat() {
    const auto system_base = [] {
        FakeRequestBasePlan base = make_base(91);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
            .frontier = 16,
        });
        return base;
    };

    FakeManager manager = make_manager(1, 2, 4);
    FakeProgram program;

    // The cold request publishes the system prefix as a shared owner.
    const ActiveRequest cold = start_active(manager, program, 91, system_base(), 1);
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 91, .frontier = 16},
        .shared_evidence        = ninfer::SharedCandidateEvidence::EngineStructural,
        .protected_rebuild_work = PrefillWork{.tokens = 16},
        .publishes_shared       = true,
        .physically_feasible    = true,
    };
    require(manager.reserve_active_capture(program, cold.lane, FakeCaptureOffer{.id = 91}, 0, {}) ==
                FakeManager::ActiveCaptureReserveResult::Reserved,
            "the cold request did not reserve the shared capture");
    {
        auto progress      = manager.progress_context_transaction(program, {});
        const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
        require(outcome.status == ContextTransactionStatus::Published,
                "the shared prefix was not published");
    }
    program.capture_assessment = FakeCaptureAssessment{};
    (void)finish_active(manager, program, cold, 16);

    // An exact repeat reuses the private continuation, as production does.
    const ActiveRequest repeat = start_active(manager, program, 91, system_base(), 2);
    (void)finish_active(manager, program, repeat, 16);

    // The new question shares only the system prefix; one preserving private action makes room.
    program.required_pressure_actions     = 1;
    program.private_pressure_alternatives = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{92, 91}, system_base(), 3);
    require(inspection.choice.has_value(), "the new question produced no admission choice");
    require(inspection.choice->summary().prefix_reuse_path == PrefixReusePath::SharedStablePrefix,
            "the new question recomputed from root while the shared prefix was resident");
}

void test_shared_fanout_keeps_owner_edges_live_across_summary_refresh() {
    FakeManager manager = make_manager(2, 3, 1);
    FakeProgram program;

    FakeRequestBasePlan seed_base = make_base(91);
    seed_base.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest seed   = start_active(manager, program, 91, seed_base, 1);
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 91, .frontier = 64},
        .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .protected_rebuild_work = PrefillWork{.tokens = 64},
        .publishes_shared       = true,
        .physically_feasible    = true,
    };
    require(manager.reserve_active_capture(program, seed.lane, FakeCaptureOffer{.id = 31}, 0, {}) ==
                FakeManager::ActiveCaptureReserveResult::Reserved,
            "shared-fanout fixture could not publish its source");
    auto capture_progress = manager.progress_context_transaction(program, {});
    const auto capture = std::get<FakeManager::ActiveCaptureOutcome>(std::move(capture_progress));
    require(capture.status == ContextTransactionStatus::Published,
            "shared-fanout fixture did not publish its source");
    (void)finish_active(manager, program, seed);

    program.report_shared_source_summary                    = true;
    program.change_shared_source_residency_on_second_report = true;
    program.reported_shared_active_references               = 0;
    const ActiveRequest first  = start_active(manager, program, 91, make_base(91), 2);
    const ActiveRequest second = start_active(manager, program, 91, make_base(91), 3);

    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 2,
            "two shared branches did not retain two logical owner edges");
    (void)finish_active(manager, program, first);
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 1,
            "first shared branch release invalidated the surviving owner edge");
    (void)finish_active(manager, program, second);
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 0,
            "shared fanout did not release both logical owner edges");
}

void test_shared_capture_combines_two_pressure_owners() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 41, make_base(41), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 42, make_base(42), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan shared_request = make_base(43);
    shared_request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 43, shared_request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 43, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 7}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "shared capture did not reserve a multi-owner pressure target");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                program.started_action_ids.size() == 2,
            "shared capture did not publish the selected two-owner target");
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 1,
            "shared capture publication did not retain the active owner reference");
    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);
}

// A shared publication that is physically infeasible has to be able to look for something to
// reclaim.  Victim enumeration used to be gated on pressure_evidence alone -- evidence that says
// whether DISPLACING a shared slot is warranted, which is a different question from whether a
// state image can be FREED.  With no explicit boundary and a single reuse domain the gate is
// false, so an infeasible capture was handed zero victims and could never plan: the shared
// frontier stopped advancing the moment the State pools saturated, while a vacant shared slot
// sat unusable behind them.
void test_infeasible_shared_capture_may_reclaim_without_pressure_evidence() {
    FakeManager manager = make_manager(1, 4, 2);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 401, make_base(401), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 402, make_base(402), 2);
    (void)finish_active(manager, program, second);

    // No ExplicitBoundary and no RequestedAutomatic: pressure_evidence is false, which is the
    // ordinary case for a single conversation walking its own context forward.
    FakeRequestBasePlan request = make_base(403);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::None,
        .frontier = 64,
    });
    const ActiveRequest active            = start_active(manager, program, 403, request, 3);
    program.required_pressure_actions     = 1;
    program.pressure_action_immediate_ns  = 0;
    program.capture_assessment            = FakeCaptureAssessment{
                   .shortlist_key          = FakeShortlistKey{.digest = 403, .frontier = 64},
                   .protected_rebuild_work = PrefillWork{.tokens = 64},
                   .publishes_shared       = true,
                   .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 81}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "an infeasible shared capture was denied any reclamation victim");
    require(program.started_action_ids.size() == program.required_pressure_actions,
            "infeasible shared capture reserved without the reclamation it needed");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published,
            "reclaiming shared capture did not publish");

    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);
}

void test_aborted_shared_capture_start_rolls_back_logical_claims() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 141, make_base(141), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 142, make_base(142), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan request = make_base(143);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 143, request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 143, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };
    program.abort_capture_start = true;

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 13}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Skipped &&
                !manager.context_transaction_kind() && !program.has_context_transaction(),
            "aborted shared capture start retained transaction ownership");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued,
            "aborted shared capture start leaked logical victim claims");

    program.abort_capture_start = false;
    const auto retried =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 14}, 0, {});
    require(retried == FakeManager::ActiveCaptureReserveResult::Reserved,
            "aborted shared capture start leaked the shared publication slot");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published,
            "shared capture retry did not publish after rollback");

    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);
}

// Portfolio arbitration cannot justify publishing a zero-demand private checkpoint over a
// resident that still carries demand, so a saturated catalog ossifies on the first conversation.
// A resident whose conversation continued would have been consumed into the active lane, so one
// left catalogued with no active edge is reclaimable by publication order.
void test_private_only_capture_reclaims_stale_resident_by_publication_order() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest resident = start_active(manager, program, 341, make_base(341), 7);
    (void)finish_active(manager, program, resident);

    const ActiveRequest active              = start_active(manager, program, 342, make_base(342), 9);
    program.capture_feasible_after_release  = true;
    program.capture_assessment              = FakeCaptureAssessment{
                     .shortlist_key          = FakeShortlistKey{.digest = 342, .frontier = 64},
                     .protected_rebuild_work = PrefillWork{.tokens = 64},
                     .publishes_private      = true,
                     .publishes_shared       = false,
                     .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 71}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "private-only capture did not reclaim a stale resident");
    require(program.released_continuations.size() == 1,
            "stale reclamation released the wrong number of continuations");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Vacant,
            "reclaimed resident was left catalogued");

}

void test_stale_reclamation_prefers_the_oldest_publication_order() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    // Catalogued newest-first, so a correct choice cannot come from slot order.
    const ActiveRequest newer = start_active(manager, program, 351, make_base(351), 40);
    (void)finish_active(manager, program, newer);
    const ActiveRequest older = start_active(manager, program, 352, make_base(352), 5);
    (void)finish_active(manager, program, older);

    const ActiveRequest active             = start_active(manager, program, 353, make_base(353), 60);
    program.capture_feasible_after_release = true;
    program.capture_assessment             = FakeCaptureAssessment{
                    .shortlist_key          = FakeShortlistKey{.digest = 353, .frontier = 64},
                    .protected_rebuild_work = PrefillWork{.tokens = 64},
                    .publishes_private      = true,
                    .publishes_shared       = false,
                    .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 72}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "stale reclamation did not reserve");
    require(program.released_continuations.size() == 1 &&
                program.released_continuations.front() == older.sequence.id,
            "stale reclamation did not choose the oldest publication_order");

}

// A private-only candidate with no committed demand and no shared credit cannot outvalue a
// resident, so the pressure search can only return no plan after spending its whole budget on the
// engine thread. Reclamation settles it without that search.
void test_zero_value_private_capture_skips_the_pressure_search() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest resident = start_active(manager, program, 361, make_base(361), 7);
    (void)finish_active(manager, program, resident);

    const ActiveRequest active             = start_active(manager, program, 362, make_base(362), 9);
    program.capture_feasible_after_release = true;
    program.capture_assessment             = FakeCaptureAssessment{
                    .shortlist_key          = FakeShortlistKey{.digest = 362, .frontier = 64},
                    .protected_rebuild_work = PrefillWork{.tokens = 64},
                    .publishes_private      = true,
                    .publishes_shared       = false,
                    .physically_feasible    = false,
    };
    const std::uint64_t sessions_before = program.pressure_planning_sessions;

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 73}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "zero-value private capture did not reclaim a stale resident");
    require(program.pressure_planning_sessions == sessions_before,
            "zero-value private capture still ran the pressure search");
    require(program.released_continuations.size() == 1,
            "zero-value private capture released the wrong number of continuations");
}

// Shared credit gives the same private-only candidate a value of its own, so arbitration must
// still weigh it against the residents.
void test_credited_private_capture_keeps_the_pressure_search() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest resident = start_active(manager, program, 371, make_base(371), 7);
    (void)finish_active(manager, program, resident);

    const ActiveRequest active             = start_active(manager, program, 372, make_base(372), 9);
    program.capture_feasible_after_release = true;
    program.capture_assessment             = FakeCaptureAssessment{
                    .shortlist_key          = FakeShortlistKey{.digest = 372, .frontier = 64},
                    .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                    .protected_rebuild_work = PrefillWork{.tokens = 64},
                    .publishes_private      = true,
                    .publishes_shared       = false,
                    .physically_feasible    = false,
    };
    const std::uint64_t sessions_before = program.pressure_planning_sessions;

    (void)manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 74}, 0, {});
    require(program.pressure_planning_sessions > sessions_before,
            "credited private capture skipped the pressure search");
}

void test_stale_reclamation_never_takes_a_retained_active_source() {
    FakeManager manager = make_manager(2, 3, 1);
    FakeProgram program;
    const FakeCacheSessionKey first_session{1};
    const FakeCacheSessionKey second_session{2};
    const ActiveRequest seed = start_active(
        manager, program, 9, make_base(9, first_session, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, seed);
    const ActiveRequest fork = start_active(
        manager, program, 9, make_base(9, second_session, RetentionClass::LiveSession), 2);
    require(program.started_source_mode == PrivateSourceMode::Retain,
            "the resident continuation was consumed instead of retained");

    program.capture_feasible_after_release = true;
    program.capture_assessment             = FakeCaptureAssessment{
                    .shortlist_key          = FakeShortlistKey{.digest = 9, .frontier = 64},
                    .protected_rebuild_work = PrefillWork{.tokens = 64},
                    .publishes_private      = true,
                    .publishes_shared       = false,
                    .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, fork.lane, FakeCaptureOffer{.id = 73}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Skipped,
            "stale reclamation took a source an active request still holds");
    require(program.released_continuations.empty(),
            "stale reclamation released a live continuation");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "stale reclamation disturbed the retained active source");

    (void)manager.abort(program, fork.lane, fork.sequence);
}

void test_stale_reclamation_does_not_run_when_the_capture_is_feasible() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest resident = start_active(manager, program, 361, make_base(361), 3);
    (void)finish_active(manager, program, resident);

    const ActiveRequest active             = start_active(manager, program, 362, make_base(362), 4);
    program.capture_feasible_after_release = true;
    program.capture_assessment             = FakeCaptureAssessment{
                    .shortlist_key          = FakeShortlistKey{.digest = 362, .frontier = 64},
                    .protected_rebuild_work = PrefillWork{.tokens = 64},
                    .publishes_private      = true,
                    .publishes_shared       = false,
                    .physically_feasible    = true,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 74}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "a feasible private capture did not reserve");
    require(program.released_continuations.empty(),
            "stale reclamation ran even though the capture was already feasible");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "a feasible capture reclaimed a resident it did not need");

}

// An admission that had to degrade a resident is pressure deciding which residents to keep; the
// optional capture of that same request must not reclaim one of them.
void test_stale_reclamation_leaves_residents_kept_by_admission_pressure() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest resident = start_active(manager, program, 381, make_base(381), 1);
    (void)finish_active(manager, program, resident);

    program.required_pressure_actions = 1;
    const ActiveRequest active        = start_active(manager, program, 382, make_base(382), 2);
    RuntimeStats admitted;
    manager.populate_runtime_stats(program, admitted);
    require(admitted.pressure_private_owners_degraded == 1 &&
                admitted.pressure_private_owners_evicted == 0,
            "admission did not keep the resident under pressure");

    program.required_pressure_actions      = 0;
    program.capture_feasible_after_release = true;
    program.capture_assessment             = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 382, .frontier = 64},
        .protected_rebuild_work = PrefillWork{.tokens = 64},
        .publishes_private      = true,
        .publishes_shared       = false,
        .physically_feasible    = false,
    };
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 75}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Skipped,
            "a capture reclaimed a resident its own admission pressure had kept");
    require(program.released_continuations.empty() &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "the resident kept by admission pressure was released");
}

// Under --recency-eviction the stale reclaim ranks residents by the owner recency the pressure path
// uses, so an older publication that a later request reused outlives a newer one nobody reused.
// The released checkpoints count as pressure drops like any other eviction.
void test_stale_reclamation_ranks_by_recency_under_recency_eviction() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    manager.enable_recency_eviction();
    const FakeCacheSessionKey first_session{1};
    const FakeCacheSessionKey second_session{2};
    const ActiveRequest reused = start_active(
        manager, program, 9, make_base(9, first_session, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, reused);
    const ActiveRequest unused = start_active(manager, program, 391, make_base(391), 2);
    (void)finish_active(manager, program, unused);
    const ActiveRequest fork = start_active(
        manager, program, 9, make_base(9, second_session, RetentionClass::LiveSession), 3);
    require(program.started_source_mode == PrivateSourceMode::Retain,
            "the reused continuation was consumed instead of retained");
    (void)finish_active(manager, program, fork);

    const ActiveRequest active             = start_active(manager, program, 392, make_base(392), 4);
    program.capture_feasible_after_release = true;
    program.capture_assessment             = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 392, .frontier = 64},
        .protected_rebuild_work = PrefillWork{.tokens = 64},
        .publishes_private      = true,
        .publishes_shared       = false,
        .physically_feasible    = false,
    };
    RuntimeStats before;
    manager.populate_runtime_stats(program, before);
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 76}, 0, {});
    RuntimeStats after;
    manager.populate_runtime_stats(program, after);
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "recency-ranked stale reclamation did not reserve");
    require(program.released_continuations.size() == 1 &&
                program.released_continuations.front() == unused.sequence.id,
            "stale reclamation ignored that a later request reused the older resident");
    require(after.pressure_private_owners_evicted == before.pressure_private_owners_evicted + 1 &&
                after.pressure_checkpoints_dropped > before.pressure_checkpoints_dropped,
            "stale reclamation did not count the checkpoints it dropped");
}

void test_capture_result_is_validated_before_any_adoption() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 241, make_base(241), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 242, make_base(242), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan request = make_base(243);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 243, request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 243, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };
    program.malform_last_capture_private_victim = true;

    require(manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 17}, 0,
                                           {}) == FakeManager::ActiveCaptureReserveResult::Reserved,
            "malformed capture fixture could not reserve pressure");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed capture fixture did not claim both owners");

    bool rejected = false;
    try {
        (void)manager.progress_context_transaction(program, {});
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "malformed capture result was accepted");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed capture result partially adopted an earlier victim");
}

void test_capture_result_is_adopted_by_owner_identity() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 251, make_base(251), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 252, make_base(252), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan request = make_base(253);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 253, request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.reverse_pressure_results     = true;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 253, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };

    require(manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 27}, 0,
                                           {}) == FakeManager::ActiveCaptureReserveResult::Reserved,
            "reordered capture fixture could not reserve pressure");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued,
            "reordered capture results changed victim availability");
    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);

    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{251}, make_base(251), 4);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered capture result attached the first summary to another owner");
    }
    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{252}, make_base(252), 5);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered capture result attached the second summary to another owner");
    }
}

void test_terminal_fallback_releases_failed_retention() {
    FakeManager manager = make_manager(1, 1);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 4, make_base(4), 1);
    manager.mark_terminal_pending(active.lane);
    program.finish_fail_next      = true;
    const FakeFinishResult result = manager.finish(program, active.lane, active.sequence);
    require(result.status == ConsumeStatus::Consumed &&
                result.disposition == FinishDisposition::Released,
            "failed retention did not converge to a released terminal result");
    require(result.timings.value == 7 && result.speculative.value == 9,
            "terminal fallback lost abort accounting");
    require(program.abort_calls == 1 &&
                manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Free &&
                manager.catalog_state(0) == FakeManager::CatalogState::Vacant,
            "terminal fallback did not free every logical owner");
}

void test_abort_salvage_publishes_reusable_endpoint() {
    FakeManager manager = make_manager(1, 1);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 41, make_base(41), 1);

    program.abort_salvage_next   = true;
    const FakeAbortResult result = manager.abort(program, active.lane, active.sequence);
    require(result.status == ConsumeStatus::Consumed && result.salvaged,
            "abort did not report its salvaged continuation");
    require(result.timings.value == 7 && result.speculative.value == 9,
            "salvage did not carry abort accounting");
    require(program.abort_calls == 1 &&
                manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Free &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "salvaged abort did not free the lane or catalogue its endpoint");

    auto reuse = manager.inspect(program, FakePreparedPrompt{41}, make_base(41), 2);
    require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
            "salvaged endpoint was not reusable by the retrying prompt");
}

void test_terminal_fallback_salvage_catalogues() {
    FakeManager manager = make_manager(1, 1);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 42, make_base(42), 1);
    manager.mark_terminal_pending(active.lane);
    program.finish_fail_next      = true;
    program.abort_salvage_next    = true;
    const FakeFinishResult result = manager.finish(program, active.lane, active.sequence);
    require(result.status == ConsumeStatus::Consumed &&
                result.disposition == FinishDisposition::Catalogued,
            "salvaged terminal fallback was not catalogued");
    require(result.timings.value == 7 && result.speculative.value == 9,
            "salvaged terminal fallback lost abort accounting");
    require(program.abort_calls == 1 &&
                manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Free &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "salvaged terminal fallback did not free every logical owner");

    auto reuse = manager.inspect(program, FakePreparedPrompt{42}, make_base(42), 2);
    require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
            "salvaged terminal fallback endpoint was not reusable");
}

void test_abort_salvage_earns_disposable_retention_under_pressure() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest salvaged = start_active(
        manager, program, 1, make_base(1, FakeCacheSessionKey{1}, RetentionClass::LiveSession), 1);
    program.abort_salvage_next = true;
    (void)manager.abort(program, salvaged.lane, salvaged.sequence);
    const ActiveRequest recent = start_active(
        manager, program, 2, make_base(2, std::nullopt, RetentionClass::RecentPrivate), 2);
    (void)finish_active(manager, program, recent);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{3}, make_base(3), 3);
    require(inspection.choice.has_value(),
            "salvage retention pressure did not find a feasible prefix");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{3}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + salvaged.sequence.id,
            "salvage did not earn Disposable retention below LiveSession");
}

void test_full_anchor_set_replaces_the_least_coverage_anchor() {
    // Anchors at 1000, 5000, 5200 and 9000 with a new capture at 12000. Dropping 5200 costs
    // (5200 - 5000) * (9000 - 5200); every other anchor covers more. The old rule always gave up
    // the deepest anchor (1000), letting the set drift towards the endpoint.
    FakeManager manager = make_manager(1, 4, 0);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 91, make_base(91), 1);
    const auto anchor          = [](std::uint32_t frontier, std::uint32_t ordinal) {
        return CheckpointRef{
            .kind = CheckpointKind::LongAnchor, .frontier = frontier, .ordinal = ordinal};
    };
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key                  = FakeShortlistKey{.digest = 91, .frontier = 12000},
        .protected_rebuild_work         = PrefillWork{.tokens = 12000},
        .private_replacement_candidates = {anchor(1000, 1), anchor(5000, 2), anchor(5200, 3),
                                           anchor(9000, 4)},
        .publishes_private              = true,
        .physically_feasible            = true,
    };
    (void)manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 1}, 0, {});
    require(program.last_capture_private_replacement == anchor(5200, 3),
            "full anchor set did not give up the anchor whose loss costs the least coverage");
}

void test_terminal_settlement_waits_for_open_resource_transaction() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 31, make_base(31), 1);

    auto inspection = manager.inspect(program, FakePreparedPrompt{32}, make_base(32), 2);
    require(inspection.choice.has_value(), "concurrent materialization choice was not produced");
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{32}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "concurrent materialization was not reserved");

    const auto capture =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 9}, true, {});
    require(capture == FakeManager::ActiveCaptureReserveResult::Skipped &&
                program.skipped_captures == 1 &&
                manager.context_transaction_kind() ==
                    ninfer::runtime::ContextTransactionKind::Materialization,
            "optional capture was not skipped behind the open materialization");

    manager.mark_terminal_pending(active.lane);
    bool rejected = false;
    try {
        (void)manager.finish(program, active.lane, active.sequence);
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected && manager.lane_state(active.lane) ==
                            ninfer::runtime::LogicalLaneState::TerminalPending,
            "terminal settlement changed topology during an open resource transaction");
}

void test_commit_and_discard_terminal_states() {
    FakeManager manager = make_manager(2, 2);
    FakeProgram program;
    const ActiveRequest first  = start_active(manager, program, 1, make_base(1), 1);
    const ActiveRequest second = start_active(manager, program, 2, make_base(2), 2);
    const std::array<LaneId, 2> lanes{first.lane, second.lane};
    FakeCommitResult commit;
    commit.row_count           = 2;
    commit.rows[0].disposition = CommitDisposition::Active;
    commit.rows[1].disposition = CommitDisposition::Finishable;
    manager.apply_commit(lanes, commit);
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Active &&
                manager.lane_state(second.lane) ==
                    ninfer::runtime::LogicalLaneState::TerminalPending,
            "row-aligned commit did not establish terminal pending state");
    (void)manager.finish(program, second.lane, second.sequence);

    FakeDiscardResult discard{.status = ConsumeStatus::Consumed, .row_count = 1};
    const std::array<LaneId, 1> remaining{first.lane};
    manager.apply_discard(remaining, discard);
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Free,
            "discard did not release cancelled active membership");
}

void test_backfill_proof_and_stats_follow_program_revision() {
    FakeManager manager = make_manager();
    FakeProgram program;
    auto inspection = manager.inspect(program, FakePreparedPrompt{8}, make_base(8), 1);
    require(inspection.choice.has_value(), "backfill test did not produce a candidate plan");
    const std::array<FakeSequenceHandle, 0> borrowers{};
    auto proof =
        manager.prove_persistent_backfill(program, make_base(99), *inspection.choice, borrowers);
    require(proof && proof->resource_revision() == program.resource_revision(),
            "Program did not seal a current-revision persistent proof");
    program.invalidate_resources();
    proof =
        manager.prove_persistent_backfill(program, make_base(99), *inspection.choice, borrowers);
    require(!proof, "resource revision change did not invalidate persistent proof");

    program.usage = FakePhysicalUsage{
        .device_state_slots            = 3,
        .host_state_slots              = 2,
        .device_main_kv_pages          = 11,
        .device_backend_kv_pages       = 5,
        .device_main_kv_lease_pages    = 4,
        .device_backend_kv_lease_pages = 2,
        .host_kv_bytes                 = 4096,
    };
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.device_state_occupied_slots == 3 && stats.host_state_occupied_slots == 2 &&
                stats.device_main_kv_occupied_pages == 11 &&
                stats.device_backend_kv_occupied_pages == 5 &&
                stats.device_main_kv_lease_pages == 4 && stats.device_backend_kv_lease_pages == 2 &&
                stats.host_kv_occupied_bytes == 4096,
            "runtime physical gauges did not come directly from Program");
}

void test_shortlist_collision_requires_program_exact_verification() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 55, make_base(123), 1);
    (void)finish_active(manager, program, seed);

    auto collision = manager.inspect(program, FakePreparedPrompt{99}, make_base(55), 2);
    require(collision.choice && collision.choice->summary().reusable_prompt_tokens == 0,
            "shortlist collision bypassed Program exact identity verification");
}

// Enumerates raw owner choices independently of the search frontier/generator and portfolio fold.
// Each owner can keep its cache, spill it (one relief unit), or drop it (two relief units).
void test_complete_search_against_small_exhaustive_oracle() {
    using Planner              = ninfer::runtime::MaterializationPlanner<FakeModelContract>;
    constexpr std::uint64_t ms = 1'000'000;
    const std::array<std::uint64_t, 3> rebuild{600 * ms, 200 * ms, 100 * ms};
    const std::array<std::uint64_t, 3> spill{40 * ms, 200 * ms, 10 * ms};
    const std::array<std::uint64_t, 3> drop{ms, 2 * ms, 3 * ms};
    for (unsigned weight_rotation = 0; weight_rotation < 3; ++weight_rotation) {
        for (unsigned required = 0; required <= 3; ++required) {
            for (bool full_catalog : {false, true}) {
                FakeProgram program;
                program.required_pressure_actions       = required;
                program.eviction_pressure_action_units  = 2;
                program.pressure_checkpoint_recovery_ns = 1'000 * ms;
                std::array<FakeContinuationHandle, 3> handles;
                std::array<const FakeContinuationHandle*, 3> owners;
                std::array<PlanningOwnerId, 3> ids;
                std::array<ninfer::runtime::MaterializationOwnerPolicy, 3> policies;
                std::array<ninfer::runtime::MaterializationCheckpointPolicy, 3> checkpoints;
                const std::array<unsigned, 3> weights{1, 4, 16};
                for (unsigned i = 0; i < 3; ++i) {
                    handles[i]     = FakeContinuationHandle{i + 1, 0};
                    owners[i]      = &handles[i];
                    ids[i]         = {.value = i};
                    policies[i]    = {.owner                    = ids[i],
                                      .private_retention_weight = weights[(i + weight_rotation) % 3]};
                    checkpoints[i] = {.owner       = ids[i],
                                      .checkpoint  = {.kind     = CheckpointKind::SessionEndpoint,
                                                      .frontier = 16,
                                                      .ordinal  = 0},
                                      .demand_mask = 1,
                                      .rebuild_ns  = rebuild[i]};
                    program.owner_decisions.push_back({i + 1,
                                                       {{.id = 1000 + i, .immediate_ns = spill[i]},
                                                        {.id                  = 2000 + i,
                                                         .immediate_ns        = drop[i],
                                                         .degradation_units   = 4,
                                                         .dropped_checkpoints = 1,
                                                         .evicts_continuation = true}}});
                }
                FakeAdmissionCandidate root, reuse;
                set_fake_machine_costs(root.identity.machine_work, 800 * ms, 800 * ms);
                set_fake_machine_costs(reuse.identity.machine_work, 100 * ms, 100 * ms);
                reuse.private_source_id                          = 1;
                reuse.value.reusable_prompt_tokens               = 48;
                reuse.identity.machine_work.reused_prompt_tokens = 48;
                for (auto* candidate : {&root, &reuse}) {
                    candidate->identity.physical_status =
                        required == 0 ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                                      : ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
                    candidate->identity.expandable = required != 0;
                }
                const std::array candidates{
                    Planner::CandidateInput{.candidate = &root, .id = {.value = 0}},
                    Planner::CandidateInput{
                        .candidate = &reuse, .id = {.value = 1}, .stable_ordinal = 1}};
                const auto inputs = [&]() -> Planner::PressureInputs {
                    return {.private_owners    = owners,
                            .private_owner_ids = ids,
                            .owner_policy      = policies,
                            .checkpoint_policy = checkpoints};
                };
                const auto goal =
                    [&](PlanningCandidateId candidate, PrivateSourceMode,
                        std::span<const ninfer::runtime::PressureOwnerOutcome> outcomes)
                    -> std::optional<Planner::LogicalGoal> {
                    if (candidate.value == 1 || !full_catalog ||
                        std::any_of(outcomes.begin(), outcomes.end(), [](auto outcome) {
                            return outcome.disposition == VictimDisposition::Evicted;
                        })) {
                        return Planner::LogicalGoal{.publication_slot = 0};
                    }
                    return std::nullopt;
                };
                std::uint64_t oracle = UINT64_MAX;
                for (unsigned source = 0; source < 2; ++source) {
                    for (unsigned raw = 0; raw < 27; ++raw) {
                        unsigned digits = raw, relief = 0;
                        bool frees_slot = false, protects_source = true;
                        std::uint64_t cost                    = (source ? 100 : 800) * ms;
                        std::uint64_t remaining_public_saving = 0;
                        for (unsigned owner = 0; owner < 3; ++owner) {
                            unsigned choice = digits % 3;
                            digits /= 3;
                            if (source == 1 && owner == 0 && choice != 0) {
                                protects_source = false;
                            }
                            relief += choice;
                            if (choice == 2) {
                                frees_slot = true;
                                cost += drop[owner] +
                                        rebuild[owner] * weights[(owner + weight_rotation) % 3];
                            } else {
                                remaining_public_saving =
                                    std::max(remaining_public_saving, rebuild[owner]);
                                if (choice == 1) { cost += spill[owner]; }
                            }
                        }
                        if (!protects_source || relief < required ||
                            (full_catalog && source == 0 && !frees_slot)) {
                            continue;
                        }
                        cost += rebuild[0] - remaining_public_saving;
                        oracle = std::min(oracle, cost);
                    }
                }
                Planner planner;
                auto allowance     = ninfer::runtime::PlanningAllowance::boundary(0);
                allowance.limit_ns = 5 * ms;
                auto result =
                    planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                                 inputs, goal, Planner::Clock::now(), allowance);
                require(result && result->plan,
                        "exhaustive-oracle problem lost its feasible fallback");
                if (result->diagnostics.predicted_total_ns != oracle) {
                    std::cerr << "oracle rotation=" << weight_rotation << " relief=" << required
                              << " full=" << full_catalog << " expected=" << oracle
                              << " selected=" << result->diagnostics.predicted_total_ns << '\n';
                }
                require(result->diagnostics.predicted_total_ns == oracle,
                        "complete search missed the independent small-problem optimum");
                require(
                    std::none_of(result->plan->private_owner_ids.begin(),
                                 result->plan->private_owner_ids.end(),
                                 [&](auto id) { return result->candidate.value == 1 && id == 1; }),
                    "construction included the selected source as a victim");
            }
        }
    }
}

void test_publication_only_pressure_constructs_adoptable_target() {
    constexpr unsigned owners = 7;
    FakeManager manager       = make_manager(1, owners);
    FakeProgram program;
    for (unsigned i = 0; i < owners; ++i) {
        const auto active = start_active(manager, program, 500 + i, make_base(500 + i), i + 1);
        (void)finish_active(manager, program, active);
    }
    require(program.required_pressure_actions == 0, "publication fixture has physical pressure");
    auto allowance     = ninfer::runtime::PlanningAllowance::boundary(0);
    allowance.limit_ns = 5'000'000;
    auto result = manager.inspect(program, FakePreparedPrompt{600}, make_base(600), 20, allowance);
    require(result.choice.has_value(),
            "publication-only pressure did not produce an adoptable target");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*result.choice),
                                          FakePreparedPrompt{600}, {});
    require(program.started_action_ids.size() == 1 && program.started_action_ids.front() >= 2000,
            "publication-only closure did not release exactly one private slot");
}

// Drives one full request lifecycle that publishes a shared prefix for `digest` at frontier 64
// and finishes it, leaving the shared prefix catalogued. No private continuation is published
// (publish_continuation cleared) so reuse can only come from the shared catalog.
void publish_shared_prefix(FakeManager& manager, FakeProgram& program, std::uint32_t digest,
                           std::uint64_t publication_order, std::uint32_t offer_id,
                           ninfer::SharedCandidateEvidence evidence, std::uint32_t frontier = 64) {
    FakeRequestBasePlan base        = make_base(digest);
    base.value.publish_continuation = false;
    base.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = evidence,
        .frontier = frontier,
    });
    const ActiveRequest active = start_active(manager, program, digest, base, publication_order);
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .shared_evidence        = evidence,
        .protected_rebuild_work = PrefillWork{.tokens = frontier},
        .publishes_shared       = true,
        .physically_feasible    = true,
    };
    const auto reserved = manager.reserve_active_capture(program, active.lane,
                                                         FakeCaptureOffer{.id = offer_id}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "shared capture was not reserved");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published,
            "shared capture was not published");
    (void)finish_active(manager, program, active);
}

void require_shared_reuse(FakeManager& manager, FakeProgram& program, std::uint32_t digest,
                          std::uint64_t publication_order, bool expected, const char* message) {
    auto inspection =
        manager.inspect(program, FakePreparedPrompt{digest}, make_base(digest), publication_order);
    require(inspection.choice.has_value(), "reuse inspection produced no admission choice");
    const bool reused = inspection.choice->summary().prefix_reuse_path ==
                            ninfer::PrefixReusePath::SharedStablePrefix &&
                        inspection.choice->summary().reusable_prompt_tokens == 64;
    require(reused == expected, message);
}

void test_escape_hatch_ranks_shared_prefixes_with_private_owners() {
    // Shared prefixes share the private owners' recency order and get a demote-to-host outcome,
    // so a ladder rung sacrifices the oldest owners of either kind instead of destroying every
    // shared prefix unconditionally, and spares any sacrificed owner the rung does not need. The
    // shared prefix is published at the fake's finish frontier because the fake pressure
    // assessment reports every owner's checkpoint at that frontier.
    const auto run = [](bool shared_first) {
        FakeManager manager = make_manager(1, 4, 1);
        manager.enable_recency_eviction();
        FakeProgram program;
        std::uint64_t order = 1;
        if (shared_first) {
            publish_shared_prefix(manager, program, 71, order++, 1,
                                  ninfer::SharedCandidateEvidence::DefaultAutomatic,
                                  program.finish_frontier);
        }
        const ActiveRequest a = start_active(manager, program, 161, make_base(161), order++);
        (void)finish_active(manager, program, a);
        const ActiveRequest b = start_active(manager, program, 162, make_base(162), order++);
        (void)finish_active(manager, program, b);
        if (!shared_first) {
            publish_shared_prefix(manager, program, 71, order++, 1,
                                  ninfer::SharedCandidateEvidence::DefaultAutomatic,
                                  program.finish_frontier);
        }
        const ActiveRequest c = start_active(manager, program, 163, make_base(163), order++);
        (void)finish_active(manager, program, c);

        program.ladder_feasibility_mode        = true;
        program.min_feasible_sacrifice         = 1;
        program.required_pressure_actions      = 16; // above the search's max pressure
        program.eviction_pressure_action_units = 1;
        program.private_pressure_alternatives  = 1;
        auto inspection = manager.inspect(program, FakePreparedPrompt{164}, make_base(164), order);
        require(inspection.choice.has_value(), "escape-hatch recency ladder found no rung");
        program.abort_start = true;
        (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                              FakePreparedPrompt{164}, {});
        const auto private_evicted = [&program](const ActiveRequest& request) {
            return std::find(program.started_action_ids.begin(), program.started_action_ids.end(),
                             2000U + request.sequence.id) != program.started_action_ids.end();
        };
        const bool shared_evicted =
            std::any_of(program.started_action_ids.begin(), program.started_action_ids.end(),
                        [](std::uint64_t id) { return id >= 4000U && id < 5000U; });
        return std::array<bool, 4>{shared_evicted, private_evicted(a), private_evicted(b),
                                   private_evicted(c)};
    };
    require(run(false) == std::array<bool, 4>{false, true, false, false},
            "rung destroyed a recent shared prefix instead of the oldest private owner");
    // With the shared prefix published first it is the oldest owner, but this admission's
    // pressure is a private catalog slot: only a released private owner relieves it. The smallest
    // adoptable rung gives up the shared prefix and the oldest private owner (the shared
    // publisher's own continuation); sparing then keeps the shared prefix, whose eviction the rung
    // does not need, so no shared prefix and none of a, b, c are evicted.
    require(run(true) == std::array<bool, 4>{false, false, false, false},
            "rung destroyed a shared prefix whose eviction the admission did not need");
}

void test_automatic_shared_capture_reclaims_oldest_catalog_entry() {
    FakeManager manager = make_manager(1, 4, 2);
    FakeProgram program;

    publish_shared_prefix(manager, program, 71, 1, 1,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    publish_shared_prefix(manager, program, 72, 2, 2,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);

    // The catalog (2 slots) is now full of automatic-evidence prefixes. A third distinct
    // conversation must reclaim the oldest publication instead of losing its shared prefix
    // forever; before the fix its capture was silently dropped and the catalog stayed frozen.
    publish_shared_prefix(manager, program, 73, 3, 3,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    require(program.released_shared_prefix_keys == std::vector<std::uint32_t>{71},
            "saturated catalog did not reclaim the oldest automatic shared prefix");
    require_shared_reuse(manager, program, 73, 4, true,
                         "reclaimed conversation has no shared prefix reuse");
    require_shared_reuse(manager, program, 71, 5, false,
                         "reclaimed shared prefix was still offered for reuse");

    // The catalog must keep rotating for every new conversation, not freeze again after one
    // reclaim.
    publish_shared_prefix(manager, program, 74, 6, 4,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    require(program.released_shared_prefix_keys == std::vector<std::uint32_t>{71, 72},
            "second saturated conversation did not reclaim the next-oldest prefix");
    require_shared_reuse(manager, program, 72, 7, false,
                         "second reclaimed prefix was still offered for reuse");
}

void test_explicit_credit_shared_entry_survives_automatic_reclaim() {
    FakeManager manager = make_manager(1, 4, 2);
    FakeProgram program;

    publish_shared_prefix(manager, program, 71, 1, 1,
                          ninfer::SharedCandidateEvidence::ExplicitBoundary);
    publish_shared_prefix(manager, program, 72, 2, 2,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);

    // The explicit-credit entry (71) holds client-declared intent and must not be reclaimed by
    // ordinary automatic traffic; the automatic entry (72) is the LRU victim instead.
    publish_shared_prefix(manager, program, 73, 3, 3,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    require(program.released_shared_prefix_keys == std::vector<std::uint32_t>{72},
            "automatic reclaim evicted an explicit-credit shared prefix");
    require_shared_reuse(manager, program, 71, 4, true,
                         "explicit-credit shared prefix lost reuse after reclaim");
}

void test_shared_capture_seal_failure_skips_instead_of_throwing() {
    // A shared capture is optional. When the seal window cannot be claimed, or the seal's
    // revalidation fails anyway, the scenario must not plan: the capture falls back to its private
    // baseline or is skipped, and nothing is published -- it must not throw out of the worker.
    const auto run = [](bool deny_window) {
        FakeManager manager = make_manager(1, 4, 2);
        FakeProgram program;
        FakeRequestBasePlan base        = make_base(81);
        base.value.publish_continuation = false;
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic,
            .frontier = 64,
        });
        const ActiveRequest active = start_active(manager, program, 81, base, 1);
        program.capture_assessment = FakeCaptureAssessment{
            .shortlist_key          = FakeShortlistKey{.digest = 81, .frontier = 64},
            .shared_evidence        = ninfer::SharedCandidateEvidence::DefaultAutomatic,
            .protected_rebuild_work = PrefillWork{.tokens = 64},
            .publishes_shared       = true,
            .physically_feasible    = true,
        };
        program.deny_seal_window          = deny_window;
        program.fail_capture_seal         = !deny_window;
        const std::uint64_t claims_before = program.seal_window_claims;
        const auto reserved =
            manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 1}, 0, {});
        require(program.seal_window_claims > claims_before,
                "shared capture sealed without claiming the seal window");
        program.deny_seal_window  = false;
        program.fail_capture_seal = false;
        if (reserved == FakeManager::ActiveCaptureReserveResult::Reserved) {
            auto progress      = manager.progress_context_transaction(program, {});
            const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
            require(outcome.status == ContextTransactionStatus::Published,
                    "private baseline capture was not published");
        }
        (void)finish_active(manager, program, active);
        require_shared_reuse(manager, program, 81, 2, false,
                             "unsealed shared capture was published");
    };
    run(true);
    run(false);
}

// A running answer whose Device KV lease ran out of pool space takes pages back from retained
// cache: least recently used idle owners first, never an owner that frees no Device page.
void test_lease_reclaim_releases_least_recent_idle_owners() {
    FakeManager manager = make_manager(1, 4);
    FakeProgram program;
    for (std::uint32_t key : {81U, 82U, 83U}) {
        const ActiveRequest request = start_active(manager, program, key, make_base(key), key);
        (void)finish_active(manager, program, request);
    }
    program.retained_kv_pages[82] = {};

    require(manager.reclaim_device_kv_for_lease(program, 1, 0) == 1 &&
                program.released_continuation_keys == std::vector<std::uint32_t>{81},
            "lease reclaim did not release only the least recently used owner");
    require(manager.reclaim_device_kv_for_lease(program, 2, 0) == 1 &&
                program.released_continuation_keys == std::vector<std::uint32_t>{81, 83},
            "lease reclaim released an owner that frees no Device page");
    require(manager.reclaim_device_kv_for_lease(program, 1, 0) == 0,
            "lease reclaim released an owner with no Device page to give back");
    auto inspection = manager.inspect(program, FakePreparedPrompt{81}, make_base(81), 90);
    require(inspection.choice.has_value() &&
                inspection.choice->summary().reusable_prompt_tokens == 0,
            "a reclaimed owner was still offered for reuse");
}

// Two turns of one session run concurrently and the later-submitted one finishes first, so it
// holds the session binding. The earlier turn publishing afterwards is a stale branch: finishing
// last must not make it outrank the session's current continuation under pressure.
void test_superseded_session_turn_ranks_below_its_binding() {
    FakeManager manager = make_manager(2, 4);
    FakeProgram program;
    const FakeCacheSessionKey session{7};
    const ActiveRequest older =
        start_active(manager, program, 91, make_base(91, session, RetentionClass::LiveSession), 1);
    const ActiveRequest newer =
        start_active(manager, program, 92, make_base(92, session, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, newer);
    (void)finish_active(manager, program, older);

    require(manager.reclaim_device_kv_for_lease(program, 1, 0) == 1 &&
                program.released_continuation_keys == std::vector<std::uint32_t>{91},
            "a stale session turn outranked the session's current binding");
}

void test_automatic_reclaim_picks_the_least_recently_used_entry() {
    FakeManager manager = make_manager(1, 4, 2);
    FakeProgram program;

    publish_shared_prefix(manager, program, 71, 1, 1,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    publish_shared_prefix(manager, program, 72, 2, 2,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    // A later conversation reuses 71, so 72 is now the least recently used entry even though
    // 71 was published first.
    const ActiveRequest reuse = start_active(manager, program, 71, make_base(71), 3);
    (void)finish_active(manager, program, reuse);

    publish_shared_prefix(manager, program, 73, 4, 3,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    require(program.released_shared_prefix_keys == std::vector<std::uint32_t>{72},
            "automatic reclaim chose by publication order instead of by use");
}

void test_automatic_reclaim_waits_for_a_planned_capture() {
    FakeManager manager = make_manager(1, 4, 2);
    FakeProgram program;

    publish_shared_prefix(manager, program, 71, 1, 1,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    publish_shared_prefix(manager, program, 72, 2, 2,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);

    // A third conversation's capture cannot fit, so no scenario plans. The saturated catalog
    // must not give up an entry for a publication that never happens.
    FakeRequestBasePlan base        = make_base(73);
    base.value.publish_continuation = false;
    base.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic,
        .frontier = 64,
    });
    const ActiveRequest active = start_active(manager, program, 73, base, 3);
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 73, .frontier = 64},
        .shared_evidence        = ninfer::SharedCandidateEvidence::DefaultAutomatic,
        .protected_rebuild_work = PrefillWork{.tokens = 64},
        .publishes_shared       = true,
        .physically_feasible    = false,
    };
    program.required_pressure_actions = 1; // the capture only fits with pressure it cannot get
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 3}, 0, {});
    program.required_pressure_actions = 0;
    if (reserved == FakeManager::ActiveCaptureReserveResult::Reserved) {
        // Only the private baseline can proceed; the shared publication did not plan.
        auto progress      = manager.progress_context_transaction(program, {});
        const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
        require(outcome.status == ContextTransactionStatus::Published,
                "private baseline capture was not published");
    }
    require(program.released_shared_prefix_keys.empty(),
            "automatic reclaim released a prefix for a capture that did not plan");
    (void)finish_active(manager, program, active);
    require_shared_reuse(manager, program, 71, 4, true,
                         "shared prefix was lost to a capture that never published");
}

// A request that extends the only automatic shared prefix must not reclaim it for a capture of
// its own longer prefix, even when it resumed from a deeper private checkpoint instead: every
// request that capture could serve also matches the shorter prefix.
void test_automatic_reclaim_spares_the_prefix_the_capture_extends() {
    FakeManager manager = make_manager(1, 4, 1);
    FakeProgram program;

    publish_shared_prefix(manager, program, 71, 1, 1,
                          ninfer::SharedCandidateEvidence::DefaultAutomatic);
    const ActiveRequest first = start_active(manager, program, 71, make_base(71), 2);
    (void)finish_active(manager, program, first, 96);

    FakeRequestBasePlan base        = make_base(71);
    base.value.prompt_tokens        = 128;
    base.value.publish_continuation = false;
    base.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic,
        .frontier = 128,
    });
    auto inspection = manager.inspect(program, FakePreparedPrompt{71}, base, 3);
    require(inspection.choice.has_value() &&
                inspection.choice->summary().prefix_reuse_path !=
                    ninfer::PrefixReusePath::SharedStablePrefix &&
                inspection.choice->summary().reusable_prompt_tokens > 64,
            "the extending request did not resume from its deeper private checkpoint");
    const ActiveRequest active = start_active(manager, program, 71, base, 3);
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 71, .frontier = 128},
        .shared_evidence        = ninfer::SharedCandidateEvidence::DefaultAutomatic,
        .protected_rebuild_work = PrefillWork{.tokens = 128},
        .publishes_shared       = true,
        .physically_feasible    = true,
    };
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 2}, 0, {});
    if (reserved == FakeManager::ActiveCaptureReserveResult::Reserved) {
        auto progress = manager.progress_context_transaction(program, {});
        (void)std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    }
    require(program.released_shared_prefix_keys.empty(),
            "an automatic reclaim released the shared prefix its own capture extends");
    (void)finish_active(manager, program, active);
    require_shared_reuse(manager, program, 71, 4, true,
                         "the extended shared prefix lost reuse to its own conversation");
}

} // namespace

// Two private victims with different rebuild costs, and Host room for only one demotion: the
// portfolio objective (each checkpoint's rebuild_ns) must demote the costly victim and evict the
// cheap one, and when Host takes no demotion the one required eviction must fall on the cheap one.
void test_value_aware_pressure_demotes_high_value_victim_over_eviction() {
    using Planner                  = ninfer::runtime::MaterializationPlanner<FakeModelContract>;
    constexpr std::uint64_t ms     = 1'000'000;
    constexpr std::uint32_t weight = 4;
    constexpr std::uint64_t rebuild_high = 400 * ms;
    constexpr std::uint64_t rebuild_low  = 40 * ms;
    // Above both rebuild costs, so an evicted victim loses its full rebuild value in the
    // portfolio fold while a retained/demoted victim (zero restore cost) loses nothing.
    constexpr std::uint64_t recovery_cost = 1000 * ms;

    // Owner 1 carries the high re-prefill value; owner 2 the low value.
    const std::array<FakeContinuationHandle, 2> handles{
        FakeContinuationHandle{1, 0},
        FakeContinuationHandle{2, 0},
    };
    const std::array<const FakeContinuationHandle*, 2> owners{&handles[0], &handles[1]};
    const std::array<PlanningOwnerId, 2> ids{
        PlanningOwnerId{.value = 0},
        PlanningOwnerId{.value = 1},
    };
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 2> policies{
        ninfer::runtime::MaterializationOwnerPolicy{.owner                    = ids[0],
                                                    .private_retention_weight = weight},
        ninfer::runtime::MaterializationOwnerPolicy{.owner                    = ids[1],
                                                    .private_retention_weight = weight},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 2> checkpoints{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner       = ids[0],
            .checkpoint  = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                         .frontier = 16,
                                         .ordinal  = 0},
            .demand_mask = 1,
            .rebuild_ns  = rebuild_high},
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner       = ids[1],
            .checkpoint  = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                         .frontier = 16,
                                         .ordinal  = 0},
            .demand_mask = 1,
            .rebuild_ns  = rebuild_low},
    };
    // Each owner can demote-to-host (preserve, one relief unit) or evict-and-drop (one relief
    // unit). Both owners are touched under primary pressure; the host constraint decides the mix.
    const std::vector<std::pair<std::uint32_t, std::vector<FakeTargetDecision>>> owner_decisions{
        {1,
         {{.id = 1000U + 1, .immediate_ns = 0, .degradation_units = 1},
          {.id                  = 2000U + 1,
           .immediate_ns        = 0,
           .degradation_units   = 4,
           .dropped_checkpoints = 1,
           .evicts_continuation = true}}},
        {2,
         {{.id = 1000U + 2, .immediate_ns = 0, .degradation_units = 1},
          {.id                  = 2000U + 2,
           .immediate_ns        = 0,
           .degradation_units   = 4,
           .dropped_checkpoints = 1,
           .evicts_continuation = true}}},
    };
    const auto inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = owners,
            .private_owner_ids = ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = policies,
            .checkpoint_policy = checkpoints,
        };
    };

    // Primary: host room holds exactly one demote, so one victim must evict. The value-aware
    // objective must demote the high-value owner and evict the low-value owner.
    {
        FakeProgram program;
        program.required_pressure_actions       = 2;
        program.eviction_pressure_action_units  = 1;
        program.pressure_action_immediate_ns    = 0;
        program.pressure_checkpoint_recovery_ns = recovery_cost;
        program.owner_decisions                 = owner_decisions;

        FakeAdmissionCandidate root;
        set_fake_machine_costs(root.identity.machine_work, 100 * ms, 100 * ms);
        root.identity.physical_status = ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        root.identity.expandable      = true;
        root.identity.assessment_digest = 7;
        const std::array<Planner::CandidateInput, 1> candidates{
            Planner::CandidateInput{.candidate               = &root,
                                    .id                      = PlanningCandidateId{.value = 0},
                                    .stable_ordinal          = 0,
                                    .current_session_binding = false},
        };
        const auto goal = [](PlanningCandidateId, PrivateSourceMode,
                             std::span<const ninfer::runtime::PressureOwnerOutcome> outcomes)
            -> std::optional<Planner::LogicalGoal> {
            std::uint32_t evicted = 0;
            for (const auto& outcome : outcomes) {
                if (outcome.disposition == VictimDisposition::Evicted) { ++evicted; }
            }
            if (evicted == 0) { return std::nullopt; } // host can't hold every demote
            return Planner::LogicalGoal{.publication_slot = 0};
        };
        Planner planner;
        auto allowance     = ninfer::runtime::PlanningAllowance::boundary(0);
        allowance.limit_ns = 100 * ms;
        const auto result =
            planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0, inputs,
                         goal, Planner::Clock::now(), allowance);
        require(result && result->plan,
                "value-aware pressure search found no demote/evict closure");
        const auto& plan = *result->plan;
        require(plan.private_owner_ids.size() == 2 && plan.private_actions.size() == 2,
                "value-aware pressure closure did not act on both victims");
        std::uint64_t action_for_high = 0, action_for_low = 0;
        for (std::size_t index = 0; index < plan.private_owner_ids.size(); ++index) {
            if (plan.private_owner_ids[index] == 1) {
                action_for_high = plan.private_actions[index].id;
            }
            if (plan.private_owner_ids[index] == 2) {
                action_for_low = plan.private_actions[index].id;
            }
        }
        require(
            action_for_high == 1000U + 1,
            "planner evicted (or failed to demote) the HIGH-value victim instead of demoting it");
        require(action_for_low == 2000U + 2,
                "planner demoted the LOW-value victim and evicted a higher-value one");
        // The evicted victim is the low-value owner, so the value-aware cost is the candidate
        // immediate plus rebuild_low * retention_weight; the retained high-value owner keeps the
        // public portfolio value, so it adds no future loss.
        const std::uint64_t expected = 100 * ms + static_cast<std::uint64_t>(weight) * rebuild_low;
        require(result->diagnostics.predicted_total_ns == expected,
                "value-aware cost model mis-priced the demote-high/evict-low closure");
    }

    // Fallback: host is full (no demote successor), so the victim that must be sacrificed evicts.
    // With relief = 1 the cheaper (low-value) victim evicts, preserving the value-aware outcome.
    {
        FakeProgram program;
        program.required_pressure_actions       = 1;
        program.eviction_pressure_action_units  = 1;
        program.pressure_action_immediate_ns    = 0;
        program.pressure_checkpoint_recovery_ns = recovery_cost;
        program.owner_decisions                 = owner_decisions;

        FakeAdmissionCandidate root;
        set_fake_machine_costs(root.identity.machine_work, 100 * ms, 100 * ms);
        root.identity.physical_status = ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        root.identity.expandable      = true;
        root.identity.assessment_digest = 7;
        const std::array<Planner::CandidateInput, 1> candidates{
            Planner::CandidateInput{.candidate               = &root,
                                    .id                      = PlanningCandidateId{.value = 0},
                                    .stable_ordinal          = 0,
                                    .current_session_binding = false},
        };
        const auto goal = [](PlanningCandidateId, PrivateSourceMode,
                             std::span<const ninfer::runtime::PressureOwnerOutcome> outcomes)
            -> std::optional<Planner::LogicalGoal> {
            std::uint32_t evicted = 0;
            for (const auto& outcome : outcomes) {
                if (outcome.disposition == VictimDisposition::Evicted) { ++evicted; }
            }
            if (evicted == 0) { return std::nullopt; } // host full: a demote is impossible
            return Planner::LogicalGoal{.publication_slot = 0};
        };
        Planner planner;
        auto allowance     = ninfer::runtime::PlanningAllowance::boundary(0);
        allowance.limit_ns = 100 * ms;
        const auto result =
            planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0, inputs,
                         goal, Planner::Clock::now(), allowance);
        require(result && result->plan, "full-host pressure search found no evicting closure");
        const auto& plan = *result->plan;
        require(plan.private_owner_ids.size() == 1,
                "full-host closure evicted more than the one required victim");
        require(plan.private_owner_ids[0] == 2 && plan.private_actions[0].id == 2000U + 2,
                "full-host closure did not evict the cheaper (low-value) victim");
        // Same value-aware pricing as the primary closure: the sacrificed victim is the
        // low-value owner, so the cost is the candidate immediate plus rebuild_low * weight.
        const std::uint64_t expected = 100 * ms + static_cast<std::uint64_t>(weight) * rebuild_low;
        require(result->diagnostics.predicted_total_ns == expected,
                "full-host closure was not priced as a low-value eviction");
    }
}

// The ranking --value-aware-demote charges to an eviction: private victims ordered by rebuild
// cost from 0, shared victims 0, ties in victim order.
void test_value_weights_rank_private_victims_by_rebuild_cost() {
    using ninfer::models::qwen3_5::detail::value_weights_for_victims;

    // Costs: A=high, B=low, C=mid, D=shared. Expected weights: A=3, B=0, C=1, D=0.
    {
        const std::array<std::uint8_t, 4> is_shared = {0, 0, 0, 1};
        const std::array<std::uint64_t, 4> cost     = {9000, 100, 5000, 0};
        const auto weights                          = value_weights_for_victims(is_shared, cost);
        require(weights.size() == 4, "weight vector size mismatch");
        // 3 non-shared victims -> ranks are 0..2; the highest cost takes the top rank (2).
        require(weights[0] == 2, "high-cost private victim did not get the top rank");
        require(weights[1] == 0, "lowest-cost private victim did not get rank 0");
        require(weights[2] == 1, "mid-cost private victim did not get rank 1");
        require(weights[3] == 0, "shared victim must stay weight 0");
    }

    // All shared -> all weights 0 (nothing to rank).
    {
        const std::array<std::uint8_t, 2> is_shared = {1, 1};
        const std::array<std::uint64_t, 2> cost     = {123, 456};
        const auto weights                          = value_weights_for_victims(is_shared, cost);
        require(weights[0] == 0 && weights[1] == 0, "all-shared victims must all be weight 0");
    }

    // Ties are deterministic (the victim index breaks them): equal costs keep ascending order.
    {
        const std::array<std::uint8_t, 3> is_shared = {0, 0, 0};
        const std::array<std::uint64_t, 3> cost     = {7, 7, 7};
        const auto weights                          = value_weights_for_victims(is_shared, cost);
        require(weights[0] == 0 && weights[1] == 1 && weights[2] == 2,
                "equal-cost victims must keep deterministic ascending order");
    }

    // Empty: no victims, no weights.
    {
        const std::array<std::uint8_t, 0> is_shared = {};
        const std::array<std::uint64_t, 0> cost     = {};
        const auto weights                          = value_weights_for_victims(is_shared, cost);
        require(weights.empty(), "no victims -> no weights");
    }
}

int main() {
    TimerResolution timer_resolution;
    run_test("independent complete-target oracle",
             test_complete_search_against_small_exhaustive_oracle);
    run_test("publication-only construction",
             test_publication_only_pressure_constructs_adoptable_target);
    run_test("value-aware demotes high-value victim over eviction",
             test_value_aware_pressure_demotes_high_value_victim_over_eviction);
    run_test("value-aware ranking of private victims by rebuild cost",
             test_value_weights_rank_private_victims_by_rebuild_cost);
    run_test("private checkpoint identity loss",
             test_private_portfolio_loss_keeps_checkpoint_identity_fixed);
    run_test("unreachable checkpoint carries no value",
             test_unreachable_checkpoint_carries_no_value);
    run_test("portfolio demand and owner aggregation", test_portfolio_demand_and_owner_aggregation);
    run_test("shared capture private transition loss",
             test_shared_capture_subtracts_private_transition_loss);
    run_test("shared capture committed target budget",
             test_shared_capture_budget_bounds_committed_canonical_targets);
    run_test("equal lower-bound tie-break",
             test_equal_lower_bound_does_not_short_circuit_tie_break);
    run_test("machine cost is selection-only",
             test_machine_cost_changes_selection_without_changing_physical_assessment);
    run_test("candidate-stratified reuse closure",
             test_candidate_search_prefers_deep_reuse_without_eviction);
    run_test("feasible identity pressure improvement",
             test_feasible_identity_expands_when_pressure_can_remove_copy);
    run_test("dominating identity fast path",
             test_dominating_identity_does_not_build_pressure_graph);
    run_test("root lifecycle and prefix reuse", test_root_lifecycle_and_prefix_reuse);
    run_test("stale revision is retryable", test_stale_revision_is_retryable);
    run_test("materialization abort preserves source", test_materialization_abort_preserves_source);
    run_test("committed victim survives abort", test_committed_victim_survives_transaction_abort);
    run_test("uncommitted pressure acknowledgement",
             test_uncommitted_pressure_acknowledgement_is_not_degradation);
    run_test("aborted source is not a hit",
             test_aborted_source_selection_does_not_create_hit_history);
    run_test("auxiliary Host pressure", test_auxiliary_host_pressure_waits_without_rejecting_request);
    run_test("retained source protection", test_retained_source_is_protected_until_terminal);
    run_test("session publication order", test_session_publication_order_controls_tied_source);
    run_test("canonical pressure", test_canonical_pressure_starts_with_disposable_owner);
    run_test("all preserving pressure alternatives",
             test_pressure_tries_every_preserving_alternative_before_eviction);
    run_test("cumulative owner target",
             test_cumulative_owner_target_closes_pressure_without_eviction);
    run_test("joint two-owner pressure", test_two_owners_jointly_close_pressure);
    run_test("validate complete materialization result before adoption",
             test_materialization_result_is_validated_before_any_adoption);
    run_test("materialization result checkpoint identity",
             test_materialization_result_binds_exact_checkpoint_identity);
    run_test("materialization result owner identity",
             test_materialization_result_is_adopted_by_owner_identity);
    run_test("guided deep retention",
             test_guided_pressure_reaches_deep_retention_before_maximal_fallback);
    run_test("combined target exact repricing",
             test_combined_target_reprices_cancelled_pressure_copy);
    run_test("in-progress and capture", test_in_progress_adoption_and_private_capture);
    run_test("projected shared marginal value",
             test_projected_nested_shared_candidates_use_marginal_value);
    run_test("observed shared independent domains",
             test_observed_shared_candidate_requires_independent_domains);
    run_test("zero-prefill private promotion",
             test_repeated_private_reuse_selects_zero_prefill_shared_promotion);
    run_test("shared prefix after an exact repeat",
             test_shared_prefix_serves_a_new_question_after_an_exact_repeat);
    run_test("automatic shared capture LRU reclaim",
             test_automatic_shared_capture_reclaims_oldest_catalog_entry);
    run_test("explicit credit survives automatic reclaim",
             test_explicit_credit_shared_entry_survives_automatic_reclaim);
    run_test("shared fanout owner edges",
             test_shared_fanout_keeps_owner_edges_live_across_summary_refresh);
    run_test("shared capture multi-owner pressure",
             test_shared_capture_combines_two_pressure_owners);
    run_test("infeasible shared capture may reclaim without pressure evidence",
             test_infeasible_shared_capture_may_reclaim_without_pressure_evidence);
    run_test("aborted shared capture logical rollback",
             test_aborted_shared_capture_start_rolls_back_logical_claims);
    run_test("stale private reclamation by publication order",
             test_private_only_capture_reclaims_stale_resident_by_publication_order);
    run_test("stale reclamation prefers oldest publication order",
             test_stale_reclamation_prefers_the_oldest_publication_order);
    run_test("zero-value private capture skips the pressure search",
             test_zero_value_private_capture_skips_the_pressure_search);
    run_test("credited private capture keeps the pressure search",
             test_credited_private_capture_keeps_the_pressure_search);
    run_test("stale reclamation never takes a retained active source",
             test_stale_reclamation_never_takes_a_retained_active_source);
    run_test("stale reclamation is not run for a feasible capture",
             test_stale_reclamation_does_not_run_when_the_capture_is_feasible);
    run_test("stale reclamation leaves residents kept by admission pressure",
             test_stale_reclamation_leaves_residents_kept_by_admission_pressure);
    run_test("stale reclamation ranks by recency under recency eviction",
             test_stale_reclamation_ranks_by_recency_under_recency_eviction);
    run_test("validate complete capture result before adoption",
             test_capture_result_is_validated_before_any_adoption);
    run_test("capture result owner identity", test_capture_result_is_adopted_by_owner_identity);
    run_test("terminal fallback", test_terminal_fallback_releases_failed_retention);
    run_test("abort salvage publishes reusable endpoint",
             test_abort_salvage_publishes_reusable_endpoint);
    run_test("terminal fallback salvage", test_terminal_fallback_salvage_catalogues);
    run_test("abort salvage earns disposable retention",
             test_abort_salvage_earns_disposable_retention_under_pressure);
    run_test("full anchor set replaces the least-coverage anchor",
             test_full_anchor_set_replaces_the_least_coverage_anchor);
    run_test("terminal waits for resource transaction",
             test_terminal_settlement_waits_for_open_resource_transaction);
    run_test("commit and discard", test_commit_and_discard_terminal_states);
    run_test("backfill proof and stats", test_backfill_proof_and_stats_follow_program_revision);
    run_test("shortlist exact verification",
             test_shortlist_collision_requires_program_exact_verification);
    run_test("private owner recency ranking",
             test_rank_private_owners_by_recency_ranks_every_private_owner);
    run_test("publication counts as recency for the ladder",
             test_publication_counts_as_recency_for_the_ladder);
    run_test("incremental eviction is LRU-licensed",
             test_incremental_eviction_is_licensed_by_the_lru_tail);
    run_test("escape hatch sacrifices oldest first and demotes the rest",
             test_escape_hatch_sacrifices_oldest_first_and_demotes_the_rest);
    run_test("escape hatch ranks shared prefixes with private owners",
             test_escape_hatch_ranks_shared_prefixes_with_private_owners);
    run_test("shared capture seal failure skips instead of throwing",
             test_shared_capture_seal_failure_skips_instead_of_throwing);
    run_test("automatic reclaim picks the least recently used entry",
             test_automatic_reclaim_picks_the_least_recently_used_entry);
    run_test("lease reclaim releases least recent idle owners",
             test_lease_reclaim_releases_least_recent_idle_owners);
    run_test("superseded session turn ranks below its binding",
             test_superseded_session_turn_ranks_below_its_binding);
    run_test("automatic reclaim waits for a planned capture",
             test_automatic_reclaim_waits_for_a_planned_capture);
    run_test("automatic reclaim spares the prefix the capture extends",
             test_automatic_reclaim_spares_the_prefix_the_capture_extends);
    run_test("escape hatch clears all when nothing fits",
             test_escape_hatch_clears_all_when_nothing_fits);
    run_test("escape hatch ladder fits the committed target budget",
             test_preserved_prefix_ladder_targets_fit_the_committed_target_budget);
    if (failures != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
