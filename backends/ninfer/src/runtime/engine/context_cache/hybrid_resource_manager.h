#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/context_cache/materialization_budget.h"
#include "runtime/engine/context_cache/resource_manager.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::runtime {

// Engine-side driver of the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md §8).
// It presents the ResourceManager surface the Engine core uses, so scheduling, prefill, decode
// and commit orchestration are shared by both cache modes. Retention policy lives in the
// Program's prefix index; this class only tracks lane ownership and forwards admission and
// terminal settlement.
template <class ModelContract>
class HybridResourceManager {
public:
    using Program                      = typename ModelContract::Program;
    using PreparedPrompt               = typename ModelContract::PreparedPrompt;
    using RequestBasePlan              = typename ModelContract::RequestBasePlan;
    using PersistentBackfillProof      = typename ModelContract::PersistentBackfillProof;
    using SequenceHandle               = typename ModelContract::SequenceHandle;
    using CaptureOffer                 = typename ModelContract::CaptureOffer;
    using StartResult                  = typename ModelContract::StartResult;
    using FinishResult                 = typename ModelContract::FinishResult;
    using AbortResult                  = typename ModelContract::AbortResult;
    using Quote                        = typename ModelContract::HybridAdmissionQuote;
    using ProgramMaterializationResult = typename ModelContract::MaterializationResult;

    class Choice {
    public:
        Choice(Choice&&) noexcept        = default;
        Choice& operator=(Choice&&)      = delete;
        Choice(const Choice&)            = delete;
        Choice& operator=(const Choice&) = delete;

        [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return quote_.summary; }

        [[nodiscard]] LaneId destination() const noexcept { return quote_.destination; }

        [[nodiscard]] bool needs_transfer() const noexcept { return false; }

    private:
        explicit Choice(Quote&& quote) : quote_(std::move(quote)) {}

        Quote quote_;

        friend class HybridResourceManager;
    };

    class PublishedActivation {
    public:
        PublishedActivation(PublishedActivation&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), result_(std::move(other.result_)),
              destination_(other.destination_) {}

        PublishedActivation& operator=(PublishedActivation&&)      = delete;
        PublishedActivation(const PublishedActivation&)            = delete;
        PublishedActivation& operator=(const PublishedActivation&) = delete;

        [[nodiscard]] const SequenceHandle& sequence() const {
            if (!result_) { throw std::logic_error("published activation is empty"); }
            return result_->sequence;
        }

    private:
        PublishedActivation(HybridResourceManager& owner, StartResult&& result, LaneId destination)
            : owner_(&owner), result_(std::move(result)), destination_(destination) {}

        HybridResourceManager* owner_ = nullptr;
        std::optional<StartResult> result_;
        LaneId destination_{};

        friend class HybridResourceManager;
    };

    struct MaterializationOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
        std::optional<PublishedActivation> activation;
        MaterializationDiagnostics diagnostics;
        std::string failure;
    };

    enum class MaterializationReserveResult : std::uint8_t {
        Reserved,
        Stale,
        Aborted,
    };

    enum class ActiveCaptureReserveResult : std::uint8_t {
        Reserved,
        Skipped,
    };

    struct ActiveCaptureOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    };

    using ContextTransactionOutcome =
        std::variant<ContextTransactionInProgress, MaterializationOutcome, ActiveCaptureOutcome>;

    struct Inspection {
        Readiness readiness = Readiness::TemporarilyBlocked;
        std::optional<Choice> choice;
    };

    explicit HybridResourceManager(std::uint32_t lane_count) : lane_count_(lane_count) {
        if (lane_count == 0 || lane_count > kMaximumConcurrency) {
            throw std::invalid_argument("hybrid resource-manager lane count is invalid");
        }
    }

    [[nodiscard]] Inspection inspect(Program& program, const PreparedPrompt& prompt,
                                     const RequestBasePlan& base, std::uint64_t publication_order,
                                     PlanningAllowance /*allowance*/ = {}) {
        if (open_lane_ || program.has_context_transaction()) {
            return {.readiness = Readiness::TemporarilyBlocked};
        }
        if (publication_order == 0) {
            throw std::invalid_argument("request publication order is zero");
        }
        if (!program.isolated_request_feasible(base)) {
            return {.readiness = Readiness::PermanentlyInfeasible};
        }
        std::optional<LaneId> destination;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (lanes_[lane] == Lane::Free) {
                destination = LaneId{lane};
                break;
            }
        }
        if (!destination) { return {.readiness = Readiness::TemporarilyBlocked}; }
        Quote quote = program.hybrid_quote(prompt, base, *destination);
        if (quote.readiness != Readiness::Ready) { return {.readiness = quote.readiness}; }
        return {.readiness = Readiness::Ready, .choice = Choice(std::move(quote))};
    }

    // Backfill past a blocked head needs a persistent feasibility proof; the hybrid mode does
    // not issue one, so a blocked head is never overtaken.
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(Program&, const RequestBasePlan&, const Choice&,
                              std::span<const SequenceHandle>) const {
        return std::nullopt;
    }

    [[nodiscard]] MaterializationReserveResult
    reserve_materialization(Program& program, Choice&& choice, PreparedPrompt&& prompt,
                            CancellationFlagView cancellation) {
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("hybrid admission overlaps an open resource transaction");
        }
        const LaneId lane = choice.destination();
        require_lane(lane, Lane::Free);
        const ContextTransactionReserveStatus status = program.hybrid_reserve_materialization(
            std::move(choice.quote_), std::move(prompt), cancellation);
        if (status == ContextTransactionReserveStatus::Aborted) {
            return cancellation.requested() ? MaterializationReserveResult::Aborted
                                            : MaterializationReserveResult::Stale;
        }
        lanes_[lane.value] = Lane::Materializing;
        open_lane_         = lane;
        return MaterializationReserveResult::Reserved;
    }

    [[nodiscard]] std::optional<ContextTransactionKind> context_transaction_kind() const noexcept {
        if (open_lane_) { return ContextTransactionKind::Materialization; }
        return std::nullopt;
    }

    [[nodiscard]] ContextTransactionOutcome
    progress_context_transaction(Program& program, CancellationFlagView cancellation) {
        if (!open_lane_ || !program.has_context_transaction()) {
            throw std::logic_error("hybrid manager has no progressable admission");
        }
        auto progress = program.progress_context_transaction(cancellation);
        if (std::holds_alternative<ContextTransactionInProgress>(progress)) {
            return ContextTransactionInProgress{};
        }
        auto* result = std::get_if<ProgramMaterializationResult>(&progress);
        if (result == nullptr) {
            throw std::logic_error("hybrid admission returned a non-materialization result");
        }
        const LaneId lane = *open_lane_;
        MaterializationOutcome outcome;
        outcome.status      = result->status;
        outcome.diagnostics = result->diagnostics;
        outcome.failure     = result->failure;
        if (result->status == ContextTransactionStatus::Published) {
            if (!result->published) {
                throw std::logic_error("published hybrid admission has no sequence");
            }
            outcome.activation.emplace(
                PublishedActivation(*this, std::move(*result->published), lane));
            return outcome;
        }
        lanes_[lane.value] = Lane::Free;
        open_lane_.reset();
        return outcome;
    }

    void adopt(Program& program, PublishedActivation&& activation) noexcept {
        if (activation.owner_ != this || !activation.result_ || !open_lane_ ||
            activation.destination_.value != open_lane_->value ||
            lanes_[activation.destination_.value] != Lane::Materializing) {
            std::terminate();
        }
        lanes_[activation.destination_.value] = Lane::Active;
        activation.result_.reset();
        activation.owner_ = nullptr;
        open_lane_.reset();
        program.finalize_context_transaction();
    }

    // Hybrid prefill never offers captures; a stray offer is declined.
    [[nodiscard]] ActiveCaptureReserveResult reserve_active_capture(Program& program, LaneId,
                                                                    CaptureOffer&& offer,
                                                                    std::uint32_t,
                                                                    CancellationFlagView) {
        program.skip_capture(std::move(offer));
        return ActiveCaptureReserveResult::Skipped;
    }

    // An admission in progress already holds every Device page it needs (staged with the
    // quote), so active leases may reclaim unpinned cached blocks while its Host restore runs.
    std::uint32_t reclaim_device_kv_for_lease(Program& program, std::uint32_t main_pages,
                                              std::uint32_t backend_pages) {
        return program.hybrid_reclaim_device_kv(main_pages, backend_pages);
    }

    [[nodiscard]] FinishResult finish(Program& program, LaneId lane, SequenceHandle sequence) {
        require_lane(lane, Lane::TerminalPending);
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("terminal finish overlaps an open resource transaction");
        }
        FinishResult result = program.finish(sequence);
        if (result.status == ConsumeStatus::Consumed) {
            lanes_[lane.value] = Lane::Free;
            return result;
        }
        AbortResult discarded = program.abort(sequence);
        if (discarded.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program could neither publish nor discard a sequence");
        }
        lanes_[lane.value] = Lane::Free;
        FinishResult released;
        released.status      = ConsumeStatus::Consumed;
        released.disposition = FinishDisposition::Released;
        released.timings     = discarded.timings;
        released.speculative = std::move(discarded.speculative);
        return released;
    }

    [[nodiscard]] AbortResult abort(Program& program, LaneId lane, SequenceHandle sequence) {
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("terminal abort overlaps an open resource transaction");
        }
        if (lanes_.at(lane.value) != Lane::Active &&
            lanes_.at(lane.value) != Lane::TerminalPending) {
            throw std::logic_error("aborted lane has no active owner");
        }
        AbortResult result = program.abort(sequence);
        if (result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program did not consume aborted sequence");
        }
        lanes_[lane.value] = Lane::Free;
        return result;
    }

    void apply_commit(std::span<const LaneId> lanes,
                      const typename ModelContract::CommitResult& result) {
        if (lanes.size() != result.row_count) {
            throw std::logic_error("commit result membership is not row aligned");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const LaneId lane = lanes[row];
            require_lane(lane, Lane::Active);
            switch (result.rows[row].disposition) {
            case CommitDisposition::Active:
                break;
            case CommitDisposition::Finishable:
                lanes_[lane.value] = Lane::TerminalPending;
                break;
            case CommitDisposition::CancelledReleased:
                lanes_[lane.value] = Lane::Free;
                break;
            }
        }
    }

    void apply_discard(std::span<const LaneId> lanes,
                       const typename ModelContract::DiscardResult& result) {
        if (lanes.size() != result.row_count || result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("pending discard did not consume its membership");
        }
        for (const LaneId lane : lanes) { release_lane(lane); }
    }

    void release_failed_commit(std::span<const LaneId> lanes) noexcept {
        for (const LaneId lane : lanes) {
            if (lane.value < lane_count_ && (lanes_[lane.value] == Lane::Active ||
                                             lanes_[lane.value] == Lane::TerminalPending)) {
                lanes_[lane.value] = Lane::Free;
            }
        }
    }

    void populate_runtime_stats(Program& program, RuntimeStats& out) const noexcept {
        const auto usage                     = program.physical_usage();
        out.device_state_occupied_slots      = usage.device_state_slots;
        out.host_state_occupied_slots        = usage.host_state_slots;
        out.device_main_kv_occupied_pages    = usage.device_main_kv_pages;
        out.device_backend_kv_occupied_pages = usage.device_backend_kv_pages;
        out.device_main_kv_lease_pages       = usage.device_main_kv_lease_pages;
        out.device_backend_kv_lease_pages    = usage.device_backend_kv_lease_pages;
        out.host_kv_occupied_bytes           = usage.host_kv_bytes;
        const auto hybrid                    = program.hybrid_stats();
        out.hybrid_cached_blocks             = hybrid.device_resident_blocks;
        out.hybrid_evictable_blocks          = hybrid.device_evictable_blocks;
        out.hybrid_tree_blocks               = hybrid.nodes;
        out.hybrid_snapshots                 = hybrid.snapshots;
        out.hybrid_host_capacity_bytes       = hybrid.host_slab_bytes * hybrid.host_slabs;
        out.hybrid_host_used_bytes =
            hybrid.host_slab_bytes * (hybrid.host_slabs - hybrid.host_free_slabs);
        out.host_kv_occupied_bytes         = out.hybrid_host_used_bytes;
        out.hybrid_snapshot_hits           = hybrid.snapshot_hits;
        out.hybrid_reused_tokens           = hybrid.reused_tokens;
        out.hybrid_blocks_inserted         = hybrid.blocks_inserted;
        out.hybrid_blocks_reattached       = hybrid.blocks_reattached;
        out.hybrid_blocks_duplicate        = hybrid.blocks_duplicate;
        out.hybrid_taps_created            = hybrid.taps_created;
        out.hybrid_taps_skipped            = hybrid.taps_skipped;
        out.hybrid_endpoints_created       = hybrid.endpoints_created;
        out.hybrid_host_image_writes       = hybrid.host_image_writes;
        out.hybrid_host_block_writes       = hybrid.host_block_writes;
        out.hybrid_host_image_restores     = hybrid.host_image_restores;
        out.hybrid_host_block_restores     = hybrid.host_block_restores;
        out.hybrid_host_write_bytes        = hybrid.host_write_bytes;
        out.hybrid_host_restore_bytes      = hybrid.host_restore_bytes;
        out.hybrid_evicted_blocks          = hybrid.evicted_blocks;
        out.hybrid_host_snapshot_evictions = hybrid.host_snapshot_evictions;
        out.hybrid_host_dead_reclaims      = hybrid.host_dead_reclaims;
        out.hybrid_unbacked_node_losses    = hybrid.unbacked_node_losses;
    }

    [[nodiscard]] LogicalLaneState lane_state(LaneId lane) const noexcept {
        return lane.value < lane_count_ ? lanes_[lane.value] : LogicalLaneState::Free;
    }

    void clear_after_program_cleanup() noexcept {
        open_lane_.reset();
        lanes_.fill(Lane::Free);
    }

private:
    using Lane = LogicalLaneState;

    void require_lane(LaneId lane, Lane expected) const {
        if (lane.value >= lane_count_ || lanes_[lane.value] != expected) {
            throw std::logic_error("hybrid lane is not in its expected ownership state");
        }
    }

    void release_lane(LaneId lane) {
        if (lane.value >= lane_count_ ||
            (lanes_[lane.value] != Lane::Active && lanes_[lane.value] != Lane::TerminalPending)) {
            throw std::logic_error("released lane has no active owner");
        }
        lanes_[lane.value] = Lane::Free;
    }

    std::uint32_t lane_count_ = 0;
    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::optional<LaneId> open_lane_;
};

} // namespace ninfer::runtime
