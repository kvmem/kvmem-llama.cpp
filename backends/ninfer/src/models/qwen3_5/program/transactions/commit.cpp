#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/planning/rebuild_work.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::models::qwen3_5::detail {

PendingBatch ProgramImpl::wrap_pending(std::span<const std::uint32_t> lanes,
                                       const runtime::BatchedGeneratedRound& round) {
    if (pending_transaction_ || lanes.empty() || lanes.size() > max_concurrency) {
        throw std::logic_error("Program already owns a pending transaction");
    }
    PendingTransaction transaction;
    transaction.id   = next_transaction_id_++;
    transaction.size = lanes.size();
    std::array<SequenceHandle, kMaximumConcurrency> handles{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("pending transaction membership is invalid");
        }
        transaction.lanes[row]  = lane;
        transaction.epochs[row] = lane_epochs[lane];
        handles[row] =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
    }
    pending_transaction_ = transaction;
    return ContractAccess::make_pending(
        this, transaction.id, std::span<const SequenceHandle>(handles.data(), lanes.size()),
        round.tokens, round.row_counts, round.row_stride, round.timing);
}

PrefillProgress ProgramImpl::wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step) {
    PrefillProgress out;
    out.summary                 = step.summary;
    out.processed_prompt_tokens = step.processed_prompt_tokens;
    out.complete                = step.complete;
    out.timing                  = step.timing;
    out.first_token_logprobs    = std::move(step.first_token_logprobs);
    if (step.complete) {
        const std::array<std::uint32_t, 1> lanes{lane};
        const runtime::BatchedGeneratedRound round{
            .tokens     = step.round.tokens,
            .row_counts = {},
            .row_stride = 1,
        };
        out.pending.emplace(wrap_pending(lanes, round));
    } else if (requests[lane].prefill && requests[lane].prefill->pending_capture_offer != 0) {
        out.capture.emplace(
            ContractAccess::make_capture_offer(this, runtime::LaneId{lane}, lane_epochs[lane],
                                               requests[lane].prefill->pending_capture_offer));
    }
    return out;
}

StartResult ProgramImpl::start_request(MaterializationTransaction& transaction) {
    std::optional<std::uint32_t> destination = transaction.destination.value;
    std::optional<std::uint32_t> continuation_index;
    try {
        if (!transaction.prepared || !transaction.plan || !destination ||
            *destination >= max_concurrency) {
            throw std::invalid_argument("materialization transaction is not publishable");
        }
        const std::uint32_t lane              = *destination;
        const AdmissionCandidateImpl& details = *transaction.plan->impl_;
        if (details.destination_epoch != lane_epochs[lane] ||
            details.has_source != transaction.has_source ||
            details.has_shared_source != transaction.has_shared_source) {
            throw std::logic_error("admission plan physical epoch is stale");
        }
        if (requests[lane].lifecycle != Lifecycle::Empty ||
            active_continuations[lane] < continuation_capacity) {
            throw std::logic_error("admission destination is not free");
        }
        if (transaction.has_source &&
            transaction.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            if (transaction.source_index >= continuation_capacity ||
                continuation_slots[transaction.source_index].role !=
                    ContinuationSlotRole::Catalogued ||
                continuation_slots[transaction.source_index].generation !=
                    transaction.source_generation ||
                transaction.source_index != details.source_index ||
                transaction.source_generation != details.source_generation) {
                throw std::logic_error("admission source capability is stale");
            }
            continuation_index                           = transaction.source_index;
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        } else {
            continuation_index = transaction.root_continuation_index;
            if (!continuation_index || transaction.root_waiting_for_victim ||
                continuation_slots[*continuation_index].role !=
                    ContinuationSlotRole::ReservedMaterialization) {
                throw std::logic_error("materialization continuation reservation is unavailable");
            }
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        }

        const detail::PhysicalResources active = details.demand.active_entitlement;
        // Consuming or recycling the source (truncating it in place, releasing it after its COW)
        // can leave pages it shared with another active sequence referenced by that sequence alone.
        const ActiveExclusiveBaseline baseline = active_exclusive_baseline();
        active_continuations[lane]             = *continuation_index;
        SequenceState& sequence                = continuation_states[*continuation_index];
        sequence.lane                          = lane;
        transaction.root_continuation_index.reset();
        start_sequence(lane, sequence, transaction);
        credit_active_ownership_transfers(baseline);
        detail::PhysicalResources actual         = owner_exclusive_resources(sequence);
        actual.device.active_lanes               = 1;
        const detail::PhysicalResources expected = active;
        if (actual != expected) {
            throw std::logic_error("materialized sequence does not match its active entitlement");
        }
        if (details.reuse != ReusePath::Root) {
            if (transaction.state_restored) {
                ++transaction.operations.state_restores;
            } else if (details.source_mode == runtime::PrivateSourceMode::Retain ||
                       transaction.has_shared_source || details.state_fork_required) {
                ++transaction.operations.state_forks;
                ++transaction.operations.historical_fork_hits;
            } else {
                ++transaction.operations.state_moves;
            }
        }
        requests[lane].active_resources   = active;
        requests[lane].optional_resources = details.active_optional_resources;
        invalidate_lane(lane);
        const SequenceHandle handle =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
        return StartResult{.sequence = handle};
    } catch (...) {
        if (destination && *destination < max_concurrency) {
            const std::uint32_t lane = *destination;
            if (active_continuations[lane] < continuation_capacity) {
                clear_lane_best_effort(active_sequence(lane), requests[lane]);
            } else if (continuation_index) {
                release_continuation_slot_best_effort(*continuation_index);
            }
            invalidate_lane(*destination);
        }
        throw;
    }
}

PendingBatch ProgramImpl::decode(std::span<const SequenceHandle> members,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        budgets.size() != members.size()) {
        throw std::invalid_argument("decode membership is invalid");
    }
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("decode sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("decode membership is duplicate or not active");
        }
        lanes[row] = lane;
    }
    const auto lane_span = std::span<const std::uint32_t>(lanes.data(), members.size());
    try {
        runtime::BatchedGeneratedRound round = decode_raw(lane_span, budgets, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += round.timing; }
        return wrap_pending(lane_span, std::move(round));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(lane_span);
        pending_transaction_.reset();
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

// Begin and ordinary rounds may already have provisional identity through the accepted extent;
// speculative and forced spans arrive with identity at their base. Both are Program-owned pending
// states, and this is their single accepted-prefix identity commit.
void ProgramImpl::commit_generated_prefix_identity(
    SequenceState& sequence, std::uint32_t base_ledger_frontier,
    std::span<const TokenId> accepted_tokens,
    std::optional<std::uint32_t> prefix_execution_split_after) {
    if (base_ledger_frontier > sequence.ledger.size() ||
        accepted_tokens.size() > sequence.ledger.size() - base_ledger_frontier ||
        sequence.ledger.size() != base_ledger_frontier + accepted_tokens.size() ||
        !std::equal(accepted_tokens.begin(), accepted_tokens.end(),
                    sequence.ledger.begin() + static_cast<std::ptrdiff_t>(base_ledger_frontier)) ||
        (prefix_execution_split_after &&
         (*prefix_execution_split_after == 0 ||
          *prefix_execution_split_after > accepted_tokens.size()))) {
        throw std::logic_error("committed generated-prefix identity has an invalid span");
    }
    const bool already_appended = sequence.prefix_identity.size() == sequence.ledger.size() &&
                                  sequence.prefix_digests.size() == sequence.ledger.size();
    const bool awaits_append = sequence.prefix_identity.size() == base_ledger_frontier &&
                               sequence.prefix_digests.size() == base_ledger_frontier;
    if (!already_appended && !awaits_append) {
        throw std::logic_error("generated-prefix identity is not at its base or committed extent");
    }
    if (already_appended && !prefix_execution_split_after) { return; }
    sequence.prefix_identity.truncate(base_ledger_frontier);
    sequence.prefix_digests.truncate(base_ledger_frontier);
    sequence.prefix_identity.append_generated(accepted_tokens.size(), sequence.rope_delta,
                                              prefix_execution_split_after);
    sequence.prefix_digests.append_generated(accepted_tokens, sequence.rope_delta,
                                             prefix_execution_split_after);
    if (sequence.prefix_identity.size() != sequence.ledger.size() ||
        sequence.prefix_digests.size() != sequence.ledger.size()) {
        throw std::logic_error("committed generated-prefix identity changed the ledger shape");
    }
}

runtime::ExecutionTiming ProgramImpl::append_forced_tokens(
    std::span<const SequenceHandle> members, std::span<const TokenId> row_major_tokens,
    std::uint32_t row_stride, std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        row_stride == 0 || prefix_execution_splits.size() != members.size() ||
        row_major_tokens.size() != static_cast<std::size_t>(row_stride) * members.size()) {
        throw std::invalid_argument("forced-token membership is invalid");
    }

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("forced-token sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("forced-token membership is duplicate or not active");
        }
        const SequenceState& sequence = active_sequence(lane);
        if (sequence.execution_frontier == std::numeric_limits<std::uint32_t>::max() ||
            sequence.ledger_frontier != sequence.execution_frontier + 1U ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != sequence.execution_frontier) ||
            (is_masked_draft_backend(speculative_backend) &&
             sequence.dflash_context_frontier > sequence.execution_frontier) ||
            static_cast<std::uint64_t>(sequence.execution_frontier) + row_stride > capacity) {
            throw std::logic_error("forced-token sequence frontier is invalid");
        }
        validate_licensed_tokens(row_major_tokens.subspan(row * row_stride, row_stride));
        if (prefix_execution_splits[row] &&
            (*prefix_execution_splits[row] == 0 || *prefix_execution_splits[row] > row_stride)) {
            throw std::logic_error("forced-token execution split is outside its row");
        }
        lanes[row] = lane;
    }

    const bool count_forced_tokens = std::any_of(
        lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(members.size()),
        [&](std::uint32_t lane) { return requests[lane].sampling_host.token_counts != nullptr; });
    if (count_forced_tokens) {
        work.reset();
        Tensor forced_ids =
            work.alloc(DType::I32, {checked_i32(static_cast<std::uint32_t>(row_major_tokens.size()),
                                                "forced-token batch exceeds int32")});
        CUDA_CHECK(cudaMemcpyAsync(forced_ids.data, row_major_tokens.data(), forced_ids.bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        for (std::size_t row = 0; row < members.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (requests[lane].sampling_host.token_counts == nullptr) { continue; }
            Tensor ids    = forced_ids.slice(0, static_cast<std::int32_t>(row * row_stride),
                                             static_cast<std::int32_t>(row_stride));
            Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(lane), 1)
                                .view({dimension(parameters.model.resources().public_token_count)});
            ops::increment_token_counts(ids, counts, device.stream);
        }
        work.reset();
    }

    try {
        for (std::size_t row = 0; row < members.size(); ++row) {
            timing.resume_submit();
            const std::uint32_t lane = lanes[row];
            SequenceState& sequence  = active_sequence(lane);
            RequestControl& request  = requests[lane];
            const std::span<const TokenId> forced =
                row_major_tokens.subspan(row * row_stride, row_stride);
            const std::uint32_t base_ledger_frontier = sequence.ledger_frontier;
            const std::uint32_t base                 = sequence.execution_frontier;
            const std::uint32_t end                  = base + row_stride;
            const auto started                       = Clock::now();

            if (is_masked_draft_backend(speculative_backend) &&
                sequence.dflash_context_frontier < base) {
                const std::array<std::uint32_t, 1> append_lanes{lane};
                const std::array<std::uint32_t, 1> append_starts{sequence.dflash_context_frontier};
                const std::array<std::uint32_t, 1> append_counts{base -
                                                                 sequence.dflash_context_frontier};
                enqueue_dflash_context_append(append_lanes, append_starts, append_counts);
                timing.begin_wait();
                device.synchronize();
                timing.end_wait();
                sequence.dflash_context_frontier = base;
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                work.reset();
                timing.resume_submit();
            }

            if (!kvmem_window_tokens) ensure_sequence_kv_mapped(sequence, end, backend_kv_cache() ? end : 0U);
            // Forced tokens run through Prefill, which reads the shared step table-row scalars.
            bind_sequence_kv(sequence);

            sequence.ledger.insert(sequence.ledger.end(), forced.begin(), forced.end());
            if (sequence.ledger.size() != static_cast<std::size_t>(end) + 1U) {
                throw std::logic_error("forced-token continuation ledger has an invalid shape");
            }

            if (is_masked_draft_backend(speculative_backend)) {
                if (!dflash || !io.dflash_decode || !sequence.kv ||
                    (backend_kv_cache() && !sequence.kv->backend)) {
                    throw std::logic_error("DFlash forced continuation state is incomplete");
                }
                upload_dflash_prefill_controls(sequence);
            }

            std::uint32_t cursor = base;
            while (cursor < end) {
                const std::uint32_t count           = std::min(prefill_chunk, end - cursor);
                if (kvmem_window_tokens) prepare_window(sequence, cursor, cursor + count);
                const StateImageSelectors selectors = state_selectors(sequence);
                execution::PrefillContext schedule_state{
                    {device, parameters, work, state_images->linear(0),
                     replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
                     proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel},
                    text_kv_view(sequence),
                    mtp_kv_view(sequence),
                    decoder->text_kv,
                    decoder->mtp_cache(),
                    dflash ? &*dflash : nullptr,
                    cursor,
                    nullptr,
                    nullptr,
                    selectors.source,
                    selectors.destination,
                    0,
                    dflash_host_ingress};
                schedule_state.cache_position_shift = sequence.window.removed_tokens;
                schedule_state.execution.memory_statistics = memory_statistics ? &*memory_statistics : nullptr;
                mark_workspace_usage(speculative_backend == SpeculativeBackend::Mtp
                                         ? workspace_plan.mtp_prefill
                                         : workspace_plan.text_prefill);
                if (is_masked_draft_backend(speculative_backend)) {
                    mark_workspace_usage(workspace_plan.dflash_context);
                }
                const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                    schedule_state, sequence.ledger, count, std::nullopt, false);
                if (result.finalized || result.processed_tokens == 0 ||
                    result.processed_tokens > count) {
                    throw std::logic_error("forced-token prefill made invalid progress");
                }
                commit_memory_statistics(sequence, cursor, cursor + result.processed_tokens);
                cursor += result.processed_tokens;
                sequence.text_kv_valid = cursor;
                if (speculative_backend == SpeculativeBackend::Mtp) {
                    sequence.mtp_kv_valid = cursor;
                } else if (is_masked_draft_backend(speculative_backend)) {
                    sequence.dflash_context_frontier = cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                settle_state_fork(sequence);
                copy_tail(sequence,
                          prefill_hidden.slice(
                              1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
            }
            timing.begin_wait();
            device.synchronize();
            timing.end_wait();
            work.reset();

            commit_generated_prefix_identity(sequence, base_ledger_frontier, forced,
                                             prefix_execution_splits[row]);
            advance_rebuild_work(sequence, end, prefill_chunk);
            sequence.execution_frontier = end;
            sequence.ledger_frontier    = end + 1U;
            sequence.mtp_draft_count    = 0;
            sequence.tail_hidden_valid  = true;
            if (sequence.ledger.size() != sequence.ledger_frontier ||
                sequence.prefix_identity.size() != sequence.ledger_frontier ||
                sequence.prefix_digests.size() != sequence.ledger_frontier ||
                sequence.ledger.back() != forced.back()) {
                throw std::logic_error("forced-token commit did not establish a valid frontier");
            }
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            request.timings.decode_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        return timing.finish();
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        work.reset();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(lanes.data(), members.size()));
        throw;
    }
}

CommitResult ProgramImpl::commit(PendingBatch&& pending,
                                 std::span<const runtime::CommitDecision> decisions,
                                 runtime::CommitObservation observation,
                                 runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    const auto input_rows       = ContractAccess::rows(pending);
    const std::size_t row_count = input_rows.size();
    for (std::size_t row = 0; row < row_count; ++row) { members[row] = input_rows[row]; }
    const bool valid = valid_pending(pending);
    ContractAccess::consume(pending);

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::array<GenerationTimings, kMaximumConcurrency> timings{};
    std::array<SpeculativeStats, kMaximumConcurrency> speculative{};
    std::array<PendingKind, kMaximumConcurrency> pending_kinds{};
    const auto release_members = [&]() noexcept {
        std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
        std::size_t failed_count = 0;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (ContractAccess::owner(members[row]) != this) { continue; }
            const std::uint32_t lane = ContractAccess::lane(members[row]).value;
            if (lane >= max_concurrency) { continue; }
            failed_lanes[failed_count++] = lane;
        }
        clear_execution_failure_lanes(
            std::span<const std::uint32_t>(failed_lanes.data(), failed_count));
        pending_transaction_.reset();
    };

    try {
        if (!valid || row_count == 0 || row_count > max_concurrency ||
            decisions.size() != row_count) {
            throw std::logic_error("pending transaction capability or decision shape is invalid");
        }
        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane                = ContractAccess::lane(members[row]).value;
            lanes[row]                              = lane;
            const PendingCandidate& candidate       = requests[lane].pending;
            pending_kinds[row]                      = candidate.kind;
            const runtime::CommitDecision& decision = decisions[row];
            if (decision.cancelled && has_context_transaction()) {
                throw std::logic_error(
                    "active cancellation overlaps the global context transaction");
            }
            if ((decision.cancelled && (decision.accepted_tokens != 0 || !decision.terminal)) ||
                (!decision.cancelled &&
                 (decision.accepted_tokens == 0 || decision.accepted_tokens > candidate.produced ||
                  (!decision.terminal && decision.accepted_tokens != candidate.produced))) ||
                (decision.prefix_execution_split_after &&
                 (decision.cancelled || *decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("pending transaction decision is invalid");
            }
            accepted[row]                = decision.accepted_tokens;
            terminal[row]                = decision.terminal ? 1U : 0U;
            cancelled[row]               = decision.cancelled ? 1U : 0U;
            prefix_execution_splits[row] = decision.prefix_execution_split_after;
            if (decision.cancelled) {
                timings[row]     = requests[lane].timings;
                speculative[row] = std::move(requests[lane].speculative_stats);
            }
        }

        timing.pause();
        timing.include(
            resolve_pending_raw(std::span<const std::uint32_t>(lanes.data(), row_count),
                                std::span<const std::uint32_t>(accepted.data(), row_count),
                                std::span<const std::uint8_t>(terminal.data(), row_count),
                                std::span<const std::uint8_t>(cancelled.data(), row_count),
                                std::span<const std::optional<std::uint32_t>>(
                                    prefix_execution_splits.data(), row_count),
                                failed_timing));
        timing.resume_post();
        pending_transaction_.reset();

        CommitResult out;
        out.row_count          = row_count;
        bool released_resource = false;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (decisions[row].cancelled) {
                invalidate_lane(lanes[row]);
                released_resource = true;
                out.rows[row]     = CommitRowResult{
                        .disposition = runtime::CommitDisposition::CancelledReleased,
                        .timings     = timings[row],
                        .speculative = std::move(speculative[row]),
                };
            } else if (decisions[row].terminal) {
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            }

            if (pending_kinds[row] != PendingKind::Begin || decisions[row].cancelled) { continue; }
            RequestControl& request = requests[lanes[row]];
            if (decisions[row].terminal) {
                request.prefill.reset();
                continue;
            }
            if (!request.prefill) { continue; }
            RequestControl::Prefill& prefill = *request.prefill;
            if (prefill.cursor != prefill.prompt_tokens ||
                prefill.next_capture >= prefill.capture_groups.size() ||
                prefill.capture_groups[prefill.next_capture].frontier != prefill.prompt_tokens ||
                prefill.pending_capture_offer != 0) {
                throw std::logic_error("prompt-frontier capture carrier is inconsistent");
            }
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            prefill.pending_capture_offer = next_capture_offer_id_;
            out.captures[row].emplace(ContractAccess::make_capture_offer(
                this, runtime::LaneId{lanes[row]}, lane_epochs[lanes[row]],
                prefill.pending_capture_offer));
        }
        if (released_resource) { advance_resource_revision(); }
        out.timing = timing.finish();
        return out;
    } catch (...) {
        timing.resume_post();
        release_members();
        throw;
    }
}

DiscardResult ProgramImpl::abort_pending(PendingBatch&& pending) noexcept {
    DiscardResult out;
    const auto rows  = ContractAccess::rows(pending);
    const bool valid = valid_pending(pending);
    out.row_count    = std::min<std::size_t>(rows.size(), kMaximumConcurrency);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    for (std::size_t row = 0; row < out.row_count; ++row) { members[row] = rows[row]; }
    ContractAccess::consume(pending);
    if (!valid) { return out; }
    std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
    for (std::size_t row = 0; row < out.row_count; ++row) {
        failed_lanes[row] = ContractAccess::lane(members[row]).value;
    }
    const bool deferred_to_fail_all = has_context_transaction();
    clear_execution_failure_lanes(
        std::span<const std::uint32_t>(failed_lanes.data(), out.row_count));
    pending_transaction_.reset();
    if (deferred_to_fail_all) { return out; }
    if (out.row_count != 0) { advance_resource_revision(); }
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

FinishResult ProgramImpl::finish(SequenceHandle sequence) noexcept {
    FinishResult out;
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane               = ContractAccess::lane(sequence).value;
    RequestControl& request                = requests[lane];
    SequenceState& state                   = active_sequence(lane);
    const std::uint32_t continuation_index = active_continuations[lane];
    if (request.lifecycle != Lifecycle::Finishable) { return out; }
    if (hybrid_) {
        // Hybrid mode retains context in the prefix index, never as a catalogued continuation.
        out.timings     = request.timings;
        out.speculative = std::move(request.speculative_stats);
        if (!hybrid_finish_lane(state, request, lane, true)) { return out; }
        out.disposition = runtime::FinishDisposition::Released;
        invalidate_lane(lane);
        advance_resource_revision();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }
    if (!request.publish_continuation) {
        retain_memory_history(state, request);
        if (!clear_lane_strict(state, request)) { return out; }
        out.disposition = runtime::FinishDisposition::Released;
        out.timings     = request.timings;
        out.speculative = std::move(request.speculative_stats);
        invalidate_lane(lane);
        advance_resource_revision();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }
    // Every valid terminal execution path settles a borrowed materialization Fork first. A
    // borrowed source cannot become this continuation's direct endpoint; fall back to terminal
    // discard if that publication invariant was not established.
    if (state.state.fork_pending && state.state.borrows_read()) { return out; }
    try {
        if (state.state.fork_pending) {
            const StateImageHandle source      = state.state.read;
            const StateImageHandle destination = state.state.write;
            state_store->abort_fork(source, destination);
            if (!state_store->release(destination)) { return out; }
            // An active-capture source is still this sequence's primary lifetime. Publishing it
            // as the endpoint retains that direct ownership; surviving checkpoint references
            // still prevent exclusive attribution and release.
            state.state = ActiveStateBinding{.read = source, .write = source};
        }
    } catch (...) { return out; }
    if (!publish_active_continuation(state, request, lane, continuation_index, out.summary)) {
        return out;
    }
    out.continuation.emplace(ContractAccess::make_continuation(
        this, continuation_index, continuation_slots[continuation_index].generation));
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    out.disposition = runtime::FinishDisposition::Catalogued;
    out.status      = runtime::ConsumeStatus::Consumed;
    return out;
}

bool ProgramImpl::publish_active_continuation(SequenceState& state, RequestControl& request,
                                              std::uint32_t lane, std::uint32_t continuation_index,
                                              qwen3_5::ContinuationSummary& summary) noexcept {
    try {
        summary.long_anchors.reserve(state.long_anchors.size());
    } catch (...) { return false; }
    try {
        if (state.reserved_state) {
            if (!state_store->release(*state.reserved_state)) { return false; }
            state.reserved_state.reset();
        }
        if (state.rewrite_state && *state.rewrite_state == state.state.read) {
            if (state_store->checkpoint_references(*state.rewrite_state) == 0) { return false; }
            state_store->release_checkpoint_reference(*state.rewrite_state);
            state.rewrite_state.reset();
            state.rewrite_checkpoint = {};
        }
        if (state_store->role(state.state.read) == StateImageRole::ActiveMutable) {
            state_store->freeze(state.state.read);
        } else if (state_store->role(state.state.read) != StateImageRole::CheckpointImmutable) {
            return false;
        }
        state.endpoint_valid = true;
        refresh_state_views(state);
        text_kv_addresses->set_checkpoint_requirement(state.kv->text, state.execution_frontier);
        if (state.kv->backend) {
            backend_kv_addresses->set_checkpoint_requirement(*state.kv->backend,
                                                             backend_kv_valid(state));
        }
        populate_continuation_summary(state, summary);
        summary.active_references = 0;
    } catch (...) { return false; }
    release_active_shared_references(state);
    release_sequence_growth_entitlement(state);
    unbind_sequence_kv(state);
    request.active_resources                    = {};
    request.optional_resources                  = {};
    // A published lane is free: any staged prefill bookkeeping belongs to the
    // finished request. The abort/salvage path publishes without going through
    // the commit decision loop that clears it on normal completion, so the
    // release itself must leave none behind.
    request.prefill.reset();
    request.lifecycle                           = Lifecycle::Empty;
    request.pending                             = {};
    continuation_slots[continuation_index].role = ContinuationSlotRole::Catalogued;
    active_continuations[lane]                  = continuation_capacity;
    invalidate_lane(lane);
    advance_resource_revision();
    return true;
}

bool ProgramImpl::salvage_continuation(SequenceState& state, RequestControl& request,
                                       std::uint32_t lane, std::uint32_t continuation_index,
                                       qwen3_5::AbortResult& out) noexcept {
    // Salvage publishes the in-place active state as the continuation endpoint. A pending
    // materialization Fork is skipped: its read source belongs to an external owner and the
    // destination copy duplicates a checkpoint the catalog already holds.
    if (!request.publish_continuation || state.state.fork_pending ||
        state.state.read != state.state.write || !state.kv) {
        return false;
    }
    const Lifecycle lifecycle = request.lifecycle;
    std::uint32_t frontier    = 0;
    if (lifecycle == Lifecycle::Prefilling) {
        if (!request.prefill || request.prefill->pending_capture_offer != 0) { return false; }
        frontier = request.prefill->cursor;
        if (frontier < kSalvageMinFrontier || state.text_kv_valid != frontier ||
            frontier > request.prefill->prompt_tokens) {
            return false;
        }
        // Staged prefill keeps the committed frontier and rebuild work at the materialization
        // base; bring both to the salvaged cursor before publishing.
        try {
            state.rebuild_work = rebuild_work_at_frontier(
                request.prefill->prompt, frontier, prefill_chunk, request.prefill->capture_groups,
                request.prefill->prompt.identity.rewrite_execution_frontiers);
            std::uint32_t tail_begin = 0;
            for (const CaptureGroup& group : request.prefill->capture_groups) {
                runtime_support::include_rebuild_boundary(tail_begin, group.frontier, frontier);
            }
            for (const std::uint32_t boundary :
                 request.prefill->prompt.identity.rewrite_execution_frontiers) {
                runtime_support::include_rebuild_boundary(tail_begin, boundary, frontier);
            }
            state.rebuild_tail_begin = tail_begin;
            state.execution_frontier = frontier;
        } catch (...) { return false; }
        // A DFlash draft context that lags the salvaged frontier cannot be truncated to it on
        // materialization, so only a caught-up backend can be published.
        if (speculative_backend == SpeculativeBackend::DFlash &&
            state.dflash_context_frontier < frontier) {
            return false;
        }
    } else if (lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Finishable) {
        frontier = state.execution_frontier;
        if (frontier < kSalvageMinFrontier || state.text_kv_valid != frontier) { return false; }
        if (speculative_backend == SpeculativeBackend::Mtp) {
            if (state.mtp_kv_valid + 1 < frontier) { return false; }
        } else if (speculative_backend == SpeculativeBackend::DFlash) {
            if (state.dflash_context_frontier < frontier) { return false; }
        }
    } else {
        return false;
    }
    if (!publish_active_continuation(state, request, lane, continuation_index, out.summary)) {
        return false;
    }
    out.continuation.emplace(ContractAccess::make_continuation(
        this, continuation_index, continuation_slots[continuation_index].generation));
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    out.salvaged    = true;
    return true;
}

AbortResult ProgramImpl::abort(SequenceHandle sequence) noexcept {
    AbortResult out;
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    RequestControl& request  = requests[lane];
    if (request.lifecycle == Lifecycle::Pending || request.lifecycle == Lifecycle::Empty) {
        return out;
    }
    SequenceState& state = active_sequence(lane);
    const std::uint32_t continuation_index = active_continuations[lane];
    if (hybrid_) {
        // The committed state is publishable as an endpoint when no model unit is in flight.
        out.timings     = request.timings;
        out.speculative = std::move(request.speculative_stats);
        const bool consistent =
            (request.lifecycle == Lifecycle::Active || request.lifecycle == Lifecycle::Finishable ||
             (request.lifecycle == Lifecycle::Prefilling && request.prefill &&
              state.text_kv_valid == request.prefill->cursor)) &&
            !state.state.fork_pending;
        if (!hybrid_finish_lane(state, request, lane, consistent)) { return out; }
        invalidate_lane(lane);
        advance_resource_revision();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }
    if (salvage_continuation(state, request, lane, continuation_index, out)) {
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }
    retain_memory_history(state, request);
    if (!clear_lane_strict(state, request)) { return out; }
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    invalidate_lane(lane);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

ReleaseResult ProgramImpl::release_continuation(ContinuationHandle&& continuation) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(continuation);
    const std::uint64_t generation = ContractAccess::epoch(continuation);
    const bool valid               = !has_context_transaction() && !pending_transaction_ &&
                       valid_continuation(continuation) && !materialization_pins(index, generation);
    if (!valid) { return out; }
    try {
        if (!can_release_continuation_slot_strict(index)) { return out; }
    } catch (...) { return out; }
    release_continuation_slot_strict(index);
    ContractAccess::consume(continuation);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

bool ProgramImpl::can_release_shared_prefix_state(std::uint32_t index,
                                                  SharedPrefixSlotRole expected_role) const {
    if (index >= shared_prefix_capacity || !state_store || !text_kv_addresses ||
        shared_prefix_slots[index].role != expected_role) {
        return false;
    }
    const SharedPrefixState& shared = shared_prefix_states[index];
    if (shared.active_references != 0 || !shared.kv || !shared.identity ||
        !state_store->valid(shared.state) || !text_kv_addresses->can_release(shared.kv->text) ||
        (shared.kv->backend &&
         (!backend_kv_addresses || !backend_kv_addresses->can_release(*shared.kv->backend)))) {
        return false;
    }
    const std::uint32_t state_references = state_store->checkpoint_references(shared.state);
    return state_references != 0 &&
           (state_references != 1 ||
            state_store->can_release_after_checkpoint_references(shared.state, 1));
}

detail::PhysicalResources
ProgramImpl::release_shared_prefix_state_strict(std::uint32_t index,
                                                SharedPrefixSlotRole expected_role) noexcept {
    try {
        if (!can_release_shared_prefix_state(index, expected_role)) { std::terminate(); }
        SharedPrefixState& shared               = shared_prefix_states[index];
        SharedPrefixSlot& slot                  = shared_prefix_slots[index];
        const detail::PhysicalResources removed = owner_exclusive_resources(shared);
        const ActiveExclusiveBaseline baseline  = active_exclusive_baseline();
        const bool last_state_reference = state_store->checkpoint_references(shared.state) == 1;
        if (shared.kv->backend && !backend_kv_addresses->release(*shared.kv->backend)) {
            std::terminate();
        }
        if (!text_kv_addresses->release(shared.kv->text)) { std::terminate(); }
        state_store->release_checkpoint_reference(shared.state);
        if (last_state_reference && !state_store->release(shared.state)) { std::terminate(); }

        shared    = SharedPrefixState{};
        slot.role = SharedPrefixSlotRole::Free;
        if (++slot.generation == 0) { ++slot.generation; }
        if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        credit_active_ownership_transfers(baseline);
        return removed;
    } catch (...) { std::terminate(); }
}

ReleaseResult ProgramImpl::release_shared_prefix(SharedPrefixHandle&& handle) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(handle);
    const std::uint64_t generation = ContractAccess::epoch(handle);
    const bool valid =
        !has_context_transaction() && !pending_transaction_ && valid_shared_prefix(handle);
    if (!valid || index >= shared_prefix_capacity ||
        shared_prefix_slots[index].generation != generation) {
        return out;
    }
    try {
        if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
            return out;
        }
    } catch (...) { return out; }
    (void)release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
    ContractAccess::consume(handle);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

std::optional<qwen3_5::PhysicalUsageSnapshot>
ProgramImpl::fail_all_cleanup(ProgramCleanup cleanup) noexcept {
    pending_transaction_.reset();
    if (auto* transaction = std::get_if<ActiveCaptureTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted) { synchronize_transfer_streams(); }
        abort_active_capture(*transaction);
    }
    if (auto* transaction = std::get_if<MaterializationTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted) { synchronize_transfer_streams(); }
        release_materialization_staging(*transaction);
    }
    if (auto* transaction = std::get_if<HybridMaterializationTransaction>(&context_transaction_)) {
        hybrid_abort_materialization(*transaction);
    }
    context_transaction_.emplace<std::monostate>();
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] < continuation_capacity) {
            clear_lane_best_effort(active_sequence(lane), requests[lane]);
        }
        invalidate_lane(lane);
    }
    if (hybrid_) {
        if (cleanup == ProgramCleanup::Shutdown) { save_hybrid_cache_for_shutdown(); }
        if (device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(device.transfer_stream);
        }
        hybrid_->clear();
    }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Free) {
            release_continuation_slot_best_effort(index);
        }
    }
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        if (shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) { continue; }
        shared_prefix_states[index].active_references = 0;
        auto handle =
            ContractAccess::make_shared_prefix(this, index, shared_prefix_slots[index].generation);
        (void)release_shared_prefix(std::move(handle));
    }

    // Every owner is gone now, so anything a store still holds is unreachable: a best-effort
    // release declined an object whose state a thrown invariant left inconsistent. Left in place,
    // it would keep an idle Engine from admitting the next request.
    if (context_stores_idle()) { return std::nullopt; }
    const qwen3_5::PhysicalUsageSnapshot leaked = physical_usage();
    (void)rebuild_context_stores();
    return leaked;
}

bool ProgramImpl::context_stores_idle() const noexcept {
    const auto empty = [](const auto& store) { return store == nullptr || store->occupied() == 0; };
    return physical_occupancy() == detail::PhysicalResources{} && empty(text_kv_pages) &&
           empty(text_kv_addresses) && empty(backend_kv_pages) && empty(backend_kv_addresses) &&
           empty(state_store);
}

// Replaces the context stores with empty ones of the same capacity. Destroying the old stores
// returns their Device page leases and reservations, execution rows and Host KV allocations to the
// pools through RAII; the Host StateImage pool, whose slots the store tracks by handle, is reset
// explicitly. The page pools, execution tables, Vision loans and the Disk tier live outside the
// stores and are untouched. Only valid once no owner, lane or context transaction remains.
bool ProgramImpl::rebuild_context_stores() noexcept {
    std::unique_ptr<LogicalKVPageStore> text_pages;
    std::unique_ptr<KVAddressSpaceStore> text_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_addresses;
    std::unique_ptr<HostKVExtentStore> extents;
    std::unique_ptr<StateImageStore> states;
    try {
        text_pages     = std::make_unique<LogicalKVPageStore>(decoder->text_kv.page_pool(),
                                                              text_kv_pages->capacity());
        text_addresses = std::make_unique<KVAddressSpaceStore>(
            *text_pages, decoder->text_kv.execution_tables(), text_kv_addresses->capacity(),
            decoder->text_kv.execution_tables().logical_page_capacity());
        if (qwen3_5::PagedKVCache* backend = backend_kv_cache()) {
            backend_pages     = std::make_unique<LogicalKVPageStore>(backend->page_pool(),
                                                                     backend_kv_pages->capacity());
            backend_addresses = std::make_unique<KVAddressSpaceStore>(
                *backend_pages, backend->execution_tables(), backend_kv_addresses->capacity(),
                backend->execution_tables().logical_page_capacity());
        }
        if (host_kv_extents) {
            extents = std::make_unique<HostKVExtentStore>(*host_kv_arena,
                                                          host_kv_extents->descriptor_capacity());
        }
        states = std::make_unique<StateImageStore>(*state_images, host_state_images.get(),
                                                   state_store->capacity());
        // A Disk tier write may still read a Host replica of an old store.
        if (disk_kv) { disk_kv->wait_idle(); }
    } catch (...) { return false; }

    // Copies still in flight may target pages about to return to the pools.
    for (std::size_t rank = 0; rank < transfer_streams.size(); ++rank) {
        (void)cudaStreamSynchronize(compute_streams[rank]);
        (void)cudaStreamSynchronize(transfer_streams[rank]);
    }

    // Handles into the old stores must not alias objects of the new ones.
    for (SequenceState& sequence : continuation_states) {
        sequence.kv.reset();
        sequence.state = {};
        sequence.rewrite_state.reset();
        sequence.reserved_state.reset();
        sequence.tail_hidden               = {};
        sequence.rewrite_checkpoint_hidden = {};
        sequence.long_anchors.clear();
        sequence.shared_prefix_references.clear();
    }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Free) {
            retire_continuation_slot(index);
        }
    }
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        shared_prefix_states[index] = SharedPrefixState{};
        SharedPrefixSlot& slot      = shared_prefix_slots[index];
        if (slot.role != SharedPrefixSlotRole::Free) {
            slot.role = SharedPrefixSlotRole::Free;
            if (++slot.generation == 0) { ++slot.generation; }
        }
    }

    // Addresses hold reservations against the page pools, so they go before the pages.
    backend_kv_addresses = std::move(backend_addresses);
    text_kv_addresses    = std::move(text_addresses);
    host_kv_extents      = std::move(extents);
    backend_kv_pages     = std::move(backend_pages);
    text_kv_pages        = std::move(text_pages);
    state_store          = std::move(states);
    if (host_state_images) { host_state_images->release_all(); }
    // The hybrid prefix cache binds the stores by pointer; fail_all_cleanup already released
    // every reference it held, so it is recreated empty against the new stores.
    hybrid_lanes_.fill(HybridLaneState{});
    if (hybrid_) {
        hybrid_.reset();
        try {
            create_hybrid_prefix_cache(StartupObserver{});
        } catch (...) { return false; }
    }
    advance_resource_revision();
    return true;
}


} // namespace ninfer::models::qwen3_5::detail
