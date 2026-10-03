#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"
#include "ninfer/ops/rope.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;

struct DFlashPersistentLayout {
    std::optional<qwen3_5::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    // ReplaySSM records, one layout per state shard: the first here, the rest in `extra`.
    std::optional<GdnReplayRecordLayout> replay_records;
    std::vector<GdnReplayRecordLayout> extra_replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_5::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> memory_key_sums;
    std::optional<TensorLayout> memory_candidate_keys;
    std::optional<TensorLayout> memory_candidate_origin;
    std::optional<TensorLayout> memory_prefill_origin;
    std::optional<TensorLayout> memory_query_sums;
    std::optional<TensorLayout> memory_ranges;
    std::optional<TensorLayout> score_hidden;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    std::optional<TensorLayout> grammar_masks;
    // Pinned Host cost of one complete Main Text KV page group, taken from the same planned
    // geometry the Device pool binds. The Host RAM budget compares one StateImage against the Main
    // pages a re-prefill of the gap it covers would pin, and that comparison is denominated in
    // Main Text pages: the stride's plane inventory (and therefore its group-scale term) follows
    // --kv-dtype, so it is carried rather than re-derived from configuration.
    std::size_t host_kv_text_page_stride = 0;
    std::size_t bytes            = 0;
    // Persistent bytes on each further device (device 1 first): the KV planes, block-table copy and
    // recurrent state of the layers that stage owns. Empty on one device.
    std::vector<std::size_t> extra_rank_bytes;
    std::size_t kv_payload_bytes = 0;
    // Arena offset just past the last page-major KV plane. Everything an overlay Vision window may
    // borrow from free KV lies below it; stores interleaved there are simply never selected.
    std::size_t lendable_kv_end_bytes = 0;
};

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    // Resident: folded into capacity after the general region. Overlay: the per-window encode
    // plan, borrowed per item, and nothing but the MTP bridge column lives in this workspace.
    std::optional<VisionWorkspacePlan> vision;
    bool vision_resident = true;
    // Overlay only: the staged visual column of a multimodal MTP bridge, past the general region.
    std::size_t vision_bridge_offset = 0;
    std::size_t vision_bridge_bytes  = 0;
    std::size_t capacity             = 0;
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t kvmem_window_tokens = 0;
    KvmemOptions kvmem;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    bool fast_prefill_kernel                = false;
    std::uint32_t draft_window              = 0;
    std::uint32_t lookup_ngram             = 0;
    MtpDraftPolicy mtp_policy               = MtpDraftPolicy::Fixed;
    std::uint32_t ngram_draft_window        = 0;
    std::uint32_t ngram_min_match           = 12;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    ops::RopeYarn rope_yarn;
    std::uint32_t mtp_attention_window = 0;
    models::LoadOptions features;
    bool use_cuda_graph    = true;
    // Nonzero replaces the computed per-profile CUDA Graph allowance in total.
    std::size_t cuda_graph_allowance_bytes = 0;
    bool causal_scoring    = false;
    bool structured_output = false;
    int device             = 0;
    ContextCacheOptions context_cache;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {

struct SequencePlanImpl {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t kvmem_window_tokens = 0;
    KvmemOptions kvmem;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t main_page_groups          = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    bool fast_prefill_kernel                = false;
    std::uint32_t draft_window              = 0;
    std::uint32_t lookup_ngram             = 0;
    MtpDraftPolicy mtp_policy               = MtpDraftPolicy::Fixed;
    // Copy proposals verified alongside the neural drafter; zero disables them.
    std::uint32_t ngram_draft_window        = 0;
    std::uint32_t ngram_min_match           = 12;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    ops::RopeYarn rope_yarn;
    std::uint32_t mtp_attention_window = 0;
    models::LoadOptions features;
    bool use_cuda_graph    = true;
    bool causal_scoring    = false;
    bool structured_output = false;
    int device             = 0;
    ContextCacheOptions context_cache;
    PersistentLayout persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
    // What each further device reserves (device 1 first): its persistent state, the scratch its
    // stage runs in, and its share of the graph allowance.
    std::vector<std::size_t> extra_rank_reservation_bytes;
};

// The widest round a decode frame verifies: the draft window, or the copy window when wider.
[[nodiscard]] inline std::uint32_t widest_verify_window(const SequencePlanImpl& plan) noexcept {
    return std::max(plan.draft_window, plan.ngram_draft_window);
}

// The widest forward pass a pipeline stage boundary carries: prefill columns, or every lane's
// verification columns. Sizes the boundary links and the scratch each stage needs around its layers.
[[nodiscard]] inline std::uint64_t stage_boundary_columns(const SequencePlanImpl& plan) noexcept {
    return std::max<std::uint64_t>(std::min(plan.prefill_chunk, plan.capacity),
                                   static_cast<std::uint64_t>(plan.max_concurrency) *
                                       (widest_verify_window(plan) + 1U));
}

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl> minimum;
};

} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_5::detail {


// Largest merged-token count one media item may occupy under these startup options.
[[nodiscard]] std::uint32_t vision_item_token_bound(std::uint32_t capacity,
                                                    const models::LoadOptions& features);

[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

// Resolves the single Host RAM budget into the plan's context-cache shape: one StateImage per
// position it retains and Host KV for everything else. StateImages are the fixed per-position cost
// of a checkpoint and Host KV the per-token cost, so the budget first covers the inventory the
// capture path creates — 2 + A images per private owner plus one per shared entry — then, when the
// engine anchors message boundaries automatically, spends the remaining StateImage headroom under
// the half-budget cap on extra long anchors per owner, and gives Host KV the remainder. Growing A
// without re-sizing the pool in the same pass would leave the Host StateImage pool undersized for
// the anchors the capture path then creates, so both are resolved together.
//
// A never drops below the configured count: the budget adds anchors, it does not remove them, and
// without automatic anchoring nothing would create more than clients mark, so A stays as
// configured. One extra anchor pays for itself only while the gap it covers exceeds
// the Main KV pages one StateImage is worth, which bounds the useful count by the logical page
// space; when even the mandatory inventory exceeds half the budget, the configured count is kept
// so the caller's rejection reports the real shortfall.
//
// `state_image_bytes` is one complete Host StateImage, `host_kv_group_bytes` one Main Text Host KV
// page group, and the capacities are the already-normalized private/shared continuation catalogs.
void resolve_host_cache_budget(ContextCacheOptions& cache, std::uint32_t private_capacity,
                               std::uint32_t shared_capacity, std::uint32_t capacity,
                               std::uint64_t state_image_bytes, std::uint64_t host_kv_group_bytes);

} // namespace ninfer::models::qwen3_5::detail
