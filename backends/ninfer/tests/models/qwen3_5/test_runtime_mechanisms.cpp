#include "core/layout.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/speculative/mtp_alignment.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/program/vision_control.h"

#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/planning/rebuild_work.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::DecoderStateSpec decoder_spec(ninfer::KvCacheStorage storage, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 256,
        .kv_storage                = storage,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 = q36::plan_decoder_state(
        bf16_builder, decoder_spec(ninfer::KvCacheStorage::BFloat16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pages.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pages.spec.page_group_count == 5 &&
               bf16.text_kv.execution_tables.front().spec.logical_page_capacity == 3 &&
               bf16.text_kv.execution_tables.front().spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(std::all_of(bf16.text_kv.pages.planes.begin(), bf16.text_kv.pages.planes.end(),
                       [](const ninfer::DeviceKVPlaneLayout& plane) {
                           return plane.geometry.dtype == ninfer::DType::BF16;
                       }),
           "BF16 KV has no scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninfer::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 = q36::plan_decoder_state(
        int8_builder, decoder_spec(ninfer::KvCacheStorage::Int8Group64, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pages.planes.size() == 8 &&
               int8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pages.planes.size() == 4 &&
               int8.mtp_kv->pages.spec.page_group_count == 4 &&
               int8.mtp_kv->execution_tables.front().spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.mtp_kv->pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");

    q36::DecoderStateSpec fp8_spec = decoder_spec(ninfer::KvCacheStorage::Fp8E4M3Row256, true);
    ninfer::LayoutBuilder fp8_builder;
    const q36::DecoderStateLayout fp8 = q36::plan_decoder_state(fp8_builder, fp8_spec);
    (void)fp8_builder.finish(256);
    expect(fp8.text_kv.pages.planes.size() == 8 &&
               fp8.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               fp8.text_kv.pages.planes[2].geometry.leading_extent == 1,
           "FP8 Text KV has row-scaled code and scale planes per layer");
    expect(fp8.mtp_kv && fp8.mtp_kv->pages.planes.size() == 4 &&
               fp8.mtp_kv->pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.leading_extent == 1,
           "FP8 MTP KV has row-scaled code and scale planes");
    expect(fp8.kv_payload_bytes() == fp8.text_kv.payload_bytes() + fp8.mtp_kv->payload_bytes(),
           "FP8 Text/MTP KV payload accounting");

    ninfer::LayoutBuilder rk8v4_builder;
    const q36::DecoderStateLayout rk8v4 = q36::plan_decoder_state(
        rk8v4_builder, decoder_spec(ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, true));
    (void)rk8v4_builder.finish(256);
    expect(rk8v4.text_kv.pages.planes.size() == 8 &&
               rk8v4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::I8 &&
               rk8v4.text_kv.pages.planes[0].geometry.leading_extent == 256 &&
               rk8v4.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::U8 &&
               rk8v4.text_kv.pages.planes[1].geometry.leading_extent == 128,
           "rk8v4 Text KV has a rotated INT8 key plane and a packed int4 value plane");
    expect(rk8v4.kv_payload_bytes() ==
               rk8v4.text_kv.payload_bytes() + rk8v4.mtp_kv->payload_bytes(),
           "rk8v4 Text/MTP KV payload accounting");

    ninfer::LayoutBuilder rk4v4_builder;
    const q36::DecoderStateLayout rk4v4 =
        q36::plan_decoder_state(rk4v4_builder, decoder_spec(ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value, true));
    (void)rk4v4_builder.finish(256);
    expect(rk4v4.text_kv.pages.planes.size() == 8 &&
               rk4v4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::U8 &&
               rk4v4.text_kv.pages.planes[0].geometry.leading_extent == 128 &&
               rk4v4.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::U8 &&
               rk4v4.text_kv.pages.planes[1].geometry.leading_extent == 128,
           "rk4v4 Text KV has packed Lloyd-Max key and packed int4 value planes");
    expect(rk4v4.kv_payload_bytes() * 1000 < rk8v4.kv_payload_bytes() * 700,
           "rk4v4 KV is at least 30% smaller than rk8v4");

    ninfer::LayoutBuilder e8_builder;
    const q36::DecoderStateLayout e8 = q36::plan_decoder_state(
        e8_builder, decoder_spec(ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8, true));
    (void)e8_builder.finish(256);
    expect(e8.text_kv.pages.planes.size() == 8 &&
               e8.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::U8 &&
               e8.text_kv.pages.planes[0].geometry.leading_extent == 128 &&
               e8.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::U8 &&
               e8.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               e8.text_kv.pages.planes[2].geometry.leading_extent == 4 &&
               e8.text_kv.pages.planes[3].geometry.leading_extent == 8,
           "rk4v4-e8 Text KV packs both code planes and keeps G64 key / G32 value scales");
    expect(e8.kv_payload_bytes() == rk4v4.kv_payload_bytes(),
           "rk4v4-e8 Text/MTP KV is rk4v4's size");

    ninfer::LayoutBuilder root_builder;
    const q36::DecoderStateLayout root = q36::plan_decoder_state(
        root_builder, decoder_spec(ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value, true));
    (void)root_builder.finish(256);
    expect(root.text_kv.pages.planes.size() == 8 &&
               root.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::U8 &&
               root.text_kv.pages.planes[0].geometry.leading_extent == 64 &&
               root.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               root.text_kv.pages.planes[2].geometry.leading_extent == 4 &&
               root.text_kv.pages.planes[3].geometry.leading_extent == 8,
           "rk2v4-e8 Text KV keeps 64 key code bytes and G64 key / G32 value scales");
    expect(root.kv_payload_bytes() < e8.kv_payload_bytes(),
           "rk2v4-e8 Text/MTP KV is smaller than rk4v4-e8");

    ninfer::LayoutBuilder nvfp4_builder;
    const q36::DecoderStateLayout nvfp4 = q36::plan_decoder_state(
        nvfp4_builder, decoder_spec(ninfer::KvCacheStorage::Nvfp4Group16, true));
    (void)nvfp4_builder.finish(256);
    expect(nvfp4.text_kv.pages.planes.size() == 8 &&
               nvfp4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[0].geometry.leading_extent == 128 &&
               nvfp4.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               nvfp4.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[2].geometry.leading_extent == 16 &&
               nvfp4.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::U8 &&
               nvfp4.text_kv.pages.planes[3].geometry.leading_extent == 16,
           "NVFP4 Text KV has packed e2m1 key/value planes and raw-byte group-16 scale planes");
    expect(nvfp4.kv_payload_bytes() ==
               nvfp4.text_kv.payload_bytes() + nvfp4.mtp_kv->payload_bytes(),
           "NVFP4 Text/MTP KV payload accounting");

    ninfer::LayoutBuilder k8v4_builder;
    const q36::DecoderStateLayout k8v4 = q36::plan_decoder_state(
        k8v4_builder, decoder_spec(ninfer::KvCacheStorage::Fp8KeyNvfp4Value, true));
    (void)k8v4_builder.finish(256);
    expect(k8v4.text_kv.pages.planes.size() == 8 &&
               k8v4.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               k8v4.text_kv.pages.planes[0].geometry.leading_extent == 256 &&
               k8v4.text_kv.pages.planes[1].geometry.dtype == ninfer::DType::U8 &&
               k8v4.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               k8v4.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               k8v4.text_kv.pages.planes[2].geometry.leading_extent == 1 &&
               k8v4.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::U8 &&
               k8v4.text_kv.pages.planes[3].geometry.leading_extent == 16,
           "K8V4 Text KV has an FP8 key plane and a packed e2m1 value plane with mismatched "
           "scale codings");
    expect(k8v4.kv_payload_bytes() ==
               k8v4.text_kv.payload_bytes() + k8v4.mtp_kv->payload_bytes(),
           "K8V4 Text/MTP KV payload accounting");
}

void test_round_layout() {
    static_assert(offsetof(q36::DFlashDecodeIngress, anchors) % 16 == 0);
    static_assert(offsetof(q36::DFlashDecodeIngress, execution_frontiers) % 16 == 0);
    static_assert(offsetof(q36::DFlashDecodeIngress, sampling) % 16 == 0);
    ninfer::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{.hidden       = 32,
                                     .output_rows  = 128,
                                     .draft_window = 5,
                                     .backend      = ninfer::SpeculativeBackend::Mtp});
    const ninfer::TensorRegion exact_prefill =
        builder.add_tensor(ninfer::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.logits.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.mtp->draft_tokens.region.offset,
           "exact prefill extension retains established round-region order");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    ninfer::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder, q36::RoundStateSpec{.hidden       = 32,
                                                 .output_rows  = 128,
                                                 .draft_window = 15,
                                                 .backend = ninfer::SpeculativeBackend::DFlash});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->produced_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");

    ninfer::LayoutBuilder scoring_builder;
    auto scoring = q36::begin_round_state_layout(
        scoring_builder, {.hidden = 32, .output_rows = 128, .causal_scoring = true});
    q36::complete_round_state_layout(scoring_builder, scoring);
    expect(scoring.rope_delta.region.bytes != 0 && scoring.text_kv_table_row.region.bytes != 0,
           "scoring keeps its Text prefill controls");
    expect(!scoring.ordinary && !scoring.mtp_decode && !scoring.dflash_decode &&
               scoring.token.region.bytes == 0 && scoring.logits.region.bytes == 0,
           "scoring does not reserve generation frames or sampled output");
    for (const std::uint32_t rows : {1U, 2U, 8U}) {
        ninfer::LayoutBuilder copy_builder;
        auto copy_layout = q36::begin_round_state_layout(
            copy_builder, {.hidden         = 32,
                           .output_rows    = 128,
                           .batch_capacity = rows,
                           .draft_window   = 15,
                           .backend        = ninfer::SpeculativeBackend::DFlash2});
        q36::complete_round_state_layout(copy_builder, copy_layout);
        const auto bytes = copy_builder.finish(256);
        std::vector<std::byte> storage(bytes + 255);
        auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        address      = (address + 255) & ~std::uintptr_t(255);
        q36::RoundState copy({reinterpret_cast<void*>(address), bytes}, copy_layout);
        const int capacity = static_cast<int>(rows);
        for (std::uint32_t k = 1; k <= 15; ++k) {
            const auto narrow = copy.dflash_decode->narrowed(k);
            expect(narrow.target_logits.ne[1] == static_cast<int>(k + 1) &&
                       narrow.target_logits.ne[2] == capacity &&
                       narrow.draft_tokens.ne[0] == static_cast<int>(k) &&
                       narrow.draft_tokens.ne[1] == capacity &&
                       narrow.candidate_ids.ne[1] == static_cast<int>(k) &&
                       narrow.candidate_ids.ne[2] == capacity &&
                       narrow.licensed_tokens.ne[0] == static_cast<int>(k + 1) &&
                       narrow.verify_ids.ne[1] == capacity,
                   "narrowed frame shapes cover every row at the round width");
            expect(narrow.target_logits.is_contiguous() && narrow.target_hidden.is_contiguous() &&
                       narrow.candidate_ids.is_contiguous() && narrow.proposal_q.is_contiguous() &&
                       narrow.draft_tokens.is_contiguous(),
                   "narrowed frame rows are dense at the round width");
            expect(narrow.target_logits.data == copy.dflash_decode->target_logits.data &&
                       narrow.draft_tokens.data == copy.dflash_decode->draft_tokens.data &&
                       narrow.append_positions.ne[0] == 16 &&
                       narrow.append_positions.ne[1] == capacity,
                   "narrowed frame aliases storage and keeps native catch-up capacity");
            expect(reinterpret_cast<std::uintptr_t>(narrow.anchors.data) % 16 == 0 &&
                       reinterpret_cast<std::uintptr_t>(narrow.execution_frontiers.data) % 16 == 0,
                   "narrowed frame controls retain vector alignment");
        }
        for (std::uint32_t k : {0, 16}) {
            bool rejected = false;
            try {
                (void)copy.dflash_decode->narrowed(k);
            } catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected, "invalid narrowed frame width rejected");
        }
    }
    ninfer::LayoutBuilder mtp_copy_builder;
    auto mtp_copy_layout = q36::begin_round_state_layout(
        mtp_copy_builder, {.hidden         = 32,
                           .output_rows    = 128,
                           .batch_capacity = 1,
                           .draft_window   = 5,
                           .verify_window  = 15,
                           .backend        = ninfer::SpeculativeBackend::Mtp});
    q36::complete_round_state_layout(mtp_copy_builder, mtp_copy_layout);
    const auto mtp_bytes = mtp_copy_builder.finish(256);
    std::vector<std::byte> mtp_storage(mtp_bytes + 255);
    const auto mtp_address =
        (reinterpret_cast<std::uintptr_t>(mtp_storage.data()) + 255) & ~std::uintptr_t(255);
    q36::RoundState mtp_copy({reinterpret_cast<void*>(mtp_address), mtp_bytes}, mtp_copy_layout);
    expect(mtp_copy.mtp_decode->ar_positions.ne[1] == 4 &&
               mtp_copy.mtp_decode->ar_rope_positions.ne[1] == 4 &&
               mtp_copy.mtp_decode->ar_valid_columns.ne[1] == 4,
           "MTP copy frame allocates the neural AR steps, not the copy width");
    for (std::uint32_t k = 1; k <= 15; ++k) {
        const auto frame = mtp_copy.mtp_decode->verification_view(k);
        expect(frame.target_logits.ne[1] == static_cast<int>(k + 1) &&
                   frame.current_drafts.ne[0] == static_cast<int>(k) &&
                   frame.alignment_ids.ne[0] == static_cast<int>(k + 1) &&
                   frame.next_drafts.ne[1] == 5 && frame.ar_positions.ne[1] == 4,
               "MTP copy verification and next proposal dimensions are independent");
        expect(frame.target_hidden.data == mtp_copy.mtp_decode->target_hidden.data &&
                   frame.target_logits.is_contiguous() && frame.alignment_hidden.is_contiguous(),
               "MTP copy frame remains a dense prefix");
    }
    for (const std::uint32_t k : {0U, 16U}) {
        bool rejected = false;
        try {
            (void)mtp_copy.mtp_decode->verification_view(k);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "MTP invalid verification width rejected");
    }
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControlPlan plan =
        q36::plan_vision_control(prompt, {.spatial_merge_size = 2, .position_grid_side = 48});
    const q36::VisionControl control = q36::build_vision_control(prompt, plan, 0);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");

    const q36::VisionControl suffix = q36::build_vision_control(prompt, plan, 1);
    expect(suffix.prepared_item_begin == 1 && suffix.items.size() == 1 &&
               suffix.items[0].patch_begin == control.items[1].patch_begin &&
               suffix.items[0].scatter_indices == control.items[1].scatter_indices &&
               suffix.items[0].position_ids == control.items[1].position_ids,
           "Vision suffix control contents");
}

q36::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                          static_cast<std::uint8_t>(q36::PromptModality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    q36::VisionItem item{.modality    = q36::PromptModality::Image,
                         .grid        = {.temporal = 1, .height = 2, .width = 4},
                         .patch_begin = 0,
                         .patch_count = 8,
                         .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(q36::PreparedPromptData& prompt, ninfer::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    q36::PreparedPromptData original    = identity_prompt();
    std::vector<ninfer::TokenId> ledger = original.token_ids;
    q36::detail::ResidentPrefixIdentity resident;
    q36::detail::PrefixShortlistDigests digests;
    resident.reserve(16);
    resident.assign(original);
    digests.reserve(16);
    digests.assign(original);

    expect(q36::detail::prefix_matches(original, ledger, resident, original.token_ids.size()),
           "identical multimodal prefix identity");

    q36::PreparedPromptData changed_media = identity_prompt(2);
    expect(!q36::detail::prefix_matches(changed_media, ledger, resident,
                                        changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(q36::detail::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!q36::detail::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!q36::detail::prefix_matches(changed_position, ledger, resident,
                                        changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    q36::PreparedPromptData changed_decomposition              = identity_prompt();
    changed_decomposition.identity.rewrite_execution_frontiers = {1};
    expect(!q36::detail::prefix_matches(changed_decomposition, ledger, resident,
                                        changed_decomposition.token_ids.size()),
           "different GDN execution decomposition must not reuse resident state");
    changed_decomposition.identity.rewrite_execution_frontiers = {4};
    expect(q36::detail::prefix_matches(changed_decomposition, ledger, resident, 3),
           "execution decomposition wholly after the frontier changed prefix identity");

    q36::PreparedPromptData resident_future              = identity_prompt();
    resident_future.identity.rewrite_execution_frontiers = {1, 4};
    q36::detail::ResidentPrefixIdentity resident_with_future;
    resident_with_future.assign(resident_future);
    q36::PreparedPromptData incoming_future              = identity_prompt();
    incoming_future.identity.rewrite_execution_frontiers = {1, 3};
    expect(q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 1),
           "resident execution decomposition after the frontier changed prefix identity");
    expect(!q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 3),
           "different execution decomposition inside the frontier reused resident state");

    q36::detail::PrefixShortlistDigests future_digest;
    future_digest.assign(resident_future);
    q36::detail::PrefixShortlistDigests incoming_digest;
    incoming_digest.assign(incoming_future);
    expect(future_digest.at(1) == incoming_digest.at(1),
           "future execution boundaries changed an earlier content shortlist");
    expect(future_digest.at(3) != incoming_digest.at(3),
           "different in-prefix execution boundaries shared a shortlist digest");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    const std::array<ninfer::TokenId, 1> generated{12};
    digests.append_generated(generated, original.rope_delta);
    append_text_token(original, 12, 4);
    q36::detail::PrefixShortlistDigests rebuilt;
    rebuilt.assign(original);
    expect(digests.at(ledger.size()) == rebuilt.at(ledger.size()),
           "incremental generated-token shortlist diverged from a full rebuild");
    expect(q36::detail::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    q36::PreparedPromptData accepted_rebuild = identity_prompt();
    const std::array<ninfer::TokenId, 3> proposed{20, 21, 22};
    const std::span<const ninfer::TokenId> accepted(proposed.data(), 2);
    std::vector<ninfer::TokenId> accepted_ledger = accepted_rebuild.token_ids;
    accepted_ledger.insert(accepted_ledger.end(), accepted.begin(), accepted.end());
    q36::detail::ResidentPrefixIdentity accepted_resident;
    q36::detail::PrefixShortlistDigests accepted_digests;
    accepted_resident.assign(accepted_rebuild);
    accepted_digests.assign(accepted_rebuild);
    accepted_resident.append_generated(accepted.size(), accepted_rebuild.rope_delta, 1);
    accepted_digests.append_generated(accepted, accepted_rebuild.rope_delta, 1);
    append_text_token(accepted_rebuild, accepted[0], 4);
    append_text_token(accepted_rebuild, accepted[1], 5);
    accepted_rebuild.identity.rewrite_execution_frontiers = {5};
    q36::detail::PrefixShortlistDigests rebuilt_accepted_digests;
    rebuilt_accepted_digests.assign(accepted_rebuild);
    expect(q36::detail::prefix_matches(accepted_rebuild, accepted_ledger, accepted_resident,
                                       accepted_ledger.size()) &&
               accepted_digests.at(accepted_ledger.size()) ==
                   rebuilt_accepted_digests.at(accepted_ledger.size()),
           "accepted generated prefix and rebuilt history formed different cache identities");

    const q36::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    digests.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    q36::detail::PrefixShortlistDigests prompt_digest;
    prompt_digest.assign(prompt_only);
    expect(digests.at(ledger.size()) == prompt_digest.at(ledger.size()),
           "truncated shortlist did not restore the original frontier digest");
    expect(q36::detail::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}

void test_rebuild_work_prompt_frontier_boundary() {
    constexpr std::uint32_t prompt_tokens = 100;
    constexpr std::uint32_t prefill_chunk = 2048;
    std::uint32_t tail_begin              = 0;
    q36::runtime_support::include_rebuild_boundary(tail_begin, prompt_tokens, prompt_tokens);
    expect(tail_begin == prompt_tokens,
           "prompt-frontier rebuild boundary was not retained for continuation growth");

    ninfer::runtime::PrefillWork work =
        ninfer::runtime::make_prefill_work(0, prompt_tokens, 0, 0, prefill_chunk);
    q36::runtime_support::advance_segmented_rebuild_work(work, tail_begin, prompt_tokens,
                                                         prompt_tokens + 1, prefill_chunk);
    const ninfer::runtime::PrefillWork exact =
        ninfer::runtime::make_prefill_work(0, prompt_tokens + 1, 0, 0, prefill_chunk);
    expect(work.chunks == 2 && work.tokens == exact.tokens &&
               work.attention_pairs == exact.attention_pairs,
           "continuation growth did not preserve the prompt-frontier rebuild split");
}

} // namespace

int main() {
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_prefix_identity();
    test_rebuild_work_prompt_frontier_boundary();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}
