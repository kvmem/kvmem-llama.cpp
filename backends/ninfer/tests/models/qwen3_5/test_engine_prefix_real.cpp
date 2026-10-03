#include "guarded_main.h"
#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// The anchor-spacing option only exists from the spacing commit on. Setting it through this
// helper lets the same scenarios build against the pre-fix tree for an A/B baseline, where the
// spacing silently stays at the old last-L rule.
template <class CacheOptions>
void set_anchor_spacing(CacheOptions& cache, std::uint32_t spacing) {
    if constexpr (requires { cache.long_anchor_min_spacing_tokens; }) {
        cache.long_anchor_min_spacing_tokens = spacing;
    }
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 1024;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.enable_vision                    = true;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 4;
    return options;
}

ninfer::EngineOptions host_restore_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    // This fixture asserts a complete Host restore. Extra Device checkpoint slots can leave the
    // selected endpoint resident and turn the pressure request into a different valid scenario.
    options.context_cache.device_state_slots     = 0;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions shared_replacement_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 4;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions anthropic_prefix_regression_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 2048;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk                    = 512;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 1;
    options.context_cache.host_state_slots   = 4;
    options.context_cache.host_kv_capacity_bytes            = 512ULL << 20;
    options.context_cache.max_private_continuations         = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions shared_rewrite_materialization_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 100000;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(100000);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 2;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions private_long_anchor_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::None;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 4;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 1;
    return options;
}

ninfer::EngineOptions automatic_long_anchor_engine_options(const char* artifact) {
    ninfer::EngineOptions options            = private_long_anchor_engine_options(artifact);
    options.context_cache.device_state_slots = 8;
    options.context_cache.max_long_anchors_per_continuation = 2;
    options.context_cache.automatic_long_anchors            = true;
    // The scenario's messages are a few dozen tokens; disable the anchor spacing so both
    // interior boundaries are anchored, as the scenario asserts.
    set_anchor_spacing(options.context_cache, 0);
    // Written against releasing the checkpoint a conversation diverged from, which this line keeps
    // opt-in: without it the second branch's endpoint keeps its value and the third branch is
    // prefilled from Root instead of resuming from the shared anchor.
    options.context_cache.release_diverged_checkpoints = true;
    return options;
}

ninfer::EngineOptions salvage_mid_prefill_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 8192;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    // One 1024-token chunk commits exactly kSalvageMinFrontier, so a cancel sampled at the
    // first chunk boundary lands mid-prefill at a salvage-eligible frontier.
    options.prefill_chunk                                   = 1024;
    options.speculative.backend                             = ninfer::SpeculativeBackend::None;
    options.max_concurrency                                 = 1;
    options.max_pending_requests                            = 1;
    options.context_cache.device_state_slots                = 8;
    options.context_cache.host_state_slots                  = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions last_alias_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 3;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions concurrent_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 512;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
    options.max_concurrency                  = 8;
    options.max_pending_requests             = 8;
    options.context_cache.device_state_slots = 16;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 8;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions pressure_resume_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 8192;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 8ULL << 30;
    options.context_cache.max_private_continuations         = 4;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::EngineOptions private_checkpoint_pressure_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 8192;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(16384);
    options.prefill_chunk                    = 1024;
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 4;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

// Small device KV capacity so a handful of retained private prefixes overflow the cache and
// force pressure eviction. No host fallback: retained prefixes must be evicted, not spilled.
ninfer::EngineOptions recency_retention_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 4096;
    // KV-capacity pressure (not the conversation cap) is what the retention order governs. A safe,
    // known-valid capacity (4096 page groups) that the five long retained prefixes overflow, so
    // eviction is driven by KV capacity through the value-based planner.
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    // Enough Device state slots for every conversation's checkpoints, so KV capacity forces the
    // evictions: a private checkpoint that finds the state pool full reclaims an idle
    // conversation itself, which would decide the scenario before KV pressure does.
    options.context_cache.device_state_slots                = 16;
    options.context_cache.host_state_slots                  = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 8;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    // The scenario exercises the recency-ordered escape hatch, which is opt-in here.
    options.context_cache.recency_eviction = true;
    return options;
}

std::vector<std::uint8_t> gradient_ppm(int width = 64, int height = 64) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput chinese_chat(bool enable_thinking) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "你好，简单介绍一下你自己。", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

struct RegisteredFrontendGoldens {
    std::uint32_t thinking_tokens;
    std::uint32_t no_thinking_tokens;
    ninfer::PrefixReusePath host_restore_path;
};

std::optional<RegisteredFrontendGoldens> registered_frontend_goldens(
    std::string_view model_name) {
    if (model_name == "qwen3.6-27b") {
        return RegisteredFrontendGoldens{16, 18, ninfer::PrefixReusePath::PrivateTurnClosure};
    }
    if (model_name == "qwen3.8-27b") {
        return RegisteredFrontendGoldens{58, 18, ninfer::PrefixReusePath::PrivateEndpoint};
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Realistic agent-scenario helpers: long multi-turn OpenAI-chat conversations with a leading
// system prompt and a fixed tool set (the shared stable prefix), a growing conversation with tool
// calls, thinking enabled and preserved, streaming. These helpers build that shape with small
// caches so the caching fixes (shared replacement, #251 reclaim, recency-ordered retention,
// cost-scaled search budget, checkpoint eviction, stats publication) are exercised under
// contention.

ninfer::EngineOptions agent_engine_options(const char* artifact, std::uint32_t shared_prefixes,
                                           std::uint32_t private_continuations,
                                           std::uint32_t device_slots, std::uint32_t kv_capacity) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(kv_capacity);
    options.prefill_chunk             = 256;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.max_concurrency           = 2;
    options.max_pending_requests      = 2;
    options.context_cache.device_state_slots                = device_slots;
    options.context_cache.host_state_slots                  = 8;
    options.context_cache.host_kv_capacity_bytes            = 256ULL << 20;
    options.context_cache.max_private_continuations         = private_continuations;
    options.context_cache.max_shared_prefixes               = shared_prefixes;
    options.context_cache.max_long_anchors_per_continuation = 0;
    // The agent scenarios were written against recency eviction, the cost-scaled admission search
    // and growing Device KV leases, which this line keeps opt-in.
    options.context_cache.recency_eviction          = true;
    options.context_cache.thorough_admission_search = true;
    options.context_cache.kv_lease_growth           = true;
    return options;
}

std::string agent_tool_json(std::string_view name, std::string_view description) {
    std::string json = "{\"type\":\"function\",\"function\":{\"name\":\"";
    json += name;
    json += "\",\"description\":\"";
    json += description;
    json += "\",\"parameters\":{\"type\":\"object\",\"properties\":{\"value\":{\"type\":\"string\"}"
            "},\"required\":[\"value\"]}}}";
    return json;
}

ninfer::ChatMessage agent_text_message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

// Builds a realistic agent prompt: a leading System message + the tool definitions (the shared
// stable prefix, marked at the ToolBoundary after the last tool) + a multi-turn conversation.
// Thinking is on and preserved, mirroring the production agent.
ninfer::PromptInput agent_prompt(std::string system, std::vector<std::string> tools,
                                 std::vector<ninfer::ChatMessage> conversation,
                                 bool enable_thinking) {
    ninfer::PromptInput input;
    input.messages.push_back(agent_text_message(ninfer::ChatRole::System, std::move(system)));
    for (auto& turn : conversation) { input.messages.push_back(std::move(turn)); }
    input.options.enable_thinking   = enable_thinking;
    input.options.preserve_thinking = true;
    for (auto& tool : tools) { input.options.tool_jsons.push_back(std::move(tool)); }
    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
        .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
        .after_tool_count = static_cast<std::uint32_t>(tools.size()),
    });
    input.context_cache.session_key = "agent-session";
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    return input;
}

int exercise_registered_frontend(const ninfer::Engine& engine) {
    const auto thinking_tokens    = engine.count_tokens(chinese_chat(true));
    const auto no_thinking_tokens = engine.count_tokens(chinese_chat(false));
    const auto model_name         = engine.load_summary().model_name;
    const auto goldens            = registered_frontend_goldens(model_name);
    if (!goldens) {
        std::cerr << "registered tokenizer/chat template has no prompt golden: model="
                  << model_name << '\n';
        return 1;
    }
    if (thinking_tokens != goldens->thinking_tokens ||
        no_thinking_tokens != goldens->no_thinking_tokens) {
        std::cerr << "registered tokenizer/chat template changed a prompt golden: model="
                  << model_name << " thinking=" << thinking_tokens << "/"
                  << goldens->thinking_tokens << " no_thinking=" << no_thinking_tokens << "/"
                  << goldens->no_thinking_tokens << '\n';
        return 1;
    }
    return 0;
}

class ObservationSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart start) override {
        if (started_) { valid_ = false; }
        started_ = true;
        start_   = start;
    }

    void progress(ninfer::PromptProgress progress) override {
        if (!started_ || timing_seen_ ||
            progress.total_prompt_tokens != start_.prompt.prompt_tokens ||
            progress.reused_prompt_tokens != start_.reused_prompt_tokens ||
            progress.processed_prompt_tokens < last_processed_ ||
            progress.processed_prompt_tokens > progress.total_prompt_tokens ||
            progress.elapsed_ns < last_progress_elapsed_ns_) {
            valid_ = false;
        }
        last_processed_           = progress.processed_prompt_tokens;
        last_progress_elapsed_ns_ = progress.elapsed_ns;
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (!started_ || last_processed_ != start_.prompt.prompt_tokens ||
            (timing_seen_ && (timing.generated_tokens < last_timing_.generated_tokens ||
                              timing.prompt_elapsed_ns != last_timing_.prompt_elapsed_ns ||
                              timing.generation_elapsed_ns < last_timing_.generation_elapsed_ns))) {
            valid_ = false;
        }
        timing_seen_ = true;
        last_timing_ = timing;
    }

    void publish(ninfer::OutputDelta) override {
        if (!timing_seen_) { valid_ = false; }
    }

    [[nodiscard]] bool valid_for(const ninfer::GenerationResult& result) const {
        return valid_ && started_ && timing_seen_ && start_.reused_prompt_tokens == 0 &&
               last_processed_ == start_.prompt.prompt_tokens &&
               last_timing_.generated_tokens == result.generated_token_ids.size() &&
               result.timings.prompt_wall_seconds > 0.0 &&
               result.timings.generation_wall_seconds >= 0.0;
    }

private:
    ninfer::GenerationStart start_;
    ninfer::GenerationTimingObservation last_timing_;
    std::uint32_t last_processed_           = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    bool started_                           = false;
    bool timing_seen_                       = false;
    bool valid_                             = true;
};

int exercise_stream_observations(ninfer::Engine& engine) {
    std::vector<ninfer::TokenId> prompt(2050, 198);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const ninfer::GenerationObservationOptions observation{
        .phase_timings = true, .live_timings = true, .prompt_progress = true};

    ObservationSink sink;
    ninfer::GenerationHandle generation =
        engine.submit(engine.prepare_tokens(std::move(prompt)), std::move(request),
                      ninfer::OutputConsumerMode::Streaming, observation);
    const ninfer::GenerationResult result = generation.wait(&sink);
    if (result.generated_token_ids.size() != 3 || !sink.valid_for(result)) {
        std::cerr
            << "stream observations lost prompt progress, commit timing, or publication order\n";
        return 1;
    }
    return 0;
}

int exercise_full_prefill_chunk(ninfer::Engine& engine) {
    constexpr std::size_t kChunkTokens = 1024;
    std::vector<ninfer::TokenId> prompt(kChunkTokens, 198);
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(std::move(prompt)), options);
    if (result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "full-chunk prefill did not complete through the planned workspace\n";
        return 1;
    }
    return 0;
}

int exercise_abandoned_handle_capacity(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    {
        auto abandoned = engine.submit(engine.prepare_tokens(prompt), request);
        if (!abandoned) {
            std::cerr << "abandonment fixture did not create a generation handle\n";
            return 1;
        }
    }
    const auto crossed = engine.generate(engine.prepare_tokens(prompt), request);
    if (crossed.generated_token_ids.size() != 1) {
        std::cerr << "request after an abandoned handle did not complete\n";
        return 1;
    }

    auto first      = engine.submit(engine.prepare_tokens(prompt), request);
    auto second     = engine.submit(engine.prepare_tokens(prompt), request);
    bool overloaded = false;
    try {
        auto third = engine.submit(engine.prepare_tokens(prompt), request);
        (void)third;
    } catch (const ninfer::RequestError& error) {
        overloaded = error.kind() == ninfer::RequestErrorKind::Overloaded;
    }
    if (!overloaded) {
        std::cerr << "outstanding capacity was released twice or not enforced\n";
        return 1;
    }
    if (first.wait().generated_token_ids.size() != 1 ||
        second.wait().generated_token_ids.size() != 1) {
        std::cerr << "requests retained after the overload check did not complete\n";
        return 1;
    }
    return 0;
}

int exercise_zero_suffix_reuse(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    ninfer::RequestOptions baseline_options;
    baseline_options.execution.requested_output_tokens = 8;
    baseline_options.execution.sampling.temperature    = 0.0F;
    baseline_options.execution.allow_prefix_reuse      = true;
    baseline_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), baseline_options);
    if (baseline.generated_token_ids.size() != 8) {
        std::cerr << "zero-suffix baseline did not generate eight tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> exact_frontier = prompt;
    exact_frontier.insert(exact_frontier.end(), baseline.generated_token_ids.begin(),
                          baseline.generated_token_ids.end() - 1);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 2;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(exact_frontier), reuse_options);
    if (reused.reused_prompt_tokens != exact_frontier.size()) {
        std::cerr << "zero-suffix reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << exact_frontier.size() << '\n';
        return 1;
    }
    if (reused.generated_token_ids.size() != 2 ||
        reused.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "zero-suffix reuse did not resume from the retained target frontier\n";
        return 1;
    }
    return 0;
}

int exercise_prefix(ninfer::Engine& engine) {
    ninfer::RequestOptions first_options;
    first_options.execution.requested_output_tokens = 5;
    first_options.execution.sampling.temperature    = 0.0F;
    first_options.stop.include_model_defaults       = false;

    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), first_options);
    if (first.generated_token_ids.size() != 5) {
        std::cerr << "first request did not generate five tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 5;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), reuse_options);

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "append reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }

    if (const int result = exercise_zero_suffix_reuse(engine, prompt); result != 0) {
        return result;
    }

    return 0;
}

int exercise_host_restore(const char* artifact) {
    ninfer::Engine engine(host_restore_engine_options(artifact));
    const auto model_name = engine.load_summary().model_name;
    const auto goldens    = registered_frontend_goldens(model_name);
    if (!goldens) {
        std::cerr << "Host-restore fixture has no reuse-path golden: model=" << model_name << '\n';
        return 1;
    }
    auto options = [](std::uint32_t outputs, bool reuse) {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = outputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = reuse;
        request.stop.include_model_defaults       = false;
        return request;
    };

    const auto retained_input = [] {
        std::string text;
        text.reserve(6U * 300U);
        for (std::uint32_t index = 0; index < 300; ++index) { text += "alpha "; }
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "host-restore-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };

    const ninfer::GenerationResult retained =
        engine.generate(engine.prepare(retained_input()), options(5, true));
    if (retained.prompt.prompt_tokens <= 256 || retained.generated_token_ids.size() != 5) {
        std::cerr << "Host-restore source request did not complete\n";
        return 1;
    }

    // Whether the retained reply re-renders byte-identically, making the longer endpoint reusable
    // instead of the turn closure, depends on the chat template, so each model names its path.
    ninfer::PromptInput continuation = retained_input();
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = retained.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = retained.content, .media = {}});
    continuation.messages.push_back(std::move(assistant));
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    continuation.messages.push_back(std::move(followup));

    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult pressure_result =
        engine.generate(engine.prepare(continuation), options(2, false));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    if (pressure_result.generated_token_ids.size() != 2 ||
        after_pressure.state_d2h_count <= before_pressure.state_d2h_count ||
        after_pressure.main_kv_d2h_pages <= before_pressure.main_kv_d2h_pages ||
        after_pressure.backend_kv_d2h_pages <= before_pressure.backend_kv_d2h_pages) {
        std::cerr << "Host pressure did not demote the complete MTP checkpoint: state="
                  << after_pressure.state_d2h_count << " main=" << after_pressure.main_kv_d2h_pages
                  << " backend=" << after_pressure.backend_kv_d2h_pages
                  << " degraded=" << after_pressure.pressure_private_owners_degraded
                  << " evicted=" << after_pressure.pressure_private_owners_evicted << '\n';
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(std::move(continuation)), options(2, true));
    const ninfer::RuntimeStats after_restore = engine.runtime_stats();
    if (restored.generated_token_ids.size() != 2 ||
        restored.prefix_reuse_path != goldens->host_restore_path ||
        restored.reused_prompt_tokens == 0 ||
        after_restore.state_h2d_count <= after_pressure.state_h2d_count ||
        after_restore.main_kv_h2d_pages <= after_pressure.main_kv_h2d_pages ||
        after_restore.backend_kv_h2d_pages <= after_pressure.backend_kv_h2d_pages) {
        std::cerr << "Complete MTP checkpoint was not materialized from Host: path="
                  << static_cast<int>(restored.prefix_reuse_path) << '/'
                  << static_cast<int>(goldens->host_restore_path)
                  << " reused=" << restored.reused_prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size()
                  << " state_d2h=" << before_pressure.state_d2h_count << '/'
                  << after_pressure.state_d2h_count << '/' << after_restore.state_d2h_count
                  << " state_h2d=" << before_pressure.state_h2d_count << '/'
                  << after_pressure.state_h2d_count << '/' << after_restore.state_h2d_count
                  << " state_d2d=" << before_pressure.state_d2d_count << '/'
                  << after_pressure.state_d2d_count << '/' << after_restore.state_d2d_count
                  << " slots=" << after_pressure.device_state_occupied_slots << ':'
                  << after_pressure.host_state_occupied_slots << '/'
                  << after_restore.device_state_occupied_slots << ':'
                  << after_restore.host_state_occupied_slots
                  << " main=" << after_pressure.main_kv_h2d_pages << '/'
                  << after_restore.main_kv_h2d_pages
                  << " backend=" << after_pressure.backend_kv_h2d_pages << '/'
                  << after_restore.backend_kv_h2d_pages
                  << " host_kv=" << after_pressure.host_kv_occupied_bytes << '/'
                  << after_restore.host_kv_occupied_bytes
                  << " degraded=" << after_restore.pressure_private_owners_degraded
                  << " evicted=" << after_restore.pressure_private_owners_evicted
                  << " | pressure d2h state=" << after_pressure.state_d2h_count
                  << " main=" << after_pressure.main_kv_d2h_pages
                  << " backend=" << after_pressure.backend_kv_d2h_pages
                  << " | source prompt=" << retained.prompt.prompt_tokens
                  << " pressure prompt=" << pressure_result.prompt.prompt_tokens
                  << " restored prompt=" << restored.prompt.prompt_tokens << '\n';
        return 1;
    }

    // The uncached pressure request and checkpoint resume use different valid prefill splits, so
    // the pressure result is a completion and transfer trigger rather than an exact-token oracle.
    return 0;
}

int exercise_shared_replacement_and_full_capacity_reuse(const char* artifact) {
    ninfer::EngineOptions engine_options = shared_replacement_engine_options(artifact);
    engine_options.max_context           = 1024;
    engine_options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    engine_options.context_cache.max_private_continuations = 1;
    ninfer::Engine engine(std::move(engine_options));
    ninfer::RequestOptions capture_request;
    capture_request.execution.requested_output_tokens = 1;
    capture_request.execution.sampling.temperature    = 0.0F;
    capture_request.execution.allow_prefix_reuse      = true;
    capture_request.stop.include_model_defaults       = false;

    const auto plain_prompt = [](std::string text) {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        return input;
    };
    const auto tool_prompt = [](std::string tool_json, std::string question) {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.options.tool_jsons.push_back(std::move(tool_json));
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = 1,
        });
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        return input;
    };

    std::string observed_text;
    for (std::uint32_t index = 0; index < 4; ++index) { observed_text += "observed-prefix "; }
    const ninfer::GenerationResult observed_first =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    const ninfer::RuntimeStats after_observed_first = engine.runtime_stats();
    const ninfer::GenerationResult observed_second =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    const ninfer::RuntimeStats after_observed_second = engine.runtime_stats();
    if (observed_first.generated_token_ids.size() != 1 ||
        observed_second.generated_token_ids.size() != 1 ||
        after_observed_first.active_captures_completed != 1 ||
        after_observed_second.active_captures_completed !=
            after_observed_first.active_captures_completed + 1U) {
        std::cerr << "observed-prefix private/shared capture sequence changed: "
                  << after_observed_first.active_captures_completed << '/'
                  << after_observed_second.active_captures_completed << '\n';
        return 1;
    }

    const ninfer::GenerationResult observed_filler = engine.generate(
        engine.prepare(plain_prompt("Unrelated private endpoint.")), capture_request);
    const ninfer::GenerationResult observed_reuse =
        engine.generate(engine.prepare(plain_prompt(observed_text)), capture_request);
    if (observed_filler.generated_token_ids.size() != 1 ||
        observed_reuse.generated_token_ids.size() != 1 ||
        observed_reuse.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        observed_reuse.reused_prompt_tokens == 0) {
        std::cerr << "promoted shared prefix was not reusable after private eviction: path="
                  << static_cast<int>(observed_reuse.prefix_reuse_path)
                  << " reused=" << observed_reuse.reused_prompt_tokens << '\n';
        return 1;
    }

    std::string long_description;
    for (std::uint32_t index = 0; index < 240; ++index) { long_description += "stable-schema "; }
    const std::string bravo_tool =
        std::string(R"({"type":"function","function":{"name":"bravo","description":")") +
        long_description +
        R"(","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})";
    const ninfer::GenerationResult replacement = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), capture_request);
    const ninfer::RuntimeStats after_replacement = engine.runtime_stats();
    if (replacement.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not produce its deterministic stop token\n";
        return 1;
    }
    // Remove the exact private endpoint without publishing another shared marker. The following
    // identical Bravo prompt must therefore materialize from the retained shared prefix.
    const ninfer::GenerationResult filled =
        engine.generate(engine.prepare(plain_prompt("Another private endpoint.")), capture_request);
    const ninfer::RuntimeStats after_filler = engine.runtime_stats();
    if (filled.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not displace the exact private endpoint\n";
        return 1;
    }
    ninfer::RequestOptions full_capacity_request = capture_request;
    full_capacity_request.execution.requested_output_tokens =
        std::numeric_limits<std::uint32_t>::max();
    full_capacity_request.stop.token_ids          = {replacement.generated_token_ids.front()};
    full_capacity_request.stop.publish_stop_token = true;
    const ninfer::GenerationResult reused         = engine.generate(
        engine.prepare(tool_prompt(bravo_tool, "Use bravo once.")), full_capacity_request);
    const ninfer::RuntimeStats after_reuse = engine.runtime_stats();

    if (reused.generated_token_ids.size() != 1) {
        std::cerr << "shared replacement fixture did not complete all requests\n";
        return 1;
    }
    if (reused.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        reused.reused_prompt_tokens == 0 || reused.reused_prompt_tokens % 64U == 0) {
        std::cerr << "single-slot shared replacement was not reusable at a non-aligned frontier: "
                  << "path=" << static_cast<int>(reused.prefix_reuse_path)
                  << " reused=" << reused.reused_prompt_tokens
                  << " captures=" << after_replacement.active_captures_completed << '/'
                  << after_filler.active_captures_completed << '/'
                  << after_reuse.active_captures_completed
                  << " shared_evicted=" << after_replacement.pressure_shared_owners_evicted << '/'
                  << after_filler.pressure_shared_owners_evicted << '/'
                  << after_reuse.pressure_shared_owners_evicted
                  << " shared_degraded=" << after_replacement.pressure_shared_owners_degraded << '/'
                  << after_filler.pressure_shared_owners_degraded << '/'
                  << after_reuse.pressure_shared_owners_degraded
                  << " private_evicted=" << after_replacement.pressure_private_owners_evicted << '/'
                  << after_filler.pressure_private_owners_evicted << '/'
                  << after_reuse.pressure_private_owners_evicted
                  << " refs=" << after_replacement.shared_active_references << '/'
                  << after_filler.shared_active_references << '/'
                  << after_reuse.shared_active_references
                  << " targets=" << reused.materialization.targets_evaluated
                  << " degradation=" << reused.materialization.selected_degradation_units
                  << " maximal=" << reused.materialization.selected_maximal_fallback << " stop="
                  << ninfer::materialization_stop_reason_name(reused.materialization.stop_reason)
                  << '\n';
        return 1;
    }
    if (after_replacement.active_captures_completed < 2 ||
        after_reuse.active_captures_completed < after_replacement.active_captures_completed) {
        std::cerr << "shared replacement capture was skipped under full State/KV capacity: "
                  << after_replacement.active_captures_completed << '/'
                  << after_reuse.active_captures_completed << '\n';
        return 1;
    }
    if (after_replacement.shared_active_references != 0 ||
        after_filler.shared_active_references != 0 || after_reuse.shared_active_references != 0) {
        std::cerr << "completed shared-prefix requests leaked active references: "
                  << after_replacement.shared_active_references << '/'
                  << after_filler.shared_active_references << '/'
                  << after_reuse.shared_active_references << '\n';
        return 1;
    }
    return 0;
}

// Issue #251: the shared stable-prefix catalog saturated by automatic (marker-free) traffic.
// With max_shared_prefixes == 1, a second automatic prefix can only be catalogued by reclaiming
// the first; without that reclamation the second is dropped and its shared-prefix reuse freezes
// for the life of the engine. Both prefixes here carry no cache marker, so they are
// automatic-evidence (not explicit-credit) -- the exact traffic the reclaim path serves.
int exercise_shared_saturation_reclaim(const char* artifact) {
    std::cout << "shared-saturation-reclaim: two automatic prefixes against a 1-slot catalog\n";
    ninfer::EngineOptions engine_options = shared_replacement_engine_options(artifact);
    engine_options.max_context           = 1024;
    engine_options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    engine_options.context_cache.max_private_continuations = 1;
    ninfer::Engine engine(std::move(engine_options));
    ninfer::RequestOptions capture_request;
    capture_request.execution.requested_output_tokens = 1;
    capture_request.execution.sampling.temperature    = 0.0F;
    capture_request.execution.allow_prefix_reuse      = true;
    capture_request.stop.include_model_defaults       = false;

    const auto plain_prompt = [](std::string text) {
        ninfer::PromptInput input;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        return input;
    };

    std::string prefix_a;
    for (std::uint32_t index = 0; index < 40; ++index) { prefix_a += "alpha-stable "; }
    std::string prefix_b;
    for (std::uint32_t index = 0; index < 40; ++index) { prefix_b += "bravo-stable "; }

    const auto gen = [&](const std::string& text) {
        return engine.generate(engine.prepare(plain_prompt(text)), capture_request);
    };

    // Prefix A captures and becomes the single shared catalog entry.
    (void)gen(prefix_a);
    (void)gen(prefix_a);
    (void)gen("private filler endpoint one.");
    const auto a_reuse = gen(prefix_a);
    if (a_reuse.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        a_reuse.reused_prompt_tokens == 0) {
        std::cerr << "shared-saturation: first automatic prefix did not reach the shared catalog: "
                  << "path=" << static_cast<int>(a_reuse.prefix_reuse_path)
                  << " reused=" << a_reuse.reused_prompt_tokens << '\n';
        return 1;
    }

    // The catalog is now full. Prefix B (also automatic) can only be catalogued by reclaiming A;
    // without the reclaim B is dropped and its shared-prefix reuse freezes (issue #251).
    const ninfer::RuntimeStats before_b = engine.runtime_stats();
    (void)gen(prefix_b);
    (void)gen(prefix_b);
    (void)gen("private filler endpoint two.");
    const auto b_reuse                 = gen(prefix_b);
    const ninfer::RuntimeStats after_b = engine.runtime_stats();

    if (b_reuse.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        b_reuse.reused_prompt_tokens == 0) {
        std::cerr << "shared-saturation: second automatic prefix froze out after catalog "
                  << "saturation (issue #251): path=" << static_cast<int>(b_reuse.prefix_reuse_path)
                  << " reused=" << b_reuse.reused_prompt_tokens
                  << " shared_evicted=" << after_b.pressure_shared_owners_evicted << '\n';
        return 1;
    }
    if (after_b.pressure_shared_owners_evicted <= before_b.pressure_shared_owners_evicted) {
        std::cerr << "shared-saturation: no shared entry was reclaimed to make room for the "
                  << "second automatic prefix: shared_evicted="
                  << after_b.pressure_shared_owners_evicted << '\n';
        return 1;
    }
    std::cout << "shared-saturation-reclaim: OK -- second automatic prefix reclaimed the first\n";
    return 0;
}

int exercise_anthropic_prefix_regression(const char* artifact) {
    ninfer::Engine engine(anthropic_prefix_regression_engine_options(artifact));
    // The default catalog holds every shared candidate one request can prepare: the explicit
    // markers plus the Engine-automatic tool, instruction and full-prompt candidates.
    if (!engine.options().context_cache.max_shared_prefixes ||
        *engine.options().context_cache.max_shared_prefixes !=
            ninfer::kMaximumPreparedPromptCacheCandidatesPerRequest) {
        std::cerr << "single-concurrency Engine did not size its default shared catalog for one "
                     "request's full candidate set\n";
        return 1;
    }

    const auto conversation = [](bool followup, const ninfer::GenerationResult* first = nullptr) {
        ninfer::PromptInput input;
        input.options.enable_thinking   = true;
        input.options.preserve_thinking = true;
        input.context_cache.session_key = "anthropic-prefix-regression";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;

        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                 .text  = "Reply with exactly the word blue.",
                                                 .media = {}});
        input.messages.push_back(std::move(user));
        if (!followup) { return input; }
        if (first == nullptr) { throw std::logic_error("followup fixture has no source result"); }

        ninfer::ChatMessage assistant;
        assistant.role              = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = first->reasoning;
        if (!first->content.empty()) {
            assistant.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = first->content, .media = {}});
        }
        input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage next;
        next.role = ninfer::ChatRole::User;
        next.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                 .text  = "Now reply with exactly the word green.",
                                                 .media = {}});
        input.messages.push_back(std::move(next));
        return input;
    };
    const auto generation_options = [](std::uint32_t outputs, bool model_stops) {
        ninfer::RequestOptions options;
        options.execution.requested_output_tokens = outputs;
        options.execution.sampling.temperature    = 0.0F;
        options.execution.allow_prefix_reuse      = true;
        options.stop.include_model_defaults       = model_stops;
        return options;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation(false)), generation_options(192, true));
    if (first.reasoning.empty() || first.content.empty() ||
        first.finish_reason != ninfer::FinishReason::StopToken) {
        std::cerr << "endpoint regression fixture did not produce a closed reasoning response: "
                  << "reasoning=" << first.reasoning.size() << " content=" << first.content.size()
                  << " finish=" << static_cast<int>(first.finish_reason) << '\n';
        return 1;
    }
    const ninfer::RuntimeStats before_followup = engine.runtime_stats();
    const ninfer::GenerationResult followup =
        engine.generate(engine.prepare(conversation(true, &first)), generation_options(1, false));
    const ninfer::RuntimeStats after_followup = engine.runtime_stats();
    const std::uint64_t followup_prefill =
        after_followup.computed_prefill_tokens - before_followup.computed_prefill_tokens;
    if (followup.generated_token_ids.size() != 1 ||
        followup.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint ||
        followup.reused_prompt_tokens == 0 ||
        followup_prefill != followup.prompt.prompt_tokens - followup.reused_prompt_tokens) {
        std::cerr << "model-output reasoning frontier did not resume from PrivateEndpoint: path="
                  << static_cast<int>(followup.prefix_reuse_path)
                  << " reused=" << followup.reused_prompt_tokens
                  << " prompt=" << followup.prompt.prompt_tokens << " computed=" << followup_prefill
                  << '\n';
        return 1;
    }

    const auto tool_definition = [](std::string name, std::string word) {
        std::string description;
        for (std::uint32_t index = 0; index < 160; ++index) {
            description += word;
            description.push_back(' ');
        }
        return std::string(R"({"type":"function","function":{"name":")") + name +
               R"(","description":")" + description +
               R"(","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}})";
    };
    const std::string alpha   = tool_definition("alpha", "stable-alpha");
    const std::string bravo   = tool_definition("bravo", "stable-bravo");
    const std::string charlie = tool_definition("charlie", "branch-charlie");
    const auto tool_prompt    = [](std::vector<std::string> tools, std::string question) {
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.options.tool_jsons      = std::move(tools);
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        for (std::uint32_t count = 1; count <= input.options.tool_jsons.size(); ++count) {
            input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                   .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                   .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                   .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
                   .after_tool_count = count,
            });
        }
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
               .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
        input.messages.push_back(std::move(user));
        return input;
    };
    const auto filler_prompt = [](std::string text) {
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        return input;
    };

    const ninfer::RequestOptions one_token = generation_options(1, false);
    const ninfer::GenerationResult seed    = engine.generate(
        engine.prepare(tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninfer::GenerationResult first_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private seed continuation.")), one_token);
    if (seed.generated_token_ids.size() != 1 || first_filler.generated_token_ids.size() != 1) {
        std::cerr << "shared compact fixture did not establish its seed and filler\n";
        return 1;
    }

    const ninfer::RuntimeStats before_late = engine.runtime_stats();
    const ninfer::GenerationResult late    = engine.generate(
        engine.prepare(tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninfer::RuntimeStats after_late = engine.runtime_stats();
    const std::uint64_t late_prefill =
        after_late.computed_prefill_tokens - before_late.computed_prefill_tokens;
    const ninfer::GenerationResult second_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private late continuation.")), one_token);
    const ninfer::RuntimeStats before_branch = engine.runtime_stats();
    const ninfer::GenerationResult branch    = engine.generate(
        engine.prepare(tool_prompt({alpha, charlie}, "Use one listed function.")), one_token);
    const ninfer::RuntimeStats after_branch = engine.runtime_stats();
    const std::uint64_t branch_prefill =
        after_branch.computed_prefill_tokens - before_branch.computed_prefill_tokens;

    if (late.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        branch.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
        late.generated_token_ids.size() != 1 || branch.generated_token_ids.size() != 1 ||
        late.reused_prompt_tokens <= branch.reused_prompt_tokens ||
        branch.reused_prompt_tokens == 0 || second_filler.generated_token_ids.size() != 1 ||
        late_prefill != late.prompt.prompt_tokens - late.reused_prompt_tokens ||
        branch_prefill != branch.prompt.prompt_tokens - branch.reused_prompt_tokens) {
        std::cerr << "default shared catalog did not retain nested compact frontiers: late_path="
                  << static_cast<int>(late.prefix_reuse_path)
                  << " late_reused=" << late.reused_prompt_tokens
                  << " late_prompt=" << late.prompt.prompt_tokens
                  << " late_computed=" << late_prefill
                  << " branch_path=" << static_cast<int>(branch.prefix_reuse_path)
                  << " branch_reused=" << branch.reused_prompt_tokens
                  << " branch_prompt=" << branch.prompt.prompt_tokens
                  << " branch_computed=" << branch_prefill << '\n';
        return 1;
    }
    return 0;
}

int exercise_shared_rewrite_materialization(const char* artifact) {
    ninfer::Engine engine(shared_rewrite_materialization_engine_options(artifact));

    std::vector<std::string> tools;
    tools.reserve(40);
    tools.push_back(
        R"({"type":"function","function":{"name":"read_chunk","description":"Read the next diagnostic chunk. Always use this tool until told done.","parameters":{"type":"object","properties":{"chunk":{"type":"integer"}},"required":["chunk"]}}})");
    for (std::uint32_t index = 1; index < 40; ++index) {
        tools.push_back(
            std::string(R"({"type":"function","function":{"name":"unused_tool_)") +
            std::to_string(index) +
            R"(","description":"Unused diagnostic tool.","parameters":{"type":"object","properties":{"value":{"type":"string"}}}}})");
    }

    constexpr std::string_view system_text =
        "You are testing a tool loop. On every turn call read_chunk exactly once with the next "
        "integer chunk number. Do not finish or answer in prose.";
    std::vector<ninfer::ChatMessage> messages;
    ninfer::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::string(system_text), .media = {}});
    messages.push_back(std::move(system));
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Start by calling read_chunk with chunk 1.",
        .media = {},
    });
    messages.push_back(std::move(user));

    const auto input = [&](bool latest_is_tool) {
        ninfer::PromptInput prompt;
        prompt.messages                  = messages;
        prompt.options.enable_thinking   = true;
        prompt.options.preserve_thinking = true;
        prompt.options.tool_jsons        = tools;
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
            .leading_instruction_bytes = static_cast<std::uint32_t>(system_text.size()),
        });
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(tools.size()),
        });
        if (latest_is_tool) {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(messages.size()),
                .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence            = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        } else {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count      = static_cast<std::uint32_t>(messages.size()),
                .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence                 = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                .after_message_part_count = 1,
            });
        }
        return prompt;
    };
    const auto request_options = [] {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 16384;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.thinking.budget         = 1024;
        request.execution.allow_prefix_reuse      = true;
        return request;
    };
    const auto append_result = [&](const ninfer::GenerationResult& result, std::uint32_t turn) {
        if (result.tool_calls.size() != 1) {
            throw std::logic_error("shared rewrite fixture did not produce one tool call");
        }
        ninfer::ChatMessage assistant;
        assistant.role              = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = result.reasoning + "\nHistory normalized by the client.";
        if (!result.content.empty()) {
            assistant.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        }
        const std::string call_id = "call_" + std::to_string(turn);
        assistant.tool_calls.push_back(ninfer::ToolCall{
            .id             = call_id,
            .name           = result.tool_calls.front().name,
            .arguments_json = result.tool_calls.front().arguments_json,
        });
        messages.push_back(std::move(assistant));

        std::string diagnostic;
        diagnostic.reserve(64000);
        for (std::uint32_t line = 0; line < 500; ++line) {
            diagnostic += "chunk=" + std::to_string(turn) + " line=" + std::to_string(line) +
                          " key=value abcdefghijklmnopqrstuvwxyz0123456789 "
                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ9876543210\n";
        }
        ninfer::ChatMessage tool_result;
        tool_result.role         = ninfer::ChatRole::Tool;
        tool_result.tool_call_id = call_id;
        tool_result.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(diagnostic), .media = {}});
        messages.push_back(std::move(tool_result));
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input(false)), request_options());
    append_result(first, 1);
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(input(true)), request_options());
    append_result(second, 2);
    const ninfer::RuntimeStats before_third = engine.runtime_stats();
    const ninfer::GenerationResult third =
        engine.generate(engine.prepare(input(true)), request_options());
    const ninfer::RuntimeStats after_third = engine.runtime_stats();

    if (first.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        second.reused_prompt_tokens == 0 ||
        third.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        third.reused_prompt_tokens == 0 || third.generated_token_ids.empty() ||
        after_third.historical_fork_hits <= before_third.historical_fork_hits) {
        std::cerr << "shared/private rewrite alias did not materialize through its active Fork: "
                  << "first_path=" << static_cast<int>(first.prefix_reuse_path)
                  << " second_path=" << static_cast<int>(second.prefix_reuse_path)
                  << " second_reused=" << second.reused_prompt_tokens
                  << " third_path=" << static_cast<int>(third.prefix_reuse_path)
                  << " third_reused=" << third.reused_prompt_tokens
                  << " third_outputs=" << third.generated_token_ids.size()
                  << " forks=" << before_third.historical_fork_hits << '/'
                  << after_third.historical_fork_hits << '\n';
        return 1;
    }
    return 0;
}

int exercise_private_long_anchor_capture_and_replacement(const char* artifact) {
    ninfer::Engine engine(private_long_anchor_engine_options(artifact));

    const auto input = [](std::vector<std::string> turns,
                          std::optional<std::uint32_t> marker_after) {
        ninfer::PromptInput prompt;
        for (std::string& text : turns) {
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            message.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking = false;
        if (marker_after) {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = *marker_after,
                .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        }
        return prompt;
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    constexpr std::string_view stable =
        "This is the stable conversation prefix retained for a later branch.";
    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(input({std::string(stable), "Follow the original branch."}, 1)), request);
    if (source.generated_token_ids.size() != 1 ||
        source.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "first private long-anchor capture did not complete from Root\n";
        return 1;
    }

    ninfer::PromptInput replacement_input =
        input({std::string(stable), "Follow the replacement branch.",
               "This suffix belongs only to the replacement source."},
              2);
    // Name the replacement lineage so the final request selects this continuation rather than
    // legitimately preferring the older anonymous branch when the private catalog is full.
    replacement_input.context_cache.session_key = "private-long-anchor-replacement";
    replacement_input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    const ninfer::GenerationResult replacement =
        engine.generate(engine.prepare(std::move(replacement_input)), request);
    if (replacement.generated_token_ids.size() != 1 ||
        replacement.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        replacement.reused_prompt_tokens == 0 ||
        replacement.reused_prompt_tokens >= replacement.prompt.prompt_tokens) {
        std::cerr << "private long anchor was not selected before full-capacity replacement: path="
                  << static_cast<int>(replacement.prefix_reuse_path)
                  << " reused=" << replacement.reused_prompt_tokens
                  << " prompt=" << replacement.prompt.prompt_tokens << '\n';
        return 1;
    }

    ninfer::PromptInput replaced_input =
        input({std::string(stable), "Follow the replacement branch.",
               "Continue through a different branch suffix."},
              std::nullopt);
    replaced_input.context_cache.session_key = "private-long-anchor-replacement";
    replaced_input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    const ninfer::GenerationResult replaced =
        engine.generate(engine.prepare(std::move(replaced_input)), request);
    if (replaced.generated_token_ids.size() != 1 ||
        replaced.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        replaced.reused_prompt_tokens <= replacement.reused_prompt_tokens ||
        replaced.reused_prompt_tokens >= replaced.prompt.prompt_tokens) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "replacement private long anchor was not reusable: path="
                  << static_cast<int>(replaced.prefix_reuse_path)
                  << " first_reused=" << replacement.reused_prompt_tokens
                  << " replaced_reused=" << replaced.reused_prompt_tokens
                  << " prompt=" << replaced.prompt.prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    return 0;
}

int exercise_engine_automatic_long_anchor_capture(const char* artifact) {
    ninfer::Engine engine(automatic_long_anchor_engine_options(artifact));

    const auto input = [](std::vector<std::string> turns) {
        ninfer::PromptInput prompt;
        for (std::string& text : turns) {
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            message.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking = false;
        return prompt;
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    constexpr std::string_view stable =
        "This is the stable conversation prefix retained for a later branch.";
    constexpr std::string_view branch = "Continue through the shared middle branch.";

    // No markers anywhere: the engine synthesizes the anchors at the last two message
    // boundaries of each request, so the first request must complete from Root and leave
    // the middle-boundary anchor behind for the later branches.
    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(input({std::string(stable), std::string(branch), "First unique suffix."})),
        request);
    if (source.generated_token_ids.size() != 1 ||
        source.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "first engine-automatic long-anchor capture did not complete from Root\n";
        return 1;
    }

    const auto followup = [&](std::string_view suffix) {
        ninfer::PromptInput prompt =
            input({std::string(stable), std::string(branch), std::string(suffix)});
        prompt.context_cache.session_key = "engine-automatic-long-anchor";
        prompt.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return engine.generate(engine.prepare(std::move(prompt)), request);
    };
    const ninfer::GenerationResult second = followup("Second unique suffix.");
    if (second.generated_token_ids.size() != 1 ||
        second.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        second.reused_prompt_tokens == 0 ||
        second.reused_prompt_tokens >= second.prompt.prompt_tokens) {
        std::cerr << "engine-automatic long anchor was not selected: path="
                  << static_cast<int>(second.prefix_reuse_path)
                  << " reused=" << second.reused_prompt_tokens
                  << " prompt=" << second.prompt.prompt_tokens << '\n';
        return 1;
    }
    // The third branch shares only the middle boundary with the first request; it must resume
    // from the same retained anchor rather than from the second request's deeper state.
    const ninfer::GenerationResult third = followup("Third unique suffix.");
    if (third.generated_token_ids.size() != 1 ||
        third.prefix_reuse_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        third.reused_prompt_tokens != second.reused_prompt_tokens) {
        std::cerr << "engine-automatic long anchor did not serve the third branch: path="
                  << static_cast<int>(third.prefix_reuse_path)
                  << " second_reused=" << second.reused_prompt_tokens
                  << " third_reused=" << third.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_salvage_mid_prefill(const char* artifact) {
    ninfer::Engine engine(salvage_mid_prefill_engine_options(artifact));

    const auto input = [](std::string text) {
        ninfer::PromptInput prompt;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        prompt.messages.push_back(std::move(user));
        prompt.options.enable_thinking = false;
        return prompt;
    };
    const auto token_count = [&](const std::string& text) {
        return engine.count_tokens(input(text));
    };

    // Grow the prompt until prefill spans at least two full 1024-token chunks, so a cancel
    // sampled at a chunk boundary lands mid-prefill with a committed frontier of at least
    // kSalvageMinFrontier.
    std::string prompt_text;
    for (std::uint32_t index = 0; index < 800; ++index) { prompt_text += "alpha "; }
    while (token_count(prompt_text) < 2048) { prompt_text += "beta gamma delta epsilon zeta; "; }

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    ninfer::PromptInput cancelled_input       = input(prompt_text);
    cancelled_input.context_cache.session_key = "salvage-mid-prefill";
    cancelled_input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;

    // The engine samples the flag once per execution unit. Admission takes milliseconds, while
    // the first 1024-token chunk takes well over 100 ms, so a deadline after 75 ms cancels the
    // request mid-prefill rather than before admission or after the prompt is fully prefilled.
    const auto cancel_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(75);
    const ninfer::CancellationView cancellation{
        [cancel_deadline] { return std::chrono::steady_clock::now() >= cancel_deadline; }};

    const ninfer::RuntimeStats before = engine.runtime_stats();
    const ninfer::GenerationResult cancelled =
        engine.generate(engine.prepare(std::move(cancelled_input)), request, nullptr, cancellation);
    const ninfer::RuntimeStats after_cancel = engine.runtime_stats();
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled ||
        cancelled.prompt.prompt_tokens < 2048 ||
        after_cancel.salvaged_continuations != before.salvaged_continuations + 1) {
        std::cerr << "mid-prefill cancel did not salvage the committed frontier: reason="
                  << static_cast<int>(cancelled.finish_reason)
                  << " prompt=" << cancelled.prompt.prompt_tokens
                  << " salvaged=" << before.salvaged_continuations << '/'
                  << after_cancel.salvaged_continuations << '\n';
        return 1;
    }

    // The retry names the same session, so it must resume from the salvaged endpoint instead of
    // reprefilling from Root.
    ninfer::PromptInput retry_input       = input(prompt_text);
    retry_input.context_cache.session_key = "salvage-mid-prefill";
    retry_input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    const ninfer::GenerationResult retry =
        engine.generate(engine.prepare(std::move(retry_input)), request);
    if (retry.generated_token_ids.size() != 1 ||
        retry.prefix_reuse_path == ninfer::PrefixReusePath::Root ||
        retry.reused_prompt_tokens < 1024 ||
        retry.reused_prompt_tokens >= retry.prompt.prompt_tokens) {
        std::cerr << "retry did not reuse the salvaged frontier: path="
                  << static_cast<int>(retry.prefix_reuse_path)
                  << " reused=" << retry.reused_prompt_tokens
                  << " prompt=" << retry.prompt.prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_last_private_alias_eviction(const char* artifact) {
    ninfer::Engine engine(last_alias_engine_options(artifact));
    std::string prompt_text;
    prompt_text.reserve(6U * 300U + 8U);
    for (std::uint32_t index = 0; index < 300; ++index) { prompt_text += "alpha "; }

    const auto input = [&](std::string session, ninfer::CacheRetentionHint retention) {
        ninfer::PromptInput prompt;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = prompt_text, .media = {}});
        prompt.messages.push_back(std::move(user));
        prompt.options.enable_thinking   = false;
        prompt.context_cache.session_key = std::move(session);
        prompt.context_cache.retention   = retention;
        return prompt;
    };
    if (engine.count_tokens(input("probe", ninfer::CacheRetentionHint::Disposable)) % 64U == 0) {
        prompt_text += "beta";
    }

    ninfer::RequestOptions one_token;
    one_token.execution.requested_output_tokens = 1;
    one_token.execution.sampling.temperature    = 0.0F;
    one_token.execution.allow_prefix_reuse      = true;
    one_token.stop.include_model_defaults       = false;

    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(input("last-alias-source", ninfer::CacheRetentionHint::LiveSession)),
        one_token);
    const ninfer::GenerationResult branch = engine.generate(
        engine.prepare(input("last-alias-branch", ninfer::CacheRetentionHint::Disposable)),
        one_token);
    if (source.generated_token_ids.size() != 1 || branch.generated_token_ids.size() != 1 ||
        branch.reused_prompt_tokens == 0 ||
        (branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint)) {
        std::cerr << "last-alias fixture did not establish two private prefix aliases: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens << '\n';
        return 1;
    }

    ninfer::RequestOptions full_capacity            = one_token;
    full_capacity.execution.requested_output_tokens = std::numeric_limits<std::uint32_t>::max();
    full_capacity.stop.token_ids                    = {source.generated_token_ids.front()};
    full_capacity.stop.publish_stop_token           = true;
    const ninfer::RuntimeStats before               = engine.runtime_stats();
    const ninfer::GenerationResult consumed         = engine.generate(
        engine.prepare(input("last-alias-source", ninfer::CacheRetentionHint::LiveSession)),
        full_capacity);
    const ninfer::RuntimeStats after = engine.runtime_stats();
    if (consumed.generated_token_ids.size() != 1 || consumed.reused_prompt_tokens == 0 ||
        (consumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         consumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint) ||
        after.pressure_private_owners_evicted <= before.pressure_private_owners_evicted ||
        after.device_main_kv_occupied_pages == 0 || after.device_main_kv_occupied_pages > 8 ||
        after.device_backend_kv_occupied_pages == 0 || after.device_backend_kv_occupied_pages > 8) {
        std::cerr << "last private prefix alias did not transfer into the active entitlement: path="
                  << static_cast<int>(consumed.prefix_reuse_path)
                  << " reused=" << consumed.reused_prompt_tokens
                  << " evictions=" << before.pressure_private_owners_evicted << '/'
                  << after.pressure_private_owners_evicted
                  << " main=" << after.device_main_kv_occupied_pages
                  << " backend=" << after.device_backend_kv_occupied_pages << '\n';
        return 1;
    }
    return 0;
}

enum class RewriteCheckpointCacheTopology : std::uint8_t {
    PrivateOnly,
    SharedAlias,
};

int exercise_rewrite_checkpoints(ninfer::Engine& engine, RewriteCheckpointCacheTopology topology) {
    const bool shared_alias = topology == RewriteCheckpointCacheTopology::SharedAlias;
    const ninfer::RuntimeStats initial_stats = engine.runtime_stats();

    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto assistant_call = [&](std::string reasoning, std::string id, std::string key) {
        ninfer::ChatMessage message = text_message(ninfer::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(ninfer::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    auto input_with_history = [&](int completed_responses, bool preserve_thinking) {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (completed_responses >= 1) {
            input.messages.push_back(
                assistant_call("The first lookup should be alpha.", "call_alpha", "alpha"));
            ninfer::ChatMessage tool =
                text_message(ninfer::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
            tool.tool_call_id = "call_alpha";
            input.messages.push_back(std::move(tool));
        }
        if (completed_responses >= 2) {
            input.messages.push_back(
                assistant_call("The alpha result requests beta.", "call_beta", "beta"));
            ninfer::ChatMessage tool = text_message(ninfer::ChatRole::Tool, "{\"value\":25}");
            tool.tool_call_id        = "call_beta";
            input.messages.push_back(std::move(tool));
        }
        input.options.preserve_thinking = preserve_thinking;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return input;
    };
    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 4;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input_with_history(0, true)), options(true));
    if (first.generated_token_ids.size() != 4 ||
        first.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "response-checkpoint source request did not complete from a cold lane\n";
        return 1;
    }

    const ninfer::GenerationResult exact_replay =
        engine.generate(engine.prepare(input_with_history(0, false)), options(true));
    if (exact_replay.generated_token_ids.size() != 4 ||
        exact_replay.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        exact_replay.reused_prompt_tokens == 0) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "pre-generation response checkpoint was not restored on an exact replay: "
                  << "path=" << static_cast<int>(exact_replay.prefix_reuse_path)
                  << " reused=" << exact_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    const ninfer::GenerationResult exact_baseline =
        engine.generate(engine.prepare(input_with_history(0, false)), options(false));
    // Replay can rebuild the MTP bridge at T=1; capture can also split the source prefill.
    // Each route must finish within its own budget and publish a valid reuse frontier.
    if (exact_baseline.generated_token_ids.size() != 4 ||
        exact_baseline.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        exact_baseline.reused_prompt_tokens != 0) {
        std::cerr << "uncached response-checkpoint baseline did not complete from Root: path="
                  << static_cast<int>(exact_baseline.prefix_reuse_path)
                  << " reused=" << exact_baseline.reused_prompt_tokens
                  << " outputs=" << exact_baseline.generated_token_ids.size() << '\n';
        return 1;
    }

    const ninfer::RuntimeStats before_first_replay = engine.runtime_stats();
    const ninfer::GenerationResult first_replay =
        engine.generate(engine.prepare(input_with_history(1, true)), options(true));
    const ninfer::RuntimeStats after_first_replay = engine.runtime_stats();
    const ninfer::PrefixReusePath expected_first_replay =
        shared_alias ? ninfer::PrefixReusePath::SharedStablePrefix
                     : ninfer::PrefixReusePath::PrivateResponseReplay;
    if (first_replay.generated_token_ids.size() != 4 ||
        first_replay.prefix_reuse_path != expected_first_replay ||
        first_replay.reused_prompt_tokens == 0 ||
        (shared_alias && first_replay.reused_prompt_tokens <= exact_replay.reused_prompt_tokens)) {
        std::cerr << "normalized first response selected the wrong cache frontier: path="
                  << static_cast<int>(first_replay.prefix_reuse_path)
                  << " expected=" << static_cast<int>(expected_first_replay)
                  << " reused=" << first_replay.reused_prompt_tokens << '\n';
        return 1;
    }
    if (shared_alias &&
        after_first_replay.historical_fork_hits <= before_first_replay.historical_fork_hits &&
        after_first_replay.state_restores <= before_first_replay.state_restores) {
        std::cerr << "shared rewrite source had no StateImage materialization transition: forks="
                  << before_first_replay.historical_fork_hits << '/'
                  << after_first_replay.historical_fork_hits
                  << " restores=" << before_first_replay.state_restores << '/'
                  << after_first_replay.state_restores << '\n';
        return 1;
    }

    const ninfer::GenerationResult second_replay =
        engine.generate(engine.prepare(input_with_history(2, true)), options(true));
    if (second_replay.generated_token_ids.size() != 4 ||
        second_replay.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        second_replay.reused_prompt_tokens <= first_replay.reused_prompt_tokens) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        std::cerr << "rolling response checkpoint did not advance across the tool loop: first="
                  << first_replay.reused_prompt_tokens
                  << " second=" << second_replay.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }

    const ninfer::GenerationResult mode_change =
        engine.generate(engine.prepare(input_with_history(2, false)), options(true));
    if (mode_change.generated_token_ids.size() != 4 ||
        mode_change.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay ||
        mode_change.reused_prompt_tokens == 0) {
        std::cerr << "preserve-thinking policy change discarded a compatible response checkpoint: "
                  << "path=" << static_cast<int>(mode_change.prefix_reuse_path)
                  << " reused=" << mode_change.reused_prompt_tokens << '\n';
        return 1;
    }

    if (shared_alias) {
        const ninfer::RuntimeStats final_stats   = engine.runtime_stats();
        const std::uint64_t initial_degradations = initial_stats.pressure_private_owners_degraded +
                                                   initial_stats.pressure_shared_owners_degraded;
        const std::uint64_t final_degradations = final_stats.pressure_private_owners_degraded +
                                                 final_stats.pressure_shared_owners_degraded;
        if (final_degradations <= initial_degradations ||
            final_stats.pressure_private_owners_evicted !=
                initial_stats.pressure_private_owners_evicted ||
            final_stats.pressure_shared_owners_evicted !=
                initial_stats.pressure_shared_owners_evicted ||
            final_stats.pressure_checkpoints_dropped !=
                initial_stats.pressure_checkpoints_dropped ||
            final_stats.active_captures_aborted != initial_stats.active_captures_aborted) {
            std::cerr << "shared/rewrite rotation did not preserve both cache owners: degraded="
                      << initial_degradations << '/' << final_degradations
                      << " private_evicted=" << initial_stats.pressure_private_owners_evicted << '/'
                      << final_stats.pressure_private_owners_evicted
                      << " shared_evicted=" << initial_stats.pressure_shared_owners_evicted << '/'
                      << final_stats.pressure_shared_owners_evicted
                      << " checkpoint_drops=" << initial_stats.pressure_checkpoints_dropped << '/'
                      << final_stats.pressure_checkpoints_dropped
                      << " capture_aborts=" << initial_stats.active_captures_aborted << '/'
                      << final_stats.active_captures_aborted << '\n';
            return 1;
        }
    }

    return 0;
}

int exercise_rewrite_branch(const char* artifact) {
    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    const auto input = [&](bool branch) {
        ninfer::PromptInput value;
        value.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (branch) {
            value.messages.push_back(text_message(ninfer::ChatRole::User,
                                                  "Summarize the conversation before answering."));
        }
        value.options.preserve_thinking = true;
        value.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return value;
    };
    const auto options = [](bool reuse) {
        ninfer::RequestOptions value;
        value.execution.requested_output_tokens = 4;
        value.execution.sampling.temperature    = 0.0F;
        value.execution.allow_prefix_reuse      = reuse;
        value.stop.include_model_defaults       = false;
        return value;
    };

    ninfer::EngineOptions configured             = engine_options(artifact);
    configured.context_cache.device_state_slots  = 2;
    configured.context_cache.max_shared_prefixes = 0;
    ninfer::Engine engine(std::move(configured));
    const ninfer::GenerationResult source =
        engine.generate(engine.prepare(input(false)), options(true));
    if (source.generated_token_ids.size() != 4 ||
        source.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "rewrite branch source did not establish a response checkpoint\n";
        return 1;
    }
    const ninfer::GenerationResult branch =
        engine.generate(engine.prepare(input(true)), options(true));
    const ninfer::GenerationResult branch_baseline =
        engine.generate(engine.prepare(input(true)), options(false));
    if (branch.generated_token_ids.size() != 4 || branch.reused_prompt_tokens == 0 ||
        branch.reused_prompt_tokens >= branch.prompt.prompt_tokens ||
        (branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateResponseReplay &&
         branch.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure) ||
        branch_baseline.generated_token_ids.size() != 4 ||
        branch_baseline.reused_prompt_tokens != 0) {
        std::cerr << "replacement user suffix did not reuse the stable conversation prefix: path="
                  << static_cast<int>(branch.prefix_reuse_path)
                  << " reused=" << branch.reused_prompt_tokens
                  << " prompt=" << branch.prompt.prompt_tokens
                  << " target_count=" << branch.materialization.targets_evaluated
                  << " degradation=" << branch.materialization.selected_degradation_units
                  << " maximal=" << branch.materialization.selected_maximal_fallback << " stop="
                  << ninfer::materialization_stop_reason_name(branch.materialization.stop_reason)
                  << " now_ns=" << branch.materialization.predicted_now_ns
                  << " future_ns=" << branch.materialization.predicted_future_loss_ns << '\n';
        return 1;
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    const auto image_bytes = gradient_ppm();
    auto image_part        = [](const std::vector<std::uint8_t>& bytes, std::string name) {
        ninfer::MessagePart image;
        image.kind              = ninfer::MessagePartKind::Media;
        image.media.kind        = ninfer::MediaKind::Image;
        image.media.bytes       = bytes;
        image.media.media_type  = "image/x-portable-pixmap";
        image.media.source_name = std::move(name);
        return image;
    };
    auto assistant_message = [](const ninfer::GenerationResult& result) {
        ninfer::ChatMessage message;
        message.role              = ninfer::ChatRole::Assistant;
        message.reasoning_content = result.reasoning;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        return message;
    };
    auto first_input = [&](const std::vector<std::uint8_t>& bytes) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(image_part(bytes, "inline.ppm"));
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "vision-prefix-real";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return input;
    };
    auto followup_input = [&](const std::vector<std::uint8_t>& bytes,
                              const ninfer::GenerationResult& first) {
        ninfer::PromptInput input = first_input(bytes);
        input.messages.push_back(assistant_message(first));
        ninfer::ChatMessage followup;
        followup.role = ninfer::ChatRole::User;
        followup.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "Give one more detail.", .media = {}});
        input.messages.push_back(std::move(followup));
        return input;
    };
    auto appended_media_input =
        [&](const std::vector<std::uint8_t>& old_bytes, const ninfer::GenerationResult& first,
            const ninfer::GenerationResult& second, const std::vector<std::uint8_t>& new_bytes) {
            ninfer::PromptInput input = followup_input(old_bytes, first);
            input.messages.push_back(assistant_message(second));
            ninfer::ChatMessage followup;
            followup.role = ninfer::ChatRole::User;
            followup.parts.push_back(image_part(new_bytes, "second.ppm"));
            followup.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = "Compare the images.", .media = {}});
            input.messages.push_back(std::move(followup));
            return input;
        };

    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 2;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    // The 1024 merged Vision columns begin after the chat prefix, so the same item necessarily
    // crosses a 1024-token prefill boundary. Its host payload may be released after the first
    // encode, while later chunks must continue to reuse the resident Vision transient.
    ninfer::RequestOptions cross_chunk_options            = options(false);
    cross_chunk_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult cross_chunk =
        engine.generate(engine.prepare(first_input(gradient_ppm(1024, 1024))), cross_chunk_options);
    if (!cross_chunk.prompt.has_media || cross_chunk.generated_token_ids.size() != 1) {
        std::cerr << "cross-chunk Vision item did not complete after releasing its host payload\n";
        return 1;
    }

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_input(image_bytes)), options(true));
    if (!first.prompt.has_media || first.generated_token_ids.size() != 2 ||
        first.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "real Vision request did not complete through the public Engine\n";
        return 1;
    }

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(followup_input(image_bytes, first)), options(true));
    if (reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0.0 ||
        reused.generated_token_ids.size() != 2) {
        std::cerr << "same-media continuation did not reuse the resident Vision prefix: reused="
                  << reused.reused_prompt_tokens << " vision=" << reused.timings.vision_seconds
                  << '\n';
        return 1;
    }

    std::vector<std::uint8_t> second_image = image_bytes;
    second_image.back() ^= 0x5aU;
    const ninfer::GenerationResult appended = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(true));
    if (appended.reused_prompt_tokens == 0 || !(appended.timings.vision_seconds > 0.0) ||
        appended.generated_token_ids.size() != 2) {
        std::cerr << "new-media suffix did not preserve the old multimodal prefix: reused="
                  << appended.reused_prompt_tokens << " vision=" << appended.timings.vision_seconds
                  << '\n';
        return 1;
    }

    const ninfer::GenerationResult baseline = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(false));
    if (baseline.generated_token_ids.size() != 2 || baseline.reused_prompt_tokens != 0 ||
        !(baseline.timings.vision_seconds > 0.0)) {
        std::cerr << "uncached multimodal prefill did not recompute its media\n";
        return 1;
    }

    std::vector<std::uint8_t> changed_prefix = image_bytes;
    changed_prefix[changed_prefix.size() - 2] ^= 0x33U;
    const ninfer::GenerationResult miss = engine.generate(
        engine.prepare(appended_media_input(changed_prefix, first, reused, second_image)),
        options(true));
    if (miss.reused_prompt_tokens != 0) {
        std::cerr << "changed media content incorrectly reused placeholder-token KV\n";
        return 1;
    }

    ninfer::RequestOptions mtp_options            = options(false);
    mtp_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult mtp_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), mtp_options);
    if (mtp_baseline.generated_token_ids.size() != 5 ||
        mtp_baseline.generated_token_ids[0] == mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal stop fixture did not produce distinct leading tokens\n";
        return 1;
    }
    ninfer::RequestOptions stop_options       = mtp_options;
    stop_options.execution.allow_prefix_reuse = true;
    stop_options.stop.token_ids.push_back(mtp_baseline.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare(first_input(image_bytes)), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.empty() || stopped.speculative.rounds == 0 ||
        stopped.generated_token_ids.back() != stop_options.stop.token_ids.front()) {
        std::cerr << "multimodal custom stop did not terminate at the selected token\n";
        return 1;
    }
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare(followup_input(image_bytes, stopped)), options(true));
    if (stopped_reuse.reused_prompt_tokens == 0 || stopped_reuse.timings.vision_seconds != 0.0) {
        std::cerr << "multimodal stop discarded its reusable boundary: reused="
                  << stopped_reuse.reused_prompt_tokens
                  << " vision=" << stopped_reuse.timings.vision_seconds << '\n';
        return 1;
    }

    // Exact registered rendering prefix before the first image-pad column:
    // <|im_start|>user\n<|vision_start|>. Reusing it places the MTP bridge directly on the first
    // Vision merger column rather than on an ordinary token embedding.
    const std::vector<ninfer::TokenId> visual_prefix{248045, 846, 198, 248053};
    ninfer::RequestOptions source_options            = options(true);
    source_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult bridge_source =
        engine.generate(engine.prepare_tokens(visual_prefix), source_options);
    ninfer::RequestOptions bridge_options            = options(true);
    bridge_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult visual_bridge =
        engine.generate(engine.prepare(first_input(image_bytes)), bridge_options);
    if (bridge_source.generated_token_ids.size() != 1 ||
        visual_bridge.reused_prompt_tokens != visual_prefix.size() ||
        !(visual_bridge.timings.vision_seconds > 0.0) || visual_bridge.speculative.rounds == 0) {
        std::cerr << "visual MTP bridge did not append the prefix and enter speculative decode: "
                  << "source_outputs=" << bridge_source.generated_token_ids.size()
                  << " reused=" << visual_bridge.reused_prompt_tokens
                  << " vision=" << visual_bridge.timings.vision_seconds
                  << " rounds=" << visual_bridge.speculative.rounds
                  << " fallbacks=" << visual_bridge.speculative.fallback_steps << '\n';
        return 1;
    }
    if (visual_bridge.generated_token_ids.size() != 5 ||
        visual_bridge.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "visual MTP bridge did not commit the requested output budget\n";
        return 1;
    }
    const auto bridge_followup =
        engine.generate(engine.prepare(followup_input(image_bytes, visual_bridge)), options(true));
    if (bridge_followup.reused_prompt_tokens == 0 ||
        bridge_followup.timings.vision_seconds != 0.0 ||
        bridge_followup.generated_token_ids.size() != 2) {
        std::cerr << "visual MTP bridge lost its retained continuation\n";
        return 1;
    }
    return 0;
}

ninfer::PromptInput session_turn(std::string session, std::string question) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = std::move(session);
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    return input;
}

ninfer::PromptInput pressure_turn(std::string text, std::string session,
                                  ninfer::CacheRetentionHint retention) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    if (!session.empty()) { input.context_cache.session_key = std::move(session); }
    input.context_cache.retention = retention;
    return input;
}

std::optional<std::string> exact_repeated_prompt_text(const ninfer::Engine& engine,
                                                      std::uint32_t target_tokens,
                                                      std::string_view word) {
    const auto text = [word](std::uint32_t repetitions) {
        std::string value;
        value.reserve(static_cast<std::size_t>(repetitions) * (word.size() + 1U));
        for (std::uint32_t index = 0; index < repetitions; ++index) {
            value.push_back(' ');
            value.append(word);
        }
        return value;
    };
    const auto count = [&](std::uint32_t repetitions) {
        return engine.count_tokens(
            pressure_turn(text(repetitions), "", ninfer::CacheRetentionHint::Disposable));
    };

    std::uint32_t low  = 0;
    std::uint32_t high = target_tokens;
    while (low <= high) {
        const std::uint32_t middle = low + (high - low) / 2U;
        const std::uint32_t tokens = count(middle);
        if (tokens == target_tokens) { return text(middle); }
        if (tokens < target_tokens) {
            low = middle + 1U;
        } else {
            if (middle == 0) { break; }
            high = middle - 1U;
        }
    }
    return std::nullopt;
}

ninfer::RequestOptions fixed_output(std::uint32_t tokens, bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int exercise_pressure_partial_spill_and_resume(const char* artifact) {
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 31;
    constexpr std::uint32_t kShortPromptTokens = 350;
    ninfer::Engine engine(pressure_resume_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_a_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    const std::optional<std::string> short_b_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "charlie");
    if (!long_text || !short_a_text || !short_b_text) {
        std::cerr << "pressure-resume fixture could not construct exact prompt geometry\n";
        return 1;
    }

    const ninfer::GenerationResult long_result = engine.generate(
        engine.prepare(pressure_turn(*long_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(kLongOutputTokens));
    if (long_result.prompt.prompt_tokens != kLongPromptTokens ||
        long_result.generated_token_ids.size() != kLongOutputTokens) {
        std::cerr << "pressure-resume long source did not establish its 121-page endpoint: prompt="
                  << long_result.prompt.prompt_tokens
                  << " output=" << long_result.generated_token_ids.size() << '\n';
        return 1;
    }

    const ninfer::GenerationResult short_a =
        engine.generate(engine.prepare(pressure_turn(*short_a_text, "pressure-short-a",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    if (short_a.prompt.prompt_tokens != kShortPromptTokens ||
        short_a.generated_token_ids.size() != 1) {
        std::cerr << "pressure-resume short source did not establish its six-page reservation\n";
        return 1;
    }

    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult short_b =
        engine.generate(engine.prepare(pressure_turn(*short_b_text, "pressure-short-b",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    const std::uint64_t pressure_main_pages =
        after_pressure.main_kv_d2h_pages - before_pressure.main_kv_d2h_pages;
    const std::uint64_t pressure_spill_pages =
        after_pressure.pressure_spill_pages - before_pressure.pressure_spill_pages;
    const std::uint64_t pressure_drops =
        after_pressure.pressure_checkpoints_dropped - before_pressure.pressure_checkpoints_dropped;
    const std::uint64_t pressure_degraded = after_pressure.pressure_private_owners_degraded -
                                            before_pressure.pressure_private_owners_degraded;
    const std::uint64_t pressure_evicted = after_pressure.pressure_private_owners_evicted -
                                           before_pressure.pressure_private_owners_evicted;
    if (short_b.generated_token_ids.size() != 1 || pressure_main_pages != 4 ||
        pressure_spill_pages != 4 || pressure_drops != 1 || pressure_degraded != 1 ||
        pressure_evicted != 0 ||
        after_pressure.state_d2h_count != before_pressure.state_d2h_count ||
        short_b.materialization.selected_maximal_fallback) {
        std::cerr << "pressure-resume did not select endpoint-drop plus four-page spill: main="
                  << pressure_main_pages << " spill=" << pressure_spill_pages
                  << " drops=" << pressure_drops << " degraded=" << pressure_degraded
                  << " evicted=" << pressure_evicted
                  << " state=" << (after_pressure.state_d2h_count - before_pressure.state_d2h_count)
                  << " device_pages=" << before_pressure.device_main_kv_occupied_pages << '/'
                  << after_pressure.device_main_kv_occupied_pages
                  << " maximal=" << short_b.materialization.selected_maximal_fallback
                  << " budget=" << short_b.materialization.budget_exhausted << '\n';
        return 1;
    }
    const ninfer::RuntimeStats before_resume = engine.runtime_stats();
    const ninfer::GenerationResult resumed   = engine.generate(
        engine.prepare(pressure_turn(*long_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(1));
    const ninfer::RuntimeStats after_resume = engine.runtime_stats();
    const std::uint64_t restored_pages =
        after_resume.main_kv_h2d_pages - before_resume.main_kv_h2d_pages;
    const std::uint32_t reused_pages = (resumed.reused_prompt_tokens + 63U) / 64U;
    if (resumed.generated_token_ids.size() != 1 ||
        resumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
        reused_pages != 120 || restored_pages != 4) {
        std::cerr << "pressure-resume did not restore the retained turn closure: path="
                  << static_cast<int>(resumed.prefix_reuse_path)
                  << " reused=" << resumed.reused_prompt_tokens << " reused_pages=" << reused_pages
                  << " restored=" << restored_pages << '\n';
        return 1;
    }
    return 0;
}

int exercise_materialization_source_pressure_protection(const char* artifact) {
    // The source occupies 121 pages and the second owner occupies six, leaving one free page. The
    // branch reuses the source's 120-page turn closure but needs two suffix pages. Under the old
    // guided closure, the source's unprotected endpoint tail was selected as the one-page Host KV
    // victim even though the same continuation was the materialization source.
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 31;
    constexpr std::uint32_t kShortPromptTokens = 350;
    ninfer::Engine engine(pressure_resume_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    if (!long_text || !short_text) {
        std::cerr << "source-pressure fixture could not construct exact prompt geometry\n";
        return 1;
    }

    const ninfer::GenerationResult source =
        engine.generate(engine.prepare(pressure_turn(*long_text, "source-pressure-origin",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(kLongOutputTokens));
    const ninfer::GenerationResult resident =
        engine.generate(engine.prepare(pressure_turn(*short_text, "source-pressure-resident",
                                                     ninfer::CacheRetentionHint::LiveSession)),
                        fixed_output(1));
    const ninfer::RuntimeStats before_branch = engine.runtime_stats();
    if (source.prompt.prompt_tokens != kLongPromptTokens ||
        source.generated_token_ids.size() != kLongOutputTokens ||
        resident.prompt.prompt_tokens != kShortPromptTokens ||
        resident.generated_token_ids.size() != 1 ||
        before_branch.device_main_kv_occupied_pages != 127) {
        std::cerr << "source-pressure fixture did not establish 127 resident pages: source="
                  << source.prompt.prompt_tokens << '+' << source.generated_token_ids.size()
                  << " resident=" << resident.prompt.prompt_tokens << '+'
                  << resident.generated_token_ids.size()
                  << " pages=" << before_branch.device_main_kv_occupied_pages << '\n';
        return 1;
    }

    ninfer::PromptInput branch = pressure_turn(*long_text, "source-pressure-branch",
                                               ninfer::CacheRetentionHint::LiveSession);
    std::string suffix;
    for (std::uint32_t index = 0; index < 96; ++index) { suffix += " delta"; }
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(suffix), .media = {}});
    branch.messages.push_back(std::move(followup));

    const ninfer::GenerationResult branched =
        engine.generate(engine.prepare(std::move(branch)), fixed_output(1));
    const ninfer::RuntimeStats after_branch = engine.runtime_stats();
    const std::uint64_t demoted_pages =
        after_branch.main_kv_d2h_pages - before_branch.main_kv_d2h_pages;
    const std::uint64_t degraded = after_branch.pressure_private_owners_degraded -
                                   before_branch.pressure_private_owners_degraded;
    const bool private_partial_source =
        branched.prefix_reuse_path == ninfer::PrefixReusePath::PrivateResponseReplay ||
        branched.prefix_reuse_path == ninfer::PrefixReusePath::PrivateTurnClosure;
    if (branched.generated_token_ids.size() != 1 || !private_partial_source ||
        branched.reused_prompt_tokens == 0 ||
        branched.reused_prompt_tokens >= branched.prompt.prompt_tokens || demoted_pages == 0 ||
        degraded == 0 || branched.materialization.selected_maximal_fallback) {
        std::cerr << "source-pressure branch did not preserve its source under guided Host KV "
                     "pressure: path="
                  << static_cast<int>(branched.prefix_reuse_path)
                  << " reused=" << branched.reused_prompt_tokens
                  << " prompt=" << branched.prompt.prompt_tokens << " demoted=" << demoted_pages
                  << " degraded=" << degraded
                  << " maximal=" << branched.materialization.selected_maximal_fallback << " stop="
                  << ninfer::materialization_stop_reason_name(branched.materialization.stop_reason)
                  << '\n';
        return 1;
    }
    return 0;
}

int exercise_private_checkpoint_pressure_retention(const char* artifact) {
    constexpr std::uint32_t kLongPromptTokens  = 7683;
    constexpr std::uint32_t kLongOutputTokens  = 16;
    constexpr std::uint32_t kShortPromptTokens = 350;
    constexpr std::uint32_t kShortOutputTokens = 256;
    ninfer::Engine engine(private_checkpoint_pressure_engine_options(artifact));

    const std::optional<std::string> long_text =
        exact_repeated_prompt_text(engine, kLongPromptTokens, "alpha");
    const std::optional<std::string> short_b_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "bravo");
    const std::optional<std::string> short_c_text =
        exact_repeated_prompt_text(engine, kShortPromptTokens, "charlie");
    if (!long_text || !short_b_text || !short_c_text) {
        std::cerr << "private-checkpoint pressure fixture could not construct prompt geometry\n";
        return 1;
    }

    const std::string session             = "private-checkpoint-pressure-source";
    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(pressure_turn(*long_text, session, ninfer::CacheRetentionHint::LiveSession)),
        fixed_output(kLongOutputTokens));
    if (source.prompt.prompt_tokens != kLongPromptTokens ||
        source.generated_token_ids.size() != kLongOutputTokens) {
        std::cerr << "private-checkpoint pressure source did not establish its long session\n";
        return 1;
    }

    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    auto short_b                               = engine.submit(
        engine.prepare(pressure_turn(*short_b_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(kShortOutputTokens));
    auto short_c = engine.submit(
        engine.prepare(pressure_turn(*short_c_text, "", ninfer::CacheRetentionHint::Disposable)),
        fixed_output(kShortOutputTokens));
    const ninfer::GenerationResult short_b_result = short_b.wait();
    const ninfer::GenerationResult short_c_result = short_c.wait();
    const ninfer::RuntimeStats after_pressure     = engine.runtime_stats();
    if (short_b_result.generated_token_ids.size() != kShortOutputTokens ||
        short_c_result.generated_token_ids.size() != kShortOutputTokens ||
        after_pressure.pressure_checkpoints_dropped <=
            before_pressure.pressure_checkpoints_dropped ||
        after_pressure.pressure_private_owners_degraded <=
            before_pressure.pressure_private_owners_degraded ||
        after_pressure.pressure_private_owners_evicted !=
            before_pressure.pressure_private_owners_evicted) {
        std::cerr << "private-checkpoint pressure did not produce a retained checkpoint "
                     "degradation: drops="
                  << before_pressure.pressure_checkpoints_dropped << '/'
                  << after_pressure.pressure_checkpoints_dropped
                  << " degraded=" << before_pressure.pressure_private_owners_degraded << '/'
                  << after_pressure.pressure_private_owners_degraded
                  << " evicted=" << before_pressure.pressure_private_owners_evicted << '/'
                  << after_pressure.pressure_private_owners_evicted << '\n';
        return 1;
    }

    ninfer::PromptInput resume =
        pressure_turn(*long_text, session, ninfer::CacheRetentionHint::LiveSession);
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = source.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = source.content, .media = {}});
    resume.messages.push_back(std::move(assistant));
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Return the retained answer in one line.",
        .media = {},
    });
    resume.messages.push_back(std::move(followup));
    const ninfer::GenerationResult resumed =
        engine.generate(engine.prepare(std::move(resume)), fixed_output(1));
    if (resumed.generated_token_ids.size() != 1 ||
        resumed.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
        resumed.reused_prompt_tokens == 0) {
        std::cerr << "private checkpoint pressure discarded the reusable turn closure: path="
                  << static_cast<int>(resumed.prefix_reuse_path)
                  << " reused=" << resumed.reused_prompt_tokens
                  << " future_ns=" << resumed.materialization.predicted_future_loss_ns << '\n';
        return 1;
    }
    return 0;
}

int exercise_concurrent_resource_settlement(const char* artifact) {
    ninfer::Engine engine(concurrent_engine_options(artifact));

    constexpr std::string_view kSession = "publication-order-real";
    constexpr std::string_view kOlderQuestion =
        "Describe deterministic scheduling using exactly one concise paragraph.";
    constexpr std::string_view kNewerQuestion =
        "Describe prefix caching using exactly one concise paragraph.";
    auto older = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kOlderQuestion))),
        fixed_output(24));
    auto newer = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    const ninfer::GenerationResult newer_result = newer.wait();
    const ninfer::GenerationResult older_result = older.wait();
    if (newer_result.generated_token_ids.size() != 2 ||
        older_result.generated_token_ids.size() != 24) {
        std::cerr << "concurrent session requests did not reach staggered terminal boundaries\n";
        return 1;
    }

    for (std::uint32_t index = 0; index < 6; ++index) {
        const std::string suffix              = std::to_string(index);
        const ninfer::GenerationResult filler = engine.generate(
            engine.prepare(session_turn("publication-filler-" + suffix,
                                        "Give one deterministic token for filler " + suffix + '.')),
            fixed_output(1));
        if (filler.generated_token_ids.size() != 1) {
            std::cerr << "session-order catalog filler did not complete\n";
            return 1;
        }
    }
    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult pressure    = engine.generate(
        engine.prepare(session_turn("publication-pressure",
                                       "Give one deterministic token for the pressure request.")),
        fixed_output(1));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    if (pressure.generated_token_ids.size() != 1 ||
        after_pressure.pressure_private_owners_evicted <=
            before_pressure.pressure_private_owners_evicted) {
        std::cerr << "full session catalog did not execute its canonical eviction\n";
        return 1;
    }

    const ninfer::GenerationResult replay = engine.generate(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    if (replay.generated_token_ids.size() != 2 || replay.reused_prompt_tokens == 0 ||
        replay.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << "late older finish exposed the newer session binding to pressure: path="
                  << static_cast<int>(replay.prefix_reuse_path)
                  << " reused=" << replay.reused_prompt_tokens << '\n';
        return 1;
    }

    {
        std::vector<ninfer::TokenId> long_prompt(400, 198);
        auto cancelled =
            engine.submit(engine.prepare_tokens(std::move(long_prompt)), fixed_output(32, false));
        if (!cancelled) {
            std::cerr << "materialization cancellation fixture did not create a handle\n";
            return 1;
        }
    }
    const ninfer::GenerationResult after_cancel = engine.generate(
        engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198}), fixed_output(1, false));
    if (after_cancel.generated_token_ids.size() != 1) {
        std::cerr << "request after materialization cancellation did not complete\n";
        return 1;
    }

    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(8);
    for (std::uint32_t row = 0; row < 8; ++row) {
        std::vector<ninfer::TokenId> prompt{
            248045, 846, 198, static_cast<ninfer::TokenId>(1000 + row), 248046, 198};
        handles.push_back(
            engine.submit(engine.prepare_tokens(std::move(prompt)), fixed_output(row + 1, false)));
    }
    for (std::uint32_t row = 0; row < handles.size(); ++row) {
        const ninfer::GenerationResult result = handles[row].wait();
        if (result.generated_token_ids.size() != row + 1 ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            std::cerr << "C=8 staggered row " << row << " did not terminate independently\n";
            return 1;
        }
    }
    const ninfer::RuntimeStats settled = engine.runtime_stats();
    if (settled.running_requests != 0 || settled.materializing_requests != 0 ||
        settled.prefilling_requests != 0 || settled.decode_ready_requests != 0 ||
        settled.capture_pending_requests != 0 || settled.terminal_pending_requests != 0) {
        std::cerr << "C=8 terminal settlement left live logical membership: running="
                  << settled.running_requests << " materializing=" << settled.materializing_requests
                  << " prefill=" << settled.prefilling_requests
                  << " decode=" << settled.decode_ready_requests
                  << " capture=" << settled.capture_pending_requests
                  << " terminal=" << settled.terminal_pending_requests << '\n';
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.architecture != "Qwen3_5ForCausalLM" || load.model_name.empty() ||
        load.weight_formats.empty() || load.host_to_device_bytes == 0 ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.architecture
                  << " weights=" << load.prefill_signature << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    const auto* vision = memory.vision_workspace ? &*memory.vision_workspace : nullptr;
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || vision == nullptr ||
        vision->aggregate_prompt_tokens != 4096 || vision->max_item_tokens != 4096 ||
        vision->general_capacity_bytes == 0 || vision->encode_peak_bytes == 0 ||
        vision->handoff_offset_bytes > memory.workspace.capacity_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.workspace.capacity_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int exercise_artifact(const char* artifact) {
    {
        ninfer::EngineOptions options             = engine_options(artifact);
        options.context_cache.device_state_slots  = 2;
        options.context_cache.max_shared_prefixes = 0;
        ninfer::Engine engine(std::move(options));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        if (const int result = exercise_registered_frontend(engine); result != 0) { return result; }
        if (const int result = exercise_stream_observations(engine); result != 0) { return result; }
        if (const int result = exercise_full_prefill_chunk(engine); result != 0) { return result; }
        if (const int result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::PrivateOnly);
            result != 0) {
            return result;
        }
        if (const int result = exercise_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_abandoned_handle_capacity(engine); result != 0) {
            return result;
        }
    }
    if (const int result = exercise_rewrite_branch(artifact); result != 0) { return result; }
    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = exercise_vision(engine); result != 0) { return result; }
    }
    if (const int result = exercise_host_restore(artifact); result != 0) { return result; }
    {
        // Production C=1/H=1 topology: repeated exact use promotes the shared prefix under one
        // cache Device slot; its Fork/Restore and the later ResponseReplay must then rotate
        // without a session identity or dropping either owner.
        ninfer::Engine engine(shared_replacement_engine_options(artifact));
        if (const int result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::SharedAlias);
            result != 0) {
            return result;
        }
    }
    if (const int result = exercise_shared_replacement_and_full_capacity_reuse(artifact);
        result != 0) {
        return result;
    }
    if (const int result = exercise_private_long_anchor_capture_and_replacement(artifact);
        result != 0) {
        return result;
    }
    if (const int result = exercise_engine_automatic_long_anchor_capture(artifact); result != 0) {
        return result;
    }
    if (const int result = exercise_salvage_mid_prefill(artifact); result != 0) { return result; }
    if (const int result = exercise_last_private_alias_eviction(artifact); result != 0) {
        return result;
    }
    if (const int result = exercise_concurrent_resource_settlement(artifact); result != 0) {
        return result;
    }
    return 0;
}

namespace {

struct RecencyPhaseOutcome {
    bool constructed               = false;
    std::uint32_t main_reused_mid  = 0;
    std::uint32_t main_reused      = 0;
    std::uint32_t main_path        = 0;
    std::uint64_t evicted_delta    = 0;
    std::uint64_t degraded_delta   = 0;
    std::uint64_t checkpoints_drop = 0;
    std::uint32_t device_occupied  = 0;
};

// Fills the device cache with several retained private prefixes, makes the shortest one the most
// recently hit (highest last_hit_epoch), applies further pressure, then re-sends it. The escape
// hatch ranks private owners by recency and sacrifices the oldest first, so the most recent
// prefix is never the eviction victim. Diagnostics go to stderr (unbuffered) so they survive
// capture even when the assertions below fail.
RecencyPhaseOutcome run_recency_phase(const char* artifact, const char* label) {
    RecencyPhaseOutcome outcome;
    ninfer::Engine engine(recency_retention_engine_options(artifact));

    // Four long old conversations plus a short MAIN: their combined KV is ~2.4x the 4096 capacity,
    // so the cache runs out of room to merely degrade and must fully evict owners through the
    // value-based pressure path.
    const auto old_a = exact_repeated_prompt_text(engine, 1500, "alpha");
    const auto old_b = exact_repeated_prompt_text(engine, 1500, "bravo");
    const auto old_c = exact_repeated_prompt_text(engine, 1500, "charlie");
    const auto old_d = exact_repeated_prompt_text(engine, 1500, "delta");
    const auto main  = exact_repeated_prompt_text(engine, 128, "omega");
    if (!old_a || !old_b || !old_c || !old_d || !main) {
        std::cerr << label << ": could not construct exact prompt geometry\n";
        return outcome;
    }
    outcome.constructed = true;

    const auto gen = [&](std::string text, std::string session,
                         ninfer::CacheRetentionHint retention, std::uint32_t outputs) {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = outputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = true;
        request.stop.include_model_defaults       = false;
        return engine.generate(
            engine.prepare(pressure_turn(std::move(text), std::move(session), retention)), request);
    };
    const auto query_main = [&]() -> std::pair<std::uint32_t, std::uint32_t> {
        const auto result = gen(*main, "main", ninfer::CacheRetentionHint::LiveSession, 1);
        return {result.reused_prompt_tokens, static_cast<std::uint32_t>(result.prefix_reuse_path)};
    };

    const ninfer::RuntimeStats before = engine.runtime_stats();
    // Fill the cap (max_private_continuations) with two long old conversations and the short MAIN.
    (void)gen(*old_a, "old-1", ninfer::CacheRetentionHint::LiveSession, 1);
    (void)gen(*old_b, "old-2", ninfer::CacheRetentionHint::LiveSession, 1);
    (void)gen(*main, "main", ninfer::CacheRetentionHint::LiveSession, 1);
    {
        // Re-hit MAIN so it carries the highest last_hit_epoch (the "most recent conversation");
        // also confirm MAIN is actually retained/reusable before the pressure arrives.
        const auto mid          = query_main();
        outcome.main_reused_mid = mid.first;
        outcome.main_path       = mid.second;
    }
    // The final two long conversations overflow the cap and force evictions for which MAIN is a
    // candidate victim (it is the shortest prefix, hence the lowest reuse value); the recency
    // order is what keeps it: the older conversations are sacrificed first.
    (void)gen(*old_c, "old-3", ninfer::CacheRetentionHint::LiveSession, 1);
    (void)gen(*old_d, "old-4", ninfer::CacheRetentionHint::LiveSession, 1);
    const ninfer::RuntimeStats after = engine.runtime_stats();
    outcome.evicted_delta =
        after.pressure_private_owners_evicted - before.pressure_private_owners_evicted;
    outcome.degraded_delta =
        after.pressure_private_owners_degraded - before.pressure_private_owners_degraded;
    outcome.checkpoints_drop =
        after.pressure_checkpoints_dropped - before.pressure_checkpoints_dropped;
    outcome.device_occupied = after.device_state_occupied_slots;

    const auto final_main = query_main();
    outcome.main_reused   = final_main.first;
    std::cerr << label << ": mid_reused=" << outcome.main_reused_mid
              << " final_reused=" << outcome.main_reused << " path=" << outcome.main_path
              << " evicted_delta=" << outcome.evicted_delta
              << " degraded_delta=" << outcome.degraded_delta
              << " checkpoints_drop=" << outcome.checkpoints_drop
              << " device_occupied=" << outcome.device_occupied << std::flush;
    return outcome;
}

int exercise_recency_retention(const char* artifact) {
    std::cout << "recency-retention: the most recent prefix survives cache pressure\n";
    const RecencyPhaseOutcome phase = run_recency_phase(artifact, "recency");
    if (!phase.constructed) {
        std::cerr << "recency-retention: fixture could not be constructed\n";
        return 1;
    }

    // The escape hatch sacrifices the oldest private owners first, so the most recent prefix is
    // never the eviction victim while an older one can be given up; the same pressure must still
    // evict at least one owner, or the fixture did not overflow.
    if (phase.main_reused == 0) {
        std::cerr << "recency-retention: the most recent prefix was NOT retained under pressure "
                     "(main_reused="
                  << phase.main_reused << ")\n";
        return 1;
    }
    if (phase.evicted_delta < 1) {
        std::cerr << "recency-retention: pressure evicted no prefix (fixture did not overflow; "
                     "evicted_delta="
                  << phase.evicted_delta << ")\n";
        return 1;
    }

    std::cout << "recency-retention: OK -- the most recent prefix survived (reused="
              << phase.main_reused << " of 128 tokens) while evicting " << phase.evicted_delta
              << " older prefix(es)\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Realistic agent scenarios. Each mirrors an agent traffic shape -- a shared system + tools
// prefix, a growing multi-turn conversation with tool calls, thinking on and preserved -- and
// drives the caching fixes into contention: a full shared catalog, saturated private
// continuations, KV-capacity pressure, and concurrent requests. The deterministic guarantees
// asserted here are the observable ones: shared-prefix reuse, pressure evictions, and zero leaked
// shared_active_references after every completed request (stats published before the caller
// wakes).

// 1. A multi-turn agent conversation reuses its shared system + tools prefix on every turn after
//    the first. The released-lane snapshot is published before the caller wakes, so a fresh
//    runtime_stats() read after each completed turn must show zero active references.
//    prefix_reuse_path is the whole-prompt reuse path: for a growing conversation the shared
//    head is replayed under the private response-replay checkpoint, so assert the observable
//    guarantee (tokens actually reused, non-Root path) rather than a specific shared path.
int exercise_agent_multi_turn(const char* artifact) {
    std::cout << "agent-multi-turn: shared system+tools prefix across 4 tool-calling turns\n";
    ninfer::Engine engine(agent_engine_options(artifact, /*shared=*/4, /*private=*/4,
                                               /*device=*/2, /*kv=*/4096));
    const std::string system = "You are a concise weather assistant. Always call a tool first.";
    std::vector<std::string> tools = {
        agent_tool_json("get_current_weather", "Get the current weather for a city."),
        agent_tool_json("get_forecast", "Get the forecast for a city and a date."),
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    std::vector<ninfer::ChatMessage> history;
    bool saw_shared_reuse = false;
    for (std::uint32_t turn = 1; turn <= 4; ++turn) {
        history.push_back(agent_text_message(
            ninfer::ChatRole::User, "What is the weather in city " + std::to_string(turn) + "?"));
        auto prepared     = engine.prepare(agent_prompt(system, tools, history, true));
        const auto result = engine.generate(std::move(prepared), request);
        if (result.generated_token_ids.size() != 1) {
            std::cerr << "agent-multi-turn: turn " << turn
                      << " did not produce its deterministic token\n";
            return 1;
        }
        history.push_back(agent_text_message(ninfer::ChatRole::Assistant,
                                             "Calling get_current_weather for city " +
                                                 std::to_string(turn) + "."));
        const auto stats = engine.runtime_stats();
        if (turn >= 2) {
            if (result.reused_prompt_tokens == 0 ||
                result.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
                std::cerr << "agent-multi-turn: turn " << turn
                          << " did not reuse the shared stable prefix: path="
                          << static_cast<int>(result.prefix_reuse_path)
                          << " reused=" << result.reused_prompt_tokens << '\n';
                return 1;
            }
            saw_shared_reuse = true;
        }
        if (stats.shared_active_references != 0) {
            std::cerr << "agent-multi-turn: turn " << turn
                      << " leaked active references: " << stats.shared_active_references << '\n';
            return 1;
        }
    }
    if (!saw_shared_reuse) {
        std::cerr << "agent-multi-turn: no turn reused the shared stable prefix\n";
        return 1;
    }
    std::cout
        << "agent-multi-turn: OK -- shared prefix reused on turns 2-4, no leaked references\n";
    return 0;
}

// 2. Five long, distinct agent conversations saturate a small KV capacity (the real contention
//    axis in the production agent is private/host continuations under KV pressure). The pressure
//    path evicts the lowest-value prefixes; the recency-ladder escape hatch sacrifices the oldest
//    private owners first, which keeps the most recent conversation reusable. No references leak.
int exercise_agent_private_continuations(const char* artifact) {
    std::cout
        << "agent-private-continuations: saturate the private cache, preserve the most recent\n";
    ninfer::Engine engine(agent_engine_options(artifact, /*shared=*/2, /*private=*/3,
                                               /*device=*/2, /*kv=*/4096));
    const std::string system = "You are a research assistant that reads files and summarizes them.";
    const std::string read_tool    = agent_tool_json("read_file", "Read a file by path.");
    std::vector<std::string> tools = {read_tool};
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    // Five long, distinct conversations (each a new private continuation). Their combined KV
    // overflows the 4096 capacity, so the cache runs out of room to merely degrade and must fully
    // evict owners through the value-based pressure path.
    std::vector<std::string> conversations;
    for (std::uint32_t doc = 0; doc < 5; ++doc) {
        std::string body;
        for (std::uint32_t line = 0; line < 160; ++line) {
            body += "file-" + std::to_string(doc) + " line " + std::to_string(line) +
                    " describes a deterministic datum. ";
        }
        conversations.push_back(std::move(body));
    }
    const auto stats_before = engine.runtime_stats();
    for (std::size_t doc = 0; doc < conversations.size(); ++doc) {
        std::vector<ninfer::ChatMessage> history = {
            agent_text_message(ninfer::ChatRole::User,
                               "Summarize this file:\n" + conversations[doc]),
        };
        const auto result =
            engine.generate(engine.prepare(agent_prompt(system, tools, history, true)), request);
        if (result.generated_token_ids.size() != 1) {
            std::cerr << "agent-private-continuations: document " << doc
                      << " did not produce its deterministic token\n";
            return 1;
        }
        if (engine.runtime_stats().shared_active_references != 0) {
            std::cerr << "agent-private-continuations: document " << doc
                      << " leaked active references: "
                      << engine.runtime_stats().shared_active_references << '\n';
            return 1;
        }
    }
    const auto stats_after = engine.runtime_stats();
    const std::uint64_t evicted_delta =
        stats_after.pressure_private_owners_evicted - stats_before.pressure_private_owners_evicted;
    if (evicted_delta == 0) {
        std::cerr << "agent-private-continuations: the saturated private cache evicted nothing "
                     "(fixture did not overflow)\n";
        return 1;
    }
    // The most recent conversation must still be reusable (the ladder sacrifices the older ones).
    std::vector<ninfer::ChatMessage> recent = {
        agent_text_message(ninfer::ChatRole::User, "Summarize this file:\n" + conversations.back()),
    };
    const auto recent_reuse =
        engine.generate(engine.prepare(agent_prompt(system, tools, recent, true)), request);
    if (recent_reuse.reused_prompt_tokens == 0) {
        std::cerr << "agent-private-continuations: the most recent conversation was not retained "
                     "under pressure (reused="
                  << recent_reuse.reused_prompt_tokens << ")\n";
        return 1;
    }
    if (engine.runtime_stats().shared_active_references != 0) {
        std::cerr << "agent-private-continuations: leaked active references after the final reuse: "
                  << engine.runtime_stats().shared_active_references << '\n';
        return 1;
    }
    std::cout << "agent-private-continuations: OK -- evicted " << evicted_delta
              << " private prefix(es), most recent still reusable, no leaked references\n";
    return 0;
}

// 3. Two concurrent agent requests (max_concurrency = 2, the production value) with a shared
//    system + tools prefix and distinct conversations settle at terminal boundaries. After both
//    complete, the published stats must show no live logical membership and zero leaked references
//    (the stats-publication fix under concurrent settlement).
int exercise_agent_concurrent(const char* artifact) {
    std::cout
        << "agent-concurrent: two concurrent agent requests settle without leaked references\n";
    ninfer::Engine engine(agent_engine_options(artifact, /*shared=*/4, /*private=*/4,
                                               /*device=*/2, /*kv=*/4096));
    const std::string system       = "You are a code-review assistant.";
    std::vector<std::string> tools = {
        agent_tool_json("read_file", "Read a file by path."),
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    // Two concurrent requests share the same system + tools prefix (the shared stable prefix) but
    // have distinct conversations. Both are submitted before either completes.
    std::vector<ninfer::ChatMessage> alpha = {
        agent_text_message(ninfer::ChatRole::User, "Review this snippet:\nint a = 1;"),
    };
    std::vector<ninfer::ChatMessage> bravo = {
        agent_text_message(ninfer::ChatRole::User, "Review this snippet:\nint b = 2;"),
    };
    auto alpha_handle =
        engine.submit(engine.prepare(agent_prompt(system, tools, alpha, true)), request);
    auto bravo_handle =
        engine.submit(engine.prepare(agent_prompt(system, tools, bravo, true)), request);
    const auto alpha_result = alpha_handle.wait();
    const auto bravo_result = bravo_handle.wait();
    if (alpha_result.generated_token_ids.size() != 1 ||
        bravo_result.generated_token_ids.size() != 1) {
        std::cerr
            << "agent-concurrent: a concurrent request did not produce its deterministic token\n";
        return 1;
    }
    const auto settled = engine.runtime_stats();
    if (settled.running_requests != 0 || settled.prefilling_requests != 0 ||
        settled.decode_ready_requests != 0 || settled.terminal_pending_requests != 0) {
        std::cerr
            << "agent-concurrent: concurrent settlement left live logical membership: running="
            << settled.running_requests << " prefill=" << settled.prefilling_requests
            << " decode=" << settled.decode_ready_requests
            << " terminal=" << settled.terminal_pending_requests << '\n';
        return 1;
    }
    if (settled.shared_active_references != 0) {
        std::cerr << "agent-concurrent: concurrent settlement leaked active references: "
                  << settled.shared_active_references << '\n';
        return 1;
    }
    std::cout << "agent-concurrent: OK -- two concurrent agent requests settled cleanly\n";
    return 0;
}

// 4. Four long, distinct agent conversations overflow a very small KV capacity, driving the
//    pressure-eviction and salvage paths. The observable guarantee: pressure evictions occur and
//    no references leak after the settled requests.
int exercise_agent_kv_pressure(const char* artifact) {
    std::cout << "agent-kv-pressure: long agent conversations overflow the KV capacity\n";
    ninfer::Engine engine(agent_engine_options(artifact, /*shared=*/2, /*private=*/4,
                                               /*device=*/2, /*kv=*/4096));
    const std::string system       = "You are a log-analysis assistant.";
    std::vector<std::string> tools = {
        agent_tool_json("query_logs", "Query the log stream by a filter."),
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    // The engine requires kv_capacity >= max_context (4096). Four long, distinct conversations
    // (~2000 tokens each, ~8000 total) overflow that 4096-token KV capacity; the pressure path
    // evicts the lowest-value prefixes and salvages what it can.
    const auto stats_before = engine.runtime_stats();
    for (std::uint32_t doc = 0; doc < 4; ++doc) {
        std::string body;
        for (std::uint32_t line = 0; line < 200; ++line) {
            body += "log-" + std::to_string(doc) + " entry " + std::to_string(line) +
                    " records a deterministic event. ";
        }
        std::vector<ninfer::ChatMessage> history = {
            agent_text_message(ninfer::ChatRole::User, "Analyze these logs:\n" + body),
        };
        const auto result =
            engine.generate(engine.prepare(agent_prompt(system, tools, history, true)), request);
        if (result.generated_token_ids.size() != 1) {
            std::cerr << "agent-kv-pressure: document " << doc
                      << " did not produce its deterministic token\n";
            return 1;
        }
    }
    const auto stats_after = engine.runtime_stats();
    if (stats_after.shared_active_references != 0) {
        std::cerr << "agent-kv-pressure: leaked active references under KV pressure: "
                  << stats_after.shared_active_references << '\n';
        return 1;
    }
    const std::uint64_t private_evicted =
        stats_after.pressure_private_owners_evicted - stats_before.pressure_private_owners_evicted;
    const std::uint64_t shared_evicted =
        stats_after.pressure_shared_owners_evicted - stats_before.pressure_shared_owners_evicted;
    if (private_evicted == 0 && shared_evicted == 0) {
        std::cerr << "agent-kv-pressure: KV overflow evicted nothing (fixture did not overflow)\n";
        return 1;
    }
    std::cout << "agent-kv-pressure: OK -- evicted private=" << private_evicted
              << " shared=" << shared_evicted << ", no leaked references\n";
    return 0;
}

// 5. Two agent conversations saturate the shared stable-prefix catalog (max_shared_prefixes = 1).
//    The system + tool prefix is a marker-free (automatic) shared candidate, so the issue #251
//    reclaim fires when the second automatic prefix must be catalogued against a full catalog:
//    the oldest automatic entry is reclaimed to make room, and the newest is reusable.
int exercise_agent_shared_catalog(const char* artifact) {
    std::cout << "agent-shared-catalog: automatic agent prefixes against a 1-slot catalog\n";
    ninfer::Engine engine(agent_engine_options(artifact, /*shared=*/1, /*private=*/2,
                                               /*device=*/2, /*kv=*/4096));
    const std::string system     = "You are a deterministic tool dispatcher.";
    const std::string alpha_tool = agent_tool_json("alpha_tool", "Perform the alpha operation.");
    const std::string bravo_tool = agent_tool_json("bravo_tool", "Perform the bravo operation.");
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    // No explicit shared-prefix marker: the system + tool prefix is an automatic shared candidate
    // (the exact traffic the #251 reclaim serves).
    const auto gen = [&](std::string tool, std::string question) {
        ninfer::PromptInput input;
        input.messages.push_back(agent_text_message(ninfer::ChatRole::System, system));
        input.messages.push_back(agent_text_message(ninfer::ChatRole::User, std::move(question)));
        input.options.enable_thinking   = true;
        input.options.preserve_thinking = true;
        input.options.tool_jsons.push_back(std::move(tool));
        input.context_cache.session_key = "agent-session";
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
        return engine.generate(engine.prepare(std::move(input)), request);
    };

    // Prefix A (alpha tool) captures and becomes the single shared catalog entry. The whole-prompt
    // private endpoint is also resident (LiveSession + session_key), so the 4th alpha call is
    // served via that endpoint (the System+tool head is reused, but the User tail makes the private
    // endpoint the cheaper whole-prompt candidate) -- the whole-prompt reuse path is
    // PrivateEndpoint, not SharedStablePrefix. Admission of the automatic System+tool prefix to the
    // shared catalog is verified by the phase-2 reclaim below: the bravo prefix can only be
    // catalogued by reclaiming the alpha entry (issue #251).
    (void)gen(alpha_tool, "Dispatch the alpha operation.");
    (void)gen(alpha_tool, "Dispatch the alpha operation.");
    (void)gen(alpha_tool, "Dispatch the alpha operation once more.");
    const auto alpha_reuse = gen(alpha_tool, "Dispatch the alpha operation.");
    if (alpha_reuse.reused_prompt_tokens == 0) {
        std::cerr << "agent-shared-catalog: first automatic agent prefix was not reused: path="
                  << static_cast<int>(alpha_reuse.prefix_reuse_path)
                  << " reused=" << alpha_reuse.reused_prompt_tokens << '\n';
        return 1;
    }

    // The catalog is now full. Prefix B (bravo tool, also automatic) can only be catalogued by
    // reclaiming A; without the reclaim B is dropped and its shared-prefix reuse freezes (#251).
    const auto before_b = engine.runtime_stats();
    (void)gen(bravo_tool, "Dispatch the bravo operation.");
    (void)gen(bravo_tool, "Dispatch the bravo operation.");
    (void)gen(bravo_tool, "Dispatch the bravo operation once more.");
    const auto bravo_reuse = gen(bravo_tool, "Dispatch the bravo operation.");
    const auto after_b     = engine.runtime_stats();
    // As with the alpha phase, the 4th bravo call is served via its whole-prompt private endpoint
    // (PrivateEndpoint), not the shared catalog. The #251 guarantee is that the bravo prefix was
    // admitted by reclaiming the alpha entry (pressure_shared_owners_evicted delta), not that this
    // particular request's whole-prompt path is SharedStablePrefix.
    if (bravo_reuse.reused_prompt_tokens == 0) {
        std::cerr << "agent-shared-catalog: second automatic agent prefix was not reused: path="
                  << static_cast<int>(bravo_reuse.prefix_reuse_path)
                  << " reused=" << bravo_reuse.reused_prompt_tokens
                  << " shared_evicted=" << after_b.pressure_shared_owners_evicted << '\n';
        return 1;
    }
    if (after_b.pressure_shared_owners_evicted <= before_b.pressure_shared_owners_evicted) {
        std::cerr << "agent-shared-catalog: no shared entry was reclaimed for the second "
                     "automatic prefix: shared_evicted="
                  << after_b.pressure_shared_owners_evicted << '\n';
        return 1;
    }
    if (after_b.shared_active_references != 0) {
        std::cerr << "agent-shared-catalog: leaked active references: "
                  << after_b.shared_active_references << '\n';
        return 1;
    }
    std::cout << "agent-shared-catalog: OK -- second automatic agent prefix reclaimed the first\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Prefix-cache review fixes (2026-09-23). One scenario per fix, each asserting the observable
// behaviour the fix changes. Every scenario prints its measurements on one "review-*" line so the
// runner log records the numbers even when an assertion passes.

namespace review {

ninfer::RequestOptions one_token() { return fixed_output(1); }

ninfer::PromptInput user_turns(std::vector<std::string> turns, std::string session = {}) {
    ninfer::PromptInput input;
    for (std::string& text : turns) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(message));
    }
    input.options.enable_thinking = false;
    if (!session.empty()) {
        input.context_cache.session_key = std::move(session);
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    }
    return input;
}

std::string repeated(std::string_view word, std::uint32_t count) {
    std::string text;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (index != 0) { text.push_back(' '); }
        text.append(word);
    }
    return text;
}

} // namespace review

// Fix 1: a conversation that has only been *published* (never hit since) is recent. Old code
// ranked every never-hit owner at epoch 0, i.e. in catalog-slot order, so the ladder licence could
// give up the newest conversation. MAIN is published after two long conversations and never
// re-hit; two more long conversations then overflow the device KV (no Host tier, so pressure must
// evict). MAIN must still be reusable, and at least one older conversation must have been evicted.
int exercise_review_publication_recency(const char* artifact) {
    std::cout << "review-publication-recency: a just-published conversation outranks older ones\n";
    ninfer::Engine engine(recency_retention_engine_options(artifact));
    const auto old_a = exact_repeated_prompt_text(engine, 1500, "alpha");
    const auto old_b = exact_repeated_prompt_text(engine, 1500, "bravo");
    const auto old_c = exact_repeated_prompt_text(engine, 1500, "charlie");
    const auto old_d = exact_repeated_prompt_text(engine, 1500, "delta");
    const auto main  = exact_repeated_prompt_text(engine, 128, "omega");
    if (!old_a || !old_b || !old_c || !old_d || !main) {
        std::cerr << "review-publication-recency: could not construct exact prompt geometry\n";
        return 1;
    }
    const auto gen = [&](const std::string& text, const char* session) {
        return engine.generate(
            engine.prepare(pressure_turn(text, session, ninfer::CacheRetentionHint::LiveSession)),
            review::one_token());
    };
    const ninfer::RuntimeStats before = engine.runtime_stats();
    (void)gen(*old_a, "old-1");
    (void)gen(*old_b, "old-2");
    (void)gen(*main, "main"); // published, never hit again before the pressure
    (void)gen(*old_c, "old-3");
    (void)gen(*old_d, "old-4");
    const ninfer::RuntimeStats after = engine.runtime_stats();
    const std::uint64_t evicted =
        after.pressure_private_owners_evicted - before.pressure_private_owners_evicted;
    const auto final_main = gen(*main, "main");
    std::cout << "review-publication-recency: main_reused=" << final_main.reused_prompt_tokens
              << " path=" << static_cast<int>(final_main.prefix_reuse_path)
              << " private_evicted=" << evicted << '\n';
    if (evicted < 1) {
        std::cerr << "review-publication-recency: pressure evicted nothing; fixture did not "
                     "overflow\n";
        return 1;
    }
    if (final_main.reused_prompt_tokens == 0) {
        std::cerr << "review-publication-recency: the most recently published conversation was "
                     "evicted while older ones survived\n";
        return 1;
    }
    std::cout << "review-publication-recency: OK\n";
    return 0;
}

// Fix 2: the escape-hatch ladder used to evict every shared prefix on every rung. A client-marked
// system+tools prefix is captured, then long private conversations overflow the device KV so each
// admission plans under pressure (with a Host tier to demote into). A new conversation with the
// same system+tools must still reuse the shared prefix, and no shared owner may have been evicted.
int exercise_review_shared_survives_ladder(const char* artifact) {
    std::cout << "review-shared-survives-ladder: private pressure keeps the shared prefix\n";
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 4096;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.kv_cache                         = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 2;
    options.context_cache.host_state_slots   = 8;
    options.context_cache.host_kv_capacity_bytes            = 64ULL << 20;
    options.context_cache.max_private_continuations         = 6;
    options.context_cache.max_shared_prefixes               = 2;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.recency_eviction                  = true;
    ninfer::Engine engine(std::move(options));

    const std::string system =
        "You are a careful operations assistant. " + review::repeated("Follow the runbook.", 40);
    const std::vector<std::string> tools = {
        agent_tool_json("lookup_host", "Look up a host in the inventory."),
        agent_tool_json("restart_service", "Restart a service on a host."),
    };
    std::uint32_t conversation = 0;
    const auto agent           = [&](std::string question) {
        std::vector<ninfer::ChatMessage> history;
        history.push_back(agent_text_message(ninfer::ChatRole::User, std::move(question)));
        auto prompt                      = agent_prompt(system, tools, std::move(history), false);
        prompt.context_cache.session_key = "review-agent-" + std::to_string(++conversation);
        return engine.generate(engine.prepare(std::move(prompt)), review::one_token());
    };
    (void)agent("First question about host alpha.");
    const auto warm = agent("Second question about host bravo.");
    if (warm.reused_prompt_tokens == 0) {
        std::cerr << "review-shared-survives-ladder: shared prefix was never captured (path="
                  << static_cast<int>(warm.prefix_reuse_path) << ")\n";
        return 1;
    }

    const auto long_a = exact_repeated_prompt_text(engine, 1500, "kilo");
    const auto long_b = exact_repeated_prompt_text(engine, 1500, "lima");
    const auto long_c = exact_repeated_prompt_text(engine, 1500, "mike");
    if (!long_a || !long_b || !long_c) {
        std::cerr << "review-shared-survives-ladder: could not construct exact prompt geometry\n";
        return 1;
    }
    const ninfer::RuntimeStats before = engine.runtime_stats();
    for (const auto* text : {&*long_a, &*long_b, &*long_c}) {
        (void)engine.generate(
            engine.prepare(pressure_turn(*text, "", ninfer::CacheRetentionHint::Default)),
            review::one_token());
    }
    const ninfer::RuntimeStats after = engine.runtime_stats();
    const auto final_result          = agent("Third question about host charlie.");
    const std::uint64_t private_pressure =
        (after.pressure_private_owners_evicted - before.pressure_private_owners_evicted) +
        (after.pressure_private_owners_degraded - before.pressure_private_owners_degraded);
    const std::uint64_t shared_evicted =
        after.pressure_shared_owners_evicted - before.pressure_shared_owners_evicted;
    std::cout << "review-shared-survives-ladder: warm_reused=" << warm.reused_prompt_tokens
              << " final_reused=" << final_result.reused_prompt_tokens
              << " final_path=" << static_cast<int>(final_result.prefix_reuse_path)
              << " private_pressure=" << private_pressure << " shared_evicted=" << shared_evicted
              << " shared_degraded="
              << (after.pressure_shared_owners_degraded - before.pressure_shared_owners_degraded)
              << '\n';
    if (private_pressure == 0) {
        std::cerr << "review-shared-survives-ladder: no private pressure; fixture did not "
                     "overflow\n";
        return 1;
    }
    if (shared_evicted != 0 || final_result.reused_prompt_tokens < warm.reused_prompt_tokens) {
        std::cerr << "review-shared-survives-ladder: the shared prefix was destroyed by private "
                     "pressure\n";
        return 1;
    }
    std::cout << "review-shared-survives-ladder: OK\n";
    return 0;
}

// Fix 3: automatic shared-prefix reclaim picks the least recently *used* entry. Two automatic
// prefixes fill a 2-slot catalog, the first is hit again, then a third needs a slot: the second
// (least recently used) must be reclaimed and the first must stay reusable. Old code reclaimed the
// first-published entry.
int exercise_review_shared_reclaim_lru(const char* artifact) {
    std::cout << "review-shared-reclaim-lru: the least recently used automatic prefix goes\n";
    ninfer::EngineOptions engine_options = shared_replacement_engine_options(artifact);
    // KV room for the active lease plus the shared snapshots. With kv_capacity == max_context the
    // Device KV lease (prompt + max(prefill_chunk, 4096) output tokens, capped at max_context)
    // takes the whole pool and every shared capture is skipped; a second context of pages (which
    // requires max_concurrency 2) leaves room for the catalog.
    engine_options.max_context          = 4096;
    engine_options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    engine_options.max_concurrency      = 2;
    engine_options.max_pending_requests = 2;
    engine_options.context_cache.max_private_continuations = 2;
    engine_options.context_cache.max_shared_prefixes       = 2;
    // Room for every StateImage the fixture keeps (2 per private continuation, 1 per shared
    // prefix) so no state-slot pressure competes with the catalog-slot reclaim under test.
    engine_options.context_cache.host_state_slots   = 16;
    engine_options.context_cache.device_state_slots = 16;
    ninfer::Engine engine(std::move(engine_options));
    const auto plain = [](const std::string& text) {
        auto input                    = review::user_turns({text});
        input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
        return input;
    };
    const auto gen = [&](const std::string& text) {
        auto result = engine.generate(engine.prepare(plain(text)), review::one_token());
        if (std::getenv("NINFER_REVIEW_TRACE") != nullptr) {
            const ninfer::RuntimeStats stats = engine.runtime_stats();
            std::cout << "  trace: '" << text.substr(0, 14)
                      << "' path=" << static_cast<int>(result.prefix_reuse_path)
                      << " reused=" << result.reused_prompt_tokens
                      << " prompt=" << result.prompt.prompt_tokens
                      << " captures=" << stats.active_captures_completed
                      << " shared_evicted=" << stats.pressure_shared_owners_evicted
                      << " shared_degraded=" << stats.pressure_shared_owners_degraded
                      << " private_evicted=" << stats.pressure_private_owners_evicted
                      << " searches=" << stats.pressure_searches
                      << " maximal=" << stats.pressure_maximal_fallback_selections
                      << " dev_state=" << stats.device_state_occupied_slots
                      << " host_state=" << stats.host_state_occupied_slots
                      << " dev_kv=" << stats.device_main_kv_occupied_pages
                      << " lease=" << stats.device_main_kv_lease_pages << '\n';
        }
        return result;
    };
    // Two private continuation slots: two distinct fillers push a prefix's private endpoint out,
    // so a later reuse of the prefix can only come from the shared catalog.
    std::uint32_t filler_index = 0;
    const auto flush_private   = [&] {
        for (int index = 0; index < 2; ++index) {
            (void)gen("private filler endpoint " + std::to_string(++filler_index) + ".");
        }
    };
    const auto publish = [&](const std::string& prefix) {
        (void)gen(prefix);
        (void)gen(prefix);
        flush_private();
    };
    const std::string a = review::repeated("alpha-stable", 40);
    const std::string b = review::repeated("bravo-stable", 40);
    const std::string c = review::repeated("charlie-stable", 40);
    publish(a);
    publish(b);
    const auto a_hit = gen(a); // A becomes the most recently used shared entry
    flush_private();
    const ninfer::RuntimeStats before = engine.runtime_stats();
    publish(c);
    const ninfer::RuntimeStats after = engine.runtime_stats();
    const auto c_reuse               = gen(c);
    flush_private();
    const auto a_reuse = gen(a);
    flush_private();
    const auto b_reuse = gen(b);
    const auto shared  = [](const ninfer::GenerationResult& result) {
        return result.prefix_reuse_path == ninfer::PrefixReusePath::SharedStablePrefix &&
               result.reused_prompt_tokens != 0;
    };
    const ninfer::RuntimeStats final_stats = engine.runtime_stats();
    std::cout << "review-shared-reclaim-lru: captures_completed="
              << final_stats.active_captures_completed
              << " captures_skipped=" << final_stats.active_captures_skipped
              << " a_hit_path=" << static_cast<int>(a_hit.prefix_reuse_path)
              << " a_hit=" << shared(a_hit) << " c_shared=" << shared(c_reuse)
              << " a_shared=" << shared(a_reuse) << " b_shared=" << shared(b_reuse)
              << " shared_evicted="
              << (after.pressure_shared_owners_evicted - before.pressure_shared_owners_evicted)
              << '\n';
    if (!shared(a_hit)) {
        std::cerr << "review-shared-reclaim-lru: prefix A never reached the shared catalog\n";
        return 1;
    }
    if (!shared(c_reuse) || !shared(a_reuse) || shared(b_reuse)) {
        std::cerr << "review-shared-reclaim-lru: reclaim did not give up the least recently used "
                     "entry (B)\n";
        return 1;
    }
    std::cout << "review-shared-reclaim-lru: OK\n";
    return 0;
}

// Fix 5: automatic long anchors are spaced geometrically back from the prompt end. A conversation
// of two ~1500-token messages followed by four short ones is prefilled from Root, then re-sent
// with the second long message rewritten. With the default spacing the boundary after the first
// long message is anchored, so the rewrite resumes from it; with spacing 0 (the old rule: last L
// boundaries) only the short tail is anchored and the rewrite falls back further. The same
// fixture runs on both engines and records the capture counts.
struct AnchorArmOutcome {
    std::uint64_t captures_first         = 0;
    std::uint32_t rewrite_reused         = 0;
    ninfer::PrefixReusePath rewrite_path = ninfer::PrefixReusePath::Root;
    std::uint32_t first_long_tokens      = 0;
    std::uint32_t prompt_tokens          = 0;
};

AnchorArmOutcome run_anchor_arm(const char* artifact, std::uint32_t spacing) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 8192;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk                    = 1024;
    options.speculative.backend              = ninfer::SpeculativeBackend::None;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 8;
    options.context_cache.host_state_slots   = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 4;
    options.context_cache.automatic_long_anchors            = true;
    set_anchor_spacing(options.context_cache, spacing);
    ninfer::Engine engine(std::move(options));

    const std::string first_long        = review::repeated("alpha", 1500);
    const std::string second_long       = review::repeated("bravo", 1500);
    const std::string rewritten         = review::repeated("zulu", 1500);
    const std::vector<std::string> tail = {"Short note one.", "Short note two.",
                                           "Short note three.", "Short note four."};
    const auto conversation             = [&](const std::string& second) {
        std::vector<std::string> turns{first_long, second};
        turns.insert(turns.end(), tail.begin(), tail.end());
        return review::user_turns(std::move(turns), "review-anchor");
    };
    AnchorArmOutcome outcome;
    outcome.first_long_tokens         = engine.count_tokens(review::user_turns({first_long}));
    const ninfer::RuntimeStats before = engine.runtime_stats();
    const auto first =
        engine.generate(engine.prepare(conversation(second_long)), review::one_token());
    const ninfer::RuntimeStats after = engine.runtime_stats();
    outcome.captures_first = after.active_captures_completed - before.active_captures_completed;
    outcome.prompt_tokens  = first.prompt.prompt_tokens;
    const auto rewrite =
        engine.generate(engine.prepare(conversation(rewritten)), review::one_token());
    outcome.rewrite_reused = rewrite.reused_prompt_tokens;
    outcome.rewrite_path   = rewrite.prefix_reuse_path;
    return outcome;
}

int exercise_review_anchor_spacing(const char* artifact) {
    std::cout << "review-anchor-spacing: spaced anchors reach deep history\n";
    const AnchorArmOutcome spaced   = run_anchor_arm(artifact, 1024);
    const AnchorArmOutcome unspaced = run_anchor_arm(artifact, 0);
    std::cout << "review-anchor-spacing: prompt=" << spaced.prompt_tokens << " first_long_tokens~"
              << spaced.first_long_tokens << " | spaced: captures=" << spaced.captures_first
              << " rewrite_reused=" << spaced.rewrite_reused
              << " rewrite_path=" << static_cast<int>(spaced.rewrite_path)
              << " | unspaced: captures=" << unspaced.captures_first
              << " rewrite_reused=" << unspaced.rewrite_reused
              << " rewrite_path=" << static_cast<int>(unspaced.rewrite_path) << '\n';
    if (spaced.rewrite_path != ninfer::PrefixReusePath::PrivateLongAnchor ||
        spaced.rewrite_reused + 64U < spaced.first_long_tokens) {
        std::cerr << "review-anchor-spacing: the rewrite did not resume from the deep anchor\n";
        return 1;
    }
    if (unspaced.rewrite_reused >= spaced.rewrite_reused) {
        std::cerr << "review-anchor-spacing: spacing did not improve deep-rewrite reuse over the "
                     "last-L rule (fixture does not discriminate)\n";
        return 1;
    }
    if (spaced.captures_first >= unspaced.captures_first) {
        std::cerr << "review-anchor-spacing: spacing did not reduce the captures on the first "
                     "prefill\n";
        return 1;
    }
    std::cout << "review-anchor-spacing: OK\n";
    return 0;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const char* selected            = std::getenv("NINFER_PREFIX_REAL_SCENARIO");
    const std::string_view scenario = selected ? selected : "all";
    int result                      = 0;
    try {
        if (scenario == "vision") {
            ninfer::Engine engine(engine_options(artifact));
            result = exercise_vision(engine);
        } else if (scenario == "all") {
            result = exercise_artifact(artifact);
        } else if (scenario == "host-restore") {
            result = exercise_host_restore(artifact);
        } else if (scenario == "concurrent") {
            result = exercise_concurrent_resource_settlement(artifact);
        } else if (scenario == "anthropic-prefix-regression") {
            result = exercise_anthropic_prefix_regression(artifact);
        } else if (scenario == "shared-rewrite-materialization") {
            result = exercise_shared_rewrite_materialization(artifact);
        } else if (scenario == "pressure-resume") {
            result = exercise_pressure_partial_spill_and_resume(artifact);
        } else if (scenario == "private-checkpoint-pressure") {
            result = exercise_private_checkpoint_pressure_retention(artifact);
        } else if (scenario == "recency-retention") {
            result = exercise_recency_retention(artifact);
        } else if (scenario == "source-pressure-protection") {
            result = exercise_materialization_source_pressure_protection(artifact);
        } else if (scenario == "shared-replacement") {
            result = exercise_shared_replacement_and_full_capacity_reuse(artifact);
        } else if (scenario == "shared-saturation-reclaim") {
            result = exercise_shared_saturation_reclaim(artifact);
        } else if (scenario == "private-long-anchor") {
            result = exercise_private_long_anchor_capture_and_replacement(artifact);
        } else if (scenario == "engine-automatic-long-anchor") {
            result = exercise_engine_automatic_long_anchor_capture(artifact);
        } else if (scenario == "salvage-mid-prefill") {
            result = exercise_salvage_mid_prefill(artifact);
        } else if (scenario == "rewrite-checkpoint-shared") {
            ninfer::Engine engine(shared_replacement_engine_options(artifact));
            result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::SharedAlias);
        } else if (scenario == "rewrite-checkpoint") {
            auto options                              = engine_options(artifact);
            options.context_cache.device_state_slots  = 2;
            options.context_cache.max_shared_prefixes = 0;
            ninfer::Engine engine(std::move(options));
            result =
                exercise_rewrite_checkpoints(engine, RewriteCheckpointCacheTopology::PrivateOnly);
        } else if (scenario == "stream-observations") {
            auto options          = engine_options(artifact);
            options.enable_vision = false;
            options.context_cache = ninfer::ContextCacheOptions{.enabled = false};
            ninfer::Engine engine(std::move(options));
            result = exercise_stream_observations(engine);
        } else if (scenario == "agent-multi-turn") {
            result = exercise_agent_multi_turn(artifact);
        } else if (scenario == "agent-private-continuations") {
            result = exercise_agent_private_continuations(artifact);
        } else if (scenario == "agent-concurrent") {
            result = exercise_agent_concurrent(artifact);
        } else if (scenario == "agent-kv-pressure") {
            result = exercise_agent_kv_pressure(artifact);
        } else if (scenario == "agent-shared-catalog") {
            result = exercise_agent_shared_catalog(artifact);
        } else if (scenario == "review-publication-recency") {
            result = exercise_review_publication_recency(artifact);
        } else if (scenario == "review-shared-survives-ladder") {
            result = exercise_review_shared_survives_ladder(artifact);
        } else if (scenario == "review-shared-reclaim-lru") {
            result = exercise_review_shared_reclaim_lru(artifact);
        } else if (scenario == "review-anchor-spacing") {
            result = exercise_review_anchor_spacing(artifact);
        } else {
            throw std::invalid_argument("unknown prefix integration scenario");
        }
    } catch (const std::exception& e) {
        std::cerr << "uncaught exception in scenario '" << scenario << "': " << e.what() << '\n';
        std::cerr.flush();
        return 1;
    }
    if (result == 0) { std::cout << "ok\n"; }
    return result;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
