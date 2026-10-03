#include "ninfer/engine.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

ninfer::PromptInput copy_prompt(const std::string& source) {
    ninfer::PromptInput input;
    ninfer::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "The final answer must contain only the requested file, with no explanation or "
                 "Markdown fences. Keep any reasoning in the thinking section.",
        .media = {}});
    input.messages.push_back(std::move(system));
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text,
        .text = "Return an exact, byte-for-byte copy of the Python file below. Preserve all "
                "whitespace and do not add text before or after it.\n\n" +
                source,
        .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = true;
    input.options.preserve_thinking = true;
    return input;
}

} // namespace

int main(int argc, char** argv) {
    const auto* artifact = std::getenv("NINFER_NGRAM_TEST_WEIGHTS");
    if (!artifact || !*artifact) { return 77; }
    try {
        const std::string backend = argc > 1 ? argv[1] : "mtp";
        const unsigned neural     = argc > 2 ? std::stoul(argv[2]) : 3U;
        const unsigned ngram      = argc > 3 ? std::stoul(argv[3]) : 15U;
        const bool strict_fresh   = argc == 5 && std::string(argv[4]) == "--strict-fresh";
        require(argc <= 5 && (argc != 5 || strict_fresh), "unsupported fixture arguments");
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 8192;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        options.max_concurrency = 1;
        options.prefill_chunk   = 1024;
        options.enable_vision   = false;
        options.kv_cache        = ninfer::KvCacheStorage::Nvfp4Group16;
        options.use_cuda_graph  = true;
        if (backend == "mtp") {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        } else if (backend == "dflash") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        } else if (backend == "dflash2") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        } else {
            throw std::invalid_argument("unsupported fixture backend");
        }
        options.speculative.draft_tokens                        = neural;
        options.speculative.ngram_draft_tokens                  = ngram;
        options.speculative.proposal_head                       = ninfer::ProposalHead::Optimized;
        options.context_cache.device_state_slots                = 1;
        options.context_cache.host_state_slots                  = 4;
        options.context_cache.host_kv_capacity_bytes            = 256ULL << 20;
        options.context_cache.max_private_continuations         = 2;
        options.context_cache.max_shared_prefixes               = 0;
        options.context_cache.max_long_anchors_per_continuation = 0;
        ninfer::Engine engine(options);
        std::string source;
        for (unsigned i = 0; i < 20; ++i) {
            source += "def transform_" + std::to_string(i) +
                      "(value):\n    offset = " + std::to_string(i + 17) + "\n    return value * " +
                      std::to_string(i + 2) + " + offset\n\n";
        }
        unsigned forced     = 0;
        bool fresh_identity = true;
        for (const unsigned budget : {1U, 4U, 15U, 16U, 31U, 32U, 63U}) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens    = 512;
            request.execution.sampling.temperature       = 0;
            request.execution.sampling.presence_penalty  = 0;
            request.execution.sampling.frequency_penalty = 0;
            request.execution.thinking.budget            = budget;
            request.execution.allow_prefix_reuse         = false;
            const auto fresh = engine.generate(engine.prepare(copy_prompt(source)), request);
            request.execution.allow_prefix_reuse = true;
            // A request with reuse disabled does not publish a cached continuation.
            const auto seeded   = engine.generate(engine.prepare(copy_prompt(source)), request);
            const auto reused   = engine.generate(engine.prepare(copy_prompt(source)), request);
            const auto repeated = engine.generate(engine.prepare(copy_prompt(source)), request);
            std::cout << "thinking budget=" << budget
                      << " fresh_reuse=" << fresh.reused_prompt_tokens
                      << " seeded_reuse=" << seeded.reused_prompt_tokens
                      << " reused=" << reused.reused_prompt_tokens
                      << " fresh_tokens=" << fresh.generated_token_ids.size()
                      << " reused_tokens=" << reused.generated_token_ids.size()
                      << " injected=" << fresh.thinking.injected_tokens
                      << " ngram_accepted=" << fresh.speculative.ngram_accepted_tokens << std::endl;
            require(fresh.reused_prompt_tokens == 0, "fresh control unexpectedly reused a prefix");
            for (const auto* result : {&fresh, &seeded, &reused, &repeated}) {
                require(result->thinking.configured_budget == budget &&
                            result->thinking.model_thinking_tokens <= budget,
                        "thinking budget accounting escaped its bound");
                require(result->generated_token_ids.size() <= 512,
                        "injected control escaped the total output budget");
                if (result->content.size() <= 64 || !source.starts_with(result->content)) {
                    std::cerr << "answer detail budget=" << budget
                              << " reused=" << result->reused_prompt_tokens
                              << " reasoning=" << std::quoted(result->reasoning)
                              << " content=" << std::quoted(result->content) << std::endl;
                }
                require(result->content.size() > 64 && source.starts_with(result->content),
                        "post-thinking output is not an exact source prefix");
                require(ngram == 0 || result->speculative.ngram_accepted_tokens >
                                          result->thinking.model_thinking_tokens +
                                              result->thinking.injected_tokens,
                        "fixture did not accept ngram drafts beyond its thinking/control prefix");
                if (result->thinking.applied) {
                    ++forced;
                    require(result->thinking.model_thinking_tokens == budget &&
                                result->thinking.injected_tokens > 0,
                            "forced close did not reach its cap and inject control tokens");
                } else {
                    require(result->thinking.injected_tokens == 0,
                            "natural close unexpectedly injected control tokens");
                }
            }
            if (seeded.generated_token_ids != fresh.generated_token_ids ||
                reused.generated_token_ids != fresh.generated_token_ids) {
                fresh_identity = false;
                std::cerr << "identity detail budget=" << budget << " seeded_equals_reused="
                          << (seeded.generated_token_ids == reused.generated_token_ids)
                          << " fresh_content_equals_seeded=" << (fresh.content == seeded.content)
                          << " fresh_content_equals_reused=" << (fresh.content == reused.content)
                          << " fresh_reasoning=" << std::quoted(fresh.reasoning)
                          << " seeded_reasoning=" << std::quoted(seeded.reasoning)
                          << " reused_reasoning=" << std::quoted(reused.reasoning) << std::endl;
            }
            require(reused.reused_prompt_tokens > 0 && repeated.reused_prompt_tokens > 0 &&
                        reused.generated_token_ids == repeated.generated_token_ids,
                    "cached repeat changed the controlled continuation");
        }
        require(forced > 0, "thinking fixture did not force a control transition");
        std::cout << "ngram thinking-boundary checks passed backend=" << backend
                  << " neural=" << neural << " ngram=" << ngram
                  << " fresh_identity=" << fresh_identity << std::endl;
        if (strict_fresh && !fresh_identity) {
            std::cerr << "strict fresh/cached identity diagnostic failed" << std::endl;
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
