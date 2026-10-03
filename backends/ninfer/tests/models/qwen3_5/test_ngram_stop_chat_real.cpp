#include "ninfer/engine.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
// A digit ends a tokenizer word, so extending this prefix cannot merge its final token.
const std::string kAssistantPrefix = "def transform_0(value):\n    offset = 17";

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(unsigned count, bool reuse) {
    ninfer::RequestOptions result;
    result.execution.requested_output_tokens    = count;
    result.execution.allow_prefix_reuse         = reuse;
    result.execution.sampling.temperature       = 0;
    result.execution.sampling.presence_penalty  = 0;
    result.execution.sampling.frequency_penalty = 0;
    // Continue the complete committed prefix, including the caller's stop token.
    result.stop.publish_stop_token = true;
    return result;
}

ninfer::PromptInput prompt(const std::string& source,
                           const std::optional<std::string>& continuation = std::nullopt) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text,
        .text = "Repeat the following Python file exactly. Output only the file, with no "
                "explanation or Markdown fences. Preserve every space and newline.\n\n" +
                source,
        .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.options.preserve_thinking = true;
    ninfer::ChatMessage assistant;
    assistant.role = ninfer::ChatRole::Assistant;
    assistant.parts.push_back(
        ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                            .text  = kAssistantPrefix + continuation.value_or(""),
                            .media = {}});
    input.messages.push_back(std::move(assistant));
    input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    return input;
}
} // namespace

int main(int argc, char** argv) {
    const auto* artifact = std::getenv("NINFER_NGRAM_TEST_WEIGHTS");
    if (!artifact || !*artifact) { return 77; }
    try {
        const std::string backend         = argc > 1 ? argv[1] : "mtp";
        const std::string_view width_text = argc > 2 ? argv[2] : "15";
        unsigned ngram                    = 0;
        const auto parsed =
            std::from_chars(width_text.data(), width_text.data() + width_text.size(), ngram);
        require(parsed.ec == std::errc{} && parsed.ptr == width_text.data() + width_text.size() &&
                    ngram <= 63,
                "ngram width must be an integer in 0..63");
        bool strict_fresh = false, graphs = true;
        for (int i = 3; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (option == "--strict-fresh" && !strict_fresh) {
                strict_fresh = true;
            } else if (option == "--no-cuda-graph" && graphs) {
                graphs = false;
            } else {
                throw std::invalid_argument("invalid or duplicate test option");
            }
        }
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 4096;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.max_concurrency = 1;
        options.prefill_chunk   = 1024;
        options.enable_vision   = false;
        options.use_cuda_graph  = graphs;
        options.kv_cache        = ninfer::KvCacheStorage::Nvfp4Group16;
        if (backend == "mtp") {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        } else if (backend == "dflash") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        } else if (backend == "dflash2") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        } else {
            throw std::invalid_argument("unsupported backend");
        }
        options.speculative.draft_tokens                        = 5;
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
        auto prepared              = engine.prepare(prompt(source));
        const auto prompt_tokens   = prepared.summary().prompt_tokens;
        const auto reference       = engine.generate(std::move(prepared), request(256, true));
        const auto reference_text  = kAssistantPrefix + reference.content;
        const auto reference_count = reference.generated_token_ids.size();
        const bool reference_budget =
            reference_count > 16 && reference_count <= 256 &&
            ((reference.finish_reason == ninfer::FinishReason::OutputLimit &&
              reference_count == 256) ||
             reference.finish_reason == ninfer::FinishReason::StopToken);
        if (!reference_budget || !source.starts_with(reference_text)) {
            const auto mismatch = std::mismatch(reference_text.begin(), reference_text.end(),
                                                source.begin(), source.end());
            const auto position = static_cast<std::size_t>(mismatch.first - reference_text.begin());
            std::cerr << "reference diagnostic backend=" << backend << " ngram=" << ngram
                      << " graphs=" << graphs << " tokens=" << reference.generated_token_ids.size()
                      << " finish=" << static_cast<int>(reference.finish_reason)
                      << " source_prefix=" << source.starts_with(reference_text)
                      << " first_difference=" << position
                      << " expected=" << std::quoted(source.substr(position, 96))
                      << " actual=" << std::quoted(reference_text.substr(position, 96))
                      << " content=" << std::quoted(reference.content) << std::endl;
        }
        // Natural stops can leave a shorter reference. The case-count assertion below
        // still requires four distinct, verified partial stops and continuations.
        require(reference_budget, "reference is too short or has an invalid finish/output budget");
        require(source.starts_with(reference_text), "reference is not an exact source prefix");
        const auto selection_end = std::min<std::size_t>(96, reference_count - 1);
        std::cout << "reference tokens=" << reference_count
                  << " finish=" << static_cast<int>(reference.finish_reason)
                  << " stop_selection_end=" << selection_end << " exact=1" << std::endl;
        require(ngram == 0 || reference.speculative.ngram_accepted_tokens > 0,
                "reference did not exercise ngram");
        for (const unsigned budget : {1U, 2U}) {
            const auto tail =
                engine.generate(engine.prepare(prompt(source)), request(budget, true));
            require(tail.generated_token_ids.size() == budget &&
                        source.starts_with(kAssistantPrefix + tail.content),
                    "anchor-only tail changed the output budget or source prefix");
            require(tail.speculative.ngram_rounds == 0 &&
                        tail.speculative.ngram_drafted_tokens == 0,
                    "ngram ran when no draft fit the remaining output budget");
        }
        // A stop token must first occur at its index, or the stop would fire earlier. A natural
        // stop can leave fewer than four such tokens past index 16 (an RTX 5090 ends this copy
        // after 33 tokens); only then do earlier tokens fill the four cases. They skip tokens
        // that end in whitespace: there, a stop at the blank line after the first function was
        // continued with a second blank line instead of the next function.
        const auto first_occurrence = [&](std::size_t i) {
            const auto begin = reference.generated_token_ids.begin();
            return std::find(begin, begin + i, reference.generated_token_ids[i]) == begin + i;
        };
        std::vector<std::size_t> stop_indices;
        for (std::size_t i = 16; i < selection_end; ++i) {
            if (first_occurrence(i)) { stop_indices.push_back(i); }
        }
        const std::size_t late_indices = stop_indices.size();
        if (late_indices < 4) {
            for (std::size_t i = std::min<std::size_t>(16, selection_end); i-- > 1;) {
                const auto bytes = engine.token_bytes(reference.generated_token_ids[i]);
                if (first_occurrence(i) && !bytes.empty() &&
                    std::isspace(static_cast<unsigned char>(bytes.back())) == 0) {
                    stop_indices.push_back(i);
                }
            }
        }
        unsigned cases = 0, partial_cases = 0, fresh_differences = 0;
        std::uint64_t accepted = 0;
        for (std::size_t position = 0; position < stop_indices.size(); ++position) {
            if (cases == 12 || (position >= late_indices && cases >= 4)) { break; }
            const std::size_t i = stop_indices[position];
            const auto token    = reference.generated_token_ids[i];
            auto stop           = request(256, true);
            stop.stop.token_ids.push_back(token);
            const auto stopped = engine.generate(engine.prepare(prompt(source)), stop);
            require(stopped.finish_reason == ninfer::FinishReason::StopToken &&
                        stopped.generated_token_ids.size() == i + 1 &&
                        std::equal(stopped.generated_token_ids.begin(),
                                   stopped.generated_token_ids.end(),
                                   reference.generated_token_ids.begin()) &&
                        source.starts_with(kAssistantPrefix + stopped.content),
                    "stopped answer differs from exact reference prefix");
            const auto tail     = [&] { return engine.prepare(prompt(source, stopped.content)); };
            const auto reused   = engine.generate(tail(), request(16, true));
            const auto fresh    = engine.generate(tail(), request(16, false));
            const auto repeated = engine.generate(engine.prepare(prompt(source)), stop);
            const auto reused_again = engine.generate(tail(), request(16, true));
            require(repeated.generated_token_ids == stopped.generated_token_ids &&
                        repeated.content == stopped.content,
                    "repeated stop changed output");
            const auto limited =
                engine.generate(engine.prepare(prompt(source)), request(i + 1, true));
            require(limited.generated_token_ids == stopped.generated_token_ids &&
                        limited.content == stopped.content,
                    "output-limit and token-stop prefixes differ");
            const auto limited_tail = engine.generate(tail(), request(16, true));
            const auto equal        = [](const auto& a, const auto& b) {
                return a.generated_token_ids == b.generated_token_ids && a.content == b.content &&
                       a.finish_reason == b.finish_reason;
            };
            require(equal(reused, reused_again), "repeated retained continuation changed output");
            if (!equal(reused, limited_tail)) {
                std::cerr << "stop/limit diagnostic index=" << i
                          << " stopped=" << std::quoted(stopped.content)
                          << " reused_count=" << reused.generated_token_ids.size()
                          << " reused_finish=" << static_cast<int>(reused.finish_reason)
                          << " limited_count=" << limited_tail.generated_token_ids.size()
                          << " limited_finish=" << static_cast<int>(limited_tail.finish_reason)
                          << " reused=" << std::quoted(reused.content)
                          << " limited=" << std::quoted(limited_tail.content) << std::endl;
            }
            require(equal(reused, limited_tail),
                    "token-stop and output-limit continuations differ");
            fresh_differences += !equal(reused, fresh);
            std::cout << "continuation counts index=" << i
                      << " reused=" << reused.generated_token_ids.size()
                      << " fresh=" << fresh.generated_token_ids.size()
                      << " repeated=" << reused_again.generated_token_ids.size()
                      << " limited=" << limited_tail.generated_token_ids.size() << std::endl;
            for (const auto* result : {&reused, &fresh, &reused_again, &limited_tail}) {
                if (!source.starts_with(kAssistantPrefix + stopped.content + result->content)) {
                    std::cerr << "source mismatch index=" << i
                              << " stopped=" << std::quoted(stopped.content)
                              << " continuation=" << std::quoted(result->content)
                              << " reused=" << result->reused_prompt_tokens << std::endl;
                }
                const auto count = result->generated_token_ids.size();
                require(count > 0 && count <= 16 &&
                            ((result->finish_reason == ninfer::FinishReason::OutputLimit &&
                              count == 16) ||
                             result->finish_reason == ninfer::FinishReason::StopToken),
                        "continuation violated its output budget or finish contract");
                require(source.starts_with(kAssistantPrefix + stopped.content + result->content),
                        "continuation is not an exact source prefix");
                accepted += result->speculative.ngram_accepted_tokens;
            }
            std::cout << "cache diagnostic index=" << i << " prompt=" << prompt_tokens
                      << " stopped=" << stopped.generated_token_ids.size()
                      << " follow_prompt=" << tail().summary().prompt_tokens
                      << " reused=" << reused.reused_prompt_tokens
                      << " repeated=" << reused_again.reused_prompt_tokens
                      << " limited=" << limited_tail.reused_prompt_tokens << std::endl;
            require(fresh.reused_prompt_tokens == 0 &&
                        reused.reused_prompt_tokens + 2 >= prompt_tokens + i + 1 &&
                        reused_again.reused_prompt_tokens + 2 >= prompt_tokens + i + 1 &&
                        limited_tail.reused_prompt_tokens + 2 >= prompt_tokens + i + 1,
                    "stop continuation did not reuse its generated prefix");
            const auto licensed = 1 + stopped.speculative.rounds +
                                  stopped.speculative.accepted_tokens +
                                  stopped.speculative.fallback_steps;
            partial_cases += stopped.generated_token_ids.size() < licensed;
            std::cout << "stop index=" << i
                      << " partial=" << (stopped.generated_token_ids.size() < licensed)
                      << " reused=" << reused.reused_prompt_tokens << " exact=1" << std::endl;
            ++cases;
        }
        require(cases >= 4 && partial_cases > 0, "insufficient partial-stop cases");
        require(ngram == 0 || accepted > 0, "continuations never accepted ngram drafts");
        std::cout << "canonical stop controls passed cases=" << cases
                  << " partial=" << partial_cases << " ngram_accepted=" << accepted
                  << " fresh_differences=" << fresh_differences << std::endl;
        // Upstream also differs between fresh prefill and a retained decode prefix.
        // Keep that stronger numerical diagnostic separate from commit/reuse correctness.
        return strict_fresh && fresh_differences ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
