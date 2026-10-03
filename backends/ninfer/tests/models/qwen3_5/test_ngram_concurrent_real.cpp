#include "ninfer/engine.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Real-artifact coverage for ngram copy drafting at --max-concurrency > 1.
//
// Every part runs on ONE Engine: a single ~20 GiB load serves all parts, so the test can share the
// GPU with a live serve. The decisive correctness contract is that a multi-request round must not
// corrupt any lane: at temperature 0 two lanes with the same prompt in the same batch decode to
// identical tokens, and a copy lane reproduces an exact source prefix.
namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

// A digit ends a tokenizer word, so this prefix cannot merge with the following token.
std::string assistant_prefix(int seed) {
    return "def transform_" + std::to_string(seed) + "(value):\n    offset = " +
           std::to_string(seed + 17);
}

std::string make_source(int seed, unsigned count = 20) {
    std::string source;
    for (unsigned i = 0; i < count; ++i) {
        const int value = seed + static_cast<int>(i);
        source += "def transform_" + std::to_string(value) + "(value):\n    offset = " +
                  std::to_string(value + 17) + "\n    return value * " +
                  std::to_string(value + 2) + " + offset\n\n";
    }
    return source;
}

// Byte offset where assistant_prefix + continuation first stops being a prefix of the source.
std::size_t first_source_difference(const std::string& source, const std::string& produced) {
    std::size_t i = 0;
    while (i < source.size() && i < produced.size() && source[i] == produced[i]) { ++i; }
    return i;
}

ninfer::RequestOptions request(unsigned output, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = output;
    options.execution.sampling.temperature       = 0.0F;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.execution.allow_prefix_reuse         = reuse;
    options.stop.include_model_defaults          = false;
    options.stop.publish_stop_token              = true;
    return options;
}

// A copy ends where the model closes its turn: models differ in how much of the file they repeat
// before they do, and what follows the turn end is no copy.
ninfer::RequestOptions copy_request(unsigned output, bool reuse = true) {
    ninfer::RequestOptions options      = request(output, reuse);
    options.stop.include_model_defaults = true;
    options.stop.publish_stop_token     = false;
    return options;
}

ninfer::PromptInput copy_prompt(const std::string& source, int seed) {
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
    assistant.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                  .text  = assistant_prefix(seed),
                                                  .media = {}});
    input.messages.push_back(std::move(assistant));
    input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
    return input;
}

ninfer::PromptInput freeform_prompt(const std::string& text) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.options.preserve_thinking = true;
    return input;
}

ninfer::EngineOptions options_for(const char* artifact, const std::string& backend, unsigned width,
                                  unsigned concurrency) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    // The context window is configurable so the test can run beside a live serve with reduced VRAM.
    unsigned max_context = 4096;
    if (const auto* ctx_env = std::getenv("NINFER_NGRAM_TEST_MAX_CONTEXT")) {
        max_context = std::max(512u, static_cast<unsigned>(std::strtoul(ctx_env, nullptr, 10)));
    }
    options.max_context         = max_context;
    options.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency     = concurrency;
    // Mixed rounds need a lane admitted while another still prefills.
    options.concurrent_prefill  = true;
    options.prefill_chunk       = 1024;
    options.enable_vision       = false;
    options.use_cuda_graph      = true;
    if (std::getenv("NINFER_NGRAM_TEST_NO_GRAPH")) { options.use_cuda_graph = false; }
    options.kv_cache            = ninfer::KvCacheStorage::Nvfp4Group16;
    if (backend == "mtp") {
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    } else if (backend == "dflash") {
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
    } else if (backend == "dflash2") {
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    } else {
        throw std::invalid_argument("unsupported backend");
    }
    options.speculative.draft_tokens       = backend == "mtp" ? 3 : 5;
    options.speculative.ngram_draft_tokens = width;
    options.speculative.proposal_head      = ninfer::ProposalHead::Optimized;
    // NINFER_NGRAM_TEST_CONTEXT_CACHE=hybrid runs the same checks on the hybrid prefix cache.
    if (const char* cache = std::getenv("NINFER_NGRAM_TEST_CONTEXT_CACHE");
        cache != nullptr && std::string_view(cache) == "hybrid") {
        options.kv_capacity =
            ninfer::KvCapacityPolicy::explicit_capacity(max_context * concurrency);
        options.context_cache.mode                    = ninfer::ContextCacheMode::Hybrid;
        options.context_cache.host_cache_budget_bytes = 1ULL << 30;
        return options;
    }
    options.context_cache.device_state_slots                  = 2;
    options.context_cache.host_state_slots                    = 8;
    options.context_cache.host_kv_capacity_bytes              = 256ULL << 20;
    options.context_cache.max_private_continuations           = 8;
    options.context_cache.max_shared_prefixes                 = 0;
    options.context_cache.max_long_anchors_per_continuation   = 0;
    return options;
}

std::size_t distinct_tokens(const std::vector<ninfer::TokenId>& ids) {
    std::size_t distinct = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        bool seen = false;
        for (std::size_t j = 0; j < i && !seen; ++j) { seen = ids[j] == ids[i]; }
        if (!seen) { ++distinct; }
    }
    return distinct;
}

// Longest run of one repeated token — a degenerate loop ("gibberish") shows up here.
std::size_t max_token_run(const std::vector<ninfer::TokenId>& ids) {
    std::size_t best = 0, run = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        run = (i > 0 && ids[i] == ids[i - 1]) ? run + 1 : 1;
        best = std::max(best, run);
    }
    return best;
}
} // namespace

int main(int argc, char** argv) {
    const auto* artifact = std::getenv("NINFER_NGRAM_TEST_WEIGHTS");
    if (!artifact || !*artifact) { artifact = std::getenv("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS"); }
    if (!artifact || !*artifact) { return 77; }
    try {
        const std::string backend         = argc > 1 ? argv[1] : "dflash2";
        const std::string_view width_text = argc > 2 ? argv[2] : "15";
        unsigned width                    = 0;
        const auto parsed =
            std::from_chars(width_text.data(), width_text.data() + width_text.size(), width);
        require(parsed.ec == std::errc{} && parsed.ptr == width_text.data() + width_text.size() &&
                    width <= 15,
                "concurrent ngram width must be an integer in 0..15 (0 = baseline)");
        const bool baseline = width == 0;
        const std::string_view concurrency_text = argc > 3 ? argv[3] : "2";
        unsigned concurrency                    = 0;
        const auto parsed_concurrency =
            std::from_chars(concurrency_text.data(), concurrency_text.data() + concurrency_text.size(),
                            concurrency);
        require(parsed_concurrency.ec == std::errc{} &&
                    parsed_concurrency.ptr == concurrency_text.data() + concurrency_text.size() &&
                    concurrency >= 2 && concurrency <= 8,
                "concurrency must be an integer in 2..8");

        ninfer::Engine engine(options_for(artifact, backend, width, concurrency));
        const std::string source = make_source(0);

        if (!baseline) {
        // Part 1: single-lane reference on the shared engine.
        std::vector<ninfer::TokenId> c1_tokens;
        {
            const auto result =
                engine.generate(engine.prepare(copy_prompt(source, 0)), copy_request(256));
            c1_tokens         = result.generated_token_ids;
            require(source.starts_with(assistant_prefix(0) + result.content),
                    "C1 reference is not an exact source prefix");
            require(result.speculative.ngram_accepted_tokens > 0,
                    "C1 reference did not engage ngram");
            std::cout << "c1 backend=" << backend << " width=" << width
                      << " tokens=" << c1_tokens.size()
                      << " ngram_accepted=" << result.speculative.ngram_accepted_tokens << "\n";
        }

        // Part 2: a single request on the C2 engine must reproduce the C1 path token-for-token.
        {
            const auto result =
                engine.generate(engine.prepare(copy_prompt(source, 0)), copy_request(256));
            require(source.starts_with(assistant_prefix(0) + result.content),
                    "C2 single request is not an exact source prefix");
            require(result.speculative.ngram_accepted_tokens > 0,
                    "C2 single request did not engage ngram");
            require(result.generated_token_ids == c1_tokens,
                    "C2 single-lane output is not token-identical to C1");
            std::cout << "c2-single tokens=" << result.generated_token_ids.size()
                      << " ngram_accepted=" << result.speculative.ngram_accepted_tokens
                      << " identical_to_c1=1\n";
        }

        // Part 3: two distinct copy requests in flight concurrently (a batch>1 ngram round).
        {
            const int seed_a = 100, seed_b = 200;
            const std::string source_a = make_source(seed_a);
            const std::string source_b = make_source(seed_b);
            auto handle_a =
                engine.submit(engine.prepare(copy_prompt(source_a, seed_a)), copy_request(256));
            auto handle_b =
                engine.submit(engine.prepare(copy_prompt(source_b, seed_b)), copy_request(256));
            const auto result_a = handle_a.wait();
            const auto result_b = handle_b.wait();
            require(source_a.starts_with(assistant_prefix(seed_a) + result_a.content),
                    "concurrent copy lane A is not an exact source prefix");
            require(source_b.starts_with(assistant_prefix(seed_b) + result_b.content),
                    "concurrent copy lane B is not an exact source prefix");
            const std::uint64_t total = result_a.speculative.ngram_accepted_tokens +
                                        result_b.speculative.ngram_accepted_tokens;
            require(total > 0, "concurrent copy requests did not engage ngram");
            std::cout << "c2-concurrent lanes=2 combined_ngram_accepted=" << total << "\n";
        }

        // Part 3b: a copy lane and a free-form lane in flight together, so ngram rounds carry a
        // row without a copy proposal. The copy lane must stay an exact source prefix. With a
        // masked drafter the free-form row keeps its neural proposal in those rounds instead of
        // decoding one token: only a final budget-limited round may verify no draft. The copy lane
        // offers its rewrite-checkpoint capture under the state pressure left by the cached
        // continuations above, and the free-form lane is admitted during its prefill. Whether that
        // capture commits or is skipped, the copy lane's later prefill steps must address its own
        // KV row rather than the row the other lane's staging bound last.
        {
            const int seed_a            = 300;
            const std::string source_a  = make_source(seed_a);
            const std::string freeform =
                "Explain in four short sentences how a hash table resolves collisions.";
            auto handle_a =
                engine.submit(engine.prepare(copy_prompt(source_a, seed_a)), copy_request(256));
            auto handle_b = engine.submit(engine.prepare(freeform_prompt(freeform)),
                                          request(192, false));
            const auto result_a = handle_a.wait();
            const auto result_b = handle_b.wait();
            const std::string produced_a = assistant_prefix(seed_a) + result_a.content;
            if (!source_a.starts_with(produced_a)) {
                const std::size_t at = first_source_difference(source_a, produced_a);
                const std::size_t from = at > 40 ? at - 40 : 0;
                std::cerr << "mixed copy lane diverges at byte " << at << " of " << produced_a.size()
                          << "\n  source:   [" << source_a.substr(from, 80) << "]\n  produced: ["
                          << produced_a.substr(from, 80) << "]\n";
            }
            require(source_a.starts_with(produced_a),
                    "mixed-round copy lane is not an exact source prefix");
            require(result_a.speculative.ngram_accepted_tokens > 0,
                    "mixed-round copy lane did not engage ngram");
            require(result_b.generated_token_ids.size() >= 64 &&
                        distinct_tokens(result_b.generated_token_ids) >= 16,
                    "mixed-round free-form lane output looks degenerate");
            const bool masked_drafter = backend != "mtp";
            require(!masked_drafter || result_b.speculative.fallback_steps <= 1,
                    "mixed-round free-form lane lost its neural proposal in ngram rounds");
            std::cout << "c2-mixed copy_ngram_accepted=" << result_a.speculative.ngram_accepted_tokens
                      << " freeform_rounds=" << result_b.speculative.rounds
                      << " freeform_fallback=" << result_b.speculative.fallback_steps
                      << " freeform_accepted=" << result_b.speculative.accepted_tokens << "\n";
        }
        } // if (!baseline)

        // Part 4: two concurrent free-form requests with no source in the prompt. This drives the
        // all-neural batch>1 round at the frame's native width. At temperature 0 two identical
        // prompts should decode to identical tokens whenever they share a batch round; a non-empty
        // first-divergence index with heavy co-batching would indicate lane corruption. Run with
        // width 0 as a baseline: the same comparison on an ngram-disabled engine tells whether any
        // divergence is intrinsic to batched decoding or introduced by the ngram paths.
        {
            const std::string freeform =
                "Describe in three short sentences why the sky appears blue during a clear day.";
            const auto stats_before = engine.runtime_stats();
            auto handle_a = engine.submit(engine.prepare(freeform_prompt(freeform)),
                                          request(192, false));
            auto handle_b = engine.submit(engine.prepare(freeform_prompt(freeform)),
                                          request(192, false));
            const auto result_a = handle_a.wait();
            const auto result_b = handle_b.wait();
            const auto stats_after  = engine.runtime_stats();
            const std::uint64_t rounds = stats_after.decode_rounds - stats_before.decode_rounds;
            const std::uint64_t row_rounds =
                stats_after.decode_row_rounds - stats_before.decode_row_rounds;
            require(result_a.generated_token_ids.size() >= 64 &&
                        result_b.generated_token_ids.size() >= 64,
                    "W-wide neural lane produced too few tokens");
            require(distinct_tokens(result_a.generated_token_ids) >= 16 &&
                        distinct_tokens(result_b.generated_token_ids) >= 16,
                    "W-wide neural lane output looks degenerate (token loop)");
            const auto& a = result_a.generated_token_ids;
            const auto& b = result_b.generated_token_ids;
            std::size_t first_diff = a.size();
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                if (a[i] != b[i]) { first_diff = i; break; }
            }
            const bool identical = a == b;
            std::cout << (baseline ? "baseline" : "c2") << "-wide-neural lanes=2 rounds=" << rounds
                      << " row_rounds=" << row_rounds << " tokens=" << a.size() << " identical="
                      << (identical ? 1 : 0) << " first_diff="
                      << (identical ? -1 : static_cast<long long>(first_diff)) << "\n";
            // Batched lanes are not guaranteed to share every round (they are prefetched and
            // admitted independently), and a batch-size change alone changes greedy near-ties —
            // the baseline run with ngram disabled diverges the same way. So cross-lane identity
            // is informational here; the corruption-sensitive contract is that no lane collapses
            // into a repeated-token loop.
            require(max_token_run(a) <= 16 && max_token_run(b) <= 16,
                    "free-form lane output collapsed into a token loop");
            require(distinct_tokens(a) >= 24 && distinct_tokens(b) >= 24,
                    "free-form lane output has too little variety");
        }

        if (!baseline) {
            // Part 5: a longer concurrent copy soak. Two long copies run together; both must remain
            // exact source prefixes up to where the model closes its turn, which catches slow
            // per-round state drift that a short run would miss. The source is long enough that the
            // whole output budget stays inside it (past the source end the model would legitimately
            // continue freely).
            {
                const int seed_a = 300, seed_b = 400;
                const std::string source_a = make_source(seed_a, 60);
                const std::string source_b = make_source(seed_b, 60);
                auto handle_a         = engine.submit(engine.prepare(copy_prompt(source_a, seed_a)),
                                                      copy_request(640, false));
                auto handle_b         = engine.submit(engine.prepare(copy_prompt(source_b, seed_b)),
                                                      copy_request(640, false));
                const auto result_a   = handle_a.wait();
                const auto result_b   = handle_b.wait();
                const auto produced_a = assistant_prefix(seed_a) + result_a.content;
                const auto produced_b = assistant_prefix(seed_b) + result_b.content;
                const auto diff_a     = first_source_difference(source_a, produced_a);
                const auto diff_b     = first_source_difference(source_b, produced_b);
                std::cout << "c2-soak lanes=2 tokens_a=" << result_a.generated_token_ids.size()
                          << " tokens_b=" << result_b.generated_token_ids.size()
                          << " source_bytes=" << source_a.size() << " first_diff_a=" << diff_a
                          << " first_diff_b=" << diff_b << "\n";
                require(source_a.starts_with(produced_a),
                        "long soak lane A drifted off its source prefix");
                require(source_b.starts_with(produced_b),
                        "long soak lane B drifted off its source prefix");
            }

        // Part 6: a long concurrent free-form soak. Two lanes decode hundreds of tokens together
        // with no source to copy. This is the direct probe for the reported "descends into
        // gibberish after a while" failure: a per-lane state or draft leak would eventually make a
        // lane repeat one token or produce a tiny alphabet. Both lanes must stay non-degenerate for
        // the whole run.
        {
            const std::string freeform =
                "Write a short essay about how rivers shape the land they flow through.";
            auto handle_a = engine.submit(engine.prepare(freeform_prompt(freeform)),
                                          request(512, false));
            auto handle_b = engine.submit(engine.prepare(freeform_prompt(freeform)),
                                          request(512, false));
            const auto result_a = handle_a.wait();
            const auto result_b = handle_b.wait();
            const auto& a        = result_a.generated_token_ids;
            const auto& b        = result_b.generated_token_ids;
            require(a.size() >= 256 && b.size() >= 256, "long soak produced too few tokens");
            std::cout << "c2-freeform-soak lanes=2 tokens=" << a.size()
                      << " distinct=" << distinct_tokens(a) << " max_run=" << max_token_run(a)
                      << " rows_ok=" << (max_token_run(b) <= 16) << "\n";
            require(max_token_run(a) <= 16 && max_token_run(b) <= 16,
                    "long free-form soak collapsed into a token loop (gibberish drift)");
            require(distinct_tokens(a) >= 48 && distinct_tokens(b) >= 48,
                    "long free-form soak lost output variety (gibberish drift)");
        }
        } // if (!baseline)

        std::cout << "ngram concurrent tests passed; backend=" << backend << " width=" << width
                  << " concurrency=" << concurrency << "\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
