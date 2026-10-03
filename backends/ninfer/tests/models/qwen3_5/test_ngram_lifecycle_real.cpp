#include "ninfer/engine.h"
#include "speculative_page_boundary.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

std::uint64_t ngram_rounds(const ninfer::GenerationResult& result) {
    return result.speculative.ngram_rounds;
}

std::uint64_t ngram_accepted(const ninfer::GenerationResult& result) {
    return result.speculative.ngram_accepted_tokens;
}

std::string copy_prefill(const std::string& source, const std::string& prefix) {
    return "Copy this Python file exactly, preserving every space and newline:\n\n" + source +
           "Output only the same file, without explanation or Markdown:\n\n" + prefix;
}

ninfer::RequestOptions request(unsigned output, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens    = output;
    options.execution.sampling.temperature       = 0;
    options.execution.sampling.presence_penalty  = 0;
    options.execution.sampling.frequency_penalty = 0;
    options.execution.allow_prefix_reuse         = reuse;
    options.stop.include_model_defaults          = false;
    return options;
}

struct Sink : ninfer::OutputSink {
    std::atomic<std::size_t> bytes{0};

    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override { bytes += delta.text.size(); }
};
} // namespace

int main(int argc, char** argv) {
    const auto* artifact = std::getenv("NINFER_NGRAM_TEST_WEIGHTS");
    if (!artifact || !*artifact) { artifact = std::getenv("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS"); }
    if (!artifact || !*artifact) { return 77; }
    try {
        const bool strict_fresh = argc > 1 && std::string(argv[argc - 1]) == "--strict-fresh";
        if (strict_fresh) { --argc; }
        const unsigned ngram_k = argc > 1 ? std::stoul(argv[1]) : 15U;
        ninfer::EngineOptions options;
        options.artifact_path       = artifact;
        options.max_context         = 4096;
        options.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.max_concurrency     = 1;
        options.prefill_chunk       = 1024;
        options.enable_vision       = false;
        options.kv_cache            = ninfer::KvCacheStorage::Nvfp4Group16;
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        const std::string backend   = argc > 8 ? argv[8] : "dflash2";
        if (backend == "mtp") {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        } else if (backend == "dflash") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        } else if (backend != "dflash2") {
            throw std::invalid_argument("unsupported fixture backend");
        }
        options.speculative.draft_tokens       = 5;
        options.speculative.ngram_draft_tokens = ngram_k;
        options.speculative.proposal_head      = ninfer::ProposalHead::Optimized;
        if (argc > 2) { options.speculative.draft_tokens = std::stoul(argv[2]); }
        if (argc > 3) { options.use_cuda_graph = std::stoi(argv[3]) != 0; }
        if (argc > 4 && std::stoi(argv[4]) == 0) {
            options.speculative.proposal_head = ninfer::ProposalHead::Full;
        }
        const std::string codec = argc > 5 ? argv[5] : "nvfp4";
        if (codec == "bf16") {
            options.kv_cache = ninfer::KvCacheStorage::BFloat16;
        } else if (codec == "int8") {
            options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        } else if (codec == "fp8") {
            options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        } else if (codec == "k8v4") {
            options.kv_cache = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
        } else if (codec != "nvfp4") {
            throw std::invalid_argument("unsupported fixture codec");
        }
        if (argc > 6) { options.speculative.ngram_min_match = std::stoul(argv[6]); }
        std::cout << "ngram=" << ngram_k << " artifact=" << artifact << " backend=" << backend
                  << " neural=" << options.speculative.draft_tokens
                  << " graph=" << options.use_cuda_graph << " full_head="
                  << (options.speculative.proposal_head == ninfer::ProposalHead::Full)
                  << " codec=" << codec << std::endl;
        options.context_cache.device_state_slots                = 0;
        options.context_cache.host_state_slots                  = 4;
        options.context_cache.host_kv_capacity_bytes            = 256ULL << 20;
        options.context_cache.max_private_continuations         = 2;
        options.context_cache.max_shared_prefixes               = 0;
        options.context_cache.max_long_anchors_per_continuation = 0;
        std::string text;
        for (int i = 0; i < 240; ++i) {
            text += "Reference record " + std::to_string(i) + ": the status remains unchanged.\n";
        }
        std::string source;
        for (int i = 0; i < 20; ++i) {
            source += "def transform_" + std::to_string(i) +
                      "(value):\n    offset = " + std::to_string(i + 17) + "\n    return value * " +
                      std::to_string(i + 2) + " + offset\n\n";
        }
        // Enter generation with a source match longer than the default admission threshold.
        text += copy_prefill(source, "def transform_0(value):\n    offset = 17");
        if (argc > 9) {
            if (std::string(argv[9]) == "concurrency" ||
                std::string(argv[9]) == "concurrency-raw") {
                require(ngram_k == 0, "multi-slot regression must leave ngram disabled");
                const bool canonical    = std::string(argv[9]) == "concurrency";
                options.max_concurrency = 2;
                ninfer::Engine engine(options);
                std::vector<std::vector<ninfer::TokenId>> prompts, references;
                std::vector<ninfer::PromptInput> chat_prompts;
                std::vector<std::string> reference_texts;
                const auto prepare = [&](std::size_t row) {
                    return canonical ? engine.prepare(chat_prompts[row])
                                     : engine.prepare_tokens(prompts[row]);
                };
                for (const std::string name : {"alpha", "beta"}) {
                    std::string file;
                    for (int i = 0; i < (canonical ? 20 : 8); ++i) {
                        file += "def " + name + "_" + std::to_string(i) +
                                "(value):\n    return value + " + std::to_string(i + 17) + "\n\n";
                    }
                    prompts.push_back(engine.tokenize_text(
                        "Here is a Python file:\n\n" + file +
                        "Here is the same Python file again, unchanged:\n\ndef " + name +
                        "_0(value):\n"));
                    ninfer::PromptInput input;
                    ninfer::ChatMessage user;
                    user.role = ninfer::ChatRole::User;
                    user.parts.push_back(ninfer::MessagePart{
                        .kind = ninfer::MessagePartKind::Text,
                        .text = "Repeat the following Python file exactly. Output only the file, "
                                "with no "
                                "explanation or Markdown fences. Do not make any changes.\n\n" +
                                file,
                        .media = {}});
                    input.messages.push_back(std::move(user));
                    input.options.enable_thinking = false;
                    input.context_cache.retention = ninfer::CacheRetentionHint::Disposable;
                    chat_prompts.push_back(std::move(input));
                    const auto reference =
                        engine.generate(prepare(prompts.size() - 1), request(96, false));
                    references.push_back(reference.generated_token_ids);
                    reference_texts.push_back(reference.content);
                }
                require(references[0].size() == 96 && references[1].size() == 96 &&
                            references[0] != references[1],
                        "multi-slot reference fixture is not distinct");
                const auto before        = engine.runtime_stats();
                auto first               = engine.submit(prepare(0), request(96, false));
                auto second              = engine.submit(prepare(1), request(96, false));
                const auto first_result  = first.wait();
                const auto second_result = second.wait();
                const bool exact         = first_result.generated_token_ids == references[0] &&
                                   second_result.generated_token_ids == references[1];
                std::cout << "canonical=" << canonical << " batch/single equality: tokens=" << exact
                          << " text0=" << (first_result.content == reference_texts[0])
                          << " text1=" << (second_result.content == reference_texts[1])
                          << std::endl;
                if (!exact) {
                    std::cout << "C2_REFERENCE_ALPHA_BEGIN\n"
                              << reference_texts[0] << "\nC2_ACTUAL_ALPHA_BEGIN\n"
                              << first_result.content << "\nC2_REFERENCE_BETA_BEGIN\n"
                              << reference_texts[1] << "\nC2_ACTUAL_BETA_BEGIN\n"
                              << second_result.content << "\nC2_OUTPUTS_END" << std::endl;
                }
                require(first_result.generated_token_ids.size() == 96 &&
                            second_result.generated_token_ids.size() == 96,
                        "batched row escaped its output budget");
                const auto after = engine.runtime_stats();
                require(after.decode_row_rounds - before.decode_row_rounds >
                            after.decode_rounds - before.decode_rounds,
                        "multi-slot test never batched two rows");
                require(engine.generate(prepare(0), request(96, false)).generated_token_ids ==
                            references[0],
                        "multi-slot server failed to return to a single-row batch");
                std::cout << "multi-slot budgets, batching and single-row recovery passed rounds="
                          << after.decode_rounds - before.decode_rounds
                          << " row_rounds=" << after.decode_row_rounds - before.decode_row_rounds
                          << std::endl;
                if (!exact) {
                    std::cerr << "batch/single token equivalence remains unresolved" << std::endl;
                    return 2;
                }
                return 0;
            }
            require(std::string(argv[9]) == "sanitizer-smoke", "unsupported fixture mode");
            ninfer::Engine engine(options);
            const auto prompt = engine.tokenize_text(
                copy_prefill(source, "def transform_0(value):\n    offset = 17"));
            // One prefill-sampled token must leave room for every draft plus its bonus.
            const auto output_budget = std::max(64U, ngram_k + 2U);
            const auto first =
                engine.generate(engine.prepare_tokens(prompt), request(output_budget, false));
            const auto second =
                engine.generate(engine.prepare_tokens(prompt), request(output_budget, false));
            require(first.generated_token_ids.size() == output_budget &&
                        first.generated_token_ids == second.generated_token_ids &&
                        (ngram_k == 0 || (ngram_accepted(first) > 0 && ngram_accepted(second) > 0)),
                    "real-model sanitizer smoke did not repeat an active ngram continuation");
            if (ngram_k == 63) {
                for (const auto* result : {&first, &second}) {
                    require(result->speculative.accepted_per_position.size() >= ngram_k &&
                                result->speculative.accepted_per_position[ngram_k - 1] > 0,
                            "wide sanitizer smoke did not accept the last ngram draft");
                }
            }
            std::cout << "real-model sanitizer smoke passed output_budget=" << output_budget
                      << " ngram_accepted=" << ngram_accepted(first) + ngram_accepted(second)
                      << " last_draft_accepted="
                      << (ngram_k == 0
                              ? 0
                              : first.speculative.accepted_per_position.at(ngram_k - 1) +
                                    second.speculative.accepted_per_position.at(ngram_k - 1))
                      << std::endl;
            return 0;
        }
        std::vector<ninfer::TokenId> device_retained, device_restored;
        {
            auto control_options                                 = options;
            control_options.context_cache.device_state_slots     = 2;
            control_options.context_cache.host_state_slots       = 0;
            control_options.context_cache.host_kv_capacity_bytes = 0;
            ninfer::Engine control(control_options);
            auto control_prompt = control.tokenize_text(text);
            device_retained =
                control.generate(control.prepare_tokens(control_prompt), request(256, true))
                    .generated_token_ids;
            control_prompt.insert(control_prompt.end(), device_retained.begin(),
                                  device_retained.end());
            control_prompt.push_back(198);
            const auto control_restore =
                control.generate(control.prepare_tokens(control_prompt), request(8, true));
            device_restored = control_restore.generated_token_ids;
            const auto control_fresh =
                control.generate(control.prepare_tokens(control_prompt), request(8, false));
            const auto stats = control.runtime_stats();
            require(control_restore.reused_prompt_tokens > 0 && stats.state_h2d_count == 0 &&
                        stats.main_kv_h2d_pages == 0,
                    "device control did not retain its checkpoint");
            std::cout << "device control: reused=" << control_restore.reused_prompt_tokens
                      << " equals_fresh=" << (device_restored == control_fresh.generated_token_ids)
                      << std::endl;
        }
        ninfer::Engine engine(options);
        const auto prompt = engine.tokenize_text(text);
        require(prompt.size() > 2048 && prompt.size() + 512 < 4096,
                "pressure fixture size invalid");
        const auto retained = engine.generate(engine.prepare_tokens(prompt), request(256, true));
        if (retained.generated_token_ids.size() != 256 ||
            (ngram_k != 0 && ngram_accepted(retained) == 0)) {
            std::cerr << "retained-source diagnostic tokens=" << retained.generated_token_ids.size()
                      << " ngram_rounds=" << ngram_rounds(retained)
                      << " ngram_accepted=" << ngram_accepted(retained)
                      << " content=" << std::quoted(retained.content) << std::endl;
        }
        require(retained.generated_token_ids.size() == 256 &&
                    (ngram_k == 0 || ngram_accepted(retained) > 0),
                "retained source did not use ngram");
        require(retained.generated_token_ids == device_retained,
                "host/device control workloads already differ before spill");
        auto follow = prompt;
        follow.insert(follow.end(), retained.generated_token_ids.begin(),
                      retained.generated_token_ids.end());
        follow.push_back(198);
        const auto before   = engine.runtime_stats();
        const auto pressure = engine.generate(engine.prepare_tokens(follow), request(8, false));
        const auto spilled  = engine.runtime_stats();
        const auto restored = engine.generate(engine.prepare_tokens(follow), request(8, true));
        const auto after    = engine.runtime_stats();
        std::cout << "host spill: state=" << spilled.state_d2h_count - before.state_d2h_count
                  << " kv_pages=" << spilled.main_kv_d2h_pages - before.main_kv_d2h_pages
                  << " restore_state=" << after.state_h2d_count - spilled.state_h2d_count
                  << " restore_pages=" << after.main_kv_h2d_pages - spilled.main_kv_h2d_pages
                  << " reused=" << restored.reused_prompt_tokens << std::endl;
        require(spilled.state_d2h_count > 0 && spilled.main_kv_d2h_pages > before.main_kv_d2h_pages,
                "host spill not exercised");
        require(after.state_h2d_count > spilled.state_h2d_count &&
                    after.main_kv_h2d_pages > spilled.main_kv_h2d_pages &&
                    restored.reused_prompt_tokens > 0,
                "host restore not exercised");
        require(restored.generated_token_ids == device_restored,
                "host restore differs from the retained-on-device control");
        bool fresh_equivalent = restored.generated_token_ids == pressure.generated_token_ids;
        std::cout << "host/device retained match; equals_fresh=" << fresh_equivalent << std::endl;

        bool partial = false;
        for (std::size_t i = 16; i < 96; ++i) {
            const auto token = retained.generated_token_ids[i];
            if (std::find(retained.generated_token_ids.begin(),
                          retained.generated_token_ids.begin() + i,
                          token) != retained.generated_token_ids.begin() + i) {
                continue;
            }
            auto stop = request(256, true);
            stop.stop.token_ids.push_back(token);
            const auto result   = engine.generate(engine.prepare_tokens(prompt), stop);
            const auto licensed = 1 + result.speculative.rounds +
                                  result.speculative.accepted_tokens +
                                  result.speculative.fallback_steps;
            if (result.finish_reason != ninfer::FinishReason::StopToken ||
                result.generated_token_ids.size() >= licensed ||
                (ngram_k != 0 && ngram_rounds(result) == 0)) {
                continue;
            }
            auto tail = prompt;
            tail.insert(tail.end(), result.generated_token_ids.begin(),
                        result.generated_token_ids.end());
            tail.push_back(198);
            const auto reuse = engine.generate(engine.prepare_tokens(tail), request(8, true));
            const auto fresh = engine.generate(engine.prepare_tokens(tail), request(8, false));
            const bool stop_fresh_equal = reuse.generated_token_ids == fresh.generated_token_ids;
            fresh_equivalent            = fresh_equivalent && stop_fresh_equal;
            std::cout << "partial stop equals_fresh=" << stop_fresh_equal << std::endl;
            partial = true;
            std::cout << "partial stop exercised at output=" << result.generated_token_ids.size()
                      << std::endl;
            break;
        }
        require(partial, "partial stop fixture did not exercise ngram");
        Sink sink;
        const auto cancelled =
            engine.generate(engine.prepare_tokens(prompt), request(512, true), &sink,
                            ninfer::CancellationView([&] { return sink.bytes.load() >= 128; }));
        require(cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                    cancelled.generated_token_ids.size() < 512 &&
                    (ngram_k == 0 || ngram_rounds(cancelled) > 0),
                "stream cancellation did not interrupt an ngram request");
        const auto recovered = engine.generate(engine.prepare_tokens(prompt), request(256, false));
        require(recovered.generated_token_ids == retained.generated_token_ids,
                "cancelled request contaminated subsequent fresh generation");
        std::cout << "ngram cancellation and recovery passed" << std::endl;

        std::vector<std::vector<ninfer::TokenId>> queued_prompts, expected;
        for (const std::string name : {"alpha", "beta", "gamma"}) {
            std::string file;
            for (int i = 0; i < 12; ++i) {
                file += "def " + name + "_" + std::to_string(i) + "(value):\n    return value + " +
                        std::to_string(i + 17) + "\n\n";
            }
            queued_prompts.push_back(engine.tokenize_text(
                copy_prefill(file, "def " + name + "_0(value):\n    return value + 17")));
            expected.push_back(
                engine.generate(engine.prepare_tokens(queued_prompts.back()), request(96, false))
                    .generated_token_ids);
        }
        require(expected[0] != expected[1] && expected[1] != expected[2],
                "queued isolation fixture did not produce distinct outputs");
        std::vector<ninfer::GenerationHandle> handles;
        for (const auto& queued : queued_prompts) {
            handles.push_back(engine.submit(engine.prepare_tokens(queued), request(96, false)));
        }
        for (std::size_t i = 0; i < handles.size(); ++i) {
            require(handles[i].wait().generated_token_ids == expected[i],
                    "queued request changed an independent continuation");
        }
        auto first = engine.submit(engine.prepare_tokens(prompt), request(256, false));
        {
            auto abandoned =
                engine.submit(engine.prepare_tokens(queued_prompts[1]), request(256, false));
        }
        auto last = engine.submit(engine.prepare_tokens(queued_prompts[2]), request(96, false));
        require(first.wait().generated_token_ids == retained.generated_token_ids &&
                    last.wait().generated_token_ids == expected[2],
                "abandoned handle corrupted neighboring requests");
        std::cout << "queued isolation and abandoned-handle recovery passed" << std::endl;
        const unsigned soak         = argc > 7 ? std::stoul(argv[7]) : 0U;
        std::uint64_t soak_accepted = 0;
        for (unsigned i = 0; i < soak; ++i) {
            const auto row = i % queued_prompts.size();
            if (i % 7 == 0) {
                auto abandoned = engine.submit(engine.prepare_tokens(queued_prompts[(row + 1) % 3]),
                                               request(96, false));
            }
            const auto result =
                engine.generate(engine.prepare_tokens(queued_prompts[row]), request(24, false));
            require(result.generated_token_ids.size() == 24 &&
                        std::equal(result.generated_token_ids.begin(),
                                   result.generated_token_ids.end(), expected[row].begin()),
                    "soak request changed its reference continuation");
            soak_accepted += ngram_accepted(result);
        }
        if (soak != 0) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            ninfer::RuntimeStats stats;
            do {
                stats = engine.runtime_stats();
                if (stats.running_requests == 0 && stats.waiting_requests == 0 &&
                    stats.terminal_pending_requests == 0) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } while (std::chrono::steady_clock::now() < deadline);
            require(stats.running_requests == 0 && stats.waiting_requests == 0 &&
                        stats.terminal_pending_requests == 0,
                    "soak did not drain its request queue");
            require(stats.device_state_occupied_slots <= options.max_concurrency &&
                        stats.host_state_occupied_slots <= 4 &&
                        stats.host_kv_occupied_bytes <= (256ULL << 20) &&
                        stats.device_main_kv_occupied_pages <= 64,
                    "soak exceeded its bounded context pools");
            require(ngram_k == 0 || soak_accepted > 0,
                    "soak fixture did not accept ngram proposals");
            std::cout << "soak passed requests=" << soak << " ngram_accepted=" << soak_accepted
                      << std::endl;
        }
        const unsigned maximum_budget = std::max(32U, ngram_k + 1);
        for (unsigned budget = 1; budget <= maximum_budget; ++budget) {
            const auto result =
                engine.generate(engine.prepare_tokens(queued_prompts[0]), request(budget, false));
            require(result.finish_reason == ninfer::FinishReason::OutputLimit &&
                        result.generated_token_ids.size() == budget &&
                        std::equal(result.generated_token_ids.begin(),
                                   result.generated_token_ids.end(), expected[0].begin()),
                    "output budget changed or exceeded its licensed reference prefix");
        }
        ninfer::test::speculative_page_boundary(engine);
        for (const unsigned gap : {1U, 4U, 15U, 16U, 31U, 32U, 33U, 63U, 64U}) {
            std::vector<ninfer::TokenId> tail_prompt(options.max_context - gap, 198);
            tail_prompt.back() = prompt.back();
            const auto tail =
                engine.generate(engine.prepare_tokens(tail_prompt), request(gap + 8, false));
            require(tail.finish_reason == ninfer::FinishReason::ContextCapacity &&
                        tail.generated_token_ids.size() == gap + 1,
                    "speculative verification escaped the context capacity tail");
        }
        std::cout << "all budgets 1.." << maximum_budget
                  << ", partial page boundary and context tails passed" << std::endl;
        std::cout << "retained-state lifecycle checks passed; fresh_identity=" << fresh_equivalent
                  << std::endl;
        if (strict_fresh && !fresh_equivalent) {
            std::cerr << "strict fresh/cached identity diagnostic failed" << std::endl;
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
