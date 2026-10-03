#include "ninfer/engine.h"
#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void add(ninfer::PromptInput& input, ninfer::ChatRole role, const std::string& text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = text});
    input.messages.push_back(std::move(message));
}
struct Sink final : ninfer::OutputSink {
    std::uint32_t stop_after = 0;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (stop_after && value.generated_tokens >= stop_after) cancelled = true;
    }
};
}
int main(int argc, char** argv) {
    try {
        require(argc == 5 || argc == 6, "artifact int8|rk8v4 eager|graph ngram-width [mtp-drafts]");
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.device_profile = "off"; options.max_context = 8192; options.prefill_chunk = 128;
        options.context_cache.enabled = false;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = 2, .verify_transfers = true};
        options.kv_cache = std::string(argv[2]) == "rk8v4"
            ? ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 : ninfer::KvCacheStorage::Int8Group64;
        options.use_cuda_graph = std::string(argv[3]) == "graph";
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = argc == 6 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 3;
        options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(std::stoul(argv[4]));
        options.speculative.ngram_min_match = 4;
        ninfer::Engine engine(options);
        ninfer::PromptInput prompt;
        prompt.options.enable_thinking = false;
        prompt.context_cache.session_key = "copy-history";
        add(prompt, ninfer::ChatRole::System, "When asked to copy text, reproduce it verbatim, without explanations or quotation marks.");
        for (int i = 0; i < 8; ++i) {
            std::string filler;
            for (int j = 0; j < 8; ++j) filler += "The observatory records pulsars, galaxies and meteor showers. ";
            add(prompt, ninfer::ChatRole::User, filler);
            add(prompt, ninfer::ChatRole::Assistant, "Recorded.");
        }
        std::string passage = "The violet observatory opens at sunrise. ";
        for (int i = 0; i < 12; ++i)
            passage += "Its telescope observes distant galaxies, while the astronomers record the changing light of stars. ";
        add(prompt, ninfer::ChatRole::User, "Store this passage for exact copying later:\n" + passage);
        add(prompt, ninfer::ChatRole::Assistant, "The passage is stored.");
        add(prompt, ninfer::ChatRole::User, "Copy the stored passage exactly. Start with: The violet observatory opens at sunrise.");
        const auto run = [&](const char* name, const ninfer::PromptInput& input, bool hit,
                             std::uint32_t count = 192, std::uint32_t cancel = 0, float temperature = 0.0F) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = count;
            request.execution.sampling.temperature = temperature;
            request.execution.sampling.top_k = 20; request.execution.sampling.top_p = 0.9F;
            request.execution.sampling.seed = 42;
            request.execution.allow_prefix_reuse = true;
            request.stop.include_model_defaults = false;
            Sink sink; sink.stop_after = cancel;
            const auto result = engine.submit(engine.prepare(input), request,
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true})
                .wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
            const auto memory = engine.memory_summary().kvmem;
            require(engine.is_available(), "ngram made Engine unavailable");
            require((result.reused_prompt_tokens > 0) == hit, "ngram history reuse decision changed");
            require(memory.host_payload_bytes <= options.kvmem.host_bytes &&
                memory.resident_pages == 0 && memory.mtp_resident_pages == 0, "ngram leaked resources");
            require(cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                           : result.generated_token_ids.size() == count, "ngram output termination mismatch");
            std::cout << "MEMORY_NGRAM case=" << name << " reused=" << result.reused_prompt_tokens
                      << " generated=" << result.generated_token_ids.size()
                      << " rounds=" << result.speculative.ngram_rounds
                      << " drafted=" << result.speculative.ngram_drafted_tokens
                      << " accepted=" << result.speculative.ngram_accepted_tokens << std::endl;
            return result;
        };
        const auto cold = run("cold-copy", prompt, false);
        require(cold.content.find("violet observatory") != std::string::npos, "copy instruction failed");
        require(cold.speculative.ngram_rounds > 0 &&
                cold.speculative.ngram_accepted_tokens > 3 * cold.speculative.ngram_rounds,
                "wide ngram path did not accept more than neural draft width");
        const auto repeat = run("repeat-copy", prompt, true);
        require(cold.generated_token_ids == repeat.generated_token_ids, "restored ngram changed greedy output");
        run("cancel-copy", prompt, true, 192, 16);
        const auto recovered = run("recover-copy", prompt, true);
        require(cold.generated_token_ids == recovered.generated_token_ids, "cancelled ngram polluted history");
        for (auto count : {1U, 2U, 3U, 17U, 65U}) {
            const auto truncated = run("output-limit", prompt, true, count);
            require(truncated.generated_token_ids.front() == cold.generated_token_ids.front(),
                    "output truncation changed prompt's first token");
        }
        const auto sampled = run("sampling", prompt, true, 96, 0, 0.8F);
        const auto sampled_again = run("sampling-repeat", prompt, true, 96, 0, 0.8F);
        require(sampled.generated_token_ids == sampled_again.generated_token_ids,
                "ngram sampling did not restore logical RNG positions");
        auto edited = prompt;
        edited.messages[1].parts[0].text = "This archive was edited. The new observatory is amber.";
        run("edited-prefix", edited, false, 32);
        auto code = edited;
        code.messages.back().parts[0].text = "Copy the following Python code exactly, without markdown:\n";
        for (int i = 0; i < 12; ++i)
            code.messages.back().parts[0].text += "def square_" + std::to_string(i) + "(value):\n    return value * value\n\n";
        const auto copied_code = run("copy-code", code, true, 192);
        require(copied_code.speculative.ngram_accepted_tokens > 0, "code did not exercise ngram");
        std::cout << "MEMORY_NGRAM PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl; return 1;
    }
}
