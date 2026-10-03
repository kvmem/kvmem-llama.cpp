#include "ninfer/engine.h"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
ninfer::PromptInput history(const std::string& key) {
    ninfer::PromptInput input;
    input.context_cache.session_key = key;
    const auto add = [&](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
        input.messages.push_back(std::move(message));
    };
    add(ninfer::ChatRole::System, "Read archive " + key + " and answer briefly.");
    for (int entry = 0; entry < 16; ++entry) {
        std::string text = "Archive entry " + std::to_string(entry) + ". ";
        for (int repeat = 0; repeat < 8; ++repeat)
            text += entry < 4 ? "The violet observatory studies pulsars, cosmic rays and distant galaxies. "
                             : "The coastal office files invoices, transport schedules and warehouse inventories. ";
        add(ninfer::ChatRole::User, text);
        add(ninfer::ChatRole::Assistant, "Recorded.");
    }
    add(ninfer::ChatRole::User, "What does the violet observatory study?");
    return input;
}
struct Sink final : ninfer::OutputSink {
    bool cancel_enabled = false;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (cancel_enabled && value.generated_tokens >= 8) cancelled.store(true);
    }
};
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "artifact draft-count(0|3)");
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 16384;
        options.prefill_chunk = 128;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 230ULL << 20, .retained_sessions = 3, .verify_transfers = true};
        options.context_cache.enabled = false;
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        ninfer::Engine engine(options);
        const auto run = [&](const char* name, ninfer::PromptInput input, bool expected_reuse,
                             bool allow_reuse = true, bool cancel = false) {
            ninfer::RequestOptions request;
            request.execution.allow_prefix_reuse = allow_reuse;
            request.execution.requested_output_tokens = cancel ? 130 : 24;
            request.execution.sampling.temperature = 0;
            request.stop.include_model_defaults = false;
            Sink sink;
            sink.cancel_enabled = cancel;
            auto result = engine.submit(engine.prepare(std::move(input)), request,
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true})
                .wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
            require(engine.is_available(), "Engine became unavailable");
            require((result.reused_prompt_tokens != 0) == expected_reuse, "wrong session reuse decision");
            require(cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                           : result.generated_token_ids.size() == 24, "wrong termination");
            const auto memory = engine.memory_summary().kvmem;
            require(memory.host_payload_bytes <= options.kvmem.host_bytes, "global H exceeded");
            require(memory.retained_sessions <= 3, "session count limit exceeded");
            require(memory.resident_pages == 0 && memory.mtp_resident_pages == 0, "device pages leaked");
            std::cout << "MEMORY_SESSION " << name << " reused=" << result.reused_prompt_tokens
                      << " retained=" << memory.retained_sessions << " H=" << memory.host_payload_bytes
                      << " evictions=" << memory.history_evictions << std::endl;
            return result;
        };
        auto a = history("A"), b = history("B"), c = history("C"), d = history("D");
        const auto first = run("A-cold", a, false);
        run("B-cold", b, false);
        require(first.generated_token_ids == run("A-return", a, true).generated_token_ids,
                "A/B/A changed the trusted result");
        run("C-cold", c, false);
        run("D-pressure", d, false);
        require(engine.memory_summary().kvmem.history_evictions > 0, "Host pressure did not evict");
        run("B-evicted", b, false);
        run("C-return", c, true);
        run("C-cancel", c, true, true, true);
        run("C-resume", c, true);
        auto no_update = c;
        no_update.context_cache.update_session_index = false;
        no_update.messages.front().parts.front().text = "Read the temporary branch carefully.";
        run("C-no-update", no_update, false);
        run("C-index-preserved", c, true);
        run("C-clear", c, false, false);
        run("C-cleared", c, false);
        auto other_key = c;
        other_key.context_cache.session_key = "same-text-other-key";
        run("different-key", other_key, false);
        auto edited = other_key;
        edited.messages[1].parts[0].text = "The archive was edited.";
        run("edited-prefix", edited, false);
        run("edited-repeat", edited, true);
        std::cout << "MEMORY_SESSIONS PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
