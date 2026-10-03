#include "ninfer/engine.h"
#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Sink final : ninfer::OutputSink {
    bool cancel = false;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (cancel && value.generated_tokens >= 12) cancelled = true;
    }
};
}
int main(int argc, char** argv) {
    try {
        require(argc >= 7 && argc <= 9, "artifact drafts(0..4) int8|rk8v4 cache-dir cold|hit|error|cancel|clear quota-MiB [ngram-width] [short]");
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 4096; options.prefill_chunk = 128;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        options.kv_cache = std::string(argv[3]) == "rk8v4"
            ? ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 : ninfer::KvCacheStorage::Int8Group64;
        options.context_cache.enabled = false;
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = 2, .verify_transfers = true,
            .disk_path = argv[4], .disk_bytes = std::stoull(argv[6]) << 20};
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        if (argc >= 8) options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(std::stoul(argv[7]));
        ninfer::Engine engine(options);
        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        input.context_cache.session_key = "disk-history";
        const auto add = [&](ninfer::ChatRole role, std::string text) {
            ninfer::ChatMessage message;
            message.role = role;
            message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
            input.messages.push_back(std::move(message));
        };
        add(ninfer::ChatRole::System, "The archive access code is ORCHID-9491. Remember it.");
        for (int entry = 0; entry < (argc == 9 && std::string(argv[8]) == "short" ? 0 : 8); ++entry) {
            std::string text = "Entry " + std::to_string(entry) + ". ";
            for (int i = 0; i < 8; ++i) text += "The observatory records pulsars, galaxies and meteor showers. ";
            add(ninfer::ChatRole::User, text); add(ninfer::ChatRole::Assistant, "Recorded.");
        }
        add(ninfer::ChatRole::User, "Start with the access code, then describe the observatory.");
        const std::string mode = argv[5];
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = mode == "cancel" ? 128 : 32;
        request.execution.sampling.temperature = 0;
        request.execution.allow_prefix_reuse = mode != "clear";
        request.stop.include_model_defaults = false;
        Sink sink; sink.cancel = mode == "cancel";
        const auto result = engine.submit(engine.prepare(input), request,
            ninfer::OutputConsumerMode::Streaming, {.live_timings = true})
            .wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
        const auto memory = engine.memory_summary().kvmem;
        const bool hit = mode == "hit" || mode == "cancel";
        require((result.reused_prompt_tokens > 0) == hit, "wrong disk reuse decision");
        require((memory.disk_hits != 0) == hit, "disk hit counter does not match reuse");
        require(engine.is_available() && memory.resident_pages == 0 && memory.mtp_resident_pages == 0,
                "snapshot path leaked active resources or failed the Engine");
        require(memory.host_payload_bytes <= options.kvmem.host_bytes, "snapshot import exceeded H");
        require(mode == "cancel" ? result.finish_reason == ninfer::FinishReason::Cancelled
                                 : result.generated_token_ids.size() == 32, "wrong generation termination");
        if (mode != "cancel") require(result.content.find("ORCHID-9491") != std::string::npos, "wrong recovered archive code");
        if (mode == "error") require(memory.disk_errors > 0, "snapshot failure was not reported");
        if (mode != "error" && mode != "clear") require(memory.disk_writes > 0, "trusted checkpoint was not persisted");
        if (mode == "clear") require(memory.disk_writes == 0, "explicit clear republished the discarded history");
        std::cout << "MEMORY_DISK mode=" << mode << " reused=" << result.reused_prompt_tokens
                  << " hits=" << memory.disk_hits << " writes=" << memory.disk_writes
                  << " errors=" << memory.disk_errors << " host=" << memory.host_payload_bytes
                  << " accepted=" << result.speculative.accepted_tokens << std::endl;
        std::cout << "MEMORY_DISK PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
