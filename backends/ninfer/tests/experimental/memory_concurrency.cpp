#include "ninfer/engine.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ninfer::PromptInput history(const std::string& key, const std::string& code) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.context_cache.session_key = key;
    const auto add = [&](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
        input.messages.push_back(std::move(message));
    };
    add(ninfer::ChatRole::System, "Read archive " + key + ". Its access code is " + code + ".");
    for (int entry = 0; entry < 16; ++entry) {
        std::string text = "Entry " + std::to_string(entry) + ". ";
        for (int repeat = 0; repeat < 8; ++repeat)
            text += "The observatory files reports about pulsars, cosmic rays and distant galaxies. ";
        add(ninfer::ChatRole::User, text);
        add(ninfer::ChatRole::Assistant, "Recorded.");
    }
    add(ninfer::ChatRole::User, "Start with the archive access code, then explain what the observatory studies.");
    return input;
}
struct Sink final : ninfer::OutputSink {
    bool cancel_enabled = false;
    std::atomic<std::uint32_t>* started = nullptr;
    std::uint32_t row_bit = 0;
    std::uint32_t cancel_after_started = 0;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (value.generated_tokens && started) started->fetch_or(row_bit);
        if (cancel_enabled && value.generated_tokens >= 12 &&
            (!started || (started->load() & cancel_after_started) == cancel_after_started))
            cancelled.store(true);
    }
};
struct Work {
    ninfer::PromptInput prompt;
    std::uint32_t output = 96;
    bool cancel = false;
    std::string expected_code;
};
}

int main(int argc, char** argv) {
    try {
        require(argc == 5, "artifact draft-count(0|3) concurrency(2|4) host-MiB");
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        const auto concurrency = static_cast<std::uint32_t>(std::stoul(argv[3]));
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 8192;
        options.max_concurrency = concurrency;
        // This correctness run synchronously verifies every transferred KV/state
        // byte; queued requests must survive that diagnostic cost.
        options.pending_timeout_ms = 600000;
        options.prefill_chunk = 128;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512 * concurrency);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = std::stoull(argv[4]) << 20, .retained_sessions = 4,
            .verify_transfers = true};
        options.context_cache.enabled = false;
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        ninfer::Engine engine(options);
        const auto batch = [&](const char* label, std::vector<Work> jobs, bool expect_batch) {
            std::vector<ninfer::PreparedPrompt> prepared;
            for (auto& job : jobs) prepared.push_back(engine.prepare(job.prompt));
            const auto before = engine.runtime_stats();
            std::vector<std::future<ninfer::GenerationResult>> futures;
            std::atomic<std::uint32_t> started{0};
            const auto cancel_after_started = expect_batch ? (1U << jobs.size()) - 1U : 0U;
            for (std::size_t row = 0; row < jobs.size(); ++row) {
                ninfer::RequestOptions request;
                request.execution.requested_output_tokens = jobs[row].output;
                request.execution.sampling.temperature = 0;
                request.stop.include_model_defaults = false;
                auto handle = engine.submit(std::move(prepared[row]), request,
                    ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
                futures.push_back(std::async(std::launch::async,
                    [handle = std::move(handle), cancel = jobs[row].cancel, row,
                     &started, cancel_after_started]() mutable {
                        Sink sink;
                        sink.cancel_enabled = cancel;
                        sink.started = &started;
                        sink.row_bit = 1U << row;
                        sink.cancel_after_started = cancel_after_started;
                        return handle.wait(&sink, ninfer::CancellationView([&] {
                            return sink.cancelled.load();
                        }));
                    }));
            }
            std::uint32_t peak_running = 0, peak_waiting = 0;
            while (std::any_of(futures.begin(), futures.end(), [](auto& future) {
                return future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
            })) {
                const auto stats = engine.runtime_stats();
                peak_running = std::max(peak_running, stats.running_requests);
                peak_waiting = std::max(peak_waiting, stats.waiting_requests);
                require(stats.kvmem.host_payload_bytes <= options.kvmem.host_bytes,
                        "concurrent archives exceeded global H");
                require(stats.kvmem.active_host_reservation_bytes <= options.kvmem.host_bytes,
                        "admission overcommitted global H");
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            std::vector<ninfer::GenerationResult> results;
            for (std::size_t row = 0; row < futures.size(); ++row) {
                auto result = futures[row].get();
                require(jobs[row].cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                                        : result.generated_token_ids.size() == jobs[row].output,
                        "one lane affected another lane's termination");
                if (!jobs[row].cancel && drafts)
                    require(result.speculative.accepted_tokens > 0, "MTP was silently disabled");
                if (!jobs[row].cancel && !jobs[row].expected_code.empty())
                    require(result.content.find(jobs[row].expected_code) != std::string::npos,
                            "request returned the wrong archive access code");
                results.push_back(std::move(result));
            }
            require(engine.is_available(), "concurrent run made Engine unavailable");
            const auto after = engine.runtime_stats();
            const auto rounds = after.decode_rounds - before.decode_rounds;
            const auto rows = after.decode_row_rounds - before.decode_row_rounds;
            std::cout << "MEMORY_BATCH " << label << " rounds=" << rounds << " rows=" << rows
                      << " peak_running=" << peak_running << " peak_waiting=" << peak_waiting << std::endl;
            if (expect_batch) require(rows > rounds && peak_running >= 2,
                                      "requests did not execute in a compact decode batch");
            if (options.kvmem.host_bytes < (256ULL << 20) && jobs.size() > 1)
                require(peak_running == 1 && peak_waiting > 0,
                        "Host pressure did not queue requests until a reservation was released");
            const auto memory = engine.memory_summary().kvmem;
            require(memory.resident_pages == 0 && memory.mtp_resident_pages == 0 &&
                    memory.active_host_reservation_bytes == 0, "terminal lanes retained active resources");
            std::cout << "MEMORY_CONCURRENCY " << label << " rounds=" << rounds << " rows=" << rows
                      << " peak_running=" << peak_running << " peak_waiting=" << peak_waiting
                      << " host=" << memory.host_payload_bytes << std::endl;
            return results;
        };
        const bool pressure = options.kvmem.host_bytes < (256ULL << 20);
        std::vector<Work> initial;
        for (std::uint32_t row = 0; row < concurrency; ++row)
            initial.push_back({history("lane-" + std::to_string(row), "ORCHID-" + std::to_string(7100 + row)),
                               192 - row * 16, false, "ORCHID-" + std::to_string(7100 + row)});
        batch("mixed-prefill", initial, !pressure);
        initial.front().cancel = true;
        batch("cancel-one", initial, !pressure);
        initial.front().cancel = false;
        for (auto& job : initial) job.output = 32;
        batch("resume-peers", initial, !pressure);
        if (!pressure) {
            auto older = history("shared", "OLDER-8010");
            auto newer = history("shared", "NEWER-9020");
            batch("publication-order", {{older, 160, false, "OLDER-8010"},
                                        {newer, 24, false, "NEWER-9020"}}, true);
            const auto repeat = batch("newer-retained", {{newer, 24, false}}, false);
            require(repeat.front().reused_prompt_tokens > 0, "older late finish replaced newer session history");
        }
        std::cout << "MEMORY_CONCURRENCY PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
