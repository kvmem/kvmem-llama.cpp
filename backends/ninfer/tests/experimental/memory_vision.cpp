#include "ninfer/engine.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ninfer::OwnedMedia media(const char* path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "image fixture is missing");
    ninfer::OwnedMedia value;
    value.bytes.assign(std::istreambuf_iterator<char>(file), {});
    value.media_type = "image/png";
    value.source_name = "image.png"; // Content identity must not depend on the file name.
    return value;
}
void add(ninfer::PromptInput& input, ninfer::ChatRole role, std::string text,
         std::vector<ninfer::OwnedMedia> images = {}) {
    ninfer::ChatMessage message;
    message.role = role;
    for (auto& image : images)
        message.parts.push_back({.kind = ninfer::MessagePartKind::Media, .media = std::move(image)});
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
}
void filler(ninfer::PromptInput& input) {
    for (int index = 0; index < 12; ++index) {
        std::string text = "Archive entry " + std::to_string(index) + ". ";
        for (int count = 0; count < 8; ++count)
            text += "The observatory records pulsars, galaxies, stars and meteor showers. ";
        add(input, ninfer::ChatRole::User, text);
        add(input, ninfer::ChatRole::Assistant, "Recorded.");
    }
}
struct Sink final : ninfer::OutputSink {
    bool cancel_enabled = false;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (cancel_enabled && value.generated_tokens >= 12) cancelled.store(true);
    }
};
}

int main(int argc, char** argv) {
    try {
        require(argc == 6, "artifact draft-count(0|3) native|kvmem red.png blue.png");
        const auto red = media(argv[4]), blue = media(argv[5]);
        const bool memory = std::string(argv[3]) == "kvmem";
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 8192;
        options.prefill_chunk = 128;
        options.enable_vision = true;
        options.vision_max_merged_tokens = 256;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(memory ? 896 : 8192);
        options.context_cache.enabled = false;
        if (memory) options.kvmem = {.selected_tokens = 768, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = 4, .verify_transfers = true};
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        ninfer::Engine engine(options);
        const auto input = [](const char* key) {
            ninfer::PromptInput result;
            result.options.enable_thinking = false;
            result.context_cache.session_key = key;
            add(result, ninfer::ChatRole::System, "Answer questions about the supplied images in English.");
            return result;
        };
        const auto run = [&](const char* label, ninfer::PromptInput prompt,
                             std::vector<std::string> expected, bool reused, bool cancel = false) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = cancel ? 160 : 48;
            request.execution.sampling.temperature = 0;
            request.stop.include_model_defaults = false;
            Sink sink;
            sink.cancel_enabled = cancel;
            const auto result = engine.submit(engine.prepare(std::move(prompt)), request,
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true})
                .wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
            require(engine.is_available(), "image request made Engine unavailable");
            require((result.reused_prompt_tokens != 0) == (memory && reused), "wrong image reuse decision");
            require(cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                           : result.generated_token_ids.size() == 48, "wrong image request termination");
            auto text = result.content;
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
            for (const auto& word : expected)
                require(text.find(word) != std::string::npos, "image answer did not match fixture");
            if (memory) {
                const auto usage = engine.memory_summary().kvmem;
                require(usage.resident_pages == 0 && usage.mtp_resident_pages == 0 &&
                    usage.host_payload_bytes <= options.kvmem.host_bytes, "image request leaked or exceeded resources");
            }
            std::cout << "MEMORY_VISION " << label << " reused=" << result.reused_prompt_tokens
                      << " output=" << result.generated_token_ids.size() << " accepted="
                      << result.speculative.accepted_tokens << " text=" << result.content << std::endl;
        };
        auto short_image = input("short");
        add(short_image, ninfer::ChatRole::User, "What color fills this image?", {red});
        run("single", short_image, {"red"}, false);

        auto old_image = input("old-image");
        add(old_image, ninfer::ChatRole::User, "Remember this image for a later question.", {red});
        add(old_image, ninfer::ChatRole::Assistant, "I will remember it.");
        filler(old_image);
        add(old_image, ninfer::ChatRole::User, "What color filled the image at the start?");
        run("old-image-cold", old_image, {"red"}, false);
        run("old-image-repeat", old_image, {"red"}, true);
        auto replacement = old_image;
        replacement.messages[1].parts[0].media = blue;
        run("same-placeholder-new-media", replacement, {"blue"}, false);
        run("replacement-repeat", replacement, {"blue"}, true);
        run("cancel", replacement, {}, true, true);
        run("cancel-recovery", replacement, {"blue"}, true);

        auto query_image = input("query-image");
        filler(query_image);
        add(query_image, ninfer::ChatRole::User, "What color fills this image?", {red});
        run("query-image-replay", query_image, {"red"}, false);
        run("query-image-repeat", query_image, {"red"}, true);
        auto multiple = input("multiple");
        filler(multiple);
        add(multiple, ninfer::ChatRole::User, "Name the colors of the first and second images, in order.", {red, blue});
        run("multiple-query-images", multiple, {"red", "blue"}, false);
        if (memory) {
            auto oversized = input("over-budget");
            add(oversized, ninfer::ChatRole::User, "Describe all three images.", {red, blue, red});
            bool rejected = false;
            try {
                ninfer::RequestOptions request;
                request.execution.requested_output_tokens = 16;
                (void)engine.generate(engine.prepare(std::move(oversized)), request);
            } catch (const ninfer::RequestError& error) {
                rejected = error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded;
            }
            require(rejected && engine.is_available(), "image budget failure was not rejected at admission");
            run("after-budget-error", short_image, {"red"}, false);
        }
        std::cout << "MEMORY_VISION PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
