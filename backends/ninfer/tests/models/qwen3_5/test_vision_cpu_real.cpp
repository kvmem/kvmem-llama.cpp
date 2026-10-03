#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// The CPU Vision residency against the resident device encoder on the same image and prompt: the
// first generated token must agree, its log probability within a small margin, and the most
// likely alternatives must largely coincide. The CPU encoder computes in FP32 where the device
// rounds activations to BF16, so equality is not expected.
namespace {

std::vector<std::uint8_t> test_image(int width, int height) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // A red disc on a blue-to-green gradient: something a model can name.
            const int dx = x - width / 2, dy = y - height / 2;
            const bool disc = dx * dx + dy * dy < (width / 4) * (width / 4);
            ppm.push_back(disc ? 230 : 20);
            ppm.push_back(disc ? 30 : static_cast<std::uint8_t>(255 * y / height));
            ppm.push_back(disc ? 30 : static_cast<std::uint8_t>(255 - 255 * y / height));
        }
    }
    return ppm;
}

struct Outcome {
    ninfer::GenerationResult result;
    double load_seconds = 0.0;
};

Outcome run_with(const char* artifact, ninfer::VisionResidency residency) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 2048;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk                    = 512;
    options.enable_vision                    = true;
    options.vision_residency                 = residency;
    options.vision_max_merged_tokens         = 256;
    options.max_concurrency                  = 1;
    options.max_pending_requests             = 1;
    options.context_cache.device_state_slots = 1;
    ninfer::Engine engine(std::move(options));

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = test_image(448, 448);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "disc.ppm";
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                .text  = "What shape and color is in the middle?",
                                                .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens  = 12;
    request.execution.sampling.temperature     = 0.0F;
    request.execution.allow_prefix_reuse       = false;
    request.execution.first_token_top_logprobs = 8;
    request.stop.include_model_defaults        = false;
    return Outcome{engine.generate(engine.prepare(std::move(input)), request),
                   engine.load_summary().load_seconds};
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    const Outcome device = run_with(artifact, ninfer::VisionResidency::Resident);
    const Outcome cpu    = run_with(artifact, ninfer::VisionResidency::Cpu);
    std::cout << "resident vision " << device.result.timings.vision_seconds
              << "s: " << device.result.content << '\n'
              << "cpu vision " << cpu.result.timings.vision_seconds << "s (load "
              << cpu.load_seconds << "s): " << cpu.result.content << '\n';
    if (!device.result.first_token_logprobs || !cpu.result.first_token_logprobs ||
        !device.result.prompt.has_media || !cpu.result.prompt.has_media) {
        std::cerr << "a run reported no media or no first-token log probabilities\n";
        return 1;
    }
    const auto& a      = *device.result.first_token_logprobs;
    const auto& b      = *cpu.result.first_token_logprobs;
    std::size_t shared = 0;
    for (const auto& candidate : a.top) {
        shared += static_cast<std::size_t>(
            std::any_of(b.top.begin(), b.top.end(),
                        [&](const auto& other) { return other.token == candidate.token; }));
    }
    std::cout << "first token " << a.selected.token << '/' << b.selected.token << " logprob "
              << a.selected.logprob << '/' << b.selected.logprob << ", top-8 shared " << shared
              << '\n';
    if (a.selected.token != b.selected.token ||
        std::abs(a.selected.logprob - b.selected.logprob) > 0.25F || shared < 6) {
        std::cerr << "the CPU encoder's embeddings steer the model away from the device's\n";
        return 1;
    }
    if (cpu.result.timings.vision_seconds <= 0.0) {
        std::cerr << "the CPU run reported no Vision time\n";
        return 1;
    }
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
