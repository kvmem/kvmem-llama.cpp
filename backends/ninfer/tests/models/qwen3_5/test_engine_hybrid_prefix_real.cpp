// Real-artifact scenarios for the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md
// §13.3). Requires NINFER_TEST_ARTIFACT; NINFER_HYBRID_REAL_SCENARIO selects one scenario.

#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kPrefillChunk = 512;
// Tokens per cached KV block.
constexpr std::uint32_t kBlock = 64;

// NINFER_HYBRID_KV_DTYPE selects the KV storage every scenario runs with (default bf16), so the
// Host tier's page records and restores are exercised for each profile.
ninfer::KvCacheStorage kv_storage = ninfer::KvCacheStorage::BFloat16;

bool select_kv_storage(std::string_view name) {
    if (name == "bf16") {
        kv_storage = ninfer::KvCacheStorage::BFloat16;
    } else if (name == "int8") {
        kv_storage = ninfer::KvCacheStorage::Int8Group64;
    } else if (name == "fp8") {
        kv_storage = ninfer::KvCacheStorage::Fp8E4M3Row256;
    } else if (name == "nvfp4") {
        kv_storage = ninfer::KvCacheStorage::Nvfp4Group16;
    } else if (name == "k8v4") {
        kv_storage = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
    } else {
        return false;
    }
    return true;
}

// Deterministic ordinary-vocabulary tokens; the content only has to be reproducible.
std::vector<ninfer::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t index = 0; index < count; ++index) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<ninfer::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

ninfer::EngineOptions hybrid_options(const char* artifact, ninfer::SpeculativeBackend backend,
                                     std::uint32_t kv_tokens, std::size_t host_bytes,
                                     std::uint32_t device_slots) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.prefill_chunk        = kPrefillChunk;
    options.kv_cache             = kv_storage;
    options.speculative.backend  = backend;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    if (backend == ninfer::SpeculativeBackend::Mtp) {
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    } else if (backend == ninfer::SpeculativeBackend::DFlash2) {
        options.speculative.draft_tokens = 7;
    }
    options.context_cache.mode                         = ninfer::ContextCacheMode::Hybrid;
    options.context_cache.host_cache_budget_bytes      = host_bytes;
    options.context_cache.hybrid.device_snapshot_slots = device_slots;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Observed {
    std::vector<ninfer::TokenId> tokens;
    std::uint32_t reused = 0;
    ninfer::RuntimeStats stats;
};

// A later request restores a flexible prompt-tail snapshot the first request captured at its last
// prefill chunk boundary. The Device-resident engine restores it with Device copies; the
// constrained engine is first forced to evict the snapshot image and most of its blocks, so the
// same snapshot comes back through the Host slabs. Both restores carry identical bytes and resume
// at the same frontier with the same chunking, so greedy generation must match token for token.
int exercise_restore_exact(const char* artifact, ninfer::SpeculativeBackend backend) {
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 1);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 2);
    second.insert(second.end(), suffix.begin(), suffix.end());
    // 3900 prompt tokens need 61 of the constrained engine's 64 pages, evicting all but the
    // first few of the first request's cached blocks.
    const std::vector<ninfer::TokenId> pressure = synthetic_tokens(3900, 3);

    const auto run = [&](ninfer::EngineOptions options, bool apply_pressure) {
        ninfer::Engine engine(std::move(options));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        if (apply_pressure) { (void)engine.generate(engine.prepare_tokens(pressure), greedy(4)); }
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(second), greedy(24));
        return Observed{result.generated_token_ids, result.reused_prompt_tokens,
                        engine.runtime_stats()};
    };
    const Observed device = run(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8), false);
    const Observed host   = run(hybrid_options(artifact, backend, 4096, 1ULL << 30, 1), true);

    // The first prompt's chunks end at 512 and 1024; the final chunk [1024, 1500) holds the
    // flexible prompt-tail tap, realized at its start.
    constexpr std::uint32_t kExpectedReuse = 2 * kPrefillChunk;
    int failures                           = 0;
    if (device.reused != kExpectedReuse || host.reused != kExpectedReuse) {
        std::cerr << "restore-exact: reused device=" << device.reused << " host=" << host.reused
                  << ", expected the prompt-tail snapshot at " << kExpectedReuse << '\n';
        ++failures;
    }
    if (device.stats.hybrid_host_image_restores != 0 ||
        device.stats.hybrid_host_block_restores != 0) {
        std::cerr << "restore-exact: the Device-resident engine restored from Host\n";
        ++failures;
    }
    if (host.stats.hybrid_host_image_restores == 0 || host.stats.hybrid_host_block_restores == 0) {
        std::cerr << "restore-exact: the constrained engine did not restore through Host slabs"
                  << " (images=" << host.stats.hybrid_host_image_restores
                  << " blocks=" << host.stats.hybrid_host_block_restores << ")\n";
        ++failures;
    }
    if (device.tokens.size() != 24 || device.tokens != host.tokens) {
        std::cerr << "restore-exact: Host and Device restores generated different tokens\n";
        ++failures;
    }
    return failures;
}

// A restarted Engine resumes from the Host tier its predecessor saved: the restored snapshot is
// the same bytes, so greedy generation matches an Engine that never restarted. A file written for
// a different build identity is ignored.
int exercise_persist(const char* artifact) {
    const std::filesystem::path file =
        std::filesystem::temp_directory_path() / "ninfer-hybrid-persist-real-test.bin";
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const std::vector<ninfer::TokenId> first = synthetic_tokens(1500, 4);
    std::vector<ninfer::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<ninfer::TokenId> suffix = synthetic_tokens(300, 5);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const auto options = [&](const char* identity) {
        ninfer::EngineOptions result =
            hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        if (identity != nullptr) {
            result.context_cache.hybrid.persistent_file     = file;
            result.context_cache.hybrid.persistent_identity = identity;
        }
        return result;
    };

    std::vector<ninfer::TokenId> reference;
    {
        ninfer::Engine engine(options(nullptr));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        reference = engine.generate(engine.prepare_tokens(second), greedy(24)).generated_token_ids;
    }
    {
        ninfer::Engine saver(options("persist-test"));
        (void)saver.generate(saver.prepare_tokens(first), greedy(8));
    }
    int failures = 0;
    if (!std::filesystem::exists(file)) {
        std::cerr << "persist: the Engine did not save its Host tier\n";
        return 1;
    }
    {
        ninfer::Engine loader(options("persist-test"));
        const ninfer::LoadSummary summary = loader.load_summary();
        const ninfer::GenerationResult resumed =
            loader.generate(loader.prepare_tokens(second), greedy(24));
        const ninfer::RuntimeStats stats = loader.runtime_stats();
        if (!summary.prefix_cache.restored || summary.prefix_cache.snapshots == 0) {
            std::cerr << "persist: the saved Host tier was not restored ("
                      << summary.prefix_cache.message << ")\n";
            ++failures;
        }
        if (resumed.reused_prompt_tokens != 2 * kPrefillChunk ||
            stats.hybrid_host_block_restores == 0 || stats.hybrid_host_image_restores == 0) {
            std::cerr << "persist: the restarted Engine reused " << resumed.reused_prompt_tokens
                      << " tokens (blocks restored " << stats.hybrid_host_block_restores << ")\n";
            ++failures;
        }
        if (resumed.generated_token_ids != reference) {
            std::cerr << "persist: the restarted Engine generated different tokens\n";
            ++failures;
        }
    }
    {
        ninfer::Engine foreign(options("another-build"));
        const ninfer::LoadSummary summary = foreign.load_summary();
        if (summary.prefix_cache.restored || summary.prefix_cache.message.empty() ||
            foreign.generate(foreign.prepare_tokens(second), greedy(4)).reused_prompt_tokens != 0) {
            std::cerr << "persist: a file from another build was not ignored\n";
            ++failures;
        }
    }
    std::filesystem::remove(file, ignored);
    return failures;
}

// Two requests submitted together share a prefix nothing has cached yet. The second waits for
// the first's snapshot where the prompts diverge and resumes from it instead of prefilling the
// prefix again (spec §12.2).
// - On a chunk boundary both runs compute the same chunks, and the leader stops after its first
//   token so the follower never shares a decode round: the follower's output must equal an
//   uncached run's.
// - Inside a block, the snapshot goes on the block boundary below, where it publishes at once.
int exercise_coalesce(const char* artifact, std::uint32_t shared_tokens, bool compare_output) {
    const std::vector<ninfer::TokenId> shared        = synthetic_tokens(shared_tokens, 6);
    std::vector<ninfer::TokenId> first               = shared;
    std::vector<ninfer::TokenId> second              = shared;
    const std::vector<ninfer::TokenId> first_suffix  = synthetic_tokens(40, 7);
    const std::vector<ninfer::TokenId> second_suffix = synthetic_tokens(40, 8);
    first.insert(first.end(), first_suffix.begin(), first_suffix.end());
    second.insert(second.end(), second_suffix.begin(), second_suffix.end());
    const std::uint32_t expected = shared_tokens / kBlock * kBlock;

    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency = 2;
    // The follower is admitted only while the leader still prefills.
    options.concurrent_prefill = true;
    std::vector<ninfer::TokenId> reference;
    if (compare_output) {
        ninfer::Engine engine(options);
        reference = engine.generate(engine.prepare_tokens(second), greedy(16)).generated_token_ids;
    }
    ninfer::Engine engine(options);
    ninfer::GenerationHandle leader    = engine.submit(engine.prepare_tokens(first), greedy(1));
    ninfer::GenerationHandle follower  = engine.submit(engine.prepare_tokens(second), greedy(16));
    const ninfer::GenerationResult led = leader.wait();
    const ninfer::GenerationResult followed = follower.wait();
    int failures                            = 0;
    if (led.reused_prompt_tokens != 0 || followed.reused_prompt_tokens != expected) {
        std::cerr << "coalesce " << shared_tokens << ": the leader reused "
                  << led.reused_prompt_tokens << " tokens and the follower "
                  << followed.reused_prompt_tokens << " (expected 0 and " << expected << ")\n";
        ++failures;
    }
    if (compare_output && followed.generated_token_ids != reference) {
        std::cerr << "coalesce " << shared_tokens
                  << ": the follower generated different tokens than an uncached run\n";
        ++failures;
    }
    return failures;
}

// Two requests submitted together with unrelated prompts: the second is admitted between the
// first one's prefill chunks, so their chunks interleave on shared execution scalars. Each must
// generate what it generates alone. The second stops after its first token so the first decodes
// alone, as in its reference.
int exercise_interleaved(const char* artifact) {
    const std::vector<ninfer::TokenId> first  = synthetic_tokens(3 * kPrefillChunk + 100, 9);
    const std::vector<ninfer::TokenId> second = synthetic_tokens(3 * kPrefillChunk + 60, 10);
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency    = 2;
    options.concurrent_prefill = true;
    std::vector<ninfer::TokenId> first_alone;
    std::vector<ninfer::TokenId> second_alone;
    {
        ninfer::Engine engine(options);
        first_alone = engine.generate(engine.prepare_tokens(first), greedy(16)).generated_token_ids;
        second_alone =
            engine.generate(engine.prepare_tokens(second), greedy(1)).generated_token_ids;
    }
    ninfer::Engine engine(options);
    ninfer::GenerationHandle a = engine.submit(engine.prepare_tokens(first), greedy(16));
    ninfer::GenerationHandle b = engine.submit(engine.prepare_tokens(second), greedy(1));
    const ninfer::GenerationResult first_together  = a.wait();
    const ninfer::GenerationResult second_together = b.wait();
    int failures                                   = 0;
    if (first_together.generated_token_ids != first_alone ||
        second_together.generated_token_ids != second_alone) {
        std::cerr
            << "interleaved: concurrent prefills generated different tokens than alone (first "
            << (first_together.generated_token_ids == first_alone ? "same" : "differs")
            << ", second "
            << (second_together.generated_token_ids == second_alone ? "same" : "differs") << ")\n";
        ++failures;
    }
    return failures;
}

ninfer::ChatMessage text_message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

// Protocol adapters shape the hints the way OpenAI default caching does: the Engine's automatic
// Legacy shared-prefix candidates are disabled and an automatic marker follows the last message.
// Hybrid taps must keep their semantic placement under those hints.
bool protocol_hints = false;

ninfer::PromptInput conversation(std::vector<ninfer::ChatMessage> messages) {
    ninfer::PromptInput input;
    input.messages                = std::move(messages);
    input.options.enable_thinking = false;
    if (protocol_hints) {
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
            .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence            = ninfer::SharedCandidateEvidence::DefaultAutomatic,
            .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
        });
    }
    return input;
}

// Exact semantic taps: the next turn of a conversation resumes at the previous turn's generation
// opener (not 64-token-floored), and a new conversation with the same leading system block resumes
// at that block's end.
int exercise_turns(const char* artifact, std::size_t host_bytes) {
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 8192, host_bytes, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    ninfer::Engine engine(std::move(options));

    std::string system = "You are a careful assistant for a small engineering team.";
    for (int sentence = 0; sentence < 60; ++sentence) {
        system += " Rule " + std::to_string(sentence) +
                  ": answer precisely, cite the relevant component, and keep replies short.";
    }
    const ninfer::ChatMessage leading = text_message(ninfer::ChatRole::System, system);
    const ninfer::ChatMessage first_user =
        text_message(ninfer::ChatRole::User, "Name three prime numbers.");

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation({leading, first_user})), greedy(24));
    const ninfer::GenerationResult second = engine.generate(
        engine.prepare(conversation({leading, first_user,
                                     text_message(ninfer::ChatRole::Assistant, first.content),
                                     text_message(ninfer::ChatRole::User, "Now name two more.")})),
        greedy(8));
    const ninfer::GenerationResult other = engine.generate(
        engine.prepare(
            conversation({leading, text_message(ninfer::ChatRole::User, "What is a hash table?")})),
        greedy(8));
    const std::uint32_t other_user_tokens =
        engine
            .prepare(conversation({text_message(ninfer::ChatRole::User, "What is a hash table?")}))
            .summary()
            .prompt_tokens;

    int failures = 0;
    // The next turn resumes at the first turn's generation opener, a few tokens before its end, or
    // at the earliest boundary of the opener's cluster (the system-block end after a short user
    // turn), at most the 64-token tap separation earlier.
    if (second.reused_prompt_tokens + 64U + 16U < first.prompt.prompt_tokens ||
        second.reused_prompt_tokens >= second.prompt.prompt_tokens) {
        std::cerr << "turns: second turn reused " << second.reused_prompt_tokens
                  << " of the first turn's " << first.prompt.prompt_tokens
                  << " prompt tokens; expected the generation opener\n";
        ++failures;
    }
    // The shared system block ends where the new user turn begins.
    if (other.reused_prompt_tokens + other_user_tokens + 16U < other.prompt.prompt_tokens) {
        std::cerr << "turns: a new conversation reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens; expected the system block end\n";
        ++failures;
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    if (stats.hybrid_taps_created == 0 || stats.hybrid_snapshot_hits < 2) {
        std::cerr << "turns: taps=" << stats.hybrid_taps_created
                  << " hits=" << stats.hybrid_snapshot_hits << '\n';
        ++failures;
    }
    return failures;
}

std::vector<std::uint8_t> gradient_ppm(int size, int phase) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(size) + ' ' + std::to_string(size) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < size * size; ++index) {
        ppm.push_back(static_cast<std::uint8_t>((index + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>(((index / size) * 3 + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7 + phase * 5) & 0xff));
    }
    return ppm;
}

// Vision identity: the same image and text resume at the generation opener, while identical text
// tokens with a different image cannot reuse anything from the image onwards (its blocks carry the
// image's content key) and never resume inside the image's span.
int exercise_vision(const char* artifact) {
    ninfer::EngineOptions options =
        hybrid_options(artifact, ninfer::SpeculativeBackend::None, 8192, 1ULL << 30, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    options.enable_vision = true;
    ninfer::Engine engine(std::move(options));

    std::string system = "You describe images for an accessibility service.";
    for (int sentence = 0; sentence < 30; ++sentence) {
        system +=
            " Rule " + std::to_string(sentence) + ": name colours, shapes and layout plainly.";
    }
    const auto prompt = [&](const std::vector<std::uint8_t>& image, const char* question) {
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        ninfer::MessagePart media;
        media.kind              = ninfer::MessagePartKind::Media;
        media.media.kind        = ninfer::MediaKind::Image;
        media.media.bytes       = image;
        media.media.media_type  = "image/x-portable-pixmap";
        media.media.source_name = "image.ppm";
        user.parts.push_back(std::move(media));
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = question, .media = {}});
        return conversation({text_message(ninfer::ChatRole::System, system), std::move(user)});
    };
    const std::vector<std::uint8_t> first_image  = gradient_ppm(512, 0);
    const std::vector<std::uint8_t> second_image = gradient_ppm(512, 97);

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const ninfer::GenerationResult same =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare(prompt(second_image, "Describe the image.")), greedy(4));

    int failures = 0;
    if (same.reused_prompt_tokens + 64U < first.prompt.prompt_tokens) {
        std::cerr << "vision: the same image and text reused " << same.reused_prompt_tokens
                  << " of " << first.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    // A 512x512 image is 256 merged tokens; a different image must lose at least those.
    if (other.prompt.prompt_tokens != first.prompt.prompt_tokens ||
        other.reused_prompt_tokens + 256U > first.prompt.prompt_tokens) {
        std::cerr << "vision: a different image reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    if (first.generated_token_ids.size() != 4 || other.generated_token_ids.size() != 4) {
        std::cerr << "vision: generation did not complete\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    if (const char* storage = std::getenv("NINFER_HYBRID_KV_DTYPE");
        storage != nullptr && *storage != '\0' && !select_kv_storage(storage)) {
        std::cerr << "unknown NINFER_HYBRID_KV_DTYPE " << storage << '\n';
        return 1;
    }
    const char* selected            = std::getenv("NINFER_HYBRID_REAL_SCENARIO");
    const std::string_view scenario = selected != nullptr && *selected != '\0' ? selected : "all";
    constexpr std::array<std::string_view, 11> kScenarios{"all",
                                                          "restore-exact",
                                                          "restore-exact-mtp",
                                                          "restore-exact-dflash2",
                                                          "turns",
                                                          "turns-device-only",
                                                          "vision",
                                                          "persist",
                                                          "turns-protocol",
                                                          "coalesce",
                                                          "interleaved"};
    if (std::find(kScenarios.begin(), kScenarios.end(), scenario) == kScenarios.end()) {
        std::cerr << "unknown NINFER_HYBRID_REAL_SCENARIO " << scenario << '\n';
        return 1;
    }
    int failures = 0;
    try {
        const bool all = scenario == "all";
        if (all || scenario == "restore-exact") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::None);
        }
        if (all || scenario == "restore-exact-mtp") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "restore-exact-dflash2") {
            failures += exercise_restore_exact(artifact, ninfer::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "turns") { failures += exercise_turns(artifact, 1ULL << 30); }
        if (all || scenario == "turns-device-only") { failures += exercise_turns(artifact, 0); }
        if (all || scenario == "vision") { failures += exercise_vision(artifact); }
        if (all || scenario == "persist") { failures += exercise_persist(artifact); }
        if (all || scenario == "interleaved") { failures += exercise_interleaved(artifact); }
        if (all || scenario == "coalesce") {
            failures += exercise_coalesce(artifact, 6 * kPrefillChunk, true);
            failures += exercise_coalesce(artifact, 3000, false);
        }
        if (all || scenario == "turns-protocol") {
            protocol_hints = true;
            failures += exercise_turns(artifact, 1ULL << 30);
            protocol_hints = false;
        }
    } catch (const std::exception& error) {
        std::cerr << "hybrid prefix real test failed: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "hybrid prefix real tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
