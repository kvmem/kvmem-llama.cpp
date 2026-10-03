#include "serve/generation_service.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;
    const auto memory_defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(memory_defaults.cuda_memory_policy == ninfer::CudaMemoryPolicy::DriverDefault &&
                          memory_defaults.cuda_vram_reserve_bytes == 0 &&
                          memory_defaults.cuda_memory_probe_step_bytes == 0 &&
                          memory_defaults.context_cache.mode == ninfer::ContextCacheMode::Legacy &&
                          memory_defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit,
                      "CUDA memory policy defaults changed");
    const auto resident = parse({"ninfer-serve", "model.ninfer",
                                 "--cuda-memory-policy", "strict", "--kv-capacity", "auto"});
    failures += check(resident.cuda_memory_policy == ninfer::CudaMemoryPolicy::StrictVram &&
                          resident.cuda_vram_reserve_bytes == (64ULL << 20) &&
                          resident.cuda_memory_probe_step_bytes == (128ULL << 20) &&
                          resident.context_cache.mode == ninfer::ContextCacheMode::Hybrid &&
                          resident.kv_capacity.automatic_headroom_bytes == (64ULL << 20),
                      "strict must enable hybrid caching and use its own automatic reserve");
    const auto mixed = parse({"ninfer-serve", "model.ninfer",
                              "--cuda-memory-policy", "mixed", "--kv-capacity", "auto"});
    failures += check(mixed.cuda_memory_policy == ninfer::CudaMemoryPolicy::Mixed &&
                          mixed.cuda_vram_reserve_bytes == 0 &&
                          mixed.cuda_memory_probe_step_bytes == 0 &&
                          mixed.kv_capacity.automatic_headroom_bytes == 0,
                      "mixed must not inherit strict probing or default-policy headroom");
    const auto ordinary = parse({"ninfer-serve", "model.ninfer",
                                 "--cuda-memory-policy", "default", "--kv-capacity", "auto",
                                 "--kv-headroom-mib", "100"});
    failures += check(ordinary.cuda_memory_policy == ninfer::CudaMemoryPolicy::DriverDefault &&
                          ordinary.kv_capacity.automatic_headroom_bytes == (100ULL << 20),
                      "default policy must retain explicit automatic-KV headroom");
    const auto custom = parse({"ninfer-serve", "model.ninfer",
                               "--cuda-memory-policy", "strict-0-1", "--kv-capacity", "auto"});
    failures += check(custom.cuda_vram_reserve_bytes == 0 &&
                          custom.cuda_memory_probe_step_bytes == (1ULL << 20) &&
                          custom.kv_capacity.automatic_headroom_bytes == 0,
                      "strict custom reserve and step minimum were not preserved");
    const auto custom_auto = parse({"ninfer-serve", "model.ninfer",
                                    "--cuda-memory-policy", "strict-00072-00256", "--kv-capacity", "auto"});
    failures += check(custom_auto.cuda_vram_reserve_bytes == (72ULL << 20) &&
                          custom_auto.cuda_memory_probe_step_bytes == (256ULL << 20) &&
                          custom_auto.kv_capacity.automatic_headroom_bytes == (72ULL << 20),
                      "custom strict decimal values must control auto-KV headroom");
    const auto maximum_mib = std::numeric_limits<std::size_t>::max() >> 20;
    const auto memory_boundaries = parse({"ninfer-serve", "model.ninfer",
                                          "--cuda-memory-policy", "strict-" + std::to_string(maximum_mib) + "-16384"});
    failures += check(memory_boundaries.cuda_memory_policy == ninfer::CudaMemoryPolicy::StrictVram &&
                          memory_boundaries.cuda_vram_reserve_bytes == (maximum_mib << 20) &&
                          memory_boundaries.cuda_memory_probe_step_bytes == (16384ULL << 20),
                      "valid CUDA memory parameter boundaries were rejected or truncated");
    const auto reset_policy = parse({"ninfer-serve", "model.ninfer",
                                     "--cuda-memory-policy", "strict-7-8", "--cuda-memory-policy", "default"});
    failures += check(reset_policy.cuda_memory_policy == ninfer::CudaMemoryPolicy::DriverDefault &&
                          reset_policy.cuda_vram_reserve_bytes == 0 &&
                          reset_policy.cuda_memory_probe_step_bytes == 0 &&
                          reset_policy.context_cache.mode == ninfer::ContextCacheMode::Legacy,
                      "last policy selection must not retain strict side effects");
    for (const auto& extra : std::vector<std::vector<std::string>>{
             {"--cuda-memory-policy", "unknown"},
             {"--cuda-memory-policy", "strict-64"}, {"--cuda-memory-policy", "strict--128"},
             {"--cuda-memory-policy", "strict-64-"}, {"--cuda-memory-policy", "strict-64-0"},
             {"--cuda-memory-policy", "strict-64-16385"}, {"--cuda-memory-policy", "strict-64-128-1"},
             {"--cuda-memory-policy", "strict-+64-128"}, {"--cuda-memory-policy", "strict- 64-128"},
             {"--cuda-memory-policy", "strict-1.5-128"}, {"--cuda-memory-policy", "strict-64--1"},
             {"--cuda-memory-policy", "strict-18446744073709551615-128"},
             {"--cuda-memory-policy", "strict-" + std::to_string(maximum_mib + 1) + "-128"},
             {"--cuda-memory-policy", "strict-64-18446744073709551615"},
             {"--cuda-memory-policy", "strict", "--vision"},
             {"--cuda-memory-policy", "mixed", "--vision"},
             {"--cuda-memory-policy", "strict", "--wddm-evictable-budget"},
             {"--cuda-memory-policy", "mixed", "--wddm-evictable-budget"},
             {"--cuda-memory-policy", "strict", "--devices", "0,1"},
             {"--cuda-memory-policy", "mixed", "--devices", "0,1"},
             {"--cuda-memory-policy", "strict", "--use-original-prefix-caching"},
             {"--no-prefix-reuse", "--cuda-memory-policy", "strict"},
             {"--cuda-memory-policy", "strict", "--kv-capacity", "auto", "--kv-headroom-mib", "64"},
             {"--cuda-memory-policy", "mixed", "--kv-capacity", "auto", "--vram-headroom-mib", "0"}}) {
        std::vector<std::string> arguments{"ninfer-serve", "model.ninfer"};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        bool rejected = false;
        try {
            (void)parse(arguments);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid or conflicting CUDA residency options admitted");
    }
    const auto memory_help = serve_usage_text("ninfer-serve");
    failures += check(memory_help.find("strict-RESERVE-STEP") != std::string::npos,
                      "Serve help omits custom strict policy syntax");
    const auto resident_engine = make_engine_options(resident);
    failures += check(resident_engine.cuda_memory_policy == ninfer::CudaMemoryPolicy::StrictVram &&
                          resident_engine.cuda_vram_reserve_bytes == (64ULL << 20) &&
                          resident_engine.cuda_memory_probe_step_bytes == (128ULL << 20) &&
                          resident_engine.context_cache.mode == ninfer::ContextCacheMode::Hybrid &&
                          resident_engine.kv_capacity.automatic_headroom_bytes == (64ULL << 20),
                      "serve-to-engine conversion lost strict policy semantics");
    const auto mixed_engine = make_engine_options(mixed);
    failures += check(mixed_engine.cuda_memory_policy == ninfer::CudaMemoryPolicy::Mixed &&
                          mixed_engine.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          mixed_engine.kv_capacity.automatic_headroom_bytes == 0,
                      "serve-to-engine conversion lost mixed policy semantics");
    const auto implicit_strict = parse({"ninfer-serve", "model.ninfer", "--cuda-memory-policy", "strict",
                                        "--device-snapshot-slots", "2"});
    failures += check(implicit_strict.context_cache.mode == ninfer::ContextCacheMode::Hybrid &&
                          implicit_strict.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          implicit_strict.kv_capacity.automatic_headroom_bytes == (64ULL << 20),
                      "strict hybrid activation must precede KV default and hybrid-option validation");
    const auto archive = parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens",
                                "5", "--ngram-draft-tokens", "63", "--ngram-archive-mib", "512",
                                "--ngram-session-mib", "128", "--ngram-native-sessions"});
    failures += check(archive.speculative.ngram_archive_bytes == (512ULL << 20) &&
                          archive.speculative.ngram_session_bytes == (128ULL << 20) &&
                          archive.ngram_native_sessions,
                      "archive capacities or native opt-in not preserved");
    for (const auto& extra : std::vector<std::vector<std::string>>{
             {"--ngram-native-sessions"},
             {"--ngram-archive-mib", "64"},
             {"--ngram-archive-mib", "512", "--ngram-session-mib", "0"},
             {"--ngram-archive-mib", "18446744073709551615"}}) {
        std::vector<std::string> args{"ninfer-serve",
                                      "model.ninfer",
                                      "--spec",
                                      "mtp",
                                      "--draft-tokens",
                                      "5",
                                      "--ngram-draft-tokens",
                                      "63"};
        args.insert(args.end(), extra.begin(), extra.end());
        bool rejected = false;
        try {
            (void)parse(args);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid archive contract admitted");
    }
    const auto ngram = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2", "--draft-tokens",
                              "5", "--ngram-draft-tokens", "15", "--ngram-min-match", "12"});
    failures +=
        check(ngram.speculative.ngram_draft_tokens == 15 && ngram.speculative.draft_tokens == 5 &&
                  ngram.speculative.ngram_min_match == 12,
              "ngram and neural widths were not kept separate");
    for (const auto& tail : std::vector<std::vector<std::string>>{
             {"--ngram-draft-tokens", "64"},
             {"--ngram-min-match", "3"},
             {"--max-concurrency", "2", "--ngram-draft-tokens", "63"},
             {"--spec", "none"}}) {
        std::vector<std::string> arguments{"ninfer-serve",
                                           "model.ninfer",
                                           "--spec",
                                           "dflash2",
                                           "--draft-tokens",
                                           "5",
                                           "--ngram-draft-tokens",
                                           "15"};
        arguments.insert(arguments.end(), tail.begin(), tail.end());
        bool rejected = false;
        try {
            (void)parse(arguments);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "unsupported ngram configuration admitted");
    }
    // A width up to 15 fits every batch; wider widths are admitted for one active request only.
    for (const std::string backend : {"mtp", "dflash", "dflash2"}) {
        const auto concurrent =
            parse({"ninfer-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3",
                   "--ngram-draft-tokens", "15", "--max-concurrency", "4"});
        failures += check(concurrent.speculative.ngram_draft_tokens == 15 &&
                              concurrent.max_concurrency == 4,
                          "ngram with concurrency>1 was not admitted");
        const auto defaulted = parse({"ninfer-serve", "model.ninfer", "--spec", backend,
                                      "--draft-tokens", "3", "--max-concurrency", "8"});
        failures += check(defaulted.speculative.ngram_draft_tokens == 15,
                          "ngram drafting is not on by default beside a drafter");
        const auto disabled = parse({"ninfer-serve", "model.ninfer", "--spec", backend,
                                     "--draft-tokens", "3", "--ngram-draft-tokens", "0"});
        failures += check(disabled.speculative.ngram_draft_tokens == 0,
                          "explicit ngram-off beside a drafter was overridden");
    }
    failures += check(parse({"ninfer-serve", "model.ninfer"}).speculative.ngram_draft_tokens == 0,
                      "ngram default needs a neural backend");
    const auto mtp_wide_concurrent = [&] {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3",
                         "--ngram-draft-tokens", "63", "--max-concurrency", "4"});
        } catch (const std::invalid_argument&) { rejected = true; }
        return rejected;
    }();
    failures += check(mtp_wide_concurrent, "wide MTP ngram with concurrency>1 was admitted");

    const auto mtp_ngram = parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens",
                                  "3", "--ngram-draft-tokens", "15"});
    failures += check(mtp_ngram.speculative.draft_tokens == 3 &&
                          mtp_ngram.speculative.ngram_draft_tokens == 15,
                      "MTP and ngram widths were not kept separate");
    for (const std::string backend : {"mtp", "dflash", "dflash2"}) {
        const int neural_limit = 15;
        for (int neural = 1; neural <= neural_limit; ++neural) {
            for (int lookup = 1; lookup <= 63; ++lookup) {
                const auto options =
                    parse({"ninfer-serve", "model.ninfer", "--spec", backend, "--draft-tokens",
                           std::to_string(neural), "--ngram-draft-tokens", std::to_string(lookup)});
                failures += check(options.speculative.draft_tokens == neural &&
                                      options.speculative.ngram_draft_tokens == lookup,
                                  "valid neural/ngram width pair changed");
            }
        }
        for (const auto& tail : std::vector<std::vector<std::string>>{
                 {"--draft-tokens", std::to_string(neural_limit + 1)},
                 {"--ngram-draft-tokens", "-1"},
                 {"--ngram-draft-tokens", "64"},
                 {"--ngram-min-match", "65"}}) {
            std::vector<std::string> arguments{"ninfer-serve",
                                               "model.ninfer",
                                               "--spec",
                                               backend,
                                               "--draft-tokens",
                                               "3",
                                               "--ngram-draft-tokens",
                                               "15"};
            arguments.insert(arguments.end(), tail.begin(), tail.end());
            bool rejected = false;
            try {
                (void)parse(arguments);
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "unsupported neural/ngram pair admitted");
        }
        // The GDN conv-record workspace caps a multi-request verify at 16 columns, so any backend
        // rejects an ngram width above 15 once concurrency exceeds one.
        bool wide_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3",
                         "--ngram-draft-tokens", "63", "--max-concurrency", "2"});
        } catch (const std::invalid_argument&) { wide_rejected = true; }
        failures += check(wide_rejected, "wide ngram with concurrency>1 admitted");
        const auto wide_single =
            parse({"ninfer-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3",
                   "--ngram-draft-tokens", "63", "--max-concurrency", "1"});
        failures += check(wide_single.speculative.ngram_draft_tokens == 63,
                          "wide ngram rejected at concurrency one");
        const auto disabled =
            parse({"ninfer-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3",
                   "--ngram-draft-tokens", "0", "--max-concurrency", "8"});
        failures += check(disabled.speculative.ngram_draft_tokens == 0,
                          "disabled ngram changed multi-slot configuration");
    }
    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.speculative.ngram_draft_tokens == 0,
                      "ngram is enabled without a neural backend");
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.context_cost_presets.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.media_cache_bytes == ninfer::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == ninfer::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.context_cache.host_state_slots == ninfer::kDefaultHostStateSlots &&
                          defaults.context_cache.host_kv_capacity_bytes ==
                              ninfer::kDefaultHostKvCapacityBytes,
                      "Host context-cache defaults mismatch");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions fp8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "fp8"});
    failures += check(fp8.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string kv_help = serve_usage_text("ninfer-serve");
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos,
                      "serve help omits a production KV storage mode");

    // Every one of these parses without error whether or not the service carries it to the Engine,
    // so the mapping from parsed options to Engine options is what has to be checked.
    const ninfer::EngineOptions default_engine = make_engine_options(defaults);
    failures += check(!default_engine.prefill_cublas && default_engine.prefill_cublas_projections &&
                          default_engine.speculative.lookup_ngram == 0,
                      "the cuBLAS prefill route or context lookup is on by default in serving");
    const ServeOptions route =
        parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3",
               "--lookup-ngram", "5", "--prefill-cublas", "--no-prefill-cublas-projections"});
    const ninfer::EngineOptions route_engine = make_engine_options(route);
    failures += check(route_engine.speculative.lookup_ngram == 5 &&
                          route_engine.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          route_engine.speculative.draft_tokens == 3,
                      "--lookup-ngram did not reach the Engine options next to --spec");
    failures += check(route_engine.prefill_cublas && !route_engine.prefill_cublas_projections,
                      "the cuBLAS prefill controls did not reach the Engine options");
    for (const char* flag : {"--prefill-cublas", "--no-prefill-cublas-projections", "--lookup-ngram"}) {
        failures += check(kv_help.find(flag) != std::string::npos,
                          "serve help omits an accepted prefill or drafting control");
    }

    // rk8v4 still parses to its storage value; the engine rejects it in
    // target_kv_cache_profile so the failure names the unported feature rather than an
    // unknown --kv-dtype token.
    const ServeOptions rotor = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk8v4"});
    failures += check(rotor.kv_cache == ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                      "--kv-dtype rk8v4 did not select rotated K8/V4 storage");
    failures += check(defaults.kv_cache == ninfer::KvCacheStorage::BFloat16,
                      "rk8v4 unexpectedly changed the default KV storage");
    const ServeOptions lloyd = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk4v4"});
    failures += check(lloyd.kv_cache == ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value,
                      "--kv-dtype rk4v4 did not select rotated Lloyd-Max K4/V4 storage");
    const ServeOptions e8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk4v4-e8"});
    failures += check(e8.kv_cache == ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8,
                      "--kv-dtype rk4v4-e8 did not select rotated K4/V4 E8 storage");
    failures += check(kv_help.find("rk4v4-e8") != std::string::npos,
                      "serve help omits --kv-dtype rk4v4-e8");
    const ServeOptions root = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "rk2v4-e8"});
    failures += check(root.kv_cache == ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value,
                      "--kv-dtype rk2v4-e8 did not select rotated E8-root key storage");
    failures += check(kv_help.find("rk2v4-e8") != std::string::npos,
                      "serve help omits --kv-dtype rk2v4-e8");
    const ServeOptions yarn = parse({"ninfer-serve", "model.ninfer", "--rope-yarn"});
    failures += check(yarn.rope_yarn && !defaults.rope_yarn, "--rope-yarn did not select YaRN");
    failures += check(defaults.rope_yarn_factor == 1.0F, "serving YaRN factor must default to 1");
    for (const auto* factor : {"1", "2.5", "4"}) {
        const ServeOptions fixed =
            parse({"ninfer-serve", "model.ninfer", "--rope-yarn-factor", factor});
        failures += check(fixed.rope_yarn_factor == std::stof(factor) &&
                              fixed.max_context == 8192 &&
                              fixed.kv_capacity.explicit_tokens == 8192 &&
                              make_engine_options(fixed).rope_yarn_factor == std::stof(factor),
                          "a YaRN factor must reach the Engine without growing the context");
    }
    for (const char* factor : std::initializer_list<const char*>{
             "0", "0.99", "4.01", "-1", "nan", "inf", "-inf", "1e999", "2x", "", nullptr}) {
        bool rejected = false;
        try {
            if (factor == nullptr) {
                (void)parse({"ninfer-serve", "model.ninfer", "--rope-yarn-factor"});
            } else {
                (void)parse({"ninfer-serve", "model.ninfer", "--rope-yarn-factor", factor});
            }
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an invalid or missing YaRN factor was accepted");
    }
    const ServeOptions wddm = parse({"ninfer-serve", "model.ninfer", "--wddm-evictable-budget"});
    failures += check(wddm.wddm_evictable_budget && !defaults.wddm_evictable_budget,
                      "--wddm-evictable-budget did not select the WDDM budget");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"ninfer-serve", "model.ninfer", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.context_cost_presets == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");
    failures += check(defaults.thinking_budget_message.empty(),
                      "thinking budget message is unexpectedly configured by default");
    const ServeOptions thinking_message =
        parse({"ninfer-serve", "model.ninfer", "--thinking-budget-message",
               "Time to stop thinking. I must act now:"});
    failures += check(thinking_message.thinking_budget_message ==
                          "Time to stop thinking. I must act now:",
                      "--thinking-budget-message did not preserve its value");
    bool empty_thinking_message_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--thinking-budget-message", ""});
    } catch (const std::invalid_argument&) { empty_thinking_message_rejected = true; }
    failures += check(empty_thinking_message_rejected, "empty --thinking-budget-message accepted");

    failures += check(parse({"ninfer-serve", "model.ninfer", "--default-max-tokens", "0"})
                              .default_max_tokens == kUnboundedOutputTokens,
                      "--default-max-tokens 0 does not leave the output bounded only by context");

    const ServeOptions reasoning_effort =
        parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", "low"});
    failures += check(reasoning_effort.default_reasoning_effort == RequestedReasoningEffort::Low,
                      "--default-reasoning-effort did not preserve its value");
    failures += check(!parse({"ninfer-serve", "model.ninfer"}).default_reasoning_effort,
                      "the server default reasoning effort is not unset by default");
    for (const std::vector<std::string>& rejected :
         {std::vector<std::string>{"ninfer-serve", "model.ninfer", "--default-reasoning-effort",
                                   "extreme"},
          std::vector<std::string>{"ninfer-serve", "model.ninfer", "--no-thinking",
                                   "--default-reasoning-effort", "low"}}) {
        bool thrown = false;
        try {
            (void)parse(rejected);
        } catch (const std::invalid_argument&) { thrown = true; }
        failures += check(thrown, "an invalid --default-reasoning-effort was accepted");
    }
    failures += check(parse({"ninfer-serve", "model.ninfer", "--no-thinking",
                             "--default-reasoning-effort", "none"})
                              .default_reasoning_effort == RequestedReasoningEffort::None,
                      "--default-reasoning-effort none was refused beside --no-thinking");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += check(options.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              options.speculative.draft_tokens == k &&
                              options.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "serve options did not preserve DFlash2 configuration");
    }

    for (const auto k : {6U, 15U}) {
        const auto options =
            parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", std::to_string(k)});
        failures += check(options.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                              options.speculative.draft_tokens == k,
                          "serve options did not accept an MTP window past five");
    }
    bool mtp_sixteen_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "16"});
    } catch (const std::invalid_argument&) { mtp_sixteen_rejected = true; }
    failures += check(mtp_sixteen_rejected, "an MTP window of 16 was accepted");

    const ServeOptions dflash_vision = parse(
        {"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 15,
                      "serve options did not preserve combined DFlash and Vision features");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse({"ninfer-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--usage-chunk-choice",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--log-stats-panel",
                                           "off",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.context_cache.host_state_slots == 0 &&
                          configured.context_cache.host_kv_capacity_bytes == 0,
                      "root-only server mode retained default Host capacities");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures += check(!defaults.webui_mcp_proxy,
                      "the WebUI MCP relay reaches arbitrary hosts and must be off by default");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--webui-mcp-proxy"}).webui_mcp_proxy,
                      "--webui-mcp-proxy did not enable the relay");
    failures += check(serve_usage_text("ninfer-serve").find("--webui-mcp-proxy") !=
                          std::string::npos,
                      "serve help omits --webui-mcp-proxy");
    failures += check(configured.usage_chunk_choice,
                      "--usage-chunk-choice did not reach serving options");
    failures += check(configured.preserve_thinking == true,
                      "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(!configured.log_stats_panel &&
                          !parse({"ninfer-serve", "model.ninfer"}).log_stats_panel &&
                          parse({"ninfer-serve", "model.ninfer", "--log-stats-panel", "on"})
                              .log_stats_panel,
                      "--log-stats-panel did not reach serving options or is not off by default");
    bool invalid_panel_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--log-stats-panel", "yes"});
    } catch (const std::invalid_argument&) { invalid_panel_rejected = true; }
    failures +=
        check(invalid_panel_rejected, "--log-stats-panel accepted a value other than on|off");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging = parse({"ninfer-serve", "model.ninfer", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "log level did not reach serving options");

    const ServeOptions context_cache =
        parse({"ninfer-serve", "model.ninfer", "--device-state-slots", "3", "--host-state-slots",
               "5", "--host-kv-mib", "64", "--max-private-continuations", "9",
               "--max-shared-prefixes", "4", "--max-long-anchors-per-continuation", "2"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_state_slots == 5 &&
                          context_cache.context_cache.host_kv_capacity_bytes == (64ULL << 20) &&
                          context_cache.context_cache.max_private_continuations == 9 &&
                          context_cache.context_cache.max_shared_prefixes == 4 &&
                          context_cache.context_cache.max_long_anchors_per_continuation == 2,
                      "context-cache capacities did not reach serving options");
    const ServeOptions host_cache_budget =
        parse({"ninfer-serve", "model.ninfer", "--host-cache-mib", "24576",
               "--max-private-continuations", "32", "--max-long-anchors-per-continuation", "8"});
    failures += check(
        host_cache_budget.context_cache.enabled &&
            host_cache_budget.context_cache.host_cache_budget_bytes == (24576ULL << 20) &&
            // The budget mode owns the two RAM components; the parse layer must leave them at
            // their defaults rather than half-apply a rejected alternative.
            host_cache_budget.context_cache.host_state_slots == ninfer::kDefaultHostStateSlots &&
            host_cache_budget.context_cache.host_kv_capacity_bytes ==
                ninfer::kDefaultHostKvCapacityBytes,
        "host cache budget did not reach serving options");
    bool budget_with_state_slots_rejected = false;
    try {
        (void)parse(
            {"ninfer-serve", "model.ninfer", "--host-cache-mib", "64", "--host-state-slots", "5"});
    } catch (const std::invalid_argument&) { budget_with_state_slots_rejected = true; }
    failures +=
        check(budget_with_state_slots_rejected, "host cache budget accepted --host-state-slots");
    bool budget_with_host_kv_rejected = false;
    try {
        (void)parse(
            {"ninfer-serve", "model.ninfer", "--host-cache-mib", "64", "--host-kv-mib", "64"});
    } catch (const std::invalid_argument&) { budget_with_host_kv_rejected = true; }
    failures += check(budget_with_host_kv_rejected, "host cache budget accepted --host-kv-mib");
    bool disabled_budget_rejected = false;
    try {
        (void)parse(
            {"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--host-cache-mib", "64"});
    } catch (const std::invalid_argument&) { disabled_budget_rejected = true; }
    failures +=
        check(disabled_budget_rejected, "root-only server mode accepted a host cache budget");
    bool disabled_cache_capacity_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--host-kv-mib", "64"});
    } catch (const std::invalid_argument&) { disabled_cache_capacity_rejected = true; }
    failures += check(disabled_cache_capacity_rejected,
                      "root-only server mode accepted context-cache capacity options");

    const ServeOptions disk = parse({"ninfer-serve", "model.ninfer", "--disk-kv-path", "/tmp/l3",
                                     "--disk-kv-gib", "3", "--disk-kv-restore"});
    failures += check(disk.context_cache.disk_kv_path == "/tmp/l3" &&
                          disk.context_cache.disk_kv_capacity_bytes == (3ULL << 30U) &&
                          disk.context_cache.disk_kv_restore,
                      "disk tier options did not reach serving options");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--disk-kv-path", "/tmp/l3",
                             "--disk-kv-directstorage"})
                          .context_cache.disk_kv_directstorage,
                      "--disk-kv-directstorage did not reach serving options");
    failures += check(defaults.context_cache.disk_kv_path.empty() &&
                          !defaults.context_cache.disk_kv_restore &&
                          !defaults.context_cache.disk_kv_directstorage,
                      "the disk tier must be off by default");
    failures += check(!defaults.context_cache.rolling_retention &&
                          parse({"ninfer-serve", "model.ninfer", "--context-cache-policy",
                                 "rolling"})
                              .context_cache.rolling_retention &&
                          !parse({"ninfer-serve", "model.ninfer", "--context-cache-policy",
                                  "default"})
                               .context_cache.rolling_retention,
                      "the rolling retention policy did not reach serving options");
    failures += check(!defaults.context_cache.release_diverged_checkpoints &&
                          parse({"ninfer-serve", "model.ninfer", "--release-diverged-checkpoints"})
                              .context_cache.release_diverged_checkpoints,
                      "--release-diverged-checkpoints did not reach serving options");
    for (const auto& invalid : std::vector<std::vector<std::string>>{
             {"--context-cache-policy", "lru"},
             {"--context-cache-policy", "rolling", "--no-prefix-reuse"},
             {"--release-diverged-checkpoints", "--no-prefix-reuse"},
             {"--chat-template", ""},
             {"--disk-kv-restore"},
             {"--disk-kv-directstorage"},
             {"--disk-kv-gib", "8"},
             {"--disk-kv-path", "/tmp/l3", "--disk-kv-gib", "0"},
             {"--disk-kv-path", "/tmp/l3", "--no-prefix-reuse"}}) {
        std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
        args.insert(args.end(), invalid.begin(), invalid.end());
        bool rejected = false;
        try {
            (void)parse(args);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an inconsistent disk tier configuration was accepted");
    }

    failures += check(!parse({"ninfer-serve", "model.ninfer"}).auto_prefix_grid,
                      "automatic prefix grid was on without --auto-prefix-grid");
    failures +=
        check(parse({"ninfer-serve", "model.ninfer", "--auto-prefix-grid"}).auto_prefix_grid,
              "--auto-prefix-grid did not reach serving options");
    bool grid_without_reuse_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--auto-prefix-grid"});
    } catch (const std::invalid_argument&) { grid_without_reuse_rejected = true; }
    failures += check(grid_without_reuse_rejected,
                      "--auto-prefix-grid was accepted with prefix reuse disabled");
    // Hybrid prefix cache selection and its own capacities.
    failures += check(parse({"ninfer-serve", "model.ninfer"}).context_cache.mode ==
                          ninfer::ContextCacheMode::Legacy,
                      "the default prefix cache must remain the Legacy mode");
    const ServeOptions hybrid =
        parse({"ninfer-serve", "model.ninfer", "--use-alt-prefix-caching", "--host-cache-mib",
               "4096", "--device-snapshot-slots", "3", "--cache-taps-per-request", "5",
               "--cache-tap-ladder", "8192", "--cache-tap-min-gap", "512"});
    failures += check(hybrid.context_cache.enabled &&
                          hybrid.context_cache.mode == ninfer::ContextCacheMode::Hybrid &&
                          hybrid.context_cache.host_cache_budget_bytes == (4096ULL << 20) &&
                          hybrid.context_cache.hybrid.device_snapshot_slots == 3 &&
                          hybrid.context_cache.hybrid.max_new_taps == 5 &&
                          hybrid.context_cache.hybrid.tap_ladder_tokens == 8192 &&
                          hybrid.context_cache.hybrid.tap_min_gap_tokens == 512,
                      "hybrid prefix-cache options did not reach serving options");
    // The mode alone is a complete configuration: the KV pool defaults to free VRAM (the Device
    // block cache) and every hybrid tuning value is left for the Engine to derive.
    const ServeOptions hybrid_minimal =
        parse({"ninfer-serve", "model.ninfer", "--use-alt-prefix-caching"});
    failures += check(hybrid_minimal.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          hybrid_minimal.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes &&
                          !hybrid_minimal.context_cache.host_cache_budget_bytes &&
                          !hybrid_minimal.context_cache.hybrid.device_snapshot_slots &&
                          !hybrid_minimal.context_cache.hybrid.max_new_taps &&
                          !hybrid_minimal.context_cache.hybrid.tap_ladder_tokens &&
                          !hybrid_minimal.context_cache.hybrid.tap_min_gap_tokens,
                      "hybrid mode alone must size KV automatically and leave tuning derived");
    const ServeOptions hybrid_explicit_kv =
        parse({"ninfer-serve", "model.ninfer", "--use-alt-prefix-caching", "--max-context", "8192",
               "--kv-capacity", "16384", "--host-cache-mib", "0"});
    failures += check(hybrid_explicit_kv.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          hybrid_explicit_kv.kv_capacity.explicit_tokens == 16384 &&
                          hybrid_explicit_kv.context_cache.host_cache_budget_bytes == 0U,
                      "hybrid mode must keep an explicit KV capacity and a zero Host budget");
    failures += check(parse({"ninfer-serve", "model.ninfer"}).kv_capacity.mode ==
                          ninfer::KvCapacityMode::Explicit,
                      "the default cache must keep the explicit max-context KV capacity");
    for (const std::vector<std::string>& legacy_flag :
         std::vector<std::vector<std::string>>{{"--device-state-slots", "2"},
                                               {"--host-state-slots", "2"},
                                               {"--host-kv-mib", "64"},
                                               {"--max-private-continuations", "4"},
                                               {"--max-shared-prefixes", "4"},
                                               {"--max-long-anchors-per-continuation", "2"},
                                               {"--long-anchor-spacing", "0"},
                                               {"--auto-long-anchors"},
                                               {"--auto-prefix-grid"},
                                               {"--derive-session-keys"},
                                               {"--context-cache-policy", "rolling"},
                                               {"--release-diverged-checkpoints"},
                                               {"--thorough-admission-search"},
                                               {"--recency-eviction"},
                                               {"--value-aware-demote"},
                                               {"--disk-kv-path", "/tmp/l3"}}) {
        std::vector<std::string> arguments{"ninfer-serve", "model.ninfer",
                                           "--use-alt-prefix-caching"};
        arguments.insert(arguments.end(), legacy_flag.begin(), legacy_flag.end());
        bool rejected = false;
        try {
            (void)parse(std::move(arguments));
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "hybrid prefix cache accepted a Legacy capacity flag");
    }
    // The cache file resolves to an absolute path at launch, so the shutdown save writes where
    // startup read. A path with backslashes and a drive, as a Windows shell passes a quoted
    // argument, names the same file.
    const auto cache_file = [&](std::vector<std::string> extra) {
        std::vector<std::string> arguments{"ninfer-serve", "model.ninfer",
                                           "--use-alt-prefix-caching"};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return parse(std::move(arguments)).context_cache.hybrid.persistent_file;
    };
    failures +=
        check(cache_file({"--prefix-cache-file", "file.cache"}) ==
                  (std::filesystem::current_path() / "file.cache").lexically_normal(),
              "a relative --prefix-cache-file did not resolve against the launch directory");
    const std::filesystem::path absolute =
        (std::filesystem::temp_directory_path() / "ninfer-serve-options.cache").lexically_normal();
    failures += check(cache_file({"--prefix-cache-file", absolute.string()}) == absolute,
                      "an absolute --prefix-cache-file was not kept as given");
    const auto rejected_cache_file = [&](std::vector<std::string> extra) {
        try {
            (void)cache_file(std::move(extra));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(rejected_cache_file({"--prefix-cache-file",
                                           (std::filesystem::temp_directory_path() /
                                            "ninfer-missing-directory-for-test" / "file.cache")
                                               .string()}),
                      "--prefix-cache-file accepted a file in a directory that does not exist");
    failures += check(rejected_cache_file(
                          {"--prefix-cache-file", std::filesystem::temp_directory_path().string()}),
                      "--prefix-cache-file accepted a directory");
    failures +=
        check(rejected_cache_file({"--prefix-cache-file", "file.cache", "--host-cache-mib", "0"}),
              "--prefix-cache-file was accepted without a Host tier to save");
    for (const std::vector<std::string>& hybrid_flag :
         std::vector<std::vector<std::string>>{{"--prefix-cache-file", "file.cache"},
                                               {"--device-snapshot-slots", "2"},
                                               {"--cache-taps-per-request", "2"},
                                               {"--cache-tap-ladder", "2048"},
                                               {"--cache-tap-min-gap", "64"}}) {
        std::vector<std::string> arguments{"ninfer-serve", "model.ninfer"};
        arguments.insert(arguments.end(), hybrid_flag.begin(), hybrid_flag.end());
        bool rejected = false;
        try {
            (void)parse(std::move(arguments));
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "a hybrid prefix-cache flag was accepted without the mode");
    }
    bool hybrid_without_reuse_rejected = false;
    try {
        (void)parse(
            {"ninfer-serve", "model.ninfer", "--use-alt-prefix-caching", "--no-prefix-reuse"});
    } catch (const std::invalid_argument&) { hybrid_without_reuse_rejected = true; }
    failures += check(hybrid_without_reuse_rejected,
                      "hybrid prefix cache was accepted together with --no-prefix-reuse");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--use-original-prefix-caching",
                             "--host-kv-mib", "64"})
                              .context_cache.mode == ninfer::ContextCacheMode::Legacy,
                      "--use-original-prefix-caching did not keep the checkpoint catalog");
    bool both_caches_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--use-original-prefix-caching",
                     "--use-alt-prefix-caching"});
    } catch (const std::invalid_argument&) { both_caches_rejected = true; }
    failures += check(both_caches_rejected, "both prefix caching systems were accepted together");
    bool hybrid_pipeline_rejected = false;
    try {
        (void)parse(
            {"ninfer-serve", "model.ninfer", "--use-alt-prefix-caching", "--devices", "0,1"});
    } catch (const std::invalid_argument&) { hybrid_pipeline_rejected = true; }
    failures +=
        check(hybrid_pipeline_rejected, "hybrid prefix cache was accepted across pipeline stages");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "20", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 20 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--top-k", "21"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens   = 1;
    const auto semantics = resolve_prompt_semantics(request, defaults);
    failures += check(!semantics.reasoning_effort && semantics.enable_thinking == true,
                      "omitted thinking did not resolve to the enabled template default");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const ninfer::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking = resolve_prompt_semantics(request, thinking_budget);
    failures += check(!non_thinking.reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = RequestedReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults);
    failures += check(explicit_effort.reasoning_effort == ninfer::ReasoningEffort::Low &&
                          explicit_effort.enable_thinking == true,
                      "explicit reasoning effort did not remain the effective effort");
    request.reasoning_effort.reset();
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == true,
                      "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == false,
                      "request preserve-thinking override did not win");

    // Client effort vocabularies are wider than the three rungs the maintained Qwen templates
    // accept: OpenAI and Claude Code send 'high', pi sends 'minimal' and 'max'. Those collapse
    // onto the nearest rung rather than making the template raise.
    const auto collapses = [&](RequestedReasoningEffort wire) {
        GenerationRequest aliased = GenerationRequest{};
        aliased.max_tokens        = 1;
        aliased.reasoning_effort  = wire;
        return resolve_prompt_semantics(aliased, defaults).reasoning_effort;
    };
    failures += check(collapses(RequestedReasoningEffort::Minimal) == ninfer::ReasoningEffort::Low,
                      "'minimal' did not collapse onto the template's low rung");
    failures += check(collapses(RequestedReasoningEffort::High) == ninfer::ReasoningEffort::XHigh,
                      "'high' did not collapse onto the template's xhigh rung");
    failures += check(collapses(RequestedReasoningEffort::Max) == ninfer::ReasoningEffort::XHigh,
                      "'max' did not collapse onto the template's xhigh rung");
    failures += check(collapses(RequestedReasoningEffort::Medium) == ninfer::ReasoningEffort::Medium,
                      "'medium' did not pass through unchanged");
    failures += check(collapses(RequestedReasoningEffort::None) == ninfer::ReasoningEffort::None,
                      "'none' did not pass through unchanged");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--auto-prefix-grid") != std::string::npos,
              "serve help omits --auto-prefix-grid");

    // --devices selects the ordered CUDA devices the model's pipeline stages run on. Only the shape
    // is parsed here; matching compute capability is the engine's to validate, because it needs real
    // devices.
    failures += check(parse({"ninfer-serve", "model.ninfer"}).devices.empty(),
                      "devices defaulted to a non-empty list");
    const ServeOptions one_device = parse({"ninfer-serve", "model.ninfer", "--devices", "3"});
    failures += check(one_device.devices.size() == 1 && one_device.devices[0] == 3,
                      "--devices with one entry did not reach serving options");
    const ServeOptions pair = parse({"ninfer-serve", "model.ninfer", "--devices", "1,2"});
    failures += check(pair.devices.size() == 2 && pair.devices[0] == 1 && pair.devices[1] == 2,
                      "--devices did not preserve the ordered pair");
    const ServeOptions triple = parse({"ninfer-serve", "model.ninfer", "--devices", "0,1,2"});
    failures += check(triple.devices.size() == 3 && triple.devices[2] == 2,
                      "--devices did not accept more than two stages");

    for (const auto& [value, why] : std::vector<std::pair<std::string, const char*>>{
             {"0,1,2,3,4,5,6,7,8", "more devices than a context holds"},
             {"", "an empty list"},
             {"1,", "a trailing comma"}}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--devices", value});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, why);
    }

    // "0,0" is deliberately permitted: it puts both ranks on one card so the split path can be
    // exercised on a single-GPU machine.
    const ServeOptions same_card = parse({"ninfer-serve", "model.ninfer", "--devices", "0,0"});
    failures += check(same_card.devices.size() == 2 && same_card.devices[0] == 0 &&
                           same_card.devices[1] == 0,
                      "--devices 0,0 was not accepted for single-card split coverage");

    // --stage-layers gives the layers per stage, one count per device.
    const ServeOptions staged =
        parse({"ninfer-serve", "model.ninfer", "--devices", "0,0", "--stage-layers", "20,44"});
    failures += check(staged.stage_layers == std::vector<std::uint32_t>({20, 44}),
                      "--stage-layers did not reach serving options");
    bool stage_layers_alone_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--stage-layers", "20,44"});
    } catch (const std::invalid_argument&) { stage_layers_alone_rejected = true; }
    failures += check(stage_layers_alone_rejected,
                      "--stage-layers without a multi-device split was accepted");
    bool zero_stage_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--devices", "0,0", "--stage-layers", "0,64"});
    } catch (const std::invalid_argument&) { zero_stage_rejected = true; }
    failures += check(zero_stage_rejected, "a stage with no layers was accepted");

    bool exclusive_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--device", "0", "--devices", "0,1"});
    } catch (const std::invalid_argument&) { exclusive_rejected = true; }
    failures += check(exclusive_rejected, "--device and --devices were accepted together");

    // Every option the parser accepts is described in the grouped help.
    const std::string serve_help = serve_usage_text("ninfer-serve");
    for (const std::string_view flag : {"--adaptive-mtp",
                                        "--api-key",
                                        "--assistant-prefill",
                                        "--auto-long-anchors",
                                        "--auto-prefix-grid",
                                        "--cache-tap-ladder",
                                        "--cache-tap-min-gap",
                                        "--cache-taps-per-request",
                                        "--chat-template",
                                        "--concurrent-prefill",
                                        "--context-cache-policy",
                                        "--context-cost-presets",
                                        "--cors",
                                        "--cuda-graph-allowance-mib",
                                        "--default-max-tokens",
                                        "--default-reasoning-effort",
                                        "--default-thinking-budget",
                                        "--derive-session-keys",
                                        "--device",
                                        "--device-profile",
                                        "--device-profile-path",
                                        "--device-snapshot-slots",
                                        "--device-state-slots",
                                        "--devices",
                                        "--disk-kv-directstorage",
                                        "--disk-kv-gib",
                                        "--disk-kv-path",
                                        "--disk-kv-restore",
                                        "--draft-tokens",
                                        "--embedding-q4",
                                        "--embedding-q6",
                                        "--fast-prefill-kernel",
                                        "--first-token-logprobs",
                                        "--frequency-penalty",
                                        "--gdn-state-fp16",
                                        "--greedy",
                                        "--host",
                                        "--host-cache-mib",
                                        "--host-kv-mib",
                                        "--host-state-slots",
                                        "--kv-capacity",
                                        "--kv-dtype",
                                        "--kv-headroom-mib",
                                        "--kv-lease-growth",
                                        "--lenient-assistant-history",
                                        "--lm-head-draft",
                                        "--lm-head-q4",
                                        "--lm-head-q6",
                                        "--log-colours",
                                        "--log-level",
                                        "--log-stats-interval-ms",
                                        "--log-stats-panel",
                                        "--long-anchor-spacing",
                                        "--lookup-ngram",
                                        "--max-cache-markers-per-request",
                                        "--max-concurrency",
                                        "--max-context",
                                        "--max-long-anchors-per-continuation",
                                        "--max-pending-requests",
                                        "--max-private-continuations",
                                        "--max-request-mib",
                                        "--max-shared-prefixes",
                                        "--media-cache-mib",
                                        "--media-live-mib",
                                        "--media-preprocess-threads",
                                        "--min-p",
                                        "--mlp-a8-decode",
                                        "--mtp-attention-window",
                                        "--model-id",
                                        "--mtp-experts-q4",
                                        "--ngram-archive-mib",
                                        "--ngram-draft-tokens",
                                        "--ngram-min-match",
                                        "--ngram-native-sessions",
                                        "--ngram-session-mib",
                                        "--no-cuda-graph",
                                        "--no-prefill-a8",
                                        "--no-prefill-cublas-projections",
                                        "--no-prefix-reuse",
                                        "--no-thinking",
                                        "--no-webui",
                                        "--pending-timeout-ms",
                                        "--port",
                                        "--post-thinking",
                                        "--post-thinking-sampler",
                                        "--post-thinking-temperature",
                                        "--post-thinking-top-k",
                                        "--post-thinking-top-p",
                                        "--prefill-chunk",
                                        "--prefill-cublas",
                                        "--prefix-cache-file",
                                        "--presence-penalty",
                                        "--preserve-thinking",
                                        "--recency-eviction",
                                        "--recover-invariant-failures",
                                        "--release-diverged-checkpoints",
                                        "--request-log-jsonl",
                                        "--request-log-keep",
                                        "--request-log-max-mib",
                                        "--response-store-max-mib",
                                        "--response-store-max-records",
                                        "--rope-scaling-factor",
                                        "--rope-scaling-original-context",
                                        "--rope-yarn",
                                        "--rope-yarn-factor",
                                        "--seed",
                                        "--spec",
                                        "--stage-layers",
                                        "--stats-port",
                                        "--structured-output",
                                        "--temperature",
                                        "--thinking-budget-message",
                                        "--thorough-admission-search",
                                        "--top-k",
                                        "--top-p",
                                        "--unconstrained-response-format",
                                        "--usage-chunk-choice",
                                        "--use-alt-prefix-caching",
                                        "--use-original-prefix-caching",
                                        "--value-aware-demote",
                                        "--vision",
                                        "--vision-cpu",
                                        "--vision-max-merged",
                                        "--vision-offload",
                                        "--vision-residency",
                                        "--vram-headroom-mib",
                                        "--wddm-evictable-budget",
                                        "--webui-mcp-proxy"}) {
        const std::string message = "serve help omits " + std::string(flag);
        failures += check(serve_help.find(flag) != std::string::npos, message.c_str());
    }
    for (const char* section :
         {"MODEL & CONTEXT", "PRECISION & KERNELS", "KV CACHE", "CONTEXT CACHE",
          "SPECULATIVE DECODING", "VISION", "SAMPLING & THINKING", "API BEHAVIOR",
          "NETWORK & LIMITS", "LOGGING"}) {
        failures +=
            check(serve_help.find(section) != std::string::npos, "serve help omits a category");
    }
    failures += check(serve_usage_text("ninfer-serve").find("--devices") != std::string::npos,
                      "serve help omits --devices");
    failures += check(serve_usage_text("ninfer-serve").find("--host-kv-mib") != std::string::npos,
                      "serve help omits context-cache capacities");
    failures += check(serve_usage_text("ninfer-serve").find("device-state=max-concurrency") !=
                          std::string::npos,
                      "serve help omits context-cache defaults");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(serve_usage_text("ninfer-serve").find("--thinking-budget-message") !=
                          std::string::npos,
                      "serve help omits --thinking-budget-message");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures += check(serve_usage_text("ninfer-serve").find("--usage-chunk-choice") !=
                          std::string::npos,
                      "serve help omits --usage-chunk-choice");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--log-stats-panel") != std::string::npos,
                      "serve help omits --log-stats-panel");
    failures += check(serve_usage_text("ninfer-serve").find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(serve_usage_text("ninfer-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--context-cost-presets") != std::string::npos,
              "serve help omits external context-cost presets");
    failures += check(serve_usage_text("ninfer-serve").find("metadata.name") != std::string::npos,
                      "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures += check(logged.request_log_max_mib == 0 && logged.request_log_keep == 4,
                      "request log rotation was not off by default");
    const ServeOptions rotated =
        parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl", "requests.jsonl",
               "--request-log-max-mib", "64", "--request-log-keep", "2"});
    failures += check(rotated.request_log_max_mib == 64 && rotated.request_log_keep == 2,
                      "request log rotation options did not reach serving options");
    bool rotation_without_log_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--request-log-max-mib", "64"});
    } catch (const std::invalid_argument&) { rotation_without_log_rejected = true; }
    failures += check(rotation_without_log_rejected,
                      "--request-log-max-mib was accepted without a request log");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    failures += check(!defaults.fast_prefill_kernel, "the fast prefill kernel must default off");
    failures += check(defaults.speculative.mtp_policy == ninfer::MtpDraftPolicy::Fixed &&
                          parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens",
                                 "5", "--adaptive-mtp"})
                                  .speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive,
                      "--adaptive-mtp did not default off or was not preserved");
    bool adaptive_without_mtp_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2", "--draft-tokens", "7",
                     "--adaptive-mtp"});
    } catch (const std::invalid_argument&) { adaptive_without_mtp_rejected = true; }
    failures += check(adaptive_without_mtp_rejected, "--adaptive-mtp was accepted without MTP");
    failures += check(
        !defaults.first_token_logprobs &&
            parse({"ninfer-serve", "model.ninfer", "--first-token-logprobs"}).first_token_logprobs,
        "--first-token-logprobs did not default off or was not preserved");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--fast-prefill-kernel"})
                          .fast_prefill_kernel,
                      "--fast-prefill-kernel was not preserved");
    failures += check(serve_usage_text("ninfer-serve").find("--fast-prefill-kernel") !=
                          std::string::npos,
                      "serve help omits --fast-prefill-kernel");

    const ServeOptions graph_allowance =
        parse({"ninfer-serve", "model.ninfer", "--cuda-graph-allowance-mib", "512"});
    failures += check(graph_allowance.cuda_graph_allowance_mib == 512 &&
                          graph_allowance.use_cuda_graph,
                      "--cuda-graph-allowance-mib did not preserve its value");
    failures += check(defaults.cuda_graph_allowance_mib == 0,
                      "omitted --cuda-graph-allowance-mib must keep the computed allowance");
    failures += check(serve_usage_text("ninfer-serve").find("--cuda-graph-allowance-mib") !=
                          std::string::npos,
                      "serve help omits --cuda-graph-allowance-mib");
    for (const auto& extra : std::vector<std::vector<std::string>>{
             {"--no-cuda-graph", "--cuda-graph-allowance-mib", "512"},
             {"--cuda-graph-allowance-mib", "18446744073709551615"},
             {"--cuda-graph-allowance-mib", "not-a-number"}}) {
        std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
        args.insert(args.end(), extra.begin(), extra.end());
        bool rejected = false;
        try {
            (void)parse(args);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid --cuda-graph-allowance-mib contract admitted");
    }

    if (failures == 0) { std::cout << "ok\n"; }

    {
        const ServeOptions overlay =
            parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-residency", "overlay",
                   "--vision-max-merged", "12288"});
        failures += check(overlay.vision_residency == ninfer::VisionResidency::Overlay,
                          "--vision-residency overlay did not select overlay residency");
        failures += check(overlay.vision_max_merged_tokens == 12288U,
                          "--vision-max-merged did not set the merged-token budget");
        const ServeOptions resident = parse({"ninfer-serve", "model.ninfer", "--vision"});
        failures += check(resident.vision_residency == ninfer::VisionResidency::Resident,
                          "vision residency does not default to resident");
        failures += check(resident.vision_max_merged_tokens == 16384U,
                          "vision merged-token budget does not default to the item maximum");
        bool overlay_without_vision_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--vision-residency", "overlay"});
        } catch (const std::invalid_argument&) { overlay_without_vision_rejected = true; }
        failures += check(overlay_without_vision_rejected,
                          "--vision-residency overlay without --vision was accepted");
        bool small_budget_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-max-merged", "32"});
        } catch (const std::invalid_argument&) { small_budget_rejected = true; }
        failures += check(small_budget_rejected, "--vision-max-merged 32 was accepted");
        bool unknown_mode_rejected = false;
        try {
            (void)parse(
                {"ninfer-serve", "model.ninfer", "--vision", "--vision-residency", "sometimes"});
        } catch (const std::invalid_argument&) { unknown_mode_rejected = true; }
        failures += check(unknown_mode_rejected, "--vision-residency sometimes was accepted");
        failures += check(parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-offload",
                                 "on"})
                                  .vision_residency == ninfer::VisionResidency::Overlay,
                          "--vision-offload on is not an alias of overlay residency");
        failures += check(parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-residency",
                                 "overlay", "--vision-offload", "off"})
                                  .vision_residency == ninfer::VisionResidency::Resident,
                          "--vision-offload off is not an alias of resident residency");
        bool offload_mode_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-offload", "yes"});
        } catch (const std::invalid_argument&) { offload_mode_rejected = true; }
        failures += check(offload_mode_rejected, "--vision-offload yes was accepted");
        const ServeOptions cpu =
            parse({"ninfer-serve", "model.ninfer", "--vision", "--vision-residency", "cpu"});
        failures += check(cpu.vision_residency == ninfer::VisionResidency::Cpu &&
                              cpu.vision_max_merged_tokens == 256U,
                          "--vision-residency cpu did not select CPU residency at 256 tokens");
        const ServeOptions cpu_alias =
            parse({"ninfer-serve", "model.ninfer", "--vision-cpu", "--vision-max-merged", "1024"});
        failures += check(cpu_alias.enable_vision &&
                              cpu_alias.vision_residency == ninfer::VisionResidency::Cpu &&
                              cpu_alias.vision_max_merged_tokens == 1024U,
                          "--vision-cpu is not --vision with CPU residency, or it overrode an "
                          "explicit --vision-max-merged");
        bool cpu_without_vision_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--vision-residency", "cpu"});
        } catch (const std::invalid_argument&) { cpu_without_vision_rejected = true; }
        failures += check(cpu_without_vision_rejected,
                          "--vision-residency cpu without --vision was accepted");
        failures += check(parse({"ninfer-serve", "model.ninfer", "--concurrent-prefill"})
                                  .concurrent_prefill &&
                              !parse({"ninfer-serve", "model.ninfer"}).concurrent_prefill,
                          "--concurrent-prefill was not an off-by-default switch");
        failures += check(parse({"ninfer-serve", "model.ninfer", "--recover-invariant-failures"})
                                  .recover_invariant_failures &&
                              !parse({"ninfer-serve", "model.ninfer"}).recover_invariant_failures,
                          "--recover-invariant-failures was not an off-by-default switch");
        failures += check(parse({"ninfer-serve", "model.ninfer", "--thorough-admission-search"})
                                  .context_cache.thorough_admission_search &&
                              !parse({"ninfer-serve", "model.ninfer"})
                                   .context_cache.thorough_admission_search,
                          "--thorough-admission-search was not an off-by-default switch");
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--recency-eviction"})
                          .context_cache.recency_eviction &&
                      !parse({"ninfer-serve", "model.ninfer"}).context_cache.recency_eviction,
                  "--recency-eviction was not an off-by-default switch");
        failures += check(
            parse({"ninfer-serve", "model.ninfer", "--stats-port", "8081"}).stats_port == 8081 &&
                parse({"ninfer-serve", "model.ninfer"}).stats_port == 0,
            "--stats-port was not an off-by-default port");
        for (const char* stats_port : {"8080", "65536"}) {
            bool rejected = false;
            try {
                (void)parse({"ninfer-serve", "model.ninfer", "--stats-port", stats_port});
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "--stats-port accepted the main port or an invalid port");
        }
        {
            const ServeOptions interpolated =
                parse({"ninfer-serve", "model.ninfer", "--rope-scaling-factor", "2.5",
                       "--rope-scaling-original-context", "131072"});
            failures += check(interpolated.rope_scaling_factor == 2.5F &&
                                  interpolated.rope_scaling_original_context == 131072U,
                              "--rope-scaling-factor/--rope-scaling-original-context not parsed");
            const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
            failures += check(defaults.rope_scaling_factor == 1.0F &&
                                  defaults.rope_scaling_original_context == 0U,
                              "position interpolation is not off by default");
            for (const char* bad : {"0.5", "33", "nan"}) {
                bool rejected = false;
                try {
                    (void)parse({"ninfer-serve", "model.ninfer", "--rope-scaling-factor", bad});
                } catch (const std::invalid_argument&) { rejected = true; }
                const std::string message = std::string("--rope-scaling-factor accepted ") + bad;
                failures += check(rejected, message.c_str());
            }
        }
        {
            failures += check(
                parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3",
                       "--mtp-attention-window", "4096"})
                            .speculative.mtp_attention_window == 4096 &&
                    parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3"})
                            .speculative.mtp_attention_window == 0,
                "--mtp-attention-window was not an off-by-default MTP option");
            for (const std::vector<std::string>& bad :
                 {std::vector<std::string>{"--mtp-attention-window", "4096"},
                  std::vector<std::string>{"--spec", "mtp", "--draft-tokens", "5",
                                           "--mtp-attention-window", "5"},
                  std::vector<std::string>{"--spec", "dflash2", "--draft-tokens", "5",
                                           "--mtp-attention-window", "4096"}}) {
                std::vector<std::string> arguments{"ninfer-serve", "model.ninfer"};
                arguments.insert(arguments.end(), bad.begin(), bad.end());
                bool rejected = false;
                try {
                    (void)parse(std::move(arguments));
                } catch (const std::invalid_argument&) { rejected = true; }
                failures += check(rejected, "an invalid --mtp-attention-window was accepted");
            }
        }
        {
            failures += check(parse({"ninfer-serve", "model.ninfer", "--derive-session-keys"})
                                      .derive_session_keys &&
                                  !parse({"ninfer-serve", "model.ninfer"}).derive_session_keys,
                              "--derive-session-keys was not an off-by-default switch");
            bool rejected = false;
            try {
                (void)parse(
                    {"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--derive-session-keys"});
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "--derive-session-keys accepted without prefix reuse");

            const auto turn = [](ninfer::ChatRole role, std::string text) {
                ChatTurn out;
                out.role = role;
                ContentPart part;
                part.text = std::move(text);
                out.content.push_back(std::move(part));
                return out;
            };
            GenerationRequest first;
            first.messages          = {turn(ninfer::ChatRole::System, "be brief"),
                                       turn(ninfer::ChatRole::User, "hello")};
            GenerationRequest later = first;
            later.messages.push_back(turn(ninfer::ChatRole::Assistant, "hi"));
            later.messages.push_back(turn(ninfer::ChatRole::User, "and more"));
            GenerationRequest other = first;
            other.messages[1]       = turn(ninfer::ChatRole::User, "goodbye");
            GenerationRequest image = first;
            ContentPart picture;
            picture.kind = ContentKind::Image;
            image.messages[1].content.push_back(picture);
            const auto key = derived_session_key(first);
            failures +=
                check(key && key->size() == 19 && key->starts_with("cs-") &&
                          derived_session_key(later) == key && derived_session_key(image) == key &&
                          derived_session_key(other) != key,
                      "a derived session key is not stable per conversation");
            GenerationRequest silent;
            silent.messages = {turn(ninfer::ChatRole::System, "be brief"),
                               turn(ninfer::ChatRole::User, "")};
            failures += check(!derived_session_key(silent),
                              "a conversation without user text got a derived session key");
        }
        {
            failures += check(!parse({"ninfer-serve", "model.ninfer"}).post_thinking_overrides,
                              "post-thinking sampling is not off by default");
            const ServeOptions preset = parse({"ninfer-serve", "model.ninfer", "--post-thinking"});
            failures += check(preset.post_thinking_overrides &&
                                  !preset.post_thinking_overrides->temperature &&
                                  !preset.post_thinking_overrides->top_k,
                              "--post-thinking did not select the bare preset");
            const ServeOptions fields =
                parse({"ninfer-serve", "model.ninfer", "--post-thinking-temperature", "0.3",
                       "--post-thinking-sampler", "top_p=0.9,top_k=10,presence=1.5"});
            failures += check(fields.post_thinking_overrides &&
                                  fields.post_thinking_overrides->temperature == 0.3F &&
                                  fields.post_thinking_overrides->top_p == 0.9F &&
                                  fields.post_thinking_overrides->top_k == 10 &&
                                  fields.post_thinking_overrides->presence_penalty == 1.5F &&
                                  !fields.post_thinking_overrides->min_p,
                              "--post-thinking-* fields were not parsed");
            for (const std::vector<std::string>& bad :
                 {std::vector<std::string>{"--post-thinking-temperature", "2.5"},
                  std::vector<std::string>{"--post-thinking-top-k", "21"},
                  std::vector<std::string>{"--post-thinking-sampler", "temp"},
                  std::vector<std::string>{"--post-thinking-sampler", "seed=1"},
                  std::vector<std::string>{"--post-thinking-sampler", "top_p=nan"}}) {
                bool rejected = false;
                try {
                    (void)parse({"ninfer-serve", "model.ninfer", bad[0].c_str(), bad[1].c_str()});
                } catch (const std::invalid_argument&) { rejected = true; }
                const std::string message = "accepted " + bad[0] + " " + bad[1];
                failures += check(rejected, message.c_str());
            }
        }
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--value-aware-demote"})
                          .context_cache.value_aware_demote &&
                      !parse({"ninfer-serve", "model.ninfer"}).context_cache.value_aware_demote,
                  "--value-aware-demote was not an off-by-default switch");
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--kv-lease-growth"})
                          .context_cache.kv_lease_growth &&
                      !parse({"ninfer-serve", "model.ninfer"}).context_cache.kv_lease_growth,
                  "--kv-lease-growth was not an off-by-default switch");
        failures += check(!parse({"ninfer-serve", "model.ninfer"}).log_colours &&
                              parse({"ninfer-serve", "model.ninfer", "--log-colours", "on"})
                                      .log_colours == true &&
                              parse({"ninfer-serve", "model.ninfer", "--log-colours", "off"})
                                      .log_colours == false,
                          "--log-colours was not an unset-by-default on|off switch");
        bool bad_colours_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--log-colours", "auto"});
        } catch (const std::invalid_argument&) { bad_colours_rejected = true; }
        failures += check(bad_colours_rejected, "--log-colours accepted a value other than on|off");
        const ServeOptions automatic_anchors =
            parse({"ninfer-serve", "model.ninfer", "--auto-long-anchors", "--long-anchor-spacing",
                   "4096"});
        failures +=
            check(!parse({"ninfer-serve", "model.ninfer"}).context_cache.automatic_long_anchors &&
                      parse({"ninfer-serve", "model.ninfer"})
                              .context_cache.long_anchor_min_spacing_tokens == 1024U &&
                      automatic_anchors.context_cache.automatic_long_anchors &&
                      automatic_anchors.context_cache.long_anchor_min_spacing_tokens == 4096U &&
                      parse({"ninfer-serve", "model.ninfer", "--auto-long-anchors",
                             "--long-anchor-spacing", "0"})
                              .context_cache.long_anchor_min_spacing_tokens == 0U,
                  "--auto-long-anchors was not an off-by-default switch carrying its spacing");
        bool spacing_without_anchors_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--long-anchor-spacing", "512"});
        } catch (const std::invalid_argument&) { spacing_without_anchors_rejected = true; }
        failures += check(spacing_without_anchors_rejected,
                          "--long-anchor-spacing was accepted without --auto-long-anchors");
        failures += check(parse({"ninfer-serve", "model.ninfer", "--assistant-prefill"}).assistant_prefill &&
                              !parse({"ninfer-serve", "model.ninfer"}).assistant_prefill,
                          "--assistant-prefill was not an off-by-default switch");
        failures += check(parse({"ninfer-serve", "model.ninfer", "--unconstrained-response-format"})
                              .unconstrained_response_format,
                          "--unconstrained-response-format was not preserved");
        bool unconstrained_with_structured_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--structured-output",
                         "--unconstrained-response-format"});
        } catch (const std::invalid_argument&) { unconstrained_with_structured_rejected = true; }
        failures += check(unconstrained_with_structured_rejected,
                          "--unconstrained-response-format was accepted with --structured-output");
        const ServeOptions headroom = parse({"ninfer-serve", "model.ninfer", "--kv-capacity",
                                             "auto", "--vram-headroom-mib", "512"});
        failures += check(headroom.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                              headroom.kv_capacity.automatic_headroom_bytes == (512ULL << 20),
                          "--vram-headroom-mib is not an alias of --kv-headroom-mib");
    }

    return failures == 0 ? 0 : 1;
}
