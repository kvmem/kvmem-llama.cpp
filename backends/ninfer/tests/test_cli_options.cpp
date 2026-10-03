#include "options.h"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int run_tests() {
    int failures = 0;
    const auto memory_defaults = parse({"ninfer-cli", "model.ninfer", "--prompt", "x"});
    failures += check(memory_defaults.cuda_memory_policy == ninfer::CudaMemoryPolicy::DriverDefault &&
                          memory_defaults.cuda_vram_reserve_bytes == 0 &&
                          memory_defaults.cuda_memory_probe_step_bytes == 0 &&
                          !memory_defaults.use_alt_prefix_caching &&
                          memory_defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit,
                      "CUDA memory policy defaults changed");
    const auto resident = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                 "--cuda-memory-policy", "strict", "--kv-capacity", "auto"});
    failures += check(resident.cuda_memory_policy == ninfer::CudaMemoryPolicy::StrictVram &&
                          resident.cuda_vram_reserve_bytes == (64ULL << 20) &&
                          resident.cuda_memory_probe_step_bytes == (128ULL << 20) &&
                          resident.use_alt_prefix_caching &&
                          resident.kv_capacity.automatic_headroom_bytes == (64ULL << 20),
                      "strict must enable hybrid caching and use its own automatic reserve");
    const auto mixed = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                              "--cuda-memory-policy", "mixed", "--kv-capacity", "auto"});
    failures += check(mixed.cuda_memory_policy == ninfer::CudaMemoryPolicy::Mixed &&
                          mixed.cuda_vram_reserve_bytes == 0 &&
                          mixed.cuda_memory_probe_step_bytes == 0 &&
                          mixed.kv_capacity.automatic_headroom_bytes == 0,
                      "mixed must not inherit strict probing or default-policy headroom");
    const auto ordinary = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                 "--cuda-memory-policy", "default", "--kv-capacity", "auto",
                                 "--kv-headroom-mib", "100"});
    failures += check(ordinary.cuda_memory_policy == ninfer::CudaMemoryPolicy::DriverDefault &&
                          ordinary.kv_capacity.automatic_headroom_bytes == (100ULL << 20),
                      "default policy must retain explicit automatic-KV headroom");
    const auto custom = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                               "--cuda-memory-policy", "strict-0-1", "--kv-capacity", "auto"});
    failures += check(custom.cuda_vram_reserve_bytes == 0 &&
                          custom.cuda_memory_probe_step_bytes == (1ULL << 20) &&
                          custom.kv_capacity.automatic_headroom_bytes == 0,
                      "strict custom reserve and step minimum were not preserved");
    const auto custom_auto = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                    "--cuda-memory-policy", "strict-00072-00256", "--kv-capacity", "auto"});
    failures += check(custom_auto.cuda_vram_reserve_bytes == (72ULL << 20) &&
                          custom_auto.cuda_memory_probe_step_bytes == (256ULL << 20) &&
                          custom_auto.kv_capacity.automatic_headroom_bytes == (72ULL << 20),
                      "custom strict decimal values must control auto-KV headroom");
    const auto maximum_mib = std::numeric_limits<std::size_t>::max() >> 20;
    const auto memory_boundaries = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                          "--cuda-memory-policy", "strict-" + std::to_string(maximum_mib) + "-16384"});
    failures += check(memory_boundaries.cuda_memory_policy == ninfer::CudaMemoryPolicy::StrictVram &&
                          memory_boundaries.cuda_vram_reserve_bytes == (maximum_mib << 20) &&
                          memory_boundaries.cuda_memory_probe_step_bytes == (16384ULL << 20),
                      "valid CUDA memory parameter boundaries were rejected or truncated");
    const auto reset_policy = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                     "--cuda-memory-policy", "strict-7-8", "--cuda-memory-policy", "default"});
    failures += check(reset_policy.cuda_memory_policy == ninfer::CudaMemoryPolicy::DriverDefault &&
                          reset_policy.cuda_vram_reserve_bytes == 0 &&
                          reset_policy.cuda_memory_probe_step_bytes == 0 &&
                          !reset_policy.use_alt_prefix_caching,
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
        std::vector<std::string> arguments{"ninfer-cli", "model.ninfer", "--prompt", "x"};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        bool rejected = false;
        try {
            (void)parse(arguments);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid or conflicting CUDA residency options admitted");
    }
    const auto memory_help = ninfer::cli::usage_text("ninfer");
    failures += check(memory_help.find("strict-RESERVE-STEP") != std::string::npos,
                      "CLI help omits custom strict policy syntax");
    for (const auto* backend : {"mtp", "dflash", "dflash2"}) {
        for (unsigned width = 0; width <= 63; ++width) {
            const auto mixed =
                parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                       "--draft-tokens", "5", "--ngram-draft-tokens", std::to_string(width)});
            failures += check(mixed.speculative.draft_tokens == 5 &&
                                  mixed.speculative.ngram_draft_tokens == width,
                              "CLI failed to preserve independent neural/ngram widths");
        }
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                                   "--draft-tokens", "5", "--ngram-draft-tokens", "64"});
                  }),
                  "CLI admitted unsupported ngram width");
        failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                                 "--draft-tokens", "3"})
                                  .speculative.ngram_draft_tokens == 15,
                          "CLI ngram drafting is not on by default beside a drafter");
    }
    failures += check(
        parse({"ninfer-cli", "model.ninfer", "--prompt", "x"}).speculative.ngram_draft_tokens == 0,
        "CLI ngram default needs a neural backend");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--ngram-draft-tokens 0..63") !=
                          std::string::npos,
                      "CLI help has a stale ngram width limit");
    failures +=
        check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--ngram-draft-tokens", "0"})
                      .speculative.ngram_draft_tokens == 0,
              "explicit ngram-off must not require a neural backend");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                             "--draft-tokens", "3", "--ngram-draft-tokens", "0"})
                              .speculative.ngram_draft_tokens == 0,
                      "explicit ngram-off beside a drafter was overridden");
    const auto ngram = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "dflash2",
                              "--draft-tokens", "5", "--ngram-draft-tokens", "15"});
    failures +=
        check(ngram.speculative.ngram_draft_tokens == 15 && ngram.speculative.draft_tokens == 5,
              "CLI ngram widths changed");
    const auto mtp_ngram = parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                  "--draft-tokens", "5", "--ngram-draft-tokens", "1"});
    failures += check(mtp_ngram.speculative.draft_tokens == 5 &&
                          mtp_ngram.speculative.ngram_draft_tokens == 1,
                      "CLI narrow ngram/wide MTP pair changed");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                               "--draft-tokens", "16", "--ngram-draft-tokens", "15"});
              }),
              "CLI admitted unsupported MTP neural width");
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--vision",
                             "--vision-offload", "on"})
                              .vision_residency == ninfer::VisionResidency::Overlay,
                      "--vision-offload on is not an alias of overlay residency");
    {
        const ninfer::cli::Options cpu =
            parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--vision-cpu"});
        failures +=
            check(cpu.enable_vision && cpu.vision_residency == ninfer::VisionResidency::Cpu &&
                      cpu.vision_max_merged_tokens == 256U,
                  "--vision-cpu did not select CPU residency at 256 merged tokens");
    }
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--vision",
                                       "--vision-offload", "maybe"});
                      }),
                      "--vision-offload maybe was accepted");
    const ninfer::cli::Options headroom = parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                                 "--kv-capacity", "auto", "--vram-headroom-mib",
                                                 "256"});
    failures += check(headroom.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          headroom.kv_capacity.automatic_headroom_bytes == (256ULL << 20),
                      "--vram-headroom-mib is not an alias of --kv-headroom-mib");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options dflash_vision =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "x"}).rope_yarn_factor ==
                          1.0F,
                      "CLI YaRN factor must default to 1");
    for (const auto* factor : {"1", "2.5", "4"}) {
        const auto yarn =
            parse({"ninfer-cli", "model.ninfer", "--prompt", "x", "--rope-yarn-factor", factor});
        failures += check(yarn.rope_yarn_factor == std::stof(factor) && yarn.max_context == 2048 &&
                              yarn.kv_capacity.explicit_tokens == 2048,
                          "a YaRN factor must not grow the default context or KV");
    }
    for (const auto* factor : {"0", "0.99", "4.01", "-1", "nan", "inf", "-inf", "1e999", "2x", ""}) {
        failures += check(rejects([&] {
                              (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "x",
                                           "--rope-yarn-factor", factor});
                          }),
                          "invalid YaRN factor accepted");
    }
    failures += check(rejects([] {
                          (void)parse(
                              {"ninfer-cli", "model.ninfer", "--prompt", "x", "--rope-yarn-factor"});
                      }),
                      "missing YaRN factor accepted");
    // Every option the parser accepts is described in the grouped help.
    for (const std::string_view flag : {"--chat-template",
                                        "--device",
                                        "--devices",
                                        "--draft-tokens",
                                        "--embedding-q4",
                                        "--embedding-q6",
                                        "--frequency-penalty",
                                        "--gdn-state-fp16",
                                        "--greedy",
                                        "--json",
                                        "--json-schema",
                                        "--kv-capacity",
                                        "--kv-dtype",
                                        "--kv-headroom-mib",
                                        "--lm-head-draft",
                                        "--lm-head-q4",
                                        "--lm-head-q6",
                                        "--log-colours",
                                        "--log-level",
                                        "--lookup-ngram",
                                        "--max-context",
                                        "--max-new",
                                        "--messages",
                                        "--min-p",
                                        "--mlp-a8-decode",
                                        "--mtp-attention-window",
                                        "--mtp-experts-q4",
                                        "--ngram-draft-tokens",
                                        "--ngram-min-match",
                                        "--no-cuda-graph",
                                        "--no-prefill-a8",
                                        "--no-prefill-cublas-projections",
                                        "--no-thinking",
                                        "--post-thinking",
                                        "--post-thinking-sampler",
                                        "--post-thinking-temperature",
                                        "--post-thinking-top-k",
                                        "--post-thinking-top-p",
                                        "--prefill-chunk",
                                        "--prefill-cublas",
                                        "--presence-penalty",
                                        "--print-token-ids",
                                        "--prompt",
                                        "--raw-output",
                                        "--reasoning-effort",
                                        "--reasoning-stop",
                                        "--rope-yarn",
                                        "--rope-yarn-factor",
                                        "--seed",
                                        "--spec",
                                        "--stage-layers",
                                        "--stop",
                                        "--stop-token-id",
                                        "--temperature",
                                        "--thinking-budget",
                                        "--top-k",
                                        "--top-p",
                                        "--vision",
                                        "--vision-max-merged",
                                        "--vision-offload",
                                        "--vision-residency",
                                        "--vram-headroom-mib",
                                        "--wddm-evictable-budget"}) {
        const std::string message = "CLI help omits " + std::string(flag);
        failures += check(help.find(flag) != std::string::npos, message.c_str());
    }
    for (const char* section :
         {"INPUT", "CONTEXT", "KV CACHE", "SPECULATIVE DECODING", "PRECISION & KERNELS", "SAMPLING",
          "THINKING", "OUTPUT", "VISION", "LOGGING"}) {
        failures += check(help.find(section) != std::string::npos, "CLI help omits a category");
    }
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninfer::cli::Options route_defaults =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(!route_defaults.prefill_cublas && route_defaults.prefill_cublas_projections &&
                          route_defaults.speculative.lookup_ngram == 0,
                      "the cuBLAS prefill route or context lookup is on by default");
    const ninfer::cli::Options route =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--spec", "mtp", "--draft-tokens",
               "3", "--lookup-ngram", "5", "--prefill-cublas", "--no-prefill-cublas-projections"});
    failures += check(route.prefill_cublas && !route.prefill_cublas_projections &&
                          route.speculative.lookup_ngram == 5 &&
                          route.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          route.speculative.draft_tokens == 3,
                      "CLI did not parse the cuBLAS prefill and context-lookup controls");
    for (const char* flag : {"--prefill-cublas", "--no-prefill-cublas-projections", "--lookup-ngram"}) {
        failures += check(help.find(flag) != std::string::npos,
                          "CLI help omits an accepted prefill or drafting control");
    }
    const ninfer::cli::Options logging =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(help.find("--log-colours") != std::string::npos,
                      "CLI help omits the log-colours control");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--device",
                                       "1", "--devices", "0,1"});
                      }),
                      "--device and --devices were accepted together");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices",
                               "4294967296"});
              }),
              "--devices silently narrowed an out-of-range id instead of rejecting it");
    const ninfer::cli::Options same_card = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,0"});
    failures += check(same_card.devices.size() == 2 && same_card.devices[0] == 0 &&
                           same_card.devices[1] == 0,
                      "--devices 0,0 was not accepted for single-card split coverage");
    const ninfer::cli::Options three = parse(
        {"ninfer-cli", "model.ninfer", "--prompt", "hello", "--devices", "0,1,2"});
    failures += check(three.devices.size() == 3, "--devices did not accept more than two stages");
    const ninfer::cli::Options staged = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                               "--devices", "0,0", "--stage-layers", "20,44"});
    failures += check(staged.stage_layers == std::vector<std::uint32_t>({20, 44}),
                      "--stage-layers did not reach the options");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--stage-layers", "20,44"});
                      }),
                      "--stage-layers without a multi-device split was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--devices", "0,0", "--stage-layers", "0,64"});
                      }),
                      "a stage with no layers was accepted");
    failures += check(parse({"ninfer", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                             "--draft-tokens", "3", "--mtp-attention-window", "2048"})
                              .speculative.mtp_attention_window == 2048,
                      "CLI --mtp-attention-window not parsed");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "x",
                                       "--mtp-attention-window", "2048"});
                      }),
                      "CLI --mtp-attention-window accepted without --spec mtp");
    failures += check(!parse({"ninfer", "model.ninfer", "--prompt", "x"}).post_thinking_sampling,
                      "CLI post-thinking sampling is not off by default");
    const auto post_thinking =
        parse({"ninfer", "model.ninfer", "--prompt", "x", "--post-thinking-top-k", "5",
               "--post-thinking-sampler", "temp=0.4,frequency=-1", "--greedy"});
    failures += check(post_thinking.post_thinking_sampling &&
                          post_thinking.post_thinking_sampling->top_k == 5 &&
                          post_thinking.post_thinking_sampling->frequency_penalty == -1.0F &&
                          post_thinking.post_thinking_sampling->temperature == 0.0F,
                      "CLI post-thinking fields or --greedy over them");
    failures += check(rejects([] {
                          (void)parse({"ninfer", "model.ninfer", "--prompt", "x",
                                       "--post-thinking-sampler", "top_k=3,"});
                      }),
                      "a trailing empty post-thinking field was accepted");
    const auto structured = parse({"ninfer", "model.ninfer", "--prompt", "hello", "--json"});
    failures +=
        check(structured.structured_output.kind == ninfer::StructuredOutputKind::JsonObject &&
                  structured.enable_thinking == false,
              "CLI JSON mode");
    const auto structured_reasoning = parse(
        {"ninfer", "model.ninfer", "--prompt", "hello", "--json", "--reasoning-effort", "medium"});
    failures += check(structured_reasoning.enable_thinking != false &&
                          structured_reasoning.reasoning_effort == ninfer::ReasoningEffort::Medium,
                      "CLI structured output preserves explicit reasoning");
    failures += check(
        rejects([] {
            (void)parse({"ninfer", "model.ninfer", "--prompt", "x", "--json", "--raw-output"});
        }),
        "raw JSON output should be rejected");
    return failures == 0 ? 0 : 1;
}

int main() {
    try {
        return run_tests();
    } catch (const std::exception& error) {
        std::cerr << "Unexpected CLI test exception: " << error.what() << '\n';
        return 1;
    }
}
