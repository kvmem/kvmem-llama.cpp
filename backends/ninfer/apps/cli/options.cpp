#include "options.h"
#include "product/cuda_memory_options.h"
#include "product/post_thinking_options.h"
#include "product/rope_yarn_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace ninfer::cli {
namespace {

std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const char* text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

// Same shape as serve's --devices: the ordered devices the model's pipeline stages run on.
// Repeating an id puts several stages on one card, which exercises the split path without a
// second GPU. Existence is checked at engine startup; this only parses the shape.
constexpr std::size_t kMaximumDevices = 8;
std::vector<int> parse_device_list(std::string_view text) {
    std::vector<int> devices;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece =
            text.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                               : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--devices entries must not be empty"); }
        const std::string entry(piece);
        const std::uint64_t raw = parse_u64(entry.c_str(), "devices");
        if (raw > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("invalid devices: " + entry);
        }
        devices.push_back(static_cast<int>(raw));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    if (devices.empty() || devices.size() > kMaximumDevices) {
        throw std::invalid_argument("--devices takes between one and " +
                                    std::to_string(kMaximumDevices) + " CUDA device ids");
    }
    // Repeated ids are deliberately permitted: they put several stages on one card, which saves no
    // memory but exercises the whole split path on a single-GPU machine.
    return devices;
}

// "30,34": the layers each pipeline stage owns, one count per device in --devices. Whether they add
// up to the model's layers is checked when the model loads; this only parses the shape.
std::vector<std::uint32_t> parse_stage_layers(std::string_view text) {
    std::vector<std::uint32_t> counts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece =
            text.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                               : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--stage-layers entries must not be empty"); }
        const std::string entry(piece);
        counts.push_back(parse_u32(entry.c_str(), "stage-layers", false));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return counts;
}

int parse_device(const char* text) {
    const std::uint64_t value = parse_u64(text, "device");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid device: ") + text);
    }
    return static_cast<int>(value);
}

float parse_float(const char* text, std::string_view label, float minimum, float maximum) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<float>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane. Opt-in.
    if (text == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    // rk4v4: rotated 4-bit Lloyd-Max keys with rk8v4's packed int4 value plane. Opt-in.
    if (text == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    // rk4v4-e8: rotated E8-snapped int4 keys with the rk8v4 value plane. Opt-in.
    if (text == "rk4v4-e8") { return KvCacheStorage::RotatedInt4KeyInt4ValueE8; }
    // rk2v4-e8: rotated keys as E8 root codes, two bytes per eight dimensions. Opt-in.
    if (text == "rk2v4-e8") { return KvCacheStorage::RotatedE8RootKeyInt4Value; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

ReasoningEffort parse_reasoning_effort(std::string_view text) {
    if (text == "none") { return ReasoningEffort::None; }
    if (text == "minimal") { return ReasoningEffort::Minimal; }
    if (text == "high") { return ReasoningEffort::High; }
    if (text == "max") { return ReasoningEffort::Max; }
    if (text == "low") { return ReasoningEffort::Low; }
    if (text == "medium") { return ReasoningEffort::Medium; }
    if (text == "xhigh") { return ReasoningEffort::XHigh; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--messages <messages.json>) [options]\n"
           "\n"
           "Runs one generation: the answer streams to stdout, reasoning and diagnostics\n"
           "to stderr.\n"
           "  --help, -h                    show this help and exit\n"
           "\n"
           "INPUT\n"
           "  --prompt TEXT                 one user message\n"
           "  --messages FILE               a JSON chat transcript; message content accepts\n"
           "                                text, image/image_url and video/video_url parts,\n"
           "                                whose sources may be local paths, HTTP(S) URLs\n"
           "                                or base64 data URIs\n"
           "  --chat-template FILE          replace the artifact's chat template\n"
           "\n"
           "CONTEXT\n"
           "  --max-context N               context ceiling in tokens (default 2048)\n"
           "  --prefill-chunk N             prefill chunk in tokens, a multiple of 128\n"
           "                                (default 1024)\n"
           "  --max-new N                   cap on generated tokens (default 128)\n"
           "  --device N                    CUDA device index (default 0)\n"
           "  --devices A,B,...             one pipeline stage per listed GPU, each owning\n"
           "                                its layers' weights, KV cache and state, the\n"
           "                                first also the embedding, head and round state;\n"
           "                                Linux only, and a repeated id (--devices 0,0)\n"
           "                                exercises the path on one GPU\n"
           "  --stage-layers A,B,...        layers per stage (default: split by each GPU's\n"
           "                                free memory)\n"
           "  --no-cuda-graph               decode without CUDA Graphs\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto          KV capacity in tokens (default --max-context);\n"
           "                                auto follows --cuda-memory-policy\n"
           "  --kv-headroom-mib N           default policy only: auto KV leaves free (default\n"
           "                                " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           "); alias --vram-headroom-mib\n"
           "  --kv-dtype T                  bf16 (default), int8, fp8, rk8v4, rk4v4,\n"
           "                                rk4v4-e8, rk2v4-e8, nvfp4 or k8v4\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2     speculative decoding backend\n"
           "  --draft-tokens N              drafts per round, 1..15\n"
           "  --lm-head-draft               draft with the optimized proposal head\n"
           "  --mtp-attention-window N      the MTP draft head attends to its first 64 keys\n"
           "                                and the newest N before its query, not the\n"
           "                                whole history; verification is unchanged\n"
           "                                (default 0: whole history)\n"
           "  --ngram-draft-tokens 0..63    copy up to that many tokens per round from\n"
           "                                earlier prompt, tool-result or output text that\n"
           "                                the last --ngram-min-match tokens match,\n"
           "                                verified alongside --spec; on with 15 whenever\n"
           "                                --spec is set, 0 disables it\n"
           "  --ngram-min-match N           shortest match a copy is drawn from, 4..64\n"
           "                                (default 12)\n"
           "  --lookup-ngram N              context-lookup drafting alongside --spec: the\n"
           "                                last N tokens are matched against the sequence\n"
           "                                so far and what followed is proposed (default 0:\n"
           "                                off)\n"
           "\n"
           "PRECISION & KERNELS (off by default)\n"
           "  --lm-head-q4 | --lm-head-q6   store the output head as Q4 or Q6 while loading\n"
           "  --embedding-q4 | --embedding-q6\n"
           "                                store the token embedding as Q4 or Q6 while\n"
           "                                loading\n"
           "  --mtp-experts-q4              Qwen3.6-35B-A3B: store the MTP layer's routed\n"
           "                                experts in the text layers' formats\n"
           "  --gdn-state-fp16              keep the GDN recurrent state in FP16\n"
           "  --mlp-a8-decode               integer-activation MLP gate_up at decode and\n"
           "                                verify widths\n"
           "  --no-prefill-a8               return full prefill tiles to their A16 routes,\n"
           "                                which is how the integer routes are measured on\n"
           "                                a whole request\n"
           "  --prefill-cublas              hand wide prefill GEMMs to cuBLAS: a large\n"
           "                                prefill speedup for a small perplexity cost;\n"
           "                                wants a larger --prefill-chunk\n"
           "  --no-prefill-cublas-projections\n"
           "                                with --prefill-cublas, keep the attention and\n"
           "                                GDN input projections off that route\n"
           "  --rope-yarn                   past the model's native window, apply Qwen's\n"
           "                                YaRN at factor --max-context / native instead of\n"
           "                                unscaled RoPE\n"
           "  --rope-yarn-factor F          apply YaRN at this fixed factor, 1..4, to every\n"
           "                                position whatever --max-context is (default 1:\n"
           "                                as --rope-yarn decides)\n"
           "  --rope-scaling-factor F       instead of YaRN, interpolate positions past\n"
           "                                --rope-scaling-original-context linearly by F,\n"
           "                                1..32 (default 1: off)\n"
           "  --rope-scaling-original-context N\n"
           "                                the interpolation threshold (default: the\n"
           "                                model's native window)\n"
           "  --cuda-memory-policy MODE    behavior when dedicated GPU memory is insufficient:\n"
           "    default                     plan against CUDA-reported free memory; ordinary\n"
           "                                driver behavior, no dedicated-residency guarantee\n"
           "    mixed                       attempt requested allocations beyond CUDA free;\n"
           "                                allow Windows to borrow system RAM for device data.\n"
           "                                Placement is driver-managed, not forced to Shared;\n"
           "                                inference may become much slower. Auto KV is capped\n"
           "                                at one page-rounded context window per concurrent\n"
           "                                request; allocation failure is reported, not retried\n"
           "    strict                      require observed dedicated-VRAM residency; equals\n"
           "                                strict-64-128 (64 MiB reserve, 128 MiB probe step)\n"
           "    strict-RESERVE-STEP          customize both values in MiB; RESERVE >= 0,\n"
           "                                STEP 1..16384. Example: strict-128-64. Reserve is\n"
           "                                verified at startup, not permanently locked; step\n"
           "                                controls probes, final device arenas stay contiguous\n"
           "                                strict selects its required cache automatically.\n"
           "                                mixed/strict: Windows single-GPU text, no WDDM mode.\n"
           "  --use-alt-prefix-caching     hybrid device context cache\n"
           "                                CLI uses no persistent Host retention tier\n"
           "  --wddm-evictable-budget       Windows D3D12 builds: budget against dedicated\n"
           "                                memory, holding arenas resident\n"
           "\n"
           "SAMPLING\n"
           "  --temperature F               0..2\n"
           "  --top-p F                     0..1\n"
           "  --top-k N                     0..20\n"
           "  --min-p F                     0..1\n"
           "  --presence-penalty F          -2..2\n"
           "  --frequency-penalty F         -2..2\n"
           "  --seed N                      fixed random seed\n"
           "  --greedy                      force temperature 0 (exact argmax)\n"
           "  --post-thinking               sample the answer of a thinking request with the\n"
           "                                post-thinking preset (temperature 0.2) from the\n"
           "                                token after its reasoning closes (default off)\n"
           "  --post-thinking-temperature F 0..2; implies --post-thinking\n"
           "  --post-thinking-top-p F       0..1; implies --post-thinking\n"
           "  --post-thinking-top-k N       0..20; implies --post-thinking\n"
           "  --post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F]\n"
           "                                the same fields in one flag\n"
           "  --stop-token-id N             stop on this token id; repeatable\n"
           "  --stop TEXT                   stop on this answer text; repeatable\n"
           "  --reasoning-stop TEXT         stop on this reasoning text; repeatable\n"
           "\n"
           "THINKING (off by default; an explicit effort or budget enables it)\n"
           "  --no-thinking                 disable thinking\n"
           "  --reasoning-effort E          none, minimal, low, medium, high, xhigh or max\n"
           "  --thinking-budget N           cap model-origin thinking tokens; inserted\n"
           "                                control tokens count toward --max-new\n"
           "\n"
           "OUTPUT\n"
           "  --json                        constrain the answer to a JSON object\n"
           "  --json-schema FILE            enforce a supported JSON schema\n"
           "  --raw-output                  print the raw generated text without framing\n"
           "  --print-token-ids             also print the generated token ids\n"
           "\n"
           "VISION (off by default)\n"
           "  --vision                      accept images and video and load the Vision GPU\n"
           "                                allocations\n"
           "  --vision-residency resident|overlay|cpu\n"
           "                                overlay keeps the Vision tower in host memory\n"
           "                                and borrows device memory per image (alias\n"
           "                                --vision-offload); cpu runs it on CPU threads\n"
           "                                with no device Vision memory, and caps\n"
           "                                --vision-max-merged at 256 unless given\n"
           "  --vision-cpu                  --vision with --vision-residency cpu\n"
           "  --vision-max-merged N         merged tokens of one media item (default 16384);\n"
           "                                larger media is downscaled\n"
           "\n"
           "LOGGING\n"
           "  --log-level L                 trace, debug, info (default), warning, error,\n"
           "                                critical or off\n"
           "  --log-colours on|off          colour the statistics on stderr (default off)\n"
           "\n"
           "NOTES\n"
           "  The precision options trade speed or memory for quality;\n"
           "  docs/maintainer/quality-trade-experiments.md measures each.\n"
           "  Sampling defaults come from the loaded model and thinking mode; flags override\n"
           "  individual fields.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument(".ninfer model path is required"); }
    options.artifact_path     = argv[1];
    bool kv_capacity_explicit = false;
    bool device_explicit      = false;
    bool ngram_width_explicit       = false;
    bool vision_max_merged_explicit = false;
    std::optional<std::size_t> kv_headroom_mib;
    const char* unsupported_prefix_flag = nullptr;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--chat-template") {
            options.chat_template_path = value(arg);
        } else if (arg == "--messages") {
            options.messages_path = value(arg);
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--kv-headroom-mib" || arg == "--vram-headroom-mib") {
            // --vram-headroom-mib is the Wallawalla47 fork's name for the same headroom.
            const std::uint64_t mib = parse_u64(value(arg), arg.substr(2));
            if (mib > (std::numeric_limits<std::size_t>::max() >> 20)) {
                throw std::invalid_argument(std::string(arg) + " is out of range");
            }
            kv_headroom_mib = static_cast<std::size_t>(mib);
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device  = parse_device(value(arg));
            device_explicit = true;
        } else if (arg == "--devices") {
            options.devices = parse_device_list(value(arg));
        } else if (arg == "--stage-layers") {
            options.stage_layers = parse_stage_layers(value(arg));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens =
                parse_u32(value(arg), "ngram-draft-tokens", true);
            ngram_width_explicit = true;
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = parse_u32(value(arg), "ngram-min-match");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--lm-head-q4") {
            options.lm_head_q4 = true;
        } else if (arg == "--lm-head-q6") {
            options.lm_head_q6 = true;
        } else if (arg == "--embedding-q4") {
            options.embedding_q4 = true;
        } else if (arg == "--embedding-q6") {
            options.embedding_q6 = true;
        } else if (arg == "--mtp-experts-q4") {
            options.mtp_experts_q4 = true;
        } else if (arg == "--gdn-state-fp16") {
            options.gdn_state_fp16 = true;
        } else if (arg == "--rope-yarn") {
            options.rope_yarn = true;
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor = product::parse_rope_yarn_factor(value(arg));
        } else if (arg == "--rope-scaling-factor") {
            options.rope_scaling_factor = product::parse_rope_scaling_factor(value(arg));
        } else if (arg == "--rope-scaling-original-context") {
            options.rope_scaling_original_context =
                product::parse_rope_scaling_original_context(value(arg));
        } else if (arg == "--use-alt-prefix-caching") {
            options.use_alt_prefix_caching = true;
        } else if (arg == "--use-original-prefix-caching" || arg == "--no-prefix-reuse") {
            unsupported_prefix_flag = argv[i];
        } else if (arg == "--cuda-memory-policy") {
            const auto policy = product::parse_cuda_memory_options(value(arg));
            options.cuda_memory_policy = policy.policy;
            options.cuda_vram_reserve_bytes = policy.reserve_bytes;
            options.cuda_memory_probe_step_bytes = policy.probe_step_bytes;
        } else if (arg == "--wddm-evictable-budget") {
            options.wddm_evictable_budget = true;
        } else if (arg == "--mlp-a8-decode") {
            options.mlp_a8_decode = true;
        } else if (arg == "--no-prefill-a8") {
            options.prefill_a8 = false;
        } else if (arg == "--lookup-ngram") {
            options.speculative.lookup_ngram = parse_u32(value("--lookup-ngram"), "lookup-ngram");
        } else if (arg == "--mtp-attention-window") {
            options.speculative.mtp_attention_window =
                parse_u32(value(arg), "mtp-attention-window", true);
        } else if (arg == "--prefill-cublas") {
            options.prefill_cublas = true;
        } else if (arg == "--no-prefill-cublas-projections") {
            options.prefill_cublas_projections = false;
        } else if (arg == "--json" || arg == "--json-schema") {
            if (options.structured_output.kind != StructuredOutputKind::None) {
                throw std::invalid_argument("choose exactly one structured output mode");
            }
            options.structured_output.kind = arg == "--json" ? StructuredOutputKind::JsonObject
                                                             : StructuredOutputKind::JsonSchema;
            if (arg == "--json-schema") {
                std::ifstream schema(value(arg));
                if (!schema) { throw std::invalid_argument("cannot read JSON schema file"); }
                options.structured_output.schema.assign(std::istreambuf_iterator<char>(schema), {});
            }
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--log-colours") {
            const std::string_view mode = value(arg);
            if (mode == "on") {
                options.log_colours = true;
            } else if (mode == "off") {
                options.log_colours = false;
            } else {
                throw std::invalid_argument("--log-colours accepts on or off");
            }
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value(arg), "thinking-budget");
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-residency") {
            const std::string_view mode = value(arg);
            if (mode == "resident") {
                options.vision_residency = VisionResidency::Resident;
            } else if (mode == "overlay") {
                options.vision_residency = VisionResidency::Overlay;
            } else if (mode == "cpu") {
                options.vision_residency = VisionResidency::Cpu;
            } else {
                throw std::invalid_argument("--vision-residency must be resident, overlay or cpu");
            }
        } else if (arg == "--vision-cpu") {
            // The gzenz fork's switch for Vision on CPU threads.
            options.enable_vision    = true;
            options.vision_residency = VisionResidency::Cpu;
        } else if (arg == "--vision-offload") {
            // The Wallawalla47 fork's switch for the same overlay residency.
            const std::string_view mode = value(arg);
            if (mode == "off") {
                options.vision_residency = VisionResidency::Resident;
            } else if (mode == "on") {
                options.vision_residency = VisionResidency::Overlay;
            } else {
                throw std::invalid_argument("--vision-offload must be on or off");
            }
        } else if (arg == "--vision-max-merged") {
            options.vision_max_merged_tokens = parse_u32(value(arg), "vision-max-merged");
            vision_max_merged_explicit       = true;
            if (options.vision_max_merged_tokens < 64 || options.vision_max_merged_tokens > 16384) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 16384]");
            }
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_float(value(arg), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_float(value(arg), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            const std::uint32_t top_k = parse_u32(value(arg), "top-k", true);
            if (top_k > 20) { throw std::invalid_argument("--top-k must be in [0,20]"); }
            options.sampling.top_k = static_cast<std::int32_t>(top_k);
        } else if (arg == "--min-p") {
            options.sampling.min_p = parse_float(value(arg), "min-p", 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            options.sampling.presence_penalty =
                parse_float(value(arg), "presence-penalty", -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            options.sampling.frequency_penalty =
                parse_float(value(arg), "frequency-penalty", -2.0F, 2.0F);
        } else if (arg == "--seed") {
            options.sampling.seed = parse_u64(value(arg), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--post-thinking") {
            if (!options.post_thinking_sampling) { options.post_thinking_sampling.emplace(); }
        } else if (arg == "--post-thinking-temperature" || arg == "--post-thinking-top-p" ||
                   arg == "--post-thinking-top-k") {
            const std::string_view key = arg == "--post-thinking-temperature" ? "temp"
                                         : arg == "--post-thinking-top-p"     ? "top_p"
                                                                              : "top_k";
            if (!options.post_thinking_sampling) { options.post_thinking_sampling.emplace(); }
            product::set_post_thinking_field(*options.post_thinking_sampling, key, value(arg));
        } else if (arg == "--post-thinking-sampler") {
            if (!options.post_thinking_sampling) { options.post_thinking_sampling.emplace(); }
            product::apply_post_thinking_sampler(value(arg), *options.post_thinking_sampling);
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(value(arg));
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (!options.devices.empty() && device_explicit) {
        throw std::invalid_argument("--device and --devices are mutually exclusive");
    }
    if (options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault &&
        (options.wddm_evictable_budget || options.enable_vision || options.devices.size() > 1)) {
        throw std::invalid_argument(
            "--cuda-memory-policy mixed/strict requires one GPU, text-only inference and "
            "ordinary CUDA allocation (no --wddm-evictable-budget)");
    }
    if (unsupported_prefix_flag != nullptr) {
        throw std::invalid_argument(
            std::string(unsupported_prefix_flag) +
            (options.cuda_memory_policy == CudaMemoryPolicy::StrictVram
                 ? " conflicts with --cuda-memory-policy strict, which enables hybrid prefix caching"
                 : " is not supported by ninfer CLI"));
    }
    if (options.cuda_memory_policy == CudaMemoryPolicy::StrictVram) {
        options.use_alt_prefix_caching = true;
    } else {
        options.cuda_vram_reserve_bytes = 0;
        options.cuda_memory_probe_step_bytes = 0;
    }
    if (options.use_alt_prefix_caching && options.devices.size() > 1) {
        throw std::invalid_argument("--use-alt-prefix-caching requires one GPU");
    }
    if (!options.stage_layers.empty() && options.devices.size() < 2) {
        throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
    }
    if (kv_headroom_mib.has_value()) {
        if (options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault) {
            throw std::invalid_argument("--kv-headroom-mib is available only with --cuda-memory-policy default");
        }
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
            throw std::invalid_argument("--kv-headroom-mib requires --kv-capacity auto");
        }
        options.kv_capacity = KvCapacityPolicy::automatic(*kv_headroom_mib << 20);
    } else if (options.kv_capacity.mode == KvCapacityMode::Automatic &&
               options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault) {
        options.kv_capacity = KvCapacityPolicy::automatic(options.cuda_vram_reserve_bytes);
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    product::apply_default_ngram_draft_tokens(options.speculative, ngram_width_explicit);
    product::validate_speculative_cli_options(options.speculative);
    if (options.vision_residency != VisionResidency::Resident && !options.enable_vision) {
        throw std::invalid_argument("--vision-residency overlay or cpu requires --vision");
    }
    // A CPU encode grows with the square of an item's patches: 256 merged tokens (about 512x512
    // pixels) keeps one to a few seconds.
    if (options.vision_residency == VisionResidency::Cpu && !vision_max_merged_explicit) {
        options.vision_max_merged_tokens = 256;
    }
    if (options.structured_output.kind != StructuredOutputKind::None) {
        if (options.raw_output || !options.stop_strings.empty() ||
            !options.stop_token_ids.empty()) {
            throw std::invalid_argument(
                "structured output requires decoded text and default stops");
        }
        if (!options.reasoning_effort && !options.thinking_budget) {
            options.enable_thinking = false;
        }
    }
    if (options.enable_thinking == false && options.reasoning_effort &&
        *options.reasoning_effort != ReasoningEffort::None) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (options.reasoning_effort == ReasoningEffort::None) options.enable_thinking = false;
    if (options.enable_thinking == false && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.greedy) {
        options.sampling.temperature = 0.0F;
        if (options.post_thinking_sampling) { options.post_thinking_sampling->temperature = 0.0F; }
    }
    return options;
}

} // namespace ninfer::cli
