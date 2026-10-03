#include "serve/serve_options.h"
#include "product/cuda_memory_options.h"
#include "product/post_thinking_options.h"
#include "product/rope_yarn_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane. Opt-in.
    if (value == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    // rk4v4: rotated 4-bit Lloyd-Max keys with rk8v4's packed int4 value plane. Opt-in.
    if (value == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    // rk4v4-e8: rotated E8-snapped int4 keys with the rk8v4 value plane. Opt-in.
    if (value == "rk4v4-e8") { return KvCacheStorage::RotatedInt4KeyInt4ValueE8; }
    // rk2v4-e8: rotated keys as E8 root codes, two bytes per eight dimensions. Opt-in.
    if (value == "rk2v4-e8") { return KvCacheStorage::RotatedE8RootKeyInt4Value; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [options]\n"
           "\n"
           "Serves the OpenAI Responses/Chat Completions and Anthropic Messages APIs.\n"
           "  --help, -h                    show this help and exit\n"
           "\n"
           "MODEL & CONTEXT\n"
           "  --kvmem-budget N              selected history B, tokens aligned to 64\n"
           "  --kvmem-gen-reserve N         append workspace R, tokens aligned to 64\n"
           "  --kvmem-host-mib N            Host native KV payload budget H, MiB\n"
           "  --kvmem-sessions N            retained Host histories, 1..16 (default 4)\n"
           "  --kvmem-disk-path PATH        cold text session snapshots (C1, device-profile off)\n"
           "  --kvmem-disk-mib N            hard disk quota including temporary replacements\n"
           "                                KVMem uses B+R device tokens and its own\n"
           "                                verified history reuse; one lane, text,\n"
           "                                BF16/INT8, no speculative decoding\n"
           "  --kvmem-verify-transfers      check restored KV/GDN bytes (diagnostic)\n"
           "  --max-context N               logical context ceiling of each request (default\n"
           "                                8192)\n"
           "  --max-concurrency N           requests decoded together, 1..8 (default 1)\n"
           "  --prefill-chunk N             prefill chunk in tokens, a multiple of 128\n"
           "                                (default 1024)\n"
           "  --default-max-tokens N        output limit of a request that sets none\n"
           "                                (default " +
           std::to_string(kDefaultMaxTokens) +
           "; 0 generates until the\n"
           "                                context runs out)\n"
           "  --model-id ID                 public model name instead of the artifact's\n"
           "                                metadata.name\n"
           "  --chat-template FILE          replace the artifact's chat template at startup\n"
           "  --device N                    CUDA device index (default 0)\n"
           "  --devices A,B,...             one pipeline stage per listed device, 2..8\n"
           "                                (Linux); excludes --device\n"
           "  --stage-layers A,B,...        layers per stage in --devices order (default:\n"
           "                                split by free memory)\n"
           "  --no-cuda-graph               decode without CUDA Graphs (on by default)\n"
           "  --cuda-graph-allowance-mib N  CUDA Graph driver-state allowance taken from the\n"
           "                                KV sizing budget (default 0: computed per\n"
           "                                profile)\n"
           "  --device-profile auto|off|calibrate\n"
           "                                per-GPU route profile: auto uses the stored or\n"
           "                                built-in one and calibrates a device that has\n"
           "                                none; off keeps the compiled routes; calibrate\n"
           "                                measures anew (default auto)\n"
           "  --device-profile-path FILE    profile file instead of the user cache\n"
           "  --context-cost-presets FILE   runtime context-cost presets over the\n"
           "                                compiled-in values\n"
           "\n"
           "PRECISION & KERNELS\n"
           "  --fast-prefill-kernel         prefill an int8 or rk* KV cache with the fast\n"
           "                                prompt-attention kernel (FP16 PV per 64-key\n"
           "                                tile) and round --prefill-chunk down to whole\n"
           "                                attention waves\n"
           "  --prefill-cublas              hand wide prefill GEMMs to cuBLAS: a large\n"
           "                                prefill speedup for a small perplexity cost;\n"
           "                                wants a larger --prefill-chunk\n"
           "  --no-prefill-cublas-projections\n"
           "                                with --prefill-cublas, keep the attention and\n"
           "                                GDN input projections off that route\n"
           "  --no-prefill-a8               prefill every projection on the A16 routes\n"
           "                                instead of the integer-activation ones\n"
           "  --mlp-a8-decode               integer-activation MLP gate_up at decode and\n"
           "                                verify widths\n"
           "  --lm-head-q4 | --lm-head-q6   store the output head as Q4 or Q6 while loading\n"
           "                                (Q4 costs +0.69% perplexity, Q6 +0.01%)\n"
           "  --embedding-q4 | --embedding-q6\n"
           "                                store the token embedding as Q4 or Q6 while\n"
           "                                loading\n"
           "  --mtp-experts-q4              Qwen3.6-35B-A3B: store the MTP layer's routed\n"
           "                                experts in the text layers' formats\n"
           "  --gdn-state-fp16              keep the GDN recurrent state in FP16 (halves\n"
           "                                each host state image)\n"
           "  --rope-yarn                   past the model's native window, apply Qwen's\n"
           "                                YaRN at factor max-context / native instead of\n"
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
           "                                --host-cache-mib independently sizes CPU context\n"
           "                                retention, not device-data system-RAM fallback.\n"
           "                                mixed/strict: Windows single-GPU text, no WDDM mode.\n"
           "  --wddm-evictable-budget       Windows D3D12 builds: budget against dedicated\n"
           "                                memory, holding arenas resident\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto          shared KV capacity in tokens (default\n"
           "                                --max-context, or auto with\n"
           "                                Hybrid/strict); auto follows --cuda-memory-policy\n"
           "  --kv-headroom-mib N           default policy only: auto KV leaves free (default\n"
           "                                " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           "); alias --vram-headroom-mib\n"
           "  --kv-dtype T                  KV storage: bf16 (default), int8, fp8, rk8v4,\n"
           "                                rk4v4, rk4v4-e8, rk2v4-e8, nvfp4 or k8v4\n"
           "\n"
           "CONTEXT CACHE\n"
           "  --no-prefix-reuse             disable compatible-prefix caching (on by\n"
           "                                default); excludes the options below\n"
           "  --device-state-slots N        checkpoint states on the device beyond the\n"
           "                                active lanes (default max-concurrency)\n"
           "  --host-state-slots N          pinned host checkpoint states (default 8)\n"
           "  --host-kv-mib N               pinned host KV in MiB (default 8192)\n"
           "  --host-cache-mib N            one pinned host RAM ceiling instead of the two\n"
           "                                above: host states for every checkpoint the\n"
           "                                capture path creates (at most half), more\n"
           "                                automatic anchors with the headroom, host KV the\n"
           "                                rest; with --use-alt-prefix-caching the pool KV\n"
           "                                blocks and state snapshots share (default 8192,\n"
           "                                0 keeps the cache on the device)\n"
           "  --max-private-continuations N private continuation catalog (default 2x\n"
           "                                max-concurrency)\n"
           "  --max-shared-prefixes N       shared stable-prefix catalog (default\n"
           "                                max(max-concurrency,7))\n"
           "  --max-long-anchors-per-continuation N\n"
           "                                long anchors kept per continuation (default 2, 4\n"
           "                                with --auto-long-anchors)\n"
           "  --auto-long-anchors           anchor up to that many message boundaries of\n"
           "                                every request, so a request that rewrites\n"
           "                                earlier history resumes from the nearest anchor\n"
           "                                instead of root (off)\n"
           "  --long-anchor-spacing N       with --auto-long-anchors, minimum tokens between\n"
           "                                anchors, doubling per anchor back from the\n"
           "                                prompt end (default 1024; 0 anchors every\n"
           "                                boundary)\n"
           "  --max-cache-markers-per-request N\n"
           "                                client cache markers one request may carry\n"
           "                                (default 4)\n"
           "  --auto-prefix-grid            offer shared candidates on a token grid, so\n"
           "                                callers whose prompts start alike share a prefix\n"
           "                                without a client hint; a grid frontier is\n"
           "                                published once two callers ask for it\n"
           "  --derive-session-keys         give a request without a session key one derived\n"
           "                                from its instructions and first user message,\n"
           "                                so the conversation keeps a session lineage\n"
           "                                and live-session retention (default off)\n"
           "  --context-cache-policy default|rolling\n"
           "                                rolling: a capture that extends a resident\n"
           "                                checkpoint inherits its demand, for one\n"
           "                                conversation whose prompt only grows\n"
           "  --release-diverged-checkpoints\n"
           "                                a conversation's private checkpoint that its\n"
           "                                next prompt diverges from keeps no value and\n"
           "                                goes first\n"
           "  --thorough-admission-search   search up to 250 ms for a new request's reuse\n"
           "                                plan even while others decode (otherwise 10 ms),\n"
           "                                longer for a costly request, exploring every\n"
           "                                eligible option\n"
           "  --recency-eviction            under pressure give up the least recently used\n"
           "                                owners first, only as many as needed, and demote\n"
           "                                kept ones to Host\n"
           "  --value-aware-demote          under pressure prefer evicting the conversations\n"
           "                                cheapest to rebuild and demoting the costliest\n"
           "  --kv-lease-growth             reserve a 4096-token output window and extend it\n"
           "                                instead of the whole max_tokens budget; an answer\n"
           "                                the pool cannot extend ends with length\n"
           "  --concurrent-prefill          admit waiting requests to free lanes while other\n"
           "                                requests prefill\n"
           "  --disk-kv-path DIR            disk tier: evicted continuations write their KV\n"
           "                                and states there, per artifact and profile, and\n"
           "                                survive restarts\n"
           "  --disk-kv-gib N               disk tier budget (default 64)\n"
           "  --disk-kv-restore             seed a new request's matching prefix from the\n"
           "                                disk tier\n"
           "  --disk-kv-directstorage       read restores through DirectStorage (Windows\n"
           "                                builds with NINFER_DIRECTSTORAGE)\n"
           "  --use-alt-prefix-caching      the hybrid prefix cache instead of the checkpoint\n"
           "                                catalog: content-addressed KV blocks and sparse\n"
           "                                state snapshots; free VRAM becomes block cache\n"
           "                                and --host-cache-mib sizes the Host tier;\n"
           "                                excludes the catalog options above\n"
           "  --use-original-prefix-caching the checkpoint catalog, which is the default;\n"
           "                                accepted for command lines that name it\n"
           "  --device-snapshot-slots N     hybrid: device state snapshot slots (default\n"
           "                                concurrency + 1; + 2 without a Host tier)\n"
           "  --cache-taps-per-request N    hybrid: new prefill snapshots per request\n"
           "                                (default 8; 2 without a Host tier)\n"
           "  --cache-tap-ladder N          hybrid: ladder base in tokens for history\n"
           "                                snapshots (default max(4096, 2x prefill chunk))\n"
           "  --cache-tap-min-gap N         hybrid: minimum tokens between ladder snapshots\n"
           "                                (default max(1024, prefill chunk))\n"
           "  --prefix-cache-file PATH      hybrid: restore the Host tier from PATH at\n"
           "                                startup when this binary wrote it for the same\n"
           "                                artifact and KV format, and save it there on a\n"
           "                                clean shutdown (default off)\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2     speculative decoding backend\n"
           "  --draft-tokens N              drafts per round, 1..15\n"
           "  --lm-head-draft               draft with the optimized proposal head\n"
           "  --adaptive-mtp                each MTP round verifies 3..--draft-tokens\n"
           "                                drafts, the width the drafts' measured survival\n"
           "                                and round cost favor; greedy output is unchanged\n"
           "  --mtp-attention-window N      the MTP draft head attends to its first 64 keys\n"
           "                                and the newest N before its query, not the\n"
           "                                whole history; verification is unchanged\n"
           "                                (default 0: whole history)\n"
           "  --ngram-draft-tokens N        copy up to N tokens (1..63) per round from\n"
           "                                earlier prompt, tool-result or output text,\n"
           "                                verified alongside --spec; on with 15 whenever\n"
           "                                --spec is set, 0 disables it, above 15 needs\n"
           "                                --max-concurrency 1\n"
           "  --ngram-min-match N           shortest match a copy is drawn from, 4..64\n"
           "                                (default 12)\n"
           "  --ngram-archive-mib N         keep finished requests' copy sources in RAM for\n"
           "                                later requests naming the same\n"
           "                                X-NInfer-Draft-Session (default 0: off)\n"
           "  --ngram-session-mib N         per-session share of that archive (default 128)\n"
           "  --ngram-native-sessions       also recognize the session identities Kilo,\n"
           "                                Codex and Claude send\n"
           "  --lookup-ngram N              context-lookup drafting alongside --spec: the\n"
           "                                last N tokens are matched against the sequence\n"
           "                                so far and what followed is proposed (default 0:\n"
           "                                off)\n"
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
           "  --vision-max-merged N         merged tokens of one media item, 64..16384\n"
           "                                (default 16384); larger media is downscaled\n"
           "  --media-cache-mib N           retained prepared media (default 1024; 0\n"
           "                                disables)\n"
           "  --media-live-mib N            all live prepared media payloads (default 2048)\n"
           "  --media-preprocess-threads N  media preprocessing workers (default 0: auto, at\n"
           "                                most 16)\n"
           "\n"
           "SAMPLING & THINKING\n"
           "  --temperature F               0..2\n"
           "  --top-p F                     0..1\n"
           "  --top-k N                     0..20\n"
           "  --min-p F                     0..1\n"
           "  --presence-penalty F          -2..2\n"
           "  --frequency-penalty F         -2..2\n"
           "  --seed N                      seed of a request that sets none (default: fresh\n"
           "                                per request)\n"
           "  --greedy                      force temperature 0 (exact argmax)\n"
           "  --post-thinking               sample a thinking request's answer with the\n"
           "                                post-thinking preset (temperature 0.2) from the\n"
           "                                token after its reasoning closes (default off;\n"
           "                                a request opts in with a post_thinking object)\n"
           "  --post-thinking-temperature F 0..2; implies --post-thinking\n"
           "  --post-thinking-top-p F       0..1; implies --post-thinking\n"
           "  --post-thinking-top-k N       0..20; implies --post-thinking\n"
           "  --post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F]\n"
           "                                the same fields in one flag\n"
           "  --no-thinking                 disable thinking by default\n"
           "  --preserve-thinking           retain closed-turn assistant reasoning in later\n"
           "                                prompts\n"
           "  --default-thinking-budget N   cap model-origin thinking of thinking requests;\n"
           "                                control tokens count toward the output limit\n"
           "  --thinking-budget-message TEXT\n"
           "                                notice a request gets at its thinking budget;\n"
           "                                the canonical </think> close is appended when\n"
           "                                missing\n"
           "  --default-reasoning-effort E  none, minimal, low, medium, high, xhigh or max\n"
           "                                for requests that set no effort and keep\n"
           "                                thinking on\n"
           "\n"
           "API BEHAVIOR\n"
           "  --structured-output           accept JSON and JSON Schema response formats;\n"
           "                                reserves the grammar masks and adds a grammar\n"
           "                                stage to every DFlash round\n"
           "  --unconstrained-response-format\n"
           "                                without --structured-output, generate such a\n"
           "                                request unconstrained instead of refusing it\n"
           "  --assistant-prefill           continue a Chat Completions request's trailing\n"
           "                                assistant message in place, as /v1/messages\n"
           "                                does; needs thinking disabled\n"
           "  --lenient-assistant-history   accept Responses input whose assistant text or\n"
           "                                reasoning follows function_call Items; it joins\n"
           "                                that turn, rendered before its calls\n"
           "  --first-token-logprobs        accept Chat Completions top_logprobs\n"
           "                                (non-streaming) and report the first token's log\n"
           "                                probability with its alternatives\n"
           "  --usage-chunk-choice          give the streamed usage chunk a zero-delta\n"
           "                                choice, for strict parsers that reject\n"
           "                                choices:[]\n"
           "\n"
           "NETWORK & LIMITS\n"
           "  --host H                      listen address (default 127.0.0.1)\n"
           "  --port N                      listen port (default 8080)\n"
           "  --stats-port N                also serve /health, /stats, /v1/load and /metrics\n"
           "                                on port N with a worker of their own (default off)\n"
           "  --api-key KEY                 require this bearer or x-api-key value (default:\n"
           "                                none)\n"
           "  --cors                        send permissive CORS headers for browser clients\n"
           "  --no-webui                    do not serve a WebUI built in with\n"
           "                                NINFER_WEBUI_DIR on GET /\n"
           "  --webui-mcp-proxy             relay the WebUI's MCP traffic at /cors-proxy;\n"
           "                                http targets only, reaches any host and carries\n"
           "                                no API key, so only on a trusted bind\n"
           "  --max-request-mib N           request body limit, enforced before parsing\n"
           "                                (default 384)\n"
           "  --max-pending-requests N      requests waiting for admission (default 16)\n"
           "  --pending-timeout-ms N        preparation-plus-admission wait (default 600000)\n"
           "  --recover-invariant-failures  on a broken internal invariant, fail the active\n"
           "                                requests and keep serving the queue instead of\n"
           "                                failing the engine\n"
           "  --response-store-max-records N\n"
           "                                retained Responses objects (default 1024)\n"
           "  --response-store-max-mib N    Responses state budget (default 256)\n"
           "\n"
           "LOGGING\n"
           "  --request-log-jsonl FILE      append full-precision server and request records\n"
           "  --request-log-max-mib N       rotate the request log at N MiB (default 0: one\n"
           "                                unbounded file)\n"
           "  --request-log-keep N          rotated request logs kept (default 4)\n"
           "  --log-stats-interval-ms N     throughput report interval (default 5000; 0\n"
           "                                disables)\n"
           "  --log-level L                 trace, debug, info (default), warning, error,\n"
           "                                critical or off\n"
           "  --log-colours on|off          colour the console log's levels and statistics\n"
           "                                (default: levels on a console, statistics off)\n"
           "  --log-stats-panel on|off      pin session statistics beneath the console log on\n"
           "                                an interactive terminal (default off)\n"
           "\n"
           "NOTES\n"
           "  Sampler defaults come from the loaded model and the resolved thinking mode;\n"
           "  server flags and request fields override individual values.\n"
           "  context cache defaults: device-state=max-concurrency, private=2x concurrency,\n"
           "  shared=max(max-concurrency,7), anchors=2 (4 with --auto-long-anchors),\n"
           "  Host state=8 slots, Host KV=8192 MiB.\n";
}

// "1,2,3" selects the ordered devices the model's pipeline stages run on; the first also holds the
// embedding, head and round state. One entry is accepted and is equivalent to --device. The engine
// validates matching compute capability at startup; this only parses the shape.
constexpr std::size_t kMaximumDevices = 8;
std::vector<int> parse_device_list(std::string_view value) {
    std::vector<int> devices;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = value.find(',', start);
        const std::string_view piece =
            value.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                                : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--devices entries must not be empty"); }
        const std::string entry(piece);
        devices.push_back(parse_nonnegative_int(entry.c_str(), "devices"));
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
        const int count = parse_nonnegative_int(entry.c_str(), "stage-layers");
        if (count == 0) { throw std::invalid_argument("--stage-layers counts must be positive"); }
        counts.push_back(static_cast<std::uint32_t>(count));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return counts;
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit  = false;
    bool kv_capacity_explicit         = false;
    bool device_explicit              = false;
    bool context_capacity_explicit    = false;
    bool host_state_slots_explicit    = false;
    bool host_kv_mib_explicit         = false;
    bool host_cache_budget_explicit   = false;
    bool long_anchor_spacing_explicit = false;
    bool ngram_width_explicit         = false;
    bool vision_max_merged_explicit   = false;
    std::optional<std::size_t> kv_headroom_mib;
    // Last flag seen that belongs to only one prefix-cache mode, for the cross-mode error.
    const char* legacy_cache_flag  = nullptr;
    const char* hybrid_option_flag = nullptr;
    bool original_cache_selected   = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        // Keep each dispatch chain below MSVC's nested-block limit.
        bool first_group_handled = true;
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--stats-port") {
            options.stats_port = parse_nonnegative_int(require_value("--stats-port"), "stats-port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--kvmem-budget") {
            options.kvmem.selected_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-budget"), "kvmem-budget"));
        } else if (arg == "--kvmem-gen-reserve") {
            options.kvmem.reserve_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-gen-reserve"), "kvmem-gen-reserve"));
        } else if (arg == "--kvmem-sessions") {
            options.kvmem.retained_sessions = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-sessions"), "kvmem-sessions"));
        } else if (arg == "--kvmem-disk-path") {
            options.kvmem.disk_path = require_value("--kvmem-disk-path");
        } else if (arg == "--kvmem-disk-mib") {
            const auto mib = parse_u64(require_value("--kvmem-disk-mib"), "kvmem-disk-mib");
            if (mib > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20))
                throw std::invalid_argument("--kvmem-disk-mib is out of range");
            options.kvmem.disk_bytes = mib << 20;
        } else if (arg == "--kvmem-host-mib") {
            const auto mib = parse_u64(require_value("--kvmem-host-mib"), "kvmem-host-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--kvmem-host-mib is out of range");
            }
            options.kvmem.host_bytes = static_cast<std::size_t>(mib * (1ULL << 20));
        } else if (arg == "--kvmem-verify-transfers") {
            options.kvmem.verify_transfers = true;
        } else if (arg == "--kv-headroom-mib" || arg == "--vram-headroom-mib") {
            // --vram-headroom-mib is the Wallawalla47 fork's name for the same headroom.
            const std::uint64_t mib = parse_u64(require_value(arg.c_str()), arg.c_str() + 2);
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument(arg + " is out of range");
            }
            kv_headroom_mib = static_cast<std::size_t>(mib);
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--first-token-logprobs") {
            options.first_token_logprobs = true;
        } else if (arg == "--fast-prefill-kernel") {
            options.fast_prefill_kernel = true;
        } else if (arg == "--device-profile") {
            options.device_profile = require_value("--device-profile");
            if (options.device_profile != "auto" && options.device_profile != "off" &&
                options.device_profile != "calibrate") {
                throw std::invalid_argument("--device-profile must be auto, off or calibrate");
            }
        } else if (arg == "--device-profile-path") {
            options.device_profile_path = require_value("--device-profile-path");
            if (options.device_profile_path.empty()) {
                throw std::invalid_argument("--device-profile-path must not be empty");
            }
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--log-colours") {
            const std::string_view value = require_value("--log-colours");
            if (value == "on") {
                options.log_colours = true;
            } else if (value == "off") {
                options.log_colours = false;
            } else {
                throw std::invalid_argument("--log-colours accepts on or off");
            }
        } else if (arg == "--log-stats-panel") {
            const std::string_view value = require_value("--log-stats-panel");
            if (value == "on") {
                options.log_stats_panel = true;
            } else if (value == "off") {
                options.log_stats_panel = false;
            } else {
                throw std::invalid_argument("--log-stats-panel accepts on or off");
            }
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--use-alt-prefix-caching") {
            options.context_cache.mode = ContextCacheMode::Hybrid;
        } else if (arg == "--use-original-prefix-caching") {
            original_cache_selected = true;
        } else if (arg == "--device-snapshot-slots") {
            options.context_cache.hybrid.device_snapshot_slots =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--device-snapshot-slots"), "device-snapshot-slots"));
            hybrid_option_flag = "--device-snapshot-slots";
        } else if (arg == "--cache-taps-per-request") {
            options.context_cache.hybrid.max_new_taps =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--cache-taps-per-request"), "cache-taps-per-request"));
            hybrid_option_flag = "--cache-taps-per-request";
        } else if (arg == "--cache-tap-ladder") {
            options.context_cache.hybrid.tap_ladder_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-ladder"), "cache-tap-ladder"));
            hybrid_option_flag = "--cache-tap-ladder";
        } else if (arg == "--prefix-cache-file") {
            options.context_cache.hybrid.persistent_file = require_value("--prefix-cache-file");
            if (options.context_cache.hybrid.persistent_file.empty()) {
                throw std::invalid_argument("--prefix-cache-file must not be empty");
            }
            hybrid_option_flag = "--prefix-cache-file";
        } else if (arg == "--cache-tap-min-gap") {
            options.context_cache.hybrid.tap_min_gap_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-min-gap"), "cache-tap-min-gap"));
            hybrid_option_flag = "--cache-tap-min-gap";
        } else if (arg == "--device-state-slots") {
            legacy_cache_flag                        = "--device-state-slots";
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-state-slots") {
            legacy_cache_flag                      = "--host-state-slots";
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
            host_state_slots_explicit = true;
        } else if (arg == "--host-kv-mib") {
            legacy_cache_flag       = "--host-kv-mib";
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
            host_kv_mib_explicit                         = true;
        } else if (arg == "--host-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--host-cache-mib"), "host-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-cache-mib is out of range");
            }
            options.context_cache.host_cache_budget_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                     = true;
            host_cache_budget_explicit                    = true;
        } else if (arg == "--disk-kv-path") {
            legacy_cache_flag                  = "--disk-kv-path";
            options.context_cache.disk_kv_path = require_value("--disk-kv-path");
            if (options.context_cache.disk_kv_path.empty()) {
                throw std::invalid_argument("--disk-kv-path must not be empty");
            }
        } else if (arg == "--disk-kv-gib") {
            const std::uint64_t gib = parse_u64(require_value("--disk-kv-gib"), "disk-kv-gib");
            if (gib == 0 || gib > (std::numeric_limits<std::uint64_t>::max() >> 30U)) {
                throw std::invalid_argument("--disk-kv-gib must be positive and in range");
            }
            options.context_cache.disk_kv_capacity_bytes = gib << 30U;
        } else if (arg == "--disk-kv-restore") {
            options.context_cache.disk_kv_restore = true;
        } else if (arg == "--disk-kv-directstorage") {
            options.context_cache.disk_kv_directstorage = true;
        } else if (arg == "--context-cache-policy") {
            legacy_cache_flag        = "--context-cache-policy";
            const std::string policy = require_value("--context-cache-policy");
            if (policy != "default" && policy != "rolling") {
                throw std::invalid_argument("--context-cache-policy must be default or rolling");
            }
            options.context_cache.rolling_retention = policy == "rolling";
            context_capacity_explicit               = true;
        } else if (arg == "--release-diverged-checkpoints") {
            legacy_cache_flag                                  = "--release-diverged-checkpoints";
            options.context_cache.release_diverged_checkpoints = true;
            context_capacity_explicit                          = true;
        } else if (arg == "--concurrent-prefill") {
            options.concurrent_prefill = true;
        } else if (arg == "--recover-invariant-failures") {
            options.recover_invariant_failures = true;
        } else if (arg == "--thorough-admission-search") {
            legacy_cache_flag                               = "--thorough-admission-search";
            options.context_cache.thorough_admission_search = true;
            context_capacity_explicit                       = true;
        } else if (arg == "--recency-eviction") {
            legacy_cache_flag                      = "--recency-eviction";
            options.context_cache.recency_eviction = true;
            context_capacity_explicit              = true;
        } else if (arg == "--value-aware-demote") {
            legacy_cache_flag                        = "--value-aware-demote";
            options.context_cache.value_aware_demote = true;
            context_capacity_explicit                = true;
        } else if (arg == "--kv-lease-growth") {
            options.context_cache.kv_lease_growth = true;
            context_capacity_explicit             = true;
        } else if (arg == "--max-private-continuations") {
            legacy_cache_flag = "--max-private-continuations";
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
        } else if (arg == "--max-shared-prefixes") {
            legacy_cache_flag = "--max-shared-prefixes";
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            legacy_cache_flag = "--max-long-anchors-per-continuation";
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
        } else if (arg == "--auto-long-anchors") {
            legacy_cache_flag                            = "--auto-long-anchors";
            options.context_cache.automatic_long_anchors = true;
            context_capacity_explicit                    = true;
        } else if (arg == "--long-anchor-spacing") {
            legacy_cache_flag                                    = "--long-anchor-spacing";
            options.context_cache.long_anchor_min_spacing_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--long-anchor-spacing"),
                                      "long-anchor-spacing"));
            long_anchor_spacing_explicit = true;
        } else if (arg == "--max-cache-markers-per-request") {
            options.context_cache.max_cache_markers_per_request = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-cache-markers-per-request"),
                                      "max-cache-markers-per-request"));
            context_capacity_explicit = true;
        } else {
            first_group_handled = false;
        }
        if (first_group_handled) { continue; }
        if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--request-log-max-mib") {
            options.request_log_max_mib = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--request-log-max-mib"), "request-log-max-mib"));
        } else if (arg == "--request-log-keep") {
            options.request_log_keep = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--request-log-keep"), "request-log-keep"));
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
            device_explicit = true;
        } else if (arg == "--devices") {
            options.devices = parse_device_list(require_value("--devices"));
        } else if (arg == "--stage-layers") {
            options.stage_layers = parse_stage_layers(require_value("--stage-layers"));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-draft-tokens"), "ngram-draft-tokens"));
            ngram_width_explicit = true;
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-min-match"), "ngram-min-match"));
        } else if (arg == "--ngram-archive-mib" || arg == "--ngram-session-mib") {
            const auto mib = parse_u64(require_value(arg.c_str()), arg.c_str());
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("ngram archive capacity is out of range");
            }
            auto& bytes = arg == "--ngram-archive-mib" ? options.speculative.ngram_archive_bytes
                                                       : options.speculative.ngram_session_bytes;
            bytes       = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--ngram-native-sessions") {
            options.ngram_native_sessions = true;
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            if (options.default_max_tokens == 0) {
                options.default_max_tokens = kUnboundedOutputTokens;
            }
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--default-reasoning-effort") {
            const std::string value = require_value("--default-reasoning-effort");
            const auto effort       = parse_requested_reasoning_effort(value);
            if (!effort) {
                throw std::invalid_argument("--default-reasoning-effort must be none, minimal, low, "
                                            "medium, high, xhigh, or max");
            }
            options.default_reasoning_effort = *effort;
        } else if (arg == "--thinking-budget-message") {
            options.thinking_budget_message = require_value("--thinking-budget-message");
            if (options.thinking_budget_message.empty()) {
                throw std::invalid_argument("--thinking-budget-message must not be empty");
            }
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-residency") {
            const std::string_view mode = require_value("--vision-residency");
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
            const std::string_view mode = require_value("--vision-offload");
            if (mode == "off") {
                options.vision_residency = VisionResidency::Resident;
            } else if (mode == "on") {
                options.vision_residency = VisionResidency::Overlay;
            } else {
                throw std::invalid_argument("--vision-offload must be on or off");
            }
        } else if (arg == "--vision-max-merged") {
            const std::uint64_t merged =
                parse_u64(require_value("--vision-max-merged"), "vision-max-merged");
            if (merged < 64 || merged > 16384) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 16384]");
            }
            options.vision_max_merged_tokens = static_cast<std::uint32_t>(merged);
            vision_max_merged_explicit       = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--cuda-graph-allowance-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--cuda-graph-allowance-mib"), "cuda-graph-allowance-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--cuda-graph-allowance-mib is out of range");
            }
            options.cuda_graph_allowance_mib = mib;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--lenient-assistant-history") {
            options.lenient_assistant_history = true;
        } else if (arg == "--derive-session-keys") {
            legacy_cache_flag           = "--derive-session-keys";
            options.derive_session_keys = true;
        } else if (arg == "--auto-prefix-grid") {
            legacy_cache_flag        = "--auto-prefix-grid";
            options.auto_prefix_grid = true;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--adaptive-mtp") {
            options.speculative.mtp_policy = MtpDraftPolicy::Adaptive;
        } else if (arg == "--mtp-attention-window") {
            options.speculative.mtp_attention_window =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--mtp-attention-window"), "mtp-attention-window"));
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
            options.rope_yarn_factor =
                product::parse_rope_yarn_factor(require_value("--rope-yarn-factor"));
        } else if (arg == "--rope-scaling-factor") {
            options.rope_scaling_factor =
                product::parse_rope_scaling_factor(require_value("--rope-scaling-factor"));
        } else if (arg == "--rope-scaling-original-context") {
            options.rope_scaling_original_context = product::parse_rope_scaling_original_context(
                require_value("--rope-scaling-original-context"));
        } else if (arg == "--cuda-memory-policy") {
            const auto policy = product::parse_cuda_memory_options(require_value(arg.c_str()));
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
            options.speculative.lookup_ngram = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--lookup-ngram"), "lookup-ngram"));
        } else if (arg == "--prefill-cublas") {
            options.prefill_cublas = true;
        } else if (arg == "--no-prefill-cublas-projections") {
            options.prefill_cublas_projections = false;
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
            if (options.chat_template_path.empty()) {
                throw std::invalid_argument("--chat-template must not be empty");
            }
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--no-webui") {
            options.enable_webui = false;
        } else if (arg == "--webui-mcp-proxy") {
            options.webui_mcp_proxy = true;
        } else if (arg == "--structured-output") {
            options.structured_output = true;
        } else if (arg == "--unconstrained-response-format") {
            options.unconstrained_response_format = true;
        } else if (arg == "--assistant-prefill") {
            options.assistant_prefill = true;
        } else if (arg == "--usage-chunk-choice") {
            options.usage_chunk_choice = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--post-thinking") {
            if (!options.post_thinking_overrides) { options.post_thinking_overrides.emplace(); }
        } else if (arg == "--post-thinking-temperature" || arg == "--post-thinking-top-p" ||
                   arg == "--post-thinking-top-k") {
            const std::string_view key = arg == "--post-thinking-temperature" ? "temp"
                                         : arg == "--post-thinking-top-p"     ? "top_p"
                                                                              : "top_k";
            if (!options.post_thinking_overrides) { options.post_thinking_overrides.emplace(); }
            product::set_post_thinking_field(*options.post_thinking_overrides, key,
                                             require_value(arg.c_str()));
        } else if (arg == "--post-thinking-sampler") {
            if (!options.post_thinking_overrides) { options.post_thinking_overrides.emplace(); }
            product::apply_post_thinking_sampler(require_value("--post-thinking-sampler"),
                                                 *options.post_thinking_overrides);
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    // Resolve policy-implied cache mode before choosing the KV default or validating cache flags.
    if (options.cuda_memory_policy == CudaMemoryPolicy::StrictVram) {
        if (original_cache_selected || !options.allow_prefix_reuse) {
            throw std::invalid_argument(
                "--cuda-memory-policy strict enables hybrid prefix caching and conflicts with "
                "--use-original-prefix-caching and --no-prefix-reuse");
        }
        options.context_cache.mode = ContextCacheMode::Hybrid;
    } else {
        options.cuda_vram_reserve_bytes = 0;
        options.cuda_memory_probe_step_bytes = 0;
    }
    if (options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault &&
        (options.wddm_evictable_budget || options.enable_vision || options.devices.size() > 1)) {
        throw std::invalid_argument(
            "--cuda-memory-policy mixed/strict requires one GPU, text-only inference and "
            "ordinary CUDA allocation (no --wddm-evictable-budget)");
    }
    if (!kv_capacity_explicit) {
        // Hybrid defaults to automatic sizing. The memory policy decides whether runtime may
        // grow beyond active-context capacity; mixed keeps that bound instead of filling VRAM.
        options.kv_capacity = options.context_cache.mode == ContextCacheMode::Hybrid
                                  ? KvCapacityPolicy::automatic()
                                  : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (original_cache_selected && options.context_cache.mode == ContextCacheMode::Hybrid) {
        throw std::invalid_argument(
            "--use-original-prefix-caching and --use-alt-prefix-caching select different prefix "
            "caching systems");
    }
    if (options.context_cache.mode == ContextCacheMode::Hybrid) {
        if (legacy_cache_flag != nullptr) {
            throw std::invalid_argument(std::string(legacy_cache_flag) +
                                        " configures the default prefix cache and cannot be "
                                        "combined with --use-alt-prefix-caching");
        }
        if (!options.allow_prefix_reuse) {
            throw std::invalid_argument(
                "--use-alt-prefix-caching cannot be combined with --no-prefix-reuse");
        }
        if (options.devices.size() > 1) {
            throw std::invalid_argument(
                "--use-alt-prefix-caching runs on one device and cannot be combined with "
                "pipeline --devices");
        }
        std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        if (!file.empty()) {
            if (host_cache_budget_explicit && options.context_cache.host_cache_budget_bytes == 0) {
                throw std::invalid_argument(
                    "--prefix-cache-file saves the Host tier, which --host-cache-mib 0 removes");
            }
            // Resolved now, so the save at shutdown writes where startup read, and checked now,
            // so an unusable location fails at launch rather than after a session of caching.
            file = std::filesystem::absolute(file).lexically_normal();
            std::error_code error;
            if (std::filesystem::is_directory(file, error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            " is a directory; name a file in it");
            }
            if (!std::filesystem::is_directory(file.parent_path(), error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            ": the directory " + file.parent_path().string() +
                                            " does not exist");
            }
        }
    } else if (hybrid_option_flag != nullptr) {
        throw std::invalid_argument(std::string(hybrid_option_flag) +
                                    " requires --use-alt-prefix-caching");
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
    if (long_anchor_spacing_explicit && !options.context_cache.automatic_long_anchors) {
        throw std::invalid_argument("--long-anchor-spacing requires --auto-long-anchors");
    }
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        if (options.auto_prefix_grid) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --auto-prefix-grid");
        }
        if (options.derive_session_keys) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --derive-session-keys");
        }
        if (!options.context_cache.disk_kv_path.empty()) {
            throw std::invalid_argument("--no-prefix-reuse cannot be combined with --disk-kv-path");
        }
        options.context_cache.enabled                = false;
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    }
    if (options.context_cache.disk_kv_path.empty() &&
        (options.context_cache.disk_kv_restore || options.context_cache.disk_kv_directstorage ||
         options.context_cache.disk_kv_capacity_bytes != 0)) {
        throw std::invalid_argument(
            "--disk-kv-restore, --disk-kv-directstorage and --disk-kv-gib need --disk-kv-path");
    }
    if (!options.devices.empty() && device_explicit) {
        throw std::invalid_argument("--device and --devices are mutually exclusive");
    }
    if (!options.stage_layers.empty() && options.devices.size() < 2) {
        throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
    }
    if (options.request_log_max_mib != 0 && options.request_log_jsonl.empty()) {
        throw std::invalid_argument("--request-log-max-mib requires --request-log-jsonl");
    }
    if (host_cache_budget_explicit) {
        // The budget is the one host RAM ceiling; the two component flags would silently
        // fight it, and their independent-allocation semantics are exactly what the budget
        // exists to replace.
        if (host_state_slots_explicit || host_kv_mib_explicit) {
            throw std::invalid_argument(
                "--host-cache-mib cannot be combined with --host-state-slots or --host-kv-mib: "
                "the budget derives both Host state slots and Host KV bytes");
        }
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.stats_port != 0 &&
        (options.stats_port > 65535 || options.stats_port == options.port)) {
        throw std::invalid_argument("--stats-port must be in [1,65535] and differ from --port");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (options.unconstrained_response_format && options.structured_output) {
        throw std::invalid_argument(
            "--unconstrained-response-format conflicts with --structured-output");
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
    if (options.enable_thinking == false && options.default_reasoning_effort &&
        *options.default_reasoning_effort != RequestedReasoningEffort::None) {
        throw std::invalid_argument("--default-reasoning-effort conflicts with --no-thinking");
    }
    if (options.cuda_graph_allowance_mib != 0 && !options.use_cuda_graph) {
        throw std::invalid_argument(
            "--cuda-graph-allowance-mib requires CUDA graphs (omit --no-cuda-graph)");
    }
    // The GDN conv-record workspace admits at most 16 verification columns when the batch holds
    // more than one request, so a wider ngram proposal is admitted only for one active request.
    if (options.speculative.ngram_draft_tokens > 15 && options.max_concurrency != 1) {
        throw std::invalid_argument("--ngram-draft-tokens above 15 requires --max-concurrency 1");
    }
    if (options.ngram_native_sessions && options.speculative.ngram_archive_bytes == 0) {
        throw std::invalid_argument("--ngram-native-sessions requires --ngram-archive-mib");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    if (options.kvmem.selected_tokens) {
        if (options.kvmem.disk_path.empty() != (options.kvmem.disk_bytes == 0) ||
            (!options.kvmem.disk_path.empty() &&
             (options.max_concurrency != 1 || options.enable_vision || options.device_profile != "off"))) {
            throw std::invalid_argument("KVMem cold snapshots require a path, nonzero disk quota, one text lane and --device-profile off");
        }
        const auto window = options.kvmem.device_tokens();
        const bool ordinary = options.speculative.backend == SpeculativeBackend::None &&
            options.speculative.draft_tokens == 0;
        const bool fixed_mtp = options.speculative.backend == SpeculativeBackend::Mtp &&
            (options.speculative.draft_tokens >= 1 && options.speculative.draft_tokens <= 4) &&
            options.speculative.mtp_policy == MtpDraftPolicy::Fixed;
        const bool memory_ngram_supported = options.speculative.ngram_draft_tokens == 0 ||
            (fixed_mtp && options.max_concurrency == 1 && !options.enable_vision &&
             options.speculative.ngram_draft_tokens <= 63 &&
             options.speculative.ngram_min_match >= 4 && options.speculative.ngram_min_match <= 64 &&
             std::max(options.speculative.draft_tokens, options.speculative.ngram_draft_tokens) +
                 options.speculative.draft_tokens <= options.kvmem.reserve_tokens);
        const bool memory_spec_supported = (ordinary || fixed_mtp) && memory_ngram_supported &&
            options.speculative.lookup_ngram == 0 && options.speculative.mtp_attention_window == 0;

        if (legacy_cache_flag || hybrid_option_flag || original_cache_selected ||
            options.context_cache.mode != ContextCacheMode::Legacy ||
            options.devices.size() > 1 || options.max_concurrency == 0 || options.max_concurrency > 4 ||
            (options.enable_vision && (options.max_concurrency != 1 ||
                options.vision_residency != VisionResidency::Resident)) ||
            !memory_spec_supported || options.concurrent_prefill ||
            options.fast_prefill_kernel || options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault ||
            (options.kv_cache != KvCacheStorage::BFloat16 && options.kv_cache != KvCacheStorage::Int8Group64 &&
             options.kv_cache != KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) ||
            (options.kv_cache == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 &&
             (options.max_concurrency != 1 || options.enable_vision))) {
            throw std::invalid_argument("KVMem requires ordinary or fixed MTP1..4 single-GPU generation, BF16/INT8 or single-lane text RK8V4, images on one resident-vision lane, "
                "and exclusive ownership of history caching");
        }
        if (options.kvmem.selected_tokens < 128 || options.kvmem.selected_tokens % 64 ||
            options.kvmem.reserve_tokens < 64 || options.kvmem.reserve_tokens % 64 ||
            window < 256 || window > options.max_context || !options.kvmem.host_bytes ||
            options.kvmem.retained_sessions == 0 || options.kvmem.retained_sessions > 16 ||
            options.prefill_chunk > options.kvmem.reserve_tokens ||
            (kv_capacity_explicit && (options.kv_capacity.mode != KvCapacityMode::Explicit ||
                                     options.kv_capacity.explicit_tokens != window * options.max_concurrency))) {
            throw std::invalid_argument("KVMem requires aligned B/R, nonzero H, prefill <= R, "
                                        "and device capacity B+R <= max-context");
        }
        if (window * options.max_concurrency > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("KVMem global device capacity exceeds uint32");
        }
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(window * options.max_concurrency));
        // --no-prefix-reuse still controls request reuse. Only the native catalog
        // is disabled here; KVMem owns the current session's trusted checkpoint.
        options.context_cache.enabled = false;
    } else if (options.kvmem.reserve_tokens || options.kvmem.host_bytes || options.kvmem.verify_transfers ||
         !options.kvmem.disk_path.empty() || options.kvmem.disk_bytes) {
        throw std::invalid_argument("KVMem auxiliary options require --kvmem-budget");
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

} // namespace ninfer::serve
