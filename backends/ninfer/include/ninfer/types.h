#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// ContextCacheHints::allow_engine_prefix_grid proposes at most this many shared candidates, at
// multiples of a stride that starts at one grid page and doubles until the count fits.
inline constexpr std::uint32_t kPrefixGridCandidates = 8;
inline constexpr std::uint32_t kPrefixGridPageTokens = 256;
// Explicit markers plus the engine's automatic tool/leading-instruction/full-prompt candidates:
// the shared-prefix opportunities one request can materialize on its own. Prefix-grid points are
// excluded because they carry only observed evidence and need a second reuse domain.
inline constexpr std::size_t kMaximumPreparedPromptCacheCandidatesPerRequest = 7;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes    = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes     = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes      = 2ULL << 30;
inline constexpr std::uint32_t kDefaultHostStateSlots    = 8;
inline constexpr std::size_t kDefaultHostKvCapacityBytes = 8ULL << 30;
// Largest text hidden size among shipped targets (27B). Used only to conservatively size the
// pinned cross-rank staging buffer for a model-parallel `DeviceContext` from the configured
// prefill chunk, before the target's actual hidden size is known; a wider future target just
// widens this.
inline constexpr std::size_t kMaxSupportedResidualHiddenSize = 5120;
// BF16: the residual stream's dtype for every shipped target, independent of KV cache storage.
inline constexpr std::size_t kResidualStreamBytesPerElement = 2;
// Pinned Host tier of the hybrid prefix cache when --host-cache-mib is not given.
inline constexpr std::size_t kDefaultHybridHostCacheBytes = 8ULL << 30;

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane over a group-32
    // scale. Ported onto the kv_cache_append Op that owns KV quantization, and opt-in through
    // --kv-dtype rk8v4. See docs/rtx-3090-windows.md.
    RotatedInt8KeyInt4ValueGroup64,
    // NVFP4 e2m1, group-16 scale, both K and V planes packed two codes per byte.
    Nvfp4Group16,
    // K8V4: FP8 E4M3 key plane (same coding as Fp8E4M3Row256) paired with an NVFP4 value plane.
    Fp8KeyNvfp4Value,
    // rk4v4 (this fork only): rotated keys as 4-bit indices into a Lloyd-Max codebook with a
    // group-64 FP16 scale, paired with rk8v4's packed signed int4 group-32 value plane. Opt-in
    // through --kv-dtype rk4v4.
    RotatedLloyd4KeyInt4Value,
    // rk4v4-e8: rotated keys snapped per 8-dimension block to the E8 lattice and stored as packed
    // signed int4 over a group-64 scale, paired with the rk8v4 packed int4 value plane. Opt-in
    // through --kv-dtype rk4v4-e8. Appended last so the existing values keep their numbering.
    RotatedInt4KeyInt4ValueE8,
    // rk2v4-e8: rotated keys stored per 8-dimension block as the nearest E8 root, a log-radius and
    // a residual axis (two bytes per block) over a group-64 scale, paired with the rk8v4 packed
    // int4 value plane. Opt-in through --kv-dtype rk2v4-e8.
    RotatedE8RootKeyInt4Value,
};

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

// How many of its drafts an MTP round verifies: always all of them, or a width the Engine adapts
// to the drafts' measured survival and the measured round cost. Greedy output does not depend on
// it.
enum class MtpDraftPolicy : std::uint8_t {
    Fixed,
    Adaptive,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: MTP, DFlash and DFlash2 1..15 (query width K+1).
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
    // Context-lookup drafting: match this many trailing tokens against the sequence so far and
    // propose whatever followed the last time they appeared. 0 disables it. It costs no device
    // work, it is exact (verify rejects a wrong guess), and it is strongest exactly where a draft
    // head is weakest -- output that repeats the input. Used as a draft source alongside the
    // configured backend, preferred whenever it finds a match.
    std::uint32_t lookup_ngram = 0;
    // MTP only; Adaptive verifies 1..draft_tokens drafts per round.
    MtpDraftPolicy mtp_policy = MtpDraftPolicy::Fixed;
    // Optional target-verified copy proposals with MTP, DFlash or DFlash2, C1 only.
    // Zero disables the proposer; enabled draft width is 1..63 and minimum match 4..64.
    std::uint32_t ngram_draft_tokens = 0;
    std::uint32_t ngram_min_match    = 12;
    // CPU-only retention, separate from KV. Zero keeps request-local drafting.
    std::size_t ngram_archive_bytes = 0;
    std::size_t ngram_session_bytes = 128ULL << 20;
    // MTP only: the draft head attends to its first 64 keys and the newest this many keys before
    // its query instead of the whole history; target verification keeps full attention, so the
    // committed tokens do not depend on it. Zero reads the whole history.
    std::uint32_t mtp_attention_window = 0;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostStatePin,
    HostKvPin,
    CudaGraphPrepare,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

enum class DiagnosticLevel : std::uint8_t {
    Debug,
    Info,
    Warning,
    Error,
};

// One runtime diagnostic from the Engine: the device route profile it installs, recovery from out
// of memory or a failed worker step, and context stores rebuilt after that recovery.
struct Diagnostic {
    DiagnosticLevel level = DiagnosticLevel::Info;
    std::string message;
};

struct DiagnosticObserver {
    // Receives every diagnostic, from the worker thread or the constructing one; the product
    // decides which levels to show. Without a callback, Info and above go to stderr. Callback
    // exceptions are ignored so a logging failure cannot disturb execution.
    std::function<void(const Diagnostic& diagnostic)> callback;
};

// Prefix-cache implementation selected at Engine construction. Legacy is the owner/checkpoint
// ResourceManager (docs/maintainer/resource-scheduling-and-context-cache.md). Hybrid is the
// content-addressed block tree with sparse state snapshots
// (docs/maintainer/hybrid-prefix-cache-spec.md).
enum class ContextCacheMode : std::uint8_t {
    Legacy,
    Hybrid,
};

// Hybrid-mode tuning. Every field is optional: Engine construction derives the unset ones from
// max_concurrency, prefill_chunk and whether a Host tier exists (host_cache_budget_bytes, default
// kDefaultHybridHostCacheBytes, 0 disables it), and Engine::options() reports the effective
// values. Legacy-only fields of ContextCacheOptions are rejected in Hybrid mode.
struct HybridPrefixCacheOptions {
    // Device StateImage slots holding inactive snapshots (and tap/endpoint staging). Total Device
    // StateImage capacity is max_concurrency + device_snapshot_slots. Default: one per request
    // lane plus one staging slot with a Host tier, plus two without one (Device slots are then
    // the only snapshot storage).
    std::optional<std::uint32_t> device_snapshot_slots;
    // New prefill state snapshots one request may create. Default 8 with a Host tier, 2 without.
    std::optional<std::uint32_t> max_new_taps;
    // Geometric ladder base G: ladder taps target prompt_tokens - G * 2^k. Default
    // max(4096, 2 * prefill_chunk): ladder taps land on prefill chunk boundaries.
    std::optional<std::uint32_t> tap_ladder_tokens;
    // Ladder taps closer than this to another snapshot on the same path are skipped. Default
    // max(1024, prefill_chunk).
    std::optional<std::uint32_t> tap_min_gap_tokens;
    // Opt-in persistence of the Host tier: saved to this file when the Engine shuts down cleanly
    // and restored from it at startup, but only when it was written for the same artifact, KV and
    // state formats and `persistent_identity` (the product binary's build). Empty disables it.
    std::filesystem::path persistent_file;
    std::string persistent_identity;
};

struct ContextCacheOptions {
    // Engine resolves every optional once at construction. With C=max_concurrency, the enabled
    // defaults are H=C, R=8, Host KV=8 GiB, P=2C, S=max(C,7) and L=2 (4 with automatic anchors);
    // Engine::options() returns those effective values.
    bool enabled = true;
    ContextCacheMode mode = ContextCacheMode::Legacy;
    HybridPrefixCacheOptions hybrid;
    // Extra Device checkpoint StateImage slots H. Total Device StateImage capacity is C + H.
    std::optional<std::uint32_t> device_state_slots;
    // Host StateImages and Host KV bytes are independently configured pinned-memory capacities.
    std::uint32_t host_state_slots     = kDefaultHostStateSlots;
    std::size_t host_kv_capacity_bytes = kDefaultHostKvCapacityBytes;
    // Single host RAM ceiling for the whole retention tier. In Hybrid mode it is the pinned Host
    // slab pool that KV blocks and state snapshots share (default kDefaultHybridHostCacheBytes, 0
    // disables the Host tier). In Legacy mode, when engaged it is authoritative:
    // the plan sizes the Host state pool from the checkpoint inventory the capture path creates
    // (2 + anchors per private owner plus one per shared entry), spends the remaining state
    // headroom under the half-budget cap on extra long anchors per owner, gives Host KV the
    // remainder, and rejects a plan whose state footprint exceeds half the budget.
    // `host_state_slots` and `host_kv_capacity_bytes` are ignored in that mode.
    std::optional<std::size_t> host_cache_budget_bytes;
    // Bounded private/shared logical catalogs and per-continuation long-anchor count. An engaged
    // host-cache budget raises the anchor count within the state inventory it funds.
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;
    // Input-complexity bound; this does not reserve checkpoint storage.
    std::optional<std::uint32_t> max_cache_markers_per_request;
    // Disk (L3) tier under the Host tier. An empty path disables it. With a path, an evicted
    // private continuation writes its KV prefix chain and its endpoint, rewrite and early anchor
    // StateImages to per-family files there, keyed by the prefix digest, and they persist across
    // restarts; the Engine keeps them apart per artifact and execution profile.
    std::filesystem::path disk_kv_path;
    std::uint64_t disk_kv_capacity_bytes = 0; // zero with a path selects 64 GiB
    // Read side: a request with no resident prefix that finds [0, E) restorable on disk seeds
    // those KV pages and the StateImage at E and prefills only [E, prompt). Off leaves the tier
    // write-only.
    bool disk_kv_restore = false;
    // Restore reads go through Microsoft DirectStorage (Windows builds with NINFER_DIRECTSTORAGE);
    // any other build refuses it at load. A failed batch falls back to mapped reads.
    bool disk_kv_directstorage = false;
    // Retention for one long-lived conversation whose prompt only grows: a capture that extends a
    // resident the request matched exactly at that resident's frontier inherits the resident's
    // demand, so rolling the frontier forward is not valued against its own history. Off by
    // default: when conversations share a prefix, one conversation's extension can evict the
    // prefix the others depend on.
    bool rolling_retention = false;
    // A private checkpoint of a conversation whose next prompt diverges from it at the checkpoint's
    // frontier keeps no retention value, so it is the first to go when the cache needs room. Off
    // by default: a client that switches back to an earlier branch of the conversation, or another
    // conversation that shares that prefix, could still have reused it.
    bool release_diverged_checkpoints = false;
    // Admission searches for a reuse plan for up to 250 ms at every boundary, and a request whose
    // plan is worth more earns a longer base grant, instead of 50 ms at an idle boundary and 10 ms
    // while other requests run. Off by default: a new request can then pause running decode for
    // up to that long, where a missed large prefix costs its whole re-prefill.
    bool thorough_admission_search = false;
    // Pressure ranks private conversations and shared prefixes in one least-recently-used order
    // (latest hit or publication). A request that does not fit gives up the fewest oldest owners
    // that make it fit and spares any of them it does not need, instead of clearing every owner
    // it cannot reuse; kept owners, shared prefixes included, are demoted to Host where it has
    // room, and incremental search may fully evict only inside that sacrificed tail and only an
    // owner Host cannot take. Off by default: the economic search alone chooses what goes.
    bool recency_eviction = false;
    // Pressure search charges evicting a private conversation its rank by rebuild cost among the
    // request's victims (0 for the cheapest), so among otherwise equal plans the search keeps the
    // conversations most expensive to rebuild, demoted to Host, and evicts the cheapest. Off by
    // default: it reorders the bounded search, which can then settle on a different plan.
    bool value_aware_demote = false;
    // An active request's Device KV entitlement is its prompt plus a bounded window of its output
    // (4096 tokens, or the prefill chunk when larger), extended at decode-round boundaries, rather
    // than prompt plus the whole max_tokens budget. When the pool cannot extend it, idle retained
    // owners are released least recently used first; if the smallest step still does not fit, the
    // answer ends with finish_reason length before max_tokens. Off by default for that reason.
    bool kv_lease_growth = false;
    // The engine anchors message boundaries itself: every request offers private long anchors at
    // up to L message boundaries, on the grid long_anchor_min_spacing_tokens sets, so a later
    // request that rewrites history there resumes from the anchor instead of root. L then defaults
    // to 4. Off by default: each anchor costs a prefill split and a StateImage whether or not the
    // client ever rewrites its history.
    bool automatic_long_anchors = false;
    // Minimum token gap between automatic long anchors, doubling per anchor walking back from the
    // prompt end (gap k >= spacing * 2^k), so the grid is sparse near the end and still reaches
    // deep history. Zero anchors every one of the last L message boundaries.
    std::uint32_t long_anchor_min_spacing_tokens = 1024;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

enum class VisionResidency : std::uint8_t {
    Resident, // fixed Vision GPU allocations for the process lifetime
    Overlay,  // tower host-pinned; each image borrows device memory inside a bounded window
    Cpu,      // tower decoded to FP32 host memory and run on CPU threads; no device Vision memory
};

enum class CudaMemoryPolicy : std::uint8_t {
    DriverDefault,
    StrictVram,
    Mixed,
};

// Optional sparse KV working set. B and R are independent whole-page budgets;
// Host retains native page records, including quantization scale planes.
struct KvmemOptions {
    std::uint32_t selected_tokens = 0; // B; zero disables KVMem
    std::uint32_t reserve_tokens = 0;  // R; also covers each prefill chunk
    std::uint64_t host_bytes = 0;      // H; hard bound on archived native payload
    std::uint32_t retained_sessions = 4; // bounded inactive Host histories, 1..16
    bool verify_transfers = false;    // correctness diagnostics, off in product runs
    // Optional cold snapshots of named inactive text sessions. Temporary writes
    // also count toward disk_bytes. No online KV paging or DirectStorage.
    std::filesystem::path disk_path;
    std::uint64_t disk_bytes = 0;
    [[nodiscard]] std::uint64_t device_tokens() const noexcept {
        return static_cast<std::uint64_t>(selected_tokens) + reserve_tokens;
    }
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    // Message the model receives when it hits its thinking budget, before the canonical
    // </think> close the frontend appends when the message lacks it. Empty preserves the
    // model's built-in end-of-thinking control suffix.
    std::string thinking_budget_message;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    // Empty or one entry keeps the single-device route and `device` selects it. Several entries
    // split the model's layers into that many pipeline stages, one per device, in the given order:
    // the first holds the embedding, head and round state. Matching compute capability is required
    // at construction; peer access is only probed and recorded as a capability -- boundary
    // transfers stage through pinned host memory when it is unavailable.
    std::vector<int> devices;
    // Layers per stage, one count per entry of `devices`. Empty lets the engine choose from each
    // device's free memory.
    std::vector<std::uint32_t> stage_layers;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    // Past the model's native window (max_position_embeddings), up to four times it: positions run
    // unscaled RoPE, or with rope_yarn the whole window takes Qwen's YaRN at factor
    // max_context / native (ops::RopeYarn).
    // Single-device bounded working sets; original RoPE positions stay logical.
    KvmemOptions kvmem;
    bool rope_yarn = false;
    // A fixed YaRN factor in (1,4] for every position of the text and MTP layers, whatever
    // max_context is: Qwen documents one factor per deployment rather than one per window. 1 leaves
    // the choice to rope_yarn.
    float rope_yarn_factor = 1.0F;
    // Linear position interpolation instead of YaRN: for the text and MTP layers a position above
    // rope_scaling_original_context rotates at original + (p - original) / rope_scaling_factor,
    // while positions up to it keep their exact angles. 1 disables it; it excludes rope_yarn and
    // rope_yarn_factor and does not raise the window cap of four times the native window.
    float rope_scaling_factor = 1.0F;
    // The interpolation threshold; 0 is the model's native window.
    std::uint32_t rope_scaling_original_context = 0;
    // Reserves the grammar mask planes that JSON and JSON Schema constrained requests sample
    // through. Off, those requests are refused and a DFlash round carries no grammar stage.
    bool structured_output = false;
    // Admit waiting requests to free lanes while other requests prefill, instead of holding
    // admission until the staged prefill finishes.
    bool concurrent_prefill = false;
    // A broken internal invariant (std::logic_error) in the worker fails the active and
    // materializing requests and keeps serving the queue, as an out-of-memory failure does, instead
    // of failing the Engine. Eight consecutive recoveries without a completed unit still fail it.
    bool recover_invariant_failures = false;
    // Windows builds with NINFER_D3D12_RESIDENCY only: allocate the device arenas from a D3D12
    // heap held resident at maximum priority and budget the runtime against the adapter's
    // dedicated memory less the weights and a 512 MiB desktop floor, not against what is free now,
    // on the grounds that WDDM will evict other allocations. Not for a GPU that drives the
    // desktop, whose allocations are often not evictable.
    bool wddm_evictable_budget         = false;
    // StrictVram is a Windows single-device residency guard and selects the Hybrid prefix cache.
    // Mixed is Windows single-device text generation using ordinary cudaMalloc: it attempts the
    // requested capacity even above reported CUDA free and accepts driver paging to Shared memory.
    // Mixed automatic capacity covers at most max_concurrency full max_context windows, without
    // expanding an idle prefix cache or probing system RAM. Normalization sets its automatic
    // headroom to zero; StrictVram sets it to cuda_vram_reserve_bytes. Only DriverDefault uses
    // the independently configured automatic KV headroom.
    // StrictVram probes may exceed reported free, but only a verified resident layout is admitted.
    CudaMemoryPolicy cuda_memory_policy = CudaMemoryPolicy::DriverDefault;
    std::size_t cuda_vram_reserve_bytes = 64ULL << 20;
    std::size_t cuda_memory_probe_step_bytes = 128ULL << 20;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    // Prefill with the fast INT8-KV prompt-attention kernel and round prefill_chunk down to whole
    // prompt-attention waves. Off keeps the default kernel and the requested chunk.
    bool fast_prefill_kernel           = false;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    // Speed-for-quality trades, opt-in and off by default. Measured in
    // docs/maintainer/quality-trade-experiments.md: lm_head_q4 costs +0.69% perplexity for a
    // C8 decode gain (~3%, real chat prompts); gdn_state_fp16 is free within measurement noise
    // (bit-identical greedy output, unchanged perplexity) for a ~2% C8 gain and a halved
    // per-slot host state image.
    bool lm_head_q4                        = false;
    bool gdn_state_fp16                    = false;
    // Device-memory trades that store a W8 vocabulary matrix at a narrower group-64 width while
    // the weights load (the artifact is unchanged). lm_head_q6 saves ~26% of the output head for
    // +0.01% perplexity; embedding_q4 halves the token embedding with no measurable perplexity
    // change. lm_head_q4 and lm_head_q6 are mutually exclusive.
    bool lm_head_q6                        = false;
    bool embedding_q4                      = false;
    // Q6G64 token embedding: the narrower-model option where Q4 costs measurable perplexity
    // (the 35B-A3B's 2048-wide table). Mutually exclusive with embedding_q4.
    bool embedding_q6                      = false;
    // Qwen3.6-35B-A3B only: the MTP draft layer's routed experts are stored W8 in the artifact;
    // transcode them to the text layers' formats (Q4 gate_up, Q6 down) at load. Drafts are
    // verified exactly, so this can change acceptance but never the output distribution.
    bool mtp_experts_q4                    = false;
    // Integer-activation MLP gate_up at decode and verify widths. Measured 4-6% off that Op from
    // sixteen columns up; see docs/maintainer/quality-trade-experiments.md.
    bool mlp_a8_decode                     = false;
    // Integer-activation routes at full prefill tiles, on by default where a route is registered.
    // Clearing it returns prefill to the A16 routes, which is the only way to measure what the
    // integer routes are worth on a whole request rather than per Op.
    bool prefill_a8                        = true;
    // Hand wide prefill GEMMs to cuBLAS instead of this fork's integer mainloop: the weight is
    // materialised as int8 with one scale per row and the activations quantised per token, which
    // runs about twice as fast (ops/linear_swiglu/q4cublas/w4_cublas_prefill.h) and is a further
    // quality trade on top of prefill_a8 -- hence off by default. Its dequantise pass is
    // weight-sized, so it wants a large prefill_chunk to amortise; the two belong together.
    bool prefill_cublas                    = false;
    // Extends prefill_cublas to the attention and GDN input projections, which hold about a fifth
    // of the linear parameters. Worth +13% prefill for +0.071% perplexity on top of what the route
    // already costs, the GDN half carrying nearly all of both. Separable because that is a
    // different trade from the MLP one and an owner may want only the cheaper half.
    bool prefill_cublas_projections        = true;
    // Largest merged-token count one media item may occupy; larger media is downscaled at
    // preprocessing. Also bounds the overlay window.
    std::uint32_t vision_max_merged_tokens = 16384;
    bool use_cuda_graph                    = true;
    // Explicit total CUDA Graph driver-state allowance in bytes; zero keeps the
    // computed per-profile allowance.
    std::size_t cuda_graph_allowance_bytes = 0;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    // Per-GPU route profile (ops/common/device_route.h). "auto" installs the profile measured for
    // this device -- from device_profile_path, then from the table compiled into the binary -- and
    // calibrates a device that has none once, storing the result at device_profile_path; "off"
    // keeps the compiled routes; "calibrate" measures anew. An empty path selects the user cache
    // ($NINFER_DEVICE_PROFILES, else $XDG_CACHE_HOME or ~/.cache, /ninfer/device-profiles.json).
    std::string device_profile = "auto";
    std::filesystem::path device_profile_path;
    StartupObserver startup_observer;
    DiagnosticObserver diagnostic_observer;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;
    // Applies from the token after a thinking request closes its reasoning block, when the
    // request or the server asks for post-thinking sampling.
    SamplingPreset post_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

enum class StructuredOutputKind : std::uint8_t { None, JsonObject, JsonSchema };

struct StructuredOutputOptions {
    StructuredOutputKind kind = StructuredOutputKind::None;
    std::string schema; // JSON Schema for JsonSchema, otherwise empty
};

inline constexpr std::uint32_t kMaximumFirstTokenTopLogprobs = 20;

struct ExecutionOptions {
    StructuredOutputOptions structured_output;
    SamplingOverrides sampling;
    // Present on a thinking request: from the token after the model closes its reasoning block,
    // sampling switches to these values, omitted ones taken from the post-thinking preset and an
    // omitted seed from the request. Ignored when thinking is off.
    std::optional<SamplingOverrides> post_thinking_sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
    // Report the first generated token's log probability and this many likely alternatives
    // (at most kMaximumFirstTokenTopLogprobs); zero reports nothing. Sampling is unaffected.
    std::uint32_t first_token_top_logprobs = 0;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

struct NgramSessionHints {
    // Explicit local conversation identity, at most 256 bytes. Never inferred from
    // prompt similarity or the KV session key. Empty means request-local drafting.
    std::string key;
    bool reset = false;
    std::string parent;
    std::uint64_t parent_generation = 0;
    // Retention-enabled requests mix the sampling seed with fresh request entropy:
    // prior generated proposals must not reuse the random draws that produced them.
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    NgramSessionHints ngram_session;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Terminal interpretation of model-origin tool-call markup. Parameter schemas guide JSON
// normalization but do not validate the call; only a structure/identity failure can return a
// complete marker region to ordinary content.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen = false;
    // A forced call whose closing tag the model never emitted was closed by the decoder. Only the
    // outer tag is ever supplied, so no argument byte is invented.
    bool forced_call_closed                     = false;
    std::uint32_t structured_call_count         = 0;
    std::uint32_t empty_arguments_omitted       = 0;
    std::uint32_t schema_mismatch_arguments     = 0;
    // Repeated parameters of a call, resolved to their last value.
    std::uint32_t duplicate_parameters_repaired = 0;
    // Why the strict pass rejected the region. With recovered set, the region still produced
    // calls and nothing went out as text.
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;
    bool recovered                              = false;
    // Calls the recovery pass kept although the strict pass would not: an undeclared name, a
    // parameter closed by its function, a call missing only its outer tag.
    std::uint32_t recovered_call_count = 0;
    // A call the recovery pass could not read either was replaced by a call to the reserved error
    // tool, so the client answers with an error and the model gets a turn to retry.
    bool malformed_call_reported = false;
    // Text after the last call; the model is predicting a tool result there, so it is dropped.
    bool trailing_content_dropped = false;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    // Function the caller selected. Its call opener is appended to the generation prompt, so the
    // answer can only continue inside that call. Requires a new assistant turn with thinking off.
    std::string forced_tool_name;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
    EngineObserved     = 1U << 4U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Structural shared candidates the Engine derives from the prompt's own shape: after the
    // leading System/Developer block, after the tool definitions, and at the full prompt frontier.
    // A protocol that carries its own write policy still wants these, because its policy only
    // governs where *its* marker goes; turning them off leaves a request whose only candidate sits
    // at the end of its own prompt, which no differing request can ever match.
    bool allow_engine_automatic_shared_prefixes = true;
    // Propose additional shared candidates on a content-independent token grid so two prompts that
    // merely start alike converge on the same frontier. Grid candidates carry EngineObserved
    // evidence only, so they are never speculatively materialized: a grid frontier is published
    // solely once two distinct reuse domains have both asked for it.
    bool allow_engine_prefix_grid = false;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

// A context-cache store (Device or Host StateImage slots, Host KV extents, Paged KV pages, transfer
// lanes) could not honor a placement the planner had already accounted for. Materialization aborts,
// only the affected request fails (Overloaded), and the Engine keeps serving. Derives from
// std::bad_alloc so existing exhaustion handling keeps working; what() names the exhausted store.
class ContextCacheExhausted final : public std::bad_alloc {
public:
    explicit ContextCacheExhausted(std::string message) : message_(std::move(message)) {}
    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live observations require a Streaming consumer;
// phase timings may also be retained for an Aggregate terminal response.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    // Overlay vision residency: per-request sums over the image windows.
    std::uint32_t overlay_windows      = 0;
    double overlay_window_seconds      = 0.0;
    double overlay_evict_seconds       = 0.0;
    double overlay_restore_seconds     = 0.0;
    std::size_t overlay_evicted_bytes  = 0;
    std::size_t overlay_staged_bytes   = 0;
    // Windows that had to borrow the text weights, which stalls every other lane.
    std::uint32_t overlay_exclusive_windows = 0;
    double decode_seconds      = 0.0;
    // Request wall phases at committed model-state boundaries. Prompt begins when admission and
    // its exact reuse choice are published and ends at the first accepted output token. Generation
    // spans the first through last accepted output token and therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
    // Adaptive MTP: rounds verified at each width 1..draft_window, and width changes.
    bool adaptive                    = false;
    std::uint64_t window_transitions = 0;
    std::vector<std::uint64_t> rounds_per_window;
    std::uint64_t ngram_rounds                  = 0;
    std::uint64_t ngram_drafted_tokens          = 0;
    std::uint64_t ngram_accepted_tokens         = 0;
    std::uint64_t ngram_archive_rounds          = 0;
    std::uint64_t ngram_archive_drafted_tokens  = 0;
    std::uint64_t ngram_archive_accepted_tokens = 0;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
    // The request switched to its post-thinking sampling when its reasoning block closed.
    bool post_thinking_sampling = false;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

// Why bounded pressure planning stopped for the materialization decision committed to one request.
enum class MaterializationStopReason : std::uint8_t {
    NoPressure,
    QueueExhausted,
    TargetBudget,
    ExpansionCapacity,
    TimeBudget,
    InsufficientExpectedGain,
    WorkBudget,
};

[[nodiscard]] inline constexpr const char*
materialization_stop_reason_name(MaterializationStopReason reason) noexcept {
    switch (reason) {
    case MaterializationStopReason::NoPressure:
        return "no_pressure";
    case MaterializationStopReason::QueueExhausted:
        return "queue_exhausted";
    case MaterializationStopReason::TargetBudget:
        return "target_budget";
    case MaterializationStopReason::ExpansionCapacity:
        return "expansion_capacity";
    case MaterializationStopReason::TimeBudget:
        return "time_budget";
    case MaterializationStopReason::InsufficientExpectedGain:
        return "insufficient_expected_gain";
    case MaterializationStopReason::WorkBudget:
        return "work_budget";
    }
    return "no_pressure";
}

enum class MaterializationSearchPhase : std::uint8_t {
    None,
    Setup,
    Construction,
    Assessment,
    Expansion,
    Refinement,
};

[[nodiscard]] inline constexpr const char*
materialization_search_phase_name(MaterializationSearchPhase phase) noexcept {
    switch (phase) {
    case MaterializationSearchPhase::None:
        return "none";
    case MaterializationSearchPhase::Setup:
        return "setup";
    case MaterializationSearchPhase::Construction:
        return "construction";
    case MaterializationSearchPhase::Assessment:
        return "assessment";
    case MaterializationSearchPhase::Expansion:
        return "expansion";
    case MaterializationSearchPhase::Refinement:
        return "refinement";
    }
    return "none";
}

struct MaterializationDiagnostics {
    std::uint64_t predicted_now_ns           = 0;
    std::uint64_t predicted_future_loss_ns   = 0;
    std::uint64_t predicted_total_ns         = 0;
    std::uint32_t targets_evaluated          = 0;
    std::uint64_t projection_work            = 0;
    std::uint64_t planning_elapsed_ns        = 0;
    std::uint64_t search_elapsed_ns          = 0;
    MaterializationStopReason stop_reason    = MaterializationStopReason::NoPressure;
    bool budget_exhausted                    = false;
    std::uint32_t selected_degradation_units = 0;
    bool selected_maximal_fallback           = false;

    std::uint64_t initial_predicted_total_ns = 0;
    std::optional<std::uint64_t> first_improvement_ns;
    std::uint32_t incumbent_improvements         = 0;
    std::uint64_t search_work                    = 0;
    std::uint64_t search_granted_ns              = 0;
    std::uint32_t search_renewals                = 0;
    bool search_discovery_used                   = false;
    std::uint64_t search_overshoot_ns            = 0;
    MaterializationSearchPhase search_stop_phase = MaterializationSearchPhase::None;
    bool search_boundary_limited                 = false;

    // Hybrid prefix cache admission. `cached_prefix_tokens` is the longest prompt prefix held as
    // cached KV blocks whether or not it was reusable: reuse also needs a state snapshot inside
    // it, so a gap to the reused token count is prefix lost to snapshot placement.
    // `restored_host_bytes` is what the admission copied back from the Host tier; the copies
    // overlap the request's first prefill pass, so their time is part of its prefill.
    std::uint32_t cached_prefix_tokens = 0;
    std::uint64_t restored_host_bytes  = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

// A token and its log probability under a raw next-token distribution.
struct TokenLogprob {
    TokenId token = 0;
    float logprob = 0.0F;
};

// The first generated token and the most likely alternatives, under the full distribution at the
// prompt's last position before temperature, penalties or filters.
struct FirstTokenLogprobs {
    TokenLogprob selected;
    std::vector<TokenLogprob> top;
};

struct NgramArchiveStats {
    bool enabled              = false;
    bool bound                = false;
    bool published            = false;
    std::uint64_t generation  = 0;
    std::size_t sources       = 0;
    std::size_t session_bytes = 0;
    std::size_t total_bytes   = 0;
    // Effective seed after request-domain separation; archive state is also needed for replay.
    std::optional<std::uint64_t> sampling_seed;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    NgramArchiveStats ngram_archive;
    ThinkingBudgetStats thinking;
    // Present when ExecutionOptions::first_token_top_logprobs asked for it.
    std::optional<FirstTokenLogprobs> first_token_logprobs;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
    VisionResidency residency             = VisionResidency::Resident;
    std::size_t window_capacity_bytes     = 0; // overlay: device bytes one window borrows
    std::size_t pinned_weight_bytes       = 0; // overlay: host-pinned tower bytes
    std::size_t mirror_bytes              = 0; // overlay: pinned mirror of the borrowable tail
};

// Latest observed process residency, not a permanent physical-pinning guarantee. Intentional
// pinned Host allocations are established before shared_baseline_bytes; device-driven growth
// above that baseline invalidates a strict candidate. Zero counters while disabled are unknown.
struct CudaResidencySummary {
    bool enabled = false;
    bool verified = false;
    std::size_t cuda_free_bytes = 0;
    std::size_t cuda_total_bytes = 0;
    std::size_t dedicated_bytes = 0;
    std::size_t shared_bytes = 0;
    std::size_t shared_baseline_bytes = 0;
    std::size_t device_allocated_bytes = 0;
    std::size_t host_pool_bytes = 0;
    std::size_t host_used_bytes = 0;
    std::size_t verified_reserve_bytes = 0;
};

// Gauges of the current KVMem history (active or retained), not process-lifetime
// counters. Payload H excludes statistics, checkpoint images and native workspace.
struct KvmemMemorySummary {
    bool enabled = false;
    std::uint32_t selected_tokens = 0, reserve_tokens = 0;
    std::uint64_t host_payload_budget_bytes = 0, host_payload_bytes = 0;
    std::uint64_t statistics_bytes = 0, checkpoint_bytes = 0;
    std::uint64_t transfer_staging_bytes = 0;
    std::uint64_t statistics_staging_bytes = 0;
    std::uint32_t mtp_resident_pages = 0;
    std::uint32_t retained_sessions = 0;
    std::uint64_t history_hits = 0, history_misses = 0, history_evictions = 0;
    std::uint64_t disk_hits = 0, disk_writes = 0, disk_errors = 0;
    std::uint64_t active_host_reservation_bytes = 0;
    std::uint32_t evaluated_tokens = 0, checkpoint_tokens = 0, resident_pages = 0;
    std::uint32_t history_swaps = 0;
    std::uint64_t history_spilled_bytes = 0, history_restored_bytes = 0;
};

struct MemorySummary {
    KvmemMemorySummary kvmem;
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // Device memory CUDA Graph preparation actually took at startup (free memory before minus
    // after instantiating, uploading and launching every executable); 0 without CUDA Graphs.
    std::size_t cuda_graph_measured_bytes         = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::uint32_t host_state_capacity_slots       = 0;
    std::uint32_t host_state_occupied_slots       = 0;
    std::size_t host_kv_capacity_bytes            = 0;
    std::size_t host_kv_occupied_bytes            = 0;
    // Host retention-tier unit costs. Every Host StateImage slot pins image_bytes regardless of
    // the prefix depth it resumes, and Host KV pins bytes per page group, so the split of a
    // budget between the two is the depth-versus-positions trade made visible.
    std::size_t host_state_image_bytes   = 0;
    std::size_t host_kv_page_group_bytes = 0;
    // Engaged only when the single host RAM budget mode is active.
    std::size_t host_cache_budget_bytes = 0;
    CudaResidencySummary cuda_residency;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t program_submit_ns       = 0;
    std::uint64_t program_post_ns         = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t device_wait_ns          = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t admission_policy_ns           = 0;
    std::uint64_t context_progress_ns           = 0;
    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t admission_policy_invocations  = 0;
    std::uint64_t context_progress_invocations  = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    KvmemMemorySummary kvmem;
    RuntimeHostWorkStats host_work;
    // Published by the worker so monitoring never takes the execution lock or queries CUDA/PDH.
    CudaResidencySummary cuda_residency;
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;

    // The first kQueueReportCap waiting requests in submission order, with how long each had
    // waited when the snapshot was published; the Engine republishes at least once a second
    // while requests wait.
    struct QueueEntry {
        std::uint64_t request_id = 0;
        double wait_seconds      = 0.0;
    };

    static constexpr std::size_t kQueueReportCap = 16;
    std::array<QueueEntry, kQueueReportCap> queue{};
    std::uint32_t queue_entries             = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;
    // Materializations aborted because a context-cache store rejected the placement.
    std::uint64_t context_cache_exhausted_requests = 0;
    // A capture the Program declined because the offer was not physically feasible. Unlike an
    // abort this is a silent retention loss, so it needs its own counter to be observable.
    std::uint64_t active_captures_skipped = 0;

    std::uint64_t root_selections                    = 0;
    std::uint64_t private_endpoint_selections        = 0;
    std::uint64_t private_turn_closure_selections    = 0;
    std::uint64_t private_response_replay_selections = 0;
    std::uint64_t private_long_anchor_selections     = 0;
    std::uint64_t shared_stable_prefix_selections    = 0;
    std::uint64_t reused_prompt_tokens               = 0;
    std::uint32_t last_selected_frontier_tokens      = 0;
    // Aborted requests whose live state was published as a continuation endpoint.
    std::uint64_t salvaged_continuations = 0;

    std::uint64_t state_moves     = 0;
    std::uint64_t state_forks     = 0;
    std::uint64_t state_restores  = 0;
    std::uint64_t state_d2h_count = 0;
    std::uint64_t state_h2d_count = 0;
    std::uint64_t state_d2d_count = 0;
    std::uint64_t state_d2h_bytes = 0;
    std::uint64_t state_h2d_bytes = 0;
    std::uint64_t state_d2d_bytes = 0;
    double state_d2h_seconds      = 0.0;
    double state_h2d_seconds      = 0.0;
    double state_d2d_seconds      = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages                 = 0;
    std::uint64_t partial_tail_cow_pages               = 0;
    std::uint32_t device_state_occupied_slots          = 0;
    std::uint32_t host_state_occupied_slots            = 0;
    std::uint32_t device_main_kv_occupied_pages        = 0;
    std::uint32_t device_backend_kv_occupied_pages     = 0;
    // Un-written growth reservation held by active requests, split out of the occupied totals
    // above. Occupancy alone cannot distinguish KV that exists from KV a request is merely still
    // entitled to, which is what made context-cache starvation invisible in the request log.
    std::uint32_t device_main_kv_lease_pages           = 0;
    std::uint32_t device_backend_kv_lease_pages        = 0;
    std::size_t host_kv_occupied_bytes                 = 0;
    std::uint64_t pressure_private_owners_degraded     = 0;
    std::uint64_t pressure_private_owners_demoted      = 0;
    std::uint64_t pressure_private_owners_evicted      = 0;
    std::uint64_t pressure_shared_owners_degraded      = 0;
    std::uint64_t pressure_shared_owners_evicted       = 0;
    std::uint64_t pressure_checkpoints_dropped         = 0;
    std::uint64_t pressure_searches                    = 0;
    std::uint64_t pressure_search_budget_exhaustions   = 0;
    std::uint64_t pressure_maximal_fallback_selections = 0;
    std::uint32_t shared_active_references             = 0;
    std::uint64_t historical_fork_hits                 = 0;
    double actual_context_transfer_seconds             = 0.0;

    // Hybrid prefix cache (ContextCacheMode::Hybrid); zero in Legacy mode. Block and snapshot
    // gauges are absolute; the rest are cumulative event counters.
    std::uint32_t hybrid_cached_blocks           = 0; // Device-resident tree blocks
    std::uint32_t hybrid_evictable_blocks        = 0;
    std::uint32_t hybrid_tree_blocks             = 0; // Device or Host
    std::uint32_t hybrid_snapshots               = 0;
    std::uint64_t hybrid_host_capacity_bytes     = 0;
    std::uint64_t hybrid_host_used_bytes         = 0;
    std::uint64_t hybrid_snapshot_hits           = 0;
    std::uint64_t hybrid_reused_tokens           = 0;
    std::uint64_t hybrid_blocks_inserted         = 0;
    std::uint64_t hybrid_blocks_reattached       = 0;
    std::uint64_t hybrid_blocks_duplicate        = 0;
    std::uint64_t hybrid_taps_created            = 0;
    std::uint64_t hybrid_taps_skipped            = 0;
    std::uint64_t hybrid_endpoints_created       = 0;
    std::uint64_t hybrid_host_image_writes       = 0;
    std::uint64_t hybrid_host_block_writes       = 0;
    std::uint64_t hybrid_host_image_restores     = 0;
    std::uint64_t hybrid_host_block_restores     = 0;
    std::uint64_t hybrid_host_write_bytes        = 0;
    std::uint64_t hybrid_host_restore_bytes      = 0;
    std::uint64_t hybrid_evicted_blocks          = 0;
    std::uint64_t hybrid_host_snapshot_evictions = 0;
    std::uint64_t hybrid_host_dead_reclaims      = 0;
    std::uint64_t hybrid_unbacked_node_losses    = 0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

// Static facts about the loaded model, independent of the current request context and memory
// layout. The Engine derives the model identity (model_id) and the dimension facts
// (vocab_size, embedding_size, native_context) from the loaded model, and measures the
// parameters, weight bytes, and weights profile from the artifact's tensor inventory. Serving
// renders these into the OpenAI-compatible /v1/models model object and its llama.cpp-compatible
// `meta` field.
struct ModelMetadata {
    std::string model_id;    // Artifact model name (directory metadata "name").
    std::string weights_id;  // Encoded formats of the artifact tensors (meta ftype).
    std::uint64_t vocab_size     = 0; // Tokenizer token domain (meta n_vocab).
    std::uint64_t embedding_size = 0; // Model embedding width (meta n_embd).
    std::uint64_t native_context = 0; // Model native/training context (meta n_ctx_train).
    std::uint64_t parameters     = 0; // Total logical weight elements (meta n_params).
    std::uint64_t weight_bytes   = 0; // Encoded weight payload bytes (meta size).
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::uint64_t pinned_weight_bytes  = 0; // host-pinned weights (overlay vision tower)
    std::uint64_t overlay_window_bytes = 0; // device bytes one overlay vision window borrows
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;

    // Hybrid prefix cache restored from its persistent file at startup.
    struct PrefixCacheRestore {
        bool attempted = false;
        bool restored  = false;
        // Why nothing was restored (no file yet, incompatible file, I/O error).
        std::string message;
        std::uint64_t blocks    = 0;
        std::uint64_t snapshots = 0;
        std::uint64_t bytes     = 0;
        double seconds          = 0.0;
    } prefix_cache;
};

} // namespace ninfer
