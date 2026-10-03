#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"
#include "runtime/engine/model_instance.h"

#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// Oracle for the Host RAM budget split. Every expectation below is computed by hand from the
// resolver's contract: one StateImage per retained position (2 + A per private owner plus one per
// shared entry), a state footprint capped at half the budget, extra anchors bought with the
// headroom under that cap at one image per private owner each, and a usefulness ceiling of
// pages(capacity) / pages(buyback tokens) - 2 where one image buys back
// state_image_bytes / host_kv_group_bytes page groups of re-prefill.
//
// The unit costs are round numbers on purpose: one image is 1 MiB and one Main page group is
// 256 KiB, so the image is exactly four page groups, buyback tokens are 4 * 64 = 256, and one page
// group is 4 logical pages. Nothing here mirrors the implementation's expression; the values are
// the arithmetic a reader can repeat.
constexpr std::uint64_t kImageBytes = 1ULL << 20; // 1 MiB per Host StateImage.
constexpr std::uint64_t kGroupBytes = 1ULL << 18; // 256 KiB per Main Host KV page group.
constexpr std::uint64_t kMiB        = 1ULL << 20;

ninfer::ContextCacheOptions budgeted(std::size_t budget_mib, std::uint32_t private_capacity,
                                     std::uint32_t shared_capacity, std::uint32_t configured,
                                     bool automatic_anchors = true) {
    ninfer::ContextCacheOptions cache;
    cache.automatic_long_anchors            = automatic_anchors;
    cache.host_cache_budget_bytes           = static_cast<std::size_t>(budget_mib * kMiB);
    cache.max_private_continuations         = private_capacity;
    cache.max_shared_prefixes               = shared_capacity;
    cache.max_long_anchors_per_continuation = configured;
    return cache;
}

void resolve(ninfer::ContextCacheOptions& cache, std::uint32_t capacity) {
    ninfer::models::qwen3_5::detail::resolve_host_cache_budget(
        cache, *cache.max_private_continuations, *cache.max_shared_prefixes, capacity, kImageBytes,
        kGroupBytes);
}

int check_split(const ninfer::ContextCacheOptions& cache, std::uint32_t anchors,
                std::uint32_t state_slots, std::uint64_t host_kv_mib, const char* message) {
    const std::uint64_t host_kv = cache.host_kv_capacity_bytes;
    if (*cache.max_long_anchors_per_continuation == anchors &&
        cache.host_state_slots == state_slots && host_kv == host_kv_mib * kMiB) {
        return 0;
    }
    std::cerr << message << ": anchors " << *cache.max_long_anchors_per_continuation << " (want "
              << anchors << "), state slots " << cache.host_state_slots << " (want " << state_slots
              << "), host KV " << host_kv << " B (want " << host_kv_mib << " MiB)\n";
    return 1;
}

} // namespace

int main() {
    using ninfer::EngineOptions;
    using ninfer::kMaximumPreparedPromptCacheCandidatesPerRequest;
    using ninfer::runtime::normalize_engine_options;

    int failures = 0;

#ifdef _WIN32
    // Strict selects Hybrid before deriving snapshot/tap capacities. Host cache remains a
    // separate choice, including zero, and strict's own reserve defines automatic headroom.
    for (const std::size_t host_bytes : {0ULL, 6ULL << 30}) {
        EngineOptions options;
        options.cuda_memory_policy = ninfer::CudaMemoryPolicy::StrictVram;
        options.kv_capacity = ninfer::KvCapacityPolicy::automatic();
        options.context_cache.host_cache_budget_bytes = host_bytes;
        const auto out = normalize_engine_options(options);
        failures += check(out.context_cache.enabled &&
                              out.context_cache.mode == ninfer::ContextCacheMode::Hybrid &&
                              out.context_cache.host_cache_budget_bytes == host_bytes &&
                              out.context_cache.hybrid.device_snapshot_slots == (host_bytes == 0 ? 3U : 2U) &&
                              out.kv_capacity.automatic_headroom_bytes == (64ULL << 20),
                          "strict did not select Hybrid while preserving Host cache and 64 MiB reserve");
    }
    {
        EngineOptions options;
        options.cuda_memory_policy = ninfer::CudaMemoryPolicy::StrictVram;
        options.cuda_vram_reserve_bytes = 96ULL << 20;
        options.kv_capacity = ninfer::KvCapacityPolicy::automatic();
        const auto out = normalize_engine_options(options);
        failures += check(out.kv_capacity.automatic_headroom_bytes == (96ULL << 20),
                          "strict custom reserve did not replace default KV headroom");
    }
    {
        EngineOptions options;
        options.cuda_memory_policy = ninfer::CudaMemoryPolicy::Mixed;
        options.kv_capacity = ninfer::KvCapacityPolicy::automatic();
        options.cuda_memory_probe_step_bytes = 0; // Mixed must not use strict-only knobs.
        options.cuda_vram_reserve_bytes = 123ULL << 20;
        options.context_cache.host_cache_budget_bytes = 3ULL << 30;
        const auto out = normalize_engine_options(options);
        failures += check(out.kv_capacity.automatic_headroom_bytes == 0 &&
                              out.context_cache.host_cache_budget_bytes == (3ULL << 30) &&
                              out.context_cache.mode == ninfer::ContextCacheMode::Legacy,
                          "mixed changed Host cache, forced Hybrid, or retained automatic headroom");
    }
    for (int variant = 0; variant < 5; ++variant) {
        EngineOptions options;
        options.cuda_memory_policy = variant == 4 ? ninfer::CudaMemoryPolicy::StrictVram
                                                  : ninfer::CudaMemoryPolicy::Mixed;
        if (variant == 0) { options.enable_vision = true; }
        if (variant == 1) { options.devices = {0, 1}; }
        if (variant == 2) { options.wddm_evictable_budget = true; }
        if (variant == 3) { options.purpose = ninfer::EnginePurpose::CausalScoring; }
        if (variant == 4) { options.context_cache.enabled = false; }
        bool rejected = false;
        try { (void)normalize_engine_options(options); }
        catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "memory policy accepted an unsupported deployment or disabled strict cache");
    }
#else
    for (const auto mode : {ninfer::CudaMemoryPolicy::Mixed, ninfer::CudaMemoryPolicy::StrictVram}) {
        EngineOptions options;
        options.cuda_memory_policy = mode;
        bool rejected = false;
        try { (void)normalize_engine_options(options); }
        catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "Windows memory policy normalized on another platform");
    }
#endif
    {
        EngineOptions options;
        options.kv_capacity = ninfer::KvCapacityPolicy::automatic(1234);
        const auto out = normalize_engine_options(options);
        failures += check(out.cuda_memory_policy == ninfer::CudaMemoryPolicy::DriverDefault &&
                              out.context_cache.mode == ninfer::ContextCacheMode::Legacy &&
                              out.kv_capacity.automatic_headroom_bytes == 1234,
                          "default memory policy normalization changed");
    }

    // A single request can materialize up to kMaximumPreparedPromptCacheCandidatesPerRequest
    // distinct shared-prefix candidates on its own: four explicit markers plus the engine's
    // tool/leading-instruction/full-prompt automatic candidates. The default
    // Engine-wide shared catalog must be able to hold at least one request's own candidates even
    // at the smallest concurrency, or ordinary DefaultAutomatic-evidence traffic starts losing
    // cache hits to its own prior turns as soon as the catalog fills.
    for (const std::uint32_t concurrency : {1U, 2U, 8U}) {
        EngineOptions options;
        options.max_concurrency = concurrency;
        const EngineOptions normalized = normalize_engine_options(options);
        const std::uint32_t default_shared_prefixes = *normalized.context_cache.max_shared_prefixes;
        failures += check(
            default_shared_prefixes >=
                static_cast<std::uint32_t>(kMaximumPreparedPromptCacheCandidatesPerRequest),
            "default shared-prefix catalog capacity is smaller than one request's own candidate ceiling");
        failures += check(default_shared_prefixes >= concurrency,
                          "default shared-prefix catalog capacity did not cover active concurrency");
    }

    // An explicit override is still respected verbatim, including a deliberately small value.
    {
        EngineOptions options;
        options.max_concurrency               = 1;
        options.context_cache.max_shared_prefixes = 1;
        const EngineOptions normalized = normalize_engine_options(options);
        failures += check(*normalized.context_cache.max_shared_prefixes == 1,
                          "explicit max_shared_prefixes override was not preserved");
    }

    // A disabled context cache still normalizes to a root-only zero capacity.
    {
        EngineOptions options;
        options.max_concurrency        = 1;
        options.context_cache.enabled  = false;
        const EngineOptions normalized = normalize_engine_options(options);
        failures += check(*normalized.context_cache.max_shared_prefixes == 0,
                          "disabled context cache did not normalize shared-prefix capacity to zero");
    }

    // A comfortable budget buys anchors. 64 MiB halves to a 32 MiB state cap; the mandatory
    // 2 + 4 per owner plus 7 shared is 19 images = 19 MiB, leaving 13 images of headroom = 6 extra
    // anchors per owner, so A = 4 + 6 = 10 and the pool holds (2 + 10) * 2 + 7 = 31 images. The
    // usefulness ceiling at 32768 tokens is 512 pages / 4 pages per 256 tokens - 2 = 126, far
    // above the budget's 10, so the budget binds.
    {
        ninfer::ContextCacheOptions cache = budgeted(64, 2, 7, 4);
        resolve(cache, 32768);
        failures += check_split(cache, 10, 31, 33, "comfortable budget did not buy anchors");
    }

    // The half-budget cap is respected exactly when no shared entry competes with the per-owner
    // images: 64 MiB / 2 is exactly (2 + 14) * 2 images of 1 MiB.
    {
        ninfer::ContextCacheOptions cache = budgeted(64, 2, 0, 4);
        resolve(cache, 32768);
        failures += check_split(cache, 14, 32, 32,
                                "budget-funded anchors did not fill the state cap exactly");
    }

    // A budget that exactly covers the mandatory inventory buys nothing extra and still starts.
    {
        ninfer::ContextCacheOptions cache = budgeted(38, 2, 7, 4);
        resolve(cache, 32768);
        failures += check_split(cache, 4, 19, 19,
                                "boundary budget did not keep the configured anchor count");
    }

    // A budget below the mandatory inventory falls back to the configured count and rejects, so
    // the caller reports the real shortfall instead of an inventory the budget never funded.
    {
        bool rejected                     = false;
        ninfer::ContextCacheOptions cache = budgeted(37, 2, 7, 4);
        try {
            resolve(cache, 32768);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an undersized budget did not reject the checkpoint inventory");
    }

    // An explicitly disabled anchor count stays disabled: the frontend anchors nothing, so extra
    // anchors would only pin images nothing creates. The 11 images of endpoint, rewrite and shared
    // inventory fit the 12 MiB half-budget and Host KV takes the rest.
    {
        ninfer::ContextCacheOptions cache = budgeted(64, 2, 7, 0);
        resolve(cache, 32768);
        failures += check_split(cache, 0, 11, 53, "an explicitly disabled anchor count was grown");
    }

    // Without automatic anchoring only client markers create anchors, so the budget keeps the
    // configured count: (2 + 2) * 2 + 7 = 15 images, Host KV the other 49 MiB.
    {
        ninfer::ContextCacheOptions cache = budgeted(64, 2, 7, 2, false);
        resolve(cache, 32768);
        failures +=
            check_split(cache, 2, 15, 49, "the budget grew anchors that no automatic grid creates");
    }

    // The payoff ceiling binds before the budget: at 4096 tokens the logical space holds
    // 64 / 4 - 2 = 14 useful anchors, so a 256 MiB budget stops there and the rest stays Host KV.
    {
        ninfer::ContextCacheOptions cache = budgeted(256, 2, 7, 4);
        resolve(cache, 4096);
        failures +=
            check_split(cache, 14, 39, 217, "the anchor payoff ceiling did not bound the budget");
    }

    // The ceiling never lowers a count the caller configured, even one above it.
    {
        ninfer::ContextCacheOptions cache = budgeted(256, 2, 7, 20);
        resolve(cache, 4096);
        failures +=
            check_split(cache, 20, 51, 205, "the anchor payoff ceiling lowered a configured count");
    }

    // The production nvidia deployment shape: 52,000 MiB budget with the real int8 page-group
    // (2,162,688 B) and DFlash2 StateImage (195,897,344 B) unit costs, at --max-concurrency 2
    // (4 private continuations, 7 shared prefixes) and 240,000-token capacity. Hand oracle: the
    // half budget is 27,262,976,000 B; the mandatory inventory at 4 anchors is 31 images =
    // 6,072,817,664 B, leaving 108 images of headroom = 27 extra anchors, so A = 31. The payoff
    // ceiling is pages(240000) / pages(5760 buyback tokens) - 2 = 3750 / 90 - 2 = 39, so the
    // budget binds. The pool holds (2 + 31) * 4 + 7 = 139 images = 27,229,730,816 B of state,
    // and Host KV gets 52000 * 2^20 - that = 27,296,221,184 B.
    {
        ninfer::ContextCacheOptions cache;
        cache.host_cache_budget_bytes           = static_cast<std::size_t>(52000ULL * kMiB);
        cache.max_private_continuations         = 4;
        cache.max_shared_prefixes               = 7;
        cache.max_long_anchors_per_continuation = 4;
        cache.automatic_long_anchors            = true;
        ninfer::models::qwen3_5::detail::resolve_host_cache_budget(
            cache, *cache.max_private_continuations, *cache.max_shared_prefixes, 240000,
            195897344ULL, 2162688ULL);
        failures += check(
            *cache.max_long_anchors_per_continuation == 31 && cache.host_state_slots == 139 &&
                cache.host_kv_capacity_bytes == 27296221184ULL,
            "production budget did not resolve to the hand-computed 31 anchors / 139 images / "
            "27,296,221,184 B Host KV split");
    }

    // Hybrid mode derives every tuning value from the rest of the configuration. With the default
    // Host tier (8 GiB) at concurrency 2 and a 2048-token chunk: one resident snapshot per lane
    // plus one staging slot = 3, 8 taps, ladder max(4096, 2 * 2048) = 4096 and minimum gap
    // max(1024, 2048) = 2048. The Legacy catalogs and Host pools are empty.
    {
        EngineOptions options;
        options.max_concurrency                = 2;
        options.prefill_chunk                  = 2048;
        options.context_cache.mode             = ninfer::ContextCacheMode::Hybrid;
        const EngineOptions normalized         = normalize_engine_options(options);
        const ninfer::ContextCacheOptions& out = normalized.context_cache;
        failures +=
            check(out.host_cache_budget_bytes == ninfer::kDefaultHybridHostCacheBytes &&
                      out.hybrid.device_snapshot_slots == 3U && out.device_state_slots == 3U &&
                      out.hybrid.max_new_taps == 8U && out.hybrid.tap_ladder_tokens == 4096U &&
                      out.hybrid.tap_min_gap_tokens == 2048U && out.host_state_slots == 0 &&
                      out.host_kv_capacity_bytes == 0 && out.max_private_continuations == 2U &&
                      out.max_shared_prefixes == 0U && out.max_long_anchors_per_continuation == 0U,
                  "hybrid defaults did not derive from concurrency, chunk and Host tier");
    }

    // Without a Host tier Device slots are the only snapshot storage: two spare slots and a
    // two-tap budget. A large chunk coarsens the ladder: max(4096, 2 * 8192) and max(1024, 8192).
    {
        EngineOptions options;
        options.max_concurrency                       = 8;
        options.prefill_chunk                         = 8192;
        options.context_cache.mode                    = ninfer::ContextCacheMode::Hybrid;
        options.context_cache.host_cache_budget_bytes = 0;
        const ninfer::ContextCacheOptions out = normalize_engine_options(options).context_cache;
        failures +=
            check(out.host_cache_budget_bytes == 0U && out.hybrid.device_snapshot_slots == 10U &&
                      out.hybrid.max_new_taps == 2U && out.hybrid.tap_ladder_tokens == 16384U &&
                      out.hybrid.tap_min_gap_tokens == 8192U,
                  "Device-only hybrid defaults are wrong");
    }

    // Explicit hybrid overrides are kept verbatim.
    {
        EngineOptions options;
        options.max_concurrency                            = 4;
        options.context_cache.mode                         = ninfer::ContextCacheMode::Hybrid;
        options.context_cache.hybrid.device_snapshot_slots = 12;
        options.context_cache.hybrid.max_new_taps          = 3;
        options.context_cache.hybrid.tap_ladder_tokens     = 8192;
        options.context_cache.hybrid.tap_min_gap_tokens    = 512;
        const ninfer::ContextCacheOptions out = normalize_engine_options(options).context_cache;
        failures +=
            check(out.hybrid.device_snapshot_slots == 12U && out.device_state_slots == 12U &&
                      out.hybrid.max_new_taps == 3U && out.hybrid.tap_ladder_tokens == 8192U &&
                      out.hybrid.tap_min_gap_tokens == 512U,
                  "explicit hybrid overrides were not preserved");
    }

    // Hybrid mode rejects Legacy capacities and out-of-range tuning.
    for (int variant = 0; variant < 4; ++variant) {
        EngineOptions options;
        options.context_cache.mode = ninfer::ContextCacheMode::Hybrid;
        if (variant == 0) { options.context_cache.device_state_slots = 2; }
        if (variant == 1) { options.context_cache.max_private_continuations = 4; }
        if (variant == 2) { options.context_cache.hybrid.device_snapshot_slots = 65; }
        if (variant == 3) { options.context_cache.hybrid.tap_min_gap_tokens = 16; }
        bool rejected = false;
        try {
            (void)normalize_engine_options(options);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "hybrid normalization accepted an invalid capacity");
    }

    // The hybrid Host budget buys whole slabs; it must hold one snapshot (image slabs + tail slab)
    // plus one block. A 2 MiB slab and a 7-slab image need 9 slabs = 18 MiB; 0 disables the tier.
    {
        ninfer::models::qwen3_5::detail::HybridHostLayout layout;
        layout.slab_bytes  = 2ULL << 20;
        layout.image_slabs = 7;
        using ninfer::models::qwen3_5::detail::hybrid_host_slabs;
        failures += check(hybrid_host_slabs(layout, 0) == 0, "a zero Host budget must disable");
        failures += check(hybrid_host_slabs(layout, 18ULL << 20) == 9U,
                          "the minimum Host budget did not buy exactly one snapshot and a block");
        failures += check(hybrid_host_slabs(layout, (21ULL << 20) - 1U) == 10U,
                          "a Host budget must buy whole slabs");
        bool rejected = false;
        try {
            (void)hybrid_host_slabs(layout, (18ULL << 20) - 1U);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "a Host budget below one snapshot was accepted");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
