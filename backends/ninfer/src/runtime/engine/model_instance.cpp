#include "runtime/engine/model_instance.h"
#include "calibration/device_calibration.h"
#include "core/arena.h"
#include "ops/common/device_route.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/device_profile.h"
#include "runtime/engine/diagnostics.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <chrono>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_memory_policy(const EngineOptions& options) {
    switch (options.cuda_memory_policy) {
    case CudaMemoryPolicy::DriverDefault:
        break;
    case CudaMemoryPolicy::Mixed:
#ifndef _WIN32
        throw std::invalid_argument("mixed CUDA memory requires Windows driver paging");
#endif
        if (options.purpose != EnginePurpose::Generation || options.devices.size() > 1 ||
            options.enable_vision || options.wddm_evictable_budget) {
            throw std::invalid_argument(
                "mixed CUDA memory supports single-GPU text generation and ordinary CUDA "
                "allocation only; scoring, Vision, pipeline devices and wddm-evictable-budget "
                "are not supported");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("mixed CUDA memory requires zero automatic KV headroom");
        }
        break;
    case CudaMemoryPolicy::StrictVram:
#ifndef _WIN32
        throw std::invalid_argument("strict requires Windows GPU residency counters");
#endif
        if (options.purpose != EnginePurpose::Generation || options.devices.size() > 1 ||
            options.enable_vision || options.wddm_evictable_budget ||
            !options.context_cache.enabled || options.context_cache.mode != ContextCacheMode::Hybrid) {
            throw std::invalid_argument(
                "strict supports single-GPU text generation with the Hybrid prefix cache "
                "only; scoring, Vision, pipeline devices, Legacy/disabled cache and "
                "wddm-evictable-budget are not supported");
        }
        if (options.cuda_memory_probe_step_bytes == 0) {
            throw std::invalid_argument("strict probe step must be positive");
        }
        break;
    default:
        throw std::invalid_argument("unknown CUDA memory policy");
    }
}

void validate_options(const EngineOptions& options) {
    validate_memory_policy(options);
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.cuda_graph_allowance_bytes != 0 && !options.use_cuda_graph) {
        throw std::invalid_argument(
            "Engine cuda_graph_allowance_bytes requires CUDA graphs to be enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    if (!std::isfinite(options.rope_yarn_factor) || options.rope_yarn_factor < 1.0F ||
        options.rope_yarn_factor > 4.0F) {
        throw std::invalid_argument("Engine rope_yarn_factor must be finite and in [1,4]");
    }
    if (!std::isfinite(options.rope_scaling_factor) || options.rope_scaling_factor < 1.0F ||
        options.rope_scaling_factor > 32.0F) {
        throw std::invalid_argument("Engine rope_scaling_factor must be finite and in [1,32]");
    }
    if (options.rope_scaling_factor > 1.0F &&
        (options.rope_yarn || options.rope_yarn_factor > 1.0F)) {
        throw std::invalid_argument(
            "Engine rope_scaling_factor excludes rope_yarn and rope_yarn_factor");
    }
}

// The hybrid index ranks admission sources and values snapshots with the same calibrated
// prefill and Host-to-Device coefficients the Legacy ResourceManager prices materialization with.
// Uncalibrated (zero) terms keep the index's generic defaults.
prefix_cache::CacheCostModel hybrid_cache_cost(const ContextMachineCostModel& model) {
    constexpr double kSecondsPerNs = 1.0e-9;
    constexpr double kQ32          = 4294967296.0;
    prefix_cache::CacheCostModel cost;
    if (model.prefill.chunk_ns != 0) {
        cost.chunk_seconds = static_cast<double>(model.prefill.chunk_ns) * kSecondsPerNs;
    }
    if (model.prefill.token_ns_q32 != 0) {
        cost.token_seconds = static_cast<double>(model.prefill.token_ns_q32) / kQ32 * kSecondsPerNs;
    }
    if (model.prefill.attention_pair_ns_q32 != 0) {
        cost.attention_pair_seconds =
            static_cast<double>(model.prefill.attention_pair_ns_q32) / kQ32 * kSecondsPerNs;
    }
    const ContextTransferCost& h2d =
        model.transfer[static_cast<std::size_t>(ContextTransferDirection::HostToDevice)];
    if (h2d.ns_per_byte_q32 != 0) {
        cost.h2d_bytes_per_second =
            1.0 / (static_cast<double>(h2d.ns_per_byte_q32) / kQ32 * kSecondsPerNs);
    }
    if (h2d.batch_ns != 0) {
        cost.transfer_batch_seconds = static_cast<double>(h2d.batch_ns) * kSecondsPerNs;
    }
    return cost;
}

// Everything the bytes of a persisted hybrid Host tier depend on besides its geometry (which the
// file records itself): the exact artifact, its execution signature, the KV and speculative
// formats, RoPE scaling and the product binary's build identity. Any difference makes the saved
// state meaningless, so the file is ignored.
std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature) {
    std::error_code error;
    const auto size = std::filesystem::file_size(options.artifact_path, error);
    const auto time = std::filesystem::last_write_time(options.artifact_path, error);
    std::string out = "artifact=" + std::filesystem::absolute(options.artifact_path).string();
    out += ";size=" + std::to_string(error ? 0U : size);
    out += ";mtime=" +
           std::to_string(error ? 0LL : static_cast<long long>(time.time_since_epoch().count()));
    out += ";signature=" + signature;
    out += ";kv=" + std::to_string(static_cast<int>(options.kv_cache));
    out += ";speculative=" + std::to_string(static_cast<int>(options.speculative.backend));
    out += ";yarn=" + std::to_string(options.rope_yarn_factor);
    out += ";interpolation=" + std::to_string(options.rope_scaling_factor) + "@" +
           std::to_string(options.rope_scaling_original_context);
    out += ";build=" + options.context_cache.hybrid.persistent_identity;
    return out;
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// Free memory on each rank's device, in rank order. Ranks that share a physical device (a test mode
// that exercises the multi-stage path on one card) split what is free between them, since each
// one's budget is spent from the same memory. Under the WDDM evictable budget the primary device
// may also count what other processes hold beyond the desktop floor: all of it but the resident
// weights.
std::vector<std::size_t> free_bytes_by_rank(const DeviceContext& device,
                                            bool evictable_budget      = false,
                                            std::size_t resident_bytes = 0) {
    std::vector<std::size_t> out;
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < device.size(); ++other) {
            if (device.same_physical_device(rank, other)) { ++sharing; }
        }
        DeviceBinding bind(device.rank(rank).device);
        std::size_t free_bytes = current_free_device_bytes();
        if (evictable_budget && rank == 0) {
            constexpr std::size_t kDesktopFloor = 512ULL << 20;
            std::size_t free_now                = 0;
            std::size_t total                   = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_now, &total));
            if (total > resident_bytes + kDesktopFloor) {
                free_bytes = std::max(free_bytes, total - resident_bytes - kDesktopFloor);
            }
        }
        out.push_back(free_bytes / sharing);
    }
    return out;
}

// Disk-tier pages are keyed by token digests, which say nothing about the weights that produced
// them: each artifact gets its own directory so one model can never restore another's KV.
EngineOptions artifact_scoped_disk_tier(const EngineOptions& options,
                                        const artifact::ArtifactId& artifact) {
    EngineOptions scoped = options;
    if (scoped.context_cache.disk_kv_path.empty()) { return scoped; }
    std::string name = "artifact_";
    for (const std::byte byte : artifact) {
        char hex[3];
        std::snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned>(byte));
        name += hex;
    }
    scoped.context_cache.disk_kv_path /= name;
    return scoped;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    if (!options.kvmem.selected_tokens &&
        (options.kvmem.reserve_tokens || options.kvmem.host_bytes || options.kvmem.verify_transfers ||
         !options.kvmem.disk_path.empty() || options.kvmem.disk_bytes)) {
        throw std::invalid_argument("KVMem auxiliary budgets require selected_tokens");
    }
    if (options.kvmem.selected_tokens != 0) {
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

        if (options.purpose != EnginePurpose::Generation || options.max_concurrency == 0 || options.max_concurrency > 4 ||
            options.devices.size() > 1 || options.concurrent_prefill ||
            (options.enable_vision && (options.max_concurrency != 1 ||
                options.vision_residency != VisionResidency::Resident)) ||
            !memory_spec_supported ||
            options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault ||
            options.kv_capacity.mode != KvCapacityMode::Explicit ||
            options.kv_capacity.explicit_tokens != window * options.max_concurrency || window % 64 != 0 || window < 256 ||
            window > options.max_context || options.kvmem.selected_tokens < 128 ||
            options.kvmem.selected_tokens % 64 != 0 || options.kvmem.reserve_tokens < 64 ||
            options.kvmem.reserve_tokens % 64 != 0 || !options.kvmem.host_bytes ||
            options.kvmem.retained_sessions == 0 || options.kvmem.retained_sessions > 16 ||
            options.prefill_chunk == 0 || options.prefill_chunk % 64 != 0 ||
            options.prefill_chunk > options.kvmem.reserve_tokens ||
            options.context_cache.enabled || options.fast_prefill_kernel ||
            (options.kv_cache != KvCacheStorage::BFloat16 && options.kv_cache != KvCacheStorage::Int8Group64 &&
             options.kv_cache != KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) ||
            (options.kv_cache == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 &&
             (options.max_concurrency != 1 || options.enable_vision))) {
            throw std::invalid_argument("KVMem requires explicit B+R page-aligned "
                "capacity per lane, nonzero global H, aligned prefill <= R, BF16/INT8 or single-lane text RK8V4, 1..4 text lanes or one resident-vision lane, ordinary/MTP1..4, "
                "and disabled prefix cache");
        }
    }
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache = options.context_cache;
    if (options.cuda_memory_policy == CudaMemoryPolicy::StrictVram) {
        if (!cache.enabled) {
            throw std::invalid_argument("strict CUDA memory requires prefix reuse and cannot disable the context cache");
        }
        // Select the cache before deriving its capacities. Host cache remains an independent
        // setting: an explicit zero or nonzero budget must survive this normalization unchanged.
        cache.mode = ContextCacheMode::Hybrid;
    }
    if (options.kv_capacity.mode == KvCapacityMode::Automatic) {
        if (options.cuda_memory_policy == CudaMemoryPolicy::StrictVram) {
            options.kv_capacity.automatic_headroom_bytes = options.cuda_vram_reserve_bytes;
        } else if (options.cuda_memory_policy == CudaMemoryPolicy::Mixed) {
            options.kv_capacity.automatic_headroom_bytes = 0;
        }
    }
    validate_memory_policy(options);
    if (options.speculative.ngram_archive_bytes != 0 &&
        options.speculative.ngram_draft_tokens == 0) {
        throw std::invalid_argument(
            "the cross-request ngram archive requires ngram drafting; enable --ngram-draft-tokens");
    }
    if (options.speculative.ngram_archive_bytes != 0 &&
        (options.speculative.ngram_session_bytes < (1ULL << 20) ||
         options.speculative.ngram_session_bytes > options.speculative.ngram_archive_bytes)) {
        throw std::invalid_argument("ngram session capacity must be between 1 MiB and the total "
                                    "archive capacity");
    }
    // The GDN conv-record workspace admits at most 16 verification columns once a batch holds
    // more than one request.
    if (options.speculative.ngram_draft_tokens > 15 && options.max_concurrency != 1) {
        throw std::invalid_argument("ngram draft widths above 15 require engine concurrency one");
    }
    const std::uint32_t concurrency = options.max_concurrency;
    if (cache.enabled && cache.mode == ContextCacheMode::Hybrid) {
        if (cache.device_state_slots || cache.max_private_continuations ||
            cache.max_shared_prefixes || cache.max_long_anchors_per_continuation) {
            throw std::invalid_argument(
                "the hybrid prefix cache does not accept Legacy capacity options (Device state "
                "slots, private/shared catalogs, long anchors)");
        }
        if (cache.rolling_retention || cache.release_diverged_checkpoints ||
            cache.thorough_admission_search || cache.recency_eviction || cache.value_aware_demote ||
            cache.automatic_long_anchors || !cache.disk_kv_path.empty()) {
            throw std::invalid_argument(
                "the hybrid prefix cache does not accept Legacy cache policies (rolling retention, "
                "diverged-checkpoint release, thorough admission search, recency eviction, "
                "value-aware demotion, automatic long anchors, disk tier)");
        }
        // One pinned Host slab pool serves blocks and snapshots alike; its size is the only
        // capacity a deployment has to choose (docs/maintainer/hybrid-prefix-cache-spec.md §5.4).
        cache.host_cache_budget_bytes =
            cache.host_cache_budget_bytes.value_or(kDefaultHybridHostCacheBytes);
        const bool host_tier             = *cache.host_cache_budget_bytes != 0;
        HybridPrefixCacheOptions& hybrid = cache.hybrid;
        // One resident snapshot per request lane keeps every live conversation's latest
        // endpoint restorable without PCIe traffic; one more slot stages taps and endpoints while
        // their Host copies are written. Without a Host tier these slots are the only snapshot
        // storage, so one more is kept for shared prefixes.
        hybrid.device_snapshot_slots =
            hybrid.device_snapshot_slots.value_or(concurrency + (host_tier ? 1U : 2U));
        // Taps without a Host tier would evict other conversations' resident snapshots.
        hybrid.max_new_taps = hybrid.max_new_taps.value_or(host_tier ? 8U : 2U);
        // Ladder taps are realized on prefill chunk boundaries, so the ladder never refines below
        // the chunk: coarser ladders only waste snapshots on taps that share one boundary.
        const std::uint32_t chunk = std::max<std::uint32_t>(options.prefill_chunk, 64U);
        hybrid.tap_ladder_tokens =
            hybrid.tap_ladder_tokens.value_or(std::max<std::uint32_t>(4096U, 2U * chunk));
        hybrid.tap_min_gap_tokens =
            hybrid.tap_min_gap_tokens.value_or(std::max<std::uint32_t>(1024U, chunk));
        if (*hybrid.device_snapshot_slots == 0 || *hybrid.device_snapshot_slots > 64) {
            throw std::invalid_argument("hybrid device snapshot slots must be in [1,64]");
        }
        if (*hybrid.max_new_taps > 64) {
            throw std::invalid_argument("hybrid taps per request must be at most 64");
        }
        if (*hybrid.tap_ladder_tokens < 64 || *hybrid.tap_min_gap_tokens < 64) {
            throw std::invalid_argument(
                "hybrid tap ladder and minimum gap must be at least 64 tokens");
        }
        // The hybrid tree keeps no catalog of owners: every lane holds one active continuation
        // and retained context lives in the Program's prefix index. The Legacy Host pools are
        // replaced by the hybrid slab pool.
        cache.device_state_slots                = hybrid.device_snapshot_slots;
        cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        return options;
    }
    if (!cache.enabled) {
        cache.mode = ContextCacheMode::Legacy;
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0) ||
            cache.host_cache_budget_bytes) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);
        // Without a catalog no continuation is ever evicted or restored.
        cache.disk_kv_path.clear();
        cache.disk_kv_restore       = false;
        cache.disk_kv_directstorage = false;
        cache.rolling_retention            = false;
        cache.release_diverged_checkpoints = false;
        cache.thorough_admission_search    = false;
        cache.recency_eviction             = false;
        cache.value_aware_demote           = false;
        cache.kv_lease_growth              = false;
        cache.automatic_long_anchors       = false;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(std::max(
        concurrency, static_cast<std::uint32_t>(kMaximumPreparedPromptCacheCandidatesPerRequest)));
    cache.max_long_anchors_per_continuation =
        cache.max_long_anchors_per_continuation.value_or(cache.automatic_long_anchors ? 4U : 2U);
    cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

ModelInstance::ModelInstance(std::unique_ptr<models::qwen3_5::Model> source,
                             const EngineOptions& options)
    : model(std::move(source)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(),
          {.chat_template_path       = options.chat_template_path,
           .architecture             = model->config().text.architecture,
           .vision_enabled           = options.enable_vision,
           .max_context              = options.max_context,
           .media_cache_bytes        = options.media_cache_bytes,
           .media_live_bytes         = options.media_live_bytes,
           .media_preprocess_threads = options.media_preprocess_threads,
           .vision_max_merged_tokens = options.vision_max_merged_tokens,
           .thinking_budget_message  = options.thinking_budget_message,
           .ngram_sources_enabled    = options.speculative.ngram_draft_tokens != 0,
           .ngram_archive_enabled    = options.speculative.ngram_archive_bytes != 0,
           .automatic_long_anchors =
               options.context_cache.automatic_long_anchors
                   ? options.context_cache.max_long_anchors_per_continuation.value_or(0U)
                   : 0U,
           .long_anchor_min_spacing_tokens =
               options.context_cache.long_anchor_min_spacing_tokens})),
      capacity(options.max_context) {}

ModelInstance::~ModelInstance() = default;

namespace {

// Installs each rank's GPU route profile before any Op runs (see EngineOptions::device_profile).
// A device with no measured profile is calibrated once, in this process, before the weights claim
// its memory; the result goes to the profile file for the next start.
void install_device_route_profile_on(const EngineOptions& options, int device) {
    if (options.device_profile == "off") {
        ops::install_device_route_profile(device, nullptr);
        return;
    }
    const bool calibrate = options.device_profile == "calibrate";
    if (!calibrate && options.device_profile != "auto") {
        throw std::invalid_argument("device_profile must be auto, off or calibrate");
    }
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const std::string hardware_class = context_cost_hardware_class(props.name, props.major, props.minor);
    const int multiprocessors        = props.multiProcessorCount;
    const std::filesystem::path path = options.device_profile_path.empty()
                                           ? default_device_profile_path()
                                           : options.device_profile_path;
    std::optional<ops::DeviceRouteProfile> profile;
    if (!calibrate) {
        try {
            profile = find_device_route_profile(hardware_class, multiprocessors, path);
        } catch (const std::exception& error) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "device profile %s ignored: %s", path.string().c_str(),
                               error.what());
        }
    }
    if (!profile) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "calibrating routes for %s (%d SMs)", hardware_class.c_str(),
                           multiprocessors);
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(device));
        ops::install_device_route_profile(device, nullptr);
        calibration::CalibrationOptions calibration_options;
        profile = calibration::calibrate_device_routes(calibration_options);
        CUDA_CHECK(cudaSetDevice(previous));
        profile->hardware_class = hardware_class;
        try {
            upsert_device_route_profile_atomic(path, *profile);
        } catch (const std::exception& error) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "device profile not saved to %s: %s", path.string().c_str(),
                               error.what());
        }
    }
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                       "device profile %s: %zu routed keys (%s)", hardware_class.c_str(),
                       profile->routes.size(), profile->origin.c_str());
    ops::install_device_route_profile(
        device, std::make_shared<const ops::DeviceRouteProfile>(std::move(*profile)));
}

void install_device_route_profile_for(const EngineOptions& options, const DeviceContext& device) {
    std::vector<int> installed;
    for (const int id : device.device_ids()) {
        if (std::find(installed.begin(), installed.end(), id) != installed.end()) { continue; }
        install_device_route_profile_on(options, id);
        installed.push_back(id);
    }
}

} // namespace

ConstructedModel construct_model(const EngineOptions& requested, DeviceContext& device) {
    validate_options(requested);
    install_device_route_profile_for(requested, device);
    std::shared_ptr<core::ResidentMemorySession> residency;
    if (requested.cuda_memory_policy == CudaMemoryPolicy::StrictVram) {
        constexpr std::size_t staging_and_control = 320ULL << 20;
        const std::size_t host_cache = requested.context_cache.host_cache_budget_bytes.value();
        if (host_cache > std::numeric_limits<std::size_t>::max() - staging_and_control) {
            throw std::overflow_error("strict Host pool capacity overflows size_t");
        }
        const std::size_t reserve =
            requested.kv_capacity.mode == KvCapacityMode::Automatic
                ? std::max(requested.cuda_vram_reserve_bytes,
                           requested.kv_capacity.automatic_headroom_bytes)
                : requested.cuda_vram_reserve_bytes;
        residency = std::make_shared<core::ResidentMemorySession>(core::ResidentMemoryConfig{
            .device = device.rank(0).device,
            .reserve_bytes = reserve,
            .host_pool_bytes = host_cache + staging_and_control,
            .probe_step_bytes = requested.cuda_memory_probe_step_bytes,
        });
        publish_diagnostic(requested.diagnostic_observer, DiagnosticLevel::Info,
                           "strict: fixed Host pool %.1f MiB, verified reserve %.1f MiB",
                           static_cast<double>(host_cache + staging_and_control) / 1048576.0,
                           static_cast<double>(reserve) / 1048576.0);
    }
    core::ResidentMemoryBinding resident_binding(residency);
    const auto start = Clock::now();
    // Every later stage of startup checks its options against the ones the model was loaded with, so
    // the stage split is decided once, here, and carried in the options from then on.
    EngineOptions options = requested;
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    if (options.devices.size() > 1 && options.stage_layers.empty()) {
        const std::vector<std::size_t> free_now = free_bytes_by_rank(device);
        const std::vector<std::uint64_t> free_bytes(free_now.begin(), free_now.end());
        const models::qwen3_5::StageSizing sizing{
            .kv_storage = options.kv_cache,
            .state_slots =
                options.max_concurrency + options.context_cache.device_state_slots.value_or(0U)};
        options.stage_layers = models::qwen3_5::default_stage_layers(
            reader, models::load_options(options), sizing, free_bytes);
    }
    core::set_wddm_residency_lock_enabled(options.wddm_evictable_budget);
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    if (residency && model->storage_stats().pinned_bytes != 0) {
        throw std::invalid_argument(
            "strict requires all selected weights resident on the device; "
            "the artifact materialized pinned Host weights");
    }
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<ModelInstance>(std::move(model), options);
    instance->residency = residency;
    if (residency) { residency->verify_working_set(false); }
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const std::size_t overlay_window_bytes =
        models::qwen3_5::prepare_vision_overlay(instance->parameters, device, options);
    const auto signature = models::qwen3_5::prefill_signature(*instance->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    const EngineOptions planning_options =
        artifact_scoped_disk_tier(options, instance->model->info().artifact_id);
    const auto curve = models::qwen3_5::make_sequence_planner(
        instance->parameters, device, planning_options).capacity_curve();
    std::vector<std::size_t> free_by_rank =
        free_bytes_by_rank(device, options.wddm_evictable_budget,
                           instance->model->storage_stats().device_capacity_bytes);
    KvCapacityPolicy capacity_policy = options.kv_capacity;
    if (residency) {
        const auto memory = residency->stats();
        const std::size_t physical_upper_bound =
            memory.cuda_total_bytes > memory.device_allocated_bytes
                ? memory.cuda_total_bytes - memory.device_allocated_bytes : 0;
        if (capacity_policy.mode == KvCapacityMode::Automatic) {
            capacity_policy.automatic_headroom_bytes =
                std::max(capacity_policy.automatic_headroom_bytes,
                         options.cuda_vram_reserve_bytes);
            // Temporary probe blocks are released before constructing real contiguous arenas.
            // The probe returns held bytes, without subtracting the reserve a second time.
            free_by_rank.front() = residency->probe_available_bytes(physical_upper_bound);
            residency->verify_working_set(false);
        } else {
            // WDDM free is an estimate, not an allocation limit. This upper bound rejects only
            // impossible sizes; actual allocation and working-set verification decide admission.
            free_by_rank.front() = physical_upper_bound > options.cuda_vram_reserve_bytes
                ? physical_upper_bound - options.cuda_vram_reserve_bytes : 0;
        }
    }
    auto resolution = options.cuda_memory_policy == CudaMemoryPolicy::Mixed
        ? resolve_mixed_kv_capacity(capacity_policy, curve, options.max_context,
                                   options.max_concurrency, free_by_rank.front())
        : resolve_kv_capacity(capacity_policy, curve, free_by_rank.front(),
                              std::span<const std::size_t>(free_by_rank).subspan(1));
    EngineOptions resolved = options;
    LoadSummary::PrefixCacheRestore restore;
    constexpr unsigned maximum_attempts = 8;
    for (unsigned attempt = 0; ; ++attempt) {
        auto planner = models::qwen3_5::make_sequence_planner(
            instance->parameters, device, planning_options);
        auto sequence = std::move(planner).finalize(resolution.main_page_groups);
        if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
            sequence.kv_capacity() != resolution.resolved_tokens ||
            !std::ranges::equal(sequence.extra_rank_reservation_bytes(),
                                resolution.extra_rank_reservation_bytes)) {
            throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
        }
        resolved.context_cache = sequence.context_cache_options();
        if (attempt == 0) { planning.complete(); }
        StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
        const auto retry_capacity = [&](const std::runtime_error& error) {
            device.synchronize();
            instance->program.reset();
            if (!residency || capacity_policy.mode != KvCapacityMode::Automatic ||
                resolution.main_page_groups <= curve.minimum_main_page_groups ||
                attempt + 1 >= maximum_attempts) {
                throw core::VramCapacityError(
                    "strict could not admit max_context=" + std::to_string(options.max_context) +
                    " with the requested KV capacity and reserve; context was not reduced: " +
                    error.what());
            }
            // A poisoned CUDA context or displaced weight working set aborts recovery.
            residency->verify_working_set(false);
            const std::size_t stride = curve.bytes_per_additional_main_page_group;
            const std::size_t step_pages =
                1 + (options.cuda_memory_probe_step_bytes - 1) / stride;
            const std::size_t extra_pages =
                resolution.main_page_groups - curve.minimum_main_page_groups;
            const auto next_pages = attempt + 2 == maximum_attempts
                ? curve.minimum_main_page_groups
                : resolution.main_page_groups - static_cast<std::uint32_t>(
                      std::min(extra_pages, step_pages));
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "strict: candidate %u KV tokens failed (%s); retrying %u tokens "
                               "without reducing max_context %u",
                               resolution.resolved_tokens, error.what(),
                               curve.resolved_tokens(next_pages), options.max_context);
            const std::size_t next_budget = curve.reservation_bytes(next_pages) +
                                           capacity_policy.automatic_headroom_bytes;
            resolution = resolve_kv_capacity(capacity_policy, curve, next_budget);
            resolution.available_after_weights_bytes = free_by_rank.front();
            resolution.planned_slack_bytes = free_by_rank.front() -
                                             resolution.runtime_reservation_bytes;
        };
        try {
            instance->program = models::qwen3_5::create_program(
                instance->parameters, std::move(sequence), device, options.startup_observer);
            restore = {};
            if (resolved.context_cache.enabled &&
                resolved.context_cache.mode == ContextCacheMode::Hybrid) {
                instance->program->set_hybrid_cost(hybrid_cache_cost(context_cost.model));
                instance->program->set_hybrid_coalesce_wait_limit(
                    static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0);
                const std::filesystem::path& file = resolved.context_cache.hybrid.persistent_file;
                if (!file.empty()) {
                    const auto loaded = instance->program->attach_hybrid_cache_file(
                        file, hybrid_cache_fingerprint(options, signature));
                    restore = LoadSummary::PrefixCacheRestore{
                        .attempted = true, .restored = loaded.ok, .message = loaded.message,
                        .blocks = loaded.blocks, .snapshots = loaded.snapshots,
                        .bytes = loaded.bytes, .seconds = loaded.seconds,
                    };
                }
            }
            device.synchronize();
            if (residency) {
                // Include all KV backing and existing weights after Graph and Host cache setup.
                residency->verify_working_set(true);
                const auto memory = residency->stats();
                publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                   "strict: admitted %u KV tokens, %.1f MiB device allocations, "
                                   "Shared %.1f / baseline %.1f MiB, reserve %.1f MiB, %u retries",
                                   resolution.resolved_tokens,
                                   static_cast<double>(memory.device_allocated_bytes) / 1048576.0,
                                   static_cast<double>(memory.shared_bytes) / 1048576.0,
                                   static_cast<double>(memory.shared_baseline_bytes) / 1048576.0,
                                   static_cast<double>(memory.verified_reserve_bytes) / 1048576.0,
                                   attempt);
            }
            program.complete();
            break;
        } catch (const core::VramCapacityError& error) {
            retry_capacity(error);
        } catch (const CudaError& error) {
            if (!residency || error.status() != cudaErrorMemoryAllocation) { throw; }
            // Do not let a recoverable Graph allocation error leak through a later
            // kernel's cudaGetLastError. Any distinct pending execution fault is fatal.
            const auto pending = cudaGetLastError();
            if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) {
                CUDA_CHECK(pending);
            }
            retry_capacity(error);
        }
    }
    // Publish only the fully constructed and verified candidate's shape.
    instance->frontend.publish_long_anchor_limit(
        resolved.context_cache.max_long_anchors_per_continuation.value_or(0));
    instance->kv_capacity_resolution = resolution;
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->model->config().text.architecture);
    summary.model_name   = instance->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.pinned_weight_bytes  = stats.pinned_bytes;
    summary.overlay_window_bytes = overlay_window_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    summary.prefix_cache         = std::move(restore);

    // Static /v1/models metadata: the model identity and dimension facts come from the loaded
    // model; the parameters, weight bytes, and weights profile are pure functions of the
    // artifact's tensor inventory.
    const auto& text_config = instance->model->config().text;
    ModelMetadata metadata;
    metadata.model_id       = summary.model_name;
    metadata.vocab_size     = text_config.vocab_size;
    metadata.embedding_size = text_config.hidden_size;
    metadata.native_context = text_config.max_position_embeddings;
    std::uint64_t parameters   = 0;
    std::uint64_t weight_bytes = 0;
    std::set<std::string> tensor_formats;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; } // Non-weight resources (tokenizer, templates).
        std::uint64_t elements = 1;
        for (const auto dimension : tensor->shape) { elements *= dimension; }
        parameters     += elements;
        weight_bytes   += tensor->bytes;
        tensor_formats.emplace(tensor->format);
    }
    metadata.parameters   = parameters;
    metadata.weight_bytes = weight_bytes;
    for (const auto& format : tensor_formats) {
        if (!metadata.weights_id.empty()) { metadata.weights_id += "+"; }
        metadata.weights_id += format;
    }
    return {std::move(instance), std::move(summary), std::move(metadata),
            std::move(context_cost.model), std::move(resolved)};
}

} // namespace ninfer::runtime
