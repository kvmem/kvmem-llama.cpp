#pragma once

// Host-side KVMem runtime: block table + CPU/NVMe tiers + reselect cadence.
// finish_reselect always runs stage_out before stage_in. GPU copies go through
// KvMemBackend (no-op in P0).

#include "kvmem/kvmem_backend.hpp"
#include "kvmem/kvmem_store.hpp"
#if defined(_WIN32)
// The POSIX implementation cannot be built on Windows; see the header.
#include "kvmem/nvme_kv_tier_win.hpp"
#else
#include "kvmem/nvme_kv_tier.hpp"
#endif
#include "kvmem/pinned_kv_tier.hpp"

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kvmem {

struct KvMemRuntimeConfig {
    KvMemStoreConfig store;
    uint64_t cpu_bytes = 0;
    uint64_t nvme_bytes = 0;
    std::string nvme_dir;
};

class KvMemRuntime {
public:
    explicit KvMemRuntime(KvMemRuntimeConfig cfg,
                          KvMemBackend *backend = nullptr);

    const KvMemRuntimeConfig &config() const { return cfg_; }
    KvMemStore &store() { return store_; }
    const KvMemStore &store() const { return store_; }
    const KvMemPlan &last_plan() const { return last_plan_; }

    PinnedKvTier *cpu_tier() { return cpu_tier_.get(); }
    NvmeKvTier *nvme_tier() { return nvme_tier_.get(); }

    void register_append(uint32_t n_tokens);
    std::vector<KvMemDroppedBlock> truncate_to(uint32_t token_pos);

    // Recency / retrieval selection. `mandatory` consumes budget slots.
    KvMemPlan prepare_reselect(const std::vector<uint32_t> &mandatory = {},
                               bool force_raw_refresh = false);
    std::vector<uint32_t> preview_reselect(const std::vector<uint32_t> & mandatory = {}) const;
    KvMemPlan prepare_selection(const std::vector<uint32_t> & selected, bool force_raw_refresh = false);
    bool commit_resident_selection(const std::vector<uint32_t> & selected);
    KvMemPlan prepare_prefill_pressure(
        const std::vector<uint32_t> &mandatory = {});

    // If the next prefill chunk would overflow the semantic budget, the hard
    // GPU pool, or the high watermark, build a sink+tail pressure plan.
    // Does not apply it; caller runs finish_reselect (evict-before-stage-in).
    bool maybe_offload_during_prefill(
        uint32_t incoming_tokens,
        uint32_t resident_tokens,
        uint32_t pool_tokens,
        const std::vector<uint32_t> &mandatory = {});

    // Copy outgoing GPU blocks to CPU/NVMe. Does not free GPU slots yet so
    // the adapter can seq_rm the cells afterwards.
    void spill_outgoing();
    // Free spilled GPU slots then stage-in (host/NVMe -> GPU).
    void admit_incoming();
    // spill_outgoing + admit_incoming (host tests; no llama seq_rm).
    void finish_reselect();

    void reselect() {
        prepare_reselect();
        finish_reselect();
    }

private:
    void stage_out(uint32_t block_id);
    void stage_in(uint32_t block_id);
    void start_prefetch();
    void wait_prefetch();
    uint8_t *cpu_ptr(int32_t slot);
    const uint8_t *cpu_ptr(int32_t slot) const;
    bool spill_bytes_to_nvme(uint32_t block_id, const void *data);
    void trace_tier(const char *tag, uint32_t block_id, int32_t slot) const;

    KvMemRuntimeConfig cfg_;
    KvMemStore store_;
    KvMemBackend *backend_;
    KvMemBackend null_backend_;
    std::unique_ptr<PinnedKvTier> cpu_tier_;
    std::unique_ptr<NvmeKvTier> nvme_tier_;
    KvMemPlan last_plan_;
    bool pending_ = false;
    bool trace_ = false;
    uint64_t slot_bytes_ = 0;
    std::vector<uint8_t> cpu_arena_;
    std::vector<uint8_t> scratch_;
    std::unordered_map<uint32_t, std::shared_ptr<std::vector<uint8_t>>> prefetch_buf_;
    std::vector<std::future<void>> prefetch_futs_;
    std::vector<int32_t> pending_gpu_frees_;
};

} // namespace kvmem
