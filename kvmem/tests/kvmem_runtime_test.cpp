#include "kvmem/kvmem_runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <filesystem>
#include <vector>

using namespace kvmem;

static int g_fail = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

struct RecordingBackend : KvMemBackend {
    int32_t next = 0;
    std::vector<int32_t> allocs;
    std::vector<int32_t> frees;
    std::vector<std::string> ops;

    int32_t alloc_gpu_slot() override {
        const int32_t s = next++;
        allocs.push_back(s);
        ops.push_back("alloc");
        return s;
    }
    void free_gpu_slot(int32_t slot) override {
        frees.push_back(slot);
        ops.push_back("free");
    }
};

static KvMemRuntimeConfig make_cfg() {
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 32;
    cfg.store.select_budget = 32 * 4;
    cfg.store.sink_blocks = 1;
    cfg.store.recent_blocks = 1;
    cfg.store.estimated_block_bytes = 1024;
    cfg.cpu_bytes = 1024 * 16;
    return cfg;
}

static void test_stage_out_before_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    be.allocs.clear();
    be.frees.clear();

    auto plan = rt.prepare_reselect();
    CHECK(plan.remaps.size() == 4);
    rt.finish_reselect();

    // All GPU frees (evictions) must precede any new alloc.
    bool saw_alloc = false;
    bool order_ok = true;
    for (const auto &op : be.ops) {
        if (op == "alloc") saw_alloc = true;
        if (op == "free" && saw_alloc) order_ok = false;
    }
    CHECK(order_ok);
    CHECK(!be.frees.empty());
}

static void test_high_overlap_skips_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    rt.reselect();
    const auto first = rt.last_plan();
    CHECK(first.remaps.size() == 4);

    auto second = rt.prepare_reselect();
    CHECK(second.stage_in.empty());
    CHECK(second.gpu_reused_blocks == 4);
    for (const auto &rm : second.remaps) {
        CHECK(rm.skip || rm.working_k_resident);
    }
    rt.finish_reselect();
}

static void test_pressure_keeps_sink_and_tail() {
    KvMemRuntime rt(make_cfg());
    rt.register_append(32 * 10);
    auto plan = rt.prepare_prefill_pressure();
    CHECK(plan.remaps.size() == 4);
    CHECK(plan.remaps.front().block_id == 0);
    CHECK(plan.remaps.back().block_id == 9);
    rt.finish_reselect();
}

static void test_maybe_offload_evicts_before_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    be.allocs.clear();
    be.frees.clear();

    const uint32_t pool = 32 * 4;
    CHECK(rt.maybe_offload_during_prefill(/*incoming=*/32, /*resident=*/32 * 10, pool));
    CHECK(rt.last_plan().remaps.size() == 4);
    rt.finish_reselect();

    bool saw_alloc = false;
    bool order_ok = true;
    for (const auto &op : be.ops) {
        if (op == "alloc") saw_alloc = true;
        if (op == "free" && saw_alloc) order_ok = false;
    }
    CHECK(order_ok);
    CHECK(!be.frees.empty());

    KvMemRuntime idle(make_cfg());
    idle.register_append(32 * 2);
    CHECK(!idle.maybe_offload_during_prefill(32, 32, 32 * 8));
}

#if KVMEM_ENABLE_NVME
static void test_cpu_full_spills_to_nvme_and_roundtrips() {
    struct MemoryBackend : KvMemBackend {
        uint64_t slot_bytes = 64;
        int32_t next = 0;
        std::map<int32_t, std::vector<uint8_t>> gpu;
        int32_t alloc_gpu_slot() override {
            const int32_t s = next++;
            gpu[s].assign(slot_bytes, static_cast<uint8_t>(s + 1));
            return s;
        }
        void free_gpu_slot(int32_t slot) override { gpu.erase(slot); }
        void copy_block_to_host(uint32_t, int32_t gpu_slot, void *host,
                                uint64_t bytes) override {
            auto it = gpu.find(gpu_slot);
            if (it == gpu.end() || !host) return;
            std::memcpy(host, it->second.data(),
                        static_cast<size_t>(std::min(bytes, slot_bytes)));
        }
        void copy_block_from_host(uint32_t, int32_t gpu_slot, const void *host,
                                  uint64_t bytes) override {
            if (!host) return;
            gpu[gpu_slot].assign(static_cast<const uint8_t *>(host),
                                 static_cast<const uint8_t *>(host) +
                                     static_cast<size_t>(bytes));
        }
    };

    const std::string dir = (std::filesystem::temp_directory_path() / "kvmem_p32_nvme").string();

    MemoryBackend be;
    KvMemRuntimeConfig cfg = make_cfg();
    cfg.store.estimated_block_bytes = 64;
    cfg.cpu_bytes = 64 * 2;          // two CPU slots
    cfg.nvme_bytes = 64 * 8;
    cfg.nvme_dir = dir;
    KvMemRuntime rt(cfg, &be);
    CHECK(rt.cpu_tier() && rt.cpu_tier()->enabled());
    CHECK(rt.nvme_tier() && rt.nvme_tier()->enabled());

    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }

    auto plan = rt.prepare_prefill_pressure();
    CHECK(plan.remaps.size() == 4);
    rt.finish_reselect();

    uint32_t on_nvme = 0;
    for (const auto &b : rt.store().blocks()) {
        if (b.nvme_slot >= 0 || b.tier == KvTier::SSD) {
            ++on_nvme;
        }
    }
    CHECK(on_nvme > 0);

    // Bring a spilled middle block back via retrieval scores.
    std::vector<double> scores(10, 0.0);
    scores[4] = 100.0;
    rt.store().set_retrieval_scores(scores);
    auto back = rt.prepare_reselect();
    bool staged = false;
    for (uint32_t id : back.stage_in) {
        if (id == 4) staged = true;
    }
    CHECK(staged);
    rt.finish_reselect();
    CHECK(rt.store().blocks()[4].gpu_slot >= 0);
    const auto &payload = be.gpu[rt.store().blocks()[4].gpu_slot];
    CHECK(!payload.empty());
    // Original GPU slot for block 4 was 4, filled with byte 5.
    CHECK(payload[0] == 5);
}

#endif

static void test_selection_preview_and_resident_commit() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 3 + 7);
    for (uint32_t id = 0; id < rt.store().block_count(); ++id) {
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    const auto before = rt.store().blocks();
    const auto selected = rt.preview_reselect();
    CHECK(selected.size() == 4);
    CHECK(be.ops.empty());
    for (uint32_t id = 0; id < before.size(); ++id) {
        const auto & after = rt.store().blocks()[id];
        CHECK(after.gpu_slot == before[id].gpu_slot);
        CHECK(after.in_working_set == before[id].in_working_set);
        CHECK(after.baked_pos == before[id].baked_pos);
    }
    CHECK(!rt.commit_resident_selection({0, 1, 2}));
    CHECK(!rt.store().blocks()[0].in_working_set);
    CHECK(rt.commit_resident_selection(selected));
    CHECK(be.ops.empty());
    CHECK(rt.last_plan().total_window_tokens == 103);
    CHECK(rt.store().blocks().back().n_tokens == 7);
    auto pending = rt.prepare_selection(selected);
    CHECK(!rt.commit_resident_selection(selected));
    rt.finish_reselect();
    CHECK(rt.commit_resident_selection(selected));
}

// Drain contract the adapter's host-store swap depends on: an empty selection
// stages out every GPU-resident block and returns every slot, and re-selecting
// the same set afterwards asks for all of them back.
static void test_full_drain_then_restage() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 3 + 7);
    const uint32_t n = rt.store().block_count();
    CHECK(n == 4);
    std::vector<uint32_t> resident;
    for (uint32_t id = 0; id < n; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
        resident.push_back(id);
    }
    be.ops.clear();
    be.allocs.clear();
    be.frees.clear();

    const auto drain = rt.prepare_selection({});
    CHECK(drain.stage_out.size() == n);
    CHECK(drain.stage_in.empty());
    CHECK(drain.total_window_tokens == 0);
    CHECK(rt.pending());
    rt.finish_reselect();
    CHECK(!rt.pending());
    CHECK(be.frees.size() == n);
    CHECK(be.allocs.empty());
    for (const auto &b : rt.store().blocks()) {
        CHECK(b.tier != KvTier::GPU);
        CHECK(!b.in_working_set);
        // A detached store must not name a slot another conversation now
        // holds. set_block_tier clears it on the way off GPU.
        CHECK(b.gpu_slot == -1);
    }
    CHECK(rt.store().total_tokens() == 32 * 3 + 7);

    be.ops.clear();
    be.frees.clear();
    const auto restage = rt.prepare_selection(resident);
    CHECK(restage.stage_in.size() == n);
    CHECK(restage.stage_out.empty());
    for (uint32_t i = 0; i < n; ++i) {
        CHECK(restage.stage_in[i] == resident[i]);
    }
    rt.finish_reselect();
    CHECK(be.allocs.size() == n);
    CHECK(be.frees.empty());
    for (uint32_t i = 0; i < n; ++i) {
        const auto &b = rt.store().blocks()[resident[i]];
        CHECK(b.tier == KvTier::GPU);
        CHECK(b.gpu_slot == be.allocs[i]);
    }

    // The adapter's post-drain guard demotes a block that somehow still names
    // a GPU slot instead of only clearing the slot, because tier GPU with
    // gpu_slot -1 is read three incompatible ways downstream (stage_in wants a
    // fresh slot, resident_tokens counts it as absent, set_selection will not
    // stage it out again). Pin what that demotion does: the slot goes, the
    // lower-tier handles stay.
    const KvMemBlock still_resident = rt.store().blocks()[0];
    CHECK(still_resident.tier == KvTier::GPU && still_resident.gpu_slot >= 0);
    rt.store().set_block_tier(0, KvTier::CPU, still_resident.cpu_slot, still_resident.nvme_slot);
    const KvMemBlock &demoted = rt.store().blocks()[0];
    CHECK(demoted.tier == KvTier::CPU && demoted.gpu_slot == -1);
    CHECK(demoted.cpu_slot == still_resident.cpu_slot);
    CHECK(demoted.nvme_slot == still_resident.nvme_slot);
}

// A prepared plan the caller abandons instead of applying, which is what the
// adapter's store swap does when the drain behind it throws. The pending
// marker must go with the plan: while it is set, conv_can_drain refuses the
// next swap with plan_pending and the fast resident-commit path stays closed.
static void test_discard_pending_plan() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 3 + 7);
    for (uint32_t id = 0; id < rt.store().block_count(); ++id) {
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    const auto selected = rt.preview_reselect();
    CHECK(rt.commit_resident_selection(selected));
    be.frees.clear();

    rt.prepare_selection(selected);
    CHECK(rt.pending());
    CHECK(!rt.commit_resident_selection(selected));
    rt.discard_pending();
    CHECK(!rt.pending());
    CHECK(rt.commit_resident_selection(selected));
    // The abandoned plan is gone, not deferred: applying its second half now
    // hands nothing back through the backend, because the slots it had queued
    // belong to a caller that rebuilds its whole free-slot list.
    rt.admit_incoming();
    CHECK(be.frees.empty());
}

int main() {
    test_selection_preview_and_resident_commit();
    test_full_drain_then_restage();
    test_discard_pending_plan();
    test_stage_out_before_stage_in();
    test_high_overlap_skips_stage_in();
    test_pressure_keeps_sink_and_tail();
    test_maybe_offload_evicts_before_stage_in();
#if KVMEM_ENABLE_NVME
    test_cpu_full_spills_to_nvme_and_roundtrips();
#endif
    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
