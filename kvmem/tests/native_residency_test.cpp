#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/memory_contract.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace kvmem;

static void check(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what);
}
template<class F> static void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "expected rejection");
}

static void unchanged(const KvMemStore & store, const std::vector<KvMemBlock> & before,
                      bool membership = true) {
    check(store.block_count() == before.size(), "catalog changed");
    for (size_t i = 0; i < before.size(); ++i) {
        const auto & a = before[i];
        const auto & b = store.blocks()[i];
        check(a.baked_pos == b.baked_pos && a.remap_count == b.remap_count &&
              a.remap_abs_delta == b.remap_abs_delta, "native path changed re-RoPE bookkeeping");
        check(a.orig_pos_start == b.orig_pos_start && a.n_tokens == b.n_tokens,
              "native path changed logical positions");
        if (membership) check(a.in_working_set == b.in_working_set, "planning published membership");
    }
}

// Independently check the physical set equations, including blocks with slots
// that have never been selected and appended blocks with no evaluated KV yet.
static void selection_properties() {
    std::mt19937 random(23);
    for (bool reuse : {false, true}) {
        for (unsigned trial = 0; trial < 200; ++trial) {
            KvMemStoreConfig cfg;
            cfg.block_tokens = 32;
            cfg.optimize_stage_in = reuse;
            cfg.immutable_source_k = true;
            KvMemStore store(cfg);
            store.register_append(32 * 7 + 5);
            std::vector<uint32_t> selected, resident;
            for (uint32_t id = 0; id < store.block_count(); ++id) {
                if (random() % 2) {
                    store.set_block_gpu_slot(id, static_cast<int32_t>(id));
                    resident.push_back(id);
                }
                if (random() % 2) selected.push_back(id);
                store.record_block_rerope(id, 10000 - id * 31);
            }
            const auto before = store.blocks();
            std::shuffle(selected.begin(), selected.end(), random);
            const auto plan = store.plan_native_selection(selected);
            std::sort(selected.begin(), selected.end());
            check(plan.selected == selected, "selection order");
            uint32_t tokens = 0, kept = 0;
            for (uint32_t id = 0; id < store.block_count(); ++id) {
                const bool r = std::find(resident.begin(), resident.end(), id) != resident.end();
                const bool s = std::find(selected.begin(), selected.end(), id) != selected.end();
                const bool out = std::find(plan.stage_out.begin(), plan.stage_out.end(), id) != plan.stage_out.end();
                const bool in = std::find(plan.stage_in.begin(), plan.stage_in.end(), id) != plan.stage_in.end();
                check(out == (r && (!s || !reuse)), "incorrect eviction");
                check(in == (s && (!r || !reuse)), "incorrect admission");
                kept += r && s && reuse;
                if (s) tokens += store.blocks()[id].n_tokens;
            }
            check(plan.gpu_reused_blocks == kept && plan.total_window_tokens == tokens, "accounting");
            unchanged(store, before);
            rejects([&] { store.plan_native_selection({0, 0}); });
            rejects([&] { store.plan_native_selection({store.block_count()}); });
            unchanged(store, before);
        }
    }
    rejects([] { plan_residency({1, 1}, {1}); });
    rejects([] { plan_residency({1}, {2, 2}); });
}

struct Slots : KvMemBackend {
    static constexpr size_t bytes = 64;
    std::array<std::array<uint8_t, bytes>, 4> data{};
    std::array<bool, 4> used{};
    std::vector<char> events;
    bool fail_alloc = false, fail_restore = false;
    int32_t alloc_gpu_slot() override {
        if (fail_alloc) return -1;
        for (int32_t i = 0; i < 4; ++i) if (!used[i]) {
            used[i] = true; events.push_back('A'); return i;
        }
        return -1;
    }
    void free_gpu_slot(int32_t slot) override {
        check(slot >= 0 && slot < 4 && used[slot], "double or invalid free");
        used[slot] = false; events.push_back('F');
    }
    void copy_block_to_host(uint32_t, int32_t slot, void * dst, uint64_t n) override {
        check(n == bytes && used[slot], "invalid spill");
        std::memcpy(dst, data[slot].data(), bytes); events.push_back('S');
    }
    void copy_block_from_host(uint32_t, int32_t slot, const void * src, uint64_t n) override {
        check(n == bytes && used[slot], "invalid restore");
        if (fail_restore) throw std::runtime_error("injected copy failure");
        std::memcpy(data[slot].data(), src, bytes); events.push_back('R');
    }
};

static void native_bytes_and_detach(bool reuse) {
    Slots first, second;
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 32;
    cfg.store.select_budget = 96;
    cfg.store.prefill_budget = 96;
    cfg.store.sink_blocks = 1;
    cfg.store.recent_blocks = 1;
    cfg.store.optimize_stage_in = reuse;
    cfg.store.estimated_block_bytes = Slots::bytes;
    cfg.cpu_bytes = Slots::bytes * 8;
    KvMemRuntime rt(cfg, &first);
    rt.register_append(101);
    for (uint32_t id = 0; id < 4; ++id) {
        const auto slot = first.alloc_gpu_slot();
        rt.store().set_block_gpu_slot(id, slot);
        for (size_t j = 0; j < Slots::bytes; ++j) first.data[slot][j] = uint8_t(id * 61 + j);
        rt.store().record_block_rerope(id, 1700 + id);
    }
    const auto before = rt.store().blocks();
    rt.prepare_native_selection({0, 3});
    unchanged(rt.store(), before);
    rejects([&] { rt.prepare_native_selection({1, 2}); });
    rejects([&] { rt.prepare_selection({1, 2}); });
    rejects([&] { rt.rebind_backend(&second); });
    rt.discard_pending();
    unchanged(rt.store(), before);

    for (const std::vector<uint32_t> selected : {std::vector<uint32_t>{0, 3}, {1, 2}, {0, 2, 3}, {}}) {
        rt.prepare_native_selection(selected);
        first.events.clear();
        rt.spill_outgoing();
        check(std::find(first.events.begin(), first.events.end(), 'F') == first.events.end(),
              "spill reclaimed source before completion");
        rt.admit_incoming();
        bool allocated = false;
        for (char event : first.events) {
            if (event == 'A') allocated = true;
            if (event == 'F') check(!allocated, "free after allocation");
        }
        for (auto id : selected) {
            const auto & block = rt.store().blocks()[id];
            check(block.gpu_slot >= 0 && block.in_working_set, "missing selected slot");
            for (size_t j = 0; j < Slots::bytes; ++j)
                check(first.data[block.gpu_slot][j] == uint8_t(id * 61 + j), "native bytes changed");
        }
        unchanged(rt.store(), before, false);
    }
    // A failed admission must not discard its only host copy or leak a slot.
    for (bool allocation_failure : {true, false}) {
        first.fail_alloc = allocation_failure;
        first.fail_restore = !allocation_failure;
        const auto host_slot = rt.store().blocks()[0].cpu_slot;
        check(host_slot >= 0, "missing host source before failure");
        rt.prepare_native_selection({0});
        rt.spill_outgoing();
        rejects([&] { rt.admit_incoming(); });
        check(rt.pending() && rt.store().blocks()[0].cpu_slot == host_slot &&
              rt.store().blocks()[0].gpu_slot < 0 && !rt.store().blocks()[0].in_working_set,
              "failed admission changed source ownership");
        check(std::none_of(first.used.begin(), first.used.end(), [](bool v) { return v; }), "failure leaked a slot");
        rt.discard_pending();
    }
    first.fail_alloc = first.fail_restore = false;
    rt.rebind_backend(nullptr);
    rt.rebind_backend(&second);
    rt.prepare_native_selection({0, 1, 2, 3});
    rt.finish_reselect();
    for (const auto & b : rt.store().blocks()) {
        for (size_t j = 0; j < Slots::bytes; ++j)
            check(second.data[b.gpu_slot][j] == uint8_t(b.block_id * 61 + j), "detach/attach changed bytes");
    }
    unchanged(rt.store(), before, false);
    // Pressure planning includes appended rows with no native bytes yet.
    rt.register_append(32);
    check(rt.maybe_offload_native_during_prefill(32, 101, 128, {3, 4}), "expected pressure");
    rt.finish_reselect();
    check(rt.store().blocks()[4].gpu_slot >= 0, "new prefill block not admitted");
    rt.truncate_to(0);
    check(std::none_of(second.used.begin(), second.used.end(), [](bool v) { return v; }), "slot leak");
}

int main() {
    try {
        selection_properties();
        native_bytes_and_detach(false);
        native_bytes_and_detach(true);
        std::cout << "native residency: positions, bytes, pressure, reload and detach passed\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
