#include "memory_test_support.hpp"
#include "kvmem/kvmem_store.hpp"
#include <algorithm>
#include <map>
#include <type_traits>

static_assert(std::is_abstract<MemoryBackend>::value, "production backend must implement the contract");
static_assert(std::is_abstract<MemoryTransfer>::value, "transfer operations cannot default to success");

struct Witness { int prepares = 0, aborts = 0, releases = 0, publishes = 0; bool destroyed = false; };
struct TestBackend : MemoryBackend {
    MemoryDescriptor desc;
    MemorySnapshot state;
    NativeKvArchive host;
    std::map<uint64_t, NativeKvPayload> device;
    std::vector<uint8_t> recurrent_state = {9, 8, 7, 6, 5};
    std::shared_ptr<Witness> witness = std::make_shared<Witness>();
    bool complete = false, fail_poll = false, fail_restore_start = false, fail_publish = false;
    bool fail_prepare = false, null_prepare = false, busy = false;

    explicit TestBackend(int profile) : desc(::descriptor(profile)),
        host(desc.identity, {42, 3}, desc.layout, desc.layout.record_bytes * 6) {
        state.stamp = {true, {42, 3}, 1, 0, uint64_t(desc.layout.block_tokens) * 5 + 1, 91};
        for (uint64_t i = 0; i < 6; ++i) {
            auto b = block(desc, i, i == 5 ? 1 : 0);
            state.blocks.push_back({b, i >= 4 ? 1u : 0u});
            if (i < 4) { device.emplace(i, payload(desc, b)); state.resident.push_back(i); }
            else host.put(payload(desc, b));
        }
    }
    ~TestBackend() override { CHECK(!busy); witness->destroyed = true; }
    MemoryDescriptor descriptor() const override { return desc; }
    MemorySnapshot snapshot() const override { return state; }
    MemoryStamp stamp() const noexcept override { return state.stamp; }
    void invalidate() noexcept override { state.stamp.valid = false; }
    MemoryBudget budget() const {
        const uint64_t bt = desc.layout.block_tokens;
        return {bt * 3, bt, bt * 4, desc.layout.record_bytes * 6};
    }
    struct Transfer : MemoryTransfer {
        TestBackend & be;
        WorkingSetPlan plan;
        NativeKvArchive previous_host;
        bool spilling = false, restoring = false, spill_done = false, destructive = false, done = false;
        Transfer(TestBackend & backend, const WorkingSetPlan & p)
            : be(backend), plan(p), previous_host(backend.host) {}
        ~Transfer() override { CHECK(done); }
        void start_spill() override { CHECK(!spilling); spilling = true; }
        TransferPoll poll() override {
            CHECK(spilling || restoring);
            if (be.fail_poll) return {TransferStatus::Failed, "injected copy failure"};
            if (!be.complete) return {TransferStatus::Pending, {}};
            be.complete = false;
            if (spilling) {
                for (const auto & b : plan.spill) be.host.put(be.device.at(b.id));
                spill_done = true;
            } else {
                for (const auto & b : plan.restore) be.device.emplace(b.id, be.host.get(b));
                CHECK(be.device.size() <= 4); // no hidden full-history device mirror
            }
            return {TransferStatus::Complete, {}};
        }
        void start_restore() override {
            CHECK(spill_done);
            spilling = false;
            restoring = true;
            destructive = !plan.evict.empty() || !plan.restore.empty();
            for (const auto & b : plan.evict) { be.device.erase(b.id); ++be.witness->releases; }
            if (be.fail_restore_start) throw MemoryError(MemoryErrorCode::TransferFailed, "restore launch failed");
        }
        void publish(const ExecutionView & view) override {
            be.state.stamp = view.stamp;
            if (be.fail_publish) throw MemoryError(MemoryErrorCode::TransferFailed, "publication failed");
            be.state.resident.clear();
            for (const auto & b : view.blocks) be.state.resident.push_back(b.block.id);
            for (auto & entry : be.state.blocks) {
                try { be.host.get(entry.block); entry.host_version = entry.block.content_version; }
                catch (const MemoryError &) { entry.host_version = 0; }
            }
            ++be.witness->publishes;
            done = true;
            be.busy = false;
        }
        AbortResult abort() noexcept override {
            ++be.witness->aborts;
            // A fixture retains a host backup until the destructive boundary;
            // it deliberately cannot reconstruct the old device view afterwards.
            if (!destructive) be.host = std::move(previous_host);
            done = true;
            be.busy = false;
            return destructive ? AbortResult::SessionInvalidated : AbortResult::PreviousViewPreserved;
        }
    };
    std::unique_ptr<MemoryTransfer> prepare(const WorkingSetPlan & p) override {
        if (busy) throw MemoryError(MemoryErrorCode::Pending, "engine already has a transfer owner");
        if (fail_prepare) throw MemoryError(MemoryErrorCode::BudgetExceeded, "native reservation failed");
        if (null_prepare) return nullptr;
        auto ticket = std::make_unique<Transfer>(*this, p);
        ++witness->prepares;
        busy = true;
        return ticket;
    }
};

static void ready(MemorySession & session, TestBackend & be) {
    CHECK(session.advance() == MemoryPhase::Spilling);
    be.complete = true;
    CHECK(session.advance() == MemoryPhase::Restoring);
    be.complete = true;
    CHECK(session.advance() == MemoryPhase::Ready);
}
static void success(int profile) {
    auto be = std::make_shared<TestBackend>(profile);
    MemorySession session(be);
    const auto old = be->stamp();
    const auto recurrent = be->recurrent_state;
    const auto & p = session.prepare(old, {5, 0, 4}, {0, 5}, be->budget());
    CHECK(p.spill.size() == 3 && p.evict.size() == 3 && p.restore.size() == 2);
    CHECK(p.host_payload_bytes_required == 5 * be->desc.layout.record_bytes);
    CHECK(p.device_tokens_required == 4 * be->desc.layout.block_tokens);
    CHECK(p.next.blocks[0].block.id == 0 && p.next.blocks[1].block.id == 4 && p.next.blocks[2].block.id == 5);
    CHECK(p.next.blocks[1].block.original_start == 4 * be->desc.layout.block_tokens);
    CHECK(p.next.blocks[1].compact_start == be->desc.layout.block_tokens);
    CHECK(p.next.visible_tokens == 2 * be->desc.layout.block_tokens + 1);
    CHECK(p.next.stamp.frontier == old.frontier);
    rejects(MemoryErrorCode::Pending, [&] { session.publish(); });
    rejects(MemoryErrorCode::Pending, [&] { session.prepare(old, {0}, {}, be->budget()); });
    CHECK(session.advance() == MemoryPhase::Spilling);
    CHECK(session.advance() == MemoryPhase::Spilling);
    CHECK(be->witness->releases == 0 && be->device.size() == 4 && be->stamp() == old);
    be->complete = true;
    CHECK(session.advance() == MemoryPhase::Restoring);
    CHECK(be->witness->releases == 3 && be->device.size() == 1);
    rejects(MemoryErrorCode::Pending, [&] { session.publish(); });
    CHECK(session.advance() == MemoryPhase::Restoring);
    be->complete = true;
    CHECK(session.advance() == MemoryPhase::Ready);
    CHECK(be->stamp() == old);
    const auto view = session.publish();
    CHECK(view.stamp.view_version == old.view_version + 1 && be->stamp() == view.stamp);
    CHECK(be->state.resident == std::vector<uint64_t>({0, 4, 5}));
    for (auto id : be->state.resident)
        CHECK(be->device.at(id).bytes() == payload(be->desc, be->state.blocks[id].block).bytes());
    CHECK(be->recurrent_state == recurrent && !be->busy);
    // A second selection restores the just-evicted original bytes via the same pool.
    session.prepare(be->stamp(), {0, 1, 2}, {0}, be->budget());
    ready(session, *be);
    session.publish();
    CHECK(be->device.at(1).bytes() == payload(be->desc, be->state.blocks[1].block).bytes());
    CHECK(be->witness->publishes == 2);
}
static void planning_failures() {
    auto be = std::make_shared<TestBackend>(2);
    MemorySession s(be);
    const auto prepare = [&](std::vector<uint64_t> ids, std::vector<uint64_t> mandatory, MemoryBudget budget) {
        s.prepare(be->stamp(), ids, mandatory, budget);
    };
    auto budget = be->budget();
    rejects(MemoryErrorCode::InvalidPlan, [&] { prepare({0, 0}, {}, budget); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { prepare({99}, {}, budget); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { prepare({0, 4}, {5}, budget); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { prepare({0, 4}, {0, 0}, budget); });
    auto small = budget; small.host_payload_bytes = be->desc.layout.record_bytes * 5 - 1;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { prepare({0, 4, 5}, {}, small); });
    small = budget; small.device_tokens = 192;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { prepare({0, 4, 5}, {}, small); });
    small = budget; small.selected_tokens = 64;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { prepare({0, 4, 5}, {}, small); });
    small = budget; small.selected_tokens = 65;
    rejects(MemoryErrorCode::InvalidPlan, [&] { prepare({0}, {}, small); });
    small = budget; small.reserve_tokens = 65; // reserve rounds to two pages
    rejects(MemoryErrorCode::BudgetExceeded, [&] { prepare({0, 4, 5}, {}, small); });
    be->state.blocks[4].host_version = 0;
    rejects(MemoryErrorCode::Stale, [&] { prepare({0, 4}, {}, budget); });
    rejects(MemoryErrorCode::Stale, [&] { prepare({0}, {}, budget); }); // cannot silently lose cold history
    be->state.blocks[4].host_version = 1;
    auto stale = be->stamp(); ++be->state.stamp.history_revision;
    rejects(MemoryErrorCode::Stale, [&] { s.prepare(stale, {0}, {}, budget); });
    CHECK(be->witness->prepares == 0 && be->device.size() == 4 && !be->busy);
    auto desc = be->desc; desc.layout.block_tokens = 32;
    rejects(MemoryErrorCode::Unsupported, [&] { desc.validate(); });
    desc = be->desc; desc.capabilities.sparse_view = false;
    rejects(MemoryErrorCode::Unsupported, [&] { desc.validate(); });
    be->fail_prepare = true;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { prepare({0, 4, 5}, {}, budget); });
    be->fail_prepare = false; be->null_prepare = true;
    rejects(MemoryErrorCode::TransferFailed, [&] { prepare({0, 4, 5}, {}, budget); });
    CHECK(s.phase() == MemoryPhase::Idle && be->stamp().valid);
}
static void no_op_and_park() {
    auto be = std::make_shared<TestBackend>(0);
    MemorySession s(be);
    auto budget = be->budget(); budget.selected_tokens = 128; budget.reserve_tokens = 0;
    const auto & same = s.prepare(be->stamp(), {0, 1, 2, 3}, {}, budget);
    CHECK(same.spill.empty() && same.evict.empty() && same.restore.empty());
    ready(s, *be); s.publish();
    CHECK(be->witness->releases == 0 && be->device.size() == 4);
    const auto & park = s.prepare(be->stamp(), {}, {}, budget);
    CHECK(park.next.visible_tokens == 0 && park.evict.size() == 4);
    ready(s, *be); s.publish();
    CHECK(be->device.empty() && be->host.bytes() == 6 * be->desc.layout.record_bytes);
    s.prepare(be->stamp(), {0, 4, 5}, {}, be->budget());
    ready(s, *be); s.publish();
    CHECK(be->device.size() == 3 && be->stamp().frontier == 161);
}
static void failure_lifecycle() {
    for (int failure = 0; failure < 6; ++failure) {
        auto be = std::make_shared<TestBackend>(0);
        MemorySession s(be);
        s.prepare(be->stamp(), {0, 4, 5}, {}, be->budget());
        if (failure == 0) { // stale plan before destructive work
            ++be->state.stamp.history_revision;
            rejects(MemoryErrorCode::Stale, [&] { s.advance(); });
            CHECK(s.phase() == MemoryPhase::Idle && be->device.size() == 4);
        } else if (failure == 1) { // failed outgoing transfer preserves source pages
            s.advance(); be->fail_poll = true;
            rejects(MemoryErrorCode::TransferFailed, [&] { s.advance(); });
            CHECK(s.phase() == MemoryPhase::Idle && be->witness->releases == 0);
        } else if (failure == 2) {
            s.advance(); be->complete = true; be->fail_restore_start = true;
            rejects(MemoryErrorCode::TransferFailed, [&] { s.advance(); });
        } else if (failure == 3) {
            s.advance(); be->complete = true; s.advance(); be->fail_poll = true;
            rejects(MemoryErrorCode::TransferFailed, [&] { s.advance(); });
        } else if (failure == 4) {
            ready(s, *be); be->fail_publish = true;
            rejects(MemoryErrorCode::TransferFailed, [&] { s.publish(); });
        } else {
            be->invalidate();
            rejects(MemoryErrorCode::InvalidSession, [&] { s.advance(); });
        }
        CHECK(!be->busy && be->witness->aborts == 1 && s.pending_plan() == nullptr);
        if (failure >= 2) {
            CHECK(s.phase() == MemoryPhase::Invalid && !be->stamp().valid);
            rejects(MemoryErrorCode::InvalidSession, [&] { s.prepare(be->stamp(), {0}, {}, be->budget()); });
        }
    }
    auto be = std::make_shared<TestBackend>(0);
    const auto witness = be->witness;
    {
        MemorySession s(be);
        s.prepare(be->stamp(), {0, 4, 5}, {}, be->budget());
        s.advance();
        be.reset(); // session keeps the backend alive until abort drains outstanding work
        CHECK(!witness->destroyed);
    }
    CHECK(witness->destroyed && witness->aborts == 1);
    be = std::make_shared<TestBackend>(0);
    MemorySession s(be);
    s.prepare(be->stamp(), {0, 4, 5}, {}, be->budget());
    s.disconnect();
    CHECK(!be->busy && be->stamp().valid);
    rejects(MemoryErrorCode::Disconnected, [&] { s.advance(); });
    rejects(MemoryErrorCode::Disconnected, [&] { MemorySession absent(nullptr); });
    {
        auto cancel_backend = std::make_shared<TestBackend>(1);
        MemorySession cancel(cancel_backend);
        cancel.prepare(cancel_backend->stamp(), {0, 4, 5}, {}, cancel_backend->budget());
        cancel.advance(); cancel_backend->complete = true; cancel.advance();
        CHECK(cancel.abort() == AbortResult::SessionInvalidated);
        CHECK(!cancel_backend->busy && !cancel_backend->stamp().valid);
    }
}
static void policy_and_checkpoint() {
    auto be = std::make_shared<TestBackend>(0);
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32; cfg.select_budget = 96; cfg.sink_blocks = 1; cfg.recent_blocks = 1;
    cfg.select_method = KvMemMethod::Recency;
    KvMemStore policy(cfg);
    policy.register_append(161);
    const auto before = policy.blocks();
    MemorySession s(be);
    const auto & p = s.prepare_reselect(be->stamp(), policy, {0, 5}, be->budget());
    const auto expected = policy.pick_topk_blocks({0, 5});
    CHECK(p.next.blocks.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) CHECK(p.next.blocks[i].block.id == expected[i]);
    for (size_t i = 0; i < before.size(); ++i) {
        CHECK(policy.blocks()[i].baked_pos == before[i].baked_pos);
        CHECK(policy.blocks()[i].in_working_set == before[i].in_working_set);
        CHECK(policy.blocks()[i].remap_count == before[i].remap_count);
    }
    CHECK(s.abort() == AbortResult::PreviousViewPreserved);
    auto retrieval_cfg = cfg; retrieval_cfg.select_method = KvMemMethod::Retrieval;
    KvMemStore retrieval(retrieval_cfg); retrieval.register_append(161);
    rejects(MemoryErrorCode::Unsupported, [&] { s.prepare_reselect(be->stamp(), retrieval, {}, be->budget()); });
    policy.register_append(1);
    rejects(MemoryErrorCode::Stale, [&] { s.prepare_reselect(be->stamp(), policy, {}, be->budget()); });
    CheckpointIdentity ck{be->desc.identity, be->desc.layout, be->stamp()};
    CHECK(ck.compatible_with(ck));
    auto invalid = ck; invalid.layout.version = 0;
    CHECK(!invalid.compatible_with(invalid));
    for (int i = 0; i < 7; ++i) {
        auto changed = ck;
        if (i == 0) changed.backend.model = "different-model";
        if (i == 1) ++changed.layout.version;
        if (i == 2) ++changed.stamp.session.incarnation;
        if (i == 3) ++changed.stamp.execution_history;
        if (i == 4) ++changed.stamp.frontier;
        if (i == 5) ++changed.stamp.view_version;
        if (i == 6) changed.stamp.valid = false;
        CHECK(!ck.compatible_with(changed));
    }
}
int main() {
    for (int profile = 0; profile < 3; ++profile) success(profile);
    planning_failures();
    no_op_and_park();
    failure_lifecycle();
    policy_and_checkpoint();
    std::puts("memory contract: three native layouts, publication, rollback and policy checks passed");
}
