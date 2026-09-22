// Decision table for the multi-conversation host-store policy, plus the LRU
// and byte accounting the caps rest on. Freestanding by design, so it runs on
// a machine with no CUDA toolchain and no llama.cpp checkout:
//
//   c++ -std=c++17 -I tools -o /tmp/cst tests/conversation-store-test.cpp && /tmp/cst
//
#include "kvmem-conversation-store.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

static const char * g_row = "-";

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d [%s]: %s\n", \
        __LINE__, g_row, #x); std::abort(); } } while (0)

static const uint64_t MiB = 1ull << 20;
static const uint64_t GiB = 1ull << 30;

static kvmem_store_match store_fixture(int id, int rows, int last_n_gen,
                                       std::vector<int> ckpt_rows, uint64_t bytes, uint64_t used) {
    kvmem_store_match match;
    match.id         = id;
    match.rows       = rows;
    match.live_row   = rows;
    match.last_n_gen = last_n_gen;
    match.ckpt_rows  = std::move(ckpt_rows);
    match.bytes      = bytes;
    match.used       = used;
    return match;
}

static const kvmem_store_match * store_by_id(const std::vector<kvmem_store_match> & stores, int id) {
    for (const auto & store : stores) {
        if (store.id == id) {
            return &store;
        }
    }
    return nullptr;
}

// Every row below goes through here, so the invariants are asserted once per
// decision rather than restated per case.
static kvmem_store_plan select(const std::vector<kvmem_store_match> & stores, int eval_end, bool spec_ok,
                               const std::string & client_id, const kvmem_store_limits & limits) {
    const kvmem_store_plan plan = kvmem_store_select(stores, eval_end, spec_ok, client_id, limits);
    CHECK(plan.reason != nullptr && *plan.reason);
    CHECK(plan.keep <= plan.lcp);
    if (plan.action == kvmem_store_action::extend) {
        CHECK(plan.id >= 0);
        // keep_cap can be negative on a short prompt, so this only holds where
        // a checkpoint was actually selected.
        const kvmem_store_match * chosen = store_by_id(stores, plan.id);
        CHECK(chosen != nullptr);
        CHECK(plan.keep > 0 && plan.keep <= kvmem_store_keep_cap(*chosen, eval_end, spec_ok));
    } else {
        // A fork always asks the caller to allocate; nothing is reset in place.
        CHECK(plan.id == -1 && plan.keep == 0);
    }
    uint64_t previous_used = 0;
    for (size_t i = 0; i < plan.evict.size(); ++i) {
        const int id = plan.evict[i];
        CHECK(id != plan.id);
        // The caller cannot execute an eviction of the store the GPU working
        // set is bound to, so a plan must never ask for one.
        CHECK(id != limits.attached);
        const kvmem_store_match * victim = store_by_id(stores, id);
        CHECK(victim != nullptr);
        CHECK(i == 0 || victim->used >= previous_used); // ascending by recency
        previous_used = victim->used;
        for (size_t j = 0; j < i; ++j) {
            CHECK(plan.evict[j] != id); // unique
        }
    }
    const bool allocating = plan.action == kvmem_store_action::fresh && plan.id < 0;
    const int live = (int) stores.size() - (int) plan.evict.size() + (allocating ? 1 : 0);
    const int protected_stores = (plan.id >= 0 ? 1 : 0) +
            (limits.attached >= 0 && limits.attached != plan.id &&
             store_by_id(stores, limits.attached) != nullptr ? 1 : 0);
    const int victims_left = (int) stores.size() - protected_stores - (int) plan.evict.size();
    CHECK(live <= limits.max_stores || victims_left == 0);
    return plan;
}

static void test_decision_table() {
    const kvmem_store_limits many{4, 0};

    kvmem_store_match A  = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
    kvmem_store_match B  = store_fixture(2, 9000, 100, {0, 8900}, 2 * GiB, 20);
    kvmem_store_match B2 = store_fixture(2, 9000, 100, {0, 200, 8900}, 2 * GiB, 20);
    kvmem_store_match A2 = store_fixture(1, 4000, 200, {0, 1000}, 1 * GiB, 10);
    kvmem_store_match A3 = store_fixture(1, 4000, 200, {}, 1 * GiB, 10);
    kvmem_store_match C  = store_fixture(3, 500, 0, {400}, 64 * MiB, 5);

    {
        g_row = "T1 empty table";
        const kvmem_store_plan plan = select({}, 1000, false, "", many);
        CHECK(plan.action == kvmem_store_action::fresh);
        CHECK(plan.id == -1 && plan.lcp == 0 && plan.keep == 0);
        CHECK(plan.evict.empty());
        CHECK(std::string(plan.reason) == "new_conversation");
    }
    {
        g_row = "T2 continuation resumes the newest checkpoint";
        A.lcp = 3950;
        const kvmem_store_plan plan = select({A}, 4300, false, "", many);
        CHECK(plan.action == kvmem_store_action::extend);
        CHECK(plan.id == 1 && plan.lcp == 3950 && plan.keep == 3800);
        CHECK(plan.evict.empty());
    }
    {
        g_row = "T3 keep_cap boundary with spec";
        A.lcp = 3950;
        // keep_cap = min(3950, 4000, 3801) = 3801, and the checkpoint at 3800 fits.
        kvmem_store_plan plan = select({A}, 3801, true, "", many);
        CHECK(plan.action == kvmem_store_action::extend && plan.keep == 3800);
        // keep_cap == the checkpoint row exactly: the comparison is <=, not <.
        plan = select({A}, 3800, true, "", many);
        CHECK(plan.action == kvmem_store_action::extend && plan.keep == 3800);
        // One row earlier and only row 0 survives the cap, which is nothing to
        // resume, so the request forks instead of truncating A.
        plan = select({A}, 3799, true, "", many);
        CHECK(plan.action == kvmem_store_action::fresh);
        CHECK(plan.id == -1 && plan.keep == 0);
        CHECK(std::string(plan.reason) == "fork_zero_keep");
    }
    {
        g_row = "T4 zero keep";
        A.lcp = 3950;
        // keep_cap = 99, so only the row-0 checkpoint fits and there is no
        // reason to truncate A to nothing.
        const kvmem_store_plan plan = select({A}, 100, false, "", many);
        CHECK(plan.action == kvmem_store_action::fresh);
        CHECK(plan.id == -1 && plan.keep == 0);
        CHECK(std::string(plan.reason) == "fork_zero_keep");
        CHECK(plan.evict.empty());
    }
    {
        g_row = "T5 no recurrent checkpoint is a miss whatever the prefix";
        A3.lcp = 3950;
        const kvmem_store_plan plan = select({A3}, 4300, false, "", many);
        CHECK(plan.action == kvmem_store_action::fresh);
        CHECK(plan.id == -1 && plan.evict.empty());
        CHECK(std::string(plan.reason) == "no_recurrent_checkpoint");
    }
    {
        g_row = "T6 shared prefix with no usable checkpoint forks";
        B.lcp = 300;
        const kvmem_store_plan plan = select({B}, 20000, false, "", many);
        CHECK(plan.action == kvmem_store_action::fresh);
        CHECK(plan.id == -1 && plan.keep == 0 && plan.evict.empty());
        CHECK(std::string(plan.reason) == "fork_zero_keep");
    }
    {
        g_row = "T7 a shared system prompt forks instead of truncating";
        B2.lcp = 300;
        // rows - lcp = 8700, far above last_n_gen + 64, so this prompt is a
        // branch of B2 and not its continuation.
        const kvmem_store_plan plan = select({B2}, 20000, false, "", many);
        CHECK(plan.action == kvmem_store_action::fresh);
        CHECK(plan.id == -1 && plan.keep == 0);
        CHECK(std::string(plan.reason) == "fork_shared_prefix");
        // Today's single-store server attaches at the low checkpoint instead
        // and truncates B2 from 9000 rows down to 200. That path does not run
        // through this policy at all, so the transcription below is where the
        // shared arithmetic is pinned; here only the fork decision is.
        int keep = 0;
        CHECK(kvmem_store_checkpoint(B2, kvmem_store_keep_cap(B2, 20000, false), keep));
        CHECK(keep == 200 && !kvmem_store_continuation(B2));
    }
    {
        g_row = "T8 an old checkpoint is still a continuation";
        A2.lcp = 3950;
        // The fork test is about prompt identity (lcp against rows), never
        // about how far back the usable checkpoint sits.
        const kvmem_store_plan plan = select({A2}, 4300, false, "", many);
        CHECK(plan.action == kvmem_store_action::extend);
        CHECK(plan.id == 1 && plan.keep == 1000 && plan.lcp == 3950);
    }
    {
        g_row = "T9 ranking picks the larger keep";
        A.lcp = 3950;
        C.lcp = 500;
        const kvmem_store_plan plan = select({A, C}, 4300, false, "", many);
        CHECK(plan.action == kvmem_store_action::extend);
        CHECK(plan.id == 1 && plan.keep == 3800);
        CHECK(plan.evict.empty());
    }
    {
        g_row = "T10 equal keep breaks toward the more recent store";
        kvmem_store_match older = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
        kvmem_store_match newer = store_fixture(2, 4000, 200, {0, 3800}, 1 * GiB, 20);
        older.lcp = newer.lcp = 3950;
        kvmem_store_plan plan = select({older, newer}, 4300, false, "", many);
        CHECK(plan.action == kvmem_store_action::extend && plan.id == 2);
        plan = select({newer, older}, 4300, false, "", many);
        CHECK(plan.action == kvmem_store_action::extend && plan.id == 2);
    }
    {
        g_row = "T11 a continuation beats a non-continuation at any keep";
        B2.lcp = 300;
        C.lcp = 500;
        const kvmem_store_plan plan = select({B2, C}, 20000, false, "", many);
        CHECK(plan.action == kvmem_store_action::extend);
        CHECK(plan.id == 3 && plan.keep == 400);
    }
    {
        g_row = "T12 count cap evicts exactly one, the least recently used";
        kvmem_store_match a = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
        kvmem_store_match b = store_fixture(2, 9000, 100, {0, 8900}, 2 * GiB, 20);
        kvmem_store_match c = store_fixture(3, 500, 0, {400}, 64 * MiB, 5);
        a.lcp = b.lcp = c.lcp = 0; // nothing to continue, so the plan forks
        const kvmem_store_plan plan = select({a, b, c}, 20000, false, "", {3, 0});
        CHECK(plan.action == kvmem_store_action::fresh && plan.id == -1);
        CHECK(plan.evict == std::vector<int>({3}));
    }
    {
        g_row = "T13 count cap leaves room for the store being allocated";
        std::vector<kvmem_store_match> stores = {
            store_fixture(1, 100, 0, {0}, 1 * MiB, 40),
            store_fixture(2, 100, 0, {0}, 1 * MiB, 10),
            store_fixture(3, 100, 0, {0}, 1 * MiB, 30),
            store_fixture(4, 100, 0, {0}, 1 * MiB, 20),
        };
        const kvmem_store_plan plan = select(stores, 20000, false, "", {2, 0});
        CHECK(plan.action == kvmem_store_action::fresh && plan.id == -1);
        // Three go, not two: the fourth slot is the one being allocated, so
        // four survivors plus the new store would exceed the cap of two.
        CHECK(plan.evict == std::vector<int>({2, 4, 3}));
    }
    {
        g_row = "T14 byte cap stops at <=, not <";
        kvmem_store_match a = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
        kvmem_store_match b = store_fixture(2, 9000, 100, {0, 8900}, 2 * GiB, 20);
        kvmem_store_match c = store_fixture(3, 500, 0, {400}, GiB / 2, 5);
        a.lcp = b.lcp = c.lcp = 0;
        const kvmem_store_plan plan = select({a, b, c}, 20000, false, "", {8, 3 * GiB});
        CHECK(plan.action == kvmem_store_action::fresh);
        // 3.5 GiB over a 3 GiB cap; dropping the least recent reaches exactly
        // the cap, which is not over it.
        CHECK(plan.evict == std::vector<int>({3}));
    }
    {
        g_row = "T15 the plan target is never evicted";
        kvmem_store_match a = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
        a.lcp = 3950;
        const kvmem_store_limits tight{8, 1 * MiB};
        const kvmem_store_plan plan = select({a}, 4300, false, "", tight);
        CHECK(plan.action == kvmem_store_action::extend && plan.id == 1);
        CHECK(plan.evict.empty());
        // A single conversation larger than the cap still runs; the caller only
        // warns, which is today's uncapped behavior rather than a failed request.
        kvmem_store_table table;
        const int id = table.add(7);
        table.set_bytes(id, 1 * GiB);
        CHECK(table.bytes_total() > tight.max_bytes);
        CHECK(table.count() <= tight.max_stores);
    }
    {
        g_row = "T15b the attached store is never evicted";
        // Reported independently by three reviewers. The byte loop used to
        // walk every store but the target, so the store the GPU working set is
        // bound to could be named as a victim. The caller declines that
        // eviction (an attached store is released through the memory object,
        // not through a detached adapter handle) while the table dropped its
        // row anyway, which left the host store alive and unreachable.
        kvmem_store_match attached = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
        kvmem_store_match other    = store_fixture(2, 4000, 200, {0, 3800}, 1 * GiB, 20);
        attached.lcp = 0;    // nothing to continue here
        other.lcp    = 3950; // ... so the request extends the other store
        kvmem_store_limits limits{4, 1 * MiB};
        limits.attached = attached.id;
        kvmem_store_plan plan = select({attached, other}, 4300, false, "", limits);
        CHECK(plan.action == kvmem_store_action::extend && plan.id == other.id);
        // 2 GiB against a 1 MiB cap, and still nothing is evicted: both live
        // stores are spoken for this request. The caller brings the table back
        // inside the cap after the reply, when the outgoing store is parked.
        CHECK(plan.evict.empty());
        // The count cap is bounded the same way.
        kvmem_store_limits counted{1, 0};
        counted.attached = attached.id;
        plan = select({attached, other}, 4300, false, "", counted);
        CHECK(plan.action == kvmem_store_action::extend && plan.id == other.id);
        CHECK(plan.evict.empty());
        // Without the hint the same inputs name the attached store, which is
        // the shape of the bug rather than a decision the caller can execute.
        limits.attached = -1;
        plan = select({attached, other}, 4300, false, "", limits);
        CHECK(plan.evict == std::vector<int>({attached.id}));
    }
    {
        g_row = "T16 a client id restricts the candidate set";
        kvmem_store_match a = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
        kvmem_store_match b = store_fixture(2, 9000, 100, {0, 8900}, 2 * GiB, 20);
        a.client_id = "x";
        b.client_id = "y";
        a.lcp = 300;   // shared prefix only
        b.lcp = 8950;  // genuine continuation
        // The hint never lets a request attach to a store it does not name,
        // even when that store has the longer prefix.
        kvmem_store_plan plan = select({a, b}, 9300, false, "x", many);
        CHECK(plan.action == kvmem_store_action::fresh && plan.id == -1);
        CHECK(std::string(plan.reason) == "fork_zero_keep");
        // A bound id that does continue its store is the cheap path.
        plan = select({a, b}, 9300, false, "y", many);
        CHECK(plan.action == kvmem_store_action::extend && plan.id == 2);
        CHECK(std::string(plan.reason) == "client_id");
        // An unknown id is not an error: it degrades to inferred matching.
        plan = select({a, b}, 9300, false, "z", many);
        CHECK(plan.action == kvmem_store_action::extend && plan.id == 2);
        CHECK(std::string(plan.reason) == "extend");
        // And it still never bypasses the checkpoint rule.
        kvmem_store_match b_no_ckpt = b;
        b_no_ckpt.ckpt_rows.clear();
        plan = select({a, b_no_ckpt}, 9300, false, "y", many);
        CHECK(plan.action == kvmem_store_action::fresh && plan.id == -1);
        CHECK(std::string(plan.reason) == "no_recurrent_checkpoint");
    }
}

// kvmem.conversation_id was an unrecognized key at v0.16.0-rc3, so a request
// carrying one was served. It still must be: a value the policy cannot use as
// a key is dropped here, never reported to the caller as an error.
static void test_client_id_never_fails_a_request() {
    g_row = "client id normalization";
    CHECK(kvmem_store_client_id("abc") == "abc");
    CHECK(kvmem_store_client_id("a-b_c.7:8@9+0~1") == "a-b_c.7:8@9+0~1");
    CHECK(kvmem_store_client_id(std::string(128, 'x')).size() == 128);
    CHECK(kvmem_store_client_id("").empty());
    CHECK(kvmem_store_client_id(std::string(129, 'x')).empty());
    CHECK(kvmem_store_client_id("has space").empty());
    CHECK(kvmem_store_client_id(std::string("nul\0byte", 8)).empty());
    CHECK(kvmem_store_client_id("\x7f").empty());       // DEL
    CHECK(kvmem_store_client_id("caf\xc3\xa9").empty()); // any non-ASCII byte
    // A dropped id is exactly an absent one: both fall back to the token
    // prefix, and neither bypasses the checkpoint rule.
    const kvmem_store_limits many{4, 0};
    kvmem_store_match a = store_fixture(1, 4000, 200, {0, 3800}, 1 * GiB, 10);
    a.lcp = 3950;
    a.client_id = "bound";
    const kvmem_store_plan plan = select({a}, 4300, false,
            kvmem_store_client_id("not printable\n"), many);
    CHECK(plan.action == kvmem_store_action::extend && plan.id == 1);
    CHECK(std::string(plan.reason) == "extend");
}

// Today's checkpoint selection, transcribed from tools/kvmem-multimodal-server.h:246-256
// at v0.16.0-rc3. Default identity is not proved by this transcription but by
// construction: conversation_begin_request returns before the policy when
// --kvmem-conversations is absent, so the single-store server never reaches it.
// What the sweep below does pin is the two helpers the live policy shares with
// that path, kvmem_store_keep_cap and kvmem_store_checkpoint, plus the rule
// that the policy never extends on a row this expression would not select.
static std::pair<bool, int> today_checkpoint(const std::vector<int> & ckpt, int lcp, int live_row,
                                             int eval_end, bool spec) {
    const int keep = std::min({lcp, live_row, eval_end - (spec ? 0 : 1)});
    bool found = false;
    int  row   = 0;
    for (int r : ckpt) {
        if (r <= keep && (!found || r > row)) {
            row   = r;
            found = true;
        }
    }
    return {found, row};
}

static void test_checkpoint_arithmetic() {
    g_row = "T18 checkpoint arithmetic sweep";
    const kvmem_store_limits many{4, 0};
    int rows_checked = 0;
    for (int rows : {0, 1, 500, 4000}) {
        const std::vector<std::vector<int>> checkpoints = {{}, {0}, {0, rows / 2}, {rows}};
        for (const auto & ckpt : checkpoints) {
            for (int gen : {0, 1, 200}) {
                for (int lcp = 0; lcp <= rows; lcp += 7) {
                    for (int eval_end : {0, 1, lcp, rows, rows + 300}) {
                        for (bool spec : {false, true}) {
                            kvmem_store_match store = store_fixture(1, rows, gen, ckpt, 1 * MiB, 5);
                            store.lcp = lcp;
                            const auto expected = today_checkpoint(ckpt, lcp, rows, eval_end, spec);
                            int keep = 0;
                            const bool found = kvmem_store_checkpoint(
                                    store, kvmem_store_keep_cap(store, eval_end, spec), keep);
                            CHECK(found == expected.first);
                            CHECK(keep == expected.second);
                            ++rows_checked;
                            // The policy adds the continuation rule and the
                            // keep > 0 rule on top, and nothing else: it never
                            // extends on a row this expression would not pick.
                            const kvmem_store_plan plan =
                                    kvmem_store_select({store}, eval_end, spec, "", many);
                            if (plan.action == kvmem_store_action::extend) {
                                CHECK(plan.id == 1 && plan.lcp == lcp);
                                CHECK(expected.first && plan.keep == expected.second);
                                CHECK(plan.keep > 0 && kvmem_store_continuation(store));
                            } else {
                                CHECK(plan.id == -1 && plan.keep == 0);
                            }
                            CHECK(plan.evict.empty());
                            // The same input with no store at all is a fresh
                            // conversation, never an extend.
                            const kvmem_store_plan empty =
                                    kvmem_store_select({}, eval_end, spec, "", many);
                            CHECK(empty.action == kvmem_store_action::fresh && empty.id == -1);
                            CHECK(std::string(empty.reason) == "new_conversation");
                        }
                    }
                }
                if (rows == 0) break; // lcp and the checkpoint sets collapse
            }
        }
    }
    CHECK(rows_checked > 1000);
    std::printf("checkpoint arithmetic: %d combinations\n", rows_checked);
}

// The continuation bound clamps last_n_gen at zero, the way the server's
// suffix_slack does ((uint32_t) std::max(0, st.last_n_gen) + 64u). Without the
// clamp a negative last_n_gen shrinks the window instead of leaving it at 64,
// and an unsigned cast of the sum would wrap it to a bound nothing fails.
static void test_continuation_clamps_last_n_gen() {
    g_row = "T19 continuation clamps a negative last_n_gen";
    for (int gen : {0, -1, -64, -1000}) {
        kvmem_store_match store = store_fixture(1, 4000, gen, {0, 2000}, 1 * MiB, 5);
        store.lcp = 4000 - 64;
        CHECK(kvmem_store_continuation(store));   // exactly at the bound
        store.lcp = 4000 - 65;
        CHECK(!kvmem_store_continuation(store));  // one row past it
    }
    // A positive last_n_gen widens the window by exactly that many rows.
    kvmem_store_match store = store_fixture(1, 4000, 200, {0, 2000}, 1 * MiB, 5);
    store.lcp = 4000 - 264;
    CHECK(kvmem_store_continuation(store));
    store.lcp = 4000 - 265;
    CHECK(!kvmem_store_continuation(store));
}

static void test_table_lru_and_bytes() {
    g_row = "table LRU and byte accounting";
    kvmem_store_table table;
    std::vector<int> ids;
    for (int i = 0; i < 6; ++i) {
        ids.push_back(table.add(100 + i));
    }
    CHECK(table.count() == 6);
    CHECK(table.bytes_total() == 0);
    // Touch in a scripted order; lru_order is least recently used first.
    const int order[] = {4, 0, 5, 2, 1, 3};
    uint64_t clock = 0;
    for (int i : order) {
        table.touch(ids[i], ++clock);
    }
    CHECK(table.lru_order() == std::vector<int>({ids[4], ids[0], ids[5], ids[2], ids[1], ids[3]}));

    // The accumulator must be uint64_t; pin that by arithmetic rather than by
    // inspection.
    const uint64_t big = 1ull << 40;
    table.set_bytes(ids[0], big);
    table.set_bytes(ids[1], big + 1);
    table.set_bytes(ids[2], UINT64_MAX / 4);
    CHECK(table.bytes_total() == big + (big + 1) + UINT64_MAX / 4);

    kvmem_store_plan plan;
    plan.action = kvmem_store_action::extend;
    plan.id     = ids[3];
    plan.evict  = {ids[0], ids[2]};
    const uint64_t dropped = big + UINT64_MAX / 4;
    const uint64_t before  = table.bytes_total();
    // resolve() only names the store to run on. It must not drop a row: the
    // caller has to destroy each victim's adapter handle in the same step, and
    // an entry erased without that is a host store nothing can reach again.
    CHECK(table.resolve(plan) == ids[3]);
    CHECK(table.count() == 6 && table.bytes_total() == before);
    for (int id : plan.evict) {
        CHECK(table.erase(id));
    }
    CHECK(table.bytes_total() == before - dropped);
    CHECK(table.count() == 4);
    CHECK(table.find(ids[0]) == nullptr && table.find(ids[2]) == nullptr);

    // Ids are never reused, so a stale id from an earlier plan is rejected
    // rather than silently matched by a later store.
    const int added = table.add(200);
    CHECK(added != ids[0] && added != ids[2]);
    kvmem_store_plan stale;
    stale.action = kvmem_store_action::extend;
    stale.id     = ids[0];
    stale.evict  = {ids[1]};
    CHECK(table.resolve(stale) == -1);
    CHECK(table.find(ids[1]) != nullptr);

    // A fresh plan asks the caller to allocate.
    kvmem_store_plan fresh;
    fresh.evict = {ids[1]};
    CHECK(table.resolve(fresh) == -1);
    CHECK(table.find(ids[1]) != nullptr);
    CHECK(table.erase(ids[1]));
    CHECK(table.erase(ids[1]) == false);
}

int main() {
    test_decision_table();
    test_client_id_never_fails_a_request();
    test_checkpoint_arithmetic();
    test_continuation_clamps_last_n_gen();
    test_table_lru_and_bytes();
    std::puts("conversation store selection, LRU and byte accounting PASS");
}
