#pragma once

// Selection policy for N host KV stores, one GPU working set, time-multiplexed.
// Freestanding on purpose: only the four standard headers below, no llama.h, no
// mtmd.h, no ServerState. The server computes every input (the longest common
// token prefix in particular, which is media-aware and lives in
// kvmem_prompt::common_prefix) and applies the plan; this file only decides.
//
//   c++ -std=c++17 -I tools -o /tmp/cst tests/conversation-store-test.cpp && /tmp/cst
//
// The three formulas are the ones the single-store server already uses, quoted
// from v0.16.0-rc3:
//
// Cited by symbol rather than by line, the way docs/multi-conversation-kv-cache.md
// does, so the references survive an edit to either file:
//
//   keep_cap     = min({lcp, live_row, eval_end - (spec_ok ? 0 : 1)})
//              `keep` in run_prefill_multimodal, kvmem-multimodal-server.h
//   found, keep  = the largest checkpoint row <= keep_cap
//              the st.mm_checkpoints scan that follows it, same function
//   continuation = (rows - lcp) <= max(0, last_n_gen) + 64
//              `suffix_slack` in run_prefill_retrieval, llama-kvmem-server.cpp
//
// Nothing here runs on the default path: conversation_begin_request returns
// before the policy when --kvmem-conversations is absent, so default behavior
// is identical by construction rather than by two expressions agreeing. What
// the two paths do share is the first two formulas above, and
// tests/conversation-store-test.cpp pins those two helpers against a local
// transcription of today's expression over an exhaustive sweep.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

struct kvmem_store_limits {
    int      max_stores = 1; // live host stores; the server calls the policy only above 1
    uint64_t max_bytes  = 0; // accounted host bytes across stores; 0 = count cap only
    // The store the GPU working set is bound to right now, which eviction must
    // never name: it is released through the memory object and not through a
    // detached adapter handle, so a caller cannot execute that eviction.
    int      attached   = -1;
};

// One live host store, described for this request. The caller fills it from
// that conversation's own payload.
struct kvmem_store_match {
    int      id         = -1; // table id, not the adapter store handle
    int      lcp        = 0;  // rows of this request's prompt matching this store
    int      rows       = 0;  // cached rows (prompt + generated) of this store
    int      live_row   = 0;  // rows actually live in the store
    int      last_n_gen = 0;  // generated rows of this store's last turn
    std::vector<int> ckpt_rows;
    uint64_t bytes      = 0;  // accounted host bytes of this store
    uint64_t used       = 0;  // LRU stamp; larger is more recent
    std::string client_id;    // bound kvmem.conversation_id, empty when inferred
};

enum class kvmem_store_action {
    extend, // run on plan.id and resume at plan.keep without clearing it
    fresh,  // the request gets an empty store; plan.id is always -1, asking
            // the caller to allocate one, or to reuse and clear its least
            // recently used store when no further store is available
};

struct kvmem_store_plan {
    kvmem_store_action action = kvmem_store_action::fresh;
    int  id    = -1;
    int  lcp   = 0;
    int  keep  = 0;
    // Least-recently-used first, never plan.id and never limits.attached.
    // Count evictions come before byte evictions, so the whole list is
    // ascending by `used`.
    std::vector<int> evict;
    const char * reason = "new_conversation";
};

// A client-supplied conversation id is an optimization and must never fail a
// request: a value the policy cannot use as a key is dropped, which leaves the
// request matched by its token prefix exactly as it was before this feature
// existed. Printable ASCII only, because the id is echoed into trace lines.
inline std::string kvmem_store_client_id(const std::string & raw) {
    if (raw.empty() || raw.size() > 128) {
        return {};
    }
    for (unsigned char c : raw) {
        if (c < 33 || c > 126) {
            return {};
        }
    }
    return raw;
}

inline int kvmem_store_keep_cap(const kvmem_store_match & store, int eval_end, bool spec_ok) {
    return std::min({store.lcp, store.live_row, eval_end - (spec_ok ? 0 : 1)});
}

// Largest checkpoint row at or before keep_cap. False means the hybrid model
// cannot be carried to the match point, which is a hard constraint and not a
// policy choice (tools/kvmem-multimodal-server.h:258-262).
inline bool kvmem_store_checkpoint(const kvmem_store_match & store, int keep_cap, int & keep) {
    bool found = false;
    keep = 0;
    for (int row : store.ckpt_rows) {
        if (row <= keep_cap && (!found || row > keep)) {
            keep = row;
            found = true;
        }
    }
    return found;
}

// Absolute, not proportional: a prompt matching most of a long store is a
// branch, not a continuation, and truncating the rest is the destruction this
// feature exists to prevent. The bound is the two legitimate ways a
// continuation's tail shrinks, the dropped last assistant turn and a few
// re-templated prompt tokens.
inline bool kvmem_store_continuation(const kvmem_store_match & store) {
    return store.rows - store.lcp <= std::max(0, store.last_n_gen) + 64;
}

// Least-recently-used eviction, count cap first and byte cap second. Neither
// the plan's target nor the caller's attached store is ever a victim: the
// target is what this request decodes against, and the attached one still owns
// the GPU working set, so a plan naming either would be one the caller cannot
// execute. A single conversation larger than the byte cap therefore still runs
// and the caller only warns; that reproduces today's uncapped single-store
// behavior instead of failing a request. The attached store becomes evictable
// again as soon as the next request attaches a different one.
inline void kvmem_store_plan_evictions(const std::vector<kvmem_store_match> & stores,
                                       const kvmem_store_limits & limits,
                                       bool allocating, kvmem_store_plan & plan) {
    std::vector<const kvmem_store_match *> victims;
    for (const auto & store : stores) {
        if (store.id != plan.id && store.id != limits.attached) {
            victims.push_back(&store);
        }
    }
    std::sort(victims.begin(), victims.end(), [](const kvmem_store_match * a, const kvmem_store_match * b) {
        return a->used != b->used ? a->used < b->used : a->id < b->id;
    });
    uint64_t total = 0;
    for (const auto & store : stores) {
        total += store.bytes;
    }
    // A fresh plan adds one store, so one slot must be free for it.
    int live = (int) stores.size() + (allocating ? 1 : 0);
    size_t next = 0;
    while (live > limits.max_stores && next < victims.size()) {
        total -= victims[next]->bytes;
        plan.evict.push_back(victims[next]->id);
        ++next;
        --live;
    }
    while (limits.max_bytes != 0 && total > limits.max_bytes && next < victims.size()) {
        total -= victims[next]->bytes;
        plan.evict.push_back(victims[next]->id);
        ++next;
    }
}

// Pure: no side effects, no logging, no globals. Eviction is planned here and
// performed by the caller.
inline kvmem_store_plan kvmem_store_select(const std::vector<kvmem_store_match> & stores,
                                           int eval_end, bool spec_ok,
                                           const std::string & client_id,
                                           const kvmem_store_limits & limits) {
    kvmem_store_plan plan;
    // Rule 5: an explicit conversation id is a pure optimization that
    // restricts the candidate set. An unknown id degrades to inferred
    // matching, and a bound id never bypasses the checkpoint rule.
    std::vector<const kvmem_store_match *> candidates;
    if (!client_id.empty()) {
        for (const auto & store : stores) {
            if (store.client_id == client_id) {
                candidates.push_back(&store);
            }
        }
    }
    const bool id_bound = !candidates.empty();
    if (!id_bound) {
        for (const auto & store : stores) {
            candidates.push_back(&store);
        }
    }

    const kvmem_store_match * best = nullptr;
    int  best_keep     = 0;
    int  rejected_keep = 0;
    bool any_found     = false;
    for (const kvmem_store_match * store : candidates) {
        int keep = 0;
        const bool found = kvmem_store_checkpoint(*store, kvmem_store_keep_cap(*store, eval_end, spec_ok), keep);
        if (found) {
            any_found = true;
            rejected_keep = std::max(rejected_keep, keep);
        }
        // keep == 0 has nothing to resume, so extending buys nothing and only
        // risks truncating the store. Rank by keep, which is the prefix the
        // hybrid model can actually restore; lcp is aspiration.
        if (!found || keep <= 0 || !kvmem_store_continuation(*store)) {
            continue;
        }
        if (!best || keep > best_keep || (keep == best_keep && store->used > best->used)) {
            best = store;
            best_keep = keep;
        }
    }
    if (best != nullptr) {
        plan.action = kvmem_store_action::extend;
        plan.id     = best->id;
        plan.lcp    = best->lcp;
        plan.keep   = best_keep;
        plan.reason = id_bound ? "client_id" : "extend";
        kvmem_store_plan_evictions(stores, limits, false, plan);
        return plan;
    }
    plan.action = kvmem_store_action::fresh;
    plan.id     = -1;
    plan.lcp    = 0;
    plan.keep   = 0;
    plan.reason = stores.empty()          ? "new_conversation"
                : !any_found              ? "no_recurrent_checkpoint"
                : rejected_keep == 0      ? "fork_zero_keep"
                                          : "fork_shared_prefix";
    kvmem_store_plan_evictions(stores, limits, true, plan);
    return plan;
}

inline const char * kvmem_store_action_name(kvmem_store_action action) {
    return action == kvmem_store_action::extend ? "extend" : "fresh";
}

// Bookkeeping for the live host stores: the adapter handle, the accounted host
// bytes and the LRU stamp. Ids are allocated monotonically and never reused, so
// a stale id from an earlier plan cannot match a later store.
class kvmem_store_table {
public:
    struct entry {
        int      id       = -1;
        int32_t  store_id = 0; // adapter host-store handle
        uint64_t bytes    = 0;
        uint64_t used     = 0;
    };

    int add(int32_t store_id) {
        entry added;
        added.id       = next_id_++;
        added.store_id = store_id;
        entries_.push_back(added);
        return added.id;
    }

    int count() const { return (int) entries_.size(); }

    const std::vector<entry> & entries() const { return entries_; }

    const entry * find(int id) const {
        for (const auto & held : entries_) {
            if (held.id == id) {
                return &held;
            }
        }
        return nullptr;
    }

    uint64_t bytes_total() const {
        uint64_t total = 0;
        for (const auto & held : entries_) {
            total += held.bytes;
        }
        return total;
    }

    void touch(int id, uint64_t tick) {
        if (entry * held = mutable_find(id)) {
            held->used = tick;
        }
    }

    void set_bytes(int id, uint64_t bytes) {
        if (entry * held = mutable_find(id)) {
            held->bytes = bytes;
        }
    }

    // Least recently used first.
    std::vector<int> lru_order() const {
        std::vector<const entry *> order;
        for (const auto & held : entries_) {
            order.push_back(&held);
        }
        std::sort(order.begin(), order.end(), [](const entry * a, const entry * b) {
            return a->used != b->used ? a->used < b->used : a->id < b->id;
        });
        std::vector<int> ids;
        for (const entry * held : order) {
            ids.push_back(held->id);
        }
        return ids;
    }

    bool erase(int id) {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->id == id) {
                entries_.erase(it);
                return true;
            }
        }
        return false;
    }

    // Reports the store to run on. -1 means the caller must allocate one
    // (plan.id == -1), or that the plan names an entry this table no longer
    // holds. Eviction is deliberately not performed here: releasing a store is
    // more than dropping a row, so the caller walks plan.evict itself and
    // erase()s only what it actually released. A resolver that also erased
    // would let the plan and its execution disagree, and an entry erased
    // without its adapter handle being destroyed is unreachable for good.
    int resolve(const kvmem_store_plan & plan) const {
        return plan.id >= 0 && find(plan.id) == nullptr ? -1 : plan.id;
    }

private:
    entry * mutable_find(int id) {
        for (auto & held : entries_) {
            if (held.id == id) {
                return &held;
            }
        }
        return nullptr;
    }

    std::vector<entry> entries_;
    int next_id_ = 0;
};
