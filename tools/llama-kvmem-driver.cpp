#include "llama-kvmem-driver.h"
#include "llama-kvmem-diag.h"
#include "kvmem/session_memory.hpp"
#include "common.h"
#include "log.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>

std::vector<llama_token> llama_driver_tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> out;
    if (n <= 0) {
        return out;
    }
    out.resize((size_t) n);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), n, add_special, true);
    return out;
}

static int decode_span(llama_context * ctx, const llama_token * toks, int pos0, int pos1, int n_batch,
                       const char * what, kvmem::RequestControl * io = nullptr);
static int decode_span_maybe_spec(LlamaEngineState & st, const llama_token * toks, int pos0, int pos1,
                                  const char * what, kvmem::RequestControl * io = nullptr);
static void multimodal_commit(LlamaEngineState & st, const std::vector<llama_token> & gen);

static bool gdn_sync_to(LlamaEngineState & st, const std::vector<llama_token> & prompt, int n_past,
                        kvmem::RequestControl * io = nullptr) {
    if (!llama_kvmem_has_recurrent()) {
        return true;
    }
    const llama_pos want = n_past - 1;
    llama_pos rmax = llama_kvmem_recr_pos_max();
    std::vector<uint8_t> carry;
    llama_pos draft_rows = n_past;
    if (st.spec.ok) {
        common_speculative_get_state(st.spec.spec, 0, carry);
        if (carry.size() >= sizeof(draft_rows)) std::memcpy(&draft_rows, carry.data(), sizeof(draft_rows));
    }
    if (st.recurrent_cache_valid && rmax == want && draft_rows == n_past) return true;
    const std::vector<uint8_t> * saved_carry = nullptr;
    const uint8_t * blob = nullptr;
    size_t blob_n = 0;
    int ckpt_pos = -1;
    auto consider = [&](const std::vector<uint8_t> & buf, int pos, const std::vector<uint8_t> & saved) {
        if (buf.empty() || pos < 0 || pos > want) {
            return;
        }
        if (pos >= ckpt_pos) {
            blob = buf.data();
            blob_n = buf.size();
            ckpt_pos = pos;
            saved_carry = &saved;
        }
    };
    consider(st.gdn_ckpt, st.gdn_ckpt_pos, st.gdn_carry);
    consider(st.gdn_ckpt_query, st.gdn_ckpt_query_pos, st.gdn_query_carry);
    if (blob == nullptr) {
        fprintf(stderr, "KVMEM_TRACE gdn_sync fail rmax=%d want=%d ckpt_pos=%d query_pos=%d\n",
                (int) rmax, (int) want, st.gdn_ckpt_pos, st.gdn_ckpt_query_pos);
        return false;
    }
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    if (llama_state_seq_set_data_ext(st.ctx, blob, blob_n, 0, fl) != blob_n) {
        fprintf(stderr, "GDN catch-up restore failed\n");
        return false;
    }
    const int from = ckpt_pos + 1;
    if (st.spec.ok) {
        if (!saved_carry || saved_carry->empty()) return false;
        common_speculative_set_state(st.spec.spec, 0, *saved_carry);
        if (!llama_kvmem_remove_logical(st.spec.ctx_dft, from, -1)) return false;
    }
    if (from < n_past) {
        llama_kvmem_set_replay(true);
        const int rc = decode_span_maybe_spec(st, prompt.data(), from, n_past, "gdn-catchup", io);
        llama_kvmem_set_replay(false);
        if (rc == KVMEM_DECODE_ABORT) {
            if (io) {
                io->aborted = true;
            }
            return false;
        }
        if (rc != 0) {
            return false;
        }
    }
    rmax = llama_kvmem_recr_pos_max();
    kvmem_diag("KVMEM_TRACE gdn_sync ckpt_pos=%d from=%d n_past=%d rmax=%d\n",
            ckpt_pos, from, n_past, (int) rmax);
    st.recurrent_cache_valid = rmax == want;
    return st.recurrent_cache_valid;
}

static int common_token_prefix(const std::vector<llama_token> & a,
                               const std::vector<llama_token> & b) {
    const int n = (int) std::min(a.size(), b.size());
    int i = 0;
    while (i < n && a[(size_t) i] == b[(size_t) i]) {
        i++;
    }
    return i;
}

// Exchange the active conversation's payload with `conv`. Called twice per
// switch: once to park the outgoing conversation, once to install the incoming
// one. Swapping is an involution, so calling it twice on the same entry undoes
// it, which is how a refused switch is rolled back.
void llama_driver_swap_conversation(LlamaEngineState & st, kvmem_conversation & conv) {
    st.cached_tokens.swap(conv.cached_tokens);
    st.cached_prompt.swap(conv.cached_prompt);
    st.mm_checkpoints.swap(conv.mm_checkpoints);
    std::swap(st.mm_live_row, conv.mm_live_row);
    st.mm_live_checkpoint.swap(conv.mm_live_checkpoint);
    st.mm_query.swap(conv.mm_query);
    st.gdn_ckpt.swap(conv.gdn_ckpt);
    st.gdn_carry.swap(conv.gdn_carry);
    st.gdn_query_carry.swap(conv.gdn_query_carry);
    std::swap(st.gdn_ckpt_pos, conv.gdn_ckpt_pos);
    st.gdn_ckpt_query.swap(conv.gdn_ckpt_query);
    std::swap(st.gdn_ckpt_query_pos, conv.gdn_ckpt_query_pos);
    std::swap(st.last_query_begin, conv.last_query_begin);
    std::swap(st.last_query_end, conv.last_query_end);
    st.last_user_text.swap(conv.last_user_text);
    std::swap(st.last_n_gen, conv.last_n_gen);
    st.recurrent_cache_valid = false;
}

// Accounted host bytes of one conversation: the adapter's raw K/V store plus
// the driver-owned recurrent checkpoints and token history. RawKvStore walks
// blocks times layers under its own mutex, so this runs once per request, at
// the commit, and once more for an eviction's trace line. Selection reads the
// figure the last commit stored in the table instead of recomputing it.
static uint64_t conversation_bytes(const LlamaEngineState & st, int id) {
    const kvmem_store_table::entry * held = st.conv_table.find(id);
    if (!held) {
        return 0;
    }
    uint64_t bytes = st.kparams.enabled ? llama_kvmem_store_bytes(held->store_id) : 0;
    const auto entry = st.conv.find(id);
    if (entry == st.conv.end()) {
        return bytes;
    }
    const bool active = id == st.conv_active;
    const auto & checkpoints = active ? st.mm_checkpoints : entry->second.mm_checkpoints;
    std::set<const MultimodalCheckpointData *> unique;
    for (const auto & checkpoint : checkpoints) {
        if (checkpoint.data && unique.insert(checkpoint.data.get()).second) {
            bytes += checkpoint.data->bytes();
        }
    }
    const auto & live = active ? st.mm_live_checkpoint : entry->second.mm_live_checkpoint;
    if (live && unique.insert(live.get()).second) bytes += live->bytes();
    const auto & tokens = active ? st.cached_tokens : entry->second.cached_tokens;
    bytes += (uint64_t) tokens.capacity() * sizeof(llama_token);
    const auto & conv = entry->second;
    bytes += sizeof(kvmem_conversation) + conv.client_id.capacity();
    if (conv.payload) bytes += conv.payload->metadata_bytes();
    bytes += (active ? st.gdn_ckpt : conv.gdn_ckpt).capacity();
    bytes += (active ? st.gdn_carry : conv.gdn_carry).capacity();
    bytes += (active ? st.gdn_query_carry : conv.gdn_query_carry).capacity();
    bytes += (active ? st.gdn_ckpt_query : conv.gdn_ckpt_query).capacity();
    bytes += (active ? st.last_user_text : conv.last_user_text).capacity();
    bytes += checkpoints.capacity()*sizeof(MultimodalCheckpoint);
    const auto & prompt = active ? st.cached_prompt : conv.cached_prompt;
    if (prompt) bytes += prompt->index_bytes();
    const auto & query = active ? st.mm_query : conv.mm_query;
    if (query) {
        bytes += sizeof(MultimodalQuery) + query->user.capacity() + query->state.count.capacity()*sizeof(uint32_t);
        bytes += query->state.sum.capacity()*sizeof(std::vector<float>);
        bytes += query->state.ranges.capacity()*sizeof(llama_kvmem_row_range);
        for (const auto & sum : query->state.sum) bytes += sum.capacity()*sizeof(float);
        if (query->prefix && query->prefix != prompt) bytes += query->prefix->index_bytes();
        for (const auto & media : query->media) bytes += sizeof(media) + media.second.capacity();
    }
    return bytes;
}

// Describe one live conversation for the selection policy, reading that
// conversation's own payload: LlamaEngineState's fields when it is the attached one,
// the parked entry otherwise.
static kvmem_store_match conversation_match(const LlamaEngineState & st, int id, const kvmem_prompt & prompt) {
    kvmem_store_match match;
    match.id = id;
    const kvmem_store_table::entry * held = st.conv_table.find(id);
    const auto entry = st.conv.find(id);
    if (!held || entry == st.conv.end()) {
        return match;
    }
    const kvmem_conversation & conv = entry->second;
    if (conv.payload && conv.payload->invalid) return match;
    const bool active = id == st.conv_active;
    match.used = held->used;
    // The accounted figure from that conversation's last commit, not a fresh
    // walk: llama_kvmem_store_bytes takes the store mutex and walks blocks
    // times layers, and this runs once per live store on the prefill latency
    // path. Only the attached conversation's footprint can have moved since,
    // and conversation_commit refreshes exactly that one.
    match.bytes = held->bytes;
    match.client_id = conv.client_id;
    match.rows = (int) (active ? st.cached_tokens.size() : conv.cached_tokens.size());
    match.last_n_gen = active ? st.last_n_gen : conv.last_n_gen;
    if (st.vision || st.query_policy_user) {
        // Default path: the media-aware prefix and the recurrent checkpoint
        // rows run_prefill_multimodal would use (llama-kvmem-driver.cpp, run_prefill_multimodal).
        const auto & cached = active ? st.cached_prompt : conv.cached_prompt;
        match.lcp = cached ? (int) prompt.common_prefix(*cached) : 0;
        match.live_row = active ? st.mm_live_row : conv.mm_live_row;
        for (const auto & checkpoint : (active ? st.mm_checkpoints : conv.mm_checkpoints)) {
            match.ckpt_rows.push_back(checkpoint.row);
        }
    } else {
        // Legacy path: the token prefix, gated by the rows the store actually
        // holds and by a GDN snapshot at or before the match point, which is
        // gdn_sync_to's consider() pair.
        const auto & cached = active ? st.cached_tokens : conv.cached_tokens;
        match.lcp = common_token_prefix(cached, prompt.tokens);
        const uint32_t stored = active
                ? (st.kparams.enabled ? llama_kvmem_store_n_tokens() : (uint32_t) st.cached_tokens.size())
                : conv.stored;
        match.live_row = (int) stored;
        if (!llama_kvmem_has_recurrent()) {
            match.ckpt_rows.push_back(match.lcp);
        } else {
            const int gen_start = active ? st.gdn_ckpt_pos : conv.gdn_ckpt_pos;
            const int query = active ? st.gdn_ckpt_query_pos : conv.gdn_ckpt_query_pos;
            const bool have_gen = !(active ? st.gdn_ckpt : conv.gdn_ckpt).empty() || (!active && conv.cold && conv.disk_gen);
            const bool have_query = !(active ? st.gdn_ckpt_query : conv.gdn_ckpt_query).empty() || (!active && conv.cold && conv.disk_query);
            if (gen_start >= 0 && have_gen) {
                match.ckpt_rows.push_back(gen_start + 1);
            }
            if (query >= 0 && have_query) {
                match.ckpt_rows.push_back(query + 1);
            }
        }
    }
    return match;
}

// Least recently used conversation that is not the attached one.
static int conversation_lru_victim(const LlamaEngineState & st) {
    for (int id : st.conv_table.lru_order()) {
        if (id != st.conv_active) {
            return id;
        }
    }
    return -1;
}

// Frees one parked conversation and reports whether it actually went. Never
// the attached one: llama_memory_clear reaches the attached host store, so an
// inactive store is released through the adapter handle instead, and a table
// row dropped without that release would leave a host store nothing can reach.
static bool conversation_evict(LlamaEngineState & st, int id, const char * reason) {
    if (id == st.conv_active) {
        return false;
    }
    const kvmem_store_table::entry * held = st.conv_table.find(id);
    if (!held) {
        return false;
    }
    const uint32_t rows = st.kparams.enabled ? llama_kvmem_store_rows(held->store_id) : 0;
    const uint64_t bytes = conversation_bytes(st, id);
    const int32_t store_id = held->store_id;
    // Partial deletion must never leave a selectable cache with missing KV.
    if (st.conv.at(id).payload) st.conv.at(id).payload->invalid = true;
    if (st.session_files && !st.session_files->erase(id)) {
        ++st.conv_counts.disk_errors;
        LOG_WRN("srv    KVMEM cannot remove session file id=%d; retaining quota charge\n", id);
        return false;
    }
    if (st.kparams.enabled && !llama_kvmem_store_destroy(store_id)) {
        // Dropping the row anyway would leave the bundle in the adapter's pool
        // with nothing naming it: one runtime with its pinned arena and two
        // host mirrors, leaked for the life of the process. Report the refusal
        // so the caller's skip path runs.
        LOG_WRN("srv    KVMEM conv=%d store=%d refused destroy; row kept\n", id, (int) store_id);
        return false;
    }
    st.conv_table.erase(id);
    st.conv.erase(id);
    ++st.conv_counts.evictions;
    kvmem_diag("KVMEM_TRACE store_evict id=%d rows=%u bytes=%llu reason=%s\n",
            id, rows, (unsigned long long) bytes, reason);
    return true;
}

// Drop a parked conversation's payload, keeping only its identity. The store
// it described no longer holds those rows, so the next request that selects it
// must take an ordinary cache miss rather than resume a checkpoint against KV
// that was never restored. Assigning a default entry is deliberate: it keeps
// this complete as kvmem_conversation grows.
void llama_driver_drop_conversation(kvmem_conversation & conv) {
    kvmem_conversation kept;
    kept.client_id = conv.client_id;
    conv = std::move(kept);
}

void llama_driver_publish_conversations(LlamaEngineState & st) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    st.conv_counts.count = st.conv_table.count();
    st.conv_counts.max = st.conv_limits.max_stores;
    st.conv_counts.active = st.conv_active;
    st.conv_counts.bytes = st.conv_table.bytes_total();
    st.conv_counts.bytes_max = st.conv_limits.max_bytes;
    if (st.session_files) {
        st.conv_counts.disk_bytes = st.session_files->bytes();
        st.conv_counts.disk_bytes_max = st.session_files->limit();
    }
    st.conv_stats.publish(st.conv_counts);
}

void llama_driver_clear(LlamaEngineState & st);


// Included after LlamaEngineState and conversation bookkeeping. Frozen allocations
// stay in place; failures leave resumable RAM/disk manifests, never attached
// partial stores. No whole-session serialization buffer is allocated.
static kvmem_session_payload & session_freeze(LlamaEngineState & st, int id) {
    auto & conv = st.conv.at(id);
    if (conv.payload) return *conv.payload;
    if (id == st.conv_active) throw std::runtime_error("cannot freeze active session");
    std::vector<kvmem::SnapshotBuffer> buffers;
    std::set<const MultimodalCheckpointData *> unique;
    auto checkpoint = [&](const std::shared_ptr<const MultimodalCheckpointData> & p) {
        if (!p || !unique.insert(p.get()).second) return;
        // Created mutable by multimodal_checkpoint(). Live/rollback aliases
        // were cleared after parking. Only this conversation owns references.
        auto & data = *const_cast<MultimodalCheckpointData *>(p.get());
        auto add = [&](auto & vector) {
            if (!vector.capacity()) return;
            auto b = kvmem::SnapshotBuffer::bind(vector);
            if (data.accounting) {
                b.accounting = data.accounting.get();
                b.account = [](void * p, int64_t delta) {
                    auto & a = *static_cast<MultimodalCheckpointAccounting *>(p);
                    if (delta < 0) a.live_bytes -= size_t(-delta); else a.add(size_t(delta));
                };
            }
            buffers.push_back(b);
        };
        add(data.recurrent); add(data.draft_carry); add(data.tail_mean.values);
    };
    for (const auto & c : conv.mm_checkpoints) checkpoint(c.data);
    checkpoint(conv.mm_live_checkpoint);
    for (auto * v : {&conv.gdn_ckpt, &conv.gdn_carry, &conv.gdn_query_carry, &conv.gdn_ckpt_query})
        if (v->capacity()) buffers.push_back(kvmem::SnapshotBuffer::bind(*v));
    const int32_t store = st.conv_table.find(id)->store_id;
    llama_kvmem_store_freeze(store, buffers);
    try {
        conv.payload = std::make_unique<kvmem_session_payload>(id, ++st.session_generation, buffers);
    } catch (...) { llama_kvmem_store_thaw(store); throw; }
    conv.disk_gen = !conv.gdn_ckpt.empty(); conv.disk_query = !conv.gdn_ckpt_query.empty();
    conv.cold = true; // also covers partially migrated and frozen, still-hot stores
    return *conv.payload;
}

static void session_refresh_bytes(LlamaEngineState & st) {
    for (const auto & entry : st.conv_table.entries())
        st.conv_table.set_bytes(entry.id, conversation_bytes(st, entry.id));
}

static void session_warn_budget(LlamaEngineState & st) {
    const bool over = st.conv_limits.max_bytes && st.conv_table.bytes_total() > st.conv_limits.max_bytes;
    if (over && !st.conv_budget_warned)
        LOG_WRN("srv    KVMEM session RAM soft limit exceeded (bytes=%llu cap=%llu); retaining active KV and cold metadata\n",
            (unsigned long long)st.conv_table.bytes_total(), (unsigned long long)st.conv_limits.max_bytes);
    st.conv_budget_warned = over;
}

// Plan the final placement BEFORE choosing a transfer order or applying LRU.
// Retention uses measured RAM and exact restore sizes, just like RAM-only mode.
// Worst-case future generation estimates must not evict reusable sessions.
static void session_balance(LlamaEngineState & st, int target) {
    const auto started = std::chrono::steady_clock::now();
    auto & files = *st.session_files;
    auto * incoming = target >= 0 && target != st.conv_active ? st.conv.at(target).payload.get() : nullptr;
    session_refresh_bytes(st);
    const uint64_t restore = incoming ? incoming->restore_bytes() : 0;
    uint64_t projected = st.conv_table.bytes_total() + restore;
    const uint64_t ram_free = kvmem::session_memory_available();
    uint64_t freed_ram = 0;
    std::vector<kvmem_session_payload *> outgoing;
    for (int id : st.conv_table.lru_order()) {
        if (id == target || id == st.conv_active) continue;
        const bool soft_pressure = st.conv_limits.max_bytes && projected > st.conv_limits.max_bytes;
        if (!soft_pressure && restore <= ram_free + freed_ram) break;
        auto & p = session_freeze(st, id);
        if (p.invalid || !p.ram_bytes()) continue;
        projected -= std::min(projected, p.ram_bytes()); freed_ram += p.ram_bytes();
        outgoing.push_back(&p);
    }
    // Unfinished temp files still have their original RAM source. A failed
    // cleanup remains an I/O error, never an eviction request.
    for (auto * p : outgoing) for (uint32_t i = 0; i < p->chunks.size(); ++i)
        if (p->chunks[i].in_ram && files.contains_chunk(p->id, i) && !files.ready(p->id, i) && !files.erase_chunk(p->id, i))
            throw std::runtime_error("cannot remove unfinished session snapshot");

    const uint64_t disk_available = files.available();
    const uint64_t disk_limit = std::min(files.limit(), files.bytes() + std::min(disk_available, UINT64_MAX - files.bytes()));
    uint64_t final_disk = files.bytes() - (incoming ? files.session_bytes(target) : 0);
    for (auto * p : outgoing) final_disk += p->disk_bytes() - files.session_bytes(p->id);
    uint64_t disk_used = files.bytes(), evicted_ram = 0;
    std::vector<int> evictions;
    // Only FINAL capacity pressure permits LRU. Credit target files that will
    // be freed by restore, including the RAM 10 / target 15 / disk quota 20 case.
    for (int id : st.conv_table.lru_order()) {
        if (final_disk <= disk_limit) break;
        if (id == target || id == st.conv_active) continue;
        const auto it = std::find_if(outgoing.begin(), outgoing.end(), [&](auto * p) { return p->id == id; });
        const uint64_t saved = it != outgoing.end() ? (*it)->disk_bytes() : files.session_bytes(id);
        if (!saved) continue;
        final_disk -= saved; disk_used -= files.session_bytes(id);
        evicted_ram += conversation_bytes(st, id); evictions.push_back(id);
        if (it != outgoing.end()) outgoing.erase(it);
    }
    if (final_disk > disk_limit) throw std::runtime_error("cannot free final session disk capacity");
    const auto plan = kvmem_plan_session_transfer(files, incoming, outgoing,
        kvmem::session_memory_available() + evicted_ram, disk_used, disk_limit);
    const char * route = plan.route == kvmem_session_plan::path::ram_first ? "ram_first" :
        plan.route == kvmem_session_plan::path::exchange ? "exchange" : "disk_first";
    kvmem_diag("KVMEM_TRACE session_transfer target=%d path=%s moves=%zu peak_disk=%llu final_disk=%llu\n",
        target, route, plan.moves.size(), (unsigned long long)plan.peak_disk, (unsigned long long)final_disk);
    for (int id : evictions)
        if (!conversation_evict(st, id, "disk_lru")) throw std::runtime_error("cannot remove LRU session snapshot");
    try {
        kvmem_execute_session_transfer(files, plan, kvmem::session_memory_available);
    } catch (...) { session_refresh_bytes(st); throw; }
    session_refresh_bytes(st);
    for (auto * p : outgoing) {
        ++st.conv_counts.spills;
        kvmem_diag("KVMEM_TRACE session_spill id=%d disk_bytes=%llu ram_bytes=%llu ms=%.2f\n", p->id,
            (unsigned long long)files.session_bytes(p->id), (unsigned long long)st.conv_table.find(p->id)->bytes,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
    }
    if (incoming) {
        if (!incoming->complete() || files.contains(target)) throw std::runtime_error("incomplete session restore");
        llama_kvmem_store_thaw(st.conv_table.find(target)->store_id);
        auto & conv = st.conv.at(target); conv.payload.reset(); conv.cold = false;
        st.conv_table.set_bytes(target, conversation_bytes(st, target));
        ++st.conv_counts.restores;
        kvmem_diag("KVMEM_TRACE session_restore id=%d rows=%u ms=%.2f\n", target, conv.stored,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
    }
}

// Post-request enforcement must not invalidate the completed active cache.
static void session_make_room(LlamaEngineState & st, int target) {
    try { session_balance(st, target); }
    catch (const std::exception & e) {
        ++st.conv_counts.disk_errors; session_refresh_bytes(st);
        LOG_WRN("srv    KVMEM idle session migration deferred: %s\n", e.what());
    }
    session_warn_budget(st);
}

void llama_driver_begin_disk_request(LlamaEngineState & st, const kvmem_prompt & prompt,
                                  const std::string & client_id, int predict) {
    const uint64_t rows = uint64_t(prompt.tokens.size()) + uint64_t(std::max(0, predict)) +
        (st.spec.ok ? uint64_t(std::max(0, st.spec_n_max)) + 1 : 0);
    if (rows > UINT32_MAX) throw std::invalid_argument("session token count overflow");
    const uint64_t reserve = llama_kvmem_store_capacity(uint32_t(rows));
    std::vector<kvmem_store_match> matches;
    for (const auto & entry : st.conv_table.entries()) matches.push_back(conversation_match(st, entry.id, prompt));
    auto limits = st.conv_limits; limits.max_bytes = 0; limits.attached = st.conv_active;
    const auto selection = kvmem_store_select(matches, int(prompt.tokens.size())-(st.spec.ok ? 1 : 0), st.spec.ok, client_id, limits);
    int target = selection.id;
    if (target != st.conv_active && st.conv_active >= 0) {
        if (!st.mm_committed || st.conv_table.find(st.conv_active)->store_id != llama_kvmem_store_current())
            throw std::runtime_error("session switch requires a committed active conversation");
        const int outgoing = st.conv_active;
        llama_driver_swap_conversation(st, st.conv.at(outgoing));
        if (!llama_kvmem_store_park()) {
            llama_driver_swap_conversation(st, st.conv.at(outgoing));
            if (!llama_kvmem_store_n_tokens() && !st.cached_tokens.empty()) llama_driver_clear(st);
            throw std::runtime_error("could not park active session");
        }
        st.conv_active = -1;
        st.mm_live_checkpoint.reset(); st.mm_rollback.reset(); st.mm_rollback_prompt.reset(); st.mm_pending_query.reset();
        if (!llama_kvmem_store_rows(st.conv_table.find(outgoing)->store_id)) llama_driver_drop_conversation(st.conv.at(outgoing));
        st.conv_table.set_bytes(outgoing, conversation_bytes(st, outgoing));
    }
    for (int victim : st.conv_table.lru_order()) {
        if (st.conv_table.count() + (target < 0 ? 1 : 0) <= st.conv_limits.max_stores) break;
        if (victim != target && victim != st.conv_active) conversation_evict(st, victim, "lru");
    }
    if (st.conv_table.count() + (target < 0 ? 1 : 0) > st.conv_limits.max_stores)
        throw std::runtime_error("cannot free session count capacity");
    try {
        // A hot/fresh request does not depend on optional idle demotion. The
        // RAM cap stays soft even when that background retention work fails.
        if (target < 0 || !st.conv.at(target).payload) session_make_room(st, target);
        else session_balance(st, target);
    } catch (const kvmem_session_corrupt & e) {
        ++st.conv_counts.disk_errors;
        LOG_WRN("srv    KVMEM session id=%d corrupt: %s; cache miss\n", target, e.what());
        if (target < 0 || !conversation_evict(st, target, "invalid_snapshot"))
            throw std::runtime_error("cannot remove invalid session snapshot");
        target = -1;
        session_balance(st, target);
    } catch (...) {
        ++st.conv_counts.disk_errors; session_refresh_bytes(st); throw;
    }
    if (target < 0) {
        const int32_t store = llama_kvmem_store_create();
        if (store < 0) throw std::runtime_error("cannot allocate session store");
        target = st.conv_table.add(store); st.conv.emplace(target, kvmem_conversation{});
    }
    if (target != st.conv_active) {
        const int32_t store = st.conv_table.find(target)->store_id;
        const bool restored = llama_kvmem_store_switch(store);
        if (llama_kvmem_store_current() != store) throw std::runtime_error("cannot attach session store");
        llama_driver_swap_conversation(st, st.conv.at(target)); st.conv_active = target;
        st.mm_live_checkpoint.reset();
        if (!restored) llama_driver_clear(st);
        ++st.conv_counts.switches;
    }
    st.conv_table.touch(target, ++st.conv_clock);
    if (selection.id == target) ++st.conv_counts.extends; else ++st.conv_counts.forks;
    llama_driver_publish_conversations(st);
    kvmem_diag("KVMEM_TRACE session_select id=%d keep=%d reserve=%llu\n", target,
        selection.id == target ? selection.keep : 0, (unsigned long long)reserve);
}


// Runs when the conversation's footprint is final for the committed turn.
static void conversation_enforce_budget(LlamaEngineState & st) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    if (st.session_files) {
        session_make_room(st, st.conv_active);
        return;
    }
    while (st.conv_table.count() > st.conv_limits.max_stores) {
        const int victim = conversation_lru_victim(st);
        if (victim < 0 || !conversation_evict(st, victim, "lru")) {
            break;
        }
    }
    while (st.conv_limits.max_bytes != 0 && st.conv_table.bytes_total() > st.conv_limits.max_bytes) {
        const int victim = conversation_lru_victim(st);
        if (victim < 0) {
            // The attached conversation alone exceeds the cap. Report it and
            // serve the request: that reproduces today's uncapped single-store
            // behavior instead of failing a request the server would serve.
            if (!st.conv_budget_warned) {
                st.conv_budget_warned = true;
                LOG_WRN("srv    KVMEM one conversation exceeds --kvmem-conversations-gb (bytes=%llu cap=%llu); nothing evicted\n",
                        (unsigned long long) st.conv_table.bytes_total(),
                        (unsigned long long) st.conv_limits.max_bytes);
            }
            break;
        }
        if (!conversation_evict(st, victim, "bytes")) {
            break;
        }
    }
}

static void conversation_commit(LlamaEngineState & st, uint32_t stored) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    const auto entry = st.conv.find(st.conv_active);
    if (entry == st.conv.end()) {
        return;
    }
    if (st.session_files) {
        if (st.cached_prompt && st.cached_prompt->has_media()) st.cached_prompt = st.cached_prompt->cache_index();
        if (st.mm_query && st.mm_query->prefix && st.mm_query->prefix->has_media()) {
            auto query = std::make_shared<MultimodalQuery>(*st.mm_query);
            query->prefix = query->prefix->cache_index(); st.mm_query = std::move(query);
        }
    }
    entry->second.stored = stored;
    if (!st.turn_conversation_id.empty()) {
        // One id names one conversation: a client that reuses an id after its
        // own context was compacted must not leave two stores answering to it.
        for (auto & other : st.conv) {
            if (other.first != st.conv_active && other.second.client_id == st.turn_conversation_id) {
                other.second.client_id.clear();
            }
        }
        entry->second.client_id = st.turn_conversation_id;
    }
    st.conv_table.touch(st.conv_active, ++st.conv_clock);
    st.conv_table.set_bytes(st.conv_active, conversation_bytes(st, st.conv_active));
    conversation_enforce_budget(st);
    llama_driver_publish_conversations(st);
}

void llama_driver_clear(LlamaEngineState & st) {
    llama_memory_t mem = llama_get_memory(st.ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            llama_memory_clear(md, true);
        }
    }
    st.cached_tokens.clear();
    st.cached_prompt.reset();
    st.mm_checkpoints.clear();
    st.mm_live_row = 0;
    st.mm_live_checkpoint.reset();
    st.mm_query.reset();
    st.mm_pending_query.reset();
    st.gdn_ckpt.clear();
    st.gdn_carry.clear();
    st.gdn_query_carry.clear();
    if (st.spec.ok) {
        std::vector<uint8_t> carry;
        common_speculative_get_state(st.spec.spec, 0, carry);
        std::fill(carry.begin(), carry.end(), 0);
        common_speculative_set_state(st.spec.spec, 0, carry);
    }
    st.gdn_ckpt_pos = -1;
    st.gdn_ckpt_query.clear();
    st.gdn_ckpt_query_pos = -1;
    st.last_query_begin = -1;
    st.last_query_end = -1;
    st.last_user_text.clear();
    st.last_n_gen = 0;
    st.recurrent_cache_valid = true;
    // Clears the attached conversation only: llama_memory_clear reaches the
    // host store that is bound right now, and every st.* field above is that
    // conversation's payload.
    if (!st.conv.empty()) {
        const auto entry = st.conv.find(st.conv_active);
        if (entry != st.conv.end()) {
            entry->second.stored = 0;
        }
        ++st.conv_counts.resets;
    }
}

// Pick the conversation this request continues and attach its host store.
// Called once per request, under the inference lock, after the prompt is
// parsed and validated and before anything reads or writes KVMem state.
//
// With --kvmem-conversations absent this returns before touching anything:
// no table, no bookkeeping, no store switch, no policy, no trace. Default
// behavior is then identical by construction rather than by argument.
void llama_driver_begin_request(LlamaEngineState & st, const kvmem_prompt & prompt,
                                       const std::string & client_id) {
    if (st.conv_limits.max_stores <= 1) {
        return;
    }
    if (st.conv.find(st.conv_active) == st.conv.end()) {
        return; // never armed, or arming failed at startup
    }
    const int eval_end = (int) prompt.tokens.size() - (st.spec.ok ? 1 : 0);
    std::vector<kvmem_store_match> matches;
    matches.reserve(st.conv_table.entries().size());
    for (const auto & held : st.conv_table.entries()) {
        matches.push_back(conversation_match(st, held.id, prompt));
    }
    // cache_reset is deliberately not a separate action: the policy still maps
    // the request to a conversation, and the fork path below clears that one
    // store. One client resetting its own history must not wipe another
    // client's store.
    kvmem_store_limits limits = st.conv_limits;
    limits.attached = st.conv_active;
    const kvmem_store_plan plan = kvmem_store_select(matches, eval_end, st.spec.ok,
            client_id, limits);
    const bool allocating = plan.action == kvmem_store_action::fresh && plan.id < 0;
    // Two of the conditions the switch needs are already knowable: the
    // previous request's rows must be committed, and conv_active must name the
    // store the adapter actually has attached. Check them before the plan is
    // executed, because a refusal after the fact would have destroyed an LRU
    // victim and created a store for a conversation the request then does not
    // move to. resolve() is pure and plan.evict never names plan.id, so asking
    // it here gives the same answer it gives below.
    const kvmem_store_table::entry * attached = st.conv_table.find(st.conv_active);
    const int32_t attached_store = attached ? attached->store_id : -1;
    const bool would_switch = allocating || st.conv_table.resolve(plan) != st.conv_active;
    const bool refused = would_switch &&
            (!st.mm_committed || attached_store != llama_kvmem_store_current());
    if (refused) {
        LOG_WRN("srv    KVMEM store switch refused action=%s conv=%d committed=%d parked=%d active=%d\n",
                kvmem_store_action_name(plan.action), st.conv_table.resolve(plan),
                (int) st.mm_committed, (int) attached_store, (int) llama_kvmem_store_current());
    }
    if (!refused) {
        const int room = (int) matches.size() - st.conv_limits.max_stores + (allocating ? 1 : 0);
        int evicted = 0;
        for (int id : plan.evict) {
            if (!conversation_evict(st, id, evicted < room ? "lru" : "bytes")) {
                // The policy excludes both the target and the attached store, so
                // this is unreachable. Never fall back to dropping the row: the
                // adapter handle would survive with nothing naming it.
                LOG_WRN("srv    KVMEM eviction plan named conv=%d, which cannot be released; skipped\n", id);
                continue;
            }
            ++evicted;
        }
    }
    // Eviction is executed above, one entry at a time, so resolve() only names
    // the target and rejects a stale id from an earlier plan. A refused plan
    // stays on the attached conversation and neither evicts nor allocates; the
    // caps it declined to enforce are enforced again at the next turn that
    // commits. That is not necessarily this one: conversation_commit() runs
    // only from llama_driver_commit(), so a turn that fails after the mapping never
    // re-measures, and the byte overage stays until some later turn commits.
    int target = refused ? st.conv_active : st.conv_table.resolve(plan);
    // Every way this request can end up on the attached conversation after
    // planning another one. None of them forks, parks or creates anything, so
    // none of them may be counted as a fork.
    bool fell_back = refused;
    bool force_reset = false;
    if (!refused && allocating) {
        const int32_t store_id = llama_kvmem_store_create();
        if (store_id >= 0) {
            target = st.conv_table.add(store_id);
            st.conv.emplace(target, kvmem_conversation{});
        } else {
            // No further host store available. Reuse the least recently used
            // one, cleared below once the switch has actually attached it.
            target = conversation_lru_victim(st);
            force_reset = target >= 0;
        }
    }
    if (target < 0 || st.conv.find(target) == st.conv.end()) {
        target = st.conv_active; // stale plan id: fall back to today's path
        force_reset = false;
        fell_back = true;
    }
    bool switched = false;
    bool restaged = false;
    if (target != st.conv_active) {
        const int parked_id = st.conv_active;
        const kvmem_store_table::entry * parked = st.conv_table.find(parked_id);
        const int32_t parked_store = parked ? parked->store_id : -1;
        const kvmem_store_table::entry * held = st.conv_table.find(target);
        const auto outgoing = st.conv.find(parked_id);
        const auto incoming = st.conv.find(target);
        // st.mm_committed and the conv_active/adapter-active agreement were
        // checked above, before the plan was executed, and nothing since can
        // have changed the adapter's active store: store_create appends and
        // store_destroy refuses the active id. What is left to check is a
        // table entry or a payload row this switch needs and cannot find. On
        // any of those the switch would report success without moving anything
        // (llama_kvmem_store_switch short-circuits a switch to the active
        // store) and both failure detectors below would read clean while this
        // conversation decoded against another one's KV.
        if (!held || outgoing == st.conv.end() || incoming == st.conv.end()) {
            // A handle the adapter does not know, or a row the table and the
            // payload map disagree about. Stay where we are rather than switch
            // on ambiguous state.
            LOG_WRN("srv    KVMEM store switch refused conv=%d held=%d outgoing=%d incoming=%d\n",
                    target, (int) (held != nullptr), (int) (outgoing != st.conv.end()),
                    (int) (incoming != st.conv.end()));
            target = st.conv_active;
            force_reset = false;
            fell_back = true;
        } else {
            llama_driver_swap_conversation(st, outgoing->second);
            restaged = llama_kvmem_store_switch(held->store_id);
            if (llama_kvmem_store_current() == held->store_id) {
                llama_driver_swap_conversation(st, incoming->second);
                st.conv_active = target;
                // The context's recurrent state still belongs to the previous
                // conversation, so multimodal_restore's live short-circuit
                // must not skip the restore (llama-kvmem-driver.cpp, checkpoint restore).
                st.mm_live_checkpoint.reset();
                switched = true;
                ++st.conv_counts.switches;
                if (!restaged && (st.mm_live_row > 0 || !st.cached_tokens.empty())) {
                    // Attached but holding no rows: either this store was
                    // already empty or the adapter could not rebuild its
                    // working set from host RAM. The payload would claim rows
                    // the KV no longer has, so drop it and let prefill take
                    // its ordinary cache-miss path.
                    kvmem_diag("KVMEM_TRACE store_stale id=%d rows=%d reason=working_set_not_rebuilt\n",
                            target, st.mm_live_row);
                    llama_driver_clear(st);
                    force_reset = false; // already empty, and resets count once
                }
                // The switch clears the outgoing store when the adapter cannot
                // drain it safely, and the bool above describes the incoming
                // store only. Cross-check what the parked handle still holds
                // instead of trusting it: a payload claiming rows its store no
                // longer has would resume a checkpoint against KV that was
                // never restored, and nothing later reconciles the two.
                if (parked_store >= 0 && llama_kvmem_store_rows(parked_store) == 0 &&
                    !outgoing->second.cached_tokens.empty()) {
                    kvmem_diag("KVMEM_TRACE store_wiped id=%d rows=%d reason=outgoing_not_drainable\n",
                            parked_id, (int) outgoing->second.cached_tokens.size());
                    llama_driver_drop_conversation(outgoing->second);
                }
            } else {
                // Nothing moved. Put the previous conversation back and let
                // prefill decide against it, as a single-store server would.
                llama_driver_swap_conversation(st, outgoing->second);
                LOG_WRN("srv    KVMEM store switch failed conv=%d store=%d\n",
                        target, (int) held->store_id);
                // The same cross-check the success branch runs, for the same
                // reason: swap_conv() clears the outgoing store before the
                // attach whenever it cannot be drained, so a throw out of the
                // attach leaves this branch restoring a payload that claims
                // rows the store no longer has. The outgoing conversation is
                // the attached one again here, so its payload lives in st.*.
                if (parked_store >= 0 && llama_kvmem_store_rows(parked_store) == 0 &&
                    !st.cached_tokens.empty()) {
                    kvmem_diag("KVMEM_TRACE store_wiped id=%d rows=%d reason=switch_failed\n",
                            parked_id, (int) st.cached_tokens.size());
                    llama_driver_clear(st);
                }
                target = st.conv_active;
                force_reset = false;
                fell_back = true;
            }
        }
    }
    if (force_reset) {
        // Clear the reused store here rather than through
        // st.mm_reset_requested: only run_prefill_multimodal consumes that
        // flag, so on the legacy retrieval path it would never be read and the
        // prefill would extend a live conversation's rows with an unrelated
        // prompt, truncating its tail with no eviction accounting. This
        // clears the attached store and its payload and counts the reset.
        kvmem_diag("KVMEM_TRACE store_reuse id=%d reason=no_store_available\n", target);
        llama_driver_clear(st);
    }
    st.conv_table.touch(target, ++st.conv_clock);
    if (fell_back) {
        ++st.conv_counts.refusals;
    } else if (plan.action == kvmem_store_action::extend && target == plan.id) {
        ++st.conv_counts.extends;
    } else {
        ++st.conv_counts.forks;
    }
    llama_driver_publish_conversations(st);
    kvmem_diag("KVMEM_TRACE store_select action=%s id=%d lcp=%d keep=%d stores=%zu "
            "bytes=%llu reason=%s switched=%d restaged=%d\n",
            kvmem_store_action_name(plan.action), target, plan.lcp, plan.keep,
            st.conv_table.entries().size(), (unsigned long long) st.conv_table.bytes_total(),
            plan.reason, (int) switched, (int) restaged);
}

// Persist GDN after a successful prefill (eval_end-1) for the next turn's
// suffix rewind. Intra-turn query rewind uses a local snapshot, not this slot.
static void persist_gdn_ckpt_gen_start(LlamaEngineState & st, int eval_end) {
    if (!st.ctx || eval_end <= 0 || !llama_kvmem_has_recurrent()) {
        return;
    }
    llama_synchronize(st.ctx);
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t sz = llama_state_seq_get_size_ext(st.ctx, 0, fl);
    if (sz == 0) {
        kvmem_diag("KVMEM_TRACE gdn_ckpt gen_start skipped size=0 eval_end=%d\n", eval_end);
        return;
    }
    std::vector<uint8_t> buf(sz);
    if (llama_state_seq_get_data_ext(st.ctx, buf.data(), sz, 0, fl) != sz) {
        fprintf(stderr, "KVMEM_TRACE gdn_ckpt gen_start copy failed eval_end=%d\n", eval_end);
        return;
    }
    st.gdn_ckpt.swap(buf);
    if (st.spec.ok) common_speculative_get_state(st.spec.spec, 0, st.gdn_carry);
    st.gdn_ckpt_pos = eval_end - 1;
    kvmem_diag("KVMEM_TRACE gdn_ckpt pos_end=%d bytes=%zu ckpt_pos=%d what=gen_start\n",
            eval_end, sz, st.gdn_ckpt_pos);
}

void llama_driver_commit(LlamaEngineState & st, const std::vector<llama_token> & prompt,
                          const std::vector<llama_token> & gen) {
    st.cached_tokens = prompt;
    st.cached_tokens.insert(st.cached_tokens.end(), gen.begin(), gen.end());
    st.last_n_gen = (int) gen.size();
    if (st.vision || st.query_policy_user) multimodal_commit(st, gen);
    else st.cached_prompt = st.active_prompt->with_generated(gen);
    const uint32_t stored = llama_kvmem_store_n_tokens();
    conversation_commit(st, stored);
    kvmem_diag("KVMEM_TRACE cache_commit n_prompt=%d n_gen=%d n_cached=%d stored=%u\n",
            (int) prompt.size(), (int) gen.size(), (int) st.cached_tokens.size(),
            stored);
}

static int decode_span(llama_context * ctx, const llama_token * toks, int pos0, int pos1, int n_batch,
                       const char * what, kvmem::RequestControl * io) {
    if (pos0 >= pos1) {
        return 0;
    }
    if (n_batch <= 0) {
        n_batch = 512;
    }
    // Explicit pos: T5 query sits in the middle of the prompt (last user, then
    // assistant tool XML + role=tool). llama_batch_get_one would append at
    // seq_pos_max+1 and miss the hole after seq_rm(q0,q1).
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int n_pos = pos0;
    while (n_pos < pos1) {
        if (!kvmem::poll_request(io)) {
            kvmem_diag("KVMEM_TRACE stream_abort phase=prefill pos=%d what=%s\n",
                    n_pos, what ? what : "");
            llama_batch_free(batch);
            return KVMEM_DECODE_ABORT;
        }
        const int n = std::min(n_batch, pos1 - n_pos);
        common_batch_clear(batch);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, toks[n_pos + i], n_pos + i, { 0 }, i == n - 1);
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "llama_decode(%s) failed rc=%d at pos=%d n=%d\n", what, rc, n_pos, n);
            llama_batch_free(batch);
            return rc;
        }
        n_pos += n;
    }
    llama_batch_free(batch);
    return 0;
}

static int decode_span_maybe_spec(LlamaEngineState & st, const llama_token * toks, int pos0, int pos1,
                                 const char * what, kvmem::RequestControl * io) {
    if (st.spec.ok) {
        auto abort_fn = [io]() { return !kvmem::poll_request(io); };
        return kvmem_spec_decode_span(st.ctx, st.spec.spec, toks, pos0, pos1, st.n_batch, what, abort_fn);
    }
    return decode_span(st.ctx, toks, pos0, pos1, st.n_batch, what, io);
}



#include <set>
#include <numeric>

void llama_driver_validate_capacity(const LlamaEngineState & st, const kvmem_prompt & prompt, int end) {
    if (!st.kparams.enabled || !st.kparams.budget || !prompt.has_media()) return;
    const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
    const uint32_t budget = st.kparams.budget / block;
    std::vector<std::pair<uint32_t, uint32_t>> groups;
    for (const auto & range : prompt.media_ranges()) {
        const uint32_t lo = (range.first ? range.first - 1 : 0) / block;
        const uint32_t hi = (std::min<uint32_t>(end, range.second + 1) + block - 1) / block;
        if (!groups.empty() && lo < groups.back().second) groups.back().second = hi;
        else groups.emplace_back(lo, hi);
    }
    const uint32_t sink = std::max(1u, st.kparams.sink_tokens / block);
    for (const auto & group : groups) {
        if (group.second - group.first + std::min(group.first, sink) > budget)
            throw std::invalid_argument("image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget");
    }
    // Text suffixes may be trimmed after the first pass. Only an image group
    // that cannot fit intact with the sink is a hard capacity error.
}

// Included after the single-slot server state and stream helpers.
static MultimodalCheckpoint multimodal_checkpoint(LlamaEngineState & st, int row) {
    MultimodalCheckpoint result;
    result.row = row;
    if (st.mm_live_checkpoint && st.mm_live_row == row) {
        result.data = st.mm_live_checkpoint;
        ++st.mm_perf.shared;
        return result;
    }
    kvmem_scoped_ms timer(st.mm_perf.save_ms);
    ++st.mm_perf.saves;
    auto data = std::make_shared<MultimodalCheckpointData>();
    llama_synchronize(st.ctx);
    {
        kvmem_scoped_ms mean_timer(st.mm_perf.mean_ms);
        if (!llama_kvmem_get_tail_mean(row, data->tail_mean))
            throw std::runtime_error("mean-K checkpoint is not an accepted prefix");
    }
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t size = llama_state_seq_get_size_ext(st.ctx, 0, flags);
    data->recurrent.resize(size);
    if (llama_state_seq_get_data_ext(st.ctx, data->recurrent.data(), size, 0, flags) != size) {
        throw std::runtime_error("multimodal recurrent checkpoint failed");
    }
    {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        if (st.spec.ok && !common_speculative_get_state(st.spec.spec, 0, data->draft_carry)) {
            throw std::runtime_error("MTP carry checkpoint failed");
        }
    }
    data->accounting = st.mm_checkpoint_accounting;
    data->accounting->add(data->bytes());
    result.data = std::move(data);
    st.mm_live_checkpoint = result.data;
    return result;
}

static void multimodal_remember(LlamaEngineState & st, MultimodalCheckpoint checkpoint) {
    auto & entries = st.mm_checkpoints;
    entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto & entry) {
        return entry.row >= checkpoint.row;
    }), entries.end());
    entries.push_back(std::move(checkpoint));
    if (entries.size() > 4) {
        const auto media_count = std::count_if(entries.begin(), entries.end(), [](const auto & e) { return e.media_boundary; });
        auto victim = std::find_if(entries.begin(), entries.end(), [&](const auto & e) {
            return e.media_boundary == (media_count > 2);
        });
        entries.erase(victim == entries.end() ? entries.begin() : victim);
    }
}

static void multimodal_restore(LlamaEngineState & st, const MultimodalCheckpoint & checkpoint, bool truncate) {
    kvmem_scoped_ms timer(st.mm_perf.restore_ms);
    const bool live = st.mm_live_checkpoint == checkpoint.data && st.mm_live_row == checkpoint.row;
    if (!checkpoint.data) throw std::runtime_error("missing multimodal checkpoint data");
    if (!llama_kvmem_tail_mean_valid(checkpoint.row, checkpoint.data->tail_mean))
        throw std::runtime_error("stale or foreign mean-K/recurrent checkpoint");
    if (live) ++st.mm_perf.restore_skips;
    else ++st.mm_perf.restores;
    llama_synchronize(st.ctx);
    if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
    llama_kvmem_decode_mean_flush();
    llama_kvmem_decode_mean_discard();
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const auto & data = checkpoint.data->recurrent;
    if (!live && llama_state_seq_set_data_ext(st.ctx, data.data(), data.size(), 0, flags) != data.size()) {
        throw std::runtime_error("multimodal recurrent restore failed");
    }
    if (!llama_kvmem_remove_logical(st.ctx, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted target rows");
    }
    if (st.spec.ctx_dft && !llama_kvmem_remove_logical(st.spec.ctx_dft, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted MTP rows");
    }
    if (st.spec.ok && !live) {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        common_speculative_set_state(st.spec.spec, 0, checkpoint.data->draft_carry);
    }
    if (truncate) {
        llama_kvmem_truncate_cached(checkpoint.row);
        if (!llama_kvmem_set_tail_mean(checkpoint.row, checkpoint.data->tail_mean))
            throw std::runtime_error("mean-K checkpoint restore rejected");
    }
    st.mm_live_row = checkpoint.row;
    st.mm_live_checkpoint = checkpoint.data;
    st.recurrent_cache_valid = true;
}

void llama_driver_finish(LlamaEngineState & st) {
    if (st.mm_committed || !st.mm_rollback) return;
    try {
        llama_kvmem_set_replay(false);
        multimodal_restore(st, *st.mm_rollback, true);
        llama_kvmem_begin_cached_turn();
        st.mm_query.reset();
        st.mm_pending_query.reset();
        st.cached_prompt = st.mm_rollback_prompt;
        st.cached_tokens = st.cached_prompt ? st.cached_prompt->tokens : std::vector<llama_token>{};
        st.cached_tokens.resize(std::min(st.cached_tokens.size(), (size_t) st.mm_live_row));
        multimodal_remember(st, *st.mm_rollback);
        kvmem_diag("KVMEM_TRACE multimodal_rollback context=%p row=%d\n", (void *) st.ctx, st.mm_live_row);
        kvmem_diag("KVMEM_CHECKPOINT_ROLLBACK live_bytes=%zu peak_bytes=%zu\n",
                st.mm_checkpoint_accounting->live_bytes.load(), st.mm_checkpoint_accounting->peak_bytes.load());
        st.mm_committed = true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        LOG_ERR("srv    KVMEM_TRACE multimodal_rollback_failed error=%s\n", e.what());
        // A failed recurrent restore cannot remain a reusable prefix.
        try { llama_driver_clear(st); }
        catch (const std::exception & clear_error) {
            LOG_ERR("srv    KVMEM cache clear failed: %s\n", clear_error.what());
            st.cached_tokens.clear();
            st.cached_prompt.reset();
            st.mm_checkpoints.clear();
            st.mm_live_checkpoint.reset();
            st.mm_live_row = 0;
            st.gdn_ckpt_pos = st.gdn_ckpt_query_pos = -1;
        }
        st.mm_committed = true;
    }
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
}

static int multimodal_decode_span(LlamaEngineState & st, int begin, int end, bool replay, kvmem::RequestControl * io) {
    const auto & prompt = *st.active_prompt;
    auto dispatch = [&](llama_batch batch) -> int {
        if (!kvmem::poll_request(io)) return KVMEM_DECODE_ABORT;
        kvmem_diag("KVMEM_TRACE multimodal_decode context=%p rows=[%d,%d) model_pos=%d image=%d replay=%d\n",
                (void *) st.ctx, batch.logical_pos[0], batch.logical_pos[batch.n_tokens - 1] + 1,
                batch.pos[0], batch.token == nullptr, replay);
        const auto start = std::chrono::steady_clock::now();
        const bool diagnostic = llama_kvmem_get_transfer_stats().enabled;
        st.mm_live_checkpoint.reset();
        int rc = llama_decode(st.ctx, batch);
        if (diagnostic) {
            llama_synchronize(st.ctx);
            st.mm_perf.target_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        const auto draft_start = std::chrono::steady_clock::now();
        if (rc == 0 && st.spec.ok && !common_speculative_process(st.spec.spec, batch)) rc = -1;
        if (diagnostic && st.spec.ok) {
            llama_synchronize(st.spec.ctx_dft);
            st.mm_perf.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - draft_start).count();
        }
        llama_synchronize(st.ctx);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        (replay ? st.mm_perf.replay_ms : st.mm_perf.first_ms) += elapsed;
        kvmem_diag("KVMEM_TRACE multimodal_compute rows=%d elapsed_ms=%.3f image=%d replay=%d\n",
                batch.n_tokens, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                batch.token == nullptr, replay);
        if (rc == 0) {
            st.mm_live_row = batch.logical_pos[batch.n_tokens - 1] + 1;
            if (replay) st.mm_replayed += batch.n_tokens;
            else {
                const int tail = std::clamp(st.mm_lcp - batch.logical_pos[0], 0, batch.n_tokens);
                st.mm_tail_replayed += tail;
                if (batch.token) st.mm_new_text += batch.n_tokens - tail;
                else st.mm_new_image += batch.n_tokens - tail;
                // Count actual first-pass work, including reconstruction after the
                // reused checkpoint; exclude the later retrieval query replay.
                if (io && io->on_prefill) io->on_prefill(batch.n_tokens, (int) prompt.tokens.size() - batch.logical_pos[0]);
            }
        }
        return rc;
    };
    int row = begin;
    while (row < end) {
        if (!kvmem::poll_request(io)) return KVMEM_DECODE_ABORT;
        if (prompt.tokens[row] == LLAMA_TOKEN_NULL) {
            const int next = (int) prompt.media_end(row);
            if (next > end) throw std::runtime_error("prefill boundary splits an image");
            if (!replay) {
                auto checkpoint = multimodal_checkpoint(st, row);
                checkpoint.media_boundary = true;
                multimodal_remember(st, std::move(checkpoint));
            }
            const int rc = st.vision->decode(st.ctx, prompt, row, st.n_batch, dispatch);
            if (rc != 0) return rc;
            row = next;
            continue;
        }
        const int limit = std::min(end, row + st.n_batch);
        int next = row;
        while (next < limit && prompt.tokens[next] != LLAMA_TOKEN_NULL) ++next;
        std::vector<llama_pos> pos(next - row), logical(next - row);
        const auto pos0 = prompt.model_pos(row);
        for (int i = row; i < next; ++i) {
            pos[i - row] = pos0 + i - row;
            logical[i - row] = i;
        }
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.tokens.data()) + row, next - row);
        batch.pos = pos.data();
        batch.logical_pos = logical.data();
        std::vector<int32_t> n_seq(next - row, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seq_ids(next - row, &seq);
        std::vector<int8_t> outputs(next - row, 0);
        outputs.back() = !st.spec.ok;
        batch.n_seq_id = n_seq.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        const int rc = dispatch(batch);
        if (rc != 0) return rc;
        row = next;
    }
    return 0;
}

static bool run_prefill_multimodal(LlamaEngineState & st, kvmem::RequestControl * io, int * n_cache_hit) {
    const auto started = std::chrono::steady_clock::now();
    st.mm_perf = {};
    const auto copies_before = llama_kvmem_get_transfer_stats();
    st.mm_checkpoint_accounting->peak_bytes.store(st.mm_checkpoint_accounting->live_bytes.load());
    st.mm_pending_query.reset();
    try {
        st.mm_error.clear();
        st.mm_error_status = 500;
        if (st.mm_reset_requested) {
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=explicit_cache_reset\n", (void *) st.ctx);
            llama_driver_clear(st);
            st.mm_reset_requested = false;
        }
        st.mm_new_text = st.mm_new_image = st.mm_replayed = st.mm_tail_replayed = 0;
        const auto & prompt = *st.active_prompt;
        const int eval_end = (int) prompt.tokens.size() - (st.spec.ok ? 1 : 0);
        const int lcp = st.cached_prompt ? (int) prompt.common_prefix(*st.cached_prompt) : 0;
        st.mm_lcp = lcp;
        // Sequence checkpoints do not restore logits. Ordinary decoding must evaluate
        // at least one token; MTP evaluates the pending prompt token in spec_generate.
        const int keep = std::min({lcp, st.mm_live_row, eval_end - (st.spec.ok ? 0 : 1)});
        MultimodalCheckpoint base;
        bool found = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.select_checkpoint_ms);
            for (const auto & checkpoint : st.mm_checkpoints) {
                if (checkpoint.row <= keep && (!found || checkpoint.row > base.row) && checkpoint.data &&
                        llama_kvmem_tail_mean_valid(checkpoint.row, checkpoint.data->tail_mean)) {
                    base = checkpoint;
                    found = true;
                }
            }
        }
        if (!found) {
            // Shared template tokens do not identify a conversation. Like llama-server,
            // treat a missing recurrent checkpoint as a cache miss and evaluate the supplied prompt.
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=%s lcp=%d keep=%d cached_rows=%zu live_rows=%d oldest_checkpoint=%d checkpoint_count=%zu\n",
                    (void *) st.ctx, st.cached_prompt ? "no_recurrent_checkpoint" : "new_conversation",
                    lcp, keep, st.cached_tokens.size(), st.mm_live_row,
                    st.mm_checkpoints.empty() ? -1 : st.mm_checkpoints.front().row, st.mm_checkpoints.size());
            llama_driver_clear(st);
            st.mm_lcp = 0; // No old rows survived the reset; count all evaluated rows as new.
            base = multimodal_checkpoint(st, 0);
        }
        st.mm_rollback = std::make_shared<MultimodalCheckpoint>(base);
        st.mm_rollback_prompt = st.cached_prompt ? st.cached_prompt->prefix(base.row) : nullptr;
        st.mm_committed = false;
        multimodal_restore(st, base, true);
        llama_kvmem_begin_cached_turn();
        std::vector<uint32_t> starts, ends;
        const auto ranges = prompt.media_ranges();
        for (const auto & range : ranges) {
            starts.push_back(range.first > 0 ? range.first - 1 : 0);
            ends.push_back(std::min<uint32_t>(prompt.tokens.size(), range.second + 1));
        }
        llama_kvmem_set_media_ranges(starts.data(), ends.data(), starts.size());
        const int user_begin = std::clamp(st.kparams.query_begin, 0, eval_end);
        const int user_end = std::clamp(st.kparams.query_end, user_begin, eval_end);
        const auto media = prompt.media_identity();
        const auto cached_query = st.mm_query;
        const bool same_query = st.query_policy_user && st.turn_query_exact && cached_query &&
            cached_query->begin == user_begin && cached_query->end == user_end &&
            cached_query->force == st.kparams.force_pos && cached_query->user == st.turn_last_user &&
            base.row >= user_end && cached_query->media == media &&
            prompt.common_prefix(*cached_query->prefix) >= (size_t) user_end;
        const bool capture_user = st.query_policy_user && st.turn_query_exact &&
            user_begin >= base.row && user_end > user_begin;
        int query = std::max(base.row, std::min(st.kparams.query_begin, eval_end));
        if (!ranges.empty()) query = std::max(query, (int) ranges.back().second);
        query = std::min(query, eval_end);
        const bool retrieve = st.kparams.enabled && st.kparams.method == 1 && query < eval_end;
        llama_kvmem_set_request_span(query, eval_end, st.kparams.force_pos);
        llama_kvmem_turn_spans spans;
        spans.query = {{query, eval_end}};
        spans.mandatory = {{query, eval_end}};
        spans.replay_begin = query;
        if (st.query_policy_user) {
            spans.query = capture_user || same_query
                ? std::vector<llama_kvmem_row_range>{{user_begin, user_end}}
                : std::vector<llama_kvmem_row_range>{{std::max(query, eval_end - st.query_max_tokens), eval_end}};
        }
        llama_kvmem_set_turn_spans(spans);
        std::string path = "legacy", reason = "legacy_requested";
        std::string reuse_fallback = "none";
        bool reused_query = false;
        const bool imported_query = same_query && st.kparams.enabled && st.kparams.method == 1 &&
            llama_kvmem_set_query(cached_query->state);
        if (imported_query) {
            kvmem_scoped_ms timer(st.mm_perf.decision_ms);
            if (llama_kvmem_can_append(eval_end, st.turn_generation_rows, false, reason)) {
                path = "keep_selected";
                reused_query = true;
            } else {
                auto select_spans = spans;
                // Q predates these tool observations. Do not let stale query
                // scores discard newly learned facts during a fast reselect.
                // If the observed tail no longer fits the selection budget,
                // use a fresh suffix probe instead. The old Q source itself
                // is not a replay dependency and need not consume mandatory slots.
                select_spans.mandatory = {{user_end, base.row}};
                const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
                if (base.row % block) select_spans.mandatory.push_back({base.row - 1, base.row});
                llama_kvmem_set_turn_spans(select_spans);
                const auto selection = llama_kvmem_preview_retrieval();
                if (llama_kvmem_selection_fits(selection, eval_end, st.turn_generation_rows)) {
                    {
                        kvmem_scoped_ms retrieval_timer(st.mm_perf.retrieval_ms);
                        llama_kvmem_apply_selection(selection);
                    }
                    llama_kvmem_begin_cached_turn_keep_query();
                    llama_kvmem_freeze_query(true);
                    reused_query = llama_kvmem_can_append(eval_end, st.turn_generation_rows, false, reason);
                    if (reused_query) path = "cached_q_reselect";
                } else reason = "observed_tail_or_append_exceeds_capacity";
            }
            if (reused_query) {
                st.mm_pending_query = cached_query;
                llama_kvmem_keep_selected();
            }
        }
        if (same_query && !reused_query) {
            reuse_fallback = imported_query ? reason : "invalid_query_state";
            llama_kvmem_reset_query();
            llama_kvmem_freeze_query(false);
            spans.query = {{std::max(query, eval_end - st.query_max_tokens), eval_end}};
            llama_kvmem_set_turn_spans(spans);
        }
        bool all_resident = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.decision_ms);
            all_resident = !reused_query && retrieve && st.query_replay_auto &&
                llama_kvmem_can_append(eval_end, st.turn_generation_rows, true, reason);
        }
        if (all_resident) llama_kvmem_keep_selected();
        if (n_cache_hit) *n_cache_hit = base.row;
        if (multimodal_decode_span(st, base.row, query, false, io) != 0) throw std::runtime_error("multimodal prefill failed or cancelled");
        auto query_checkpoint = multimodal_checkpoint(st, query);
        multimodal_remember(st, query_checkpoint);
        const auto probe_view = llama_kvmem_get_attention_view();
        if (multimodal_decode_span(st, query, eval_end, false, io) != 0) throw std::runtime_error("multimodal query prefill failed or cancelled");
        if (reused_query) {
            if (!llama_kvmem_commit_resident(false)) throw std::runtime_error("incomplete KV after query continuation");
        } else if (retrieve) {
            bool replay = true;
            bool replay_fits = true;
            {
                kvmem_scoped_ms timer(st.mm_perf.retrieval_ms);
                if (all_resident && llama_kvmem_commit_resident()) {
                    path = "all_resident";
                    replay = false;
                } else {
                    const auto selection = llama_kvmem_preview_retrieval();
                    if (st.query_replay_auto && llama_kvmem_commit_unchanged(probe_view, selection)) {
                        path = "unchanged_selection";
                        reason = "same_attention_view";
                        replay = false;
                    } else {
                        path = "query_replay";
                        if (st.query_replay_auto) reason = "selection_or_attention_view_changed";
                        // Check the actual selection: complete image groups also
                        // consume slots, so a text-only budget estimate is insufficient.
                        const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
                        for (uint32_t id = query / block; id < ((uint32_t) eval_end + block - 1) / block; ++id) {
                            if (!std::binary_search(selection.blocks.begin(), selection.blocks.end(), id)) {
                                replay_fits = false;
                                break;
                            }
                        }
                        llama_kvmem_apply_selection(selection);
                    }
                }
            }
            if (replay && !replay_fits) {
                // Selection may trim the mandatory suffix when a long tool history
                // exceeds the retrieval budget. Keep the completed first-pass
                // recurrent state, logits and MTP carry; restoring the query
                // checkpoint would require replaying rows with no resident slot.
                path = "query_replay_skipped";
                reason = "replay_exceeds_budget";
                replay = false;
                kvmem_diag("KVMEM_TRACE replay_skipped reason=over_budget query=[%d,%d) replay_rows=%d budget_tokens=%u\n",
                        query, eval_end, eval_end - query, st.kparams.budget);
            }
            if (replay) {
                multimodal_restore(st, query_checkpoint, false);
                llama_kvmem_set_replay(true);
                const int rc = multimodal_decode_span(st, query, eval_end, true, io);
                llama_kvmem_set_replay(false);
                if (rc != 0) throw std::runtime_error("multimodal query replay failed or cancelled");
            }
        }
        if (!reused_query && capture_user && st.kparams.enabled && st.kparams.method == 1) {
            auto saved = std::make_shared<MultimodalQuery>();
            if (llama_kvmem_get_query(saved->state) &&
                    *std::max_element(saved->state.count.begin(), saved->state.count.end()) == (uint32_t) (user_end - user_begin)) {
                saved->begin = user_begin;
                saved->end = user_end;
                saved->force = st.kparams.force_pos;
                saved->user = st.turn_last_user;
                saved->media = media;
                saved->prefix = prompt.prefix(user_end);
                st.mm_pending_query = std::move(saved);
            }
        }
        const auto & counts = st.mm_pending_query ? st.mm_pending_query->state.count : std::vector<uint32_t>{};
        const uint32_t q_rows = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());
        const char * source = !st.query_policy_user ? "legacy_suffix" : reused_query ? "cached_user" :
            capture_user ? "user" : "bootstrap_suffix";
        if (!retrieve && !reused_query) reason = "no_query_suffix";
        kvmem_diag("KVMEM_PREFILL_DECISION path=%s reason=%s reuse_fallback=%s append=[%d,%d) query=[%d,%d) feature=[%d,%d) query_source=%s query_reused=%d q_rows=%u replay_rows=%u decision_ms=%.3f\n",
                path.c_str(), reason.c_str(), reuse_fallback.c_str(), base.row, eval_end, query, eval_end,
                spans.query.front().begin, spans.query.back().end, source, reused_query, q_rows, st.mm_replayed, st.mm_perf.decision_ms);
        llama_kvmem_pin_working_set();
        multimodal_remember(st, multimodal_checkpoint(st, eval_end));
        std::vector<uint8_t> carry;
        llama_pos synced = 0;
        if (st.spec.ok) {
            common_speculative_get_state(st.spec.spec, 0, carry);
            if (carry.size() < sizeof(synced)) throw std::runtime_error("MTP carry missing");
            std::memcpy(&synced, carry.data(), sizeof(synced));
            if (synced != eval_end) throw std::runtime_error("MTP has unsynchronized visual rows");
        }
        kvmem_diag("KVMEM_TRACE multimodal_prefill context=%p prefix_hit_rows=%d lcp=%d new_text_rows=%u new_image_rows=%u replayed_rows=%u vision_encode_calls=%u encoder_ms=%.2f logical_cursor=%d model_cursor=%d mtp_synced_rows=%d cached_tail_rows=%u replay_reason=%s embedding_cache_bytes=%zu checkpoint_bytes=%zu\n",
                (void *) st.ctx, base.row, lcp, st.mm_new_text, st.mm_new_image, st.mm_replayed,
                prompt.encode_calls, prompt.encode_ms,
                eval_end, prompt.model_pos(eval_end), synced, st.mm_tail_replayed, st.mm_replayed ? reason.c_str() : "none",
                st.vision ? st.vision->cache_bytes() : 0,
                std::accumulate(st.mm_checkpoints.begin(), st.mm_checkpoints.end(), size_t(0),
                    [](size_t n, const auto & c) { return n + c.data->bytes(); }));
        std::set<const MultimodalCheckpointData *> unique;
        size_t unique_bytes = 0;
        size_t ref_bytes = 0;
        auto count = [&](const MultimodalCheckpoint & cp) {
            if (cp.data && unique.insert(cp.data.get()).second) {
                unique_bytes += cp.data->bytes();
                ref_bytes += cp.data->bytes() * cp.data.use_count();
            }
        };
        for (const auto & cp : st.mm_checkpoints) count(cp);
        if (st.mm_rollback) count(*st.mm_rollback);
        count(base);
        count(query_checkpoint);
        const auto & p = st.mm_perf;
        kvmem_diag("KVMEM_PREFILL_PERF total_ms=%.3f first_ms=%.3f replay_ms=%.3f retrieval_ms=%.3f checkpoint_select_ms=%.3f checkpoint_save_ms=%.3f checkpoint_restore_ms=%.3f mean_nested_ms=%.3f carry_nested_ms=%.3f saves=%u restores=%u shared=%u restore_skips=%u checkpoint_unique_bytes=%zu\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count(),
                p.first_ms, p.replay_ms, p.retrieval_ms, p.select_checkpoint_ms, p.save_ms, p.restore_ms,
                p.mean_ms, p.carry_ms, p.saves, p.restores, p.shared, p.restore_skips, unique_bytes);
        kvmem_diag("KVMEM_CHECKPOINT_MEMORY ref_bytes=%zu unique_bytes=%zu live_bytes=%zu peak_bytes=%zu\n",
                ref_bytes, unique_bytes, st.mm_checkpoint_accounting->live_bytes.load(), st.mm_checkpoint_accounting->peak_bytes.load());
        if (copies_before.enabled) {
            const auto copies = llama_kvmem_get_transfer_stats();
            fprintf(stderr, "KVMEM_PREFILL_PHASE path=%s first_ms=%.3f replay_ms=%.3f retrieval_ms=%.3f replay_rows=%u query=[%d,%d)\n",
                    path.c_str(), p.first_ms, p.replay_ms, p.retrieval_ms,
                    st.mm_replayed, query, eval_end);
            fprintf(stderr, "KVMEM_PREFILL_DIAGNOSTIC target_ms=%.3f draft_ms=%.3f extra_sync=1 adapter_h2d_bytes=%llu adapter_d2h_bytes=%llu adapter_d2d_bytes=%llu h2d_calls=%llu d2h_calls=%llu d2d_calls=%llu\n",
                    p.target_ms, p.draft_ms,
                    (unsigned long long) (copies.bytes[0] - copies_before.bytes[0]),
                    (unsigned long long) (copies.bytes[1] - copies_before.bytes[1]),
                    (unsigned long long) (copies.bytes[2] - copies_before.bytes[2]),
                    (unsigned long long) (copies.calls[0] - copies_before.calls[0]),
                    (unsigned long long) (copies.calls[1] - copies_before.calls[1]),
                    (unsigned long long) (copies.calls[2] - copies_before.calls[2]));
        }
        return true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        if (dynamic_cast<const std::invalid_argument *>(&e)) st.mm_error_status = 400;
        LOG_ERR("srv    KVMEM_TRACE multimodal_error error=%s\n", e.what());
        llama_driver_finish(st);
        return false;
    }
}

static void multimodal_commit(LlamaEngineState & st, const std::vector<llama_token> & gen) {
    st.cached_prompt = st.active_prompt->with_generated(gen);
    st.mm_live_row = st.kparams.enabled ? (int) llama_kvmem_store_n_tokens()
        : st.mm_live_row;
    multimodal_remember(st, multimodal_checkpoint(st, st.mm_live_row));
    st.mm_query = st.mm_pending_query;
    st.mm_pending_query.reset();
    st.mm_committed = true;
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
    kvmem_diag("KVMEM_CHECKPOINT_COMMIT live_bytes=%zu peak_bytes=%zu\n",
            st.mm_checkpoint_accounting->live_bytes.load(), st.mm_checkpoint_accounting->peak_bytes.load());
}

int llama_driver_decode_generated(LlamaEngineState & st, llama_token id, int row) {
    st.mm_live_checkpoint.reset();
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_pos logical = row;
    llama_pos pos = st.active_prompt->model_pos(row);
    batch.logical_pos = &logical;
    batch.pos = &pos;
    const int rc = llama_decode(st.ctx, batch);
    if (rc == 0) st.mm_live_row = row + 1;
    return rc;
}



bool llama_driver_prepare(LlamaEngineState & st, const std::vector<llama_token> & prompt,
                                 kvmem::RequestControl * io, int * n_cache_hit) {
    if (st.vision || st.query_policy_user) return run_prefill_multimodal(st, io, n_cache_hit);
    if (!kvmem::poll_request(io)) {
        kvmem_diag("KVMEM_TRACE stream_abort phase=prefill_start n_prompt=%d\n",
                (int) prompt.size());
        return false;
    }
    const int n_prompt = (int) prompt.size();
    llama_context * ctx = st.ctx;
    const int eval_end = st.spec.ok ? n_prompt - 1 : n_prompt;

    const bool do_retr = st.kparams.enabled && st.kparams.method == 1 && st.kparams.query_begin > 0;
    int q0 = st.kparams.query_begin;
    int q1 = st.kparams.query_end;
    if (q0 < 0) {
        q0 = 0;
    }
    if (q1 <= q0 || q1 > eval_end) {
        q1 = eval_end;
    }
    const bool replay_fits = llama_kvmem_query_replay_fits(
            (uint32_t) std::max(q0, 0), (uint32_t) std::max(eval_end, 0));

    int n_past = 0;
    bool reused = false;
    uint32_t stored = st.kparams.enabled ? llama_kvmem_store_n_tokens()
                                         : (uint32_t) st.cached_tokens.size();
    if (!st.cached_tokens.empty() && n_prompt > 1) {
        const int lcp = common_token_prefix(st.cached_tokens, prompt);
        kvmem_diag("KVMEM_TRACE prefix_try lcp=%d n_cached=%d stored=%u n_prompt=%d\n",
                lcp, (int) st.cached_tokens.size(), stored, n_prompt);
        reused = lcp > 0 && lcp < n_prompt && stored >= (uint32_t) lcp;
        if (reused) {
            n_past = lcp;
        }
    }

    llama_pos kv_smax = -1;
    if (llama_memory_t mem = llama_get_memory(ctx)) {
        kv_smax = llama_memory_seq_pos_max(mem, 0);
    }
    const llama_pos gdn_rmax = llama_kvmem_has_recurrent()
            ? llama_kvmem_recr_pos_max()
            : (n_past > 0 ? (llama_pos) (n_past - 1) : (llama_pos) -1);
    const bool same_query = !st.last_user_text.empty() &&
            st.last_user_text == st.turn_last_user;
    const bool gdn_at_tip = !llama_kvmem_has_recurrent() ||
            (st.recurrent_cache_valid && n_past > 0 && gdn_rmax == (llama_pos) (n_past - 1));
    const bool kv_at_tip = n_past > 0 && kv_smax >= (llama_pos) (n_past - 1);
    // Same last-user: keep the GPU window and only prefill the new tail.
    // Suffix after query is recency (recent_tokens), not skip-gated.
    // New user / miss / GDN not at tip / no gen slots → full retrieval.
    const int n_cached = (int) st.cached_tokens.size();
    // Continuation: LCP covers the previous cache except last gen (thinking
    // stripped / re-templated). +64 is a few prompt-side template tokens.
    // Compact leaves LCP far short of n_cached.
    const uint32_t suffix_slack = (uint32_t) std::max(0, st.last_n_gen) + 64u;
    const bool suffix_cont = reused
            && (uint32_t) (n_cached - n_past) <= suffix_slack;
    if (reused && !suffix_cont) {
        kvmem_diag("KVMEM_TRACE prefix_rewrite drop_reuse=1 n_past=%d n_cached=%d "
                "n_prompt=%d last_n_gen=%d slack=%u same_query=%d\n",
                n_past, n_cached, n_prompt, st.last_n_gen, suffix_slack,
                (int) same_query);
        llama_driver_clear(st);
        n_past = 0;
        reused = false;
    }
    const uint32_t n_new_tok = (uint32_t) std::max(0, eval_end - n_past);
    const uint32_t bt = std::max(1u, st.kparams.block_tokens);
    const uint32_t need_slots = n_new_tok == 0 ? 0u : (n_new_tok + bt - 1) / bt;
    const uint32_t free_slots = llama_kvmem_free_slots();
    bool past_query = n_past > q1;
    bool warm_skip = do_retr && reused && suffix_cont && same_query && past_query &&
            gdn_at_tip && kv_at_tip && free_slots >= need_slots;

    if (reused) {
        if (st.kparams.enabled) {
            if (past_query && same_query) {
                llama_kvmem_begin_cached_turn_keep_query();
            } else {
                llama_kvmem_begin_cached_turn();
            }
            if (warm_skip) {
                llama_kvmem_keep_selected();
            }
        }
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_seq_rm(mem, 0, n_past, -1);
        }
        if (st.spec.ctx_dft) {
            llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
            if (md) {
                llama_memory_seq_rm(md, 0, n_past, -1);
            }
        }
        if (st.kparams.enabled) {
            llama_kvmem_truncate_cached((uint32_t) n_past);
        }
        // Continuation already has GDN at n_past. Catch-up from query would
        // llama_decode at q0 while seq_pos_max is n_past-1 (M-RoPE X < Y).
        if (!warm_skip && !gdn_sync_to(st, prompt, n_past, io)) {
            if (io && io->aborted) {
                return false;
            }
            reused = false;
            warm_skip = false;
        }
    }
    if (!reused) {
        llama_driver_clear(st);
        n_past = 0;
        warm_skip = false;
        past_query = false;
    }
    // DeepSeek usage: prefix cache hit = kept LCP (n_past). Full miss if reuse
    // was dropped (gdn_sync fail / empty cache).
    if (n_cache_hit) {
        *n_cache_hit = n_past < 0 ? 0 : n_past;
        if (*n_cache_hit > n_prompt) {
            *n_cache_hit = n_prompt;
        }
    }

    // GDN ckpt/rewind only on the first pass of this query. Continuation
    // already has GDN at n_past; replaying the decode suffix is recency, not
    // a mandatory catch-up.
    const bool recr_ckpt = do_retr && replay_fits && llama_kvmem_has_recurrent() &&
            !warm_skip && !past_query;
    kvmem_diag("KVMEM_TRACE prefix_reuse reused=%d n_past=%d n_prompt=%d n_cached=%d "
            "n_new=%d stored=%u query=[%d,%d) replay_fits=%d warm_skip=%d "
            "same_query=%d suffix_cont=%d last_n_gen=%d slack=%u "
            "gdn_rmax=%d kv_smax=%d free_slots=%u need_slots=%u\n",
            (int) reused, n_past, n_prompt, (int) st.cached_tokens.size(),
            eval_end - n_past, llama_kvmem_store_n_tokens(),
            q0, q1, (int) replay_fits, (int) warm_skip, (int) same_query,
            (int) suffix_cont, st.last_n_gen, suffix_slack,
            (int) gdn_rmax, (int) kv_smax, free_slots, need_slots);

    auto take_rc = [&](int rc) -> bool {
        if (rc == KVMEM_DECODE_ABORT) {
            if (io) {
                io->aborted = true;
            }
            return false;
        }
        return rc == 0;
    };
    auto dec = [&](int a, int b, const char * what) -> bool {
        if (a < 0) {
            a = 0;
        }
        if (b > eval_end) {
            b = eval_end;
        }
        if (a >= b) {
            return true;
        }
        return take_rc(decode_span_maybe_spec(st, prompt.data(), a, b, what, io));
    };
    // Hole-fill / recapture of positions that may already sit in KV. Trunk
    // only: MTP draft is M-RoPE and cannot decode Y while X (seq_pos_max)
    // is still ahead of Y.
    auto replay = [&](int a, int b, const char * what) -> bool {
        if (a < 0) {
            a = 0;
        }
        if (b > eval_end) {
            b = eval_end;
        }
        if (a >= b) {
            return true;
        }
        llama_kvmem_set_replay(true);
        const int rc = decode_span(st.ctx, prompt.data(), a, b, st.n_batch, what, io);
        llama_kvmem_set_replay(false);
        return take_rc(rc);
    };

    auto note_prefill = [&]() {
        llama_synchronize(ctx);
        const llama_perf_context_data p = llama_perf_context(ctx);
        const int d = p.n_p_eval - st.perf_p_eval;
        st.perf_p_eval = p.n_p_eval;
        kvmem_diag("KVMEM_TRACE prefix_prefill n_p_eval=%d reused=%d n_past=%d n_new=%d\n",
                d, (int) reused, n_past, eval_end - n_past);
    };
    auto commit_last_query = [&](bool ok) {
        if (ok) {
            st.last_query_begin = q0;
            st.last_query_end = q1;
            st.last_user_text = st.turn_last_user;
        } else {
            st.last_query_begin = -1;
            st.last_query_end = -1;
            st.last_user_text.clear();
        }
    };

    if (warm_skip) {
        kvmem_diag("KVMEM_TRACE query_replay_skip_same query=[%d,%d) n_past=%d n_new=%d\n",
                q0, q1, n_past, eval_end - n_past);
        if (!dec(n_past, eval_end, "prefill-tail")) {
            return false;
        }
        if (st.spec.ctx_dft) {
            llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
            if (md) {
                kvmem_diag("KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                        llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
            }
        }
        llama_kvmem_pin_working_set();
        note_prefill();
        commit_last_query(true);
        persist_gdn_ckpt_gen_start(st, eval_end);
        return true;
    }

    // Same last-user, query already in the prefix, but skip could not keep
    // the window (usually gen-reserve full). Reuse the captured Q for top-k;
    // do not llama_decode at q0 (M-RoPE requires seq_pos_max < q0).
    if (do_retr && reused && suffix_cont && same_query && past_query) {
        kvmem_diag("KVMEM_TRACE query_reuse_q reselect=1 query=[%d,%d) n_past=%d n_new=%d\n",
                q0, q1, n_past, eval_end - n_past);
        llama_kvmem_apply_retrieval(ctx);
        if (!dec(n_past, eval_end, "prefill-tail")) {
            return false;
        }
        if (st.spec.ctx_dft) {
            llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
            if (md) {
                kvmem_diag("KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                        llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
            }
        }
        llama_kvmem_pin_working_set();
        note_prefill();
        commit_last_query(true);
        persist_gdn_ckpt_gen_start(st, eval_end);
        return true;
    }

    if (!do_retr) {
        if (!dec(n_past, eval_end, reused ? "prefill-suffix" : "prefill")) {
            return false;
        }
        note_prefill();
        commit_last_query(false);
        persist_gdn_ckpt_gen_start(st, eval_end);
        return true;
    }

    llama_kvmem_reset_query();
    if (!dec(n_past, q0, reused ? "prefill-suffix" : "prefill")) {
        return false;
    }
    std::vector<uint8_t> gdn_ckpt;
    if (recr_ckpt) {
        if (n_past > q0 && !gdn_sync_to(st, prompt, q0, io)) {
            LOG_ERR("srv    KVMEM_TRACE gdn_sync to query_begin failed\n");
            return false;
        }
        llama_synchronize(ctx);
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t sz = llama_state_seq_get_size_ext(ctx, 0, fl);
        if (sz == 0) {
            fprintf(stderr, "GDN checkpoint size 0\n");
            return false;
        }
        gdn_ckpt.resize(sz);
        if (llama_state_seq_get_data_ext(ctx, gdn_ckpt.data(), sz, 0, fl) != sz) {
            fprintf(stderr, "GDN checkpoint copy failed\n");
            return false;
        }
        st.gdn_ckpt_query = gdn_ckpt;
        if (st.spec.ok) common_speculative_get_state(st.spec.spec, 0, st.gdn_query_carry);
        st.gdn_ckpt_query_pos = q0 > 0 ? q0 - 1 : -1;
        kvmem_diag("KVMEM_TRACE gdn_ckpt_query pos_end=%d bytes=%zu query_begin=%d ckpt_pos=%d\n",
                q0, sz, q0, st.gdn_ckpt_query_pos);
    }
    // Query may already sit inside the reused prefix (T5: last user, then
    // assistant tool XML + role=tool). Recapture Q over the cached part
    // (replay skips K/mean); prefill only the missing tail of the span.
    if (n_past > q0 && n_past < q1) {
        if (!replay(q0, n_past, "query-q-capture")) {
            return false;
        }
    }
    if (n_past < q1) {
        if (!dec(std::max(n_past, q0), q1, "prefill-query")) {
            return false;
        }
    } else if (!replay(q0, q1, "query-q-capture")) {
        return false;
    }
    kvmem_diag("KVMEM_TRACE query_q_capture n_past=%d query=[%d,%d) recapture=%d\n",
            n_past, q0, q1, (int) (n_past > q0));
    llama_synchronize(ctx);
    {
        const llama_perf_context_data p = llama_perf_context(ctx);
        const int d = p.n_p_eval - st.perf_p_eval;
        st.perf_p_eval = p.n_p_eval;
        kvmem_diag("KVMEM_TRACE prefix_prefill n_p_eval=%d reused=%d n_past=%d n_new=%d\n",
                d, (int) reused, n_past, eval_end - n_past);
    }

    const int tail0 = std::max(n_past, q1);
    const uint32_t tail_tok = (uint32_t) std::max(0, eval_end - tail0);
    const uint32_t gen_res = std::max(1u, st.kparams.gen_reserve);
    const bool tail_fits_gen = tail_tok <= gen_res;

    if (tail_fits_gen) {
        llama_kvmem_apply_retrieval(ctx);
        if (!replay_fits) {
            kvmem_diag("KVMEM_TRACE query_replay_skip query=[%d,%d) eval_end=%d "
                    "(sink+suffix exceeds GPU budget)\n",
                    q0, q1, eval_end);
        } else {
            if (recr_ckpt && !gdn_ckpt.empty()) {
                const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
                if (llama_state_seq_set_data_ext(ctx, gdn_ckpt.data(), gdn_ckpt.size(), 0, fl) != gdn_ckpt.size()) {
                    fprintf(stderr, "GDN restore failed\n");
                    return false;
                }
                kvmem_diag("KVMEM_TRACE gdn_restore bytes=%zu\n", gdn_ckpt.size());
            }
            llama_memory_t mem = llama_get_memory(ctx);
            if (mem) {
                kvmem_diag("KVMEM_TRACE before_seq_rm seq_pos=[%d,%d] query=[%d,%d)\n",
                        llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                        q0, q1);
                llama_memory_seq_rm(mem, 0, q0, q1);
                kvmem_diag("KVMEM_TRACE after_seq_rm seq_pos=[%d,%d] auto_pos0=%d\n",
                        llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0),
                        llama_memory_seq_pos_max(mem, 0) + 1);
            }
            if (st.spec.ctx_dft && !past_query) {
                llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
                if (md) {
                    llama_memory_seq_rm(md, 0, q0, q1);
                    kvmem_diag("KVMEM_TRACE mtp_after_seq_rm seq_pos=[%d,%d]\n",
                            llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
                }
            }
            if (!replay(q0, q1, "query replay")) {
                return false;
            }
            llama_synchronize(ctx);
            kvmem_diag("KVMEM_TRACE query_replay begin=%d n=%d recr_ckpt=%d\n",
                    q0, q1 - q0, (int) recr_ckpt);
            if (st.spec.ctx_dft && !past_query) {
                // First pass of this query: seq_rm left a hole in the draft cache.
                // M-RoPE cannot fill it while a suffix remains, so drop [q0, inf)
                // and append in order up to q1. Continuation keeps draft suffix.
                llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
                if (md) {
                    llama_memory_seq_rm(md, 0, q0, -1);
                }
                if (q1 > q0) {
                    const int rc = decode_span(st.spec.ctx_dft, prompt.data(), q0, q1, st.n_batch,
                                               "mtp-resync", io);
                    if (!take_rc(rc)) {
                        return false;
                    }
                    kvmem_diag("KVMEM_TRACE mtp_resync query=[%d,%d) to=%d\n", q0, q1, q1);
                }
            }
        }
        if (!dec(tail0, eval_end, "prefill-tail")) {
            return false;
        }
    } else {
        // Compact / long history after last-user: tail is not this turn's
        // decode slack. Prefill with spill, then retrieve.
        kvmem_diag("KVMEM_TRACE prefill_tail_offload n=%u gen_reserve=%u query=[%d,%d)\n",
                tail_tok, gen_res, q0, q1);
        if (!dec(tail0, eval_end, "prefill-tail")) {
            return false;
        }
        llama_kvmem_apply_retrieval(ctx);
    }
    if (st.spec.ctx_dft) {
        llama_memory_t md = llama_get_memory(st.spec.ctx_dft);
        if (md) {
            kvmem_diag("KVMEM_TRACE mtp_after_query_replay seq_pos=[%d,%d]\n",
                    llama_memory_seq_pos_min(md, 0), llama_memory_seq_pos_max(md, 0));
        }
    }
    commit_last_query(true);
    persist_gdn_ckpt_gen_start(st, eval_end);
    return true;
}

static std::string trim_copy(const std::string & s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) {
        ++a;
    }
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\n' || s[b - 1] == '\r' || s[b - 1] == '\t')) {
        --b;
    }
    return s.substr(a, b - a);
}

// Last ChatML user *role block* in the rendered prompt, not rfind(content).
// Thinking/tool text can quote last_user; only <|im_start|>user ... <|im_end|>
// counts. If last_user is set, pick the last block whose content equals it
// (two identical user turns → the later block).
static bool find_last_user_role_block(const std::string & prompt, const std::string & last_user,
                                      size_t & content0, size_t & content1, int & n_blocks, int & pick) {
    static const char * hdrs[] = {
        "<|im_start|>user\n",
        "<|im_start|>user\r\n",
        "<|im_start|>user",
    };
    struct Blk {
        size_t c0;
        size_t c1;
    };
    std::vector<Blk> blks;
    size_t search = 0;
    while (search < prompt.size()) {
        size_t best = std::string::npos;
        size_t best_len = 0;
        for (const char * h : hdrs) {
            const size_t n = std::strlen(h);
            const size_t p = prompt.find(h, search);
            if (p == std::string::npos) {
                continue;
            }
            if (best == std::string::npos || p < best || (p == best && n > best_len)) {
                best = p;
                best_len = n;
            }
        }
        if (best == std::string::npos) {
            break;
        }
        const size_t c0 = best + best_len;
        const size_t end = prompt.find("<|im_end|>", c0);
        if (end == std::string::npos) {
            break;
        }
        blks.push_back(Blk{c0, end});
        search = best + 1;
    }
    n_blocks = (int) blks.size();
    pick = -1;
    if (blks.empty()) {
        return false;
    }
    const std::string want = trim_copy(last_user);
    if (!want.empty()) {
        for (int i = n_blocks - 1; i >= 0; --i) {
            const std::string got = trim_copy(prompt.substr(blks[(size_t) i].c0,
                    blks[(size_t) i].c1 - blks[(size_t) i].c0));
            if (got == want) {
                pick = i;
                break;
            }
        }
    }
    if (pick < 0) {
        // No exact content match (template wrapping). Do not fall back to
        // the last user-role header: Qwen tools are often rendered as user
        // + <tool_response>. Leave the caller to query-last fallback.
        if (!want.empty()) {
            return false;
        }
        pick = n_blocks - 1;
    }
    content0 = blks[(size_t) pick].c0;
    content1 = blks[(size_t) pick].c1;
    return content0 < content1;
}

void llama_driver_query_span(LlamaEngineState & st, const std::string & prompt, const std::string & last_user,
                              const std::vector<llama_token> & toks, int & qbegin, int & qend) {
    qbegin = -1;
    qend = (int) toks.size();
    size_t c0 = 0;
    size_t c1 = 0;
    int n_blocks = 0;
    int pick = -1;
    if (find_last_user_role_block(prompt, last_user, c0, c1, n_blocks, pick)) {
        const std::string prefix = prompt.substr(0, c0);
        const std::string through = prompt.substr(0, c1);
        qbegin = (int) llama_driver_tokenize(st.vocab, prefix, true).size();
        qend = (int) llama_driver_tokenize(st.vocab, through, true).size();
        kvmem_diag("KVMEM_TRACE query_loc method=role_block n_user_blocks=%d pick=%d "
                "span=[%zu,%zu) tokens=[%d,%d)\n",
                n_blocks, pick, c0, c1, qbegin, qend);
    }
    if (qend > (int) toks.size()) {
        qend = (int) toks.size();
    }
    if (qbegin < 0 || qend <= qbegin) {
        qend = (int) toks.size();
        const int last = std::min(st.query_last_fallback, qend);
        qbegin = qend > last ? qend - last : 0;
        kvmem_diag("KVMEM_TRACE query_loc method=query_last tokens=[%d,%d)\n",
                qbegin, qend);
    }
    if (qbegin >= qend) {
        qbegin = 0;
    }
}

bool llama_driver_native_query_span(const LlamaEngineState & st, const std::string & formatted,
                                      const common_chat_templates_inputs & inputs, const kvmem_prompt & prompt,
                                      int & begin, int & end) {
    // Render the structured prefix ending at the real last user. This retains
    // native vision wrappers and does not confuse tool_response user-style
    // blocks with an actual user message. Require an exact causal prefix match.
    auto prefix_inputs = inputs;
    while (!prefix_inputs.messages.empty() && prefix_inputs.messages.back().role != "user")
        prefix_inputs.messages.pop_back();
    if (prefix_inputs.messages.empty()) return false;
    prefix_inputs.add_generation_prompt = false;
    std::string prefix;
    try {
        prefix = common_chat_templates_apply(st.tmpls.get(), prefix_inputs).prompt;
    } catch (const std::exception &) {
        return false; // A template may require the full trailing tool sequence.
    }
    size_t c0 = 0, c1 = 0;
    int count = 0, pick = -1;
    if (!find_last_user_role_block(prefix, "", c0, c1, count, pick) ||
            formatted.compare(0, c1, prefix, 0, c1) != 0) return false;
    int full_count = 0, unused = -1;
    if (!find_last_user_role_block(formatted, "", c0, c1, full_count, unused)) return false;
    common_chat_msg_delimiters delimiters;
    delimiters.add(COMMON_CHAT_ROLE_USER, "<|im_start|>user");
    delimiters.add(COMMON_CHAT_ROLE_UNKNOWN, "<|im_end|>");
    delimiters.tokenize(st.vocab);
    const auto spans = prompt.message_spans(delimiters);
    std::vector<common_chat_msg_span> users;
    for (const auto & span : spans.spans) if (span.role == COMMON_CHAT_ROLE_USER) users.push_back(span);
    if ((int) users.size() != full_count || pick < 0 || pick >= (int) users.size()) return false;
    begin = users[pick].pos + delimiters.delimiters.front().tokens.size();
    end = users[pick].pos + users[pick].len;
    for (const auto & image : prompt.media_ranges()) {
        if ((int) image.first >= begin && (int) image.second <= end) begin = image.second;
    }
    if (begin >= end) return false;
    std::string text;
    for (int row = begin; row < end; ++row)
        text += common_token_to_piece(st.vocab, prompt.tokens[row], false);
    if (trim_copy(text).empty()) return false;
    kvmem_diag("KVMEM_TRACE query_loc method=native_role pick=%d tokens=[%d,%d)\n", pick, begin, end);
    return true;
}

void llama_driver_clamp_query(const LlamaEngineState & st, int & qbegin, int & qend) {
    const int cap = st.query_max_tokens;
    if (cap > 0 && qend > qbegin && (qend - qbegin) > cap) {
        kvmem_diag("KVMEM_TRACE query_clamp span=[%d,%d) tokens=%d cap=%d -> [%d,%d)\n",
                qbegin, qend, qend - qbegin, cap, qend - cap, qend);
        qbegin = qend - cap;
    }
}
