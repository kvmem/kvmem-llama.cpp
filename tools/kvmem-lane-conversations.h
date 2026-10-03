#pragma once

// Included after the server's conversation payload and checkpoint helpers.
class kvmem_lane_conversations {
    using bundle_ptr = std::unique_ptr<llama_kvmem_store_bundle, decltype(&llama_kvmem_store_bundle_free)>;
    struct snapshot {
        kvmem_store_match match;
        std::shared_ptr<const kvmem_prompt> prompt;
    };
    struct entry {
        int lane = -1;
        bool busy = false, switching = false;
        size_t refs = 0;
        uint64_t used = 0, bytes = 0;
        std::string client_id;
        bundle_ptr parked {nullptr, llama_kvmem_store_bundle_free};
        kvmem_conversation payload;
        snapshot committed;
    };
public:
    struct operation {
        int lane = -1, id = -1, outgoing = -1;
        bool fresh = false, added = false, attached = false, ready = false, finished = false;
        size_t prepare_begin = 0;
    };
    class reference {
    public:
        ~reference() {
            std::lock_guard<std::mutex> lock(owner_.mu_);
            if (registered_ && !key_.empty()) {
                auto it = owner_.waiting_keys_.find(key_);
                if (it != owner_.waiting_keys_.end() && --it->second == 0) owner_.waiting_keys_.erase(it);
            }
            auto it = owner_.entries_.find(id_);
            if (it != owner_.entries_.end() && id_ >= 0) --it->second.refs;
        }
    private:
        friend class kvmem_lane_conversations;
        reference(kvmem_lane_conversations & owner, std::string key) : owner_(owner), key_(std::move(key)) {}
        kvmem_lane_conversations & owner_;
        std::string key_;
        int id_ = -1;
        bool registered_ = false;
    };
    kvmem_lane_conversations(const std::vector<ServerState *> & lanes, int count, uint64_t bytes)
        : lanes_(lanes), resident_(lanes.size()), max_(count), bytes_max_(bytes) {
        for (size_t lane = 0; lane < lanes.size(); ++lane) {
            const int id = next_id_++;
            auto & held = entries_.emplace(id, entry{}).first->second;
            held.lane = (int)lane;
            kvmem_execution_scope execution(lanes[lane]->execution.get());
            held.bytes = llama_kvmem_store_active_bytes();
            if (lane == 0) has_recurrent_ = llama_kvmem_has_recurrent();
            resident_[lane] = id;
        }
    }
    std::shared_ptr<reference> pin(const std::string & key) {
        auto result = std::shared_ptr<reference>(new reference(*this, key));
        std::lock_guard<std::mutex> lock(mu_);
        if (!key.empty()) ++waiting_keys_[key];
        result->registered_ = true;
        return result;
    }
    size_t preview(const kvmem_prompt & prompt, const std::string & key, bool reset,
                   const std::shared_ptr<reference> & ref) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto plan = plan_locked(prompt, key);
        if (!reset && ref && ref->id_ < 0 && plan.id >= 0) {
            ref->id_ = plan.id;
            ++entries_.at(plan.id).refs;
        }
        return reset ? 0 : (size_t) std::max(0, plan.keep);
    }
    int select(const kvmem_prompt & prompt, const std::string & key, bool reset,
               const std::vector<bool> & busy, bool reserve, operation & op) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto plan = plan_locked(prompt, key);
        int id = reset ? -1 : plan.id;
        bool fresh = reset || plan.action == kvmem_store_action::fresh;
        if (id < 0 && !key.empty()) {
            for (const auto & item : entries_) if (item.second.client_id == key) { id = item.first; break; }
        }
        if (id < 0) {
            for (const auto & item : entries_) {
                const auto & held = item.second;
                if (held.client_id.empty() && held.committed.match.rows == 0 && eligible(held, busy)) {
                    id = item.first; break;
                }
            }
        }
        if (id < 0 && entries_.size() >= (size_t)max_) {
            for (const auto & item : entries_) {
                if (!eligible(item.second, busy) || protected_entry(item.second)) continue;
                if (id < 0 || item.second.used < entries_.at(id).used) id = item.first;
            }
            if (id < 0) return -1;
        }
        if (id >= 0 && !eligible(entries_.at(id), busy)) return -1;
        const size_t begin = fresh ? 0 : (size_t) std::max(0, plan.keep);
        op.prepare_begin = begin;
        if (!prompt.media_ready(begin)) return -2;
        int lane = id >= 0 ? entries_.at(id).lane : -1;
        if (lane < 0) {
            for (size_t candidate = 0; candidate < busy.size(); ++candidate) {
                const auto & outgoing = entries_.at(resident_[candidate]);
                if (!busy[candidate] && !outgoing.busy && !outgoing.switching) { lane = (int)candidate; break; }
            }
        }
        if (lane < 0) return -1;
        if (!reserve) return lane;
        op.added = id < 0;
        if (op.added) {
            id = next_id_;
            entries_.emplace(id, entry{});
            ++next_id_;
        }
        op.lane = lane;
        op.id = id;
        op.outgoing = resident_[lane];
        op.fresh = fresh;
        entries_.at(id).busy = true;
        if (id != op.outgoing) entries_.at(op.outgoing).switching = true;
        return lane;
    }
    void attach(operation & op, ServerState & st, const std::string & key) {
        std::string client_id = key; // Allocate before changing either owner.
        entry * incoming;
        entry * outgoing;
        {
            std::lock_guard<std::mutex> lock(mu_);
            incoming = &entries_.at(op.id);
            outgoing = &entries_.at(op.outgoing);
        }
        bool restaged = true;
        if (op.id == op.outgoing) {
            if (op.fresh) llama_driver_clear(st);
        } else {
            bundle_ptr bundle {nullptr, llama_kvmem_store_bundle_free};
            bundle = std::move(incoming->parked);
            if (!bundle) bundle.reset(llama_kvmem_store_bundle_create());
            if (op.fresh) {
                llama_kvmem_store_bundle_reset(bundle.get());
                llama_driver_drop_conversation(incoming->payload);
            }
            llama_driver_swap_conversation(st, outgoing->payload);
            try {
                restaged = llama_kvmem_store_bundle_swap(bundle.get());
            } catch (...) {
                incoming->parked = std::move(bundle);
                llama_driver_swap_conversation(st, outgoing->payload);
                if (!llama_kvmem_store_n_tokens() && !st.cached_tokens.empty()) llama_driver_clear(st);
                throw;
            }
            outgoing->parked = std::move(bundle);
            llama_driver_swap_conversation(st, incoming->payload);
            {
                std::lock_guard<std::mutex> lock(mu_);
                // The handover has happened even if subsequent repair fails.
                incoming->lane = op.lane;
                outgoing->lane = -1;
                resident_[op.lane] = op.id;
                op.attached = true;
            }
            st.mm_live_checkpoint.reset();
            if (!restaged && !st.cached_tokens.empty()) llama_driver_clear(st);
            if (!llama_kvmem_store_bundle_rows(outgoing->parked.get())) llama_driver_drop_conversation(outgoing->payload);
        }
        st.mm_reset_requested = false;
        std::lock_guard<std::mutex> lock(mu_);
        // Publish the actual owners before any accounting allocation can fail.
        incoming->lane = op.lane;
        resident_[op.lane] = op.id;
        op.attached = true;
        if (op.fresh && !op.added && (!incoming->client_id.empty() || incoming->committed.match.rows > 0)) ++evictions_;
        incoming->client_id.swap(client_id);
        if (op.id != op.outgoing) {
            outgoing->lane = -1;
            outgoing->committed = describe(outgoing->payload, llama_kvmem_store_bundle_rows(outgoing->parked.get()));
            outgoing->bytes = payload_bytes(outgoing->payload) + llama_kvmem_store_bundle_bytes(outgoing->parked.get());
            outgoing->switching = false;
            ++switches_;
        }
        if (op.fresh) ++forks_; else ++extends_;
        op.ready = true;
        kvmem_diag("KVMEM_LANE_STORE lane=%d out=%d in=%d fresh=%d restaged=%d stores=%zu\n",
                    op.lane, op.outgoing, op.id, (int)op.fresh, (int)restaged, entries_.size());
    }
    void finish(operation & op, ServerState & st) noexcept {
        if (op.finished) return;
        op.finished = true;
        try {
            kvmem_execution_scope execution(st.execution.get());
            llama_driver_finish(st);
            if (st.cached_prompt) st.cached_prompt = st.cached_prompt->cache_index();
            if (st.mm_query && st.mm_query->prefix) {
                auto query = std::make_shared<MultimodalQuery>(*st.mm_query);
                query->prefix = query->prefix->cache_index();
                st.mm_query = std::move(query);
            }
            auto committed = op.attached && !op.ready ? snapshot{} : describe(st, llama_kvmem_store_n_tokens());
            const uint64_t bytes = payload_bytes(st) + llama_kvmem_store_active_bytes();
            std::lock_guard<std::mutex> lock(mu_);
            const int active = op.attached ? op.id : op.outgoing;
            auto & held = entries_.at(active);
            held.committed = std::move(committed);
            held.bytes = bytes;
            held.used = ++clock_;
            if (!op.attached && op.added) entries_.erase(op.id);
            else entries_.at(op.id).busy = false;
            held.switching = false;
            if (!op.ready) {
                auto target = entries_.find(op.id);
                if (target != entries_.end()) target->second.committed = snapshot{};
                if (op.attached && op.outgoing != op.id) entries_.at(op.outgoing).committed = snapshot{};
            }
            entries_.at(op.outgoing).switching = false;
            enforce_bytes();
        } catch (const std::exception & e) {
            LOG_ERR("srv    KVMEM lane commit failed: %s\n", e.what());
            std::lock_guard<std::mutex> lock(mu_);
            if (!op.attached && op.added) entries_.erase(op.id);
            auto it = entries_.find(op.attached ? op.id : op.outgoing);
            if (it != entries_.end()) { it->second.busy = false; it->second.committed = snapshot{}; }
            auto target = entries_.find(op.id);
            if (target != entries_.end()) target->second.busy = false;
            auto out = entries_.find(op.outgoing);
            if (out != entries_.end()) { out->second.switching = false; out->second.committed = snapshot{}; }
        }
        st.active_prompt.reset();
    }
    json status(int lane = -1) const {
        std::lock_guard<std::mutex> lock(mu_);
        json result = {{"count", entries_.size()}, {"max", max_}, {"bytes", bytes_locked()},
            {"bytes_max", bytes_max_}, {"switches", switches_}, {"evictions", evictions_},
            {"extends", extends_}, {"forks", forks_}};
        if (lane >= 0) {
            const int id = resident_.at((size_t)lane);
            result["resident"] = id;
            result["switching"] = entries_.at(id).switching;
        }
        return result;
    }
private:
    mutable std::mutex mu_;
    std::vector<ServerState *> lanes_;
    std::vector<int> resident_;
    std::map<int, entry> entries_;
    std::map<std::string, size_t> waiting_keys_;
    int max_, next_id_ = 0;
    bool warned_bytes_ = false, has_recurrent_ = false;
    uint64_t bytes_max_, clock_ = 0, switches_ = 0, evictions_ = 0, extends_ = 0, forks_ = 0;
    kvmem_store_plan plan_locked(const kvmem_prompt & prompt, const std::string & key) const {
        std::vector<kvmem_store_match> matches;
        for (const auto & item : entries_) {
            auto match = item.second.committed.match;
            match.id = item.first;
            match.used = item.second.used;
            match.bytes = item.second.bytes;
            match.client_id = item.second.client_id;
            match.lcp = item.second.committed.prompt ? (int)prompt.common_prefix(*item.second.committed.prompt) : 0;
            if (!has_recurrent_ && !lanes_[0]->vision && !lanes_[0]->query_policy_user)
                match.ckpt_rows.push_back(match.lcp);
            matches.push_back(std::move(match));
        }
        kvmem_store_limits limits;
        limits.max_stores = std::numeric_limits<int>::max();
        return kvmem_store_select(matches, (int)prompt.tokens.size() - (lanes_[0]->spec.ok ? 1 : 0),
                                    lanes_[0]->spec.ok, key, limits);
    }
    bool eligible(const entry & held, const std::vector<bool> & busy) const {
        return !held.busy && !held.switching && (held.lane < 0 || !busy[(size_t)held.lane]);
    }
    bool protected_entry(const entry & held) const {
        return held.refs > 0 || (!held.client_id.empty() && waiting_keys_.count(held.client_id));
    }
    template<class Source> static snapshot describe(const Source & source, uint32_t stored) {
        snapshot result;
        result.prompt = source.cached_prompt;
        result.match.rows = (int)source.cached_tokens.size();
        result.match.last_n_gen = source.last_n_gen;
        // The legacy text prefill uses GDN checkpoints, not mm_live_row.
        result.match.live_row = (int)stored;
        for (const auto & checkpoint : source.mm_checkpoints) result.match.ckpt_rows.push_back(checkpoint.row);
        if (source.gdn_ckpt_pos >= 0 && !source.gdn_ckpt.empty()) result.match.ckpt_rows.push_back(source.gdn_ckpt_pos + 1);
        if (source.gdn_ckpt_query_pos >= 0 && !source.gdn_ckpt_query.empty()) result.match.ckpt_rows.push_back(source.gdn_ckpt_query_pos + 1);
        return result;
    }
    template<class Source> static uint64_t payload_bytes(const Source & source) {
        uint64_t bytes = source.cached_tokens.capacity()*sizeof(llama_token);
        bytes += source.gdn_ckpt.capacity() + source.gdn_carry.capacity() + source.gdn_query_carry.capacity();
        bytes += source.gdn_ckpt_query.capacity() + source.last_user_text.capacity();
        bytes += source.mm_checkpoints.capacity()*sizeof(MultimodalCheckpoint);
        std::set<const MultimodalCheckpointData *> unique;
        for (const auto & checkpoint : source.mm_checkpoints) {
            if (checkpoint.data && unique.insert(checkpoint.data.get()).second) bytes += checkpoint.data->bytes();
        }
        if (source.mm_live_checkpoint && unique.insert(source.mm_live_checkpoint.get()).second) bytes += source.mm_live_checkpoint->bytes();
        if (source.cached_prompt) bytes += source.cached_prompt->index_bytes();
        if (source.mm_query) {
            bytes += sizeof(MultimodalQuery) + source.mm_query->user.capacity();
            bytes += source.mm_query->state.count.capacity()*sizeof(uint32_t);
            for (const auto & sum : source.mm_query->state.sum) bytes += sizeof(sum) + sum.capacity()*sizeof(float);
            if (source.mm_query->prefix) bytes += source.mm_query->prefix->index_bytes();
        }
        return bytes;
    }
    uint64_t bytes_locked() const {
        uint64_t bytes = 0;
        for (const auto & item : entries_) bytes += item.second.bytes;
        return bytes;
    }
    void enforce_bytes() {
        while (bytes_max_ && bytes_locked() > bytes_max_) {
            auto victim = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                const auto & held = it->second;
                if (held.lane >= 0 || held.busy || held.switching || protected_entry(held)) continue;
                if (victim == entries_.end() || held.used < victim->second.used) victim = it;
            }
            if (victim == entries_.end()) break;
            entries_.erase(victim);
            ++evictions_;
        }
        const bool over = bytes_max_ && bytes_locked() > bytes_max_;
        if (over && !warned_bytes_) LOG_WRN("srv    KVMEM retained lane/queued stores exceed soft host byte cap\n");
        warned_bytes_ = over;
    }
};
