#pragma once

#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-kvmem.h"

#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

struct ggml_backend_sched;

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

// MTP draft KV as a lockstep follower of the target slot-pool.
//
// kv_size and the slot count are copied from target (budget + gen_reserve),
// never recomputed from draft n_ctx. The same block_id maps to the same slot
// index and the same original pos on the cell. Follower does not alloc/free
// slots.
class llama_memory_kvmem_mtp : public llama_memory_i {
public:
    llama_memory_kvmem_mtp(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams,
            llama_memory_kvmem * target);

    ~llama_memory_kvmem_mtp() override;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t kv_size() const { return kv_size_; }
    llama_kv_cache * get_kv() { return kv_.get(); }
    llama_memory_kvmem * target() { return target_; }

    bool is_mtp_layer(int il) const { return il >= 0 && (uint32_t) il >= n_layer_trunk_; }
    void register_capture(struct ggml_tensor * t, int il, char which);
    void capture_on_new_graph();
    void harvest_pending(struct ggml_backend_sched * sched);
    void harvest_flush();
    uint32_t harvest_perf_n_ubatch() const { return perf_n_ubatch_; }
    int64_t harvest_perf_sync_us() const { return perf_sync_us_; }
    uint64_t harvest_perf_nvme_bytes() const { return perf_nvme_bytes_; }
    uint64_t harvest_perf_nvme_syscalls() const { return perf_nvme_syscalls_; }
    void on_stage_out(uint32_t block_id);
    void harvest_resident_v();
    // A block with no packed draft K in the mirror is occupied anyway, which
    // is what the single-store server has always done: the cells hold this
    // conversation's own decoded draft K. After a store swap they hold the
    // previous conversation's, so such a block is left out for as long as its
    // slot is tainted (see slot_tainted_ below).
    void follow_retrieval();
    void detach_target() { target_ = nullptr; }

    // Retrieval layout: keep native MTP GPU KV for blocks that still sit
    // at src_slot, D2D them to dst_slot (same permutation as the trunk).
    struct LayoutMove {
        int32_t src_slot = -1;
        int32_t dst_slot = -1;
        uint32_t n_tokens = 0;
    };
    bool slot_holds(int32_t slot, uint32_t orig_pos) const;
    bool layout_d2d(const LayoutMove * moves, size_t n_moves);
    void occupy_block(uint32_t block_id);
    bool remove_logical(llama_pos p0, llama_pos p1) {
        if (target_) target_->note_attention_change();
        return kv_->seq_rm_logical(0, p0, p1);
    }
    // raw_ is null between the two halves of a store swap, and stays null for
    // a conversation whose mirror could not be allocated (attach_conv logs and
    // continues, since the mirror is an accept-rate input and not a
    // correctness one). Every deref of it is guarded: the two accessors here,
    // the early returns in harvest_k(), harvest_v() and write_block_to_gpu(),
    // the coverage test in follow_retrieval(), and the null checks in clear()
    // and harvest_flush().
    void truncate_cached(uint32_t n) { if (raw_) raw_->truncate_to(n); }
    void invalidate_packed_from(uint32_t n) { if (raw_) raw_->invalidate_packed_from(n); }
    // Store swap, driven by the target one statement apart from its own, as
    // truncate_cached above is. The follower mirror is keyed by the target's
    // block ids, so it only means anything next to the trunk store it was
    // harvested against. harvest_flush() here is the follower's own write
    // barrier: on_stage_out() has just memcpy'd packed draft K/V into raw_ and
    // those writes may still be queued on the store's io thread.
    std::unique_ptr<kvmem::RawKvStore> swap_raw(std::unique_ptr<kvmem::RawKvStore> in) {
        harvest_flush();
        std::unique_ptr<kvmem::RawKvStore> out = std::move(raw_);
        raw_ = std::move(in);
        // The detach emptied the draft cells with seq_rm and nothing zeroed
        // the storage, so every slot still holds the outgoing conversation's
        // packed draft K until something writes over it. slot_tainted_ is
        // sized once, in the constructor, so marking the whole pool here
        // neither allocates nor throws: the trunk's detach calls this after it
        // has already moved its own runtime and host mirror out, where a throw
        // would leave the trunk attached with no store at all.
        std::fill(slot_tainted_.begin(), slot_tainted_.end(), true);
        return out;
    }
    std::unique_ptr<kvmem::RawKvStore> make_raw() const {
        return std::make_unique<kvmem::RawKvStore>(raw_cfg_);
    }
    // Empty the draft cells but keep the mirror, unlike clear(bool) which also
    // wipes raw_. Called from the target's detach after its own drain, which
    // is where on_stage_out() mirrors each resident block -- every one of
    // them, unless this conversation has no mirror to harvest into at all.
    // pos_queue_ is the only per-ubatch state that can outlive a request
    // here: pending_capture_ is emptied by harvest_pending() on every ubatch
    // and register_capture() on the follower is a no-op.
    void drop_gpu() {
        if (target_) target_->note_attention_change();
        (void) kv_->seq_rm(0, -1, -1);
        pos_queue_.clear();
    }

private:
    bool fill_from_target(const llama_ubatch & ubatch, llama_kv_cache::slot_info & out);
    void write_block_to_gpu(uint32_t block_id);
    void harvest_k(uint32_t block_id);
    void harvest_v(uint32_t block_id);

    // True while this slot's cells may still hold another conversation's
    // packed draft K. swap_raw() marks the whole pool and write_block_to_gpu()
    // clears one slot once a full block of this conversation's packed draft K
    // has been written over it; a partial block leaves the tail cells as they
    // were, so its slot stays marked. "The whole pool" is n_slots_ entries,
    // copied from the target rather than recomputed here: the trunk's pool is
    // ceil(kv_size_ / block_tokens_), so sizing this with that division
    // truncated left the highest slot unmarked whenever kv_size_ is not a
    // whole number of blocks. An index outside the vector would be a layout
    // disagreement with the trunk; it reads as tainted, which costs draft
    // context and can never hand the draft layer another conversation's K.
    // Only follow_retrieval's coverage-miss branch reads this, and it errs
    // towards leaving a slot marked: the cost is draft context the follower
    // does not have, never a wrong answer from the target model, which
    // verifies every drafted token. Every entry is false until the first
    // store swap, so nothing changes without --kvmem-conversations.
    bool slot_tainted(int32_t slot) const {
        if (slot < 0 || (size_t) slot >= slot_tainted_.size()) {
            return true;
        }
        return slot_tainted_[(size_t) slot];
    }
    void clear_slot_taint(int32_t slot) {
        if (slot >= 0 && (size_t) slot < slot_tainted_.size()) {
            slot_tainted_[(size_t) slot] = false;
        }
    }

    const llama_model & model_;
    llama_memory_kvmem * target_ = nullptr;
    uint32_t kv_size_ = 0;
    // The target's slot count, not a recomputation of it: slot_tainted_ is
    // indexed by the trunk's gpu_slot, so the two must agree exactly.
    uint32_t n_slots_ = 0;
    uint32_t block_tokens_ = 32;
    uint32_t n_layer_trunk_ = 0;
    uint32_t il_graph_ = 0;
    uint32_t n_embd_k_ = 0;
    uint32_t n_embd_v_ = 0;
    ggml_type type_k_ = GGML_TYPE_F16;
    ggml_type type_v_ = GGML_TYPE_F16;
    bool v_trans_ = false;
    bool trace_ = false;
    kvmem::RopeConfig rope_{};
    std::unique_ptr<llama_kv_cache> kv_;
    kvmem::RawKvStoreConfig raw_cfg_{};
    std::unique_ptr<kvmem::RawKvStore> raw_;
    std::vector<bool> slot_tainted_;

    struct CaptureNode {
        ggml_tensor * t = nullptr;
        char which = 0;
    };
    std::vector<CaptureNode> pending_capture_;
    std::vector<std::vector<llama_pos>> pos_queue_;
    std::vector<llama_pos> cur_pos_;
    uint32_t perf_n_ubatch_ = 0;
    int64_t perf_sync_us_ = 0;
    uint64_t perf_nvme_bytes_ = 0;
    uint64_t perf_nvme_syscalls_ = 0;
};
