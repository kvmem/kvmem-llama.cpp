#pragma once

#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-kvmem-hooks.h"

#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

#include <condition_variable>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct llama_model;
struct llama_cparams;
struct llama_memory_params;
struct ggml_tensor;
class llama_memory_recurrent;
class llama_memory_kvmem_mtp;

// Bounded block-slot pool over a llama_kv_cache.
//
// GPU attention cache size = min(n_ctx, budget + gen_reserve). Each logical
// KVMem block occupies one slot of `block_tokens` cells. Reselect never packs
// cells into [0, W); resident blocks keep their slot. init_batch returns a
// llama_kv_cache_context so llama-graph.cpp can static_cast as usual.
class llama_memory_kvmem : public llama_memory_i {
public:
    llama_memory_kvmem(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams,
            llama_kv_cache * ext_kv = nullptr);

    ~llama_memory_kvmem() override;

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

    llama_kv_cache * get_kv() { return kv_; }
    kvmem::KvMemRuntime & runtime() { return *runtime_; }
    const kvmem::KvMemRuntime & runtime() const { return *runtime_; }

    uint32_t kv_size() const { return kv_size_; }
    uint32_t block_tokens() const { return block_tokens_; }
    uint32_t n_slots() const { return n_slots_; }

    // Slot-pool prepare used by both the dense KVMem memory and the hybrid
    // wrapper (attn half). Fills per-ubatch slot_info and the capture pos queue.
    bool prepare_ubatches(
            const std::vector<llama_ubatch> & ubatches,
            uint32_t n_new_tokens,
            llama_kv_cache::slot_info_vec_t & sinfos);
    void reset_policy();

    int32_t alloc_slot();
    void free_slot(int32_t slot);
    int32_t peek_free_slot() const;
    // Map original pos to (gpu_slot, offset in block). For a token the target
    // has not appended yet (MTP draft), predict the slot the next alloc would
    // take without popping the free list. Returns false if no mapping exists.
    bool slot_for_orig_pos(llama_pos pos, int32_t * slot, uint32_t * off) const;
    const kvmem::KvMemStore & store() const { return runtime_->store(); }

    void note_ubatch_pos(const std::vector<llama_pos> & pos);
    void reset_query_acc();
    void register_capture(struct ggml_tensor * t, int il, char which);
    void capture_on_new_graph();
    bool capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) const;
    void harvest_pending(struct ggml_backend_sched * sched);
    void harvest_flush();
    void harvest_perf_print_sum();
    void harvest_capture(struct ggml_tensor * t, int il, char which);
    void apply_retrieval();
    void set_turn_spans(const llama_kvmem_turn_spans & spans);
    bool query_contains(llama_pos row) const;
    bool query_overlaps(uint32_t n, const llama_pos * rows) const;
    llama_kvmem_attention_view attention_view(bool canonical = true) const;
    bool can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason) const;
    llama_kvmem_selection preview_retrieval();
    bool selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t generation_rows) const;
    bool commit_unchanged(const llama_kvmem_attention_view & view, const llama_kvmem_selection & selection);
    void apply_selection(const llama_kvmem_selection & selection);
    bool commit_resident(bool canonical = true);
    bool get_query(llama_kvmem_query_state & state);
    bool set_query(const llama_kvmem_query_state & state);
    void freeze_query(bool frozen) { query_frozen_ = frozen; }
    // The follower also invalidates a pending proof when old draft KV changes.
    void note_attention_change() { ++attention_epoch_; }
    bool query_replay_fits(uint32_t query_begin, uint32_t prompt_end) const;
    void dump_kv_compare(int32_t block_id, bool writeback_test = false);
    void trace_working_set(const char * tag) const;
    void set_replay(bool replay);
    bool replay() const { return replay_; }
    bool want_prefill_capture() const {
        return prefill_capture_ && !retrieval_pinned_ && !replay_;
    }
    // Recapture Q for retrieval even while replaying a cached query span.
    bool want_q_capture() const {
        return method_ == 1 && !retrieval_pinned_ && !query_frozen_ &&
            (explicit_spans_ ? !turn_spans_.query.empty() : query_begin_ >= 0);
    }
    bool want_decode_mean() const {
        return retrieval_pinned_ && method_ == 1 && !replay_;
    }
    void end_prefill_capture() { prefill_capture_ = false; }
    // Next request continues this sequence: flush decode mean, unpin
    // retrieval, harvest mean-K for the suffix. Does not wipe prefix KV.
    void begin_cached_turn(bool reset_query = true);
    // Same-query skip: keep the retrieved GPU window. Prefill still harvests
    // the new tail; recency pressure must not evict selected history.
    void keep_selected_window() { keep_selected_ = true; }
    // After skip/reselect prefill: pin so decode mean-K uses gen_reserve.
    void pin_working_set() {
        retrieval_pinned_ = true;
        keep_selected_ = true;
    }
    size_t free_slot_count() const { return free_slots_.size(); }
    void truncate_cached(uint32_t n_past);
    void occupy_in(llama_kv_cache * cache, uint32_t block_id);
    llama_pos model_pos(uint32_t logical_pos) const;
    bool remove_logical(llama_context * ctx, llama_pos begin, llama_pos end);
    uint32_t store_n_tokens() const {
        return runtime_ ? runtime_->store().total_tokens() : 0;
    }
    llama_pos recr_pos_max() const;
    void decode_mean_commit(uint32_t n_keep);
    void decode_mean_discard();
    void decode_mean_flush();
    void set_recurrent(llama_memory_recurrent * recr);
    bool has_recurrent() const { return recr_ != nullptr; }
    bool gdn_replay_enabled() const;
    bool gdn_replay_begin(llama_pos start, uint32_t width);
    bool gdn_replay_commit(llama_context * ctx, uint32_t n_keep);
    void set_query_span(int32_t begin, int32_t end) {
        harvest_flush();
        explicit_spans_ = false;
        query_begin_ = begin;
        query_end_ = end;
    }
    void set_force_pos(int32_t pos) { force_pos_ = pos; }
    void set_mtp_follower(llama_memory_kvmem_mtp * mtp) { mtp_ = mtp; }
    llama_memory_kvmem_mtp * mtp_follower() { return mtp_; }

    const kvmem::RopeConfig & rope() const { return rope_; }
    ggml_type type_k() const { return type_k_; }
    ggml_type type_v() const { return type_v_; }
    bool v_trans() const { return v_trans_; }
    uint32_t n_embd_k() const { return n_embd_k_; }
    uint32_t n_embd_v() const { return n_embd_v_; }

    kvmem::RawKvStore & raw() { return *raw_; }

private:
    friend struct kvmem_transfer_test_access;
    struct SlotBackend : public kvmem::KvMemBackend {
        llama_memory_kvmem * owner = nullptr;
        int32_t alloc_gpu_slot() override { return owner->alloc_slot(); }
        void free_gpu_slot(int32_t slot) override { owner->free_slot(slot); }
        void copy_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes) override {
            owner->copy_gpu_block_to_host(block_id, gpu_slot, host, bytes);
        }
        void copy_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes) override {
            owner->copy_gpu_block_from_host(block_id, gpu_slot, host, bytes);
        }
    };

    uint32_t resident_tokens() const;
    bool prepare_working_set(uint32_t n_new_tokens);
    void apply_plan_to_kv(const kvmem::KvMemPlan & plan);
    // Place GPU-resident blocks into slots 0..N-1 in orig_pos order.
    // Resident KV is copied slot-to-slot; cold blocks memcpy packed GPU K/V.
    bool layout_gpu_slots_by_orig_pos();
    bool gpu_kv_already_resident(uint32_t block_id) const;
    bool gpu_kv_complete(uint32_t block_id, const llama_kv_cache * cache) const;
    std::vector<uint32_t> retrieval_mandatory() const;
    void occupy_block_cells(uint32_t block_id);
    void reset_slots();
    void trace_plan(const char * tag, const kvmem::KvMemPlan & plan) const;
    void write_block_to_gpu(uint32_t block_id);
    void harvest_gpu_v(uint32_t block_id);
    void harvest_gpu_v_commit();
    void harvest_gpu_v_flush_slab();
    void harvest_write_batch();
    // After a prefill graph, enqueue packed K/V D2H for GPU-resident
    // full blocks. Does not wait; apply_plan / retrieval commit.
    void harvest_full_blocks_async();
    void decode_mean_ingest(struct ggml_backend_sched * sched);
    void decode_mean_reset();
    void decode_mean_add_range(uint32_t tok0, uint32_t n_add);
    void decode_mean_zero_acc();
    bool decode_mean_add_host_layer(struct ggml_tensor * t, int il, uint32_t tok0, uint32_t n_add);
    void decode_mean_print_sum();
    struct HarvestVJob {
        uint32_t pos0 = 0;
        uint32_t n = 0;
        uint32_t il = 0;
        bool is_k = false;
        const uint8_t * gpu_src = nullptr;
        size_t nbytes = 0;
        size_t pin_off = 0;
        ggml_tensor * vt = nullptr;
        size_t tensor_off = 0;
    };
    struct HarvestVBatch {
        int slot = -1;
        std::vector<HarvestVJob> jobs;
    };
    std::vector<HarvestVJob> harvest_v_jobs_;
    HarvestVBatch harvest_v_pending_;
    // 1 while a block's packed D2H is queued or in flight (until commit).
    std::vector<uint8_t> harvest_gpu_queued_;
    void copy_gpu_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes);
    void copy_gpu_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes);
    void score_retrieval();
    bool read_gpu_block(uint32_t block_id, uint32_t il, bool is_k, std::vector<float> & out) const;
    static void kv_stats(const char * tag, const float * a, const float * b, size_t n);
    static void tensor_to_f32_token_major(const struct ggml_tensor * t, std::vector<float> & out);
    static void bytes_to_f32_token_major(const uint8_t * data, ggml_type type,
                                         int64_t d, int64_t h, int64_t n,
                                         size_t nb0, size_t nb1, size_t nb2,
                                         std::vector<float> & out);
    static void bytes_to_f16_token_major(const uint8_t * data, ggml_type type,
                                         int64_t d, int64_t h, int64_t n,
                                         size_t nb0, size_t nb1, size_t nb2,
                                         std::vector<uint16_t> & out);
    void harvest_from_host(int il, char which, const uint8_t * host,
                           ggml_type type, int64_t d, int64_t h, int64_t n,
                           size_t nb0, size_t nb1, size_t nb2);
    bool d2h_init();
    void d2h_free();
    void d2h_commit(int slot);
    bool d2h_submit(struct ggml_backend * be);
    bool harvest_perf_on() const { return perf_.enabled; }
    void harvest_perf_emit_graph_line();
    void retr_perf_print();
    void harvest_worker_start();
    void harvest_worker_stop();
    void harvest_loop();
    void harvest_wait_slot(int slot);
    bool harvest_worker_on() const;

    struct HarvestPerf {
        bool enabled = false;
        bool sum_printed = false;
        bool graph_line_printed = false;
        uint32_t n_ubatch = 0;
        uint32_t n_tok = 0;
        int64_t harvest_entry_us = 0;
        int64_t sync_us = 0;
        int64_t d2d_us = 0;
        int64_t snap_wait_us = 0;
        int64_t d2h_submit_us = 0;
        int64_t commit_us = 0;
        int64_t d2h_wait_us = 0;
        int64_t pack_us = 0;
        int64_t nvme_us = 0;
        uint64_t nvme_bytes = 0;
        uint64_t nvme_syscalls = 0;
        int64_t last_sync_us = 0;
        int64_t last_d2d_us = 0;
        int64_t last_snap_wait_us = 0;
        int64_t last_d2h_submit_us = 0;
        int64_t last_commit_us = 0;
        int64_t last_d2h_wait_us = 0;
        int64_t last_pack_us = 0;
        int64_t last_nvme_us = 0;
        uint64_t last_nvme_bytes = 0;
        uint64_t last_nvme_syscalls = 0;
        uint32_t n_pressure = 0;
        uint32_t n_pressure_out = 0;
    };

    struct RetrPerf {
        bool enabled = false;
        int64_t total_us = 0;
        int64_t flush_us = 0;
        int64_t score_us = 0;
        int64_t plan_us = 0;
        int64_t stage_out_us = 0;
        int64_t stage_out_gpu_us = 0;
        int64_t stage_out_host_us = 0;
        int64_t admit_us = 0;
        int64_t seq_rm_us = 0;
        int64_t occupy_us = 0;
        int64_t layout_d2h_us = 0;
        int64_t layout_h2d_us = 0;
        int64_t copy_us = 0;
        int64_t rope_us = 0;
        int64_t hadamard_us = 0;
        int64_t set_us = 0;
        int64_t mtp_us = 0;
        int64_t dump_us = 0;
        uint32_t n_move = 0;
        uint32_t n_raw = 0;
        uint32_t n_skip = 0;
        uint32_t n_stage_in = 0;
        int laid_out = 0;
    };

    const llama_model & model_;
    uint32_t block_tokens_ = 128;
    uint32_t kv_size_ = 0;
    uint32_t n_slots_ = 0;
    bool multi_gpu_ = false;
    bool trace_ = false;

    std::unique_ptr<llama_kv_cache> kv_owned_;
    llama_kv_cache * kv_ = nullptr;
    llama_memory_recurrent * recr_ = nullptr;
    struct GdnReplay;
    std::unique_ptr<GdnReplay> gdn_replay_;
    llama_memory_kvmem_mtp * mtp_ = nullptr;
    SlotBackend backend_;
    std::unique_ptr<kvmem::KvMemRuntime> runtime_;
    std::unique_ptr<kvmem::RawKvStore> raw_;
    std::vector<int32_t> free_slots_;
    struct RowPosition {
        std::array<llama_pos, 4> pos{};
        llama_token token = LLAMA_TOKEN_NULL;
        bool spatial = false;
    };
    std::vector<RowPosition> row_positions_;

    kvmem::RopeConfig rope_{};
    uint32_t n_layer_ = 0;
    uint32_t n_embd_k_ = 0;
    uint32_t n_embd_v_ = 0;
    uint32_t n_head_ = 0;
    uint32_t n_head_kv_ = 0;
    uint32_t n_embd_head_ = 0;
    ggml_type type_k_ = GGML_TYPE_F16;
    ggml_type type_v_ = GGML_TYPE_F16;
    bool v_trans_ = false;
    bool replay_ = false;
    bool retrieval_pinned_ = false;
    bool keep_selected_ = false;
    bool prefill_capture_ = true;
    int32_t method_ = 0;
    int32_t query_begin_ = -1;
    int32_t query_end_ = -1;
    int32_t force_pos_ = -1;
    llama_kvmem_turn_spans turn_spans_;
    bool explicit_spans_ = false;
    bool query_frozen_ = false;
    uint64_t attention_epoch_ = 0;

    std::vector<std::vector<llama_pos>> pos_queue_;
    std::vector<llama_pos> cur_pos_;
    struct CaptureNode {
        ggml_tensor * t = nullptr;
        int il = 0;
        char which = 0;
    };
    std::vector<CaptureNode> pending_capture_;
    std::vector<CaptureNode> decode_mean_pending_k_;
    std::vector<llama_pos> decode_mean_pending_pos_;
    uint32_t decode_mean_n_ = 0;
    uint32_t decode_mean_block_ = ~0u;
    uint32_t decode_mean_pos0_ = 0;
    std::vector<std::vector<float>> decode_mean_host_;
    std::vector<uint8_t> decode_mean_src_; // 0 none, 1 gpu, 2 host
    struct DecodeMeanStats {
        uint32_t n_tok = 0;
        uint32_t n_flush = 0;
        uint32_t n_gpu = 0;
        uint32_t n_host = 0;
        uint32_t n_miss = 0;
        uint32_t last_block = ~0u;
        uint32_t last_n = 0;
        uint32_t last_layers = 0;
        float last_rms = 0.0f;
        bool printed = false;
    };
    DecodeMeanStats decode_mean_stats_;
    bool graph_has_q_ = false;
    bool graph_has_k_ = false;
    bool graph_has_record_ = false;
    struct CaptureD2hPipe;
    std::unique_ptr<CaptureD2hPipe> d2h_;
    struct HarvestWorker {
        std::mutex mu;
        std::condition_variable cv;
        std::vector<int> q;
        std::thread th;
        bool stop = false;
    };
    std::unique_ptr<HarvestWorker> harvest_w_;
    HarvestPerf perf_;
    RetrPerf retr_;
    std::vector<std::vector<float>> q_sum_;
    std::vector<uint32_t> q_count_;
};

// GPU attn-cache cell count for a KVMem slot pool (budget + gen_reserve,
// clamped to n_ctx on identity). Hybrid uses this as llama_memory_hybrid's
// attn kv_size so the two halves agree.
uint32_t llama_kvmem_pool_cells(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams);
