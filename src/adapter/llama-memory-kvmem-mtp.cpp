#include "llama-memory-kvmem-mtp.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-kvmem-capture.h"
#include "llama-kvmem-hooks.h"
#include "llama-kvmem-stagein.h"
#include "llama-kvmem-transfer.h"
#include "llama-model.h"

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include "llama-kvmem-gpu.h"
#include "llama-kvmem-diag.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static uint8_t * kvmem_mtp_cuda_ptr(ggml_tensor * t) {
    if (!t || !t->data) {
        return nullptr;
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf || ggml_backend_buffer_is_host(buf)) {
        return nullptr;
    }
    return static_cast<uint8_t *>(t->data);
}

llama_memory_kvmem_mtp::llama_memory_kvmem_mtp(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams,
        llama_memory_kvmem * target) :
    model_(model),
    target_(target) {
    GGML_ASSERT(target_ && "MTP follower requires a KVMem target");
    trace_ = getenv("KVMEM_TRACE") != nullptr;
    kv_size_ = target_->kv_size();
    block_tokens_ = target_->block_tokens();
    if (block_tokens_ == 0) {
        block_tokens_ = 32;
    }
    n_layer_trunk_ = model.hparams.n_layer();
    il_graph_ = n_layer_trunk_;
    // Only slots follow the target. Byte layout follows the draft cache.
    n_embd_k_ = model.hparams.n_embd_k_gqa(il_graph_);
    n_embd_v_ = model.hparams.n_embd_v_gqa(il_graph_);
    type_k_ = params.type_k;
    type_v_ = params.type_v;
    v_trans_ = !cparams.flash_attn;
    rope_ = target_->rope();

    const uint32_t n_layer = n_layer_trunk_;
    llama_kv_cache::layer_filter_cb filter =
            [n_layer](uint32_t il) { return il >= n_layer; };

    kv_ = std::make_unique<llama_kv_cache>(
            model,
            model.hparams,
            type_k_,
            type_v_,
            v_trans_,
            cparams.offload_kqv,
            /* unified */ true,
            kv_size_,
            /* n_seq_max */ 1,
            /* n_pad */ 1,
            model.hparams.n_swa,
            model.hparams.swa_type,
            nullptr,
            filter,
            nullptr,
            nullptr,
            "kvmem-mtp");

    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const ggml_tensor * kt = kv_->get_k_storage(il_graph_);
    const ggml_tensor * vt = kv_->get_v_storage(il_graph_);
    if (!kt || !vt || kt->type != type_k_ || vt->type != type_v_ ||
            kt->ne[0] != n_embd_k_ || kt->ne[1] != kv_size_ || kt->nb[1] != krow ||
            (!v_trans_ && (vt->ne[0] != n_embd_v_ || vt->ne[1] != kv_size_ || vt->nb[1] != vrow))) {
        throw std::runtime_error("KVMem MTP cache layout does not match packed K/V transfers");
    }

    kvmem::RawKvStoreConfig rcfg;
    rcfg.n_layer = std::max(1u, model.hparams.n_layer_nextn);
    rcfg.n_embd_k = n_embd_k_;
    rcfg.n_embd_v = n_embd_v_;
    rcfg.block_tokens = block_tokens_;
    if (ggml_is_quantized(type_k_)) {
        rcfg.k_row_bytes = ggml_row_size(type_k_, n_embd_k_);
    }
    rcfg.k_gpu_row_bytes = ggml_row_size(type_k_, n_embd_k_);
    if (!v_trans_) {
        rcfg.v_gpu_row_bytes = ggml_row_size(type_v_, n_embd_v_);
    }
    const llama_kvmem_params * kp = llama_kvmem_get_params();
    if (kp && kp->raw_k_nvme) {
        const uint64_t ntok = std::max<uint64_t>(cparams.n_ctx, kv_size_);
        const uint64_t need =
                static_cast<uint64_t>(rcfg.n_layer) * ntok *
                (static_cast<uint64_t>(rcfg.n_embd_k) + rcfg.n_embd_v) *
                sizeof(uint16_t);
        rcfg.nvme_bytes = need + 32ull * 1024ull * 1024ull;
        rcfg.nvme_dir = (kp->nvme_dir && kp->nvme_dir[0])
                ? kp->nvme_dir
                : "/tmp/kvmem_nvme";
        rcfg.nvme_file = "kvmem_raw_mtp_k.bin";
    }
    raw_ = std::make_unique<kvmem::RawKvStore>(rcfg);

    size_t bytes = 0;
    for (const auto & kv : kv_->memory_breakdown()) {
        bytes += kv.second;
    }
    const uint32_t n_layers = (uint32_t) kv_->get_layer_ids().size();
    kvmem_diag("KVMEM_TRACE mtp_pool cells=%u target_cells=%u n_ctx=%u bytes=%zu layers=%u block_tokens=%u"
            " type_k=%s type_v=%s k_row_bytes=%zu v_row_bytes=%zu v_trans=%d\n",
            kv_size_, target_->kv_size(), cparams.n_ctx, bytes, n_layers, block_tokens_,
            ggml_type_name(kt->type), ggml_type_name(vt->type), krow, vrow, (int) v_trans_);
    LLAMA_LOG_INFO(
            "%s: KVMem MTP follower cells=%u target_cells=%u n_ctx=%u bytes=%.2f MiB layers=%u\n",
            __func__, kv_size_, target_->kv_size(), cparams.n_ctx,
            bytes / (1024.0 * 1024.0), n_layers);
    target_->set_mtp_follower(this);
    kvmem_mtp_bind(this);
}

llama_memory_kvmem_mtp::~llama_memory_kvmem_mtp() {
    kvmem_mtp_unbind(this);
    if (target_) {
        target_->set_mtp_follower(nullptr);
        target_ = nullptr;
    }
}

bool llama_memory_kvmem_mtp::fill_from_target(
        const llama_ubatch & ubatch,
        llama_kv_cache::slot_info & out) {
    if (!target_ || ubatch.n_tokens == 0 || !ubatch.pos) {
        return false;
    }
    out.s0 = 0;
    out.s1 = 0;
    out.resize(1);
    out.strm[0] = 0;
    out.idxs[0].clear();
    out.idxs[0].reserve(ubatch.n_tokens);

    int32_t first_slot = -1;
    uint32_t first_cell = 0;
    std::vector<llama_pos> pos_note;
    pos_note.reserve(ubatch.n_tokens);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id && ubatch.n_seq_id[i] > 1) {
            LLAMA_LOG_ERROR("%s: KVMem MTP is single-sequence only\n", __func__);
            return false;
        }
        const llama_pos pos = ubatch.logical_pos ? ubatch.logical_pos[i] : ubatch.pos[i];
        int32_t slot = -1;
        uint32_t off = 0;
        if (!target_->slot_for_orig_pos(pos, &slot, &off)) {
            LLAMA_LOG_ERROR("%s: no target slot for pos %d\n", __func__, (int) pos);
            return false;
        }
        const uint32_t cell = static_cast<uint32_t>(slot) * block_tokens_ + off;
        if (cell >= kv_size_) {
            LLAMA_LOG_ERROR("%s: cell %u >= kv_size %u (slot %d pos %d)\n",
                    __func__, cell, kv_size_, (int) slot, (int) pos);
            return false;
        }
        if (i == 0) {
            first_slot = slot;
            first_cell = cell;
        }
        out.idxs[0].push_back(cell);
        pos_note.push_back(pos);
    }
    pos_queue_.push_back(std::move(pos_note));
    if (trace_ && ubatch.n_tokens > 0) {
        kvmem_diag("KVMEM_TRACE mtp_occupy n=%u first_pos=%d first_slot=%d first_cell=%u "
                "tgt_slot=%d kv_size=%u\n",
                ubatch.n_tokens, (int) ubatch.pos[0], (int) first_slot, first_cell,
                (int) first_slot, kv_size_);
    }
    return out.idxs[0].size() == ubatch.n_tokens;
}

llama_memory_context_ptr llama_memory_kvmem_mtp::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);
    do {
        balloc.split_reset();
        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }
        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }
        llama_kv_cache::slot_info_vec_t sinfos;
        sinfos.reserve(ubatches.size());
        bool ok = true;
        for (const auto & ubatch : ubatches) {
            llama_kv_cache::slot_info sinfo;
            if (!fill_from_target(ubatch, sinfo)) {
                ok = false;
                break;
            }
            sinfos.push_back(std::move(sinfo));
        }
        if (!ok) {
            break;
        }
        return std::make_unique<llama_kv_cache_context>(
                kv_.get(), std::move(sinfos), std::move(ubatches));
    } while (false);
    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_kvmem_mtp::init_full() {
    return kv_->init_full();
}

llama_memory_context_ptr llama_memory_kvmem_mtp::init_update(llama_context * lctx, bool optimize) {
    return kv_->init_update(lctx, optimize);
}

void llama_memory_kvmem_mtp::clear(bool data) {
    if (target_) target_->note_attention_change();
    kv_->clear(data);
    raw_->clear();
    pos_queue_.clear();
}

bool llama_memory_kvmem_mtp::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (target_ && kv_->seq_pos_max(seq_id) >= std::max<llama_pos>(0, p0) &&
            (p1 < 0 || kv_->seq_pos_min(seq_id) < p1)) target_->note_attention_change();
    return kv_->seq_rm(seq_id, p0, p1);
}

void llama_memory_kvmem_mtp::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (target_) target_->note_attention_change();
    kv_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_kvmem_mtp::seq_keep(llama_seq_id seq_id) {
    if (target_) target_->note_attention_change();
    kv_->seq_keep(seq_id);
}

void llama_memory_kvmem_mtp::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (target_) target_->note_attention_change();
    kv_->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_kvmem_mtp::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (target_) target_->note_attention_change();
    kv_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_kvmem_mtp::seq_pos_min(llama_seq_id seq_id) const {
    return kv_->seq_pos_min(seq_id);
}

llama_pos llama_memory_kvmem_mtp::seq_pos_max(llama_seq_id seq_id) const {
    return kv_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_kvmem_mtp::memory_breakdown() const {
    return kv_->memory_breakdown();
}

void llama_memory_kvmem_mtp::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_->state_write(io, seq_id, flags);
}

void llama_memory_kvmem_mtp::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (target_) target_->note_attention_change();
    kv_->state_read(io, seq_id, flags);
}

void llama_memory_kvmem_mtp::register_capture(struct ggml_tensor * t, int il, char which) {
    GGML_UNUSED(t);
    GGML_UNUSED(il);
    GGML_UNUSED(which);
}

void llama_memory_kvmem_mtp::capture_on_new_graph() {
    pending_capture_.clear();
}

void llama_memory_kvmem_mtp::harvest_flush() {
    if (raw_) {
        raw_->wait_writes();
    }
}

void llama_memory_kvmem_mtp::harvest_pending(struct ggml_backend_sched * sched) {
    // Packed GPU K/V are copied at stage-out. Prefill does not store
    // unrotated MTP K (orig-pos restore is memcpy).
    GGML_UNUSED(sched);
    pending_capture_.clear();
    pos_queue_.clear();
}

void llama_memory_kvmem_mtp::occupy_block(uint32_t block_id) {
    if (target_ && kv_) target_->occupy_in(kv_.get(), block_id);
}

void llama_memory_kvmem_mtp::harvest_k(uint32_t block_id) {
    if (!target_ || !kv_ || !raw_) {
        return;
    }
    const auto & st = target_->store();
    if (block_id >= st.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = st.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx0 = (uint32_t) blk.gpu_slot * block_tokens_;
    if (idx0 >= cells.size() || cells.is_empty(idx0)
            || cells.ext_get(idx0).logical_pos != (llama_pos) blk.orig_pos_start) {
        return;
    }
    if (raw_->has_k_gpu(block_id, 0, blk.n_tokens)) {
        return;
    }
    ggml_tensor * kt = kv_->get_k_storage((int32_t) il_graph_);
    if (!kt) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    const uint32_t cell0 = (uint32_t) blk.gpu_slot * block_tokens_;
    const size_t row = ggml_row_size(type_k_, n_embd_k_);
    std::vector<uint8_t> packed((size_t) nt * row);
    kvmem_tensor_get(kt, packed.data(), (size_t) cell0 * row, (size_t) nt * row);
    raw_->write_layer_k_gpu(blk.orig_pos_start, nt, 0, packed.data());
}

void llama_memory_kvmem_mtp::harvest_v(uint32_t block_id) {
    if (!target_ || !kv_ || !raw_ || v_trans_) {
        return;
    }
    const auto & st = target_->store();
    if (block_id >= st.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = st.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx0 = (uint32_t) blk.gpu_slot * block_tokens_;
    if (idx0 >= cells.size() || cells.is_empty(idx0)
            || cells.ext_get(idx0).logical_pos != (llama_pos) blk.orig_pos_start) {
        return;
    }
    if (raw_->has_v_gpu(block_id, 0, blk.n_tokens)) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    const uint32_t cell0 = (uint32_t) blk.gpu_slot * block_tokens_;
    ggml_tensor * vt = kv_->get_v_storage((int32_t) il_graph_);
    if (!vt) {
        return;
    }
    const size_t row = ggml_row_size(type_v_, n_embd_v_);
    std::vector<uint8_t> packed((size_t) nt * row);
    kvmem_tensor_get(vt, packed.data(), (size_t) cell0 * row, (size_t) nt * row);
    raw_->write_layer_v_gpu(blk.orig_pos_start, nt, 0, packed.data());
}

bool llama_memory_kvmem_mtp::slot_holds(int32_t slot, uint32_t orig_pos) const {
    if (!kv_ || slot < 0) {
        return false;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx = (uint32_t) slot * block_tokens_;
    if (idx >= cells.size() || cells.is_empty(idx)) {
        return false;
    }
    return cells.ext_get(idx).logical_pos == (llama_pos) orig_pos;
}

bool llama_memory_kvmem_mtp::layout_d2d(const LayoutMove * moves, size_t n_moves) {
    if (!kv_ || !moves || n_moves == 0) {
        return true;
    }
    ggml_tensor * kt = kv_->get_k_storage((int32_t) il_graph_);
    ggml_tensor * vt = kv_->get_v_storage((int32_t) il_graph_);
    uint8_t * kbase = kvmem_mtp_cuda_ptr(kt);
    uint8_t * vbase = (vt && !v_trans_) ? kvmem_mtp_cuda_ptr(vt) : nullptr;
    if (kt && !kbase) {
        return false;
    }
    if (vt && !v_trans_ && !vbase) {
        return false;
    }
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint64_t kspan = (uint64_t) block_tokens_ * krow;
    const uint64_t vspan = (uint64_t) block_tokens_ * vrow;
    const uint64_t stride = kspan + vspan;
    uint8_t * scratch = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&scratch),
                   n_moves * (size_t) stride) != cudaSuccess) {
        return false;
    }
    kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                            std::max(krow, vrow) * (size_t) block_tokens_);
    std::vector<const void *> gsrc;
    std::vector<void *> gdst;
    std::vector<size_t> gbytes;
    std::vector<const void *> ssrc;
    std::vector<void *> sdst;
    std::vector<size_t> sbytes;
    gsrc.reserve(n_moves * 2);
    gdst.reserve(n_moves * 2);
    gbytes.reserve(n_moves * 2);
    ssrc.reserve(n_moves * 2);
    sdst.reserve(n_moves * 2);
    sbytes.reserve(n_moves * 2);
    for (size_t i = 0; i < n_moves; ++i) {
        const uint32_t nt = moves[i].n_tokens;
        if (nt == 0 || moves[i].src_slot < 0 || moves[i].dst_slot < 0) {
            continue;
        }
        const uint32_t src0 = (uint32_t) moves[i].src_slot * block_tokens_;
        const uint32_t dst0 = (uint32_t) moves[i].dst_slot * block_tokens_;
        uint8_t * slot_sc = scratch + i * (size_t) stride;
        if (kbase && krow) {
            const size_t nb = (size_t) nt * krow;
            gsrc.push_back(kbase + (size_t) src0 * krow);
            gdst.push_back(slot_sc);
            gbytes.push_back(nb);
            ssrc.push_back(slot_sc);
            sdst.push_back(kbase + (size_t) dst0 * krow);
            sbytes.push_back(nb);
        }
        if (vbase && vrow) {
            const size_t nb = (size_t) nt * vrow;
            gsrc.push_back(vbase + (size_t) src0 * vrow);
            gdst.push_back(slot_sc + (size_t) kspan);
            gbytes.push_back(nb);
            ssrc.push_back(slot_sc + (size_t) kspan);
            sdst.push_back(vbase + (size_t) dst0 * vrow);
            sbytes.push_back(nb);
        }
    }
    bool ok = true;
    const int ng = (int) gsrc.size();
    if (ng > 0 && !kvmem_d2d_batched(gsrc.data(), gdst.data(), gbytes.data(), ng)) {
        for (int j = 0; j < ng && ok; ++j) {
            ok = kvmem_copy_async(gdst[j], gsrc[j], gbytes[j],
                                 cudaMemcpyDeviceToDevice) == cudaSuccess;
        }
    }
    if (ok && cudaDeviceSynchronize() != cudaSuccess) {
        ok = false;
    }
    const int ns = (int) ssrc.size();
    if (ok && ns > 0 && !kvmem_d2d_batched(ssrc.data(), sdst.data(), sbytes.data(), ns)) {
        for (int j = 0; j < ns && ok; ++j) {
            ok = kvmem_copy_async(sdst[j], ssrc[j], sbytes[j],
                                 cudaMemcpyDeviceToDevice) == cudaSuccess;
        }
    }
    if (ok && cudaDeviceSynchronize() != cudaSuccess) {
        ok = false;
    }
    cudaFree(scratch);
    return ok;
}

void llama_memory_kvmem_mtp::write_block_to_gpu(uint32_t block_id) {
    if (!target_ || !kv_ || !raw_) {
        return;
    }
    const auto & st = target_->store();
    if (block_id >= st.block_count() || !raw_->has_block(block_id)) {
        return;
    }
    const kvmem::KvMemBlock & blk = st.blocks()[block_id];
    if (blk.gpu_slot < 0) {
        return;
    }
    target_->note_attention_change();
    const uint32_t nt = blk.n_tokens;
    occupy_block(block_id);
    ggml_tensor * kt = kv_->get_k_storage((int32_t) il_graph_);
    ggml_tensor * vt = kv_->get_v_storage((int32_t) il_graph_);
    if (!kt) {
        return;
    }
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = (uint32_t) blk.gpu_slot * block_tokens_;

    // Cold blocks: copy the GPU-format K saved at stage-out (already
    // RoPE+Hadamard+quant). Do not rebuild from unrotated raw-K — that path
    // does not match the live MTP graph and zeroes accept rate.
    std::vector<uint8_t> kpack((size_t) nt * krow);
    if (raw_->copy_k_gpu(block_id, 0, kpack.data(), nt)) {
        kvmem_tensor_set(kt, kpack.data(), cell0 * krow, nt * krow);
    }

    std::vector<uint8_t> vpack((size_t) nt * vrow);
    if (vt && !v_trans_ && raw_->copy_v_gpu(block_id, 0, vpack.data(), nt)) {
        kvmem_tensor_set(vt, vpack.data(), cell0 * vrow, nt * vrow);
    }
}

void llama_memory_kvmem_mtp::on_stage_out(uint32_t block_id) {
    if (target_) target_->note_attention_change();
    if (!target_) {
        return;
    }
    harvest_k(block_id);
    harvest_v(block_id);
    const auto & st = target_->store();
    if (block_id >= st.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & b = st.blocks()[block_id];
    if (b.n_tokens > 0) {
        kv_->seq_rm_logical(0, (llama_pos) b.orig_pos_start, (llama_pos) b.orig_pos_end());
    }
}

void llama_memory_kvmem_mtp::harvest_resident_v() {
    if (!target_ || !kv_) {
        return;
    }
    // Only copy V from cells that still hold this block. Newly admitted
    // stage_in slots are empty here; overwriting raw would wipe the V that
    // on_stage_out saved when the block left the pool during prefill.
    const llama_kv_cells & cells = kv_->get_cells(0);
    for (const auto & b : target_->store().blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        const uint32_t idx = (uint32_t) b.gpu_slot * block_tokens_;
        if (idx >= cells.size() || cells.is_empty(idx)) {
            continue;
        }
        if (cells.pos_get(idx) != (llama_pos) b.orig_pos_start) {
            continue;
        }
        harvest_k(b.block_id);
        harvest_v(b.block_id);
    }
}

void llama_memory_kvmem_mtp::follow_retrieval() {
    if (!target_ || !kv_) {
        return;
    }
    // Do not seq_rm the whole band. Resident blocks already hold the
    // graph-written GPU K (optionally D2D-packed by layout). Cold stage-in
    // blocks copy packed GPU K saved at stage-out.
    uint32_t n_gpu = 0;
    uint32_t n_keep = 0;
    uint32_t n_host = 0;
    uint32_t n_miss = 0;
    if (trace_) {
        kvmem_diag("KVMEM_TRACE mtp_selected");
    }
    for (const auto & b : target_->store().blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        n_gpu++;
        if (trace_) {
            fprintf(stderr, " %u", b.block_id);
        }
        if (slot_holds(b.gpu_slot, b.orig_pos_start)) {
            n_keep++;
            continue;
        }
        if (raw_ && raw_->has_k_gpu(b.block_id, 0)) {
            write_block_to_gpu(b.block_id);
            n_host++;
        } else {
            occupy_block(b.block_id);
            n_miss++;
        }
    }
    if (trace_) {
        fprintf(stderr, "\n");
    }
    kvmem_diag("KVMEM_TRACE mtp_follow n_gpu=%u n_keep=%u n_host=%u n_writeback=%u n_no_raw=%u seq_pos=[%d,%d]\n",
            n_gpu, n_keep, n_host, n_keep + n_host, n_miss,
            (int) kv_->seq_pos_min(0), (int) kv_->seq_pos_max(0));
}
