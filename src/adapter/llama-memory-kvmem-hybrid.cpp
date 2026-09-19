#include "llama-memory-kvmem-hybrid.h"

#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <limits>
#include <cstring>
#include <stdexcept>

static bool use_gdn_replay(const llama_model & model, const llama_cparams & cp) {
    const int mode = llama_kvmem_get_params()->mtp_state;
    if (mode == 0) return false;
    const auto & h = model.hparams;
    bool supported = model.arch == LLM_ARCH_QWEN35 && h.n_layer() == 64 && cp.n_seq_max == 1 &&
        cp.n_rs_seq > 0 && cp.n_rs_seq <= 5 && cp.n_ubatch >= cp.n_rs_seq + 1 && cp.offload_kqv && h.ssm_d_inner == 6144 &&
        h.ssm_d_state == 128 && h.ssm_n_group == 16 && h.ssm_dt_rank == 48 && h.ssm_d_conv == 4;
    ggml_backend_dev_t device = nullptr;
    for (uint32_t il = 0; supported && il < h.n_layer(); ++il) {
        if (!h.is_recr(il)) continue;
        auto * dev = model.dev_layer(il);
        // Compare against the backend's own registry name rather than the
        // literal "CUDA": the GDN record/fold kernels live in
        // ggml/src/ggml-cuda/gated_delta_net.cu, which the HIP backend compiles
        // from the very same source, and GGML_CUDA_NAME is "ROCm" there.
        // ggml-cuda.cu registers the backend with exactly this string.
        supported = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), GGML_CUDA_NAME) == 0 && (!device || device == dev);
        device = dev;
    }
    if (!supported && mode == 2) throw std::runtime_error("GDN replay requires single-sequence CUDA Qwen 27B with MTP 1-5 and all recurrent layers on one GPU");
    // Keep automatic selection on snapshots until the replay regression suite passes.
    return supported && mode == 2;
}

llama_memory_kvmem_hybrid::llama_memory_kvmem_hybrid(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) :
    llama_memory_hybrid(
            model,
            params.type_k,
            params.type_v,
            !cparams.flash_attn,
            llama_kvmem_pool_cells(model, params, cparams),
            /* n_pad */ 1,
            model.hparams.n_swa,
            model.hparams.swa_type,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            std::max((uint32_t) 1, cparams.n_seq_max),
            cparams.n_seq_max,
            cparams.n_rs_seq,
            cparams.offload_kqv,
            /* unified */ true,
            [&](int32_t il) {
                return il < (int32_t) model.hparams.n_layer() && !model.hparams.is_recr(il);
            },
            [&](int32_t il) {
                return il < (int32_t) model.hparams.n_layer() && model.hparams.is_recr(il);
            }, use_gdn_replay(model, cparams)) {
    attn_kvmem_ = std::make_unique<llama_memory_kvmem>(
            model, params, cparams, get_mem_attn());
    attn_kvmem_->set_recurrent(get_mem_recr());
    LLAMA_LOG_INFO("%s: KVMem hybrid (attn=slot-pool recr=stock) n_rs_seq=%u\n",
            __func__, cparams.n_rs_seq);
}

llama_memory_context_ptr llama_memory_kvmem_hybrid::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (embd_all) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Keep GDN rollback snapshots valid: trailing (1 + n_rs_seq)
                // tokens of each seq stay in one ubatch.
                const bool unified = (get_mem_attn()->get_n_stream() == 1);
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;
                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (!get_mem_recr()->prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        llama_kv_cache::slot_info_vec_t sinfos;
        if (!attn_kvmem_->prepare_ubatches(ubatches, balloc.get_n_tokens(), sinfos)) {
            LLAMA_LOG_ERROR("%s: failed to prepare KVMem attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

void llama_memory_kvmem_hybrid::clear(bool data) {
    llama_memory_hybrid::clear(data);
    if (attn_kvmem_) {
        attn_kvmem_->reset_policy();
    }
}

bool llama_memory_kvmem_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const llama_pos p0n = p0 < 0 ? 0 : p0;
    const bool full = seq_id <= 0 && p0n == 0 && p1 < 0;
    llama_memory_recurrent * recr = get_mem_recr();
    if (full) {
        if (!recr->seq_rm(seq_id, p0, p1)) {
            return false;
        }
        return attn_kvmem_ ? attn_kvmem_->seq_rm(seq_id, p0, p1)
                           : get_mem_attn()->seq_rm(seq_id, p0, p1);
    }
    // Query-replay holes must not touch GDN (P4-2 restores it separately).
    // MTP verify reject is a short open suffix (length 1..n_rs_seq): roll
    // GDN back on the GPU snapshot planes instead of a 150 MiB host dump.
    const uint32_t n_rs = recr->n_rs_seq;
    const llama_pos rmax = recr->seq_pos_max(seq_id);
    const llama_pos p1x = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if (n_rs > 0 && p0n > 0 && rmax >= 0 && p0n <= rmax && p1x > rmax) {
        const llama_pos rollback = rmax - (p0n - 1);
        if (rollback >= 1 && rollback <= (llama_pos) n_rs) {
            if (!recr->seq_rm(seq_id, p0, p1)) {
                return false;
            }
        }
    }
    return attn_kvmem_ ? attn_kvmem_->seq_rm(seq_id, p0, p1)
                       : get_mem_attn()->seq_rm(seq_id, p0, p1);
}
