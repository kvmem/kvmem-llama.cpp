#include "llama-memory-kvmem.h"
#include "llama-memory-kvmem-hybrid.h"
#include "llama-memory-kvmem-mtp.h"

#include "llama-kvmem-batch.h"
#include "llama-kvmem-capture.h"
#include "llama-kvmem-factory.h"
#include "llama-kvmem-hooks.h"
#include "llama-kvmem-quant.h"
#include "llama-kvmem-stagein.h"
#include "llama-kvmem-transfer.h"

#include "llama-arch.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"

#include "llama.h"

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include "llama-kvmem-diag.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

static llama_kvmem_params g_kvmem_params = {};

struct llama_memory_kvmem::GdnReplay {
    ggml_backend_buffer_ptr descriptors;
    cudaStream_t stream = nullptr;
    int layers = 0;
    int device = 0;
    uint64_t folds = 0;
    int64_t fold_us = 0;
    ~GdnReplay() {
        if (stream) {
            cudaSetDevice(device);
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
};

void llama_memory_kvmem::set_recurrent(llama_memory_recurrent * recr) {
    recr_ = recr;
    if (!recr) return;
    size_t recurrent_bytes = 0, conv_bytes = 0;
    for (size_t il = 0; il < recr->r_l.size(); ++il) {
        if (recr->r_l[il]) conv_bytes += ggml_nbytes(recr->r_l[il]);
        if (recr->s_l[il]) recurrent_bytes += ggml_nbytes(recr->s_l[il]);
    }
    const size_t planes = recr->n_rs_seq + 1;
    LLAMA_LOG_INFO("KVMEM_GDN_ALLOCATION mode=%s recurrent_bytes=%zu conv_bytes=%zu rollback_bytes=%zu\n",
            recr->replay_capacity ? "replay" : "snapshots", recurrent_bytes / planes, conv_bytes / planes,
            (recurrent_bytes + conv_bytes) / planes * (planes - 1));
    if (!recr->replay_capacity) return;
    auto replay = std::make_unique<GdnReplay>();
    std::vector<ggml_cuda_gdn_replay_layer> layers;
    size_t records = 0, states = 0;
    ggml_backend_buffer_type_t buft = nullptr;
    for (size_t il = 0; il < recr->r_l.size(); ++il) {
        if (!recr->r_l[il]) continue;
        const auto & r = recr->replay_l[il];
        layers.push_back({static_cast<float *>(recr->s_l[il]->data), static_cast<float *>(recr->r_l[il]->data),
                static_cast<float *>(r[0]->data), static_cast<float *>(r[1]->data), static_cast<float *>(r[2]->data),
                static_cast<float *>(r[3]->data), static_cast<float *>(r[4]->data)});
        for (auto * t : r) records += ggml_nbytes(t);
        states += ggml_nbytes(recr->r_l[il]) + ggml_nbytes(recr->s_l[il]);
        buft = ggml_backend_buffer_get_type(recr->s_l[il]->buffer);
    }
    if (layers.empty()) throw std::runtime_error("GDN replay has no recurrent layers");
    cudaPointerAttributes attrs{};
    if (cudaPointerGetAttributes(&attrs, layers.front().state) != cudaSuccess ||
            cudaSetDevice(attrs.device) != cudaSuccess) throw std::runtime_error("cannot select GDN replay device");
    replay->device = attrs.device;
    replay->layers = layers.size();
    replay->descriptors.reset(ggml_backend_buft_alloc_buffer(buft, layers.size() * sizeof(layers[0])));
    if (!replay->descriptors || cudaStreamCreateWithFlags(&replay->stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMemcpy(ggml_backend_buffer_get_base(replay->descriptors.get()), layers.data(), layers.size() * sizeof(layers[0]), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("cannot allocate GDN replay descriptors");
    }
    LLAMA_LOG_INFO("KVMEM_GDN_MEMORY mode=replay layers=%d capacity=%u state_bytes=%zu record_bytes=%zu descriptor_bytes=%zu\n",
            replay->layers, recr->replay_capacity, states, records, layers.size() * sizeof(layers[0]));
    gdn_replay_ = std::move(replay);
}

bool llama_memory_kvmem::gdn_replay_enabled() const {
    return gdn_replay_ != nullptr;
}

bool llama_memory_kvmem::gdn_replay_begin(llama_pos start, uint32_t width) {
    return gdn_replay_ && recr_->replay_begin(start, width);
}

bool llama_memory_kvmem::gdn_replay_commit(llama_context * ctx, uint32_t n_keep) {
    if (!gdn_replay_ || !recr_->replay_recording || n_keep > recr_->replay_width) return false;
    llama_synchronize(ctx);
    auto & replay = *gdn_replay_;
    const int64_t started = ggml_time_us();
    const auto * layers = static_cast<const ggml_cuda_gdn_replay_layer *>(ggml_backend_buffer_get_base(replay.descriptors.get()));
    if (cudaSetDevice(replay.device) != cudaSuccess ||
            !ggml_backend_cuda_gdn_fold(layers, replay.layers, n_keep, recr_->replay_capacity, replay.stream) ||
            cudaStreamSynchronize(replay.stream) != cudaSuccess) {
        recr_->replay_poisoned = true;
        recr_->replay_finish(0);
        return false;
    }
    recr_->replay_finish(n_keep);
    replay.fold_us += ggml_time_us() - started;
    replay.folds += n_keep != 0;
    return true;
}
static std::atomic<uint64_t> transfer_bytes[3]{};
static std::atomic<uint64_t> transfer_calls[3]{};

static bool transfer_stats_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("KVMEM_PERF");
        return e && e[0] && e[0] != '0';
    }();
    return enabled;
}

void kvmem_record_transfer(int kind, uint64_t bytes) {
    if (!transfer_stats_enabled() || kind < 1 || kind > 3 || !bytes) return;
    transfer_bytes[kind - 1].fetch_add(bytes, std::memory_order_relaxed);
    transfer_calls[kind - 1].fetch_add(1, std::memory_order_relaxed);
}

llama_kvmem_transfer_stats llama_kvmem_get_transfer_stats() {
    llama_kvmem_transfer_stats stats;
    stats.enabled = transfer_stats_enabled();
    for (int i = 0; i < 3; ++i) {
        stats.bytes[i] = transfer_bytes[i].load(std::memory_order_relaxed);
        stats.calls[i] = transfer_calls[i].load(std::memory_order_relaxed);
    }
    return stats;
}

struct llama_memory_kvmem::CaptureD2hPipe {
    struct Item {
        int il = 0;
        char which = 0;
        size_t offset = 0;
        size_t nbytes = 0;
        int64_t d = 0;
        int64_t h = 0;
        int64_t n = 0;
        size_t nb0 = 0;
        size_t nb1 = 0;
        size_t nb2 = 0;
        ggml_type type = GGML_TYPE_F16;
        bool from_gpu = false;
    };
    struct Slot {
        uint8_t * gpu = nullptr;
        uint8_t * pin = nullptr;
        size_t cap = 0;
        cudaEvent_t done = nullptr;
        bool inflight = false;
        std::vector<Item> items;
        std::vector<llama_pos> pos;
    };
    cudaStream_t stream = nullptr;
    cudaEvent_t snap = nullptr;
    ggml_backend_event_t compute_done = nullptr;
    ggml_backend_event_t snap_be = nullptr;
    Slot slots[2];
    int next = 0;
    int device = 0;
    bool ok = false;
};

static bool kvmem_cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "KVMEM D2H %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

static uint8_t * kvmem_cuda_tensor_ptr(ggml_tensor * t) {
    if (!t || !t->data) {
        return nullptr;
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf || ggml_backend_buffer_is_host(buf)) {
        return nullptr;
    }
    return static_cast<uint8_t *>(t->data);
}

static bool kvmem_d2d(uint8_t * dst, const uint8_t * src, size_t n, cudaStream_t st) {
    if (!dst || !src || n == 0) {
        return true;
    }
    return kvmem_cuda_ok(kvmem_copy_async(dst, src, n, cudaMemcpyDeviceToDevice, st), "layout D2D");
}

static void kvmem_stagein_flush_sync(int64_t * copy_us, int64_t * rope_us,
                                     int64_t * hadamard_us, int64_t * set_us) {
    kvmem_stagein_flush(copy_us, rope_us, hadamard_us, set_us);
    const int64_t t0 = ggml_time_us();
    kvmem_stagein_sync();
    if (set_us) {
        *set_us += ggml_time_us() - t0;
    }
}

static bool kvmem_harvest_sync_old() {
    const char * e = getenv("KVMEM_HARVEST_SYNC");
    return e && e[0] != '\0' && e[0] != '0';
}

static cudaEvent_t kvmem_ggml_cuda_event(ggml_backend_event_t ev) {
    return ev ? static_cast<cudaEvent_t>(ev->context) : nullptr;
}

static bool kvmem_env_perf() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("KVMEM_PERF");
        v = (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    return v == 1;
}

void llama_kvmem_set_params(const struct llama_kvmem_params * params) {
    if (params) {
        g_kvmem_params = *params;
    } else {
        g_kvmem_params = {};
    }
}

const struct llama_kvmem_params * llama_kvmem_get_params(void) {
    return &g_kvmem_params;
}

static uint32_t kvmem_align_tokens(uint32_t tokens, uint32_t block_tokens) {
    if (block_tokens == 0) {
        return 0;
    }
    if (tokens < block_tokens) {
        return block_tokens;
    }
    return (tokens / block_tokens) * block_tokens;
}

static double kvmem_default_ratio(float v, double fallback) {
    return v > 0.0f ? static_cast<double>(v) : fallback;
}

static uint64_t kvmem_first_gpu_total_bytes() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free_m = 0;
            size_t total_m = 0;
            ggml_backend_dev_memory(dev, &free_m, &total_m);
            return static_cast<uint64_t>(total_m);
        }
    }
    return 0;
}

static kvmem::KvMemRuntimeConfig make_runtime_cfg(
        uint32_t block_tokens,
        uint32_t budget,
        uint32_t sink_tokens,
        uint32_t recent_tokens,
        uint64_t block_bytes) {
    kvmem::KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = block_tokens;
    cfg.store.select_budget = budget;
    cfg.store.prefill_budget = budget;
    cfg.store.select_method = g_kvmem_params.method == 1
        ? kvmem::KvMemMethod::Retrieval
        : kvmem::KvMemMethod::Recency;
    cfg.store.sink_blocks = sink_tokens == 0
        ? 1u
        : std::max(1u, sink_tokens / block_tokens);
    cfg.store.recent_blocks = recent_tokens / block_tokens;
    cfg.store.gen_budget = 0;
    cfg.store.gpu_memory_ratio = kvmem_default_ratio(g_kvmem_params.gpu_memory_ratio, 0.50);
    cfg.store.gpu_high_watermark = kvmem_default_ratio(g_kvmem_params.gpu_high_watermark, 0.95);
    cfg.store.gpu_low_watermark = kvmem_default_ratio(g_kvmem_params.gpu_low_watermark, 0.85);
    cfg.store.estimated_block_bytes = block_bytes;
    cfg.store.gpu_resident_block_bytes = block_bytes;
    cfg.cpu_bytes = g_kvmem_params.cpu_bytes;
    // GPU-block NVMe spill is separate from raw-K NVMe. When raw-K owns the
    // SSD budget, skip the 2 MiB GPU-format copies (V already lives in raw).
    if (!g_kvmem_params.raw_k_nvme) {
        cfg.nvme_bytes = g_kvmem_params.nvme_bytes;
        if (g_kvmem_params.nvme_dir && g_kvmem_params.nvme_dir[0]) {
            cfg.nvme_dir = g_kvmem_params.nvme_dir;
        } else if (cfg.nvme_bytes > 0) {
            cfg.nvme_dir = "/tmp/kvmem_nvme";
        }
    }
    return cfg;
}

static bool kvmem_cache_has_layer(const llama_kv_cache * kv, int32_t il) {
    if (!kv) {
        return false;
    }
    for (uint32_t id : kv->get_layer_ids()) {
        if (static_cast<int32_t>(id) == il) {
            return true;
        }
    }
    return false;
}

static uint32_t kvmem_first_attn_layer(const llama_model & model) {
    const uint32_t n = model.hparams.n_layer();
    for (uint32_t il = 0; il < n; ++il) {
        if (!model.hparams.is_recr(il)) {
            return il;
        }
    }
    return 0;
}

static uint32_t kvmem_n_attn_layers(const llama_model & model) {
    const uint32_t n = model.hparams.n_layer();
    uint32_t c = 0;
    for (uint32_t il = 0; il < n; ++il) {
        if (!model.hparams.is_recr(il)) {
            ++c;
        }
    }
    return c == 0 ? n : c;
}

struct kvmem_pool_plan {
    uint32_t block_tokens = 32;
    uint32_t budget = 0;
    uint32_t gen_reserve = 0;
    uint32_t kv_size = 0;
    uint32_t n_slots = 0;
    uint64_t block_bytes = 0;
    uint32_t cap_blocks = 0;
    uint64_t gpu_total = 0;
};

static kvmem_pool_plan kvmem_compute_pool(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) {
    kvmem_pool_plan p;
    p.block_tokens = g_kvmem_params.block_tokens ? g_kvmem_params.block_tokens : 32u;
    uint32_t budget = g_kvmem_params.budget;
    if (budget == 0) {
        budget = kvmem_align_tokens(cparams.n_ctx_seq, p.block_tokens);
        if (budget == 0) {
            budget = p.block_tokens;
        }
    } else {
        budget = kvmem_align_tokens(budget, p.block_tokens);
    }
    uint32_t gen_reserve = g_kvmem_params.gen_reserve ? g_kvmem_params.gen_reserve : 256u;
    gen_reserve = kvmem_align_tokens(gen_reserve, p.block_tokens);

    const uint32_t il0 = kvmem_first_attn_layer(model);
    const uint32_t n_attn = kvmem_n_attn_layers(model);
    const uint32_t n_embd_k = model.hparams.n_embd_k_gqa(il0);
    const uint32_t n_embd_v = model.hparams.n_embd_v_gqa(il0);
    const uint64_t k_row = ggml_row_size(params.type_k, n_embd_k);
    const uint64_t v_row = ggml_row_size(params.type_v, n_embd_v);
    p.block_bytes = static_cast<uint64_t>(n_attn) * (k_row + v_row) * p.block_tokens;

    const double ratio = kvmem_default_ratio(g_kvmem_params.gpu_memory_ratio, 0.50);
    p.gpu_total = kvmem_first_gpu_total_bytes();
    if (p.gpu_total > 0 && p.block_bytes > 0 && ratio > 0.0) {
        p.cap_blocks = static_cast<uint32_t>(
                (p.gpu_total * ratio) / std::max(p.block_bytes, uint64_t{1}));
    }

    uint32_t pool = budget + gen_reserve;
    if (pool > cparams.n_ctx_seq && g_kvmem_params.budget == 0) {
        pool = cparams.n_ctx_seq;
    }
    if (p.cap_blocks > 0) {
        const uint32_t cap_tokens = p.cap_blocks * p.block_tokens;
        if (pool > cap_tokens) {
            pool = cap_tokens;
            if (budget >= pool) {
                gen_reserve = std::min(gen_reserve, p.block_tokens);
                budget = pool > gen_reserve ? pool - gen_reserve : pool;
            } else if (budget + gen_reserve > pool) {
                gen_reserve = pool - budget;
            }
            budget = kvmem_align_tokens(std::max(budget, p.block_tokens), p.block_tokens);
            gen_reserve = kvmem_align_tokens(std::max(gen_reserve, p.block_tokens), p.block_tokens);
            pool = budget + gen_reserve;
            if (pool > cap_tokens) {
                pool = cap_tokens;
            }
        }
    }
    p.budget = budget;
    p.gen_reserve = gen_reserve;
    p.kv_size = std::max(pool, 1u);
    p.n_slots = (p.kv_size + p.block_tokens - 1) / p.block_tokens;
    if (p.n_slots * p.block_tokens < p.kv_size) {
        p.n_slots += 1;
    }
    return p;
}

uint32_t llama_kvmem_pool_cells(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) {
    return kvmem_compute_pool(model, params, cparams).kv_size;
}

llama_memory_i * llama_memory_kvmem_maybe_create(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) {
    if (!g_kvmem_params.enabled) {
        return nullptr;
    }
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        llama_memory_kvmem * tgt = nullptr;
        if (cparams.ctx_other) {
            llama_memory_t mem = llama_get_memory(cparams.ctx_other);
            if (auto * hyb = dynamic_cast<llama_memory_kvmem_hybrid *>(mem)) {
                tgt = hyb->attn_kvmem();
            } else {
                tgt = dynamic_cast<llama_memory_kvmem *>(mem);
            }
        }
        if (!tgt) {
            return nullptr;
        }
        return new llama_memory_kvmem_mtp(model, params, cparams, tgt);
    }
    if (llm_arch_is_recurrent(model.arch)) {
        LLAMA_LOG_WARN("%s: KVMem skips purely recurrent arch %s\n",
                __func__, llm_arch_name(model.arch));
        return nullptr;
    }
    if (cparams.n_seq_max > 1) {
        LLAMA_LOG_WARN("%s: KVMem requires n_seq_max=1 (got %u)\n",
                __func__, cparams.n_seq_max);
        return nullptr;
    }
    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        LLAMA_LOG_WARN("%s: KVMem skips SWA models\n", __func__);
        return nullptr;
    }
    if (llm_arch_is_hybrid(model.arch)) {
        return new llama_memory_kvmem_hybrid(model, params, cparams);
    }
    return new llama_memory_kvmem(model, params, cparams);
}

llama_memory_kvmem::llama_memory_kvmem(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams,
        llama_kv_cache * ext_kv) :
    model_(model) {
    backend_.owner = this;
    trace_ = getenv("KVMEM_TRACE") != nullptr;
    perf_.enabled = kvmem_env_perf();
    retr_.enabled = perf_.enabled;
    harvest_perf_emit_graph_line();

    const kvmem_pool_plan pool = kvmem_compute_pool(model, params, cparams);
    block_tokens_ = pool.block_tokens;
    kv_size_ = pool.kv_size;
    n_slots_ = pool.n_slots;

    auto rt_cfg = make_runtime_cfg(
            block_tokens_, pool.budget, g_kvmem_params.sink_tokens, g_kvmem_params.recent_tokens,
            pool.block_bytes);
    rt_cfg.store.estimated_gpu_block_capacity = pool.cap_blocks;
    runtime_ = std::make_unique<kvmem::KvMemRuntime>(rt_cfg, &backend_);

    if (ext_kv) {
        kv_ = ext_kv;
        if (kv_->get_size() != kv_size_) {
            LLAMA_LOG_WARN("%s: borrowed attn cache size %u != planned pool %u\n",
                    __func__, kv_->get_size(), kv_size_);
            kv_size_ = kv_->get_size();
            n_slots_ = (kv_size_ + block_tokens_ - 1) / block_tokens_;
            if (n_slots_ * block_tokens_ < kv_size_) {
                n_slots_ += 1;
            }
        }
    } else {
        kv_owned_ = std::make_unique<llama_kv_cache>(
                model,
                model.hparams,
                params.type_k,
                params.type_v,
                !cparams.flash_attn,
                cparams.offload_kqv,
                /* unified */ true,
                kv_size_,
                /* n_seq_max */ 1,
                /* n_pad */ 1,
                model.hparams.n_swa,
                model.hparams.swa_type,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                "kvmem");
        kv_ = kv_owned_.get();
    }

    reset_slots();

    const uint32_t il0 = kvmem_first_attn_layer(model);
    n_layer_ = model.hparams.n_layer();
    n_embd_head_ = model.hparams.n_embd_head_k(il0);
    n_head_kv_ = model.hparams.n_head_kv(il0);
    n_head_ = model.hparams.n_head(il0);
    n_embd_k_ = model.hparams.n_embd_k_gqa(il0);
    n_embd_v_ = model.hparams.n_embd_v_gqa(il0);
    rope_.n_rot = model.hparams.n_rot(il0);
    rope_.n_embd_head = n_embd_head_;
    rope_.n_head_kv = n_head_kv_;
    rope_.freq_base = cparams.rope_freq_base;
    rope_.freq_scale = cparams.rope_freq_scale;
    type_k_ = params.type_k;
    type_v_ = params.type_v;
    v_trans_ = !cparams.flash_attn;
    method_ = g_kvmem_params.method;
    query_begin_ = g_kvmem_params.query_begin;
    query_end_ = g_kvmem_params.query_end;
    force_pos_ = g_kvmem_params.force_pos;
    kvmem::RawKvStoreConfig rcfg;
    rcfg.n_layer = n_layer_;
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
    if (g_kvmem_params.raw_k_nvme) {
        rcfg.nvme_bytes = g_kvmem_params.nvme_bytes
                ? g_kvmem_params.nvme_bytes
                : (32ull * 1024ull * 1024ull * 1024ull);
        rcfg.nvme_dir = (g_kvmem_params.nvme_dir && g_kvmem_params.nvme_dir[0])
                ? g_kvmem_params.nvme_dir
                : "/tmp/kvmem_nvme";
        rcfg.nvme_file = "kvmem_raw_k.bin";
    }
    raw_ = std::make_unique<kvmem::RawKvStore>(rcfg);
    q_sum_.assign(n_layer_, std::vector<float>(n_head_ * n_embd_head_, 0.0f));
    q_count_.assign(n_layer_, 0);
    kvmem_capture_bind(this);

    size_t kv_bytes = 0;
    if (kv_) {
        for (const auto & kv : kv_->memory_breakdown()) {
            kv_bytes += kv.second;
        }
    }
    LLAMA_LOG_INFO(
            "%s: KVMem slot-pool cells=%u slots=%u block_tokens=%u budget=%u gen_reserve=%u sink_blocks=%u method=%s harvest_v=%d type_k=%s type_v=%s n_embd_k=%u attn_layers=%u%s\n",
            __func__, kv_size_, n_slots_, block_tokens_, pool.budget, pool.gen_reserve,
            rt_cfg.store.sink_blocks,
            method_ == 1 ? "retrieval" : "recency",
            (int) g_kvmem_params.harvest_v,
            ggml_type_name(type_k_), ggml_type_name(type_v_),
            n_embd_k_, kvmem_n_attn_layers(model),
            ext_kv ? " hybrid_attn" : "");
    kvmem_diag("KVMEM_KV_BYTES bytes=%zu cells=%u slots=%u budget=%u pool=%u "
            "ratio=%.2f high=%.2f low=%.2f cap_blocks=%u gpu_total=%llu block_bytes=%llu\n",
            kv_bytes, kv_size_, n_slots_, pool.budget, kv_size_,
            rt_cfg.store.gpu_memory_ratio,
            rt_cfg.store.gpu_high_watermark,
            rt_cfg.store.gpu_low_watermark,
            pool.cap_blocks,
            (unsigned long long) pool.gpu_total,
            (unsigned long long) pool.block_bytes);
}

llama_memory_kvmem::~llama_memory_kvmem() {
    decode_mean_flush();
    decode_mean_discard();
    decode_mean_print_sum();
    harvest_flush();
    harvest_worker_stop();
    harvest_perf_print_sum();
    d2h_free();
    kvmem_stagein_gpu_free();
    if (mtp_) {
        mtp_->detach_target();
        mtp_ = nullptr;
    }
    kvmem_capture_unbind(this);
}

void llama_memory_kvmem::reset_slots() {
    free_slots_.clear();
    free_slots_.reserve(n_slots_);
    for (int32_t i = static_cast<int32_t>(n_slots_) - 1; i >= 0; --i) {
        free_slots_.push_back(i);
    }
}

void llama_memory_kvmem::reset_policy() {
    ++attention_epoch_;
    explicit_spans_ = false;
    query_frozen_ = false;
    turn_spans_ = {};
    row_positions_.clear();
    decode_mean_reset();
    reset_query_acc();
    if (raw_) {
        raw_->clear();
    }
    if (runtime_) {
        runtime_->truncate_to(0);
    }
    reset_slots();
    retrieval_pinned_ = false;
    keep_selected_ = false;
    prefill_capture_ = true;
}

void llama_memory_kvmem::begin_cached_turn(bool reset_query) {
    harvest_flush();
    query_frozen_ = false;
    decode_mean_flush();
    decode_mean_discard();
    if (reset_query) {
        reset_query_acc();
    }
    retrieval_pinned_ = false;
    keep_selected_ = false;
    prefill_capture_ = true;
}

void llama_memory_kvmem::truncate_cached(uint32_t n_past) {
    if (runtime_ && n_past >= runtime_->store().total_tokens()) return;
    ++attention_epoch_;
    harvest_flush();
    harvest_gpu_v_commit();
    if (row_positions_.size() > n_past) row_positions_.resize(n_past);
    if (mtp_) mtp_->truncate_cached(n_past);
    if (raw_) {
        raw_->truncate_to(n_past);
    }
    if (runtime_ && n_past < runtime_->store().total_tokens()) {
        runtime_->truncate_to(n_past);
    }
}

void llama_memory_kvmem::set_replay(bool replay) {
    if (replay && !replay_) {
        harvest_flush();
        harvest_gpu_v_commit();
        ++attention_epoch_;
        const uint32_t begin = std::max(0, explicit_spans_ ? turn_spans_.replay_begin : query_begin_);
        if (raw_) raw_->invalidate_packed_from(begin);
        if (mtp_) mtp_->invalidate_packed_from(begin);
    }
    replay_ = replay;
}

llama_pos llama_memory_kvmem::recr_pos_max() const {
    return recr_ ? recr_->seq_pos_max(0) : (llama_pos) -1;
}

int32_t llama_memory_kvmem::alloc_slot() {
    if (free_slots_.empty()) {
        return -1;
    }
    const int32_t slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
}

void llama_memory_kvmem::free_slot(int32_t slot) {
    if (slot < 0) {
        return;
    }
    free_slots_.push_back(slot);
}

int32_t llama_memory_kvmem::peek_free_slot() const {
    if (free_slots_.empty()) {
        return -1;
    }
    return free_slots_.back();
}

bool llama_memory_kvmem::slot_for_orig_pos(llama_pos pos, int32_t * slot, uint32_t * off) const {
    if (!slot || !off || pos < 0 || !runtime_) {
        return false;
    }
    const auto & st = runtime_->store();
    const uint32_t bt = block_tokens_;
    if (bt == 0) {
        return false;
    }
    const int32_t bid = st.block_id_containing(static_cast<uint32_t>(pos));
    if (bid >= 0) {
        const kvmem::KvMemBlock & b = st.blocks()[static_cast<uint32_t>(bid)];
        if (b.gpu_slot < 0) {
            return false;
        }
        *slot = b.gpu_slot;
        *off = static_cast<uint32_t>(pos) - b.orig_pos_start;
        return *off < bt;
    }
    // Token not yet on the target store (MTP draft before verify). Stay in
    // the last GPU block if it still has room; otherwise peek the next free
    // slot without popping so target's next alloc_slot() takes the same one.
    if (!st.blocks().empty()) {
        const kvmem::KvMemBlock & last = st.blocks().back();
        if (last.gpu_slot >= 0 &&
            static_cast<uint32_t>(pos) >= last.orig_pos_start &&
            static_cast<uint32_t>(pos) < last.orig_pos_start + bt) {
            *slot = last.gpu_slot;
            *off = static_cast<uint32_t>(pos) - last.orig_pos_start;
            return true;
        }
    }
    const uint32_t orig = (static_cast<uint32_t>(pos) / bt) * bt;
    *off = static_cast<uint32_t>(pos) - orig;
    *slot = peek_free_slot();
    return *slot >= 0 && *off < bt;
}

uint32_t llama_memory_kvmem::resident_tokens() const {
    uint32_t n = 0;
    for (const auto & b : runtime_->store().blocks()) {
        if (b.gpu_slot >= 0) {
            n += b.n_tokens;
        }
    }
    return n;
}

void llama_memory_kvmem::trace_plan(const char * tag, const kvmem::KvMemPlan & plan) const {
    if (!trace_) {
        return;
    }
    uint32_t skip = 0;
    for (const auto & r : plan.remaps) {
        if (r.skip) {
            skip++;
        }
    }
    kvmem_diag("KVMEM_TRACE %s stage_in=%zu stage_out=%zu skip=%u gpu_reused=%u window=%u free_slots=%zu\n",
            tag,
            plan.stage_in.size(),
            plan.stage_out.size(),
            skip,
            plan.gpu_reused_blocks,
            plan.total_window_tokens,
            free_slots_.size());
}

void llama_memory_kvmem::apply_plan_to_kv(const kvmem::KvMemPlan & plan) {
    if (!plan.stage_in.empty() || !plan.stage_out.empty()) ++attention_epoch_;
    auto & store = runtime_->store();
    for (uint32_t id : plan.stage_out) {
        if (id == decode_mean_block_) {
            decode_mean_flush();
            break;
        }
    }
    {
        const int64_t t0 = ggml_time_us();
        for (uint32_t id : plan.stage_out) {
            harvest_gpu_v(id);
            if (mtp_) {
                mtp_->on_stage_out(id);
            }
        }
        harvest_gpu_v_commit();
        runtime_->spill_outgoing();
        if (retr_.enabled) {
            retr_.stage_out_us += ggml_time_us() - t0;
        }
    }
    {
        const int64_t t0 = ggml_time_us();
        for (uint32_t id : plan.stage_out) {
            if (id >= store.block_count()) {
                continue;
            }
            const kvmem::KvMemBlock & b = store.blocks()[id];
            if (b.n_tokens > 0) {
                kv_->seq_rm_logical(0, static_cast<llama_pos>(b.orig_pos_start),
                            static_cast<llama_pos>(b.orig_pos_end()));
            }
        }
        if (retr_.enabled) {
            retr_.seq_rm_us += ggml_time_us() - t0;
        }
    }
    {
        const int64_t t0 = ggml_time_us();
        runtime_->admit_incoming();
        if (retr_.enabled) {
            retr_.admit_us += ggml_time_us() - t0;
        }
    }
}

bool llama_memory_kvmem::gpu_kv_already_resident(uint32_t block_id) const {
    if (!kv_ || block_id >= runtime_->store().block_count()) {
        return false;
    }
    const kvmem::KvMemBlock & b = runtime_->store().blocks()[block_id];
    if (b.gpu_slot < 0 || b.n_tokens == 0) {
        return false;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx = static_cast<uint32_t>(b.gpu_slot) * block_tokens_;
    if (idx >= cells.size() || cells.is_empty(idx)) {
        return false;
    }
    return cells.ext_get(idx).logical_pos == static_cast<llama_pos>(b.orig_pos_start);
}

void llama_memory_kvmem::occupy_in(llama_kv_cache * cache, uint32_t block_id) {
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.resize(1);
    sinfo.strm[0] = 0;
    sinfo.idxs[0].resize(nt);
    std::vector<llama_pos> pos(4*nt), logical(nt);
    std::vector<int32_t> n_seq_id(nt, 1);
    std::vector<llama_seq_id> seq_data(nt, 0);
    std::vector<llama_seq_id *> seq_id(nt);
    std::vector<int8_t> output(nt, 0);
    std::vector<llama_token> tok(nt, 0);
    llama_seq_id seq0 = 0;
    for (uint32_t t = 0; t < nt; ++t) {
        sinfo.idxs[0][t] = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_ + t;
        logical[t] = blk.orig_pos_start + t;
        if ((size_t) logical[t] >= row_positions_.size()) {
            throw std::runtime_error("missing cache row position metadata");
        }
        const auto & row = row_positions_[logical[t]];
        for (uint32_t j = 0; j < 4; ++j) pos[j*nt+t] = row.pos[j];
        tok[t] = row.token;
        seq_id[t] = &seq_data[t];
    }
    llama_ubatch ub{};
    ub.n_tokens = nt;
    ub.n_seq_tokens = nt;
    ub.n_seqs = 1;
    ub.n_seqs_unq = 1;
    ub.n_pos = 4;
    ub.logical_pos = logical.data();
    ub.pos = pos.data();
    ub.n_seq_id = n_seq_id.data();
    ub.seq_id = seq_id.data();
    ub.seq_id_unq = &seq0;
    ub.output = output.data();
    ub.token = tok.data();
    {
        const int64_t t0 = ggml_time_us();
        cache->apply_ubatch(sinfo, ub);
        if (retr_.enabled) {
            retr_.occupy_us += ggml_time_us() - t0;
        }
    }
}

void llama_memory_kvmem::occupy_block_cells(uint32_t block_id) {
    occupy_in(kv_, block_id);
}

llama_pos llama_memory_kvmem::model_pos(uint32_t logical) const {
    if (logical < row_positions_.size()) return row_positions_[logical].pos[0];
    if (row_positions_.empty()) return (llama_pos) logical;
    const auto & last = row_positions_.back();
    // Appending text after a media chunk uses the helper's explicit cursor.
    return last.pos[0] + 1 + (llama_pos) (logical - row_positions_.size());
}

bool llama_memory_kvmem::remove_logical(llama_context * ctx, llama_pos begin, llama_pos end) {
    ++attention_epoch_;
    if (mtp_ && llama_get_memory(ctx) == mtp_) return mtp_->remove_logical(begin, end);
    // Recurrent rollback is only valid across consecutive text positions.
    if (recr_ && end < 0 && begin > 0 && (size_t) begin < row_positions_.size()) {
        const auto p = model_pos(begin);
        const auto prev = model_pos(begin - 1);
        const auto rmax = recr_->seq_pos_max(0);
        if (p == prev + 1 && rmax >= p && rmax - prev <= (llama_pos) recr_->n_rs_seq) {
            if (!recr_->seq_rm(0, p, -1)) return false;
        }
    }
    return kv_->seq_rm_logical(0, begin, end);
}

bool llama_memory_kvmem::layout_gpu_slots_by_orig_pos() {
    auto & store = runtime_->store();
    struct Item {
        uint32_t id;
        uint32_t orig;
        int32_t slot;
        uint32_t n;
        bool resident;
    };
    std::vector<Item> items;
    for (const auto & b : store.blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        items.push_back({b.block_id, b.orig_pos_start, b.gpu_slot, b.n_tokens,
                         gpu_kv_already_resident(b.block_id)});
    }
    if (items.empty()) {
        return false;
    }
    std::sort(items.begin(), items.end(),
              [](const Item & a, const Item & b) { return a.orig < b.orig; });
    bool sorted = true;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].slot != static_cast<int32_t>(i)) {
            sorted = false;
            break;
        }
    }
    if (sorted) {
        return false;
    }

    ++attention_epoch_;
    std::vector<uint8_t> mtp_res(items.size(), 0);
    bool mtp_d2d_ok = false;
    if (mtp_) {
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].resident &&
                mtp_->slot_holds(items[i].slot, items[i].orig)) {
                mtp_res[i] = 1;
            }
        }
    }

    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint64_t kspan = static_cast<uint64_t>(block_tokens_) * krow;
    const uint64_t vspan = static_cast<uint64_t>(block_tokens_) * vrow;
    const uint64_t scratch_stride = kspan + vspan;

    std::vector<size_t> res_ix(items.size(), static_cast<size_t>(-1));
    size_t n_res = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].resident) {
            res_ix[i] = n_res++;
        }
    }

    bool d2d_ok = n_res > 0;
    uint8_t * scratch = nullptr;
    if (d2d_ok) {
        if (cudaMalloc(reinterpret_cast<void **>(&scratch),
                       static_cast<size_t>(n_res) * scratch_stride) != cudaSuccess) {
            scratch = nullptr;
            d2d_ok = false;
            LLAMA_LOG_WARN("%s: layout scratch cudaMalloc failed, host fallback\n", __func__);
        }
        kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                                std::max(krow, vrow) * (size_t) block_tokens_);
    }

    if (d2d_ok) {
        cudaStream_t st = cudaStreamPerThread;
        bool copy_ok = true;
        for (uint32_t il = 0; il < n_layer_ && copy_ok; ++il) {
            if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
                continue;
            }
            ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
            ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
            uint8_t * kbase = kt ? kvmem_cuda_tensor_ptr(kt) : nullptr;
            uint8_t * vbase = (vt && !v_trans_) ? kvmem_cuda_tensor_ptr(vt) : nullptr;
            if (kt && !kbase) {
                copy_ok = false;
                break;
            }
            if (vt && !v_trans_ && !vbase) {
                copy_ok = false;
                break;
            }
            std::vector<const void *> gsrc;
            std::vector<void *> gdst;
            std::vector<size_t> gbytes;
            std::vector<const void *> ssrc;
            std::vector<void *> sdst;
            std::vector<size_t> sbytes;
            gsrc.reserve(n_res * 2);
            gdst.reserve(n_res * 2);
            gbytes.reserve(n_res * 2);
            ssrc.reserve(n_res * 2);
            sdst.reserve(n_res * 2);
            sbytes.reserve(n_res * 2);
            for (size_t i = 0; i < items.size(); ++i) {
                if (res_ix[i] == static_cast<size_t>(-1)) {
                    continue;
                }
                const uint32_t nt = items[i].n;
                const uint32_t src0 = static_cast<uint32_t>(items[i].slot) * block_tokens_;
                const uint32_t dst0 = static_cast<uint32_t>(i) * block_tokens_;
                uint8_t * slot_sc = scratch + res_ix[i] * scratch_stride;
                if (kbase && krow && nt) {
                    const size_t nb = static_cast<size_t>(nt) * krow;
                    gsrc.push_back(kbase + static_cast<size_t>(src0) * krow);
                    gdst.push_back(slot_sc);
                    gbytes.push_back(nb);
                    ssrc.push_back(slot_sc);
                    sdst.push_back(kbase + static_cast<size_t>(dst0) * krow);
                    sbytes.push_back(nb);
                }
                if (vbase && vrow && nt) {
                    const size_t nb = static_cast<size_t>(nt) * vrow;
                    gsrc.push_back(vbase + static_cast<size_t>(src0) * vrow);
                    gdst.push_back(slot_sc + kspan);
                    gbytes.push_back(nb);
                    ssrc.push_back(slot_sc + kspan);
                    sdst.push_back(vbase + static_cast<size_t>(dst0) * vrow);
                    sbytes.push_back(nb);
                }
            }
            const int64_t t_g = ggml_time_us();
            const int ng = static_cast<int>(gsrc.size());
            if (!kvmem_d2d_batched(gsrc.data(), gdst.data(), gbytes.data(), ng)) {
                for (int j = 0; j < ng && copy_ok; ++j) {
                    copy_ok = kvmem_d2d(static_cast<uint8_t *>(gdst[j]),
                                        static_cast<const uint8_t *>(gsrc[j]),
                                        gbytes[j], st);
                }
            }
            if (!copy_ok || cudaStreamSynchronize(st) != cudaSuccess) {
                copy_ok = false;
                break;
            }
            if (retr_.enabled) {
                retr_.layout_d2h_us += ggml_time_us() - t_g;
            }
            const int64_t t_s = ggml_time_us();
            const int ns = static_cast<int>(ssrc.size());
            if (!kvmem_d2d_batched(ssrc.data(), sdst.data(), sbytes.data(), ns)) {
                for (int j = 0; j < ns && copy_ok; ++j) {
                    copy_ok = kvmem_d2d(static_cast<uint8_t *>(sdst[j]),
                                        static_cast<const uint8_t *>(ssrc[j]),
                                        sbytes[j], st);
                }
            }
            if (!copy_ok || cudaStreamSynchronize(st) != cudaSuccess) {
                copy_ok = false;
                break;
            }
            if (retr_.enabled) {
                retr_.layout_h2d_us += ggml_time_us() - t_s;
            }
        }
        cudaFree(scratch);
        scratch = nullptr;
        d2d_ok = copy_ok;
        if (!copy_ok) {
            LLAMA_LOG_ERROR("%s: layout D2D failed; reloading residents from raw-K\n", __func__);
        }
        if (d2d_ok && mtp_) {
            std::vector<llama_memory_kvmem_mtp::LayoutMove> mm;
            for (size_t i = 0; i < items.size(); ++i) {
                if (!mtp_res[i]) {
                    continue;
                }
                mm.push_back({items[i].slot, static_cast<int32_t>(i), items[i].n});
            }
            mtp_d2d_ok = mm.empty() || mtp_->layout_d2d(mm.data(), mm.size());
            if (!mtp_d2d_ok) {
                LLAMA_LOG_WARN("%s: MTP layout D2D failed; restoring those blocks from raw\n",
                               __func__);
            }
        }
    } else if (n_res > 0) {
        const uint64_t payload_bytes =
                static_cast<uint64_t>(n_layer_) * (krow + vrow) * block_tokens_;
        std::vector<std::vector<uint8_t>> payloads(items.size());
        const int64_t t_d2h = ggml_time_us();
        for (size_t i = 0; i < items.size(); ++i) {
            if (res_ix[i] == static_cast<size_t>(-1) || payload_bytes == 0) {
                continue;
            }
            payloads[i].assign(static_cast<size_t>(payload_bytes), 0);
            copy_gpu_block_to_host(items[i].id, items[i].slot,
                                   payloads[i].data(), payload_bytes);
        }
        if (retr_.enabled) {
            retr_.layout_d2h_us += ggml_time_us() - t_d2h;
        }
        {
            const int64_t t0 = ggml_time_us();
            for (const auto & it : items) {
                kv_->seq_rm_logical(0, static_cast<llama_pos>(it.orig),
                            static_cast<llama_pos>(it.orig + it.n));
                if (mtp_) {
                    mtp_->remove_logical(static_cast<llama_pos>(it.orig),
                                 static_cast<llama_pos>(it.orig + it.n));
                }
                store.set_block_gpu_slot(it.id, -1);
            }
            if (retr_.enabled) {
                retr_.seq_rm_us += ggml_time_us() - t0;
            }
        }
        reset_slots();
        if (trace_) {
            kvmem_diag("KVMEM_TRACE layout_orig_pos");
            for (size_t i = 0; i < items.size(); ++i) {
                fprintf(stderr, " %u", items[i].id);
            }
            fprintf(stderr, "\n");
        }
        uint32_t n_move = 0;
        uint32_t n_raw = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            const int32_t slot = alloc_slot();
            if (slot < 0) {
                LLAMA_LOG_ERROR("%s: no GPU slot while laying out block %u\n",
                                __func__, items[i].id);
                return false;
            }
            store.set_block_gpu_slot(items[i].id, slot);
            store.set_block_tier(items[i].id, kvmem::KvTier::GPU, -1,
                                 store.blocks()[items[i].id].nvme_slot);
            if (items[i].resident && !payloads[i].empty()) {
                occupy_block_cells(items[i].id);
                const int64_t t_h2d = ggml_time_us();
                copy_gpu_block_from_host(items[i].id, slot,
                                         payloads[i].data(), payload_bytes);
                if (retr_.enabled) {
                    retr_.layout_h2d_us += ggml_time_us() - t_h2d;
                }
                n_move++;
            } else {
                write_block_to_gpu(items[i].id);
                n_raw++;
            }
        }
        if (retr_.enabled) {
            retr_.n_move += n_move;
            retr_.n_raw += n_raw;
            retr_.laid_out = 1;
        }
        kvmem_stagein_flush_sync(retr_.enabled ? &retr_.copy_us : nullptr,
                                 retr_.enabled ? &retr_.rope_us : nullptr,
                                 retr_.enabled ? &retr_.hadamard_us : nullptr,
                                 retr_.enabled ? &retr_.set_us : nullptr);
        if (trace_) {
            kvmem_diag("KVMEM_TRACE layout_writeback move=%u raw=%u\n", n_move, n_raw);
        }
        return true;
    }

    {
        const int64_t t0 = ggml_time_us();
        for (const auto & it : items) {
            kv_->seq_rm_logical(0, static_cast<llama_pos>(it.orig),
                        static_cast<llama_pos>(it.orig + it.n));
            if (mtp_) {
                mtp_->remove_logical(static_cast<llama_pos>(it.orig),
                             static_cast<llama_pos>(it.orig + it.n));
            }
            store.set_block_gpu_slot(it.id, -1);
        }
        if (retr_.enabled) {
            retr_.seq_rm_us += ggml_time_us() - t0;
        }
    }
    reset_slots();
    if (trace_) {
        kvmem_diag("KVMEM_TRACE layout_orig_pos");
        for (size_t i = 0; i < items.size(); ++i) {
            fprintf(stderr, " %u", items[i].id);
        }
        fprintf(stderr, "\n");
    }
    uint32_t n_move = 0;
    uint32_t n_raw = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const int32_t slot = alloc_slot();
        if (slot < 0) {
            LLAMA_LOG_ERROR("%s: no GPU slot while laying out block %u\n",
                            __func__, items[i].id);
            return false;
        }
        store.set_block_gpu_slot(items[i].id, slot);
        store.set_block_tier(items[i].id, kvmem::KvTier::GPU, -1,
                             store.blocks()[items[i].id].nvme_slot);
        if (items[i].resident && d2d_ok) {
            occupy_block_cells(items[i].id);
            if (mtp_ && mtp_res[i] && mtp_d2d_ok) {
                mtp_->occupy_block(items[i].id);
            }
            n_move++;
        } else {
            write_block_to_gpu(items[i].id);
            n_raw++;
        }
    }
    if (retr_.enabled) {
        retr_.n_move += n_move;
        retr_.n_raw += n_raw;
        retr_.laid_out = 1;
    }
    kvmem_stagein_flush_sync(retr_.enabled ? &retr_.copy_us : nullptr,
                             retr_.enabled ? &retr_.rope_us : nullptr,
                             retr_.enabled ? &retr_.hadamard_us : nullptr,
                             retr_.enabled ? &retr_.set_us : nullptr);
    if (trace_) {
        kvmem_diag("KVMEM_TRACE layout_writeback move=%u raw=%u d2d=%d\n",
                n_move, n_raw, (int) d2d_ok);
    }
    return true;
}

bool llama_memory_kvmem::prepare_working_set(uint32_t n_new_tokens) {
    if (n_new_tokens == 0) {
        return true;
    }
    if (replay_) {
        return true;
    }
    auto & store = runtime_->store();
    const uint32_t t0 = store.total_tokens();
    if (!replay_) {
        runtime_->register_append(n_new_tokens);
    }
    const uint32_t t1 = store.total_tokens();

    std::vector<uint32_t> incoming;
    for (const auto & b : store.blocks()) {
        if (b.orig_pos_end() > t0 && b.orig_pos_start < t1) {
            incoming.push_back(b.block_id);
        }
    }

    const uint32_t budget_blocks = store.prefill_budget_blocks();
    bool need_offload = false;
    // After retrieval the host store still holds every historical block, so
    // block_count() > budget is true and a recency pressure reselect would
    // drop the resurrected needle on the first generated token. Pin the
    // working set and place decode tokens into gen_reserve slots.
    if (!retrieval_pinned_ && !keep_selected_) {
        try {
            need_offload = runtime_->maybe_offload_during_prefill(
                    n_new_tokens, resident_tokens(), kv_size_, incoming);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: KVMem reselect failed: %s\n", __func__, e.what());
            runtime_->truncate_to(t0);
            return false;
        }
    }

    if (trace_) {
        kvmem_diag("KVMEM_TRACE append n=%u total=%u resident=%u incoming_blocks=%zu "
                "over_budget=%d need_offload=%d free_slots=%zu\n",
                n_new_tokens, t1, resident_tokens(), incoming.size(),
                (int) (store.block_count() > budget_blocks), (int) need_offload,
                free_slots_.size());
    }

    if (need_offload) {
        try {
            const kvmem::KvMemPlan & plan = runtime_->last_plan();
            trace_plan("prefill_pressure", plan);
            apply_plan_to_kv(plan);
            if (perf_.enabled) {
                perf_.n_pressure += 1;
                perf_.n_pressure_out += static_cast<uint32_t>(plan.stage_out.size());
            }
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: KVMem reselect failed: %s\n", __func__, e.what());
            runtime_->truncate_to(t0);
            return false;
        }
    } else {
        for (uint32_t id : incoming) {
            if (store.blocks()[id].gpu_slot >= 0) {
                continue;
            }
            const int32_t slot = alloc_slot();
            if (slot < 0) {
                // Pinned retrieval will not evict the working set. gen_reserve
                // exhaustion is a known v1 limit (see docs/architecture.md).
                LLAMA_LOG_ERROR("%s: no free GPU slot for block %u\n", __func__, id);
                runtime_->truncate_to(t0);
                return false;
            }
            store.set_block_gpu_slot(id, slot);
            store.set_block_tier(id, kvmem::KvTier::GPU, -1, store.blocks()[id].nvme_slot);
        }
    }

    for (uint32_t id : incoming) {
        if (id < store.block_count() && store.blocks()[id].gpu_slot < 0) {
            LLAMA_LOG_ERROR("%s: incoming block %u was not placed on GPU\n", __func__, id);
            runtime_->truncate_to(t0);
            return false;
        }
    }
    return true;
}

bool llama_memory_kvmem::prepare_ubatches(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t n_new_tokens,
        llama_kv_cache::slot_info_vec_t & sinfos) {
    const uint32_t old_rows = store_n_tokens();
    for (const auto & ub : ubatches) {
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_pos row = ub.logical_pos ? ub.logical_pos[i] : ub.pos[i];
            if (row >= 0 && (uint32_t) row < old_rows) { ++attention_epoch_; break; }
        }
    }
    if (!prepare_working_set(n_new_tokens)) {
        return false;
    }
    for (const auto & ub : ubatches) {
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_pos logical = ub.logical_pos ? ub.logical_pos[i] : ub.pos[i];
            if (logical < 0 || (uint32_t) logical >= runtime_->store().total_tokens()) return false;
            if ((size_t) logical >= row_positions_.size()) row_positions_.resize(logical + 1);
            auto & row = row_positions_[logical];
            row.spatial = ub.is_pos_2d();
            for (uint32_t j = 0; j < 4; ++j) row.pos[j] = ub.pos[(j < ub.n_pos ? j*ub.n_tokens : 0) + i];
            row.token = ub.token ? ub.token[i] : LLAMA_TOKEN_NULL;
        }
    }
    sinfos.clear();
    sinfos.reserve(ubatches.size());
    for (const auto & ubatch : ubatches) {
        llama_kv_cache::slot_info sinfo;
        if (!kvmem_fill_slot_info(runtime_->store(), block_tokens_, kv_size_, ubatch, sinfo)) {
            return false;
        }
        sinfos.push_back(std::move(sinfo));
    }
    pos_queue_.clear();
    for (const auto & ubatch : ubatches) {
        std::vector<llama_pos> p(ubatch.n_tokens);
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            p[i] = ubatch.logical_pos ? ubatch.logical_pos[i] : ubatch.pos[i];
        }
        pos_queue_.push_back(std::move(p));
    }
    return true;
}

llama_memory_context_ptr llama_memory_kvmem::init_batch(
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
        if (!prepare_ubatches(ubatches, balloc.get_n_tokens(), sinfos)) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                kv_, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_kvmem::init_full() {
    return kv_->init_full();
}

llama_memory_context_ptr llama_memory_kvmem::init_update(llama_context * lctx, bool optimize) {
    return kv_->init_update(lctx, optimize);
}

void llama_memory_kvmem::clear(bool data) {
    harvest_flush();
    harvest_gpu_v_commit();
    if (kv_) {
        kv_->clear(data);
    }
    reset_policy();
}

bool llama_memory_kvmem::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    ++attention_epoch_;
    const bool ok = kv_->seq_rm(seq_id, p0, p1);
    if (!ok) {
        return false;
    }
    // Query replay seq_rm's the query span, which can be the whole prompt on
    // short inputs. That must not wipe KvMemStore metadata / GPU slots.
    if (!replay_ &&
        seq_id <= 0 && p0 <= 0 &&
        (p1 < 0 || p1 >= static_cast<llama_pos>(runtime_->store().total_tokens()))) {
        runtime_->truncate_to(0);
        reset_slots();
        retrieval_pinned_ = false;
    }
    return true;
}

void llama_memory_kvmem::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    ++attention_epoch_;
    kv_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_kvmem::seq_keep(llama_seq_id seq_id) {
    ++attention_epoch_;
    kv_->seq_keep(seq_id);
}

void llama_memory_kvmem::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    ++attention_epoch_;
    kv_->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_kvmem::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    ++attention_epoch_;
    kv_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_kvmem::seq_pos_min(llama_seq_id seq_id) const {
    return kv_->seq_pos_min(seq_id);
}

llama_pos llama_memory_kvmem::seq_pos_max(llama_seq_id seq_id) const {
    return kv_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_kvmem::memory_breakdown() const {
    return kv_->memory_breakdown();
}

void llama_memory_kvmem::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv_->state_write(io, seq_id, flags);
}

void llama_memory_kvmem::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ++attention_epoch_;
    kv_->state_read(io, seq_id, flags);
}

void llama_memory_kvmem::note_ubatch_pos(const std::vector<llama_pos> & pos) {
    pos_queue_.push_back(pos);
}

void llama_memory_kvmem::register_capture(ggml_tensor * t, int il, char which) {
    if (!t) {
        return;
    }
    pending_capture_.push_back({t, il, which});
    if (which == 'q') {
        graph_has_q_ = true;
    }
    if (which == 'k') {
        graph_has_k_ = true;
    }
}

void llama_memory_kvmem::capture_on_new_graph() {
    pending_capture_.clear();
    graph_has_q_ = false;
    graph_has_k_ = false;
    graph_has_record_ = recr_ && recr_->replay_recording;
}

bool llama_memory_kvmem::capture_can_reuse(uint32_t n_tokens, uint32_t n_pos,
                                           const llama_pos * pos) const {
    if (graph_has_record_ != bool(recr_ && recr_->replay_recording)) return false;
    // After pin, n=1 decode must not reuse a graph built without K capture
    // (last prefill ubatch of 1 token, or query replay with capture off).
    if (want_decode_mean() && !graph_has_k_) {
        return false;
    }
    // Post-pin query replay must not keep rebuilding just because the ubatch
    // overlaps the query span — Q nodes are off after pin. T5 recapture
    // (want_q_capture, including replay) still requires a Q graph.
    const bool need_q = want_q_capture() &&
            llama_kvmem_ubatch_needs_q_capture(n_tokens, n_pos, pos);
    return need_q == graph_has_q_;
}

bool llama_memory_kvmem::d2h_init() {
    if (d2h_ && d2h_->ok) {
        return true;
    }
    if (!d2h_) {
        d2h_ = std::make_unique<CaptureD2hPipe>();
    }
    if (d2h_->stream == nullptr) {
        if (!kvmem_cuda_ok(cudaStreamCreateWithFlags(&d2h_->stream, cudaStreamNonBlocking),
                           "stream")) {
            return false;
        }
    }
    if (d2h_->snap == nullptr) {
        if (!kvmem_cuda_ok(cudaEventCreateWithFlags(&d2h_->snap, cudaEventDisableTiming),
                           "snap")) {
            return false;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (d2h_->slots[i].done == nullptr) {
            if (!kvmem_cuda_ok(cudaEventCreateWithFlags(&d2h_->slots[i].done, cudaEventDisableTiming),
                               "event")) {
                return false;
            }
        }
    }
    if (cudaGetDevice(&d2h_->device) != cudaSuccess) {
        d2h_->device = 0;
    }
    d2h_->ok = true;
    harvest_worker_start();
    return true;
}

void llama_memory_kvmem::d2h_free() {
    if (!d2h_) {
        return;
    }
    if (d2h_->stream) {
        cudaStreamSynchronize(d2h_->stream);
    }
    for (int i = 0; i < 2; ++i) {
        auto & s = d2h_->slots[i];
        if (s.gpu) {
            cudaFree(s.gpu);
            s.gpu = nullptr;
        }
        if (s.pin) {
            cudaFreeHost(s.pin);
            s.pin = nullptr;
        }
        if (s.done) {
            cudaEventDestroy(s.done);
            s.done = nullptr;
        }
        s.cap = 0;
        s.inflight = false;
    }
    if (d2h_->snap) {
        cudaEventDestroy(d2h_->snap);
        d2h_->snap = nullptr;
    }
    if (d2h_->compute_done) {
        ggml_backend_event_free(d2h_->compute_done);
        d2h_->compute_done = nullptr;
    }
    if (d2h_->snap_be) {
        ggml_backend_event_free(d2h_->snap_be);
        d2h_->snap_be = nullptr;
    }
    if (d2h_->stream) {
        cudaStreamDestroy(d2h_->stream);
        d2h_->stream = nullptr;
    }
    d2h_->ok = false;
}

bool llama_memory_kvmem::harvest_worker_on() const {
    return harvest_w_ && harvest_w_->th.joinable();
}

void llama_memory_kvmem::harvest_worker_start() {
    if (kvmem_harvest_sync_old() || harvest_worker_on()) {
        return;
    }
    if (!harvest_w_) {
        harvest_w_ = std::make_unique<HarvestWorker>();
    }
    harvest_w_->stop = false;
    harvest_w_->th = std::thread([this] { harvest_loop(); });
}

void llama_memory_kvmem::harvest_worker_stop() {
    if (!harvest_w_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(harvest_w_->mu);
        harvest_w_->stop = true;
    }
    harvest_w_->cv.notify_all();
    if (harvest_w_->th.joinable()) {
        harvest_w_->th.join();
    }
}

void llama_memory_kvmem::harvest_wait_slot(int slot) {
    if (!harvest_w_ || !d2h_ || slot < 0 || slot > 1) {
        return;
    }
    std::unique_lock<std::mutex> lk(harvest_w_->mu);
    harvest_w_->cv.wait(lk, [&] {
        return !d2h_->slots[slot].inflight;
    });
}

void llama_memory_kvmem::harvest_loop() {
    if (d2h_) {
        kvmem_cuda_ok(cudaSetDevice(d2h_->device), "harvest worker set device");
    }
    while (true) {
        int slot = -1;
        {
            std::unique_lock<std::mutex> lk(harvest_w_->mu);
            harvest_w_->cv.wait(lk, [&] {
                return harvest_w_->stop || !harvest_w_->q.empty();
            });
            if (harvest_w_->q.empty() && harvest_w_->stop) {
                break;
            }
            slot = harvest_w_->q.front();
            harvest_w_->q.erase(harvest_w_->q.begin());
        }
        d2h_commit(slot);
    }
}

void llama_memory_kvmem::d2h_commit(int slot) {
    if (!d2h_ || slot < 0 || slot > 1) {
        return;
    }
    auto & s = d2h_->slots[slot];
    if (!s.inflight) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    int64_t wait_us = 0;
    if (s.done) {
        const int64_t tw = ggml_time_us();
        cudaEventSynchronize(s.done);
        wait_us = ggml_time_us() - tw;
    }
    cur_pos_ = s.pos;
    const uint64_t nv0 = raw_ ? raw_->nvme_wait_ns() : 0;
    const uint64_t sc0 = raw_ ? raw_->nvme_syscalls() : 0;
    const uint64_t b0 = raw_ ? raw_->nvme_bytes_written() : 0;
    const int64_t tp = ggml_time_us();
    for (const auto & it : s.items) {
        if (it.which == 'q') {
            harvest_from_host(it.il, it.which, s.pin + it.offset, it.type,
                              it.d, it.h, it.n, it.nb0, it.nb1, it.nb2);
        }
    }
    for (const auto & it : s.items) {
        if (it.which != 'q') {
            harvest_from_host(it.il, it.which, s.pin + it.offset, it.type,
                              it.d, it.h, it.n, it.nb0, it.nb1, it.nb2);
        }
    }
    const int64_t pack_wall_us = ggml_time_us() - tp;
    const uint64_t nvme_us = raw_ ? (raw_->nvme_wait_ns() - nv0) / 1000 : 0;
    perf_.last_d2h_wait_us = wait_us;
    perf_.last_nvme_us = static_cast<int64_t>(nvme_us);
    perf_.last_pack_us = pack_wall_us > static_cast<int64_t>(nvme_us)
            ? pack_wall_us - static_cast<int64_t>(nvme_us) : 0;
    perf_.last_nvme_bytes = raw_ ? (raw_->nvme_bytes_written() - b0) : 0;
    perf_.last_nvme_syscalls = raw_ ? (raw_->nvme_syscalls() - sc0) : 0;
    perf_.last_commit_us = ggml_time_us() - t0;
    perf_.d2h_wait_us += perf_.last_d2h_wait_us;
    perf_.pack_us += perf_.last_pack_us;
    perf_.nvme_us += perf_.last_nvme_us;
    perf_.nvme_bytes += perf_.last_nvme_bytes;
    perf_.nvme_syscalls += perf_.last_nvme_syscalls;
    perf_.commit_us += perf_.last_commit_us;
    s.items.clear();
    s.pos.clear();
    if (harvest_w_) {
        std::lock_guard<std::mutex> lk(harvest_w_->mu);
        s.inflight = false;
        harvest_w_->cv.notify_all();
    } else {
        s.inflight = false;
    }
}

void llama_memory_kvmem::harvest_flush() {
    if (harvest_worker_on() && d2h_) {
        std::unique_lock<std::mutex> lk(harvest_w_->mu);
        harvest_w_->cv.wait(lk, [&] {
            return harvest_w_->q.empty() &&
                    !d2h_->slots[0].inflight &&
                    !d2h_->slots[1].inflight;
        });
    } else if (d2h_) {
        d2h_commit(1 - d2h_->next);
        d2h_commit(d2h_->next);
    }
    if (raw_) {
        raw_->wait_writes();
    }
    if (mtp_) {
        mtp_->harvest_flush();
    }
    harvest_perf_print_sum();
}

void llama_memory_kvmem::harvest_perf_emit_graph_line() {
    if (!perf_.enabled || perf_.graph_line_printed) {
        return;
    }
    perf_.graph_line_printed = true;
    // Capture Y/N needs GGML_LOG_LEVEL=DEBUG; this line is the adapter signal.
    kvmem_diag("KVMEM_CUDA_GRAPH compiled=1 captured=-1 resets=-1\n");
}

void llama_memory_kvmem::harvest_perf_print_sum() {
    if (!perf_.enabled || perf_.sum_printed || perf_.n_ubatch == 0) {
        return;
    }
    perf_.sum_printed = true;
    int64_t mtp_sync_us = 0;
    uint64_t mtp_nvme_bytes = 0;
    uint64_t mtp_nvme_syscalls = 0;
    uint32_t mtp_n = 0;
    if (mtp_) {
        mtp_n = mtp_->harvest_perf_n_ubatch();
        mtp_sync_us = mtp_->harvest_perf_sync_us();
        mtp_nvme_bytes = mtp_->harvest_perf_nvme_bytes();
        mtp_nvme_syscalls = mtp_->harvest_perf_nvme_syscalls();
    }
    kvmem_diag("KVMEM_HARVEST_SUM n_ubatch=%u n_tok=%u "
            "sync_ms=%.3f d2d_ms=%.3f d2h_wait_ms=%.3f pack_ms=%.3f nvme_ms=%.3f "
            "nvme_bytes=%llu nvme_syscalls=%llu "
            "n_pressure=%u n_pressure_out=%u "
            "mtp_n_ubatch=%u mtp_sync_ms=%.3f mtp_nvme_bytes=%llu mtp_nvme_syscalls=%llu\n",
            perf_.n_ubatch, perf_.n_tok,
            perf_.sync_us / 1000.0, perf_.d2d_us / 1000.0,
            perf_.d2h_wait_us / 1000.0, perf_.pack_us / 1000.0,
            perf_.nvme_us / 1000.0,
            (unsigned long long) perf_.nvme_bytes,
            (unsigned long long) perf_.nvme_syscalls,
            perf_.n_pressure, perf_.n_pressure_out,
            mtp_n, mtp_sync_us / 1000.0,
            (unsigned long long) mtp_nvme_bytes,
            (unsigned long long) mtp_nvme_syscalls);
}

bool llama_memory_kvmem::d2h_submit(ggml_backend_t be) {
    if (!d2h_init() || pending_capture_.empty() || pos_queue_.empty()) {
        return false;
    }
    size_t bytes = 0;
    for (const CaptureNode & n : pending_capture_) {
        if (n.t && n.which != 'v') {
            bytes += ggml_nbytes(n.t);
        }
    }
    if (bytes == 0) {
        return false;
    }
    auto & s = d2h_->slots[d2h_->next];
    if (s.inflight) {
        if (harvest_worker_on()) {
            const int64_t tw = ggml_time_us();
            harvest_wait_slot(d2h_->next);
            perf_.last_commit_us = ggml_time_us() - tw;
            perf_.commit_us += perf_.last_commit_us;
        } else {
            d2h_commit(d2h_->next);
        }
    }
    perf_.last_sync_us = 0;
    perf_.last_d2d_us = 0;
    perf_.last_snap_wait_us = 0;
    perf_.last_d2h_submit_us = 0;
    const int64_t t_submit0 = ggml_time_us();
    if (s.cap < bytes) {
        if (s.gpu) {
            cudaFree(s.gpu);
            s.gpu = nullptr;
        }
        if (s.pin) {
            cudaFreeHost(s.pin);
            s.pin = nullptr;
        }
        if (!kvmem_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&s.gpu), bytes), "gpu staging") ||
            !kvmem_cuda_ok(cudaMallocHost(reinterpret_cast<void **>(&s.pin), bytes), "pinned host")) {
            s.cap = 0;
            return false;
        }
        s.cap = bytes;
        kvmem_diag("KVMEM_CAPTURE_MEMORY mode=raw slot0_bytes=%zu slot1_bytes=%zu last_bytes=%zu pinned_bytes=%zu\n",
                d2h_->slots[0].cap, d2h_->slots[1].cap, bytes, d2h_->slots[0].cap + d2h_->slots[1].cap);
    }
    if (be && !d2h_->compute_done) {
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        if (dev) {
            d2h_->compute_done = ggml_backend_event_new(dev);
            d2h_->snap_be = ggml_backend_event_new(dev);
        }
    }
    const bool old_sync = kvmem_harvest_sync_old() || !be || !d2h_->compute_done ||
            !kvmem_ggml_cuda_event(d2h_->compute_done);
    {
        const int64_t t_sync = ggml_time_us();
        if (old_sync) {
            if (be) {
                ggml_backend_synchronize(be);
            } else {
                cudaDeviceSynchronize();
            }
        } else {
            ggml_backend_event_record(d2h_->compute_done, be);
            cudaEvent_t cde = kvmem_ggml_cuda_event(d2h_->compute_done);
            if (cde) {
                cudaStreamWaitEvent(d2h_->stream, cde, 0);
            } else {
                ggml_backend_synchronize(be);
            }
        }
        perf_.last_sync_us = ggml_time_us() - t_sync;
        perf_.sync_us += perf_.last_sync_us;
    }
    s.items.clear();
    s.pos = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    size_t off = 0;
    uint32_t n_q = 0, n_k = 0, n_v = 0, n_host = 0, n_dev = 0;
    const int64_t t_d2d = ggml_time_us();
    for (const CaptureNode & n : pending_capture_) {
        if (!n.t || n.which == 'v') {
            continue;
        }
        CaptureD2hPipe::Item it;
        it.il = n.il;
        it.which = n.which;
        it.offset = off;
        it.nbytes = ggml_nbytes(n.t);
        it.d = n.t->ne[0];
        it.h = n.t->ne[1];
        it.n = n.t->ne[2];
        it.nb0 = n.t->nb[0];
        it.nb1 = n.t->nb[1];
        it.nb2 = n.t->nb[2];
        it.type = n.t->type;
        ggml_backend_buffer_t buf = n.t->view_src ? n.t->view_src->buffer : n.t->buffer;
        const uint8_t * src = static_cast<const uint8_t *>(n.t->data);
        if (buf && ggml_backend_buffer_is_host(buf)) {
            memcpy(s.pin + off, src, it.nbytes);
            n_host++;
        } else {
            if (!kvmem_cuda_ok(kvmem_copy_async(s.gpu + off, src, it.nbytes,
                                               cudaMemcpyDeviceToDevice, d2h_->stream),
                               "D2D")) {
                pos_queue_.insert(pos_queue_.begin(), std::move(s.pos));
                return false;
            }
            it.from_gpu = true;
            n_dev++;
        }
        if (n.which == 'q') {
            n_q++;
        } else if (n.which == 'k') {
            n_k++;
        } else if (n.which == 'v') {
            n_v++;
        }
        s.items.push_back(it);
        off += it.nbytes;
    }
    perf_.last_d2d_us = ggml_time_us() - t_d2d;
    perf_.d2d_us += perf_.last_d2d_us;
    if (n_dev > 0) {
        if (!kvmem_cuda_ok(cudaEventRecord(d2h_->snap, d2h_->stream), "snap record")) {
            return false;
        }
        const int64_t t_snap = ggml_time_us();
        cudaEvent_t snap_cuda = kvmem_ggml_cuda_event(d2h_->snap_be);
        if (!old_sync && be && snap_cuda) {
            // Record the ggml event on the harvest stream (never
            // ggml_backend_event_record(snap_be, be) — that is the compute stream).
            if (!kvmem_cuda_ok(cudaEventRecord(snap_cuda, d2h_->stream), "snap_be record")) {
                return false;
            }
            ggml_backend_event_wait(be, d2h_->snap_be);
        } else if (!kvmem_cuda_ok(cudaEventSynchronize(d2h_->snap), "snap wait")) {
            return false;
        }
        perf_.last_snap_wait_us = ggml_time_us() - t_snap;
        perf_.snap_wait_us += perf_.last_snap_wait_us;
        if (n_host == 0) {
            if (!kvmem_cuda_ok(kvmem_copy_async(s.pin, s.gpu, off, cudaMemcpyDeviceToHost, d2h_->stream),
                               "D2H")) {
                return false;
            }
        } else {
            for (const auto & it : s.items) {
                if (!it.from_gpu) {
                    continue;
                }
                if (!kvmem_cuda_ok(kvmem_copy_async(s.pin + it.offset, s.gpu + it.offset, it.nbytes,
                                                   cudaMemcpyDeviceToHost, d2h_->stream),
                                   "D2H item")) {
                    return false;
                }
            }
        }
    }
    if (!kvmem_cuda_ok(cudaEventRecord(s.done, d2h_->stream), "record")) {
        return false;
    }
    const int submitted = d2h_->next;
    if (harvest_worker_on()) {
        {
            std::lock_guard<std::mutex> lk(harvest_w_->mu);
            s.inflight = true;
            harvest_w_->q.push_back(submitted);
        }
        harvest_w_->cv.notify_one();
    } else {
        s.inflight = true;
    }
    if (!old_sync && d2h_->compute_done) {
        ggml_backend_event_synchronize(d2h_->compute_done);
    }
    perf_.last_d2h_submit_us = ggml_time_us() - t_submit0;
    perf_.d2h_submit_us += perf_.last_d2h_submit_us;
    if (trace_) {
        kvmem_diag("KVMEM_TRACE harvest n=%zu q=%u k=%u v=%u host=%u gpu=%u bytes=%zu n_pos=%zu async=1 slot=%d\n",
                s.items.size(), n_q, n_k, n_v, n_host, n_dev, off, s.pos.size(), submitted);
    }
    d2h_->next = 1 - d2h_->next;
    return true;
}

void llama_memory_kvmem::harvest_pending(ggml_backend_sched_t sched) {
    if (want_decode_mean()) {
        decode_mean_ingest(sched);
        return;
    }
    const int64_t t_entry = ggml_time_us();
    perf_.last_commit_us = 0;
    perf_.last_d2h_wait_us = 0;
    perf_.last_pack_us = 0;
    perf_.last_nvme_us = 0;
    perf_.last_nvme_bytes = 0;
    perf_.last_nvme_syscalls = 0;
    perf_.last_sync_us = 0;
    perf_.last_d2d_us = 0;
    perf_.last_snap_wait_us = 0;
    perf_.last_d2h_submit_us = 0;
    if (d2h_ && d2h_->ok && !harvest_worker_on()) {
        d2h_commit(d2h_->next == 0 ? 1 : 0);
    }
    if (pending_capture_.empty() || pos_queue_.empty()) {
        harvest_flush();
        if (sched) {
            ggml_backend_sched_synchronize(sched);
        }
        harvest_full_blocks_async();
        return;
    }
    ggml_backend_t be = nullptr;
    if (sched) {
        for (const CaptureNode & n : pending_capture_) {
            if (n.t) {
                be = ggml_backend_sched_get_tensor_backend(sched, n.t);
                if (be) {
                    break;
                }
            }
        }
    }
    const uint32_t n_pos = pos_queue_.empty() ? 0u : (uint32_t) pos_queue_.front().size();
    bool submitted = d2h_submit(be);
    if (!submitted) {
        cur_pos_ = std::move(pos_queue_.front());
        pos_queue_.erase(pos_queue_.begin());
        for (const CaptureNode & n : pending_capture_) {
            harvest_capture(n.t, n.il, n.which);
        }
        if (sched) {
            ggml_backend_sched_synchronize(sched);
        }
    }
    harvest_full_blocks_async();
    const int64_t entry_us = ggml_time_us() - t_entry;
    perf_.harvest_entry_us += entry_us;
    perf_.n_ubatch += 1;
    perf_.n_tok += n_pos;
    if (perf_.enabled) {
        kvmem_diag("KVMEM_HARVEST ubatch=%u n=%u is_mtp=0 "
                "harvest_entry_us=%lld sync_us=%lld d2d_us=%lld snap_wait_us=%lld "
                "d2h_submit_us=%lld commit_us=%lld pack_us=%lld "
                "nvme_us=%lld nvme_bytes=%llu nvme_syscalls=%llu\n",
                perf_.n_ubatch, n_pos,
                (long long) entry_us,
                (long long) perf_.last_sync_us,
                (long long) perf_.last_d2d_us,
                (long long) perf_.last_snap_wait_us,
                (long long) perf_.last_d2h_submit_us,
                (long long) perf_.last_commit_us,
                (long long) perf_.last_pack_us,
                (long long) perf_.last_nvme_us,
                (unsigned long long) perf_.last_nvme_bytes,
                (unsigned long long) perf_.last_nvme_syscalls);
    }
}

void llama_memory_kvmem::reset_query_acc() {
    for (auto & s : q_sum_) {
        std::fill(s.begin(), s.end(), 0.0f);
    }
    std::fill(q_count_.begin(), q_count_.end(), 0);
}

void llama_memory_kvmem::bytes_to_f32_token_major(const uint8_t * data, ggml_type type,
                                                  int64_t d, int64_t h, int64_t n,
                                                  size_t nb0, size_t nb1, size_t nb2,
                                                  std::vector<float> & out) {
    out.assign(static_cast<size_t>(n * h * d), 0.0f);
    if (!data) {
        return;
    }
    for (int64_t tok = 0; tok < n; ++tok) {
        for (int64_t head = 0; head < h; ++head) {
            for (int64_t dim = 0; dim < d; ++dim) {
                const size_t off = static_cast<size_t>(tok * nb2 + head * nb1 + dim * nb0);
                float val = 0.0f;
                if (type == GGML_TYPE_F32) {
                    val = *reinterpret_cast<const float *>(data + off);
                } else if (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + off));
                } else if (type == GGML_TYPE_BF16) {
                    val = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data + off));
                }
                out[static_cast<size_t>(tok * h * d + head * d + dim)] = val;
            }
        }
    }
}

void llama_memory_kvmem::bytes_to_f16_token_major(const uint8_t * data, ggml_type type,
                                                  int64_t d, int64_t h, int64_t n,
                                                  size_t nb0, size_t nb1, size_t nb2,
                                                  std::vector<uint16_t> & out) {
    out.assign(static_cast<size_t>(n * h * d), 0);
    if (!data || type != GGML_TYPE_F16) {
        return;
    }
    const bool packed = nb0 == sizeof(uint16_t) &&
            nb1 == static_cast<size_t>(d) * sizeof(uint16_t) &&
            nb2 == static_cast<size_t>(h) * static_cast<size_t>(d) * sizeof(uint16_t);
    if (packed) {
        std::memcpy(out.data(), data, out.size() * sizeof(uint16_t));
        return;
    }
    for (int64_t tok = 0; tok < n; ++tok) {
        for (int64_t head = 0; head < h; ++head) {
            for (int64_t dim = 0; dim < d; ++dim) {
                const size_t off = static_cast<size_t>(tok * nb2 + head * nb1 + dim * nb0);
                out[static_cast<size_t>(tok * h * d + head * d + dim)] =
                        *reinterpret_cast<const uint16_t *>(data + off);
            }
        }
    }
}

void llama_memory_kvmem::tensor_to_f32_token_major(const ggml_tensor * t, std::vector<float> & out) {
    std::vector<uint8_t> tmp;
    const uint8_t * data = nullptr;
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        data = static_cast<const uint8_t *>(t->data);
    } else {
        tmp.resize(ggml_nbytes(t));
        kvmem_tensor_get(t, tmp.data(), 0, tmp.size());
        data = tmp.data();
    }
    bytes_to_f32_token_major(data, t->type, t->ne[0], t->ne[1], t->ne[2],
                             t->nb[0], t->nb[1], t->nb[2], out);
}

void llama_memory_kvmem::harvest_from_host(int il, char which, const uint8_t * host,
                                           ggml_type type, int64_t d, int64_t h, int64_t ntok,
                                           size_t nb0, size_t nb1, size_t nb2) {
    if (which == 'v') {
        return;
    }
    if (!host || il < 0 || static_cast<uint32_t>(il) >= n_layer_ || cur_pos_.empty()) {
        return;
    }
    const uint32_t n = static_cast<uint32_t>(cur_pos_.size());
    const uint32_t pos0 = static_cast<uint32_t>(cur_pos_[0]);
    if (which == 'k') {
        std::vector<float> flat;
        if (type == GGML_TYPE_F16) {
            std::vector<uint16_t> flat16;
            bytes_to_f16_token_major(host, type, d, h, ntok, nb0, nb1, nb2, flat16);
            flat.resize(flat16.size());
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(flat16.data()),
                                  flat.data(), static_cast<int64_t>(flat16.size()));
        } else {
            bytes_to_f32_token_major(host, type, d, h, ntok, nb0, nb1, nb2, flat);
        }
        if (!flat.empty()) {
            raw_->write_layer_mean_k(pos0, n, static_cast<uint32_t>(il), flat.data());
        }
    } else if (which == 'q') {
        std::vector<float> flat;
        bytes_to_f32_token_major(host, type, d, h, ntok, nb0, nb1, nb2, flat);
        const uint32_t qdim = n_head_ * n_embd_head_;
        if (flat.size() < static_cast<size_t>(n) * qdim) {
            return;
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (!query_contains(cur_pos_[i])) continue;
            float * dst = q_sum_[static_cast<uint32_t>(il)].data();
            const float * src = flat.data() + i * qdim;
            for (uint32_t d0 = 0; d0 < qdim; ++d0) {
                dst[d0] += src[d0];
            }
            q_count_[static_cast<uint32_t>(il)]++;
        }
    }
}

void llama_memory_kvmem::harvest_capture(ggml_tensor * t, int il, char which) {
    if (which == 'v') {
        return;
    }
    if (!t || il < 0 || static_cast<uint32_t>(il) >= n_layer_) {
        return;
    }
    if (cur_pos_.empty()) {
        return;
    }
    std::vector<float> flat;
    tensor_to_f32_token_major(t, flat);
    const uint32_t n = static_cast<uint32_t>(cur_pos_.size());
    if (n == 0) {
        return;
    }
    const uint32_t pos0 = static_cast<uint32_t>(cur_pos_[0]);

    if (which == 'k') {
        raw_->write_layer_mean_k(pos0, n, static_cast<uint32_t>(il), flat.data());
        static bool dumped = false;
        if (!dumped && getenv("KVMEM_DUMP_CAPTURE") && il == 0) {
            dumped = true;
            double acc = 0;
            for (float x : flat) {
                acc += static_cast<double>(x) * x;
            }
            kvmem_diag("KVMEM_DUMP k_prerope layer0 n=%u rms=%.6f shape_elems=%zu\n",
                    n, std::sqrt(acc / std::max<size_t>(flat.size(), 1)), flat.size());
        }
    } else if (which == 'q') {
        const uint32_t qdim = n_head_ * n_embd_head_;
        if (flat.size() < static_cast<size_t>(n) * qdim) {
            return;
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (!query_contains(cur_pos_[i])) continue;
            float * dst = q_sum_[static_cast<uint32_t>(il)].data();
            const float * src = flat.data() + i * qdim;
            for (uint32_t d = 0; d < qdim; ++d) {
                dst[d] += src[d];
            }
            q_count_[static_cast<uint32_t>(il)]++;
        }
    }
}

void llama_memory_kvmem::harvest_write_batch() {
    HarvestVBatch & b = harvest_v_pending_;
    if (b.jobs.empty()) {
        b.slot = -1;
        return;
    }
    auto fallback = [&](const HarvestVJob & j) {
        if (j.nbytes == 0) {
            return;
        }
        std::vector<uint8_t> packed(j.nbytes);
        auto commit = [&](const uint8_t * src) {
            if (j.is_k) {
                raw_->write_layer_k_gpu(j.pos0, j.n, j.il, src);
            } else {
                raw_->write_layer_v_gpu(j.pos0, j.n, j.il, src);
            }
        };
        if (j.gpu_src &&
            kvmem_copy(packed.data(), j.gpu_src, j.nbytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            commit(packed.data());
            return;
        }
        if (j.vt) {
            kvmem_tensor_get(j.vt, packed.data(), j.tensor_off, j.nbytes);
            commit(packed.data());
        }
    };
    const uint8_t * base = (b.slot >= 0) ? kvmem_stageout_slot_base(b.slot) : nullptr;
    const int64_t t0 = ggml_time_us();
    for (const HarvestVJob & j : b.jobs) {
        const uint8_t * p = (base && j.nbytes) ? base + j.pin_off : nullptr;
        if (p) {
            if (j.is_k) {
                raw_->write_layer_k_gpu(j.pos0, j.n, j.il, p);
            } else {
                raw_->write_layer_v_gpu(j.pos0, j.n, j.il, p);
            }
        } else {
            fallback(j);
        }
    }
    if (retr_.enabled) {
        retr_.stage_out_host_us += ggml_time_us() - t0;
    }
    b.jobs.clear();
    b.slot = -1;
}

void llama_memory_kvmem::harvest_gpu_v_flush_slab() {
    if (harvest_v_jobs_.empty()) {
        return;
    }
    int64_t * gpu_us = retr_.enabled ? &retr_.stage_out_gpu_us : nullptr;
    if (harvest_v_pending_.slot >= 0) {
        if (!kvmem_stageout_wait(harvest_v_pending_.slot, gpu_us)) {
            harvest_v_pending_.slot = -1;
        }
        const int slot = kvmem_stageout_submit(gpu_us);
        HarvestVBatch next;
        next.slot = slot;
        next.jobs = std::move(harvest_v_jobs_);
        harvest_v_jobs_.clear();
        harvest_write_batch();
        harvest_v_pending_ = std::move(next);
        if (harvest_v_pending_.slot < 0) {
            harvest_write_batch();
        }
        return;
    }
    const int slot = kvmem_stageout_submit(gpu_us);
    harvest_v_pending_.slot = slot;
    harvest_v_pending_.jobs = std::move(harvest_v_jobs_);
    harvest_v_jobs_.clear();
    if (harvest_v_pending_.slot < 0) {
        harvest_write_batch();
    }
}

void llama_memory_kvmem::harvest_gpu_v_commit() {
    harvest_gpu_v_flush_slab();
    int64_t * gpu_us = retr_.enabled ? &retr_.stage_out_gpu_us : nullptr;
    if (harvest_v_pending_.slot >= 0) {
        if (!kvmem_stageout_wait(harvest_v_pending_.slot, gpu_us)) {
            harvest_v_pending_.slot = -1;
        }
        harvest_write_batch();
    } else if (!harvest_v_pending_.jobs.empty()) {
        harvest_write_batch();
    }
    kvmem_stageout_clear();
    harvest_gpu_queued_.clear();
}

void llama_memory_kvmem::harvest_gpu_v(uint32_t block_id) {
    if (v_trans_ || !raw_ || !kv_) {
        return;
    }
    if (block_id < harvest_gpu_queued_.size() && harvest_gpu_queued_[block_id]) {
        return;
    }
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    if (idx >= cells.size() || cells.is_empty(idx)) {
        return;
    }
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t nt = blk.n_tokens;
    const uint32_t cell0 = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                            std::max(krow, vrow) * (size_t) block_tokens_);
    auto enqueue_or_get = [&](ggml_tensor * t, uint32_t il, uint32_t nget, bool is_k) {
        const size_t row = is_k ? krow : vrow;
        const size_t nbytes = static_cast<size_t>(nget) * row;
        const size_t toff = static_cast<size_t>(cell0) * row;
        uint8_t * base = kvmem_cuda_tensor_ptr(t);
        const uint8_t * gpu_src = (base && nbytes > 0) ? base + toff : nullptr;
        HarvestVJob job;
        job.pos0 = blk.orig_pos_start;
        job.n = nget;
        job.il = il;
        job.is_k = is_k;
        job.gpu_src = gpu_src;
        job.nbytes = nbytes;
        job.pin_off = kvmem_stageout_used();
        job.vt = t;
        job.tensor_off = toff;
        if (gpu_src && kvmem_stageout_enqueue(gpu_src, nbytes)) {
            harvest_v_jobs_.push_back(job);
            return;
        }
        if (gpu_src && (!harvest_v_jobs_.empty() || harvest_v_pending_.slot >= 0)) {
            harvest_gpu_v_flush_slab();
            job.pin_off = kvmem_stageout_used();
            if (kvmem_stageout_enqueue(gpu_src, nbytes)) {
                harvest_v_jobs_.push_back(job);
                return;
            }
        }
        std::vector<uint8_t> packed(nbytes);
        kvmem_tensor_get(t, packed.data(), toff, nbytes);
        if (is_k) {
            raw_->write_layer_k_gpu(blk.orig_pos_start, nget, il, packed.data());
        } else {
            raw_->write_layer_v_gpu(blk.orig_pos_start, nget, il, packed.data());
        }
    };
    uint32_t n_ok = 0;
    uint32_t n_skip = 0;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        const uint32_t nget = (cell0 < kv_size_)
                ? std::min(nt, kv_size_ - cell0) : 0;
        if (nget == 0) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        if (kt && krow && !raw_->has_k_gpu(block_id, il, nget)) {
            enqueue_or_get(kt, il, nget, true);
            n_ok++;
        } else {
            n_skip++;
        }
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        if (vt && !v_trans_ && vrow && !raw_->has_v_gpu(block_id, il, nget)) {
            enqueue_or_get(vt, il, nget, false);
            n_ok++;
        } else {
            n_skip++;
        }
    }
    if (n_ok > 0) {
        if (harvest_gpu_queued_.size() <= block_id) {
            harvest_gpu_queued_.resize(block_id + 1);
        }
        harvest_gpu_queued_[block_id] = 1;
    }
    if (trace_) {
        kvmem_diag("KVMEM_TRACE harvest_kv block=%u slot=%d n=%u layers=%u skip=%u\n",
                block_id, blk.gpu_slot, blk.n_tokens, n_ok, n_skip);
    }
}

void llama_memory_kvmem::harvest_full_blocks_async() {
    if (retrieval_pinned_ || replay_ || v_trans_ || !raw_ || !kv_ || !runtime_) {
        return;
    }
    auto & store = runtime_->store();
    uint32_t n_enq = 0;
    for (const auto & b : store.blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens != block_tokens_) {
            continue;
        }
        if (b.block_id < harvest_gpu_queued_.size() && harvest_gpu_queued_[b.block_id]) {
            continue;
        }
        bool need = false;
        for (uint32_t il = 0; il < n_layer_; ++il) {
            if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
                continue;
            }
            if (!raw_->has_k_gpu(b.block_id, il, b.n_tokens) ||
                (!v_trans_ && !raw_->has_v_gpu(b.block_id, il, b.n_tokens))) {
                need = true;
                break;
            }
        }
        if (!need) {
            continue;
        }
        harvest_gpu_v(b.block_id);
        n_enq++;
    }
    if (!harvest_v_jobs_.empty()) {
        harvest_gpu_v_flush_slab();
    }
    if (trace_ && n_enq > 0) {
        kvmem_diag("KVMEM_TRACE harvest_full_async n=%u pending_slot=%d\n",
                n_enq, harvest_v_pending_.slot);
    }
}

void llama_memory_kvmem::decode_mean_reset() {
    decode_mean_flush();
    decode_mean_discard();
    decode_mean_n_ = 0;
    decode_mean_block_ = ~0u;
    decode_mean_pos0_ = 0;
}

void llama_memory_kvmem::decode_mean_discard() {
    decode_mean_pending_k_.clear();
    decode_mean_pending_pos_.clear();
}

void llama_memory_kvmem::decode_mean_zero_acc() {
    if (kvmem_meank_ready(n_layer_, n_embd_k_)) {
        for (uint32_t il = 0; il < n_layer_; ++il) {
            kvmem_meank_zero(il);
        }
    }
    decode_mean_host_.assign(n_layer_, std::vector<float>(n_embd_k_, 0.0f));
    decode_mean_src_.assign(n_layer_, 0);
}

void llama_memory_kvmem::decode_mean_print_sum() {
    if (decode_mean_stats_.printed) {
        return;
    }
    decode_mean_stats_.printed = true;
    kvmem_diag("KVMEM_DECODE_MEAN n_tok=%u n_flush=%u n_gpu=%u n_host=%u n_miss=%u "
            "last_block=%d last_n=%u last_layers=%u last_rms=%.6f\n",
            decode_mean_stats_.n_tok, decode_mean_stats_.n_flush,
            decode_mean_stats_.n_gpu, decode_mean_stats_.n_host,
            decode_mean_stats_.n_miss,
            decode_mean_stats_.last_block == ~0u
                    ? -1 : static_cast<int>(decode_mean_stats_.last_block),
            decode_mean_stats_.last_n, decode_mean_stats_.last_layers,
            decode_mean_stats_.last_rms);
}

bool llama_memory_kvmem::decode_mean_add_host_layer(ggml_tensor * t, int il,
                                                   uint32_t tok0, uint32_t n_add) {
    if (!t || il < 0 || static_cast<uint32_t>(il) >= n_layer_ || n_add == 0) {
        return false;
    }
    if (decode_mean_host_.size() != n_layer_) {
        decode_mean_host_.assign(n_layer_, std::vector<float>(n_embd_k_, 0.0f));
        decode_mean_src_.assign(n_layer_, 0);
    }
    const uint32_t uil = static_cast<uint32_t>(il);
    if (decode_mean_src_[uil] == 1) {
        if (decode_mean_host_[uil].size() != n_embd_k_) {
            decode_mean_host_[uil].assign(n_embd_k_, 0.0f);
        }
        kvmem_stagein_sync();
        if (!kvmem_meank_d2h(uil, decode_mean_host_[uil].data(), n_embd_k_)) {
            return false;
        }
        kvmem_stagein_sync();
        kvmem_meank_zero(uil);
        decode_mean_src_[uil] = 2;
    }
    std::vector<float> flat;
    tensor_to_f32_token_major(t, flat);
    const size_t need = (static_cast<size_t>(tok0) + n_add) * n_embd_k_;
    if (flat.size() < need) {
        return false;
    }
    if (decode_mean_host_[uil].size() != n_embd_k_) {
        decode_mean_host_[uil].assign(n_embd_k_, 0.0f);
    }
    const float * src = flat.data() + static_cast<size_t>(tok0) * n_embd_k_;
    for (uint32_t t_i = 0; t_i < n_add; ++t_i) {
        for (uint32_t d = 0; d < n_embd_k_; ++d) {
            decode_mean_host_[uil][d] += src[static_cast<size_t>(t_i) * n_embd_k_ + d];
        }
    }
    decode_mean_src_[uil] = 2;
    decode_mean_stats_.n_host += n_add;
    return true;
}

void llama_memory_kvmem::decode_mean_ingest(ggml_backend_sched_t sched) {
    if (pending_capture_.empty() || pos_queue_.empty()) {
        decode_mean_stats_.n_miss++;
        if (trace_ || decode_mean_stats_.n_miss <= 3) {
            kvmem_diag("KVMEM_DECODE_MEAN ingest_skip capture=%zu posq=%zu "
                    "pinned=%d replay=%d graph_k=%d\n",
                    pending_capture_.size(), pos_queue_.size(),
                    (int) retrieval_pinned_, (int) replay_, (int) graph_has_k_);
        }
        return;
    }
    if (sched) {
        ggml_backend_sched_synchronize(sched);
    }
    decode_mean_pending_pos_ = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    decode_mean_pending_k_.clear();
    for (const CaptureNode & n : pending_capture_) {
        if (n.t && n.which == 'k') {
            decode_mean_pending_k_.push_back(n);
        }
    }
    if (decode_mean_pending_k_.empty()) {
        decode_mean_stats_.n_miss++;
        if (trace_ || decode_mean_stats_.n_miss <= 3) {
            kvmem_diag("KVMEM_DECODE_MEAN ingest_skip no_k_nodes capture=%zu\n",
                    pending_capture_.size());
        }
        decode_mean_pending_pos_.clear();
        return;
    }
    if (trace_ && decode_mean_stats_.n_tok == 0 && !decode_mean_pending_k_.empty()) {
        const CaptureNode & n0 = decode_mean_pending_k_.front();
        kvmem_diag("KVMEM_TRACE decode_mean_ingest n_pos=%zu n_k=%zu type=%s gpu=%d\n",
                decode_mean_pending_pos_.size(), decode_mean_pending_k_.size(),
                n0.t ? ggml_type_name(n0.t->type) : "?",
                n0.t && kvmem_cuda_tensor_ptr(n0.t) ? 1 : 0);
    }
    if (decode_mean_pending_pos_.size() == 1) {
        decode_mean_commit(1);
    }
}

void llama_memory_kvmem::decode_mean_add_range(uint32_t tok0, uint32_t n_add) {
    if (n_add == 0 || decode_mean_pending_pos_.empty() || !raw_ ||
        tok0 >= decode_mean_pending_pos_.size()) {
        return;
    }
    const llama_pos pos0 = decode_mean_pending_pos_[tok0];
    const uint32_t bid = static_cast<uint32_t>(pos0) / block_tokens_;
    if (decode_mean_block_ != bid) {
        if (decode_mean_n_ > 0) {
            decode_mean_flush();
        }
        decode_mean_zero_acc();
        decode_mean_block_ = bid;
        decode_mean_pos0_ = static_cast<uint32_t>(pos0);
        decode_mean_n_ = 0;
    }
    const bool gpu_ready = kvmem_meank_ready(n_layer_, n_embd_k_);
    bool any = false;
    for (const CaptureNode & n : decode_mean_pending_k_) {
        if (!n.t || static_cast<uint32_t>(n.il) >= n_layer_) {
            continue;
        }
        bool ok = false;
        if (gpu_ready) {
            const uint8_t * gpu = kvmem_cuda_tensor_ptr(n.t);
            if (gpu &&
                kvmem_meank_add(static_cast<uint32_t>(n.il), n.t->type, gpu,
                                tok0, n_add, n_embd_k_, n.t->ne[0],
                                n.t->nb[0], n.t->nb[1], n.t->nb[2])) {
                if (decode_mean_src_.size() != n_layer_) {
                    decode_mean_src_.assign(n_layer_, 0);
                }
                decode_mean_src_[static_cast<uint32_t>(n.il)] = 1;
                decode_mean_stats_.n_gpu += n_add;
                ok = true;
            }
        }
        if (!ok) {
            ok = decode_mean_add_host_layer(n.t, n.il, tok0, n_add);
        }
        any = any || ok;
    }
    if (!any) {
        decode_mean_stats_.n_miss++;
        if (trace_ || decode_mean_stats_.n_miss <= 3) {
            kvmem_diag("KVMEM_DECODE_MEAN add_fail n_add=%u n_k=%zu gpu_ready=%d\n",
                    n_add, decode_mean_pending_k_.size(), (int) gpu_ready);
        }
        return;
    }
    decode_mean_n_ += n_add;
    decode_mean_stats_.n_tok += n_add;
}

void llama_memory_kvmem::decode_mean_commit(uint32_t n_keep) {
    if (n_keep == 0 || decode_mean_pending_pos_.empty()) {
        decode_mean_discard();
        return;
    }
    n_keep = std::min(n_keep, static_cast<uint32_t>(decode_mean_pending_pos_.size()));
    uint32_t i = 0;
    while (i < n_keep) {
        const uint32_t pos = static_cast<uint32_t>(decode_mean_pending_pos_[i]);
        const uint32_t bid = pos / block_tokens_;
        uint32_t take = 1;
        while (i + take < n_keep) {
            const uint32_t p = static_cast<uint32_t>(decode_mean_pending_pos_[i + take]);
            if (p / block_tokens_ != bid) {
                break;
            }
            take++;
        }
        decode_mean_add_range(i, take);
        // Accepted tokens only. Store n_tokens can include MTP drafts that
        // seq_rm will drop; do not flush on that watermark.
        if (decode_mean_block_ == bid && decode_mean_n_ > 0 &&
            (decode_mean_pos0_ % block_tokens_) + decode_mean_n_ >= block_tokens_) {
            decode_mean_flush();
        }
        i += take;
    }
    decode_mean_discard();
}

void llama_memory_kvmem::decode_mean_flush() {
    if (decode_mean_n_ == 0 || !raw_ || decode_mean_block_ == ~0u) {
        return;
    }
    kvmem_stagein_sync();
    std::vector<float> sum(n_embd_k_, 0.0f);
    uint32_t n_ok = 0;
    uint32_t n_gpu = 0;
    uint32_t n_host = 0;
    float rms = 0.0f;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        const uint8_t src = (il < decode_mean_src_.size()) ? decode_mean_src_[il] : 0;
        bool got = false;
        if (src == 2 && il < decode_mean_host_.size() &&
            decode_mean_host_[il].size() == n_embd_k_) {
            std::memcpy(sum.data(), decode_mean_host_[il].data(),
                        n_embd_k_ * sizeof(float));
            got = true;
            n_host++;
        } else if (src == 1 && kvmem_meank_d2h(il, sum.data(), n_embd_k_)) {
            kvmem_stagein_sync();
            got = true;
            n_gpu++;
        }
        if (!got) {
            continue;
        }
        raw_->write_layer_mean_sum(decode_mean_pos0_, decode_mean_n_, il, sum.data());
        kvmem_meank_zero(il);
        if (n_ok == 0 && n_embd_k_ > 0) {
            double acc = 0;
            for (uint32_t d = 0; d < n_embd_k_; ++d) {
                acc += static_cast<double>(sum[d]) * sum[d];
            }
            rms = std::sqrt(acc / static_cast<double>(n_embd_k_)) /
                    static_cast<float>(std::max(1u, decode_mean_n_));
        }
        n_ok++;
    }
    kvmem_diag("KVMEM_DECODE_MEAN flush block=%u n=%u pos0=%u layers=%u gpu=%u host=%u rms=%.6f\n",
            decode_mean_block_, decode_mean_n_, decode_mean_pos0_, n_ok, n_gpu, n_host, rms);
    if (trace_) {
        kvmem_diag("KVMEM_TRACE decode_mean_flush block=%u n=%u pos0=%u layers=%u\n",
                decode_mean_block_, decode_mean_n_, decode_mean_pos0_, n_ok);
    }
    decode_mean_stats_.n_flush++;
    decode_mean_stats_.last_block = decode_mean_block_;
    decode_mean_stats_.last_n = decode_mean_n_;
    decode_mean_stats_.last_layers = n_ok;
    decode_mean_stats_.last_rms = rms;
    decode_mean_n_ = 0;
    decode_mean_block_ = ~0u;
    decode_mean_src_.assign(n_layer_, 0);
    for (auto & h : decode_mean_host_) {
        std::fill(h.begin(), h.end(), 0.0f);
    }
}

void llama_memory_kvmem::write_block_to_gpu(uint32_t block_id) {
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || !raw_->has_block(block_id)) {
        return;
    }
    ++attention_epoch_;
    occupy_block_cells(block_id);
    const uint32_t nt = blk.n_tokens;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    std::vector<uint8_t> kpack(static_cast<size_t>(nt) * krow);
    std::vector<uint8_t> vpack(static_cast<size_t>(nt) * vrow);
    const uint32_t cell0 = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                            std::max(krow, vrow) * (size_t) block_tokens_);
    int64_t * sacc = retr_.enabled ? &retr_.set_us : nullptr;
    auto write_packed = [&](ggml_tensor * t, uint8_t * base, const uint8_t * host,
                            size_t row, size_t nbytes) {
        if (!t || !host || nbytes == 0 || cell0 >= kv_size_) {
            return;
        }
        uint8_t * dst = base ? base + static_cast<size_t>(cell0) * row : nullptr;
        if (dst && kvmem_stagein_enqueue_v(host, nbytes, dst, sacc)) {
            return;
        }
        const int64_t t_set = ggml_time_us();
        if (!(dst && kvmem_stagein_h2d_bytes(dst, host, nbytes))) {
            kvmem_tensor_set(t, host, static_cast<size_t>(cell0) * row, nbytes);
        }
        if (retr_.enabled) {
            retr_.set_us += ggml_time_us() - t_set;
        }
    };
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        const int64_t t_copy = ggml_time_us();
        const bool have_k = raw_->copy_k_gpu(block_id, il, kpack.data(), nt);
        const bool have_v = !v_trans_ && raw_->copy_v_gpu(block_id, il, vpack.data(), nt);
        if (retr_.enabled) {
            retr_.copy_us += ggml_time_us() - t_copy;
        }
        if (!have_k && !have_v) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        uint8_t * kbase = kvmem_cuda_tensor_ptr(kt);
        uint8_t * vbase = (vt && !v_trans_) ? kvmem_cuda_tensor_ptr(vt) : nullptr;
        if (have_k) {
            write_packed(kt, kbase, kpack.data(), krow, krow * (size_t) nt);
        }
        if (have_v) {
            write_packed(vt, vbase, vpack.data(), vrow, vrow * (size_t) nt);
        }
    }
    if (trace_) {
        kvmem_diag("KVMEM_TRACE stage_in_packed block=%u slot=%d n=%u orig=%u\n",
                block_id, blk.gpu_slot, nt, blk.orig_pos_start);
    }
}

void llama_memory_kvmem::copy_gpu_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                                void * host, uint64_t bytes) {
    if (!host || !kv_ || gpu_slot < 0 || bytes == 0) {
        return;
    }
    (void) block_id;
    auto * dst = static_cast<uint8_t *>(host);
    uint64_t off = 0;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = static_cast<uint32_t>(gpu_slot) * block_tokens_;
    const uint32_t nget = (cell0 < kv_size_)
            ? std::min(block_tokens_, kv_size_ - cell0) : 0;
    const uint64_t kspan = static_cast<uint64_t>(block_tokens_) * krow;
    const uint64_t vspan = static_cast<uint64_t>(block_tokens_) * vrow;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        if (off + kspan > bytes) {
            return;
        }
        if (kt && nget > 0) {
            kvmem_tensor_get(kt, dst + off, cell0 * krow, nget * krow);
        }
        off += kspan;
        if (off + vspan > bytes) {
            return;
        }
        if (vt && !v_trans_ && nget > 0) {
            kvmem_tensor_get(vt, dst + off, cell0 * vrow, nget * vrow);
        }
        off += vspan;
    }
}

void llama_memory_kvmem::copy_gpu_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                                  const void * host, uint64_t bytes) {
    if (!host || !kv_ || gpu_slot < 0 || bytes == 0) {
        return;
    }
    (void) block_id;
    const auto * src = static_cast<const uint8_t *>(host);
    uint64_t off = 0;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = static_cast<uint32_t>(gpu_slot) * block_tokens_;
    const uint32_t nset = (cell0 < kv_size_)
            ? std::min(block_tokens_, kv_size_ - cell0) : 0;
    const uint64_t kspan = static_cast<uint64_t>(block_tokens_) * krow;
    const uint64_t vspan = static_cast<uint64_t>(block_tokens_) * vrow;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        if (off + kspan > bytes) {
            return;
        }
        if (kt && nset > 0) {
            kvmem_tensor_set(kt, src + off, cell0 * krow, nset * krow);
        }
        off += kspan;
        if (off + vspan > bytes) {
            return;
        }
        if (vt && !v_trans_ && nset > 0) {
            kvmem_tensor_set(vt, src + off, cell0 * vrow, nset * vrow);
        }
        off += vspan;
    }
}

void llama_memory_kvmem::score_retrieval() {
    auto & store = runtime_->store();
    const uint32_t nblk = store.block_count();
    std::vector<double> scores(nblk, 0.0);
    std::vector<float> mk(n_embd_k_, 0.0f);
    const uint32_t g = n_head_kv_ == 0 ? 1 : n_head_ / n_head_kv_;
    for (uint32_t b = 0; b < nblk; ++b) {
        if (!raw_->has_block(b)) {
            continue;
        }
        double acc = 0;
        uint32_t nlay = 0;
        for (uint32_t il = 0; il < n_layer_; ++il) {
            if (q_count_[il] == 0) {
                continue;
            }
            raw_->mean_k(b, il, mk.data());
            double layer = 0;
            const float invq = 1.0f / static_cast<float>(q_count_[il] * std::max(1u, g));
            for (uint32_t h = 0; h < n_head_kv_; ++h) {
                std::vector<float> qh(n_embd_head_, 0.0f);
                for (uint32_t gi = 0; gi < g; ++gi) {
                    const uint32_t qh_i = h * g + gi;
                    if (qh_i >= n_head_) {
                        break;
                    }
                    const float * q = q_sum_[il].data() + qh_i * n_embd_head_;
                    for (uint32_t d = 0; d < n_embd_head_; ++d) {
                        qh[d] += q[d];
                    }
                }
                const float * kh = mk.data() + h * n_embd_head_;
                float dot = 0, nq = 0, nk = 0;
                for (uint32_t d = 0; d < n_embd_head_; ++d) {
                    const float qv = qh[d] * invq;
                    dot += qv * kh[d];
                    nq += qv * qv;
                    nk += kh[d] * kh[d];
                }
                layer += dot / (std::sqrt(nq) * std::sqrt(nk) + 1e-6f);
            }
            acc += layer / static_cast<double>(std::max(1u, n_head_kv_));
            nlay++;
        }
        scores[b] = nlay ? acc / nlay : 0;
    }
    store.set_retrieval_scores(scores);
}

void llama_memory_kvmem::retr_perf_print() {
    if (!retr_.enabled) {
        return;
    }
    kvmem_diag("KVMEM_RETR_SUM total_ms=%.3f flush_ms=%.3f score_ms=%.3f plan_ms=%.3f "
            "stage_out_ms=%.3f stage_out_gpu_ms=%.3f stage_out_host_ms=%.3f "
            "admit_ms=%.3f seq_rm_ms=%.3f occupy_ms=%.3f "
            "layout_d2h_ms=%.3f layout_h2d_ms=%.3f "
            "copy_ms=%.3f rope_ms=%.3f hadamard_ms=%.3f set_ms=%.3f mtp_ms=%.3f dump_ms=%.3f "
            "n_move=%u n_raw=%u n_skip=%u n_stage_in=%u laid_out=%d\n",
            retr_.total_us / 1000.0, retr_.flush_us / 1000.0, retr_.score_us / 1000.0,
            retr_.plan_us / 1000.0, retr_.stage_out_us / 1000.0,
            retr_.stage_out_gpu_us / 1000.0, retr_.stage_out_host_us / 1000.0,
            retr_.admit_us / 1000.0, retr_.seq_rm_us / 1000.0, retr_.occupy_us / 1000.0,
            retr_.layout_d2h_us / 1000.0, retr_.layout_h2d_us / 1000.0,
            retr_.copy_us / 1000.0, retr_.rope_us / 1000.0, retr_.hadamard_us / 1000.0,
            retr_.set_us / 1000.0, retr_.mtp_us / 1000.0, retr_.dump_us / 1000.0,
            retr_.n_move, retr_.n_raw, retr_.n_skip, retr_.n_stage_in, retr_.laid_out);
}

void llama_memory_kvmem::set_turn_spans(const llama_kvmem_turn_spans & spans) {
    for (const auto & ranges : {spans.query, spans.mandatory}) {
        for (const auto & r : ranges) {
            if (r.begin < 0 || r.end < r.begin) throw std::invalid_argument("invalid KVMem row range");
        }
    }
    if (spans.replay_begin < 0) throw std::invalid_argument("invalid KVMem replay start");
    harvest_flush();
    turn_spans_ = spans;
    explicit_spans_ = true;
}

bool llama_memory_kvmem::query_contains(llama_pos row) const {
    if (query_frozen_) return false;
    if (!explicit_spans_) return query_begin_ >= 0 && row >= query_begin_ && (query_end_ <= 0 || row < query_end_);
    for (const auto & r : turn_spans_.query) if (row >= r.begin && row < r.end) return true;
    return false;
}

bool llama_memory_kvmem::query_overlaps(uint32_t n, const llama_pos * rows) const {
    if (!want_q_capture() || !n) return false;
    if (!rows) return true;
    for (uint32_t i = 0; i < n; ++i) if (query_contains(rows[i])) return true;
    return false;
}

bool llama_memory_kvmem::gpu_kv_complete(uint32_t id, const llama_kv_cache * cache) const {
    if (!cache || id >= store().block_count()) return false;
    const auto & b = store().blocks()[id];
    if (b.gpu_slot < 0 || b.tier != kvmem::KvTier::GPU || b.in_flight || !b.n_tokens) return false;
    const auto & cells = cache->get_cells(0);
    const uint64_t first = (uint64_t) b.gpu_slot * block_tokens_;
    if (first + b.n_tokens > cells.size() || b.orig_pos_end() > row_positions_.size()) return false;
    for (uint32_t i = 0; i < b.n_tokens; ++i) {
        const uint32_t cell = first + i, row = b.orig_pos_start + i;
        if (cells.is_empty(cell) || !cells.seq_has(cell, 0)) return false;
        const auto & ext = cells.ext_get(cell);
        const auto & pos = row_positions_[row].pos;
        if (ext.logical_pos != (llama_pos) row || cells.pos_get(cell) != pos[0] ||
                (row_positions_[row].spatial && (ext.y != pos[1] || ext.x != pos[2]))) return false;
    }
    return true;
}

llama_kvmem_attention_view llama_memory_kvmem::attention_view(bool canonical) const {
    llama_kvmem_attention_view view;
    view.epoch = attention_epoch_;
    view.rows = store_n_tokens();
    view.valid = kv_ && runtime_ && store().config().optimize_stage_in;
    uint32_t resident_rows = 0;
    for (const auto & b : store().blocks()) {
        if (b.gpu_slot < 0 || !b.n_tokens) continue;
        resident_rows += b.n_tokens;
        if ((canonical && b.gpu_slot != (int32_t) view.blocks.size()) || !gpu_kv_complete(b.block_id, kv_) ||
                (mtp_ && !gpu_kv_complete(b.block_id, mtp_->get_kv()))) view.valid = false;
        view.blocks.push_back(b.block_id);
    }
    // Extra live cells (for example an untrimmed speculative tail) are also
    // part of attention. A proof must account for them, not just known blocks.
    if (kv_ && kv_->get_cells(0).get_used() != resident_rows) view.valid = false;
    if (mtp_ && mtp_->get_kv()->get_cells(0).get_used() != resident_rows) view.valid = false;
    return view;
}

bool llama_memory_kvmem::can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason) const {
    auto fail = [&](const char * why) { reason = why; return false; };
    if (!runtime_ || !kv_ || end < store_n_tokens()) return fail("invalid_boundary");
    const auto view = attention_view(all_history);
    if (!view.valid) return fail("coverage_or_layout");
    const auto & s = store();
    const uint64_t end_blocks = ((uint64_t) end + block_tokens_ - 1) / block_tokens_;
    if (view.rows % block_tokens_ && !gpu_kv_complete(view.rows / block_tokens_, kv_)) return fail("partial_tail_missing");
    if (all_history) {
        if (view.blocks.size() != s.block_count()) return fail("history_not_resident");
        if (s.budget_blocks() && end_blocks > s.budget_blocks()) return fail("selection_budget");
        if (s.prefill_needs_offload(resident_tokens(), end - view.rows, kv_size_)) return fail("prefill_pressure");
    }
    const uint64_t final_blocks = ((uint64_t) end + generation_rows + block_tokens_ - 1) / block_tokens_;
    if (final_blocks > n_slots_ + (uint64_t) s.block_count()) return fail("insufficient_slots");
    uint64_t needed = 0;
    for (uint64_t id = view.rows / block_tokens_; id < final_blocks; ++id) {
        if (id >= s.block_count() || s.blocks()[id].gpu_slot < 0) ++needed;
    }
    if (needed > free_slots_.size()) return fail("insufficient_slots");
    reason = all_history ? "all_resident" : "same_query";
    return true;
}

std::vector<uint32_t> llama_memory_kvmem::retrieval_mandatory() const {
    const auto & s = store();
    std::vector<llama_kvmem_row_range> ranges;
    if (explicit_spans_) ranges = turn_spans_.mandatory;
    else if (query_begin_ >= 0) ranges.push_back({query_begin_, query_end_ > 0 ? query_end_ : (int32_t) s.total_tokens()});
    std::vector<uint32_t> mandatory;
    for (const auto & b : s.blocks()) {
        for (const auto & r : ranges) {
            if ((int64_t) b.orig_pos_end() > r.begin && (int64_t) b.orig_pos_start < r.end) {
                mandatory.push_back(b.block_id);
                break;
            }
        }
    }
    if (force_pos_ >= 0) {
        const int32_t id = s.block_id_containing(force_pos_);
        if (id >= 0) mandatory.push_back(id);
    }
    return mandatory;
}

llama_kvmem_selection llama_memory_kvmem::preview_retrieval() {
    const int64_t t0 = ggml_time_us();
    const bool enabled = retr_.enabled;
    retr_ = RetrPerf{};
    retr_.enabled = enabled;
    harvest_flush();
    const int64_t t1 = ggml_time_us();
    if (method_ == 1) score_retrieval();
    const int64_t t2 = ggml_time_us();
    llama_kvmem_selection selection;
    selection.epoch = attention_epoch_;
    selection.rows = store_n_tokens();
    selection.blocks = runtime_->preview_reselect(retrieval_mandatory());
    if (enabled) {
        retr_.flush_us = t1 - t0;
        retr_.score_us = t2 - t1;
        retr_.plan_us = ggml_time_us() - t2;
        retr_.total_us = ggml_time_us() - t0;
    }
    return selection;
}

bool llama_memory_kvmem::selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t generation_rows) const {
    if (selection.epoch != attention_epoch_ || selection.rows != store_n_tokens() || end < selection.rows) return false;
    // The selector can trim mandatory ranges to its budget. Such a plan is
    // unsuitable for continuation even if it fits the physical slot pool.
    for (uint32_t id : retrieval_mandatory()) {
        if (!std::binary_search(selection.blocks.begin(), selection.blocks.end(), id)) return false;
    }
    const uint64_t final = ((uint64_t) end + generation_rows + block_tokens_ - 1) / block_tokens_;
    const uint64_t first = selection.rows / block_tokens_;
    if (final < first || final - first > n_slots_) return false;
    uint64_t count = selection.blocks.size();
    for (uint64_t id = first; id < final; ++id) {
        if (!std::binary_search(selection.blocks.begin(), selection.blocks.end(), (uint32_t) id)) ++count;
    }
    return count <= n_slots_;
}

bool llama_memory_kvmem::commit_unchanged(const llama_kvmem_attention_view & before,
                                        const llama_kvmem_selection & selection) {
    if (!before.valid || before.epoch != attention_epoch_ || before.rows > store_n_tokens() ||
            selection.epoch != attention_epoch_ || selection.rows != store_n_tokens()) return false;
    const auto after = attention_view();
    if (!after.valid || after.blocks != selection.blocks) return false;
    for (uint32_t id : before.blocks) {
        if (!std::binary_search(after.blocks.begin(), after.blocks.end(), id)) return false;
    }
    return runtime_->commit_resident_selection(selection.blocks);
}

bool llama_memory_kvmem::commit_resident(bool canonical) {
    harvest_flush();
    const auto view = attention_view(canonical);
    return view.valid && runtime_->commit_resident_selection(view.blocks);
}

bool llama_memory_kvmem::get_query(llama_kvmem_query_state & state) {
    harvest_flush();
    bool any = false;
    uint32_t rows = 0;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, il)) continue;
        if (!q_count_[il] || (rows && rows != q_count_[il])) return false;
        rows = q_count_[il];
        any = true;
    }
    if (!any) return false;
    state.sum = q_sum_;
    state.count = q_count_;
    return true;
}

bool llama_memory_kvmem::set_query(const llama_kvmem_query_state & state) {
    if (state.sum.size() != n_layer_ || state.count.size() != n_layer_) return false;
    uint32_t rows = 0;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        const auto & v = state.sum[il];
        if (v.size() != n_head_ * n_embd_head_ ||
                !std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); })) return false;
        if (!kvmem_cache_has_layer(kv_, il)) continue;
        if (!state.count[il] || (rows && rows != state.count[il])) return false;
        rows = state.count[il];
    }
    if (!rows) return false;
    harvest_flush();
    q_sum_ = state.sum;
    q_count_ = state.count;
    query_frozen_ = true;
    return true;
}

void llama_memory_kvmem::apply_retrieval() {
    if (method_ != 1) { harvest_flush(); return; }
    apply_selection(preview_retrieval());
}

void llama_memory_kvmem::apply_selection(const llama_kvmem_selection & selection) {
    if (selection.epoch != attention_epoch_ || selection.rows != store_n_tokens()) {
        throw std::runtime_error("stale KVMem selection");
    }
    const int64_t t_all = ggml_time_us();
    kvmem::KvMemPlan plan;
    {
        const int64_t t0 = ggml_time_us();
        plan = runtime_->prepare_selection(selection.blocks);
        if (retr_.enabled) {
            retr_.plan_us += ggml_time_us() - t0;
            retr_.n_stage_in = static_cast<uint32_t>(plan.stage_in.size());
            for (const auto & r : plan.remaps) {
                if (r.skip) {
                    retr_.n_skip++;
                }
            }
        }
    }
    trace_plan("retrieval", plan);
    if (trace_) {
        kvmem_diag("KVMEM_TRACE selected");
        for (const auto & r : plan.remaps) {
            fprintf(stderr, " %u", r.block_id);
        }
        fprintf(stderr, "\n");
    }
    apply_plan_to_kv(plan);
    auto & store = runtime_->store();
    retrieval_pinned_ = true;
    if (mtp_) {
        const int64_t t0 = ggml_time_us();
        mtp_->harvest_resident_v();
        if (retr_.enabled) {
            retr_.mtp_us += ggml_time_us() - t0;
        }
    }
    // Native GPU KV of a reused block at orig_pos != 0 is the ground truth for
    // host RoPE + capture layout. Prefer a block that is NOT in stage_in so the
    // slot still holds prefill bytes. Must run before layout rewrite.
    if (trace_) {
        const int64_t t_dump = ggml_time_us();
        std::vector<bool> is_stage_in(store.block_count(), false);
        for (uint32_t id : plan.stage_in) {
            if (id < is_stage_in.size()) {
                is_stage_in[id] = true;
            }
        }
        for (const auto & r : plan.remaps) {
            if (r.block_id >= store.block_count() || is_stage_in[r.block_id]) {
                continue;
            }
            const kvmem::KvMemBlock & b = store.blocks()[r.block_id];
            if (b.gpu_slot < 0 || b.orig_pos_start == 0 || !raw_->has_block(r.block_id)) {
                continue;
            }
            kvmem_diag("KVMEM_KV --- native reused block %u orig=%u skip=%d (GPU vs packed) ---\n",
                    r.block_id, b.orig_pos_start, (int) r.skip);
            dump_kv_compare(static_cast<int32_t>(r.block_id), false);
            break;
        }
        if (retr_.enabled) {
            retr_.dump_us += ggml_time_us() - t_dump;
        }
    }
    const bool laid_out = layout_gpu_slots_by_orig_pos();
    uint32_t n_raw = 0;
    uint32_t n_skip = 0;
    if (!laid_out) {
        for (uint32_t id : plan.stage_in) {
            if (id >= store.block_count() || store.blocks()[id].gpu_slot < 0) {
                continue;
            }
            if (gpu_kv_already_resident(id)) {
                n_skip++;
                continue;
            }
            write_block_to_gpu(id);
            n_raw++;
        }
        kvmem_stagein_flush_sync(retr_.enabled ? &retr_.copy_us : nullptr,
                                 retr_.enabled ? &retr_.rope_us : nullptr,
                                 retr_.enabled ? &retr_.hadamard_us : nullptr,
                                 retr_.enabled ? &retr_.set_us : nullptr);
        if (retr_.enabled) {
            retr_.n_raw += n_raw;
            retr_.n_skip += n_skip;
            retr_.laid_out = 0;
        }
    }
    if (trace_) {
        kvmem_diag("KVMEM_TRACE writeback laid_out=%d raw=%u skip_resident=%u stage_in=%zu\n",
                (int) laid_out, n_raw, n_skip, plan.stage_in.size());
    }
    if (trace_) {
        const int64_t t_dump = ggml_time_us();
        for (uint32_t id : plan.stage_in) {
            if (id < store.block_count() && store.blocks()[id].gpu_slot >= 0) {
                kvmem_diag("KVMEM_KV --- after stage_in_packed block %u ---\n", id);
                dump_kv_compare(static_cast<int32_t>(id), false);
                break;
            }
        }
        trace_working_set("after_retrieval");
        if (retr_.enabled) {
            retr_.dump_us += ggml_time_us() - t_dump;
        }
    }
    if (mtp_) {
        const int64_t t0 = ggml_time_us();
        mtp_->follow_retrieval();
        if (retr_.enabled) {
            retr_.mtp_us += ggml_time_us() - t0;
        }
    }
    if (retr_.enabled) {
        retr_.total_us += ggml_time_us() - t_all;
        retr_perf_print();
    }
}

void llama_memory_kvmem::trace_working_set(const char * tag) const {
    if (!kv_ || !runtime_) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    kvmem_diag("KVMEM_TRACE %s cells used=%u used_max_p1=%u size=%u seq_pos=[%d,%d]\n",
            tag,
            cells.get_used(), cells.used_max_p1(), cells.size(),
            kv_->seq_pos_min(0), kv_->seq_pos_max(0));
    if (recr_) {
        kvmem_diag("KVMEM_TRACE %s recr_seq_pos=[%d,%d]\n",
                tag, recr_->seq_pos_min(0), recr_->seq_pos_max(0));
    }
    const auto & store = runtime_->store();
    for (const auto & b : store.blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        const uint32_t idx = static_cast<uint32_t>(b.gpu_slot) * block_tokens_;
        const bool empty = idx >= cells.size() || cells.is_empty(idx);
        llama_pos p_last = -1;
        bool last_empty = true;
        if (!empty && b.n_tokens > 0) {
            const uint32_t last = idx + b.n_tokens - 1;
            last_empty = last >= cells.size() || cells.is_empty(last);
            p_last = last_empty ? -1 : cells.pos_get(last);
        }
        kvmem_diag("KVMEM_TRACE occupy tag=%s block=%u slot=%d cell=%u empty=%d pos=%d last_empty=%d last_pos=%d n=%u orig=%u\n",
                tag, b.block_id, b.gpu_slot, idx, (int) empty,
                empty ? -1 : (int) cells.pos_get(idx),
                (int) last_empty, (int) p_last, b.n_tokens, b.orig_pos_start);
    }
}

void llama_memory_kvmem::kv_stats(const char * tag, const float * a, const float * b, size_t n) {
    if (!a || !b || n == 0) {
        kvmem_diag("KVMEM_KV %s missing n=%zu\n", tag, n);
        return;
    }
    double dot = 0, na = 0, nb = 0, se = 0, maxabs = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i];
        const double db = b[i];
        dot += da * db;
        na += da * da;
        nb += db * db;
        const double e = da - db;
        se += e * e;
        maxabs = std::max(maxabs, std::abs(e));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    const double rmse = std::sqrt(se / static_cast<double>(n));
    const double rms_a = std::sqrt(na / static_cast<double>(n));
    kvmem_diag("KVMEM_KV %s n=%zu cos=%.6f rmse=%.6f maxabs=%.6f rms_gpu=%.6f\n",
            tag, n, cos, rmse, maxabs, rms_a);
}

bool llama_memory_kvmem::read_gpu_block(uint32_t block_id, uint32_t il, bool is_k,
                                        std::vector<float> & out) const {
    const auto & store = runtime_->store();
    if (block_id >= store.block_count() || il >= n_layer_ ||
        !kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
        return false;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0) {
        return false;
    }
    ggml_tensor * t = is_k ? kv_->get_k_storage(static_cast<int32_t>(il))
                           : kv_->get_v_storage(static_cast<int32_t>(il));
    if (!t) {
        return false;
    }
    const uint32_t dim = is_k ? n_embd_k_ : n_embd_v_;
    const ggml_type ty = is_k ? type_k_ : type_v_;
    const size_t row = ggml_row_size(ty, dim);
    const uint32_t nt = blk.n_tokens;
    const uint32_t cell0 = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    out.assign(static_cast<size_t>(nt) * dim, 0.0f);
    if (nt == 0 || cell0 >= kv_size_) {
        return false;
    }
    const uint32_t nget = std::min(nt, kv_size_ - cell0);
    std::vector<uint8_t> raw(static_cast<size_t>(nget) * row);
    kvmem_tensor_get(t, raw.data(), static_cast<size_t>(cell0) * row,
                            static_cast<size_t>(nget) * row);
    if (!kvmem_cache_unpack_rows(ty, raw.data(), out.data(),
                                 static_cast<int64_t>(nget),
                                 static_cast<int64_t>(dim))) {
        return false;
    }
    if (kvmem_attn_rot_on(ty, static_cast<int>(n_embd_head_))) {
        const int nrot = is_k
                ? kvmem_hadamard_nrot_k(static_cast<int>(n_embd_head_))
                : kvmem_hadamard_nrot_v(static_cast<int>(n_embd_head_));
        kvmem_hadamard_rows(out.data(), static_cast<int64_t>(nget),
                            static_cast<int>(n_head_kv_), static_cast<int>(n_embd_head_),
                            nrot);
    }
    return true;
}

void llama_memory_kvmem::dump_kv_compare(int32_t block_id, bool writeback_test) {
    auto & store = runtime_->store();
    if (block_id < 0 && force_pos_ >= 0) {
        block_id = store.block_id_containing(static_cast<uint32_t>(force_pos_));
    }
    if (block_id < 0) {
        block_id = 0;
    }
    const uint32_t bid = static_cast<uint32_t>(block_id);
    if (bid >= store.block_count()) {
        kvmem_diag("KVMEM_KV dump: no block %d\n", block_id);
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[bid];
    bool any_pk = false;
    bool any_pv = false;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        any_pk = any_pk || raw_->has_k_gpu(bid, il);
        any_pv = any_pv || raw_->has_v_gpu(bid, il);
    }
    kvmem_diag("KVMEM_KV dump block=%u slot=%d n=%u orig=%u packed_k=%d packed_v=%d v_trans=%d type_k=%s pos0=%d\n",
            bid, blk.gpu_slot, blk.n_tokens, blk.orig_pos_start,
            (int) any_pk, (int) any_pv,
            (int) v_trans_, ggml_type_name(type_k_),
            static_cast<int>(blk.orig_pos_start));

    const uint32_t nt = blk.n_tokens;
    const size_t nk = static_cast<size_t>(nt) * n_embd_k_;
    const size_t nv = static_cast<size_t>(nt) * n_embd_v_;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = (blk.gpu_slot >= 0)
            ? static_cast<uint32_t>(blk.gpu_slot) * block_tokens_ : 0;
    std::vector<float> gpu_k, gpu_v, packed_f;

    auto unpack_packed = [&](ggml_type ty, const uint8_t * packed, uint32_t ntok,
                             uint32_t dim, bool is_k, std::vector<float> & out) -> bool {
        out.assign(static_cast<size_t>(ntok) * dim, 0.0f);
        if (!kvmem_cache_unpack_rows(ty, packed, out.data(),
                                     static_cast<int64_t>(ntok),
                                     static_cast<int64_t>(dim))) {
            return false;
        }
        if (kvmem_attn_rot_on(ty, static_cast<int>(n_embd_head_))) {
            const int nrot = is_k
                    ? kvmem_hadamard_nrot_k(static_cast<int>(n_embd_head_))
                    : kvmem_hadamard_nrot_v(static_cast<int>(n_embd_head_));
            kvmem_hadamard_rows(out.data(), static_cast<int64_t>(ntok),
                                static_cast<int>(n_head_kv_),
                                static_cast<int>(n_embd_head_), nrot);
        }
        return true;
    };

    const uint32_t layers_show[] = {0, n_layer_ / 2, n_layer_ > 0 ? n_layer_ - 1 : 0};
    for (uint32_t li = 0; li < 3; ++li) {
        const uint32_t il = layers_show[li];
        if (il >= n_layer_ || !kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        if (!read_gpu_block(bid, il, true, gpu_k)) {
            kvmem_diag("KVMEM_KV L%u K gpu read fail\n", il);
            continue;
        }
        char tag[64];
        std::vector<uint8_t> kpack(static_cast<size_t>(nt) * krow);
        if (raw_->copy_k_gpu(bid, il, kpack.data(), nt)) {
            ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
            size_t nmis = 0;
            if (kt && cell0 < kv_size_) {
                std::vector<uint8_t> kgpu(kpack.size());
                kvmem_tensor_get(kt, kgpu.data(),
                                        static_cast<size_t>(cell0) * krow,
                                        kpack.size());
                for (size_t i = 0; i < kpack.size(); ++i) {
                    nmis += (kpack[i] != kgpu[i]);
                }
            }
            kvmem_diag("KVMEM_KV L%u K packed_bytes mismatch=%zu / %zu\n",
                    il, nmis, kpack.size());
            if (unpack_packed(type_k_, kpack.data(), nt, n_embd_k_, true, packed_f)) {
                snprintf(tag, sizeof(tag), "L%u K packed_vs_gpu", il);
                kv_stats(tag, gpu_k.data(), packed_f.data(), nk);
            }
        } else {
            kvmem_diag("KVMEM_KV L%u no packed K\n", il);
        }
        if (v_trans_) {
            continue;
        }
        std::vector<uint8_t> vpack(static_cast<size_t>(nt) * vrow);
        if (raw_->copy_v_gpu(bid, il, vpack.data(), nt) &&
            read_gpu_block(bid, il, false, gpu_v) &&
            unpack_packed(type_v_, vpack.data(), nt, n_embd_v_, false, packed_f)) {
            snprintf(tag, sizeof(tag), "L%u V packed_vs_gpu", il);
            kv_stats(tag, gpu_v.data(), packed_f.data(), nv);
        }
    }

    uint32_t il_wb = 0;
    while (il_wb < n_layer_ &&
           (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il_wb)) ||
            !raw_->has_k_gpu(bid, il_wb))) {
        il_wb++;
    }
    if (writeback_test && il_wb < n_layer_ && read_gpu_block(bid, il_wb, true, gpu_k)) {
        std::vector<float> before = gpu_k;
        write_block_to_gpu(bid);
        kvmem_stagein_flush_sync(nullptr, nullptr, nullptr, nullptr);
        std::vector<float> after;
        if (read_gpu_block(bid, il_wb, true, after)) {
            kv_stats("K gpu_after_writeback_vs_gpu_before", before.data(), after.data(),
                     before.size());
        }
    }
}

void llama_kvmem_dump_kv_compare(struct llama_context * /*ctx*/, int32_t block_id) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->dump_kv_compare(block_id, false);
    }
}

void llama_kvmem_dump_kv_writeback(struct llama_context * /*ctx*/, int32_t block_id) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->dump_kv_compare(block_id, true);
    }
}

void llama_kvmem_apply_retrieval(struct llama_context * /*ctx*/) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->apply_retrieval();
    }
}

bool llama_memory_kvmem::query_replay_fits(uint32_t query_begin, uint32_t prompt_end) const {
    if (!runtime_ || query_begin >= prompt_end) {
        return true;
    }
    const kvmem::KvMemStore & store = runtime_->store();
    const uint32_t bt = std::max(1u, store.config().block_tokens);
    const uint32_t budget = store.budget_blocks();
    if (budget == 0) {
        return true;
    }
    const uint64_t total_blocks =
            (static_cast<uint64_t>(prompt_end) + bt - 1) / bt;
    const uint64_t boundary_block = query_begin / bt;
    if (boundary_block >= total_blocks) {
        return true;
    }
    const uint64_t sink =
            std::min<uint64_t>(store.config().sink_blocks, total_blocks);
    const uint64_t suffix = total_blocks - boundary_block;
    const uint64_t required =
            boundary_block < sink ? total_blocks : sink + suffix;
    return required <= static_cast<uint64_t>(budget);
}

bool llama_kvmem_query_replay_fits(uint32_t query_begin, uint32_t prompt_end) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return mem->query_replay_fits(query_begin, prompt_end);
    }
    return true;
}

void llama_kvmem_set_replay(bool replay) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->set_replay(replay);
    }
}

void llama_kvmem_trace_cells(struct llama_context * /*ctx*/, const char * tag) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->trace_working_set(tag ? tag : "cells");
    }
}

bool llama_kvmem_has_recurrent(void) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    return mem && mem->has_recurrent();
}

bool llama_kvmem_gdn_replay_enabled(void) {
    auto * mem = kvmem_capture_active();
    return mem && mem->gdn_replay_enabled();
}

bool llama_kvmem_gdn_replay_begin(llama_pos start, uint32_t width) {
    auto * mem = kvmem_capture_active();
    return mem && mem->gdn_replay_begin(start, width);
}

bool llama_kvmem_gdn_replay_commit(llama_context * ctx, uint32_t n_keep) {
    auto * mem = kvmem_capture_active();
    return mem && mem->gdn_replay_commit(ctx, n_keep);
}

bool llama_kvmem_want_decode_mean(void) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    return mem && mem->want_decode_mean();
}

void llama_kvmem_decode_mean_commit(uint32_t n_keep) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->decode_mean_commit(n_keep);
    }
}

void llama_kvmem_decode_mean_discard(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->decode_mean_discard();
    }
}

void llama_kvmem_decode_mean_flush(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->decode_mean_flush();
    }
}

bool llama_kvmem_want_prefill_capture(void) {
    const llama_kvmem_params * kp = llama_kvmem_get_params();
    if (!kp || !kp->enabled) {
        return false;
    }
    if (kp->method != 1 && !kp->harvest_v && getenv("KVMEM_DUMP_CAPTURE") == nullptr) {
        return false;
    }
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem) {
        return false;
    }
    return mem->want_prefill_capture();
}

bool llama_kvmem_want_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem || !mem->want_q_capture()) {
        return false;
    }
    return llama_kvmem_ubatch_needs_q_capture(n_tokens, n_pos, pos);
}

void llama_kvmem_reset_query(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->reset_query_acc();
    }
}

void llama_kvmem_end_prefill_capture(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->end_prefill_capture();
    }
}

void llama_kvmem_begin_cached_turn(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->begin_cached_turn(true);
    }
}

void llama_kvmem_begin_cached_turn_keep_query(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->begin_cached_turn(false);
    }
}

void llama_kvmem_keep_selected(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->keep_selected_window();
    }
}

void llama_kvmem_pin_working_set(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->pin_working_set();
    }
}

uint32_t llama_kvmem_free_slots(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return (uint32_t) mem->free_slot_count();
    }
    return 0;
}

uint32_t llama_kvmem_store_n_tokens(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return mem->store_n_tokens();
    }
    return 0;
}

void llama_kvmem_truncate_cached(uint32_t n_past) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->truncate_cached(n_past);
    }
}

llama_pos llama_kvmem_recr_pos_max(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return mem->recr_pos_max();
    }
    return -1;
}

void llama_kvmem_set_request_span(int32_t query_begin, int32_t query_end, int32_t force_pos) {
    g_kvmem_params.query_begin = query_begin;
    g_kvmem_params.query_end = query_end;
    g_kvmem_params.force_pos = force_pos;
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->set_query_span(query_begin, query_end);
        mem->set_force_pos(force_pos);
    }
}

void llama_kvmem_set_turn_spans(const llama_kvmem_turn_spans & spans) {
    if (auto * mem = kvmem_capture_active()) mem->set_turn_spans(spans);
}

llama_kvmem_attention_view llama_kvmem_get_attention_view() {
    if (auto * mem = kvmem_capture_active()) return mem->attention_view();
    return {};
}

bool llama_kvmem_can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason) {
    if (auto * mem = kvmem_capture_active()) return mem->can_append(end, generation_rows, all_history, reason);
    reason = "no_kvmem";
    return false;
}

llama_kvmem_selection llama_kvmem_preview_retrieval() {
    if (auto * mem = kvmem_capture_active()) return mem->preview_retrieval();
    return {};
}

bool llama_kvmem_selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t generation_rows) {
    if (auto * mem = kvmem_capture_active()) return mem->selection_fits(selection, end, generation_rows);
    return false;
}

bool llama_kvmem_commit_unchanged(const llama_kvmem_attention_view & view, const llama_kvmem_selection & selection) {
    if (auto * mem = kvmem_capture_active()) return mem->commit_unchanged(view, selection);
    return false;
}

void llama_kvmem_apply_selection(const llama_kvmem_selection & selection) {
    if (auto * mem = kvmem_capture_active()) mem->apply_selection(selection);
}

bool llama_kvmem_commit_resident(bool canonical) {
    if (auto * mem = kvmem_capture_active()) return mem->commit_resident(canonical);
    return false;
}

bool llama_kvmem_get_query(llama_kvmem_query_state & state) {
    if (auto * mem = kvmem_capture_active()) return mem->get_query(state);
    return false;
}

bool llama_kvmem_set_query(const llama_kvmem_query_state & state) {
    if (auto * mem = kvmem_capture_active()) return mem->set_query(state);
    return false;
}

void llama_kvmem_freeze_query(bool frozen) {
    if (auto * mem = kvmem_capture_active()) mem->freeze_query(frozen);
}

llama_pos llama_kvmem_model_pos(uint32_t logical) {
    auto * mem = kvmem_capture_active();
    return mem ? mem->model_pos(logical) : (llama_pos) logical;
}

bool llama_kvmem_remove_logical(llama_context * ctx, llama_pos begin, llama_pos end) {
    auto * mem = kvmem_capture_active();
    if (mem) return mem->remove_logical(ctx, begin, end);
    auto * native = llama_get_memory(ctx);
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(native);
    auto * kv = hybrid ? hybrid->get_mem_attn() : dynamic_cast<llama_kv_cache *>(native);
    if (!kv) return llama_memory_seq_rm(native, 0, begin, end);
    if (hybrid && end < 0) {
        const auto & cells = kv->get_cells(0);
        llama_pos first = INT32_MAX;
        llama_pos model_begin = -1;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) continue;
            const auto row = cells.ext_get(i).logical_pos;
            if (row >= begin && row < first) { first = row; model_begin = cells.pos_get(i); }
        }
        auto * recurrent = hybrid->get_mem_recr();
        if (model_begin >= 0 && recurrent->seq_pos_max(0) >= model_begin
                && !recurrent->seq_rm(0, model_begin, -1)) return false;
    }
    return kv->seq_rm_logical(0, begin, end);
}

void llama_kvmem_set_media_ranges(const uint32_t * starts, const uint32_t * ends, size_t count) {
    if (auto * mem = kvmem_capture_active()) {
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (size_t i = 0; i < count; ++i) ranges.emplace_back(starts[i], ends[i]);
        mem->runtime().store().set_media_ranges(std::move(ranges));
    }
}

void llama_kvmem_get_tail_mean(uint32_t row, std::vector<float> & state) {
    if (auto * mem = kvmem_capture_active()) {
        mem->harvest_flush();
        mem->decode_mean_flush();
        state = mem->raw().mean_checkpoint(row);
    }
}

void llama_kvmem_set_tail_mean(uint32_t row, const std::vector<float> & state) {
    if (auto * mem = kvmem_capture_active()) mem->raw().restore_mean_checkpoint(row, state);
}
