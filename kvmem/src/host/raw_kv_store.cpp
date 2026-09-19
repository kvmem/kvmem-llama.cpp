#include "kvmem/raw_kv_store.hpp"
#if defined(_WIN32)
// The POSIX implementation cannot be built on Windows; see the header.
#include "kvmem/nvme_kv_tier_win.hpp"
#else
#include "kvmem/nvme_kv_tier.hpp"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {
// steady_clock is the portable spelling of CLOCK_MONOTONIC: it never steps
// backwards and ignores wall-clock adjustments. libstdc++ implements it as
// clock_gettime(CLOCK_MONOTONIC) on Linux, so behaviour there is unchanged;
// MSVC has neither clock_gettime nor CLOCK_MONOTONIC.
uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
} // namespace

namespace kvmem {
namespace {

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if ((x & 0x7fffffffu) == 0) {
        return static_cast<uint16_t>(sign);
    }
    if (((x >> 23) & 0xffu) == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u | (man ? 0x200u : 0));
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        man |= 0x800000u;
        const uint32_t t = static_cast<uint32_t>(14 - exp);
        uint32_t half = man >> t;
        const uint32_t rem = man & ((1u << t) - 1u);
        if (rem > (1u << (t - 1)) || (rem == (1u << (t - 1)) && (half & 1u))) {
            half += 1;
        }
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t half = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
        half += 1;
    }
    return static_cast<uint16_t>(sign | half);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3ffu;
            out = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

void pack_f32(const float * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f32_to_f16(src[i]);
    }
}

void unpack_f16(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = f16_to_f32(src[i]);
    }
}

} // namespace

RawKvStore::RawKvStore(RawKvStoreConfig cfg) : cfg_(std::move(cfg)) {
#if !KVMEM_ENABLE_NVME
    if (cfg_.nvme_bytes) throw std::runtime_error("NVMe offload is disabled in this build");
#endif
    if (cfg_.nvme_bytes > 0 && !cfg_.nvme_dir.empty() && cfg_.block_tokens > 0) {
        NvmeKvTierConfig ncfg;
        ncfg.dir = cfg_.nvme_dir;
        ncfg.file_name = cfg_.nvme_file.empty() ? "kvmem_raw_k.bin" : cfg_.nvme_file;
        ncfg.total_bytes = cfg_.nvme_bytes;
        ncfg.slot_bytes = std::max(k_slot_bytes(), v_slot_bytes());
        if (cfg_.v_gpu_row_bytes) {
            ncfg.slot_bytes = std::max(ncfg.slot_bytes, v_gpu_slot_bytes());
        }
        ncfg.drop_page_cache = true;
        if (ncfg.slot_bytes == 0) {
            return;
        }
        nvme_ = std::make_unique<NvmeKvTier>(ncfg);
        if (nvme_->enabled()) {
            std::fprintf(stderr,
                         "KVMEM_RAW_NVME dir=%s file=%s bytes=%llu slots=%u slot_bytes=%llu\n",
                         cfg_.nvme_dir.c_str(), ncfg.file_name.c_str(),
                         (unsigned long long) cfg_.nvme_bytes,
                         nvme_->slot_count(),
                         (unsigned long long) ncfg.slot_bytes);
            if (!io_sync_inline()) {
                io_thread_ = std::thread([this] { io_loop(); });
            }
        }
    }
}

RawKvStore::~RawKvStore() {
    stop_io_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (io_thread_.joinable()) {
        io_thread_.join();
    }
}

bool RawKvStore::io_sync_inline() const {
    const char * e = std::getenv("KVMEM_HARVEST_SYNC");
    return e && e[0] != '\0' && e[0] != '0';
}

bool RawKvStore::nvme_enabled() const {
    return nvme_ && nvme_->enabled();
}

uint32_t RawKvStore::nvme_key(uint32_t block_id, uint32_t il, bool is_v) const {
    return block_id * (cfg_.n_layer * 2u + 2u) + il * 2u + (is_v ? 1u : 0u);
}

uint64_t RawKvStore::k_row_bytes() const {
    return cfg_.k_row_bytes ? cfg_.k_row_bytes
                            : static_cast<uint64_t>(cfg_.n_embd_k) * sizeof(uint16_t);
}

bool RawKvStore::k_is_f16() const {
    return k_row_bytes() == static_cast<uint64_t>(cfg_.n_embd_k) * sizeof(uint16_t);
}

uint64_t RawKvStore::k_slot_bytes() const {
    return static_cast<uint64_t>(cfg_.block_tokens) * k_row_bytes();
}

uint64_t RawKvStore::v_slot_bytes() const {
    return static_cast<uint64_t>(cfg_.block_tokens) * cfg_.n_embd_v * sizeof(uint16_t);
}

uint64_t RawKvStore::v_gpu_slot_bytes() const {
    return static_cast<uint64_t>(cfg_.block_tokens) * cfg_.v_gpu_row_bytes;
}

void RawKvStore::ensure_blocks(uint32_t block_count) {
    if (blocks_.size() >= block_count) {
        return;
    }
    const size_t old = blocks_.size();
    blocks_.resize(block_count);
    for (size_t i = old; i < blocks_.size(); ++i) {
        blocks_[i].layers.resize(cfg_.n_layer);
    }
}

void RawKvStore::capture_mean_f16(LayerBlk & lb) const {
    if (!k_is_f16() || lb.k.empty() || cfg_.n_embd_k == 0) {
        return;
    }
    const uint32_t nt = lb.n_tokens;
    if (nt == 0) {
        return;
    }
    const uint64_t row = k_row_bytes();
    lb.k_sum.assign(cfg_.n_embd_k, 0.0f);
    for (uint32_t t = 0; t < nt; ++t) {
        const auto * src = reinterpret_cast<const uint16_t *>(
                lb.k.data() + static_cast<size_t>(t) * static_cast<size_t>(row));
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            lb.k_sum[d] += f16_to_f32(src[d]);
        }
    }
    lb.mean_tokens = nt;
}

void RawKvStore::add_mean_f32(LayerBlk & lb, uint32_t off, uint32_t take,
                              const float * k) {
    if (!k || take == 0 || cfg_.n_embd_k == 0) {
        return;
    }
    if (lb.k_sum.size() != cfg_.n_embd_k) {
        lb.k_sum.assign(cfg_.n_embd_k, 0.0f);
    }
    if (off == 0) {
        std::fill(lb.k_sum.begin(), lb.k_sum.end(), 0.0f);
    }
    for (uint32_t t = 0; t < take; ++t) {
        const float * row = k + t * cfg_.n_embd_k;
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            lb.k_sum[d] += row[d];
        }
    }
    const uint32_t nt = off == 0 ? take : std::max(lb.mean_tokens, off + take);
    lb.mean_tokens = nt;
}

void RawKvStore::enqueue_flush(uint32_t key, std::vector<uint8_t> && data,
                               uint64_t bytes, uint32_t block_id, uint32_t il,
                               bool is_v) {
    IoJob job;
    job.key = key;
    job.block_id = block_id;
    job.il = il;
    job.is_v = is_v;
    job.bytes = bytes;
    job.data = std::move(data);
    q_.push_back(std::move(job));
    cv_.notify_one();
}

void RawKvStore::maybe_flush_k(uint32_t block_id, uint32_t il) {
    if (!nvme_enabled() || block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return;
    }
    LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.k_on_nvme || lb.k_flushing || lb.k.empty() ||
        lb.n_tokens < cfg_.block_tokens) {
        return;
    }
    const uint64_t nbytes = k_slot_bytes();
    if (lb.k.size() < nbytes) {
        return;
    }
    if (k_is_f16() && lb.k_sum.size() != cfg_.n_embd_k) {
        capture_mean_f16(lb);
    }
    if (io_sync_inline() || !io_thread_.joinable()) {
        const uint64_t t0 = monotonic_ns();
        nvme_->write_block(nvme_key(block_id, il, false), lb.k.data(), nbytes);
        nvme_wait_ns_ += monotonic_ns() - t0;
        nvme_syscalls_ += 1;
        nvme_k_bytes_ += nbytes;
        lb.k.clear();
        lb.k.shrink_to_fit();
        lb.k_on_nvme = true;
        return;
    }
    lb.k_flushing = true;
    std::vector<uint8_t> blob(nbytes);
    std::memcpy(blob.data(), lb.k.data(), nbytes);
    lb.k.clear();
    lb.k.shrink_to_fit();
    enqueue_flush(nvme_key(block_id, il, false), std::move(blob), nbytes,
                  block_id, il, false);
}

void RawKvStore::maybe_flush_v(uint32_t block_id, uint32_t il) {
    if (!nvme_enabled() || block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return;
    }
    LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.v_on_nvme || lb.v_flushing || lb.n_tokens < cfg_.block_tokens) {
        return;
    }
    const bool gpu = lb.v_gpu_fmt && !lb.v_gpu.empty();
    const uint64_t nbytes = gpu ? v_gpu_slot_bytes() : v_slot_bytes();
    if (gpu) {
        if (lb.v_gpu.size() < nbytes) {
            return;
        }
    } else if (lb.v.empty() || lb.v.size() * sizeof(uint16_t) < nbytes) {
        return;
    }
    const void * src = gpu ? static_cast<const void *>(lb.v_gpu.data())
                           : static_cast<const void *>(lb.v.data());
    if (io_sync_inline() || !io_thread_.joinable()) {
        const uint64_t t0 = monotonic_ns();
        nvme_->write_block(nvme_key(block_id, il, true), src, nbytes);
        nvme_wait_ns_ += monotonic_ns() - t0;
        nvme_syscalls_ += 1;
        nvme_v_bytes_ += nbytes;
        lb.v.clear();
        lb.v.shrink_to_fit();
        lb.v_gpu.clear();
        lb.v_gpu.shrink_to_fit();
        lb.v_on_nvme = true;
        return;
    }
    lb.v_flushing = true;
    std::vector<uint8_t> blob;
    if (gpu) {
        blob = std::move(lb.v_gpu);
        if (blob.size() > nbytes) {
            blob.resize(nbytes);
        }
    } else {
        blob.assign(nbytes, 0);
        std::memcpy(blob.data(), lb.v.data(), nbytes);
        lb.v.clear();
        lb.v.shrink_to_fit();
    }
    lb.v_gpu.clear();
    lb.v_gpu.shrink_to_fit();
    enqueue_flush(nvme_key(block_id, il, true), std::move(blob), nbytes,
                  block_id, il, true);
}

void RawKvStore::io_loop() {
    while (true) {
        std::vector<IoJob> batch;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] {
                return stop_io_.load(std::memory_order_acquire) || !q_.empty();
            });
            if (q_.empty() && stop_io_.load(std::memory_order_acquire)) {
                break;
            }
            batch.swap(q_);
            inflight_ = batch.size();
        }
        if (batch.empty() || !nvme_) {
            continue;
        }
        std::vector<NvmeIoSpan> spans;
        spans.reserve(batch.size());
        uint64_t slab_bytes = 0;
        for (auto & job : batch) {
            auto p = nvme_->place_block(job.key);
            if (p.slot < 0) {
                continue;
            }
            NvmeIoSpan sp;
            sp.slot = p.slot;
            sp.buffer_offset = slab_bytes;
            sp.bytes = job.bytes;
            spans.push_back(sp);
            slab_bytes += job.bytes;
        }
        std::vector<size_t> order(spans.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return spans[a].slot < spans[b].slot;
        });
        std::vector<uint8_t> slab(slab_bytes);
        std::vector<NvmeIoSpan> sorted;
        sorted.reserve(spans.size());
        uint64_t off = 0;
        for (size_t idx : order) {
            NvmeIoSpan sp = spans[idx];
            std::memcpy(slab.data() + off, batch[idx].data.data(),
                        static_cast<size_t>(batch[idx].bytes));
            sp.buffer_offset = off;
            sorted.push_back(sp);
            off += batch[idx].bytes;
        }
        NvmeBatchIoStats st;
        const uint64_t t0 = monotonic_ns();
        nvme_->write_spans(sorted, slab.data(), slab.size(), &st);
        const uint64_t dt = monotonic_ns() - t0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            nvme_wait_ns_ += st.duration_ns ? st.duration_ns : dt;
            nvme_syscalls_ += st.syscalls ? st.syscalls : sorted.size();
            for (auto & job : batch) {
                if (job.block_id >= blocks_.size() || job.il >= cfg_.n_layer) {
                    continue;
                }
                LayerBlk & lb = blocks_[job.block_id].layers[job.il];
                if (job.is_v) {
                    lb.v_flushing = false;
                    lb.v_on_nvme = true;
                    nvme_v_bytes_ += job.bytes;
                } else {
                    lb.k_flushing = false;
                    lb.k_on_nvme = true;
                    nvme_k_bytes_ += job.bytes;
                }
            }
            inflight_ = 0;
            cv_.notify_all();
        }
    }
}

void RawKvStore::wait_writes() {
    if (!nvme_enabled() || io_sync_inline() || !io_thread_.joinable()) {
        return;
    }
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return q_.empty() && inflight_ == 0; });
}

void RawKvStore::write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                                    const float * k, const float * v) {
    std::unique_lock<std::mutex> lk(mu_);
    if (n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        if (k) {
            cv_.wait(lk, [&] { return !blocks_[bid].layers[il].k_flushing; });
        }
        if (v) {
            cv_.wait(lk, [&] { return !blocks_[bid].layers[il].v_flushing; });
        }
        LayerBlk & lb = blocks_[bid].layers[il];
        if (k && k_is_f16()) {
            const uint64_t row = k_row_bytes();
            const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
            if (lb.k.size() < need) {
                if (lb.k_on_nvme) {
                    lb.k.assign(need, 0);
                    load_k_nvme(bid, il, lb.k.data());
                    lb.k_on_nvme = false;
                } else {
                    lb.k.assign(need, 0);
                }
            }
            pack_f32(k + done * cfg_.n_embd_k,
                     reinterpret_cast<uint16_t *>(
                             lb.k.data() + static_cast<size_t>(off) * static_cast<size_t>(row)),
                     static_cast<size_t>(take) * cfg_.n_embd_k);
            add_mean_f32(lb, off, take, k + done * cfg_.n_embd_k);
        }
        if (v) {
            lb.v_gpu.clear();
            lb.v_gpu.shrink_to_fit();
            if (lb.v_on_nvme && lb.v_gpu_fmt) {
                lb.v_on_nvme = false;
            }
            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_v;
            if (lb.v.size() < need) {
                if (lb.v_on_nvme) {
                    lb.v.assign(need, 0);
                    load_v_nvme(bid, il, lb.v.data());
                    lb.v_on_nvme = false;
                } else {
                    lb.v.assign(need, 0);
                }
            }
            pack_f32(v + done * cfg_.n_embd_v,
                     lb.v.data() + off * cfg_.n_embd_v,
                     static_cast<size_t>(take) * cfg_.n_embd_v);
            lb.v_gpu_fmt = false;
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        maybe_flush_k(bid, il);
        maybe_flush_v(bid, il);
        done += take;
    }
}

void RawKvStore::write_layer_tokens_f16(uint32_t pos0, uint32_t n, uint32_t il,
                                        const uint16_t * k, const uint16_t * v) {
    std::unique_lock<std::mutex> lk(mu_);
    if (n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        if (k) {
            cv_.wait(lk, [&] { return !blocks_[bid].layers[il].k_flushing; });
        }
        if (v) {
            cv_.wait(lk, [&] { return !blocks_[bid].layers[il].v_flushing; });
        }
        LayerBlk & lb = blocks_[bid].layers[il];
        if (k && k_is_f16()) {
            const uint64_t row = k_row_bytes();
            const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
            if (lb.k.size() < need) {
                if (lb.k_on_nvme) {
                    lb.k.assign(need, 0);
                    load_k_nvme(bid, il, lb.k.data());
                    lb.k_on_nvme = false;
                } else {
                    lb.k.assign(need, 0);
                }
            }
            std::memcpy(lb.k.data() + static_cast<size_t>(off) * static_cast<size_t>(row),
                        k + done * cfg_.n_embd_k,
                        static_cast<size_t>(take) * static_cast<size_t>(row));
        }
        if (v) {
            lb.v_gpu.clear();
            lb.v_gpu.shrink_to_fit();
            if (lb.v_on_nvme && lb.v_gpu_fmt) {
                lb.v_on_nvme = false;
            }
            const size_t need = static_cast<size_t>(bt) * cfg_.n_embd_v;
            if (lb.v.size() < need) {
                if (lb.v_on_nvme) {
                    lb.v.assign(need, 0);
                    load_v_nvme(bid, il, lb.v.data());
                    lb.v_on_nvme = false;
                } else {
                    lb.v.assign(need, 0);
                }
            }
            std::memcpy(lb.v.data() + off * cfg_.n_embd_v,
                        v + done * cfg_.n_embd_v,
                        static_cast<size_t>(take) * cfg_.n_embd_v * sizeof(uint16_t));
            lb.v_gpu_fmt = false;
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        if (k && k_is_f16()) {
            capture_mean_f16(lb);
        }
        maybe_flush_k(bid, il);
        maybe_flush_v(bid, il);
        done += take;
    }
}

void RawKvStore::write_layer_k_rows(uint32_t pos0, uint32_t n, uint32_t il,
                                    const uint8_t * k, const float * k_f32) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!k || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        k_row_bytes() == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint64_t row = k_row_bytes();
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        cv_.wait(lk, [&] { return !blocks_[bid].layers[il].k_flushing; });
        LayerBlk & lb = blocks_[bid].layers[il];
        const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
        if (lb.k.size() < need) {
            if (lb.k_on_nvme) {
                lb.k.assign(need, 0);
                load_k_nvme(bid, il, lb.k.data());
                lb.k_on_nvme = false;
            } else {
                lb.k.assign(need, 0);
            }
        }
        std::memcpy(lb.k.data() + static_cast<size_t>(off) * static_cast<size_t>(row),
                    k + static_cast<size_t>(done) * static_cast<size_t>(row),
                    static_cast<size_t>(take) * static_cast<size_t>(row));
        if (k_f32) {
            add_mean_f32(lb, off, take, k_f32 + done * cfg_.n_embd_k);
        }
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        maybe_flush_k(bid, il);
        done += take;
    }
}

void RawKvStore::write_layer_v_gpu(uint32_t pos0, uint32_t n, uint32_t il,
                                   const uint8_t * v) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!v || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.v_gpu_row_bytes == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint64_t row = cfg_.v_gpu_row_bytes;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        cv_.wait(lk, [&] { return !blocks_[bid].layers[il].v_flushing; });
        LayerBlk & lb = blocks_[bid].layers[il];
        lb.v.clear();
        const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
        const size_t nbytes = static_cast<size_t>(take) * static_cast<size_t>(row);
        const uint8_t * src = v + static_cast<size_t>(done) * static_cast<size_t>(row);
        if (off == 0 && take == bt && !(lb.v_on_nvme && lb.v_gpu_fmt)) {
            if (lb.v_on_nvme) {
                lb.v_on_nvme = false;
            }
            lb.v_gpu.assign(src, src + need);
        } else {
            if (lb.v_gpu.size() < need) {
                if (lb.v_on_nvme && lb.v_gpu_fmt) {
                    lb.v_gpu.assign(need, 0);
                    load_v_gpu_nvme(bid, il, lb.v_gpu.data());
                    lb.v_on_nvme = false;
                } else {
                    if (lb.v_on_nvme) {
                        lb.v_on_nvme = false;
                    }
                    lb.v_gpu.resize(need);
                }
            }
            std::memcpy(lb.v_gpu.data() + static_cast<size_t>(off) * row, src, nbytes);
        }
        lb.v_gpu_fmt = true;
        lb.v_gpu_tokens = std::max(lb.v_gpu_tokens, off + take);
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        maybe_flush_v(bid, il);
        done += take;
    }
}

void RawKvStore::write_layer_k_gpu(uint32_t pos0, uint32_t n, uint32_t il,
                                   const uint8_t * k) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!k || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.k_gpu_row_bytes == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint64_t row = cfg_.k_gpu_row_bytes;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        LayerBlk & lb = blocks_[bid].layers[il];
        const size_t need = static_cast<size_t>(bt) * static_cast<size_t>(row);
        const size_t nbytes = static_cast<size_t>(take) * static_cast<size_t>(row);
        const uint8_t * src = k + static_cast<size_t>(done) * static_cast<size_t>(row);
        if (off == 0 && take == bt) {
            lb.k_gpu.assign(src, src + need);
        } else {
            if (lb.k_gpu.size() < need) {
                lb.k_gpu.assign(need, 0);
            }
            std::memcpy(lb.k_gpu.data() + static_cast<size_t>(off) * row, src, nbytes);
        }
        lb.k_gpu_fmt = true;
        lb.k_gpu_tokens = std::max(lb.k_gpu_tokens, off + take);
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        done += take;
    }
}

void RawKvStore::write_layer_mean_k(uint32_t pos0, uint32_t n, uint32_t il,
                                    const float * k) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!k || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.n_embd_k == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    uint32_t done = 0;
    while (done < n) {
        const uint32_t pos = pos0 + done;
        const uint32_t bid = pos / bt;
        const uint32_t off = pos % bt;
        const uint32_t take = std::min(n - done, bt - off);
        ensure_blocks(bid + 1);
        LayerBlk & lb = blocks_[bid].layers[il];
        add_mean_f32(lb, off, take, k + done * cfg_.n_embd_k);
        lb.n_tokens = std::max(lb.n_tokens, off + take);
        done += take;
    }
}

void RawKvStore::write_layer_mean_sum(uint32_t pos0, uint32_t n, uint32_t il,
                                      const float * sum) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!sum || n == 0 || il >= cfg_.n_layer || cfg_.block_tokens == 0 ||
        cfg_.n_embd_k == 0) {
        return;
    }
    const uint32_t bt = cfg_.block_tokens;
    const uint32_t bid = pos0 / bt;
    const uint32_t off = pos0 % bt;
    const uint32_t take = std::min(n, bt - off);
    ensure_blocks(bid + 1);
    LayerBlk & lb = blocks_[bid].layers[il];
    if (lb.k_sum.size() != cfg_.n_embd_k) {
        lb.k_sum.assign(cfg_.n_embd_k, 0.0f);
    }
    if (off == 0) {
        std::fill(lb.k_sum.begin(), lb.k_sum.end(), 0.0f);
    }
    for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
        lb.k_sum[d] += sum[d];
    }
    lb.n_tokens = std::max(lb.n_tokens, off + take);
    lb.mean_tokens = off == 0 ? take : std::max(lb.mean_tokens, off + take);
}

bool RawKvStore::has_block(uint32_t block_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size()) {
        return false;
    }
    for (const auto & lb : blocks_[block_id].layers) {
        if (lb.n_tokens > 0 &&
            (!lb.k.empty() || lb.k_on_nvme || lb.k_flushing || lb.k_gpu_fmt ||
             (lb.mean_tokens > 0 && !lb.k_sum.empty()))) {
            return true;
        }
    }
    return false;
}

bool RawKvStore::has_k(uint32_t block_id, uint32_t il) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return !lb.k.empty() || lb.k_on_nvme || lb.k_flushing;
}

bool RawKvStore::has_k_gpu(uint32_t block_id, uint32_t il, uint32_t n) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return lb.k_gpu_fmt && !lb.k_gpu.empty() && lb.k_gpu_tokens >= n;
}

bool RawKvStore::has_v(uint32_t block_id, uint32_t il) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return !lb.v.empty() || !lb.v_gpu.empty() || lb.v_on_nvme || lb.v_flushing;
}

bool RawKvStore::has_v_gpu(uint32_t block_id, uint32_t il, uint32_t n) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    return lb.v_gpu_fmt && lb.v_gpu_tokens >= n;
}

uint32_t RawKvStore::n_tokens(uint32_t block_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size()) {
        return 0;
    }
    uint32_t n = 0;
    for (const auto & lb : blocks_[block_id].layers) {
        n = std::max(n, lb.n_tokens);
    }
    return n;
}

bool RawKvStore::load_k_nvme(uint32_t block_id, uint32_t il, uint8_t * dst) const {
    if (!nvme_enabled() || !dst) {
        return false;
    }
    try {
        nvme_->read_block(nvme_key(block_id, il, false), dst, k_slot_bytes());
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool RawKvStore::load_v_nvme(uint32_t block_id, uint32_t il, uint16_t * dst) const {
    if (!nvme_enabled() || !dst) {
        return false;
    }
    try {
        nvme_->read_block(nvme_key(block_id, il, true), dst, v_slot_bytes());
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool RawKvStore::load_v_gpu_nvme(uint32_t block_id, uint32_t il, uint8_t * dst) const {
    if (!nvme_enabled() || !dst || cfg_.v_gpu_row_bytes == 0) {
        return false;
    }
    try {
        nvme_->read_block(nvme_key(block_id, il, true), dst, v_gpu_slot_bytes());
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool RawKvStore::copy_k(uint32_t block_id, uint32_t il, float * out) const {
    if (!out || !k_is_f16()) {
        return false;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    cv_.wait(lk, [&] {
        const LayerBlk & x = blocks_[block_id].layers[il];
        return !x.k_flushing;
    });
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.k.empty() && !lb.k_on_nvme) {
        return false;
    }
    const uint32_t nt = lb.n_tokens;
    const uint8_t * raw = nullptr;
    if (!lb.k.empty()) {
        raw = lb.k.data();
    } else {
        io8_.assign(static_cast<size_t>(k_slot_bytes()), 0);
        if (!load_k_nvme(block_id, il, io8_.data())) {
            return false;
        }
        raw = io8_.data();
    }
    unpack_f16(reinterpret_cast<const uint16_t *>(raw), out,
               static_cast<size_t>(nt) * cfg_.n_embd_k);
    return true;
}

bool RawKvStore::copy_v(uint32_t block_id, uint32_t il, float * out) const {
    if (!out) {
        return false;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    cv_.wait(lk, [&] {
        const LayerBlk & x = blocks_[block_id].layers[il];
        return !x.v_flushing;
    });
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (lb.v_gpu_fmt && lb.v.empty()) {
        return false;
    }
    if (lb.v.empty() && !lb.v_on_nvme) {
        return false;
    }
    const uint32_t nt = lb.n_tokens;
    const uint16_t * src = nullptr;
    if (!lb.v.empty()) {
        src = lb.v.data();
    } else {
        io_.assign(static_cast<size_t>(cfg_.block_tokens) * cfg_.n_embd_v, 0);
        if (!load_v_nvme(block_id, il, io_.data())) {
            return false;
        }
        src = io_.data();
    }
    unpack_f16(src, out, static_cast<size_t>(nt) * cfg_.n_embd_v);
    return true;
}

bool RawKvStore::copy_k_rows(uint32_t block_id, uint32_t il, uint8_t * out,
                             uint32_t n) const {
    if (!out || n == 0 || k_row_bytes() == 0) {
        return false;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    cv_.wait(lk, [&] {
        const LayerBlk & x = blocks_[block_id].layers[il];
        return !x.k_flushing;
    });
    const LayerBlk & lb = blocks_[block_id].layers[il];
    const uint64_t row = k_row_bytes();
    const uint32_t nt = std::min(n, lb.n_tokens);
    const uint8_t * src = nullptr;
    if (!lb.k.empty()) {
        src = lb.k.data();
    } else if (lb.k_on_nvme) {
        io8_.assign(static_cast<size_t>(k_slot_bytes()), 0);
        if (!load_k_nvme(block_id, il, io8_.data())) {
            return false;
        }
        src = io8_.data();
    } else {
        return false;
    }
    std::memcpy(out, src, static_cast<size_t>(nt) * static_cast<size_t>(row));
    return true;
}

bool RawKvStore::copy_k_gpu(uint32_t block_id, uint32_t il, uint8_t * out,
                            uint32_t n) const {
    if (!out || n == 0 || cfg_.k_gpu_row_bytes == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (!lb.k_gpu_fmt || lb.k_gpu.empty() || lb.k_gpu_tokens < n) {
        return false;
    }
    const uint64_t row = cfg_.k_gpu_row_bytes;
    const uint32_t nt = n;
    if (lb.k_gpu.size() < static_cast<size_t>(nt) * static_cast<size_t>(row)) {
        return false;
    }
    std::memcpy(out, lb.k_gpu.data(), static_cast<size_t>(nt) * static_cast<size_t>(row));
    return true;
}

bool RawKvStore::copy_v_gpu(uint32_t block_id, uint32_t il, uint8_t * out,
                            uint32_t n) const {
    if (!out || n == 0 || cfg_.v_gpu_row_bytes == 0) {
        return false;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        return false;
    }
    cv_.wait(lk, [&] {
        const LayerBlk & x = blocks_[block_id].layers[il];
        return !x.v_flushing;
    });
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (!lb.v_gpu_fmt || lb.v_gpu_tokens < n) {
        return false;
    }
    const uint64_t row = cfg_.v_gpu_row_bytes;
    const uint32_t nt = n;
    const uint8_t * src = nullptr;
    if (!lb.v_gpu.empty()) {
        src = lb.v_gpu.data();
    } else if (lb.v_on_nvme) {
        io8_.assign(static_cast<size_t>(v_gpu_slot_bytes()), 0);
        if (!load_v_gpu_nvme(block_id, il, io8_.data())) {
            return false;
        }
        src = io8_.data();
    } else {
        return false;
    }
    std::memcpy(out, src, static_cast<size_t>(nt) * static_cast<size_t>(row));
    return true;
}

void RawKvStore::mean_k(uint32_t block_id, uint32_t il, float * out) const {
    if (!out) {
        return;
    }
    std::unique_lock<std::mutex> lk(mu_);
    if (block_id >= blocks_.size() || il >= cfg_.n_layer) {
        std::fill_n(out, cfg_.n_embd_k, 0.0f);
        return;
    }
    const auto normalize = [&](const LayerBlk & lb) {
        if (lb.mean_tokens == 0 || lb.k_sum.size() != cfg_.n_embd_k) {
            return false;
        }
        // Keep the same FP32 reciprocal and multiplication as the write path
        // used for its cached mean. Never normalize the accumulator in place.
        const float inv = 1.0f / static_cast<float>(lb.mean_tokens);
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            out[d] = lb.k_sum[d] * inv;
        }
        return true;
    };
    if (normalize(blocks_[block_id].layers[il])) {
        return;
    }
    std::fill_n(out, cfg_.n_embd_k, 0.0f);
    if (!k_is_f16()) {
        return;
    }
    cv_.wait(lk, [&] {
        return block_id >= blocks_.size() || !blocks_[block_id].layers[il].k_flushing;
    });
    if (block_id >= blocks_.size()) {
        return;
    }
    const LayerBlk & lb = blocks_[block_id].layers[il];
    if (normalize(lb)) {
        return;
    }
    const uint32_t nt = lb.n_tokens;
    if (nt == 0 || cfg_.n_embd_k == 0) {
        return;
    }
    const uint8_t * raw = nullptr;
    if (!lb.k.empty()) {
        raw = lb.k.data();
    } else if (lb.k_on_nvme) {
        io8_.assign(static_cast<size_t>(k_slot_bytes()), 0);
        if (!load_k_nvme(block_id, il, io8_.data())) {
            return;
        }
        raw = io8_.data();
    } else {
        return;
    }
    const uint64_t row = k_row_bytes();
    std::fill(out, out + cfg_.n_embd_k, 0.0f);
    for (uint32_t t = 0; t < nt; ++t) {
        const auto * src = reinterpret_cast<const uint16_t *>(
                raw + static_cast<size_t>(t) * static_cast<size_t>(row));
        for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
            out[d] += f16_to_f32(src[d]);
        }
    }
    const float inv = 1.0f / static_cast<float>(nt);
    for (uint32_t d = 0; d < cfg_.n_embd_k; ++d) {
        out[d] *= inv;
    }
}

size_t RawKvStore::bytes_k() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = nvme_k_bytes_;
    for (const auto & b : blocks_) {
        for (const auto & lb : b.layers) {
            n += lb.k.size();
            n += lb.k_gpu.size();
            n += lb.k_sum.size() * sizeof(float);
        }
    }
    return n;
}

uint64_t RawKvStore::nvme_bytes_written() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nvme_k_bytes_ + nvme_v_bytes_;
}

uint64_t RawKvStore::nvme_syscalls() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nvme_syscalls_;
}

uint64_t RawKvStore::nvme_wait_ns() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nvme_wait_ns_;
}

size_t RawKvStore::bytes_v() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = nvme_v_bytes_;
    for (const auto & b : blocks_) {
        for (const auto & lb : b.layers) {
            n += lb.v.size() * sizeof(uint16_t);
            n += lb.v_gpu.size();
        }
    }
    return n;
}

void RawKvStore::truncate_to(uint32_t token_pos) {
    wait_writes();
    std::lock_guard<std::mutex> lk(mu_);
    if (token_pos == 0 || cfg_.block_tokens == 0) {
        blocks_.clear();
        return;
    }
    const uint32_t keep = (token_pos + cfg_.block_tokens - 1) / cfg_.block_tokens;
    if (blocks_.size() > keep) {
        blocks_.resize(keep);
    }
    const uint32_t tail = token_pos % cfg_.block_tokens;
    if (tail && blocks_.size() == keep) {
        for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
            auto & lb = blocks_.back().layers[il];
            lb.n_tokens = std::min(lb.n_tokens, tail);
            lb.k_gpu_tokens = std::min(lb.k_gpu_tokens, tail);
            lb.v_gpu_tokens = std::min(lb.v_gpu_tokens, tail);
            if (lb.mean_tokens > tail) {
                lb.k_sum.clear();
                lb.mean_tokens = 0;
                if (k_is_f16()) {
                    if (lb.k.empty() && lb.k_on_nvme) {
                        lb.k.resize(k_slot_bytes());
                        if (!load_k_nvme(keep - 1, il, lb.k.data())) lb.k.clear();
                        lb.k_on_nvme = false;
                    }
                    capture_mean_f16(lb);
                }
            }
        }
    }
}

void RawKvStore::invalidate_packed_from(uint32_t token_pos) {
    wait_writes();
    std::lock_guard<std::mutex> lk(mu_);
    for (uint32_t bid = token_pos / cfg_.block_tokens; bid < blocks_.size(); ++bid) {
        const uint32_t keep = bid == token_pos / cfg_.block_tokens ? token_pos % cfg_.block_tokens : 0;
        for (auto & lb : blocks_[bid].layers) {
            lb.k_gpu_tokens = std::min(lb.k_gpu_tokens, keep);
            lb.v_gpu_tokens = std::min(lb.v_gpu_tokens, keep);
        }
    }
}

std::vector<float> RawKvStore::mean_checkpoint(uint32_t token_pos) const {
    std::lock_guard<std::mutex> lk(mu_);
    const uint32_t bid = token_pos / cfg_.block_tokens;
    if (token_pos % cfg_.block_tokens == 0) return {};
    const size_t stride = 1 + cfg_.n_embd_k;
    std::vector<float> state(cfg_.n_layer * stride, 0);
    if (bid >= blocks_.size()) return state;
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        const auto & lb = blocks_[bid].layers[il];
        float * dst = state.data() + il * stride;
        dst[0] = lb.mean_tokens;
        if (!lb.k_sum.empty()) std::copy(lb.k_sum.begin(), lb.k_sum.end(), dst + 1);
    }
    return state;
}

void RawKvStore::restore_mean_checkpoint(uint32_t token_pos, const std::vector<float> & state) {
    if (state.empty()) return;
    std::lock_guard<std::mutex> lk(mu_);
    const size_t stride = 1 + cfg_.n_embd_k;
    if (state.size() != cfg_.n_layer * stride) throw std::runtime_error("invalid mean-K checkpoint");
    const uint32_t bid = token_pos / cfg_.block_tokens;
    ensure_blocks(bid + 1);
    for (uint32_t il = 0; il < cfg_.n_layer; ++il) {
        auto & lb = blocks_[bid].layers[il];
        const float * src = state.data() + il * stride;
        lb.mean_tokens = (uint32_t) src[0];
        if (lb.mean_tokens > 0) {
            lb.k_sum.assign(src + 1, src + stride);
        } else {
            lb.k_sum.clear();
        }
    }
}

void RawKvStore::clear() {
    wait_writes();
    std::lock_guard<std::mutex> lk(mu_);
    blocks_.clear();
    nvme_k_bytes_ = 0;
    nvme_v_bytes_ = 0;
    nvme_syscalls_ = 0;
    nvme_wait_ns_ = 0;
    if (nvme_) {
        nvme_->clear();
    }
}

} // namespace kvmem
