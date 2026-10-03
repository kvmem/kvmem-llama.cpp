#pragma once

// Host spill for attention KV, indexed by logical block then layer.
//
// Product path: ordered K sum (F32) at first write, mean-K computed on read;
// packed GPU-format K/V (q8_0 etc)
// copied at stage-out. Restore is memcpy; orig pos does not need unrotated K.
// `write_layer_k_rows` (unrotated token-major K) remains for tests only.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace kvmem {

class NvmeKvTier;
class SnapshotWriter;
class SnapshotReader;
struct SnapshotBuffer;

struct RawKvStoreConfig {
    uint32_t n_layer = 0;
    uint32_t n_embd_k = 0;
    uint32_t n_embd_v = 0;
    uint32_t block_tokens = 128;
    uint64_t k_row_bytes = 0;      // 0 = FP16 (n_embd_k * 2); else opaque K row
    uint64_t k_gpu_row_bytes = 0;  // 0 = no packed GPU K; else bytes per token row
    uint64_t v_gpu_row_bytes = 0;  // 0 = no packed GPU V; else bytes per token row
    uint64_t nvme_bytes = 0;
    std::string nvme_dir;
    std::string nvme_file = "kvmem_raw_k.bin";
};

class RawKvStore {
public:
    explicit RawKvStore(RawKvStoreConfig cfg);
    ~RawKvStore();

    RawKvStore(const RawKvStore &) = delete;
    RawKvStore &operator=(const RawKvStore &) = delete;

    const RawKvStoreConfig &config() const { return cfg_; }
    bool nvme_enabled() const;

    void ensure_blocks(uint32_t block_count);
    // Reserve packed destinations without marking any row valid. Caller has
    // drained capture/I/O writers before preparing a native transfer.
    void reserve_packed(uint32_t block_id, const std::vector<uint32_t> & layers);

    void write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                            const float * k, const float * v);
    void write_layer_tokens_f16(uint32_t pos0, uint32_t n, uint32_t il,
                                const uint16_t * k, const uint16_t * v);
    // Opaque unrotated K rows (`k_row_bytes`). Optional F32 is mean only.
    void write_layer_k_rows(uint32_t pos0, uint32_t n, uint32_t il,
                            const uint8_t * k, const float * k_f32);
    void write_layer_v_gpu(uint32_t pos0, uint32_t n, uint32_t il, const uint8_t * v);
    // Packed GPU-format K (RoPE+Hadamard+quant already applied). RAM only.
    void write_layer_k_gpu(uint32_t pos0, uint32_t n, uint32_t il, const uint8_t * k);
    // Prefill mean-K only (no raw-K rows). k is token-major F32, n_embd_k per token.
    void write_layer_mean_k(uint32_t pos0, uint32_t n, uint32_t il, const float * k);
    // Decode running-sum: `sum` is already reduced over `n` tokens in one block.
    void write_layer_mean_sum(uint32_t pos0, uint32_t n, uint32_t il, const float * sum);

    bool has_block(uint32_t block_id) const;
    bool has_k(uint32_t block_id, uint32_t il) const;
    bool has_k_gpu(uint32_t block_id, uint32_t il, uint32_t n = 1) const;
    bool has_v(uint32_t block_id, uint32_t il) const;
    bool has_v_gpu(uint32_t block_id, uint32_t il, uint32_t n = 1) const;
    uint32_t n_tokens(uint32_t block_id) const;

    bool copy_k(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_v(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_k_rows(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;
    bool copy_k_gpu(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;
    bool copy_v_gpu(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;

    // Normalize the stored sum by mean_tokens; absent statistics return zeros.
    void mean_k(uint32_t block_id, uint32_t il, float * out) const;

    size_t bytes_k() const;
    size_t bytes_v() const;
    size_t allocated_bytes() const;
    uint64_t capacity_bytes(uint32_t tokens, uint32_t populated_layers = UINT32_MAX) const;
    void snapshot_write(SnapshotWriter & out);
    void snapshot_read(SnapshotReader & in, uint32_t max_blocks);
    // Caller freezes the detached store until all bindings have been restored.
    void snapshot_buffers(std::vector<SnapshotBuffer> & buffers);

    uint64_t nvme_bytes_written() const;
    uint64_t nvme_syscalls() const;
    uint64_t nvme_wait_ns() const;

    void wait_writes();
    void clear();
    // Preserve only the valid prefix, including a partial last block.
    void truncate_to(uint32_t token_pos);
    void invalidate_packed_from(uint32_t token_pos);
    // In-process tail checkpoint: per layer, valid count followed by the F32 sum.
    std::vector<float> mean_checkpoint(uint32_t token_pos) const;
    void restore_mean_checkpoint(uint32_t token_pos, const std::vector<float> & state);

private:
    struct LayerBlk {
        uint32_t n_tokens = 0;
        uint32_t k_gpu_tokens = 0;
        uint32_t v_gpu_tokens = 0;
        uint32_t mean_tokens = 0;
        std::vector<uint8_t> k;
        std::vector<uint16_t> v;
        std::vector<uint8_t> k_gpu;
        std::vector<uint8_t> v_gpu;
        std::vector<float> k_sum;
        bool k_on_nvme = false;
        bool v_on_nvme = false;
        bool k_gpu_fmt = false;
        bool v_gpu_fmt = false;
        bool k_flushing = false;
        bool v_flushing = false;
    };
    struct BlockRaw {
        std::vector<LayerBlk> layers;
    };

    uint32_t nvme_key(uint32_t block_id, uint32_t il, bool is_v) const;
    uint64_t k_row_bytes() const;
    uint64_t k_slot_bytes() const;
    uint64_t v_slot_bytes() const;
    uint64_t v_gpu_slot_bytes() const;
    bool k_is_f16() const;
    void maybe_flush_k(uint32_t block_id, uint32_t il);
    void maybe_flush_v(uint32_t block_id, uint32_t il);
    void capture_mean_f16(LayerBlk & lb) const;
    void add_mean_f32(LayerBlk & lb, uint32_t off, uint32_t take, const float * k);
    bool load_k_nvme(uint32_t block_id, uint32_t il, uint8_t * dst) const;
    bool load_v_nvme(uint32_t block_id, uint32_t il, uint16_t * dst) const;
    bool load_v_gpu_nvme(uint32_t block_id, uint32_t il, uint8_t * dst) const;
    void enqueue_flush(uint32_t key, std::vector<uint8_t> && data, uint64_t bytes,
                       uint32_t block_id, uint32_t il, bool is_v);
    void io_loop();
    bool io_sync_inline() const;

    struct IoJob {
        uint32_t key = 0;
        uint32_t block_id = 0;
        uint32_t il = 0;
        bool is_v = false;
        uint64_t bytes = 0;
        std::vector<uint8_t> data;
    };

    RawKvStoreConfig cfg_;
    std::vector<BlockRaw> blocks_;
    std::unique_ptr<NvmeKvTier> nvme_;
    mutable std::vector<uint16_t> io_;
    mutable std::vector<uint8_t> io8_;
    size_t nvme_k_bytes_ = 0;
    size_t nvme_v_bytes_ = 0;
    uint64_t nvme_syscalls_ = 0;
    uint64_t nvme_wait_ns_ = 0;
    mutable std::mutex mu_;
    mutable std::condition_variable cv_;
    std::vector<IoJob> q_;
    size_t inflight_ = 0;
    std::thread io_thread_;
    std::atomic<bool> stop_io_{false};
};

} // namespace kvmem
