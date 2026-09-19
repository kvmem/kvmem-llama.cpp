#pragma once

// NVMe KV tier -- Windows stand-in.
//
// kvmem/include/kvmem/nvme_kv_tier.hpp is a POSIX component: it is built on
// positional pread/pwrite, O_DIRECT, posix_fadvise(DONTNEED) and
// sync_file_range, and it relies on 64-bit off_t. None of that has an
// equivalent on Windows:
//
//   * pread/pwrite  -- possible via FILE_FLAG_OVERLAPPED, but only if the file
//                      is opened through CreateFileW instead of the CRT, which
//                      would fork the whole open/preallocate/unlink policy;
//   * O_DIRECT      -- no analogue (FILE_FLAG_NO_BUFFERING has different
//                      alignment and lifetime rules);
//   * posix_fadvise(DONTNEED) / sync_file_range -- no analogue at all, and
//                      these carry the "don't keep a second copy of the arena
//                      in the page cache" contract that `drop_page_cache`
//                      exists to provide;
//   * off_t         -- MSVC defines it as 32-bit `long`, so arena offsets
//                      beyond 2 GiB would silently truncate.
//
// Rather than ship a port that quietly drops the cache-control and durability
// guarantees the Linux version makes, the tier is declared unavailable here.
// It is only reachable when a caller actually asks for an arena
// (`nvme_bytes > 0`, or raw-K pinned to NVMe), and that request now fails
// loudly at construction instead of running with different semantics.
//
// This matches the upstream README, which already documents NVMe offload as
// not implemented. Everything else in the tree -- the CPU pinned tier, the raw
// KV store, retrieval, MTP and the GPU stage-in/stage-out paths -- is
// unaffected.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kvmem {

struct NvmeKvTierConfig {
    std::string dir;
    std::string file_name = "kvmem_nvme.bin";
    uint64_t total_bytes = 0;
    uint64_t slot_bytes = 0;
    bool durable = false;
    bool direct_mapped = false;
    bool read_only = false;
    bool preallocate = false;
    std::string overlay_dir;
    std::string overlay_file_name = "kvmem_overlay.bin";
    bool drop_page_cache = false;
    bool direct_read = false;
};

struct NvmeSlotPlacement {
    int32_t slot = -1;
    int32_t evicted_block = -1;
};

struct NvmeIoSpan {
    int32_t slot = -1;
    uint64_t buffer_offset = 0;
    uint64_t bytes = 0;
};

struct NvmeBatchIoStats {
    uint64_t bytes = 0;
    uint64_t syscalls = 0;
    uint64_t duration_ns = 0;
    uint64_t cache_drop_bytes = 0;
    uint64_t cache_drop_failures = 0;
    uint64_t cpu_copy_bytes = 0;
    uint64_t cpu_copy_ns = 0;
    uint32_t cpu_copy_blocks = 0;
};

class NvmeKvTier {
public:
    explicit NvmeKvTier(NvmeKvTierConfig cfg) : cfg_(std::move(cfg)) {
        const bool wants_arena = cfg_.slot_bytes > 0 &&
                                 cfg_.total_bytes >= cfg_.slot_bytes &&
                                 !cfg_.dir.empty();
        if (wants_arena) {
            throw std::runtime_error(
                "KVMem's NVMe KV tier is not available in Windows builds. It is "
                "implemented on POSIX positional I/O (pread/pwrite), O_DIRECT, "
                "posix_fadvise(DONTNEED) and sync_file_range, which have no "
                "equivalent here, and MSVC's 32-bit off_t would truncate arena "
                "offsets past 2 GiB. Drop --kvmem-nvme-bytes / --kvmem-nvme-dir "
                "(and --kvmem-raw-k-nvme) to run on the CPU pinned tier and VRAM "
                "only.");
        }
    }

    // Always false: no arena was ever opened.
    bool enabled() const { return false; }
    uint32_t slot_count() const { return 0; }
    uint64_t slot_bytes() const { return cfg_.slot_bytes; }
    const std::string & path() const { return path_; }
    bool drops_page_cache() const { return cfg_.drop_page_cache; }
    bool direct_mapped() const { return cfg_.direct_mapped; }
    bool read_only() const { return cfg_.read_only; }
    bool has_overlay() const { return false; }
    bool direct_reads() const { return false; }

    uint32_t free_slots() const { return 0; }
    uint32_t used_slots() const { return 0; }
    uint64_t slot_offset(int32_t slot) const {
        return static_cast<uint64_t>(slot) * cfg_.slot_bytes;
    }
    int32_t block_slot(uint32_t) const { return -1; }
    int32_t lru_victim() const { return -1; }

    NvmeSlotPlacement place_block(uint32_t) { return NvmeSlotPlacement{}; }
    NvmeSlotPlacement place_block_evicting(uint32_t) { return NvmeSlotPlacement{}; }

    [[noreturn]] void mark_present_range(uint32_t, uint32_t) { unavailable(); }
    void release_block(uint32_t) {}
    void clear() {}
    void touch(uint32_t) {}

    [[noreturn]] void write_block(uint32_t, const void *, uint64_t) { unavailable(); }
    [[noreturn]] void read_block(uint32_t, void *, uint64_t) { unavailable(); }

    [[noreturn]] void write_slot(int32_t, const void *, uint64_t) const { unavailable(); }
    [[noreturn]] void write_slot_range(int32_t, uint64_t, const void *, uint64_t) const { unavailable(); }
    [[noreturn]] void read_slot(int32_t, void *, uint64_t) const { unavailable(); }
    [[noreturn]] void read_slot_range(int32_t, uint64_t, void *, uint64_t) const { unavailable(); }

    [[noreturn]] void write_spans(const std::vector<NvmeIoSpan> &, const void *, uint64_t,
                                  NvmeBatchIoStats * = nullptr) const { unavailable(); }
    [[noreturn]] void read_spans(const std::vector<NvmeIoSpan> &, void *, uint64_t,
                                 NvmeBatchIoStats * = nullptr) const { unavailable(); }

private:
    [[noreturn]] static void unavailable() {
        throw std::runtime_error(
            "KVMem's NVMe KV tier is not available in Windows builds");
    }

    NvmeKvTierConfig cfg_;
    std::string path_;
};

}  // namespace kvmem
