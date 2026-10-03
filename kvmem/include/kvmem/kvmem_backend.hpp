#pragma once

// Legacy slot hooks for the existing llama adapter and parked metadata runtimes.
// The default no-op behavior is retained for those existing callers only.
// New backend integration uses MemoryBackend in memory_contract.hpp, whose
// operations are all mandatory and whose transfers have explicit completion.

#include <cstdint>

namespace kvmem {

class KvMemBackend {
public:
    virtual ~KvMemBackend() = default;

    virtual int32_t alloc_gpu_slot() { return -1; }
    virtual void free_gpu_slot(int32_t /*slot*/) {}

    // Legacy byte-copy hooks; do not use the defaults as a production backend.
    virtual void copy_block_to_host(uint32_t /*block_id*/,
                                    int32_t /*gpu_slot*/,
                                    void * /*host*/,
                                    uint64_t /*bytes*/) {}
    virtual void copy_block_from_host(uint32_t /*block_id*/,
                                      int32_t /*gpu_slot*/,
                                      const void * /*host*/,
                                      uint64_t /*bytes*/) {}
};

} // namespace kvmem
