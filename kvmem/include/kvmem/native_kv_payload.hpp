#pragma once
#include "kvmem/memory_contract.hpp"
#include <cstddef>
#include <map>
#include <vector>

namespace kvmem {
class RawKvStore;
struct RawKvStoreConfig;

// Opaque native bytes, including all declared scale planes and padding. This type
// has no mean-K/Q statistics and never performs quantization or RoPE transforms.
class NativeKvPayload {
public:
    NativeKvPayload(BackendIdentity backend, PayloadLayout layout, BlockDescriptor block,
                    std::vector<uint8_t> bytes);
    const BackendIdentity & backend() const noexcept { return backend_; }
    const PayloadLayout & layout() const noexcept { return layout_; }
    const BlockDescriptor & block() const noexcept { return block_; }
    const std::vector<uint8_t> & bytes() const noexcept { return bytes_; }
private:
    BackendIdentity backend_;
    PayloadLayout layout_;
    BlockDescriptor block_;
    std::vector<uint8_t> bytes_;
};

// Portable bounded host implementation. Engine adapters may instead use their
// native pinned stores, with the same identity/version/completion contract.
// References returned by get are invalidated by put/erase; not async transfer leases.
class NativeKvArchive {
public:
    NativeKvArchive(BackendIdentity backend, SessionIdentity session, PayloadLayout layout,
                    uint64_t capacity_bytes);
    void put(NativeKvPayload payload); // atomic replacement after validation, rejects stale writes
    const NativeKvPayload & get(const BlockDescriptor & expected) const;
    void erase(uint64_t block_id);
    uint64_t bytes() const noexcept { return bytes_; }
    const PayloadLayout & layout() const noexcept { return layout_; }
private:
    BackendIdentity backend_;
    SessionIdentity session_;
    PayloadLayout layout_;
    uint64_t capacity_;
    uint64_t bytes_ = 0;
    std::map<uint64_t, NativeKvPayload> records_;
};

// Compatibility with the existing llama host mirror: K/V are packed token rows;
// quantization scales already embedded in those rows remain opaque. Only supplied
// attention layers participate; GDN layers must not be invented as zero-filled KV.
PayloadLayout raw_kv_payload_layout(const RawKvStoreConfig &, const std::string & format,
                                    const std::vector<uint32_t> & attention_layers);
NativeKvPayload export_raw_kv_payload(const RawKvStore &, const BackendIdentity &, const PayloadLayout &,
                                      const BlockDescriptor &);
// Caller holds the engine-safe boundary. Statistics and checkpoint state are untouched.
// Allocation/I/O failure during import invalidates the caller's attempted restore.
void import_raw_kv_payload(RawKvStore &, const BackendIdentity &, const PayloadLayout &,
                           const NativeKvPayload &);
} // namespace kvmem
