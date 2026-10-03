#pragma once
#include "kvmem/memory_contract.hpp"
#include "kvmem/native_kv_payload.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); std::abort(); } } while (0)
using namespace kvmem;
template<class F> void rejects(MemoryErrorCode code, F action) {
    bool caught = false;
    try { action(); } catch (const MemoryError & e) { caught = true; CHECK(e.code() == code); }
    CHECK(caught);
}
inline MemoryDescriptor descriptor(int profile) {
    MemoryDescriptor d;
    d.identity = {profile == 0 ? "llama-test" : "ninfer-test", "fixture-1", "qwen-test", "state-v1"};
    if (profile == 0) {
        RawKvStoreConfig cfg;
        cfg.n_layer = 3;
        cfg.block_tokens = 32;
        cfg.k_gpu_row_bytes = 34;
        cfg.v_gpu_row_bytes = 68;
        d.layout = raw_kv_payload_layout(cfg, "llama-packed-q8-fixture", {0, 2});
    } else {
        auto & l = d.layout;
        l.format = profile == 1 ? "ninfer-bf16-fixture" : "ninfer-int8-g64-fixture";
        l.version = 1;
        l.block_tokens = 64;
        l.alignment = 256;
        // Two attention layers of the real D256/KV4 geometry, opaque per-page planes.
        for (uint32_t layer : {3u, 7u}) {
            for (auto kind : {PayloadPlaneKind::Key, PayloadPlaneKind::Value,
                              PayloadPlaneKind::KeyScale, PayloadPlaneKind::ValueScale}) {
                const bool scale = kind == PayloadPlaneKind::KeyScale || kind == PayloadPlaneKind::ValueScale;
                if (scale && profile == 1) continue;
                const uint64_t bytes = scale ? 64 * 4 * 4 * 2 : 64 * 4 * 256 * (profile == 1 ? 2 : 1);
                l.planes.push_back({layer, kind, l.record_bytes, bytes, 256, 0});
                l.record_bytes += bytes;
            }
        }
    }
    d.capabilities.page_tokens = d.layout.block_tokens; // slot allocation quantum for these fixtures
    d.capabilities.max_position = 262144;
    d.capabilities.native_payload = true;
    d.capabilities.sparse_view = true;
    d.capabilities.original_rope_positions = true;
    d.capabilities.state_checkpoint = true;
    d.capabilities.asynchronous_transfer = true;
    d.validate();
    return d;
}
inline BlockDescriptor block(const MemoryDescriptor & d, uint64_t id, uint32_t tokens = 0) {
    return {{42, 3}, id, id * d.layout.block_tokens, tokens ? tokens : d.layout.block_tokens, 1, 7};
}
inline NativeKvPayload payload(const MemoryDescriptor & d, BlockDescriptor b) {
    std::vector<uint8_t> bytes(static_cast<size_t>(d.layout.record_bytes));
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 37 + b.id * 11 + b.content_version);
    return NativeKvPayload(d.identity, d.layout, b, std::move(bytes));
}
