#include "memory_test_support.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

static void native_planes(int profile) {
    const auto d = descriptor(profile);
    auto b = block(d, 5, 1);
    NativeKvArchive archive(d.identity, b.session, d.layout, d.layout.record_bytes);
    const auto original = payload(d, b);
    archive.put(original);
    CHECK(archive.bytes() == d.layout.record_bytes);
    CHECK(archive.get(b).bytes() == original.bytes()); // all planes AND padding, exact bytes
    for (const auto & p : d.layout.planes) {
        CHECK(std::memcmp(archive.get(b).bytes().data() + p.offset,
                          original.bytes().data() + p.offset, static_cast<size_t>(p.bytes)) == 0);
    }
    ++b.statistics_version;
    CHECK(archive.get(b).bytes() == original.bytes()); // retrieval stats never alias native KV versions
    rejects(MemoryErrorCode::BudgetExceeded, [&] { archive.put(payload(d, block(d, 6))); });
    auto changed = original.bytes(); changed.back() ^= 1;
    rejects(MemoryErrorCode::Stale, [&] { archive.put(NativeKvPayload(d.identity, d.layout, b, changed)); });
    CHECK(archive.get(b).bytes() == original.bytes());
    ++b.content_version;
    rejects(MemoryErrorCode::Stale, [&] { archive.get(b); });
    archive.put(payload(d, b));
    rejects(MemoryErrorCode::Stale, [&] { archive.put(original); });
    auto other = d; other.identity.model = "same-shape-other-model";
    rejects(MemoryErrorCode::InvalidPlan, [&] { archive.put(payload(other, b)); });
    other = d; ++other.layout.version;
    rejects(MemoryErrorCode::InvalidPlan, [&] { archive.put(payload(other, b)); });
    auto foreign = b; ++foreign.session.incarnation;
    rejects(MemoryErrorCode::InvalidPlan, [&] { archive.put(payload(d, foreign)); });
    rejects(MemoryErrorCode::Stale, [&] { archive.get(foreign); });
    auto shorter = b; shorter.valid_tokens = 2;
    rejects(MemoryErrorCode::Stale, [&] { archive.get(shorter); });
    auto short_bytes = original.bytes(); short_bytes.pop_back();
    rejects(MemoryErrorCode::InvalidPlan, [&] { NativeKvPayload bad(d.identity, d.layout, b, short_bytes); });
    archive.erase(b.id);
    CHECK(archive.bytes() == 0);
    rejects(MemoryErrorCode::Stale, [&] { archive.get(b); });
}
static void invalid_layouts() {
    const auto d = descriptor(2);
    for (int i = 0; i < 7; ++i) {
        auto l = d.layout;
        if (i == 0) l.planes[1].offset = l.planes[0].offset;
        if (i == 1) l.planes[1].kind = l.planes[0].kind;
        if (i == 2) l.planes.back().bytes = l.record_bytes;
        if (i == 3) l.alignment = 3;
        if (i == 4) l.planes[1].alignment = 512;
        if (i == 5) l.planes[0].token_stride = std::numeric_limits<uint64_t>::max();
        if (i == 6) l.planes.erase(l.planes.begin()); // missing K plane for first layer
        rejects(MemoryErrorCode::InvalidPlan, [&] { l.validate(); });
    }
}
static void raw_bridge(uint32_t block_tokens, uint32_t valid_tokens) {
    RawKvStoreConfig cfg;
    cfg.block_tokens = block_tokens;
    cfg.n_layer = 4; cfg.n_embd_k = 4; cfg.n_embd_v = 4;
    cfg.k_row_bytes = 34; cfg.k_gpu_row_bytes = 34; cfg.v_gpu_row_bytes = 68;
    RawKvStore raw(cfg), restored(cfg);
    const BackendIdentity identity{"llama-test", "base", "model", "gdn-state-v1"};
    const auto layout = raw_kv_payload_layout(cfg, "packed-q8-rows", {1, 3});
    const BlockDescriptor b{{71, 1}, 1, block_tokens, valid_tokens, 4, 8};
    std::vector<uint8_t> k(valid_tokens * 34), v(valid_tokens * 68);
    for (size_t i = 0; i < k.size(); ++i) k[i] = static_cast<uint8_t>(i * 3 + 19);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(i * 7 + 31);
    std::vector<float> stats(valid_tokens * 4);
    for (size_t i = 0; i < stats.size(); ++i) stats[i] = float(int(i % 17) - 8) * 0.25f;
    for (auto layer : {1u, 3u}) {
        raw.write_layer_k_gpu(block_tokens, valid_tokens, layer, k.data());
        raw.write_layer_v_gpu(block_tokens, valid_tokens, layer, v.data());
        raw.write_layer_mean_k(block_tokens, valid_tokens, layer, stats.data());
        // Target retrieval history belongs to the target state, not the incoming KV record.
        std::vector<float> target_stats(valid_tokens * 4, 3.25f);
        restored.write_layer_mean_k(block_tokens, valid_tokens, layer, target_stats.data());
    }
    std::vector<float> before(4), after(4), source_before(4), source_after(4);
    restored.mean_k(1, 1, before.data());
    raw.mean_k(1, 1, source_before.data());
    const auto native = export_raw_kv_payload(raw, identity, layout, b);
    for (const auto & plane : layout.planes) {
        const auto & expected = plane.kind == PayloadPlaneKind::Key ? k : v;
        CHECK(std::memcmp(native.bytes().data() + plane.offset, expected.data(), expected.size()) == 0);
        const auto tail = native.bytes().begin() + static_cast<size_t>(plane.offset) + expected.size();
        CHECK(std::all_of(tail, native.bytes().begin() + static_cast<size_t>(plane.offset + plane.bytes),
                          [](uint8_t x) { return x == 0; }));
    }
    import_raw_kv_payload(restored, identity, layout, native);
    CHECK(export_raw_kv_payload(restored, identity, layout, b).bytes() == native.bytes());
    restored.mean_k(1, 1, after.data()); raw.mean_k(1, 1, source_after.data());
    CHECK(before == after && source_before == source_after);
    CHECK(!restored.has_k_gpu(1, 0) && !restored.has_v_gpu(1, 2)); // no fake KV for GDN layers
    auto wrong = identity; wrong.model = "other";
    rejects(MemoryErrorCode::InvalidPlan, [&] { import_raw_kv_payload(restored, wrong, layout, native); });
    auto wrong_layout = layout; ++wrong_layout.version;
    rejects(MemoryErrorCode::InvalidPlan, [&] { import_raw_kv_payload(restored, identity, wrong_layout, native); });
    auto missing = b; missing.valid_tokens = std::min(block_tokens, valid_tokens + 1);
    if (valid_tokens < block_tokens)
        rejects(MemoryErrorCode::Stale, [&] { export_raw_kv_payload(raw, identity, layout, missing); });
    if (valid_tokens > 1) {
        auto short_block = b; short_block.valid_tokens = 1; ++short_block.content_version;
        auto short_payload = export_raw_kv_payload(raw, identity, layout, short_block);
        rejects(MemoryErrorCode::InvalidPlan, [&] { import_raw_kv_payload(restored, identity, layout, short_payload); });
        restored.invalidate_packed_from(block_tokens);
        import_raw_kv_payload(restored, identity, layout, short_payload);
        CHECK(!restored.has_k_gpu(1, 1, 2) && !restored.has_v_gpu(1, 1, 2));
        restored.mean_k(1, 1, after.data()); CHECK(before == after);
    }
    RawKvStore incomplete(cfg);
    incomplete.write_layer_k_gpu(block_tokens, valid_tokens, 1, k.data());
    rejects(MemoryErrorCode::Stale, [&] { export_raw_kv_payload(incomplete, identity, layout, b); });
}
int main() {
    for (int profile = 0; profile < 3; ++profile) native_planes(profile);
    invalid_layouts();
    for (uint32_t bt : {32u, 64u, 128u}) {
        raw_bridge(bt, 1); raw_bridge(bt, bt - 1); raw_bridge(bt, bt);
    }
    std::puts("native payload: exact planes, identity/version checks and nine raw/statistics roundtrips passed");
}
