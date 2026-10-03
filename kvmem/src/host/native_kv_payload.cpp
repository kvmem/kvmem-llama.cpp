#include "kvmem/native_kv_payload.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <limits>
#include <set>
#include <utility>

namespace kvmem {
namespace {
[[noreturn]] void invalid(const char * message) {
    throw MemoryError(MemoryErrorCode::InvalidPlan, message);
}
void validate_block(const PayloadLayout & layout, const BlockDescriptor & block) {
    if (!block.session.id || !block.session.incarnation || !block.content_version ||
        !block.valid_tokens || block.valid_tokens > layout.block_tokens ||
        block.original_start % layout.block_tokens ||
        block.original_start > std::numeric_limits<uint64_t>::max() - block.valid_tokens)
        invalid("invalid native payload block identity/range");
}
void validate_backend(const BackendIdentity & backend) {
    if (backend.backend.empty() || backend.revision.empty() || backend.model.empty() || backend.state_schema.empty())
        invalid("native payload requires complete backend/model/state identity");
}
void validate_raw(const RawKvStoreConfig & cfg, const PayloadLayout & layout, const BlockDescriptor & b) {
    layout.validate();
    validate_block(layout, b);
    if (layout.block_tokens != cfg.block_tokens || !cfg.k_gpu_row_bytes || !cfg.v_gpu_row_bytes ||
        b.id > std::numeric_limits<uint32_t>::max() ||
        b.original_start / cfg.block_tokens != b.id ||
        b.original_start + b.valid_tokens > std::numeric_limits<uint32_t>::max())
        invalid("native payload does not fit the legacy raw KV block addressing");
    for (const auto & p : layout.planes) {
        if (p.layer >= cfg.n_layer ||
            (p.kind != PayloadPlaneKind::Key && p.kind != PayloadPlaneKind::Value))
            invalid("raw KV bridge only accepts declared K/V token-row planes");
        const auto stride = p.kind == PayloadPlaneKind::Key ? cfg.k_gpu_row_bytes : cfg.v_gpu_row_bytes;
        if (p.token_stride != stride) invalid("raw KV packed row stride differs");
    }
}
} // namespace

NativeKvPayload::NativeKvPayload(BackendIdentity backend, PayloadLayout layout, BlockDescriptor block,
                                 std::vector<uint8_t> bytes)
    : backend_(std::move(backend)), layout_(std::move(layout)), block_(block), bytes_(std::move(bytes)) {
    validate_backend(backend_);
    layout_.validate();
    validate_block(layout_, block_);
    if (bytes_.size() != layout_.record_bytes) invalid("native payload record is truncated or oversized");
}
NativeKvArchive::NativeKvArchive(BackendIdentity backend, SessionIdentity session, PayloadLayout layout,
                                 uint64_t capacity_bytes)
    : backend_(std::move(backend)), session_(session), layout_(std::move(layout)), capacity_(capacity_bytes) {
    validate_backend(backend_);
    layout_.validate();
    if (!session.id || !session.incarnation) invalid("archive requires a session incarnation");
}
void NativeKvArchive::put(NativeKvPayload payload) {
    if (!(payload.backend() == backend_) || !(payload.block().session == session_) || !(payload.layout() == layout_))
        invalid("foreign session or incompatible native layout");
    const auto id = payload.block().id;
    auto current = records_.find(id);
    if (current != records_.end()) {
        const auto & old = current->second;
        if (payload.block().original_start != old.block().original_start ||
            payload.block().content_version < old.block().content_version)
            throw MemoryError(MemoryErrorCode::Stale, "native payload write is stale or aliases another range");
        if (payload.block().content_version == old.block().content_version) {
            if (payload.block().valid_tokens != old.block().valid_tokens || payload.bytes() != old.bytes())
                throw MemoryError(MemoryErrorCode::Stale, "different bytes/frontier under the same KV version");
            return; // statistics revision is deliberately independent of native KV storage
        }
        current->second = std::move(payload);
    } else {
        if (layout_.record_bytes > capacity_ - bytes_)
            throw MemoryError(MemoryErrorCode::BudgetExceeded, "host payload archive is full");
        records_.emplace(id, std::move(payload));
        bytes_ += layout_.record_bytes;
    }
}
const NativeKvPayload & NativeKvArchive::get(const BlockDescriptor & expected) const {
    const auto it = records_.find(expected.id);
    if (it == records_.end()) throw MemoryError(MemoryErrorCode::Stale, "host payload is missing");
    const auto & actual = it->second.block();
    if (!(expected.session == actual.session) || expected.original_start != actual.original_start ||
        expected.valid_tokens != actual.valid_tokens || expected.content_version != actual.content_version)
        throw MemoryError(MemoryErrorCode::Stale, "host payload identity/version/frontier differs");
    return it->second;
}
void NativeKvArchive::erase(uint64_t id) {
    if (records_.erase(id)) bytes_ -= layout_.record_bytes;
}
PayloadLayout raw_kv_payload_layout(const RawKvStoreConfig & cfg, const std::string & format,
                                    const std::vector<uint32_t> & layers) {
    if (!cfg.block_tokens || !cfg.k_gpu_row_bytes || !cfg.v_gpu_row_bytes || layers.empty())
        invalid("raw KV layout requires packed K/V and explicit attention layers");
    PayloadLayout layout;
    layout.format = format;
    layout.version = 1;
    layout.block_tokens = cfg.block_tokens;
    std::set<uint32_t> seen;
    for (auto layer : layers) {
        if (layer >= cfg.n_layer || !seen.insert(layer).second) invalid("invalid attention layer list");
        for (auto kind : {PayloadPlaneKind::Key, PayloadPlaneKind::Value}) {
            const auto stride = kind == PayloadPlaneKind::Key ? cfg.k_gpu_row_bytes : cfg.v_gpu_row_bytes;
            if (stride > (std::numeric_limits<uint64_t>::max() - layout.record_bytes) / cfg.block_tokens)
                invalid("raw KV layout size overflow");
            const auto bytes = stride * cfg.block_tokens;
            layout.planes.push_back({layer, kind, layout.record_bytes, bytes, 1, stride});
            layout.record_bytes += bytes;
        }
    }
    layout.validate();
    return layout;
}
NativeKvPayload export_raw_kv_payload(const RawKvStore & raw, const BackendIdentity & backend,
                                      const PayloadLayout & layout,
                                      const BlockDescriptor & b) {
    validate_raw(raw.config(), layout, b);
    if (layout.record_bytes > std::numeric_limits<size_t>::max()) invalid("payload exceeds host address space");
    std::vector<uint8_t> bytes(static_cast<size_t>(layout.record_bytes), 0);
    for (const auto & plane : layout.planes) {
        auto * dst = bytes.data() + static_cast<size_t>(plane.offset);
        const bool copied = plane.kind == PayloadPlaneKind::Key
            ? raw.copy_k_gpu(static_cast<uint32_t>(b.id), plane.layer, dst, b.valid_tokens)
            : raw.copy_v_gpu(static_cast<uint32_t>(b.id), plane.layer, dst, b.valid_tokens);
        if (!copied) throw MemoryError(MemoryErrorCode::Stale, "packed KV layer is missing or has a partial frontier");
    }
    return NativeKvPayload(backend, layout, b, std::move(bytes));
}
void import_raw_kv_payload(RawKvStore & raw, const BackendIdentity & backend, const PayloadLayout & layout,
                           const NativeKvPayload & payload) {
    if (!(payload.backend() == backend) || !(payload.layout() == layout))
        invalid("incompatible backend/model/layout at raw KV import");
    const auto & b = payload.block();
    validate_raw(raw.config(), payload.layout(), b);
    // Legacy writes extend packed validity. A rollback must invalidate its old
    // tail explicitly first, otherwise a short restore would expose stale rows.
    if (b.valid_tokens < raw.config().block_tokens) {
        for (const auto & p : layout.planes) {
            const bool longer = p.kind == PayloadPlaneKind::Key
                ? raw.has_k_gpu(static_cast<uint32_t>(b.id), p.layer, b.valid_tokens + 1)
                : raw.has_v_gpu(static_cast<uint32_t>(b.id), p.layer, b.valid_tokens + 1);
            if (longer) invalid("invalidate the legacy packed tail before a shorter restore");
        }
    }
    // All format/range checks precede mutation. Native packed writes do not update mean-K.
    for (const auto & plane : payload.layout().planes) {
        const auto * src = payload.bytes().data() + static_cast<size_t>(plane.offset);
        if (plane.kind == PayloadPlaneKind::Key)
            raw.write_layer_k_gpu(static_cast<uint32_t>(b.original_start), b.valid_tokens, plane.layer, src);
        else
            raw.write_layer_v_gpu(static_cast<uint32_t>(b.original_start), b.valid_tokens, plane.layer, src);
    }
}
} // namespace kvmem
