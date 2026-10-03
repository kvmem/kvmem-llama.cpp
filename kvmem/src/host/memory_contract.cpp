#include "kvmem/memory_contract.hpp"
#include "kvmem/kvmem_store.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace kvmem {
namespace {
[[noreturn]] void fail(MemoryErrorCode code, const char * message) {
    throw MemoryError(code, message);
}
uint64_t add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        fail(MemoryErrorCode::InvalidPlan, "memory size/position overflow");
    return a + b;
}
uint64_t mul(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a)
        fail(MemoryErrorCode::InvalidPlan, "memory size overflow");
    return a * b;
}
bool power_of_two(uint64_t n) { return n && !(n & (n - 1)); }
uint64_t page_ceil(uint64_t n, uint32_t page) {
    return mul(n / page + (n % page != 0), page);
}
void valid_identity(const BackendIdentity & id) {
    if (id.backend.empty() || id.revision.empty() || id.model.empty() || id.state_schema.empty())
        fail(MemoryErrorCode::InvalidPlan, "backend/model/state identity is incomplete");
}
} // namespace

bool BackendIdentity::operator==(const BackendIdentity & b) const {
    return std::tie(backend, revision, model, state_schema) ==
           std::tie(b.backend, b.revision, b.model, b.state_schema);
}
bool SessionIdentity::operator==(const SessionIdentity & b) const {
    return id == b.id && incarnation == b.incarnation;
}
bool PayloadPlane::operator==(const PayloadPlane & b) const {
    return std::tie(layer, kind, offset, bytes, alignment, token_stride) ==
           std::tie(b.layer, b.kind, b.offset, b.bytes, b.alignment, b.token_stride);
}
bool PayloadLayout::operator==(const PayloadLayout & b) const {
    return std::tie(format, version, block_tokens, record_bytes, alignment, planes) ==
           std::tie(b.format, b.version, b.block_tokens, b.record_bytes, b.alignment, b.planes);
}
void PayloadLayout::validate() const {
    if (format.empty() || !version || !block_tokens || !record_bytes || planes.empty() ||
        !power_of_two(alignment) || record_bytes % alignment)
        fail(MemoryErrorCode::InvalidPlan, "invalid native payload layout");
    uint64_t end = 0;
    std::map<uint32_t, std::set<PayloadPlaneKind>> layers;
    for (const auto & p : planes) {
        if (!p.bytes || !power_of_two(p.alignment) || p.alignment > alignment ||
            p.offset % p.alignment || p.offset < end ||
            !layers[p.layer].insert(p.kind).second)
            fail(MemoryErrorCode::InvalidPlan, "overlapping/duplicate/misaligned payload plane");
        switch (p.kind) {
        case PayloadPlaneKind::Key: case PayloadPlaneKind::Value:
        case PayloadPlaneKind::KeyScale: case PayloadPlaneKind::ValueScale: break;
        default: fail(MemoryErrorCode::InvalidPlan, "unknown payload plane kind");
        }
        end = add(p.offset, p.bytes);
        if (end > record_bytes || (p.token_stride && mul(p.token_stride, block_tokens) > p.bytes))
            fail(MemoryErrorCode::InvalidPlan, "payload plane exceeds its record");
    }
    for (const auto & layer : layers) {
        if (!layer.second.count(PayloadPlaneKind::Key) || !layer.second.count(PayloadPlaneKind::Value))
            fail(MemoryErrorCode::InvalidPlan, "every attention layer requires key and value planes");
    }
}
void MemoryDescriptor::validate() const {
    valid_identity(identity);
    layout.validate();
    if (!capabilities.page_tokens || !capabilities.max_position ||
        layout.block_tokens % capabilities.page_tokens)
        fail(MemoryErrorCode::Unsupported, "selection block must be a whole number of native pages");
    if (!capabilities.native_payload || !capabilities.sparse_view || !capabilities.original_rope_positions)
        fail(MemoryErrorCode::Unsupported, "backend lacks the native working-set contract");
}
bool BlockDescriptor::operator==(const BlockDescriptor & b) const {
    return session == b.session &&
           std::tie(id, original_start, valid_tokens, content_version, statistics_version) ==
           std::tie(b.id, b.original_start, b.valid_tokens, b.content_version, b.statistics_version);
}
bool MemoryStamp::operator==(const MemoryStamp & b) const {
    return valid == b.valid && session == b.session &&
           std::tie(history_revision, view_version, frontier, execution_history) ==
           std::tie(b.history_revision, b.view_version, b.frontier, b.execution_history);
}
bool CheckpointIdentity::compatible_with(const CheckpointIdentity & b) const {
    if (!stamp.session.id || !stamp.session.incarnation || !stamp.history_revision || !stamp.execution_history)
        return false;
    try { valid_identity(backend); layout.validate(); }
    catch (const MemoryError &) { return false; }
    return stamp.valid && b.stamp.valid && backend == b.backend && layout == b.layout && stamp == b.stamp;
}

ResidencyDelta plan_residency(const std::vector<uint64_t> & resident,
    const std::vector<uint64_t> & selected, bool force_reload) {
    const std::set<uint64_t> current(resident.begin(), resident.end());
    const std::set<uint64_t> chosen(selected.begin(), selected.end());
    if (current.size() != resident.size() || chosen.size() != selected.size())
        fail(MemoryErrorCode::InvalidPlan, "duplicate block in residency selection");
    ResidencyDelta delta;
    for (auto id : current) {
        if (force_reload || !chosen.count(id)) delta.evict.push_back(id);
        else delta.retain.push_back(id);
    }
    for (auto id : chosen) {
        if (force_reload || !current.count(id)) delta.restore.push_back(id);
    }
    return delta;
}

WorkingSetPlan plan_working_set(const MemoryDescriptor & d, const MemorySnapshot & s,
    const std::vector<uint64_t> & selected, const std::vector<uint64_t> & mandatory,
    const MemoryBudget & budget) {
    d.validate();
    if (!s.stamp.valid) fail(MemoryErrorCode::InvalidSession, "session needs a fresh prefill");
    if (!s.stamp.session.id || !s.stamp.session.incarnation || !s.stamp.history_revision ||
        !s.stamp.execution_history || s.stamp.frontier > d.capabilities.max_position)
        fail(MemoryErrorCode::InvalidPlan, "invalid logical history identity/frontier");
    const auto page = d.capabilities.page_tokens;
    if (!budget.device_tokens || budget.device_tokens % page || budget.selected_tokens % page)
        fail(MemoryErrorCode::InvalidPlan, "working-set budget must respect native page granularity");
    std::map<uint64_t, const BlockCopies *> catalog;
    uint64_t frontier = 0, host_records = 0;
    for (const auto & item : s.blocks) {
        const auto & b = item.block;
        if (!(b.session == s.stamp.session) || !b.content_version || !b.valid_tokens ||
            b.valid_tokens > d.layout.block_tokens || b.original_start != frontier ||
            b.original_start % d.layout.block_tokens || item.host_version > b.content_version ||
            !catalog.emplace(b.id, &item).second)
            fail(MemoryErrorCode::InvalidPlan, "invalid block catalog or foreign session");
        frontier = add(b.original_start, b.valid_tokens);
        host_records += item.host_version != 0;
    }
    if (frontier != s.stamp.frontier)
        fail(MemoryErrorCode::Stale, "block catalog does not match committed logical frontier");
    std::set<uint64_t> resident, chosen, required;
    for (auto id : s.resident) {
        if (!catalog.count(id) || !resident.insert(id).second)
            fail(MemoryErrorCode::InvalidPlan, "invalid resident block set");
    }
    for (auto id : selected) {
        if (!catalog.count(id) || !chosen.insert(id).second)
            fail(MemoryErrorCode::InvalidPlan, "unknown or duplicate selected block");
    }
    for (auto id : mandatory) {
        if (!required.insert(id).second || !chosen.count(id))
            fail(MemoryErrorCode::InvalidPlan, "mandatory blocks must appear exactly once in selection");
    }
    const auto delta = plan_residency(s.resident, selected);
    const std::set<uint64_t> evicted(delta.evict.begin(), delta.evict.end());
    const std::set<uint64_t> restored(delta.restore.begin(), delta.restore.end());
    WorkingSetPlan plan;
    plan.expected = s.stamp;
    plan.budget = budget;
    plan.mandatory = mandatory;
    plan.next.stamp = s.stamp;
    plan.next.stamp.view_version = add(s.stamp.view_version, 1);
    plan.device_tokens_required = page_ceil(budget.reserve_tokens, page);
    // Iterate logical catalog, independent of selector rank/order and physical slots.
    for (const auto & item : s.blocks) {
        const auto & b = item.block;
        if (!resident.count(b.id) && item.host_version != b.content_version)
            fail(MemoryErrorCode::Stale, "cold history has no completed current host copy");
        if (chosen.count(b.id)) {
            plan.next.blocks.push_back({b, plan.next.visible_tokens});
            plan.next.visible_tokens = add(plan.next.visible_tokens, b.valid_tokens);
            plan.device_tokens_required = add(plan.device_tokens_required, page_ceil(b.valid_tokens, page));
            if (restored.count(b.id)) {
                plan.restore.push_back(b);
            }
        } else if (evicted.count(b.id)) {
            plan.evict.push_back(b);
            if (item.host_version != b.content_version) {
                plan.spill.push_back(b);
                host_records += item.host_version == 0;
            }
        }
    }
    plan.host_payload_bytes_required = mul(host_records, d.layout.record_bytes);
    if (plan.next.visible_tokens > budget.selected_tokens ||
        plan.device_tokens_required > budget.device_tokens ||
        plan.host_payload_bytes_required > budget.host_payload_bytes)
        fail(MemoryErrorCode::BudgetExceeded, "selection/reserve/host payload budget exceeded");
    return plan;
}

MemorySession::MemorySession(std::shared_ptr<MemoryBackend> backend) : backend_(std::move(backend)) {
    if (!backend_) fail(MemoryErrorCode::Disconnected, "a concrete memory backend is required");
    descriptor_ = backend_->descriptor();
    descriptor_.validate();
}
MemorySession::~MemorySession() { abort(); }
void MemorySession::require_attached() const {
    if (!backend_ || phase_ == MemoryPhase::Detached)
        fail(MemoryErrorCode::Disconnected, "memory backend is disconnected");
    if (phase_ == MemoryPhase::Invalid || !backend_->stamp().valid)
        fail(MemoryErrorCode::InvalidSession, "session invalidated; create a fresh engine session");
}
void MemorySession::require_current() const {
    require_attached();
    if (!plan_ || !(backend_->stamp() == plan_->expected))
        fail(MemoryErrorCode::Stale, "history/view changed while a transfer was prepared");
}
const WorkingSetPlan & MemorySession::prepare_snapshot(const MemorySnapshot & snapshot,
    const std::vector<uint64_t> & selected, const std::vector<uint64_t> & mandatory,
    const MemoryBudget & budget) {
    auto candidate = plan_working_set(descriptor_, snapshot, selected, mandatory, budget);
    plan_ = std::move(candidate);
    try {
        require_current();
        transfer_ = backend_->prepare(*plan_);
        if (!transfer_) fail(MemoryErrorCode::TransferFailed, "backend returned no prepared transfer");
        require_current();
        phase_ = MemoryPhase::Prepared;
        return *plan_;
    } catch (...) {
        failed();
        throw;
    }
}
const WorkingSetPlan & MemorySession::prepare(const MemoryStamp & expected,
    const std::vector<uint64_t> & selected, const std::vector<uint64_t> & mandatory,
    const MemoryBudget & budget) {
    require_attached();
    if (transfer_) fail(MemoryErrorCode::Pending, "a working-set transfer is already pending");
    auto snapshot = backend_->snapshot();
    if (!(snapshot.stamp == expected)) fail(MemoryErrorCode::Stale, "selection stamp is stale");
    return prepare_snapshot(snapshot, selected, mandatory, budget);
}
const WorkingSetPlan & MemorySession::prepare_reselect(const MemoryStamp & expected,
    const KvMemStore & policy, const std::vector<uint32_t> & mandatory, const MemoryBudget & budget) {
    require_attached();
    if (transfer_) fail(MemoryErrorCode::Pending, "a working-set transfer is already pending");
    auto snapshot = backend_->snapshot();
    if (!(snapshot.stamp == expected)) fail(MemoryErrorCode::Stale, "selection stamp is stale");
    if (policy.config().block_tokens != descriptor_.layout.block_tokens ||
        policy.blocks().size() != snapshot.blocks.size())
        fail(MemoryErrorCode::Stale, "policy and native block catalog differ");
    if (policy.config().select_method == KvMemMethod::Retrieval &&
        !descriptor_.capabilities.pre_rope_statistics)
        fail(MemoryErrorCode::Unsupported, "backend has no pre-RoPE retrieval statistics capability");
    for (size_t i = 0; i < snapshot.blocks.size(); ++i) {
        const auto & p = policy.blocks()[i];
        const auto & b = snapshot.blocks[i].block;
        if (p.block_id != b.id || p.orig_pos_start != b.original_start || p.n_tokens != b.valid_tokens)
            fail(MemoryErrorCode::Stale, "policy and native token ranges differ");
    }
    // Preserve the existing ranking algorithm, but never call the mutating legacy set_selection.
    const auto selected = policy.pick_topk_blocks(mandatory);
    return prepare_snapshot(snapshot, {selected.begin(), selected.end()},
                             {mandatory.begin(), mandatory.end()}, budget);
}
MemoryPhase MemorySession::advance() {
    try {
        require_attached();
        if (!transfer_) fail(MemoryErrorCode::InvalidPlan, "no prepared working-set transfer");
        require_current();
        if (phase_ == MemoryPhase::Prepared) {
            phase_ = MemoryPhase::Spilling;
            transfer_->start_spill();
        } else if (phase_ == MemoryPhase::Spilling || phase_ == MemoryPhase::Restoring) {
            const auto result = transfer_->poll();
            require_current();
            if (result.status == TransferStatus::Failed)
                throw MemoryError(MemoryErrorCode::TransferFailed, result.error);
            if (result.status == TransferStatus::Complete) {
                if (phase_ == MemoryPhase::Spilling) {
                    phase_ = MemoryPhase::Restoring;
                    transfer_->start_restore();
                } else {
                    phase_ = MemoryPhase::Ready;
                }
            }
        }
        return phase_;
    } catch (...) {
        failed();
        throw;
    }
}
ExecutionView MemorySession::publish() {
    try { require_attached(); } catch (...) { failed(); throw; }
    if (phase_ != MemoryPhase::Ready)
        fail(MemoryErrorCode::Pending, "cannot publish before every transfer completes");
    // Allocate the returned view before publishing anything to the engine.
    auto result = plan_->next;
    bool publishing = false;
    try {
        require_current();
        publishing = true;
        transfer_->publish(result);
        if (!(backend_->stamp() == result.stamp))
            fail(MemoryErrorCode::TransferFailed, "backend failed to publish the expected view version");
        transfer_.reset();
        plan_.reset();
        phase_ = MemoryPhase::Published;
        return result;
    } catch (...) {
        failed(publishing); // publication failure cannot promise a usable previous view
        throw;
    }
}
AbortResult MemorySession::abort() noexcept {
    auto result = phase_ == MemoryPhase::Invalid ? AbortResult::SessionInvalidated
                                                : AbortResult::PreviousViewPreserved;
    if (transfer_) result = transfer_->abort();
    if (backend_ && !backend_->stamp().valid) result = AbortResult::SessionInvalidated;
    transfer_.reset();
    plan_.reset();
    if (result == AbortResult::SessionInvalidated) {
        if (backend_) backend_->invalidate();
        phase_ = MemoryPhase::Invalid;
    } else if (phase_ != MemoryPhase::Detached && phase_ != MemoryPhase::Published) {
        phase_ = MemoryPhase::Idle;
    }
    return result;
}
void MemorySession::failed(bool force_invalidate) noexcept {
    abort();
    if (force_invalidate) {
        if (backend_) backend_->invalidate();
        phase_ = MemoryPhase::Invalid;
    }
}
void MemorySession::disconnect() noexcept {
    abort();
    backend_.reset();
    phase_ = MemoryPhase::Detached;
}
} // namespace kvmem
