#include "kvmem/kv_content.hpp"
#include <algorithm>
#include <atomic>
#include <limits>

namespace kvmem {
namespace { std::atomic<uint64_t> next_session{1}; }
KvContent::KvContent(uint32_t block_tokens) : block_tokens_(block_tokens) {
    if (!block_tokens) throw MemoryError(MemoryErrorCode::InvalidPlan, "zero content block size");
    stamp_.session.id = next_session.fetch_add(1);
    reset();
}
void KvContent::reset() noexcept {
    revoke_prefixes(0, true);
    ++stamp_.session.incarnation;
    ++stamp_.history_revision;
    ++stamp_.execution_history;
    ++stamp_.view_version;
    stamp_.valid = true;
    stamp_.frontier = reserved_ = evaluated_ = 0;
    versions_.clear();
}
void KvContent::revoke_prefixes(uint64_t keep, bool all) noexcept {
    prefixes_.erase(std::remove_if(prefixes_.begin(), prefixes_.end(), [&](const auto & weak) {
        auto p = weak.lock();
        if (!p) return true;
        if (all || p->frontier_ > keep) { p->valid_ = false; return true; }
        return false;
    }), prefixes_.end());
}
void KvContent::invalidate() noexcept {
    stamp_.valid = false;
    revoke_prefixes(0, true);
}
std::shared_ptr<const ContentPrefix> KvContent::prefix(uint64_t end) {
    if (!stamp_.valid || end > stamp_.frontier)
        throw MemoryError(MemoryErrorCode::Pending, "prefix exceeds accepted content");
    revoke_prefixes(stamp_.frontier);
    auto p = std::make_shared<ContentPrefix>();
    p->session_ = stamp_.session;
    p->frontier_ = end;
    prefixes_.push_back(p);
    return p;
}
bool KvContent::contains(const std::shared_ptr<const ContentPrefix> & p) const noexcept {
    return stamp_.valid && p && p->valid_ && p->session_ == stamp_.session && p->frontier_ <= stamp_.frontier;
}
void KvContent::reserve(uint64_t end) {
    if (!stamp_.valid) throw MemoryError(MemoryErrorCode::InvalidSession, "content session invalid");
    if (end < reserved_ || end > UINT32_MAX)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid content reservation");
    versions_.resize(end / block_tokens_ + (end % block_tokens_ != 0));
    if (reserved_ != end) ++stamp_.history_revision;
    reserved_ = end;
}
void KvContent::complete(uint64_t begin, uint64_t end, bool accepted) {
    if (!stamp_.valid) throw MemoryError(MemoryErrorCode::InvalidSession, "content session invalid");
    if (begin > stamp_.frontier || begin >= end || end > reserved_)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "completion leaves a hole or exceeds reservation");
    const uint64_t revision = ++stamp_.history_revision;
    for (uint64_t id = begin / block_tokens_; id <= (end - 1) / block_tokens_; ++id)
        versions_[id] = revision;
    evaluated_ = std::max(evaluated_, end);
    if (accepted) stamp_.frontier = std::max(stamp_.frontier, end);
}
void KvContent::accept(uint64_t end) {
    if (!stamp_.valid || end < stamp_.frontier || end > evaluated_)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "acceptance exceeds evaluated KV");
    if (end != stamp_.frontier) ++stamp_.history_revision;
    stamp_.frontier = end;
}
void KvContent::truncate(uint64_t end) {
    if (end > reserved_) throw MemoryError(MemoryErrorCode::InvalidPlan, "truncate exceeds reservation");
    if (end == reserved_) return;
    revoke_prefixes(end);
    ++stamp_.history_revision;
    ++stamp_.execution_history;
    reserved_ = end;
    stamp_.frontier = std::min(stamp_.frontier, end);
    evaluated_ = std::min(evaluated_, end);
    versions_.resize(end / block_tokens_ + (end % block_tokens_ != 0));
    if (end % block_tokens_ && end <= stamp_.frontier)
        versions_.back() = stamp_.history_revision;
}
BlockDescriptor KvContent::block(uint64_t id) const {
    const uint64_t count = stamp_.frontier / block_tokens_ + (stamp_.frontier % block_tokens_ != 0);
    if (!stamp_.valid || id >= count || !versions_[id])
        throw MemoryError(MemoryErrorCode::Stale, "block has no completed KV");
    const uint64_t begin = id * block_tokens_;
    return {stamp_.session, id, begin, uint32_t(std::min<uint64_t>(block_tokens_, stamp_.frontier - begin)),
            versions_[id], 0}; // statistics are not certified by the KV completion fence
}
void KvContent::publish(const MemoryStamp & expected, const MemoryStamp & next) {
    auto wanted = expected;
    ++wanted.view_version;
    if (!(stamp_ == expected) || !(next == wanted))
        throw MemoryError(MemoryErrorCode::Stale, "content changed before view publication");
    stamp_ = next;
}
} // namespace kvmem
