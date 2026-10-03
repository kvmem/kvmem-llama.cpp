#pragma once
#include "kvmem/memory_contract.hpp"

namespace kvmem {
// A logical-prefix lease, not a claim about physical KV or recurrent state.
// Append and execution replay preserve it. Truncation through its prefix,
// reset and invalidation revoke it permanently, including after regrowth.
class ContentPrefix {
public:
    uint64_t frontier() const noexcept { return frontier_; }
private:
    friend class KvContent;
    SessionIdentity session_;
    uint64_t frontier_ = 0;
    bool valid_ = true;
};
// Per-conversation content, independent of physical slots and policy reservations.
// complete() is called only after the engine's graph completion fence. Speculative
// evaluation does not advance the committed frontier until accept(). truncate()
// removes rejected rows; accepted mean-K remains a separate engine operation.
class KvContent {
public:
    explicit KvContent(uint32_t block_tokens);
    KvContent(const KvContent &) = delete;
    KvContent & operator=(const KvContent &) = delete;
    const MemoryStamp & stamp() const noexcept { return stamp_; }
    uint64_t reserved() const noexcept { return reserved_; }
    uint64_t evaluated() const noexcept { return evaluated_; }
    uint64_t allocated_bytes() const noexcept {
        return sizeof(*this) + versions_.capacity()*sizeof(uint64_t) +
            prefixes_.capacity()*sizeof(std::weak_ptr<ContentPrefix>) + prefixes_.size()*sizeof(ContentPrefix);
    }
    std::shared_ptr<const ContentPrefix> prefix(uint64_t end);
    bool contains(const std::shared_ptr<const ContentPrefix> & prefix) const noexcept;
    void reserve(uint64_t end);
    void complete(uint64_t begin, uint64_t end, bool accepted = true);
    void accept(uint64_t end);
    void truncate(uint64_t end);
    void reset() noexcept;
    void invalidate() noexcept;
    void change_execution() noexcept { ++stamp_.history_revision; ++stamp_.execution_history; }
    void publish(const MemoryStamp & expected, const MemoryStamp & next);
    BlockDescriptor block(uint64_t id) const;
private:
    uint32_t block_tokens_;
    uint64_t reserved_ = 0;
    uint64_t evaluated_ = 0;
    MemoryStamp stamp_;
    std::vector<uint64_t> versions_;
    std::vector<std::weak_ptr<ContentPrefix>> prefixes_;
    void revoke_prefixes(uint64_t keep, bool all = false) noexcept;
};
} // namespace kvmem
