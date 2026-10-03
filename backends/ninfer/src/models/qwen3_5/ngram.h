#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Per-request views, independent of the archive's retained per-session source limit.
inline constexpr std::size_t kNgramRequestSourceCapacity = 1024;
inline constexpr std::size_t kNgramMatchHistoryTokens    = 64;

enum class NgramSourceKind : std::uint8_t { Reasoning, Text, Generated, Tool };

struct NgramSourceView {
    std::span<const TokenId> tokens;
    NgramSourceKind kind = NgramSourceKind::Text;
};

struct NgramMatch {
    std::vector<TokenId> tokens;
    std::uint32_t matched = 0;
    std::uint64_t source  = 0;
    std::uint32_t offset  = 0;
    NgramSourceKind kind  = NgramSourceKind::Text;
    bool archived         = false;
};

struct NgramArchiveLimits {
    std::size_t session_bytes = 128ULL << 20;
    std::size_t total_bytes   = 512ULL << 20;
    std::uint32_t sessions    = 32;
    std::uint32_t sources     = 1024;
};

class NgramSnapshot {
public:
    ~NgramSnapshot();
    [[nodiscard]] NgramMatch propose(std::span<const TokenId> history, std::uint32_t maximum,
                                     std::uint32_t minimum_match = 12) const;
    [[nodiscard]] NgramMatch continue_copy(std::uint64_t source, std::uint32_t offset,
                                           std::span<const TokenId> history, std::uint32_t maximum,
                                           std::uint32_t minimum_match = 12) const;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::size_t source_count() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] std::size_t session_bytes() const noexcept;

private:
    struct Impl;
    explicit NgramSnapshot(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class NgramArchive;
};

// Engine-owned CPU storage. A request's view is fixed at admission; its private
// input sources are published only on successful completion. No target state is retained.
// The Engine serializes mutations; immutable snapshot reads and releases may overlap.
class NgramArchive {
public:
    using Limits = NgramArchiveLimits;

    class Request {
    public:
        ~Request();
        [[nodiscard]] std::shared_ptr<const NgramSnapshot> snapshot() const noexcept;

    private:
        struct Impl;
        explicit Request(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
        friend class NgramArchive;
    };

    explicit NgramArchive(NgramArchiveLimits limits = {});
    ~NgramArchive();
    NgramArchive(const NgramArchive&)            = delete;
    NgramArchive& operator=(const NgramArchive&) = delete;

    // Unknown parents, stale generations and exhausted capacity fall back to no
    // archive. Fork inheritance is explicit; an existing destination is never merged.
    [[nodiscard]] std::unique_ptr<Request> begin(std::string_view key,
                                                 std::span<const NgramSourceView> input,
                                                 std::span<const TokenId> boundaries = {},
                                                 std::string_view parent             = {},
                                                 std::uint64_t parent_generation     = 0);
    bool publish(std::unique_ptr<Request> request, std::span<const NgramSourceView> output = {});
    bool publish(std::unique_ptr<Request> request, std::span<const TokenId> output,
                 std::uint32_t reasoning_tokens);
    void clear(std::string_view key) noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] NgramArchiveStats stats(std::string_view key) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen3_5
