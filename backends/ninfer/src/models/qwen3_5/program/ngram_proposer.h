#pragma once

#include "models/qwen3_5/ngram.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// A bounded, position-indexed proposal corpus. It never owns target state.
// Absolute positions make overwritten ring entries rejectable before a read.
class NgramProposer {
public:
    using Token = std::int32_t;

    using Match = NgramMatch;

    [[nodiscard]] static bool maximal(const Match& match, std::size_t history_size,
                                      std::uint32_t maximum) {
        return match.matched == history_size && match.tokens.size() == maximum;
    }

    explicit NgramProposer(std::size_t token_capacity = 1U << 20,
                           std::size_t bucket_count   = 1U << 19)
        : tokens_(token_capacity, -1), buckets_(bucket_count) {
        if (token_capacity < 64 || bucket_count == 0 || (bucket_count & (bucket_count - 1)) != 0) {
            throw std::invalid_argument("invalid ngram corpus capacity");
        }
    }

    void boundary() { append(-1); }

    void set_boundaries(std::vector<Token> tokens) {
        boundaries_ = std::move(tokens);
        std::sort(boundaries_.begin(), boundaries_.end());
    }

    void append(Token token) {
        if (std::binary_search(boundaries_.begin(), boundaries_.end(), token)) { token = -1; }
        tokens_[end_ % tokens_.size()] = token;
        ++end_;
        if (token < 0) {
            segment_start_ = end_;
            return;
        }
        for (const auto n : widths_) {
            if (end_ - segment_start_ < n || end_ - live_begin() < n) { continue; }
            auto& bucket = buckets_[hash_at(end_ - n, n) & (buckets_.size() - 1)];
            for (std::size_t i = bucket.size() - 1; i > 0; --i) { bucket[i] = bucket[i - 1]; }
            bucket[0] = {end_, n};
        }
    }

    void ingest(std::span<const Token> tokens) {
        boundary();
        for (const Token token : tokens) { append(token); }
        boundary();
    }

    [[nodiscard]] Match propose_for_round(std::span<const Token> history, std::uint32_t maximum,
                                          std::uint32_t neural_drafts,
                                          std::uint32_t minimum_match = 12) const {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        return finish_round(propose(history, maximum + 2U, minimum_match), maximum, neural_drafts);
    }

    [[nodiscard]] static Match finish_round(Match match, std::uint32_t maximum,
                                            std::uint32_t neural_drafts) {
        if (maximum == 0) { return {}; }
        if (maximum > std::numeric_limits<std::uint32_t>::max() - 2U) {
            throw std::invalid_argument("ngram proposal lookahead overflows");
        }
        const auto available = static_cast<std::uint32_t>(match.tokens.size());
        // The target emits a bonus after the drafts. Leave one source token beyond it
        // for neural completion; source endings may need different surrounding text.
        const auto drafts = std::min(maximum, available > 2U ? available - 2U : 0U);
        if (available < maximum + 2U && drafts <= neural_drafts) { return {}; }
        match.tokens.resize(drafts);
        return match;
    }

    [[nodiscard]] Match propose(std::span<const Token> history, std::uint32_t maximum,
                                std::uint32_t minimum_match = 12) const {
        Match best;
        if (maximum == 0) { return best; }
        for (const auto n : widths_) {
            if (history.size() < n) { continue; }
            const auto tail = history.last(n);
            if (std::find_if(tail.begin(), tail.end(), [](Token t) { return t < 0; }) !=
                tail.end()) {
                continue;
            }
            const auto& bucket = buckets_[hash(tail, n) & (buckets_.size() - 1)];
            for (const auto entry : bucket) {
                if (entry.width != n || entry.end < n || entry.end - n < live_begin() ||
                    entry.end >= end_) {
                    continue;
                }
                bool equal = true;
                for (std::uint32_t j = 0; j < n; ++j) {
                    if (at(entry.end - n + j) != tail[j]) {
                        equal = false;
                        break;
                    }
                }
                if (!equal) { continue; }
                std::uint32_t matched = n;
                while (matched < history.size() && entry.end - matched > live_begin()) {
                    const auto token = at(entry.end - matched - 1);
                    if (token < 0 || token != history[history.size() - matched - 1]) { break; }
                    ++matched;
                }
                if (matched < minimum_match || matched < best.matched) { continue; }
                std::vector<Token> draft;
                for (std::uint64_t p = entry.end; p < end_ && draft.size() < maximum; ++p) {
                    const auto token = at(p);
                    if (token < 0) { break; }
                    draft.push_back(token);
                }
                if (!draft.empty() &&
                    (matched > best.matched || draft.size() > best.tokens.size())) {
                    best = {std::move(draft), matched};
                }
            }
        }
        return best;
    }

private:
    struct Entry {
        std::uint64_t end   = 0;
        std::uint32_t width = 0;
    };

    static constexpr std::array<std::uint32_t, 3> widths_{16, 8, 4};
    std::vector<Token> tokens_;
    std::vector<Token> boundaries_;
    std::vector<std::array<Entry, 4>> buckets_;
    std::uint64_t end_           = 0;
    std::uint64_t segment_start_ = 0;

    [[nodiscard]] std::uint64_t live_begin() const {
        return end_ > tokens_.size() ? end_ - tokens_.size() : 0;
    }

    [[nodiscard]] Token at(std::uint64_t p) const { return tokens_[p % tokens_.size()]; }

    static std::uint64_t mix(std::uint64_t h, Token token) {
        return (h ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
    }

    static std::uint64_t finalize_hash(std::uint64_t value) {
        // Buckets mask low bits; avalanche patterned IDs before selecting a bucket.
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    static std::uint64_t hash(std::span<const Token> tokens, std::uint32_t n) {
        std::uint64_t h = 1469598103934665603ULL ^ n;
        for (auto token : tokens) { h = mix(h, token); }
        return finalize_hash(h);
    }

    [[nodiscard]] std::uint64_t hash_at(std::uint64_t p, std::uint32_t n) const {
        std::uint64_t h = 1469598103934665603ULL ^ n;
        for (std::uint32_t j = 0; j < n; ++j) { h = mix(h, at(p + j)); }
        return finalize_hash(h);
    }
};

} // namespace ninfer::models::qwen3_5::detail
