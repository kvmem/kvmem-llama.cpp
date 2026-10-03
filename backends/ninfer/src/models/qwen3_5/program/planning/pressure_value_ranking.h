#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// One value weight per pressure victim: a private victim gets its 0-based rank among the private
// victims ordered by ascending rebuild cost (the cheapest 0), a shared victim 0. Ties keep victim
// order, so the ranking is deterministic. `is_shared` and `rebuild_cost` have equal sizes, and a
// shared victim's cost is ignored.
[[nodiscard]] inline std::vector<std::uint32_t>
value_weights_for_victims(std::span<const std::uint8_t> is_shared,
                          std::span<const std::uint64_t> rebuild_cost) {
    std::vector<std::uint32_t> weights(is_shared.size(), 0U);
    std::vector<std::tuple<std::uint64_t, std::uint32_t>> ranked;
    ranked.reserve(is_shared.size());
    for (std::size_t index = 0; index < is_shared.size(); ++index) {
        if (is_shared[index] == 0U) {
            ranked.emplace_back(rebuild_cost[index], static_cast<std::uint32_t>(index));
        }
    }
    std::sort(ranked.begin(), ranked.end());
    for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
        weights[std::get<1>(ranked[rank])] = static_cast<std::uint32_t>(rank);
    }
    return weights;
}

} // namespace ninfer::models::qwen3_5::detail
