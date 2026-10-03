#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ninfer/ops/softmax_attention.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {
std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> dflash_base_profiles(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6) { add_target_boundary(draft_window <= 11 ? 512U : 1024U); }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(max_frontier, ends);
}

bool verify_uses_chunked_small_t(std::uint32_t draft_window, std::uint32_t batch_size,
                                 std::uint32_t max_visible_keys) {
    const std::uint32_t tokens = draft_window + 1;
    if (tokens <= 6) { return false; }
    if (batch_size > 1) { return true; }
    // At batch one the route only depends on the target while the width is inside the verify
    // domain, which a copy round's wide flag extends from 16 to 64 columns; past it
    // causal_attention_resolve_route returns Prompt for every envelope. Without this line the
    // mirror claims a target dependence the route does not have, which costs a second topology
    // class at widths no verify path can request.
    if (tokens > 64) { return false; }
    const std::uint32_t prompt_visible_limit = tokens <= 12 ? 512U : 1024U;
    return max_visible_keys > prompt_visible_limit;
}

// The attention route family of a call of `columns` query columns whose visible keys end at
// `target`, as the op resolves it. Only a copy round runs past 16 columns, and it asks for the
// wide verify route.
int route_family(ops::AttentionHeadGeometry attention, KvCacheStorage storage,
                 std::uint32_t columns, std::uint32_t target) {
    return ops::causal_softmax_attention_route_family(
        attention, storage, {1U, std::max(target, 1U), false, columns > 16U}, 1,
        static_cast<std::int32_t>(columns));
}

int verify_route_family(ops::AttentionHeadGeometry attention, KvCacheStorage storage,
                        std::uint32_t verify_window, std::uint32_t target) {
    return route_family(attention, storage, verify_window + 1U, target);
}

// Largest target that still takes the route of the smallest one, or zero when the route does
// not change up to `capacity`. The route flips at most once as the target grows (prompt below a
// storage's cutoff, small-T or chunked small-T above), so bisection finds the flip.
std::uint32_t verify_route_flip_target(ops::AttentionHeadGeometry attention, KvCacheStorage storage,
                                       std::uint32_t verify_window, std::uint32_t capacity) {
    const int first = verify_route_family(attention, storage, verify_window, 1U);
    if (verify_route_family(attention, storage, verify_window, capacity) == first) { return 0U; }
    std::uint32_t same_side  = 1U;
    std::uint32_t other_side = capacity;
    while (other_side - same_side > 1U) {
        const std::uint32_t mid = same_side + (other_side - same_side) / 2U;
        if (verify_route_family(attention, storage, verify_window, mid) == first) {
            same_side = mid;
        } else {
            other_side = mid;
        }
    }
    return same_side;
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity,
                                                           ops::AttentionHeadGeometry attention,
                                                           KvCacheStorage storage) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    std::vector<GraphExecutionProfile> profiles =
        graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
    // A BF16 cache takes the prompt kernel up to 128 visible keys (the first range) and small-T
    // past it, so the class is the route: an executable cannot be updated across the change.
    for (GraphExecutionProfile& profile : profiles) {
        profile.topology_class =
            static_cast<std::uint32_t>(route_family(attention, storage, 1U, profile.max + 1U));
    }
    return profiles;
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                      std::uint32_t verify_window,
                                                      std::uint32_t draft_window,
                                                      ops::AttentionHeadGeometry attention,
                                                      KvCacheStorage storage) {
    // An adaptive round verifies fewer drafts than it proposes, and an ngram copy round more.
    if (verify_window == 0 || draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+V+K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, verify_window + draft_window);
    }
    // Target verify and MTP batch both have T=V+1 and W=E+V+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (verify_window == 3) {
        add_shifted(1029, verify_window + 1);
    } else if (verify_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, verify_window + 1);
        }
    } else if (verify_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, verify_window + 1);
        }
    }
    // instantiate_graph_family builds one executable per topology class and installs the other
    // profiles of that class through an in-place update, which cannot cross a change of node
    // count. The attention route turns on the visible-key count where the storage sets a prompt
    // cutoff (BF16 below 128 keys at four columns, the INT8 family below 256 past eight columns),
    // so the frontier breaks where the route flips and the class is the profile's route.
    const std::uint32_t flip_target =
        verify_route_flip_target(attention, storage, verify_window, capacity);
    if (flip_target != 0U) { add_shifted(flip_target, verify_window + 1); }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());

    std::vector<GraphExecutionProfile> profiles = graph_profiles_through(capacity - 1, ends);
    // Past eight verify columns the small-T attention runs in chunks whose launches change with the
    // window in more places than the route flip, and an in-place update across them fails at
    // startup (MTP with 10 or 15 drafts). Those widths give every profile its own executable, as
    // DFlash2 does.
    constexpr std::uint32_t kSmallTColumns = 8;
    if (verify_window + 1U > kSmallTColumns) {
        for (std::size_t index = 0; index < profiles.size(); ++index) {
            profiles[index].topology_class = static_cast<std::uint32_t>(index);
        }
        return profiles;
    }
    // Otherwise profiles share an executable when every attention call they capture takes the same
    // route: the target verify and the MTP batch (verify_window + 1 columns up to max + V + 1) and
    // each autoregressive draft step (one column up to max + V + 2 + step), the envelopes
    // mtp_causal_attention_envelopes gives them.
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    std::vector<std::vector<int>> signatures;
    for (GraphExecutionProfile& profile : profiles) {
        std::vector<int> signature{verify_route_family(
            attention, storage, verify_window,
            visible(static_cast<std::uint64_t>(profile.max) + verify_window + 1ULL))};
        for (std::uint32_t step = 0; step + 1 < draft_window; ++step) {
            signature.push_back(route_family(
                attention, storage, 1U,
                visible(static_cast<std::uint64_t>(profile.max) + verify_window + step + 2ULL)));
        }
        const auto found = std::find(signatures.begin(), signatures.end(), signature);
        profile.topology_class = static_cast<std::uint32_t>(found - signatures.begin());
        if (found == signatures.end()) { signatures.push_back(std::move(signature)); }
    }
    return profiles;
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                         std::uint32_t capacity,
                                                         std::uint32_t draft_window,
                                                         std::uint32_t batch_size) {
    if (capacity == 0 || draft_window == 0 || draft_window > 63 ||
        (draft_window > 15 && batch_size != 1)) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    if (backend == SpeculativeBackend::DFlash2) {
        auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
        return profiles;
    }
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool split_swa           = profile.max > 96U;
        const bool chunked_target =
            verify_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = (chunked_target ? 2U : 0U) | (split_swa ? 1U : 0U);
    }
    return profiles;
}

} // namespace ninfer::models::qwen3_5::detail
