#pragma once

// Which contiguous range of layers each pipeline stage owns, and how to choose that split.
//
// A stage owns its layers whole: weights, the KV cache of its attention layers and the recurrent
// state of its GDN layers all live on the stage's device. That is what lets a stage run its layers
// with the same kernels as a single GPU and cross to the next stage once, at a layer boundary.
//
// A one-stage plan is the identity mapping, which keeps every consumer a no-op on one GPU.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer {

struct StagePlacement {
    std::size_t stage   = 0; // which stage owns it
    std::uint32_t local = 0; // index within that stage's layers
};

class StagePlan {
public:
    // Identity: one stage owns every layer.
    explicit StagePlan(std::uint32_t layer_count)
        : layer_count_(layer_count), boundaries_{layer_count} {
        if (layer_count == 0) { throw std::invalid_argument("stage plan needs at least one layer"); }
    }

    // boundaries[i] is the exclusive end of stage i: strictly ascending, last == layer_count. A
    // stage cannot be empty; a device that would hold nothing is a planning mistake, not a
    // configuration.
    StagePlan(std::uint32_t layer_count, std::vector<std::uint32_t> boundaries)
        : layer_count_(layer_count), boundaries_(std::move(boundaries)) {
        if (layer_count_ == 0) { throw std::invalid_argument("stage plan needs at least one layer"); }
        if (boundaries_.empty()) { throw std::invalid_argument("stage plan needs a boundary"); }
        if (boundaries_.back() != layer_count_) {
            throw std::invalid_argument("stage plan must cover every layer");
        }
        std::uint32_t previous = 0;
        for (const std::uint32_t end : boundaries_) {
            if (end <= previous) {
                throw std::invalid_argument("stage plan stages must each own at least one layer");
            }
            previous = end;
        }
    }

    // From explicit per-stage layer counts (`--stage-layers 30,34`).
    static StagePlan from_layer_counts(std::uint32_t layer_count,
                                       std::span<const std::uint32_t> counts) {
        std::vector<std::uint32_t> boundaries;
        boundaries.reserve(counts.size());
        std::uint64_t total = 0;
        for (const std::uint32_t count : counts) {
            total += count;
            if (total > layer_count) {
                throw std::invalid_argument("stage layer counts exceed the model's " +
                                            std::to_string(layer_count) + " layers");
            }
            boundaries.push_back(static_cast<std::uint32_t>(total));
        }
        if (total != layer_count) {
            throw std::invalid_argument("stage layer counts sum to " + std::to_string(total) +
                                        " but the model has " + std::to_string(layer_count) +
                                        " layers");
        }
        return StagePlan(layer_count, std::move(boundaries));
    }

    // Equal layer counts, remainder spread over the leading stages.
    static StagePlan even(std::uint32_t layer_count, std::size_t stages) {
        if (stages == 0) { throw std::invalid_argument("stage plan needs at least one stage"); }
        if (stages > layer_count) { throw std::invalid_argument("more stages than layers"); }
        std::vector<std::uint32_t> boundaries;
        boundaries.reserve(stages);
        for (std::size_t stage = 0; stage < stages; ++stage) {
            boundaries.push_back(static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(layer_count) * (stage + 1)) / stages));
        }
        return StagePlan(layer_count, std::move(boundaries));
    }

    [[nodiscard]] std::size_t stages() const noexcept { return boundaries_.size(); }
    [[nodiscard]] std::uint32_t layer_count() const noexcept { return layer_count_; }
    [[nodiscard]] bool single_stage() const noexcept { return boundaries_.size() == 1; }

    [[nodiscard]] std::uint32_t stage_begin(std::size_t stage) const {
        check_stage(stage);
        return stage == 0 ? 0U : boundaries_[stage - 1];
    }
    [[nodiscard]] std::uint32_t stage_end(std::size_t stage) const {
        check_stage(stage);
        return boundaries_[stage];
    }
    [[nodiscard]] std::uint32_t stage_layers(std::size_t stage) const {
        return stage_end(stage) - stage_begin(stage);
    }

    [[nodiscard]] StagePlacement placement(std::uint32_t global_layer) const {
        if (global_layer >= layer_count_) {
            throw std::out_of_range("layer index past the end of the stage plan");
        }
        const auto stage = static_cast<std::size_t>(
            std::upper_bound(boundaries_.begin(), boundaries_.end(), global_layer) -
            boundaries_.begin());
        return StagePlacement{stage, global_layer - stage_begin(stage)};
    }

    // True when the residual stream must move to another stage after this layer. Exactly one per
    // stage boundary, which is what makes a layer split tolerant of a slow link.
    [[nodiscard]] bool crosses_after(std::uint32_t global_layer) const {
        if (global_layer + 1 >= layer_count_) { return false; }
        return placement(global_layer).stage != placement(global_layer + 1).stage;
    }

    [[nodiscard]] const std::vector<std::uint32_t>& boundaries() const noexcept {
        return boundaries_;
    }

    friend bool operator==(const StagePlan&, const StagePlan&) = default;

private:
    void check_stage(std::size_t stage) const {
        if (stage >= boundaries_.size()) { throw std::out_of_range("stage index out of range"); }
    }

    std::uint32_t layer_count_ = 0;
    std::vector<std::uint32_t> boundaries_;
};

// What one layer costs on the device that owns it.
struct LayerCost {
    // Resident weights.
    std::uint64_t weight_bytes = 0;
    // KV bytes for one page group of the layer, zero for layers without attention. A page group is
    // the unit the whole model allocates in, so total KV on a stage is this times the page groups.
    std::uint64_t kv_bytes_per_page_group = 0;
    // Recurrent state for every request slot, zero for layers without it.
    std::uint64_t state_bytes = 0;
};

// What one stage's device can spend.
struct StageBudget {
    // Bytes free on the device before anything of the model is placed on it.
    std::uint64_t available_bytes = 0;
    // Bytes the stage needs whatever layers it owns: the embedding on the first stage, the head and
    // prefill buffers on the last, workspace, graph memory.
    std::uint64_t fixed_bytes = 0;
};

struct StageSolution {
    StagePlan plan;
    // The page groups every stage can hold at once: the capacity the plan was chosen to maximise.
    // Unbounded (the maximum value) when no layer has a KV cost.
    std::uint64_t page_groups = 0;
};

namespace detail {

struct StageCosts {
    std::vector<std::uint64_t> weight_prefix{0};
    std::vector<std::uint64_t> kv_prefix{0};
    std::vector<std::uint64_t> state_prefix{0};
};

inline StageCosts prefix_costs(std::span<const LayerCost> layers) {
    StageCosts out;
    for (const LayerCost& layer : layers) {
        out.weight_prefix.push_back(out.weight_prefix.back() + layer.weight_bytes);
        out.kv_prefix.push_back(out.kv_prefix.back() + layer.kv_bytes_per_page_group);
        out.state_prefix.push_back(out.state_prefix.back() + layer.state_bytes);
    }
    return out;
}

inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
    return (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
               ? std::numeric_limits<std::uint64_t>::max()
               : a * b;
}

// Whether layers [begin, end) fit on `budget` while holding `page_groups` page groups. Saturating,
// so a huge page-group count reads as "does not fit" instead of wrapping around to "fits".
inline bool stage_fits(const StageCosts& costs, const StageBudget& budget, std::uint32_t begin,
                       std::uint32_t end, std::uint64_t page_groups) {
    std::uint64_t used = budget.fixed_bytes;
    used = saturating_add(used, costs.weight_prefix[end] - costs.weight_prefix[begin]);
    used = saturating_add(used, costs.state_prefix[end] - costs.state_prefix[begin]);
    used = saturating_add(
        used, saturating_mul(costs.kv_prefix[end] - costs.kv_prefix[begin], page_groups));
    return used <= budget.available_bytes;
}

// Feasibility of `page_groups` on `budgets`: whether some cut into non-empty contiguous stages fits
// every device. Exact rather than greedy. Filling early stages as full as possible is not safe once
// every stage must own a layer: it can leave a small middle stage no choice but to take a layer it
// cannot hold, when a slightly different cut would have fit. The instance is tiny (layers x stages
// squared), so the search costs nothing.
inline bool feasible(const StageCosts& costs, std::span<const StageBudget> budgets,
                     std::uint32_t layer_count, std::uint64_t page_groups) {
    // reachable[end]: the first `stage` stages can own exactly layers [0, end).
    std::vector<char> reachable(layer_count + 1, 0);
    reachable[0] = 1;
    for (std::size_t stage = 0; stage < budgets.size(); ++stage) {
        std::vector<char> next(layer_count + 1, 0);
        for (std::uint32_t begin = 0; begin < layer_count; ++begin) {
            if (reachable[begin] == 0) { continue; }
            // Fits is monotone in the range, so stop at the first end that does not.
            for (std::uint32_t end = begin + 1; end <= layer_count; ++end) {
                if (!stage_fits(costs, budgets[stage], begin, end, page_groups)) { break; }
                next[end] = 1;
            }
        }
        reachable = std::move(next);
    }
    return reachable[layer_count] != 0;
}

} // namespace detail

// Chooses the contiguous split that lets the most page groups fit on every stage at once, then,
// among the splits that achieve it, the one that leaves the most even fraction of each device in
// use so no card is the near-full one.
//
// Layers are placed in order and each stage owns at least one. Throws when even zero page groups do
// not fit -- the weights alone are too large for the devices given.
inline StageSolution solve_stage_plan(std::span<const LayerCost> layers,
                                      std::span<const StageBudget> budgets) {
    if (layers.empty()) { throw std::invalid_argument("stage solver needs at least one layer"); }
    if (budgets.empty()) { throw std::invalid_argument("stage solver needs at least one stage"); }
    if (budgets.size() > layers.size()) { throw std::invalid_argument("more stages than layers"); }
    const auto layer_count = static_cast<std::uint32_t>(layers.size());
    const detail::StageCosts costs = detail::prefix_costs(layers);

    if (!detail::feasible(costs, budgets, layer_count, 0)) {
        throw std::runtime_error(
            "the model's weights do not fit on the given devices even with no KV cache");
    }

    // Largest page-group count that is still feasible. Feasibility is monotone in the count.
    constexpr std::uint64_t kUnbounded = std::numeric_limits<std::uint64_t>::max();
    const bool any_kv                  = costs.kv_prefix.back() != 0;
    std::uint64_t best                 = kUnbounded;
    if (any_kv) {
        std::uint64_t low  = 0;
        std::uint64_t high = 1;
        while (high < (std::uint64_t{1} << 62) &&
               detail::feasible(costs, budgets, layer_count, high)) {
            low = high;
            high *= 2;
        }
        while (low + 1 < high) {
            const std::uint64_t middle = low + (high - low) / 2;
            if (detail::feasible(costs, budgets, layer_count, middle)) {
                low = middle;
            } else {
                high = middle;
            }
        }
        best = low;
    }

    // Among the splits that hold `best` page groups, minimise the fullest stage. Exact DP over
    // (stages placed, layers placed): the minimum achievable maximum utilisation.
    const std::uint64_t at   = any_kv ? best : 0;
    const std::size_t stages = budgets.size();
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    auto utilisation = [&](std::size_t stage, std::uint32_t begin, std::uint32_t end) -> double {
        if (!detail::stage_fits(costs, budgets[stage], begin, end, at)) { return kInfinity; }
        const long double used =
            static_cast<long double>(budgets[stage].fixed_bytes) +
            static_cast<long double>(costs.weight_prefix[end] - costs.weight_prefix[begin]) +
            static_cast<long double>(costs.state_prefix[end] - costs.state_prefix[begin]) +
            static_cast<long double>(costs.kv_prefix[end] - costs.kv_prefix[begin]) *
                static_cast<long double>(at);
        const long double capacity = static_cast<long double>(budgets[stage].available_bytes);
        return capacity > 0 ? static_cast<double>(used / capacity) : kInfinity;
    };

    std::vector<std::vector<double>> dp(stages + 1,
                                        std::vector<double>(layer_count + 1, kInfinity));
    std::vector<std::vector<std::uint32_t>> from(stages + 1,
                                                 std::vector<std::uint32_t>(layer_count + 1, 0));
    dp[0][0] = 0.0;
    for (std::size_t stage = 0; stage < stages; ++stage) {
        for (std::uint32_t end = 1; end <= layer_count; ++end) {
            for (std::uint32_t begin = 0; begin < end; ++begin) {
                if (dp[stage][begin] == kInfinity) { continue; }
                const double u = utilisation(stage, begin, end);
                if (u == kInfinity) { continue; }
                const double value = std::max(dp[stage][begin], u);
                if (value < dp[stage + 1][end]) {
                    dp[stage + 1][end]   = value;
                    from[stage + 1][end] = begin;
                }
            }
        }
    }

    std::vector<std::uint32_t> boundaries(stages);
    std::uint32_t end = layer_count;
    for (std::size_t stage = stages; stage > 0; --stage) {
        boundaries[stage - 1] = end;
        end                   = from[stage][end];
    }
    return StageSolution{StagePlan(layer_count, std::move(boundaries)), best};
}

} // namespace ninfer
