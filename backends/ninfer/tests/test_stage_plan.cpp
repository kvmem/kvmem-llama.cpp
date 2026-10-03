// Stage plan and memory-balancing solver.
//
// The solver's answer is checked against an independent oracle: enumerate every way to cut the
// layers into contiguous, non-empty stages and score each one directly. That is exponential and
// only viable on small instances, which is fine -- the point is to catch the DP or the binary search
// being wrong, and small random instances with uneven costs and budgets are where they go wrong.

#include "core/stage_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

template <class Fn>
void check_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    std::cerr << "expectation failed: " << label << " (nothing was thrown)\n";
    ++failures;
}

std::uint64_t next_random(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
}

using ninfer::LayerCost;
using ninfer::StageBudget;
using ninfer::StagePlan;

constexpr std::uint64_t kUnbounded = std::numeric_limits<std::uint64_t>::max();

struct Score {
    bool feasible;
    std::uint64_t page_groups; // kUnbounded when no stage has a KV cost
    double utilisation;        // at the page-group count `at`
};

// Direct evaluation of one partition.
Score score(const std::vector<LayerCost>& layers, const std::vector<StageBudget>& budgets,
            const std::vector<std::uint32_t>& counts, std::uint64_t at) {
    std::uint64_t page_groups = kUnbounded;
    double worst              = 0.0;
    std::size_t layer         = 0;
    for (std::size_t stage = 0; stage < counts.size(); ++stage) {
        std::uint64_t weights = 0, state = 0, kv = 0;
        for (std::uint32_t i = 0; i < counts[stage]; ++i, ++layer) {
            weights += layers[layer].weight_bytes;
            state += layers[layer].state_bytes;
            kv += layers[layer].kv_bytes_per_page_group;
        }
        const std::uint64_t base = budgets[stage].fixed_bytes + weights + state;
        if (base > budgets[stage].available_bytes) { return {false, 0, 0.0}; }
        if (kv != 0) {
            page_groups = std::min(page_groups, (budgets[stage].available_bytes - base) / kv);
        }
        if (at != kUnbounded) {
            const long double used = static_cast<long double>(base) +
                                     static_cast<long double>(kv) * static_cast<long double>(at);
            if (used > static_cast<long double>(budgets[stage].available_bytes)) {
                return {false, 0, 0.0};
            }
            worst = std::max(worst, static_cast<double>(
                                        used / static_cast<long double>(budgets[stage].available_bytes)));
        }
    }
    return {true, page_groups, worst};
}

void enumerate(std::uint32_t layers_left, std::size_t stages_left, std::vector<std::uint32_t>& counts,
               const std::function<void(const std::vector<std::uint32_t>&)>& visit) {
    if (stages_left == 1) {
        counts.push_back(layers_left);
        visit(counts);
        counts.pop_back();
        return;
    }
    for (std::uint32_t take = 1; take + (stages_left - 1) <= layers_left; ++take) {
        counts.push_back(take);
        enumerate(layers_left - take, stages_left - 1, counts, visit);
        counts.pop_back();
    }
}

void plan_basics() {
    const StagePlan one(10);
    check(one.single_stage() && one.stages() == 1, "a one-stage plan is the identity");
    check(one.placement(7).stage == 0 && one.placement(7).local == 7, "identity keeps local == global");
    check(!one.crosses_after(3), "a single stage never crosses");

    const StagePlan even = StagePlan::even(10, 3);
    check(even.stages() == 3 && even.stage_begin(0) == 0 && even.stage_end(2) == 10,
          "an even plan covers every layer");
    std::uint32_t covered = 0;
    for (std::size_t stage = 0; stage < even.stages(); ++stage) {
        check(even.stage_layers(stage) >= 3 && even.stage_layers(stage) <= 4,
              "even stages differ by at most one layer");
        covered += even.stage_layers(stage);
    }
    check(covered == 10, "even stages add up to the layer count");

    const std::uint32_t counts[] = {3, 7};
    const StagePlan split = StagePlan::from_layer_counts(10, counts);
    check(split.placement(2).stage == 0 && split.placement(3).stage == 1, "placement follows the cut");
    check(split.placement(3).local == 0 && split.placement(9).local == 6, "local index restarts per stage");
    check(split.crosses_after(2) && !split.crosses_after(3) && !split.crosses_after(9),
          "exactly one crossing per boundary");

    check_throws([] { const std::uint32_t c[] = {3, 6}; (void)StagePlan::from_layer_counts(10, c); },
                 "counts that fall short of the model are rejected");
    check_throws([] { const std::uint32_t c[] = {6, 6}; (void)StagePlan::from_layer_counts(10, c); },
                 "counts that overshoot the model are rejected");
    check_throws([] { const std::uint32_t c[] = {0, 10}; (void)StagePlan::from_layer_counts(10, c); },
                 "an empty stage is rejected");
    check_throws([] { (void)StagePlan::even(3, 4); }, "more stages than layers is rejected");
    check_throws([&] { (void)split.stage_begin(2); }, "a stage past the end is rejected");
    check_throws([&] { (void)split.placement(10); }, "a layer past the end is rejected");
}

void solver_matches_oracle() {
    std::uint64_t rng = 12345;
    int compared      = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const std::uint32_t layer_count = 2 + static_cast<std::uint32_t>(next_random(rng) % 8);
        const std::size_t stages        = 1 + static_cast<std::size_t>(next_random(rng) % 3);
        if (stages > layer_count) { continue; }

        std::vector<LayerCost> layers(layer_count);
        for (LayerCost& layer : layers) {
            layer.weight_bytes = 10 + next_random(rng) % 90;
            // Roughly one layer in three carries KV, one in two carries recurrent state.
            layer.kv_bytes_per_page_group = next_random(rng) % 3 == 0 ? 1 + next_random(rng) % 6 : 0;
            layer.state_bytes             = next_random(rng) % 2 == 0 ? next_random(rng) % 12 : 0;
        }
        std::vector<StageBudget> budgets(stages);
        for (StageBudget& budget : budgets) {
            budget.fixed_bytes     = next_random(rng) % 40;
            budget.available_bytes = 150 + next_random(rng) % 900;
        }

        // Oracle: best page-group count over every partition, then the least-full partition that
        // still holds it.
        std::uint64_t best_groups = 0;
        bool any_feasible         = false;
        std::vector<std::uint32_t> counts;
        enumerate(layer_count, stages, counts, [&](const std::vector<std::uint32_t>& c) {
            const Score s = score(layers, budgets, c, 0);
            if (!s.feasible) { return; }
            any_feasible = true;
            best_groups  = std::max(best_groups, s.page_groups);
        });

        if (!any_feasible) {
            check_throws([&] { (void)ninfer::solve_stage_plan(layers, budgets); },
                         "an unfittable model is rejected");
            continue;
        }

        double best_utilisation = std::numeric_limits<double>::infinity();
        const std::uint64_t at  = best_groups == kUnbounded ? 0 : best_groups;
        enumerate(layer_count, stages, counts, [&](const std::vector<std::uint32_t>& c) {
            const Score s = score(layers, budgets, c, at);
            if (s.feasible) { best_utilisation = std::min(best_utilisation, s.utilisation); }
        });

        const ninfer::StageSolution solution = ninfer::solve_stage_plan(layers, budgets);
        ++compared;
        if (solution.page_groups != best_groups) {
            std::cerr << "  instance: " << layer_count << " layers, " << stages << " stages; oracle "
                      << best_groups << ", solver " << solution.page_groups << "\n";
            for (const LayerCost& l : layers) {
                std::cerr << "    w=" << l.weight_bytes << " kv=" << l.kv_bytes_per_page_group
                          << " st=" << l.state_bytes << '\n';
            }
            for (const StageBudget& b : budgets) {
                std::cerr << "    avail=" << b.available_bytes << " fixed=" << b.fixed_bytes << '\n';
            }
        }
        check(solution.page_groups == best_groups, "the solver finds the largest feasible page-group count");

        std::vector<std::uint32_t> chosen;
        for (std::size_t stage = 0; stage < solution.plan.stages(); ++stage) {
            chosen.push_back(solution.plan.stage_layers(stage));
        }
        const Score s = score(layers, budgets, chosen, at);
        check(s.feasible && (best_groups == kUnbounded || s.page_groups >= best_groups),
              "the chosen split actually holds that many page groups");
        check(std::abs(s.utilisation - best_utilisation) < 1e-9,
              "among the splits that hold it, the solver picks the least-full");
    }
    check(compared > 100, "enough random instances were feasible to mean something");
}

void solver_scenarios() {
    // A 27B-shaped model: 64 layers, every fourth one full attention.
    std::vector<LayerCost> layers(64);
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const bool attention = i % 4 == 3;
        layers[i].weight_bytes            = attention ? 260ULL << 20 : 240ULL << 20;
        layers[i].kv_bytes_per_page_group = attention ? 2ULL << 20 : 0;
        layers[i].state_bytes             = attention ? 0 : 3ULL << 20;
    }
    const std::uint64_t GiB = 1ULL << 30;

    // Two identical cards and no fixed costs: an even split.
    {
        const StageBudget budgets[] = {{16 * GiB, 0}, {16 * GiB, 0}};
        const auto solution         = ninfer::solve_stage_plan(layers, budgets);
        check(solution.plan.stage_layers(0) == 32 && solution.plan.stage_layers(1) == 32,
              "identical devices split the layers evenly");
    }
    // The first card also holds the embedding: it takes fewer layers.
    {
        const StageBudget budgets[] = {{16 * GiB, 2 * GiB}, {16 * GiB, 0}};
        const auto solution         = ninfer::solve_stage_plan(layers, budgets);
        check(solution.plan.stage_layers(0) < solution.plan.stage_layers(1),
              "a fixed cost on the first stage moves layers off it");
    }
    // A larger second card takes more.
    {
        const StageBudget budgets[] = {{16 * GiB, 0}, {24 * GiB, 0}};
        const auto solution         = ninfer::solve_stage_plan(layers, budgets);
        check(solution.plan.stage_layers(1) > solution.plan.stage_layers(0),
              "a larger device takes more layers");
    }
    // Weights alone do not fit.
    {
        const StageBudget budgets[] = {{4 * GiB, 0}, {4 * GiB, 0}};
        check_throws([&] { (void)ninfer::solve_stage_plan(layers, budgets); },
                     "weights that fit nowhere are rejected");
    }
    // No KV anywhere: capacity is unbounded and the split is still balanced.
    {
        std::vector<LayerCost> plain(8, LayerCost{100, 0, 0});
        const StageBudget budgets[] = {{1000, 0}, {1000, 0}};
        const auto solution         = ninfer::solve_stage_plan(plain, budgets);
        check(solution.page_groups == kUnbounded, "no KV cost means unbounded page groups");
        check(solution.plan.stage_layers(0) == 4, "a model without KV still splits evenly");
    }
    // A huge page-group ceiling must not wrap around into "fits".
    {
        std::vector<LayerCost> tiny(4, LayerCost{1, 1, 0});
        const StageBudget budgets[] = {{std::numeric_limits<std::uint64_t>::max() / 2, 0},
                                       {std::numeric_limits<std::uint64_t>::max() / 2, 0}};
        const auto solution         = ninfer::solve_stage_plan(tiny, budgets);
        check(solution.page_groups > 1000000, "very large budgets solve without overflow");
    }
}

} // namespace

int run_stage_plan_test() {
    try {
        plan_basics();
        solver_matches_oracle();
        solver_scenarios();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "stage plan tests passed\n";
    return 0;
}
