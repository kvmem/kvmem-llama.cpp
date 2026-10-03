// Adaptive MTP only chooses how many drafts a round verifies; greedy output is the same at every
// width. What these checks defend is the choice: the survival estimate, the measured cost model,
// and that the controller widens under full acceptance and settles low under none.

#include "models/qwen3_5/program/speculative/mtp_adaptive.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace {

using ninfer::models::qwen3_5::detail::MtpAdaptiveBatchController;
using ninfer::models::qwen3_5::detail::MtpAdaptiveSignal;
using ninfer::models::qwen3_5::detail::MtpRoundCostModel;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void survival_follows_acceptance() {
    MtpAdaptiveSignal fresh;
    const float prior = fresh.expected_tokens(3, 3);
    require(prior > 1.5F && prior < 2.5F, "a fresh signal expects the prior survival");
    require(fresh.expected_tokens(3, 0) == 1.0F, "no ready draft expects only the bonus token");

    MtpAdaptiveSignal accepting;
    for (int round = 0; round < 40; ++round) { accepting.observe(5, 5); }
    require(accepting.expected_tokens(5, 5) > 5.5F, "full acceptance expects every draft");
    require(accepting.confident_tail(5), "a long success streak is a confident tail");

    MtpAdaptiveSignal rejecting;
    for (int round = 0; round < 40; ++round) { rejecting.observe(5, 0); }
    require(rejecting.expected_tokens(5, 5) < 1.1F, "no acceptance expects only the bonus token");
}

void costs_are_measured_then_scaled() {
    MtpRoundCostModel costs;
    const float shape_one   = costs.cost(1, 1);
    const float shape_three = costs.cost(1, 3);
    require(shape_three > shape_one, "the relative shape grows with the width");
    for (int round = 0; round < 8; ++round) { costs.observe(1, 3, 0.030); }
    require(std::abs(costs.cost(1, 3) - 0.030F) < 1e-6F, "a measured width reports its seconds");
    require(std::abs(costs.cost(1, 1) - 0.030F * shape_one / shape_three) < 1e-6F,
            "an unmeasured width scales from the measured one");
    require(costs.cost(2, 3) == shape_three, "batch sizes are measured separately");
}

struct Simulation {
    std::uint32_t final_window = 0;
    std::uint32_t widest       = 0;
};

// Rounds of one row with `maximum` drafts always ready; `accept` gives the accepted drafts of a
// round at a width, `seconds` its cost.
template <typename Accept, typename Seconds>
Simulation simulate(std::uint32_t maximum, Accept accept, Seconds seconds) {
    MtpAdaptiveBatchController controller;
    controller.reset(maximum);
    MtpAdaptiveSignal signal;
    Simulation out;
    const std::array<const MtpAdaptiveSignal*, 1> signals{&signal};
    const std::array<std::uint32_t, 1> available{maximum};
    const std::array<std::uint32_t, 1> room{4096};
    for (int round = 0; round < 300; ++round) {
        const std::uint32_t window = controller.select(signals, available, room, 7);
        require(window >= 1 && window <= maximum, "the selected width is outside the range");
        controller.observe_execution(1, window, seconds(window));
        signal.observe(window, accept(window));
        out.widest       = std::max(out.widest, window);
        out.final_window = window;
    }
    return out;
}

void the_controller_widens_and_narrows() {
    const auto seconds         = [](std::uint32_t window) { return 0.020 + 0.001 * window; };
    const Simulation accepting = simulate(7, [](std::uint32_t window) { return window; }, seconds);
    require(accepting.final_window >= 5, "full acceptance did not widen the window");
    const Simulation rejecting = simulate(7, [](std::uint32_t) { return 0U; }, seconds);
    require(rejecting.final_window == 3, "no acceptance did not settle at the narrowest width");
    const Simulation fixed_short =
        simulate(2, [](std::uint32_t window) { return window; }, seconds);
    require(fixed_short.widest <= 2, "the window exceeded its maximum");
}

} // namespace

int main() {
    try {
        survival_follows_acceptance();
        costs_are_measured_then_scaled();
        the_controller_widens_and_narrows();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("adaptive MTP: survival, costs and width selection ok");
    return 0;
}
