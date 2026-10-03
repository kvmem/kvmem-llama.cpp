#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>

namespace ninfer::runtime {

template <class Clock = std::chrono::steady_clock>
[[nodiscard]] std::uint64_t planning_now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

// One Engine admission boundary, including all head/backfill inspections. Correctness work and
// sealing cannot be cancelled by this allowance, but their elapsed time reduces optional headroom.
struct PlanningAllowance {
    std::uint64_t started_ns              = planning_now_ns();
    std::uint64_t limit_ns                = 50'000'000;
    std::uint32_t affected_requests       = 1;
    const std::atomic<bool>* cancellation = nullptr;
    std::uint64_t control_deadline_ns     = std::numeric_limits<std::uint64_t>::max();
    // A thorough search scales its base grant with the value at stake instead of a flat 5 ms,
    // gives every eligible node its own bounded discovery grant, and prices each request's gain
    // alone instead of sharing it across the runnable requests.
    bool thorough = false;

    static constexpr std::uint64_t kThoroughBoundaryNs = 250'000'000;

    // A thorough boundary gets the full 250 ms even while other requests run: admission happens
    // once per request, and a search stopped early can miss a reusable prefix worth tens of
    // seconds of re-prefill to save the running decode one pause of at most this allowance.
    [[nodiscard]] static PlanningAllowance boundary(std::uint32_t other_runnable,
                                                    std::uint64_t now = planning_now_ns(),
                                                    bool thorough     = false) noexcept {
        return {.started_ns        = now,
                .limit_ns          = thorough              ? kThoroughBoundaryNs
                                     : other_runnable == 0 ? 50'000'000ULL
                                                           : 10'000'000ULL,
                .affected_requests = 1U + other_runnable,
                .thorough          = thorough};
    }

    // The gain is the re-prefill one request avoids. By default the runnable requests share its
    // economic bound; a thorough boundary does not divide it, since queued re-prefills cost more
    // under load and the time limit already bounds the pause the others see.
    [[nodiscard]] std::uint32_t economic_sharing() const noexcept {
        return thorough ? 1U : std::max(1U, affected_requests);
    }

    [[nodiscard]] std::uint64_t remaining(std::uint64_t now) const noexcept {
        if ((cancellation && cancellation->load(std::memory_order_relaxed)) ||
            now >= control_deadline_ns) {
            return 0;
        }
        const auto elapsed = now > started_ns ? now - started_ns : 0;
        return std::min(elapsed < limit_ns ? limit_ns - elapsed : 0, control_deadline_ns - now);
    }
};

// Integer timestamps make wall-budget decisions reproducible without sleeping in policy tests.
// Estimates control optional work only; they never certify feasibility or a search bound.
class MaterializationSearchBudget {
public:
    MaterializationSearchBudget(PlanningAllowance allowance, std::uint64_t started,
                                std::uint64_t initial_cost) noexcept
        : allowance_(allowance), started_(started),
          granted_(std::min(allowance.thorough ? value_scaled_grant(initial_cost)
                                               : std::min(kMinimumGrantNs, economic(initial_cost)),
                            allowance.remaining(started))) {}

    [[nodiscard]] bool allow(std::uint64_t now, std::uint64_t next_operation_ns,
                             std::uint64_t completion_ns, std::uint64_t gain_ns,
                             bool complete_prediction, std::uint64_t progress,
                             bool discovery_eligible = true) noexcept {
        boundary_limited_             = false;
        const std::uint64_t elapsed   = now > started_ ? now - started_ : 0;
        const std::uint64_t remaining = allowance_.remaining(now);
        next_operation_ns             = std::max<std::uint64_t>(1, next_operation_ns);
        completion_ns                 = std::max(next_operation_ns, completion_ns);
        if (next_operation_ns > remaining) {
            boundary_limited_ = true;
            reason_           = MaterializationStopReason::TimeBudget;
            return false;
        }
        if (elapsed < granted_ && next_operation_ns <= granted_ - elapsed) { return true; }
        if (completion_ns > remaining || completion_ns > economic(gain_ns)) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        // Unknown forecasts get one bounded discovery episode, not a fresh grant for every node,
        // unless the search is thorough.
        if (!complete_prediction &&
            (!discovery_eligible || (discovery_used_ && !allowance_.thorough))) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        if (renewals_ != 0 && progress <= renewal_progress_) {
            reason_ = MaterializationStopReason::InsufficientExpectedGain;
            return false;
        }
        const auto hard = elapsed + remaining; // bounded by the allowance's representable duration
        const auto growth = std::max(granted_, next_operation_ns);
        auto added        = std::min(growth, hard > granted_ ? hard - granted_ : 0);
        if (!complete_prediction) { added = std::min<std::uint64_t>(added, 5'000'000); }
        if (added == 0 || elapsed + next_operation_ns > granted_ + added) {
            reason_ = MaterializationStopReason::TimeBudget;
            return false;
        }
        granted_ += added;
        ++renewals_;
        renewal_progress_ = progress;
        discovery_used_   = discovery_used_ || !complete_prediction;
        return true;
    }

    [[nodiscard]] std::uint64_t granted_ns() const noexcept { return granted_; }

    [[nodiscard]] std::uint32_t renewals() const noexcept { return renewals_; }

    [[nodiscard]] bool boundary_limited() const noexcept { return boundary_limited_; }

    [[nodiscard]] bool discovery_used() const noexcept { return discovery_used_; }

    [[nodiscard]] MaterializationStopReason stop_reason() const noexcept { return reason_; }

    [[nodiscard]] std::uint64_t overshoot(std::uint64_t now) const noexcept {
        const auto elapsed = now > started_ ? now - started_ : 0;
        return elapsed > granted_ ? elapsed - granted_ : 0;
    }

    // The search time a gain can justify; a saturated gain is not evidence of value.
    [[nodiscard]] std::uint64_t economic(std::uint64_t gain) const noexcept {
        if (gain == std::numeric_limits<std::uint64_t>::max()) { return 0; }
        return gain / kCostDivisor / allowance_.economic_sharing();
    }

private:
    static constexpr std::uint64_t kMinimumGrantNs = 5'000'000;
    static constexpr std::uint64_t kCostDivisor    = 20;

    // The incumbent's machine-work cost is the value a better plan can save, so an expensive one
    // earns a longer search, up to the thorough boundary, and a cheap one keeps the 5 ms floor. A
    // saturated cost is not evidence of large value and earns nothing.
    [[nodiscard]] static std::uint64_t value_scaled_grant(std::uint64_t incumbent_cost) noexcept {
        if (incumbent_cost == std::numeric_limits<std::uint64_t>::max()) { return 0; }
        return std::clamp(incumbent_cost / kCostDivisor, kMinimumGrantNs,
                          PlanningAllowance::kThoroughBoundaryNs);
    }

    bool boundary_limited_ = false;
    PlanningAllowance allowance_;
    std::uint64_t started_            = 0;
    std::uint64_t granted_            = 0;
    std::uint64_t renewal_progress_   = 0;
    std::uint32_t renewals_           = 0;
    bool discovery_used_              = false;
    MaterializationStopReason reason_ = MaterializationStopReason::TimeBudget;
};

} // namespace ninfer::runtime
