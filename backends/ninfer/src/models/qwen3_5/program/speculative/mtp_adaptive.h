#pragma once

// Adaptive MTP verification width (--adaptive-mtp). Each request keeps an estimate of how far its
// drafts survive; a batch controller picks the verification width that maximizes expected
// committed tokens per unit of round cost, moving between widths only after a candidate has won
// several rounds and a width has held for a few. Round costs are measured per batch size and
// width as the rounds run; a width not yet run is priced from a relative cost shape scaled by the
// widths that have.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::uint32_t kAdaptiveMtpMaximumDrafts = 15;
inline constexpr std::uint32_t kAdaptiveMtpMaximumBatch  = 8;

class MtpAdaptiveSignal final {
public:
    void reset() noexcept { *this = MtpAdaptiveSignal{}; }

    void observe(std::uint32_t proposed, std::uint32_t accepted) noexcept {
        proposed = std::min(proposed, kAdaptiveMtpMaximumDrafts);
        accepted = std::min(accepted, proposed);
        for (std::uint32_t position = 0; position < kAdaptiveMtpMaximumDrafts; ++position) {
            if (position < proposed) {
                const float sample = position < accepted ? 1.0F : 0.0F;
                fast_[position] += kFastAlpha * (sample - fast_[position]);
                slow_[position] += kSlowAlpha * (sample - slow_[position]);
                if (observations_[position] != UINT16_MAX) { ++observations_[position]; }
                if (sample == 1.0F) {
                    if (success_streak_[position] != UINT8_MAX) { ++success_streak_[position]; }
                } else {
                    success_streak_[position] = 0;
                }
                stale_rounds_[position] = 0;
            } else if (stale_rounds_[position] != UINT8_MAX) {
                ++stale_rounds_[position];
            }
        }
    }

    // Expected committed tokens of one round verifying `window` drafts, `available` of them ready.
    [[nodiscard]] float expected_tokens(std::uint32_t window,
                                        std::uint32_t available) const noexcept {
        const std::uint32_t extent = std::min({window, available, kAdaptiveMtpMaximumDrafts});
        float expected             = 1.0F;
        float previous             = 1.0F;
        for (std::uint32_t position = 0; position < extent; ++position) {
            const float evidence =
                std::min(1.0F, static_cast<float>(observations_[position]) / 4.0F);
            float estimate = kPriorSurvival[position] +
                             evidence * (0.75F * fast_[position] + 0.25F * slow_[position] -
                                         kPriorSurvival[position]);
            if ((observations_[position] == 0 || stale_rounds_[position] >= 4) && position != 0) {
                const float continuation = success_streak_[position - 1U] >= 2 ? 0.85F : 0.60F;
                estimate                 = std::max(estimate, previous * continuation);
            }
            estimate = std::clamp(estimate, 0.0F, previous);
            expected += estimate;
            previous = estimate;
        }
        return expected;
    }

    [[nodiscard]] bool confident_tail(std::uint32_t window) const noexcept {
        return window != 0 && window <= kAdaptiveMtpMaximumDrafts &&
               success_streak_[window - 1U] >= kWideningSuccessStreak;
    }

    [[nodiscard]] bool strong_wide_prefix() const noexcept {
        constexpr std::array<float, 3> thresholds{0.75F, 0.60F, 0.45F};
        for (std::size_t position = 0; position < thresholds.size(); ++position) {
            if (observations_[position] < 4 ||
                0.75F * fast_[position] + 0.25F * slow_[position] < thresholds[position]) {
                return false;
            }
        }
        return success_streak_[thresholds.size() - 1U] >= kWideningSuccessStreak;
    }

private:
    static constexpr float kFastAlpha                    = 0.375F;
    static constexpr float kSlowAlpha                    = 0.125F;
    static constexpr std::uint8_t kWideningSuccessStreak = 6;
    static constexpr std::array<float, kAdaptiveMtpMaximumDrafts> kPriorSurvival{
        0.54F,      0.16F,       0.08F,        0.04F,         0.02F,
        0.01F,      0.005F,      0.0025F,      0.00125F,      0.000625F,
        0.0003125F, 0.00015625F, 0.000078125F, 0.0000390625F, 0.00001953125F};

    std::array<float, kAdaptiveMtpMaximumDrafts> fast_ = kPriorSurvival;
    std::array<float, kAdaptiveMtpMaximumDrafts> slow_ = kPriorSurvival;
    std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts> observations_{};
    std::array<std::uint8_t, kAdaptiveMtpMaximumDrafts> stale_rounds_{};
    std::array<std::uint8_t, kAdaptiveMtpMaximumDrafts> success_streak_{};
};

// Round cost by batch size and verification width, in seconds once measured. A width without
// enough rounds is priced from the relative shape below, scaled by the most-run width of its batch
// size; with nothing measured every width is priced in the shape's own units.
class MtpRoundCostModel final {
public:
    void observe(std::size_t batch_size, std::uint32_t window, double seconds) noexcept {
        if (!valid(batch_size, window) || !(seconds > 0.0)) { return; }
        float& cost           = seconds_[batch_size - 1U][window - 1U];
        std::uint16_t& counts = samples_[batch_size - 1U][window - 1U];
        const auto sample     = static_cast<float>(seconds);
        cost                  = counts == 0 ? sample : cost + kAlpha * (sample - cost);
        if (counts != UINT16_MAX) { ++counts; }
    }

    [[nodiscard]] float cost(std::size_t batch_size, std::uint32_t window) const noexcept {
        if (!valid(batch_size, window)) { return 1.0F; }
        const auto& seconds = seconds_[batch_size - 1U];
        const auto& samples = samples_[batch_size - 1U];
        if (samples[window - 1U] >= kMeasuredRounds) { return seconds[window - 1U]; }
        std::size_t reference = kAdaptiveMtpMaximumDrafts;
        for (std::size_t index = 0; index < kAdaptiveMtpMaximumDrafts; ++index) {
            if (samples[index] >= kMeasuredRounds &&
                (reference == kAdaptiveMtpMaximumDrafts || samples[index] > samples[reference])) {
                reference = index;
            }
        }
        const float shape = kRelativeCost[window - 1U];
        if (reference == kAdaptiveMtpMaximumDrafts) { return shape; }
        return shape * seconds[reference] / kRelativeCost[reference];
    }

private:
    [[nodiscard]] static bool valid(std::size_t batch_size, std::uint32_t window) noexcept {
        return batch_size != 0 && batch_size <= kAdaptiveMtpMaximumBatch && window != 0 &&
               window <= kAdaptiveMtpMaximumDrafts;
    }

    static constexpr float kAlpha                  = 0.125F;
    static constexpr std::uint16_t kMeasuredRounds = 4;
    // Verification cost of 1..15 drafts relative to one, from the upstream adaptive-MTP baseline.
    static constexpr std::array<float, kAdaptiveMtpMaximumDrafts> kRelativeCost{
        1.000F, 1.075F, 1.064F, 1.125F, 1.177F, 1.227F, 1.281F, 1.334F,
        1.388F, 1.442F, 1.496F, 1.550F, 1.604F, 1.658F, 1.712F};

    std::array<std::array<float, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch> seconds_{};
    std::array<std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        samples_{};
};

class MtpAdaptiveBatchController final {
public:
    void reset(std::uint32_t maximum_window) noexcept {
        maximum_window_             = std::clamp(maximum_window, 1U, kAdaptiveMtpMaximumDrafts);
        selected_window_            = default_startup_window();
        candidate_window_           = selected_window_;
        rounds_at_window_           = 0;
        candidate_rounds_           = 0;
        execution_samples_          = {};
        last_width_execution_round_ = {};
        execution_round_            = 0;
        selection_cohort_key_       = 0;
        last_transition_from_       = selected_window_;
        last_transition_to_         = selected_window_;
    }

    void observe_execution(std::uint32_t batch_size, std::uint32_t window,
                           double seconds) noexcept {
        if (batch_size == 0 || batch_size > kAdaptiveMtpMaximumBatch || window == 0 ||
            window > maximum_window_) {
            return;
        }
        costs_.observe(batch_size, window, seconds);
        const std::size_t batch = batch_size - 1U;
        const std::size_t width = window - 1U;
        if (execution_round_ != UINT64_MAX) { ++execution_round_; }
        if (execution_samples_[batch][width] != UINT16_MAX) { ++execution_samples_[batch][width]; }
        last_width_execution_round_[batch][width] = execution_round_;
    }

    // `available` counts each row's ready drafts, `room` its remaining tokens; a new cohort key (a
    // different set of requests) restarts the selection from a short probe.
    [[nodiscard]] std::uint32_t select(std::span<const MtpAdaptiveSignal* const> signals,
                                       std::span<const std::uint32_t> available,
                                       std::span<const std::uint32_t> room,
                                       std::uint64_t cohort_key) noexcept {
        last_transition_from_ = selected_window_;
        last_transition_to_   = selected_window_;
        if (signals.empty() || signals.size() > kAdaptiveMtpMaximumBatch ||
            signals.size() != available.size() || signals.size() != room.size()) {
            return selected_window_;
        }
        if (selection_cohort_key_ != cohort_key) {
            selected_window_            = startup_window(signals, available, room);
            candidate_window_           = selected_window_;
            rounds_at_window_           = 0;
            candidate_rounds_           = 0;
            execution_samples_          = {};
            last_width_execution_round_ = {};
            execution_round_            = 0;
            selection_cohort_key_       = cohort_key;
        }
        const std::uint32_t round_window = selected_window_;

        const std::uint32_t minimum_window = minimum_selectable_window(available);
        if (selected_window_ < minimum_window) {
            selected_window_  = minimum_window;
            candidate_window_ = minimum_window;
            rounds_at_window_ = 0;
            candidate_rounds_ = 0;
        }

        std::uint32_t best_window = selected_window_;
        float best_score          = score(selected_window_, signals, available, room);
        const std::uint32_t upward_limit =
            selected_window_ == 1 && maximum_window_ >= 3 ? 3U : selected_window_ + 1U;
        for (std::uint32_t window = minimum_window; window <= maximum_window_; ++window) {
            if (window > upward_limit) { continue; }
            const float candidate = score(window, signals, available, room);
            if (candidate > best_score) {
                best_score  = candidate;
                best_window = window;
            }
        }

        if (maximum_window_ >= 6 && best_window < 6) {
            bool all_prefixes_strong = true;
            for (const MtpAdaptiveSignal* signal : signals) {
                all_prefixes_strong = all_prefixes_strong && signal->strong_wide_prefix();
            }
            const std::uint32_t floor_window =
                widest_admissible_width(6, best_window, signals.size());
            if (all_prefixes_strong && floor_window != 0) {
                best_window = floor_window;
                best_score  = score(best_window, signals, available, room);
            }
        }

        const float current_score = score(selected_window_, signals, available, room);
        bool confident_probe      = false;
        const std::size_t batch   = signals.size() - 1U;
        if (selected_window_ < maximum_window_) {
            bool has_continuing_row  = false;
            bool all_tails_confident = true;
            for (std::size_t row = 0; row < signals.size(); ++row) {
                if (available[row] < selected_window_ || room[row] <= selected_window_ + 1U) {
                    continue;
                }
                has_continuing_row = true;
                all_tails_confident =
                    all_tails_confident && signals[row]->confident_tail(selected_window_);
            }
            const std::uint32_t preferred_probe = std::min(selected_window_ + 2U, maximum_window_);
            const std::uint32_t probe_window =
                widest_admissible_width(preferred_probe, selected_window_, signals.size());
            if (probe_window != 0) {
                const std::uint64_t observed_round =
                    last_width_execution_round_[batch][probe_window - 1U];
                const bool probe_due =
                    observed_round == 0 || execution_round_ - observed_round >= 32;
                if (has_continuing_row && all_tails_confident && probe_due) {
                    best_window     = probe_window;
                    best_score      = std::max(score(best_window, signals, available, room),
                                               current_score * 1.011F);
                    confident_probe = true;
                }
            }
        }
        const bool material_gain = best_score > current_score * 1.01F;
        if (best_window != selected_window_ && material_gain) {
            if (candidate_window_ == best_window) {
                if (candidate_rounds_ != UINT8_MAX) { ++candidate_rounds_; }
            } else {
                candidate_window_ = best_window;
                candidate_rounds_ = 1;
            }
            const bool high_width_contraction =
                best_window < selected_window_ && selected_window_ >= 6;
            const bool maximum_width_probation =
                selected_window_ == maximum_window_ &&
                execution_samples_[batch][selected_window_ - 1U] <= 2;
            const std::uint8_t minimum_residency         = confident_probe           ? 2
                                                           : maximum_width_probation ? 2
                                                           : high_width_contraction  ? 12
                                                                                     : 3;
            const std::uint8_t required_candidate_rounds = confident_probe           ? 2
                                                           : maximum_width_probation ? 2
                                                                                     : 4;
            if (candidate_rounds_ >= required_candidate_rounds &&
                rounds_at_window_ >= minimum_residency) {
                selected_window_  = best_window;
                candidate_window_ = best_window;
                rounds_at_window_ = 0;
                candidate_rounds_ = 0;
            }
        } else {
            candidate_window_ = selected_window_;
            candidate_rounds_ = 0;
        }
        if (rounds_at_window_ != UINT8_MAX) { ++rounds_at_window_; }
        last_transition_from_ = round_window;
        last_transition_to_   = selected_window_;
        return selected_window_;
    }

    [[nodiscard]] std::uint32_t selected_window() const noexcept { return selected_window_; }

    [[nodiscard]] bool transitioned() const noexcept {
        return last_transition_from_ != last_transition_to_;
    }

private:
    [[nodiscard]] std::uint32_t default_startup_window() const noexcept {
        return std::min(3U, maximum_window_);
    }

    [[nodiscard]] std::uint32_t
    minimum_selectable_window(std::span<const std::uint32_t> available) const noexcept {
        if (maximum_window_ < 3) { return 1; }
        for (const std::uint32_t row_available : available) {
            if (row_available < 3) { return 1; }
        }
        return 3;
    }

    [[nodiscard]] std::uint32_t startup_window(std::span<const MtpAdaptiveSignal* const> signals,
                                               std::span<const std::uint32_t> available,
                                               std::span<const std::uint32_t> room) const noexcept {
        const std::uint32_t startup = default_startup_window();
        std::uint32_t best_window   = minimum_selectable_window(available);
        float best_score            = score(best_window, signals, available, room);
        for (std::uint32_t window = best_window + 1U; window <= startup; ++window) {
            const float candidate = score(window, signals, available, room);
            if (candidate > best_score) {
                best_score  = candidate;
                best_window = window;
            }
        }
        return best_window;
    }

    // Expected tokens per unit cost over this round and the next one at the same width.
    [[nodiscard]] float score(std::uint32_t window,
                              std::span<const MtpAdaptiveSignal* const> signals,
                              std::span<const std::uint32_t> available,
                              std::span<const std::uint32_t> room) const noexcept {
        float expected           = 0.0F;
        float future_expected    = 0.0F;
        float cost               = costs_.cost(signals.size(), window);
        std::size_t future_batch = 0;
        for (std::size_t row = 0; row < signals.size(); ++row) {
            expected += signals[row]->expected_tokens(window, available[row]);
            const std::uint32_t consumed    = std::min(window, available[row]) + 1U;
            const std::uint32_t future_room = room[row] > consumed ? room[row] - consumed : 0U;
            const std::uint32_t future_extent =
                std::min(window, future_room > 0 ? future_room - 1U : 0U);
            if (future_room != 0) {
                future_expected += signals[row]->expected_tokens(window, future_extent);
                ++future_batch;
            }
        }
        if (future_batch != 0) {
            expected += future_expected;
            cost += costs_.cost(future_batch, window);
        }
        return expected / std::max(cost, 1e-6F);
    }

    // A width whose cost exceeds what even full acceptance could repay against a narrower one.
    [[nodiscard]] bool physically_dominated(std::uint32_t window,
                                            std::size_t batch_size) const noexcept {
        const float candidate_cost = costs_.cost(batch_size, window);
        for (std::uint32_t narrower = 1; narrower < window; ++narrower) {
            const float narrower_cost = costs_.cost(batch_size, narrower);
            const float maximum_yield_ratio =
                static_cast<float>(window + 1U) / static_cast<float>(narrower + 1U);
            if (candidate_cost >= narrower_cost * maximum_yield_ratio) { return true; }
        }
        return false;
    }

    [[nodiscard]] std::uint32_t widest_admissible_width(std::uint32_t preferred,
                                                        std::uint32_t lower_exclusive,
                                                        std::size_t batch_size) const noexcept {
        for (std::uint32_t window = preferred; window > lower_exclusive; --window) {
            if (!physically_dominated(window, batch_size)) { return window; }
        }
        return 0;
    }

    MtpRoundCostModel costs_;
    std::uint32_t maximum_window_   = 1;
    std::uint32_t selected_window_  = 1;
    std::uint32_t candidate_window_ = 1;
    std::uint8_t rounds_at_window_  = 0;
    std::uint8_t candidate_rounds_  = 0;
    std::array<std::array<std::uint16_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        execution_samples_{};
    std::array<std::array<std::uint64_t, kAdaptiveMtpMaximumDrafts>, kAdaptiveMtpMaximumBatch>
        last_width_execution_round_{};
    std::uint64_t execution_round_      = 0;
    std::uint64_t selection_cohort_key_ = 0;
    std::uint32_t last_transition_from_ = 1;
    std::uint32_t last_transition_to_   = 1;
};

} // namespace ninfer::models::qwen3_5::detail
