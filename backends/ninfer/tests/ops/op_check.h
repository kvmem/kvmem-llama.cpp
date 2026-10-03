#pragma once

// Shared pointwise comparison mechanics for Op tests. Each semantic Op owns the
// concrete criterion used by its suite; this file deliberately defines no
// cross-Op tolerance presets.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ninfer::test {

struct PointwiseCriterion {
    double absolute;
    double relative;
};

struct PointwiseStats {
    double maximum_absolute_error    = 0.0;
    double maximum_relative_error    = 0.0;
    double maximum_criterion_ratio   = 0.0;
    std::int64_t maximum_error_index = -1;
    std::int64_t maximum_ratio_index = -1;
    double actual_at_maximum         = 0.0;
    double reference_at_maximum      = 0.0;
    std::int64_t first_violation     = -1;
    std::int64_t non_finite_count    = 0;
};

inline PointwiseStats compute_pointwise_stats(const double* actual, const double* reference,
                                              std::int64_t count,
                                              const PointwiseCriterion& criterion) {
    PointwiseStats stats;
    for (std::int64_t index = 0; index < count; ++index) {
        const double got      = actual[index];
        const double expected = reference[index];
        if (!std::isfinite(got) || !std::isfinite(expected)) {
            ++stats.non_finite_count;
            if (stats.first_violation < 0) stats.first_violation = index;
            continue;
        }

        const double absolute_error = std::abs(got - expected);
        const double scale          = std::max(std::abs(got), std::abs(expected));
        const double relative_error = scale == 0.0 ? 0.0 : absolute_error / scale;
        if (absolute_error > stats.maximum_absolute_error) {
            stats.maximum_absolute_error = absolute_error;
            stats.maximum_error_index    = index;
            stats.actual_at_maximum      = got;
            stats.reference_at_maximum   = expected;
        }
        stats.maximum_relative_error = std::max(stats.maximum_relative_error, relative_error);

        const double limit = criterion.absolute + criterion.relative * std::abs(expected);
        const double criterion_ratio =
            limit == 0.0 ? (absolute_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                         : absolute_error / limit;
        if (criterion_ratio > stats.maximum_criterion_ratio) {
            stats.maximum_criterion_ratio = criterion_ratio;
            stats.maximum_ratio_index     = index;
        }
        if (absolute_error > limit && stats.first_violation < 0) { stats.first_violation = index; }
    }
    return stats;
}

inline bool pointwise_passes(const PointwiseStats& stats, std::int64_t count) {
    return count > 0 && stats.non_finite_count == 0 && stats.first_violation < 0;
}

struct ReductionCriterion {
    double relative_l2;
    double gross_absolute;
    double gross_relative_to_max_reference;
};

// Floor for `gross_relative_to_max_reference` when the compared output is stored as BF16.
//
// BF16 keeps eight significand bits, so for a value in [2^e, 2^(e+1)) the representable spacing is
// 2^(e-7) and round-to-nearest costs up to half of that. Relative to the value that is at most
// 2^-8 = 3.906e-3, reached at the bottom of a binade. `gross_relative_to_max_reference` bounds the
// single worst element against the largest reference value in the tensor, so a limit below 2^-8
// requires that element to round at least as well as the FP32 oracle -- which is a property of the
// input, not of the kernel.
//
// Measured across every Op test carrying such a criterion (NINFER_OP_REPORT_STATS=1, worst case per
// criterion, expressed in the 2^-8 step above):
//
//     limit          worst observed error      worst / limit
//     0.51-1.46 ULP        0.66-1.46 ULP          0.77-0.99
//
// The bound and the error being measured are the same quantity, which is why every one of them sat
// within a fifth of failing. Two of the attention criteria were already *below* their own observed
// error and passed only because `gross_absolute` carried them, so their relative term was doing no
// work at all.
//
// Two steps leaves room for the kernel accumulating in a different order than the oracle while
// still bounding a genuinely wrong element far more tightly than any real defect: the sm_86 small-T
// INT8 regression ran 3-11x the reference, i.e. hundreds of times this bound. It also keeps the
// storage profiles ordered by lossiness -- bf16/int8/rk8v4 at 2.00 steps, fp8 at 2.30, nvfp4/k8v4
// at 2.82 -- rather than tightest-on-the-hardest.
//
// This value is not new. Several tests already spell it `2.0 * kBf16UnitRoundoff` or `2.0 / 256.0`
// (tests/ops/linear/linear_test_common.cpp, linear/test_bf16_a16.cpp, test_attn_input_proj.cpp,
// test_gdn_input_proj.cpp, test_gdn_input_proj_conv_snapshot.cpp, test_linear_topk.cu), where
// kBf16UnitRoundoff is that same 2^-8 step. Naming it once makes a house convention that half the
// tree already followed apply to the other half, instead of each Op picking a number by hand.
//
// This is only the gross bound, whose job is catching one badly wrong element. `relative_l2` is the
// criterion that constrains kernel accuracy and is deliberately untouched by this floor.
inline constexpr double kBf16GrossRelativeFloor = 2.0 / 256.0; // 2^-7, two BF16 rounding steps

struct ReductionStats {
    double relative_l2                = 0.0;
    double root_mean_squared_error    = 0.0;
    double reference_root_mean_square = 0.0;
    double maximum_absolute_error     = 0.0;
    double maximum_absolute_reference = 0.0;
    std::int64_t maximum_error_index  = -1;
    std::int64_t first_non_finite     = -1;
    double actual_at_maximum          = 0.0;
    double reference_at_maximum       = 0.0;
};

inline ReductionStats compute_reduction_stats(const double* actual, const double* reference,
                                              std::int64_t count) {
    ReductionStats stats;
    long double squared_error     = 0.0L;
    long double squared_reference = 0.0L;
    for (std::int64_t index = 0; index < count; ++index) {
        const double got      = actual[index];
        const double expected = reference[index];
        if (!std::isfinite(got) || !std::isfinite(expected)) {
            if (stats.first_non_finite < 0) stats.first_non_finite = index;
            continue;
        }

        const double error          = got - expected;
        const double absolute_error = std::abs(error);
        squared_error += static_cast<long double>(error) * error;
        squared_reference += static_cast<long double>(expected) * expected;
        stats.maximum_absolute_reference =
            std::max(stats.maximum_absolute_reference, std::abs(expected));
        if (absolute_error > stats.maximum_absolute_error) {
            stats.maximum_absolute_error = absolute_error;
            stats.maximum_error_index    = index;
            stats.actual_at_maximum      = got;
            stats.reference_at_maximum   = expected;
        }
    }
    stats.relative_l2 = std::sqrt(static_cast<double>(squared_error)) /
                        std::max(std::sqrt(static_cast<double>(squared_reference)), 1.0e-30);
    if (count > 0) {
        stats.root_mean_squared_error =
            std::sqrt(static_cast<double>(squared_error / static_cast<long double>(count)));
        stats.reference_root_mean_square =
            std::sqrt(static_cast<double>(squared_reference / static_cast<long double>(count)));
    }
    return stats;
}

inline double gross_error_limit(const ReductionStats& stats, const ReductionCriterion& criterion) {
    return criterion.gross_absolute +
           criterion.gross_relative_to_max_reference * stats.maximum_absolute_reference;
}

inline bool reduction_passes(const ReductionStats& stats, std::int64_t count,
                             const ReductionCriterion& criterion) {
    return count > 0 && stats.first_non_finite < 0 && stats.relative_l2 <= criterion.relative_l2 &&
           stats.maximum_absolute_error <= gross_error_limit(stats, criterion);
}

} // namespace ninfer::test
