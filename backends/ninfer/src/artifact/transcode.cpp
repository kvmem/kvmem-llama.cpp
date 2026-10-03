#include "artifact/transcode.h"

#include "artifact/schema.h"
#include "core/weight_view.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <thread>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr int kTargetGroup      = 64;
constexpr int kSourceGroup      = 32;
constexpr int kClippingRatios   = 25;
constexpr float kFirstRatio     = 0.70F;
constexpr float kRatioIncrement = 0.02F;

std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t round_shift_nearest_even(std::uint32_t value, int shift) {
    const std::uint32_t mask = (1U << shift) - 1U;
    const std::uint32_t half = 1U << (shift - 1);
    const std::uint32_t base = value >> shift;
    const std::uint32_t rem  = value & mask;
    return base + ((rem > half || (rem == half && (base & 1U) != 0U)) ? 1U : 0U);
}

// IEEE binary32 -> binary16, round to nearest even, overflow to infinity.
std::uint16_t float_to_half(float value) {
    const std::uint32_t x    = float_bits(value);
    const std::uint32_t sign = (x >> 16U) & 0x8000U;
    const std::uint32_t ax   = x & 0x7fffffffU;
    if (ax >= 0x7f800000U) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | ((ax & 0x007fffffU) != 0 ? 0x0200U : 0U));
    }
    int exponent       = static_cast<int>((ax >> 23U) & 0xffU) - 127 + 15;
    std::uint32_t mant = ax & 0x007fffffU;
    if (exponent <= 0) {
        if (exponent < -10) { return static_cast<std::uint16_t>(sign); }
        mant |= 0x00800000U;
        return static_cast<std::uint16_t>(sign | round_shift_nearest_even(mant, 14 - exponent));
    }
    if (exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    std::uint32_t half_mant = round_shift_nearest_even(mant, 13);
    if (half_mant == 0x0400U) {
        half_mant = 0;
        if (++exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10U) |
                                      half_mant);
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(half) & 0x8000U) << 16U;
    std::uint32_t exponent   = (static_cast<std::uint32_t>(half) >> 10U) & 0x1fU;
    std::uint32_t mant       = static_cast<std::uint32_t>(half) & 0x03ffU;
    if (exponent == 0) {
        if (mant == 0) { return bits_float(sign); }
        int e = -14;
        while ((mant & 0x0400U) == 0) {
            mant <<= 1U;
            --e;
        }
        mant &= 0x03ffU;
        return bits_float(sign | (static_cast<std::uint32_t>(e + 127) << 23U) | (mant << 13U));
    }
    if (exponent == 31) { return bits_float(sign | 0x7f800000U | (mant << 13U)); }
    return bits_float(sign | ((exponent - 15 + 127) << 23U) | (mant << 13U));
}

std::uint16_t load_u16(const std::byte* bytes) {
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(bytes[0]) |
                                      (std::to_integer<unsigned>(bytes[1]) << 8U));
}

void store_u16(std::byte* bytes, std::uint16_t value) {
    bytes[0] = static_cast<std::byte>(value & 0xffU);
    bytes[1] = static_cast<std::byte>(value >> 8U);
}

int code_for(float weight, float inverse_scale, int qmin, int qmax) {
    const float scaled = std::nearbyint(weight * inverse_scale);
    return static_cast<int>(std::clamp(scaled, static_cast<float>(qmin), static_cast<float>(qmax)));
}

struct TargetCodec {
    int bits = 0;
    int qmin = 0;
    int qmax = 0;
};

TargetCodec target_codec(QType target) {
    switch (target) {
    case QType::Q4_G64_FP16:
        return {4, -8, 7};
    case QType::Q6_G64_FP16:
        return {6, -32, 31};
    default:
        break;
    }
    throw ArtifactError("row-split transcode target must be Q4_G64_FP16 or Q6_G64_FP16");
}

void transcode_rows(const TargetCodec& codec, const WeightGeometry& source_geometry,
                    const WeightGeometry& target_geometry, std::span<const std::byte> source,
                    std::span<std::byte> destination, std::uint64_t row_begin,
                    std::uint64_t row_end) {
    const std::uint64_t padded_columns = source_geometry.padded_columns;
    const std::uint64_t target_groups  = target_geometry.padded_columns / target_geometry.group_size;
    const std::uint64_t source_groups  = source_geometry.padded_columns / source_geometry.group_size;
    const std::uint64_t low_bytes_per_group  = target_geometry.code_bytes_per_row / target_groups;
    const std::uint64_t high_bytes_per_group = target_geometry.high_bytes_per_row / target_groups;
    const std::byte* const source_codes  = source.data();
    const std::byte* const source_scales = source.data() + source_geometry.scale_offset;
    std::byte* const low                 = destination.data();
    std::byte* const high =
        codec.bits == 6 ? destination.data() + target_geometry.high_offset : nullptr;
    std::byte* const scales = destination.data() + target_geometry.scale_offset;

    std::array<float, kTargetGroup> weights{};
    std::array<int, kTargetGroup> codes{};
    for (std::uint64_t row = row_begin; row < row_end; ++row) {
        for (std::uint64_t group = 0; group < target_groups; ++group) {
            const std::uint64_t column0 = row * padded_columns + group * kTargetGroup;
            const std::uint64_t source_group0 = row * source_groups + group * 2;
            float absmax = 0.0F;
            for (int half = 0; half < 2; ++half) {
                const float scale =
                    half_to_float(load_u16(source_scales + (source_group0 + half) * 2));
                for (int lane = 0; lane < kSourceGroup; ++lane) {
                    const int index = half * kSourceGroup + lane;
                    const auto code = static_cast<std::int8_t>(
                        std::to_integer<unsigned>(source_codes[column0 + index]));
                    weights[index] = static_cast<float>(code) * scale;
                    absmax         = std::max(absmax, std::fabs(weights[index]));
                }
            }

            std::uint16_t best_half = 0;
            float best_scale        = 0.0F;
            if (absmax > 0.0F) {
                double best_error = std::numeric_limits<double>::infinity();
                for (int ratio = 0; ratio < kClippingRatios; ++ratio) {
                    const std::uint16_t candidate = float_to_half(
                        absmax / static_cast<float>(codec.qmax) *
                        (kFirstRatio + kRatioIncrement * static_cast<float>(ratio)));
                    const float scale = half_to_float(candidate);
                    if (!(scale > 0.0F) || !std::isfinite(scale)) { continue; }
                    const float inverse = 1.0F / scale;
                    double error        = 0.0;
                    for (int index = 0; index < kTargetGroup; ++index) {
                        const double residual =
                            static_cast<double>(weights[index]) -
                            static_cast<double>(scale) *
                                code_for(weights[index], inverse, codec.qmin, codec.qmax);
                        error += residual * residual;
                    }
                    if (error < best_error) {
                        best_error = error;
                        best_half  = candidate;
                        best_scale = scale;
                    }
                }
            }
            const float inverse = best_scale > 0.0F ? 1.0F / best_scale : 0.0F;
            for (int index = 0; index < kTargetGroup; ++index) {
                codes[index] = code_for(weights[index], inverse, codec.qmin, codec.qmax);
            }

            const std::uint64_t group_index = row * target_groups + group;
            std::byte* const low_group      = low + group_index * low_bytes_per_group;
            for (int index = 0; index < kTargetGroup; ++index) {
                const auto unsigned_code = static_cast<std::uint32_t>(codes[index]) &
                                           ((1U << codec.bits) - 1U);
                const auto nibble = static_cast<unsigned>(unsigned_code & 0x0fU);
                low_group[index >> 1] |= static_cast<std::byte>((index & 1) != 0 ? nibble << 4U
                                                                                 : nibble);
                if (high != nullptr) {
                    const int bit = index * 2;
                    high[group_index * high_bytes_per_group + (bit >> 3)] |=
                        static_cast<std::byte>(((unsigned_code >> 4U) & 0x03U) << (bit & 7));
                }
            }
            store_u16(scales + group_index * 2, best_half);
        }
    }
}

} // namespace

bool row_split_transcode_supported(QType source, QType target) noexcept {
    return source == QType::Q8_G32_FP16 &&
           (target == QType::Q4_G64_FP16 || target == QType::Q6_G64_FP16);
}

void transcode_row_split(QType target, std::span<const std::uint64_t> shape,
                         std::span<const std::byte> source, std::span<std::byte> destination) {
    const TargetCodec codec = target_codec(target);
    const WeightGeometry source_geometry =
        weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape);
    const WeightGeometry target_geometry = weight_geometry(target, QuantLayout::RowSplit, shape);
    if (source.size() != source_geometry.bytes || destination.size() != target_geometry.bytes) {
        throw ArtifactError("transcode payload sizes do not match the tensor shape");
    }
    if (source_geometry.padded_columns % kTargetGroup != 0 ||
        source_geometry.padded_columns != target_geometry.padded_columns) {
        throw ArtifactError("transcode requires padded columns divisible by 64");
    }
    std::fill(destination.begin(), destination.end(), std::byte{0});

    const std::uint64_t rows = shape[0];
    const std::uint64_t workers =
        std::clamp<std::uint64_t>(std::thread::hardware_concurrency(), 1, std::max<std::uint64_t>(rows, 1));
    if (workers == 1) {
        transcode_rows(codec, source_geometry, target_geometry, source, destination, 0, rows);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers - 1));
    std::vector<std::exception_ptr> failures(static_cast<std::size_t>(workers));
    for (std::uint64_t worker = 1; worker < workers; ++worker) {
        pool.emplace_back([&, worker] {
            try {
                transcode_rows(codec, source_geometry, target_geometry, source, destination,
                               rows * worker / workers, rows * (worker + 1) / workers);
            } catch (...) {
                failures[static_cast<std::size_t>(worker)] = std::current_exception();
            }
        });
    }
    try {
        transcode_rows(codec, source_geometry, target_geometry, source, destination, 0,
                       rows / workers);
    } catch (...) {
        failures[0] = std::current_exception();
    }
    for (std::thread& thread : pool) { thread.join(); }
    for (const std::exception_ptr& failure : failures) {
        if (failure) { std::rethrow_exception(failure); }
    }
}

} // namespace ninfer::artifact
