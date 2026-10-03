// Load-time W8G32 -> Q4G64 / Q6G64 transcoding (artifact/transcode.h), used by --lm-head-q4,
// --lm-head-q6, --embedding-q4 and --mtp-experts-q4 (whose routed down projection is 512 columns).
//
// 1. The encoding is the target format's row-split-k128-v1 geometry, byte for byte, and an
//    independent decoder reads it: every 64-column group's squared error against the W8 values is no
//    worse than plain absmax/qmax rounding (one of the candidate scales), for full, narrow and padded
//    column counts, including all-zero groups.
// 2. A transcoded 248320x5120 head routed through ops::linear at T = 1, 4, 8, 16, 24 and 32 meets the
//    A16 linear criterion (tests/ops/linear/linear_test_common.cpp) per output column against an fp64
//    oracle over the independently decoded weights on sampled rows.

#include "artifact/transcode.h"
#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/small_t_oracle.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
namespace qw     = ninfer::test::quantized_weight;
namespace oracle = ninfer::test::small_t_oracle;

struct Codec {
    QType qtype;
    const char* name;
};

constexpr std::array<Codec, 2> kCodecs{{
    {QType::Q4_G64_FP16, "Q4G64"},
    {QType::Q6_G64_FP16, "Q6G64"},
}};

ninfer::WeightGeometry target_geometry(const Codec& codec, std::span<const std::uint64_t> shape) {
    return ninfer::weight_geometry(codec.qtype, ninfer::QuantLayout::RowSplit, shape);
}

std::vector<std::byte> transcode(const Codec& codec, const qw::PackedWeight& source,
                                 std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(rows),
                                             static_cast<std::uint64_t>(columns)};
    const auto geometry = target_geometry(codec, shape);
    std::vector<std::byte> out(geometry.bytes);
    ninfer::artifact::transcode_row_split(
        codec.qtype, shape,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(source.payload.data()),
                                   source.payload.size()),
        out);
    return out;
}

// Decodes one row of a transcoded payload with the test packer's independent unpacker.
std::vector<float> decode_row(const Codec& codec, const std::vector<std::byte>& payload,
                              std::int32_t rows, std::int32_t columns, std::int32_t row) {
    const qw::detail::QuantSpec spec = qw::detail::quant_spec(codec.qtype);
    const std::int32_t padded        = qw::detail::align_up(columns, 128);
    const std::int32_t groups        = padded / 64;
    const std::size_t low_bytes      = static_cast<std::size_t>(rows) * groups * 32;
    const std::size_t high_bpr       = static_cast<std::size_t>(qw::detail::high_bytes_per_group(spec));
    const std::size_t high_offset    = qw::detail::align_up_size(low_bytes, 256);
    const std::size_t scale_offset =
        high_offset + qw::detail::align_up_size(static_cast<std::size_t>(rows) * groups * high_bpr, 256);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(payload.data());
    std::vector<float> out(static_cast<std::size_t>(padded));
    for (std::int32_t group = 0; group < groups; ++group) {
        const std::size_t index = static_cast<std::size_t>(row) * groups + group;
        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, bytes + scale_offset + index * 2, 2);
        const float scale = qw::detail::f16_to_f32(scale_bits);
        for (int lane = 0; lane < 64; ++lane) {
            const int code = qw::detail::unpack_lowbit_code(
                bytes + index * 32, high_bpr == 0 ? nullptr : bytes + high_offset + index * high_bpr,
                spec, lane);
            out[static_cast<std::size_t>(group) * 64 + lane] = static_cast<float>(code) * scale;
        }
    }
    return out;
}

// W8 values of one row, decoded independently from the source payload (padded columns included).
std::vector<float> source_row(const qw::PackedWeight& source, std::int32_t rows,
                              std::int32_t columns, std::int32_t row) {
    const std::int32_t padded = qw::detail::align_up(columns, 128);
    const std::int32_t groups = padded / 32;
    const std::size_t scale_offset =
        qw::detail::align_up_size(static_cast<std::size_t>(rows) * groups * 32, 256);
    std::vector<float> out(static_cast<std::size_t>(padded));
    for (std::int32_t group = 0; group < groups; ++group) {
        const std::size_t index = static_cast<std::size_t>(row) * groups + group;
        const float scale =
            qw::detail::f16_to_f32(qw::detail::load_u16_le(source.payload, scale_offset + index * 2));
        for (int lane = 0; lane < 32; ++lane) {
            const auto code = static_cast<std::int8_t>(source.payload[index * 32 + lane]);
            out[static_cast<std::size_t>(group) * 32 + lane] = static_cast<float>(code) * scale;
        }
    }
    return out;
}

int check_group_errors(const Codec& codec, const qw::PackedWeight& source,
                       const std::vector<std::byte>& encoded, std::int32_t rows,
                       std::int32_t columns, std::int32_t row, const std::string& label) {
    const qw::detail::QuantSpec spec = qw::detail::quant_spec(codec.qtype);
    const std::vector<float> reference = source_row(source, rows, columns, row);
    const std::vector<float> decoded   = decode_row(codec, encoded, rows, columns, row);
    int failures                       = 0;
    for (std::size_t group = 0; group * 64 < reference.size(); ++group) {
        float absmax = 0.0F;
        for (int lane = 0; lane < 64; ++lane) {
            absmax = std::max(absmax, std::fabs(reference[group * 64 + lane]));
        }
        const float rtn_scale =
            qw::detail::f16_to_f32(qw::detail::f32_to_f16(absmax / static_cast<float>(spec.qmax)));
        double error = 0.0, rtn_error = 0.0;
        for (int lane = 0; lane < 64; ++lane) {
            const double w = reference[group * 64 + lane];
            error += (w - decoded[group * 64 + lane]) * (w - decoded[group * 64 + lane]);
            double rtn = 0.0;
            if (rtn_scale > 0.0F) {
                const float code = std::clamp(std::nearbyint(static_cast<float>(w) / rtn_scale),
                                              static_cast<float>(spec.qmin),
                                              static_cast<float>(spec.qmax));
                rtn              = static_cast<double>(code) * rtn_scale;
            }
            rtn_error += (w - rtn) * (w - rtn);
        }
        if (error > rtn_error * (1.0 + 1e-5) + 1e-12) {
            if (failures++ < 3) {
                std::cerr << label << " row " << row << " group " << group << ": error " << error
                          << " worse than absmax/qmax rounding " << rtn_error << '\n';
            }
        }
    }
    return failures;
}

std::vector<float> random_matrix(std::int32_t rows, std::int32_t columns, std::uint64_t seed) {
    std::vector<float> values(static_cast<std::size_t>(rows) * columns);
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < values.size(); ++i) {
        state = qw::detail::mix64(state);
        // Heavy-ish tails so clipping ratios below 1.0 are actually selected.
        const double u = (static_cast<double>(state >> 11) + 0.5) / 9007199254740992.0;
        const double centered = u - 0.5;
        values[i] = static_cast<float>(0.02 * centered * (1.0 + 30.0 * centered * centered * centered * centered));
    }
    // One all-zero row exercises zero scales.
    std::fill(values.begin(), values.begin() + columns, 0.0F);
    return values;
}

int test_codec_layout() {
    int failures = 0;
    for (const auto& [rows, columns] :
         std::array<std::pair<std::int32_t, std::int32_t>, 4>{{{257, 5120}, {64, 2048}, {96, 512}, {33, 100}}}) {
        const qw::PackedWeight source =
            qw::pack_q8_g32_row_split(random_matrix(rows, columns, 0x51ULL + columns), rows, columns);
        for (const Codec& codec : kCodecs) {
            const std::string label = std::string(codec.name) + " [" + std::to_string(rows) + "," +
                                      std::to_string(columns) + "]";
            const std::vector<std::byte> encoded = transcode(codec, source, rows, columns);
            for (std::int32_t row = 0; row < rows; ++row) {
                failures += check_group_errors(codec, source, encoded, rows, columns, row, label);
            }
            const std::vector<float> zero = decode_row(codec, encoded, rows, columns, 0);
            if (std::any_of(zero.begin(), zero.end(), [](float v) { return v != 0.0F; })) {
                ++failures;
                std::cerr << label << ": all-zero row did not decode to zero\n";
            }
        }
    }
    return failures;
}

int test_routed_head(const Codec& codec) {
    constexpr std::int32_t kRows      = 248320;
    constexpr std::int32_t kColumns   = 5120;
    constexpr std::int32_t kMaxTokens = 32;
    qw::PatternedWeightOptions options;
    options.row_split_codes = qw::RowSplitCodePattern::Hashed;
    options.row_split_scale = qw::RowSplitScalePattern::Small;
    const qw::PackedWeight source =
        qw::make_patterned_weight(QType::Q8_G32_FP16, kRows, kColumns, 0x1a4dU, options);
    const std::vector<std::byte> encoded = transcode(codec, source, kRows, kColumns);

    const std::array<std::uint64_t, 2> shape{kRows, kColumns};
    const auto geometry = target_geometry(codec, shape);
    ninfer::test::GuardedDeviceBuffer device(encoded.size());
    device.copy_from_host(encoded.data(), encoded.size());
    ninfer::Weight weight{};
    auto* base              = static_cast<std::uint8_t*>(device.data());
    weight.qtype            = codec.qtype;
    weight.layout           = ninfer::QuantLayout::RowSplit;
    weight.scale_dtype      = DType::FP16;
    weight.payload          = base;
    weight.payload_bytes    = geometry.bytes;
    weight.high_plane_bytes = geometry.high_bytes;
    weight.qdata            = base;
    weight.qhigh  = geometry.high_bytes == 0 ? nullptr : base + geometry.high_offset;
    weight.scales = base + geometry.scale_offset;
    weight.group_size      = 64;
    weight.group           = 64;
    weight.ndim            = 2;
    weight.shape[0]        = kRows;
    weight.shape[1]        = kColumns;
    weight.padded_shape[0] = kRows;
    weight.padded_shape[1] = kColumns;
    weight.n               = kRows;
    weight.k               = kColumns;

    std::vector<std::int32_t> rows;
    for (std::int32_t row = 0; row < kRows; row += 997) { rows.push_back(row); }
    rows.push_back(kRows - 1);
    std::vector<float> decoded(rows.size() * kColumns);
    int failures = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        failures += check_group_errors(codec, source, encoded, kRows, kColumns, rows[r],
                                       std::string(codec.name) + " head");
        const std::vector<float> row = decode_row(codec, encoded, kRows, kColumns, rows[r]);
        std::copy(row.begin(), row.begin() + kColumns, decoded.begin() + r * kColumns);
    }

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(kColumns) * kMaxTokens);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (auto& value : activation) {
        state               = qw::detail::mix64(state);
        const int numerator = static_cast<int>(state % 255U) - 127;
        value               = oracle::f32_to_bf16_rne(static_cast<float>(numerator) * 1e-2F);
    }
    const std::vector<double> expected = oracle::project(
        decoded, static_cast<std::int32_t>(rows.size()), kColumns, activation, kMaxTokens);
    ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
    device_x.copy_from_host(activation.data(), activation.size() * 2);
    ninfer::test::GuardedDeviceBuffer device_out(static_cast<std::size_t>(kRows) * kMaxTokens * 2);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kRows) * kMaxTokens);
    for (const std::int32_t tokens : {1, 4, 8, 16, 24, 32}) {
        Tensor x(device_x.data(), DType::BF16, {kColumns, tokens});
        Tensor y(device_out.data(), DType::BF16, {kRows, tokens});
        device_out.fill(0xff);
        ninfer::ops::linear(x, weight, y, nullptr);
        ninfer::test::cuda_check(cudaDeviceSynchronize(), "transcoded vocabulary linear");
        device_out.copy_to_host(out.data(), static_cast<std::size_t>(kRows) * tokens * 2);
        // One BF16 unit roundoff of relative L2, and a gross bound of two BF16 steps against the
        // column's largest reference, exactly as the A16 linear conformance tests qualify routes.
        constexpr ninfer::test::ReductionCriterion kA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
        std::vector<double> actual(rows.size());
        std::vector<double> reference(rows.size());
        for (std::int32_t col = 0; col < tokens; ++col) {
            for (std::size_t r = 0; r < rows.size(); ++r) {
                actual[r] = oracle::bf16_to_f32(out[static_cast<std::size_t>(col) * kRows + rows[r]]);
                reference[r] = expected[static_cast<std::size_t>(col) * rows.size() + r];
            }
            const auto stats = ninfer::test::compute_reduction_stats(
                actual.data(), reference.data(), static_cast<std::int64_t>(rows.size()));
            if (!ninfer::test::reduction_passes(stats, static_cast<std::int64_t>(rows.size()), kA16)) {
                ++failures;
                std::cerr << codec.name << " T=" << tokens << " col " << col
                          << ": relative L2 " << stats.relative_l2 << ", max error "
                          << stats.maximum_absolute_error << " (limit "
                          << ninfer::test::gross_error_limit(stats, kA16) << ")\n";
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        int failures = test_codec_layout();
        if (ninfer::test::cuda_unavailable()) {
            std::cout << (failures == 0 ? "OK" : "FAIL")
                      << " vocabulary transcode codec (routed head skipped: no CUDA device)\n";
            return failures == 0 ? 0 : 1;
        }
        for (const Codec& codec : kCodecs) { failures += test_routed_head(codec); }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8G32 vocabulary transcode\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "vocabulary transcode test failed: " << error.what() << '\n';
        return 1;
    }
}
