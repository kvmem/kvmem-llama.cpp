#include "models/qwen3_5/execution/vision_cpu.h"
#include "models/qwen3_5/load/vision_cpu.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace qw = ninfer::test::quantized_weight;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::models::qwen3_5::CpuVisionWeights;
using ninfer::models::qwen3_5::VisionItemControl;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::span<const std::byte> bytes_of(const std::vector<std::uint8_t>& payload) {
    return std::as_bytes(std::span<const std::uint8_t>(payload));
}

void test_decode_matches_reference(QType format, QuantLayout layout, std::int32_t n, std::int32_t k,
                                   qw::PatternedWeightOptions options = {}) {
    const qw::PackedWeight packed = qw::make_patterned_weight(format, n, k, 11U, options);
    const std::uint64_t shape[]{static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)};
    const ninfer::WeightGeometry geometry = ninfer::weight_geometry(format, layout, shape);
    const std::string name                = "format " + std::to_string(static_cast<int>(format));
    check(packed.payload.size() >= geometry.bytes, name + ": fixture is shorter than its geometry");
    const std::vector<float> decoded =
        ninfer::models::qwen3_5::decode_tensor(geometry, bytes_of(packed.payload));
    double worst = 0.0;
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            const double expected = qw::logical_weight_fp64(packed, row, column);
            const double actual   = decoded[static_cast<std::size_t>(row) * k + column];
            worst =
                std::max(worst, std::abs(actual - expected) / std::max(1e-30, std::abs(expected)));
        }
    }
    check(worst <= 1e-6, name + ": decode differs from the reference by " + std::to_string(worst));
}

void test_contiguous_decode() {
    const std::vector<float> values{1.5F, -0.25F, 3.0e-3F, 42.0F, -7.75F, 0.0F};
    std::vector<std::uint8_t> bf16(values.size() * 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto word =
            static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(values[i]) >> 16U);
        bf16[2 * i]     = static_cast<std::uint8_t>(word & 0xffU);
        bf16[2 * i + 1] = static_cast<std::uint8_t>(word >> 8U);
    }
    const std::uint64_t shape[]{2, 3};
    const auto bf16_geometry = ninfer::weight_geometry(QType::BF16, QuantLayout::Contiguous, shape);
    const auto decoded = ninfer::models::qwen3_5::decode_tensor(bf16_geometry, bytes_of(bf16));
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float truncated =
            std::bit_cast<float>(std::bit_cast<std::uint32_t>(values[i]) & 0xffff0000U);
        check(decoded[i] == truncated, "BF16 element " + std::to_string(i) + " was not widened");
    }
    std::vector<std::uint8_t> fp32(values.size() * 4);
    std::memcpy(fp32.data(), values.data(), fp32.size());
    const auto fp32_geometry = ninfer::weight_geometry(QType::FP32, QuantLayout::Contiguous, shape);
    check(ninfer::models::qwen3_5::decode_tensor(fp32_geometry, bytes_of(fp32)) == values,
          "FP32 tensor was not copied");
}

// A small tower with the real head geometry: two 72-wide heads, a 2x2 merge.
CpuVisionWeights tiny_tower(std::uint32_t seed) {
    CpuVisionWeights w;
    w.hidden        = 144;
    w.heads         = 2;
    w.intermediate  = 96;
    w.patch_width   = 24;
    w.merge_unit    = 4;
    w.output_hidden = 40;
    w.position_rows = 16;
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const auto fill = [&](std::size_t count, float scale, float offset = 0.0F) {
        std::vector<float> out(count);
        for (float& value : out) { value = offset + scale * normal(random); }
        return out;
    };
    const auto h           = static_cast<std::size_t>(w.hidden);
    const auto i           = static_cast<std::size_t>(w.intermediate);
    const auto m           = static_cast<std::size_t>(w.merger_width());
    const auto o           = static_cast<std::size_t>(w.output_hidden);
    w.patch_embedding      = fill(h * w.patch_width, 0.2F);
    w.patch_embedding_bias = fill(h, 0.1F);
    w.position_embedding   = fill(w.position_rows * h, 0.3F);
    for (int layer = 0; layer < 2; ++layer) {
        CpuVisionWeights::Layer l;
        l.norm1_weight = fill(h, 0.1F, 1.0F);
        l.norm1_bias   = fill(h, 0.05F);
        l.qkv          = fill(3 * h * h, 0.08F);
        l.qkv_bias     = fill(3 * h, 0.05F);
        l.output       = fill(h * h, 0.08F);
        l.output_bias  = fill(h, 0.05F);
        l.norm2_weight = fill(h, 0.1F, 1.0F);
        l.norm2_bias   = fill(h, 0.05F);
        l.fc1          = fill(i * h, 0.08F);
        l.fc1_bias     = fill(i, 0.05F);
        l.fc2          = fill(h * i, 0.1F);
        l.fc2_bias     = fill(h, 0.05F);
        w.layers.push_back(std::move(l));
    }
    w.merger_norm_weight = fill(h, 0.1F, 1.0F);
    w.merger_norm_bias   = fill(h, 0.05F);
    w.merger_fc1         = fill(m * m, 0.04F);
    w.merger_fc1_bias    = fill(m, 0.05F);
    w.merger_fc2         = fill(o * m, 0.04F);
    w.merger_fc2_bias    = fill(o, 0.05F);
    w.validate();
    return w;
}

// Two frames of a 4x4 patch grid in the processor's merge order, one attention segment per frame.
VisionItemControl two_frame_control() {
    VisionItemControl control;
    constexpr std::int32_t frames = 2, side = 4, merge = 2;
    const std::int32_t patches = frames * side * side;
    control.patch_count        = static_cast<std::size_t>(patches);
    control.merged_count       = static_cast<std::size_t>(patches / (merge * merge));
    control.segment_length     = side * side;
    control.segment_count      = frames;
    control.position_ids.resize(static_cast<std::size_t>(patches) * 2);
    std::int32_t cursor = 0;
    for (std::int32_t t = 0; t < frames; ++t) {
        for (std::int32_t by = 0; by < side / merge; ++by) {
            for (std::int32_t bx = 0; bx < side / merge; ++bx) {
                for (std::int32_t iy = 0; iy < merge; ++iy) {
                    for (std::int32_t ix = 0; ix < merge; ++ix) {
                        const std::int32_t y = by * merge + iy, x = bx * merge + ix;
                        control.position_ids[static_cast<std::size_t>(cursor)]           = y;
                        control.position_ids[static_cast<std::size_t>(patches + cursor)] = x;
                        const std::int32_t base = y * side + x;
                        control.position_table_indices.insert(
                            control.position_table_indices.end(),
                            {base, (base + 1) % 16, (base + 4) % 16, (base + 5) % 16});
                        control.position_table_weights.insert(control.position_table_weights.end(),
                                                              {0.4F, 0.3F, 0.2F, 0.1F});
                        ++cursor;
                    }
                }
            }
        }
    }
    return control;
}

std::vector<std::uint16_t> patterned_patches(const CpuVisionWeights& w, std::size_t patches) {
    std::vector<std::uint16_t> out(patches * static_cast<std::size_t>(w.patch_width));
    std::mt19937 random(7U);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    for (std::uint16_t& word : out) {
        word = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(uniform(random)) >> 16U);
    }
    return out;
}

double widen(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

// A token-major FP64 transcription of the device encoder, kept deliberately naive.
std::vector<double> reference_encode(const CpuVisionWeights& w,
                                     const std::vector<std::uint16_t>& patches,
                                     const VisionItemControl& control) {
    const int P = static_cast<int>(control.patch_count), H = w.hidden, D = 72;
    const int heads = w.heads, S = control.segment_length,
              V        = static_cast<int>(control.merged_count);
    using Rows         = std::vector<std::vector<double>>;
    const auto project = [](const std::vector<float>& weight, const std::vector<float>& bias,
                            const std::vector<double>& in, int n) {
        const int k = static_cast<int>(in.size());
        std::vector<double> out(static_cast<std::size_t>(n));
        for (int r = 0; r < n; ++r) {
            double sum = bias[static_cast<std::size_t>(r)];
            for (int c = 0; c < k; ++c) {
                sum += static_cast<double>(weight[static_cast<std::size_t>(r) * k + c]) * in[c];
            }
            out[r] = sum;
        }
        return out;
    };
    const auto norm = [](const std::vector<double>& in, const std::vector<float>& weight,
                         const std::vector<float>& bias) {
        double mean = 0.0, variance = 0.0;
        for (double v : in) { mean += v; }
        mean /= static_cast<double>(in.size());
        for (double v : in) { variance += (v - mean) * (v - mean); }
        variance /= static_cast<double>(in.size());
        std::vector<double> out(in.size());
        for (std::size_t i = 0; i < in.size(); ++i) {
            out[i] = (in[i] - mean) / std::sqrt(variance + 1e-6) * weight[i] + bias[i];
        }
        return out;
    };
    Rows x(static_cast<std::size_t>(P));
    for (int p = 0; p < P; ++p) {
        std::vector<double> in(static_cast<std::size_t>(w.patch_width));
        for (int f = 0; f < w.patch_width; ++f) {
            in[f] = widen(patches[static_cast<std::size_t>(p) * w.patch_width + f]);
        }
        x[p] = project(w.patch_embedding, w.patch_embedding_bias, in, H);
        for (int c = 0; c < 4; ++c) {
            const int index = control.position_table_indices[static_cast<std::size_t>(p) * 4 + c];
            const double weight =
                control.position_table_weights[static_cast<std::size_t>(p) * 4 + c];
            for (int h = 0; h < H; ++h) {
                x[p][h] += w.position_embedding[static_cast<std::size_t>(index) * H + h] * weight;
            }
        }
    }
    for (const auto& layer : w.layers) {
        Rows q(P), k(P), v(P);
        for (int p = 0; p < P; ++p) {
            const auto qkv = project(layer.qkv, layer.qkv_bias,
                                     norm(x[p], layer.norm1_weight, layer.norm1_bias), 3 * H);
            q[p].assign(qkv.begin(), qkv.begin() + H);
            k[p].assign(qkv.begin() + H, qkv.begin() + 2 * H);
            v[p].assign(qkv.begin() + 2 * H, qkv.end());
            for (int head = 0; head < heads; ++head) {
                for (int i = 0; i < 36; ++i) {
                    const int axis         = i / 18;
                    const double frequency = std::pow(10000.0, -2.0 * (i % 18) / 36.0);
                    const double angle =
                        control.position_ids[static_cast<std::size_t>(axis) * P + p] * frequency;
                    for (auto* row : {&q[p], &k[p]}) {
                        const double a = (*row)[head * D + i], b = (*row)[head * D + i + 36];
                        (*row)[head * D + i]      = a * std::cos(angle) - b * std::sin(angle);
                        (*row)[head * D + i + 36] = b * std::cos(angle) + a * std::sin(angle);
                    }
                }
            }
        }
        for (int p = 0; p < P; ++p) {
            std::vector<double> attended(static_cast<std::size_t>(H), 0.0);
            const int s0 = (p / S) * S;
            for (int head = 0; head < heads; ++head) {
                std::vector<double> score(static_cast<std::size_t>(S));
                double top = -1e300;
                for (int j = 0; j < S; ++j) {
                    double dot = 0.0;
                    for (int d = 0; d < D; ++d) {
                        dot += q[p][head * D + d] * k[s0 + j][head * D + d];
                    }
                    score[j] = dot / std::sqrt(72.0);
                    top      = std::max(top, score[j]);
                }
                double sum = 0.0;
                for (double& value : score) { sum += (value = std::exp(value - top)); }
                for (int d = 0; d < D; ++d) {
                    double value = 0.0;
                    for (int j = 0; j < S; ++j) {
                        value += score[j] / sum * v[s0 + j][head * D + d];
                    }
                    attended[head * D + d] = value;
                }
            }
            const auto projected = project(layer.output, layer.output_bias, attended, H);
            for (int h = 0; h < H; ++h) { x[p][h] += projected[h]; }
        }
        for (int p = 0; p < P; ++p) {
            auto up = project(layer.fc1, layer.fc1_bias,
                              norm(x[p], layer.norm2_weight, layer.norm2_bias), w.intermediate);
            for (double& value : up) {
                value = 0.5 * value *
                        (1.0 + std::tanh(std::sqrt(2.0 / M_PI) *
                                         (value + 0.044715 * value * value * value)));
            }
            const auto down = project(layer.fc2, layer.fc2_bias, up, H);
            for (int h = 0; h < H; ++h) { x[p][h] += down[h]; }
        }
    }
    std::vector<double> out(static_cast<std::size_t>(V) * w.output_hidden);
    for (int token = 0; token < V; ++token) {
        std::vector<double> merged;
        for (int c = 0; c < w.merge_unit; ++c) {
            const auto normed =
                norm(x[token * w.merge_unit + c], w.merger_norm_weight, w.merger_norm_bias);
            merged.insert(merged.end(), normed.begin(), normed.end());
        }
        auto hidden = project(w.merger_fc1, w.merger_fc1_bias, merged, w.merger_width());
        for (double& value : hidden) {
            value = 0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0)));
        }
        const auto result = project(w.merger_fc2, w.merger_fc2_bias, hidden, w.output_hidden);
        std::copy(result.begin(), result.end(),
                  out.begin() + static_cast<std::ptrdiff_t>(token) * w.output_hidden);
    }
    return out;
}

void test_encoder_matches_reference() {
    namespace exec             = ninfer::models::qwen3_5::execution;
    const CpuVisionWeights w   = tiny_tower(3U);
    const VisionItemControl ct = two_frame_control();
    const auto patches         = patterned_patches(w, ct.patch_count);
    const auto single          = exec::encode_vision_on_cpu(w, patches, ct, 1);
    const auto parallel        = exec::encode_vision_on_cpu(w, patches, ct, 4);
    check(single == parallel, "the encode depends on its thread count");
    const auto expected = reference_encode(w, patches, ct);
    check(single.size() == expected.size(), "the encode returned the wrong element count");
    double scale = 0.0, worst = 0.0;
    for (double value : expected) { scale = std::max(scale, std::abs(value)); }
    for (std::size_t i = 0; i < std::min(single.size(), expected.size()); ++i) {
        worst = std::max(worst, std::abs(widen(single[i]) - expected[i]));
    }
    check(worst <= 1e-2 * scale, "the encode differs from the FP64 reference by " +
                                     std::to_string(worst) + " of " + std::to_string(scale));

    std::atomic<bool> cancelled{true};
    check(exec::encode_vision_on_cpu(w, patches, ct, 2, &cancelled).empty(),
          "a cancelled encode returned embeddings");

    auto shared             = std::make_shared<const CpuVisionWeights>(w);
    auto payload            = std::make_shared<ninfer::models::qwen3_5::PreparedMediaPayload>();
    payload->patches        = std::make_unique<std::uint16_t[]>(patches.size());
    payload->patch_elements = patches.size();
    std::copy(patches.begin(), patches.end(), payload->patches.get());
    {
        exec::CpuVisionSession session(shared);
        session.submit_item(payload, ct);
        check(session.pending(), "a submitted item is not pending");
        const auto bytes = session.complete_item();
        check(!session.pending() && bytes.size() == single.size() * 2 &&
                  std::memcmp(bytes.data(), single.data(), bytes.size()) == 0,
              "the session's embeddings differ from the synchronous encode");
        check(session.encode_seconds() > 0.0, "the session did not time its encode");
    }
    {
        // Destroying a session with an item in flight cancels and joins it.
        exec::CpuVisionSession session(shared);
        session.submit_item(payload, ct);
    }
    VisionItemControl mismatched = ct;
    mismatched.patch_count       = ct.patch_count - 4;
    bool rejected                = false;
    try {
        (void)exec::encode_vision_on_cpu(w, patches, mismatched, 1);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "an item that does not match its control was encoded");
}

} // namespace

int main() {
    try {
        test_contiguous_decode();
        for (const QType format : {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16,
                                   QType::Q8_G32_FP16, QType::T2_G128_FP16}) {
            test_decode_matches_reference(format, QuantLayout::RowSplit, 3, 200);
        }
        test_decode_matches_reference(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale, 5, 48);
        test_decode_matches_reference(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, 256, 128,
                                      {.weight_scale_divisor = 2.5F, .input_scale_divisor = 1.0F});
        test_encoder_matches_reference();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
