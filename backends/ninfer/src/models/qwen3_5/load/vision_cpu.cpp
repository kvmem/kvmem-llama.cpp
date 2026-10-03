#include "models/qwen3_5/load/vision_cpu.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::int32_t kRotaryHeadDim = 72;

float bf16_to_f32(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

float f16_to_f32(std::uint16_t word) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(word) & 0x8000U) << 16U;
    std::uint32_t exponent   = (static_cast<std::uint32_t>(word) >> 10U) & 0x1fU;
    std::uint32_t mantissa   = static_cast<std::uint32_t>(word) & 0x03ffU;
    if (exponent == 0) {
        if (mantissa == 0) { return std::bit_cast<float>(sign); }
        int shift = -14;
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1U;
            --shift;
        }
        mantissa &= 0x03ffU;
        return std::bit_cast<float>(sign | (static_cast<std::uint32_t>(shift + 127) << 23U) |
                                    (mantissa << 13U));
    }
    if (exponent == 31) { return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U)); }
    exponent = exponent - 15U + 127U;
    return std::bit_cast<float>(sign | (exponent << 23U) | (mantissa << 13U));
}

float e4m3fn_to_f32(std::uint8_t word) {
    const std::uint32_t exponent = (word >> 3U) & 0x0fU;
    const std::uint32_t mantissa = word & 0x07U;
    float magnitude              = 0.0F;
    if (exponent == 0) {
        magnitude = static_cast<float>(mantissa) * std::ldexp(1.0F, -9);
    } else if (exponent == 0x0fU && mantissa == 0x07U) {
        magnitude = std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = (1.0F + static_cast<float>(mantissa) / 8.0F) *
                    std::ldexp(1.0F, static_cast<int>(exponent) - 7);
    }
    return (word & 0x80U) != 0 ? -magnitude : magnitude;
}

float e2m1_to_f32(std::uint8_t word) {
    constexpr float kMagnitudes[]{0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = kMagnitudes[word & 0x07U];
    return (word & 0x08U) != 0 ? -magnitude : magnitude;
}

std::uint16_t load_u16(const std::byte* bytes) {
    return static_cast<std::uint16_t>(std::to_integer<std::uint16_t>(bytes[0]) |
                                      (std::to_integer<std::uint16_t>(bytes[1]) << 8U));
}

int code_bits(QType format) {
    switch (format) {
    case QType::T2_G128_FP16:
        return 2;
    case QType::Q4_G64_FP16:
        return 4;
    case QType::Q5_G64_FP16:
        return 5;
    case QType::Q6_G64_FP16:
        return 6;
    case QType::Q8_G32_FP16:
        return 8;
    default:
        return 0;
    }
}

// The signed code at `lane` of one row-split group: low nibbles in the code plane, the extra bits
// of Q5/Q6 in the high plane, whole bytes for Q8 and two-bit fields for T2.
int row_split_code(const std::byte* codes, const std::byte* high, int bits, int lane) {
    const auto byte = [](const std::byte* plane, int index) {
        return std::to_integer<std::uint32_t>(plane[index]);
    };
    if (bits == 8) { return static_cast<std::int8_t>(byte(codes, lane)); }
    if (bits == 2) {
        const std::uint32_t value = (byte(codes, lane >> 2) >> ((lane & 3) * 2)) & 0x03U;
        return (value & 0x2U) != 0 ? static_cast<int>(value) - 4 : static_cast<int>(value);
    }
    const std::uint32_t low_byte = byte(codes, lane >> 1);
    const std::uint32_t low      = (lane & 1) != 0 ? (low_byte >> 4U) : (low_byte & 0x0fU);
    std::uint32_t extra          = 0;
    if (bits == 5) {
        extra = (byte(high, lane >> 3) >> (lane & 7)) & 0x01U;
    } else if (bits == 6) {
        const int bit = lane * 2;
        extra         = (byte(high, bit >> 3) >> (bit & 7)) & 0x03U;
    }
    const std::uint32_t value = low | (extra << 4U);
    const std::uint32_t sign  = 1U << (bits - 1);
    return (value & sign) != 0 ? static_cast<int>(value) - (1 << bits) : static_cast<int>(value);
}

void require_bytes(std::span<const std::byte> bytes, std::uint64_t needed) {
    if (bytes.size() < needed) {
        throw std::invalid_argument("CPU Vision: encoded tensor is shorter than its geometry");
    }
}

void check_size(const std::vector<float>& tensor, std::size_t expected, const char* name) {
    if (tensor.size() != expected) {
        throw std::invalid_argument(std::string("CPU Vision weights: ") + name +
                                    " has the wrong element count");
    }
}

} // namespace

std::size_t CpuVisionWeights::bytes() const noexcept {
    std::size_t total = 0;
    const auto add    = [&](const std::vector<float>& tensor) { total += tensor.size() * 4; };
    for (const auto* tensor :
         {&patch_embedding, &patch_embedding_bias, &position_embedding, &merger_norm_weight,
          &merger_norm_bias, &merger_fc1, &merger_fc1_bias, &merger_fc2, &merger_fc2_bias}) {
        add(*tensor);
    }
    for (const Layer& layer : layers) {
        for (const auto* tensor :
             {&layer.norm1_weight, &layer.norm1_bias, &layer.qkv, &layer.qkv_bias, &layer.output,
              &layer.output_bias, &layer.norm2_weight, &layer.norm2_bias, &layer.fc1,
              &layer.fc1_bias, &layer.fc2, &layer.fc2_bias}) {
            add(*tensor);
        }
    }
    return total;
}

void CpuVisionWeights::validate() const {
    if (hidden <= 0 || heads <= 0 || hidden % heads != 0 || head_dim() != kRotaryHeadDim ||
        intermediate <= 0 || patch_width <= 0 || merge_unit <= 0 || output_hidden <= 0 ||
        position_rows <= 0 || layers.empty()) {
        throw std::invalid_argument(
            "CPU Vision needs the two-axis rotary tower geometry (72-wide heads)");
    }
    const auto h = static_cast<std::size_t>(hidden);
    const auto i = static_cast<std::size_t>(intermediate);
    const auto m = static_cast<std::size_t>(merger_width());
    const auto o = static_cast<std::size_t>(output_hidden);
    check_size(patch_embedding, h * static_cast<std::size_t>(patch_width), "patch_embedding");
    check_size(patch_embedding_bias, h, "patch_embedding_bias");
    check_size(position_embedding, static_cast<std::size_t>(position_rows) * h,
               "position_embedding");
    for (const Layer& layer : layers) {
        check_size(layer.norm1_weight, h, "norm1_weight");
        check_size(layer.norm1_bias, h, "norm1_bias");
        check_size(layer.qkv, 3 * h * h, "attention projections");
        check_size(layer.qkv_bias, 3 * h, "attention biases");
        check_size(layer.output, h * h, "attention output");
        check_size(layer.output_bias, h, "attention output bias");
        check_size(layer.norm2_weight, h, "norm2_weight");
        check_size(layer.norm2_bias, h, "norm2_bias");
        check_size(layer.fc1, i * h, "fc1");
        check_size(layer.fc1_bias, i, "fc1_bias");
        check_size(layer.fc2, h * i, "fc2");
        check_size(layer.fc2_bias, h, "fc2_bias");
    }
    check_size(merger_norm_weight, h, "merger norm weight");
    check_size(merger_norm_bias, h, "merger norm bias");
    check_size(merger_fc1, m * m, "merger fc1");
    check_size(merger_fc1_bias, m, "merger fc1 bias");
    check_size(merger_fc2, o * m, "merger fc2");
    check_size(merger_fc2_bias, o, "merger fc2 bias");
}

std::vector<float> decode_tensor(const WeightGeometry& geometry, std::span<const std::byte> bytes) {
    require_bytes(bytes, geometry.bytes);
    std::vector<float> out(geometry.elements);
    if (geometry.layout == QuantLayout::Contiguous) {
        if (geometry.format == QType::BF16) {
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = bf16_to_f32(load_u16(bytes.data() + 2 * i));
            }
            return out;
        }
        if (geometry.format == QType::FP32) {
            std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
            return out;
        }
        throw std::invalid_argument("CPU Vision cannot decode this contiguous tensor format");
    }
    if (geometry.shape.size() != 2) {
        throw std::invalid_argument("CPU Vision: a quantized tensor must be a matrix");
    }
    const auto n = static_cast<std::size_t>(geometry.shape[0]);
    const auto k = static_cast<std::size_t>(geometry.shape[1]);
    if (geometry.layout == QuantLayout::RowSplit && code_bits(geometry.format) != 0) {
        const int bits              = code_bits(geometry.format);
        const auto group            = static_cast<std::size_t>(geometry.group_size);
        const std::size_t groups    = geometry.padded_columns / group;
        const std::size_t extra_row = geometry.high_bytes_per_row;
        const std::size_t extra     = groups == 0 ? 0 : extra_row / groups;
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t g = 0; g < groups; ++g) {
                const std::byte* codes = bytes.data() + row * geometry.code_bytes_per_row + g * 32;
                const std::byte* high =
                    extra == 0 ? nullptr
                               : bytes.data() + geometry.high_offset + row * extra_row + g * extra;
                const float scale = f16_to_f32(
                    load_u16(bytes.data() + geometry.scale_offset + (row * groups + g) * 2));
                for (std::size_t lane = 0; lane < group; ++lane) {
                    const std::size_t column = g * group + lane;
                    if (column >= k) { break; }
                    out[row * k + column] = static_cast<float>(row_split_code(
                                                codes, high, bits, static_cast<int>(lane))) *
                                            scale;
                }
            }
        }
        return out;
    }
    if (geometry.layout == QuantLayout::RowScale && geometry.format == QType::FP8_E4M3FN_ROW_BF16) {
        for (std::size_t row = 0; row < n; ++row) {
            const float scale =
                bf16_to_f32(load_u16(bytes.data() + geometry.scale_offset + row * 2));
            for (std::size_t column = 0; column < k; ++column) {
                out[row * k + column] =
                    e4m3fn_to_f32(std::to_integer<std::uint8_t>(bytes[row * k + column])) * scale;
            }
        }
        return out;
    }
    if (geometry.layout == QuantLayout::BlockScaleK16M128x4 && geometry.format == QType::NVFP4) {
        const std::size_t divisors = geometry.divisor_count;
        std::vector<float> divisor(divisors);
        for (std::size_t index = 0; index < divisors; ++index) {
            std::uint32_t word = 0;
            std::memcpy(&word, bytes.data() + geometry.divisor_offset + index * 4, 4);
            divisor[index] = std::bit_cast<float>(word);
            if (!std::isfinite(divisor[index]) || divisor[index] <= 0.0F) {
                throw std::invalid_argument("CPU Vision: invalid NVFP4 weight divisor");
            }
        }
        const std::size_t rows_per_divisor = n / divisors;
        for (std::size_t row = 0; row < n; ++row) {
            const float inverse = 1.0F / divisor[row / rows_per_divisor];
            for (std::size_t group = 0; group < k / 16; ++group) {
                const float scale = e4m3fn_to_f32(std::to_integer<std::uint8_t>(
                                        bytes[weight_scale_offset(geometry, row, group)])) *
                                    inverse;
                for (std::size_t column = group * 16; column < group * 16 + 16; ++column) {
                    const std::uint8_t packed =
                        std::to_integer<std::uint8_t>(bytes[row * (k / 2) + column / 2]);
                    const std::uint8_t code =
                        (column & 1U) == 0 ? (packed & 0x0fU) : (packed >> 4U);
                    out[row * k + column] = e2m1_to_f32(code) * scale;
                }
            }
        }
        return out;
    }
    throw std::invalid_argument("CPU Vision cannot decode this quantized tensor representation");
}

namespace loading {

std::shared_ptr<const CpuVisionWeights>
load_cpu_vision(artifact::Binder& binder, const VisionConfig& config, const TextConfig& target) {
    const artifact::Reader& reader = binder.reader();
    std::unordered_map<std::size_t, std::vector<float>> decoded;
    const auto tensor = [&](const std::string& name, artifact::Shape shape) {
        const artifact::ParameterReference reference =
            binder.parameter(name, std::move(shape), artifact::Residency::Values);
        std::vector<float> out;
        out.reserve(static_cast<std::size_t>(reference.binding.elements));
        for (const artifact::Part& part : reference.binding.parts) {
            auto found = decoded.find(part.object.index);
            if (found == decoded.end()) {
                found =
                    decoded
                        .emplace(part.object.index, decode_tensor(reader.geometry(part.object),
                                                                  reader.read_object(part.object)))
                        .first;
            }
            out.insert(out.end(), found->second.begin() + static_cast<std::ptrdiff_t>(part.begin),
                       found->second.begin() + static_cast<std::ptrdiff_t>(part.end));
        }
        return out;
    };
    const auto projection = [&](const std::string& name, artifact::Shape shape,
                                const std::string& input) {
        for (const auto& [role, auxiliary] : binder.use(name, input).auxiliaries) {
            (void)auxiliary;
            if (role == "hadamard_signs" || role == "input_columns") {
                throw artifact::ArtifactError(name + "@" + input + ": the CPU Vision residency " +
                                              "cannot apply " + role + "; use a device residency");
            }
        }
        return tensor(name, std::move(shape));
    };
    const auto stack = [](std::vector<float> first, const std::vector<float>& second,
                          const std::vector<float>& third) {
        first.insert(first.end(), second.begin(), second.end());
        first.insert(first.end(), third.begin(), third.end());
        return first;
    };

    const auto h            = config.hidden_size;
    const auto intermediate = config.intermediate_size;
    const auto merger       = config.merger_width();
    auto weights            = std::make_shared<CpuVisionWeights>();
    weights->hidden         = static_cast<std::int32_t>(h);
    weights->heads          = static_cast<std::int32_t>(config.num_heads);
    weights->intermediate   = static_cast<std::int32_t>(intermediate);
    weights->patch_width    = static_cast<std::int32_t>(config.patch_width());
    weights->merge_unit     = static_cast<std::int32_t>(merger / h);
    weights->output_hidden  = static_cast<std::int32_t>(target.hidden_size);
    weights->position_rows  = static_cast<std::int32_t>(config.num_position_embeddings);

    weights->patch_embedding =
        projection("vision/patch_embedding", {h, config.patch_width()}, "vision/patch_input");
    weights->patch_embedding_bias = tensor("vision/patch_embedding_bias", {h});
    weights->position_embedding =
        tensor("vision/position_embedding", {config.num_position_embeddings, h});
    weights->layers.resize(config.depth);
    for (std::uint32_t index = 0; index < config.depth; ++index) {
        const std::string p            = "vision/layers/" + std::to_string(index) + "/";
        CpuVisionWeights::Layer& layer = weights->layers[index];
        layer.norm1_weight             = tensor(p + "norm1_weight", {h});
        layer.norm1_bias               = tensor(p + "norm1_bias", {h});
        layer.qkv = stack(projection(p + "attention/query", {h, h}, p + "attention_input"),
                          projection(p + "attention/key", {h, h}, p + "attention_input"),
                          projection(p + "attention/value", {h, h}, p + "attention_input"));
        layer.qkv_bias =
            stack(tensor(p + "attention/query_bias", {h}), tensor(p + "attention/key_bias", {h}),
                  tensor(p + "attention/value_bias", {h}));
        layer.output       = projection(p + "attention/output", {h, h}, p + "attention_output");
        layer.output_bias  = tensor(p + "attention/output_bias", {h});
        layer.norm2_weight = tensor(p + "norm2_weight", {h});
        layer.norm2_bias   = tensor(p + "norm2_bias", {h});
        layer.fc1          = projection(p + "mlp/fc1", {intermediate, h}, p + "mlp_input");
        layer.fc1_bias     = tensor(p + "mlp/fc1_bias", {intermediate});
        layer.fc2          = projection(p + "mlp/fc2", {h, intermediate}, p + "mlp_activation");
        layer.fc2_bias     = tensor(p + "mlp/fc2_bias", {h});
        // Each object is read and decoded once; a later layer never shares this one's objects.
        decoded.clear();
    }
    weights->merger_norm_weight = tensor("vision/merger/norm_weight", {h});
    weights->merger_norm_bias   = tensor("vision/merger/norm_bias", {h});
    weights->merger_fc1 = projection("vision/merger/fc1", {merger, merger}, "vision/merger/input");
    weights->merger_fc1_bias = tensor("vision/merger/fc1_bias", {merger});
    weights->merger_fc2 =
        projection("vision/merger/fc2", {target.hidden_size, merger}, "vision/merger/activation");
    weights->merger_fc2_bias = tensor("vision/merger/fc2_bias", {target.hidden_size});
    weights->validate();
    return weights;
}

} // namespace loading
} // namespace ninfer::models::qwen3_5
