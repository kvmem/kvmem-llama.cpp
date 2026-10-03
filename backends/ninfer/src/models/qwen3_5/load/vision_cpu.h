#pragma once

// Host FP32 Vision weights for `--vision-residency cpu`: the tower's parameters are read from the
// artifact and decoded once at load, never placed on the device. The encoder that consumes them
// lives in execution/vision_cpu.h.

#include "artifact/binder.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "models/qwen3_5/config.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Matrices are row-major [out, in], so row n holds output feature n. The query, key and value
// projections are stacked into one [3*hidden, hidden] matrix, as the device fuses them.
struct CpuVisionWeights {
    std::int32_t hidden        = 0;
    std::int32_t heads         = 0;
    std::int32_t intermediate  = 0;
    std::int32_t patch_width   = 0;
    std::int32_t merge_unit    = 0; // spatial_merge_size squared
    std::int32_t output_hidden = 0; // the text hidden size the merger projects to
    std::int32_t position_rows = 0;

    struct Layer {
        std::vector<float> norm1_weight, norm1_bias;
        std::vector<float> qkv, qkv_bias;
        std::vector<float> output, output_bias;
        std::vector<float> norm2_weight, norm2_bias;
        std::vector<float> fc1, fc1_bias;
        std::vector<float> fc2, fc2_bias;
    };

    std::vector<float> patch_embedding, patch_embedding_bias;
    std::vector<float> position_embedding; // [position_rows, hidden]
    std::vector<Layer> layers;
    std::vector<float> merger_norm_weight, merger_norm_bias;
    std::vector<float> merger_fc1, merger_fc1_bias;
    std::vector<float> merger_fc2, merger_fc2_bias;

    [[nodiscard]] std::int32_t head_dim() const noexcept { return heads != 0 ? hidden / heads : 0; }

    [[nodiscard]] std::int32_t merger_width() const noexcept { return hidden * merge_unit; }

    [[nodiscard]] std::size_t bytes() const noexcept;
    // Throws unless every tensor has the size its dimensions imply and the heads have the 72-wide
    // two-axis rotary geometry the encoder implements.
    void validate() const;
};

// Every logical element of one stored tensor as FP32, row-major: contiguous BF16 and FP32,
// row-split Q4/Q5/Q6/Q8/T2 grouped codes, row-scaled FP8 and block-scaled NVFP4. Other
// representations throw.
[[nodiscard]] std::vector<float> decode_tensor(const WeightGeometry& geometry,
                                               std::span<const std::byte> bytes);

namespace loading {

// Reads and decodes the tower without any device or Host placement demand. A projection whose
// Use carries a Hadamard rotation or an input gather is refused: the CPU encoder feeds every
// projection its plain input.
[[nodiscard]] std::shared_ptr<const CpuVisionWeights>
load_cpu_vision(artifact::Binder& binder, const VisionConfig& config, const TextConfig& target);

} // namespace loading
} // namespace ninfer::models::qwen3_5
