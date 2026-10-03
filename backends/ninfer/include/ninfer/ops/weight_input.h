#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sparse_moe.h"

#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
    // BF16 [K] signs of a Hadamard-rotated matrix: the Use multiplies the stored rows with
    // hadamard_transform(input, signs) instead of the input. Empty for an ordinary matrix. No Op
    // reads it; the caller rotates the activation (see the model's execution layer).
    Tensor hadamard_signs{};
    // INT32 [K] input gather of a GGUF matrix stored over permuted input columns (see
    // Weight::input_columns). Empty for every other matrix.
    Tensor input_columns{};
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    Tensor hadamard_signs{};
};

struct PairedProjectionWeights {
    Weight first, second;
    // Split parents reach their Op without a SingleProjectionWeight to carry the decision, so the
    // pair carries it. A16Only unless a registered route exists for these exact shapes.
    LinearPolicy policy = LinearPolicy::A16Only;
    Tensor hadamard_signs{};
};

// A fused input projection over GGUF matrices, whose logical parts may sit in different parents
// of different block types. Each part is rows of one parent landing at `row` of output `output`:
// for a GDN projection 0 = q/k/v and 1 = z; for an attention projection 0 = query, 1 = gate,
// 2 = key and 3 = value.
struct GgufProjectionPart {
    Weight weight;
    std::int32_t output = 0;
    std::int32_t row    = 0;
};

struct GgufProjectionWeights {
    std::vector<GgufProjectionPart> parts;
    LinearPolicy policy = LinearPolicy::A16Only;
    Tensor hadamard_signs{};
};

using ProjectionWeights =
    std::variant<SingleProjectionWeight, PairedProjectionWeights, GgufProjectionWeights>;

// Whether the rows of `inputs`, in order, are one contiguous region of one parent.
[[nodiscard]] bool joins(std::span<const WeightInput> inputs);

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a,
                                                                const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);
[[nodiscard]] SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down);

} // namespace ninfer::ops
