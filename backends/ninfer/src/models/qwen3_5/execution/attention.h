#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "ninfer/ops/rope.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream,
                          InputBasis basis = InputBasis::Primal);

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, cudaStream_t stream);
void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, Tensor& key, cudaStream_t stream);

// Normalize q and k and rotate them. Where the fused Op covers the geometry this is one graph node
// instead of three; everywhere else it is the three calls it replaces, which are the same
// arithmetic bit for bit. It chooses a schedule, not a result.
void text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                       const AttentionConfig& attention, float rms_norm_eps,
                       const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                       const Tensor& query, const Tensor& key, Tensor& normalized_query,
                       Tensor& normalized_key, const ops::RopeYarn& yarn, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
