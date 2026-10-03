#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/rotation.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/token_sums.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

// The fused text form is registered for the two text head geometries with a one-dimensional
// position axis. The MRoPE path and any other geometry take the three calls it replaces.
//
// It is also bounded in width. One warp owns one head, so the fused kernel stops gaining once a
// width alone fills the machine, and past that the three separate kernels - each free to choose
// its own shape - are ahead: measured on an RTX 5090, the fused form wins by 22 to 52 % through
// 256 tokens and loses by up to 22 % at 1024. The bound sits a doubling below the crossover
// because the two geometries cross at different widths. The Op itself is valid at any width; this
// is a dispatch choice, and both branches are the same arithmetic bit for bit.
constexpr std::int32_t kFusedTextQkNormRopeMaximumTokens = 256;

bool fused_text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                             const AttentionConfig& attention, std::int32_t tokens) {
    return positions.ne[1] == 1 && tokens <= kFusedTextQkNormRopeMaximumTokens &&
           attention.head_dim == 256 && rope.rotary_dim == 64 &&
           ((attention.num_attention_heads == 16 && attention.num_key_value_heads == 2) ||
            (attention.num_attention_heads == 24 && attention.num_key_value_heads == 4));
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    const Tensor& signs = projection_signs(parameters.projection);
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        return rotated_workspace_bytes(
            signs, gguf->parts.front().weight.k, last,
            ops::attn_input_proj_workspace_capacity_bytes(*gguf, first, last));
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return rotated_workspace_bytes(
            signs, weight.k, last,
            ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                          single->policy, first, last));
    }
    const auto& pair = std::get<ops::PairedProjectionWeights>(parameters.projection);
    return rotated_workspace_bytes(signs, pair.first.k, last,
                                   ops::attn_input_proj_split_workspace_capacity_bytes(
                                       pair.first.qtype, pair.first.n, pair.second.qtype,
                                       pair.second.n, pair.first.k, pair.policy, first, last));
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream, InputBasis basis) {
    auto scope = workspace.scope();
    const Tensor x =
        rotated_input(hidden, projection_signs(parameters.projection), workspace, stream, basis);
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(x, *gguf, query, gate, key, value, workspace, stream);
    } else if (const auto* pair =
                   std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(x, pair->first, pair->second, query, gate, key, value, pair->policy,
                             workspace, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(x, single.weight, query, gate, key, value, single.policy, workspace,
                             stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, yarn, query, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::RopeYarn& yarn,
               Tensor& query, Tensor& key, cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, yarn, query, key, stream);
}

void text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                       const AttentionConfig& attention, float rms_norm_eps,
                       const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                       const Tensor& query, const Tensor& key, Tensor& normalized_query,
                       Tensor& normalized_key, const ops::RopeYarn& yarn, cudaStream_t stream,
                       const MemoryStatistics* statistics, std::int32_t layer) {
    require_rope_axes(positions, rope);
    // The fused Op rotates unscaled positions at the unscaled frequencies, so YaRN or position
    // interpolation keeps the three separate calls.
    if (statistics == nullptr && !yarn.active() && fused_text_qk_norm_rope(positions, rope, attention, query.ne[2])) {
        ops::rmsnorm_rope(positions, q_norm_weight, k_norm_weight, query, key, normalized_query,
                          normalized_key, stream);
        return;
    }
    ops::rmsnorm(query, q_norm_weight, rms_norm_eps, true, normalized_query, stream);
    ops::rmsnorm(key, k_norm_weight, rms_norm_eps, true, normalized_key, stream);
    if (statistics != nullptr) {
        const auto tokens = query.ne[2];
        const auto block = static_cast<std::int32_t>(statistics->block_tokens);
        const auto buckets = (tokens + 2 * block - 2) / block;
        // Decode columns are independent candidates, potentially belonging to
        // different requests. Copy their represented keys without interpreting
        // flattened columns as a consecutive logical token range.
        Tensor origin = block == 1 ? statistics->candidate_origin
                                   : statistics->prefill_origin;
        Tensor k_sum = statistics->key_sums.slice(2, layer, 1)
                           .slice(1, 0, buckets).view({dimension(attention.key_width()), buckets});
        Tensor q_sum = statistics->query_sums.slice(2, layer, 1)
                           .view({dimension(attention.query_width()), 1});
        ops::token_sums(normalized_key.view({dimension(attention.key_width()), tokens}),
                        origin, statistics->ranges.slice(0, 0, 2), block, k_sum, stream);
        if (block != 1) {
            ops::token_sums(normalized_query.view({dimension(attention.query_width()), tokens}),
                            origin, statistics->ranges.slice(0, 2, 2), 0, q_sum, stream);
        }
    }
    ops::rope(positions, dimension(rope.rotary_dim), rope.rope_theta, yarn, normalized_query,
              normalized_key, stream);
}

} // namespace ninfer::models::qwen3_5::execution
