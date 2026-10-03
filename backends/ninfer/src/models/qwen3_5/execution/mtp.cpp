#include "models/qwen3_5/execution/mtp.h"

#include "core/layout.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/mtp_pack.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::execution {

namespace {

// A bank whose parts share no contiguous parent -- GGUF parts of different block types, or mixed
// formats such as Q8 K/V beside BF16 Q/gate -- has no packed form; each row projects on its own.
bool unpacked(const MtpProjectionParameters& parameters) {
    return parameters.rows && parameters.packed.weight.qdata == nullptr;
}

bool gguf_rows(const MtpProjectionParameters& parameters) {
    return parameters.rows && is_gguf((*parameters.rows)[1].weight.qtype);
}

std::size_t row_bytes(const MtpProjectionParameters& parameters, std::size_t index,
                      std::int32_t first, std::int32_t last) {
    const auto& p = (*parameters.rows)[index];
    const auto& w = p.weight;
    return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
}

} // namespace

std::size_t mtp_projection_workspace_bytes(const MtpProjectionParameters& parameters,
                                           std::int32_t first, std::int32_t last) {
    if (unpacked(parameters)) {
        return std::max({row_bytes(parameters, 0, first, last), row_bytes(parameters, 1, first, last),
                         row_bytes(parameters, 2, first, last), row_bytes(parameters, 3, first, last)});
    }
    const auto& p = parameters.packed;
    const auto& w = p.weight;
    if (!parameters.rows) {
        return ops::attn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first,
                                                             last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {w.n, last});
    (void)layout.alloc_bytes(
        ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_kv_workspace_bytes(const MtpProjectionParameters& parameters,
                                   const AttentionConfig& config, std::int32_t first,
                                   std::int32_t last) {
    if (gguf_rows(parameters)) {
        return std::max(row_bytes(parameters, 1, first, last), row_bytes(parameters, 3, first, last));
    }
    if (parameters.rows) {
        return ops::linear_pair_workspace_capacity_bytes((*parameters.rows)[1].weight,
                                                         (*parameters.rows)[3].weight, first, last);
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.query_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

std::size_t mtp_query_gate_workspace_bytes(const MtpProjectionParameters& parameters,
                                           const AttentionConfig& config, std::int32_t first,
                                           std::int32_t last) {
    if (parameters.rows) {
        const auto bytes = [&](std::size_t index) {
            const auto& p = (*parameters.rows)[index];
            const auto& w = p.weight;
            return ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, p.policy, first, last);
        };
        return std::max(bytes(0), bytes(2));
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc(DType::BF16, {dimension(config.key_width()), last});
    (void)layout.alloc_bytes(mtp_projection_workspace_bytes(parameters, first, last));
    return layout.peak_bytes(1);
}

void mtp_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                    const AttentionConfig& config, Tensor& query, Tensor& gate, Tensor& key,
                    Tensor& value, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& p = parameters.packed;
    if (!parameters.rows) {
        ops::attn_input_proj(hidden, p.weight, query, gate, key, value, p.policy, workspace,
                             stream);
        return;
    }
    if (unpacked(parameters)) {
        Tensor* outputs[] = {&query, &key, &gate, &value};
        for (std::size_t i = 0; i < 4; ++i) {
            auto scope    = workspace.scope();
            const auto& r = (*parameters.rows)[i];
            ops::linear(hidden, r.weight, *outputs[i], r.policy, workspace, stream);
        }
        return;
    }
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    Tensor packed      = workspace.alloc(DType::BF16, {p.weight.n, columns});
    ops::linear(hidden, p.weight, packed, p.policy, workspace, stream);
    Tensor q =
        query.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor k =
        key.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    Tensor g =
        gate.view({dimension(config.head_dim), dimension(config.num_attention_heads), columns});
    Tensor v =
        value.view({dimension(config.head_dim), dimension(config.num_key_value_heads), columns});
    ops::mtp_split_attn_in(packed, q, k, g, v, stream);
}

void mtp_kv_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                       const AttentionConfig& config, Tensor& key, Tensor& value,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    if (gguf_rows(parameters)) {
        const auto& k = (*parameters.rows)[1];
        const auto& v = (*parameters.rows)[3];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, k.weight, key, k.policy, workspace, stream);
        }
        ops::linear(hidden, v.weight, value, v.policy, workspace, stream);
        return;
    }
    if (parameters.rows) {
        ops::linear_pair(hidden, (*parameters.rows)[1].weight, (*parameters.rows)[3].weight, key,
                         value, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor query = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    Tensor gate  = workspace.alloc(DType::BF16, {dimension(config.query_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

void mtp_query_gate_projection(const Tensor& hidden, const MtpProjectionParameters& parameters,
                               const AttentionConfig& config, Tensor& query, Tensor& gate,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    if (parameters.rows) {
        const auto& q = (*parameters.rows)[0];
        const auto& g = (*parameters.rows)[2];
        {
            auto scope = workspace.scope();
            ops::linear(hidden, q.weight, query, q.policy, workspace, stream);
        }
        ops::linear(hidden, g.weight, gate, g.policy, workspace, stream);
        return;
    }
    auto scope   = workspace.scope();
    Tensor key   = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    Tensor value = workspace.alloc(DType::BF16, {dimension(config.key_width()), hidden.ne[1]});
    mtp_projection(hidden, parameters, config, query, gate, key, value, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
