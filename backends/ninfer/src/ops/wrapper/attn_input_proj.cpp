#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/t2/t2_a8.h"
#include "ops/linear/t2/t2_weight_view.h"

#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <cmath>
#include <cstddef>
#include <array>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool no_high   = weight.qhigh == nullptr && weight.high_plane_bytes == 0;
    const bool q4_planes = qtype != QType::Q4_G64_FP16 || no_high;
    const bool t2_planes = qtype != QType::T2_G128_FP16 || no_high;
    const bool q5_planes =
        qtype != QType::Q5_G64_FP16 || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    const std::int32_t group = qtype == QType::T2_G128_FP16 ? 128 : 64;
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 ||
        weight.group_size != static_cast<std::uint32_t>(group) || weight.group != group ||
        !t2_planes || weight.ndim != 2 || weight.n != rows || weight.k != 5120 ||
        weight.shape[0] != rows || weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5_G64_FP16 && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_q8_rowsplit(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                         const char* label) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || !aligned_to(weight.qdata, 16) ||
        !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void require_bf16_contiguous(const Weight& weight, std::int32_t rows, std::int32_t hidden,
                             const char* label) {
    const std::uint64_t payload_bytes = static_cast<std::uint64_t>(rows) *
                                        static_cast<std::uint64_t>(hidden) * sizeof(std::uint16_t);
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.payload_bytes < payload_bytes || weight.high_plane_bytes != 0 || weight.ndim != 2 ||
        weight.n != rows || weight.k != hidden || weight.shape[0] != rows ||
        weight.shape[1] != hidden || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != hidden || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.group_size != 0 || weight.group != 0 || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("attn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
    // The split Q4/Q5 pair has an integer-activation route; the single-parent forms decline it
    // below, where no such route is registered for their qtypes.
    case LinearPolicy::AllowA8Int:
    case LinearPolicy::AllowA8IntDecode:
    // The cuBLAS prefill route is registered only for linear_swiglu and linear_add. Everywhere else
    // this policy means exactly what AllowA8Int means, and is accepted rather than rejected so that
    // one engine-wide setting does not have to be threaded per Op.
    case LinearPolicy::AllowPrefillCublas:
        return;
    }
    throw std::invalid_argument("attn_input_proj: invalid compute policy");
}

// No single-parent qtype registers an integer-activation route, and their resolvers reject a policy
// they do not know, so the integer policies read as A16Only there.
LinearPolicy without_integer(LinearPolicy policy) {
    return allows_a8_int(policy) ? LinearPolicy::A16Only : policy;
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                            cudaStream_t stream) {
    validate_policy(policy);
    policy = without_integer(policy);
    if (weight.qtype == QType::BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        require_bf16_contiguous(weight, kRows, kHidden, "query/key/gate/value weight");
        detail::bf16_attn_input_dispatch(x, weight, q, gate, k, v, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_nvfp4_weight(weight, "nvfp4 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 attn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden = 5120;
        constexpr std::int32_t kQRows  = 6144;
        constexpr std::int32_t kKvRows = 1024;
        constexpr std::int32_t kRows   = 14336;
        const std::int32_t cols        = x.ne[1];
        if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(q, kQRows, cols, "q");
        require_matrix(gate, kQRows, cols, "gate");
        require_matrix(k, kKvRows, cols, "k");
        require_matrix(v, kKvRows, cols, "v");
        detail::validate_fp8_weight(weight, "fp8 attn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 attn_input_proj: unsupported weight shape");
        }
        detail::fp8_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden = 2048;
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 512;
    constexpr std::int32_t kRows   = 9216;
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(weight, kRows, kHidden, "query/key/gate/value weight");
    detail::q8_attn_input_dispatch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }

    switch (parent_qtype) {
    case QType::BF16:
        if (parent_rows != 14336 || input_rows != 5120) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported BF16 profile");
        }
        return 0;
    case QType::NVFP4:
        if (parent_rows != detail::Nvfp4N14336K5120::kOutputRows ||
            input_rows != detail::Nvfp4N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        if (parent_rows != detail::Fp8N14336K5120::kOutputRows ||
            input_rows != detail::Fp8N14336K5120::kInputRows) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_attn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    case QType::Q8_G32_FP16:
        if (parent_rows != 9216 || input_rows != 2048) {
            throw std::invalid_argument("attn_input_proj workspace: unsupported Q8 profile");
        }
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, min_tokens});
        (void)detail::q8_attn_input_resolve_plan(
            {input_rows, 4096, 512, parent_rows, input_rows, max_tokens});
        return 0;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::T2_G128_FP16:
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("attn_input_proj workspace: unsupported parent qtype");
}

namespace {

// Shape checks both split overloads owe their callers.
void require_split_profile(const Tensor& x, const Weight& query_key_weight,
                           const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                           Tensor& v) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4_G64_FP16, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5_G64_FP16, kQRows + kKvRows, "gate/value weight");
}

// Ternary pair: both parents are T2 (q/k rows [0,6144)/[6144,7168), gate/v likewise). The integer
// route quantises x once and splits each parent's rows into its two destinations; otherwise each
// part is projected through the T2 linear routes from a row view of its parent.
bool t2_pair_project(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    const bool qk = query_key_weight.qtype == QType::T2_G128_FP16;
    if (qk != (gate_value_weight.qtype == QType::T2_G128_FP16)) {
        throw std::invalid_argument("attn_input_proj: the two parents must share the T2 format");
    }
    if (!qk) { return false; }
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, 5120, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::T2_G128_FP16, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::T2_G128_FP16, kQRows + kKvRows, "gate/value weight");
    if (workspace != nullptr && detail::t2_a8_admits(policy) &&
        detail::t2_a8_supported(query_key_weight, cols) &&
        detail::t2_a8_supported(gate_value_weight, cols)) {
        // One quantisation of x and one pass over each parent, split into its two destinations.
        auto scope             = workspace->scope();
        const auto activations = detail::t2_a8_quantize(x, *workspace, stream);
        detail::t2_a8_project_split_pair(activations, {query_key_weight, q, 0, kQRows, k, 0},
                                         {gate_value_weight, gate, 0, kQRows, v, 0}, stream);
        return true;
    }
    const auto project = [&](const Weight& part, Tensor& out) {
        if (workspace != nullptr) {
            linear(x, part, out, policy, *workspace, stream);
        } else {
            linear(x, part, out, stream);
        }
    };
    project(detail::t2_row_view(query_key_weight, 0, kQRows), q);
    project(detail::t2_row_view(query_key_weight, kQRows, kKvRows), k);
    project(detail::t2_row_view(gate_value_weight, 0, kQRows), gate);
    project(detail::t2_row_view(gate_value_weight, kQRows, kKvRows), v);
    return true;
}

} // namespace

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);
    if (t2_pair_project(x, query_key_weight, gate_value_weight, q, gate, k, v, policy, &workspace,
                        stream)) {
        return;
    }
    require_split_profile(x, query_key_weight, gate_value_weight, q, gate, k, v);
    // Both parents split into a query/gate half and a key/value half, and both read the same
    // activations, so the cuBLAS route materialises each parent once and shares one quantisation
    // across all four destinations.
    if (allows_cublas_prefill(policy) && x.ne[1] >= kCublasPrefillMinTokens) {
        const std::int32_t q_rows  = q.ne[0];
        const std::int32_t kv_rows = k.ne[0];
        const detail::CublasProjectionDestination qk_dests[] = {
            {q.data, 0, q_rows, q_rows, 0}, {k.data, q_rows, kv_rows, kv_rows, 0}};
        const detail::CublasProjectionDestination gv_dests[] = {
            {gate.data, 0, q_rows, q_rows, 0}, {v.data, q_rows, kv_rows, kv_rows, 0}};
        const detail::CublasProjection parents[] = {
            {&query_key_weight, qk_dests, 2}, {&gate_value_weight, gv_dests, 2}};
        if (detail::w4_cublas_projection_supported(parents, 2, x.ne[1])) {
            detail::w4_cublas_projection_launch(x, parents, 2, workspace, stream);
            return;
        }
    }
    if (allows_a8_int(policy) &&
        detail::q4_q5_attn_input_a8_supported(query_key_weight, gate_value_weight, x.ne[1])) {
        detail::q4_q5_attn_input_a8_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                           workspace, stream);
        return;
    }
    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

std::size_t attn_input_proj_split_workspace_capacity_bytes(
    QType query_key_qtype, std::int32_t query_key_rows, QType gate_value_qtype,
    std::int32_t gate_value_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj workspace: invalid token interval");
    }
    if (query_key_qtype == QType::T2_G128_FP16 || gate_value_qtype == QType::T2_G128_FP16) {
        if (query_key_qtype != gate_value_qtype || query_key_rows != 7168 ||
            gate_value_rows != 7168 || input_rows != 5120) {
            throw std::invalid_argument("attn_input_proj workspace: unregistered T2 pair");
        }
        std::size_t bytes = 0;
        for (const std::int32_t rows : {6144, 1024}) {
            bytes = std::max(bytes,
                             linear_workspace_capacity_bytes(QType::T2_G128_FP16, rows, input_rows,
                                                             policy, min_tokens, max_tokens));
        }
        return bytes;
    }
    const bool registered = query_key_qtype == QType::Q4_G64_FP16 && query_key_rows == 7168 &&
                            gate_value_qtype == QType::Q5_G64_FP16 && gate_value_rows == 7168 &&
                            input_rows == 5120;
    if (!allows_a8_int(policy) || !registered) { return 0; }
    const std::size_t a8 = detail::q4_q5_attn_input_a8_workspace_capacity_bytes(min_tokens,
                                                                                max_tokens);
    if (allows_cublas_prefill(policy) && max_tokens >= kCublasPrefillMinTokens) {
        return std::max(a8, detail::w4_cublas_projection_workspace_capacity_bytes(
                                std::max(query_key_rows, gate_value_rows), input_rows,
                                std::max(min_tokens, kCublasPrefillMinTokens), max_tokens));
    }
    return a8;
}

void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    if (t2_pair_project(x, query_key_weight, gate_value_weight, q, gate, k, v,
                        LinearPolicy::A16Only, nullptr, stream)) {
        return;
    }
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(gate, kQRows, cols, "gate");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_rowsplit(query_key_weight, QType::Q4_G64_FP16, kQRows + kKvRows, "query/key weight");
    require_rowsplit(gate_value_weight, QType::Q5_G64_FP16, kQRows + kKvRows, "gate/value weight");

    detail::q4_q5_attn_input_dispatch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                      stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, policy, &workspace,
                           stream);
}

bool attn_input_proj_fused_rmsnorm_nvfp4_eligible(const Weight& weight, LinearPolicy policy,
                                                  std::int32_t tokens) {
    // T==1 fuses the GEMV consumer, T>=4 the W4A4 consumers. T==2,3 stay unfused: a fused
    // norm+SIMT variant measured a regression (see plan.h), so the gap is deliberate.
    const bool supported_tokens = tokens == 1 || tokens >= 4;
    return policy == LinearPolicy::AllowA4 && supported_tokens && weight.qtype == QType::NVFP4 &&
           weight.layout == QuantLayout::BlockScaleK16M128x4 && weight.n == 14336 &&
           weight.k == 5120;
}

void attn_input_proj_fused_rmsnorm_nvfp4(const Tensor& residual, const Tensor& norm_weight,
                                         float eps, const Weight& projection_weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                                         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (!attn_input_proj_fused_rmsnorm_nvfp4_eligible(projection_weight, policy, tokens)) {
        throw std::invalid_argument("fused rmsnorm attn_input_proj: unsupported route");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument(
            "fused rmsnorm attn_input_proj: eps must be positive and finite");
    }
    require_matrix(residual, 5120, tokens, "residual");
    if (norm_weight.dtype != DType::BF16 || norm_weight.ne[0] != 5120 || norm_weight.ne[1] != 1 ||
        norm_weight.ne[2] != 1 || norm_weight.ne[3] != 1 || !norm_weight.is_contiguous() ||
        !aligned_to(norm_weight.data, 16)) {
        throw std::invalid_argument("fused rmsnorm attn_input_proj: invalid norm weight");
    }
    require_matrix(q, 6144, tokens, "q");
    require_matrix(gate, 6144, tokens, "gate");
    require_matrix(k, 1024, tokens, "k");
    require_matrix(v, 1024, tokens, "v");
    detail::validate_nvfp4_weight(projection_weight, "fused rmsnorm attn_input_proj");
    detail::nvfp4_attn_input_fused_rmsnorm_launch(residual, norm_weight, eps, projection_weight, q,
                                                  gate, k, v, workspace, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_gate_value_weight, q, gate, k, v, LinearPolicy::A16Only,
                           nullptr, stream);
}

void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kQRows  = 4096;
    constexpr std::int32_t kKvRows = 1024;
    constexpr std::int32_t kRows   = 6144;
    const std::int32_t hidden      = x.ne[0];
    const std::int32_t cols        = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    if (hidden != 2048 && hidden != 5120) {
        throw std::invalid_argument("attn_input_proj: unsupported Q8 Q/K/V profile");
    }
    require_matrix(x, hidden, cols, "x");
    require_matrix(q, kQRows, cols, "q");
    require_matrix(k, kKvRows, cols, "k");
    require_matrix(v, kKvRows, cols, "v");
    require_q8_rowsplit(query_key_value_weight, kRows, hidden, "query/key/value weight");

    detail::q8_attn_input_dispatch(x, query_key_value_weight, q, k, v, stream);
}

namespace {

void require_gguf_attention(const GgufProjectionWeights& weights, std::int32_t hidden,
                            std::int32_t query_rows, std::int32_t kv_rows) {
    std::array<std::int64_t, 4> covered{};
    for (const auto& part : weights.parts) {
        detail::require_gguf(part.weight, "attn_input_proj GGUF part");
        const std::int32_t limit = part.output < 2 ? query_rows : kv_rows;
        if (part.weight.k != hidden || part.output < 0 || part.output > 3 || part.row < 0 ||
            part.row + part.weight.n > limit) {
            throw std::invalid_argument("attn_input_proj: GGUF part outside the attention profile");
        }
        covered[part.output] += part.weight.n;
    }
    if (covered[0] != query_rows || covered[1] != query_rows || covered[2] != kv_rows ||
        covered[3] != kv_rows) {
        throw std::invalid_argument("attn_input_proj: GGUF parts do not cover q/gate/k/v");
    }
}

} // namespace

std::size_t attn_input_proj_workspace_capacity_bytes(const GgufProjectionWeights& weights,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    std::vector<detail::GgufShape> shapes;
    for (const auto& part : weights.parts) {
        shapes.push_back({part.weight.qtype, part.weight.n, part.weight.k});
    }
    return detail::gguf_project_workspace_bytes(shapes, min_tokens, max_tokens);
}

void attn_input_proj(const Tensor& x, const GgufProjectionWeights& weights, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& workspace,
                     cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("attn_input_proj: T must be positive"); }
    const std::int32_t query_rows = q.ne[0];
    const std::int32_t kv_rows    = k.ne[0];
    require_matrix(x, x.ne[0], cols, "x");
    require_matrix(q, query_rows, cols, "q");
    require_matrix(gate, query_rows, cols, "gate");
    require_matrix(k, kv_rows, cols, "k");
    require_matrix(v, kv_rows, cols, "v");
    require_gguf_attention(weights, x.ne[0], query_rows, kv_rows);
    Tensor* outputs[] = {&q, &gate, &k, &v};
    std::vector<detail::GgufProduct> products;
    products.reserve(weights.parts.size());
    for (const auto& part : weights.parts) {
        detail::GgufProduct product;
        product.weight = &part.weight;
        product.out    = outputs[part.output];
        product.row    = part.row;
        products.push_back(product);
    }
    detail::gguf_project(x, products, workspace, stream);
}

} // namespace ninfer::ops
