#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

namespace ninfer::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear(x, p.weight, output, p.policy, workspace, stream);
}

// `wide_verification` keeps the neural path's residual activation precision in a wide
// single-row copy verification (see residual_projection_policy).
inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream,
                        InputBasis basis = InputBasis::Primal, bool wide_verification = false) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear_add(x, p.weight, residual, residual_projection_policy(p, wide_verification),
                    workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear_swiglu(x, p.weight, output, p.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
