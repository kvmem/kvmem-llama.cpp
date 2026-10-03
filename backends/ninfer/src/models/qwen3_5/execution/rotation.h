#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

// A Hadamard-rotated checkpoint stores W * diag(s) * H per 1024-block (H normalized), so a Use of
// such a matrix multiplies FWHT(s * x) instead of x. The residual stream stays in the primal
// basis; only the inputs of rotated projections are transformed. The sign vector travels with the
// prepared weight, and an empty tensor marks an ordinary matrix.
[[nodiscard]] inline bool rotated(const Tensor& signs) noexcept { return signs.data != nullptr; }

[[nodiscard]] inline const Tensor& projection_signs(const ops::ProjectionWeights& projection) {
    return std::visit([](const auto& weights) -> const Tensor& { return weights.hadamard_signs; },
                      projection);
}

// Whether a projection input still needs its rotation, or arrives rotated from the op that produced
// it (rmsnorm_hadamard and the other fused producers).
enum class InputBasis : std::uint8_t { Primal, Rotated };

// x itself, or its rotation in a workspace allocation that lives in the caller's scope.
[[nodiscard]] inline Tensor rotated_input(const Tensor& x, const Tensor& signs,
                                          WorkspaceArena& workspace, cudaStream_t stream,
                                          InputBasis basis = InputBasis::Primal) {
    if (!rotated(signs) || basis == InputBasis::Rotated) { return x; }
    Tensor out = workspace.alloc(DType::BF16, {x.ne[0], x.ne[1], x.ne[2], x.ne[3]});
    ops::hadamard_transform(x, signs, false, out, stream);
    return out;
}

// The token rows of `ids`, restored to the primal basis when the table is stored rotated.
inline void embed_tokens(const Tensor& ids, const Weight& table, const Tensor& signs, Tensor& out,
                         cudaStream_t stream) {
    if (rotated(signs)) {
        ops::embedding_rotated(ids, table, signs, out, stream);
    } else {
        ops::embedding(ids, table, out, stream);
    }
}

// Workspace bytes of rotated_input over `columns` columns of `rows` features, followed by the
// `inner` bytes the consuming projection takes from the same workspace.
[[nodiscard]] inline std::size_t rotated_workspace_bytes(const Tensor& signs, std::int32_t rows,
                                                         std::int32_t columns, std::size_t inner) {
    if (!rotated(signs)) { return inner; }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, columns});
    if (inner != 0) { (void)layout.alloc_bytes(inner); }
    return layout.peak_bytes(1);
}

} // namespace ninfer::models::qwen3_5::execution
