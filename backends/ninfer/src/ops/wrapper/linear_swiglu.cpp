#include "core/layout.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/silu_mul.h"

#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"
#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
    case LinearPolicy::AllowA8Int:
    case LinearPolicy::AllowA8IntDecode:
    // The cuBLAS prefill route is registered only for linear_swiglu and linear_add. Everywhere else
    // this policy means exactly what AllowA8Int means, and is accepted rather than rejected so that
    // one engine-wide setting does not have to be threaded per Op.
    case LinearPolicy::AllowPrefillCublas:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (is_gguf(qtype)) {
        const detail::GgufShape parent{qtype, gate_up_rows, input_rows};
        return detail::gguf_swiglu_workspace_bytes(parent, nullptr, min_tokens, max_tokens);
    }
    if (qtype == QType::T2_G128_FP16) {
        (void)linear_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                              LinearPolicy::A16Only, min_tokens, max_tokens);
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(DType::BF16, {gate_up_rows, max_tokens});
        return layout.peak_bytes(1);
    }
    if (qtype == QType::Q8_G32_FP16) {
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q4_G64_FP16) {
        // Every valid policy admits the A16 route (upstream relaxed the A16-only check); the
        // integer-A8 policies additionally admit the q4a8 prefill and small-T decode routes.
        const std::size_t a16 = detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
        const bool integer_a8 = allows_a8_int(policy);
        if (!integer_a8 || gate_up_rows != 34816 || input_rows != 5120) { return a16; }
        // The cuBLAS route holds a materialised int8 weight and an int32 output tile, so it is far
        // and away the widest claimant wherever it is admitted. Any interval reaching its width
        // gate has to reserve for it, because the resolver will pick it at that T.
        if (allows_cublas_prefill(policy) && max_tokens >= kCublasPrefillMinTokens) {
            return std::max(a16, detail::w4_cublas_prefill_workspace_capacity_bytes(
                                     gate_up_rows, input_rows,
                                     std::max(min_tokens, kCublasPrefillMinTokens), max_tokens));
        }
        // The small-T integer route stages quantised activations too, over its padded tile width.
        const auto decode_bytes = [&](std::int32_t t) -> std::size_t {
            return (policy == LinearPolicy::AllowA8IntDecode &&
                    detail::q4_linear_swiglu_small_t_i8_supported(t))
                       ? detail::q4_linear_swiglu_small_t_tiled_i8_workspace_bytes(t)
                       : 0;
        };
        // A single-T query must report exactly what that T will use, so it has to name the route
        // the resolver will pick. Across an interval the answer is an upper bound over both.
        if (min_tokens == max_tokens) {
            if (detail::q4a8_tokens_supported(min_tokens)) {
                return detail::q4a8_swiglu_workspace_capacity_bytes(min_tokens, max_tokens);
            }
            const std::size_t decode = decode_bytes(min_tokens);
            return decode != 0 ? decode : a16;
        }
        std::size_t widest = std::max(
            a16, detail::q4a8_swiglu_workspace_capacity_bytes(min_tokens, max_tokens));
        for (std::int32_t t = min_tokens; t <= max_tokens; ++t) {
            widest = std::max(widest, decode_bytes(t));
        }
        return widest;
    }
    if (qtype == QType::NVFP4 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    if (is_gguf(gate_up_weight.qtype)) {
        detail::gguf_swiglu(x, gate_up_weight, nullptr, out, ws, stream);
        return;
    }
    const bool large_shape = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                             gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                             gate_up_weight.padded_shape[1] == 5120;
    const bool q8_shape = x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                          gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                          gate_up_weight.padded_shape[1] == 2048;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !q8_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4_G64_FP16 &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool q8_weight =
        (q8_shape || large_shape) && gate_up_weight.qtype == QType::Q8_G32_FP16 &&
        gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
        gate_up_weight.qhigh == nullptr && gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool nvfp4_weight = large_shape && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight   = large_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const bool t2_weight    = large_shape && gate_up_weight.qtype == QType::T2_G128_FP16 &&
                              gate_up_weight.qhigh == nullptr &&
                              gate_up_weight.high_plane_bytes == 0 && common_row_split;
    if (!q4_weight && !q8_weight && !nvfp4_weight && !fp8_weight && !t2_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (t2_weight) {
        // Ternary rows project gate and up into one plane; SwiGLU is a separate elementwise pass
        // over its two halves. A16 by construction.
        auto scope     = ws.scope();
        Tensor gate_up = ws.alloc(DType::BF16, {gate_up_weight.n, t});
        linear(x, gate_up_weight, gate_up, stream);
        const Tensor gate = gate_up.slice(0, 0, out.ne[0]);
        const Tensor up   = gate_up.slice(0, out.ne[0], out.ne[0]);
        silu_mul(gate, up, out, stream);
        return;
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, q8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    // The cuBLAS route first, where it is admitted and wide enough to pay: about 2x the integer
    // mainloop at 4096 tokens, 1.83x at 1024, and a loss below that -- hence the width gate rather
    // than an unconditional preference. Everything narrower falls through to the routes below.
    if (allows_cublas_prefill(policy) && t >= kCublasPrefillMinTokens && q4_weight &&
        detail::w4_cublas_prefill_supported(gate_up_weight, t)) {
        detail::w4_cublas_swiglu_launch(x, gate_up_weight, out, ws, stream);
        return;
    }

    const bool integer_a8 = allows_a8_int(policy);
    if (integer_a8 && q4_weight && detail::q4a8_swiglu_supported(gate_up_weight, t)) {
        detail::q4a8_swiglu_launch(x, gate_up_weight, out, ws, stream);
        return;
    }
    // Decode and verify widths, opt-in only: the integer small-T route wins from sixteen columns
    // up and loses below, so the narrow widths stay on A16 (q4_small_t_mma_i8.cuh has the table).
    if (policy == LinearPolicy::AllowA8IntDecode && q4_weight &&
        detail::q4_linear_swiglu_small_t_i8_supported(t) &&
        gate_up_weight.n == 34816 && gate_up_weight.k == 5120) {
        detail::q4_linear_swiglu_small_t_tiled_i8_launch(x, gate_up_weight, out, ws, stream);
        return;
    }

    if (q8_weight) {
        detail::q8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
    }
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

void linear_swiglu(const Tensor& x, const Weight& gate_weight, const Weight& up_weight, Tensor& out,
                   LinearPolicy policy, WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    if (!is_gguf(gate_weight.qtype) || !is_gguf(up_weight.qtype)) {
        throw std::invalid_argument("linear_swiglu: the two-parent form is registered for GGUF");
    }
    detail::gguf_swiglu(x, gate_weight, &up_weight, out, ws, stream);
}

std::size_t linear_swiglu_pair_workspace_capacity_bytes(QType gate_qtype, QType up_qtype,
                                                        std::int32_t rows,
                                                        std::int32_t input_rows,
                                                        LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    validate_policy(policy);
    if (!is_gguf(gate_qtype) || !is_gguf(up_qtype)) {
        throw std::invalid_argument("linear_swiglu workspace: the two-parent form is GGUF only");
    }
    const detail::GgufShape gate{gate_qtype, rows, input_rows};
    const detail::GgufShape up{up_qtype, rows, input_rows};
    return detail::gguf_swiglu_workspace_bytes(gate, &up, min_tokens, max_tokens);
}

} // namespace ninfer::ops
