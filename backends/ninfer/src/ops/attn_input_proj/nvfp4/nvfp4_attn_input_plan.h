#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream);

// MMA schedule-band dispatch shared by the ordinary W4A4 launcher and the fused entry.
// Caller must guarantee 4 <= tokens < 1024 (the W4A4 sub-TMA band).
void launch_nvfp4_w4a4_mma_banded(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                  Tensor& v, Nvfp4W4a4Workspace workspace, std::int32_t tokens,
                                  cudaStream_t stream);

// Shared by the ordinary W4A4 launcher and the fused entry. This is the TMA cutoff.
[[nodiscard]] inline constexpr bool nvfp4_attn_input_tma_route(std::int32_t tokens) {
    return tokens >= 1024;
}

void launch_nvfp4_attn_input_fused_rmsnorm_quantize(const Tensor& residual,
                                                    const Tensor& norm_weight, float eps,
                                                    float input_scale_divisor,
                                                    Nvfp4W4a4Workspace workspace,
                                                    cudaStream_t stream);

void launch_nvfp4_attn_input_fused_rmsnorm_quantize_rowmajor(const Tensor& residual,
                                                             const Tensor& norm_weight, float eps,
                                                             float input_scale_divisor,
                                                             Nvfp4W4a4Workspace workspace,
                                                             cudaStream_t stream);

// Fused norm + A16 GEMV for exactly one token. No workspace use.
void nvfp4_attn_input_fused_rmsnorm_decode_launch(const Tensor& residual, const Tensor& norm_weight,
                                                  float eps, const Weight& weight, Tensor& q,
                                                  Tensor& gate, Tensor& k, Tensor& v,
                                                  cudaStream_t stream);

// NOTE: T==2,3 deliberately have no fused entry. A fused norm+SIMT variant was built and
// measured a regression (-2.0us at T=2, -6.3us at T=3 across runs): the redundant per-CTA
// norm wall cost scales with T while the saving (one launch) is fixed, so T>=2 in the
// row-parallel A16 structure crosses over. T==2,3 stay on the unfused path.
void nvfp4_attn_input_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight,
                                           float eps, const Weight& weight, Tensor& q, Tensor& gate,
                                           Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                           cudaStream_t stream);

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
