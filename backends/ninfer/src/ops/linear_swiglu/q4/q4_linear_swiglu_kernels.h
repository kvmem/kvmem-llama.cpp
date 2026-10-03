#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
// sm_86 routes 33..40 and 41..48 here (q4_linear_swiglu_plan.cpp); upstream dropped both tiles.
void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
// Wide folded tiles for the leading column blocks plus a narrow route for a small remainder.
void q4_linear_swiglu_mma_split_half_pair_r32_c128_tail_launch(const Tensor& x, const Weight& w,
                                                               Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);
// Upstream's small-T route over the unified sliced-K MMA (eight K warps, pairwise reduction, launch
// bounds per tile width), taken by the tiled launcher where fused_route_table
// "unified/q4_linear_swiglu" says so.
void q4_linear_swiglu_small_t_unified_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);
// Bench-only: forces the runtime-column-count instantiation that the launcher above drops when a
// width fills its tile exactly, so the cost of masking stays measurable rather than remembered.
void q4_linear_swiglu_small_t_tiled_masked_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                  cudaStream_t stream);

// int8 tensor-core small-T route, T=2..32. Selected by LinearPolicy::AllowA8IntDecode rather than
// by q4_linear_swiglu_resolve_plan's table, because it is a quality trade and therefore opt-in per
// engine; q4_linear_swiglu_execute_schedule also reaches it for the schedule bench.
//
// The integer small-T route wins from sixteen columns up and loses below, where a 64-k group has
// too little MMA work to pay for quantising the activations. Measured per width in
// ops/linear/q4/q4_small_t_mma_i8.cuh.
[[nodiscard]] bool q4_linear_swiglu_small_t_i8_supported(std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t q4_linear_swiglu_small_t_tiled_i8_workspace_bytes(std::int32_t tokens);
void q4_linear_swiglu_small_t_tiled_i8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
