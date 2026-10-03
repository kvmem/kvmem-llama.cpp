#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void linear_topk_q8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                            const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                                const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q4_launch(const Tensor& hidden, const Weight& head,
                           const Tensor& row_to_global_ids, const LinearTopKWorkspace& workspace,
                           cudaStream_t stream);
// The direct-rows Q4 head through the unified sliced-K MMA (fused_route_table
// "unified/q4_linear_topk").
void linear_topk_q4_unified_launch(const Tensor& hidden, const Weight& head,
                                   const Tensor& row_to_global_ids,
                                   const LinearTopKWorkspace& workspace, cudaStream_t stream);
// row_to_global_ids is null for the full head.
void linear_topk_t2_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const Tensor* row_to_global_ids, Tensor& logits,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream);
// Top sixteen of already materialized BF16 logits [rows, columns] (rows = logits.ne[0]).
void linear_topk_logits_launch(const Tensor& logits, std::int32_t valid_rows,
                               const Tensor* row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q4_m64_launch(const Tensor& hidden, const Weight& head,
                               const Tensor& row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_merge_launch(const LinearTopKWorkspace& workspace, Tensor& candidate_ids,
                              Tensor& candidate_scores, cudaStream_t stream);

} // namespace ninfer::ops::detail
