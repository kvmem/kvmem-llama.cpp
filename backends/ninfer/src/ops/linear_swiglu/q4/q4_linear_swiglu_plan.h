#pragma once

#include "core/weight.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4LinearSwiGluScheduleId {
    GemvPair,
    SmallTTiled,
    MmaSplitHalfPairR32C40,
    MmaSplitHalfPairR32C48,
    Materialized,
    MmaSplitHalfPairR32C128,
    MmaSplitHalfPairR32C128Tail,
    // int8 tensor-core small-T route, T=32 only. Not in the route table -- reachable only through
    // q4_linear_swiglu_execute_schedule, for measuring against SmallTTiled before any decision to
    // wire it in. See q4_small_t_mma_i8.cuh and docs/performance.md's tensor-rate-bound C8 finding.
    SmallTTiledI8,
    // Bench-only: SmallTTiled with the runtime column count it now avoids when the width fills its
    // tile exactly. Kept so the cost of masking stays measurable rather than remembered.
    SmallTTiledMasked,
};

struct Q4LinearSwiGluProblem {
    std::int32_t gate_up_rows;
    std::int32_t output_rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4LinearSwiGluPlan {
    Q4LinearSwiGluScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept;

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept;
Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem);

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols);

// Workspace for Materialized alone, sized directly from the column count rather than from which
// columns resolve_plan's route table currently sends there. Materialized's workspace grows
// monotonically with cols, so sizing for the widest column count in a sweep covers every narrower
// one too. Exists for callers -- benchmarks in particular -- that run Materialized outside its
// routed interval; q4_linear_swiglu_capacity_workspace_bytes reports zero for those callers and is
// the wrong function to size against.
std::size_t q4_linear_swiglu_materialized_workspace_bytes(std::int32_t gate_up_rows,
                                                          std::int32_t max_cols);

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream);

// Runs a schedule without first checking that it is the one resolve_plan would pick;
// q4_linear_swiglu_execute_plan is exactly this plus that check.
//
// It exists so a bench can time every candidate schedule at the same column count, which is the
// only way to tell whether a route boundary sits in the right place. Materialized in particular
// cannot be timed any other way: it is not a kernel but a composite -- linear() into a workspace,
// then silu_mul() over the two halves -- so a bench that called the pieces itself would be timing a
// replica of the dispatch rather than the dispatch. Mirrors q8_pair_execute_schedule, which exists
// for the same reason.
//
// Nothing on the inference path should call this: the check execute_plan adds is what keeps a plan
// from being executed against a problem it was not resolved for.
void q4_linear_swiglu_execute_schedule(Q4LinearSwiGluScheduleId schedule, const Tensor& x,
                                       const Weight& w, Tensor& out, WorkspaceArena& ws,
                                       cudaStream_t stream);
void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
