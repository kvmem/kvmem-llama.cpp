#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"
#include "core/layout.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/device_route.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4LinearSwiGluScheduleId schedule;
};

constexpr Q4LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

constexpr std::array<RouteSpec, 5> kRoutes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    // Re-measured 2026-09-11 on an RTX 3090 under Linux after the small-T MMA's rewrite (padded code
    // rows, then a permuted k order with 64-bit code loads and no int-to-float decode -- see
    // q4_ksplit_mma.cuh, formerly q4_small_t_mma.cuh). Cold, median of 31 (us):
    //
    //   T            2      4      8     12     16     20     24     25     32
    //   small_t  118.8  120.8  123.9  164.9  201.7  268.3  290.8  358.4  431.1
    //   c40      439.3  438.3  442.4  444.4  443.4  447.4  444.4  442.4  444.4
    //
    // SmallTTiled now wins to the kernel's 32-column limit, so the c40 tile keeps only 33..40. The
    // old 24/25 boundary was measured against the kernel before the rewrite (399 us at T=24, and
    // 446.5 against c40's 444.4 at T=25).
    //
    // Its 16-32-column tiles were then given shared activation slabs (KWarps 4/2, and two tiles
    // per warp at 24 columns; q4_linear_swiglu_gemv.cu has the layout sweep). Same bench (us):
    //
    //   T            9     16     17     24     32
    //   small_t  145.4  152.6  167.9  178.2  208.9     (24 then two tiles per warp: ~164)
    //   c40      452.6  444.4  451.6  445.4  447.5
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTTiled},
    {{33, 40}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40},
    {{41, 48}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48},
    // Measured on sm_86 by bench/ops/q4_linear_swiglu_schedule_bench.cu, cold, median of 9-15,
    // repeated three to four times per width because this Op is noisier than the others here.
    //
    // Upstream alternates Materialized and the c128 tile three times across 49..640. That shape is
    // implausible on its face -- the winner does not genuinely change back and forth over a
    // contiguous range -- and it does not survive measurement. Materialized lost at every width in
    // {49,128} and {257,384}, in every run:
    //
    //   T          49     64     96    128    257    320    384
    //   c128      812    823    848    809   2309   2310   2190
    //   mat       856    855    890    864   3004   2724   2398
    //   c128 by  5.1%   3.9%   4.7%   6.4%  23.1%  15.2%   8.7%
    //
    // So {49,128} and {257,384} join the c128 bands on either side.
    //
    // {513,640} used to stay on Materialized because the measurement could not separate them:
    // over four runs Materialized won 576 three times, but c128 there ranged 4147-4959 us (19.6%
    // spread) and the margins outside the single outlying run were +-2%. Re-measured 2026-09-09
    // with --spread (which did not exist then, so the spread that blocked the decision was never
    // visible) at 31-51 repetitions per point on an idle card. c128 won all three widths in four
    // independent runs; us, range of the per-run medians:
    //
    //   T            513          576          640
    //   c128     3788-3906    3876-3933    3611-3641
    //   mat      3863-3956    3950-4216    3895-4077
    //   c128 by  1.3-2.5%     1.2-7.5%     7.3-11.2%
    //
    // The min settles it: c128's fastest sample beats Materialized's fastest at every width in
    // every run, by 7-9% at 640 with no overlap (3524-3581 against 3862-3867). Sign is consistent
    // 12 times out of 12.
    //
    // **This bench overstates the margin, so do not quote it as a speedup.** Confirmed in situ by
    // profiling a single 600-token chunk (nsys, node-level graph trace, 64 layers): Materialized
    // costs 269.73 ms in q4_rowsplit_gemm_mma plus 5.16 ms in silu_and_mul_dim0_split = 274.89 ms,
    // against 268.32 ms for the c128 pair kernel. c128 still wins, but by 2.4%, not 7.5% -- and
    // both are slower per call in situ than in the bench (4193 vs 3625 us for c128, 4295 vs 3927
    // for Materialized). The bench flushes L2 before every repetition; a real prefill does not
    // arrive with a cold cache, and the flush penalises Materialized's second pass more than
    // production does. The same caveat applies to every band in this table.
    //
    // End-to-end the win is invisible: pp600 measured 886.0-888.8 tok/s with c128 against
    // 888.5-892.1 on Materialized, i.e. inside the run-to-run spread, because 6.6 ms of a ~660 ms
    // prefill is under the +-3% that unrelated kernels move between runs. Keep the change for
    // consistency and because it is right, not for a number.
    //
    // Reachability, so nobody over-values this band: q4a8_swiglu takes any width with
    // `tokens >= 128 && tokens % 128 == 0`, and --prefill-chunk must be a multiple of 128, so
    // every full prefill chunk goes to the integer-activation kernel and never reaches this table.
    // The 49..end route is reached only by decode widths and by a prompt's ragged tail chunk. That
    // is why {513,640} was both close and inconsequential for so long.
    //
    // With this the alternation is gone completely and 49..end is one route. That was the
    // suspicious thing about upstream's table to begin with.
    //
    // Upstream's 5b4303c0 (RTX 5090) re-split 33..end around its retuned Q4 linear routes, adding
    // MmaSplitHalfPairR32C128Tail for 129..168. That schedule is kept executable but is not routed
    // here: it has not been measured on sm_86.
    {{49, kAnyCols}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q4 LinearSwiGLU routes must be exact, contiguous, and closed");

bool supported_shape(const Q4LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

} // namespace

std::size_t q4_linear_swiglu_materialized_workspace_bytes(std::int32_t gate_up_rows,
                                                          std::int32_t max_cols) {
    return materialized_workspace_bytes(gate_up_rows, max_cols);
}

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        return "linear_swiglu.q4.gemv.paired_rows";
    case Q4LinearSwiGluScheduleId::SmallTTiled:
        return "linear_swiglu.q4.mma.small_t.tiled";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c40";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c48";
    case Q4LinearSwiGluScheduleId::Materialized:
        return "linear_swiglu.q4.materialized";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128.narrow_tail";
    case Q4LinearSwiGluScheduleId::SmallTTiledI8:
        return "linear_swiglu.q4.mma.small_t.tiled_i8";
    case Q4LinearSwiGluScheduleId::SmallTTiledMasked:
        return "linear_swiglu.q4.mma.small_t.tiled_masked";
    }
    return "linear_swiglu.q4.unknown";
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem) {
    if (!q4_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q4 linear_swiglu: exact problem or column count is not admitted");
    }
    // Materialized needs a workspace sized from the compiled bands, so profiles choose among the
    // workspace-free schedules only.
    using Id = Q4LinearSwiGluScheduleId;
    static constexpr std::array<DeviceRouteCandidate<Id>, 6> kCandidates{{
        {"gemv_pair", Id::GemvPair, 1},
        {"small_t_tiled", Id::SmallTTiled, 32},
        {"split_half_pair_c40", Id::MmaSplitHalfPairR32C40, 0},
        {"split_half_pair_c48", Id::MmaSplitHalfPairR32C48, 0},
        {"split_half_pair_c128", Id::MmaSplitHalfPairR32C128, 0},
        {"split_half_pair_c128_tail", Id::MmaSplitHalfPairR32C128Tail, 0},
    }};
    if (const auto* routed = routed_candidate<Id>("q4_linear_swiglu/34816x17408x5120",
                                                  problem.cols, kCandidates)) {
        return {routed->id, 0};
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q4LinearSwiGluPlan plan{
            route.schedule,
            0,
        };
        switch (route.schedule) {
        case Q4LinearSwiGluScheduleId::GemvPair:
        case Q4LinearSwiGluScheduleId::SmallTTiled:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
            return plan;
        case Q4LinearSwiGluScheduleId::Materialized:
            plan.workspace_bytes = materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
            return plan;
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
            return plan;
        case Q4LinearSwiGluScheduleId::SmallTTiledI8:
        case Q4LinearSwiGluScheduleId::SmallTTiledMasked:
            // Never in kRoutes: reached solely via execute_schedule.
            throw std::logic_error("q4 linear_swiglu: schedule is not routed");
        }
    }
    throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_swiglu: invalid column interval");
    }
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols});
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols});

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum                     = std::max(maximum, q4_linear_swiglu_resolve_plan(
                                        {gate_up_rows, output_rows, k, padded_k, endpoint})
                                                            .workspace_bytes);
    }
    return maximum;
}

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan resolved = q4_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }
    q4_linear_swiglu_execute_schedule(plan.schedule, x, w, out, ws, stream);
}

void q4_linear_swiglu_execute_schedule(Q4LinearSwiGluScheduleId schedule, const Tensor& x,
                                       const Weight& w, Tensor& out, WorkspaceArena& ws,
                                       cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTTiled:
        q4_linear_swiglu_small_t_tiled_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128Tail:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_tail_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTTiledI8:
        q4_linear_swiglu_small_t_tiled_i8_launch(x, w, out, ws, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTTiledMasked:
        q4_linear_swiglu_small_t_tiled_masked_launch(x, w, out, stream);
        return;
    }
    throw std::logic_error("q4 linear_swiglu: unknown schedule");
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan plan = q4_linear_swiglu_resolve_plan(problem);
    q4_linear_swiglu_execute_plan(plan, x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
