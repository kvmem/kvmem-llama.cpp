#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_plan.h"

#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    Q8AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 4> kTargetRoutes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 64, Q8AttnInputScheduleId::SplitKMmaDirect},
    {65, 128, Q8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, Q8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 9> kCompanionRoutes{{
    {1, 1, Q8AttnInputScheduleId::DecodeR8Direct},
    {2, 96, Q8AttnInputScheduleId::SplitKMmaDirect},
    {97, 192, Q8AttnInputScheduleId::MmaR32C64},
    {193, 288, Q8AttnInputScheduleId::MmaR64C96},
    {289, 320, Q8AttnInputScheduleId::MmaR64C64},
    {321, 384, Q8AttnInputScheduleId::MmaR64C128},
    {385, 448, Q8AttnInputScheduleId::MmaR128C64},
    {449, 560, Q8AttnInputScheduleId::MmaR128C80},
    {561, kAnyCols, Q8AttnInputScheduleId::MmaR64C128},
}};

// Measured on sm_86 by bench/ops/q8_dflash2_schedule_bench.cu, cold, median of 21. Upstream tuned
// this table on sm_120; here SmallT was routed to 48 while losing 22-35% from T=17 on, and
// R16C64K128 owned 49..63 while being the slowest kernel at every width measured.
//
// The R32C32K128 / R32C64K128 alternation below is not overfitting. The two kernels differ only in
// column tile width, so their cost is quantised into ceil(T/32) and ceil(T/64) waves respectively,
// and the wave boundaries interleave: c32 is ahead while it needs an odd number of waves for the
// same work c64 does in a whole one, and behind on the next band. Measured per-band winners:
//
//   T      17-32   33-64   65-96   97-128   129-159   160-192   193+
//   c32    77-82   131-160 163-184 214-245  250-281   313-340   365+
//   c64k   105-123 105-123 208-211 193-220  289-292   294-309   372+
//
// R64C128 is a single c128 tile at T<=128 and is the best kernel at exactly 128 (217 us against
// 220), but 129 costs it a second tile and it runs 356-361 through 192. The 1.4% it would gain at
// exactly 128 is inside this bench's run-to-run spread, so 97..128 stays on one route rather than
// carrying a single-width special case that a future merge would have to reason about.
//
// DFlash2MmaR16C64K128 is no longer selected at any width -- it never won one. It is kept rather
// than deleted so the next catch-up merge does not have to re-add it.
constexpr std::array<RouteSpec, 8> kDFlash2Routes{{
    {1, 16, Q8AttnInputScheduleId::DFlash2SmallT},
    {17, 32, Q8AttnInputScheduleId::DFlash2MmaR32C32K128},
    {33, 64, Q8AttnInputScheduleId::DFlash2MmaR32C64K128},
    {65, 96, Q8AttnInputScheduleId::DFlash2MmaR32C32K128},
    {97, 128, Q8AttnInputScheduleId::DFlash2MmaR32C64K128},
    {129, 159, Q8AttnInputScheduleId::DFlash2MmaR32C32K128},
    {160, 192, Q8AttnInputScheduleId::DFlash2MmaR32C64},
    {193, kAnyCols, Q8AttnInputScheduleId::DFlash2MmaR64C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kTargetRoutes),
              "Q8 target attention input routes must be exact and closed");
static_assert(catalog_is_closed(kCompanionRoutes),
              "Q8 companion attention input routes must be exact and closed");
static_assert(catalog_is_closed(kDFlash2Routes),
              "Q8 DFlash2 attention input routes must be exact and closed");

bool is_companion_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 2048;
}

bool is_dflash2_shape(const Q8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 5120;
}

bool supported_shape(const Q8AttnInputProblem& problem) noexcept {
    const bool target_qkgv =
        problem.query_rows == 4096 && problem.kv_rows == 512 && problem.parent_rows == 9216;
    const bool target_shape = problem.input_rows == 2048 && problem.padded_k == 2048 && target_qkgv;
    return target_shape || is_companion_shape(problem) || is_dflash2_shape(problem);
}

} // namespace

const char* q8_attn_input_schedule_name(Q8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        return "attn_input_proj.q8.decode.r8.direct.k2048";
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        return "attn_input_proj.q8.splitk.mma.r16.direct";
    case Q8AttnInputScheduleId::SimtR8C4:
        return "attn_input_proj.q8.simt.r8.c4";
    case Q8AttnInputScheduleId::MmaR32C64:
        return "attn_input_proj.q8.mma.r32.c64";
    case Q8AttnInputScheduleId::MmaR32C128:
        return "attn_input_proj.q8.mma.r32.c128";
    case Q8AttnInputScheduleId::MmaR64C64:
        return "attn_input_proj.q8.mma.r64.c64";
    case Q8AttnInputScheduleId::MmaR64C96:
        return "attn_input_proj.q8.mma.r64.c96";
    case Q8AttnInputScheduleId::MmaR64C128:
        return "attn_input_proj.q8.mma.r64.c128";
    case Q8AttnInputScheduleId::MmaR128C64:
        return "attn_input_proj.q8.mma.r128.c64";
    case Q8AttnInputScheduleId::MmaR128C80:
        return "attn_input_proj.q8.mma.r128.c80";
    case Q8AttnInputScheduleId::DFlash2SmallT:
        return "attn_input_proj.q8.dflash2.small_t";
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
        return "attn_input_proj.q8.dflash2.mma.r16.c64.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
        return "attn_input_proj.q8.dflash2.mma.r32.c32.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        return "attn_input_proj.q8.dflash2.mma.r32.c64.k128";
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
        return "attn_input_proj.q8.dflash2.mma.r32.c64";
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
        return "attn_input_proj.q8.dflash2.mma.r64.c128";
    }
    return "attn_input_proj.q8.unknown";
}

bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem) {
    if (!q8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q8 attention input: exact problem or column count is not admitted");
    }
    const auto resolve_from = [&](const auto& routes) -> Q8AttnInputPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("Q8 attention input: admitted problem has no covering route");
    };
    if (is_dflash2_shape(problem)) { return resolve_from(kDFlash2Routes); }
    if (is_companion_shape(problem)) { return resolve_from(kCompanionRoutes); }
    return resolve_from(kTargetRoutes);
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    if (problem.parent_rows != 9216 || problem.kv_rows != 512 ||
        resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact four-output problem");
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
    case Q8AttnInputScheduleId::MmaR64C64:
    case Q8AttnInputScheduleId::MmaR64C96:
    case Q8AttnInputScheduleId::MmaR128C64:
    case Q8AttnInputScheduleId::MmaR128C80:
    case Q8AttnInputScheduleId::DFlash2SmallT:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        throw std::logic_error("Q8 attention input: three-output schedule in four-output plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, gate, k, v,
                               stream);
}

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const Q8AttnInputPlan resolved = q8_attn_input_resolve_plan(problem);
    const bool companion           = is_companion_shape(problem);
    const bool dflash2             = is_dflash2_shape(problem);
    if ((!companion && !dflash2) || resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "Q8 attention input: plan does not match exact three-output problem");
    }
    if (dflash2) {
        switch (plan.schedule) {
        case Q8AttnInputScheduleId::DFlash2SmallT:
            q8_dflash2_attn_input_small_t_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
            q8_dflash2_attn_input_mma_r16_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
            q8_dflash2_attn_input_mma_r32_c32_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
            q8_dflash2_attn_input_mma_r32_c64_k128_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR32C64:
            q8_dflash2_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
            return;
        case Q8AttnInputScheduleId::DFlash2MmaR64C128:
            q8_dflash2_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
            return;
        default:
            throw std::logic_error("Q8 attention input: non-DFlash2 schedule in DFlash2 plan");
        }
    }
    switch (plan.schedule) {
    case Q8AttnInputScheduleId::DecodeR8Direct:
        q8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SplitKMmaDirect:
        q8_attn_input_splitk_mma_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::SimtR8C4:
        q8_attn_input_simt_r8_c4_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C128:
        q8_attn_input_mma_r32_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR32C64:
        q8_companion_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C64:
        q8_companion_attn_input_mma_r64_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C96:
        q8_companion_attn_input_mma_r64_c96_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C64:
        q8_companion_attn_input_mma_r128_c64_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR128C80:
        q8_companion_attn_input_mma_r128_c80_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::MmaR64C128:
        q8_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    case Q8AttnInputScheduleId::DFlash2SmallT:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64:
    case Q8AttnInputScheduleId::DFlash2MmaR64C128:
    case Q8AttnInputScheduleId::DFlash2MmaR16C64K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C32K128:
    case Q8AttnInputScheduleId::DFlash2MmaR32C64K128:
        throw std::logic_error("Q8 attention input: DFlash2 schedule in companion plan");
    }
    throw std::logic_error("Q8 attention input: unknown schedule");
}

void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const Q8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    q8_attn_input_execute_plan(q8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
