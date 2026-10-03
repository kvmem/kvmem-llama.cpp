#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/linear/q8/q8_ksplit_config.h"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_rowsplit_output.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/common/route_table.h"
#include "ops/linear/q8/q8_ksplit_mma.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include <cuda_bf16.h>
#include <array>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 5120, kGroups = 320;

__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    const int index = col * kRows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * kRows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * kRows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

using Launch = Q8Launch;

template <int InputRows, int TileColumns>
void tiled_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int Warps =
#if defined(NINFER_SM8X_COMPAT)
        // 8 warps at TileColumns=40 with InputRows=4096 needs 49664 B of static shared memory,
        // over sm_86's 49152 B cap. Four warps is what the other q8 k-split schedules already
        // fall back to on this architecture (see ops/linear/q8/q8_ksplit_config.h).
        InputRows == 4096 ? (TileColumns <= 24 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
#else
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
#endif
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry            = Q8LinearGeometry<kRows, InputRows>;
    using Schedule            = Q8KSplitSchedule<Warps, TileColumns, Warps == 8 ? 2 : 3,
                                                 Q8KSplitScaleAccess::Shared, Activation>;
    constexpr int SharedBytes = TileColumns > 64 ? sizeof(Q8KSplitSharedStorage<Schedule>) : 0;
    if constexpr (SharedBytes > 0) {
        configure_cuda_device_once([] {
            return cudaFuncSetAttribute(
                q8_ksplit_mma_kernel<Geometry, TileColumns, Schedule, Q8ContiguousOutput,
                                     Q8KSplitStoreEpilogue, Q8KSplitIdentityRows, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes);
        });
    }
    const int columns = x.ne[1];
    Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    const dim3 grid(kRows / 16, (columns + TileColumns - 1) / TileColumns);
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(grid), dim3(Schedule::kThreads), SharedBytes, stream},
        q8_ksplit_mma_kernel<Geometry, TileColumns, Schedule, Q8ContiguousOutput,
                             Q8KSplitStoreEpilogue, Q8KSplitIdentityRows, false, true>,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitStoreEpilogue{},
        Q8KSplitIdentityRows{}, columns));
    CUDA_CHECK(cudaGetLastError());
}

// Upstream's move of the tiled projection onto the unified sliced-K MMA (fused_route_table
// "unified/q8_dynamic_grouped_conv"), which stages wide tiles in dynamic shared memory.
template <int InputRows, int TileColumns>
void tiled_projection_unified(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    constexpr int Warps =
        InputRows == 4096 ? (TileColumns <= 40 ? 8 : 4) : (TileColumns <= 32 ? 8 : 4);
    constexpr Cache Activation =
        InputRows == 4096 && ((TileColumns > 24 && TileColumns <= 40) || TileColumns > 48)
            ? Cache::cg
            : Cache::ca;
    using Geometry = Q8LinearGeometry<kRows, InputRows>;
    using Schedule = Q8A16SlicedKMmaSchedule<TileColumns, Warps, 1, Warps == 8 ? 2 : 3,
                                             Q8ScaleAccess::Shared, Activation>;
    LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kRows};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, TileColumns, false>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream);
    CUDA_CHECK(cudaGetLastError());
}

// Live columns stay dynamic; only the eight-column MMA accumulator layout is specialized.
template <int C, std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&tiled_projection<C, 8 * (1 + static_cast<int>(I))>...};
}

template <int C, std::size_t... I>
constexpr auto make_unified_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &tiled_projection_unified<C, 8 * (1 + static_cast<int>(I))>...};
}

constexpr auto attention         = make_launchers<4096>(std::make_index_sequence<11>{});
constexpr auto mlp               = make_launchers<17408>(std::make_index_sequence<11>{});
constexpr auto attention_unified = make_unified_launchers<4096>(std::make_index_sequence<11>{});
constexpr auto mlp_unified       = make_unified_launchers<17408>(std::make_index_sequence<11>{});

__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    pdl::enter();
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    finish_value(row, col, width, __bfloat162float(projected[index]),
                 col % width ? __bfloat162float(projected[index - kRows]) : 0.0f, base, delta,
                 residual);
}

void materialized(Q8DynamicConvAddSchedule schedule, const Tensor& x, const Weight& weight,
                  const Tensor& base, const Tensor& delta, Tensor& residual, Tensor& projected,
                  cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({kRows, tokens});
    const bool unified =
        fused_route_table("unified/q8_dynamic_grouped_conv", tokens) == LinearRouteTable::Unified;
    switch (schedule) {
    case Q8DynamicConvAddSchedule::TiledMma: {
        const auto& launchers = x.ne[0] == 4096 ? (unified ? attention_unified : attention)
                                                : (unified ? mlp_unified : mlp);
        launchers[(tokens - 1) / 8](flat, weight, result, stream);
        break;
    }
    case Q8DynamicConvAddSchedule::MmaK128:
        if (unified) {
            launch_q8_a16_mma_r64x32_t64_k128_a1(flat, weight, result, stream);
        } else {
            launch_q8_mma_r64x32_c64_k128_a1(flat, weight, result, stream);
        }
        break;
    }
    dynamic_grouped_conv_finish_launch(projected, base, delta, residual, x.ne[1], stream);
}
} // namespace

void dynamic_grouped_conv_finish_launch(const Tensor& projected, const Tensor& base,
                                        const Tensor& delta, Tensor& residual, std::int32_t width,
                                        cudaStream_t stream) {
    const int tokens = projected.ne[1] * projected.ne[2] * projected.ne[3];
    const dim3 grid((kRows + 255) / 256, tokens);
    CUDA_CHECK(pdl::launch_consumer({dim3(grid), dim3(256), 0, stream}, finish_kernel,
                                    static_cast<const __nv_bfloat16*>(projected.data),
                                    static_cast<const __nv_bfloat16*>(base.data),
                                    static_cast<const __nv_bfloat16*>(delta.data),
                                    static_cast<__nv_bfloat16*>(residual.data), width));
    CUDA_CHECK(cudaGetLastError());
}

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base, const Tensor& delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream) {
    materialized(schedule, x, weight, base, delta, residual, projected, stream);
}
} // namespace ninfer::ops::detail
