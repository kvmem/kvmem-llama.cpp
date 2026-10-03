#include "ops/linear/t2/t2_small_t_mma.cuh"

#include "core/device.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int ColumnTiles>
void launch_small_t(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using Schedule          = T2SmallTMmaSchedule<ColumnTiles, 3>;
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = w.padded_shape[1];
    const std::int32_t cols = x.ne[1];
    if ((rows % Schedule::kRows) != 0 || (k % Schedule::kStageK) != 0 || k != x.ne[0] ||
        cols <= 0 || cols > Schedule::kColumns) {
        throw std::invalid_argument("t2 small-T MMA: unsupported shape");
    }
    const dim3 grid(static_cast<unsigned>(rows / Schedule::kRows), 1u, 1u);
    t2_small_t_mma_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k,
        cols);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_t2_small_t_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 8) {
        launch_small_t<1>(x, w, out, stream);
    } else {
        launch_small_t<2>(x, w, out, stream);
    }
}

} // namespace ninfer::ops::detail
