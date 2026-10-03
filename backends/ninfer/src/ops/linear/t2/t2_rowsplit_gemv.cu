#include "ops/linear/t2/t2_rowsplit_gemv.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule>
void launch_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    t2_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_t2_gemv_r8_w1_k5120(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_gemv<T2GemvR8W1K5120Schedule>(x, w, out, stream);
}

void launch_t2_gemv_r8_w1_k6144(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    launch_gemv<T2GemvR8W1K6144Schedule>(x, w, out, stream);
}

void launch_t2_gemv_r8_w1_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_gemv<T2GemvR8W1K17408Schedule>(x, w, out, stream);
}

void launch_t2_gemv_r4_w1_word(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_gemv<T2GemvR4W1WordSchedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
