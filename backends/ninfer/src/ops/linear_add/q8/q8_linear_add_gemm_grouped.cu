#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "ops/linear_add/q8/q8_linear_add_kernels_unified.h"
#include "ops/linear/common/route_table.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_ksplit_grouped_mma.cuh"

namespace ninfer::ops::detail {
namespace {

template <int K>
void launch(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    constexpr int kColumns     = 128;
    constexpr int kSplits      = 2;
    constexpr int kTokenGroups = 4;
    const dim3 grid(5120 / 16, div_up(x.ne[1], kColumns));
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual.data), 5120};
    q8_ksplit_grouped_mma_kernel<K, kColumns, kSplits, kTokenGroups, 1, Q8ContiguousOutput, true,
                                 true><<<grid, kSplits * kTokenGroups * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_linear_add_grouped_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                  cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_grouped_launch(x, w, residual, stream);
        return;
    }
    if (w.k == 6144) {
        launch<6144>(x, w, residual, stream);
    } else {
        launch<17408>(x, w, residual, stream);
    }
}

} // namespace ninfer::ops::detail
