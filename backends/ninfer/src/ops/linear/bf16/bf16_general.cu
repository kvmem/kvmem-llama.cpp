#include "ops/linear/bf16/bf16_general.cuh"
#include "core/device.h"
#include "ops/linear/bf16/bf16_dispatch.h"

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

// Runtime-shape bf16 GEMM used for every linear projection the specialised bf16 kernels do not
// tile. x is [tokens, k], weight is [n, k] row-major, and out is [tokens, n] token-major, matching
// the contiguous layout the specialised kernels produce.
void bf16_general_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    const std::int32_t K = weight.k;
    const std::int32_t T = x.ne[1];
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((T + kTileT - 1) / kTileT));
    bf16_general_gemm_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), N, K, T);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens) {
    (void)tokens;
    return &bf16_general_gemm;
}

} // namespace ninfer::ops::detail
