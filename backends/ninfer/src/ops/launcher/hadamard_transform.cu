// ninfer::ops - hadamard_transform launcher: one warp per (column, 1024-block), four warps per CTA.
#include "ops/launcher/hadamard_transform.h"

#include "core/device.h"
#include "ops/kernel/hadamard_producers.cuh"
#include "ops/kernel/hadamard_transform.cuh"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarpsPerCta = 4;

static_assert(kRmsnormHadamardWidth == kRmsnormHadamardFusedWidth);
static_assert(kGatedRmsnormHadamardHeadDim == kGatedRmsnormHadamardFusedHeadDim);
static_assert(kGatedRmsnormHadamardRowsPerCta == kGatedRmsnormHadamardFusedRows);

template <bool SignsAfter>
void launch(const __nv_bfloat16* x, const __nv_bfloat16* signs, __nv_bfloat16* out,
            std::int64_t items, std::int32_t blocks_per_column, cudaStream_t stream) {
    const std::int64_t ctas = (items + kWarpsPerCta - 1) / kWarpsPerCta;
    if (ctas > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("hadamard_transform: grid exceeds the CUDA launch limit");
    }
    hadamard_transform_1024_kernel<SignsAfter, kWarpsPerCta>
        <<<static_cast<unsigned>(ctas), kWarpsPerCta * kWarpSize, 0, stream>>>(x, signs, out, items,
                                                                               blocks_per_column);
}

} // namespace

void silu_mul_hadamard_launch(const Tensor& plane, const Tensor& signs, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t width             = out.ne[0];
    const std::int32_t blocks_per_column = width / kHadamardTransformBlock;
    const std::int64_t columns           = out.numel() / width;
    const std::int64_t items             = columns * blocks_per_column;
    if (items > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("silu_mul_hadamard: grid exceeds the CUDA launch limit");
    }
    silu_mul_hadamard_quarter_kernel<<<static_cast<unsigned>(items),
                                       kHadamardQuarterWarps * kWarpSize, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(plane.data),
        static_cast<const __nv_bfloat16*>(signs.data), static_cast<__nv_bfloat16*>(out.data), items,
        blocks_per_column);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_hadamard_5120_launch(const Tensor& x, const Tensor& weight, float eps,
                                  bool unit_offset, const Tensor& signs, Tensor& out,
                                  cudaStream_t stream) {
    const std::int64_t rows = x.numel() / kRmsnormHadamardWidth;
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm_hadamard: row count exceeds the CUDA grid limit");
    }
    const auto* x2 = static_cast<const __nv_bfloat162*>(x.data);
    const auto* w2 = static_cast<const __nv_bfloat162*>(weight.data);
    const auto* s  = static_cast<const __nv_bfloat16*>(signs.data);
    auto* o        = static_cast<__nv_bfloat16*>(out.data);
    if (unit_offset) {
        rmsnorm_hadamard_5120_kernel<RmsEpilogue::Offset>
            <<<static_cast<unsigned>(rows), kRmsnormHadamardThreads, 0, stream>>>(x2, w2, s, o,
                                                                                  rows, eps);
    } else {
        rmsnorm_hadamard_5120_kernel<RmsEpilogue::Plain>
            <<<static_cast<unsigned>(rows), kRmsnormHadamardThreads, 0, stream>>>(x2, w2, s, o,
                                                                                  rows, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

void gated_rmsnorm_hadamard_d128_launch(const Tensor& x, const Tensor& weight, const Tensor& z,
                                        float eps, const Tensor& signs,
                                        std::int32_t heads_per_column, Tensor& out,
                                        cudaStream_t stream) {
    const std::int64_t rows = x.numel() / kGatedRmsnormHadamardHeadDim;
    const std::int64_t ctas = rows / kGatedRmsnormHadamardRowsPerCta;
    if (ctas > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("gated_rmsnorm_hadamard: grid exceeds the CUDA launch limit");
    }
    gated_rmsnorm_hadamard_d128_kernel<<<static_cast<unsigned>(ctas),
                                         kGatedRmsnormHadamardRowsPerCta * kWarpSize, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(x.data), static_cast<const __nv_bfloat162*>(weight.data),
        static_cast<const __nv_bfloat162*>(z.data), static_cast<const __nv_bfloat16*>(signs.data),
        static_cast<__nv_bfloat16*>(out.data), heads_per_column, rows, eps);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_mul_hadamard_launch(const Tensor& gate, const Tensor& x, const Tensor& signs,
                                 Tensor& out, cudaStream_t stream) {
    const std::int32_t width             = signs.ne[0];
    const std::int32_t blocks_per_column = width / kHadamardTransformBlock;
    const std::int64_t items             = x.numel() / width * blocks_per_column;
    if (items > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("sigmoid_mul_hadamard: grid exceeds the CUDA launch limit");
    }
    sigmoid_mul_hadamard_quarter_kernel<<<static_cast<unsigned>(items),
                                          kHadamardQuarterWarps * kWarpSize, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(signs.data), static_cast<__nv_bfloat16*>(out.data), items,
        blocks_per_column);
    CUDA_CHECK(cudaGetLastError());
}

void embedding_rotated_t2_launch(const Tensor& ids, const Weight& table, const Tensor& signs,
                                 Tensor& out, cudaStream_t stream) {
    const std::int32_t width = out.ne[0];
    const std::int64_t items =
        static_cast<std::int64_t>(ids.ne[0]) * (width / kHadamardTransformBlock);
    const std::int64_t ctas = (items + kWarpsPerCta - 1) / kWarpsPerCta;
    if (ctas > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("embedding_rotated: grid exceeds the CUDA launch limit");
    }
    embed_gather_t2_hadamard_kernel<kWarpsPerCta>
        <<<static_cast<unsigned>(ctas), kWarpsPerCta * kWarpSize, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data),
            static_cast<const std::uint8_t*>(table.qdata), static_cast<const __half*>(table.scales),
            static_cast<const __nv_bfloat16*>(signs.data), static_cast<__nv_bfloat16*>(out.data),
            items, width);
    CUDA_CHECK(cudaGetLastError());
}

void hadamard_transform_launch(const Tensor& x, const Tensor& signs, bool inverse, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t k                 = x.ne[0];
    const std::int32_t blocks_per_column = k / kHadamardTransformBlock;
    const std::int64_t columns           = x.numel() / k;
    const std::int64_t items             = columns * blocks_per_column;
    const auto* x_bf16                   = static_cast<const __nv_bfloat16*>(x.data);
    const auto* signs_bf16               = static_cast<const __nv_bfloat16*>(signs.data);
    auto* out_bf16                       = static_cast<__nv_bfloat16*>(out.data);
    if (inverse) {
        launch<true>(x_bf16, signs_bf16, out_bf16, items, blocks_per_column, stream);
    } else {
        launch<false>(x_bf16, signs_bf16, out_bf16, items, blocks_per_column, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
