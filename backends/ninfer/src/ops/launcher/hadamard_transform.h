#pragma once

// ninfer::ops::detail - private launch prototype for hadamard_transform.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Geometries with a fused producer route; every other shape takes the op followed by
// hadamard_transform in place, which the fused routes reproduce bit for bit.
inline constexpr std::int32_t kRmsnormHadamardFusedWidth        = 5120;
inline constexpr std::int32_t kGatedRmsnormHadamardFusedHeadDim = 128;
inline constexpr std::int32_t kGatedRmsnormHadamardFusedRows    = 16;

void hadamard_transform_launch(const Tensor& x, const Tensor& signs, bool inverse, Tensor& out,
                               cudaStream_t stream);
void silu_mul_hadamard_launch(const Tensor& plane, const Tensor& signs, Tensor& out,
                              cudaStream_t stream);
void rmsnorm_hadamard_5120_launch(const Tensor& x, const Tensor& weight, float eps,
                                  bool unit_offset, const Tensor& signs, Tensor& out,
                                  cudaStream_t stream);
void gated_rmsnorm_hadamard_d128_launch(const Tensor& x, const Tensor& weight, const Tensor& z,
                                        float eps, const Tensor& signs,
                                        std::int32_t heads_per_column, Tensor& out,
                                        cudaStream_t stream);
void sigmoid_mul_hadamard_launch(const Tensor& gate, const Tensor& x, const Tensor& signs,
                                 Tensor& out, cudaStream_t stream);
void embedding_rotated_t2_launch(const Tensor& ids, const Weight& table, const Tensor& signs,
                                 Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
