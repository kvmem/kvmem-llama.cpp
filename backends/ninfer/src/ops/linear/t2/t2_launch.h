#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using T2Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_t2_gemv_r8_w1_k5120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_r8_w1_k6144(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_gemv_r8_w1_k17408(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_t2_gemv_r4_w1_word(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_small_t_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_small_t_v2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_mma_r64_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_t2_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
