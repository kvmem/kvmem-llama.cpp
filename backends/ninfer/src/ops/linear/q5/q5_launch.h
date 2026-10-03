#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q5Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q5_gemv_r16_s2_x(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_split4_c1_k5120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_split4_c1_k6144(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_split4_c1_k17408(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c32_s3(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r32_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

// Unified-template launches (ops/linear/common/route_table.h selects their table).
void launch_q5_a16_direct_r1_t1_w4_k17408(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t1_w4_k5120(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t1_w4_k6144(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t2_w2_k17408(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t2_w2_k6144(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t2_w4_k5120(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t3_w2_k17408(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t3_w2_k6144(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t3_w4_k5120(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r1_t4_w2_k17408(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r2_t4_w2_g8_b4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_direct_r2_t4_w4_g4_b4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_mma_r32_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_mma_r64_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_mma_r64_t96_k128_s1_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r16_t16_w2_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r16_t16_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r16_t24_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r16_t32_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r16_t8_capacity4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r16_t8_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r32_t16_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r32_t24_w4_s2_pairwise(const Tensor&, const Weight&, Tensor&,
                                                 cudaStream_t);
void launch_q5_a16_sliced_r32_t32_w2_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r32_t32_w4_s1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r32_t32_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r32_t64_w2_s1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q5_a16_sliced_r32_t64_w2_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
