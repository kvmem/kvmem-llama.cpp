#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Computes four independent linear projections for each token:
 *
 *   q[:,t]    = linear(x[:,t], query_key_weight[0:6144,:])
 *   k[:,t]    = linear(x[:,t], query_key_weight[6144:7168,:])
 *   gate[:,t] = linear(x[:,t], gate_value_weight[0:6144,:])
 *   v[:,t]    = linear(x[:,t], gate_value_weight[6144:7168,:]).
 *
 * All tensors are contiguous BF16. Shapes are x [5120,T], q/gate [6144,T], and k/v [1024,T].
 * T may be any positive value.
 * The two parent weights are RowSplit [7168,5120] with FP16 scales and group size 64:
 * query_key is Q4_G64_FP16 and gate_value is Q5_G64_FP16. The oracle exact-decodes each row and
 * evaluates every projection naively in FP64 from the represented inputs. The BF16 outputs are
 * promoted and compared directly with those ideal values; final output storage rounding belongs
 * to AttnInputProj's named A16 criterion, not the oracle. Production routes choose their private
 * accumulator and staging precision. Inputs and the four outputs must be mutually non-overlapping.
 * Current registered routes require no transient allocation. The Op has no persistent state side
 * effect.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream);

/**
 * Policy-bearing split form. AllowA8Int admits the integer-activation route for the registered
 * Q4 plus Q5 [7168,5120] pair at full prefill tiles (T a positive multiple of 128) and for the T2
 * [7168,5120] pair at 1..192 tokens (the small-T kernel) and above (padded inside); every
 * other width, shape or policy takes the A16 schedules. Size the transient storage with
 * attn_input_proj_split_workspace_capacity_bytes().
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t attn_input_proj_split_workspace_capacity_bytes(
    QType query_key_qtype, std::int32_t query_key_rows, QType gate_value_qtype,
    std::int32_t gate_value_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Computes the single-parent Q/K/output-gate/V projection.
 *
 * The parent stores rows in physical order query, key, output gate, value while the public output
 * argument order is q, gate, k, v. Every route writes the four independently contiguous final
 * allocations directly; no packed parent output is materialized. The NVFP4 A4 and FP8 A8
 * profiles may use caller-owned transient storage for their private quantized activation.
 *
 * Registered parent forms are:
 *
 * - Q8_G32_FP16 RowSplit `[9216,2048]`, with row counts `[4096,512,4096,512]`. `x` is
 *   BF16 `[2048,T]`, q/gate are BF16 `[4096,T]`, and k/v are BF16 `[512,T]`.
 * - BF16 Contiguous `[14336,5120]`, with row counts `[6144,1024,6144,1024]`. `x` is
 *   BF16 `[5120,T]`, q/gate are BF16 `[6144,T]`, and k/v are BF16 `[1024,T]`.
 * - NVFP4 BlockScaleK16M128x4 `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16.
 * - FP8_E4M3FN_ROW_BF16 RowScale `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16.
 *
 * `T` is the positive token extent of the Op contract. All three policies permit the BF16
 * and Q8_G32_FP16 A16 implementations. NVFP4 uses A16 under A16Only/AllowA8; AllowA4 permits the
 * resolver to select either a qualified A16 route or activation quantization to NVFP4 at every
 * positive T. FP8 accepts all policies at every positive T. AllowA8/AllowA4 permit the resolver to
 * choose a qualified A16 route or private activation quantization followed by A8 Tensor Core
 * computation. A16Only preserves the represented BF16 activation at every positive T; tile and
 * route cutoffs are private implementation choices, independent of speculative block width.
 *
 * The oracle evaluates every projection independently with naive FP64 accumulation from the
 * logical values represented by the persistent weight and BF16 activation. The final four BF16
 * stores belong to the Op's criterion for the selected activation-compute path.
 *
 * `workspace` is caller-owned call-scoped transient storage sized by
 * attn_input_proj_workspace_capacity_bytes(). It must not overlap the input, parent weight, or any
 * output. The Op does not allocate device memory internally.
 */
[[nodiscard]] std::size_t
attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                         std::int32_t input_rows, LinearPolicy policy,
                                         std::int32_t min_tokens, std::int32_t max_tokens);

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Whether the single-parent NVFP4 [14336,5120] projection takes a fused norm+projection route.
 * T==1 and T>=4 under AllowA4; T==2,3 deliberately stay unfused (a fused norm+SIMT variant
 * measured a regression), as do other policies and token counts.
 * This is a host-side route predicate. A true result does not validate the weight payload.
 */
[[nodiscard]] bool attn_input_proj_fused_rmsnorm_nvfp4_eligible(const Weight& weight,
                                                                LinearPolicy policy,
                                                                std::int32_t tokens);

/**
 * Computes the Offset RMSNorm and single-parent NVFP4 Q/K/output-gate/V projection together.
 * residual is contiguous BF16 [5120,T], norm_weight is contiguous BF16 [5120], and q/gate are
 * contiguous BF16 [6144,T] while k/v are contiguous BF16 [1024,T]. The parent must be NVFP4
 * BlockScaleK16M128x4 [14336,5120] under AllowA4. T==1 fuses the norm into the A16 GEMV
 * consumer, 4<=T<1024 into the RowMajor quantizer + MMA-band consumer, and T>=1024 into the
 * Tiled quantizer + TMA consumer; T==2,3 throw and use the unfused path. The input norm uses
 * rsqrtf(sum(x*x)/5120 + eps) and multiplies each element by norm_weight + 1, rounding to
 * BF16 before the consumer reads it. The exact-byte oracle is Offset rmsnorm() into BF16
 * storage, then the same-route ordinary projection. It uses the same caller-owned workspace
 * capacity as attn_input_proj_workspace_capacity_bytes() for this parent and T; the workspace
 * must not overlap any input, weight, or output. No hidden tensor is written.
 * The Op has no persistent state side effect.
 */
void attn_input_proj_fused_rmsnorm_nvfp4(const Tensor& residual, const Tensor& norm_weight,
                                         float eps, const Weight& projection_weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                                         WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Applies the A16-only single-parent Q/K/output-gate/V projection without transient workspace.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

/**
 * Three-output Q8 specialization. The Q8_G32_FP16 RowSplit parent stores rows in order
 * [query 4096, key 1024, value 1024]. Registered parent forms are [6144,2048] with BF16
 * x [2048,T] for the Qwen3.6 companion and [6144,5120] with BF16 x [5120,T] for DFlash2.
 * q is contiguous BF16 [4096,T], and k/v are contiguous BF16 [1024,T]. Every route writes the
 * three independent final allocations directly; no parent output or transient workspace is
 * materialized. T may be any positive value. Q and K remain raw projection outputs: this Op does
 * not normalize or rotate either tensor.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream);

/**
 * GGUF form: query, gate, key and value are parts of one or more GGUF parents (see
 * GgufProjectionWeights; outputs 0 = q, 1 = gate, 2 = k, 3 = v), each block type its own, projected
 * through the GGUF linear routes against one q8_1 quantization of x per activation layout.
 */
void attn_input_proj(const Tensor& x, const GgufProjectionWeights& weights, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& workspace,
                     cudaStream_t stream);

[[nodiscard]] std::size_t attn_input_proj_workspace_capacity_bytes(
    const GgufProjectionWeights& weights, std::int32_t min_tokens, std::int32_t max_tokens);

} // namespace ninfer::ops
