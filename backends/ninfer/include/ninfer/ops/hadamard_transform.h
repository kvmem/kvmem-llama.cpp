#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: hadamard_transform
 *
 * Math / indexing:
 *   For every column t and every 1024-block b of the fastest axis, with i, j in [0,1024):
 *     forward: out[b*1024+i, t] = 2^-5 * sum_j H[i][j] * signs[b*1024+j] * x[b*1024+j, t]
 *     inverse: out[b*1024+i, t] = 2^-5 * signs[b*1024+i] * sum_j H[i][j] * x[b*1024+j, t]
 *   where H[i][j] = (-1)^popcount(i & j) is the order-1024 Sylvester Walsh-Hadamard matrix.
 *   The forward form is the activation-side transform of a Hadamard-rotated checkpoint
 *   (`y = W_r · FWHT(s ⊙ a)`); the inverse form restores a rotated table row to the primal
 *   basis (`e = s ⊙ FWHT(z)`). H is symmetric and 2^-5 H is orthonormal, so the two forms are
 *   mutually inverse up to rounding.
 *
 * Logical shapes / supported domain:
 *   x and out are same-shaped contiguous BF16 tensors whose fastest extent K = ne[0] is a
 *   positive multiple of 1024; every other extent is a batch axis (T = numel / K columns).
 *   signs is contiguous BF16 [K] holding +1 or -1. The registered checkpoint uses
 *   K in {5120, 6144, 17408}; the Op accepts every positive multiple of 1024. x, signs and out
 *   are 16-byte aligned (the lanes load and store eight elements at a time).
 *
 * Numeric:
 *   The oracle evaluates the definition naively in FP64 from the represented BF16 inputs. The
 *   BF16 output is promoted and compared directly with that result; output storage rounding
 *   belongs to the Op's named reduction criterion, not the oracle. Butterfly association and
 *   accumulator precision are implementation choices.
 *
 * Effects:
 *   Writes all of out. out may alias x exactly (in place); otherwise the two must not overlap.
 *   x (when not aliased) and signs are preserved.
 *
 * Workspace:
 *   None. No persistent state side effect.
 */
void hadamard_transform(const Tensor& x, const Tensor& signs, bool inverse, Tensor& out,
                        cudaStream_t stream);

/**
 * Op: silu_mul_hadamard
 *
 * Math / indexing:
 *   With gate = plane[0:I, t] and up = plane[I:2I, t] for every column t, and a = the BF16
 *   rounding of silu(gate) * up (silu(x) = x / (1 + e^-x), exact fp32):
 *     out[b*1024+i, t] = 2^-5 * sum_j H[i][j] * signs[b*1024+j] * a[b*1024+j, t]
 *   i.e. the forward hadamard_transform of silu_mul's output, the down-projection input of a
 *   Hadamard-rotated checkpoint, in one pass over the SwiGLU plane.
 *
 * Logical shapes / supported domain:
 *   plane is contiguous BF16 [2*I, T...], out contiguous BF16 [I, T...] with I = out.ne[0] a
 *   positive multiple of 1024 and the batch extents equal; signs is contiguous BF16 [I] of
 *   +1 / -1. All three are 16-byte aligned. The registered checkpoint uses I = 17408.
 *
 * Numeric:
 *   The oracle is the composition's definition in FP64 from the represented BF16 inputs with the
 *   intermediate SwiGLU value rounded to BF16; the same reduction criterion as hadamard_transform.
 *
 * Effects:
 *   Writes all of out; out must not overlap the plane. plane and signs are preserved.
 *
 * Workspace:
 *   None. No persistent state side effect.
 */
void silu_mul_hadamard(const Tensor& plane, const Tensor& signs, Tensor& out, cudaStream_t stream);


/**
 * Producers of a Hadamard-rotated projection input
 *
 * Each op below is the named op followed by the forward hadamard_transform of its output with
 * `signs` (K = signs.ne[0], a multiple of 1024, is the projection's input width), fused where a
 * route exists and composed otherwise. Both forms round the op's output to BF16 before the
 * transform, so the result is bit-identical to running the two ops in sequence.
 *
 *   rmsnorm_hadamard:       rmsnorm over rows of K = x.ne[0], then the transform of each row.
 *   gated_rmsnorm_hadamard: gated_rmsnorm over rows of D = x.ne[0], then the transform of each
 *                           K-wide column of consecutive rows (K a multiple of D).
 *   sigmoid_mul_hadamard:   the BF16 product x * sigmoid(gate), then the transform of each
 *                           K-wide column; out may alias x exactly, never gate.
 *
 * Domain: the named op's own, plus contiguous 16-byte aligned signs and out holding whole
 * K-wide columns (out has x's element count). Fused routes: rmsnorm at K = 5120; gated_rmsnorm at
 * D = 128 with K / D a multiple of sixteen.
 *
 * Numeric: the named op's rounding followed by hadamard_transform's reduction criterion.
 *
 * Effects: writes all of out; inputs and signs are preserved (x only when out does not alias it).
 * No workspace or persistent state.
 */
void rmsnorm_hadamard(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                      const Tensor& signs, Tensor& out, cudaStream_t stream);
void gated_rmsnorm_hadamard(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                            const Tensor& signs, Tensor& out, cudaStream_t stream);
void sigmoid_mul_hadamard(const Tensor& gate, const Tensor& x, const Tensor& signs, Tensor& out,
                          cudaStream_t stream);


/**
 * Op: embedding_rotated
 *
 * Gathers rows of a Hadamard-rotated T2 table and restores them to the primal basis:
 *   out[b*1024+i, t] = 2^-5 * signs[b*1024+i] * sum_j H[i][j] * table[ids[t], b*1024+j]
 * i.e. the inverse hadamard_transform of each dequantised row. `table` is a row-split
 * t2_g128_fp16 [vocab, K] weight, `signs` BF16 [K] of +1 / -1 (K a multiple of 1024), `ids` I32 [T]
 * with every id in [0, vocab), `out` BF16 [K, T]; signs and out are 16-byte aligned. The
 * dequantised values enter the transform unrounded, so out carries one BF16 rounding. No workspace
 * or persistent state.
 */
void embedding_rotated(const Tensor& ids, const Weight& table, const Tensor& signs, Tensor& out,
                       cudaStream_t stream);

} // namespace ninfer::ops
