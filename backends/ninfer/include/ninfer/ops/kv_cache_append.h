#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Host launch-resource promise for device-selected prefix append. It bounds every device count
 * during capture/replay but neither selects nor publishes a committed frontier.
 */
struct KVCacheAppendPrefixExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Append every K/V row to single-sequence paged growing-cache storage.
 *
 * k/v are contiguous BF16 [256,4|2,T] and positions is contiguous sequential device I32 [T].
 * BF16 cache mode copies K and V bit-for-bit; both planes are BF16. INT8-G64 cache rows
 * use one scale for each contiguous 64-value group. For codec input values x, the persistent
 * INT8 group encoding is
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s.
 *
 * FP8_E4M3FN cache rows use one FP16 scale for the complete D256 row:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   a == 0: scale_bits=FP16(+0), code[i]=E4M3FN(+0)
 *   a != 0: raw_scale  = a / 448
 *           scale_bits = FP16_RNE(clamp(raw_scale, 0x1p-24, 65504))
 *           s          = FP32(scale_bits)
 *           inv        = FP32(1 / s)
 *           code[i]    = E4M3FN_RNE_SATFINITE(FP32(x[i]) * inv)
 *   decode[i] = FP32(E4M3FN(code[i])) * s.
 *
 * A cache whose value plane is DType::U8 selects the rk8v4 profile: keys keep the INT8-G64
 * encoding above and values are stored as signed 4-bit codes, two per byte, in a value plane of
 * half the head dimension. Dimension d occupies the low nibble of byte d/2 when d is even and the
 * high nibble when d is odd. Values use a 32-value group rather than the key plane's 64, so their
 * scale plane has twice the key plane's leading extent: four bits resolve a group to 15 levels, so
 * one outlier would otherwise set the step for 64 neighbours. The group encoding is the INT8 one
 * with 7 in place of 127, over that 32-value group:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 7)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I4(clamp(RNE_even(FP32(x[i]) * inv), -7, 7))
 *   decode[i]  = FP32(code[i]) * s.
 *
 * A cache whose key plane is also DType::U8 (RotatedLloyd4KeyInt4Value, rk4v4) keeps the rk8v4
 * value plane and stores each key dimension as a 4-bit index, packed with the same nibble order,
 * over the INT8-G64 key scale plane. Index = sign << 3 | m, with m in 0..7 selecting the Lloyd-Max
 * level for N(0,1) {0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326}, represented
 * by the fixed INT8 code C[m] = {6, 18, 31, 44, 58, 75, 96, 127}. For one G64 group, with lane l
 * holding x[l] and x[l+32] and every sum formed by single-rounded FP32 additions of those per-lane
 * partials in xor-butterfly order (offsets 16, 8, 4, 2, 1):
 *
 *   r          = FP32_SQRT_RN(sum(x0*x0 + x1*x1) * (1/64))
 *   inv        = r > 0 ? FP32(1 / r) : 0
 *   m[i]       = count of boundaries {0.25825, 0.52245, 0.79960, 1.09930, 1.43715, 1.84355,
 *                2.40080} <= abs(FP32(x[i]) * inv)
 *   index[i]   = (x[i] < 0 ? 8 : 0) | m[i],  c[i] = (x[i] < 0 ? -1 : 1) * C[m[i]]
 *   scale_bits = FP16_RNE(r > 0 ? sum(x0*c0 + x1*c1) / sum(c0*c0 + c1*c1) : 0)
 *   decode[i]  = FP32(c[i]) * FP32(scale_bits).
 *
 * The indices are chosen against the group RMS and the scale is the least-squares fit of the
 * chosen codes, so the decoded key is an ordinary INT8-G64 key and the INT8 QK path consumes it.
 *
 * KvCacheStorage::RotatedInt4KeyInt4ValueE8 (rk4v4-e8) has the same U8 plane shapes as rk4v4 and
 * is told apart by storage. Values are the rk8v4 packed coding above. Keys take the same rotation
 * and G64 group as INT8-G64, with scale_bits = FP16_RNE(a / 7), and each block of eight
 * consecutive dimensions y = FP32(x) * inv is snapped to the nearest E8 lattice point: the nearer,
 * by squared distance summed in FP32, of D8(y) and D8(y - 1/2) + 1/2, D8 on ties. D8 rounds each
 * coordinate to nearest-even and, when the rounded sum is odd, moves the coordinate with the
 * largest rounding error (lowest index on ties) one step towards y. The stored key code is
 * I4(clamp(RNE_even(point), -8, 7)), packed like values; the coset bit is not stored.
 *
 * KvCacheStorage::RotatedE8RootKeyInt4Value (rk2v4-e8) is selected by storage as well. Values are
 * the rk8v4 packed coding. Keys take the same rotation, G64 group and scale_bits = FP16_RNE(a / 7);
 * each block of eight consecutive rotated dimensions b is stored as two bytes at row byte 2*block:
 * the E8 root nearest to b/|b| (a pair root +-e_i+-e_j or a half root 1/2(+-1)^8 with an even
 * number of minus signs, by the larger of their scores |u_i|+|u_j| and sum|u|/2 less the smallest
 * |u| when the sign count is odd), then a byte holding the radius index -- the nearest integer to
 * 3*log2(r)+8 in [1,15] for r = |b| / (s*sqrt(8)), zero for r < 0.08 -- over the residual axis,
 * the largest coordinate of u less its projection on the root, with its sign. The exact operation
 * order is ops/kv_cache/e8_root_codec.cuh's; decode yields eight int8 codes times s.
 *
 * V uses represented BF16 source values directly as x under every profile, including rk8v4: values
 * are never rotated, so no inverse preparation is applied to the attention output. For every
 * quantized profile, K is a paired physical representation for causal Attention: its
 * implementation-owned fixed orthogonal preparation selects x, and the causal consumer applies the
 * matching private Q preparation. The transform and raw K code/scale bytes are not standalone
 * mathematical outputs. Standalone and
 * fused append produce the same consumable K representation. Every addressed code/value and scale
 * is overwritten, and no unrelated cache row is read or written. Inputs and every cache
 * plane/table are pairwise non-overlapping. The Op owns no persistent allocation, frontier,
 * request identity, or commit authority.
 *
 * NVFP4-G16 and K8V4 KV-cache profiles are also ported on this fork: both planes use the e2m1
 * codec (Nvfp4Group16) or an FP8 key paired with an e2m1 value plane (Fp8KeyNvfp4Value), each with
 * a raw E4M3-byte group-16 scale on their e2m1 plane(s). See d256_kv_cache_profile's KvCacheStorage
 * overload for the exact per-plane encoding.
 */
void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, cudaStream_t stream);

/**
 * Append device-selected BF16 prefixes to batched paged growing-cache storage.
 *
 * k/v are contiguous BF16 [128,8,T,B], positions is contiguous device I32 [T,B], and counts and
 * table_rows are contiguous device I32 [B]. For row b and i in [0,counts[b]), k/v[:, :, i, b]
 * store K and V bit-for-bit, both BF16, at logical position positions[i,b] through
 * table row table_rows[b]. The paged planes use head-major order [128,64,Nphysical,8]. No byte
 * belonging only to the rejected physical tail [counts[b],T) is written. Inputs are unchanged,
 * and the Op neither decides nor publishes a committed frontier.
 *
 * The caller guarantees T>0, B=1..8, envelope.min_count <= counts[b] <= envelope.max_count <= T,
 * sequential nonnegative live positions, materialized table entries for every position allowed by
 * the envelope, and pairwise non-aliasing of inputs and cache storage.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, cudaStream_t stream);

/**
 * Append device-selected BF16 prefixes to lane-owned cyclic storage.
 *
 * k/v, positions, counts, and their storage-conversion and mutation contracts match the paged
 * overload; lanes[b] selects the destination lane. The registered profiles have fixed geometry
 * D=128, Hkv=8 and capacity 2048 or 4096. Absolute position p maps to slot p mod capacity.
 * For a nonempty prefix, the caller guarantees that the row's existing live interval ends
 * immediately before positions[0,b]. Advancing it by counts[b] makes every overwritten old slot
 * dead, and one row commits at most the ring capacity. A zero count reads no positions or K/V
 * and changes no cache bytes. Physical T is independent of the committed prefix length.
 * Consequently, no two live writes race for one physical slot. The Op does not own or publish the
 * lane frontier.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
