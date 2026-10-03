#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

/**
 * YaRN, Qwen's way past a model's native window (Peng et al.; Hugging Face's `yarn` rope type with
 * beta_fast=32 and beta_slow=1). With s=factor, L=native_context and R=rotary_dim, the correction
 * range [lo,hi] holds the pairs whose wavelength 2*pi/theta^(-2i/R) fits between 32 times and once
 * into L, rounded outward; pair i takes
 *
 *   inv_freq[i] = theta^(-2i/R) * ((1 - ramp(i)) + ramp(i)/s),  ramp(i) = clamp((i-lo)/(hi-lo), 0,
 * 1),
 *
 * and cos and sin both carry the attention factor 0.1*ln(s)+1. factor<=1 leaves RoPE unscaled.
 *
 * Independently, linear position interpolation past a threshold T with factor f rotates a position
 * p > T at T + (p - T) / f, in floating point, so positions up to T keep their exact angles.
 * interpolation_factor<=1 leaves positions unscaled. Text 1-D and Text MRoPE at D256/R64 only.
 */
struct RopeYarn {
    float factor                          = 1.0F;
    std::uint32_t native_context          = 0;
    float interpolation_factor            = 1.0F;
    std::uint32_t interpolation_threshold = 0;

    [[nodiscard]] bool active() const noexcept {
        return factor > 1.0F || interpolation_factor > 1.0F;
    }
};

void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& q,
          Tensor& k, cudaStream_t stream);

void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& x,
          cudaStream_t stream);

} // namespace ninfer::ops
