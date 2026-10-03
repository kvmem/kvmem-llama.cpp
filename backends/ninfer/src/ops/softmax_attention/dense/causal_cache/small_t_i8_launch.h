#pragma once

// The INT8-family small-T partial launch, instantiated per query width in small_t_i8_w<N>.cu so
// the kernel's many specializations compile in parallel instead of in one translation unit.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace ninfer::ops::detail {

// Device-profile key of the INT8-family partial kernel's launch tier:
// "attn_i8_small/h24/rk8v4/w8". The axis is the launch's implementation window.
std::string small_t_i8_route_key(std::int32_t q_heads, KvCacheStorage storage, std::int32_t width);

// The CTAs per SM of a profile-routed tier at this window, or zero when the profile routes none.
int small_t_i8_routed_ctas_per_sm(std::int32_t q_heads, KvCacheStorage storage, std::int32_t width,
                                  std::int32_t window);

template <typename Geometry, int TokenTile, typename CacheInput>
void launch_small_t_i8(bool multi_batch, bool masked, const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                       PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                       std::int32_t logical_capacity, std::int32_t implementation_window,
                       std::int32_t splits, std::int32_t wave_splits, Tensor& partial_acc,
                       Tensor& partial_m, Tensor& partial_l, cudaStream_t stream);

} // namespace ninfer::ops::detail
