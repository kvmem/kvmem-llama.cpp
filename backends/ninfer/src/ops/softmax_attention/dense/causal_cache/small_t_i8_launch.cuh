#pragma once

// Definition behind small_t_i8_launch.h; included only by the per-width small_t_i8_w<N>.cu units.

#include "core/device.h" // CUDA_CHECK
#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"
#include "ops/common/device_route.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.h"

#include <cstdlib>
#include <string_view>

namespace ninfer::ops::detail {
namespace small_t_i8 {

// Sums each key tile's PV products in FP16 where the device profile's "attn_pv_f16" says "on"
// (GeForce parts accumulate FP16 at twice their FP32-accumulator rate); NINFER_SMALLT_PV_F16=0|1
// overrides the profile.
inline bool small_t_pv_f16() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_SMALLT_PV_F16");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pv_f16", 1) == "on";
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, std::int32_t wave_splits, Tensor& partial_acc,
                          Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    // A U8 value plane is the packed signed int4 coding (rk8v4 and the packed-key storages); a
    // U8 key plane is a packed key coding, told apart by storage.
    const bool packed_values = cache_v.dtype == DType::U8;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena,
                      int QkSplit = 1, bool EarlyFetch = false>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0u;
        auto issue_pv = [&]<bool PackedValues, KvKeyCoding Keys, bool PvF16>() {
            if constexpr (DynamicArena) {
                configure_cuda_device_once([&] {
                    return cudaFuncSetAttribute(
                        causal_attention_small_t_i8_tiled_kernel<
                            Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena,
                            MultiBatch, Masked, CacheInput, PackedValues, Keys, PvF16, QkSplit,
                            EarlyFetch>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
                });
            }
            causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     MultiBatch, Masked, CacheInput, PackedValues,
                                                     Keys, PvF16, QkSplit, EarlyFetch>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, wave_splits, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
        };
        auto issue = [&]<bool PackedValues, KvKeyCoding Keys>() {
            if (small_t_pv_f16()) {
                issue_pv.template operator()<PackedValues, Keys, true>();
            } else {
                issue_pv.template operator()<PackedValues, Keys, false>();
            }
        };
        if (cache.storage == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
            issue.template operator()<true, KvKeyCoding::Lloyd4>();
        } else if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
            issue.template operator()<true, KvKeyCoding::Int4E8>();
        } else if (cache.storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
            issue.template operator()<true, KvKeyCoding::RootE8>();
        } else if (packed_values) {
            issue.template operator()<true, KvKeyCoding::Int8>();
        } else {
            issue.template operator()<false, KvKeyCoding::Int8>();
        }
    };
    // A device profile names the tier per window: "<warps>x<CTAs per SM>x<key block>", with a
    // trailing "d" for the dynamic-arena variant, "q" for split QK and "e" for EarlyFetch.
    const std::string_view routed = device_route_schedule(
        small_t_i8_route_key(Geometry::QHeads, cache.storage, TokenTile), implementation_window);
    if (!routed.empty()) {
        constexpr int kRowTiles = (TokenTile * Geometry::GroupSize + 15) / 16;
        bool taken              = true;
        if constexpr (kRowTiles == 1) {
            if (routed == "2x4x32") {
                launch.template operator()<2, 4, 32, false>();
            } else if (routed == "4x2x32") {
                launch.template operator()<4, 2, 32, false>();
            } else if (routed == "8x2x32") {
                launch.template operator()<8, 2, 32, false>();
            } else if (routed == "16x1x32") {
                launch.template operator()<16, 1, 32, false>();
            } else if (routed == "2x2x32q") {
                launch.template operator()<2, 2, 32, true, 2>();
            } else if (routed == "4x2x32q") {
                launch.template operator()<4, 2, 32, true, 2>();
            } else if (routed == "4x2x32e") {
                launch.template operator()<4, 2, 32, false, 1, true>();
            } else if (routed == "8x2x32e") {
                launch.template operator()<8, 2, 32, false, 1, true>();
            } else if (routed == "16x1x32e") {
                launch.template operator()<16, 1, 32, false, 1, true>();
            } else if (routed == "4x2x32qe") {
                launch.template operator()<4, 2, 32, true, 2, true>();
            } else {
                taken = false;
            }
        } else if constexpr (kRowTiles == 2) {
            if (routed == "4x2x32") {
                launch.template operator()<4, 2, 32, false>();
            } else if (routed == "8x2x32") {
                launch.template operator()<8, 2, 32, false>();
            } else if (routed == "16x1x32") {
                launch.template operator()<16, 1, 32, false>();
            } else if (routed == "32x1x32") {
                launch.template operator()<32, 1, 32, false>();
            } else if (routed == "4x2x32q") {
                launch.template operator()<4, 2, 32, true, 2>();
            } else if (routed == "8x2x32q") {
                launch.template operator()<8, 2, 32, true, 2>();
            } else if (routed == "4x2x32e") {
                launch.template operator()<4, 2, 32, false, 1, true>();
            } else if (routed == "8x2x32e") {
                launch.template operator()<8, 2, 32, false, 1, true>();
            } else if (routed == "16x1x32e") {
                launch.template operator()<16, 1, 32, false, 1, true>();
            } else if (routed == "4x2x32qe") {
                launch.template operator()<4, 2, 32, true, 2, true>();
            } else if (routed == "8x2x32qe") {
                launch.template operator()<8, 2, 32, true, 2, true>();
            } else {
                taken = false;
            }
        } else {
            if (routed == "6x2x32") {
                launch.template operator()<6, 2, 32, false>();
            } else if (routed == "12x1x32") {
                launch.template operator()<12, 1, 32, false>();
            } else if (routed == "12x1x64d") {
                launch.template operator()<12, 1, 64, true>();
            } else if (routed == "24x1x32") {
                launch.template operator()<24, 1, 32, false>();
            } else if (routed == "6x2x32q") {
                launch.template operator()<6, 2, 32, true, 2>();
            } else if (routed == "12x1x32q") {
                launch.template operator()<12, 1, 32, true, 2>();
            } else if (routed == "6x2x32e") {
                launch.template operator()<6, 2, 32, false, 1, true>();
            } else if (routed == "12x1x32e") {
                launch.template operator()<12, 1, 32, false, 1, true>();
            } else if (routed == "6x2x32qe") {
                launch.template operator()<6, 2, 32, true, 2, true>();
            } else if (routed == "12x1x32qe") {
                launch.template operator()<12, 1, 32, true, 2, true>();
            } else {
                taken = false;
            }
        }
        if (taken) {
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace small_t_i8

template <typename Geometry, int TokenTile, typename CacheInput>
void launch_small_t_i8(bool multi_batch, bool masked, const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                       PagedKVBatchLayerView cache, const CausalSmallTInvocation& invocation,
                       std::int32_t logical_capacity, std::int32_t implementation_window,
                       std::int32_t splits, std::int32_t wave_splits, Tensor& partial_acc,
                       Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    const auto run = [&]<bool MultiBatch, bool Masked>() {
        small_t_i8::launch_tc_partial_i8<Geometry, TokenTile, MultiBatch, Masked>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, wave_splits, partial_acc, partial_m, partial_l, stream);
    };
    if (multi_batch) {
        if (masked) {
            run.template operator()<true, true>();
        } else {
            run.template operator()<true, false>();
        }
    } else if (masked) {
        run.template operator()<false, true>();
    } else {
        run.template operator()<false, false>();
    }
}

} // namespace ninfer::ops::detail
