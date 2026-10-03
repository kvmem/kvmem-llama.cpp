// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/device_route.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {
static_assert(kPromptWaveRows == CausalPromptI8FastShape<8>::Br);
static_assert(kPromptWaveRows % CausalPromptI8FastShape<4>::Br == 0);
static_assert(kPromptWaveRows % kCausalPromptI8Br == 0);
static_assert(kPromptWaveRows % kCausalPromptBr == 0);

namespace {

// Sums each key tile's PV products in FP16 on the packed-value INT8 prompt kernels where the
// device profile's "attn_pv_f16" says "on"; NINFER_PROMPT_PV_F16=0|1 overrides the profile.
bool prompt_pv_f16() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_PV_F16");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pv_f16", 1) == "on";
}

// Packs one KV head's query heads into each INT8 prompt tile (causal_attention_prompt_i8_kernel's
// PackGqa) where the device profile's "attn_pack_gqa" says "on"; NINFER_PROMPT_PACK_GQA=0|1
// overrides the profile. The fast kernel already issues a row block's heads together.
bool prompt_pack_gqa() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_PACK_GQA");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pack_gqa", 1) == "on";
}

// Both fast INT8 variants run one CTA per SM and every CTA of a launch sweeps a similar key range,
// so a launch costs about (waves) x (one CTA's sweep). A four-warp CTA sweeps in about 0.72 of an
// eight-warp CTA's time (measured on RTX 5090 at 64K context) but covers half the rows.
bool causal_attention_prompt_i8_fast_prefers_narrow(std::int32_t tokens, std::int32_t q_heads) {
    static const int multiprocessors = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    const auto waves = [&](int rows) {
        return div_up(div_up(tokens, rows) * q_heads, multiprocessors);
    };
    constexpr int NarrowCostPercent = 72;
    return waves(CausalPromptI8FastShape<4>::Br) * NarrowCostPercent <
           waves(CausalPromptI8FastShape<8>::Br) * 100;
}

// The int8 cache and rk8v4 share the fast kernel: their keys are the same INT8 G64 codes, and
// PackedValues decodes rk8v4's packed int4 value plane; Keys expands the packed key codings.
template <typename Geometry, bool PackedValues, KvKeyCoding Keys = KvKeyCoding::Int8,
          typename CacheView, typename Metadata>
void causal_attention_prompt_i8_fast_launch_for(const Tensor& q, const Tensor& positions,
                                                float scale, const CacheView& cache,
                                                Metadata metadata, Tensor& out,
                                                cudaStream_t stream) {
    configure_cuda_device_once([] {
        return cudaFuncSetAttribute(
            causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 8, PackedValues, Keys>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            CausalPromptI8FastShape<8, PackedValues>::SmemBytes);
    });
    configure_cuda_device_once([] {
        return cudaFuncSetAttribute(
            causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, 4, PackedValues, Keys>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            CausalPromptI8FastShape<4, PackedValues>::SmemBytes);
    });

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const auto launch = [&]<int Warps>() {
        using Shape = CausalPromptI8FastShape<Warps, PackedValues>;
        const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                        static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, Warps, PackedValues, Keys>
            <<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache.k_pages.data),
                static_cast<const std::int8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    };
    if (causal_attention_prompt_i8_fast_prefers_narrow(tokens, Geometry::QHeads)) {
        launch.template operator()<4>();
    } else {
        launch.template operator()<8>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out, bool fast,
                                                  cudaStream_t stream) {
    fast = causal_attention_prompt_fast_kernel(cache.storage, fast);
    if (fast && cache.storage == KvCacheStorage::Int8Group64) {
        causal_attention_prompt_i8_fast_launch_for<Geometry, false>(q, positions, scale, cache,
                                                                    metadata, out, stream);
        return;
    }
    if (fast && cache.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        causal_attention_prompt_i8_fast_launch_for<Geometry, true>(q, positions, scale, cache,
                                                                   metadata, out, stream);
        return;
    }
    if (fast && cache.storage == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
        causal_attention_prompt_i8_fast_launch_for<Geometry, true, KvKeyCoding::Lloyd4>(
            q, positions, scale, cache, metadata, out, stream);
        return;
    }
    if (fast && cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
        causal_attention_prompt_i8_fast_launch_for<Geometry, true, KvKeyCoding::Int4E8>(
            q, positions, scale, cache, metadata, out, stream);
        return;
    }
    if (fast && cache.storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
        causal_attention_prompt_i8_fast_launch_for<Geometry, true, KvKeyCoding::RootE8>(
            q, positions, scale, cache, metadata, out, stream);
        return;
    }
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    const auto tokens     = static_cast<std::int32_t>(q.ne[2]);
    if (kv_cache_is_int8_family(cache.storage)) {
        // The INT8 family is the one place that deliberately does not take its plane types from
        // ops/kv_cache/plane_types.h. One kernel serves both codings, so its value parameter is
        // std::int8_t* for int8-g64 and for rk8v4 alike, and the packed-int4 path re-casts to
        // std::uint8_t where it unpacks (causal_prompt_i4_dequant_f16x8). Substituting
        // KvValueCodeT<RotatedInt8KeyInt4ValueGroup64>, which is U8 because the profile describes
        // the plane's storage rather than this kernel's signature, would not be a cleanup. Leave
        // it; the dtype check below is the guard that matters here.
        const bool pack_gqa  = Geometry::GroupSize > 1 && prompt_pack_gqa();
        const auto launch_i8 = [&]<bool PackedValues, KvKeyCoding Keys, bool PvF16>() {
            const auto launch = [&]<bool PackGqa>() {
                const auto kernel =
                    causal_attention_prompt_i8_kernel<Geometry, Metadata, PackedValues, Keys, PvF16,
                                                      PackGqa>;
                // The kernel exceeds the default 48 KiB dynamic-smem ceiling.
                configure_cuda_device_once([&] {
                    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                kCausalPromptI8SmemBytes);
                });
                const dim3 grid =
                    PackGqa ? dim3(static_cast<unsigned>(
                                       div_up(tokens * Geometry::GroupSize, kCausalPromptI8Br)),
                                   static_cast<unsigned>(Geometry::KVHeads), 1u)
                            : dim3(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                   static_cast<unsigned>(Geometry::QHeads), 1u);
                kernel<<<grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::int8_t*>(cache_k.data),
                    static_cast<const std::int8_t*>(cache_v.data),
                    static_cast<const __half*>(cache.k_scale_pages.data),
                    static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
            };
            if (pack_gqa) {
                launch.template operator()<true>();
            } else {
                launch.template operator()<false>();
            }
        };
        // A U8 key plane is a packed key coding (rk4v4, rk4v4-e8 or rk2v4-e8), told apart by
        // storage; a U8 value plane is the packed signed int4 coding they share with rk8v4.
        const bool pv_f16 = prompt_pv_f16();
        if (cache.storage == KvCacheStorage::RotatedLloyd4KeyInt4Value && pv_f16) {
            launch_i8.template operator()<true, KvKeyCoding::Lloyd4, true>();
        } else if (cache.storage == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
            launch_i8.template operator()<true, KvKeyCoding::Lloyd4, false>();
        } else if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 && pv_f16) {
            launch_i8.template operator()<true, KvKeyCoding::Int4E8, true>();
        } else if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
            launch_i8.template operator()<true, KvKeyCoding::Int4E8, false>();
        } else if (cache.storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
            launch_i8.template operator()<true, KvKeyCoding::RootE8, false>();
        } else if (cache_v.dtype == DType::U8 && pv_f16) {
            launch_i8.template operator()<true, KvKeyCoding::Int8, true>();
        } else if (cache_v.dtype == DType::U8) {
            launch_i8.template operator()<true, KvKeyCoding::Int8, false>();
        } else {
            launch_i8.template operator()<false, KvKeyCoding::Int8, false>();
        }
    } else {
        // The kernel exceeds the default 48 KiB dynamic-smem ceiling.
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kCausalPromptSmemBytes);
        });
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested) {
    const bool supported = storage == KvCacheStorage::Int8Group64 ||
                           storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
                           storage == KvCacheStorage::RotatedLloyd4KeyInt4Value ||
                           storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
                           storage == KvCacheStorage::RotatedE8RootKeyInt4Value;
    if (!supported) { return false; }
    if (requested) { return true; }
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_FAST");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_prompt_fast", 1) == "on";
}

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out, bool fast,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, fast, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, fast, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, bool fast,
                                    cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, fast, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, fast, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
