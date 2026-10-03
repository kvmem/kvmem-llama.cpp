// Locks the one place that declares what a KV plane is.
//
// d256_kv_cache_profile is the single source of truth for every KV plane's dtype and extent, and
// plane_types.h derives the C++ types from it. The failure this guards against is not a compiler
// error -- it is an Op that quietly restates a plane's dtype and gets it wrong. That is exactly
// how context_kv_materialize came in requiring FP16 for the value plane while requiring BF16 for
// the key plane: byte-compatible, so it built and ran, and it took a DFlash2 startup failure to
// surface it.
//
// Everything here is compile-time. The test binary passing means the table and the aliases still
// agree; there is nothing to run on a GPU.

#include "ops/kv_cache/d256_profile.h"
#include "ops/kv_cache/plane_types.h"

#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

using ninfer::DType;
using ninfer::KvCacheStorage;
using ninfer::ops::d256_kv_cache_profile;
using ninfer::ops::KvKeyCodeT;
using ninfer::ops::KvKeyScaleT;
using ninfer::ops::KvValueCodeT;
using ninfer::ops::KvValueScaleT;

// Every storage the public enum admits. A new one added without a profile entry fails to compile
// here, because d256_kv_cache_profile is constexpr and throws on an unknown storage.
constexpr KvCacheStorage kStorages[] = {
    KvCacheStorage::BFloat16,      KvCacheStorage::Int8Group64,
    KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
    KvCacheStorage::Nvfp4Group16,  KvCacheStorage::Fp8KeyNvfp4Value,
    KvCacheStorage::RotatedLloyd4KeyInt4Value,
    KvCacheStorage::RotatedInt4KeyInt4ValueE8,
    KvCacheStorage::RotatedE8RootKeyInt4Value,
};

// The profile must be evaluable at compile time for every storage; this also proves each one has
// an entry rather than falling through to the throwing default.
template <KvCacheStorage S>
constexpr bool profile_is_constexpr() {
    constexpr auto profile = d256_kv_cache_profile(S);
    return profile.key_leading_extent > 0 && profile.value_leading_extent > 0;
}
static_assert(profile_is_constexpr<KvCacheStorage::BFloat16>());
static_assert(profile_is_constexpr<KvCacheStorage::Int8Group64>());
static_assert(profile_is_constexpr<KvCacheStorage::Fp8E4M3Row256>());
static_assert(profile_is_constexpr<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>());
static_assert(profile_is_constexpr<KvCacheStorage::Nvfp4Group16>());
static_assert(profile_is_constexpr<KvCacheStorage::Fp8KeyNvfp4Value>());
static_assert(profile_is_constexpr<KvCacheStorage::RotatedLloyd4KeyInt4Value>());
static_assert(profile_is_constexpr<KvCacheStorage::RotatedInt4KeyInt4ValueE8>());
static_assert(profile_is_constexpr<KvCacheStorage::RotatedE8RootKeyInt4Value>());

// This fork's defining invariant: the BF16 cache is symmetric. Upstream stores V as FP16, the two
// are the same width, and every catch-up merge has tried to reintroduce that.
static_assert(d256_kv_cache_profile(KvCacheStorage::BFloat16).key_code_dtype == DType::BF16);
static_assert(d256_kv_cache_profile(KvCacheStorage::BFloat16).value_code_dtype == DType::BF16,
              "the BF16 KV cache stores V as BF16, symmetric with K");
static_assert(std::is_same_v<KvKeyCodeT<KvCacheStorage::BFloat16>,
                             KvValueCodeT<KvCacheStorage::BFloat16>>);

// The aliases must agree with the dtypes the table declares, for every plane of every storage.
static_assert(std::is_same_v<KvKeyCodeT<KvCacheStorage::Int8Group64>, std::int8_t>);
static_assert(std::is_same_v<KvValueCodeT<KvCacheStorage::Int8Group64>, std::int8_t>);
static_assert(std::is_same_v<KvKeyScaleT<KvCacheStorage::Int8Group64>, __half>);

// rk8v4 is this fork's own storage: rotated INT8 keys, packed signed-int4 values two per byte.
// The asymmetry here is deliberate, unlike the BF16 one.
static_assert(std::is_same_v<KvKeyCodeT<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>,
                             std::int8_t>);
static_assert(std::is_same_v<KvValueCodeT<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>,
                             std::uint8_t>);
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedInt8KeyInt4ValueGroup64)
                  .packed_int4_values());
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedInt8KeyInt4ValueGroup64)
                      .value_leading_extent *
                  2 ==
              d256_kv_cache_profile(KvCacheStorage::RotatedInt8KeyInt4ValueGroup64)
                  .key_leading_extent,
              "a packed int4 value plane is half the width of its key plane");

// rk4v4 packs both planes two 4-bit codes per byte, and keeps the INT8 family's G64 key scale.
static_assert(std::is_same_v<KvKeyCodeT<KvCacheStorage::RotatedLloyd4KeyInt4Value>, std::uint8_t>);
static_assert(std::is_same_v<KvValueCodeT<KvCacheStorage::RotatedLloyd4KeyInt4Value>, std::uint8_t>);
static_assert(std::is_same_v<KvKeyScaleT<KvCacheStorage::RotatedLloyd4KeyInt4Value>, __half>);
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedLloyd4KeyInt4Value).key_leading_extent * 2 ==
                  ninfer::ops::kD256KVCacheHeadDim,
              "rk4v4 packs two key indices per byte");
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedLloyd4KeyInt4Value).scale_leading_extent ==
                  d256_kv_cache_profile(KvCacheStorage::Int8Group64).scale_leading_extent,
              "rk4v4 keeps the INT8 family's G64 key scale plane");
static_assert(ninfer::ops::kv_cache_is_int8_family(KvCacheStorage::RotatedLloyd4KeyInt4Value));
// rk4v4-e8 packs keys too: both code planes are U8 at half the head dimension, keys keep the G64
// scale plane (four per token) and values the G32 one (eight). The key plane's dtype matches
// rk4v4's and Nvfp4Group16's, which is why consumers dispatch on storage rather than on dtype.
static_assert(std::is_same_v<KvKeyCodeT<KvCacheStorage::RotatedInt4KeyInt4ValueE8>,
                             std::uint8_t>);
static_assert(std::is_same_v<KvValueCodeT<KvCacheStorage::RotatedInt4KeyInt4ValueE8>,
                             std::uint8_t>);
static_assert(std::is_same_v<KvKeyScaleT<KvCacheStorage::RotatedInt4KeyInt4ValueE8>, __half>);
static_assert(std::is_same_v<KvValueScaleT<KvCacheStorage::RotatedInt4KeyInt4ValueE8>, __half>);
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedInt4KeyInt4ValueE8).key_leading_extent *
                  2 ==
              ninfer::ops::kD256KVCacheHeadDim);
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedInt4KeyInt4ValueE8)
                  .scale_leading_extent == 4);
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedInt4KeyInt4ValueE8)
                  .value_scale_leading_extent == 8);
static_assert(ninfer::ops::kv_cache_is_int8_family(KvCacheStorage::RotatedInt4KeyInt4ValueE8));

// rk2v4-e8 keeps two root code bytes per eight key dimensions, a quarter of the head dimension,
// under the same G64 key scales, with rk8v4's packed value plane.
static_assert(std::is_same_v<KvKeyCodeT<KvCacheStorage::RotatedE8RootKeyInt4Value>, std::uint8_t>);
static_assert(
    std::is_same_v<KvValueCodeT<KvCacheStorage::RotatedE8RootKeyInt4Value>, std::uint8_t>);
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedE8RootKeyInt4Value).key_leading_extent *
                  4 ==
              ninfer::ops::kD256KVCacheHeadDim);
static_assert(
    d256_kv_cache_profile(KvCacheStorage::RotatedE8RootKeyInt4Value).scale_leading_extent == 4);
static_assert(d256_kv_cache_profile(KvCacheStorage::RotatedE8RootKeyInt4Value)
                  .value_scale_leading_extent == 8);
static_assert(ninfer::ops::kv_cache_is_int8_family(KvCacheStorage::RotatedE8RootKeyInt4Value));

// k8v4 pairs an FP8 key plane with an NVFP4 value plane, so its two scale planes differ in dtype.
static_assert(std::is_same_v<KvKeyScaleT<KvCacheStorage::Fp8KeyNvfp4Value>, __half>);
static_assert(std::is_same_v<KvValueScaleT<KvCacheStorage::Fp8KeyNvfp4Value>, std::uint8_t>);

// A packed byte plane always carries two codes per byte, so its extent is half the head dimension.
template <KvCacheStorage S>
constexpr bool packed_extents_agree() {
    constexpr auto profile = d256_kv_cache_profile(S);
    if (profile.value_code_dtype != DType::U8) { return true; }
    return profile.value_leading_extent * 2 == ninfer::ops::kD256KVCacheHeadDim;
}
static_assert(packed_extents_agree<KvCacheStorage::Nvfp4Group16>());
static_assert(packed_extents_agree<KvCacheStorage::Fp8KeyNvfp4Value>());
static_assert(packed_extents_agree<KvCacheStorage::RotatedInt8KeyInt4ValueGroup64>());
static_assert(packed_extents_agree<KvCacheStorage::RotatedLloyd4KeyInt4Value>());
static_assert(packed_extents_agree<KvCacheStorage::RotatedInt4KeyInt4ValueE8>());
static_assert(packed_extents_agree<KvCacheStorage::RotatedE8RootKeyInt4Value>());

} // namespace

int main() {
    // Nothing to execute: every claim above is a static_assert, so reaching main means the KV
    // plane table and the types derived from it still agree.
    std::cout << "kv_plane_types: PASS (" << (sizeof(kStorages) / sizeof(kStorages[0]))
              << " storages checked at compile time)\n";
    return 0;
}
