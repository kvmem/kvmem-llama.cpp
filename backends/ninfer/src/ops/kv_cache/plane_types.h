#pragma once

#include "core/dtype.h"
#include "ops/kv_cache/d256_profile.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

// One declaration of the C++ type behind a KV plane, derived from d256_kv_cache_profile so there
// is still exactly one table describing the cache. Nothing here restates a dtype; it only turns
// the dtype the profile already declares into the type a kernel must use.
//
// Why it exists. Every 16-bit KV plane is the same width, so aiming a kernel at the wrong 16-bit
// type reinterprets the bytes rather than failing to compile: it builds, it runs, and it decodes
// 100.0 as 3.39. Upstream stores the value plane as FP16 while this fork stores it symmetrically
// with the key plane as BF16, so each catch-up merge reintroduces __half on the value plane across
// a spread of files, and neither the compiler nor a grep for "__half" can separate that from the
// scale planes, where __half is correct. Routing plane pointers through these aliases makes the
// mismatch a compile error instead of a silent numerical fault.
//
// The primary template is deliberately left undefined: an unhandled dtype must fail to compile
// rather than fall back to something plausible.
template <DType D>
struct KvPlaneStorage;

template <>
struct KvPlaneStorage<DType::BF16> {
    using type = __nv_bfloat16;
};

template <>
struct KvPlaneStorage<DType::FP16> {
    using type = __half;
};

template <>
struct KvPlaneStorage<DType::I8> {
    using type = std::int8_t;
};

// Byte planes: U8 carries two 4-bit codes (signed int4 for rk8v4, e2m1 for the nvfp4 family) and
// FP8_E4M3FN carries one raw E4M3 byte. Both are addressed as bytes and decoded by their codec.
template <>
struct KvPlaneStorage<DType::U8> {
    using type = std::uint8_t;
};

template <>
struct KvPlaneStorage<DType::FP8_E4M3FN> {
    using type = std::uint8_t;
};

template <DType D>
using KvPlaneT = typename KvPlaneStorage<D>::type;

// The four planes of a cache storage, named by role. These are the types a kernel signature should
// use; getting one wrong is now a compile error at the call site rather than a silent reinterpret.
template <ninfer::KvCacheStorage S>
using KvKeyCodeT = KvPlaneT<d256_kv_cache_profile(S).key_code_dtype>;
template <ninfer::KvCacheStorage S>
using KvValueCodeT = KvPlaneT<d256_kv_cache_profile(S).value_code_dtype>;
template <ninfer::KvCacheStorage S>
using KvKeyScaleT = KvPlaneT<d256_kv_cache_profile(S).key_scale_dtype>;
template <ninfer::KvCacheStorage S>
using KvValueScaleT = KvPlaneT<d256_kv_cache_profile(S).value_scale_dtype>;

// Assert that a kernel's actual plane pointer types match what the storage declares. Call it at
// the top of a launcher: the diagnostic names the offending plane, which a raw type mismatch
// deeper in a template would not.
// Accepts either the element type or a (cv-qualified) pointer to it, since call sites naturally
// pass decltype of a plane pointer.
template <typename T>
using KvPlaneElementT = std::remove_cv_t<std::remove_pointer_t<std::remove_reference_t<T>>>;

template <ninfer::KvCacheStorage S, typename KeyCode, typename ValueCode>
constexpr void assert_kv_code_planes() {
    static_assert(std::is_same_v<KvPlaneElementT<KeyCode>, KvKeyCodeT<S>>,
                  "KV key code plane type does not match the declared cache storage");
    static_assert(std::is_same_v<KvPlaneElementT<ValueCode>, KvValueCodeT<S>>,
                  "KV value code plane type does not match the declared cache storage. This fork "
                  "stores V symmetrically with K (BF16); upstream stores V as FP16, and that "
                  "difference is byte-compatible, so it will not fail any other way.");
}

// The scale planes need the same treatment and for a sharper reason: across the six storages the
// same role takes three different types. The value scale is FP16 for the INT8 family, a raw E4M3
// byte for the NVFP4 family, and FP16 again for rk8v4 while its *key* scale is also FP16 -- so
// "the scale plane is __half" is true often enough to look like a rule and wrong on two storages.
// Fp8KeyNvfp4Value mixes them within one cache: FP16 key scale, byte value scale.
template <ninfer::KvCacheStorage S, typename KeyScale, typename ValueScale>
constexpr void assert_kv_scale_planes() {
    static_assert(std::is_same_v<KvPlaneElementT<KeyScale>, KvKeyScaleT<S>>,
                  "KV key scale plane type does not match the declared cache storage");
    static_assert(std::is_same_v<KvPlaneElementT<ValueScale>, KvValueScaleT<S>>,
                  "KV value scale plane type does not match the declared cache storage");
}

// All four planes at once, which is what a launcher wants.
template <ninfer::KvCacheStorage S, typename KeyCode, typename ValueCode, typename KeyScale,
          typename ValueScale>
constexpr void assert_kv_planes() {
    assert_kv_code_planes<S, KeyCode, ValueCode>();
    assert_kv_scale_planes<S, KeyScale, ValueScale>();
}

// This fork's invariant, stated once where it can be checked rather than only in prose: the
// BFloat16 profile is symmetric, both code planes BF16. If a merge ever flips the value plane to
// FP16, this fails at compile time in every translation unit that includes the header.
static_assert(std::is_same_v<KvKeyCodeT<ninfer::KvCacheStorage::BFloat16>, __nv_bfloat16>);
static_assert(std::is_same_v<KvValueCodeT<ninfer::KvCacheStorage::BFloat16>, __nv_bfloat16>,
              "the BF16 KV cache must store V as BF16, symmetric with K");

} // namespace ninfer::ops
