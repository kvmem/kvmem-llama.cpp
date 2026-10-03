#pragma once

#include "core/dtype.h"

#include <ninfer/types.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

// K and V are described separately because the rk8v4 and nvfp4-family profiles pair differently
// coded key and value planes (and, for k8v4, differently coded key and value *scale* planes too).
// Every other profile stores both planes identically, so their key and value fields agree.
//
// A code plane of DType::U8 denotes two 4-bit codes per byte, low nibble first (signed int4 for
// rk8v4's value plane, e2m1 for the nvfp4-family), which is why its leading extent is half the
// head dimension. Scale dtype is FP16 for every family except nvfp4-family value planes, whose
// group scale is a raw stored E4M3 byte (DType::U8).
struct D256KVCacheProfile {
    DType key_code_dtype;
    DType value_code_dtype;
    std::int32_t key_leading_extent;
    std::int32_t value_leading_extent;
    std::int32_t quant_group;
    DType key_scale_dtype;
    std::int32_t scale_leading_extent;
    // Values may use a finer group (or a different scale coding) than keys, so their scale plane
    // can differ in both dtype and width.
    DType value_scale_dtype;
    std::int32_t value_scale_leading_extent;

    [[nodiscard]] constexpr bool packed_int4_values() const {
        return value_code_dtype == DType::U8;
    }
};

// value_code_dtype selects between the INT8 and packed int4 value codings inside the INT8 family;
// callers pass the value plane's own dtype. Other families ignore it, because their value coding
// is implied by the family.
inline constexpr D256KVCacheProfile d256_kv_cache_profile(DType dtype, DType value_code_dtype) {
    switch (dtype) {
    case DType::BF16:
        return {DType::BF16, DType::BF16, kD256KVCacheHeadDim, kD256KVCacheHeadDim,
                0,           DType::FP16, 0,                   DType::FP16,
                0};
    case DType::I8:
        if (value_code_dtype == DType::U8) {
            return {DType::I8, DType::U8,   kD256KVCacheHeadDim, kD256KVCacheHeadDim / 2,
                    64,        DType::FP16, 4,                   DType::FP16,
                    8};
        }
        return {DType::I8, DType::I8,   kD256KVCacheHeadDim, kD256KVCacheHeadDim,
                64,        DType::FP16, 4,                   DType::FP16,
                4};
    case DType::FP8_E4M3FN:
        return {DType::FP8_E4M3FN, DType::FP8_E4M3FN, kD256KVCacheHeadDim, kD256KVCacheHeadDim,
                256,               DType::FP16,        1,                   DType::FP16,
                1};
    default:
        throw std::invalid_argument("unsupported D256 KV-cache dtype");
    }
}

// The storages the INT8-family kernels serve: keys reach QK as rotated INT8 codes with a G64 FP16
// scale (stored directly, or as rk4v4 Lloyd-Max indices, rk4v4-e8 int4 codes or rk2v4-e8 E8 root
// codes expanded on load), and values are INT8 or packed int4. The value plane's dtype and, for
// packed keys, the storage select the coding inside each kernel.
inline constexpr bool kv_cache_is_int8_family(ninfer::KvCacheStorage storage) {
    return storage == ninfer::KvCacheStorage::Int8Group64 ||
           storage == ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
           storage == ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value ||
           storage == ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
           storage == ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value;
}

// Overload for call sites that describe a cache whose two planes share one coding.
inline constexpr D256KVCacheProfile d256_kv_cache_profile(DType dtype) {
    return d256_kv_cache_profile(dtype, dtype);
}

// Overload for call sites that only carry the public KvCacheStorage selection (PagedKVLayerView
// and friends).
inline constexpr D256KVCacheProfile d256_kv_cache_profile(ninfer::KvCacheStorage storage) {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return d256_kv_cache_profile(DType::BF16);
    case ninfer::KvCacheStorage::Int8Group64:
        return d256_kv_cache_profile(DType::I8, DType::I8);
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return d256_kv_cache_profile(DType::FP8_E4M3FN);
    case ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return d256_kv_cache_profile(DType::I8, DType::U8);
    case ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value:
    case ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        // Both planes pack two 4-bit codes per byte (Lloyd-Max indices or E8 int4 key codes, signed
        // int4 values), with rk8v4's scale planes: G64 FP16 for keys, G32 FP16 for values.
        return {DType::U8, DType::U8,   kD256KVCacheHeadDim / 2, kD256KVCacheHeadDim / 2,
                64,        DType::FP16, 4,                       DType::FP16,
                8};
    case ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value:
        // Two E8 root code bytes per eight key dimensions under the G64 FP16 scale; the value
        // plane is rk8v4's. Dispatch on the storage, as for rk4v4 and rk4v4-e8.
        return {DType::U8,
                DType::U8,
                kD256KVCacheHeadDim / 4,
                kD256KVCacheHeadDim / 2,
                64,
                DType::FP16,
                4,
                DType::FP16,
                8};
    case ninfer::KvCacheStorage::Nvfp4Group16:
        // Both planes are e2m1-coded (2 codes/byte, so half the head dimension) with a 16-value
        // group and a raw E4M3 scale byte per group.
        return {DType::U8, DType::U8, kD256KVCacheHeadDim / 2, kD256KVCacheHeadDim / 2,
                16,        DType::U8, kD256KVCacheHeadDim / 16, DType::U8,
                kD256KVCacheHeadDim / 16};
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        // Key plane matches Fp8E4M3Row256's coding (FP16 scale); value plane matches
        // Nvfp4Group16's (raw E4M3 scale byte).
        return {DType::FP8_E4M3FN, DType::U8, kD256KVCacheHeadDim, kD256KVCacheHeadDim / 2,
                256,               DType::FP16, 1,                  DType::U8,
                kD256KVCacheHeadDim / 16};
    }
    throw std::invalid_argument("unknown KV-cache storage profile");
}

} // namespace ninfer::ops
