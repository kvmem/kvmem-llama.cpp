#pragma once

// T2G128_F16S row-split storage: 2-bit two's-complement codes, four per byte (lane i in bits
// (i % 4) * 2 of byte i / 4), 128 codes and 32 code bytes per group, one binary16 scale per group,
// no high plane. Legal codes are 00 (0), 01 (+1) and 11 (-1); 10 (-2) is outside the artifact
// language and the decode atoms are defined only for valid streams.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct T2RowSplitStorage {
    static constexpr int kGroupK             = 128;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kScaleBytesPerGroup = 2;
    static constexpr int kCodesPerByte       = 4;
    static constexpr int kCodesPerWord       = 16;
};

// The two bit planes of a packed word: bit 0 of every field is the magnitude (|w| in {0, 1}),
// bit 1 is the sign. value = magnitude - 2 * (magnitude & sign), i.e. 00 -> 0, 01 -> +1, 11 -> -1.
struct T2DecodeAtom {
    // Sixteen packed codes -> sixteen floats scaled by the group's binary16 multiplier, in k
    // order (code j is field j of the word, little-endian bytes).
    __device__ static __forceinline__ void
    decode_sixteen(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[16]) {
        const float scale     = __half2float(__ushort_as_half(scale_bits));
        const float neg_scale = -scale;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const std::uint32_t field = (packed >> (2 * j)) & 0x3u;
            weights[j]                = (field == 0u) ? 0.0f : ((field & 0x2u) ? neg_scale : scale);
        }
    }

    // Four packed codes of one byte -> two unscaled bf16 pairs {w0, w1}, {w2, w3}; callers fold
    // the group scale into the accumulation as the Q4 MMA atom does.
    __device__ static __forceinline__ void decode_quad(std::uint8_t packed, unsigned& pair01,
                                                       unsigned& pair23) {
        // bf16 patterns: +1 = 0x3F80, -1 = 0xBF80, 0 = 0x0000.
        const auto lane = [](std::uint32_t field) -> std::uint32_t {
            return (field & 0x1u) ? (0x3F80u | ((field & 0x2u) << 14)) : 0u;
        };
        pair01 = lane(packed & 0x3u) | (lane((packed >> 2) & 0x3u) << 16);
        pair23 = lane((packed >> 4) & 0x3u) | (lane((packed >> 6) & 0x3u) << 16);
    }

    __device__ static __forceinline__ void
    decode_quad(std::uint8_t packed, std::uint16_t scale_bits, float (&weights)[4]) {
        const float scale     = __half2float(__ushort_as_half(scale_bits));
        const float neg_scale = -scale;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t field = (static_cast<std::uint32_t>(packed) >> (2 * j)) & 0x3u;
            weights[j]                = (field == 0u) ? 0.0f : ((field & 0x2u) ? neg_scale : scale);
        }
    }
};

} // namespace ninfer::ops::detail
