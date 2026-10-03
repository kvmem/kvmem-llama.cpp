#pragma once

// Independent host oracle for the rk4v4 Lloyd-Max key codec (ninfer/ops/kv_cache_append.h), shared
// by the append and attention tests. It follows the specified group encoding rather than any kernel:
// lane l of a G64 group owns dimensions l and l+32, and the RMS, the index against the Lloyd-Max
// boundaries and the least-squares scale use single-rounded FP32 operations in xor-butterfly order,
// so the device result is reproduced exactly.

#include <array>
#include <cmath>
#include <cstdint>

namespace ninfer::test {

inline constexpr float kLloyd4Boundaries[7] = {0.25825f, 0.52245f, 0.79960f, 1.09930f,
                                               1.43715f, 1.84355f, 2.40080f};
inline constexpr int kLloyd4Codes[8]        = {6, 18, 31, 44, 58, 75, 96, 127};

// Signed INT8 code of a 4-bit index (sign << 3 | magnitude).
inline int lloyd4_code(std::uint8_t index) {
    return (index & 8u) != 0 ? -kLloyd4Codes[index & 7u] : kLloyd4Codes[index & 7u];
}

inline float lloyd4_butterfly_sum(std::array<float, 32> lanes) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        std::array<float, 32> next{};
        for (int lane = 0; lane < 32; ++lane) {
            next[static_cast<std::size_t>(lane)] = lanes[static_cast<std::size_t>(lane)] +
                                                   lanes[static_cast<std::size_t>(lane ^ offset)];
        }
        lanes = next;
    }
    return lanes[0];
}

struct Lloyd4Group {
    std::array<std::uint8_t, 64> index{};
    float scale = 0.0f; // least-squares FP32 scale, before its FP16 rounding
};

// Encode one rotated G64 group; x holds its 64 values in dimension order.
inline Lloyd4Group lloyd4_encode_group(const float* x) {
    Lloyd4Group out;
    std::array<float, 32> sum_sq{};
    for (int lane = 0; lane < 32; ++lane) {
        const float x0 = x[lane];
        const float x1 = x[lane + 32];
        sum_sq[static_cast<std::size_t>(lane)] = x0 * x0 + x1 * x1;
    }
    const float rms     = std::sqrt(lloyd4_butterfly_sum(sum_sq) * (1.0f / 64.0f));
    const float inv_rms = rms > 0.0f ? 1.0f / rms : 0.0f;
    std::array<float, 32> xc{};
    std::array<float, 32> cc{};
    for (int lane = 0; lane < 32; ++lane) {
        float c[2]{};
        for (int half = 0; half < 2; ++half) {
            const int d           = lane + 32 * half;
            const float magnitude = std::abs(x[d] * inv_rms);
            std::uint8_t index    = 0;
            for (const float boundary : kLloyd4Boundaries) index += magnitude >= boundary ? 1 : 0;
            index = static_cast<std::uint8_t>(index | (x[d] < 0.0f ? 8u : 0u));
            out.index[static_cast<std::size_t>(d)] = index;
            c[half] = static_cast<float>(lloyd4_code(index));
        }
        xc[static_cast<std::size_t>(lane)] = x[lane] * c[0] + x[lane + 32] * c[1];
        cc[static_cast<std::size_t>(lane)] = c[0] * c[0] + c[1] * c[1];
    }
    out.scale = rms > 0.0f ? lloyd4_butterfly_sum(xc) / lloyd4_butterfly_sum(cc) : 0.0f;
    return out;
}

} // namespace ninfer::test
