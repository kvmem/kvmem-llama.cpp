#pragma once

// Host replica of ops/kv_cache/e8_root_codec.cuh for the rk2v4-e8 tests. It performs the same
// correctly rounded operations in the same order as the device encoder (octet sums in butterfly
// order, the lowest index on every tie), so an encoded block matches the device byte for byte.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace ninfer::test::e8_root {
namespace {

using Block = std::array<float, 8>;

struct Code {
    std::uint8_t root        = 0;
    std::uint8_t radius_axis = 0;
};

struct Stats {
    int pair_roots   = 0;
    int half_roots   = 0;
    int empty_blocks = 0;
};

constexpr float kSqrt8                        = 2.82842712474619f;
constexpr float kInvSqrt2                     = 0.7071067811865475f;
constexpr float kRadiusFloor                  = 0.08f;
constexpr std::array<float, 14> kRadiusBounds = {
    0.2227247f, 0.2806155f, 0.3535534f, 0.4454494f, 0.5612310f, 0.7071068f, 0.8908987f,
    1.1224620f, 1.4142136f, 1.7817974f, 2.2449241f, 2.8284271f, 3.5635949f, 4.4898482f,
};
constexpr std::array<float, 16> kRadiusScale = {
    0.0000f, 0.0992f, 0.1250f, 0.1575f, 0.1984f, 0.2500f, 0.3150f, 0.3969f,
    0.5000f, 0.6300f, 0.7937f, 1.0000f, 1.2599f, 1.5874f, 2.0000f, 2.5198f,
};

// ((0+1)+(2+3))+((4+5)+(6+7)): the value every lane of a butterfly octet reduction ends with.
float octet_sum(const Block& x) {
    return ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
}

// The root table entry: the root times four, dimension 0 first.
std::array<int, 8> root_times_four(int root) {
    std::array<int, 8> out{};
    if (root < 112) {
        int pair = root >> 2;
        int i    = 0;
        while (pair >= 7 - i) {
            pair -= 7 - i;
            ++i;
        }
        const int j                      = i + 1 + pair;
        out[static_cast<std::size_t>(i)] = (root & 2) != 0 ? 4 : -4;
        out[static_cast<std::size_t>(j)] = (root & 1) != 0 ? 4 : -4;
        return out;
    }
    if (root < 240) {
        const int bits = root - 112;
        int minus      = 0;
        for (int d = 0; d < 7; ++d) {
            const bool positive              = ((bits >> d) & 1) != 0;
            out[static_cast<std::size_t>(d)] = positive ? 2 : -2;
            minus += positive ? 0 : 1;
        }
        out[7] = (minus & 1) != 0 ? -2 : 2;
    }
    return out;
}

Code encode(const Block& x, float scale, Stats* stats = nullptr) {
    Block squares{};
    for (std::size_t i = 0; i < 8; ++i) { squares[i] = x[i] * x[i]; }
    const float norm        = std::sqrt(octet_sum(squares));
    const float denominator = scale * kSqrt8;
    const float relative    = norm / (denominator + 1e-8f);
    int radius              = 0;
    if (relative >= kRadiusFloor) {
        radius = 1;
        for (const float bound : kRadiusBounds) { radius += relative >= bound ? 1 : 0; }
    }

    const float inverse = 1.0f / (norm + 1e-8f);
    Block u{};
    Block abs_u{};
    for (std::size_t i = 0; i < 8; ++i) {
        u[i]     = x[i] * inverse;
        abs_u[i] = std::abs(u[i]);
    }

    std::array<int, 8> order{0, 1, 2, 3, 4, 5, 6, 7};
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return abs_u[static_cast<std::size_t>(a)] > abs_u[static_cast<std::size_t>(b)];
    });
    const int pair_i = std::min(order[0], order[1]);
    const int pair_j = std::max(order[0], order[1]);
    const float pair_score =
        abs_u[static_cast<std::size_t>(order[0])] + abs_u[static_cast<std::size_t>(order[1])];
    const auto positive  = [&](int i) { return u[static_cast<std::size_t>(i)] >= 0.0f; };
    const int pair_index = pair_i * (15 - pair_i) / 2 + (pair_j - pair_i - 1);
    const int pair_code =
        (pair_index << 2) | (positive(pair_i) ? 2 : 0) | (positive(pair_j) ? 1 : 0);

    int smallest_i = 0;
    int minus      = 0;
    for (int i = 0; i < 8; ++i) {
        if (abs_u[static_cast<std::size_t>(i)] < abs_u[static_cast<std::size_t>(smallest_i)]) {
            smallest_i = i;
        }
        minus += positive(i) ? 0 : 1;
    }
    const bool odd   = (minus & 1) != 0;
    float half_score = 0.5f * octet_sum(abs_u);
    if (odd) { half_score = half_score - abs_u[static_cast<std::size_t>(smallest_i)]; }
    std::array<bool, 8> half_positive{};
    int half_bits = 0;
    for (int i = 0; i < 8; ++i) {
        half_positive[static_cast<std::size_t>(i)] =
            (odd && i == smallest_i) ? !positive(i) : positive(i);
        if (i < 7 && half_positive[static_cast<std::size_t>(i)]) { half_bits |= 1 << i; }
    }

    const bool pair_root = pair_score >= half_score;
    Block direction{};
    Block products{};
    for (int i = 0; i < 8; ++i) {
        float coordinate = 0.0f;
        if (pair_root) {
            if (i == pair_i) { coordinate = positive(pair_i) ? 1.0f : -1.0f; }
            if (i == pair_j) { coordinate = positive(pair_j) ? 1.0f : -1.0f; }
        } else {
            coordinate = half_positive[static_cast<std::size_t>(i)] ? 0.5f : -0.5f;
        }
        direction[static_cast<std::size_t>(i)] = coordinate * kInvSqrt2;
        products[static_cast<std::size_t>(i)] =
            u[static_cast<std::size_t>(i)] * direction[static_cast<std::size_t>(i)];
    }
    const float projection = octet_sum(products);
    int largest_i          = 0;
    float largest          = -1.0f;
    float residual_at      = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float shift    = projection * direction[static_cast<std::size_t>(i)];
        const float residual = u[static_cast<std::size_t>(i)] - shift;
        if (std::abs(residual) > largest) {
            largest     = std::abs(residual);
            largest_i   = i;
            residual_at = residual;
        }
    }
    const int axis = (largest_i << 1) | (residual_at >= 0.0f ? 0 : 1);

    if (radius == 0) {
        if (stats != nullptr) { ++stats->empty_blocks; }
        return {};
    }
    if (stats != nullptr) { ++(pair_root ? stats->pair_roots : stats->half_roots); }
    return {static_cast<std::uint8_t>(pair_root ? pair_code : 112 + half_bits),
            static_cast<std::uint8_t>((radius << 4) | axis)};
}

// The eight int8 codes the device decodes the block to; the key is code times the group scale.
std::array<std::int8_t, 8> decode(Code code) {
    std::array<std::int8_t, 8> out{};
    const int radius = code.radius_axis >> 4;
    if (radius == 0 || code.root >= 240) { return out; }
    const std::array<int, 8> root = root_times_four(code.root);
    const int axis_dim            = (code.radius_axis >> 1) & 7;
    const int axis_sign           = (code.radius_axis & 1) != 0 ? -1 : 1;
    for (int d = 0; d < 8; ++d) {
        const int value = root[static_cast<std::size_t>(d)] + (d == axis_dim ? axis_sign : 0);
        const float scaled =
            static_cast<float>(value) * kRadiusScale[static_cast<std::size_t>(radius)];
        out[static_cast<std::size_t>(d)] = static_cast<std::int8_t>(std::nearbyint(scaled));
    }
    return out;
}

} // namespace
} // namespace ninfer::test::e8_root
