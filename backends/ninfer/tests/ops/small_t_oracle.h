#pragma once

// fp64 oracles for the small-T MMA projection tests (test_gdn_input_small_t.cpp,
// test_attn_input_small_t.cpp). A projection's oracle is computed once for the widest column count
// a test uses; columns are independent, so a narrower T checks a prefix of it.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ninfer::test::small_t_oracle {

inline float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f                  = 0.0F;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::uint16_t f32_to_bf16_rne(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

// out[col * rows + row] = sum_k w[row, k] * x[col, k], with w row-major [rows, k] as decoded by the
// quantized-weight fixture and x column-major bf16 [k, cols].
inline std::vector<double> project(const std::vector<float>& w, std::int32_t rows, std::int32_t k,
                                   const std::vector<std::uint16_t>& x, std::int32_t cols) {
    std::vector<double> out(static_cast<std::size_t>(rows) * cols);
    for (std::int32_t col = 0; col < cols; ++col) {
        const std::uint16_t* xcol = x.data() + static_cast<std::size_t>(col) * k;
        for (std::int32_t row = 0; row < rows; ++row) {
            const float* wrow = w.data() + static_cast<std::size_t>(row) * k;
            double sum        = 0.0;
            for (std::int32_t kk = 0; kk < k; ++kk) {
                sum += static_cast<double>(wrow[kk]) * bf16_to_f32(xcol[kk]);
            }
            out[static_cast<std::size_t>(col) * rows + row] = sum;
        }
    }
    return out;
}

// One bf16 rounding step of the result, plus an absolute floor for outputs that cancel to near
// zero: the kernels read exact bf16 codes and activations, accumulate in fp32 and round once.
inline bool within(double oracle, float value) {
    return std::isfinite(value) &&
           std::fabs(static_cast<double>(value) - oracle) <= std::fabs(oracle) / 256.0 + 1.0e-6;
}

struct Miss {
    std::size_t count = 0;
    std::int32_t row  = 0;
    std::int32_t col  = 0;
    float value       = 0.0F;
    double oracle     = 0.0;
};

// Scores output rows [0, rows) of a column-major bf16 tensor (leading dimension `ld`) against
// oracle rows [oracle_row0, oracle_row0 + rows) of a projection with `oracle_rows` rows.
inline Miss score(const std::vector<std::uint16_t>& out, std::int32_t ld, std::int32_t rows,
                  std::int32_t tokens, const std::vector<double>& oracle, std::int32_t oracle_rows,
                  std::int32_t oracle_row0) {
    Miss miss;
    for (std::int32_t col = 0; col < tokens; ++col) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const float value = bf16_to_f32(out[static_cast<std::size_t>(col) * ld + row]);
            const double expected =
                oracle[static_cast<std::size_t>(col) * oracle_rows + oracle_row0 + row];
            if (!within(expected, value)) {
                if (miss.count == 0) {
                    miss.row    = row;
                    miss.col    = col;
                    miss.value  = value;
                    miss.oracle = expected;
                }
                ++miss.count;
            }
        }
    }
    return miss;
}

} // namespace ninfer::test::small_t_oracle
