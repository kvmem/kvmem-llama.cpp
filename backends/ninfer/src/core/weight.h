#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // Ternary codes {-1, 0, +1} as two-bit two's complement (0b10 unused), four per byte in
    // column order, one binary16 multiplier per 128 columns. Row-split only.
    T2_G128_FP16 = 9,
    // ggml block types kept as a GGUF stored them (GgufBlocks layout only): each row is its K /
    // block_elements blocks, byte for byte. Their products quantize the activation to ggml's q8_1.
    GGUF_Q8_0    = 10,
    GGUF_Q2_K    = 11,
    GGUF_Q3_K    = 12,
    GGUF_Q4_K    = 13,
    GGUF_Q5_K    = 14,
    GGUF_Q6_K    = 15,
    GGUF_IQ2_XXS = 16,
    GGUF_IQ2_XS  = 17,
    GGUF_IQ2_S   = 18,
    GGUF_IQ3_XXS = 19,
    GGUF_IQ3_S   = 20,
    GGUF_IQ1_S   = 21,
    GGUF_IQ1_M   = 22,
    GGUF_IQ4_NL  = 23,
    GGUF_IQ4_XS  = 24,
};

[[nodiscard]] constexpr bool is_gguf(QType format) {
    return format >= QType::GGUF_Q8_0 && format <= QType::GGUF_IQ4_XS;
}

struct GgufBlockShape {
    int elements = 0;
    int bytes    = 0;
};

// ggml-common.h's block sizes; {0, 0} for every non-GGUF format.
[[nodiscard]] constexpr GgufBlockShape gguf_block_shape(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return {32, 34};
    case QType::GGUF_Q2_K: return {256, 84};
    case QType::GGUF_Q3_K: return {256, 110};
    case QType::GGUF_Q4_K: return {256, 144};
    case QType::GGUF_Q5_K: return {256, 176};
    case QType::GGUF_Q6_K: return {256, 210};
    case QType::GGUF_IQ2_XXS: return {256, 66};
    case QType::GGUF_IQ2_XS: return {256, 74};
    case QType::GGUF_IQ2_S: return {256, 82};
    case QType::GGUF_IQ3_XXS: return {256, 98};
    case QType::GGUF_IQ3_S: return {256, 110};
    case QType::GGUF_IQ1_S: return {256, 50};
    case QType::GGUF_IQ1_M: return {256, 56};
    case QType::GGUF_IQ4_NL: return {32, 18};
    case QType::GGUF_IQ4_XS: return {256, 136};
    default: return {};
    }
}

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // RowSplit's bytes, permuted so that the code records of kRowSplitPanelRows consecutive rows
    // for one k-group are contiguous. Same size, same bytes, same scales: a block streaming a row
    // tile reads whole cache lines instead of one 32-byte record per row per group. Device-only --
    // it is produced by a load-time permute, never stored in a `.ninfer`.
    RowSplitPanel = 4,
    // Rows of whole ggml blocks, as a GGUF stores them.
    GgufBlocks = 5,
};

// Rows per stored panel. Four 32-byte records is exactly one 128-byte line, which is all the
// prefill kernels need (measured flat from 2 to 64 in tools/w4a8_marlin_probe.cu), and it is the
// least disruptive value for the GEMV decode kernels, whose blocks own four consecutive rows.
inline constexpr int kRowSplitPanelRows  = 4;
inline constexpr int kRowSplitPanelShift = 2;
static_assert((1 << kRowSplitPanelShift) == kRowSplitPanelRows,
              "the panel row count is addressed by shifting, so it must be a power of two");

// Kernels address both layouts with one expression, where a shift of zero collapses the panel form
// to the row-major one. Nothing but this decides which a kernel reads.
[[nodiscard]] constexpr int row_split_panel_shift(QuantLayout layout) {
    return layout == QuantLayout::RowSplitPanel ? kRowSplitPanelShift : 0;
}

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;

    // An NVFP4 plane assembled from several source matrices carries one divisor per source, in the
    // payload after the scales. `weight_divisors` addresses them and `weight_divisor_rows` says how
    // many consecutive rows each covers, so the divisor of row r is element r /
    // weight_divisor_rows. A plane with one source sets the rows to its own row count, which makes
    // that index zero for every row, and `weight_scale_divisor` is then the whole story. On a stack
    // `weight_scale_divisor` holds only the first source's word, so a route that reads it for any
    // other row is silently wrong by a scale factor.
    const void* weight_divisors      = nullptr;
    std::int32_t weight_divisor_rows = 0;

    // INT32 [K] input gather of a GGUF matrix whose stored columns are a permutation of its
    // input's: column c multiplies input element input_columns[c]. Null for every other matrix.
    const std::int32_t* input_columns = nullptr;
};

} // namespace ninfer
