#pragma once

// Row views over a T2G128_F16S RowSplit parent. The row-split planes are row-major per plane, so a
// contiguous row range of the parent is itself a valid RowSplit weight with offset code and scale
// pointers; the fused wrappers project the parent's parts through ops::linear with these views.

#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

inline Weight t2_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    if (parent.qtype != QType::T2_G128_FP16 || parent.layout != QuantLayout::RowSplit ||
        row_begin < 0 || rows <= 0 || row_begin + rows > parent.n) {
        throw std::invalid_argument("t2 row view: invalid parent or row range");
    }
    const std::int64_t groups_per_row = parent.padded_shape[1] / 128;
    Weight view                       = parent;
    view.n                            = rows;
    view.shape[0]                     = rows;
    view.padded_shape[0]              = rows;
    view.qdata                        = static_cast<const std::uint8_t*>(parent.qdata) +
                                        static_cast<std::int64_t>(row_begin) * groups_per_row * 32;
    view.scales                       = static_cast<const std::uint8_t*>(parent.scales) +
                                        static_cast<std::int64_t>(row_begin) * groups_per_row * 2;
    return view;
}

} // namespace ninfer::ops::detail
