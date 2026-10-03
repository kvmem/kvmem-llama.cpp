#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void paged_kv_window_rows_launch(const Tensor& block_tables, const Tensor& table_rows,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 std::int32_t sink_pages, std::int32_t window_keys,
                                 Tensor& window_tables, Tensor& window_rows,
                                 Tensor& window_positions, cudaStream_t stream);

} // namespace ninfer::ops::detail
