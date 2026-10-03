#include "ops/launcher/paged_kv_window.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_window.cuh"

namespace ninfer::ops::detail {

void paged_kv_window_rows_launch(const Tensor& block_tables, const Tensor& table_rows,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 std::int32_t sink_pages, std::int32_t window_keys,
                                 Tensor& window_tables, Tensor& window_rows,
                                 Tensor& window_positions, cudaStream_t stream) {
    const std::int32_t rows = table_rows.ne[0];
    paged_kv_window_rows_kernel<<<rows, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(positions.data),
        valid_columns.data != nullptr ? static_cast<const std::int32_t*>(valid_columns.data)
                                      : nullptr,
        block_tables.ne[0], positions.ne[0], sink_pages, window_keys,
        static_cast<std::int32_t*>(window_tables.data),
        static_cast<std::int32_t*>(window_rows.data),
        static_cast<std::int32_t*>(window_positions.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
