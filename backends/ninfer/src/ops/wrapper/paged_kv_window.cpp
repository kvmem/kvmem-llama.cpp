#include "ninfer/ops/paged_kv_window.h"

#include "ops/launcher/paged_kv_window.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_i32(const Tensor& tensor, std::int32_t ne0, std::int32_t ne1, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != ne0 || tensor.ne[1] != ne1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string("paged_kv_window_rows: ") + name +
                                    " has the wrong shape or layout");
    }
}

} // namespace

void paged_kv_window_rows(const Tensor& block_tables, const Tensor& table_rows,
                          const Tensor& positions, const Tensor& valid_columns,
                          std::int32_t sink_pages, std::int32_t window_keys, Tensor& window_tables,
                          Tensor& window_rows, Tensor& window_positions, cudaStream_t stream) {
    const std::int32_t pages = block_tables.ne[0];
    const std::int32_t rows  = table_rows.ne[0];
    const std::int32_t width = positions.ne[0];
    if (pages <= 0 || rows <= 0 || width <= 0 || sink_pages < 0 || window_keys <= 0) {
        throw std::invalid_argument("paged_kv_window_rows: empty extent or invalid window");
    }
    require_i32(block_tables, pages, block_tables.ne[1], "block_tables");
    require_i32(table_rows, rows, 1, "table_rows");
    require_i32(positions, width, rows, "positions");
    if (valid_columns.data != nullptr) { require_i32(valid_columns, rows, 1, "valid_columns"); }
    require_i32(window_tables, pages, rows, "window_tables");
    require_i32(window_rows, rows, 1, "window_rows");
    require_i32(window_positions, width, rows, "window_positions");
    detail::paged_kv_window_rows_launch(block_tables, table_rows, positions, valid_columns,
                                        sink_pages, window_keys, window_tables, window_rows,
                                        window_positions, stream);
}

} // namespace ninfer::ops
