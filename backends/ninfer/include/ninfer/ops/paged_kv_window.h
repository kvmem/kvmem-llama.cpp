#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: paged_kv_window_rows
 *
 * Rebinds B rows of a paged cache to a windowed view that the causal attention Ops read as a
 * shorter sequence of the same pages: the first sink_pages pages of each row, then its pages from
 * the one holding position max(0, p0 - window_keys) on, where p0 is the row's first live position.
 * Positions shift down by the pages skipped, so every live key a query sees keeps its offset in
 * its page and a new K/V row still lands in its own page. A row whose window starts inside its
 * sink keeps its table and positions.
 *
 * Math / indexing, for row b with table row r = table_rows[b], live columns V = valid_columns[b]
 * (W when valid_columns is empty), p0 = positions[0,b] when V > 0 and 0 otherwise:
 *   first  = max(0, p0 - window_keys) >> 6, shift = max(0, first - sink_pages);
 *   window_tables[j,b] = block_tables[j < sink_pages ? j : j + shift, r] for j + shift < P,
 *                        block_tables[j, r] otherwise (never addressed);
 *   window_positions[c,b] = positions[c,b] - 64 * shift (unchanged when smaller than that);
 *   window_rows[b] = b.
 *
 * Logical shapes:
 *   block_tables I32 [P,R]; table_rows I32 [B]; positions I32 [W,B]; valid_columns I32 [B] or an
 *   empty Tensor; window_tables I32 [P,B]; window_rows I32 [B]; window_positions I32 [W,B]. All
 *   contiguous device tensors; outputs do not alias inputs.
 *
 * Numeric:
 *   sink_pages >= 0 and window_keys > 0.
 *
 * Effects:
 *   Writes the three outputs completely.
 *
 * Workspace:
 *   None.
 */
void paged_kv_window_rows(const Tensor& block_tables, const Tensor& table_rows,
                          const Tensor& positions, const Tensor& valid_columns,
                          std::int32_t sink_pages, std::int32_t window_keys, Tensor& window_tables,
                          Tensor& window_rows, Tensor& window_positions, cudaStream_t stream);

} // namespace ninfer::ops
