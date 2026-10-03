#pragma once

#include "core/weight_view.h"

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace ninfer::artifact {

// RowSplit stores a weight row-major over K, so a block staging a row tile reads one 32-byte code
// record per row per k-group -- BM scattered transactions for one instruction. The panel layout
// permutes those records so the kRowSplitPanelRows rows of a panel are contiguous within each
// k-group, turning that into whole-line reads. It is size-preserving and tile-local, so it is
// applied in place on the device after upload and before graph capture, and no `.ninfer` ever
// declares it.
[[nodiscard]] bool row_split_panel_supported(const WeightGeometry& geometry) noexcept;

// Permutes the code plane, and the high plane when the format has one, in place. Scales are left
// row-major. Throws if the geometry cannot take the layout.
void permute_row_split_to_panel(const WeightGeometry& geometry, std::byte* data,
                                cudaStream_t stream);

} // namespace ninfer::artifact
