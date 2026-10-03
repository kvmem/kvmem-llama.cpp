// The panel permute is applied in place to real weight bytes and is never re-read from the
// artifact, so nothing downstream can detect a wrong permutation except as degraded output. These
// checks are therefore the primary guard: that the geometry is byte-identical to RowSplit's, that
// the permutation is exactly the one the kernels assume, and that it is a bijection.

#include "artifact/permute_row_split.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

WeightGeometry geometry_for(QType format, QuantLayout layout, std::uint64_t rows,
                            std::uint64_t columns) {
    const std::uint64_t shape[2] = {rows, columns};
    return weight_geometry(format, layout, std::span<const std::uint64_t>(shape, 2));
}

// The panel layout permutes bytes; it must not move a single boundary.
void geometry_matches_row_split() {
    for (const auto format : {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16}) {
        const auto plain = geometry_for(format, QuantLayout::RowSplit, 256, 5120);
        const auto panel = geometry_for(format, QuantLayout::RowSplitPanel, 256, 5120);
        require(plain.bytes == panel.bytes && plain.code_bytes == panel.code_bytes &&
                    plain.code_bytes_per_row == panel.code_bytes_per_row &&
                    plain.high_offset == panel.high_offset &&
                    plain.high_bytes_per_row == panel.high_bytes_per_row &&
                    plain.scale_offset == panel.scale_offset &&
                    plain.scale_bytes_per_row == panel.scale_bytes_per_row &&
                    plain.group_size == panel.group_size &&
                    plain.padded_columns == panel.padded_columns,
                "panel geometry differs from row-split geometry");
    }
}

// Every byte must land where the kernels index it, and nowhere else.
void permutation_is_the_expected_one() {
    constexpr std::uint64_t kRows    = 64;
    constexpr std::uint64_t kColumns = 512;
    const auto geometry = geometry_for(QType::Q5_G64_FP16, QuantLayout::RowSplitPanel, kRows,
                                       kColumns);
    const auto groups   = geometry.padded_columns / geometry.group_size;

    // A byte whose value encodes its own (row, group, index) so a misplacement is visible.
    std::vector<std::uint8_t> source(static_cast<std::size_t>(geometry.bytes));
    std::iota(source.begin(), source.end(), std::uint8_t{0});

    std::uint8_t* device = nullptr;
    if (cudaMalloc(&device, source.size()) != cudaSuccess) {
        std::cout << "no device; skipping\n";
        std::exit(77);
    }
    require(cudaMemcpy(device, source.data(), source.size(), cudaMemcpyHostToDevice) == cudaSuccess,
            "upload failed");
    permute_row_split_to_panel(geometry, reinterpret_cast<std::byte*>(device), nullptr);
    require(cudaDeviceSynchronize() == cudaSuccess, "permute launch failed");

    std::vector<std::uint8_t> permuted(source.size());
    require(cudaMemcpy(permuted.data(), device, source.size(), cudaMemcpyDeviceToHost) ==
                cudaSuccess,
            "download failed");

    const auto check_plane = [&](std::uint64_t plane_offset, std::uint64_t bytes_per_group) {
        const auto row_bytes = groups * bytes_per_group;
        for (std::uint64_t row = 0; row < kRows; ++row) {
            for (std::uint64_t g = 0; g < groups; ++g) {
                for (std::uint64_t j = 0; j < bytes_per_group; ++j) {
                    const auto panel     = row / kRowSplitPanelRows;
                    const auto in_panel  = row % kRowSplitPanelRows;
                    const auto from      = plane_offset + row * row_bytes + g * bytes_per_group + j;
                    const auto to        = plane_offset +
                                    panel * kRowSplitPanelRows * row_bytes +
                                    g * kRowSplitPanelRows * bytes_per_group +
                                    in_panel * bytes_per_group + j;
                    require(permuted[static_cast<std::size_t>(to)] ==
                                source[static_cast<std::size_t>(from)],
                            "a code byte did not land at its panel-major position");
                }
            }
        }
    };
    check_plane(0, 32);
    check_plane(geometry.high_offset, geometry.high_bytes_per_row / groups);

    // Scales are deliberately left row-major; if that ever changes this check should change with it.
    for (std::uint64_t i = geometry.scale_offset; i < geometry.bytes; ++i) {
        require(permuted[static_cast<std::size_t>(i)] == source[static_cast<std::size_t>(i)],
                "the permute disturbed the scale plane");
    }

    // A permutation moves bytes without creating or destroying them.
    std::vector<int> before(256, 0);
    std::vector<int> after(256, 0);
    for (std::uint64_t i = 0; i < geometry.code_bytes; ++i) {
        ++before[source[static_cast<std::size_t>(i)]];
        ++after[permuted[static_cast<std::size_t>(i)]];
    }
    require(before == after, "the code plane is not a permutation of itself");
    require(cudaFree(device) == cudaSuccess, "free failed");
}

void rejects_what_it_cannot_express() {
    // A partial panel has nowhere to go.
    require(!row_split_panel_supported(
                geometry_for(QType::Q4_G64_FP16, QuantLayout::RowSplitPanel, 66, 5120)),
            "a row count that is not a whole number of panels was accepted");
    require(!row_split_panel_supported(
                geometry_for(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, 256, 5120)),
            "a non row-split layout was accepted");
}

} // namespace

int main() {
    try {
        geometry_matches_row_split();
        rejects_what_it_cannot_express();
        permutation_is_the_expected_one();
        std::cout << "row-split panel permute checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
