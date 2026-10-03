#include "artifact/permute_row_split.h"

#include "artifact/schema.h"
#include "core/weight.h"

#include <cuda_runtime.h>
#include <string>

namespace ninfer::artifact {
namespace {

constexpr int kPanel   = kRowSplitPanelRows;
constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string("failed to ") + operation + ": " +
                            cudaGetErrorString(status));
    }
}

// One block owns one panel and rewrites it in place through shared memory. The permutation never
// crosses a panel, so blocks are independent and the object needs no scratch: the panel is exactly
// as large after as before. Within a panel, row-major (row, group) becomes group-major
// (group, row), which is what makes a row tile's records for one k-group contiguous.
__global__ void permute_panel_kernel(std::byte* __restrict__ data, int groups, int bytes_per_group,
                                     long long panels) {
    extern __shared__ unsigned int staged[];
    const long long panel = blockIdx.x;
    if (panel >= panels) { return; }

    const int row_bytes   = groups * bytes_per_group;
    const int panel_bytes = kPanel * row_bytes;
    auto* base            = reinterpret_cast<unsigned int*>(data + panel * panel_bytes);
    const int words       = panel_bytes / 4;

    for (int i = static_cast<int>(threadIdx.x); i < words; i += kThreads) { staged[i] = base[i]; }
    __syncthreads();

    for (int i = static_cast<int>(threadIdx.x); i < words; i += kThreads) {
        // Destination word i lands at group g, row r, byte j of that record.
        const int byte = i * 4;
        const int g    = byte / (kPanel * bytes_per_group);
        const int rem  = byte - g * (kPanel * bytes_per_group);
        const int r    = rem / bytes_per_group;
        const int j    = rem - r * bytes_per_group;
        base[i]        = staged[(r * row_bytes + g * bytes_per_group + j) / 4];
    }
}

void permute_plane(std::byte* data, int groups, int bytes_per_group, long long rows,
                   cudaStream_t stream) {
    if (!bytes_per_group || !groups || rows <= 0) { return; }
    const long long panels     = rows / kPanel;
    const std::size_t shared   = static_cast<std::size_t>(kPanel) * groups * bytes_per_group;
    if (shared > 96u * 1024u) {
        throw ArtifactError("panel permute needs more shared memory than the device offers");
    }
    check_cuda(cudaFuncSetAttribute(permute_panel_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared)),
               "raise shared memory for the panel permute");
    permute_panel_kernel<<<static_cast<unsigned>(panels), kThreads, shared, stream>>>(
        data, groups, bytes_per_group, panels);
    check_cuda(cudaGetLastError(), "launch the panel permute");
}

} // namespace

bool row_split_panel_supported(const WeightGeometry& geometry) noexcept {
    if (!is_row_split(geometry.layout) || geometry.shape.size() != 2 || !geometry.group_size) {
        return false;
    }
    // A partial panel has nowhere to go, and every consumer slices rows on a multiple of 64.
    return geometry.shape[0] % kRowSplitPanelRows == 0;
}

void permute_row_split_to_panel(const WeightGeometry& geometry, std::byte* data,
                                cudaStream_t stream) {
    if (!row_split_panel_supported(geometry)) {
        throw ArtifactError("weight cannot take the panel layout");
    }
    const auto rows   = static_cast<long long>(geometry.shape[0]);
    const int groups  = static_cast<int>(geometry.padded_columns / geometry.group_size);
    permute_plane(data, groups, 32, rows, stream);
    if (geometry.high_bytes) {
        permute_plane(data + geometry.high_offset, groups,
                      static_cast<int>(geometry.high_bytes_per_row / groups), rows, stream);
    }
    // Scales stay row-major. They are 2 bytes per row per group against 32 for the codes, and the
    // prefill kernels prefetch them once per ring refill rather than once per group, so the same
    // argument that makes the codes worth permuting does not carry to them. Measure before adding.
}

} // namespace ninfer::artifact
