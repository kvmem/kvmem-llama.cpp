// Two halves, and the second is the one that matters.
//
// The enumeration half answers "which CUDA indices are the two cards, and can they reach each
// other" -- indices for --devices must be read from a CUDA-level enumeration rather than assumed,
// since a display GPU commonly takes cuda=0 even when nvidia-smi lists it elsewhere.
//
// The DeviceContext half then constructs the real thing across those indices. That is the check
// worth running on rented hardware: peer access is a performance capability rather than a
// requirement, so a bridgeless pair must construct successfully with peer_access() == false, and
// cudaMemcpyPeer must still move bytes by staging through the host. This binary is what proves
// that on a machine we do not own.
#include "core/device.h"

#include <cstdio>
#include <exception>
#include <vector>

namespace {

// canAccessPeer only reports capability. Moving bytes is the test that matters, so do it either
// way and report the real error string on failure.
bool check_peer_copy(int source, int destination) {
    void* src = nullptr;
    void* dst = nullptr;
    constexpr std::size_t kBytes = 1u << 20;

    if (cudaSetDevice(source) != cudaSuccess) { return false; }
    if (cudaMalloc(&src, kBytes) != cudaSuccess) { return false; }
    if (cudaSetDevice(destination) != cudaSuccess) {
        cudaFree(src);
        return false;
    }
    if (cudaMalloc(&dst, kBytes) != cudaSuccess) {
        cudaFree(src);
        return false;
    }

    const cudaError_t copy = cudaMemcpyPeer(dst, destination, src, source, kBytes);
    std::printf("memcpy_peer %d->%d 1MiB=%s\n", source, destination,
                copy == cudaSuccess ? "OK" : cudaGetErrorString(copy));

    cudaSetDevice(source);
    cudaFree(src);
    cudaSetDevice(destination);
    cudaFree(dst);
    return copy == cudaSuccess;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { return 1; }
    std::printf("devices=%d\n", count);
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        char bus_id[32]{};
        cudaGetDeviceProperties(&props, device);
        cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device);
        int managed            = 0;
        int concurrent_managed = 0;
        cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device);
        cudaDeviceGetAttribute(&concurrent_managed, cudaDevAttrConcurrentManagedAccess, device);
        std::printf("cuda=%d bus=%s name=%s sm=%d%d vram_mib=%zu managed=%d concurrent_managed=%d\n",
                    device, bus_id, props.name, props.major, props.minor,
                    static_cast<std::size_t>(props.totalGlobalMem >> 20), managed,
                    concurrent_managed);
    }
    for (int source = 0; source < count; ++source) {
        for (int destination = 0; destination < count; ++destination) {
            if (source == destination) { continue; }
            int access = 0;
            cudaDeviceCanAccessPeer(&access, source, destination);
            std::printf("peer %d->%d=%d\n", source, destination, access);
        }
    }

    if (count < 2) {
        std::printf("device_context: skipped, needs 2 devices\n");
        return 0;
    }

    bool peer_copy_ok = true;
    for (int source = 0; source < 2; ++source) {
        for (int destination = 0; destination < 2; ++destination) {
            if (source != destination) {
                peer_copy_ok = check_peer_copy(source, destination) && peer_copy_ok;
            }
        }
    }
    if (!peer_copy_ok) { return 3; }

    // The real construction. A bridgeless pair must reach here and report peer_access=0 rather
    // than throwing, which is exactly what this branch changed.
    try {
        const std::vector<int> ids{0, 1};
        ninfer::DeviceContext context(ids);
        std::printf("device_context=OK size=%zu model_parallel=%d peer_access=%d\n", context.size(),
                    context.model_parallel() ? 1 : 0, context.peer_access() ? 1 : 0);
        for (std::size_t rank = 0; rank < context.size(); ++rank) {
            std::printf("rank=%zu cuda=%d stream=%p fence=%p\n", rank, context.device_ids()[rank],
                        static_cast<void*>(context.stream_for_rank(rank)),
                        static_cast<void*>(context.fence_for_rank(rank)));
        }
        std::printf("device_context_active_rank=%zu\n", context.active_rank());
    } catch (const std::exception& error) {
        std::printf("device_context=FAILED %s\n", error.what());
        return 2;
    }
    return 0;
}
