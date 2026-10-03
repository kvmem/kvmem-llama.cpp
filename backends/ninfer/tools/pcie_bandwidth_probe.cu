// Host <-> device transfer bandwidth, pageable and pinned, both directions.
//
// Why it exists. Model load time, host state slots and context-cache spill all cross the PCIe
// link, and some parts run it far below the x16 the slot suggests: the CMP 170HX ships at Gen1 x4
// (~0.85 GB/s measured) and a community unlock reports Gen2 x4. nvidia-smi says what was
// negotiated; this says what a copy actually gets, which is the number that sets the load time of
// a multi-GiB artifact.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 tools/pcie_bandwidth_probe.cu -o pcie_bandwidth_probe
// Run:
//   pcie_bandwidth_probe [--mib N]     (default 256; use a smaller N on a very slow link)

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e__ = (x);                                                                     \
        if (e__ != cudaSuccess) {                                                                  \
            std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e__), __LINE__);          \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

constexpr int kReps = 4;

// Best GB/s over kReps timed copies of `bytes`, after one warmup.
double time_copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
    CHECK(cudaMemcpy(dst, src, bytes, kind));
    double best = 0.0;
    for (int i = 0; i < kReps; ++i) {
        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a));
        CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));
        CHECK(cudaMemcpyAsync(dst, src, bytes, kind));
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        CHECK(cudaEventDestroy(a));
        CHECK(cudaEventDestroy(b));
        const double gbps = static_cast<double>(bytes) / (ms / 1000.0) / 1e9;
        if (gbps > best) { best = gbps; }
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    std::size_t mib = 256;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
            mib = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else {
            std::printf("usage: %s [--mib N]\n", argv[0]);
            return 2;
        }
    }
    const std::size_t bytes = mib << 20;

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU: %s  transfer size %zu MiB\n\n", p.name, mib);

    void* device = nullptr;
    CHECK(cudaMalloc(&device, bytes));
    std::vector<char> pageable(bytes, 1);
    void* pinned = nullptr;
    CHECK(cudaMallocHost(&pinned, bytes));
    std::memset(pinned, 1, bytes);

    const double h2d_page = time_copy(device, pageable.data(), bytes, cudaMemcpyHostToDevice);
    const double d2h_page = time_copy(pageable.data(), device, bytes, cudaMemcpyDeviceToHost);
    const double h2d_pin  = time_copy(device, pinned, bytes, cudaMemcpyHostToDevice);
    const double d2h_pin  = time_copy(pinned, device, bytes, cudaMemcpyDeviceToHost);

    std::printf("  %-28s %8.2f GB/s\n", "host->device, pageable", h2d_page);
    std::printf("  %-28s %8.2f GB/s\n", "device->host, pageable", d2h_page);
    std::printf("  %-28s %8.2f GB/s\n", "host->device, pinned", h2d_pin);
    std::printf("  %-28s %8.2f GB/s\n", "device->host, pinned", d2h_pin);

    // What the pinned rate means for the operations that cross the link.
    const double load_gib = 16.0;
    std::printf("\n  a %.0f GiB artifact upload at the pinned host->device rate: %.1f s\n", load_gib,
                load_gib * 1.073741824 / h2d_pin);
    // MB divided by GB/s is milliseconds.
    std::printf("  a 147 MiB state slot swap (one way) at the pinned rate:     %.0f ms\n",
                147.0 * 1.048576 / h2d_pin);

    CHECK(cudaFreeHost(pinned));
    CHECK(cudaFree(device));
    return 0;
}
