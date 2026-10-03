#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::core {
namespace {

// Volatile global loads force every byte of the backing to be visited without
// changing live weights, KV pages, or captured graph operands.
__global__ void scan_resident_bytes(const unsigned char* input, std::size_t bytes,
                                    unsigned int* checksum) {
    const std::size_t start = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    const auto* words = reinterpret_cast<const volatile unsigned long long*>(input);
    unsigned long long value = 0;
    for (std::size_t i = start; i < bytes / sizeof(*words); i += stride) {
        value ^= words[i];
    }
    const auto* tail = reinterpret_cast<const volatile unsigned char*>(input);
    for (std::size_t i = bytes / sizeof(*words) * sizeof(*words) + start;
         i < bytes; i += stride) {
        value ^= tail[i];
    }
    __shared__ unsigned int partial[256];
    partial[threadIdx.x] = unsigned(value) ^ unsigned(value >> 32);
    __syncthreads();
    for (unsigned int width = 128; width != 0; width >>= 1) {
        if (threadIdx.x < width) { partial[threadIdx.x] ^= partial[threadIdx.x + width]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicXor(checksum, partial[0]); }
}

} // namespace

cudaError_t launch_resident_scan(const void* pointer, std::size_t bytes, void* scratch) {
    if (bytes == 0) { return cudaSuccess; }
    scan_resident_bytes<<<256, 256>>>(static_cast<const unsigned char*>(pointer), bytes,
                                    static_cast<unsigned int*>(scratch));
    return cudaGetLastError();
}

} // namespace ninfer::core
