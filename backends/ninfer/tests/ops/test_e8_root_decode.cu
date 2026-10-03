#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer::test;

constexpr std::uint32_t kCodePairs = 65536;

__global__ void decode_all_code_pairs(uint2* table, uint2* reference) {
    const std::uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= kCodePairs) { return; }
    const std::uint32_t root        = pair >> 8;
    const std::uint32_t radius_axis = pair & 0xffu;
    table[pair]                     = ninfer::ops::kv_cache_e8_root_decode_block(root, radius_axis);
    reference[pair] = ninfer::ops::kv_cache_e8_root_decode_block_reference(root, radius_axis);
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        uint2* table     = nullptr;
        uint2* reference = nullptr;
        cuda_check(cudaMalloc(&table, kCodePairs * sizeof(uint2)), "cudaMalloc table");
        cuda_check(cudaMalloc(&reference, kCodePairs * sizeof(uint2)), "cudaMalloc reference");
        decode_all_code_pairs<<<kCodePairs / 256, 256>>>(table, reference);
        cuda_check_last_launch("decode_all_code_pairs");
        std::vector<uint2> table_host(kCodePairs);
        std::vector<uint2> reference_host(kCodePairs);
        cuda_check(cudaMemcpy(table_host.data(), table, kCodePairs * sizeof(uint2),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy table");
        cuda_check(cudaMemcpy(reference_host.data(), reference, kCodePairs * sizeof(uint2),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy reference");
        cuda_check(cudaFree(table), "cudaFree table");
        cuda_check(cudaFree(reference), "cudaFree reference");
        std::size_t failures = 0;
        for (std::uint32_t pair = 0; pair < kCodePairs; ++pair) {
            if (table_host[pair].x == reference_host[pair].x &&
                table_host[pair].y == reference_host[pair].y) {
                continue;
            }
            if (++failures <= 8) {
                std::cerr << "E8 decode mismatch at root " << (pair >> 8) << " radius_axis "
                          << (pair & 0xffu) << '\n';
            }
        }
        if (failures != 0) {
            std::cerr << failures << " E8 root code pairs decode differently from the reference\n";
            return 1;
        }
        std::cout << "E8 root table decode matches the reference for all " << kCodePairs
                  << " code pairs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E8 root decode test: " << error.what() << '\n';
        return 1;
    }
}
