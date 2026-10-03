#include "core/arena.h"
#include "core/device.h"
#include "core/resident_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t MiB = 1024 * 1024;
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
}

int main(int argc, char** argv) {
#ifndef _WIN32
    std::cout << "SKIP: strict VRAM counters require Windows WDDM\n";
    return 77;
#else
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        CUDA_CHECK(status);
        CUDA_CHECK(cudaSetDevice(0));
        ninfer::core::ResidentMemoryConfig config;
        config.host_pool_bytes = 16 * MiB;
        const bool pressure = argc == 2 && std::string_view(argv[1]) == "--pressure-probe";
        config.reserve_bytes = (pressure ? 64 : 8) * MiB;
        config.probe_step_bytes = (pressure ? 128 : 4) * MiB;
        auto session = std::make_shared<ninfer::core::ResidentMemorySession>(config);
        const auto permanent_bytes = session->stats().device_allocated_bytes;
        std::unique_ptr<ninfer::DeviceBuffer> survivor;
        {
            ninfer::core::ResidentMemoryBinding binding(session);
            bool typed_oom = false;
            try { CUDA_CHECK(cudaErrorMemoryAllocation); }
            catch (const ninfer::CudaError& error) {
                typed_oom = error.status() == cudaErrorMemoryAllocation;
            }
            require(typed_oom, "strict CUDA_CHECK lost the recoverable OOM status");
            {
                ninfer::PinnedHostBuffer first(4 * MiB), second(4 * MiB);
                ninfer::PinnedHostBuffer moved(std::move(first));
                require(first.data() == nullptr, "moved host buffer retains storage");
                require(session->stats().host_used_bytes == 8 * MiB, "host move changed ownership");
                bool rejected = false;
                try { ninfer::PinnedHostBuffer excess(9 * MiB); }
                catch (const std::exception&) { rejected = true; }
                require(rejected, "exhausted host pool silently allocated additional host backing");
                require(session->stats().host_pool_bytes == 16 * MiB, "host pool unexpectedly grew");
            }
            require(session->stats().host_used_bytes == 0, "host allocations leaked");
            { ninfer::PinnedHostBuffer coalesced(16 * MiB); }
            {
                ninfer::DeviceArena owning(8 * MiB);
                const auto before_borrow = session->stats().device_allocated_bytes;
                {
                    ninfer::DeviceArena borrowed(ninfer::DeviceSpan{owning.base(), owning.capacity()});
                    ninfer::DeviceArena moved(std::move(owning));
                    CUDA_CHECK(cudaMemset(borrowed.base(), 0xa7, borrowed.capacity()));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    require(session->stats().device_allocated_bytes == before_borrow,
                            "arena move or borrowed view allocated extra memory");
                }
            }
            require(session->stats().device_allocated_bytes == permanent_bytes, "arena ownership leaked");
            {
                ninfer::DeviceBuffer source(2 * MiB), destination(3 * MiB);
                destination = std::move(source);
                require(source.p == nullptr && source.bytes == 0,
                        "move-assigned device buffer retains ownership");
                require(session->stats().device_allocated_bytes == permanent_bytes + 2 * MiB,
                        "move assignment did not release replaced device allocation");
            }
            survivor = std::make_unique<ninfer::DeviceBuffer>(8 * MiB + 17);
            std::vector<std::uint8_t> source(survivor->bytes), destination(source.size());
            for (std::size_t i = 0; i < source.size(); ++i) {
                source[i] = static_cast<std::uint8_t>((i * 37 + i / 257) & 255);
            }
            survivor->copy_from_host(source.data(), source.size());
            session->verify_working_set();
            survivor->copy_to_host(destination.data(), destination.size());
            require(source == destination, "working-set residency scan changed model bytes");
            require(session->stats().verified, "working set did not verify");
            require(session->stats().verified_reserve_bytes == config.reserve_bytes,
                    "reserve was not included in verification");
            require(session->probe_available_bytes(12 * MiB) == 12 * MiB,
                    "bounded probe did not recover its requested candidate");
            require(session->stats().device_allocated_bytes == survivor->bytes + permanent_bytes,
                    "temporary reserve or probe allocations leaked");
            if (pressure) {
                const auto before = session->stats();
                const auto capacity = session->probe_available_bytes(before.cuda_total_bytes);
                require(capacity > 0, "physical boundary probe returned no usable capacity");
                require(session->stats().device_allocated_bytes == survivor->bytes + permanent_bytes,
                        "physical boundary probe leaked temporary allocations");
                survivor->copy_to_host(destination.data(), destination.size());
                require(source == destination, "boundary cleanup corrupted retained model data");
                session->verify_working_set();
                std::cout << "Physical boundary: " << capacity / MiB << " MiB additional, initial CUDA free "
                          << before.cuda_free_bytes / MiB << " MiB; retained data unchanged\n";
            }
        }
        require(!ninfer::core::current_resident_memory(), "scoped allocator binding leaked");
        // The owner must survive the binding, and freeing must use the original session.
        survivor.reset();
        session->verify_working_set(false);
        require(session->stats().device_allocated_bytes == permanent_bytes, "out-of-scope buffer cleanup leaked");
        require(session->stats().shared_bytes <= session->stats().shared_baseline_bytes,
                "clean test left increased Shared usage");
        std::cout << "PASS: host pool, ownership, payload scan, reserve, probe and cleanup\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
#endif
}
