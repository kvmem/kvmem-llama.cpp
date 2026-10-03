// DeviceBuffer's host-to-device copy and fill must be visible to a non-blocking stream.
//
// Both are implemented with the synchronous CUDA runtime calls -- `cudaMemcpy` and `cudaMemset` --
// whose names suggest the work is finished when they return. Neither is. CUDA documents that a
// host-to-device copy out of *pageable* memory returns once the source has been staged for DMA
// "but the DMA to final destination may not have completed", and that a device memset is
// asynchronous with respect to the host. Both complete on the legacy stream, and every stream
// `DeviceContext` owns is created with `cudaStreamNonBlocking`, which is exempt from the legacy
// stream's implicit ordering.
//
// So without an explicit completion the sequence "stage inputs, then launch on the engine stream"
// is a race, and it is the sequence every Op test's setup uses. It cost real time: an
// `attn_input_proj` case at `T=112` failed in roughly a third of full-suite runs, always in the
// graph-replay phase that re-uploads its activation, always on every output at once -- the
// signature of one shared input being read early -- and never in isolation, because an idle box
// lands the DMA in time.
//
// This test reproduces that sequence directly and asserts the outcome, rather than asserting the
// implementation. Against an unsynchronized DeviceBuffer it reads stale data in roughly 90 of
// every 100 iterations on an idle RTX 3090.
//
// The window is narrow and easy to close by accident: an earlier draft cleared the counter with
// DeviceBuffer::fill between the copy and the stream work, and that single synchronous runtime
// call in the gap was enough to make the race vanish entirely. Keep the gap empty.
//
// The `fill` half below asserts the same contract for cudaMemset, which is documented as
// asynchronous with respect to the host in the same way. It has never actually caught a stale
// read on this driver -- the memset appears to complete before the call returns -- so treat it as
// pinning the contract rather than as a reproduction.

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;

namespace {

// Counts elements that still hold the value written before the copy under test.
__global__ void count_stale(const std::uint16_t* data, int count, std::uint16_t stale_value,
                            int* out) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (data[index] == stale_value) atomicAdd(out, 1);
}

// Sized as the case that exposed this: 112 tokens of a 5,120-wide BF16 activation, 1,146,880
// bytes. Copies at or below ~256 KiB were never observed to race on an RTX 3090; every size above
// it did, so a small buffer here would pass whether the bug is present or not.
constexpr int kElements = 112 * 5120;
constexpr std::uint16_t kStale = 0xAAAA;
constexpr std::uint16_t kFresh = 0xBBBB;

// One iteration wins the race far more often than it loses even when the bug is present. Ninety
// reproduced it in every trial while developing this; the observed rate was above 70%.
constexpr int kIterations = 90;

int check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return 0;
    std::printf("FAIL %s: %s\n", what, cudaGetErrorString(status));
    return 1;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    // A non-blocking stream, exactly as DeviceContext creates.
    cudaStream_t stream = nullptr;
    if (check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create"))
        return 1;

    DeviceBuffer payload(sizeof(std::uint16_t) * kElements);
    DeviceBuffer counter(sizeof(int));
    // Poison targets standing in for the outputs an Op test clears before launching.
    DeviceBuffer poison_a(sizeof(std::uint16_t) * 4096 * 112);
    DeviceBuffer poison_b(sizeof(std::uint16_t) * 1024 * 112);

    const std::vector<std::uint16_t> stale_host(kElements, kStale);
    const std::vector<std::uint16_t> fresh_host(kElements, kFresh);

    int copy_races = 0, fill_races = 0, worst = 0;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
        payload.copy_from_host(stale_host.data(), payload.bytes);
        counter.fill(0);
        if (check(cudaDeviceSynchronize(), "settle")) return 1;

        // The copy under test, followed immediately by work on the non-blocking stream. Nothing
        // may sit between the two: a synchronous runtime call in the gap closes the window the
        // test exists to open.
        payload.copy_from_host(fresh_host.data(), payload.bytes);
        if (check(cudaMemsetAsync(poison_a.p, 0xff, poison_a.bytes, stream), "poison a")) return 1;
        if (check(cudaMemsetAsync(poison_b.p, 0xff, poison_b.bytes, stream), "poison b")) return 1;
        count_stale<<<(kElements + 255) / 256, 256, 0, stream>>>(
            static_cast<const std::uint16_t*>(payload.p), kElements, kStale,
            static_cast<int*>(counter.p));
        if (check(cudaGetLastError(), "launch")) return 1;
        if (check(cudaStreamSynchronize(stream), "stream sync")) return 1;

        int stale = 0;
        counter.copy_to_host(&stale, sizeof(stale));
        if (stale > 0) {
            ++copy_races;
            if (stale > worst) worst = stale;
        }

        // The same question for fill(): clear the payload, then read it on the stream. Every
        // element must now be zero, so any element still holding kFresh is a lost memset. The
        // counter is cleared first, for the same reason as above.
        counter.fill(0);
        if (check(cudaDeviceSynchronize(), "settle fill")) return 1;
        payload.fill(0);
        count_stale<<<(kElements + 255) / 256, 256, 0, stream>>>(
            static_cast<const std::uint16_t*>(payload.p), kElements, kFresh,
            static_cast<int*>(counter.p));
        if (check(cudaGetLastError(), "fill launch")) return 1;
        if (check(cudaStreamSynchronize(stream), "fill stream sync")) return 1;
        counter.copy_to_host(&stale, sizeof(stale));
        if (stale > 0) ++fill_races;
    }

    (void)cudaStreamDestroy(stream);

    if (copy_races != 0 || fill_races != 0) {
        std::printf("FAIL device_buffer_visibility: over %d iterations, copy_from_host was read "
                    "stale %d times (worst %d of %d elements) and fill %d times\n",
                    kIterations, copy_races, worst, kElements, fill_races);
        return 1;
    }
    std::printf("OK device_buffer_visibility (%d iterations at %d bytes)\n", kIterations,
                int(sizeof(std::uint16_t) * kElements));
    return 0;
}
