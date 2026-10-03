// The pipeline split makes one workspace arena serve several devices by swapping its backing per
// rank. That keeps every existing call site unchanged, but it moves the bump pointer, capacity and
// peak into per-rank state -- and a scope opened on one rank can outlive a switch to another.
// Getting the rollback wrong hands out overlapping workspace, which corrupts silently, so the
// switching rules are worth pinning down here rather than discovering them in a model.
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Fn>
void check_throws(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    std::cerr << "FAIL (expected throw): " << message << '\n';
    ++failures;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

} // namespace

int main() {
    int device_count      = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(err) || device_count == 0) {
        std::cout << "skip: no CUDA device\n";
        return 77;
    }
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    // A plain arena must behave exactly as before: one rank, no extra state. This is what keeps
    // the single-GPU path untouched.
    {
        ninfer::DeviceArena arena(1 << 20);
        check(arena.rank_count() == 1, "a plain arena has one rank");
        check(arena.active_rank() == 0, "a plain arena is on rank 0");
        arena.activate_rank(0); // must be a no-op rather than an error
        check_throws([&] { arena.activate_rank(1); }, "activating a missing rank throws");
        check_throws([&] { (void)arena.peak_used_for_rank(1); }, "peak for a missing rank throws");
    }

    // Two ranks over two separate allocations. Rank 1 deliberately uses its own device memory, as
    // it would on a second card.
    void* second = nullptr;
    if (cudaMalloc(&second, 1 << 20) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        return 1;
    }

    {
        ninfer::DeviceArena arena(1 << 20);
        void* first_base = arena.base();
        arena.attach_rank_storage(ninfer::DeviceSpan{second, 1 << 20});

        check(arena.rank_count() == 2, "attaching gives two ranks");
        check(arena.active_rank() == 0, "still on rank 0 after attaching");
        check(arena.base() == first_base, "rank 0 keeps its original storage");

        // Allocations must come from the active rank's storage.
        const ninfer::DeviceSpan a = arena.alloc_bytes(4096);
        check(a.data == first_base, "rank 0 allocates from rank 0 storage");
        const std::size_t used_on_first = arena.used();
        check(used_on_first >= 4096, "rank 0 recorded the allocation");

        arena.activate_rank(1);
        check(arena.active_rank() == 1, "switched to rank 1");
        check(arena.base() == second, "rank 1 uses its own storage");
        check(arena.used() == 0, "rank 1 starts empty rather than inheriting rank 0's offset");
        const ninfer::DeviceSpan b = arena.alloc_bytes(8192);
        check(b.data == second, "rank 1 allocates from rank 1 storage");

        // Switching back must restore rank 0 exactly, not reset it.
        arena.activate_rank(0);
        check(arena.used() == used_on_first, "rank 0's bump pointer survived the round trip");
        check(arena.base() == first_base, "rank 0 storage restored");

        // Peaks are tracked per rank.
        check(arena.peak_used_for_rank(1) >= 8192, "rank 1 peak recorded");
        check(arena.peak_used_for_rank(0) >= 4096, "rank 0 peak recorded");

        // reset() is a round boundary and must clear every rank, or the next round leaks the
        // previous one's workspace on the other card.
        arena.reset();
        check(arena.used() == 0, "reset cleared rank 0");
        arena.activate_rank(1);
        check(arena.used() == 0, "reset cleared rank 1 too");
    }

    // The scope-across-a-switch case, which is what run_layers actually does: an enclosing scope
    // is opened on rank 0, the loop walks onto rank 1, and the scope closes while rank 1 is live.
    {
        ninfer::DeviceArena arena(1 << 20);
        arena.attach_rank_storage(ninfer::DeviceSpan{second, 1 << 20});

        (void)arena.alloc_bytes(1024);
        const std::size_t rank0_before = arena.used();
        {
            auto outer = arena.scope();
            (void)arena.alloc_bytes(4096);
            check(arena.used() > rank0_before, "inner allocation advanced rank 0");

            arena.activate_rank(1);
            (void)arena.alloc_bytes(2048);
            const std::size_t rank1_used = arena.used();
            check(rank1_used >= 2048, "rank 1 allocated inside the scope");
            // outer goes out of scope here, while rank 1 is active.
        }
        // The scope must have rolled back rank 0, and left rank 1 alone.
        check(arena.active_rank() == 1, "closing a scope does not change the active rank");
        check(arena.used() >= 2048, "rank 1's offset was not clobbered by rank 0's scope");
        arena.activate_rank(0);
        check(arena.used() == rank0_before, "rank 0 rolled back to the scope's entry offset");
    }

    // Block 2 left `arena` active on rank 1 -- attached, non-owning storage -- when it went out of
    // scope. The arena must have freed only its own rank-0 allocation, not `second`: a double free
    // here would make this cudaFree fail (or corrupt the allocator silently), not the destructor.
    check(cudaFree(second) == cudaSuccess,
         "destroying an arena left active on a borrowed rank freed storage it does not own");

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "arena rank tests passed\n";
    return 0;
}
