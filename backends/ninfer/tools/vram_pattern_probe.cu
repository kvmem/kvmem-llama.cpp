// Fills nearly all of device memory with known patterns and reads them back, reporting mismatches
// per 1 GiB region.
//
// Why it exists. A community firmware unlock exposes 64 GB on a CMP 170HX that ships with 8 GB, and
// its own documentation warns that some of the extra HBM may have been disabled because it is
// defective rather than merely locked. A bad region does not crash anything: weights read back
// wrong and generation is silently degraded. This finds it in seconds, before any model is loaded.
//
// Four passes: a hash of the address, its complement (every bit flips, so stuck bits show), a second
// hash, and zeros. All regions are written first and verified afterwards, so each read comes after
// the whole memory has been touched. It tests what cudaMalloc hands back; regions are reported by
// allocation order, which follows the device address space closely but is not a bank map.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 tools/vram_pattern_probe.cu -o vram_pattern_probe
// Run:
//   vram_pattern_probe [--reserve-mib N]     (default 512 MiB left free for the context)
//
// Exit status is 0 only if every pass read back clean.

#include <cuda_runtime.h>

#include <cstdint>
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

constexpr std::size_t kRegionBytes = std::size_t{1} << 30;
constexpr int kThreads             = 256;

enum class Pass : int { HashA = 0, HashAComplement = 1, HashB = 2, Zeros = 3 };
constexpr int kPasses = 4;
const char* kPassName[kPasses] = {"hash A", "~hash A", "hash B", "zeros"};

__host__ __device__ inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// The value word `index` of region `region` must hold in `pass`. The region enters the hash so two
// regions never share a pattern, and a stale copy of another region's data is caught.
__host__ __device__ inline std::uint64_t expected(Pass pass, std::uint64_t region,
                                                  std::uint64_t index) {
    const std::uint64_t key = index ^ (region << 48);
    switch (pass) {
    case Pass::HashA: return splitmix64(key);
    case Pass::HashAComplement: return ~splitmix64(key);
    case Pass::HashB: return splitmix64(key ^ 0xa5a5a5a5a5a5a5a5ull);
    case Pass::Zeros: return 0ull;
    }
    return 0ull;
}

__global__ void fill_kernel(std::uint64_t* data, std::size_t words, Pass pass,
                            std::uint64_t region) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
         i += stride) {
        data[i] = expected(pass, region, i);
    }
}

// Counts mismatching words and records the lowest offending word index.
__global__ void verify_kernel(const std::uint64_t* data, std::size_t words, Pass pass,
                              std::uint64_t region, unsigned long long* error_count,
                              unsigned long long* first_bad) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    unsigned long long first = ~0ull;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
         i += stride) {
        if (data[i] != expected(pass, region, i)) {
            ++local;
            if (i < first) { first = i; }
        }
    }
    if (local != 0) {
        atomicAdd(error_count, local);
        atomicMin(first_bad, first);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::size_t reserve_mib = 512;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--reserve-mib") == 0 && i + 1 < argc) {
            reserve_mib = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else {
            std::printf("usage: %s [--reserve-mib N]\n", argv[0]);
            return 2;
        }
    }

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    std::size_t free_bytes = 0, total_bytes = 0;
    CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("GPU: %s  VRAM total %.2f GiB, free %.2f GiB\n", p.name,
                static_cast<double>(total_bytes) / (1 << 30),
                static_cast<double>(free_bytes) / (1 << 30));

    const std::size_t reserve = reserve_mib << 20;
    std::vector<void*> regions;
    std::vector<std::size_t> region_bytes;
    // Whole 1 GiB regions while they fit, then one final smaller region for the remainder.
    std::size_t remaining = free_bytes > reserve ? free_bytes - reserve : 0;
    while (remaining >= (std::size_t{64} << 20)) {
        // The kernels address whole 64-bit words.
        const std::size_t want = (remaining < kRegionBytes ? remaining : kRegionBytes) & ~std::size_t{7};
        void* ptr              = nullptr;
        if (cudaMalloc(&ptr, want) != cudaSuccess) {
            (void)cudaGetLastError();
            break;
        }
        regions.push_back(ptr);
        region_bytes.push_back(want);
        remaining -= want;
    }
    std::size_t tested = 0;
    for (std::size_t b : region_bytes) { tested += b; }
    std::printf("testing %.2f GiB in %zu regions (reserve %zu MiB)\n\n",
                static_cast<double>(tested) / (1 << 30), regions.size(), reserve_mib);
    if (regions.empty()) {
        std::printf("VERDICT vram: NOT TESTED (nothing allocatable)\n");
        return 1;
    }

    unsigned long long* counters = nullptr; // [region][2]: error count, first bad word
    CHECK(cudaMalloc(&counters, regions.size() * 2 * sizeof(unsigned long long)));
    const int blocks = p.multiProcessorCount * 8;

    std::vector<unsigned long long> total_errors(regions.size(), 0);
    std::vector<unsigned long long> first_bad(regions.size(), ~0ull);
    std::vector<int> first_bad_pass(regions.size(), -1);
    bool any_error = false;

    for (int pass = 0; pass < kPasses; ++pass) {
        const Pass which = static_cast<Pass>(pass);
        for (std::size_t r = 0; r < regions.size(); ++r) {
            fill_kernel<<<blocks, kThreads>>>(static_cast<std::uint64_t*>(regions[r]),
                                              region_bytes[r] / 8, which, r);
            CHECK(cudaGetLastError());
        }
        CHECK(cudaDeviceSynchronize());

        std::vector<unsigned long long> init(regions.size() * 2);
        for (std::size_t r = 0; r < regions.size(); ++r) {
            init[r * 2]     = 0;
            init[r * 2 + 1] = ~0ull;
        }
        CHECK(cudaMemcpy(counters, init.data(), init.size() * sizeof(unsigned long long),
                         cudaMemcpyHostToDevice));
        for (std::size_t r = 0; r < regions.size(); ++r) {
            verify_kernel<<<blocks, kThreads>>>(static_cast<const std::uint64_t*>(regions[r]),
                                                region_bytes[r] / 8, which, r, counters + r * 2,
                                                counters + r * 2 + 1);
            CHECK(cudaGetLastError());
        }
        CHECK(cudaDeviceSynchronize());

        std::vector<unsigned long long> result(regions.size() * 2);
        CHECK(cudaMemcpy(result.data(), counters, result.size() * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        unsigned long long pass_errors = 0;
        for (std::size_t r = 0; r < regions.size(); ++r) {
            pass_errors += result[r * 2];
            if (result[r * 2] != 0) {
                total_errors[r] += result[r * 2];
                if (first_bad_pass[r] < 0) {
                    first_bad_pass[r] = pass;
                    first_bad[r]      = result[r * 2 + 1];
                }
            }
        }
        std::printf("  pass %d (%-8s): %s%llu bad words\n", pass + 1, kPassName[pass],
                    pass_errors == 0 ? "clean, " : "", pass_errors);
        any_error = any_error || pass_errors != 0;
    }

    if (any_error) {
        std::printf("\nregion  size       bad words       first bad byte offset   (first seen in)\n");
        for (std::size_t r = 0; r < regions.size(); ++r) {
            if (total_errors[r] == 0) { continue; }
            std::printf("  %3zu   %5.2f GiB  %-14llu  %-22llu  %s\n", r,
                        static_cast<double>(region_bytes[r]) / (1 << 30), total_errors[r],
                        first_bad[r] * 8, kPassName[first_bad_pass[r]]);
        }
    }

    std::printf("\n");
    if (any_error) {
        std::printf("VERDICT vram: BAD   (%.2f GiB tested, errors in the regions above)\n",
                    static_cast<double>(tested) / (1 << 30));
    } else {
        std::printf("VERDICT vram: clean   (%.2f GiB tested, 4 passes, 0 errors)\n",
                    static_cast<double>(tested) / (1 << 30));
    }

    CHECK(cudaFree(counters));
    for (void* ptr : regions) { CHECK(cudaFree(ptr)); }
    return any_error ? 1 : 0;
}
