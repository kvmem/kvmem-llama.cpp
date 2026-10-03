#include "core/resident_memory.h"

#include <cuda_runtime.h>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <pdh.h>
#    include <pdhmsg.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::core {

cudaError_t launch_resident_scan(const void*, std::size_t, void*);

namespace {
constexpr std::size_t kHostAlignment = 256;
constexpr std::size_t kScratchBytes = 4096;
thread_local std::shared_ptr<ResidentMemorySession> bound_session;

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("strict: ") + operation + ": " +
                                 cudaGetErrorName(result) + ": " + cudaGetErrorString(result));
    }
}

void cleanup_error(cudaError_t result, const char* operation) noexcept {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "strict cleanup %s: %s\n", operation,
                     cudaGetErrorString(result));
    }
}

std::size_t align_host(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - (kHostAlignment - 1)) {
        throw std::overflow_error("strict: host allocation size overflow");
    }
    return (bytes + kHostAlignment - 1) & ~(kHostAlignment - 1);
}

class CleanupDevice {
public:
    explicit CleanupDevice(int device) noexcept {
        if (cudaGetDevice(&previous_) == cudaSuccess && previous_ != device) {
            changed_ = cudaSetDevice(device) == cudaSuccess;
        }
    }
    ~CleanupDevice() { if (changed_) { cleanup_error(cudaSetDevice(previous_), "restore device"); } }
private:
    int previous_ = 0;
    bool changed_ = false;
};
} // namespace

struct ResidentMemorySession::Impl {
    ResidentMemoryConfig config;
    mutable std::mutex mutex;
    ResidentMemoryStats snapshot;
    void* host_pool = nullptr;
    void* scratch = nullptr;
    std::map<std::size_t, std::size_t> free_host_ranges;
    std::map<void*, std::pair<std::size_t, std::size_t>> host_allocations;
    std::map<void*, std::size_t> allocations;
#if defined(_WIN32)
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER shared_counter = nullptr;
    PDH_HCOUNTER dedicated_counter = nullptr;
    std::wstring instance_prefix;
#endif

    explicit Impl(const ResidentMemoryConfig& value) : config(value) {}
    ~Impl() {
        CleanupDevice select(config.device);
        // All allocation owners retain the session. Remaining entries only arise
        // during failed construction or exceptional cleanup.
        if (!allocations.empty() || scratch) {
            cleanup_error(cudaDeviceSynchronize(), "session sync");
        }
        for (const auto& item : allocations) { cleanup_error(cudaFree(item.first), "device free"); }
        if (scratch) { cleanup_error(cudaFree(scratch), "scratch free"); }
        if (host_pool) { cleanup_error(cudaFreeHost(host_pool), "host pool free"); }
#if defined(_WIN32)
        if (query) { PdhCloseQuery(query); }
#endif
    }

    void require_device() const {
        int active = -1;
        cuda_check(cudaGetDevice(&active), "get current device");
        if (active != config.device) {
            throw std::runtime_error("strict: current CUDA device differs from session device");
        }
    }

#if defined(_WIN32)
    std::size_t counter_value(PDH_HCOUNTER counter) const {
        DWORD bytes = 0, count = 0;
        const auto sized = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE,
                                                       &bytes, &count, nullptr);
        if (sized != PDH_MORE_DATA || bytes == 0) {
            throw std::runtime_error("strict: GPU memory counters unavailable; refusing unverified capacity");
        }
        std::vector<unsigned char> buffer(bytes);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &bytes, &count, items)
                != ERROR_SUCCESS) {
            throw std::runtime_error("strict: GPU memory counter read failed");
        }
        std::size_t sum = 0;
        unsigned int matched = 0;
        for (DWORD i = 0; i < count; ++i) {
            std::wstring name(items[i].szName ? items[i].szName : L"");
            std::transform(name.begin(), name.end(), name.begin(),
                           [](wchar_t value) { return static_cast<wchar_t>(std::towlower(value)); });
            if (name.compare(0, instance_prefix.size(), instance_prefix) != 0) { continue; }
            const auto& value = items[i].FmtValue;
            if ((value.CStatus != PDH_CSTATUS_VALID_DATA &&
                 value.CStatus != PDH_CSTATUS_NEW_DATA) || value.largeValue < 0) {
                throw std::runtime_error("strict: invalid matching GPU memory counter");
            }
            const auto current = static_cast<std::size_t>(value.largeValue);
            if (current > std::numeric_limits<std::size_t>::max() - sum) {
                throw std::runtime_error("strict: GPU memory counter overflow");
            }
            sum += current;
            ++matched;
        }
        if (matched == 0) {
            throw std::runtime_error("strict: no valid GPU memory counter for CUDA device LUID");
        }
        return sum;
    }
#endif

    void sample() {
        require_device();
        cuda_check(cudaMemGetInfo(&snapshot.cuda_free_bytes, &snapshot.cuda_total_bytes),
                   "query CUDA memory");
#if defined(_WIN32)
        // Both counters are read from the same query collection, for this PID and
        // the CUDA device LUID only. Other adapters cannot mask spill detection.
        if (PdhCollectQueryData(query) != ERROR_SUCCESS) {
            throw std::runtime_error("strict: GPU memory counter collection failed");
        }
        snapshot.shared_bytes = counter_value(shared_counter);
        snapshot.dedicated_bytes = counter_value(dedicated_counter);
#else
        throw std::runtime_error("strict currently requires Windows GPU memory counters");
#endif
    }

    void check(const char* stage) {
        try { sample(); }
        catch (...) {
            snapshot.verified = false;
            snapshot.verified_reserve_bytes = 0;
            throw;
        }
        if (snapshot.shared_bytes > snapshot.shared_baseline_bytes) {
            snapshot.verified = false;
            snapshot.verified_reserve_bytes = 0;
            throw VramCapacityError(std::string("strict: Shared GPU memory grew during ") +
                (stage ? stage : "unknown stage") + " from " +
                std::to_string(snapshot.shared_baseline_bytes) + " to " +
                std::to_string(snapshot.shared_bytes) + " bytes; rejecting device capacity",
                VramCapacityError::Reason::Shared);
        }
    }

    void scan_all() {
        require_device();
        cuda_check(cudaDeviceSynchronize(), "order working-set scan");
        cuda_check(cudaMemset(scratch, 0, sizeof(unsigned int)), "clear scan scratch");
        for (const auto& allocation : allocations) {
            cuda_check(launch_resident_scan(allocation.first, allocation.second, scratch),
                       "launch read-only working-set scan");
        }
        cuda_check(cudaDeviceSynchronize(), "complete working-set scan");
        unsigned int checksum = 0;
        cuda_check(cudaMemcpy(&checksum, scratch, sizeof(checksum), cudaMemcpyDeviceToHost),
                   "read scan checksum");
    }

    void* allocate(std::size_t bytes) {
        require_device();
        if (bytes == 0) { return nullptr; }
        if (bytes > std::numeric_limits<std::size_t>::max() - snapshot.device_allocated_bytes) {
            throw std::overflow_error("strict: device allocation size overflow");
        }
        void* pointer = nullptr;
        const cudaError_t allocated = cudaMalloc(&pointer, bytes);
        if (allocated == cudaErrorMemoryAllocation) {
            const auto pending = cudaGetLastError();
            if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) {
                cuda_check(pending, "pending error after allocation failure");
            }
            snapshot.verified = false;
            snapshot.verified_reserve_bytes = 0;
            throw VramCapacityError("strict: cudaMalloc could not allocate " +
                                    std::to_string(bytes) + " device bytes",
                                    VramCapacityError::Reason::Allocation);
        }
        cuda_check(allocated, "allocate device memory");
        try {
            cuda_check(cudaMemset(pointer, 0, bytes), "touch entire device allocation");
            cuda_check(cudaDeviceSynchronize(), "complete device allocation touch");
            check("device allocation touch");
            allocations.emplace(pointer, bytes);
        } catch (...) {
            cleanup_error(cudaFree(pointer), "reject device allocation");
            throw;
        }
        snapshot.device_allocated_bytes += bytes;
        snapshot.verified = false;
        snapshot.verified_reserve_bytes = 0;
        return pointer;
    }

    void release(void* pointer) noexcept {
        if (!pointer) { return; }
        const auto entry = allocations.find(pointer);
        if (entry == allocations.end()) {
            std::fprintf(stderr, "strict: attempted to free an unknown device allocation\n");
            return;
        }
        CleanupDevice select(config.device);
        const auto result = cudaFree(pointer);
        cleanup_error(result, "device free");
        if (result == cudaSuccess) {
            snapshot.device_allocated_bytes -= entry->second;
            allocations.erase(entry);
        }
    }

    void initialize() {
#if !defined(_WIN32)
        throw std::runtime_error("strict currently requires Windows GPU memory counters");
#else
        if (config.probe_step_bytes == 0) {
            throw std::invalid_argument("strict: probe step must be nonzero");
        }
        require_device();
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, config.device), "query device properties");
        LUID luid{};
        std::memcpy(&luid, properties.luid, sizeof(luid));
        if (luid.HighPart == 0 && luid.LowPart == 0) {
            throw std::runtime_error("strict: CUDA device has no Windows adapter LUID");
        }
        wchar_t prefix[160]{};
        std::swprintf(prefix, sizeof(prefix) / sizeof(prefix[0]),
                      L"pid_%lu_luid_0x%08x_0x%08x_phys_", GetCurrentProcessId(),
                      static_cast<unsigned int>(luid.HighPart), luid.LowPart);
        instance_prefix = prefix;

        // Reserve host backing while the device is empty. The entire pool is CPU
        // touched, and HtoD/DtoH DMA is warmed before establishing the baseline.
        if (config.host_pool_bytes != 0) {
            config.host_pool_bytes = align_host(config.host_pool_bytes);
            cuda_check(cudaHostAlloc(&host_pool, config.host_pool_bytes, cudaHostAllocPortable),
                       "allocate fixed pinned host pool");
            std::memset(host_pool, 0, config.host_pool_bytes);
            free_host_ranges.emplace(0, config.host_pool_bytes);
        }
        snapshot.host_pool_bytes = config.host_pool_bytes;
        cuda_check(cudaMalloc(&scratch, kScratchBytes), "allocate scan scratch");
        cuda_check(cudaMemset(scratch, 0, kScratchBytes), "warm CUDA allocation");
        if (host_pool) {
            // CUDA/WDDM can defer host mapping/accounting until a range is used.
            // Warm DMA across the entire pool before the baseline, using only a
            // small temporary device block that is released before model loading.
            const auto window = std::min<std::size_t>(8ULL << 20, config.host_pool_bytes);
            void* warm = nullptr;
            cuda_check(cudaMalloc(&warm, window), "allocate host DMA warmup window");
            try {
                for (std::size_t offset = 0; offset < config.host_pool_bytes; offset += window) {
                    const auto bytes = std::min(window, config.host_pool_bytes - offset);
                    auto* host = static_cast<unsigned char*>(host_pool) + offset;
                    cuda_check(cudaMemcpy(warm, host, bytes, cudaMemcpyHostToDevice),
                               "warm full host pool HtoD");
                    cuda_check(cudaMemcpy(host, warm, bytes, cudaMemcpyDeviceToHost),
                               "warm full host pool DtoH");
                }
                cuda_check(cudaDeviceSynchronize(), "complete full host pool DMA warmup");
                cuda_check(cudaFree(warm), "release host DMA warmup window");
                warm = nullptr;
            } catch (...) {
                if (warm) { cleanup_error(cudaFree(warm), "failed host DMA warmup"); }
                throw;
            }
        }
        cuda_check(launch_resident_scan(static_cast<unsigned char*>(scratch) + 256,
                                       kScratchBytes - 256, scratch), "warm scan kernel module");
        cuda_check(cudaDeviceSynchronize(), "finish strict warmup");
        snapshot.device_allocated_bytes = kScratchBytes;

        if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) {
            throw std::runtime_error("strict: could not open GPU memory counters");
        }
        const std::wstring path = L"\\GPU Process Memory(pid_" +
                                 std::to_wstring(GetCurrentProcessId()) + L"_*)\\";
        if (PdhAddEnglishCounterW(query, (path + L"Shared Usage").c_str(), 0,
                                 &shared_counter) != ERROR_SUCCESS ||
            PdhAddEnglishCounterW(query, (path + L"Dedicated Usage").c_str(), 0,
                                 &dedicated_counter) != ERROR_SUCCESS ||
            PdhCollectQueryData(query) != ERROR_SUCCESS) {
            throw std::runtime_error("strict: could not initialize GPU memory counters");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        sample();
        auto baseline = snapshot.shared_bytes;
        for (int i = 0; i < 2; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            sample();
            if (snapshot.shared_bytes != baseline) {
                throw std::runtime_error("strict: Shared baseline did not stabilize before model allocation");
            }
        }
        snapshot.shared_baseline_bytes = baseline;
#endif
    }
};

ResidentMemorySession::ResidentMemorySession(const ResidentMemoryConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
    impl_->initialize();
}
ResidentMemorySession::~ResidentMemorySession() = default;

void* ResidentMemorySession::allocate_device(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->allocate(bytes);
}

void ResidentMemorySession::free_device(void* pointer) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->release(pointer);
}

void* ResidentMemorySession::allocate_host(std::size_t bytes) {
    if (bytes == 0) { return nullptr; }
    const auto rounded = align_host(bytes);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto range = impl_->free_host_ranges.begin(); range != impl_->free_host_ranges.end(); ++range) {
        if (range->second < rounded) { continue; }
        const auto offset = range->first;
        const auto remainder = range->second - rounded;
        auto* pointer = static_cast<unsigned char*>(impl_->host_pool) + offset;
        // Insert all potentially allocating map nodes before changing the free list.
        impl_->host_allocations.emplace(pointer, std::make_pair(offset, rounded));
        try {
            if (remainder) { impl_->free_host_ranges.emplace(offset + rounded, remainder); }
        } catch (...) {
            impl_->host_allocations.erase(pointer);
            throw;
        }
        impl_->free_host_ranges.erase(range);
        impl_->snapshot.host_used_bytes += rounded;
        return pointer;
    }
    throw std::runtime_error("strict: fixed pinned host pool exhausted while requesting " +
        std::to_string(bytes) + " bytes (used " + std::to_string(impl_->snapshot.host_used_bytes) +
        " of " + std::to_string(impl_->snapshot.host_pool_bytes) +
        "); increase cuda host pool capacity; late pinned allocations are prohibited");
}

void ResidentMemorySession::free_host(void* pointer) noexcept {
    if (!pointer) { return; }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto allocation = impl_->host_allocations.find(pointer);
    if (allocation == impl_->host_allocations.end()) {
        std::fprintf(stderr, "strict: attempted to free unknown host allocation\n");
        return;
    }
    const auto offset = allocation->second.first;
    const auto bytes = allocation->second.second;
    // cudaFreeHost used to end this backing's lifetime. With a reusable pool,
    // complete any DMA before making the subrange available to another owner.
    CleanupDevice select(impl_->config.device);
    const auto completed = cudaDeviceSynchronize();
    if (completed != cudaSuccess) {
        cleanup_error(completed, "host pool reuse sync");
        impl_->snapshot.verified = false;
        impl_->snapshot.verified_reserve_bytes = 0;
        return;
    }
    // Retain bookkeeping rather than terminate if a host bookkeeping allocation
    // fails during noexcept cleanup. The pool still releases as one owned block.
    try {
        auto entry = impl_->free_host_ranges.emplace(offset, bytes).first;
        if (entry != impl_->free_host_ranges.begin()) {
            auto previous = std::prev(entry);
            if (previous->first + previous->second == entry->first) {
                previous->second += entry->second;
                impl_->free_host_ranges.erase(entry);
                entry = previous;
            }
        }
        const auto next = std::next(entry);
        if (next != impl_->free_host_ranges.end() && entry->first + entry->second == next->first) {
            entry->second += next->second;
            impl_->free_host_ranges.erase(next);
        }
        impl_->snapshot.host_used_bytes -= bytes;
        impl_->host_allocations.erase(allocation);
    } catch (...) {
        std::fprintf(stderr, "strict: host free-list bookkeeping failed; pool block retained\n");
    }
}

void ResidentMemorySession::check(const char* stage) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check(stage);
}

void ResidentMemorySession::verify_working_set(bool with_reserve) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->snapshot.verified = false;
    impl_->snapshot.verified_reserve_bytes = 0;
    void* reserve = nullptr;
    try {
        if (with_reserve && impl_->config.reserve_bytes) {
            reserve = impl_->allocate(impl_->config.reserve_bytes);
        }
        // Visit every allocation again while the reserve block remains alive.
        // Detect displacement of old allocations by the most recently added one.
        impl_->scan_all();
        impl_->check("joint working-set verification");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        impl_->check("working-set verification hold");
        if (reserve) { impl_->release(reserve); reserve = nullptr; }
        impl_->check("after reserve release");
        impl_->snapshot.verified = true;
        if (with_reserve) { impl_->snapshot.verified_reserve_bytes = impl_->config.reserve_bytes; }
    } catch (...) {
        if (reserve) { impl_->release(reserve); }
        throw;
    }
}

std::size_t ResidentMemorySession::probe_available_bytes(std::size_t max_bytes) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check("before capacity probe");
    std::vector<void*> blocks;
    std::size_t held = 0;
    const auto step = impl_->config.probe_step_bytes;
    const auto reported_free = impl_->snapshot.cuda_free_bytes;
    const auto initial = impl_->snapshot.cuda_free_bytes > step ?
                         impl_->snapshot.cuda_free_bytes - step : step;
    auto amount = std::min(initial, max_bytes);
    unsigned int initial_backoffs = 0;
    auto cleanup = [&]() {
        for (auto block = blocks.rbegin(); block != blocks.rend(); ++block) { impl_->release(*block); }
        blocks.clear();
    };
    try {
        while (amount != 0 && held < max_bytes) {
            const auto previous_held = held;
            try {
                void* block = impl_->allocate(amount);
                try { blocks.push_back(block); }
                catch (...) { impl_->release(block); throw; }
                held += amount;
                // Sampling after all older blocks have been revisited is necessary:
                // a new allocation can evict old data instead of evicting itself.
                impl_->scan_all();
                impl_->check("capacity probe working set");
            } catch (const VramCapacityError& error) {
                // A block whose allocation succeeded but displaced an older block
                // is not part of the successfully verified candidate.
                held = previous_held;
                if (held == 0 && blocks.empty() && amount > step &&
                    error.reason() == VramCapacityError::Reason::Allocation) {
                    // A fragmented device may reject the initial large block even
                    // though smaller ones fit. Halve to a step-aligned first block;
                    // after at most seven reductions try one step, then stop.
                    ++initial_backoffs;
                    const auto smaller = initial_backoffs >= 7 ? step :
                                         std::max(step, (amount / 2 / step) * step);
                    std::fprintf(stderr, "strict: initial probe allocation failed; retrying %zu bytes instead of %zu\n",
                                 smaller, amount);
                    impl_->scan_all();
                    impl_->check("retained allocations after initial probe OOM");
                    amount = smaller;
                    continue;
                }
                std::fprintf(stderr,
                    "strict: probe boundary rejected %zu bytes after %zu verified bytes: %s\n",
                    amount, held, error.what());
                break;
            }
            amount = std::min(step, max_bytes - held);
        }
        cleanup();
        // A failed boundary trial can momentarily displace retained weights. All
        // probes are gone now; revisit weights and allow bounded counter settling
        // without ever raising the baseline. Persistent growth remains a failure.
        for (int attempt = 0; ; ++attempt) {
            impl_->scan_all();
            try {
                impl_->check("retained allocations after probe cleanup");
                break;
            } catch (const VramCapacityError&) {
                if (attempt == 2) { throw; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }
    std::fprintf(stderr,
        "strict: probe verified %zu additional bytes (initial CUDA free %zu, step %zu); "
        "temporary blocks released and retained working set rechecked\n",
        held, reported_free, step);
    return held;
}

ResidentMemoryStats ResidentMemorySession::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->snapshot;
}

ResidentMemoryBinding::ResidentMemoryBinding(std::shared_ptr<ResidentMemorySession> session)
    : previous_(std::move(bound_session)) {
    bound_session = std::move(session);
}
ResidentMemoryBinding::~ResidentMemoryBinding() { bound_session = std::move(previous_); }
std::shared_ptr<ResidentMemorySession> current_resident_memory() { return bound_session; }

} // namespace ninfer::core
