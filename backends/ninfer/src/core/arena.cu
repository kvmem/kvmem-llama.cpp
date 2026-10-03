#include "core/arena.h"
#include "core/resident_memory.h"

#include <cuda_runtime.h>

#if defined(_WIN32) && defined(NINFER_D3D12_RESIDENCY) && NINFER_D3D12_RESIDENCY
#    define NINFER_WDDM_RESIDENCY 1
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <d3d12.h>
#    include <dxgi1_6.h>
#    include <wrl/client.h>

#    include <algorithm>
#    include <atomic>
#    include <cstring>
#    include <mutex>
#    include <vector>
#else
#    define NINFER_WDDM_RESIDENCY 0
#endif

#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
    if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
    const std::size_t mask         = align - 1;
    const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
    return with_mask & ~static_cast<std::uintptr_t>(mask);
}

void free_device(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFree", cudaFree(ptr));
        ptr = nullptr;
    }
}

void free_pinned(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFreeHost", cudaFreeHost(ptr));
        ptr = nullptr;
    }
}

#if NINFER_WDDM_RESIDENCY
// A D3D12 heap imported into CUDA, after UDPSendToFailed/ninfer-4090 (a35acf6a and its follow-ups).
// It lives as long as the arena it backs: free_arena releases the mapping, the external memory and
// the heap together.
struct D3D12ResidentHeap {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device3> device;
    Microsoft::WRL::ComPtr<ID3D12Heap> heap;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    cudaExternalMemory_t memory = nullptr;
    void* pointer               = nullptr;
};

std::atomic<bool> g_wddm_residency_lock{false};
std::mutex g_resident_heaps_mutex;
std::vector<D3D12ResidentHeap> g_resident_heaps;

// WDDM keeps this much of the adapter for the desktop compositor's scanout.
constexpr UINT64 kDwmScanoutFloor          = 512ULL * 1024ULL * 1024ULL;
constexpr DWORD kResidencyWaitMilliseconds = 10000;

void release_mapping(void* pointer, cudaExternalMemory_t memory) noexcept {
    if (pointer != nullptr) { (void)cudaFree(pointer); }
    if (memory != nullptr) { (void)cudaDestroyExternalMemory(memory); }
    (void)cudaGetLastError();
}

// Allocate the arena from a D3D12 heap on the adapter of the current CUDA device, resident at the
// maximum priority and denied over-budget paging, and imported into CUDA. Any failure returns
// null and leaves the CUDA context clean, so the caller falls back to cudaMalloc.
void* allocate_resident_heap(std::size_t capacity_bytes) {
    D3D12ResidentHeap heap;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&heap.factory)))) { return nullptr; }

    int device_id = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device_id) != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }
    LUID cuda_luid{};
    std::memcpy(&cuda_luid, properties.luid, sizeof(LUID));
    Microsoft::WRL::ComPtr<IDXGIAdapter1> candidate;
    for (UINT i = 0; heap.factory->EnumAdapters1(i, &candidate) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 description{};
        candidate->GetDesc1(&description);
        if (description.AdapterLuid.LowPart == cuda_luid.LowPart &&
            description.AdapterLuid.HighPart == cuda_luid.HighPart) {
            heap.adapter = candidate;
            break;
        }
    }
    if (!heap.adapter) { return nullptr; }

    Microsoft::WRL::ComPtr<ID3D12Device> base_device;
    if (FAILED(D3D12CreateDevice(heap.adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                 IID_PPV_ARGS(&base_device))) ||
        FAILED(base_device.As(&heap.device))) {
        return nullptr;
    }

    constexpr std::size_t kHeapAlignment = 64 * 1024;
    const std::size_t heap_bytes =
        (capacity_bytes + kHeapAlignment - 1) / kHeapAlignment * kHeapAlignment;
    // Never import more than the adapter can back: an unbacked heap faults inside CUDA, and the
    // sticky fault would also break the cudaMalloc fallback.
    DXGI_ADAPTER_DESC1 adapter_description{};
    if (SUCCEEDED(heap.adapter->GetDesc1(&adapter_description))) {
        const UINT64 usable = adapter_description.DedicatedVideoMemory > kDwmScanoutFloor
                                  ? adapter_description.DedicatedVideoMemory - kDwmScanoutFloor
                                  : 0;
        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
        DXGI_QUERY_VIDEO_MEMORY_INFO memory_info{};
        if (SUCCEEDED(heap.adapter.As(&adapter3)) &&
            SUCCEEDED(
                adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory_info)) &&
            memory_info.CurrentUsage + heap_bytes > usable) {
            return nullptr;
        }
    }

    D3D12_HEAP_DESC description{};
    description.SizeInBytes                     = heap_bytes;
    description.Properties.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    description.Properties.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    description.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    description.Alignment                       = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    description.Flags                           = D3D12_HEAP_FLAG_SHARED;
    if (FAILED(heap.device->CreateHeap(&description, IID_PPV_ARGS(&heap.heap)))) { return nullptr; }

    ID3D12Pageable* pageable                = heap.heap.Get();
    const D3D12_RESIDENCY_PRIORITY priority = D3D12_RESIDENCY_PRIORITY_MAXIMUM;
    (void)heap.device->SetResidencyPriority(1, &pageable, &priority);
    if (FAILED(heap.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&heap.fence))) ||
        FAILED(heap.device->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 1, &pageable,
                                                heap.fence.Get(), 1))) {
        return nullptr;
    }
    // Residency completes asynchronously; touching the heap before the fence signals would write
    // into pages that are not yet backed.
    if (heap.fence->GetCompletedValue() < 1) {
        HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (event == nullptr) { return nullptr; }
        const bool resident =
            SUCCEEDED(heap.fence->SetEventOnCompletion(1, event)) &&
            WaitForSingleObject(event, kResidencyWaitMilliseconds) == WAIT_OBJECT_0;
        CloseHandle(event);
        if (!resident) { return nullptr; }
    }

    HANDLE shared = nullptr;
    if (FAILED(heap.device->CreateSharedHandle(heap.heap.Get(), nullptr, GENERIC_ALL, nullptr,
                                               &shared))) {
        return nullptr;
    }
    cudaExternalMemoryHandleDesc handle{};
    handle.type                = cudaExternalMemoryHandleTypeD3D12Heap;
    handle.handle.win32.handle = shared;
    handle.size                = heap_bytes;
    handle.flags               = 0; // a heap is suballocatable, not a dedicated resource
    const cudaError_t imported = cudaImportExternalMemory(&heap.memory, &handle);
    CloseHandle(shared);
    if (imported != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }

    cudaExternalMemoryBufferDesc buffer{};
    buffer.offset = 0;
    buffer.size   = capacity_bytes;
    buffer.flags  = 0;
    void* pointer = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&pointer, heap.memory, &buffer) != cudaSuccess) {
        release_mapping(nullptr, heap.memory);
        return nullptr;
    }

    // Commit every page now and check that the mapping round-trips: a heap can import cleanly and
    // still read back wrong, which downstream is silent output corruption rather than an error.
    constexpr std::size_t kProbeBytes = 4096;
    const std::size_t probe           = capacity_bytes < kProbeBytes ? capacity_bytes : kProbeBytes;
    std::vector<unsigned char> pattern(probe);
    for (std::size_t i = 0; i < probe; ++i) { pattern[i] = static_cast<unsigned char>(i * 31 + 7); }
    std::vector<unsigned char> readback(probe, 0);
    auto* bytes            = static_cast<unsigned char*>(pointer);
    const std::size_t tail = capacity_bytes - probe;
    const bool coherent =
        cudaMemset(pointer, 0, capacity_bytes) == cudaSuccess &&
        cudaMemcpy(bytes + tail, pattern.data(), probe, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(readback.data(), bytes + tail, probe, cudaMemcpyDeviceToHost) == cudaSuccess &&
        std::memcmp(pattern.data(), readback.data(), probe) == 0 &&
        cudaMemset(pointer, 0, capacity_bytes) == cudaSuccess &&
        // The fill runs on the legacy default stream, which does not order against the
        // non-blocking load stream: without this wait a weight upload could land before it.
        cudaDeviceSynchronize() == cudaSuccess;
    if (!coherent) {
        std::fprintf(stderr,
                     "D3D12 resident arena of %zu bytes failed its read-back check; "
                     "using cudaMalloc\n",
                     capacity_bytes);
        release_mapping(pointer, heap.memory);
        return nullptr;
    }

    heap.pointer = pointer;
    std::lock_guard lock(g_resident_heaps_mutex);
    g_resident_heaps.push_back(std::move(heap));
    return pointer;
}
#endif

// Free an owning arena's storage, with the D3D12 heap behind it when it came from one.
void free_arena(void*& ptr) noexcept {
#if NINFER_WDDM_RESIDENCY
    if (ptr != nullptr) {
        std::lock_guard lock(g_resident_heaps_mutex);
        const auto found =
            std::find_if(g_resident_heaps.begin(), g_resident_heaps.end(),
                         [ptr](const D3D12ResidentHeap& heap) { return heap.pointer == ptr; });
        if (found != g_resident_heaps.end()) {
            release_mapping(found->pointer, found->memory);
            g_resident_heaps.erase(found);
            ptr = nullptr;
            return;
        }
    }
#endif
    free_device(ptr);
}

} // namespace

namespace core {

#if NINFER_WDDM_RESIDENCY
void set_wddm_residency_lock_enabled(bool enabled) {
    g_wddm_residency_lock.store(enabled, std::memory_order_relaxed);
}

bool wddm_residency_lock_enabled() noexcept {
    return g_wddm_residency_lock.load(std::memory_order_relaxed);
}
#else
void set_wddm_residency_lock_enabled(bool enabled) {
    if (enabled) {
        throw std::invalid_argument(
            "--wddm-evictable-budget needs a Windows build with NINFER_D3D12_RESIDENCY=ON");
    }
}

bool wddm_residency_lock_enabled() noexcept { return false; }
#endif

} // namespace core

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }
    resident_session_ = core::current_resident_memory();
    if (resident_session_) {
        p = resident_session_->allocate_device(bytes);
        return;
    }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }
    p = ptr;
}

DeviceBuffer::~DeviceBuffer() {
    if (resident_session_) { resident_session_->free_device(p); }
    else { free_device(p); }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : p(other.p), bytes(other.bytes), resident_session_(std::move(other.resident_session_)) {
    other.p     = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    if (resident_session_) { resident_session_->free_device(p); }
    else { free_device(p); }
    resident_session_ = std::move(other.resident_session_);
    p     = other.p;
    bytes = other.bytes;

    other.p     = nullptr;
    other.bytes = 0;
    return *this;
}

// The same contract for a fill: CUDA documents `cudaMemset` on device memory as asynchronous with
// respect to the host, and it completes on the legacy stream, which a non-blocking stream does not
// wait for. Unlike `copy_from_host` below this one has *not* been observed to race on this box --
// tests/test_device_buffer_visibility.cu exercises it 90 times per run and has never caught it, so
// on this driver the memset appears to complete before the call returns. That is an implementation
// detail rather than a guarantee, and the sync costs nothing on a path every caller uses at setup,
// so make the contract explicit rather than depend on the observation.
void DeviceBuffer::fill(int byte_value) {
    if (bytes == 0) { return; }
    const cudaError_t err = cudaMemset(p, byte_value, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
    }
    const cudaError_t sync = cudaStreamSynchronize(nullptr);
    if (sync != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("fill completion sync failed", sync));
    }
}

// `cudaMemcpy` does not finish the transfer before it returns. CUDA's documented
// synchronization behaviour for a host-to-device copy out of *pageable* memory is that the call
// returns once the source has been staged for DMA, "but the DMA to final destination may not have
// completed". That trailing DMA rides the legacy stream, and a stream created with
// `cudaStreamNonBlocking` -- which is every stream `DeviceContext` owns -- is exempt from the
// legacy stream's implicit ordering. So work enqueued on one of those streams immediately after
// this call can, and does, read the destination before the copy lands.
//
// Measured on this RTX 3090 with a faithful replica of the sequence
// tests/ops/test_attn_input_proj.cpp runs at graph replay phase 1 -- a 1,146,880-byte pageable
// H2D, three `cudaMemsetAsync` on a non-blocking stream, then a captured graph launched on it --
// the reader saw stale bytes in 146 of 200 iterations, up to 17.8% of the buffer, on an
// *otherwise idle* GPU. Copies at or below ~256 KiB never raced; every size above it did.
// tests/test_device_buffer_visibility.cu is that experiment, reduced.
//
// Synchronizing the legacy stream here costs one round trip on a path that is load-time setup in
// every caller, and makes the method mean what its name says: on return, the bytes are visible to
// any stream. Callers wanting an overlapped copy should use an explicitly stream-ordered one
// (`cudaMemcpy2DAsync`, as the paged-KV and state-image pools already do) rather than this.
void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
    require_range(byte_offset, count, "host-to-device copy");
    if (count == 0) { return; }
    void* destination     = static_cast<std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy host-to-device failed", err));
    }
    const cudaError_t sync = cudaStreamSynchronize(nullptr);
    if (sync != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("host-to-device copy completion sync failed", sync));
    }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count,
                                std::size_t byte_offset) const {
    require_range(byte_offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const void* source    = static_cast<const std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy device-to-host failed", err));
    }
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count,
                                 const char* operation) const {
    if (byte_offset > bytes || count > bytes - byte_offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.off_), saved_rank_(arena.active_rank_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ == nullptr) { return; }
    if (saved_rank_ == arena_->active_rank_) {
        if (saved_offset_ <= arena_->off_) { arena_->off_ = saved_offset_; }
        return;
    }
    // The arena moved to another rank while this scope was open. Roll back the rank the scope was
    // actually taken on, not whichever one happens to be active now -- restoring a foreign bump
    // pointer would hand out overlapping workspace on the next allocation.
    if (saved_rank_ < arena_->ranks_.size()) {
        RankBacking& backing = arena_->ranks_[saved_rank_];
        if (saved_offset_ <= backing.off) { backing.off = saved_offset_; }
    }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_), saved_rank_(other.saved_rank_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    resident_session_ = core::current_resident_memory();
    if (resident_session_) {
        base_ = resident_session_->allocate_device(capacity_bytes);
        owned_base_ = base_;
        cap_ = capacity_bytes;
        return;
    }
    void* ptr = nullptr;
#if NINFER_WDDM_RESIDENCY
    if (core::wddm_residency_lock_enabled()) { ptr = allocate_resident_heap(capacity_bytes); }
#endif
    if (ptr == nullptr) {
        const cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
        }
#if NINFER_WDDM_RESIDENCY
        // Under the evictable budget, commit the pages now so WDDM makes room at startup; the wait
        // orders the fill before uploads on the non-blocking load stream.
        if (core::wddm_residency_lock_enabled()) {
            const cudaError_t fill = cudaMemset(ptr, 0, capacity_bytes);
            const cudaError_t done = fill == cudaSuccess ? cudaDeviceSynchronize() : fill;
            if (done != cudaSuccess) {
                log_cuda_error("cudaFree", cudaFree(ptr));
                throw std::runtime_error(cuda_error_message("cudaMemset failed", done));
            }
        }
#endif
    }

    base_       = ptr;
    owned_base_ = ptr;
    cap_        = capacity_bytes;
    off_        = 0;
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), cap_(storage.bytes), owns_(false) {
    if (base_ == nullptr || cap_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() {
    if (owns_) {
        if (resident_session_) { resident_session_->free_device(owned_base_); }
        else { free_arena(owned_base_); }
    }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_), owned_base_(other.owned_base_),
      resident_session_(std::move(other.resident_session_)), ranks_(std::move(other.ranks_)),
      active_rank_(other.active_rank_) {
    other.base_        = nullptr;
    other.cap_         = 0;
    other.off_         = 0;
    other.peak_        = 0;
    other.owns_        = true;
    other.owned_base_  = nullptr;
    other.ranks_.clear();
    other.active_rank_ = 0;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    if (owns_) {
        if (resident_session_) { resident_session_->free_device(owned_base_); }
        else { free_arena(owned_base_); }
    }
    resident_session_ = std::move(other.resident_session_);
    base_       = other.base_;
    cap_        = other.cap_;
    off_        = other.off_;
    peak_       = other.peak_;
    owns_       = other.owns_;
    owned_base_ = other.owned_base_;

    ranks_       = std::move(other.ranks_);
    active_rank_ = other.active_rank_;

    other.base_        = nullptr;
    other.cap_         = 0;
    other.off_         = 0;
    other.peak_        = 0;
    other.owns_        = true;
    other.owned_base_  = nullptr;
    other.ranks_.clear();
    other.active_rank_ = 0;
    return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("DeviceArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }

    const std::uintptr_t base_addr           = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t current_addr        = checked_add_uintptr(base_addr, off_);
    const std::uintptr_t aligned_addr        = align_up_addr(current_addr, align);
    const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
    if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("arena aligned offset overflows size_t");
    }
    const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
        throw std::overflow_error("arena allocation end offset overflows size_t");
    }
    const std::size_t end = aligned_offset + bytes;
    if (end > cap_) { throw std::bad_alloc(); }

    auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
    off_      = end;
    if (off_ > peak_) { peak_ = off_; }
    return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }

void DeviceArena::reset() noexcept {
    off_ = 0;
    for (RankBacking& backing : ranks_) { backing.off = 0; }
}

void* DeviceArena::base() const noexcept { return base_; }

std::size_t DeviceArena::used() const noexcept { return off_; }

std::size_t DeviceArena::capacity() const noexcept { return cap_; }

std::size_t DeviceArena::peak_used() const noexcept { return peak_; }

void DeviceArena::reset_peak() noexcept { peak_ = off_; }

void DeviceArena::store_active_rank() noexcept {
    if (active_rank_ < ranks_.size()) {
        RankBacking& backing = ranks_[active_rank_];
        backing.base         = base_;
        backing.cap          = cap_;
        backing.off          = off_;
        backing.peak         = peak_;
    }
}

void DeviceArena::attach_rank_storage(DeviceSpan storage) {
    if (storage.data == nullptr || storage.bytes == 0) {
        throw std::invalid_argument("attached rank storage must be a nonempty device span");
    }
    if (ranks_.empty()) {
        // First attach materializes rank 0 from the arena's own storage, so indices line up with
        // pipeline ranks from here on.
        ranks_.push_back(RankBacking{base_, cap_, off_, peak_});
        active_rank_ = 0;
    }
    ranks_.push_back(RankBacking{storage.data, storage.bytes, 0, 0});
}

void DeviceArena::activate_rank(std::size_t rank) {
    if (ranks_.empty()) {
        // Single-rank arenas only ever have rank 0; anything else is a planning bug worth hearing
        // about rather than silently ignoring.
        if (rank != 0) { throw std::out_of_range("arena has no storage for that pipeline rank"); }
        return;
    }
    if (rank >= ranks_.size()) {
        throw std::out_of_range("arena has no storage for that pipeline rank");
    }
    if (rank == active_rank_) { return; }

    store_active_rank();
    const RankBacking& next = ranks_[rank];
    base_                   = next.base;
    cap_                    = next.cap;
    off_                    = next.off;
    peak_                   = next.peak;
    active_rank_            = rank;
}

std::size_t DeviceArena::active_rank() const noexcept { return active_rank_; }

std::size_t DeviceArena::rank_count() const noexcept {
    return ranks_.empty() ? 1U : ranks_.size();
}

std::size_t DeviceArena::peak_used_for_rank(std::size_t rank) const {
    if (ranks_.empty()) {
        if (rank != 0) { throw std::out_of_range("arena has no storage for that pipeline rank"); }
        return peak_;
    }
    if (rank >= ranks_.size()) {
        throw std::out_of_range("arena has no storage for that pipeline rank");
    }
    // The active rank's live counters have not been written back yet.
    return rank == active_rank_ ? peak_ : ranks_[rank].peak;
}

ScopedArenaRank::ScopedArenaRank(DeviceArena& arena, std::size_t rank)
    : arena_(arena), previous_rank_(arena.active_rank()) {
    arena_.activate_rank(rank);
}

ScopedArenaRank::~ScopedArenaRank() noexcept {
    if (arena_.active_rank() != previous_rank_) { arena_.activate_rank(previous_rank_); }
}

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }
    resident_session_ = core::current_resident_memory();
    if (resident_session_) {
        data_ = resident_session_->allocate_host(size_bytes);
        size_ = size_bytes;
        return;
    }

    // A pending error from an earlier call is returned by whatever runs next, so read and clear it
    // first. Without this, a failure here can be somebody else's error wearing this message.
    const cudaError_t pending = cudaGetLastError();

    // Portable, so the buffer counts as pinned in every device's context rather than only the one
    // that was current when it was allocated. With several ranks, ingress frames are copied to each
    // stage's device from one host buffer, and a copy that treats it as pageable is neither fast nor
    // capturable. (The error text below keeps its `cudaMallocHost` wording: launcher docs and the
    // test guard match on it.)
    void* ptr             = nullptr;
    const cudaError_t err = cudaHostAlloc(&ptr, size_bytes, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        // Say the size, and say which memory. `cudaErrorMemoryAllocation: out of memory` reads as
        // a VRAM shortfall and sends people to nvidia-smi when the shortfall is pinned system RAM.
        // But the converse also happens and used to be asserted away here: on Windows/WDDM this
        // allocation is mapped into the GPU's address space and charged against the card, so a
        // large pin fails once the device is nearly full even with tens of GiB of system RAM free
        // -- measured on a 24 GiB RTX 3090, where resident-device plus pinned-host lands within a
        // few hundred MiB of the card's capacity every time. `cudaErrorAlreadyMapped` is the
        // failure that shape produces. Report free VRAM so the two are distinguishable rather than
        // asserting which one it is.
        std::size_t free_device = 0, total_device = 0;
        const cudaError_t info = cudaMemGetInfo(&free_device, &total_device);
        std::string prefix     = "cudaMallocHost failed to pin " +
                             std::to_string((size_bytes + (1ULL << 20) - 1) >> 20) +
                             " MiB of host memory";
        if (info == cudaSuccess) {
            prefix += " (device has " + std::to_string(free_device >> 20) + " MiB of " +
                      std::to_string(total_device >> 20) + " MiB free; on Windows a pinned host "
                      "allocation is mapped into the GPU address space and competes with it)";
        } else {
            prefix += " (system RAM, pinned)";
        }
        if (pending != cudaSuccess) {
            prefix += ", after an unretrieved earlier error (";
            prefix += cudaGetErrorName(pending);
            prefix += ")";
        }
        throw std::runtime_error(cuda_error_message(prefix.c_str(), err));
    }

    // The pending error read above is now gone for good -- `cudaGetLastError()` clears it, and
    // nothing else has seen it since. A successful pin does not mean it never mattered: it means
    // an earlier, unrelated CUDA call failed and nobody checked. Report it rather than let it
    // disappear silently.
    if (pending != cudaSuccess) {
        std::fprintf(stderr,
                      "ninfer: pinning %zu bytes of host memory succeeded, but cleared an "
                      "unretrieved earlier CUDA error: %s: %s\n",
                      size_bytes, cudaGetErrorName(pending), cudaGetErrorString(pending));
    }

    data_ = ptr;
    size_ = size_bytes;
}

PinnedHostBuffer::~PinnedHostBuffer() {
    if (resident_session_) { resident_session_->free_host(data_); }
    else { free_pinned(data_); }
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_), resident_session_(std::move(other.resident_session_)) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    if (resident_session_) { resident_session_->free_host(data_); }
    else { free_pinned(data_); }
    resident_session_ = std::move(other.resident_session_);
    data_ = other.data_;
    size_ = other.size_;

    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void* PinnedHostBuffer::data() const noexcept { return data_; }

std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

} // namespace ninfer
