#pragma once

// KVMem GPU runtime portability shim.
//
// The adapter is written against the CUDA runtime API because that is what the
// upstream KVMem port targets. ROCm builds of this tree reuse the very same
// source: llama.cpp's HIP backend compiles ggml/src/ggml-cuda/*.cu with AMD
// clang (see ggml/src/ggml-hip/CMakeLists.txt) and relies on its own
// CUDA -> HIP name mapping in ggml/src/ggml-cuda/vendors/hip.h.
//
// This header does the same job for the out-of-tree adapter so the adapter
// sources stay single-source for both backends. Nothing here changes behaviour
// on a CUDA build: the HIP branch is only taken when GGML_USE_HIP or the AMD
// platform macro is defined.
//
// Only the subset of the runtime API actually used by the adapter is mapped.
// If new CUDA calls are added to the adapter, add the matching mapping here or
// the HIP build will fail to compile (which is the intended, loud failure).

// ---------------------------------------------------------------------------
// Host (Apple unified memory / Metal) branch.
//
// On Apple silicon the Metal backend allocates its buffers with
// MTLResourceStorageModeShared: ggml_backend_buffer_get_base() returns an
// address the CPU can read and write directly, and there is no separate device
// memory. The stage-in pipeline therefore needs no device code at all -- the
// very same kernels run as ordinary host loops and every "copy" is a memcpy.
// This is the phase-1 port: functionally complete, not yet GPU-accelerated.
// ---------------------------------------------------------------------------
#if defined(KVMEM_GPU_BACKEND_HOST)

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// --- kernel launch emulation ----------------------------------------------
//
// Kernels keep their CUDA shape: they read blockIdx/threadIdx/blockDim and
// return early when their linear index is out of range. The launch macros below
// drive them from a host loop with one emulated thread at a time, so the kernel
// bodies stay single-source for CUDA, HIP and this backend.
struct kvmem_idx3 {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
};

inline kvmem_idx3 kvmem_blockIdx_v{};
inline kvmem_idx3 kvmem_threadIdx_v{};
inline kvmem_idx3 kvmem_blockDim_v{1, 1, 1};

#define blockIdx  kvmem_blockIdx_v
#define threadIdx kvmem_threadIdx_v
#define blockDim  kvmem_blockDim_v

struct kvmem_uint4 { uint32_t x, y, z, w; };
#define uint4 kvmem_uint4

struct kvmem_dim3 {
    uint32_t x, y, z;
    kvmem_dim3(uint32_t x_ = 0, uint32_t y_ = 0, uint32_t z_ = 0) : x(x_), y(y_), z(z_) {}
};
#define dim3 kvmem_dim3

// Half precision: ggml's own host conversion is bit-identical to what the
// device kernels compute, so the numerics do not change.
struct kvmem_half_t {
    uint16_t bits = 0;
};
inline float __half2float(const kvmem_half_t & h) { return ggml_fp16_to_fp32(h.bits); }
inline kvmem_half_t __float2half(float f) { kvmem_half_t h; h.bits = ggml_fp32_to_fp16(f); return h; }

using __nv_bfloat16 = ggml_bf16_t;
using nv_bfloat16   = ggml_bf16_t;
inline float __bfloat162float(ggml_bf16_t v) { return ggml_bf16_to_fp32(v); }

// sincosf is a CUDA/GNU extension; the Metal SDK only spells it __sincosf.
inline void sincosf(float x, float * s, float * c) { __sincosf(x, s, c); }

#define KVMEM_GLOBAL
#define KVMEM_ALIGN(n) __attribute__((aligned(n)))

#define KVMEM_LAUNCH1(kernel, ntotal, threads, ...)                              \
    do {                                                                         \
        const int64_t  kvmem_n_  = (int64_t) (ntotal);                            \
        const uint32_t kvmem_t_  = (uint32_t) (threads);                          \
        kvmem_blockDim_v.x = kvmem_t_;                                            \
        kvmem_blockDim_v.y = 1;                                                   \
        kvmem_blockDim_v.z = 1;                                                   \
        const uint32_t kvmem_gx_ = kvmem_n_ > 0                                   \
            ? (uint32_t) ((kvmem_n_ + kvmem_t_ - 1) / kvmem_t_) : 0u;             \
        kvmem_blockIdx_v.y = 0;                                                   \
        kvmem_blockIdx_v.z = 0;                                                   \
        for (uint32_t kvmem_bx_ = 0; kvmem_bx_ < kvmem_gx_; ++kvmem_bx_) {        \
            kvmem_blockIdx_v.x = kvmem_bx_;                                       \
            for (uint32_t kvmem_tx_ = 0; kvmem_tx_ < kvmem_t_; ++kvmem_tx_) {     \
                kvmem_threadIdx_v.x = kvmem_tx_;                                  \
                kernel(__VA_ARGS__);                                              \
            }                                                                     \
        }                                                                         \
        kvmem_blockIdx_v.x = 0;                                                   \
        kvmem_threadIdx_v.x = 0;                                                  \
    } while (0)

#define KVMEM_LAUNCH2(kernel, nx, ny, threads, ...)                              \
    do {                                                                         \
        const uint32_t kvmem_t_ = (uint32_t) (threads);                           \
        kvmem_blockDim_v.x = kvmem_t_;                                            \
        kvmem_blockDim_v.y = 1;                                                   \
        kvmem_blockDim_v.z = 1;                                                   \
        kvmem_blockIdx_v.z = 0;                                                   \
        for (uint32_t kvmem_by_ = 0; kvmem_by_ < (uint32_t) (ny); ++kvmem_by_) {   \
            kvmem_blockIdx_v.y = kvmem_by_;                                       \
            for (uint32_t kvmem_bx_ = 0; kvmem_bx_ < (uint32_t) (nx); ++kvmem_bx_) {\
                kvmem_blockIdx_v.x = kvmem_bx_;                                   \
                for (uint32_t kvmem_tx_ = 0; kvmem_tx_ < kvmem_t_; ++kvmem_tx_) { \
                    kvmem_threadIdx_v.x = kvmem_tx_;                              \
                    kernel(__VA_ARGS__);                                          \
                }                                                                 \
            }                                                                     \
        }                                                                         \
        kvmem_blockIdx_v.x = 0;                                                   \
        kvmem_blockIdx_v.y = 0;                                                   \
        kvmem_threadIdx_v.x = 0;                                                  \
    } while (0)

// --- runtime API ----------------------------------------------------------
using cudaError_t = int;
using cudaStream_t = struct kvmem_host_stream *;
using cudaEvent_t  = struct kvmem_host_event *;
using cudaMemcpyKind = int;

enum : int {
    cudaSuccess               = 0,
    cudaErrorMemoryAllocation = 2,
};
enum : int {
    cudaMemcpyHostToDevice   = 0,
    cudaMemcpyDeviceToHost   = 1,
    cudaMemcpyDeviceToDevice = 2,
};
enum : unsigned {
    cudaStreamNonBlocking  = 1u,
    cudaEventDisableTiming = 2u,
};
enum : int {
    cudaMemoryTypeHost   = 1,
    cudaMemoryTypeDevice = 2,
};

// Treated as device memory on purpose: the Metal backend reports its buffers as
// device buffers (ggml_backend_buffer_is_host() is false) even though the
// address is host-writable. Callers that gate on "is this a device pointer"
// must take the device path, which here is a plain memcpy.
struct cudaPointerAttributes {
    int   type;
    int   device;
    void *devicePointer;
    void *hostPointer;
};

inline const char * cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess:               return "host shim: success";
        case cudaErrorMemoryAllocation: return "host shim: allocation failure";
        default:                        return "host shim: unknown error";
    }
}

inline cudaError_t cudaGetLastError() { return cudaSuccess; }

inline cudaError_t cudaMemcpy(void * dst, const void * src, size_t n, cudaMemcpyKind) {
    if (n) { std::memcpy(dst, src, n); }
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t n,
                                   cudaMemcpyKind, cudaStream_t = nullptr) {
    if (n) { std::memcpy(dst, src, n); }
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void * dst, int value, size_t n, cudaStream_t = nullptr) {
    if (n) { std::memset(dst, value, n); }
    return cudaSuccess;
}

inline cudaError_t cudaMalloc(void ** ptr, size_t n) {
    void * p = n ? std::malloc(n) : nullptr;
    if (!p) { return cudaErrorMemoryAllocation; }
    *ptr = p;
    return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void ** ptr, size_t n) { return cudaMalloc(ptr, n); }
inline cudaError_t cudaFree(void * ptr)     { std::free(ptr); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void * ptr) { std::free(ptr); return cudaSuccess; }

// Everything is synchronous, so stream and event handles are opaque tokens.
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t * s, unsigned) {
    *s = reinterpret_cast<cudaStream_t>(1);
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t)                    { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t)                { return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t * e, unsigned) {
    *e = reinterpret_cast<cudaEvent_t>(1);
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t)   { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t)            { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t)                { return cudaSuccess; }

inline cudaError_t cudaSetDevice(int)        { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int * dev)  { if (dev) { *dev = 0; } return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize()   { return cudaSuccess; }

inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * a, const void * ptr) {
    if (!a) { return cudaSuccess; }
    a->type          = ptr ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
    a->device        = 0;
    a->devicePointer = const_cast<void *>(ptr);
    a->hostPointer   = const_cast<void *>(ptr);
    return cudaSuccess;
}

#define cudaStreamPerThread nullptr

#elif defined(GGML_USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

// The kernels spell bfloat16 the CUDA way; HIP's equivalent is __hip_bfloat16.
// ggml-cuda/vendors/hip.h defines nv_bfloat16 the same way.
typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat16 nv_bfloat16;

// --- runtime API ----------------------------------------------------------
#define cudaError_t                hipError_t
#define cudaSuccess                hipSuccess
#define cudaErrorMemoryAllocation  hipErrorMemoryAllocation
#define cudaGetErrorString         hipGetErrorString
#define cudaGetLastError           hipGetLastError

#define cudaMemcpyKind             hipMemcpyKind
#define cudaMemcpyHostToDevice     hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost     hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice   hipMemcpyDeviceToDevice
#define cudaMemcpy                 hipMemcpy
#define cudaMemcpyAsync            hipMemcpyAsync

#define cudaMalloc                 hipMalloc
#define cudaFree                   hipFree
#define cudaMallocHost(ptr, size)  hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaFreeHost               hipHostFree

#define cudaMemsetAsync            hipMemsetAsync

#define cudaStream_t               hipStream_t
#define cudaStreamPerThread        hipStreamPerThread
#define cudaStreamCreateWithFlags  hipStreamCreateWithFlags
#define cudaStreamDestroy          hipStreamDestroy
#define cudaStreamSynchronize      hipStreamSynchronize
#define cudaStreamWaitEvent        hipStreamWaitEvent
#define cudaStreamNonBlocking      hipStreamNonBlocking

#define cudaEvent_t                hipEvent_t
#define cudaEventCreateWithFlags   hipEventCreateWithFlags
#define cudaEventDisableTiming     hipEventDisableTiming
#define cudaEventRecord            hipEventRecord
#define cudaEventSynchronize       hipEventSynchronize
#define cudaEventDestroy           hipEventDestroy

#define cudaSetDevice              hipSetDevice
#define cudaGetDevice              hipGetDevice
#define cudaDeviceSynchronize      hipDeviceSynchronize

#define cudaPointerAttributes      hipPointerAttribute_t
#define cudaPointerGetAttributes   hipPointerGetAttributes
#define cudaMemoryTypeDevice       hipMemoryTypeDevice

#else

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#endif  // GGML_USE_HIP

// --- device-code spellings shared by the CUDA and HIP builds ---------------
#if !defined(KVMEM_GPU_BACKEND_HOST)

#define KVMEM_GLOBAL __global__
#define KVMEM_ALIGN(n) __align__(n)

// One-dimensional launch: ntotal elements, `threads` per block, on the
// adapter's per-thread stream.
#define KVMEM_LAUNCH1(kernel, ntotal, threads, ...)                              \
    kernel<<<(int) (((int64_t) (ntotal) + (threads) - 1) / (threads)),           \
             (threads), 0, stream()>>>(__VA_ARGS__)

// Two-dimensional launch used by the RoPE kernel: grid.x over tokens,
// grid.y over heads.
#define KVMEM_LAUNCH2(kernel, nx, ny, threads, ...)                              \
    kernel<<<dim3((nx), (ny), 1), (threads), 0, stream()>>>(__VA_ARGS__)

#endif  // !KVMEM_GPU_BACKEND_HOST
