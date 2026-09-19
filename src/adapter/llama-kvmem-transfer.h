#pragma once

#include "llama-kvmem-hooks.h"
#include "llama-kvmem-gpu.h"
#include "ggml-backend.h"

// Optional KVMEM_PERF accounting at executed copy call sites, not plan entries.
// Native model/backend transfers outside the adapter are deliberately excluded.
inline cudaError_t kvmem_copy_async(void * dst, const void * src, size_t n,
                                   cudaMemcpyKind kind, cudaStream_t stream = nullptr) {
    const auto rc = cudaMemcpyAsync(dst, src, n, kind, stream);
    if (rc == cudaSuccess) kvmem_record_transfer(kind, n);
    return rc;
}
inline cudaError_t kvmem_copy(void * dst, const void * src, size_t n, cudaMemcpyKind kind) {
    const auto rc = cudaMemcpy(dst, src, n, kind);
    if (rc == cudaSuccess) kvmem_record_transfer(kind, n);
    return rc;
}
inline bool kvmem_tensor_on_device(const ggml_tensor * t) {
    const auto buffer = t->buffer ? t->buffer : t->view_src ? t->view_src->buffer : nullptr;
    return buffer && !ggml_backend_buffer_is_host(buffer);
}
inline void kvmem_tensor_get(const ggml_tensor * t, void * data, size_t offset, size_t size) {
    ggml_backend_tensor_get(t, data, offset, size);
    if (kvmem_tensor_on_device(t)) kvmem_record_transfer(cudaMemcpyDeviceToHost, size);
}
inline void kvmem_tensor_set(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    ggml_backend_tensor_set(t, data, offset, size);
    if (kvmem_tensor_on_device(t)) kvmem_record_transfer(cudaMemcpyHostToDevice, size);
}
