// Generic (non-CUDA) implementation of the KVMem stage-in interface.
//
// Built when LLAMA_KVMEM is on but GGML_CUDA is off (e.g. -DGGML_VULKAN=ON).
// Every GPU-pipeline entry point reports "not available", so callers take
// their existing host fallbacks: host dequant/RoPE/Hadamard plus
// ggml_backend_tensor_get/set transfers (see llama-kvmem-transfer.h).
// Correct on any ggml backend; slower than the CUDA pipeline because
// transfers are synchronous and staging math runs on the host.

#include "llama-kvmem-stagein.h"

bool kvmem_stagein_gpu_ready(size_t /*n_f32*/, size_t /*n_packed*/) {
    return false;
}

void kvmem_stagein_gpu_free() {
}

bool kvmem_stagein_fwht_ok(int /*nrot*/) {
    return false;
}

bool kvmem_stagein_quant_ok(ggml_type /*ty*/) {
    return false;
}

bool kvmem_stagein_h2d_f32(const float * /*host*/, int64_t /*n*/) {
    return false;
}

bool kvmem_stagein_h2d_packed(const void * /*host*/, size_t /*n*/) {
    return false;
}

bool kvmem_stagein_dequant(ggml_type /*ty*/, int64_t /*n_rows*/, int64_t /*n_embd*/) {
    return false;
}

bool kvmem_stagein_rope_neox(int64_t /*n_tokens*/, int /*n_head*/, int /*n_embd_head*/,
                             int /*n_rot*/, int32_t /*pos0*/, const float * /*theta*/,
                             int /*n_theta*/) {
    return false;
}

bool kvmem_stagein_fwht(int64_t /*n_rows*/, int64_t /*n_embd*/, int /*nrot*/) {
    return false;
}

bool kvmem_stagein_quantize(ggml_type /*ty*/, void * /*gpu_dst*/,
                            int64_t /*n_rows*/, int64_t /*n_embd*/) {
    return false;
}

bool kvmem_stagein_h2d_bytes(void * /*gpu_dst*/, const void * /*host*/, size_t /*n*/) {
    return false;
}

void kvmem_stagein_sync() {
}

bool kvmem_stagein_enqueue_k(
        ggml_type /*ty*/, const void * /*packed*/, size_t /*nbytes*/, uint8_t * /*dst*/,
        int64_t /*nt*/, int64_t /*n_embd*/, int /*nrot*/,
        int /*n_head*/, int /*n_embd_head*/, int /*n_rot_rope*/, int32_t /*pos0*/,
        const float * /*theta*/, int /*n_theta*/,
        int64_t * /*copy_us*/, int64_t * /*rope_us*/, int64_t * /*hadamard_us*/,
        int64_t * /*set_us*/) {
    return false;
}

bool kvmem_stagein_enqueue_v(const void * /*packed*/, size_t /*nbytes*/, uint8_t * /*dst*/,
                             int64_t * /*set_us*/) {
    return false;
}

bool kvmem_stagein_flush(int64_t * /*copy_us*/, int64_t * /*rope_us*/,
                         int64_t * /*hadamard_us*/, int64_t * /*set_us*/) {
    return false;
}

bool kvmem_stageout_enqueue(const void * /*gpu_src*/, size_t /*nbytes*/) {
    return false;
}

size_t kvmem_stageout_used() {
    return 0;
}

int kvmem_stageout_submit(int64_t * /*copy_us*/) {
    return -1;
}

bool kvmem_stageout_wait(int /*slot*/, int64_t * /*copy_us*/) {
    return false;
}

const uint8_t * kvmem_stageout_slot_base(int /*slot*/) {
    return nullptr;
}

void kvmem_stageout_clear() {
}

bool kvmem_d2d_batched(const void * const * /*src*/, void * const * /*dst*/,
                       const size_t * /*nbytes*/, int /*n*/) {
    return false;
}

bool kvmem_meank_ready(uint32_t /*n_layer*/, uint32_t /*n_embd*/) {
    return false;
}

void kvmem_meank_free() {
}

void kvmem_meank_zero(uint32_t /*il*/) {
}

bool kvmem_meank_add(uint32_t /*il*/, ggml_type /*ty*/, const void * /*gpu_k*/,
                     uint32_t /*tok0*/, uint32_t /*n_keep*/, uint32_t /*n_embd*/,
                     int64_t /*ne0*/, size_t /*nb0*/, size_t /*nb1*/, size_t /*nb2*/) {
    return false;
}

bool kvmem_meank_d2h(uint32_t /*il*/, float * /*host*/, uint32_t /*n_embd*/) {
    return false;
}
