#pragma once

#include "llama.h"

#include <vector>

// Must match the definition in llama-memory-kvmem.h (`class`, not `struct`):
// the two keywords produce different mangling under the Microsoft C++ ABI.
class llama_memory_kvmem;
class llama_memory_kvmem_mtp;
struct ggml_tensor;

void kvmem_capture_bind(llama_memory_kvmem * mem);
void kvmem_capture_unbind(llama_memory_kvmem * mem);
llama_memory_kvmem * kvmem_capture_active();
void kvmem_mtp_bind(llama_memory_kvmem_mtp * mem);
void kvmem_mtp_unbind(llama_memory_kvmem_mtp * mem);
void kvmem_capture_note_ubatch(const std::vector<llama_pos> & pos);
void kvmem_capture_reset_q();
void kvmem_capture_register(struct ggml_tensor * t, int il, char which);
void kvmem_capture_on_new_graph(int is_mtp);
void kvmem_capture_harvest_ubatch(struct ggml_backend_sched * sched, int is_mtp);
bool kvmem_ubatch_needs_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos);
bool kvmem_capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos, int is_mtp);
