#pragma once

// Thin hook used by llama_model::create_memory when LLAMA_KVMEM is enabled.
// Implementation lives out-of-tree in src/adapter/.

struct llama_memory_i;
struct llama_model;
struct llama_memory_params;
struct llama_cparams;

#if defined(LLAMA_KVMEM)
llama_memory_i * llama_memory_kvmem_maybe_create(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams);
#endif
