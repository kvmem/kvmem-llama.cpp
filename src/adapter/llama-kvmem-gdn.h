#pragma once

#include "ggml-backend.h"

struct llama_kvmem_gdn_layer {
    ggml_tensor * state;
    ggml_tensor * conv;
    const ggml_tensor * key;
    const ggml_tensor * value;
    const ggml_tensor * gate;
    const ggml_tensor * beta;
    const ggml_tensor * conv_input;
};

bool llama_kvmem_gdn_fold_ggml(
        ggml_backend_t backend,
        const llama_kvmem_gdn_layer * layers,
        int n_layers, int n_keep, int capacity,
        size_t * compute_bytes = nullptr);
