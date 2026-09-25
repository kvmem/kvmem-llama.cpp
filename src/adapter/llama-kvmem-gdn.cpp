#include "llama-kvmem-gdn.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdint>

static ggml_tensor * view_4d_prefix(
        ggml_context * ctx, const ggml_tensor * src,
        int64_t ne0, int64_t ne1, int64_t ne2) {
    const size_t element = sizeof(float);
    return ggml_view_4d(ctx, const_cast<ggml_tensor *>(src), ne0, ne1, ne2, 1,
            ne0 * element, ne0 * ne1 * element, ggml_nbytes(src), 0);
}

bool llama_kvmem_gdn_fold_ggml(
        ggml_backend_t backend,
        const llama_kvmem_gdn_layer * layers,
        int n_layers, int n_keep, int capacity,
        size_t * compute_bytes) {
    if (compute_bytes) *compute_bytes = 0;
    if (n_keep == 0) return true;
    if (!backend || !layers || n_layers <= 0 || n_keep < 0 || n_keep > capacity || capacity > 6) return false;

    ggml_backend_buffer_type_t buft = nullptr;
    auto valid = [&](const ggml_tensor * tensor, int64_t elements) {
        if (!tensor || !tensor->buffer || tensor->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(tensor) || ggml_nelements(tensor) != elements) {
            return false;
        }
        auto * tensor_buft = ggml_backend_buffer_get_type(tensor->buffer);
        if (!ggml_backend_supports_buft(backend, tensor_buft)) return false;
        if (buft && tensor_buft != buft) return false;
        buft = tensor_buft;
        return true;
    };
    for (int il = 0; il < n_layers; ++il) {
        const auto & layer = layers[il];
        if (!valid(layer.state, 128 * 128 * 48) || !valid(layer.conv, 3 * 10240) ||
                !valid(layer.key, capacity * 2048) || !valid(layer.value, capacity * 6144) ||
                !valid(layer.gate, capacity * 48) || !valid(layer.beta, capacity * 48) ||
                !valid(layer.conv_input, capacity * 10240)) {
            return false;
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ (size_t) n_layers * 32 * ggml_tensor_overhead() + ggml_graph_overhead_custom((size_t) n_layers * 32, false) + 64 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) return false;
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), (size_t) n_layers * 32, false);

    for (int il = 0; il < n_layers; ++il) {
        const auto & layer = layers[il];
        ggml_tensor * key = view_4d_prefix(ctx.get(), layer.key, 128, 16, n_keep);
        ggml_tensor * value = view_4d_prefix(ctx.get(), layer.value, 128, 48, n_keep);
        ggml_tensor * gate = view_4d_prefix(ctx.get(), layer.gate, 1, 48, n_keep);
        ggml_tensor * beta = view_4d_prefix(ctx.get(), layer.beta, 1, 48, n_keep);
        ggml_tensor * state = ggml_view_4d(ctx.get(), layer.state, 128, 128, 48, 1,
                128 * sizeof(float), 128 * 128 * sizeof(float), ggml_nbytes(layer.state), 0);
        ggml_tensor * result = ggml_gated_delta_net(ctx.get(), key, key, value, gate, beta, state, 1);
        const size_t state_offset = (size_t) 128 * 48 * n_keep * sizeof(float);
        ggml_tensor * new_state = ggml_view_4d(ctx.get(), result, 128, 128, 48, 1,
                128 * sizeof(float), 128 * 128 * sizeof(float), 128 * 128 * 48 * sizeof(float), state_offset);
        ggml_build_forward_expand(graph, ggml_cpy(ctx.get(), new_state, layer.state));

        ggml_tensor * conv_input = ggml_view_2d(ctx.get(), const_cast<ggml_tensor *>(layer.conv_input), 10240, n_keep,
                10240 * sizeof(float), 0);
        ggml_tensor * conv = ggml_reshape_2d(ctx.get(), layer.conv, 3, 10240);
        ggml_tensor * appended = ggml_concat(ctx.get(), conv,
                ggml_cont(ctx.get(), ggml_transpose(ctx.get(), conv_input)), 0);
        ggml_tensor * new_conv = ggml_view_2d(ctx.get(), appended, 3, 10240,
                (3 + n_keep) * sizeof(float), n_keep * sizeof(float));
        ggml_build_forward_expand(graph, ggml_cpy(ctx.get(), new_conv, layer.conv));
    }

    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) return false;
    }
    ggml_gallocr_ptr allocator(ggml_gallocr_new(buft));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) return false;
    if (compute_bytes) *compute_bytes = ggml_gallocr_get_buffer_size(allocator.get(), 0);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    return status == GGML_STATUS_SUCCESS;
}
