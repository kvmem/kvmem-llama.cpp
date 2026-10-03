#include "common.cuh"
#include "ggml.h"

template <int rows, int width>
static __device__ __forceinline__ float gdn_delta_f32(
        const float (&state)[rows], const float (&key)[rows], float decay, float value, float beta) {
    float partial = 0.0f;
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        partial += state[r] * key[r];
    }
    const float dot = warp_reduce_sum<width>(partial);
    return (value - decay * dot) * beta;
}

static __device__ __forceinline__ float gdn_update_f32(float state, float key, float decay, float delta) {
    return decay * state + key * delta;
}

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
