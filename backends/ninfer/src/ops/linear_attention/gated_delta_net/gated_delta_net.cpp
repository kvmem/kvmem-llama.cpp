#include "ninfer/ops/gated_delta_net.h"

#include "ninfer/ops/l2norm.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/device_route.h"
#include "ops/common/math.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/gated_delta_net/launch.h"
#include "ops/linear_attention/gated_delta_net/two_stage/launch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct Geometry {
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
};

void require_dtype(const Tensor& t, DType dtype, const char* name) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string("gated_delta_net: ") + name); }
}

void require_state_dtype(const Tensor& tensor, const char* message) {
    if (tensor.dtype != DType::FP32 && tensor.dtype != DType::FP16) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + message);
    }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string("gated_delta_net: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " data must be non-null");
    }
}

Geometry require_geometry(const Tensor& q, const Tensor& v) {
    const Geometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != detail::gated_delta_net::kStateDim) {
        throw std::invalid_argument("gated_delta_net: state/head dimension must be 128");
    }
    if (!detail::gated_delta_net::are_head_counts_valid(geometry.qk_heads, geometry.value_heads)) {
        throw std::invalid_argument(
            "gated_delta_net: value heads must be at least q/k heads and divisible by them");
    }
    if (geometry.tokens <= 0) {
        throw std::invalid_argument("gated_delta_net: T must be positive");
    }
    return geometry;
}

void require_scale(float scale) {
    const float expected_scale =
        1.0f / std::sqrt(static_cast<float>(detail::gated_delta_net::kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_net: scale must be 1/sqrt(128)");
    }
}

Geometry validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& ssm_state,
                            const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_state_dtype(ssm_state, "ssm_state must be FP32 or FP16");

    const Geometry geometry = require_geometry(q, v);
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    require_shape(ssm_state, detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                  geometry.value_heads, 1, "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

Geometry validate_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         const Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, const Tensor& out) {
    constexpr std::int32_t kMaximumBatch = 8;
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be FP32");
    require_dtype(beta, DType::FP32, "beta must be FP32");
    require_state_dtype(ssm_states, "ssm_states must be FP32 or FP16");
    require_dtype(source_state_slots, DType::I32, "source_state_slots must be I32");
    require_dtype(destination_state_slots, DType::I32, "destination_state_slots must be I32");

    const Geometry geometry  = require_geometry(q, v);
    const std::int32_t batch = q.ne[3];
    if (batch <= 0 || batch > kMaximumBatch || geometry.tokens != 1) {
        throw std::invalid_argument("gated_delta_net: batch update requires B=1..8 and W=1");
    }
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, batch, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, batch, 1, "beta");
    if (ssm_states.ne[0] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[1] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] <= 0) {
        throw std::invalid_argument("gated_delta_net: invalid shape for pooled ssm_states");
    }
    require_shape(source_state_slots, batch, 1, 1, 1, "source_state_slots");
    require_shape(destination_state_slots, batch, 1, 1, 1, "destination_state_slots");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    require_contiguous_nonnull(source_state_slots, "source_state_slots");
    require_contiguous_nonnull(destination_state_slots, "destination_state_slots");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, const Tensor& ssm_state_in,
                      const Tensor& ssm_state_out, const Tensor& out) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const Geometry geometry = validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out);
    require_state_dtype(ssm_state_in, "ssm_state_in must be FP32 or FP16");
    require_shape(ssm_state_in, detail::gated_delta_net::kStateDim,
                  detail::gated_delta_net::kStateDim, geometry.value_heads, 1, "ssm_state_in");
    require_contiguous_nonnull(ssm_state_in, "ssm_state_in");
}

struct ChunkedWorkspace {
    Tensor normalized_q;
    Tensor normalized_k;
    DeviceSpan stage;
    // FP32 running state for the chunked kernels when the stored state is FP16.
    Tensor state_fp32;
};

template <class Allocator>
ChunkedWorkspace allocate_chunked_workspace(Allocator& allocator, std::int32_t qk_heads,
                                            std::int32_t value_heads, std::int32_t tokens,
                                            bool normalize_qk) {
    ChunkedWorkspace out;
    const std::int32_t full =
        (tokens / detail::gated_delta_net::kChunkSize) * detail::gated_delta_net::kChunkSize;
    if (full == 0) { return out; }
    if (normalize_qk) {
        out.normalized_q =
            allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, qk_heads, tokens});
        out.normalized_k =
            allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, qk_heads, tokens});
    }
    out.stage =
        allocator.alloc_bytes(detail::gated_delta_net::chunked_workspace_bytes(value_heads, full));
    out.state_fp32 = allocator.alloc(DType::FP32, {detail::gated_delta_net::kStateDim,
                                                   detail::gated_delta_net::kStateDim, value_heads});
    return out;
}

// The two-stage prefill against the WY/state-passing/output pipeline, per prompt width: the device
// profile's "gdn_two_stage/h<value heads>" entry, or NINFER_GDN_TWO_STAGE=0/1 for every width.
bool two_stage_route(std::int32_t value_heads, std::int32_t tokens) {
    if (tokens < detail::gated_delta_net::two_stage::kMinTokens) { return false; }
    static const int forced = [] {
        const char* value = std::getenv("NINFER_GDN_TWO_STAGE");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("gdn_two_stage/h" + std::to_string(value_heads), tokens) == "on";
}

struct TwoStageWorkspace {
    DeviceSpan packets;
    // FP32 running state for the two-stage kernels when either stored state is FP16.
    Tensor state_fp32;
};

template <class Allocator>
TwoStageWorkspace allocate_two_stage_workspace(Allocator& allocator, std::int32_t qk_heads,
                                               std::int32_t value_heads, std::int32_t tokens) {
    TwoStageWorkspace out;
    if (tokens < detail::gated_delta_net::two_stage::kMinTokens) { return out; }
    out.packets = allocator.alloc_bytes(
        detail::gated_delta_net::two_stage::workspace_layout(qk_heads, value_heads, tokens)
            .total_bytes);
    out.state_fp32 =
        allocator.alloc(DType::FP32, {detail::gated_delta_net::kStateDim,
                                      detail::gated_delta_net::kStateDim, value_heads});
    return out;
}

void run_two_stage(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                   const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                   const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                   cudaStream_t stream) {
    namespace two_stage       = detail::gated_delta_net::two_stage;
    auto scratch_scope        = ws.scope();
    TwoStageWorkspace scratch = allocate_two_stage_workspace(ws, q.ne[1], v.ne[1], q.ne[2]);
    const auto layout         = two_stage::workspace_layout(q.ne[1], v.ne[1], q.ne[2]);
    auto* qk = static_cast<two_stage::QkChunk*>(layout.qk.bind(scratch.packets).data);
    auto* control =
        static_cast<two_stage::ControlChunk*>(layout.control.bind(scratch.packets).data);

    const bool staged = ssm_state_in.dtype == DType::FP16 || ssm_state_out.dtype == DType::FP16;
    if (staged) {
        if (ssm_state_in.dtype == DType::FP16) {
            detail::gated_delta_net::widen_state_fp16_to_fp32(ssm_state_in, scratch.state_fp32,
                                                              stream);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(scratch.state_fp32.data, ssm_state_in.data,
                                       ssm_state_in.bytes(), cudaMemcpyDeviceToDevice, stream));
        }
    }
    const two_stage::Arguments args{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(staged ? scratch.state_fp32.data : ssm_state_in.data),
        static_cast<float*>(staged ? scratch.state_fp32.data : ssm_state_out.data),
        static_cast<__nv_bfloat16*>(out.data),
        q.ne[1],
        v.ne[1],
        q.ne[2],
        scale};
    two_stage::launch_prepare(args, qk, control, normalize_qk, stream);
    two_stage::launch_recurrence(args, qk, control, stream);
    if (staged) {
        if (ssm_state_out.dtype == DType::FP16) {
            detail::gated_delta_net::narrow_state_fp32_to_fp16(scratch.state_fp32, ssm_state_out,
                                                               stream);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(ssm_state_out.data, scratch.state_fp32.data,
                                       ssm_state_out.bytes(), cudaMemcpyDeviceToDevice, stream));
        }
    }
}

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                     std::int32_t value_heads, bool normalize_qk,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (!detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("gated_delta_net workspace: invalid profile or interval");
    }
    // Either prefill algorithm may serve any width the route admits it for.
    WorkspaceLayoutBuilder chunked;
    (void)allocate_chunked_workspace(chunked, qk_heads, value_heads, max_tokens, normalize_qk);
    WorkspaceLayoutBuilder two_stage;
    (void)allocate_two_stage_workspace(two_stage, qk_heads, value_heads, max_tokens);
    return std::max(chunked.peak_bytes(1), two_stage.peak_bytes(1));
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    if (q.ne[2] != 1) {
        gated_delta_net(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state, ssm_state, out,
                        stream);
        return;
    }
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);

    (void)ws;
    detail::gated_delta_net::launch_recurrent(q, k, v, g, beta, scale, normalize_qk, ssm_state, out,
                                              stream);
}

void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, Tensor& ssm_states,
                                  const Tensor& source_state_slots,
                                  const Tensor& destination_state_slots, Tensor& out,
                                  cudaStream_t stream) {
    validate_recurrent_batch_update(q, k, v, g, beta, scale, ssm_states, source_state_slots,
                                    destination_state_slots, out);

    detail::gated_delta_net::launch_recurrent_batch_update(q, k, v, g, beta, scale, normalize_qk,
                                                           ssm_states, source_state_slots,
                                                           destination_state_slots, out, stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     cudaStream_t stream) {
    validate_chunked(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);
    if (two_stage_route(v.ne[1], q.ne[2])) {
        run_two_stage(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state_in, ssm_state_out, out,
                      stream);
        return;
    }

    auto scratch_scope   = ws.scope();
    const std::int32_t T = q.ne[2];
    const std::int32_t T_full =
        (T / detail::gated_delta_net::kChunkSize) * detail::gated_delta_net::kChunkSize;
    ChunkedWorkspace scratch = allocate_chunked_workspace(ws, q.ne[1], v.ne[1], T, normalize_qk);
    Tensor q_compute         = q;
    Tensor k_compute         = k;
    bool recurrent_normalize = normalize_qk;
    if (normalize_qk && T_full > 0) {
        q_compute = scratch.normalized_q;
        k_compute = scratch.normalized_k;
        l2norm(q, 1.0e-6f, q_compute, stream);
        l2norm(k, 1.0e-6f, k_compute, stream);
        recurrent_normalize = false;
    }
    if (T_full > 0) {
        Tensor q_full    = q_compute.slice(2, 0, T_full);
        Tensor k_full    = k_compute.slice(2, 0, T_full);
        Tensor v_full    = v.slice(2, 0, T_full);
        Tensor g_full    = g.slice(1, 0, T_full);
        Tensor beta_full = beta.slice(1, 0, T_full);
        Tensor out_full  = out.slice(2, 0, T_full);
        if (ssm_state_out.dtype == DType::FP16) {
            // The chunked kernels read and write FP32 state; stage it through the workspace.
            if (ssm_state_in.dtype == DType::FP16) {
                detail::gated_delta_net::widen_state_fp16_to_fp32(ssm_state_in, scratch.state_fp32,
                                                                  stream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(scratch.state_fp32.data, ssm_state_in.data,
                                           ssm_state_in.bytes(), cudaMemcpyDeviceToDevice, stream));
            }
            detail::gated_delta_net::launch_chunked(q_full, k_full, v_full, g_full, beta_full,
                                                    scale, scratch.state_fp32, scratch.state_fp32,
                                                    out_full, scratch.stage.data,
                                                    scratch.stage.bytes, stream);
            detail::gated_delta_net::narrow_state_fp32_to_fp16(scratch.state_fp32, ssm_state_out,
                                                               stream);
        } else {
            detail::gated_delta_net::launch_chunked(q_full, k_full, v_full, g_full, beta_full,
                                                    scale, ssm_state_in, ssm_state_out, out_full,
                                                    scratch.stage.data, scratch.stage.bytes,
                                                    stream);
        }
    }

    const std::int32_t tail = T - T_full;
    if (tail > 0) {
        Tensor q_tail    = q_compute.slice(2, T_full, tail);
        Tensor k_tail    = k_compute.slice(2, T_full, tail);
        Tensor v_tail    = v.slice(2, T_full, tail);
        Tensor g_tail    = g.slice(1, T_full, tail);
        Tensor beta_tail = beta.slice(1, T_full, tail);
        Tensor out_tail  = out.slice(2, T_full, tail);
        // After full chunks the running state lives in ssm_state_out; a tail-only run (no full
        // chunks) reads the caller-provided ssm_state_in. Either way the tail publishes to
        // ssm_state_out.
        const Tensor& tail_in = (T_full > 0) ? ssm_state_out : ssm_state_in;
        detail::gated_delta_net::launch_recurrent_inout(q_tail, k_tail, v_tail, g_tail, beta_tail,
                                                        scale, recurrent_normalize, tail_in,
                                                        ssm_state_out, out_tail, stream);
    }
}

} // namespace ninfer::ops
