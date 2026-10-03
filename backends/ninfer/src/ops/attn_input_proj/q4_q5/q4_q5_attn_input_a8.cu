// Integer-activation route for the 27B attention input projection, on sm_86.
//
// The groupwise-int artifact binds this projection as two parents over one input:
//
//   query_key   Q4_G64_FP16 [7168, 5120] -> rows [0, 6144) -> q, rows [6144, 7168) -> k
//   gate_value  Q5_G64_FP16 [7168, 5120] -> rows [0, 6144) -> gate, rows [6144, 7168) -> v
//
// Four destinations, one activation: the quantiser runs once and all four launches read it. Only
// 16 of the 27B's 64 layers are full attention, so this is the smallest of the three projections
// an Nsight prefill profile shows -- 209 ms of 3,280 ms -- but it shares the schedule with them.

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "core/device.h"
#include "core/weight_view.h"
#include "ops/common/rowsplit_a8_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

namespace a8 = rowsplit_a8;

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kQRows      = 6144;
constexpr std::int32_t kKvRows     = 1024;
constexpr std::int32_t kParentRows = kQRows + kKvRows;

// One m-tile per warp stages 64 rows, which 6144 and 1024 both divide, and leaves the register
// budget for the widest token tile.
using Rows = a8::ContiguousRows<1>;

template <class Codec, int NT>
void launch_range(const Weight& weight, const std::int8_t* codes, const __half* scales,
                  std::int32_t tokens, std::int32_t row_begin, std::int32_t row_count,
                  __nv_bfloat16* dst, std::int32_t dst_rows, cudaStream_t stream) {
    constexpr int BN       = a8::kWarpsN * NT * 8;
    const std::size_t smem = a8::shared_bytes<Codec, 1, NT, Rows>(kHidden);
    const dim3 grid(row_count / Rows::kRowsPerBlock, tokens / BN);
    auto* kernel = a8::a8_mma_kernel<Codec, kHidden, 1, NT, Rows, a8::StoreEpilogue>;
    if (smem > 48 * 1024) {
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem));
        });
    }
    kernel<<<grid, a8::kThreads, smem, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const __half*>(weight.scales), codes, scales, tokens, Rows{row_begin},
        a8::StoreEpilogue{dst, dst_rows, 0}, row_split_panel_shift(weight.layout));
    CUDA_CHECK(cudaGetLastError());
}

template <int NT>
void launch_all(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                std::int32_t tokens, std::int8_t* codes, __half* scales, Tensor& q, Tensor& gate,
                Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr int BN = a8::kWarpsN * NT * 8;
    a8::quantize_activations<kHidden, BN><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, codes, scales);
    CUDA_CHECK(cudaGetLastError());

    launch_range<a8::Q4Codec, NT>(query_key, codes, scales, tokens, 0, kQRows,
                                  static_cast<__nv_bfloat16*>(q.data), kQRows, stream);
    launch_range<a8::Q4Codec, NT>(query_key, codes, scales, tokens, kQRows, kKvRows,
                                  static_cast<__nv_bfloat16*>(k.data), kKvRows, stream);
    launch_range<a8::Q5Codec, NT>(gate_value, codes, scales, tokens, 0, kQRows,
                                  static_cast<__nv_bfloat16*>(gate.data), kQRows, stream);
    launch_range<a8::Q5Codec, NT>(gate_value, codes, scales, tokens, kQRows, kKvRows,
                                  static_cast<__nv_bfloat16*>(v.data), kKvRows, stream);
}

} // namespace

bool q4_q5_attn_input_a8_supported(const Weight& query_key, const Weight& gate_value,
                                   std::int32_t tokens) {
    return a8::tokens_supported(tokens) && query_key.qtype == QType::Q4_G64_FP16 &&
           is_row_split(query_key.layout) && query_key.n == kParentRows &&
           query_key.k == kHidden && query_key.group == a8::kGroup && query_key.qdata != nullptr &&
           query_key.scales != nullptr && gate_value.qtype == QType::Q5_G64_FP16 &&
           is_row_split(gate_value.layout) && gate_value.n == kParentRows &&
           gate_value.k == kHidden && gate_value.group == a8::kGroup &&
           gate_value.qdata != nullptr && gate_value.qhigh != nullptr &&
           gate_value.scales != nullptr;
}

std::size_t q4_q5_attn_input_a8_workspace_capacity_bytes(std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q4_q5 attn_input a8 workspace: invalid token interval");
    }
    return a8::activation_workspace_bytes(kHidden, max_tokens);
}

void q4_q5_attn_input_a8_launch(const Tensor& x, const Weight& query_key, const Weight& gate_value,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!q4_q5_attn_input_a8_supported(query_key, gate_value, tokens)) {
        throw std::invalid_argument("q4_q5 attn_input a8: unsupported profile");
    }

    auto scope             = workspace.scope();
    const DeviceSpan codes = workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kHidden);
    const DeviceSpan scales = workspace.alloc_bytes(
        static_cast<std::size_t>(tokens) * (kHidden / a8::kGroup) * sizeof(__half));

    auto* code_data  = reinterpret_cast<std::int8_t*>(codes.data);
    auto* scale_data = reinterpret_cast<__half*>(scales.data);

    switch (a8::token_tile(tokens, 512)) {
    case 512:
        launch_all<16>(x, query_key, gate_value, tokens, code_data, scale_data, q, gate, k, v,
                       stream);
        return;
    case 256:
        launch_all<8>(x, query_key, gate_value, tokens, code_data, scale_data, q, gate, k, v,
                      stream);
        return;
    default:
        launch_all<4>(x, query_key, gate_value, tokens, code_data, scale_data, q, gate, k, v,
                      stream);
        return;
    }
}

} // namespace ninfer::ops::detail
