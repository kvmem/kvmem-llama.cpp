// Integer-activation route for the 27B GDN input projection, on sm_86.
//
// The groupwise-int artifact binds this projection as two parents over one input:
//
//   query_key  Q4_G64_FP16 [4096, 5120]  -> qkv rows [0, 4096)
//   value_z    Q5_G64_FP16 [12288, 5120] -> rows [0, 6144)     -> qkv rows [4096, 10240)
//                                           rows [6144, 12288) -> z rows [0, 6144)
//
// Both read the same x, so the activation quantisation happens once and all three launches share
// it: the quantiser is O(K*T) against the GEMMs' O(N*K*T), and at this shape that is well under a
// percent of the work. The three launches differ only in which weight they read and where their
// rows land, which is what rowsplit_a8::StoreEpilogue carries.
//
// 48 of the 27B's 64 layers are GDN, so this is the largest single projection in a prefill: an
// Nsight profile of a 4,096-token prefill put it at 749 ms of 3,280 ms with the A16 route.

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

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
constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
constexpr std::int32_t kParentRows = kValueRows + kZRows;

// One m-tile per warp over a 4x4 warp grid stages 64 rows, which every row count here divides, and
// leaves the register budget for the widest token tile.
using Rows = a8::ContiguousRows<1>;

template <class Codec, int NT>
void launch_range(const Weight& weight, const std::int8_t* codes, const __half* scales,
                  std::int32_t tokens, std::int32_t row_begin, std::int32_t row_count,
                  a8::StoreEpilogue epilogue, cudaStream_t stream) {
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
        epilogue, row_split_panel_shift(weight.layout));
    CUDA_CHECK(cudaGetLastError());
}

// The quantiser writes its activations in the tile's fragment order, so the tile width is chosen
// once and every launch of this call uses it.
template <int NT>
void launch_all(const Tensor& x, const Weight& qk, const Weight& value_z, std::int32_t tokens,
                std::int8_t* codes, __half* scales, __nv_bfloat16* qkv, __nv_bfloat16* z,
                cudaStream_t stream) {
    constexpr int BN = a8::kWarpsN * NT * 8;
    a8::quantize_activations<kHidden, BN><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, codes, scales);
    CUDA_CHECK(cudaGetLastError());

    launch_range<a8::Q4Codec, NT>(qk, codes, scales, tokens, 0, kQkRows,
                                  a8::StoreEpilogue{qkv, kQkvRows, 0}, stream);
    launch_range<a8::Q5Codec, NT>(value_z, codes, scales, tokens, 0, kValueRows,
                                  a8::StoreEpilogue{qkv, kQkvRows, kQkRows}, stream);
    launch_range<a8::Q5Codec, NT>(value_z, codes, scales, tokens, kValueRows, kZRows,
                                  a8::StoreEpilogue{z, kZRows, 0}, stream);
}

} // namespace

bool q4_q5_gdn_input_a8_supported(const Weight& qk, const Weight& value_z, std::int32_t tokens) {
    return a8::tokens_supported(tokens) && qk.qtype == QType::Q4_G64_FP16 &&
           is_row_split(qk.layout) && qk.n == kQkRows && qk.k == kHidden &&
           qk.group == a8::kGroup && qk.qdata != nullptr && qk.scales != nullptr &&
           value_z.qtype == QType::Q5_G64_FP16 && is_row_split(value_z.layout) &&
           value_z.n == kParentRows && value_z.k == kHidden && value_z.group == a8::kGroup &&
           value_z.qdata != nullptr && value_z.qhigh != nullptr && value_z.scales != nullptr;
}

std::size_t q4_q5_gdn_input_a8_workspace_capacity_bytes(std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q4_q5 gdn_input a8 workspace: invalid token interval");
    }
    return a8::activation_workspace_bytes(kHidden, max_tokens);
}

void q4_q5_gdn_input_a8_launch(const Tensor& x, const Weight& qk, const Weight& value_z, Tensor& qkv,
                               Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!q4_q5_gdn_input_a8_supported(qk, value_z, tokens)) {
        throw std::invalid_argument("q4_q5 gdn_input a8: unsupported profile");
    }

    auto scope = workspace.scope();
    const DeviceSpan codes =
        workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kHidden);
    const DeviceSpan scales = workspace.alloc_bytes(
        static_cast<std::size_t>(tokens) * (kHidden / a8::kGroup) * sizeof(__half));

    auto* code_data  = reinterpret_cast<std::int8_t*>(codes.data);
    auto* scale_data = reinterpret_cast<__half*>(scales.data);
    auto* qkv_data   = static_cast<__nv_bfloat16*>(qkv.data);
    auto* z_data     = static_cast<__nv_bfloat16*>(z.data);

    switch (a8::token_tile(tokens, 512)) {
    case 512:
        launch_all<16>(x, qk, value_z, tokens, code_data, scale_data, qkv_data, z_data, stream);
        return;
    case 256:
        launch_all<8>(x, qk, value_z, tokens, code_data, scale_data, qkv_data, z_data, stream);
        return;
    default:
        launch_all<4>(x, qk, value_z, tokens, code_data, scale_data, qkv_data, z_data, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
