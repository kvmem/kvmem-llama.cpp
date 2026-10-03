#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan_unified.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_nvfp4_a4() {
    throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace, Nvfp4ScaleLayout,
                                cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_attn_input_fused_rmsnorm_quantize(const Tensor&, const Tensor&, float, float,
                                                    Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_attn_input_fused_rmsnorm_quantize_rowmajor(const Tensor&, const Tensor&, float,
                                                             float, Nvfp4W4a4Workspace,
                                                             cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_attn_input_fused_rmsnorm_launch(const Tensor&, const Tensor&, float, const Weight&,
                                           Tensor&, Tensor&, Tensor&, Tensor&, WorkspaceArena&,
                                           cudaStream_t) {
    reject_nvfp4_a4();
}

// The same routes of upstream's unified NVFP4 family.
namespace unified {

void launch_nvfp4_a4_quantize(const Tensor&, const Weight&, Nvfp4A4Workspace, Nvfp4ScaleLayout,
                              cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId, const Nvfp4A4Operands&, __nv_bfloat16*,
                                cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_attn_input_a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                Nvfp4A4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*,
                                   __nv_bfloat16*, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_gdn_input_a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Nvfp4A4Workspace,
                               cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_add_a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace,
                                cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_swiglu_a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                   cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

} // namespace unified

} // namespace ninfer::ops::detail
