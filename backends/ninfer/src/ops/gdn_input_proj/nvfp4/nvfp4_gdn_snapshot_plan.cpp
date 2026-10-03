#include "ops/linear/common/route_table.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan_unified.h"
#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include <optional>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct Nvfp4GdnProjectedWorkspace {
    Tensor projected;
    DeviceSpan projection;
};

template <class Allocator>
Nvfp4GdnProjectedWorkspace allocate_workspace(Allocator& allocator, std::int32_t tokens,
                                              LinearPolicy policy) {
    Nvfp4GdnProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {10240, tokens}, 256);
    const std::size_t projection_bytes =
        nvfp4_gdn_input_workspace_capacity_bytes(policy, tokens, tokens);
    if (projection_bytes != 0) { out.projection = allocator.alloc_bytes(projection_bytes, 256); }
    return out;
}

// Upstream registers the A16 schedules only through T=16, where this line's materialized route
// reaches T=64 for a single-row copy verification; wider A16 calls stay on this line's routes.
constexpr std::int32_t kUnifiedA16MaxTokens = 16;

bool unified_route(LinearPolicy policy, std::int32_t tokens) {
    return (allows_a4(policy) || tokens <= kUnifiedA16MaxTokens) &&
           fused_route_table("unified/nvfp4_gdn_input", tokens) == LinearRouteTable::Unified;
}

} // namespace

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                             std::int32_t batch_size) {
    if (tokens <= 0 || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument("nvfp4 gdn conv: invalid B/T domain");
    }
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 gdn conv: invalid compute policy");
    }
    if (batch_size > 1) { return {Nvfp4GdnConvScheduleId::Materialized}; }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
        if (tokens <= 16) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
        // A single-row copy verification reaches 64 columns through the materialized route.
        if (tokens <= 64) { return {Nvfp4GdnConvScheduleId::Materialized}; }
        // Same limit and same consequence as nvfp4_linear_swiglu's -- see the note there.
        // This one is normally unreachable because the SwiGLU throws first, so a reader
        // who lands here has fixed that one and needs to know this is the next wall.
        throw std::invalid_argument(
            "nvfp4 gdn conv: the A16 route is registered only through T=64, and on sm_80/86/89 A16 "
            "is the only policy available for NVFP4 weights, so this artifact cannot serve a "
            "prefill chunk wider than 64 columns on this architecture. See TODO.md section 1.");
    }
    if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
    if (tokens <= 3) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
    return {Nvfp4GdnConvScheduleId::Materialized};
}

static std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes_own(LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn snapshot workspace: invalid token interval");
    }
    (void)nvfp4_gdn_conv_resolve_plan(policy, min_tokens, 1);
    const Nvfp4GdnConvPlan maximum_plan = nvfp4_gdn_conv_resolve_plan(policy, max_tokens, 1);
    if (maximum_plan.schedule != Nvfp4GdnConvScheduleId::Materialized) { return 0; }

    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens, policy);
    return layout.peak_bytes(1);
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    return capacity_by_table(
        min_tokens, max_tokens,
        [&](std::int32_t width) {
            return unified_route(policy, width) ? LinearRouteTable::Unified
                                                : LinearRouteTable::Legacy;
        },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_gdn_snapshot_workspace_capacity_bytes(policy, first, last)
                       : nvfp4_gdn_snapshot_workspace_capacity_bytes_own(policy, first, last);
        });
}

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                 Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    if (unified_route(policy, x.ne[1])) {
        unified::nvfp4_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                             initial_slot, snapshot_base_slot, query, key, value, z,
                                             policy, workspace, stream);
        return;
    }
    switch (nvfp4_gdn_conv_resolve_plan(policy, x.ne[1], 1).schedule) {
    case Nvfp4GdnConvScheduleId::DecodeFusedA16:
        nvfp4_gdn_snapshot_decode_launch(x, weight, conv_weight, conv_states, valid_columns,
                                         initial_slot, snapshot_base_slot, query, key, value, z,
                                         stream);
        return;
    case Nvfp4GdnConvScheduleId::SmallTFusedA16:
        nvfp4_gdn_snapshot_small_t_launch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_slot, snapshot_base_slot, query, key, value, z,
                                          stream);
        return;
    case Nvfp4GdnConvScheduleId::Materialized:
        break;
    }

    auto scope                         = workspace.scope();
    Nvfp4GdnProjectedWorkspace scratch = allocate_workspace(workspace, x.ne[1], policy);
    std::optional<WorkspaceArena> projection_workspace;
    if (scratch.projection.bytes != 0) { projection_workspace.emplace(scratch.projection); }
    nvfp4_gdn_input_dispatch(x, weight, scratch.projected, z, policy,
                             projection_workspace ? &*projection_workspace : nullptr, stream);
    nvfp4_gdn_snapshot_post_launch(scratch.projected, conv_weight, conv_states, valid_columns,
                                   initial_slot, snapshot_base_slot, query, key, value, stream);
}

} // namespace ninfer::ops::detail
