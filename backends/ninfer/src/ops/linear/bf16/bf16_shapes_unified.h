#pragma once

#include "ops/linear/bf16/bf16_launch.h"

// Upstream's BF16 shape tables over the unified templates (shapes/*_unified.cu).
namespace ninfer::ops::detail::unified {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);

} // namespace ninfer::ops::detail::unified
