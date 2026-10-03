#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 15;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;
// Abort salvage publishes the live state only when it covers enough committed work that the
// saved rebuild outweighs the checkpoint's retention cost.
inline constexpr std::uint32_t kSalvageMinFrontier = 1024;

[[nodiscard]] inline bool wide_residual_verification(TextPhase phase, std::int32_t batch,
                                                     std::int32_t first, std::int32_t last) {
    return phase == TextPhase::Verify && batch == 1 && first > 16 && first <= last && last <= 64;
}

} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {
using ContractAccess = RuntimeContractAccess;

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace ninfer::models::qwen3_5::detail
