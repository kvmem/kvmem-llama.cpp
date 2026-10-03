#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <span>

namespace ninfer::runtime {

// `available_runtime_bytes` is the primary device's free memory after weights;
// `extra_rank_available_bytes` is the same for each further device, in the order of
// `curve.extra_ranks`. The capacity is the largest that fits every device, and a failure names the
// device that could not hold it.
[[nodiscard]] KvCapacityResolution
resolve_kv_capacity(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                    std::size_t available_runtime_bytes,
                    std::span<const std::size_t> extra_rank_available_bytes = {});

// Mixed allows the driver to satisfy an allocation beyond its reported free bytes. Explicit
// capacity is preserved; automatic capacity is bounded by one full context window per lane and
// the plan's representable page count. There is no memory probe or fallback to a smaller pool.
// `available_runtime_bytes` remains the observed free memory, used only for reporting.
[[nodiscard]] KvCapacityResolution
resolve_mixed_kv_capacity(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                          std::uint32_t max_context, std::uint32_t max_concurrency,
                          std::size_t available_runtime_bytes);

} // namespace ninfer::runtime
