#pragma once

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

inline nlohmann::json cuda_residency_json(const CudaResidencySummary& memory) {
    return {{"enabled", memory.enabled},
            {"verified", memory.verified},
            {"cuda_free_bytes", memory.cuda_free_bytes},
            {"cuda_total_bytes", memory.cuda_total_bytes},
            {"dedicated_bytes", memory.dedicated_bytes},
            {"shared_bytes", memory.shared_bytes},
            {"shared_baseline_bytes", memory.shared_baseline_bytes},
            {"device_allocated_bytes", memory.device_allocated_bytes},
            {"host_pool_bytes", memory.host_pool_bytes},
            {"host_used_bytes", memory.host_used_bytes},
            {"verified_reserve_bytes", memory.verified_reserve_bytes}};
}

} // namespace ninfer::serve
