#pragma once
#include "ninfer/types.h"
#include <nlohmann/json.hpp>

namespace ninfer::serve {
inline nlohmann::json kvmem_status_json(const KvmemMemorySummary& m) {
    return {{"enabled", m.enabled}, {"selected_tokens", m.selected_tokens},
        {"reserve_tokens", m.reserve_tokens}, {"host_payload_budget_bytes", m.host_payload_budget_bytes},
        {"host_payload_bytes", m.host_payload_bytes}, {"statistics_bytes", m.statistics_bytes},
        {"checkpoint_bytes", m.checkpoint_bytes}, {"evaluated_tokens", m.evaluated_tokens},
        {"transfer_staging_bytes", m.transfer_staging_bytes},
        {"statistics_staging_bytes", m.statistics_staging_bytes},
        {"mtp_resident_pages", m.mtp_resident_pages},
        {"retained_sessions", m.retained_sessions}, {"history_hits", m.history_hits},
        {"history_misses", m.history_misses}, {"history_evictions", m.history_evictions},
        {"disk_hits", m.disk_hits}, {"disk_writes", m.disk_writes}, {"disk_errors", m.disk_errors},
        {"active_host_reservation_bytes", m.active_host_reservation_bytes},
        {"checkpoint_tokens", m.checkpoint_tokens}, {"resident_pages", m.resident_pages},
        {"history_swaps", m.history_swaps}, {"history_spilled_bytes", m.history_spilled_bytes},
        {"history_restored_bytes", m.history_restored_bytes}};
}
} // namespace ninfer::serve
