#pragma once

#include <cstddef>

namespace ninfer {

// The pinned host KV cache competes with the model for *device* memory on Windows (WDDM maps a
// pinned host allocation into the GPU's address space and charges it against the same budget --
// see the measured numbers and failure mode next to this function's caller in program_impl.h).
// Clamp the requested reservation to what is actually safe to take, rounding the result down to a
// whole number of KV pages so the caller never has to deal with a partial-page remainder.
//
// Pure arithmetic, no CUDA dependency, so it can be unit-tested without a device -- see
// tests/host/test_host_kv_clamp.cpp.
inline std::size_t clamp_host_kv_reservation_bytes(std::size_t requested_bytes,
                                                    std::size_t free_device_bytes,
                                                    std::size_t minimum_stride) noexcept {
    constexpr std::size_t kPinnedHostReserveBytes = 1ULL << 30;
    const std::size_t budget = free_device_bytes > kPinnedHostReserveBytes
                                    ? (free_device_bytes - kPinnedHostReserveBytes) / 2
                                    : 0;
    if (requested_bytes > budget) { return budget - (budget % minimum_stride); }
    return requested_bytes;
}

} // namespace ninfer
