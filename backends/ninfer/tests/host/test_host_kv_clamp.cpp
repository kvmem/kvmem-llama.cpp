// Pins the pinned-host-KV-cache clamp formula (src/core/host_kv_clamp.h) against known inputs.
// Deliberately dependency-free -- no CUDA, no project headers beyond the one under test -- so it
// can build and run on a GitHub-hosted runner. See .github/workflows/host-checks.yml.

#include "core/host_kv_clamp.h"

#include <cstdio>
#include <cstdlib>

namespace {

int failures = 0;

void expect_eq(std::size_t actual, std::size_t expected, const char* label) {
    if (actual != expected) {
        std::fprintf(stderr, "%s: expected %zu, got %zu\n", label, expected, actual);
        ++failures;
    }
}

} // namespace

int main() {
    using ninfer::clamp_host_kv_reservation_bytes;

    // Plenty of free VRAM: the request fits under the budget untouched.
    expect_eq(clamp_host_kv_reservation_bytes(1ULL << 30, 20ULL << 30, 1ULL << 20), 1ULL << 30,
              "ample headroom leaves the request untouched");

    // free_device just over the 1 GiB reserve: budget is (free - 1GiB) / 2, rounded to stride.
    // free_device = 3 GiB -> budget = 1 GiB exactly; request of 2 GiB must clamp down to it.
    expect_eq(clamp_host_kv_reservation_bytes(2ULL << 30, 3ULL << 30, 1ULL << 20), 1ULL << 30,
              "clamps to half of what remains past the 1 GiB floor");

    // free_device at or below the 1 GiB floor: budget is zero, clamping to zero is supported.
    expect_eq(clamp_host_kv_reservation_bytes(1ULL << 30, 1ULL << 30, 1ULL << 20), 0,
              "free VRAM at the floor clamps the reservation to zero");
    expect_eq(clamp_host_kv_reservation_bytes(1ULL << 30, 512ULL << 20, 1ULL << 20), 0,
              "free VRAM under the floor clamps the reservation to zero");

    // Budget not an exact multiple of the page stride: rounds down, never up. free_device is
    // 1 GiB + 10 MiB past the floor, so the budget is 5 MiB; a 3 MiB stride does not divide that
    // evenly, which is the case that exercises the remainder subtraction.
    const std::size_t free_device  = (1ULL << 30) + (10ULL << 20); // 1 GiB + 10 MiB
    const std::size_t stride       = 3ULL << 20;                   // 3 MiB pages
    const std::size_t budget       = (free_device - (1ULL << 30)) / 2; // 5 MiB
    const std::size_t expected     = budget - (budget % stride);       // rounds 5 MiB down to 3 MiB
    expect_eq(clamp_host_kv_reservation_bytes(1ULL << 30, free_device, stride), expected,
              "clamp result is rounded down to a whole number of pages");

    // Request already smaller than the budget: no clamp applied, even with a tight budget.
    expect_eq(clamp_host_kv_reservation_bytes(1ULL << 20, 3ULL << 30, 1ULL << 20), 1ULL << 20,
              "a request under budget is returned unchanged");

    if (failures == 0) { std::printf("OK host_kv_clamp: all checks passed\n"); }
    return failures == 0 ? 0 : 1;
}
