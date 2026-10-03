#pragma once

// Device route profiles: per-GPU measured choices between an Op's interchangeable schedules.
//
// Every route table in ops/ was measured on one card, and the best schedule for a width depends on
// the part: the SM count sets how many CTAs fill a wave, the memory system sets where a GEMV stops
// being latency-bound. A profile names, for one hardware class, the schedule each width band takes;
// an Op asks device_route_schedule() before falling back to its compiled table. Keys name the Op
// and shape ("q4_linear_swiglu/34816x17408x5120", "t2_i8_small/5120x17408"); the axis is the
// Op's width (query columns for projections, visible keys for attention tiers).
//
// Profiles are installed per CUDA device at Engine startup, before any Op runs, and never change
// while kernels are in flight, so lookups take no lock. A lookup answers for the calling thread's
// current device, which lets the ranks of a model split over different parts (an RTX 3090 and a
// 3090 Ti differ in SM count) each follow their own profile.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::ops {

struct DeviceRouteBand {
    std::int32_t last = 0; // inclusive upper bound; each band starts after the previous one
    std::string schedule;
};

struct DeviceRouteProfile {
    std::string hardware_class;
    std::int32_t multiprocessors = 0;
    std::string origin;
    std::map<std::string, std::vector<DeviceRouteBand>, std::less<>> routes;
};

// Installs (null: removes) the profile of `device`, or of the current device.
void install_device_route_profile(int device, std::shared_ptr<const DeviceRouteProfile> profile);
void install_device_route_profile(std::shared_ptr<const DeviceRouteProfile> profile);
[[nodiscard]] std::shared_ptr<const DeviceRouteProfile> installed_device_route_profile();

// The schedule the current device's profile names for `key` at `width`; empty when there is no
// profile, no entry for the key, or the width lies past the entry's last band.
[[nodiscard]] std::string_view device_route_schedule(std::string_view key, std::int32_t width);

// Calibration times candidate schedules with the compiled routes bypassed: while a scope is alive,
// device_route_schedule() answers `schedule` for `key` whatever the installed profile says.
class DeviceRouteForce {
public:
    DeviceRouteForce(std::string key, std::string schedule);
    ~DeviceRouteForce();
    DeviceRouteForce(const DeviceRouteForce&)            = delete;
    DeviceRouteForce& operator=(const DeviceRouteForce&) = delete;
};

// One schedule an Op lets a profile name: the profile's string, the Op's id for it, and the widest
// width the schedule is defined for (0: unbounded). Kernels do not police their width domains, so a
// profile entry outside a schedule's domain is ignored rather than trusted.
template <class Id>
struct DeviceRouteCandidate {
    std::string_view name;
    Id id;
    std::int32_t max_width = 0;
};

// The candidate the installed profile routes `width` to, or null.
template <class Id, class Candidates>
[[nodiscard]] const DeviceRouteCandidate<Id>* routed_candidate(std::string_view key,
                                                               std::int32_t width,
                                                               const Candidates& candidates) {
    const std::string_view routed = device_route_schedule(key, width);
    if (routed.empty()) { return nullptr; }
    for (const DeviceRouteCandidate<Id>& candidate : candidates) {
        if (candidate.name == routed) {
            return candidate.max_width == 0 || width <= candidate.max_width ? &candidate : nullptr;
        }
    }
    return nullptr;
}

} // namespace ninfer::ops
