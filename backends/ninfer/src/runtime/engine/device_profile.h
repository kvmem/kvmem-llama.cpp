#pragma once

// Device route profiles on disk and in the binary, and their installation at Engine startup.
//
// A profile file holds one entry per hardware class (context_cost_hardware_class of the GPU):
//
//   {"schema": "ninfer.device-route-profiles", "schema_version": 1,
//    "devices": [{"hardware_class": "nvidia-geforce-rtx-3090-sm86", "multiprocessors": 82,
//                 "origin": "...", "routes": {"<key>": [[last, "<schedule>"], ...], ...}}]}
//
// An empty schedule string keeps the compiled route for its band. Entries whose multiprocessor
// count differs from the device's are not applied: a laptop part can share a desktop part's name
// and not its SM count.

#include "ops/common/device_route.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::runtime {

[[nodiscard]] std::vector<ops::DeviceRouteProfile>
parse_device_route_profiles(std::string_view json, std::string_view source_name);
[[nodiscard]] std::string serialize_device_route_profiles(
    const std::vector<ops::DeviceRouteProfile>& profiles);

// $NINFER_DEVICE_PROFILES, else $XDG_CACHE_HOME/ninfer/device-profiles.json, else
// ~/.cache/ninfer/device-profiles.json (%LOCALAPPDATA% on Windows).
[[nodiscard]] std::filesystem::path default_device_profile_path();

// The profile for this hardware class and SM count: the file's entry when it has one, else the
// compiled table's.
[[nodiscard]] std::optional<ops::DeviceRouteProfile>
find_device_route_profile(std::string_view hardware_class, int multiprocessors,
                          const std::filesystem::path& path);

// Replaces or appends the profile's hardware-class entry in the file, atomically.
void upsert_device_route_profile_atomic(const std::filesystem::path& path,
                                        const ops::DeviceRouteProfile& profile);

} // namespace ninfer::runtime
