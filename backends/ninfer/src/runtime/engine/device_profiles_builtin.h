#pragma once

#include <string_view>

namespace ninfer::runtime {

// Device route profiles measured on the parts named in them (ninfer-calibrate) and compiled in;
// a matching entry in the user's profile file takes precedence.
[[nodiscard]] std::string_view compiled_device_route_profiles_json();

} // namespace ninfer::runtime
