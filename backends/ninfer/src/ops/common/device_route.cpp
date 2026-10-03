#include "ops/common/device_route.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr int kMaxDevices = 64;

struct Forced {
    std::string key;
    std::string schedule;
};

std::mutex& install_mutex() {
    static std::mutex mutex;
    return mutex;
}

// Owns every profile ever installed: a replaced profile stays alive, so a lookup that loaded its
// pointer just before the replacement never reads freed memory. Profiles are installed a handful
// of times per process.
std::vector<std::shared_ptr<const DeviceRouteProfile>>& installed_storage() {
    static std::vector<std::shared_ptr<const DeviceRouteProfile>> profiles;
    return profiles;
}

std::array<std::shared_ptr<const DeviceRouteProfile>, kMaxDevices>& current_storage() {
    static std::array<std::shared_ptr<const DeviceRouteProfile>, kMaxDevices> profiles;
    return profiles;
}

std::array<std::atomic<const DeviceRouteProfile*>, kMaxDevices>& installed_pointers() {
    static std::array<std::atomic<const DeviceRouteProfile*>, kMaxDevices> pointers{};
    return pointers;
}

int current_device() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return device;
}

// A forced route is set only by the calibration tool, on one thread, around synchronous timings.
Forced*& forced_route() {
    static Forced* forced = nullptr;
    return forced;
}

} // namespace

void install_device_route_profile(int device, std::shared_ptr<const DeviceRouteProfile> profile) {
    if (device < 0 || device >= kMaxDevices) { return; }
    const std::lock_guard lock(install_mutex());
    if (profile) { installed_storage().push_back(profile); }
    installed_pointers()[static_cast<std::size_t>(device)].store(profile.get(),
                                                                 std::memory_order_release);
    current_storage()[static_cast<std::size_t>(device)] = std::move(profile);
}

void install_device_route_profile(std::shared_ptr<const DeviceRouteProfile> profile) {
    install_device_route_profile(current_device(), std::move(profile));
}

std::shared_ptr<const DeviceRouteProfile> installed_device_route_profile() {
    const int device = current_device();
    if (device < 0 || device >= kMaxDevices) { return nullptr; }
    const std::lock_guard lock(install_mutex());
    return current_storage()[static_cast<std::size_t>(device)];
}

std::string_view device_route_schedule(std::string_view key, std::int32_t width) {
    if (const Forced* forced = forced_route(); forced != nullptr && forced->key == key) {
        return forced->schedule;
    }
    const int device = current_device();
    if (device < 0 || device >= kMaxDevices) { return {}; }
    const DeviceRouteProfile* profile =
        installed_pointers()[static_cast<std::size_t>(device)].load(std::memory_order_acquire);
    if (profile == nullptr) { return {}; }
    const auto entry = profile->routes.find(key);
    if (entry == profile->routes.end()) { return {}; }
    for (const DeviceRouteBand& band : entry->second) {
        if (width <= band.last) { return band.schedule; }
    }
    return {};
}

DeviceRouteForce::DeviceRouteForce(std::string key, std::string schedule) {
    delete forced_route();
    forced_route() = new Forced{std::move(key), std::move(schedule)};
}

DeviceRouteForce::~DeviceRouteForce() {
    delete forced_route();
    forced_route() = nullptr;
}

} // namespace ninfer::ops
