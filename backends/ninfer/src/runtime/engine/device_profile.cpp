#include "runtime/engine/device_profile.h"

#include "runtime/engine/device_profiles_builtin.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ninfer::runtime {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::string_view kSchema = "ninfer.device-route-profiles";
constexpr int kSchemaVersion       = 1;

int process_id() noexcept {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

[[noreturn]] void fail(std::string_view source, const std::string& message) {
    throw std::invalid_argument(std::string(source) + ": " + message);
}

ops::DeviceRouteProfile parse_device(const Json& entry, std::string_view source) {
    if (!entry.is_object() || !entry.contains("hardware_class") || !entry.contains("routes")) {
        fail(source, "a device entry needs hardware_class and routes");
    }
    ops::DeviceRouteProfile profile;
    profile.hardware_class = entry.at("hardware_class").get<std::string>();
    if (profile.hardware_class.empty()) { fail(source, "hardware_class is empty"); }
    profile.multiprocessors = entry.value("multiprocessors", 0);
    profile.origin          = entry.value("origin", std::string(source));
    const Json& routes      = entry.at("routes");
    if (!routes.is_object()) { fail(source, "routes must be an object"); }
    for (const auto& [key, bands] : routes.items()) {
        if (!bands.is_array() || bands.empty()) { fail(source, key + ": bands must be a nonempty array"); }
        std::vector<ops::DeviceRouteBand> parsed;
        std::int64_t previous = 0;
        for (const Json& band : bands) {
            if (!band.is_array() || band.size() != 2 || !band[0].is_number_integer() ||
                !band[1].is_string()) {
                fail(source, key + ": a band is [last, schedule]");
            }
            const std::int64_t last = band[0].get<std::int64_t>();
            if (last <= previous || last > std::numeric_limits<std::int32_t>::max()) {
                fail(source, key + ": band bounds must increase");
            }
            previous = last;
            parsed.push_back({static_cast<std::int32_t>(last), band[1].get<std::string>()});
        }
        profile.routes.emplace(key, std::move(parsed));
    }
    return profile;
}

Json device_json(const ops::DeviceRouteProfile& profile) {
    Json routes = Json::object();
    for (const auto& [key, bands] : profile.routes) {
        Json list = Json::array();
        for (const ops::DeviceRouteBand& band : bands) { list.push_back(Json::array({band.last, band.schedule})); }
        routes[key] = std::move(list);
    }
    return Json{{"hardware_class", profile.hardware_class},
                {"multiprocessors", profile.multiprocessors},
                {"origin", profile.origin},
                {"routes", std::move(routes)}};
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("failed to open device profiles: " + path.string()); }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::optional<ops::DeviceRouteProfile> find_in(const std::vector<ops::DeviceRouteProfile>& profiles,
                                               std::string_view hardware_class,
                                               int multiprocessors) {
    for (const ops::DeviceRouteProfile& profile : profiles) {
        if (profile.hardware_class == hardware_class &&
            (profile.multiprocessors == 0 || profile.multiprocessors == multiprocessors)) {
            return profile;
        }
    }
    return std::nullopt;
}

} // namespace

std::vector<ops::DeviceRouteProfile> parse_device_route_profiles(std::string_view json,
                                                                 std::string_view source_name) {
    Json document;
    try {
        document = Json::parse(json.begin(), json.end());
    } catch (const nlohmann::json::exception& error) {
        fail(source_name, std::string("invalid JSON: ") + error.what());
    }
    if (!document.is_object() || document.value("schema", std::string()) != kSchema ||
        document.value("schema_version", 0) != kSchemaVersion || !document.contains("devices") ||
        !document.at("devices").is_array()) {
        fail(source_name, "not a ninfer.device-route-profiles v1 document");
    }
    std::vector<ops::DeviceRouteProfile> profiles;
    for (const Json& entry : document.at("devices")) { profiles.push_back(parse_device(entry, source_name)); }
    return profiles;
}

std::string serialize_device_route_profiles(const std::vector<ops::DeviceRouteProfile>& profiles) {
    Json devices = Json::array();
    for (const ops::DeviceRouteProfile& profile : profiles) { devices.push_back(device_json(profile)); }
    const Json document{{"schema", kSchema}, {"schema_version", kSchemaVersion}, {"devices", std::move(devices)}};
    return document.dump(1) + '\n';
}

std::filesystem::path default_device_profile_path() {
    if (const char* explicit_path = std::getenv("NINFER_DEVICE_PROFILES");
        explicit_path != nullptr && explicit_path[0] != '\0') {
        return explicit_path;
    }
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA"); local != nullptr && local[0] != '\0') {
        return std::filesystem::path(local) / "ninfer" / "device-profiles.json";
    }
#endif
    if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache != nullptr && cache[0] != '\0') {
        return std::filesystem::path(cache) / "ninfer" / "device-profiles.json";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / ".cache" / "ninfer" / "device-profiles.json";
    }
    return std::filesystem::path("device-profiles.json");
}

std::optional<ops::DeviceRouteProfile> find_device_route_profile(std::string_view hardware_class,
                                                                 int multiprocessors,
                                                                 const std::filesystem::path& path) {
    std::error_code error;
    if (!path.empty() && std::filesystem::exists(path, error)) {
        const auto profiles = parse_device_route_profiles(read_file(path), path.string());
        if (auto found = find_in(profiles, hardware_class, multiprocessors)) { return found; }
    }
    static const std::vector<ops::DeviceRouteProfile> compiled =
        parse_device_route_profiles(compiled_device_route_profiles_json(), "compiled device profiles");
    return find_in(compiled, hardware_class, multiprocessors);
}

void upsert_device_route_profile_atomic(const std::filesystem::path& path,
                                        const ops::DeviceRouteProfile& profile) {
    std::vector<ops::DeviceRouteProfile> profiles;
    std::error_code error;
    if (std::filesystem::exists(path, error)) {
        profiles = parse_device_route_profiles(read_file(path), path.string());
    }
    bool replaced = false;
    for (ops::DeviceRouteProfile& existing : profiles) {
        if (existing.hardware_class == profile.hardware_class &&
            existing.multiprocessors == profile.multiprocessors) {
            existing = profile;
            replaced = true;
        }
    }
    if (!replaced) { profiles.push_back(profile); }
    const std::string serialized = serialize_device_route_profiles(profiles);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(process_id()) + "." +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("failed to write device profiles: " + temporary.string()); }
        output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
        if (!output) { throw std::runtime_error("failed to write device profiles: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, path);
}

} // namespace ninfer::runtime
