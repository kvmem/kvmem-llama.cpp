#include "ops/common/device_route.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/device_profile.h"
#include "runtime/engine/device_profiles_builtin.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

ninfer::ops::DeviceRouteProfile sample_profile() {
    ninfer::ops::DeviceRouteProfile profile;
    profile.hardware_class  = "test-gpu-sm86";
    profile.multiprocessors = 82;
    profile.origin          = "unit test";
    profile.routes["attn_i8_small/h24/rk8v4/w2"] = {{65536, ""}, {1048576, "4x2x32e"}};
    profile.routes["attn_pv_f16"]                = {{1, "on"}};
    return profile;
}

void test_round_trip() {
    const ninfer::ops::DeviceRouteProfile profile = sample_profile();
    const std::string text = ninfer::runtime::serialize_device_route_profiles({profile});
    const auto parsed      = ninfer::runtime::parse_device_route_profiles(text, "round trip");
    expect(parsed.size() == 1, "one device parses back");
    expect(parsed[0].hardware_class == profile.hardware_class, "hardware class survives");
    expect(parsed[0].multiprocessors == 82, "SM count survives");
    bool same = parsed[0].routes.size() == profile.routes.size();
    for (const auto& [key, bands] : profile.routes) {
        const auto found = parsed[0].routes.find(key);
        same = same && found != parsed[0].routes.end() && found->second.size() == bands.size();
        for (std::size_t i = 0; same && i < bands.size(); ++i) {
            same = found->second[i].last == bands[i].last && found->second[i].schedule == bands[i].schedule;
        }
    }
    expect(same, "route bands survive");
}

void test_rejects_other_documents() {
    bool threw = false;
    try {
        (void)ninfer::runtime::parse_device_route_profiles(R"({"schema": "other", "devices": []})",
                                                           "bad schema");
    } catch (const std::exception&) { threw = true; }
    expect(threw, "a document of another schema is refused");
}

void test_compiled_table_parses() {
    const auto compiled = ninfer::runtime::parse_device_route_profiles(
        ninfer::runtime::compiled_device_route_profiles_json(), "compiled");
    for (const auto& profile : compiled) {
        expect(!profile.hardware_class.empty(), "every compiled entry names its hardware class");
        expect(profile.multiprocessors > 0, "every compiled entry names its SM count");
    }
}

// The cards that ship a measured profile, by the name and compute capability the driver reports, so
// a change to the hardware-class slug cannot silently orphan their entries.
void test_builtin_parts() {
    struct Part {
        const char* name;
        int major;
        int minor;
        int multiprocessors;
    };

    const Part parts[] = {
        {"NVIDIA GeForce RTX 3090", 8, 6, 82},
        {"NVIDIA GeForce RTX 4090", 8, 9, 128},
        {"NVIDIA GeForce RTX 5090", 12, 0, 170},
        {"NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 12, 0, 188},
        {"NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", 12, 0, 188},
        {"NVIDIA RTX PRO 6000 Blackwell Server Edition", 12, 0, 188},
    };
    for (const Part& part : parts) {
        const auto found = ninfer::runtime::find_device_route_profile(
            ninfer::runtime::context_cost_hardware_class(part.name, part.major, part.minor),
            part.multiprocessors, {});
        expect(found.has_value() && !found->routes.empty(), part.name);
    }
}

void test_file_lookup() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("ninfer-device-profile-test-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    ninfer::runtime::upsert_device_route_profile_atomic(path, sample_profile());
    auto found = ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path);
    expect(found.has_value(), "the stored profile is found");
    expect(found && found->routes.count("attn_pv_f16") == 1, "its routes come back");
    expect(!ninfer::runtime::find_device_route_profile("test-gpu-sm86", 84, path).has_value(),
           "a part with another SM count does not take the entry");

    ninfer::ops::DeviceRouteProfile replacement = sample_profile();
    replacement.routes.erase("attn_pv_f16");
    ninfer::runtime::upsert_device_route_profile_atomic(path, replacement);
    found = ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path);
    expect(found && found->routes.count("attn_pv_f16") == 0, "an upsert replaces the entry");
    std::filesystem::remove(path);
}

void test_bands_and_force() {
    using ninfer::ops::device_route_schedule;
    ninfer::ops::install_device_route_profile(
        std::make_shared<const ninfer::ops::DeviceRouteProfile>(sample_profile()));
    expect(device_route_schedule("attn_i8_small/h24/rk8v4/w2", 8192).empty(),
           "a width in a compiled band keeps the compiled route");
    expect(device_route_schedule("attn_i8_small/h24/rk8v4/w2", 262144) == "4x2x32e",
           "a width in a routed band takes its schedule");
    expect(device_route_schedule("attn_i8_small/h24/rk8v4/w2", 2000000).empty(),
           "a width past the last band keeps the compiled route");
    expect(device_route_schedule("unknown/key", 1).empty(), "an unknown key keeps the compiled route");
    {
        const ninfer::ops::DeviceRouteForce force("attn_pv_f16", "");
        expect(device_route_schedule("attn_pv_f16", 1).empty(), "a forced route wins over the profile");
    }
    expect(device_route_schedule("attn_pv_f16", 1) == "on", "the profile returns after the force ends");
    ninfer::ops::install_device_route_profile(nullptr);
    expect(device_route_schedule("attn_pv_f16", 1).empty(), "removing the profile restores the tables");
}

} // namespace

int main() {
    test_round_trip();
    test_rejects_other_documents();
    test_compiled_table_parses();
    test_builtin_parts();
    test_file_lookup();
    test_bands_and_force();
    if (failures != 0) {
        std::cerr << failures << " device profile check(s) failed\n";
        return 1;
    }
    std::cout << "device profile tests passed\n";
    return 0;
}
