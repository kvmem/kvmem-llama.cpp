// ninfer-calibrate: measures this GPU's device route profile and stores it where ninfer-serve and
// ninfer look for it (EngineOptions::device_profile). The engine calibrates an unknown device on
// its own at first start; this tool refreshes a profile, writes it elsewhere, or prints it for the
// compiled table.

#include "calibration/device_calibration.h"
#include "core/device.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/device_profile.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>

namespace {

[[noreturn]] void usage(const char* program, int code) {
    std::fprintf(
        code == 0 ? stdout : stderr,
        "usage: %s [--device N] [--out PATH] [--print] [--repeat N] [--margin F]\n"
        "          [--no-ternary] [--no-groupwise] [--no-attention]\n"
        "          [--no-linear-attention] [--only PREFIX] [--detail] [--quiet]\n"
        "  Times every route a device profile can steer on this GPU and stores the profile\n"
        "  at PATH (default: $NINFER_DEVICE_PROFILES, else the user cache). --only limits\n"
        "  it to the route keys starting with PREFIX; --detail logs every candidate.\n",
        program);
    std::exit(code);
}

} // namespace

int main(int argc, char** argv) {
    int device = 0;
    std::filesystem::path out;
    bool print = false;
    bool quiet = false;
    ninfer::calibration::CalibrationOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        const auto value = [&]() -> const char* {
            if (index + 1 >= argc) { usage(argv[0], 2); }
            return argv[++index];
        };
        if (arg == "--device") {
            device = std::atoi(value());
        } else if (arg == "--out") {
            out = value();
        } else if (arg == "--print") {
            print = true;
        } else if (arg == "--repeat") {
            options.repeat = std::max(3, std::atoi(value()));
        } else if (arg == "--margin") {
            options.margin = std::atof(value());
        } else if (arg == "--no-ternary") {
            options.ternary = false;
        } else if (arg == "--no-groupwise") {
            options.groupwise = false;
        } else if (arg == "--no-attention") {
            options.attention = false;
        } else if (arg == "--no-linear-attention") {
            options.linear_attention = false;
        } else if (arg == "--only") {
            options.only = value();
        } else if (arg == "--detail") {
            options.detail = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0], 0);
        } else {
            usage(argv[0], 2);
        }
    }
    try {
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (!quiet) {
            options.log = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
        }
        ninfer::ops::DeviceRouteProfile profile = ninfer::calibration::calibrate_device_routes(options);
        profile.hardware_class = ninfer::runtime::context_cost_hardware_class(
            properties.name, properties.major, properties.minor);
        profile.origin = std::string("ninfer-calibrate on ") + properties.name;
        const std::filesystem::path path =
            out.empty() ? ninfer::runtime::default_device_profile_path() : out;
        ninfer::runtime::upsert_device_route_profile_atomic(path, profile);
        std::fprintf(stderr, "stored %zu routed keys for %s (%d SMs) in %s\n", profile.routes.size(),
                     profile.hardware_class.c_str(), profile.multiprocessors, path.string().c_str());
        if (print) {
            std::fputs(ninfer::runtime::serialize_device_route_profiles({profile}).c_str(), stdout);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer-calibrate: %s\n", error.what());
        return 1;
    }
    return 0;
}
