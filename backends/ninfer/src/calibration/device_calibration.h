#pragma once

// Device calibration: measures every route family a device profile can steer on the current GPU
// and returns the bands where a schedule other than the compiled one wins.
//
// Each family is timed through the same dispatch an inference call uses, with the family's route
// forced to one candidate at a time (ops::DeviceRouteForce), on synthetic weights of the registered
// model shapes and with L2 flushed before every sample. A candidate replaces the compiled choice
// only where it is faster by more than `margin`, so a profile records measured wins rather than
// noise.

#include "ops/common/device_route.h"

#include <functional>
#include <string>

namespace ninfer::calibration {

struct CalibrationOptions {
    bool ternary   = true; // T2G128 projections (Ternary Bonsai 2)
    bool groupwise = true; // Q4/Q5 fused projections (Qwen3.6/3.8-27B groupwise-int)
    bool attention = true; // INT8-family small-T attention launch tiers
    bool linear_attention = true; // GDN prefill algorithm per prompt width
    int warmup     = 2;
    int repeat     = 11;
    double margin  = 0.03;
    bool detail    = false; // log every candidate's time, not only the winner
    std::string only;       // calibrate only the families whose key starts with this
    std::function<void(const std::string&)> log;
};

[[nodiscard]] ops::DeviceRouteProfile calibrate_device_routes(const CalibrationOptions& options);

} // namespace ninfer::calibration
