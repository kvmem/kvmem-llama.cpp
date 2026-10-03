#include "ops/linear/common/route_table.h"

#include "ops/common/device_route.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <optional>
#include <span>
#include <string_view>

namespace ninfer::ops::detail {
namespace {

constexpr int kMaxDevices = 64;

struct WidthBand {
    std::int32_t first;
    std::int32_t last;
};

struct FamilyBands {
    std::span<const WidthBand> q4;
    std::span<const WidthBand> q5;
    std::span<const WidthBand> q6;
    std::span<const WidthBand> q8;
    std::span<const WidthBand> fp8;
    std::span<const WidthBand> nvfp4;
    std::span<const WidthBand> bf16;
};

// Widths where the unified table beat the legacy one on the named card, from two
// ninfer_linear_bench sweeps of every shape that has both tables (T 1..32, 40..256 in steps of 8,
// 320..1024 in steps of 64; graph execution, L2 flushed) averaged: at least 3% faster over the
// shapes' geometric mean and no shape more than 3% slower, where a difference under one 2 us timer
// tick counts as a tie. A band spans at least two measured widths and takes the unmeasured widths
// between them. The FP8 and NVFP4 shapes count once per policy the card runs them under: A16, and
// also A8 or A4 on the RTX 5090.
constexpr std::array<WidthBand, 1> kRtx3090Q4{{{25, 32}}};
constexpr std::array<WidthBand, 1> kRtx3090Q5{{{7, 96}}};
constexpr std::array<WidthBand, 3> kRtx3090Q6{{{4, 18}, {21, 40}, {56, 96}}};
constexpr std::array<WidthBand, 1> kRtx3090Q8{{{5, 6}}};
constexpr std::array<WidthBand, 1> kRtx3090Fp8{{{31, 1024}}};
constexpr std::array<WidthBand, 2> kRtx3090Nvfp4{{{27, 28}, {32, 1024}}};
constexpr std::array<WidthBand, 3> kRtx3090Bf16{{{9, 16}, {25, 64}, {136, 1024}}};

constexpr std::array<WidthBand, 3> kRtx4090Q4{{{25, 72}, {88, 96}, {704, 1024}}};
constexpr std::array<WidthBand, 2> kRtx4090Q5{{{9, 128}, {320, 384}}};
constexpr std::array<WidthBand, 2> kRtx4090Q6{{{7, 32}, {56, 96}}};
constexpr std::array<WidthBand, 3> kRtx4090Q8{{{7, 8}, {40, 48}, {104, 120}}};
constexpr std::array<WidthBand, 4> kRtx4090Fp8{{{9, 10}, {12, 16}, {19, 320}, {448, 1024}}};
constexpr std::array<WidthBand, 2> kRtx4090Nvfp4{{{15, 16}, {26, 1024}}};
constexpr std::array<WidthBand, 3> kRtx4090Bf16{{{14, 16}, {21, 64}, {136, 1024}}};

constexpr std::array<WidthBand, 4> kRtx5090Q4{{{25, 112}, {232, 256}, {384, 576}, {704, 768}}};
constexpr std::array<WidthBand, 1> kRtx5090Q5{{{5, 1024}}};
constexpr std::array<WidthBand, 2> kRtx5090Q6{{{4, 32}, {56, 96}}};
constexpr std::array<WidthBand, 1> kRtx5090Q8{{{1, 48}}};
constexpr std::array<WidthBand, 7> kRtx5090Fp8{
    {{5, 6}, {8, 23}, {27, 31}, {40, 128}, {232, 240}, {256, 384}, {512, 1024}}};
constexpr std::array<WidthBand, 4> kRtx5090Nvfp4{{{3, 22}, {24, 200}, {216, 224}, {320, 384}}};
constexpr std::array<WidthBand, 1> kRtx5090Bf16{{{6, 1024}}};

enum class DeviceClass : std::uint8_t {
    Unmeasured,
    Sm86,
    Sm89,
    Sm12x,
};

FamilyBands bands_for(DeviceClass device) {
    switch (device) {
    case DeviceClass::Sm86:
        return {kRtx3090Q4,  kRtx3090Q5,    kRtx3090Q6,  kRtx3090Q8,
                kRtx3090Fp8, kRtx3090Nvfp4, kRtx3090Bf16};
    case DeviceClass::Sm89:
        return {kRtx4090Q4,  kRtx4090Q5,    kRtx4090Q6,  kRtx4090Q8,
                kRtx4090Fp8, kRtx4090Nvfp4, kRtx4090Bf16};
    case DeviceClass::Sm12x:
        return {kRtx5090Q4,  kRtx5090Q5,    kRtx5090Q6,  kRtx5090Q8,
                kRtx5090Fp8, kRtx5090Nvfp4, kRtx5090Bf16};
    case DeviceClass::Unmeasured:
        break;
    }
    return {};
}

std::span<const WidthBand> family_bands(const FamilyBands& bands, LinearRouteFamily family) {
    switch (family) {
    case LinearRouteFamily::Q4:
        return bands.q4;
    case LinearRouteFamily::Q5:
        return bands.q5;
    case LinearRouteFamily::Q6:
        return bands.q6;
    case LinearRouteFamily::Q8:
        return bands.q8;
    case LinearRouteFamily::Fp8:
        return bands.fp8;
    case LinearRouteFamily::Nvfp4:
        return bands.nvfp4;
    case LinearRouteFamily::Bf16:
        return bands.bf16;
    }
    return {};
}

// -1: nothing forced; otherwise the forced table.
std::atomic<int>& forced_for_tests() {
    static std::atomic<int> forced{-1};
    return forced;
}

std::optional<LinearRouteTable> forced_table() {
    const char* value = std::getenv("NINFER_LINEAR_ROUTES");
    if (value == nullptr) return std::nullopt;
    const std::string_view name(value);
    if (name == "legacy") return LinearRouteTable::Legacy;
    if (name == "unified") return LinearRouteTable::Unified;
    return std::nullopt;
}

DeviceClass classify(int device) {
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        cudaGetLastError();
        return DeviceClass::Unmeasured;
    }
    if (major == 8 && minor == 6) return DeviceClass::Sm86;
    if (major == 8 && minor == 9) return DeviceClass::Sm89;
    if (major == 12) return DeviceClass::Sm12x;
    return DeviceClass::Unmeasured;
}

DeviceClass current_device_class() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return DeviceClass::Unmeasured;
    }
    if (device < 0 || device >= kMaxDevices) return classify(device);
    // 0: unknown; otherwise the class plus one. Racing first lookups store the same value.
    static std::array<std::atomic<std::uint8_t>, kMaxDevices> cached{};
    std::atomic<std::uint8_t>& slot = cached[static_cast<std::size_t>(device)];
    std::uint8_t value              = slot.load(std::memory_order_relaxed);
    if (value == 0) {
        value = static_cast<std::uint8_t>(static_cast<std::uint8_t>(classify(device)) + 1);
        slot.store(value, std::memory_order_relaxed);
    }
    return static_cast<DeviceClass>(value - 1);
}

std::optional<LinearRouteTable> forced_route_table() {
    if (const int value = forced_for_tests().load(std::memory_order_relaxed); value >= 0) {
        return static_cast<LinearRouteTable>(value);
    }
    static const std::optional<LinearRouteTable> forced = forced_table();
    return forced;
}

} // namespace

LinearRouteTable linear_route_table(LinearRouteFamily family, std::int32_t t) {
    if (const auto forced = forced_route_table()) return *forced;
    for (const WidthBand band : family_bands(bands_for(current_device_class()), family)) {
        if (t >= band.first && t <= band.last) return LinearRouteTable::Unified;
    }
    return LinearRouteTable::Legacy;
}

LinearRouteTable fused_route_table(std::string_view key, std::int32_t width) {
    if (const auto forced = forced_route_table()) return *forced;
    return device_route_schedule(key, width) == "unified" ? LinearRouteTable::Unified
                                                          : LinearRouteTable::Legacy;
}

void force_linear_route_table(std::optional<LinearRouteTable> table) {
    forced_for_tests().store(table ? static_cast<int>(*table) : -1, std::memory_order_relaxed);
}

} // namespace ninfer::ops::detail
