#include "calibration/device_calibration.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/linear/t2/t2_a8.h"
#include "ops/linear_add/q4/q4_linear_add_dispatch.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::calibration {
namespace {

__global__ void fill_u16_kernel(std::uint16_t* values, std::uint64_t count, std::uint16_t bits) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        values[index] = bits;
    }
}

__global__ void fill_f32_kernel(float* values, std::uint64_t count, float value) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        values[index] = value;
    }
}

__global__ void fill_bf16_ramp_kernel(std::uint16_t* values, std::uint64_t count) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        // bf16 values in [-0.5, 0.5): exact magnitudes are irrelevant to timing.
        const float value = 0.5f - static_cast<float>(index % 251) / 250.0f;
        values[index]     = static_cast<std::uint16_t>(__float_as_uint(value) >> 16);
    }
}

__global__ void fill_random_kernel(std::uint32_t* words, std::uint64_t count, std::uint32_t seed) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        std::uint32_t x = static_cast<std::uint32_t>(index) * 0x9E3779B9u ^ seed;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        words[index] = x;
    }
}

int fill_grid(std::uint64_t count) {
    return static_cast<int>(std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (count + 255) / 256)));
}

void fill_random(void* data, std::size_t bytes, std::uint32_t seed) {
    const std::uint64_t words = bytes / 4;
    fill_random_kernel<<<fill_grid(words), 256>>>(static_cast<std::uint32_t*>(data), words, seed);
    CUDA_CHECK(cudaGetLastError());
}

DeviceBuffer bf16_buffer(std::size_t elements) {
    DeviceBuffer buffer(std::max<std::size_t>(elements * 2, 256));
    fill_bf16_ramp_kernel<<<fill_grid(elements), 256>>>(static_cast<std::uint16_t*>(buffer.p),
                                                        elements);
    CUDA_CHECK(cudaGetLastError());
    return buffer;
}

// A row-split weight with the layout the artifacts use: code plane, optional high plane, FP16
// scales. Codes are a fixed byte pattern that every format decodes to valid values.
struct SyntheticWeight {
    DeviceBuffer storage;
    Weight weight{};
};

SyntheticWeight row_split_weight(QType qtype, std::int32_t n, std::int32_t k) {
    std::int32_t group      = 64;
    std::int32_t high_bytes = 0;
    switch (qtype) {
    case QType::Q4_G64_FP16:
        break;
    case QType::Q5_G64_FP16:
        high_bytes = 8;
        break;
    case QType::Q6_G64_FP16:
        high_bytes = 16;
        break;
    case QType::Q8_G32_FP16:
        group = 32;
        break;
    case QType::T2_G128_FP16:
        group = 128;
        break;
    default:
        throw std::invalid_argument("calibration: unsupported synthetic weight format");
    }
    const auto align    = [](std::uint64_t value) { return (value + 255) / 256 * 256; };
    const std::uint64_t groups =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k / group);
    const std::uint64_t low_bytes    = groups * 32;
    const std::uint64_t high_total   = groups * static_cast<std::uint64_t>(high_bytes);
    const std::uint64_t high_offset  = align(low_bytes);
    const std::uint64_t scale_offset = high_offset + align(high_total);
    const std::uint64_t total        = scale_offset + groups * 2;

    SyntheticWeight result{DeviceBuffer(static_cast<std::size_t>(total)), {}};
    auto* base = static_cast<std::uint8_t*>(result.storage.p);
    fill_random(base, low_bytes, 0x5eedu + static_cast<std::uint32_t>(n));
    if (high_total != 0) { fill_random(base + high_offset, high_total, 0xbeefu + static_cast<std::uint32_t>(k)); }
    fill_u16_kernel<<<fill_grid(groups), 256>>>(
        reinterpret_cast<std::uint16_t*>(base + scale_offset), groups, 0x3c00);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& w           = result.weight;
    w.payload           = base;
    w.payload_bytes     = total;
    w.high_plane_bytes  = high_total;
    w.qtype             = qtype;
    w.group_size        = static_cast<std::uint32_t>(group);
    w.shape[0]          = n;
    w.shape[1]          = k;
    w.padded_shape[0]   = n;
    w.padded_shape[1]   = k;
    w.ndim              = 2;
    w.qdata             = base;
    w.qhigh             = high_total == 0 ? nullptr : base + high_offset;
    w.scales            = base + scale_offset;
    w.n                 = n;
    w.k                 = k;
    w.group             = group;
    w.layout            = QuantLayout::RowSplit;
    w.scale_dtype       = DType::FP16;
    return result;
}

class Timer {
public:
    explicit Timer(const CalibrationOptions& options)
        : options_(options), flush_(flush_bytes()) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~Timer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    Timer(const Timer&)            = delete;
    Timer& operator=(const Timer&) = delete;

    // Median microseconds of `launch` with L2 flushed before each sample, or infinity when the
    // launch throws (a candidate outside its domain or an unsupported shape).
    double median_us(const std::function<void(cudaStream_t)>& launch) {
        try {
            for (int index = 0; index < options_.warmup; ++index) {
                flush();
                launch(nullptr);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> samples;
            samples.reserve(static_cast<std::size_t>(options_.repeat));
            for (int index = 0; index < options_.repeat; ++index) {
                flush();
                CUDA_CHECK(cudaEventRecord(start_, nullptr));
                launch(nullptr);
                CUDA_CHECK(cudaEventRecord(stop_, nullptr));
                CUDA_CHECK(cudaEventSynchronize(stop_));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
                samples.push_back(ms);
            }
            std::sort(samples.begin(), samples.end());
            return 1000.0 * samples[samples.size() / 2];
        } catch (const std::exception&) {
            cudaGetLastError();
            cudaDeviceSynchronize();
            return std::numeric_limits<double>::infinity();
        }
    }

private:
    static std::size_t flush_bytes() {
        int device = 0;
        int l2     = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
        return std::max<std::size_t>(64ULL << 20, static_cast<std::size_t>(l2) * 4);
    }
    void flush() { CUDA_CHECK(cudaMemsetAsync(flush_.p, 0xa5, flush_.bytes, nullptr)); }

    const CalibrationOptions& options_;
    DeviceBuffer flush_;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_  = nullptr;
};

struct Family {
    std::string key;
    std::vector<std::string> candidates;
    std::vector<std::int32_t> widths;
    std::function<void(std::int32_t, cudaStream_t)> run;
    // The BF16 output a run at this width writes, compared between the compiled route and each
    // candidate: a schedule that computes something else must never win on speed.
    std::function<std::pair<const void*, std::size_t>(std::int32_t)> output;
    // Restores any state a run accumulates into (a residual) before a checked run.
    std::function<void()> reset;
};

std::vector<float> bf16_host(std::pair<const void*, std::size_t> output) {
    std::vector<std::uint16_t> raw(output.second);
    CUDA_CHECK(cudaMemcpy(raw.data(), output.first, raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> values(raw.size());
    for (std::size_t index = 0; index < raw.size(); ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(raw[index]) << 16;
        float value              = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        values[index] = value;
    }
    return values;
}

// Largest element difference over the reference's largest magnitude; infinity for non-finite.
double relative_difference(const std::vector<float>& reference, const std::vector<float>& other) {
    double scale = 1e-6;
    double worst = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(other[index])) { return std::numeric_limits<double>::infinity(); }
        scale = std::max(scale, static_cast<double>(std::fabs(reference[index])));
        worst = std::max(worst, static_cast<double>(std::fabs(reference[index] - other[index])));
    }
    return worst / scale;
}

void report(const CalibrationOptions& options, const std::string& line) {
    if (options.log) { options.log(line); }
}

// Times the compiled choice and every candidate at each width and turns the winners into bands.
// An empty schedule is the compiled route; a candidate replaces it only when faster by `margin`.
std::vector<ops::DeviceRouteBand> sweep(const Family& family, Timer& timer,
                                        const CalibrationOptions& options) {
    std::vector<ops::DeviceRouteBand> bands;
    if (!options.only.empty() && family.key.rfind(options.only, 0) != 0) { return bands; }
    for (const std::int32_t width : family.widths) {
        const double compiled = timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
        std::vector<float> reference;
        if (family.output) {
            if (family.reset) { family.reset(); }
            family.run(width, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            reference = bf16_host(family.output(width));
        }
        double best_us = compiled;
        std::string best;
        for (const std::string& candidate : family.candidates) {
            const ops::DeviceRouteForce force(family.key, candidate);
            if (family.output) {
                try {
                    if (family.reset) { family.reset(); }
                    family.run(width, nullptr);
                    CUDA_CHECK(cudaDeviceSynchronize());
                } catch (const std::exception&) {
                    cudaGetLastError();
                    continue;
                }
                const double difference = relative_difference(reference, bf16_host(family.output(width)));
                if (!(difference <= 0.05)) {
                    char line[256];
                    std::snprintf(line, sizeof(line), "%s w=%d %s rejected: output differs by %.3g",
                                  family.key.c_str(), width, candidate.c_str(), difference);
                    report(options, line);
                    continue;
                }
            }
            const double us = timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
            if (options.detail) {
                char line[256];
                std::snprintf(line, sizeof(line), "%s w=%d %s %.1f us (compiled %.1f us)",
                              family.key.c_str(), width, candidate.c_str(), us, compiled);
                report(options, line);
            }
            if (std::isfinite(us) && us < compiled * (1.0 - options.margin) && us < best_us) {
                best_us = us;
                best    = candidate;
            }
        }
        // A win must survive a second, interleaved measurement of both routes.
        if (!best.empty()) {
            double again_compiled = 0.0;
            double again_best     = 0.0;
            for (int round = 0; round < 2; ++round) {
                again_compiled += timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
                const ops::DeviceRouteForce force(family.key, best);
                again_best += timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
            }
            if (!(again_best < again_compiled * (1.0 - options.margin / 2))) { best.clear(); }
        }
        char line[256];
        std::snprintf(line, sizeof(line), "%s w=%d compiled %.1f us -> %s %.1f us", family.key.c_str(),
                      width, compiled, best.empty() ? "compiled" : best.c_str(),
                      best.empty() ? compiled : best_us);
        report(options, line);
        if (!bands.empty() && bands.back().schedule == best) {
            bands.back().last = width;
        } else {
            bands.push_back({width, best});
        }
    }
    const bool any = std::any_of(bands.begin(), bands.end(),
                                 [](const ops::DeviceRouteBand& band) { return !band.schedule.empty(); });
    if (!any) { bands.clear(); }
    return bands;
}

std::vector<std::int32_t> decode_widths() {
    return {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
}

// Ternary Bonsai 2 27B projections, in the launches the model makes: the GDN and attention input
// pairs, the output and down projections with their residual adds, the gate/up and the LM head.
void calibrate_ternary(ops::DeviceRouteProfile& profile, Timer& timer,
                       const CalibrationOptions& options) {
    constexpr std::int32_t hidden = 5120;
    const std::vector<std::string> small = {"r16c8",   "r16c16",  "r32c32",   "r16c8w1",
                                            "r16c8s3", "r32c8",   "r32c8k4",  "r16c16w1",
                                            "r32c16",  "r16c32",  "r32c16k8"};
    constexpr std::int32_t max_tokens = 192;
    SyntheticWeight gdn_qk      = row_split_weight(QType::T2_G128_FP16, 4096, hidden);
    SyntheticWeight gdn_vz      = row_split_weight(QType::T2_G128_FP16, 12288, hidden);
    SyntheticWeight attn_qk     = row_split_weight(QType::T2_G128_FP16, 7168, hidden);
    SyntheticWeight attn_gv     = row_split_weight(QType::T2_G128_FP16, 7168, hidden);
    SyntheticWeight out_proj    = row_split_weight(QType::T2_G128_FP16, hidden, 6144);
    SyntheticWeight gate_up     = row_split_weight(QType::T2_G128_FP16, 34816, hidden);
    SyntheticWeight down        = row_split_weight(QType::T2_G128_FP16, hidden, 17408);
    SyntheticWeight head        = row_split_weight(QType::T2_G128_FP16, 248320, hidden);
    DeviceBuffer x_hidden       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer x_mixer        = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer x_intermediate = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    DeviceBuffer qkv            = bf16_buffer(static_cast<std::size_t>(10240) * max_tokens);
    DeviceBuffer z              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer q              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer gate           = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer k              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer v              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer residual       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer wide           = bf16_buffer(static_cast<std::size_t>(34816) * max_tokens);
    DeviceBuffer logits         = bf16_buffer(static_cast<std::size_t>(248320) * 16);
    WorkspaceArena workspace(256ULL << 20);

    const auto tensor = [](DeviceBuffer& buffer, std::int32_t rows, std::int32_t cols) {
        return Tensor(buffer.p, DType::BF16, {rows, cols});
    };
    const auto gdn_pair = [&](std::int32_t cols, cudaStream_t stream) {
        auto scope        = workspace.scope();
        Tensor x          = tensor(x_hidden, hidden, cols);
        Tensor qkv_plane  = tensor(qkv, 10240, cols);
        Tensor z_plane    = tensor(z, 6144, cols);
        const auto active = ops::detail::t2_a8_quantize(x, workspace, stream);
        ops::detail::t2_a8_project_split_pair(active, {gdn_qk.weight, qkv_plane, 0, 4096, qkv_plane, 0},
                                              {gdn_vz.weight, qkv_plane, 4096, 6144, z_plane, 0},
                                              stream);
    };
    const auto attn_pair = [&](std::int32_t cols, cudaStream_t stream) {
        auto scope        = workspace.scope();
        Tensor x          = tensor(x_hidden, hidden, cols);
        Tensor q_plane    = tensor(q, 6144, cols);
        Tensor gate_plane = tensor(gate, 6144, cols);
        Tensor k_plane    = tensor(k, 1024, cols);
        Tensor v_plane    = tensor(v, 1024, cols);
        const auto active = ops::detail::t2_a8_quantize(x, workspace, stream);
        ops::detail::t2_a8_project_split_pair(active, {attn_qk.weight, q_plane, 0, 6144, k_plane, 0},
                                              {attn_gv.weight, gate_plane, 0, 6144, v_plane, 0},
                                              stream);
    };
    const auto out_add = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_mixer, 6144, cols);
        Tensor res = tensor(residual, hidden, cols);
        ops::detail::t2_a8_linear_add(x, out_proj.weight, res, workspace, stream);
    };
    const auto gate_up_linear = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_hidden, hidden, cols);
        Tensor out = tensor(wide, 34816, cols);
        ops::detail::t2_a8_linear(x, gate_up.weight, out, workspace, stream);
    };
    const auto down_add = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_intermediate, 17408, cols);
        Tensor res = tensor(residual, hidden, cols);
        ops::detail::t2_a8_linear_add(x, down.weight, res, workspace, stream);
    };
    const auto head_linear = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_hidden, hidden, cols);
        Tensor out = tensor(logits, 248320, cols);
        ops::detail::t2_a8_linear(x, head.weight, out, workspace, stream);
    };

    DeviceBuffer residual_pristine = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    const auto view = [](DeviceBuffer& buffer, std::size_t rows) {
        return [&buffer, rows](std::int32_t cols) {
            return std::pair<const void*, std::size_t>{buffer.p, rows * static_cast<std::size_t>(cols)};
        };
    };
    const auto reset_residual = [&]() {
        CUDA_CHECK(cudaMemcpy(residual.p, residual_pristine.p, residual.bytes, cudaMemcpyDeviceToDevice));
    };
    const std::vector<Family> families = {
        {"t2_i8_small/4096+12288x5120", small, decode_widths(), gdn_pair, view(qkv, 10240), {}},
        {"t2_i8_small/7168+7168x5120", small, decode_widths(), attn_pair, view(q, 6144), {}},
        {"t2_i8_small/5120x6144", small, decode_widths(), out_add, view(residual, hidden), reset_residual},
        {"t2_i8_small/34816x5120", small, decode_widths(), gate_up_linear, view(wide, 34816), {}},
        {"t2_i8_small/5120x17408", small, decode_widths(), down_add, view(residual, hidden), reset_residual},
        {"t2_i8_small/248320x5120", small, {1, 2, 3, 4, 5, 6, 7, 8, 12, 16}, head_linear,
         view(logits, 248320), {}},
        {"t2_i8_route",
         {"small", "tile"},
         {16, 24, 32, 48, 64, 96, 128, 160, 192},
         [&](std::int32_t cols, cudaStream_t stream) {
             gdn_pair(cols, stream);
             out_add(cols, stream);
             gate_up_linear(cols, stream);
             down_add(cols, stream);
         },
         view(wide, 34816),
         reset_residual},
    };
    for (const Family& family : families) {
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

// Qwen3.6/3.8-27B groupwise-int fused projections at their registered shapes.
void calibrate_groupwise(ops::DeviceRouteProfile& profile, Timer& timer,
                         const CalibrationOptions& options) {
    constexpr std::int32_t hidden     = 5120;
    constexpr std::int32_t max_tokens = 64;
    SyntheticWeight attn_qk = row_split_weight(QType::Q4_G64_FP16, 7168, hidden);
    SyntheticWeight attn_gv = row_split_weight(QType::Q5_G64_FP16, 7168, hidden);
    SyntheticWeight gdn_qk  = row_split_weight(QType::Q4_G64_FP16, 4096, hidden);
    SyntheticWeight gdn_vz  = row_split_weight(QType::Q5_G64_FP16, 12288, hidden);
    SyntheticWeight out     = row_split_weight(QType::Q5_G64_FP16, hidden, 6144);
    SyntheticWeight down    = row_split_weight(QType::Q5_G64_FP16, hidden, 17408);
    SyntheticWeight gate_up = row_split_weight(QType::Q4_G64_FP16, 34816, hidden);
    SyntheticWeight out_q4            = row_split_weight(QType::Q4_G64_FP16, hidden, 6144);
    DeviceBuffer x_hidden       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer x_mixer        = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer x_intermediate = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    DeviceBuffer q              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer gate           = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer k              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer v              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer qkv            = bf16_buffer(static_cast<std::size_t>(10240) * max_tokens);
    DeviceBuffer z              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer residual       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer swiglu_out     = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    WorkspaceArena workspace(256ULL << 20);
    const auto tensor = [](DeviceBuffer& buffer, std::int32_t rows, std::int32_t cols) {
        return Tensor(buffer.p, DType::BF16, {rows, cols});
    };
    std::vector<std::int32_t> widths = decode_widths();
    widths.insert(widths.end(), {40, 48, 56, 64});
    DeviceBuffer residual_pristine = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    const auto view = [](DeviceBuffer& buffer, std::size_t rows) {
        return [&buffer, rows](std::int32_t cols) {
            return std::pair<const void*, std::size_t>{buffer.p, rows * static_cast<std::size_t>(cols)};
        };
    };
    const auto reset_residual = [&]() {
        CUDA_CHECK(cudaMemcpy(residual.p, residual_pristine.p, residual.bytes, cudaMemcpyDeviceToDevice));
    };

    const std::vector<Family> families = {
        {"q4_q5_attn_input/5120x6144x1024",
         {"parent_split_fixed", "small_t_mma", "grouped_r32_c32_s4", "grouped_r32_c64_s4",
          "mixed_r32_c64_s3", "pair_r32_c64_s3", "mixed_r64_c128_s2"},
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_hidden, hidden, cols);
             Tensor qp = tensor(q, 6144, cols), gp = tensor(gate, 6144, cols);
             Tensor kp = tensor(k, 1024, cols), vp = tensor(v, 1024, cols);
             ops::detail::q4_q5_attn_input_dispatch(x, attn_qk.weight, attn_gv.weight, qp, gp, kp, vp,
                                                    stream);
         },
         view(q, 6144),
         {}},
        {"q4_q5_gdn_input/5120x4096x12288",
         {"independent_direct", "small_t_mma", "grouped_r64_c8", "grouped_r64_c16", "grouped_r64_c32",
          "grouped_r64_c64", "grouped_r64_c128", "grouped_r32_c32_s2", "grouped_r32_c64_s4"},
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x  = tensor(x_hidden, hidden, cols);
             Tensor qp = tensor(qkv, 10240, cols), zp = tensor(z, 6144, cols);
             ops::detail::q4_q5_gdn_input_dispatch(x, gdn_qk.weight, gdn_vz.weight, qp, zp, stream);
         },
         view(qkv, 10240),
         {}},
        {"q5_linear_add/5120x6144",
         {"split2_exact", "small_t_mma", "mma_r64_c16", "mma_r64_c24", "mma_r64_c32", "mma_r64_c64",
          "mma_r64_c32_s3", "mma_r64_c32_s4", "mma_r64_c128"},
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_mixer, 6144, cols), res = tensor(residual, hidden, cols);
             ops::detail::q5_linear_add_dispatch(x, out.weight, res, workspace, stream);
         },
         view(residual, hidden),
         reset_residual},
        {"q5_linear_add/5120x17408",
         {"split2_exact", "small_t_mma", "mma_r64_c16", "mma_r64_c24", "mma_r64_c32", "mma_r64_c64",
          "mma_r64_c32_s3", "mma_r64_c32_s4", "mma_r64_c128"},
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_intermediate, 17408, cols), res = tensor(residual, hidden, cols);
             ops::detail::q5_linear_add_dispatch(x, down.weight, res, workspace, stream);
         },
         view(residual, hidden),
         reset_residual},
        {"q4_linear_swiglu/34816x17408x5120",
         {"gemv_pair", "small_t_tiled", "split_half_pair_c40", "split_half_pair_c48",
          "split_half_pair_c128", "split_half_pair_c128_tail"},
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_hidden, hidden, cols), o = tensor(swiglu_out, 17408, cols);
             ops::detail::q4_linear_swiglu_dispatch(x, gate_up.weight, o, workspace, stream);
         },
         view(swiglu_out, 17408),
         {}},
    };
    // Upstream's move of these projections onto the unified templates, against the routes the
    // families above left: one switch per Op, measured over the widths its fused launches serve.
    const auto widths_to = [&](std::int32_t last) {
        std::vector<std::int32_t> out;
        for (const std::int32_t width : widths) {
            if (width <= last) { out.push_back(width); }
        }
        return out;
    };
    const std::vector<Family> unified_families = {
        {"unified/q4_q5_attn_input",
         {"unified"},
         widths_to(12),
         families[0].run,
         families[0].output,
         {}},
        {"unified/q4_q5_gdn_input",
         {"unified"},
         widths_to(15),
         families[1].run,
         families[1].output,
         {}},
        {"unified/q5_linear_add/5120x6144",
         {"unified"},
         widths,
         families[2].run,
         families[2].output,
         families[2].reset},
        {"unified/q5_linear_add/5120x17408",
         {"unified"},
         widths,
         families[3].run,
         families[3].output,
         families[3].reset},
        {"unified/q4_linear_swiglu",
         {"unified"},
         widths_to(32),
         families[4].run,
         families[4].output,
         {}},
        {"unified/q4_linear_add",
         {"unified"},
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_mixer, 6144, cols), res = tensor(residual, hidden, cols);
             ops::detail::select_q4_linear_add(hidden, 6144, cols)(x, out_q4.weight, res, stream);
         },
         view(residual, hidden),
         reset_residual},
    };
    for (const auto* set : {&families, &unified_families}) {
        for (const Family& family : *set) {
            auto bands = sweep(family, timer, options);
            if (!bands.empty()) {
                profile.routes[family.key] = std::move(bands);
                ops::install_device_route_profile(
                    std::make_shared<const ops::DeviceRouteProfile>(profile));
            }
        }
    }
}

// The INT8-family small-T partial kernel's tiers. A serving process captures its graphs over an
// envelope as wide as the KV capacity, so the tier is chosen per envelope and has to serve every
// depth below it: each envelope is timed at its full depth and at a sixteenth of it.
// The two-stage GDN prefill against the WY/state-passing pipeline at the prompt widths prefill
// takes, for the 27B's 48 value heads and the 35B-A3B's 32 (16 query/key heads each).
void calibrate_linear_attention(ops::DeviceRouteProfile& profile, Timer& timer,
                                const CalibrationOptions& options) {
    constexpr std::int32_t state_dim       = 128;
    constexpr std::int32_t qk_heads        = 16;
    const std::vector<std::int32_t> widths = {16,  32,  48,   64,   96,   128,
                                              256, 512, 1024, 2048, 4096, 8192};
    const std::int32_t max_tokens          = widths.back();
    const float scale                      = 1.0f / std::sqrt(static_cast<float>(state_dim));
    for (const std::int32_t value_heads : {48, 32}) {
        const auto f32_buffer = [](std::size_t elements, float value) {
            DeviceBuffer buffer(std::max<std::size_t>(elements * 4, 256));
            fill_f32_kernel<<<fill_grid(elements), 256>>>(static_cast<float*>(buffer.p), elements,
                                                          value);
            CUDA_CHECK(cudaGetLastError());
            return buffer;
        };
        const std::size_t qk_elements = static_cast<std::size_t>(state_dim) * qk_heads * max_tokens;
        const std::size_t value_elements =
            static_cast<std::size_t>(state_dim) * value_heads * max_tokens;
        const std::size_t gate_elements = static_cast<std::size_t>(value_heads) * max_tokens;
        const std::size_t state_elements =
            static_cast<std::size_t>(state_dim) * state_dim * value_heads;
        DeviceBuffer q         = bf16_buffer(qk_elements);
        DeviceBuffer k         = bf16_buffer(qk_elements);
        DeviceBuffer v         = bf16_buffer(value_elements);
        DeviceBuffer out       = bf16_buffer(value_elements);
        DeviceBuffer g         = f32_buffer(gate_elements, -0.05f);
        DeviceBuffer beta      = f32_buffer(gate_elements, 0.5f);
        DeviceBuffer state_in  = f32_buffer(state_elements, 0.0f);
        DeviceBuffer state_out = f32_buffer(state_elements, 0.0f);
        WorkspaceArena workspace(
            std::max<std::size_t>(ops::gated_delta_net_workspace_capacity_bytes(
                                      qk_heads, value_heads, true, 1, max_tokens),
                                  256));
        const auto run = [&](std::int32_t tokens, cudaStream_t stream) {
            const Tensor qt(q.p, DType::BF16, {state_dim, qk_heads, tokens});
            const Tensor kt(k.p, DType::BF16, {state_dim, qk_heads, tokens});
            const Tensor vt(v.p, DType::BF16, {state_dim, value_heads, tokens});
            const Tensor gt(g.p, DType::FP32, {value_heads, tokens});
            const Tensor bt(beta.p, DType::FP32, {value_heads, tokens});
            const Tensor st_in(state_in.p, DType::FP32, {state_dim, state_dim, value_heads});
            Tensor st_out(state_out.p, DType::FP32, {state_dim, state_dim, value_heads});
            Tensor ot(out.p, DType::BF16, {state_dim, value_heads, tokens});
            ops::gated_delta_net(qt, kt, vt, gt, bt, scale, true, workspace, st_in, st_out, ot,
                                 stream);
        };
        const Family family{"gdn_two_stage/h" + std::to_string(value_heads),
                            {"on"},
                            widths,
                            run,
                            [&](std::int32_t tokens) {
                                return std::pair<const void*, std::size_t>{
                                    out.p,
                                    static_cast<std::size_t>(state_dim) * value_heads * tokens};
                            },
                            {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            // Past the widest measured prompt both algorithms scale linearly in its chunks.
            bands.back().last          = std::numeric_limits<std::int32_t>::max();
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(
                std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

void calibrate_attention(ops::DeviceRouteProfile& profile, Timer& timer,
                         const CalibrationOptions& options, KvCacheStorage storage,
                         const char* coding) {
    constexpr std::int32_t kLongestEnvelope = 4 * 262144;
    constexpr std::int32_t head_dim = 256;
    constexpr std::int32_t q_heads  = 24;
    constexpr std::int32_t kv_heads = 4;
    const std::vector<std::int32_t> envelopes = {8192, 65536, 262144};
    const std::int32_t max_window            = envelopes.back();
    const PagedKVStorageLayout layout        = paged_kv_storage_layout(storage, head_dim);
    const std::int32_t pages                 = max_window / kPagedKVPageSize;
    const auto plane_bytes = [&](std::int32_t leading, DType dtype) {
        return static_cast<std::size_t>(leading) * kv_heads * kPagedKVPageSize * pages * dtype_size(dtype);
    };
    DeviceBuffer k_pages(plane_bytes(layout.key.data_leading_extent, layout.key.data_dtype));
    DeviceBuffer v_pages(plane_bytes(layout.value.data_leading_extent, layout.value.data_dtype));
    DeviceBuffer k_scale(layout.key.has_scale()
                             ? plane_bytes(layout.key.scale_leading_extent, layout.key.scale_dtype)
                             : 256);
    DeviceBuffer v_scale(layout.value.has_scale()
                             ? plane_bytes(layout.value.scale_leading_extent, layout.value.scale_dtype)
                             : 256);
    fill_random(k_pages.p, k_pages.bytes, 0x1234u);
    fill_random(v_pages.p, v_pages.bytes, 0x5678u);
    // Scales of about 1/64 keep the scores and values in a realistic range.
    fill_u16_kernel<<<fill_grid(k_scale.bytes / 2), 256>>>(static_cast<std::uint16_t*>(k_scale.p),
                                                           k_scale.bytes / 2, 0x2400);
    fill_u16_kernel<<<fill_grid(v_scale.bytes / 2), 256>>>(static_cast<std::uint16_t*>(v_scale.p),
                                                           v_scale.bytes / 2, 0x2400);
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::int32_t> identity(static_cast<std::size_t>(pages));
    for (std::int32_t page = 0; page < pages; ++page) { identity[static_cast<std::size_t>(page)] = page; }
    DeviceBuffer block_table(identity.size() * sizeof(std::int32_t));
    block_table.copy_from_host(identity.data(), block_table.bytes);
    DeviceBuffer q = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * 8);
    DeviceBuffer out = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * 8);
    DeviceBuffer positions(8 * sizeof(std::int32_t));

    const ops::AttentionHeadGeometry geometry{head_dim, q_heads, kv_heads};
    const auto cache_view = [&](std::int32_t window) {
        const std::int32_t logical = window / kPagedKVPageSize;
        return PagedKVLayerView{
            .k_pages = Tensor(k_pages.p, layout.key.data_dtype,
                              {layout.key.data_leading_extent, kPagedKVPageSize, kv_heads, pages}),
            .v_pages = Tensor(v_pages.p, layout.value.data_dtype,
                              {layout.value.data_leading_extent, kPagedKVPageSize, kv_heads, pages}),
            .k_scale_pages = layout.key.has_scale()
                                 ? Tensor(k_scale.p, layout.key.scale_dtype,
                                          {layout.key.scale_leading_extent, kPagedKVPageSize, kv_heads,
                                           pages})
                                 : Tensor(),
            .v_scale_pages = layout.value.has_scale()
                                 ? Tensor(v_scale.p, layout.value.scale_dtype,
                                          {layout.value.scale_leading_extent, kPagedKVPageSize,
                                           kv_heads, pages})
                                 : Tensor(),
            .block_table  = Tensor(block_table.p, DType::I32, {logical}),
            .head_dim     = head_dim,
            .num_kv_heads = kv_heads,
            .storage      = storage,
        };
    };

    // FP16-accumulated PV per key tile: one 1024-token prompt chunk at 32K and a decode step at 131K.
    if (storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        constexpr std::int32_t chunk = 1024;
        // The fast-kernel comparison runs the chunk the planner aligns to the fast kernel's waves:
        // whole waves of 128-row CTAs are whole waves of the standard kernel's 64-row ones too.
        const std::int32_t wave_chunk =
            ops::causal_softmax_attention_prompt_aligned_chunk(geometry, storage, true, chunk);
        const std::int32_t max_chunk = std::max(chunk, wave_chunk);
        DeviceBuffer prompt_q   = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * max_chunk);
        DeviceBuffer prompt_out = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * max_chunk);
        DeviceBuffer prompt_positions(max_chunk * sizeof(std::int32_t));
        const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
            geometry, storage, {1, static_cast<std::uint32_t>(max_window)}, 1, 1, max_chunk);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
        const auto run = [&](std::int32_t tokens, std::int32_t depth, cudaStream_t stream) {
            std::vector<std::int32_t> host(static_cast<std::size_t>(tokens));
            for (std::int32_t token = 0; token < tokens; ++token) {
                host[static_cast<std::size_t>(token)] = depth - tokens + token;
            }
            CUDA_CHECK(cudaMemcpyAsync(prompt_positions.p, host.data(),
                                       host.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       stream));
            Tensor qt(prompt_q.p, DType::BF16, {head_dim, q_heads, tokens});
            Tensor ot(prompt_out.p, DType::BF16, {head_dim, q_heads, tokens});
            Tensor pt(prompt_positions.p, DType::I32, {tokens});
            auto scope = workspace.scope();
            ops::causal_softmax_attention_cached(qt, pt, geometry, 0.0625f, cache_view(max_window),
                                                 {1, static_cast<std::uint32_t>(max_window)},
                                                 workspace, ot, stream);
        };
        const Family family{"attn_pv_f16",
                            {"on"},
                            {1},
                            [&](std::int32_t, cudaStream_t stream) {
                                run(chunk, 32768, stream);
                                run(1, 131072, stream);
                            },
                            [&](std::int32_t) {
                                return std::pair<const void*, std::size_t>{
                                    prompt_out.p, static_cast<std::size_t>(head_dim) * q_heads};
                            },
                            {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
        // The standard prompt kernel with each KV head's query heads packed into its tiles, as the
        // PV choice above left it: one 1024-token chunk at 32K and one at 131K of context.
        const Family pack{"attn_pack_gqa",
                          {"on"},
                          {1},
                          [&](std::int32_t, cudaStream_t stream) {
                              run(chunk, 32768, stream);
                              run(chunk, 131072, stream);
                          },
                          [&](std::int32_t) {
                              return std::pair<const void*, std::size_t>{
                                  prompt_out.p, static_cast<std::size_t>(head_dim) * q_heads};
                          },
                          {}};
        auto pack_bands = sweep(pack, timer, options);
        if (!pack_bands.empty()) {
            profile.routes[pack.key] = std::move(pack_bands);
            ops::install_device_route_profile(
                std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
        // The fast prompt kernel against the standard one (as the PV and packing choices above
        // left it): one wave-aligned chunk at 32K and one at 131K of context.
        const Family fast{"attn_prompt_fast",
                          {"on"},
                          {1},
                          [&](std::int32_t, cudaStream_t stream) {
                              run(wave_chunk, 32768, stream);
                              run(wave_chunk, 131072, stream);
                          },
                          [&](std::int32_t) {
                              return std::pair<const void*, std::size_t>{
                                  prompt_out.p, static_cast<std::size_t>(head_dim) * q_heads};
                          },
                          {}};
        auto fast_bands = sweep(fast, timer, options);
        if (!fast_bands.empty()) {
            profile.routes[fast.key] = std::move(fast_bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }

    for (std::int32_t width = 1; width <= 8; ++width) {
        const std::int32_t row_tiles = (width * (q_heads / kv_heads) + 15) / 16;
        std::vector<std::string> tiers;
        if (row_tiles == 1) {
            tiers = {"2x4x32",  "4x2x32",  "8x2x32",   "16x1x32", "2x2x32q",
                     "4x2x32q", "4x2x32e", "8x2x32e",  "16x1x32e", "4x2x32qe"};
        } else if (row_tiles == 2) {
            tiers = {"4x2x32",  "8x2x32",  "16x1x32",  "32x1x32",  "4x2x32q",  "8x2x32q",
                     "4x2x32e", "8x2x32e", "16x1x32e", "4x2x32qe", "8x2x32qe"};
        } else {
            tiers = {"6x2x32",  "12x1x32",  "12x1x64d", "24x1x32",  "6x2x32q",
                     "12x1x32q", "6x2x32e", "12x1x32e", "6x2x32qe", "12x1x32qe"};
        }
        const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
            geometry, storage, {1, static_cast<std::uint32_t>(max_window)}, 1, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
        const auto attend = [&](std::int32_t envelope, std::int32_t depth, cudaStream_t stream) {
            std::vector<std::int32_t> host(static_cast<std::size_t>(width));
            for (std::int32_t token = 0; token < width; ++token) {
                host[static_cast<std::size_t>(token)] = depth - width + token;
            }
            CUDA_CHECK(cudaMemcpyAsync(positions.p, host.data(), host.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            Tensor qt(q.p, DType::BF16, {head_dim, q_heads, width});
            Tensor ot(out.p, DType::BF16, {head_dim, q_heads, width});
            Tensor pt(positions.p, DType::I32, {width});
            const PagedKVLayerView cache = cache_view(envelope);
            auto scope                   = workspace.scope();
            ops::causal_softmax_attention_cached(
                qt, pt, geometry, 0.0625f, cache,
                {1, static_cast<std::uint32_t>(envelope)}, workspace, ot, stream);
        };
        const Family family{
            "attn_i8_small/h24/" + std::string(coding) + "/w" + std::to_string(width), tiers, envelopes,
            [&](std::int32_t envelope, cudaStream_t stream) {
                attend(envelope, envelope, stream);
                attend(envelope, std::max(width + 1, envelope / 16), stream);
            },
            [&](std::int32_t) {
                return std::pair<const void*, std::size_t>{
                    out.p, static_cast<std::size_t>(head_dim) * q_heads * width};
            },
            {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            // Past the native window (YaRN, up to four times it) the kernel is as memory-bound as at
            // the widest measured envelope, so that envelope's tier serves those windows too.
            if (bands.back().last == max_window) { bands.back().last = kLongestEnvelope; }
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

} // namespace

ops::DeviceRouteProfile calibrate_device_routes(const CalibrationOptions& options) {
    const auto started = std::chrono::steady_clock::now();
    ops::DeviceRouteProfile profile;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    profile.multiprocessors = properties.multiProcessorCount;
    profile.origin          = "calibration";
    Timer timer(options);
    if (options.ternary) { calibrate_ternary(profile, timer, options); }
    if (options.groupwise) { calibrate_groupwise(profile, timer, options); }
    if (options.linear_attention) { calibrate_linear_attention(profile, timer, options); }
    if (options.attention) {
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                            "rk8v4");
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedLloyd4KeyInt4Value,
                            "rk4v4");
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedInt4KeyInt4ValueE8,
                            "rk4v4-e8");
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedE8RootKeyInt4Value,
                            "rk2v4-e8");
        calibrate_attention(profile, timer, options, KvCacheStorage::Int8Group64, "int8");
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    char line[160];
    std::snprintf(line, sizeof(line), "calibration: %zu routed keys in %.1f s",
                  profile.routes.size(), seconds);
    report(options, line);
    return profile;
}

} // namespace ninfer::calibration
