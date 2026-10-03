#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr int kHidden     = 5120;
constexpr int kParentRows = 14336;
constexpr float kEps      = 1.0e-6F;

template <class T>
int compare_bytes(const std::string& label, const T* reference, const T* fused, std::size_t count) {
    const auto expected = from_device<std::uint8_t>(reference, count * sizeof(T));
    const auto actual   = from_device<std::uint8_t>(fused, count * sizeof(T));
    if (actual == expected) { return 0; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << label << ": byte mismatch at " << i << " (reference "
                      << unsigned(expected[i]) << ", fused " << unsigned(actual[i]) << ")\n";
            break;
        }
    }
    return 1;
}

int run_case(const Weight& weight, int tokens, bool measure) {
    const auto values            = make_bf16_activation(kHidden, tokens, 871U + tokens);
    auto gains                   = make_bf16_activation(kHidden, 1, 119U);
    const auto value_bits        = bf16_bits(values);
    const auto gain_bits         = bf16_bits(gains);
    DeviceBuffer residual_buffer = to_device(value_bits);
    DeviceBuffer gain_buffer     = to_device(gain_bits);
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kHidden) * tokens * 2);
    Tensor residual(residual_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kHidden, tokens});
    ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);

    const std::size_t capacity = ops::attn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kParentRows, kHidden, ops::LinearPolicy::AllowA4, tokens, tokens);
    // The A16 route (T<4) uses no workspace; the one-byte floor keeps the arena
    // constructible while still rejecting any genuine plane allocation.
    const std::size_t floored = capacity == 0 ? 1 : capacity;
    DeviceArena reference_workspace(floored);
    DeviceArena fused_workspace(floored);
    int failures = 0;
    {
        auto reference_scope = reference_workspace.scope();
        auto fused_scope     = fused_workspace.scope();
        // Below T=4 the route carries BF16 activations (no quantizer); at T>=4 the W4A4
        // route quantizes, tiled at/above the TMA cutoff and row-major below it.
        if (tokens >= 4) {
            const bool tiled = ops::detail::nvfp4_attn_input_tma_route(tokens);
            const auto reference_planes =
                ops::detail::allocate_nvfp4_w4a4_workspace(reference_workspace, tokens, kHidden);
            const auto fused_planes =
                ops::detail::allocate_nvfp4_w4a4_workspace(fused_workspace, tokens, kHidden);
            ops::detail::launch_nvfp4_w4a4_quantize(hidden, weight, reference_planes,
                                                    tiled ? ops::detail::Nvfp4ScaleLayout::Tiled
                                                          : ops::detail::Nvfp4ScaleLayout::RowMajor,
                                                    nullptr);
            if (tiled) {
                ops::detail::launch_nvfp4_attn_input_fused_rmsnorm_quantize(
                    residual, gain, kEps, weight.input_scale_divisor, fused_planes, nullptr);
            } else {
                ops::detail::launch_nvfp4_attn_input_fused_rmsnorm_quantize_rowmajor(
                    residual, gain, kEps, weight.input_scale_divisor, fused_planes, nullptr);
            }
            cuda_synchronize();
            const std::string suffix = " T=" + std::to_string(tokens);
            failures += compare_bytes("codes" + suffix, reference_planes.codes, fused_planes.codes,
                                      static_cast<std::size_t>(tokens) * kHidden / 2);
            // Row-major planes carry no padding; only the written prefix is defined.
            const std::size_t scale_count = tiled
                                                ? reference_planes.scale_bytes
                                                : static_cast<std::size_t>(tokens) * (kHidden / 16);
            failures += compare_bytes("scales" + suffix, reference_planes.scales,
                                      fused_planes.scales, scale_count);
        }
    }

    constexpr int kQRows  = 6144;
    constexpr int kKvRows = 1024;
    DeviceBuffer reference_q(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer reference_gate(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer reference_k(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer reference_v(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer fused_q(reference_q.bytes), fused_gate(reference_gate.bytes);
    DeviceBuffer fused_k(reference_k.bytes), fused_v(reference_v.bytes);
    Tensor rq(reference_q.p, DType::BF16, {kQRows, tokens});
    Tensor rg(reference_gate.p, DType::BF16, {kQRows, tokens});
    Tensor rk(reference_k.p, DType::BF16, {kKvRows, tokens});
    Tensor rv(reference_v.p, DType::BF16, {kKvRows, tokens});
    Tensor fq(fused_q.p, DType::BF16, {kQRows, tokens});
    Tensor fg(fused_gate.p, DType::BF16, {kQRows, tokens});
    Tensor fk(fused_k.p, DType::BF16, {kKvRows, tokens});
    Tensor fv(fused_v.p, DType::BF16, {kKvRows, tokens});
    ops::attn_input_proj(hidden, weight, rq, rg, rk, rv, ops::LinearPolicy::AllowA4,
                         reference_workspace, nullptr);
    ops::attn_input_proj_fused_rmsnorm_nvfp4(residual, gain, kEps, weight, fq, fg, fk, fv,
                                             ops::LinearPolicy::AllowA4, fused_workspace, nullptr);
    cuda_synchronize();
    const std::string suffix = " T=" + std::to_string(tokens);
    failures += compare_bytes("q" + suffix, static_cast<const std::uint8_t*>(reference_q.p),
                              static_cast<const std::uint8_t*>(fused_q.p), reference_q.bytes);
    failures += compare_bytes("gate" + suffix, static_cast<const std::uint8_t*>(reference_gate.p),
                              static_cast<const std::uint8_t*>(fused_gate.p), reference_gate.bytes);
    failures += compare_bytes("k" + suffix, static_cast<const std::uint8_t*>(reference_k.p),
                              static_cast<const std::uint8_t*>(fused_k.p), reference_k.bytes);
    failures += compare_bytes("v" + suffix, static_cast<const std::uint8_t*>(reference_v.p),
                              static_cast<const std::uint8_t*>(fused_v.p), reference_v.bytes);
    if ((tokens == 1 || tokens == 1500) && failures == 0) {
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::attn_input_proj_fused_rmsnorm_nvfp4(residual, gain, kEps, weight, fq, fg, fk, fv,
                                                     ops::LinearPolicy::AllowA4, fused_workspace,
                                                     device.stream);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        failures += compare_bytes("graph q" + suffix,
                                  static_cast<const std::uint8_t*>(reference_q.p),
                                  static_cast<const std::uint8_t*>(fused_q.p), reference_q.bytes);
        failures += compare_bytes("graph gate" + suffix,
                                  static_cast<const std::uint8_t*>(reference_gate.p),
                                  static_cast<const std::uint8_t*>(fused_gate.p),
                                  reference_gate.bytes);
        failures += compare_bytes("graph k" + suffix,
                                  static_cast<const std::uint8_t*>(reference_k.p),
                                  static_cast<const std::uint8_t*>(fused_k.p), reference_k.bytes);
        failures += compare_bytes("graph v" + suffix,
                                  static_cast<const std::uint8_t*>(reference_v.p),
                                  static_cast<const std::uint8_t*>(fused_v.p), reference_v.bytes);
    }
    if (reference_workspace.used() != 0 || fused_workspace.used() != 0) {
        std::cerr << "attention input workspace scope was not released\n";
        ++failures;
    }
    if (measure && failures == 0) {
        const auto reference_stage = [&] {
            ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);
            ops::attn_input_proj(hidden, weight, rq, rg, rk, rv, ops::LinearPolicy::AllowA4,
                                 reference_workspace, nullptr);
        };
        const auto fused_stage = [&] {
            ops::attn_input_proj_fused_rmsnorm_nvfp4(residual, gain, kEps, weight, fq, fg, fk, fv,
                                                     ops::LinearPolicy::AllowA4, fused_workspace,
                                                     nullptr);
        };
        const auto median_wall_us = [](const auto& stage) {
            for (int i = 0; i < 5; ++i) {
                stage();
                cuda_synchronize();
            }
            std::vector<double> samples;
            for (int i = 0; i < 31; ++i) {
                const auto start = std::chrono::steady_clock::now();
                stage();
                cuda_synchronize();
                const auto stop = std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
            }
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2];
        };
        const double reference_us = median_wall_us(reference_stage);
        const double fused_us     = median_wall_us(fused_stage);
        std::cout << "ATTN_INPUT_STAGE T=" << tokens << " reference_us=" << reference_us
                  << " fused_us=" << fused_us << " delta_us=" << reference_us - fused_us << '\n';
    }
    return failures;
}

// T==2,3 deliberately stay unfused (measured fused-norm+SIMT regression): the predicate
// must exclude them, the fused entry must throw rather than misroute, and the unfused
// path must run clean.
int check_unfused_fallback(const Weight& weight, int tokens) {
    if (ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                          tokens)) {
        std::cerr << "T=" << tokens << " must stay on the unfused path\n";
        return 1;
    }
    const auto values            = make_bf16_activation(kHidden, tokens, 871U + tokens);
    const auto gains             = make_bf16_activation(kHidden, 1, 119U);
    DeviceBuffer residual_buffer = to_device(bf16_bits(values));
    DeviceBuffer gain_buffer     = to_device(bf16_bits(gains));
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kHidden) * tokens * 2);
    Tensor residual(residual_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kHidden, tokens});
    constexpr int kQRows  = 6144;
    constexpr int kKvRows = 1024;
    DeviceBuffer fused_q(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer fused_gate(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer fused_k(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer fused_v(static_cast<std::size_t>(kKvRows) * tokens * 2);
    Tensor fq(fused_q.p, DType::BF16, {kQRows, tokens});
    Tensor fg(fused_gate.p, DType::BF16, {kQRows, tokens});
    Tensor fk(fused_k.p, DType::BF16, {kKvRows, tokens});
    Tensor fv(fused_v.p, DType::BF16, {kKvRows, tokens});
    DeviceArena workspace(1);
    try {
        ops::attn_input_proj_fused_rmsnorm_nvfp4(residual, gain, kEps, weight, fq, fg, fk, fv,
                                                 ops::LinearPolicy::AllowA4, workspace, nullptr);
    } catch (const std::invalid_argument&) {
        ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);
        ops::attn_input_proj(hidden, weight, fq, fg, fk, fv, ops::LinearPolicy::AllowA4, workspace,
                             nullptr);
        cuda_synchronize();
        for (const double v : from_device_bf16(fused_q.p, fused_q.bytes / 2)) {
            if (!std::isfinite(v)) {
                std::cerr << "T=" << tokens << " unfused fallback produced non-finite output\n";
                return 1;
            }
        }
        return 0;
    }
    std::cerr << "T=" << tokens << " fused entry must throw\n";
    return 1;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        quantized_weight::PatternedWeightOptions options;
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::NVFP4, kParentRows,
                                                                          kHidden, 331U, options));
        const Weight weight = parent.view();
        // Eligibility reads qtype/layout/shape only (never the payload), so
        // field-override copies are safe negatives here.
        Weight wrong_qtype  = weight;
        Weight wrong_layout = weight;
        Weight wrong_n      = weight;
        Weight wrong_k      = weight;
        wrong_qtype.qtype   = QType::BF16;
        wrong_layout.layout = QuantLayout::RowSplit;
        wrong_n.n           = kParentRows - 1;
        wrong_k.k           = kHidden - 1;
        if (ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::A16Only,
                                                              1024) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                              0) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::A16Only,
                                                              1) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(wrong_qtype,
                                                              ops::LinearPolicy::AllowA4, 1024) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(wrong_layout,
                                                              ops::LinearPolicy::AllowA4, 1024) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(wrong_n, ops::LinearPolicy::AllowA4,
                                                              1024) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(wrong_k, ops::LinearPolicy::AllowA4,
                                                              1024) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(wrong_n, ops::LinearPolicy::AllowA4,
                                                              1) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                              2) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                              3) ||
            !ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                               1024) ||
            !ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                               1) ||
            !ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                               4)) {
            std::cerr << "fused route eligibility mismatch\n";
            return 1;
        }
        int failures       = 0;
        const bool measure = std::getenv("NINFER_MEASURE_FUSED_STAGE") != nullptr;
        for (const int tokens : {2, 3}) { failures += check_unfused_fallback(weight, tokens); }
        for (const int tokens : {1,   4,   5,   32,  64,  65,   96,   97,   128,  129,  192,
                                 193, 384, 385, 512, 513, 1000, 1023, 1024, 1500, 2048, 4096}) {
            failures +=
                run_case_allowing_arch_skip("fused rmsnorm attn_input_proj T=" +
                                                std::to_string(tokens),
                                            [&] { return run_case(weight, tokens, measure); });
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fused attention input test: " << error.what() << '\n';
        return 1;
    }
}
