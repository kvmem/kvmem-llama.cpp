#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/op_tester.h"
#include "core/device.h"
#include "core/decode_graph.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kBlock = 1024;

// The transform is orthonormal, so FP32 butterflies add ~1e-6 relative error and the BF16 output
// rounding dominates: the same reduction criterion the norms use.
constexpr ReductionCriterion hadamard_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ kBf16GrossRelativeFloor};
}

// Naive definition: out[i] = 2^-5 * sum_j (-1)^popcount(i&j) * (signs applied before or after).
std::vector<double> hadamard_oracle(const std::vector<float>& input,
                                    const std::vector<float>& signs, std::int32_t k,
                                    std::int64_t columns, bool inverse) {
    std::vector<double> output(input.size());
    const int blocks = k / kBlock;
    for (std::int64_t column = 0; column < columns; ++column) {
        for (int block = 0; block < blocks; ++block) {
            const std::size_t base =
                static_cast<std::size_t>(column) * k + static_cast<std::size_t>(block) * kBlock;
            for (int i = 0; i < kBlock; ++i) {
                double sum = 0.0;
                for (int j = 0; j < kBlock; ++j) {
                    const double sign_before =
                        inverse ? 1.0 : static_cast<double>(signs[block * kBlock + j]);
                    const double value = static_cast<double>(input[base + j]) * sign_before;
                    sum += (__builtin_popcount(static_cast<unsigned>(i & j)) & 1) ? -value : value;
                }
                const double sign_after =
                    inverse ? static_cast<double>(signs[block * kBlock + i]) : 1.0;
                output[base + i] = sum * (1.0 / 32.0) * sign_after;
            }
        }
    }
    return output;
}

// Butterfly evaluation in double for large cases; validated against the naive oracle below.
std::vector<double> hadamard_butterfly_oracle(const std::vector<float>& input,
                                              const std::vector<float>& signs, std::int32_t k,
                                              std::int64_t columns, bool inverse) {
    std::vector<double> output(input.size());
    const int blocks = k / kBlock;
    std::vector<double> work(kBlock);
    for (std::int64_t column = 0; column < columns; ++column) {
        for (int block = 0; block < blocks; ++block) {
            const std::size_t base =
                static_cast<std::size_t>(column) * k + static_cast<std::size_t>(block) * kBlock;
            for (int j = 0; j < kBlock; ++j) {
                work[j] = static_cast<double>(input[base + j]) *
                          (inverse ? 1.0 : static_cast<double>(signs[block * kBlock + j]));
            }
            for (int stride = 1; stride < kBlock; stride <<= 1) {
                for (int i = 0; i < kBlock; ++i) {
                    if ((i & stride) == 0) {
                        const double low  = work[i];
                        const double high = work[i + stride];
                        work[i]           = low + high;
                        work[i + stride]  = low - high;
                    }
                }
            }
            for (int i = 0; i < kBlock; ++i) {
                output[base + i] = work[i] * (1.0 / 32.0) *
                                   (inverse ? static_cast<double>(signs[block * kBlock + i]) : 1.0);
            }
        }
    }
    return output;
}

int verify_oracles_agree() {
    std::vector<float> input(static_cast<std::size_t>(2 * kBlock)), signs(kBlock);
    fill_uniform(input, 77U, -4.0F, 4.0F);
    fill_uniform(signs, 78U, -1.0F, 1.0F);
    for (auto& sign : signs) sign = sign < 0.0F ? -1.0F : 1.0F;
    round_to_bf16(input);
    int failures = 0;
    for (bool inverse : {false, true}) {
        const auto naive     = hadamard_oracle(input, signs, kBlock, 2, inverse);
        const auto butterfly = hadamard_butterfly_oracle(input, signs, kBlock, 2, inverse);
        failures += verify_pointwise(inverse ? "oracle agreement inverse" : "oracle agreement",
                                     butterfly, naive, {1.0e-9, 1.0e-12});
    }
    return failures;
}

int run_case(const std::string& label, std::int32_t k, std::int64_t columns, bool inverse,
             bool in_place, bool replay, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(k) * static_cast<std::size_t>(columns);
    std::vector<float> input(count), signs(static_cast<std::size_t>(k));
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(signs, seed + 1U, -1.0F, 1.0F);
    for (auto& sign : signs) sign = sign < 0.0F ? -1.0F : 1.0F;
    signs[0] = -1.0F; // both signs are always present
    signs[1] = 1.0F;
    round_to_bf16(input);

    std::vector<std::uint16_t> input_words(count), sign_words(static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < count; ++i) input_words[i] = f32_to_bf16(input[i]);
    for (std::size_t i = 0; i < sign_words.size(); ++i) sign_words[i] = f32_to_bf16(signs[i]);
    GuardedDeviceBuffer device_input(count * sizeof(std::uint16_t));
    device_input.copy_from_host(input_words.data(), count * sizeof(std::uint16_t));
    DeviceBuffer device_signs = to_device(sign_words);
    GuardedDeviceBuffer output(count * sizeof(std::uint16_t));
    output.fill(0xff);

    Tensor x(device_input.data(), DType::BF16, {k, static_cast<std::int32_t>(columns)});
    Tensor sign_tensor(device_signs.p, DType::BF16, {k});
    Tensor out(in_place ? device_input.data() : output.data(), DType::BF16,
               {k, static_cast<std::int32_t>(columns)});

    DeviceContext device;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] {
        ops::hadamard_transform(x, sign_tensor, inverse, out, device.stream);
    };
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : input) value = -value;
            for (std::size_t i = 0; i < count; ++i) input_words[i] = f32_to_bf16(input[i]);
        }
        device_input.copy_from_host(input_words.data(), count * sizeof(std::uint16_t));
        if (!in_place) output.fill(0xff);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
        }
        if (replay) {
            graph.launch(device.stream);
        } else {
            launch();
        }
        cuda_synchronize(device.stream);
        const auto reference = hadamard_butterfly_oracle(input, signs, k, columns, inverse);
        const void* result   = in_place ? device_input.data() : output.data();
        failures += verify_reduction(label, from_device_bf16(result, count), reference,
                                     hadamard_bf16_criterion());
        failures += device_input.verify_guards(label + " input guards");
        failures += output.verify_guards(label + " output guards");
        if (!in_place) {
            failures +=
                verify_exact((label + " preserves input").c_str(),
                             from_device<std::uint16_t>(device_input.data(), count), input_words);
        }
        failures +=
            verify_exact((label + " preserves signs").c_str(),
                         from_device<std::uint16_t>(device_signs, sign_words.size()), sign_words);
    }
    return failures;
}

// forward then inverse must return the input up to two BF16 roundings.
int run_round_trip(std::int32_t k, std::int64_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(k) * static_cast<std::size_t>(columns);
    std::vector<float> input(count), signs(static_cast<std::size_t>(k));
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(signs, seed + 1U, -1.0F, 1.0F);
    for (auto& sign : signs) sign = sign < 0.0F ? -1.0F : 1.0F;
    round_to_bf16(input);
    std::vector<std::uint16_t> sign_words(static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < sign_words.size(); ++i) sign_words[i] = f32_to_bf16(signs[i]);
    DeviceBuffer device_input = to_device_bf16(input);
    DeviceBuffer device_signs = to_device(sign_words);
    DeviceBuffer rotated(count * sizeof(std::uint16_t));
    DeviceBuffer restored(count * sizeof(std::uint16_t));
    Tensor x(device_input.p, DType::BF16, {k, static_cast<std::int32_t>(columns)});
    Tensor sign_tensor(device_signs.p, DType::BF16, {k});
    Tensor y(rotated.p, DType::BF16, {k, static_cast<std::int32_t>(columns)});
    Tensor z(restored.p, DType::BF16, {k, static_cast<std::int32_t>(columns)});
    DeviceContext device;
    ops::hadamard_transform(x, sign_tensor, false, y, device.stream);
    ops::hadamard_transform(y, sign_tensor, true, z, device.stream);
    cuda_synchronize(device.stream);
    std::vector<double> reference(count);
    for (std::size_t i = 0; i < count; ++i) reference[i] = static_cast<double>(input[i]);
    const std::string label =
        "hadamard round trip K=" + std::to_string(k) + " T=" + std::to_string(columns);
    return verify_reduction(label, from_device_bf16(restored, count), reference,
                            {/*relative_l2*/ 3.7e-3, /*gross_absolute*/ 1.0e-5,
                             /*gross_relative_to_max_reference*/ 2.0 * kBf16GrossRelativeFloor});
}

// silu(gate) * up rounded to BF16, then the forward transform: the composition's definition.
int run_silu_mul_case(std::int32_t width, std::int64_t columns, bool replay, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(columns);
    std::vector<float> plane(2 * count), signs(static_cast<std::size_t>(width));
    fill_uniform(plane, seed, -4.0F, 4.0F);
    fill_uniform(signs, seed + 1U, -1.0F, 1.0F);
    for (auto& sign : signs) sign = sign < 0.0F ? -1.0F : 1.0F;
    signs[0] = -1.0F;
    signs[1] = 1.0F;
    round_to_bf16(plane);
    std::vector<float> activation(count);
    for (std::int64_t column = 0; column < columns; ++column) {
        for (std::int32_t i = 0; i < width; ++i) {
            const float gate = plane[static_cast<std::size_t>(column) * 2 * width + i];
            const float up   = plane[static_cast<std::size_t>(column) * 2 * width + width + i];
            const float silu = gate / (1.0F + std::exp(-gate));
            activation[static_cast<std::size_t>(column) * width + i] =
                bf16_to_f32(f32_to_bf16(silu * up));
        }
    }
    std::vector<std::uint16_t> sign_words(static_cast<std::size_t>(width));
    for (std::size_t i = 0; i < sign_words.size(); ++i) sign_words[i] = f32_to_bf16(signs[i]);
    DeviceBuffer device_plane = to_device_bf16(plane);
    DeviceBuffer device_signs = to_device(sign_words);
    GuardedDeviceBuffer output(count * sizeof(std::uint16_t));
    output.fill(0xff);
    Tensor plane_tensor(device_plane.p, DType::BF16,
                        {2 * width, static_cast<std::int32_t>(columns)});
    Tensor sign_tensor(device_signs.p, DType::BF16, {width});
    Tensor out(output.data(), DType::BF16, {width, static_cast<std::int32_t>(columns)});
    DeviceContext device;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] {
        ops::silu_mul_hadamard(plane_tensor, sign_tensor, out, device.stream);
    };
    if (replay) {
        definition.capture(device.stream, launch);
        graph.instantiate(definition);
        graph.launch(device.stream);
    } else {
        launch();
    }
    cuda_synchronize(device.stream);
    const auto reference = hadamard_butterfly_oracle(activation, signs, width, columns, false);
    const std::string label =
        "silu_mul_hadamard I=" + std::to_string(width) + " T=" + std::to_string(columns);
    int failures = verify_reduction(label, from_device_bf16(output.data(), count), reference,
                                    hadamard_bf16_criterion());
    failures += output.verify_guards(label + " output guards");
    failures +=
        verify_exact((label + " preserves signs").c_str(),
                     from_device<std::uint16_t>(device_signs, sign_words.size()), sign_words);
    return failures;
}

std::vector<float> random_signs(std::int32_t width, std::uint32_t seed) {
    std::vector<float> signs(static_cast<std::size_t>(width));
    fill_uniform(signs, seed, -1.0F, 1.0F);
    for (auto& sign : signs) sign = sign < 0.0F ? -1.0F : 1.0F;
    signs[0] = -1.0F;
    signs[1] = 1.0F;
    return signs;
}

std::vector<std::uint16_t> device_words(const DeviceBuffer& buffer, std::size_t count) {
    return from_device<std::uint16_t>(buffer, count);
}

// rmsnorm_hadamard must equal rmsnorm followed by hadamard_transform word for word.
int run_rmsnorm_hadamard_case(std::int32_t width, std::int64_t columns, bool unit_offset,
                              bool replay, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(columns);
    std::vector<float> x(count), weight(static_cast<std::size_t>(width));
    fill_uniform(x, seed, -3.0F, 3.0F);
    fill_uniform(weight, seed + 1U, -0.5F, 0.5F);
    round_to_bf16(x);
    round_to_bf16(weight);
    const auto signs           = random_signs(width, seed + 2U);
    DeviceBuffer device_x      = to_device_bf16(x);
    DeviceBuffer device_weight = to_device_bf16(weight);
    DeviceBuffer device_signs  = to_device_bf16(signs);
    DeviceBuffer normalized(count * 2), composed(count * 2);
    GuardedDeviceBuffer fused(count * 2);
    fused.fill(0xff);
    const std::int32_t t = static_cast<std::int32_t>(columns);
    Tensor tx(device_x.p, DType::BF16, {width, t}), tw(device_weight.p, DType::BF16, {width}),
        ts(device_signs.p, DType::BF16, {width}), tn(normalized.p, DType::BF16, {width, t}),
        tc(composed.p, DType::BF16, {width, t}), tf(fused.data(), DType::BF16, {width, t});
    DeviceContext device;
    ops::rmsnorm(tx, tw, 1.0e-6F, unit_offset, tn, device.stream);
    ops::hadamard_transform(tn, ts, false, tc, device.stream);
    const auto launch = [&] {
        ops::rmsnorm_hadamard(tx, tw, 1.0e-6F, unit_offset, ts, tf, device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) {
        definition.capture(device.stream, launch);
        graph.instantiate(definition);
        graph.launch(device.stream);
    } else {
        launch();
    }
    cuda_synchronize(device.stream);
    const std::string label = std::string("rmsnorm_hadamard K=") + std::to_string(width) +
                              " T=" + std::to_string(columns) + (unit_offset ? " offset" : "");
    int failures            = verify_exact((label + " matches the composition").c_str(),
                                           from_device<std::uint16_t>(fused.data(), count),
                                           device_words(composed, count));
    failures += fused.verify_guards(label + " guards");
    return failures;
}

// gated_rmsnorm_hadamard over [128, heads, T] must equal gated_rmsnorm then the transform of each
// [128 * heads] column word for word.
int run_gated_rmsnorm_hadamard_case(std::int32_t heads, std::int64_t columns, std::uint32_t seed) {
    constexpr std::int32_t kD = 128;
    const std::int32_t width  = kD * heads;
    const std::size_t count   = static_cast<std::size_t>(width) * static_cast<std::size_t>(columns);
    std::vector<float> x(count), z(count), weight(kD);
    fill_uniform(x, seed, -3.0F, 3.0F);
    fill_uniform(z, seed + 1U, -4.0F, 4.0F);
    fill_uniform(weight, seed + 2U, 0.5F, 1.5F);
    round_to_bf16(x);
    round_to_bf16(z);
    round_to_bf16(weight);
    const auto signs           = random_signs(width, seed + 3U);
    DeviceBuffer device_x      = to_device_bf16(x);
    DeviceBuffer device_z      = to_device_bf16(z);
    DeviceBuffer device_weight = to_device_bf16(weight);
    DeviceBuffer device_signs  = to_device_bf16(signs);
    DeviceBuffer normalized(count * 2), composed(count * 2);
    GuardedDeviceBuffer fused(count * 2);
    fused.fill(0xff);
    const std::int32_t t = static_cast<std::int32_t>(columns);
    Tensor tx(device_x.p, DType::BF16, {kD, heads, t}), tz(device_z.p, DType::BF16, {kD, heads, t}),
        tw(device_weight.p, DType::BF16, {kD}), ts(device_signs.p, DType::BF16, {width}),
        tn(normalized.p, DType::BF16, {kD, heads, t}), tc(composed.p, DType::BF16, {width, t}),
        tf(fused.data(), DType::BF16, {width, t});
    DeviceContext device;
    ops::gated_rmsnorm(tx, tw, tz, 1.0e-6F, tn, device.stream);
    Tensor tn_columns = tn.view({width, t});
    ops::hadamard_transform(tn_columns, ts, false, tc, device.stream);
    ops::gated_rmsnorm_hadamard(tx, tw, tz, 1.0e-6F, ts, tf, device.stream);
    cuda_synchronize(device.stream);
    const std::string label =
        "gated_rmsnorm_hadamard heads=" + std::to_string(heads) + " T=" + std::to_string(columns);
    int failures = verify_exact((label + " matches the composition").c_str(),
                                from_device<std::uint16_t>(fused.data(), count),
                                device_words(composed, count));
    failures += fused.verify_guards(label + " guards");
    return failures;
}

// sigmoid_mul_hadamard must equal sigmoid_mul then the transform, in place or not.
int run_sigmoid_mul_hadamard_case(std::int32_t width, std::int64_t columns, bool in_place,
                                  std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(columns);
    std::vector<float> gate(count), x(count);
    fill_uniform(gate, seed, -6.0F, 6.0F);
    fill_uniform(x, seed + 1U, -3.0F, 3.0F);
    round_to_bf16(gate);
    round_to_bf16(x);
    const auto signs          = random_signs(width, seed + 2U);
    DeviceBuffer device_gate  = to_device_bf16(gate);
    DeviceBuffer device_x     = to_device_bf16(x);
    DeviceBuffer gated        = to_device_bf16(x);
    DeviceBuffer device_signs = to_device_bf16(signs);
    DeviceBuffer composed(count * 2);
    GuardedDeviceBuffer fused(count * 2);
    fused.fill(0xff);
    const std::int32_t t = static_cast<std::int32_t>(columns);
    Tensor tg(device_gate.p, DType::BF16, {width, t}), tx(device_x.p, DType::BF16, {width, t}),
        ts(device_signs.p, DType::BF16, {width}), tm(gated.p, DType::BF16, {width, t}),
        tc(composed.p, DType::BF16, {width, t}),
        tf(in_place ? device_x.p : fused.data(), DType::BF16, {width, t});
    DeviceContext device;
    ops::sigmoid_mul(tg, tm, device.stream);
    ops::hadamard_transform(tm, ts, false, tc, device.stream);
    ops::sigmoid_mul_hadamard(tg, tx, ts, tf, device.stream);
    cuda_synchronize(device.stream);
    const std::string label = "sigmoid_mul_hadamard K=" + std::to_string(width) +
                              " T=" + std::to_string(columns) + (in_place ? " in place" : "");
    const void* result      = in_place ? device_x.p : fused.data();
    int failures =
        verify_exact((label + " matches the composition").c_str(),
                     from_device<std::uint16_t>(result, count), device_words(composed, count));
    if (!in_place) failures += fused.verify_guards(label + " guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures                 = verify_oracles_agree();
    const std::int32_t widths[]  = {1024, 5120, 6144, 17408};
    const std::int64_t columns[] = {1, 2, 7, 8, 15, 16, 31, 32, 64, 128, 1024};
    std::uint32_t seed           = 3100U;
    for (const std::int32_t k : widths) {
        for (const std::int64_t t : columns) {
            const bool replay       = t == 1 || t == 8 || t == 16 || t == 32 || t == 128;
            const bool in_place     = t == 2 || t == 15 || t == 64;
            const std::string label = "hadamard K=" + std::to_string(k) + " T=" + std::to_string(t);
            failures += run_case(label + " forward", k, t, false, in_place, replay, seed++);
            if (t == 1 || t == 7 || t == 16 || t == 1024) {
                failures += run_case(label + " inverse", k, t, true, false, false, seed++);
            }
        }
    }
    failures += run_case("hadamard K=5120 T=1 in-place replay", 5120, 1, false, true, true, seed++);
    failures +=
        run_case("hadamard K=17408 T=32 in-place inverse", 17408, 32, true, true, false, seed++);
    for (const std::int32_t k : widths) failures += run_round_trip(k, 5, seed++);
    for (const std::int64_t t : {1, 4, 8, 9, 33}) {
        failures += run_silu_mul_case(17408, t, t == 8, seed++);
    }
    failures += run_silu_mul_case(1024, 3, false, seed++);
    for (const std::int64_t t : {1, 2, 4, 7, 8, 9, 16, 33, 1024}) {
        failures += run_rmsnorm_hadamard_case(5120, t, true, t == 1 || t == 8, seed++);
        failures += run_rmsnorm_hadamard_case(5120, t, false, false, seed++);
    }
    failures += run_rmsnorm_hadamard_case(6144, 3, true, false, seed++);
    for (const std::int64_t t : {1, 2, 5, 8, 16, 40}) {
        failures += run_gated_rmsnorm_hadamard_case(48, t, seed++);
    }
    failures += run_gated_rmsnorm_hadamard_case(8, 3, seed++);
    for (const std::int64_t t : {1, 4, 8, 9, 64}) {
        failures += run_sigmoid_mul_hadamard_case(6144, t, false, seed++);
        failures += run_sigmoid_mul_hadamard_case(6144, t, true, seed++);
    }

    std::cout << (failures ? "FAIL" : "OK") << " hadamard_transform\n";
    return failures ? 1 : 0;
}
