#include "ninfer/ops/gated_delta_net.h"

#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

// The two-stage prefill (NINFER_GDN_TWO_STAGE=1) qualified against the complete FP64 recurrence;
// test_gated_delta_net.cpp qualifies the WY/state-passing prefill and the decode paths.
namespace {

constexpr int kStateDim = 128;

enum class PrecisionProfile { Recurrent, ChunkedNormalized, ChunkedRaw };

struct Criteria {
    const char* name;
    ReductionCriterion out;
    ReductionCriterion state;
};

// Engineering limits for the represented public inputs, not universal forward-error
// bounds. The chunked profile keeps the master state in FP32, uses TF32 residual products and
// BF16 readout/update operands, and optionally materializes normalized Q/K once in BF16.
// Output has an additional final BF16 conversion. These profile-wide regression limits retain
// margin around the qualified error envelope and apply uniformly across head counts, sequence
// lengths and gate values. The recurrent profile keeps test_gated_delta_net.cpp's limits.
constexpr std::array<Criteria, 3> kCriteria{{
    {"recurrent",
     {4.1e-3, 5.0e-6, kBf16GrossRelativeFloor},
     {2.7e-3, 1.0e-5, kBf16GrossRelativeFloor}},
    {"chunked-normalized", {6.0e-3, 5.0e-6, 8.0e-3}, {4.5e-3, 1.0e-5, 9.0e-3}},
    {"chunked-raw", {4.5e-3, 5.0e-6, 7.0e-3}, {3.0e-3, 1.0e-5, 4.0e-3}},
}};

PrecisionProfile prefill_profile(int tokens, bool normalize_qk) {
    // Qualify the current public prefill boundary explicitly, including T=15/16/17.
    if (tokens < 16) return PrecisionProfile::Recurrent;
    return normalize_qk ? PrecisionProfile::ChunkedNormalized : PrecisionProfile::ChunkedRaw;
}

enum class InputPattern {
    Ordinary,
    WeakDecay,
    ZeroDecay,
    SmallBeta,
    StrongDecay,
    NearCollinear,
    ZeroState,
    ZeroQuery,
    ZeroSignal,
    ZeroBeta,
    UnitBeta,
};

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed,
                            InputPattern pattern = InputPattern::Ordinary) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    switch (pattern) {
    case InputPattern::Ordinary:
        break;
    case InputPattern::WeakDecay:
        std::fill(in.g.begin(), in.g.end(), -1.0e-6F);
        break;
    case InputPattern::ZeroDecay:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        break;
    case InputPattern::SmallBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.002F);
        break;
    case InputPattern::StrongDecay:
        std::fill(in.g.begin(), in.g.end(), -80.0F);
        break;
    case InputPattern::NearCollinear: {
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.98F);
        for (float& value : in.state) value *= 50.0F;
        const std::vector<float> base(in.k.begin(), in.k.begin() + test_case.qk_heads * kStateDim);
        for (int t = 0; t < test_case.tokens; ++t) {
            for (int qh = 0; qh < test_case.qk_heads; ++qh) {
                for (int d = 0; d < kStateDim; ++d) {
                    const auto offset = (t * test_case.qk_heads + qh) * kStateDim + d;
                    in.k[offset]      = base[qh * kStateDim + d] + 0.002F * in.k[offset];
                }
            }
        }
        round_to_bf16(in.k);
        // Make V close to the prediction from the initial state. This stresses cancellation
        // in the update; it does not change the independent mathematical oracle.
        for (int t = 0; t < test_case.tokens; ++t) {
            for (int h = 0; h < test_case.value_heads; ++h) {
                const int qh      = h / (test_case.value_heads / test_case.qk_heads);
                const auto offset = (t * test_case.qk_heads + qh) * kStateDim;
                double sum        = 1.0e-6;
                for (int d = 0; d < kStateDim; ++d)
                    sum += double(in.k[offset + d]) * in.k[offset + d];
                const double inv = test_case.normalize_qk ? 1.0 / std::sqrt(sum) : 1.0;
                for (int row = 0; row < kStateDim; ++row) {
                    double prediction = 0.0;
                    for (int d = 0; d < kStateDim; ++d)
                        prediction += double(in.state[(h * kStateDim + row) * kStateDim + d]) *
                                      in.k[offset + d] * inv;
                    in.v[(t * test_case.value_heads + h) * kStateDim + row] =
                        static_cast<float>(prediction);
                }
            }
        }
        round_to_bf16(in.v);
        break;
    }
    case InputPattern::ZeroState:
        std::fill(in.state.begin(), in.state.end(), 0.0F);
        break;
    case InputPattern::ZeroQuery:
        std::fill(in.q.begin(), in.q.end(), 0.0F);
        break;
    case InputPattern::ZeroSignal:
        std::fill(in.state.begin(), in.state.end(), 0.0F);
        std::fill(in.v.begin(), in.v.end(), 0.0F);
        break;
    case InputPattern::ZeroBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.0F);
        break;
    case InputPattern::UnitBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 1.0F);
        break;
    }
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return ninfer::test::verify_exact(label.c_str(), got, expected);
}

struct WorstError {
    double value = 0.0;
    std::string label;

    void include(double candidate, const std::string& name) {
        if (label.empty() || candidate > value) {
            value = candidate;
            label = name;
        }
    }
};

struct ProfileSummary {
    std::size_t heads_checked = 0;
    WorstError relative_l2;
    WorstError gross_ratio;
};

std::array<std::array<ProfileSummary, 2>, kCriteria.size()> summaries;

int verify_heads(const std::string& label, std::span<const double> got,
                 std::span<const double> expected, int value_heads, int tokens,
                 PrecisionProfile profile, bool state) {
    const auto profile_index = static_cast<std::size_t>(profile);
    const auto& criteria     = kCriteria[profile_index];
    const auto& criterion    = state ? criteria.state : criteria.out;
    const std::size_t head_size =
        static_cast<std::size_t>(kStateDim) * (state ? kStateDim : tokens);
    if (got.size() != head_size * value_heads || got.size() != expected.size()) {
        throw std::logic_error("GDN per-head comparison shape mismatch");
    }
    std::vector<double> head_got(head_size), head_ref(head_size);
    int failures = 0;
    for (int h = 0; h < value_heads; ++h) {
        if (state) {
            std::copy_n(got.data() + h * head_size, head_size, head_got.data());
            std::copy_n(expected.data() + h * head_size, head_size, head_ref.data());
        } else {
            for (int t = 0; t < tokens; ++t) {
                const auto source = (static_cast<std::size_t>(t) * value_heads + h) * kStateDim;
                std::copy_n(got.data() + source, kStateDim, head_got.data() + t * kStateDim);
                std::copy_n(expected.data() + source, kStateDim, head_ref.data() + t * kStateDim);
            }
        }
        const std::string head_label = label + " head=" + std::to_string(h);
        const auto stats = compute_reduction_stats(head_got.data(), head_ref.data(), head_size);
        const double gross_limit = gross_error_limit(stats, criterion);
        auto& summary            = summaries[profile_index][state ? 1 : 0];
        ++summary.heads_checked;
        summary.relative_l2.include(stats.relative_l2, head_label);
        summary.gross_ratio.include(stats.maximum_absolute_error / gross_limit, head_label);
        report_reduction_stats(head_label, head_size, stats, criterion);
        if (!reduction_passes(stats, head_size, criterion)) {
            std::cerr << head_label << ": " << criteria.name << " rel_l2=" << stats.relative_l2
                      << " limit=" << criterion.relative_l2
                      << " max_abs=" << stats.maximum_absolute_error
                      << " gross_limit=" << gross_limit << " non_finite=" << stats.first_non_finite
                      << '\n';
            ++failures;
        }
    }
    return failures;
}

void print_summaries() {
    for (std::size_t p = 0; p < kCriteria.size(); ++p) {
        for (int state = 0; state < 2; ++state) {
            const auto& summary   = summaries[p][state];
            const auto& criterion = state ? kCriteria[p].state : kCriteria[p].out;
            std::cout << "GDN_PROFILE profile=" << kCriteria[p].name
                      << " kind=" << (state ? "state" : "out")
                      << " heads_checked=" << summary.heads_checked << std::setprecision(10)
                      << " max_rel_l2=" << summary.relative_l2.value
                      << " rel_l2_limit=" << criterion.relative_l2
                      << " max_gross_ratio=" << summary.gross_ratio.value
                      << " l2_case=" << summary.relative_l2.label
                      << " gross_case=" << summary.gross_ratio.label << '\n';
        }
    }
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

gdn_ref::Result reference_prefix(const gdn_ref::Inputs& in, int tokens, float scale,
                                 bool normalize_qk) {
    if (tokens == in.tokens) return gdn_ref::evaluate(in, scale, normalize_qk);
    gdn_ref::Inputs prefix;
    prefix.head_dim    = in.head_dim;
    prefix.qk_heads    = in.qk_heads;
    prefix.value_heads = in.value_heads;
    prefix.tokens      = tokens;
    prefix.q.assign(in.q.begin(), in.q.begin() + tokens * in.qk_heads * kStateDim);
    prefix.k.assign(in.k.begin(), in.k.begin() + tokens * in.qk_heads * kStateDim);
    prefix.v.assign(in.v.begin(), in.v.begin() + tokens * in.value_heads * kStateDim);
    prefix.g.assign(in.g.begin(), in.g.begin() + tokens * in.value_heads);
    prefix.beta.assign(in.beta.begin(), in.beta.begin() + tokens * in.value_heads);
    prefix.state = in.state;
    // Re-evaluate a few logical prefixes from the original public state. Neither GPU state nor
    // a cast of an earlier FP64 result is fed back into the oracle at a call boundary.
    return gdn_ref::evaluate(prefix, scale, normalize_qk);
}

int run_case(const Case& test_case, std::uint32_t seed,
             InputPattern pattern = InputPattern::Ordinary, bool inplace = false,
             std::vector<int> partitions = {}) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed, pattern);
    if (partitions.empty()) partitions.push_back(test_case.tokens);
    if (std::accumulate(partitions.begin(), partitions.end(), 0) != test_case.tokens ||
        *std::min_element(partitions.begin(), partitions.end()) <= 0) {
        throw std::logic_error("invalid GDN continuation partition");
    }
    const std::string label =
        std::string(test_case.name) + " H=" + std::to_string(test_case.qk_heads) + "/" +
        std::to_string(test_case.value_heads) + " T=" + std::to_string(test_case.tokens) +
        (test_case.normalize_qk ? " normalized" : " raw") +
        (inplace ? " inplace" : " distinct-state");
    std::cout << "GDN_CASE " << label << " calls=" << partitions.size() << std::endl;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kStateDim));
    DeviceInputs device(in);
    GuardedDeviceBuffer state_a(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_b(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_a.copy_from_host(in.state.data(), state_a.bytes());
    out.fill(0xff);
    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor output(out.data(), DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});

    const int maximum_call     = *std::max_element(partitions.begin(), partitions.end());
    const auto workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, 1, maximum_call);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 256));
    scratch.fill(0xff);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});

    GuardedDeviceBuffer* current = &state_a;
    int failures                 = 0;
    int begin                    = 0;
    bool consumed_chunked        = false;
    for (int length : partitions) {
        auto* destination = inplace ? current : (current == &state_a ? &state_b : &state_a);
        std::vector<float> source_before;
        if (!inplace) {
            source_before = from_device<float>(current->data(), in.state.size());
            destination->fill(0xff);
        }
        Tensor input_state(current->data(), DType::FP32,
                           {kStateDim, kStateDim, test_case.value_heads});
        Tensor output_state(destination->data(), DType::FP32,
                            {kStateDim, kStateDim, test_case.value_heads});
        Tensor qs = q.slice(2, begin, length), ks = k.slice(2, begin, length);
        Tensor vs = v.slice(2, begin, length), gs = g.slice(1, begin, length);
        Tensor bs = beta.slice(1, begin, length), os = output.slice(2, begin, length);
        if (inplace) {
            ops::gated_delta_net(qs, ks, vs, gs, bs, scale, test_case.normalize_qk, workspace,
                                 output_state, os, nullptr);
        } else {
            ops::gated_delta_net(qs, ks, vs, gs, bs, scale, test_case.normalize_qk, workspace,
                                 input_state, output_state, os, nullptr);
        }
        cuda_synchronize();
        begin += length;
        consumed_chunked |=
            prefill_profile(length, test_case.normalize_qk) != PrecisionProfile::Recurrent;
        // A recurrent suffix inherits earlier chunked error. Check the complete prefix against
        // the sequence's arithmetic profile; do not reset its error by seeding the oracle with
        // the GPU's intermediate state.
        const auto profile = consumed_chunked
                                 ? prefill_profile(test_case.tokens, test_case.normalize_qk)
                                 : PrecisionProfile::Recurrent;
        const std::string checkpoint = label + " prefix=" + std::to_string(begin);
        const auto ref               = reference_prefix(in, begin, scale, test_case.normalize_qk);
        const auto output_count =
            static_cast<std::size_t>(begin) * test_case.value_heads * kStateDim;
        const auto got_output = from_device_bf16(out.data(), output_count);
        const auto got_state  = read_f32(destination->data(), in.state.size());
        failures += verify_heads(checkpoint + " out", got_output, ref.out, test_case.value_heads,
                                 begin, profile, false);
        if (partitions.size() > 1) {
            const auto offset =
                static_cast<std::size_t>(begin - length) * test_case.value_heads * kStateDim;
            failures += verify_heads(checkpoint + " current out",
                                     std::span<const double>(got_output).subspan(offset),
                                     std::span<const double>(ref.out).subspan(offset),
                                     test_case.value_heads, length, profile, false);
        }
        failures += verify_heads(checkpoint + " state", got_state, ref.final_state,
                                 test_case.value_heads, 1, profile, true);
        if (pattern == InputPattern::ZeroQuery || pattern == InputPattern::ZeroSignal) {
            failures += verify_exact(checkpoint + " exact zero output", got_output,
                                     std::vector<double>(output_count, 0.0));
        }
        if (pattern == InputPattern::ZeroSignal) {
            failures += verify_exact(checkpoint + " exact zero state", got_state,
                                     std::vector<double>(in.state.size(), 0.0));
        }
        if (pattern == InputPattern::ZeroBeta) {
            failures += verify_exact(checkpoint + " identity state", got_state, doubles(in.state));
        }
        if (!inplace) {
            failures +=
                verify_exact(checkpoint + " source state unchanged",
                             from_device<float>(current->data(), in.state.size()), source_before);
        }
        if (output_count < in.v.size()) {
            const auto* suffix = static_cast<const std::uint16_t*>(out.data()) + output_count;
            failures +=
                verify_exact(checkpoint + " unconsumed output unchanged",
                             from_device<std::uint16_t>(suffix, in.v.size() - output_count),
                             std::vector<std::uint16_t>(in.v.size() - output_count, 0xffffU));
        }
        failures += state_a.verify_guards(checkpoint + " state-a");
        failures += state_b.verify_guards(checkpoint + " state-b");
        failures += out.verify_guards(checkpoint + " out");
        failures += scratch.verify_guards(checkpoint + " workspace");
        if (workspace.used() != 0) {
            std::cerr << checkpoint << ": workspace was not released\n";
            ++failures;
        }
        current = destination;
    }
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    // The capacity query covers either prefill algorithm; this one needs at most that.
    if (workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace exceeded its capacity query\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    // Read once, at the first prefill.
#ifdef _WIN32
    _putenv_s("NINFER_GDN_TWO_STAGE", "1");
#else
    setenv("NINFER_GDN_TWO_STAGE", "1", 1);
#endif

    int failures = 0;

    for (const bool normalize_qk : {false, true}) {
        const std::size_t interval =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 15, 17);
        const std::size_t witness =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 17, 17);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the packet boundary\n";
            ++failures;
        }
    }

    failures += run_case({"27b recurrent", 16, 48, 1, true}, 12001U, InputPattern::Ordinary, true);
    failures += run_case({"27b recurrent", 16, 48, 7, false}, 12007U);
    failures += run_case({"35b recurrent", 16, 32, 15, true}, 12015U);

    struct Geometry {
        const char* name;
        int qk_heads, value_heads, tokens;
    };

    for (const auto geometry :
         {Geometry{"qwen-27b", 16, 48, 1025}, Geometry{"qwen-35b-a3b", 16, 32, 1025},
          Geometry{"kimi-k3", 96, 96, 1025}, Geometry{"glm5.3-flash", 64, 64, 1025},
          Geometry{"single-head", 1, 1, 65}, Geometry{"shared-qk", 1, 3, 65},
          Geometry{"generic-grouped-map", 5, 15, 65}}) {
        for (bool normalize : {false, true}) {
            failures += run_case({geometry.name, geometry.qk_heads, geometry.value_heads,
                                  geometry.tokens, normalize},
                                 16180U + geometry.value_heads, InputPattern::Ordinary, !normalize);
        }
    }
    for (int tokens : {15, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 4095, 4096}) {
        failures += run_case({"prefill boundary", 1, 3, tokens, true}, 90000U + tokens,
                             InputPattern::Ordinary, tokens % 2 != 0);
    }
    for (int tokens : {15, 16, 17}) {
        failures += run_case({"raw prefill boundary", 5, 15, tokens, false}, 91000U + tokens);
    }

    struct Stress {
        const char* name;
        InputPattern pattern;
        int tokens;
        std::uint32_t seed;
    };

    // Retain the previously observed error cases, including long weak/zero decay. The limits
    // above are fixed before qualification, shared by every geometry and every input pattern.
    for (const auto stress :
         {Stress{"weak decay", InputPattern::WeakDecay, 16385, 332211U},
          Stress{"zero decay", InputPattern::ZeroDecay, 16385, 332212U},
          Stress{"small beta", InputPattern::SmallBeta, 16385, 332213U},
          Stress{"strong decay", InputPattern::StrongDecay, 16385, 332214U},
          Stress{"near collinear", InputPattern::NearCollinear, 4097, 332215U}}) {
        failures += run_case({stress.name, 1, 3, stress.tokens, true}, stress.seed, stress.pattern);
    }
    for (const auto stress : {Stress{"raw weak decay", InputPattern::WeakDecay, 16385, 332211U},
                              Stress{"raw zero decay", InputPattern::ZeroDecay, 16385, 332212U}}) {
        failures +=
            run_case({stress.name, 1, 3, stress.tokens, false}, stress.seed, stress.pattern, true);
    }
    failures += run_case({"35b zero decay", 16, 32, 4097, true}, 16032U, InputPattern::ZeroDecay);
    failures +=
        run_case({"35b weak decay", 16, 32, 4097, true}, 16032U, InputPattern::WeakDecay, true);
    failures += run_case({"27b strong tail", 16, 48, 65, true}, 12065U, InputPattern::StrongDecay);
    // The first checkpoint is the aligned T=64 prefix of the identical T=65 stress input.
    failures += run_case({"27b strong continuation", 16, 48, 65, true}, 12065U,
                         InputPattern::StrongDecay, true, {64, 1});

    failures += run_case({"27b prefill continuation", 16, 48, 2049, true}, 14249U,
                         InputPattern::ZeroState, false, {1024, 1024, 1});
    failures += run_case({"weak prefill continuation", 1, 3, 4097, true}, 14297U,
                         InputPattern::WeakDecay, true, {1024, 1024, 1024, 1024, 1});
    failures += run_case({"raw uneven continuation", 5, 15, 1025, false}, 14125U,
                         InputPattern::ZeroDecay, false, {15, 17, 992, 1});
    failures += run_case({"near-zero normalization", 1, 3, 17, true, true}, 14017U);
    for (bool normalize : {false, true}) {
        failures += run_case({"zero query", 1, 3, 17, normalize}, 14317U, InputPattern::ZeroQuery);
        failures +=
            run_case({"zero signal", 1, 3, 17, normalize}, 14417U, InputPattern::ZeroSignal, true);
        failures += run_case({"zero beta", 1, 3, 17, normalize}, 14517U, InputPattern::ZeroBeta);
        failures +=
            run_case({"unit beta", 1, 3, 257, normalize}, 14657U, InputPattern::UnitBeta, true);
    }

    print_summaries();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net two-stage correctness\n";
    return failures == 0 ? 0 : 1;
}
