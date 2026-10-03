// The Q4/Q5 GDN input projection's small-T MMA schedule against an fp64 oracle, at T=1..32 on the
// registered 27B shape: query/key [4096,5120] Q4 into qkv rows [0, 4096), value/z [12288,5120] Q5
// into qkv rows [4096, 10240) (value) and z (rows [6144, 12288) of the Q5 weight). Every output
// element is checked, starting from a sentinel so an unwritten one fails; tolerance is one bf16
// rounding step (tests/ops/small_t_oracle.h).

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/small_t_oracle.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::ops::detail::Q4Q5GdnInputScheduleId;
namespace oracle = ninfer::test::small_t_oracle;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kQkRows    = 4096;
constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kValueZ    = 12288;
constexpr std::int32_t kQkvRows   = kQkRows + kValueRows;
constexpr std::int32_t kZRows     = 6144;
constexpr std::int32_t kMaxTokens = 32;

int report(const char* what, std::int32_t tokens, const oracle::Miss& miss) {
    if (miss.count == 0) return 0;
    std::cerr << what << " T=" << tokens << ": " << miss.count
              << " outputs miss the fp64 oracle, first at row " << miss.row << " col " << miss.col
              << " (" << miss.value << " vs " << miss.oracle << ")\n";
    return 1;
}

} // namespace

int main() {
    namespace qw = ninfer::test::quantized_weight;
    try {
        qw::PatternedWeightOptions options;
        options.row_split_codes = qw::RowSplitCodePattern::Hashed;
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        qw::PackedWeight host_qk =
            qw::make_patterned_weight(QType::Q4_G64_FP16, kQkRows, kHidden, 0x6d41U, options);
        qw::PackedWeight host_vz =
            qw::make_patterned_weight(QType::Q5_G64_FP16, kValueZ, kHidden, 0x6d42U, options);
        ninfer::test::GuardedDeviceBuffer device_qk(host_qk.payload.size());
        ninfer::test::GuardedDeviceBuffer device_vz(host_vz.payload.size());
        device_qk.copy_from_host(host_qk.payload.data(), host_qk.payload.size());
        device_vz.copy_from_host(host_vz.payload.data(), host_vz.payload.size());
        const ninfer::Weight qk_weight = host_qk.device_weight(device_qk.data());
        const ninfer::Weight vz_weight = host_vz.device_weight(device_vz.data());

        std::vector<std::uint16_t> activation(static_cast<std::size_t>(kHidden) * kMaxTokens);
        std::uint64_t state = 0x243f6a8885a308d3ULL;
        for (auto& value : activation) {
            state               = qw::detail::mix64(state);
            const int numerator = static_cast<int>(state % 255U) - 127;
            value = oracle::f32_to_bf16_rne(static_cast<float>(numerator) * 1e-3F);
        }
        const std::vector<double> qk_oracle = oracle::project(
            qw::decode_row_split_lowbit(host_qk.payload, kQkRows, kHidden, kHidden,
                                        QType::Q4_G64_FP16),
            kQkRows, kHidden, activation, kMaxTokens);
        const std::vector<double> vz_oracle = oracle::project(
            qw::decode_row_split_lowbit(host_vz.payload, kValueZ, kHidden, kHidden,
                                        QType::Q5_G64_FP16),
            kValueZ, kHidden, activation, kMaxTokens);

        ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
        device_x.copy_from_host(activation.data(), activation.size() * 2);
        ninfer::test::GuardedDeviceBuffer out_qkv(static_cast<std::size_t>(kQkvRows) * kMaxTokens *
                                                  2);
        ninfer::test::GuardedDeviceBuffer out_z(static_cast<std::size_t>(kZRows) * kMaxTokens * 2);

        int failures = 0;
        std::vector<std::uint16_t> qkv(static_cast<std::size_t>(kQkvRows) * kMaxTokens);
        std::vector<std::uint16_t> z(static_cast<std::size_t>(kZRows) * kMaxTokens);
        for (const std::int32_t tokens :
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 20, 24, 31, 32}) {
            Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
            Tensor oq(out_qkv.data(), DType::BF16, {kQkvRows, tokens});
            Tensor oz(out_z.data(), DType::BF16, {kZRows, tokens});
            out_qkv.fill(0xff); // a NaN sentinel, so an element the kernel never wrote fails
            out_z.fill(0xff);
            ninfer::ops::detail::q4_q5_gdn_input_execute_schedule(
                Q4Q5GdnInputScheduleId::SmallTMma, x, qk_weight, vz_weight, oq, oz, nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "GDN input small-T MMA");
            out_qkv.copy_to_host(qkv.data(), static_cast<std::size_t>(kQkvRows) * tokens * 2);
            out_z.copy_to_host(z.data(), static_cast<std::size_t>(kZRows) * tokens * 2);
            const std::vector<std::uint16_t> value(qkv.begin() + kQkRows, qkv.end());
            failures += report(
                "query/key", tokens,
                oracle::score(qkv, kQkvRows, kQkRows, tokens, qk_oracle, kQkRows, 0));
            failures += report(
                "value", tokens,
                oracle::score(value, kQkvRows, kValueRows, tokens, vz_oracle, kValueZ, 0));
            failures += report(
                "z", tokens,
                oracle::score(z, kZRows, kZRows, tokens, vz_oracle, kValueZ, kValueRows));
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " GDN input small-T MMA matches the fp64 oracle at T=1..32\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GDN input small-T MMA test failed: " << error.what() << '\n';
        return 1;
    }
}
