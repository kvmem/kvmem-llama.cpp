// The Q4/Q5 attention input projection's small-T MMA schedule against an fp64 oracle, at T=1..32 on
// the registered 27B shape: query_key [7168,5120] Q4 into q (rows < 6144) and k, gate_value
// [7168,5120] Q5 into gate and v. Every output element is checked, starting from a sentinel so an
// unwritten one fails; tolerance is one bf16 rounding step (tests/ops/small_t_oracle.h).

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
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
namespace oracle = ninfer::test::small_t_oracle;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kParent    = 7168;
constexpr std::int32_t kQueryRows = 6144;
constexpr std::int32_t kKvRows    = 1024;
constexpr std::int32_t kMaxTokens = 32;

int report(const char* what, std::int32_t tokens, const oracle::Miss& miss) {
    if (miss.count == 0) return 0;
    std::cerr << what << " T=" << tokens << ": " << miss.count
              << " outputs miss the fp64 oracle, first at row " << miss.row << " col " << miss.col
              << " (" << miss.value << " vs " << miss.oracle << ")\n";
    return 1;
}

std::vector<std::uint16_t> read(ninfer::test::GuardedDeviceBuffer& buffer, std::int32_t rows,
                                std::int32_t tokens) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * tokens);
    buffer.copy_to_host(host.data(), host.size() * 2);
    return host;
}

} // namespace

int main() {
    namespace qw = ninfer::test::quantized_weight;
    try {
        qw::PatternedWeightOptions options;
        options.row_split_codes = qw::RowSplitCodePattern::Hashed;
        options.row_split_scale = qw::RowSplitScalePattern::Small;
        qw::PackedWeight host_qk =
            qw::make_patterned_weight(QType::Q4_G64_FP16, kParent, kHidden, 0xa771U, options);
        qw::PackedWeight host_gv =
            qw::make_patterned_weight(QType::Q5_G64_FP16, kParent, kHidden, 0xa772U, options);
        ninfer::test::GuardedDeviceBuffer device_qk(host_qk.payload.size());
        ninfer::test::GuardedDeviceBuffer device_gv(host_gv.payload.size());
        device_qk.copy_from_host(host_qk.payload.data(), host_qk.payload.size());
        device_gv.copy_from_host(host_gv.payload.data(), host_gv.payload.size());
        const ninfer::Weight qk_weight = host_qk.device_weight(device_qk.data());
        const ninfer::Weight gv_weight = host_gv.device_weight(device_gv.data());

        std::vector<std::uint16_t> activation(static_cast<std::size_t>(kHidden) * kMaxTokens);
        std::uint64_t state = 0x13198a2e03707344ULL;
        for (auto& value : activation) {
            state               = qw::detail::mix64(state);
            const int numerator = static_cast<int>(state % 255U) - 127;
            value = oracle::f32_to_bf16_rne(static_cast<float>(numerator) * 1e-3F);
        }
        const std::vector<double> qk_oracle = oracle::project(
            qw::decode_row_split_lowbit(host_qk.payload, kParent, kHidden, kHidden,
                                        QType::Q4_G64_FP16),
            kParent, kHidden, activation, kMaxTokens);
        const std::vector<double> gv_oracle = oracle::project(
            qw::decode_row_split_lowbit(host_gv.payload, kParent, kHidden, kHidden,
                                        QType::Q5_G64_FP16),
            kParent, kHidden, activation, kMaxTokens);

        ninfer::test::GuardedDeviceBuffer device_x(activation.size() * 2);
        device_x.copy_from_host(activation.data(), activation.size() * 2);
        const auto bytes = [](std::int32_t rows) {
            return static_cast<std::size_t>(rows) * kMaxTokens * 2;
        };
        ninfer::test::GuardedDeviceBuffer q(bytes(kQueryRows)), gate(bytes(kQueryRows));
        ninfer::test::GuardedDeviceBuffer k(bytes(kKvRows)), v(bytes(kKvRows));

        int failures = 0;
        for (const std::int32_t tokens :
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 20, 24, 31, 32}) {
            Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
            Tensor tq(q.data(), DType::BF16, {kQueryRows, tokens});
            Tensor tg(gate.data(), DType::BF16, {kQueryRows, tokens});
            Tensor tk(k.data(), DType::BF16, {kKvRows, tokens});
            Tensor tv(v.data(), DType::BF16, {kKvRows, tokens});
            for (auto* buffer : {&q, &gate, &k, &v}) {
                buffer->fill(0xff); // a NaN sentinel, so an unwritten element fails
            }
            ninfer::ops::detail::q4_q5_attn_input_small_t_mma_launch(x, qk_weight, gv_weight, tq,
                                                                     tg, tk, tv, nullptr);
            ninfer::test::cuda_check(cudaDeviceSynchronize(), "attention input small-T MMA");
            failures += report("q", tokens,
                               oracle::score(read(q, kQueryRows, tokens), kQueryRows, kQueryRows,
                                             tokens, qk_oracle, kParent, 0));
            failures += report("k", tokens,
                               oracle::score(read(k, kKvRows, tokens), kKvRows, kKvRows, tokens,
                                             qk_oracle, kParent, kQueryRows));
            failures += report("gate", tokens,
                               oracle::score(read(gate, kQueryRows, tokens), kQueryRows,
                                             kQueryRows, tokens, gv_oracle, kParent, 0));
            failures += report("v", tokens,
                               oracle::score(read(v, kKvRows, tokens), kKvRows, kKvRows, tokens,
                                             gv_oracle, kParent, kQueryRows));
        }
        std::cout << (failures == 0 ? "OK" : "FAIL")
                  << " attention input small-T MMA matches the fp64 oracle at T=1..32\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "attention input small-T MMA test failed: " << error.what() << '\n';
        return 1;
    }
}
