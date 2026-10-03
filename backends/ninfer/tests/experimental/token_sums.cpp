#include "ninfer/ops/token_sums.h"
#include "ops/op_tester.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

void check_case(int channels, int tokens, int first, int block, bool clipped, bool graph) {
    std::mt19937 rng(913U + channels + tokens);
    std::uniform_real_distribution<float> random(-4.0F, 4.0F);
    std::vector<std::uint16_t> represented(static_cast<std::size_t>(channels) * tokens);
    for (auto& value : represented) { value = f32_to_bf16(random(rng)); }
    const auto buckets = block ? (tokens + 2 * block - 2) / block : 1;
    auto x = to_device(represented);
    auto origin_buffer = to_device(std::vector<int>{first});
    auto range_buffer = to_device(std::vector<int>{0, 1000000});
    DeviceBuffer result_buffer(static_cast<std::size_t>(channels) * buckets * sizeof(float));
    Tensor input(x.p, DType::BF16, {channels, tokens});
    Tensor origin(origin_buffer.p, DType::I32, {1});
    Tensor range(range_buffer.p, DType::I32, {2});
    Tensor output(result_buffer.p, DType::FP32, {channels, buckets});
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "stream");
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture begin");
        ops::token_sums(input, origin, range, block, output, stream);
        cuda_check(cudaStreamEndCapture(stream, &captured), "capture end");
        cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0), "instantiate");
    }
    for (int replay = 0; replay < (graph ? 3 : 1); ++replay) {
        // The original position changes across page boundaries without recapture.
        const int start = first + replay * 37;
        const int limits[] = {clipped ? start + 7 : start,
                              clipped ? start + tokens - 13 : start + tokens};
        origin_buffer.copy_from_host(&start, sizeof(start));
        range_buffer.copy_from_host(limits, sizeof(limits));
        if (graph) { cuda_check(cudaGraphLaunch(executable, stream), "graph launch"); }
        else { ops::token_sums(input, origin, range, block, output, stream); }
        cuda_check(cudaStreamSynchronize(stream), "sum completion");
        const auto actual = from_device<float>(result_buffer, output.numel());
        std::vector<double> expected(actual.size(), 0), absolute(actual.size(), 0);
        // Independent oracle bins each represented input row by original position.
        for (int row = tokens - 1; row >= 0; --row) {
            const int position = start + row;
            if (position < limits[0] || position >= limits[1]) { continue; }
            const int bucket = block ? position / block - start / block : 0;
            for (int channel = 0; channel < channels; ++channel) {
                const double value = bf16_to_f32(represented[static_cast<std::size_t>(row) * channels + channel]);
                const auto index = static_cast<std::size_t>(bucket) * channels + channel;
                expected[index] += value;
                absolute[index] += std::abs(value);
            }
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            // FP32 summation forward-error bound from the represented BF16 inputs.
            const double bound = 2.0 * tokens * std::numeric_limits<float>::epsilon() * absolute[i];
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > std::max(1e-6, bound)) {
                throw std::runtime_error("token_sums differs from the FP64 oracle");
            }
        }
    }
    if (executable) { cuda_check(cudaGraphExecDestroy(executable), "destroy graph executable"); }
    if (captured) { cuda_check(cudaGraphDestroy(captured), "destroy graph"); }
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
}

int main(int argc, char** argv) {
    try {
        if (cuda_unavailable()) { return 77; }
        int cases = 0;
        if (argc == 2 && std::string(argv[1]) == "candidates") {
            for (int channels : {512, 1024}) {
                for (int tokens : {2, 4, 8, 12, 16, 32, 64}) {
                    for (bool graph : {false, true}) {
                        check_case(channels, tokens, 0, 1, false, graph);
                        ++cases;
                    }
                }
            }
            std::cout << "TOKEN_SUMS_CANDIDATES PASS cases=" << cases << '\n';
            return 0;
        }
        for (int channels : {512, 1024, 4096, 6144}) {
            for (int tokens : {1, 63, 64, 65, 128, 511, 512, 513}) {
                for (int first : {0, 63, 262000}) {
                    for (int block : {0, 1, 64}) {
                        for (bool clipped : {false, true}) {
                            check_case(channels, tokens, first, block, clipped, false);
                            ++cases;
                        }
                    }
                }
            }
            check_case(channels, 65, 262000, 64, false, true);
            check_case(channels, 65, 262000, 0, true, true);
            for (int tokens : {1, 2, 3, 4}) {
                check_case(channels, tokens, 63, 1, false, true);
            }
            cases += 6;
        }
        std::cout << "TOKEN_SUMS PASS cases=" << cases << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TOKEN_SUMS FAIL " << error.what() << '\n';
        return 1;
    }
}
