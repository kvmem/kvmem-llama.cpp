#include "ninfer/ops/token_sums.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
__global__ void sum_tokens(const __nv_bfloat16* input, const std::int32_t* origin,
                           const std::int32_t* range, std::uint32_t block_tokens,
                           std::int64_t channels, std::int64_t tokens,
                           std::int64_t count, float* output) {
    const std::int64_t begin = origin[0];
    for (std::int64_t cell = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         cell < count; cell += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const auto channel = cell % channels;
        const auto bucket = cell / channels;
        std::int64_t first = 0, last = tokens;
        if (block_tokens) {
            const auto page_begin = (begin / block_tokens + bucket) * block_tokens;
            first = max(std::int64_t{0}, page_begin - begin);
            last = min(tokens, page_begin + block_tokens - begin);
        }
        first = max(first, static_cast<std::int64_t>(range[0]) - begin);
        last = min(last, static_cast<std::int64_t>(range[1]) - begin);
        float sum = 0.0F;
        for (auto token = first; token < last; ++token) {
            sum += __bfloat162float(input[token * channels + channel]);
        }
        output[cell] = sum;
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto first = reinterpret_cast<std::uintptr_t>(a.data);
    const auto second = reinterpret_cast<std::uintptr_t>(b.data);
    return first < second + b.bytes() && second < first + a.bytes();
}
} // namespace

void token_sums(const Tensor& input, const Tensor& origin, const Tensor& range,
                std::uint32_t block_tokens, Tensor& output, cudaStream_t stream) {
    const auto tokens = static_cast<std::int64_t>(input.ne[1]);
    const auto buckets = block_tokens
        ? (tokens + 2 * static_cast<std::int64_t>(block_tokens) - 2) / block_tokens : 1;
    if (!input.data || !origin.data || !range.data || !output.data ||
        !input.is_contiguous() || !origin.is_contiguous() || !range.is_contiguous() ||
        !output.is_contiguous() || input.dtype != DType::BF16 ||
        origin.dtype != DType::I32 || origin.numel() != 1 ||
        range.dtype != DType::I32 || range.numel() != 2 || output.dtype != DType::FP32 ||
        input.ne[0] <= 0 || tokens <= 0 || input.ne[2] != 1 || input.ne[3] != 1 ||
        output.ne[0] != input.ne[0] || output.ne[1] != buckets ||
        output.ne[2] != 1 || output.ne[3] != 1 || overlaps(input, output) ||
        overlaps(origin, output) || overlaps(range, output) || overlaps(input, origin) ||
        overlaps(input, range) || overlaps(origin, range)) {
        throw std::invalid_argument("token_sums: invalid shape, type, or aliased buffers");
    }
    constexpr int threads = 128;
    const auto blocks = static_cast<unsigned>(std::min<std::int64_t>(65535,
        (output.numel() + threads - 1) / threads));
    sum_tokens<<<blocks, threads, 0, stream>>>(static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const std::int32_t*>(origin.data), static_cast<const std::int32_t*>(range.data),
        block_tokens, input.ne[0], tokens, output.numel(), static_cast<float*>(output.data));
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("token_sums: ") + cudaGetErrorString(status));
    }
}
} // namespace ninfer::ops
