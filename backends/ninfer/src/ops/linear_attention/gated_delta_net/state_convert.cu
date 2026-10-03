#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

__global__ void widen_kernel(const __half2* __restrict__ in, float2* __restrict__ out,
                             std::int64_t pairs) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < pairs; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = __half22float2(in[i]);
    }
}

__global__ void narrow_kernel(const float2* __restrict__ in, __half2* __restrict__ out,
                              std::int64_t pairs) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < pairs; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = __float22half2_rn(in[i]);
    }
}

std::int64_t checked_pairs(const Tensor& a, const Tensor& b) {
    const std::int64_t elements = static_cast<std::int64_t>(a.bytes() / (a.dtype == DType::FP16 ? 2 : 4));
    const std::int64_t other    = static_cast<std::int64_t>(b.bytes() / (b.dtype == DType::FP16 ? 2 : 4));
    if (elements != other || (elements % 2) != 0 || !a.is_contiguous() || !b.is_contiguous()) {
        throw std::invalid_argument("GDN state convert: tensors must match and be contiguous");
    }
    return elements / 2;
}

} // namespace

void widen_state_fp16_to_fp32(const Tensor& in, Tensor& out, cudaStream_t stream) {
    if (in.dtype != DType::FP16 || out.dtype != DType::FP32) {
        throw std::invalid_argument("GDN state widen: expects FP16 -> FP32");
    }
    const std::int64_t pairs = checked_pairs(in, out);
    widen_kernel<<<256, 256, 0, stream>>>(static_cast<const __half2*>(in.data),
                                          static_cast<float2*>(out.data), pairs);
    CUDA_CHECK(cudaGetLastError());
}

void narrow_state_fp32_to_fp16(const Tensor& in, Tensor& out, cudaStream_t stream) {
    if (in.dtype != DType::FP32 || out.dtype != DType::FP16) {
        throw std::invalid_argument("GDN state narrow: expects FP32 -> FP16");
    }
    const std::int64_t pairs = checked_pairs(in, out);
    narrow_kernel<<<256, 256, 0, stream>>>(static_cast<const float2*>(in.data),
                                           static_cast<__half2*>(out.data), pairs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::gated_delta_net
