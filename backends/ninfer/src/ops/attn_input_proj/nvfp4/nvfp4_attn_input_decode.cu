#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

struct Nvfp4AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t,
                                          float result) const {
        constexpr std::int32_t kQueryRows  = 6144;
        constexpr std::int32_t kKeyRows    = 1024;
        constexpr std::int32_t kGateRows   = 6144;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        const __nv_bfloat16 result_bf16 = __float2bfloat16_rn(result);
        if (parent_row < kKeyBegin) {
            query[parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = result_bf16;
        } else {
            value[parent_row - kValueBegin] = result_bf16;
        }
    }
};

using DecodeGeometry = Nvfp4N14336K5120;
using DecodeSchedule =
    Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;

// Fused Offset-RMSNorm + GEMV for exactly one token. The norm prologue deliberately reuses the
// D=5120 RMSNorm launcher's pair ownership (256 threads x 10 pairs), accumulation order,
// reduction, and BF16 rounding, so the staged row is the exact BF16 image the standalone norm
// would write to the hidden buffer; the GEMV body below is nvfp4_gemv_kernel with x redirected
// to that staged row. Every CTA redundantly normalizes the same row (no cross-CTA sharing
// without a grid sync that would cost more than the redundant L2 reads).
template <class Geometry, class Schedule>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void rmsnorm_nvfp4_gemv_kernel(
    const __nv_bfloat162* __restrict__ residual, const __nv_bfloat162* __restrict__ norm_weight,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    float inverse_weight_divisor, float eps, Nvfp4AttentionInputOutput output) {
    static_assert(Schedule::kThreads == 256);
    constexpr int kPairsPerRow    = Geometry::kInputRows / 2;
    constexpr int kPairsPerThread = kPairsPerRow / Schedule::kThreads;
    static_assert(kPairsPerThread == 10);
    static_assert(sizeof(Nvfp4GemvSharedStorage<Geometry, Schedule>) +
                      sizeof(__nv_bfloat162) * kPairsPerRow +
                      sizeof(float) * (Schedule::kThreads / kWarpSize + 1) <=
                  48 * 1024);

    const int thread = static_cast<int>(threadIdx.x);
    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const int pair  = thread + i * Schedule::kThreads;
        values[i]       = residual[pair];
        weights[i]      = norm_weight[pair];
        const float2 xf = __bfloat1622float2(values[i]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    __shared__ float warp_sums[Schedule::kThreads / kWarpSize];
    __shared__ float inv_shared;
    __shared__ Nvfp4GemvSharedStorage<Geometry, Schedule> gemv_shared;
    __shared__ alignas(16) __nv_bfloat162 staged[kPairsPerRow];
    const float block_sum = block_reduce_sum<Schedule::kThreads>(sum, warp_sums);
    if (thread == 0) {
        inv_shared = rsqrtf(block_sum / static_cast<float>(Geometry::kInputRows) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const int pair  = thread + i * Schedule::kThreads;
        const float2 xf = __bfloat1622float2(values[i]);
        const float2 wf = __bfloat1622float2(weights[i]);
        staged[pair] =
            __floats2bfloat162_rn(rmsnorm_epilogue<RmsEpilogue::Offset>(xf.x, inv, wf.x, 0.0F),
                                  rmsnorm_epilogue<RmsEpilogue::Offset>(xf.y, inv, wf.y, 0.0F));
    }
    __syncthreads();

    constexpr int kCtasPerM128 = 128 / Schedule::kRowsPerCta;
    const int m_tile           = static_cast<int>(blockIdx.x) / kCtasPerM128;
    const int cta_in_tile      = static_cast<int>(blockIdx.x) - m_tile * kCtasPerM128;
    const int rmod_base        = cta_in_tile * (Schedule::kRowsPerCta / 4);
    stage_nvfp4_scales<Geometry, Schedule>(scales, gemv_shared, m_tile, rmod_base);

    const int lane      = thread & 31;
    const int warp      = thread >> 5;
    const int flat_row0 = warp * Schedule::kRowsPerWarp;
    int parent_rows[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        const int flat_row     = flat_row0 + local_row;
        const int rmod         = rmod_base + flat_row / 4;
        const int quartile     = flat_row & 3;
        parent_rows[local_row] = m_tile * 128 + rmod + quartile * 32;
    }

    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    compute_nvfp4_rows<Geometry, Schedule>(reinterpret_cast<const __nv_bfloat16*>(staged), codes,
                                           scales, gemv_shared, inverse_weight_divisor, parent_rows,
                                           flat_row0, lane, accumulators);

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        total = warp_reduce_sum(total);
        if (lane == 0) {
            const int parent_row = parent_rows[local_row];
            output.store(parent_row, 0, Nvfp4IdentityEpilogue{}.apply(parent_row, 0, total));
        }
    }
}

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry = DecodeGeometry;
    using Schedule = DecodeSchedule;
    static_assert((6144 % 128) == 0);
    static_assert((1024 % 128) == 0);

    const Nvfp4AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

void nvfp4_attn_input_fused_rmsnorm_decode_launch(const Tensor& residual, const Tensor& norm_weight,
                                                  float eps, const Weight& weight, Tensor& q,
                                                  Tensor& gate, Tensor& k, Tensor& v,
                                                  cudaStream_t stream) {
    using Geometry = DecodeGeometry;
    using Schedule = DecodeSchedule;
    const Nvfp4AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    rmsnorm_nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(residual.data),
        static_cast<const __nv_bfloat162*>(norm_weight.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor, eps, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
