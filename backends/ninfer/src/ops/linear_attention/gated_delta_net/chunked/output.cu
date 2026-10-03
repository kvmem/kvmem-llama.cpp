#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/output.cuh"

#include <array>
#include <atomic>
#include <cstddef>

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = output;

// One resident wave of the output kernel on the current device: its SM count times the CTAs the
// kernel keeps resident per SM (four on the parts it was tuned on). Cached per device index because
// a model split over several GPUs launches each stage on its own device.
template <bool MULTI_JOB>
std::int64_t resident_wave_ctas() {
    constexpr int kCachedDevices = 64;
    static std::array<std::atomic<std::int64_t>, kCachedDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kCachedDevices) {
        return 170 * 4;
    }
    const std::int64_t known = cache[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
    if (known > 0) { return known; }
    int sms    = 0;
    int blocks = 0;
    constexpr int smem_bytes = kernel::kernel_dims::SMEM_BYTES;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaFuncSetAttribute(kernel::output_kernel<MULTI_JOB>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel::output_kernel<MULTI_JOB>,
                                                      kernel::THREADS, smem_bytes) != cudaSuccess ||
        sms <= 0 || blocks <= 0) {
        cudaGetLastError();
        return 170 * 4;
    }
    const std::int64_t ctas = static_cast<std::int64_t>(sms) * blocks;
    cache[static_cast<std::size_t>(device)].store(ctas, std::memory_order_relaxed);
    return ctas;
}

template <bool MULTI_JOB>
cudaError_t launch_fixed(const chunk_output_config& cfg, dim3 grid, head_map qk_map, int chunks) {
    constexpr int smem_bytes = kernel::kernel_dims::SMEM_BYTES;

    cudaError_t err = cudaFuncSetAttribute(kernel::output_kernel<MULTI_JOB>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }

    const dim3 block(kernel::THREADS, 1, 1);

    kernel::output_kernel<MULTI_JOB><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.q, cfg.k, cfg.v_new, cfg.g_cumsum, cfg.h_chunk, cfg.attn_out, qk_map, cfg.scale,
        chunks);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_output(const chunk_output_config& cfg) {
    stage_validator v{"launch_output", cfg.H_qk, cfg.H_v, cfg.L};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.q == nullptr || cfg.k == nullptr || cfg.v_new == nullptr || cfg.g_cumsum == nullptr ||
        cfg.h_chunk == nullptr || cfg.attn_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map     = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const std::int64_t NT = cfg.L / BT;

    // Keep at most one resident wave of this device and distribute chunks evenly across it. Small
    // grids retain one logical job per CTA.
    const std::int64_t target_ctas    = resident_wave_ctas<true>();
    const std::int64_t logical_jobs   = NT * cfg.H_v;
    const std::int64_t jobs_per_block = (logical_jobs + target_ctas - 1) / target_ctas;
    const std::int64_t grid_chunks    = (NT + jobs_per_block - 1) / jobs_per_block;
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(grid_chunks, cfg.H_v));

    const dim3 grid(static_cast<unsigned>(grid_chunks), static_cast<unsigned>(cfg.H_v), 1);
    if (jobs_per_block == 1) {
        return launch_fixed<false>(cfg, grid, qk_map, static_cast<int>(NT));
    }
    return launch_fixed<true>(cfg, grid, qk_map, static_cast<int>(NT));
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
