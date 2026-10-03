#pragma once

// Host side of ggml's integer tensor-core matrix kernel (mmq.cuh launch_mul_mat_q), without ggml's
// backend context: the caller owns the activation, the FP32 output and the stream-k fixup plane.

#include "ggml_bridge_internal.cuh"

#include <climits>
#include <mutex>

namespace ninfer::ops::gguf::detail {

struct MatrixLaunch {
    int J                   = 0;
    int I                   = 0;
    int threads             = 0;
    std::size_t shared      = 0;
    int row_tiles           = 0;
    int column_tiles        = 0;
    bool stream_k           = false;
    int blocks              = 0;
    bool fixup              = false;
};

template <ggml_type type>
MatrixLaunch select_matrix_launch(int rows, int columns) {
    const DeviceFacts& device = device_facts();
    MatrixLaunch best;
    int best_tiles = INT_MAX;
    for (const int J : kMatrixColumnTiles) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT) { continue; }
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) { continue; }
        const int tiles = (columns + J - 1) / J;
        if (tiles < best_tiles) {
            best_tiles        = tiles;
            best.J            = J;
            best.I            = config.I;
            best.threads      = config.nthreads;
            best.shared       = shared;
            best.stream_k     = config.stream_k;
        }
        if (tiles == 1) { break; }
    }
    if (best.J == 0) { throw std::invalid_argument("gguf matrix product: no tile fits this device"); }
    best.row_tiles    = (rows + best.I - 1) / best.I;
    best.column_tiles = (columns + best.J - 1) / best.J;
    const int tiles   = best.row_tiles * best.column_tiles;
    if (best.stream_k) {
        const int waves      = (tiles + device.sm_count - 1) / device.sm_count;
        const int efficiency = 100 * tiles / (device.sm_count * waves);
        best.blocks          = efficiency >= 90 ? tiles : device.sm_count;
        best.fixup           = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns) {
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    return launch.fixup ? std::size_t(launch.blocks) * launch.J * launch.I * sizeof(float) : 0;
}

template <ggml_type type, int J>
void launch_matrix(const MatrixLaunch& launch, const char* weight, int stride_row_blocks, int rows,
                   int k, const int* activation, int columns, float* out,
                   std::int64_t out_column_stride, float* fixup, cudaStream_t stream) {
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(mul_mat_q<type, J, false>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(launch.shared)),
              "matrix shared memory");
    });
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    const uint3 blocks_fd   = init_fastdiv_values(k / qk);
    const uint3 one_fd      = init_fastdiv_values(1);
    const uint3 ntx_fd      = init_fastdiv_values(launch.column_tiles);
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const int stride        = static_cast<int>(out_column_stride);
    if (!launch.stream_k) {
        mul_mat_q<type, J, false>
            <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                weight, activation, nullptr, nullptr, out, nullptr, nullptr, blocks_fd, rows,
                columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0, one_fd,
                one_fd, 0, 0, 0, ntx_fd);
        check(cudaGetLastError(), "matrix product launch");
        return;
    }
    if (launch.fixup && fixup == nullptr) {
        throw std::invalid_argument("gguf matrix product: this launch needs a fixup plane");
    }
    mul_mat_q<type, J, false><<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
        weight, activation, nullptr, nullptr, out, launch.fixup ? fixup : nullptr, nullptr,
        blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0,
        one_fd, one_fd, 0, 0, 0, ntx_fd);
    check(cudaGetLastError(), "matrix product launch");
    if (!launch.fixup) { return; }
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, 1);
    mul_mat_q_stream_k_fixup<type, J, false><<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, fixup, blocks_fd, rows, columns, stride, one_fd, 0, one_fd, 0,
        ntx_fd);
    check(cudaGetLastError(), "matrix fixup launch");
}

template <ggml_type type>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, float* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    if (k % 256 != 0 || rows % 128 != 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        throw std::invalid_argument("gguf matrix product: unsupported geometry");
    }
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    const auto* x             = static_cast<const char*>(weight);
    const auto* y             = static_cast<const int*>(activation);
    auto* f                   = static_cast<float*>(fixup);
    const int stride_row      = static_cast<int>(row_bytes / block_bytes);
    switch (launch.J) {
    case 16:
        launch_matrix<type, 16>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    case 32:
        launch_matrix<type, 32>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    case 64:
        launch_matrix<type, 64>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    case 128:
        launch_matrix<type, 128>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("gguf matrix product: unexpected column tile");
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_MATRIX_INSTANCE(TYPE)                                                          \
    template std::size_t ninfer::ops::gguf::detail::matrix_fixup_bytes_impl<TYPE>(int, int);      \
    template void ninfer::ops::gguf::detail::matrix_product_impl<TYPE>(                           \
        const void*, std::int64_t, int, int, const void*, int, float*, std::int64_t, void*,        \
        cudaStream_t)
