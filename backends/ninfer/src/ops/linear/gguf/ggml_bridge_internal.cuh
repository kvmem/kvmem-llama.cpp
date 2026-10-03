#pragma once

// Shared by the bridge's translation units only: the vendored ggml headers and the device facts the
// matrix kernel's host side needs.

#include "ggml_bridge.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::gguf::detail {

struct DeviceFacts {
    int cc         = 0; // ggml's compute capability: 100 * major + 10 * minor
    int sm_count   = 0;
    std::size_t shared_per_block_optin = 0;
};

[[nodiscard]] const DeviceFacts& device_facts();

inline void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gguf ") + what + ": " + cudaGetErrorString(status));
    }
}

[[nodiscard]] inline ggml_type to_ggml(GgmlType type) { return static_cast<ggml_type>(type); }

// Column tiles instantiated for the matrix kernel. A product is launched with the smallest number of
// column tiles these allow, which is where ggml's own selection lands for them as well.
inline constexpr int kMatrixColumnTiles[] = {16, 32, 64, 128};

// The matrix kernel's activation carries one block_q8_1_mmq per 128 values per column; ggml over-reads
// up to one widest column tile past the end, so the plane keeps that much slack.
inline constexpr std::size_t kMatrixActivationSlack = 128 * sizeof(block_q8_1_mmq);

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns);

// Where and how the vector kernel (ggml_bridge_vec.cuh) writes its rows.
struct VecStore {
    __nv_bfloat16* bf16;
    float* f32;
    std::int64_t column_stride;
    Epilogue epilogue;
    const float* gate;
    std::int64_t gate_column_stride;
};

struct VecArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows; // output rows; a fused [gate; up] parent holds 2 * rows
    int k;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    VecStore out;
};

// Byte offset of the scales inside a planar vector activation of `columns` x `k`.
[[nodiscard]] constexpr std::size_t vec_scales_offset(int k, int columns) {
    return (std::size_t(k) * columns + 15) / 16 * 16;
}

template <ggml_type type>
void vec_launch(const VecArgs& args, int columns, bool fused, cudaStream_t stream);

__device__ __forceinline__ float vec_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ void vec_store(const VecStore& out, int row, int column, float value) {
    const std::int64_t at = column * out.column_stride + row;
    if (out.epilogue == Epilogue::GateProduct) {
        value *= vec_silu(out.gate[column * out.gate_column_stride + row]);
    }
    if (out.f32 != nullptr) {
        out.f32[at] = out.epilogue == Epilogue::Accumulate ? out.f32[at] + value : value;
        return;
    }
    if (out.epilogue == Epilogue::Accumulate) { value += __bfloat162float(out.bf16[at]); }
    out.bf16[at] = __float2bfloat16(value);
}

template <ggml_type type>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, float* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream);

} // namespace ninfer::ops::gguf::detail
