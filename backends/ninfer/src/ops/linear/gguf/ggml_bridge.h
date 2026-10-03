#pragma once

// Plain entry points into the vendored ggml CUDA kernels (third_party/ggml-quants). The ggml headers
// define their own CUDA_CHECK and friends, so only the bridge's own translation unit includes them;
// everything here is raw pointers and cudaStream_t.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::gguf {

// The block type's id in a GGUF tensor directory (ggml_type).
enum class GgmlType : int {
    Q8_0    = 8,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    IQ1_M   = 29,
};

struct BlockShape {
    int elements = 0;
    int bytes    = 0;
};

[[nodiscard]] BlockShape block_shape(GgmlType type);

// Streaming multiprocessors of the current device (cached per device).
[[nodiscard]] int multiprocessor_count();

// Whether the integer tensor-core prefill kernel exists for the type. IQ1_M has none (nor does it in
// llama.cpp); its wide products dequantize and take a BF16 GEMM.
[[nodiscard]] bool has_matrix_kernel(GgmlType type);

// Columns the vector kernel takes in one launch; wider products take the matrix kernel.
inline constexpr int kMaxVectorColumns = 8;

// Activation for the vector kernel: q8_1 blocks of 32 values, [columns][k / 32].
[[nodiscard]] std::size_t vector_activation_bytes(int k, int columns);
// `input_columns` (optional, [k]) makes element c of each column x[input_columns[c]].
void quantize_vector_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream);

enum class Epilogue : int {
    Store,       // out = y
    Accumulate,  // out = out + y
    GateProduct, // out = silu(gate) * y, gate read from a float plane
};

struct VectorOutput {
    __nv_bfloat16* bf16 = nullptr; // exactly one of bf16 / f32
    float* f32          = nullptr;
    std::int64_t column_stride = 0; // elements between consecutive columns
    Epilogue epilogue          = Epilogue::Store;
    const float* gate          = nullptr; // GateProduct only
    std::int64_t gate_column_stride = 0;
};

// out[r, c] <- epilogue(sum_k W[r, k] x[k, c]) for rows [0, rows) and columns [0, columns),
// columns <= kMaxVectorColumns. `weight` points at row 0; rows are `row_bytes` apart.
void vector_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream);

// out[r, c] = silu(gate_r . x_c) * (up_r . x_c) where gate row r and up row r are rows r and
// rows + r of one parent of this type.
void vector_swiglu(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                   const void* activation, int columns, __nv_bfloat16* out,
                   std::int64_t out_column_stride, cudaStream_t stream);

// Activation for the matrix kernel, laid out for the type's scale/sum schedule. Types that share a
// schedule share an activation, see matrix_activation_layout().
[[nodiscard]] int matrix_activation_layout(GgmlType type);
[[nodiscard]] std::size_t matrix_activation_bytes(int k, int columns);
void quantize_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream);

// Bytes of the stream-k fixup plane the matrix kernel may need for this product on the current
// device (zero when the tiles divide evenly across the machine).
[[nodiscard]] std::size_t matrix_fixup_bytes(GgmlType type, int rows, int columns);

// out[c * out_column_stride + r] = sum_k W[r, k] x[k, c] in FP32.
void matrix_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, float* out,
                    std::int64_t out_column_stride, void* fixup, cudaStream_t stream);

// out <- epilogue(in) for an FP32 [rows, columns] plane whose columns are `in_column_stride` apart.
void store_plane(const float* in, std::int64_t in_column_stride, int rows, int columns,
                 const VectorOutput& out, cudaStream_t stream);

// out[c * k + i] = x[c * k + input_columns[i]] for each of `columns` BF16 columns of width k.
void gather_columns(const __nv_bfloat16* x, int k, int columns, const std::int32_t* input_columns,
                    __nv_bfloat16* out, cudaStream_t stream);

// Rows per pass of dequantized_product for a scratch of `scratch_bytes`.
[[nodiscard]] int dequantized_rows_per_pass(int k, std::size_t scratch_bytes);

// The same product as matrix_product for a type without an integer kernel: each pass dequantizes
// up to dequantized_rows_per_pass rows into `scratch` (BF16) and runs a cuBLAS BF16 GEMM with an
// FP32 accumulator. x is BF16 [k, columns].
void dequantized_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                         const __nv_bfloat16* x, int columns, float* out,
                         std::int64_t out_column_stride, void* scratch, std::size_t scratch_bytes,
                         cudaStream_t stream);

// out[i, :] = dequantize(W[row_ids ? row_ids[i] : i, :]) as BF16, rows `out_row_stride` apart.
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __nv_bfloat16* out,
                     std::int64_t out_row_stride, cudaStream_t stream);
// The same rows in FP32, where every value is exact.
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, float* out,
                     std::int64_t out_row_stride, cudaStream_t stream);

} // namespace ninfer::ops::gguf
