// T2G128_F16S head profiles (the full head, or the indexed proposal head): the ternary head is
// projected through the public T2 Linear route into a dense BF16 logits tile, then a producer CTA
// per 128 rows keeps the stable top sixteen of every column through the same warp merge sort the
// fused W8 producer uses, keyed by global token id, and the shared merge kernels finish the
// reduction.

#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/linear_topk/grouped_ksplit_topk.cuh"
#include "ops/linear_topk/linear_topk_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kColumns = kLinearTopKT2ChunkColumns;

struct Storage {
    GroupedKSplitTopKStorage<kColumns, kWarps> topk;
    float tile[kLinearTopK * kColumns];
};

__global__ __launch_bounds__(kThreads) void t2_logits_topk_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t rows, std::int32_t valid_rows,
    std::int32_t columns, std::uint64_t* __restrict__ partial_keys, std::int32_t producer_groups,
    const std::int32_t* __restrict__ row_ids) {
    __shared__ Storage storage;
    grouped_ksplit_topk_initialize(storage.topk);

    const int tid      = static_cast<int>(threadIdx.x);
    const int cta_row0 = static_cast<int>(blockIdx.x) * kLinearTopKGroupedRows;
    for (int slice = 0; slice < kLinearTopKGroupedRows; slice += kLinearTopK) {
        const int row_begin = cta_row0 + slice;
        for (int p = tid; p < kLinearTopK * columns; p += kThreads) {
            const int column = p / kLinearTopK;
            const int r      = p - column * kLinearTopK;
            const int row    = row_begin + r;
            float value      = 0.0f;
            if (row < rows) {
                value = __bfloat162float(logits[static_cast<std::int64_t>(column) * rows + row]);
            }
            storage.tile[r * kColumns + column] = value;
        }
        __syncthreads();
        grouped_ksplit_topk_consume<kColumns, kColumns, kWarps>(
            storage.tile, storage.topk, row_begin, valid_rows, columns, row_ids);
    }
    grouped_ksplit_topk_publish(storage.topk, partial_keys, producer_groups, columns);
}

} // namespace

void linear_topk_logits_launch(const Tensor& logits, std::int32_t valid_rows,
                               const Tensor* row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const std::int32_t columns = logits.ne[1];
    if (columns <= 0 || columns > kColumns ||
        workspace.rows_per_producer != kLinearTopKGroupedRows || workspace.tile_columns != 0 ||
        logits.dtype != DType::BF16) {
        throw std::invalid_argument("linear_topk: invalid materialized-logits launch");
    }
    t2_logits_topk_kernel<<<workspace.producer_groups, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), logits.ne[0], valid_rows, columns,
        static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups,
        row_to_global_ids != nullptr ? static_cast<const std::int32_t*>(row_to_global_ids->data)
                                     : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void linear_topk_t2_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const Tensor* row_to_global_ids, Tensor& logits,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const std::int32_t columns = hidden.ne[1];
    if (columns <= 0 || columns > kColumns || logits.dtype != DType::BF16 ||
        logits.ne[0] != head.n || logits.ne[1] != columns) {
        throw std::invalid_argument("linear_topk: invalid T2 head launch");
    }
    ops::linear(hidden, head, logits, stream);
    linear_topk_logits_launch(logits, valid_rows, row_to_global_ids, workspace, stream);
}

} // namespace ninfer::ops::detail
