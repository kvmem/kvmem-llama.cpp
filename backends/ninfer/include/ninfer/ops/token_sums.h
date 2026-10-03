#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops {
// Sum represented BF16 values over token rows, producing FP32 statistics.
// input is contiguous [channels,T], origin is an I32 scalar containing the
// nonnegative original position of row zero, range is I32 [begin,end).
// Only rows whose original position lies in range contribute. If block_tokens
// is positive, output bucket j corresponds to floor(origin/block_tokens)+j;
// output is [channels,ceil((T+block_tokens-1)/block_tokens)], including zero
// padding buckets. block_tokens==0 produces one sum over the entire range.
// Outputs are overwritten, not accumulated. All buffers must be disjoint;
// origin/range contents can change between graph replays. No persistent state,
// frontier update, normalization, quantization, allocation or host copy occurs.
// Numerical reference: naive FP64 sums of the represented BF16 inputs.
void token_sums(const Tensor& input, const Tensor& origin, const Tensor& range,
                std::uint32_t block_tokens, Tensor& output, cudaStream_t stream);
} // namespace ninfer::ops
