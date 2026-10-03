#pragma once
#include "core/tensor.h"

namespace ninfer::models::qwen3_5::execution {
// Stable Program-owned buffers, borrowed by eager execution and graph capture.
// Key sums: [key_width, buckets, full_attention_layers]; query sums:
// [query_width, 1, full_attention_layers]. Ranges: [K begin,end,Q begin,end].
// Only normalized, represented BF16 values BEFORE RoPE enter these reductions.
struct MemoryStatistics {
    Tensor key_sums;
    Tensor query_sums;
    Tensor ranges;
    // Prefill aggregates complete logical blocks; speculative verification keeps
    // one column per candidate until the Frontend commits an exact prefix.
    std::uint32_t block_tokens = 64;
    Tensor candidate_origin;
    Tensor prefill_origin;
};
} // namespace ninfer::models::qwen3_5::execution
