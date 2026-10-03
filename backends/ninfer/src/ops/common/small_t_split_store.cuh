#pragma once

// Epilogue for the Q4 and Q5 small-T MMA kernels (q4_ksplit_mma.cuh, q5_small_t_mma.cuh) when one
// weight matrix feeds two column-major outputs: rows [0, split) go to `head`, rows [split, rows)
// to `tail`, each with its own leading dimension. split == rows makes it a plain strided store.
// A lane stores v.x = (row, col0), v.y = (row, col0 + 1), v.z = (row + 8, col0) and
// v.w = (row + 8, col0 + 1), skipping columns at or past `columns`.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct SmallTSplitStore {
    __nv_bfloat16* head;
    __nv_bfloat16* tail;
    std::int32_t head_ld;
    std::int32_t tail_ld;
    std::int32_t split;
    int columns;

    __device__ __forceinline__ void put(int row, int col, float value) const {
        if (row < split) {
            head[static_cast<std::int64_t>(col) * head_ld + row] = __float2bfloat16_rn(value);
        } else {
            tail[static_cast<std::int64_t>(col) * tail_ld + (row - split)] =
                __float2bfloat16_rn(value);
        }
    }

    __device__ __forceinline__ void store(int row, int col0, float4 v) const {
        if (col0 < columns) {
            put(row, col0, v.x);
            put(row + 8, col0, v.z);
        }
        if (col0 + 1 < columns) {
            put(row, col0 + 1, v.y);
            put(row + 8, col0 + 1, v.w);
        }
    }

    // The Q4 kernel's epilogue call carries its compile-time column tile.
    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 v) const {
        store(row, col0, v);
    }
};

} // namespace ninfer::ops::detail
