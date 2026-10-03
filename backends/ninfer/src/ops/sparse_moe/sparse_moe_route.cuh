#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

inline constexpr int kSparseMoeExperts = 256;
inline constexpr int kSparseMoeTopK    = 8;

struct SparseMoeRankedValue {
    float value;
    int id;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    // Bitwise, not short-circuit. `||` and `&&` are sequence points, and nvcc is free to lower them
    // to a branch; on d2's single warp nothing hides a data-dependent branch, so the whole order is
    // evaluated unconditionally and combined as a bool. Same comparisons, same total order.
    return (a.value > b.value) | ((a.value == b.value) & (a.id < b.id));
}

// Sorts one lane's eight candidates descending, branch-free.
//
// This replaced a `while`-loop insertion sort, and the reason is measured rather than stylistic:
// `ncu` attributes **15.25% of d2's active warp cycles to `branch_resolving`** and reports 22.19 of
// 32 threads active per warp, both of which are that loop -- its trip count depends on the data, so
// lanes diverge and the warp pays for every path. (The other 37.68% is `wait`, dependent
// fixed-latency instructions, which needs more warps and is a separate change.)
//
// Batcher's odd-even merge sort for eight elements: 19 compare-exchanges in six dependency levels,
// with every index a compile-time constant, so the whole thing unrolls into predicated selects with
// no branches and no data-dependent trip count. The network is also *shorter* in dependent depth
// than the insertion sort's worst case (six levels against seven).
//
// Semantics are identical, and that is load-bearing. `sparse_moe_ranked_better` is a total order
// over distinct expert ids, so a correct sorting network produces exactly the sequence the
// insertion sort did -- same set, same order -- which is what the callers downstream rely on and
// what `ninfer_sparse_moe_test` checks.
__device__ __forceinline__ void
sparse_moe_sort_eight_descending(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
    static_assert(kSparseMoeTopK == 8, "this network is specifically the eight-element one");
    // A compare-exchange that leaves the better of the pair in `i`. Written with selects rather
    // than a swap under an `if` so nvcc emits predicated moves; a branch here is the whole point.
    const auto compare_exchange = [&run](int i, int j) {
        const SparseMoeRankedValue a = run[i];
        const SparseMoeRankedValue b = run[j];
        const bool ordered           = sparse_moe_ranked_better(a, b);
        run[i]                       = ordered ? a : b;
        run[j]                       = ordered ? b : a;
    };
#pragma unroll
    for (int level = 0; level < 6; ++level) {
        // Batcher odd-even merge sort, n=8. Levels are listed explicitly because the closed form
        // is less legible than the network and this is generated once at compile time anyway.
        switch (level) {
        case 0:
            compare_exchange(0, 1); compare_exchange(2, 3);
            compare_exchange(4, 5); compare_exchange(6, 7);
            break;
        case 1:
            compare_exchange(0, 2); compare_exchange(1, 3);
            compare_exchange(4, 6); compare_exchange(5, 7);
            break;
        case 2:
            compare_exchange(1, 2); compare_exchange(5, 6);
            break;
        case 3:
            compare_exchange(0, 4); compare_exchange(1, 5);
            compare_exchange(2, 6); compare_exchange(3, 7);
            break;
        case 4:
            compare_exchange(2, 4); compare_exchange(3, 5);
            break;
        default:
            compare_exchange(1, 2); compare_exchange(3, 4); compare_exchange(5, 6);
            break;
        }
    }
}

// Merges the descending run of each lane with the run of its xor partner and keeps the better
// half. The eight exchanges of a step do not depend on each other, so the warp reaches the
// warp-wide top-8 in five merge steps instead of eight dependent reduction rounds. Ranking is a
// total order over distinct expert ids, so the selected set and its order are the same as the
// ones any other correct selection produces.
__device__ __forceinline__ void
sparse_moe_merge_ranked_runs(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        SparseMoeRankedValue merged[kSparseMoeTopK];
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            const SparseMoeRankedValue mirror = run[kSparseMoeTopK - 1 - rank];
            SparseMoeRankedValue other;
            other.value  = __shfl_xor_sync(kFullWarpMask, mirror.value, partner);
            other.id     = __shfl_xor_sync(kFullWarpMask, mirror.id, partner);
            merged[rank] = sparse_moe_ranked_better(run[rank], other) ? run[rank] : other;
        }
        // The kept half is bitonic; three compare-exchange stages restore descending order.
        //
        // Written as selects, the same way sparse_moe_sort_eight_descending's compare_exchange is.
        // This used to be `if (!better) { swap }`: a data-dependent branch per exchange, 12 per
        // merge step and 60 across the five, on a kernel with one warp -- so no other warp ever
        // covers a branch while it resolves. The sort was made branch-free for exactly this reason
        // in #88; the merge that consumes its output never was. `(rank & stride) != 0` is still a
        // plain `continue` because rank and stride are both compile-time after the unroll.
#pragma unroll
        for (int stride = kSparseMoeTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                if ((rank & stride) != 0) { continue; }
                const int partner_rank        = rank | stride;
                const SparseMoeRankedValue a  = merged[rank];
                const SparseMoeRankedValue b  = merged[partner_rank];
                const bool keep               = sparse_moe_ranked_better(a, b);
                merged[rank]                  = keep ? a : b;
                merged[partner_rank]          = keep ? b : a;
            }
        }
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) { run[rank] = merged[rank]; }
    }
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kSparseMoeTopK];
#pragma unroll
    for (int item = 0; item < kSparseMoeTopK; ++item) {
        const int id = lane + item * 32;
        local[item]  = {scores[id], id};
    }
    sparse_moe_sort_eight_descending(local);
    sparse_moe_merge_ranked_runs(local);

    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            ids[rank]             = local[rank].id;
            selected_logits[rank] = local[rank].value;
        }
    }
    __syncwarp();

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(scores[kSparseMoeExperts]); }
}

} // namespace ninfer::ops::detail
