// The eight-element sorting network in `sparse_moe_route.cuh`, run on the device over every input
// it can be handed.
//
// `sparse_moe_sort_eight_descending` replaced a data-dependent insertion sort with Batcher's
// odd-even merge sort -- 19 compare-exchanges in six levels, every index a compile-time constant.
// The correctness argument for that swap is entirely "the network is a correct network": nothing
// downstream re-checks the order, and `sparse_moe_merge_ranked_runs` takes a *descending run* as a
// precondition, so a network that sorts most inputs and not others would not fail loudly. It would
// select slightly wrong experts for some score permutations and change inference output.
//
// A review of that change asserted the schedule was invalid without alternating comparator
// directions -- which is the rule for a *bitonic* network, not for odd-even merge sort, whose
// comparators all point the same way. Rather than argue it, this test settles it by exhaustion,
// and keeps it settled: the network is now pinned against every reordering of eight distinct
// scores and against every tie pattern the id tiebreak has to resolve.
//
// Two case sets, both run through the real device function rather than a host transcription of it:
//
//   * all 8! = 40,320 permutations of eight distinct scores, which is every ordering the sort can
//     be handed when scores differ;
//   * all 3^8 = 6,561 assignments of three score values across eight distinct expert ids, which
//     is where `sparse_moe_ranked_better`'s "equal value, lower id wins" tiebreak decides the
//     order. Three values rather than two because a run can hold both a tie and a strict
//     comparison at once.
//
// One thread per case, so the whole thing is two launches and runs in milliseconds.

#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

using ninfer::ops::detail::kSparseMoeTopK;
using ninfer::ops::detail::SparseMoeRankedValue;

__global__ void sort_cases(const float* values, const int* ids, float* out_values, int* out_ids,
                           int cases) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cases) { return; }
    SparseMoeRankedValue run[kSparseMoeTopK];
    for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
        run[rank].value = values[index * kSparseMoeTopK + rank];
        run[rank].id    = ids[index * kSparseMoeTopK + rank];
    }
    ninfer::ops::detail::sparse_moe_sort_eight_descending(run);
    for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
        out_values[index * kSparseMoeTopK + rank] = run[rank].value;
        out_ids[index * kSparseMoeTopK + rank]    = run[rank].id;
    }
}

// The host mirror of `sparse_moe_ranked_better`, which is the order the network must produce.
bool better(float a_value, int a_id, float b_value, int b_id) {
    return a_value > b_value || (a_value == b_value && a_id < b_id);
}

bool check(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return false; }
    std::printf("FAIL sparse_moe_route_network: %s: %s\n", what, cudaGetErrorString(status));
    return true;
}

struct CaseSet {
    const char* name;
    std::vector<float> values;  // cases * kSparseMoeTopK
    std::vector<int> ids;
};

CaseSet all_permutations_of_distinct_scores() {
    CaseSet set{"8! distinct-score permutations", {}, {}};
    std::vector<int> order(kSparseMoeTopK);
    std::iota(order.begin(), order.end(), 0);
    do {
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            set.values.push_back(static_cast<float>(order[rank]));
            set.ids.push_back(rank);
        }
    } while (std::next_permutation(order.begin(), order.end()));
    return set;
}

CaseSet all_tie_patterns() {
    CaseSet set{"3^8 tie patterns over distinct ids", {}, {}};
    for (int encoded = 0; encoded < 6561; ++encoded) {
        int remaining = encoded;
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            set.values.push_back(static_cast<float>(remaining % 3));
            set.ids.push_back(rank);
            remaining /= 3;
        }
    }
    return set;
}

// Returns the number of cases the network got wrong, and prints the first one in full.
int run_case_set(const CaseSet& set) {
    const int cases = static_cast<int>(set.ids.size()) / kSparseMoeTopK;
    const std::size_t value_bytes = set.values.size() * sizeof(float);
    const std::size_t id_bytes    = set.ids.size() * sizeof(int);

    float* device_values     = nullptr;
    int* device_ids          = nullptr;
    float* device_out_values = nullptr;
    int* device_out_ids      = nullptr;
    if (check(cudaMalloc(&device_values, value_bytes), "cudaMalloc values") ||
        check(cudaMalloc(&device_ids, id_bytes), "cudaMalloc ids") ||
        check(cudaMalloc(&device_out_values, value_bytes), "cudaMalloc out values") ||
        check(cudaMalloc(&device_out_ids, id_bytes), "cudaMalloc out ids")) {
        return -1;
    }
    if (check(cudaMemcpy(device_values, set.values.data(), value_bytes, cudaMemcpyHostToDevice),
              "upload values") ||
        check(cudaMemcpy(device_ids, set.ids.data(), id_bytes, cudaMemcpyHostToDevice),
              "upload ids")) {
        return -1;
    }

    sort_cases<<<(cases + 255) / 256, 256>>>(device_values, device_ids, device_out_values,
                                             device_out_ids, cases);
    if (check(cudaGetLastError(), "launch") || check(cudaDeviceSynchronize(), "synchronize")) {
        return -1;
    }

    std::vector<float> out_values(set.values.size());
    std::vector<int> out_ids(set.ids.size());
    if (check(cudaMemcpy(out_values.data(), device_out_values, value_bytes, cudaMemcpyDeviceToHost),
              "download values") ||
        check(cudaMemcpy(out_ids.data(), device_out_ids, id_bytes, cudaMemcpyDeviceToHost),
              "download ids")) {
        return -1;
    }
    (void)cudaFree(device_values);
    (void)cudaFree(device_ids);
    (void)cudaFree(device_out_values);
    (void)cudaFree(device_out_ids);

    int wrong = 0;
    for (int index = 0; index < cases; ++index) {
        const int base = index * kSparseMoeTopK;
        bool ok        = true;
        // Two conditions, and the second is not redundant. Descending order alone would also be
        // satisfied by a kernel that never ran: freshly-allocated device memory reads back as
        // zeros, and eight equal values with ascending ids *is* descending under this order. So
        // also require the output to be a permutation of the input -- every id present exactly
        // once, carrying the value it went in with. A test that cannot fail is not a test.
        bool seen[kSparseMoeTopK] = {};
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            const int id = out_ids[base + rank];
            if (id < 0 || id >= kSparseMoeTopK || seen[id]) {
                ok = false;
                break;
            }
            seen[id] = true;
            // ids are the position each value entered at, so the input value for `id` is at `id`.
            if (out_values[base + rank] != set.values[base + id]) {
                ok = false;
                break;
            }
        }
        for (int rank = 0; ok && rank + 1 < kSparseMoeTopK; ++rank) {
            if (!better(out_values[base + rank], out_ids[base + rank], out_values[base + rank + 1],
                        out_ids[base + rank + 1])) {
                ok = false;
                break;
            }
        }
        if (ok) { continue; }
        if (wrong == 0) {
            std::printf("FAIL sparse_moe_route_network: %s, first bad case %d\n", set.name, index);
            std::printf("  in :");
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                std::printf(" (%g,%d)", set.values[base + rank], set.ids[base + rank]);
            }
            std::printf("\n  out:");
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                std::printf(" (%g,%d)", out_values[base + rank], out_ids[base + rank]);
            }
            std::printf("\n");
        }
        ++wrong;
    }
    return wrong;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP sparse_moe_route_network: no CUDA device\n");
        return 77;
    }

    const CaseSet sets[] = {all_permutations_of_distinct_scores(), all_tie_patterns()};
    int total_cases      = 0;
    for (const CaseSet& set : sets) {
        const int wrong = run_case_set(set);
        if (wrong < 0) { return 1; }
        if (wrong > 0) {
            std::printf("FAIL sparse_moe_route_network: %s, %d of %d cases not descending\n",
                        set.name, wrong, static_cast<int>(set.ids.size()) / kSparseMoeTopK);
            return 1;
        }
        total_cases += static_cast<int>(set.ids.size()) / kSparseMoeTopK;
    }

    std::printf("OK sparse_moe_route_network (%d cases: %d permutations, %d tie patterns)\n",
                total_cases, 40320, 6561);
    return 0;
}
