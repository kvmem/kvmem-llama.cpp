#include "ops/linear/t2/t2_small_t_v2.cuh"

#include "core/device.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

enum class V2Rows : std::uint8_t {
    R16,        // KWarps 8, one tile per warp, six CTAs per SM: the narrow widths and T <= 4
    R16Blocks4, // R16 at four CTAs per SM: more registers per thread, wins at T = 5..8
    R32,        // KWarps 4, one tile per warp
    R32K8, // KWarps 8, two tiles per warp: 32 rows over 512-k slabs, the wide outputs at T = 5..8
    R64,   // KWarps 4, two tiles per warp: the heads at T >= 5
};

template <class Schedule>
void launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = w.padded_shape[1];
    const std::int32_t cols = x.ne[1];
    if ((rows % Schedule::kRows) != 0 || (k % Schedule::kSlabK) != 0 || k != x.ne[0] || cols <= 0 ||
        cols > Schedule::kColumns) {
        throw std::invalid_argument("t2 small-T v2: unsupported shape");
    }
    const dim3 grid(static_cast<unsigned>(rows / Schedule::kRows), 1u, 1u);
    t2_small_t_v2_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k,
        cols);
    CUDA_CHECK(cudaGetLastError());
}

template <int ColumnTiles>
void launch_rows(V2Rows rows, const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    switch (rows) {
    case V2Rows::R16:
        return launch<T2SmallTv2Schedule<8, 1, ColumnTiles, 2, 6>>(x, w, out, stream);
    case V2Rows::R16Blocks4:
        return launch<T2SmallTv2Schedule<8, 1, ColumnTiles, 2, 4>>(x, w, out, stream);
    case V2Rows::R32:
        return launch<T2SmallTv2Schedule<4, 1, ColumnTiles, 2, 5>>(x, w, out, stream);
    case V2Rows::R32K8:
        return launch<T2SmallTv2Schedule<8, 2, ColumnTiles, 2, 4>>(x, w, out, stream);
    case V2Rows::R64:
        return launch<T2SmallTv2Schedule<4, 2, ColumnTiles, 2, 4>>(x, w, out, stream);
    }
    throw std::invalid_argument("t2 small-T v2: unknown row schedule");
}

// NINFER_T2_V2=r16|r16b4|r32|r32k8|r64 pins one row schedule for every shape (benchmark sweeps).
const V2Rows* pinned_rows() {
    static const V2Rows* pinned = []() -> const V2Rows* {
        static V2Rows value;
        const char* env = std::getenv("NINFER_T2_V2");
        if (env == nullptr) { return nullptr; }
        const std::string text(env);
        if (text == "r16") {
            value = V2Rows::R16;
            return &value;
        }
        if (text == "r16b4") {
            value = V2Rows::R16Blocks4;
            return &value;
        }
        if (text == "r32") {
            value = V2Rows::R32;
            return &value;
        }
        if (text == "r32k8") {
            value = V2Rows::R32K8;
            return &value;
        }
        if (text == "r64") {
            value = V2Rows::R64;
            return &value;
        }
        throw std::invalid_argument("NINFER_T2_V2 must be r16, r16b4, r32, r32k8 or r64");
    }();
    return pinned;
}

// Measured on the RTX 3090 (ninfer_linear_bench --qtype t2, cold L2). T <= 4: sixteen rows per
// CTA win every text-layer shape (gate_up 662..647 GB/s, down 565..518) and the heads. T = 5..8:
// the wide outputs take 32 rows over 512-k slabs (gate_up 601 against 551), the rest sixteen rows
// at four CTAs per SM (down 489 against 460), the heads 64 rows (713). From T = 9 the two-tile
// column band halves the tensor rate and the wider CTAs win (32 rows for the layers, 64 for the
// heads).
V2Rows rows_for(std::int32_t n, std::int32_t cols) {
    if (const V2Rows* pinned = pinned_rows()) { return *pinned; }
    const bool head = n >= 131072;
    if (cols <= 4) { return V2Rows::R16; }
    if (cols <= 8) {
        if (head) { return V2Rows::R64; }
        return n >= 12288 ? V2Rows::R32K8 : V2Rows::R16Blocks4;
    }
    return head ? V2Rows::R64 : V2Rows::R32;
}

} // namespace

void launch_t2_small_t_v2(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const V2Rows rows = rows_for(out.ne[0], x.ne[1]);
    if (x.ne[1] <= 8) {
        launch_rows<1>(rows, x, w, out, stream);
    } else {
        launch_rows<2>(rows, x, w, out, stream);
    }
}

} // namespace ninfer::ops::detail
