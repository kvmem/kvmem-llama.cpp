#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "ops/linear_add/q8/q8_linear_add_kernels_unified.h"
#include "ops/linear/common/route_table.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule, bool Full>
void launch_tt(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const std::int32_t rows     = w.n;
    const std::int32_t k        = w.k;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), rows};
    q8_rowsplit_gemm_mma_kernel<Schedule, Full, Q8Epilogue::Residual>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output, rows, k, cols, padded_k);
}

template <class Schedule>
void launch_variant(bool full, const Tensor& x, const Weight& w, Tensor& residual_out,
                    cudaStream_t stream) {
    if (full) {
        launch_tt<Schedule, true>(x, w, residual_out, stream);
    } else {
        launch_tt<Schedule, false>(x, w, residual_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_linear_add_mma_r32_c32_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r32_c32_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 32, 32, 16, 4>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c48_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r32_c48_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 48, 32, 16, 4>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c64_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r32_c64_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c80_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r32_c80_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c96_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r32_c96_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c112_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r32_c112_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 112, 32, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r32_c128_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r32_c128_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c64_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r48_c64_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<48, 64, 48, 16, 3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c80_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r48_c80_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<48, 80, 48, 16, 3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c96_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r48_c96_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<48, 96, 48, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c112_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r48_c112_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<48, 112, 48, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r48_c128_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r48_c128_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<48, 128, 48, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c32_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r64_c32_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 32, 64, 16, 3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c48_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r64_c48_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 48, 64, 16, 3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c64_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r64_c64_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c80_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r64_c80_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 80, 64, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c96_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r64_c96_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c112_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r64_c112_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 112, 64, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r64_c128_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r128_c64_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r128_c64_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_r128_c80_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream) {
    if (fused_route_table("unified/q8_linear_add", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_linear_add_mma_r128_c80_launch(x, w, residual_out, stream);
        return;
    }
    using Schedule = Q8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

// Exact-group-scale twins. `with_exact_group_scale` restates the tile above it, so the pair can
// never drift apart in anything but where the Q8G32 scale is applied -- and, where it is spelled,
// MIN_BLOCKS. The exact body holds an FP32 group partial and this tile's row scales, and left at
// the default tile's MIN_BLOCKS ptxas spends the slack on registers and drops a resident block
// (`cuobjdump -res-usage`, sm_86): r32_c64 92 -> 136 registers and 4 blocks -> 3, r32_c96 90 ->
// 136 and 3 -> 2, r64_c64 120 -> 209 and 3 -> 2. The three below are raised to the block count
// the shared-memory footprint allows, which is the one their default twin already runs at; every
// other tile here keeps its own because it was already at that count. None of them spill.
void q8_linear_add_mma_exact_r32_c64_launch(bool full, const Tensor& x, const Weight& w,
                                            Tensor& residual_out, cudaStream_t stream) {
    using Schedule =
        Q8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>::with_exact_group_scale::with_min_blocks<4>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r32_c96_launch(bool full, const Tensor& x, const Weight& w,
                                            Tensor& residual_out, cudaStream_t stream) {
    using Schedule =
        Q8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>::with_exact_group_scale::with_min_blocks<3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r32_c128_launch(bool full, const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r48_c64_launch(bool full, const Tensor& x, const Weight& w,
                                            Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<48, 64, 48, 16, 3>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r48_c96_launch(bool full, const Tensor& x, const Weight& w,
                                            Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<48, 96, 48, 16, 2>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c64_launch(bool full, const Tensor& x, const Weight& w,
                                            Tensor& residual_out, cudaStream_t stream) {
    using Schedule =
        Q8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>::with_exact_group_scale::with_min_blocks<3>;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c96_launch(bool full, const Tensor& x, const Weight& w,
                                            Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c112_launch(bool full, const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 112, 64, 16, 2>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r64_c128_launch(bool full, const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r128_c64_launch(bool full, const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

void q8_linear_add_mma_exact_r128_c80_launch(bool full, const Tensor& x, const Weight& w,
                                             Tensor& residual_out, cudaStream_t stream) {
    using Schedule = Q8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>::with_exact_group_scale;
    launch_variant<Schedule>(full, x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
