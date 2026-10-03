#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n34816_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q6_a16_simt_r8_t4_cg;
    if (tokens <= 5) return launch_q6_a16_simt_r8_t5_cg;
    if (tokens <= 6) return launch_q6_a16_simt_r8_t6_cg;
    if (tokens <= 7) return launch_q6_a16_simt_r8_t7_cg;
    if (tokens <= 16) return launch_q6_a16_mma_r32_t16_k256;
    if (tokens <= 24) return launch_q6_a16_mma_r32_t24_k256;
    if (tokens <= 32) return launch_q6_a16_mma_r32_t32_k256;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t64_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
