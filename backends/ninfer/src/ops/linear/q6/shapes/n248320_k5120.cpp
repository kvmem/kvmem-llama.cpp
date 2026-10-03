#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n248320_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q6_a16_simt_r8_t4_cg;
    if (tokens <= 5) return launch_q6_a16_simt_r8_t5_cg;
    if (tokens <= 6) return launch_q6_a16_simt_r8_t6_cg;
    if (tokens <= 7) return launch_q6_a16_simt_r8_t7_cg;
    if (tokens <= 16) return launch_q6_a16_mma_r64_t16_k128;
    if (tokens <= 24) return launch_q6_a16_mma_r64_t24_k128;
    if (tokens <= 32) return launch_q6_a16_mma_r64_t32_k128;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
