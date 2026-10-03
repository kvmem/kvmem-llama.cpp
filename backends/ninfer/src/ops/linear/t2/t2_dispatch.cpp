#include "ops/linear/t2/t2_dispatch.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// The ternary profile's registered problems (qwen3.8-27b-ternary.md §8): decode (T = 1) through
// the GEMV, the small-T bands through the SIMT route, everything wider through the MMA route.
T2Launch select_t2_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("t2 linear: unsupported shape or T"); }

    // T = 1..16 take the small-T tensor-core kernel: at T = 1 the v2 kernel streams the
    // text-layer weights at 430..660 GB/s against the SIMT GEMV's 260..430 (RTX 3090).
    // NINFER_T2_DECODE=simt keeps the GEMV at T = 1 for comparison.
    static const bool decode_simt = [] {
        const char* value = std::getenv("NINFER_T2_DECODE");
        return value != nullptr && std::string(value) == "simt";
    }();
    // NINFER_T2_SMALLT=v1 keeps the first tensor-core kernel for T = 2..16 (benchmark A/B).
    static const bool small_t_v1 = [] {
        const char* value = std::getenv("NINFER_T2_SMALLT");
        return value != nullptr && std::string(value) == "v1";
    }();
    const auto small_t = [t]() -> T2Launch {
        if (t <= 16) { return small_t_v1 ? launch_t2_small_t_mma : launch_t2_small_t_v2; }
        if (t <= 32) { return launch_t2_mma_r64_c32; }
        if (t <= 64) { return launch_t2_mma_r64_c64; }
        return launch_t2_mma_r64_c128;
    };

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
        case 4096:
        case 6144:
        case 7168:
        case 12288:
        case 14336:
        case 16384:
        case 34816:
            if (t == 1 && decode_simt) { return launch_t2_gemv_r8_w1_k5120; }
            return small_t();
        case 131072:
        case 248320:
            if (t == 1 && decode_simt) { return launch_t2_gemv_r4_w1_word; }
            return small_t();
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t == 1 && decode_simt) { return launch_t2_gemv_r8_w1_k6144; }
            return small_t();
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t == 1 && decode_simt) { return launch_t2_gemv_r8_w1_k17408; }
            return small_t();
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("t2 linear: unsupported shape or T");
}

T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA8Int:
    case LinearPolicy::AllowA8IntDecode:
    case LinearPolicy::AllowPrefillCublas:
        return select_t2_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("t2 linear: unsupported policy");
}

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    // The kernels address row-major records; the panel permutation is a Q4-family device form.
    if (w.layout != QuantLayout::RowSplit || w.qdata == nullptr || w.scales == nullptr) {
        throw std::invalid_argument("t2 linear: requires a row-split weight");
    }
    const T2Launch launch = select_t2_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
