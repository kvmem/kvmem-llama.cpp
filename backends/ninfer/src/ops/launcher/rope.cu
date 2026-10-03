// ninfer::ops - rope launcher: private token-count tuning and generic fallback.
#include "ops/launcher/rope.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/rope.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kLargeBlock               = 256;
constexpr int kFullChunkBlock           = 192;
constexpr int kSmallBlock               = 128;
constexpr int kDefaultChunkTargetTokens = 1024;
// RTX 5090 has 170 SMs and admits six of these 256-thread CTAs per SM.
constexpr int kLargeBlockWaveCapacity = 1020;

template <RopeKernelMode Mode>
inline constexpr bool kTextMode =
    Mode == RopeKernelMode::Text1D || Mode == RopeKernelMode::TextMrope ||
    Mode == RopeKernelMode::DflashText1D;

std::int64_t token_stride(const Tensor* tensor) {
    return tensor == nullptr ? 0 : tensor->nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

bool bf16x2_aligned(const Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data) & (alignof(__nv_bfloat162) - 1)) == 0 &&
           tensor.nb[2] % static_cast<std::int64_t>(alignof(__nv_bfloat162)) == 0;
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed_block(const Tensor& positions, Tensor* q, Tensor* k, int block,
                        cudaStream_t stream) {
    const int tokens = positions.ne[0];
    rope_fixed_kernel<Mode, QHeads, KHeads><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k));
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
void launch_fixed(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    const int tokens = positions.ne[0];
    int block        = kSmallBlock;
    if constexpr (kTextMode<Mode>) {
        if (tokens <= 6) {
            block = (QHeads + KHeads) * 32;
        } else if (tokens <= kLargeBlockWaveCapacity) {
            block = kLargeBlock;
        } else if (tokens <= kDefaultChunkTargetTokens) {
            block = kFullChunkBlock;
        }
        const int head_warps = (QHeads + KHeads) * 32;
        if (block > head_warps) { block = head_warps; }
        if (block > 1024) { block = 1024; }
    }
    launch_fixed_block<Mode, QHeads, KHeads>(positions, q, k, block, stream);
}

template <int HeadsPerBlock, int QHeads, int KHeads>
void launch_dflash_split(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream) {
    constexpr int kGroups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock <= 2 ? 64 : HeadsPerBlock * 32;
    const int tokens      = positions.ne[0];
    rope_fixed_split_kernel<RopeKernelMode::DflashText1D, QHeads, KHeads, HeadsPerBlock>
        <<<tokens * kGroups, kBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
            k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
            token_stride(k));
}

bool launch_fixed_pair(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                       cudaStream_t stream) {
    if (!bf16x2_aligned(q) || !bf16x2_aligned(k)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && q.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F && q.ne[1] == 32 &&
        k.ne[1] == 8) {
        const int tokens = positions.ne[0];
        if (tokens <= 16) {
            launch_dflash_split<5, 32, 8>(positions, &q, &k, stream);
        } else if (tokens <= 400) {
            launch_dflash_split<8, 32, 8>(positions, &q, &k, stream);
        } else {
            launch_fixed_block<RopeKernelMode::DflashText1D, 32, 8>(positions, &q, &k, 160, stream);
        }
        return true;
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (q.ne[1] == 24 && k.ne[1] == 4) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 24, 4>(positions, &q, &k, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 24, 4>(positions, &q, &k, stream);
                return true;
            }
        }
        if (q.ne[1] == 16 && k.ne[1] == 2) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 16, 2>(positions, &q, &k, stream);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 16, 2>(positions, &q, &k, stream);
                return true;
            }
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && q.ne[1] == 16 && k.ne[1] == 16) {
        launch_fixed<RopeKernelMode::Vision2D, 16, 16>(positions, &q, &k, stream);
        return true;
    }
    return false;
}

template <RopeKernelMode Mode, int Heads>
void launch_fixed_single(const Tensor& positions, Tensor& x, cudaStream_t stream) {
    launch_fixed<Mode, Heads, 0>(positions, &x, nullptr, stream);
}

template <int Heads>
bool launch_text_single(const Tensor& positions, int axes, Tensor& x, cudaStream_t stream) {
    if (x.ne[1] != Heads) { return false; }
    if (axes == 1) {
        launch_fixed_single<RopeKernelMode::Text1D, Heads>(positions, x, stream);
        return true;
    }
    if (axes == 3) {
        launch_fixed_single<RopeKernelMode::TextMrope, Heads>(positions, x, stream);
        return true;
    }
    return false;
}

bool launch_fixed_single_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                                  cudaStream_t stream) {
    if (!bf16x2_aligned(x)) { return false; }
    const int axes = positions.ne[1];
    if (axes == 1 && x.ne[0] == 128 && rotary_dim == 128 && theta == 1.0e7F) {
        if (x.ne[1] == 32) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 32>(positions, x, stream);
            return true;
        }
        if (x.ne[1] == 8) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 8>(positions, x, stream);
            return true;
        }
    }
    if (rotary_dim == 64 && theta == 1.0e7F) {
        if (launch_text_single<24>(positions, axes, x, stream) ||
            launch_text_single<4>(positions, axes, x, stream) ||
            launch_text_single<16>(positions, axes, x, stream) ||
            launch_text_single<2>(positions, axes, x, stream)) {
            return true;
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && x.ne[1] == 16) {
        launch_fixed_single<RopeKernelMode::Vision2D, 16>(positions, x, stream);
        return true;
    }
    return false;
}

RopeYarnKernelTable yarn_table(int rotary_dim, float theta, const RopeYarn& yarn) {
    constexpr double kPi  = 3.14159265358979323846;
    const double dim      = rotary_dim;
    const double log_base = std::log(static_cast<double>(theta));
    const auto correction = [&](double rotations) {
        return dim * std::log(yarn.native_context / (rotations * 2.0 * kPi)) / (2.0 * log_base);
    };
    const double lo = std::max(std::floor(correction(32.0)), 0.0);
    double hi       = std::min(std::ceil(correction(1.0)), dim - 1.0);
    if (hi == lo) { hi += 0.001; }
    const bool scaled = yarn.factor > 1.0F;
    RopeYarnKernelTable table{};
    for (int pair = 0; pair < rotary_dim / 2; ++pair) {
        const double frequency = std::pow(static_cast<double>(theta), -2.0 * pair / dim);
        const double ramp      = scaled ? std::clamp((pair - lo) / (hi - lo), 0.0, 1.0) : 0.0;
        table.inverse_frequency[pair] =
            static_cast<float>(frequency * ((1.0 - ramp) + ramp / (scaled ? yarn.factor : 1.0F)));
    }
    table.attention_factor  = scaled ? static_cast<float>(0.1 * std::log(yarn.factor) + 1.0) : 1.0F;
    const bool interpolated = yarn.interpolation_factor > 1.0F;
    table.interpolation_threshold = interpolated ? static_cast<float>(yarn.interpolation_threshold)
                                                 : std::numeric_limits<float>::max();
    table.interpolation_scale     = interpolated ? 1.0F / yarn.interpolation_factor : 1.0F;
    return table;
}

template <int QHeads, int KHeads, bool Mrope>
void launch_yarn_block(const Tensor& positions, Tensor* q, Tensor* k,
                       const RopeYarnKernelTable& table, cudaStream_t stream) {
    const int tokens = positions.ne[0];
    int block        = (QHeads + KHeads) * 32;
    if (block > kLargeBlock) { block = kLargeBlock; }
    rope_yarn_kernel<QHeads, KHeads, Mrope><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k), table);
}

template <int QHeads, int KHeads>
bool launch_yarn_heads(const Tensor& positions, Tensor* q, Tensor* k,
                       const RopeYarnKernelTable& table, cudaStream_t stream) {
    if ((q != nullptr && q->ne[1] != QHeads) || (k != nullptr && k->ne[1] != KHeads)) {
        return false;
    }
    if (positions.ne[1] == 1) {
        launch_yarn_block<QHeads, KHeads, false>(positions, q, k, table, stream);
    } else {
        launch_yarn_block<QHeads, KHeads, true>(positions, q, k, table, stream);
    }
    return true;
}

void launch_yarn(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn,
                 Tensor* q, Tensor* k, cudaStream_t stream) {
    const RopeYarnKernelTable table = yarn_table(rotary_dim, theta, yarn);
    const bool launched =
        k != nullptr ? launch_yarn_heads<24, 4>(positions, q, k, table, stream) ||
                           launch_yarn_heads<16, 2>(positions, q, k, table, stream)
                     : launch_yarn_heads<24, 0>(positions, q, nullptr, table, stream) ||
                           launch_yarn_heads<4, 0>(positions, q, nullptr, table, stream) ||
                           launch_yarn_heads<16, 0>(positions, q, nullptr, table, stream) ||
                           launch_yarn_heads<2, 0>(positions, q, nullptr, table, stream);
    if (!launched) {
        throw std::invalid_argument("rope: YaRN covers the 24/4 and 16/2 Text head geometries");
    }
}

void launch_generic(const Tensor& positions, int rotary_dim, float theta, Tensor* q, Tensor* k,
                    cudaStream_t stream) {
    constexpr int block = 128;
    Tensor& sample      = q != nullptr ? *q : *k;
    const int tokens    = sample.ne[2];
    rope_generic_kernel<<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), positions.ne[1],
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), sample.ne[0], rotary_dim,
        theta, q == nullptr ? 0 : q->ne[1], k == nullptr ? 0 : k->ne[1], tokens, token_stride(q),
        token_stride(k));
}

} // namespace

void rope_launch(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn,
                 Tensor& q, Tensor& k, cudaStream_t stream) {
    if (yarn.active()) {
        launch_yarn(positions, rotary_dim, theta, yarn, &q, &k, stream);
    } else if (!launch_fixed_pair(positions, rotary_dim, theta, q, k, stream)) {
        launch_generic(positions, rotary_dim, theta, &q, &k, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn,
                        Tensor& x, cudaStream_t stream) {
    if (yarn.active()) {
        launch_yarn(positions, rotary_dim, theta, yarn, &x, nullptr, stream);
    } else if (!launch_fixed_single_dispatch(positions, rotary_dim, theta, x, stream)) {
        launch_generic(positions, rotary_dim, theta, &x, nullptr, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
