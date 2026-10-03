// ninfer::ops - hadamard_transform wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"

#include "ops/launcher/hadamard_transform.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kBlock = 1024;

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("hadamard_transform: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("hadamard_transform: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

} // namespace

void silu_mul_hadamard(const Tensor& plane, const Tensor& signs, Tensor& out, cudaStream_t stream) {
    if (plane.dtype != DType::BF16 || signs.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out must be BF16");
    }
    (void)checked_numel(plane, "plane");
    (void)checked_numel(out, "out");
    (void)checked_numel(signs, "signs");
    const std::int32_t width = out.ne[0];
    if (width % kBlock != 0) {
        throw std::invalid_argument("silu_mul_hadamard: out.ne[0] must be a multiple of 1024");
    }
    if (plane.ne[0] != 2 * width || plane.ne[1] != out.ne[1] || plane.ne[2] != out.ne[2] ||
        plane.ne[3] != out.ne[3]) {
        throw std::invalid_argument("silu_mul_hadamard: plane must be [2 * out.ne[0], columns]");
    }
    if (signs.ne[0] != width || signs.ne[1] != 1 || signs.ne[2] != 1 || signs.ne[3] != 1) {
        throw std::invalid_argument("silu_mul_hadamard: signs must be 1-D with ne[0] == out.ne[0]");
    }
    if (!plane.is_contiguous() || !signs.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out must be contiguous");
    }
    if (plane.data == nullptr || signs.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out data must be non-null");
    }
    if (!aligned16(plane.data) || !aligned16(signs.data) || !aligned16(out.data)) {
        throw std::invalid_argument("silu_mul_hadamard: plane/signs/out must be 16-byte aligned");
    }
    const auto* plane_begin = static_cast<const std::uint8_t*>(plane.data);
    const auto* out_begin   = static_cast<const std::uint8_t*>(out.data);
    if (out_begin < plane_begin + plane.bytes() && plane_begin < out_begin + out.bytes()) {
        throw std::invalid_argument("silu_mul_hadamard: out must not overlap the plane");
    }
    if (signs.data == plane.data || signs.data == out.data) {
        throw std::invalid_argument("silu_mul_hadamard: signs must not alias plane or out");
    }
    detail::silu_mul_hadamard_launch(plane, signs, out, stream);
}

void hadamard_transform(const Tensor& x, const Tensor& signs, bool inverse, Tensor& out,
                        cudaStream_t stream) {
    if (x.dtype != DType::BF16 || signs.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("hadamard_transform: x/signs/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    (void)checked_numel(signs, "signs");
    for (int d = 0; d < 4; ++d) {
        if (x.ne[d] != out.ne[d]) {
            throw std::invalid_argument("hadamard_transform: x/out shapes must match");
        }
    }
    const std::int32_t k = x.ne[0];
    if (k % kBlock != 0) {
        throw std::invalid_argument("hadamard_transform: ne[0] must be a multiple of 1024");
    }
    if (signs.ne[0] != k || signs.ne[1] != 1 || signs.ne[2] != 1 || signs.ne[3] != 1) {
        throw std::invalid_argument("hadamard_transform: signs must be 1-D with ne[0] == x.ne[0]");
    }
    if (!x.is_contiguous() || !signs.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("hadamard_transform: x/signs/out must be contiguous");
    }
    if (x.data == nullptr || signs.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("hadamard_transform: x/signs/out data must be non-null");
    }
    if (signs.data == x.data || signs.data == out.data) {
        throw std::invalid_argument("hadamard_transform: signs must not alias x or out");
    }
    if (!aligned16(x.data) || !aligned16(signs.data) || !aligned16(out.data)) {
        throw std::invalid_argument("hadamard_transform: x/signs/out must be 16-byte aligned");
    }
    detail::hadamard_transform_launch(x, signs, inverse, out, stream);
}

namespace {

bool aligned4(const void* pointer) { return (reinterpret_cast<std::uintptr_t>(pointer) & 3U) == 0; }

// The fused producers take a 1-D BF16 sign vector whose width K is a multiple of 1024, and write
// a contiguous, 16-byte aligned BF16 output of whole K-wide columns.
std::int32_t require_rotation(const char* op, const Tensor& signs, const Tensor& out) {
    if (signs.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": signs/out must be BF16");
    }
    (void)checked_numel(signs, "signs");
    const std::int64_t count = checked_numel(out, "out");
    const std::int32_t width = signs.ne[0];
    if (signs.ne[1] != 1 || signs.ne[2] != 1 || signs.ne[3] != 1 || width % kBlock != 0 ||
        count % width != 0) {
        throw std::invalid_argument(std::string(op) +
                                    ": signs must be 1-D, a multiple of 1024 wide, and divide out "
                                    "into whole columns");
    }
    if (!signs.is_contiguous() || !out.is_contiguous() || signs.data == nullptr ||
        out.data == nullptr || !aligned16(signs.data) || !aligned16(out.data)) {
        throw std::invalid_argument(std::string(op) +
                                    ": signs/out must be contiguous, non-null and 16-byte aligned");
    }
    if (signs.data == out.data) {
        throw std::invalid_argument(std::string(op) + ": signs must not alias out");
    }
    return width;
}

Tensor columns_view(const Tensor& tensor, std::int32_t width) {
    return tensor.view({width, static_cast<std::int32_t>(tensor.numel() / width)});
}

} // namespace

void rmsnorm_hadamard(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                      const Tensor& signs, Tensor& out, cudaStream_t stream) {
    const std::int32_t width = require_rotation("rmsnorm_hadamard", signs, out);
    if (x.ne[0] != width) {
        throw std::invalid_argument("rmsnorm_hadamard: x.ne[0] must equal the sign width");
    }
    if (width == detail::kRmsnormHadamardFusedWidth && x.dtype == DType::BF16 &&
        weight.dtype == DType::BF16 && x.is_contiguous() && weight.is_contiguous() &&
        x.numel() == out.numel() && weight.numel() == width && x.data != nullptr &&
        weight.data != nullptr && aligned4(x.data) && aligned4(weight.data) && std::isfinite(eps) &&
        eps > 0.0f) {
        const auto* x_begin   = static_cast<const std::uint8_t*>(x.data);
        const auto* out_begin = static_cast<const std::uint8_t*>(out.data);
        if (out_begin < x_begin + x.bytes() && x_begin < out_begin + out.bytes()) {
            throw std::invalid_argument("rmsnorm_hadamard: out must not overlap x");
        }
        detail::rmsnorm_hadamard_5120_launch(x, weight, eps, unit_offset, signs, out, stream);
        return;
    }
    rmsnorm(x, weight, eps, unit_offset, out, stream);
    Tensor rows = columns_view(out, width);
    hadamard_transform(rows, signs, false, rows, stream);
}

void gated_rmsnorm_hadamard(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                            const Tensor& signs, Tensor& out, cudaStream_t stream) {
    const std::int32_t width = require_rotation("gated_rmsnorm_hadamard", signs, out);
    const std::int32_t d     = x.ne[0];
    if (d <= 0 || width % d != 0) {
        throw std::invalid_argument(
            "gated_rmsnorm_hadamard: the sign width must be a whole number of x rows");
    }
    const std::int32_t heads = width / d;
    if (d == detail::kGatedRmsnormHadamardFusedHeadDim &&
        heads % detail::kGatedRmsnormHadamardFusedRows == 0 && x.dtype == DType::BF16 &&
        z.dtype == DType::BF16 && weight.dtype == DType::BF16 && x.is_contiguous() &&
        z.is_contiguous() && weight.is_contiguous() && x.numel() == out.numel() &&
        z.numel() == out.numel() && weight.numel() == d && x.data != nullptr && z.data != nullptr &&
        weight.data != nullptr && aligned4(x.data) && aligned4(z.data) && aligned4(weight.data) &&
        std::isfinite(eps) && eps > 0.0f) {
        for (const Tensor* input : {&x, &z}) {
            const auto* in_begin  = static_cast<const std::uint8_t*>(input->data);
            const auto* out_begin = static_cast<const std::uint8_t*>(out.data);
            if (out_begin < in_begin + input->bytes() && in_begin < out_begin + out.bytes()) {
                throw std::invalid_argument("gated_rmsnorm_hadamard: out must not overlap x or z");
            }
        }
        detail::gated_rmsnorm_hadamard_d128_launch(x, weight, z, eps, signs, heads, out, stream);
        return;
    }
    Tensor rows = out.view({x.ne[0], x.ne[1], x.ne[2], x.ne[3]});
    gated_rmsnorm(x, weight, z, eps, rows, stream);
    Tensor columns = columns_view(out, width);
    hadamard_transform(columns, signs, false, columns, stream);
}

void sigmoid_mul_hadamard(const Tensor& gate, const Tensor& x, const Tensor& signs, Tensor& out,
                          cudaStream_t stream) {
    (void)require_rotation("sigmoid_mul_hadamard", signs, out);
    if (gate.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("sigmoid_mul_hadamard: gate/x must be BF16");
    }
    if (checked_numel(gate, "gate") != out.numel() || checked_numel(x, "x") != out.numel()) {
        throw std::invalid_argument("sigmoid_mul_hadamard: gate/x/out must have equal sizes");
    }
    if (!gate.is_contiguous() || !x.is_contiguous() || gate.data == nullptr || x.data == nullptr ||
        !aligned16(gate.data) || !aligned16(x.data)) {
        throw std::invalid_argument(
            "sigmoid_mul_hadamard: gate/x must be contiguous, non-null and 16-byte aligned");
    }
    const auto* gate_begin = static_cast<const std::uint8_t*>(gate.data);
    const auto* x_begin    = static_cast<const std::uint8_t*>(x.data);
    const auto* out_begin  = static_cast<const std::uint8_t*>(out.data);
    if (out_begin < gate_begin + gate.bytes() && gate_begin < out_begin + out.bytes()) {
        throw std::invalid_argument("sigmoid_mul_hadamard: out must not overlap gate");
    }
    if (out.data != x.data && out_begin < x_begin + x.bytes() &&
        x_begin < out_begin + out.bytes()) {
        throw std::invalid_argument("sigmoid_mul_hadamard: out must alias x exactly or not at all");
    }
    detail::sigmoid_mul_hadamard_launch(gate, x, signs, out, stream);
}

void embedding_rotated(const Tensor& ids, const Weight& table, const Tensor& signs, Tensor& out,
                       cudaStream_t stream) {
    const std::int32_t width = require_rotation("embedding_rotated", signs, out);
    if (ids.dtype != DType::I32 || !ids.is_contiguous() || ids.data == nullptr || ids.ne[1] != 1 ||
        ids.ne[2] != 1 || ids.ne[3] != 1) {
        throw std::invalid_argument("embedding_rotated: ids must be contiguous I32 [T]");
    }
    if (out.ne[0] != width || out.ne[1] != ids.ne[0] || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("embedding_rotated: out must be [K, T] with K the sign width");
    }
    if (table.qtype != QType::T2_G128_FP16 || table.layout != QuantLayout::RowSplit ||
        table.qdata == nullptr || table.scales == nullptr || table.k != width || table.n <= 0 ||
        table.scale_dtype != DType::FP16 || !aligned4(table.qdata)) {
        throw std::invalid_argument(
            "embedding_rotated: table must be a row-split t2_g128_fp16 [vocab, K] weight");
    }
    if (ids.ne[0] == 0) { return; }
    detail::embedding_rotated_t2_launch(ids, table, signs, out, stream);
}

} // namespace ninfer::ops
