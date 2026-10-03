#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/q8/q8_shapes.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using ShapeSelector = Q8Launch (*)(std::int32_t);

struct ShapeEntry {
    std::int32_t n;
    std::int32_t k;
    ShapeSelector select;
    // The unified-template table; shapes without one keep the legacy routes everywhere.
    ShapeSelector unified;
};

template <class Geometry>
constexpr ShapeEntry shape(ShapeSelector select, ShapeSelector unified = nullptr) {
    return {Geometry::kOutputRows, Geometry::kInputRows, select, unified};
}

constexpr std::array kShapes{
    shape<Q8N1024K2048>(select_q8_n1024_k2048, select_q8_n1024_k2048_unified),
    shape<Q8N1024K5120>(select_q8_n1024_k5120, select_q8_n1024_k5120_unified),
    shape<Q8N2048K4096>(select_q8_n2048_k4096, select_q8_n2048_k4096_unified),
    shape<Q8N2048K4608>(select_q8_n2048_k4608, select_q8_n2048_k4608_unified),
    shape<Q8N2048K16384>(select_q8_n2048_k16384, select_q8_n2048_k16384_unified),
    shape<Q8N4608K4608>(select_q8_n4608_k4608, select_q8_n4608_k4608_unified),
    shape<Q8N5120K4608>(select_q8_n5120_k4608, select_q8_n5120_k4608_unified),
    shape<Q8N5120K6144>(select_q8_n5120_k6144, select_q8_n5120_k6144_unified),
    shape<Q8N5120K10240>(select_q8_n5120_k10240, select_q8_n5120_k10240_unified),
    shape<Q8N5120K17408>(select_q8_n5120_k17408, select_q8_n5120_k17408_unified),
    shape<Q8N5120K25600>(select_q8_n5120_k25600, select_q8_n5120_k25600_unified),
    shape<Q8N6144K5120>(select_q8_n6144_k5120, select_q8_n6144_k5120_unified),
    shape<Q8N9216K2048>(select_q8_n9216_k2048, select_q8_n9216_k2048_unified),
    shape<Q8N12288K2048>(select_q8_n12288_k2048, select_q8_n12288_k2048_unified),
    shape<Q8N14336K5120>(select_q8_n14336_k5120, select_q8_n14336_k5120_unified),
    shape<Q8N17408K5120>(select_q8_n17408_k5120),
    shape<Q8N34816K5120>(select_q8_n34816_k5120, select_q8_n34816_k5120_unified),
    shape<Q8N248320K5120>(select_q8_n248320_k5120, select_q8_n248320_k5120_unified),
};
} // namespace

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n != n || entry.k != k) continue;
        if (entry.unified != nullptr &&
            linear_route_table(LinearRouteFamily::Q8, t) == LinearRouteTable::Unified) {
            return entry.unified(t);
        }
        return entry.select(t);
    }
    throw std::invalid_argument("q8 linear: unsupported shape");
}

Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q8 linear: unsupported policy");
    return select_q8_a16_launch(n, k, t);
}

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q8_launch(w.n, w.k, x.ne[1], policy)(x, w, out, stream);
}

} // namespace ninfer::ops::detail
