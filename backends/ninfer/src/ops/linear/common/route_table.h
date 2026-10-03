#pragma once

#include "ninfer/ops/linear.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ninfer::ops::detail {

// The route tables a pure Linear shape follows. The unified-template tables were measured on an
// RTX 5090; the legacy tables are this line's routes from before those templates, over the
// kernels they were tuned with.
enum class LinearRouteTable : std::uint8_t {
    Legacy,
    Unified,
};

enum class LinearRouteFamily : std::uint8_t {
    Q4,
    Q5,
    Q6,
    Q8,
    Fp8,
    Nvfp4,
    Bf16,
};

// The table for one call: by default the unified table only inside the width bands where it was
// measured faster on this device's class, and the legacy table everywhere else;
// NINFER_LINEAR_ROUTES=legacy|unified takes one table for every width.
[[nodiscard]] LinearRouteTable linear_route_table(LinearRouteFamily family, std::int32_t t);

// The launches a fused projection Op (a Linear with its consumer fused into the epilogue) takes at
// one width: its own, or upstream's move of the same routes onto the unified templates where the
// device profile's `key` entry names "unified" for the width. NINFER_LINEAR_ROUTES and
// force_linear_route_table() force one table here too.
[[nodiscard]] LinearRouteTable fused_route_table(std::string_view key, std::int32_t width);

// A workspace query over [min_width, max_width] when each width runs on the table `table_of` gives
// it: the largest of the tables' own queries (`query(table, first, last)`) over the runs of widths
// each one serves, so that an interval reserves what its widths' routes use and no more.
template <class TableOf, class Query>
[[nodiscard]] std::size_t capacity_by_table(std::int32_t min_width, std::int32_t max_width,
                                            TableOf&& table_of, Query&& query) {
    std::size_t capacity   = 0;
    std::int32_t first     = min_width;
    LinearRouteTable table = table_of(min_width);
    for (std::int32_t width = min_width + 1; width <= max_width; ++width) {
        const LinearRouteTable next = table_of(width);
        if (next == table) { continue; }
        capacity = std::max(capacity, query(table, first, width - 1));
        first    = width;
        table    = next;
    }
    return std::max(capacity, query(table, first, max_width));
}

// Upstream's fused plans predate the integer-A8 policies: some reject them, others read them as
// admitting A4. This line's FP8 and NVFP4 plans read them as A16Only, and a call forwarded to the
// unified table carries that reading.
[[nodiscard]] constexpr LinearPolicy unified_policy(LinearPolicy policy) noexcept {
    return allows_a8_int(policy) ? LinearPolicy::A16Only : policy;
}

// Tests run each shape under both tables: a forced table wins over the environment and the device
// default until it is cleared with nullopt. Not for use while other threads launch Linear Ops.
void force_linear_route_table(std::optional<LinearRouteTable> table);

} // namespace ninfer::ops::detail
