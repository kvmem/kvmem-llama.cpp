#pragma once

#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::artifact {

// A device object may be materialized in a narrower grouped format than the artifact stores. The
// plan reserves the target encoding's size, the materializer writes target bytes, and every
// consumer (bindings, weight views, arena accounting) sees only the target format. This trades
// weight precision for device memory; model loading selects it per object from startup options.
//
// The only supported source is Q8_G32_FP16 in the row-split layout; targets are Q4_G64_FP16 and
// Q6_G64_FP16 in the same layout. Logical element addressing (row * columns + column) is the same
// before and after, so bindings over parts of a transcoded object stay valid.
[[nodiscard]] bool row_split_transcode_supported(QType source, QType target) noexcept;

// Requantizes a row-split Q8_G32_FP16 payload to the target format's row-split encoding. Each
// 64-column group takes, of 25 clipping ratios of absmax/qmax in [0.70, 1.18], the fp16 scale whose
// round-to-nearest codes minimise the group's squared error against the Q8 values; plain
// absmax/qmax is one candidate. Groups are independent, so the result does not depend on the
// worker count.
void transcode_row_split(QType target, std::span<const std::uint64_t> shape,
                         std::span<const std::byte> source, std::span<std::byte> destination);

} // namespace ninfer::artifact
