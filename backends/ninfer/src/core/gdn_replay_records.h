#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer {

struct GdnReplayRecordSpec {
    std::int32_t layers          = 0;
    std::int32_t record_capacity = 0;
    std::int32_t width           = 0;
    std::int32_t conv_channels   = 0;
    std::int32_t qk_heads        = 0;
    std::int32_t value_heads     = 0;
    std::int32_t key_dim         = 0;
    std::int32_t value_dim       = 0;
};

struct GdnReplayRecordLayout {
    GdnReplayRecordSpec spec;
    TensorRegion conv;
    TensorRegion key;
    TensorRegion value;
    TensorRegion gate;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder& builder,
                                                            const GdnReplayRecordSpec& spec);

struct GdnReplayRecordLayer {
    Tensor conv;  // BF16 [conv_channels, width, rows]
    Tensor key;   // BF16 [key_dim, qk_heads, width, rows]
    Tensor value; // BF16 [value_dim, value_heads, width, rows]
    Tensor gate;  // FP32 [2, value_heads, width, rows], ordered {g, beta}
};

/**
 * Bound non-owning all-layer ReplaySSM transition records.
 *
 * Each layer owns a block of `record_capacity * width` columns in every plane, and a round packs
 * its rows inside that block at the width it records: row `r` of a round of width `w` starts at
 * column `layer * record_capacity * width + r * w`. The object owns no allocation and stores no
 * active-row or valid-prefix metadata.
 */
struct GdnReplayRecords {
    Tensor conv;
    Tensor key;
    Tensor value;
    Tensor gate;
    GdnReplayRecordSpec spec;

    GdnReplayRecords() = default;
    GdnReplayRecords(DeviceSpan backing, const GdnReplayRecordLayout& layout);

    // One layer's records for `rows` active rows of `active_width` columns. Rows are packed at
    // the active width inside the layer's block, so a round that records fewer columns than the
    // planned width must fold with the same width.
    [[nodiscard]] GdnReplayRecordLayer layer(std::int32_t layer, std::int32_t rows,
                                             std::int32_t active_width) const;
};

} // namespace ninfer
