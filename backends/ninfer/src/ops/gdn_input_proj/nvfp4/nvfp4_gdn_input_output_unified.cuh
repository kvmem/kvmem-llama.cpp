#pragma once
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail::unified {
using Nvfp4GdnInputOutput = LinearBf16SegmentedOutput<10240, 6144>;
} // namespace ninfer::ops::detail::unified
