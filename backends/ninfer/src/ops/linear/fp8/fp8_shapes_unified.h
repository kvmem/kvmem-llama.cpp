#pragma once
#include "ops/linear/fp8/fp8_shapes.h"

// Upstream's FP8 shape tables over the unified templates (shapes/*_unified.cu).
namespace ninfer::ops::detail::unified {
extern const Fp8LinearShape kFp8N14336K5120;
extern const Fp8LinearShape kFp8N16384K5120;
extern const Fp8LinearShape kFp8N34816K5120;
extern const Fp8LinearShape kFp8N5120K6144;
extern const Fp8LinearShape kFp8N5120K17408;
extern const Fp8LinearShape kFp8N248320K5120;
} // namespace ninfer::ops::detail::unified
