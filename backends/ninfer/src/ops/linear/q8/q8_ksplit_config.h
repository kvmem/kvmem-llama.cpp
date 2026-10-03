#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_geometry.h"

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8KSplitScaleAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class Q8KSplitActivationStage : std::uint8_t {
    // Stage the compile-time column extent; tiled calls zero-fill its inactive columns.
    ActiveOnly,
    // Stage the full MMA tile, including padding beyond the compile-time column extent.
    PaddedZero,
    // Stage only live columns. Inactive MMA columns are discarded by the store epilogue.
    RuntimeActive,
};

template <int KWarps, int TileTokens, int MinBlocksPerSm, Q8KSplitScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8KSplitActivationStage ActivationStage = Q8KSplitActivationStage::ActiveOnly>
struct Q8KSplitSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48 || TileTokens == 56 || TileTokens == 64 ||
                  TileTokens == 72 || TileTokens == 80 || TileTokens == 88);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
    static constexpr int kScaleBytesPerRow  = kGroupK / 16;
};

template <int TileTokens, int ActiveTokens>
using Q8KSplitDefaultSchedule = Q8KSplitSchedule<
#if defined(NINFER_SM8X_COMPAT)
    4, TileTokens, 2,
#else
    8, TileTokens, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
#endif
    (ActiveTokens > 4 ? Q8KSplitScaleAccess::Shared : Q8KSplitScaleAccess::Direct)>;

// sm_86/sm_89 cap a kernel's static shared memory at 49,152 bytes. Q8KSplitSharedStorage stages
// KWarps * TileTokens * 128 bytes of activations beside 1,024 * KWarps bytes of codes, so sixteen
// K warps overflow it from a 16-column tile and eight K warps from a 40-column tile; those are
// compile errors there, not slow paths. Shapes whose SM120 tables use such tiles declare them
// through this alias, which keeps the SM120 schedule and on sm_8x falls back to the four-K-warp,
// two-block schedule the W8 routes used on this card.
template <int KWarps, int TileTokens, int MinBlocksPerSm, Q8KSplitScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8KSplitActivationStage ActivationStage = Q8KSplitActivationStage::ActiveOnly>
using Q8KSplitSm8xFourWarpSchedule =
#if defined(NINFER_SM8X_COMPAT)
    Q8KSplitSchedule<4, TileTokens, 2, ScaleAccess, ActivationCache, WeightCache, ActivationStage>;
#else
    Q8KSplitSchedule<KWarps, TileTokens, MinBlocksPerSm, ScaleAccess, ActivationCache, WeightCache,
                     ActivationStage>;
#endif

} // namespace ninfer::ops::detail
