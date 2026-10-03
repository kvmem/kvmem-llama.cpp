#pragma once

namespace ninfer::ops::detail {

// Warp layout of the small-T MMA kernels (q4_ksplit_mma.cuh, q5_small_t_mma.cuh). A CTA is eight
// warps: KWarps of them split each K slab, one 64-k quantization group apiece, and the CTA covers
// 8 / KWarps sixteen-row MMA tiles, one per set of KWarps warps. Every tile reads the same staged
// activation slab, so a CTA stages T columns of activations once per 16 * 8 / KWarps weight rows.
//
// KWarps = 8 (one tile) keeps the most CTAs in flight, which is what a narrow verify wants. At
// T=32 it stages 32 activation columns for every 16 rows: for gate_up that is ~713 MB of L2 reads
// per call against 89 MB of weights. Fewer K warps trade CTA count for that traffic.
//
// TilesPerWarp > 1 gives each warp that many tiles over the same K group, so it loads each
// activation fragment once for all of them, and the CTA covers TilesPerWarp times the rows.
template <int KWarps, int TilesPerWarp = 1>
struct SmallTLayout {
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static_assert(TilesPerWarp == 1 || TilesPerWarp == 2);
    static constexpr int kWarps        = 8;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kKWarps       = KWarps;
    static constexpr int kTilesPerWarp = TilesPerWarp;
    static constexpr int kRowTiles     = kWarps / KWarps * TilesPerWarp;
    static constexpr int kRowsPerCta   = 16 * kRowTiles;
    static constexpr int kGroupK       = 64 * KWarps;
};

} // namespace ninfer::ops::detail
