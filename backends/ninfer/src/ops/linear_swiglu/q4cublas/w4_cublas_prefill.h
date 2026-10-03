#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Prefill by materialising the weight as int8 and handing the GEMM to cuBLAS.
//
// Our own W4A8 mainloop reaches about 147 TOP/s at its best measured shape against cuBLAS's 238 on
// the same card and problem, and the gap is structural rather than a tuning knob: with streaming
// perfectly hidden the compute path alone still costs 2,092 us, so the hand-written ceiling is near
// 1.36x the shipped kernel (tools/w4a8_marlin_probe.cu). Handing the work to cuBLAS instead is
// worth 2.0x, because the dequantise pass is weight-sized while the GEMM is token-sized, so its
// overhead amortises as the prefill chunk grows -- 1.42x / 1.67x / 1.89x / 2.00x at 512 / 1024 /
// 2048 / 4096 tokens (tools/w4_dequant_cublas_probe.cu). It therefore wants a large
// `--prefill-chunk` to pay, and the two settings belong together.
//
// **It is a quality trade and that is why it is opt-in.** cuBLAS reduces over the whole of K, so it
// cannot see a scale per 64 columns: the weight carries one scale per row and the activations one
// per token. The weight half measures 9.7e-3 to 1.1e-2 relative L2 against the exact group-64 values
// on the real 27B artifact, the median per projection kind and 1.5e-2 for the worst tensor
// (tools/w4_row_scale_error.cpp, with the route's own row scale), comparable to the 9e-3 to 2.0e-2 this
// fork already accepts from per-group activation quantisation; the activation half is the trade
// this fork has twice declined, at 1.2e-2 rising to 1.29e-1 on outlier-heavy inputs. Whether the
// two together are affordable is a perplexity question, which is why nothing here is on by default.

// WHAT THE ROUTE COVERS, AND WHAT EACH PART COSTS. Measured at pp4096, chunk 4096, kv int8, against
// a 1,736 tok/s baseline with the route off, and perplexity on the 1M corpus against 4.343155:
//
//   route off                                1,736 tok/s   4.343155
//   MLP and out_proj                         2,673  1.54x  4.346284   +0.072%
//   ... plus attention and GDN projections   2,998  1.73x  4.349944   +0.156%
//
// The second step is `prefill_cublas_projections`, separable because it is a different trade: the
// attention and GDN input projections hold about a fifth of the linear parameters, and covering
// them is worth +12% prefill for +0.084% perplexity. Nearly all of both sides of that come from
// GDN rather than attention -- attention alone measured +2.4%.
//
// UNALIGNED TOKEN COUNTS ARE WHERE THE ROUTE IS FURTHEST AHEAD, which the aligned benchmarks hide.
// The integer route's token tile wants an aligned width and loses about 40% without one; cuBLAS
// does not care. gate_up, TOP/s, integer route against this one:
//
//   tokens     512    640   1000   1536   3000   4095
//   integer   92.5   95.2   53.8   97.5   62.0   59.8
//   cuBLAS   135.1  143.0  160.2  206.1  207.4  210.2
//
// A real prompt is an arbitrary length, so its last chunk is almost always unaligned. The end-to-end
// numbers quoted elsewhere use round prompt sizes and therefore understate this.

// CHOOSING THE PREFILL CHUNK. The route's dequantise pass is weight-sized and its GEMM is
// token-sized, so everything about it is a function of the chunk. Measured end to end on the 27B at
// pp4096 (local 3090, kv int8):
//
//   chunk   baseline   cuBLAS   speedup   workspace   one prefill step
//     512      1,581    1,929     1.22x      268 MiB       ~265 ms
//    1024      1,663    2,235     1.34x      365 MiB       ~460 ms
//    2048      1,724    2,643     1.53x      543 MiB       ~775 ms
//    4096      1,736    2,753     1.59x      661 MiB     ~1,490 ms
//
// The last column is the one serving cares about. The scheduler alternates prefill and decode at
// exactly one `advance_prefill` call (runtime/engine/engine_core.h:2011-2023), which is one chunk,
// so that is how long a concurrent decode lane stalls. 4096 buys the most throughput and stalls
// other lanes for a second and a half; 512 still pays 1.22x and keeps steps short. 2048 is the
// compromise. Workspace matters only near the context ceiling: 196,608 tokens loads at 512 and
// fails at 4096, by 168 MiB.
//
// NOT DONE, and the measurement is why: inverting the prefill loops so the weight is materialised
// once per *window* of chunks rather than once per chunk. It is feasible -- prefill captures no
// CUDA graph, the layer loop is a plain for, and execution/text.cpp already has a chunk loop with
// an unconditional break. But a window has to run inside one `advance_prefill` call, so it inherits
// exactly the scheduling granularity of a chunk that size: it would buy the 4096 throughput at the
// 512 workspace while stalling decode lanes as badly as 4096 does. That leaves it a pure memory
// play, worth ~390 MiB against a restructure of the GDN state fork and capture-frontier handling.
// Not a good trade at this size; revisit if the context ceiling ever binds.

// TRIED AND REJECTED: a larger tile budget. cuBLAS re-reads the materialised weight once per token
// tile, so a budget big enough to make gate_up one tile at 4096 tokens saves ~356 MB of reads. It
// measures 6,822 us against 7,025 at the shipped budget -- inside the ~10% spread the same
// configuration showed across repeats -- and it would take the workspace past 1.1 GiB, because the
// int32 output for one gate_up tile at 4096 tokens is 570 MB on its own. Not worth it.

// TRIED AND REJECTED, 2026-09-18: overlapping the weight dequantise with the GEMM.
//
// The dequantise is memory bound and the GEMM is tensor-core bound, so splitting the weight's rows
// into blocks and materialising block b+1 on a side stream while block b's GEMM runs looks like it
// should hide a cost that is 21% of the call at 1024 tokens. It does not. Measured on gate_up
// against the unmodified serial form (cuBLAS column, us):
//
//                        T=1024   T=4096
//   serial (shipped)      2,243    7,410
//   2 blocks, no overlap  2,010    7,281
//   4 blocks, no overlap  2,255    7,817
//   2 blocks, overlapped  2,070    7,318
//   4 blocks, overlapped  2,173    7,752
//
// Overlapping is worse than not overlapping at every block count, and the block split alone is
// inside the machine's drift over the run -- the untouched A8Int arm moved 3,299 to 3,823 across
// the same sweep. Two plausible reasons, neither worth chasing: cuBLAS already moves enough bytes
// that a concurrent memory-bound kernel takes bandwidth from it rather than filling a gap, and
// splitting one GEMM into four costs more efficiency than the overlap returns.
//
// So the dequantise has to be made *rarer* rather than hidden, which is a prefill-loop question --
// materialise once per window of chunks instead of once per chunk -- not a kernel one.

// The int32 output buffer a token tile needs is n * tile * 4 bytes, so the tile is chosen per shape
// against a budget rather than fixed. A fixed 1024 costs the narrow shapes real throughput -- down
// ran 1.54x against 1.58x for out_proj at the same token count -- because a tall-thin GEMM has less
// parallelism to give cuBLAS than a wide one, and n varies by 7x across the parents here.
inline constexpr std::size_t kCublasTileBudgetBytes = 256u << 20;

[[nodiscard]] std::int32_t cublas_token_tile(std::int32_t rows, std::int32_t tokens);

[[nodiscard]] bool w4_cublas_prefill_supported(const Weight& weight, std::int32_t tokens);

[[nodiscard]] std::size_t w4_cublas_prefill_workspace_capacity_bytes(std::int32_t rows,
                                                                    std::int32_t cols,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

// out[i, t] = silu(C[i, t]) * C[i + out_rows, t], with C the int32 GEMM rescaled by
// row_scale[row] * token_scale[t].
void w4_cublas_swiglu_launch(const Tensor& x, const Weight& gate_up, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream);

// out[i, t] += C[i, t] rescaled. `residual` is read and written in place.
void w4_cublas_add_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                          WorkspaceArena& workspace, cudaStream_t stream);

// The attention and GDN input projections read one parent weight and write several destinations
// from disjoint row ranges of it, so they share an activation quantisation and, per parent, a
// single materialisation. About a fifth of this model's linear parameters live here -- query_key,
// gate_value, qk and value_z -- and they were the part of prefill the route did not reach.
struct CublasProjectionDestination {
    void* data             = nullptr; // BF16, leading dimension `leading`
    std::int32_t row_begin = 0;       // first row of the parent this destination takes
    std::int32_t rows      = 0;       // how many parent rows it takes
    std::int32_t leading   = 0;       // rows in the destination tensor; >= begin + rows
    std::int32_t begin     = 0;       // first row within the destination, for packed outputs
};

struct CublasProjection {
    const Weight* weight = nullptr;
    const CublasProjectionDestination* destinations = nullptr;
    int destination_count = 0;
};

[[nodiscard]] bool w4_cublas_projection_supported(const CublasProjection* parents, int parent_count,
                                                  std::int32_t tokens);

[[nodiscard]] std::size_t w4_cublas_projection_workspace_capacity_bytes(std::int32_t max_rows,
                                                                        std::int32_t cols,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

void w4_cublas_projection_launch(const Tensor& x, const CublasProjection* parents, int parent_count,
                                 WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
