// A Marlin-class W4A8 mainloop over a permuted weight layout, on sm_86.
//
//   C[N,T] = W[N,K] (Q4, group-64 FP16 scales) x X[K,T] (s8, one scale per (token, group))
//   N = 34816, K = 5120 -- the Qwen3.8-27B mlp/gate_up shape.
//
// This is the gate for the whole layout project. The shipped kernel reaches 117 TOP/s at T=1024;
// cuBLAS's int8 GEMM reaches 237.5 on the same card and shape while solving an easier problem
// (tools/int8_gemm_reference_probe.cu). Every knob the shipped structure has is spent -- token tile
// +39%, occupancy +6%, pipeline depth +3.9%, grid swizzle +0.5%, L2 evict-first -0.8%, interleaved
// loads -1.9% (tools/w4a8_rowsplit_probe.cu, which stays as the baseline oracle). What is left is
// structural, and this probe builds all of it at once, because measuring the pieces separately is
// exactly what produced those noise-sized numbers:
//
//   1. the weight permuted so one 8-byte shared load is a lane's whole A fragment for a row,
//   2. LOP3 dequant whose natural output order *is* the MMA's required order,
//   3. a cp.async ring of STAGES buffers with one __syncthreads per stage rather than two per group,
//   4. packed nibbles in shared, which is what frees the shared memory the ring and a wide tile need.
//
// **The tile is the other half of the prize.** Per block a tile streams BM*K/2 bytes of weight and
// BN*K of activations, so over the whole GEMM W traffic is N*T*K/(2*BN) and X traffic is N*T*K/BM.
// At the shipped 64x512, X is 94% of the bytes -- 2,852 MB against 178 MB -- because each of the 544
// row blocks re-reads the activation tile. BM is therefore the dominant lever and it is currently
// pinned low by the registers operand assembly costs:
//
//   BM x BN     W        X        total
//    64 x 512   178 MB   2,852 MB 3,030 MB   (shipped)
//   128 x 256   356 MB   1,426 MB 1,782 MB
//   256 x 256   356 MB     713 MB 1,069 MB
//   256 x 512   178 MB     713 MB   891 MB
//
// ---------------------------------------------------------------------------------------------
// RESULT, 2026-09-18: the layout axis that pays is *across* rows, not within a group.
//
// Marlin's within-group permutation (the original subject of this probe) was worth 1.07x and is
// documented below as a dead end. The thing it missed: our weights are row-major over K, so a
// warp's cp.async for one k-group reads sixteen 32-byte fragments **K/2 = 2,560 bytes apart**, i.e.
// sixteen memory transactions per instruction. PANEL_MAJOR stores each panel of rows' bytes for
// one k-group contiguously, so that same instruction reads one 512-byte run.
//
//   streaming path alone (ABLATE=2)   2,729 us  ->  1,453 us    1.88x
//   full kernel, best configuration   2,901 us  ->  2,479 us    1.26x over the shipped 3,113
//
// Best configuration: 128x128, 256 threads, 2 stages, PANEL=64. Correct at every configuration.
//
// **This re-opens everything the memory floor was masking.** Before, streaming was 2,729 against a
// 1,444 us MMA floor, so every arithmetic idea measured as noise; per-token activation scales were
// worth 1.5%. On the new layout streaming is 1,453 against 1,439, roughly 70% of it hides behind
// the MMAs, and per-token scales are worth **8%** (2,295 us / 159 TOP/s). The remaining stack at
// the best shape: MMA floor 1,439, rescale ~475, shared reads + decode ~178, exposed streaming
// ~449. If the rescale went away and streaming hid fully the kernel would be ~1,620 us / 226 TOP/s
// -- 1.9x the shipped kernel, and past cuBLAS's 1,532 on an easier problem.
//
// **PANEL is free to decouple from BM, and the win saturates at a tiny panel.** At BM=128 the full
// kernel measures 2,470 / 2,474 / 2,488 / 2,487 / 2,479 / 2,487 us for PANEL 64 / 32 / 16 / 8 / 4 /
// 2 -- flat to within 0.7% across a 32x range. Halving the transactions per warp captures the whole
// benefit; past that the compute path binds (compute-only is 2,092) and further coalescing buys
// nothing.
//
// That matters because panel-major is *not* a free permutation engine-wide: it trades the decode
// path for the prefill path. Every a8/MMA prefill consumer improves, but the whole GEMV decode
// family is built on "one warp owns one row and streams its K", which panel-major scatters -- and
// that family is already transaction-bound (q4_linear_swiglu_gemv.cu:278 records DRAM at 50% while
// the memory pipes sat at 81%). A large panel would cost 4x the line fills there. A panel of 4
// does not: the decode block already owns 4 consecutive gate rows and 4 consecutive up rows, so a
// 4-row panel is exactly one 128-byte line that the block fully consumes, and the fetch needs a
// lane remap rather than a restructure. **Pick PANEL to fit the decode block, not the prefill
// tile.**
//
// Two things this probe does not model. The Q5/Q6 high plane and the scale plane are separate
// planes with their own per-row strides, and permuting only the code plane leaves their scattered
// per-row reads in place -- the scale-ring prefetch is BM scattered 16-byte reads, exactly what
// this exists to fix, so there is unclaimed upside there. And row-range slicing is plain pointer
// arithmetic (core/weight_view.cpp:215), valid under panel-major only while row_begin % PANEL == 0;
// every current caller satisfies it and nothing enforces it.
//
// Two implementation notes that cost real time. The panel divide must be hoisted out of the issue
// lambda -- left inside it is loop-invariant arithmetic competing with the accumulators for
// registers, and it costs ~30%. And THREADS can exceed BM*2, so the surplus lanes need a guard;
// without it the wide-tile configurations run off the tile.
//
// Dead ends recorded so nobody repeats them: Marlin's within-group permutation, 1.07x; a grid
// swizzle grouping row-blocks for L2 reuse, 3,375 us against 2,901 (actively worse); more
// occupancy -- 24 warps is slower than 16; half2 scale products underflow (both scales are ~2e-3,
// their product ~6e-6 against fp16's 6.1e-5 smallest normal); hoisting the token scales into
// registers is worth nothing; independent accumulators for the two k-halves cost 24%; 1024-thread
// blocks collapse; 256x128 at 512 threads, the byte-minimal register-legal tile, is worse than
// both at 3,220 us because 70 KB of shared drops it to one block per SM.
//
// Sweep with -DBM_ROWS -DBN_TOKENS -DSTAGES_N -DTHREADS_N -DWARPS_M_N -DABLATE.
//
// Build:
//   nvcc -O3 -arch=sm_86 -allow-unsupported-compiler w4a8_marlin_probe.cu -o w4a8_marlin_probe

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int N = 34816;
constexpr int K = 5120;
#ifndef T_TOKENS
#define T_TOKENS 1024
#endif
constexpr int T = T_TOKENS;

constexpr int GROUP  = 64; // one FP16 scale per 64 codes; the rescale unit
constexpr int GROUPS = K / GROUP;

#ifndef BM_ROWS
#define BM_ROWS 128
#endif
#ifndef BN_TOKENS
#define BN_TOKENS 256
#endif
#ifndef THREADS_N
#define THREADS_N 512
#endif
#ifndef WARPS_M_N
#define WARPS_M_N 4
#endif
#ifndef STAGES_N
#define STAGES_N 3
#endif
// 0 = full kernel, 1 = MMAs and loads but no per-group rescale (int32 accumulate, converted once),
// 2 = streaming only (no MMAs, no shared reads): the G0a floor,
// 3 = compute only: every shared read, decode, MMA and rescale, but no cp.async at all. Reads stale
//     data, so it is numerically meaningless and timing-wise exactly the question -- is our compute
//     at the MMA issue floor, or only arithmetically close to it?
// 4 = compute without the LOP3 decode: raw shared words go straight into the MMA,
// 5 = MMAs and rescale only: no streaming, fragments hoisted out of the loop (no shared reads),
// 6 = MMAs only: no streaming, no shared reads, no rescale -- the pure issue floor of our own
//     instruction stream, which is the number to compare against the 1.28 ms hardware floor,
// 7 = per-token activation scale (Marlin's choice) instead of per-group: the weight scale still
//     changes every 64, so the FMA stays, but the ws*xa product per (m,n) and the two half2float
//     per n disappear into the epilogue,
// 8 = independent accumulators for the two k-halves, summed after the loop, to see whether the
//     back-to-back dependent MMAs into one s[] are stalling the tensor pipe,
// 9 = per-group activation scales in half2. MEASURED WRONG: both scales are ~2e-3 and their
//     product ~6e-6 is subnormal in fp16 (smallest normal 6.1e-5), so it flushes. Kept as a
//     recorded dead end.
// 10 = per-group activation scales kept exactly, but the token scales are hoisted into registers
//     once per group instead of being re-read from shared inside the m loop. Same arithmetic, same
//     quality; the question is how much of ABLATE=7's win was the scale *loads* rather than the
//     scale *maths*.
#ifndef ABLATE
#define ABLATE 0
#endif
// Row-blocks per rasterisation group; 0 = the plain x-major 2D grid.
#ifndef SWIZ
#define SWIZ 0
#endif
// 1 = panel-major weights: for one row-tile of BM rows, the BM*32 bytes belonging to one k-group
// are stored contiguously, so a warp's cp.async reads 512 consecutive bytes instead of sixteen
// 32-byte fragments 3 KB apart. Row-major (0) is what the artifact stores today.
#ifndef PANEL_MAJOR
#define PANEL_MAJOR 0
#endif
// Rows per stored panel. Production needs ONE value every consumer agrees on, independent of the
// tile a given kernel picks, so this is deliberately decoupled from BM: a block whose BM spans
// several panels reads BM/PANEL contiguous runs instead of one. 0 = tie it to BM.
#ifndef PANEL_ROWS
#define PANEL_ROWS 0
#endif

constexpr int BM      = BM_ROWS;
constexpr int BN      = BN_TOKENS;
constexpr int THREADS = THREADS_N;
constexpr int WARPS   = THREADS / 32;
constexpr int WARPS_M = WARPS_M_N;
constexpr int WARPS_N = WARPS / WARPS_M;
constexpr int MT      = BM / (WARPS_M * 16); // 16-row m-tiles per warp
constexpr int NT      = BN / (WARPS_N * 8);  // 8-token n-tiles per warp
constexpr int STAGES  = STAGES_N;
constexpr int PANEL   = PANEL_ROWS == 0 ? BM_ROWS : PANEL_ROWS;
static_assert(BM_ROWS % PANEL == 0, "BM must be a whole number of stored panels");

static_assert(BM % (WARPS_M * 16) == 0, "BM must divide into 16-row tiles per warp row");
static_assert(BN % (WARPS_N * 8) == 0, "BN must divide into 8-token tiles per warp column");

// A row's 32 packed bytes are 8 shared words, so eight consecutive rows would land on only four
// bank groups. 48 keeps the cp.async destination 16-byte aligned and spreads eight rows over eight.
constexpr int WROW    = 48;
constexpr int WSTAGE  = BM * WROW;
constexpr int XSTAGE  = (BN / 8) * 32 * 16; // fragment order: one contiguous run per n-tile
constexpr int XSSTAGE = BN * 2;
constexpr int STAGE   = WSTAGE + XSTAGE + XSSTAGE;

constexpr int RING_GROUPS = 8;
constexpr int RING_BYTES  = BM * RING_GROUPS * 2;
constexpr int RING_BUFS   = 2;

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(addr));
    return r;
}

__device__ __forceinline__ uint2 lds64(const void* p) {
    uint2 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "r"(addr));
    return r;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr), "l"(gmem));
}

// The whole point of the permutation: a packed word's low nibbles are already one MMA A register's
// four codes, and its high nibbles the next one's. Two masks and two vsub4 per word -- four
// instructions for eight codes, against ten for the byte-shuffling the RowSplit order forces.
__device__ __forceinline__ unsigned decode_lo(unsigned w) {
    return __vsub4(w & 0x0f0f0f0fu, 0x08080808u);
}
__device__ __forceinline__ unsigned decode_hi(unsigned w) {
    return __vsub4((w >> 4) & 0x0f0f0f0fu, 0x08080808u);
}

// One scale per (token, group of 64), written in the fragment order the GEMM reads.
__global__ void quantize_activations(const __nv_bfloat16* __restrict__ x, std::int8_t* codes,
                                     __half* scales) {
    const int token  = blockIdx.x;
    const int cb     = token / BN;
    const int tok_in = token % BN;
    const int nt     = tok_in / 8;
    const int gid    = tok_in % 8;
    for (int g = threadIdx.x; g < GROUPS; g += blockDim.x) {
        const __nv_bfloat16* src = x + static_cast<size_t>(token) * K + g * GROUP;
        float amax               = 0.0f;
        for (int j = 0; j < GROUP; ++j) { amax = fmaxf(amax, fabsf(__bfloat162float(src[j]))); }
        amax                                                        = fmaxf(amax, 1e-20f);
        scales[(static_cast<size_t>(cb) * GROUPS + g) * BN + tok_in] = __float2half(amax / 127.0f);
        const float inv  = 127.0f / amax;
        std::int8_t* tile =
            codes + ((static_cast<size_t>(cb) * GROUPS + g) * (BN / 8) + nt) * (32 * 16);
        for (int ks = 0; ks < 2; ++ks) {
            for (int hi = 0; hi < 2; ++hi) {
                for (int tig = 0; tig < 4; ++tig) {
                    std::int8_t quad[4];
                    for (int j = 0; j < 4; ++j) {
                        const float v = __bfloat162float(src[ks * 32 + hi * 16 + tig * 4 + j]) * inv;
                        quad[j] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v))));
                    }
                    *reinterpret_cast<unsigned*>(tile + (gid * 4 + tig) * 16 + ks * 8 + hi * 4) =
                        *reinterpret_cast<const unsigned*>(quad);
                }
            }
        }
    }
}

__global__ __launch_bounds__(THREADS) void w4a8_marlin(const unsigned char* __restrict__ w_perm,
                                                       const __half* __restrict__ w_scales,
                                                       const char* __restrict__ x_perm,
                                                       const __half* __restrict__ x_scales,
                                                       __nv_bfloat16* __restrict__ out) {
    extern __shared__ char smem[];
    char* const s_base = smem;
    char* const s_ring = smem + STAGES * STAGE;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2; // row within a 16-row tile
    const int tig    = lane & 3;  // which quarter of the group this lane owns
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    // Rasterisation order. The plain 2D grid launches x-major, so every block resident at once
    // shares a token tile and streams a *different* slice of the weight -- the weight, which is the
    // big stream, gets no L2 reuse at all. SWIZ walks SWIZ row-blocks x every token block as one
    // group, so a resident wave forms a patch that reuses both operands.
#if SWIZ > 0
    const int nbm      = gridDim.x;
    const int nbn      = gridDim.y;
    const int bid      = blockIdx.x + blockIdx.y * nbm;
    const int per_grp  = SWIZ * nbn;
    const int grp      = bid / per_grp;
    const int first_m  = grp * SWIZ;
    const int gsize    = (nbm - first_m) < SWIZ ? (nbm - first_m) : SWIZ;
    const int inner    = bid - grp * per_grp;
    const int row_block = (first_m + (inner % gsize)) * BM;
    const int col_block = inner / gsize;
#else
    const int row_block = blockIdx.x * BM;
    const int col_block = blockIdx.y;
#endif

    const unsigned char* const w_blk = w_perm + static_cast<size_t>(row_block) * (K / 2);
    const char* const x_blk = x_perm + static_cast<size_t>(col_block) * GROUPS * XSTAGE;
    const char* const xs_blk =
        reinterpret_cast<const char*>(x_scales) + static_cast<size_t>(col_block) * GROUPS * XSSTAGE;
    const char* const ws_blk =
        reinterpret_cast<const char*>(w_scales) + static_cast<size_t>(row_block) * GROUPS * 2;

#if PANEL_MAJOR
    // This lane's global source and shared destination, resolved once. Leaving the panel divide
    // inside the issue lambda costs ~30% -- it is loop-invariant arithmetic competing with the
    // accumulators for registers on every group.
    constexpr int WITER = (BM * 2 + THREADS - 1) / THREADS;
    const unsigned char* w_lane[WITER];
    int                  w_dst[WITER];
#pragma unroll
    for (int i = 0; i < WITER; ++i) {
        // THREADS can exceed BM*2, so the surplus lanes must stay idle rather than run off the tile.
        const int c    = tid + i * THREADS;
        const int safe = c < BM * 2 ? c : 0;
        const int row  = safe >> 1;
        const int half = safe & 1;
        w_lane[i]      = w_blk + static_cast<size_t>(row / PANEL) * PANEL * (K / 2) +
                    ((row % PANEL) * 2 + half) * 16;
        w_dst[i] = c < BM * 2 ? (row * WROW + half * 16) : -1;
    }
#endif

    auto issue = [&](int g, int buf) {
        char* const dst = s_base + buf * STAGE;
        // W stays packed in shared: half the bytes of the unpacked staging the shipped kernel uses,
        // which is what leaves room for the ring and the wider tile.
#if PANEL_MAJOR
#pragma unroll
        for (int i = 0; i < WITER; ++i) {
            if (w_dst[i] >= 0) {
                cp_async16(dst + w_dst[i],
                           w_lane[i] + static_cast<size_t>(g) * PANEL * (GROUP / 2));
            }
        }
#else
#pragma unroll
        for (int c = tid; c < BM * 2; c += THREADS) {
            const int row  = c >> 1;
            const int half = c & 1;
            cp_async16(dst + row * WROW + half * 16,
                       w_blk + static_cast<size_t>(row) * (K / 2) + g * (GROUP / 2) + half * 16);
        }
#endif
#pragma unroll
        for (int c = tid; c < XSTAGE / 16; c += THREADS) {
            cp_async16(dst + WSTAGE + c * 16, x_blk + static_cast<size_t>(g) * XSTAGE + c * 16);
        }
#pragma unroll
        for (int c = tid; c < XSSTAGE / 16; c += THREADS) {
            cp_async16(dst + WSTAGE + XSTAGE + c * 16,
                       xs_blk + static_cast<size_t>(g) * XSSTAGE + c * 16);
        }
        if (g % RING_GROUPS == 0) {
#pragma unroll
            for (int r = tid; r < BM; r += THREADS) {
                cp_async16(s_ring + ((g / RING_GROUPS) % RING_BUFS) * RING_BYTES +
                               r * RING_GROUPS * 2,
                           ws_blk + (static_cast<size_t>(r) * GROUPS + g) * 2);
            }
        }
        asm volatile("cp.async.commit_group;");
    };

#if ABLATE == 5 || ABLATE == 6
    unsigned af[MT][2][4];
    unsigned bf[NT][2][2];
#endif
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0f;

#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        if (i < GROUPS) { issue(i, i); }
    }
#if ABLATE >= 3 && ABLATE <= 6
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
#endif

    for (int g = 0; g < GROUPS; ++g) {
        // One barrier per stage, and the order is the whole point. Wait for this stage's data, then
        // barrier once -- which both publishes it and proves everyone has finished reading the slot
        // the next issue is about to overwrite -- then issue and compute with nothing between them,
        // so the copies for stage g+STAGES-1 fly while stage g's MMAs run. The shipped kernel pays
        // two barriers per group and serialises the two phases.
#if ABLATE == 5 || ABLATE == 6
        const int outstanding = 0;
        (void)outstanding;
#else
        const int outstanding = (g + STAGES - 1 < GROUPS) ? (STAGES - 2) : 0;
#endif
#if ABLATE != 5 && ABLATE != 6
        if (outstanding >= 3) {
            asm volatile("cp.async.wait_group 3;");
        } else if (outstanding == 2) {
            asm volatile("cp.async.wait_group 2;");
        } else if (outstanding == 1) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
#endif
#if ABLATE != 5 && ABLATE != 6
        __syncthreads();
#endif
#if ABLATE < 3 || ABLATE > 6
        if (g + STAGES - 1 < GROUPS) { issue(g + STAGES - 1, (g + STAGES - 1) % STAGES); }
#endif

        const char* const sa    = s_base + (g % STAGES) * STAGE;
        const char* const sb    = sa + WSTAGE;
        const __half* const sxs = reinterpret_cast<const __half*>(sb + XSTAGE);
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / RING_GROUPS) % RING_BUFS) * RING_BYTES);

#if ABLATE != 2
#if ABLATE == 5 || ABLATE == 6
        // Hoisted: assembled once, so the loop issues MMAs and nothing else.
        if (g == 0) {
#else
        unsigned af[MT][2][4];
        unsigned bf[NT][2][2];
#endif
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0 = (warp_m * MT + m) * 16 + gid;
            // One 8-byte load per row is this lane's whole fragment for both k-halves.
            const uint2 w0 = lds64(sa + r0 * WROW + tig * 8);
            const uint2 w1 = lds64(sa + (r0 + 8) * WROW + tig * 8);
#if ABLATE == 4
            af[m][0][0] = w0.x; af[m][0][1] = w1.x; af[m][0][2] = w0.x; af[m][0][3] = w1.x;
            af[m][1][0] = w0.y; af[m][1][1] = w1.y; af[m][1][2] = w0.y; af[m][1][3] = w1.y;
#else
            af[m][0][0] = decode_lo(w0.x); af[m][0][1] = decode_lo(w1.x);
            af[m][0][2] = decode_hi(w0.x); af[m][0][3] = decode_hi(w1.x);
            af[m][1][0] = decode_lo(w0.y); af[m][1][1] = decode_lo(w1.y);
            af[m][1][2] = decode_hi(w0.y); af[m][1][3] = decode_hi(w1.y);
#endif
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const uint4 b = lds128(sb + ((warp_n * NT + n) * 32 + lane) * 16);
            bf[n][0][0] = b.x; bf[n][0][1] = b.y; bf[n][1][0] = b.z; bf[n][1][1] = b.w;
        }
#if ABLATE == 5 || ABLATE == 6
        }
#endif
#if ABLATE == 10
        // One read per token pair per group, not one per (m, token pair).
        float xa_lo[NT], xa_hi[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int c = (warp_n * NT + n) * 8 + tig * 2;
            xa_lo[n]    = __half2float(sxs[c]);
            xa_hi[n]    = __half2float(sxs[c + 1]);
        }
#endif
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int sr    = (warp_m * MT + m) * 16 + gid;
#if ABLATE == 9
            const __half ws0h = ring[sr * RING_GROUPS + (g % RING_GROUPS)];
            const __half ws1h = ring[(sr + 8) * RING_GROUPS + (g % RING_GROUPS)];
#elif ABLATE != 1 && ABLATE != 6
            const float ws0 = __half2float(ring[sr * RING_GROUPS + (g % RING_GROUPS)]);
            const float ws1 = __half2float(ring[(sr + 8) * RING_GROUPS + (g % RING_GROUPS)]);
#endif
#pragma unroll
            for (int n = 0; n < NT; ++n) {
#if ABLATE == 8
                // Two chains, no dependency between them; the sum costs 4 adds per group.
                int s[4]  = {0, 0, 0, 0};
                int s2[4] = {0, 0, 0, 0};
                mma_s8(s[0], s[1], s[2], s[3], af[m][0][0], af[m][0][1], af[m][0][2], af[m][0][3],
                       bf[n][0][0], bf[n][0][1]);
                mma_s8(s2[0], s2[1], s2[2], s2[3], af[m][1][0], af[m][1][1], af[m][1][2],
                       af[m][1][3], bf[n][1][0], bf[n][1][1]);
#pragma unroll
                for (int j = 0; j < 4; ++j) { s[j] += s2[j]; }
#else
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
#endif
#if ABLATE == 1 || ABLATE == 6
                acc[m][n][0] += static_cast<float>(s[0]);
                acc[m][n][1] += static_cast<float>(s[1]);
                acc[m][n][2] += static_cast<float>(s[2]);
                acc[m][n][3] += static_cast<float>(s[3]);
#elif ABLATE == 7
                // Only the weight scale is in the loop; the token scale is a per-column constant
                // and waits for the epilogue.
                acc[m][n][0] = fmaf(static_cast<float>(s[0]), ws0, acc[m][n][0]);
                acc[m][n][1] = fmaf(static_cast<float>(s[1]), ws0, acc[m][n][1]);
                acc[m][n][2] = fmaf(static_cast<float>(s[2]), ws1, acc[m][n][2]);
                acc[m][n][3] = fmaf(static_cast<float>(s[3]), ws1, acc[m][n][3]);
#elif ABLATE == 10
                acc[m][n][0] = fmaf(static_cast<float>(s[0]), ws0 * xa_lo[n], acc[m][n][0]);
                acc[m][n][1] = fmaf(static_cast<float>(s[1]), ws0 * xa_hi[n], acc[m][n][1]);
                acc[m][n][2] = fmaf(static_cast<float>(s[2]), ws1 * xa_lo[n], acc[m][n][2]);
                acc[m][n][3] = fmaf(static_cast<float>(s[3]), ws1 * xa_hi[n], acc[m][n][3]);
#elif ABLATE == 9
                const int c        = (warp_n * NT + n) * 8 + tig * 2;
                const __half2 xa   = *reinterpret_cast<const __half2*>(sxs + c);
                const float2 p0    = __half22float2(__hmul2(__half2half2(ws0h), xa));
                const float2 p1    = __half22float2(__hmul2(__half2half2(ws1h), xa));
                acc[m][n][0]       = fmaf(static_cast<float>(s[0]), p0.x, acc[m][n][0]);
                acc[m][n][1]       = fmaf(static_cast<float>(s[1]), p0.y, acc[m][n][1]);
                acc[m][n][2]       = fmaf(static_cast<float>(s[2]), p1.x, acc[m][n][2]);
                acc[m][n][3]       = fmaf(static_cast<float>(s[3]), p1.y, acc[m][n][3]);
#else
                const int c     = (warp_n * NT + n) * 8 + tig * 2;
                const float xa0 = __half2float(sxs[c]);
                const float xa1 = __half2float(sxs[c + 1]);
                acc[m][n][0]    = fmaf(static_cast<float>(s[0]), ws0 * xa0, acc[m][n][0]);
                acc[m][n][1]    = fmaf(static_cast<float>(s[1]), ws0 * xa1, acc[m][n][1]);
                acc[m][n][2]    = fmaf(static_cast<float>(s[2]), ws1 * xa0, acc[m][n][2]);
                acc[m][n][3]    = fmaf(static_cast<float>(s[3]), ws1 * xa1, acc[m][n][3]);
#endif
            }
        }
#else
        // Streaming floor: keep every copy, barrier and loop, drop the reads and the MMAs.
        if (tid == 0x7fffffff) { acc[0][0][0] += __half2float(sxs[0]) + __half2float(ring[0]); }
#endif
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = row_block + (warp_m * MT + m) * 16 + gid;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int c0 = col_block * BN + (warp_n * NT + n) * 8 + tig * 2;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = r0 + half * 8;
                out[static_cast<size_t>(row) * T + c0]     = __float2bfloat16(acc[m][n][half * 2]);
                out[static_cast<size_t>(row) * T + c0 + 1] = __float2bfloat16(acc[m][n][half * 2 + 1]);
            }
        }
    }
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));

    const size_t w_bytes  = static_cast<size_t>(N) * K / 2;
    const size_t ws_count = static_cast<size_t>(N) * GROUPS;
    const size_t x_count  = static_cast<size_t>(T) * K;
    const size_t xs_count = static_cast<size_t>(T) * GROUPS;

    std::vector<unsigned char> hw(w_bytes);
    std::vector<__half> hws(ws_count);
    std::vector<signed char> hx(x_count);
    std::vector<__half> hxs(xs_count);
    srand(1234);
    for (size_t i = 0; i < w_bytes; ++i) hw[i] = static_cast<unsigned char>(rand() & 0xff);
    for (size_t i = 0; i < ws_count; ++i) hws[i] = __float2half(0.002f + 0.001f * ((i % 7) / 7.0f));
    for (size_t i = 0; i < x_count; ++i) hx[i] = static_cast<signed char>((rand() % 255) - 127);
    for (size_t i = 0; i < xs_count; ++i)
        hxs[i] = __float2half(0.0031f + 0.0004f * ((i % 5) / 5.0f));

    // The permutation, exactly as the load-time device kernel would apply it: within one row and one
    // group of 64, never across rows, so row slicing and the scale plane are untouched. Lane `tig`
    // owns the 8 bytes at `tig*8`; its first word's low nibbles are k = tig*4 + 0..3 (the MMA's a0)
    // and its high nibbles k = 16 + tig*4 + 0..3 (a2); the second word carries the k+32 half.
    const auto code_at = [&](int row, int k) {
        const unsigned char byte = hw[static_cast<size_t>(row) * (K / 2) + k / 2];
        return static_cast<int>((k % 2 == 0) ? (byte & 0xf) : (byte >> 4));
    };
    std::vector<unsigned char> hwp(w_bytes);
    for (int row = 0; row < N; ++row) {
        for (int g = 0; g < GROUPS; ++g) {
#if PANEL_MAJOR
            unsigned char* dst = &hwp[static_cast<size_t>(row / PANEL) * PANEL * (K / 2) +
                                      static_cast<size_t>(g) * PANEL * (GROUP / 2) +
                                      static_cast<size_t>(row % PANEL) * (GROUP / 2)];
#else
            unsigned char* dst = &hwp[static_cast<size_t>(row) * (K / 2) + g * (GROUP / 2)];
#endif
            for (int t4 = 0; t4 < 4; ++t4) {
                for (int word = 0; word < 2; ++word) {
                    for (int j = 0; j < 4; ++j) {
                        const int k_lo = g * GROUP + word * 32 + t4 * 4 + j;
                        const int k_hi = k_lo + 16;
                        dst[t4 * 8 + word * 4 + j] =
                            static_cast<unsigned char>(code_at(row, k_lo) | (code_at(row, k_hi) << 4));
                    }
                }
            }
        }
    }

    // Activations: the quantiser owns this buffer, so it writes fragment order directly.
    std::vector<signed char> hxp(x_count);
    for (int cb = 0; cb < T / BN; ++cb)
        for (int g = 0; g < GROUPS; ++g)
            for (int nt = 0; nt < BN / 8; ++nt)
                for (int l = 0; l < 32; ++l) {
                    const int gid = l >> 2, tg = l & 3;
                    const int col = cb * BN + nt * 8 + gid;
                    signed char* dst =
                        &hxp[((static_cast<size_t>(cb) * GROUPS + g) * (BN / 8) + nt) * 512 +
                             static_cast<size_t>(l) * 16];
                    for (int ks = 0; ks < 2; ++ks)
                        for (int hi = 0; hi < 2; ++hi)
                            for (int j = 0; j < 4; ++j)
                                dst[ks * 8 + hi * 4 + j] =
                                    hx[static_cast<size_t>(col) * K + g * GROUP + ks * 32 +
                                       tg * 4 + hi * 16 + j];
                }
    std::vector<__half> hxsp(xs_count);
    for (int cb = 0; cb < T / BN; ++cb)
        for (int g = 0; g < GROUPS; ++g)
            for (int t = 0; t < BN; ++t)
                hxsp[(static_cast<size_t>(cb) * GROUPS + g) * BN + t] =
                    hxs[static_cast<size_t>(cb * BN + t) * GROUPS + g];

    void *dw, *dws, *dx, *dxs, *dout, *dflush;
    CHECK(cudaMalloc(&dw, w_bytes));
    CHECK(cudaMalloc(&dws, ws_count * sizeof(__half)));
    CHECK(cudaMalloc(&dx, x_count));
    CHECK(cudaMalloc(&dxs, xs_count * sizeof(__half)));
    CHECK(cudaMalloc(&dout, static_cast<size_t>(N) * T * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dflush, 256u << 20));
    CHECK(cudaMemcpy(dw, hwp.data(), w_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dws, hws.data(), ws_count * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hxp.data(), x_count, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dxs, hxsp.data(), xs_count * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 grid(N / BM, T / BN);
    const size_t smem = STAGES * STAGE + RING_BUFS * RING_BYTES;
    CHECK(cudaFuncSetAttribute(w4a8_marlin, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem)));
    int blocks_per_sm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, w4a8_marlin, THREADS, smem));
    printf("GPU: %s  tile=%dx%d warps=%dx%d (MT=%d NT=%d) threads=%d stages=%d\n", p.name, BM, BN,
           WARPS_M, WARPS_N, MT, NT, THREADS, STAGES);
    printf("      smem=%zu B  blocks/SM=%d (%d of 48 warps)  grid=(%d,%d)\n", smem, blocks_per_sm,
           blocks_per_sm * WARPS, grid.x, grid.y);
    if (smem > 99u * 1024) {
        printf("  shared memory over the 99 KiB budget; skipping\n");
        return 2;
    }
    if (blocks_per_sm == 0) {
        printf("  configuration does not fit; skipping\n");
        return 2;
    }

    w4a8_marlin<<<grid, THREADS, smem>>>(static_cast<const unsigned char*>(dw),
                                         static_cast<const __half*>(dws),
                                         static_cast<const char*>(dx),
                                         static_cast<const __half*>(dxs),
                                         static_cast<__nv_bfloat16*>(dout));
    CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hout(static_cast<size_t>(N) * T);
    CHECK(cudaMemcpy(hout.data(), dout, static_cast<size_t>(N) * T * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    double worst = 0.0;
#if ABLATE == 0 || ABLATE == 9 || ABLATE == 10
    for (int s = 0; s < 64; ++s) {
        const int r = static_cast<int>(static_cast<size_t>(rand()) * 7919 % N);
        const int c = rand() % T;
        double ref  = 0.0;
        for (int g = 0; g < GROUPS; ++g) {
            long long dot = 0;
            for (int j = 0; j < GROUP; ++j) {
                const int k = g * GROUP + j;
                dot += static_cast<long long>(code_at(r, k) - 8) * hx[static_cast<size_t>(c) * K + k];
            }
            ref += static_cast<double>(dot) *
                   static_cast<double>(__half2float(hws[static_cast<size_t>(r) * GROUPS + g])) *
                   static_cast<double>(__half2float(hxs[static_cast<size_t>(c) * GROUPS + g]));
        }
        const double got = static_cast<double>(__bfloat162float(hout[static_cast<size_t>(r) * T + c]));
        worst = std::max(worst, std::abs(got - ref) / std::max(std::abs(ref), 1e-6));
    }
    printf("correctness: 64 sampled outputs, worst relative error %.3e  (%s)\n", worst,
           worst < 5e-3 ? "OK" : "MISMATCH");
    if (worst >= 5e-3) {
        printf("  aborting: a fast wrong kernel is the hazard this probe exists to catch\n");
        return 1;
    }
#else
    printf("correctness: skipped (ablation)\n");
#endif

    const int reps = 12;
    std::vector<float> ms(reps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaMemsetAsync(dflush, i & 0xff, 256u << 20));
        CHECK(cudaEventRecord(a));
        w4a8_marlin<<<grid, THREADS, smem>>>(static_cast<const unsigned char*>(dw),
                                             static_cast<const __half*>(dws),
                                             static_cast<const char*>(dx),
                                             static_cast<const __half*>(dxs),
                                             static_cast<__nv_bfloat16*>(dout));
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double us  = ms[reps / 2] * 1000.0;
    const double ops = 2.0 * N * K * T;
    printf("\n  %-44s %9.1f us   %6.2f TOP/s\n", "W4A8 permuted layout, Marlin-class mainloop", us,
           ops / (us * 1e-6) / 1e12);
    printf("  %-44s %9.1f us   %6.2f TOP/s\n", "shipped kernel (64x512, 2 stages)", 3113.0,
           ops / (3113.0e-6) / 1e12);
    printf("  %-44s %9.1f us   %6.2f TOP/s\n", "cuBLAS int8 (easier problem, 2x the bytes)", 1531.9,
           ops / (1531.9e-6) / 1e12);
    printf("  %-44s %9.2fx\n", "vs shipped", 3113.0 / us);
    return 0;
}
