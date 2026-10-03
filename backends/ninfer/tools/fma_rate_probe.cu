// Measures the non-tensor arithmetic rates that decide whether a GPU's compute has been throttled
// the way the CMP 170HX's is: FP32 FMA, FP32 mul+add issued as separate instructions, FP32 mul and
// add alone, INT32 multiply-add (IMAD), and packed FP16 / BF16 FMA.
//
// Why it exists. The CMP 170HX ships with FP32 FMA throttled to roughly 1/16 of its multiply/add
// rate (0.39 vs 6.2 TFLOPS in the community measurements), and a community firmware unlock reports
// switching the "FMA/IMLA" throttle off. IMLA is integer multiply-add, which every kernel here uses
// for indexing and most use for dequantization, so it is measured too. A rented card may or may not
// carry the unlock; the verdict lines at the bottom say which.
//
// Every operation is an explicit intrinsic, so the result does not depend on -fmad: __fmaf_rn is
// always an FFMA, and __fadd_rn(__fmul_rn(...)) is never contracted into one. Each warp runs
// kChains independent dependent-chains so instruction latency is hidden and the number is issue
// throughput. Inputs are runtime values, so nothing folds.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 tools/fma_rate_probe.cu -o fma_rate_probe
// Use -arch=sm_86 / sm_89 on those parts; sm_80 is the CMP 170HX (GA100).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e__ = (x);                                                                     \
        if (e__ != cudaSuccess) {                                                                  \
            std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e__), __LINE__);          \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {

constexpr int kChains = 8;
constexpr int kIters  = 16384;
constexpr int kReps   = 5;

// Each Op provides the value type, one dependent step, how many arithmetic operations that step
// counts as (an FMA is 2, a packed half2 FMA is 4), and a conversion used only to keep the chain
// live. Step constants are chosen so the chain stays finite and normal for kIters steps.
struct Ffma {
    using T = float;
    static constexpr double kOps = 2.0;
    __device__ static T step(T c, T x, T y) { return __fmaf_rn(c, x, y); }
    __device__ static float live(T c) { return c; }
};
// c*x + y as two instructions. The multiply depends on c, so it cannot be hoisted out of the loop.
struct MulAdd {
    using T = float;
    static constexpr double kOps = 2.0;
    __device__ static T step(T c, T x, T y) { return __fadd_rn(__fmul_rn(c, x), y); }
    __device__ static float live(T c) { return c; }
};
struct Fmul {
    using T = float;
    static constexpr double kOps = 1.0;
    __device__ static T step(T c, T x, T) { return __fmul_rn(c, x); }
    __device__ static float live(T c) { return c; }
};
struct Fadd {
    using T = float;
    static constexpr double kOps = 1.0;
    __device__ static T step(T c, T, T y) { return __fadd_rn(c, y); }
    __device__ static float live(T c) { return c; }
};
// Unsigned so wraparound is defined; the compiler emits IMAD for a*b+c.
struct Imad {
    using T = unsigned;
    static constexpr double kOps = 2.0;
    __device__ static T step(T c, T x, T y) { return c * x + y; }
    __device__ static float live(T c) { return static_cast<float>(c); }
};
struct Hfma2 {
    using T = __half2;
    static constexpr double kOps = 4.0;
    __device__ static T step(T c, T x, T y) { return __hfma2(c, x, y); }
    __device__ static float live(T c) { return __low2float(c) + __high2float(c); }
};
struct Bfma2 {
    using T = __nv_bfloat162;
    static constexpr double kOps = 4.0;
    __device__ static T step(T c, T x, T y) { return __hfma2(c, x, y); }
    __device__ static float live(T c) { return __low2float(c) + __high2float(c); }
};

template <class Op>
__global__ void chain_kernel(float* sink, typename Op::T x, typename Op::T y) {
    typename Op::T c[kChains];
#pragma unroll
    for (int i = 0; i < kChains; ++i) { c[i] = y; }

    for (int it = 0; it < kIters; ++it) {
#pragma unroll
        for (int i = 0; i < kChains; ++i) { c[i] = Op::step(c[i], x, y); }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kChains; ++i) { acc += Op::live(c[i]); }
    if (acc == 1234.5678f) { sink[0] = acc; }
}

// Returns achieved arithmetic operations per second, in units of 1e12.
template <class Op>
double run(int blocks, int threads, float* sink, typename Op::T x, typename Op::T y) {
    chain_kernel<Op><<<blocks, threads>>>(sink, x, y);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    double best = 0.0;
    for (int trial = 0; trial < 3; ++trial) {
        cudaEvent_t a, b;
        CHECK(cudaEventCreate(&a));
        CHECK(cudaEventCreate(&b));
        CHECK(cudaEventRecord(a));
        for (int r = 0; r < kReps; ++r) { chain_kernel<Op><<<blocks, threads>>>(sink, x, y); }
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        CHECK(cudaEventDestroy(a));
        CHECK(cudaEventDestroy(b));

        const double lanes = static_cast<double>(blocks) * threads;
        const double ops   = lanes * kIters * kChains * kReps * Op::kOps;
        const double rate  = ops / (ms / 1000.0) / 1e12;
        if (rate > best) { best = rate; }
    }
    return best;
}

// FP32 FFMA lanes per SM by architecture; used only to state what an unthrottled part would do.
int fp32_lanes_per_sm(int major, int minor) {
    if (major == 8 && minor == 0) { return 64; }
    if (major == 8) { return 128; } // sm_86 / sm_89
    return 0;
}

} // namespace

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    // cudaDeviceProp dropped clockRate in CUDA 13; the attribute exists in every toolkit.
    int sm_khz = 0;
    CHECK(cudaDeviceGetAttribute(&sm_khz, cudaDevAttrClockRate, 0));
    const double ghz = sm_khz / 1.0e6;
    std::printf("GPU: %s  sm_%d%d  %d SMs  %.0f MHz\n\n", p.name, p.major, p.minor,
                p.multiProcessorCount, sm_khz / 1000.0);

    const int threads = 256;
    const int blocks  = p.multiProcessorCount * 8;
    float* sink       = nullptr;
    CHECK(cudaMalloc(&sink, sizeof(float)));

    // |x| < 1 keeps c*x+y bounded; the values are runtime arguments so the chains cannot fold.
    const float x_mac = 0.999f, y_mac = 0.001f;
    const float x_mul = 1.0000001f, y_add = 1.0e-7f;

    const double ffma   = run<Ffma>(blocks, threads, sink, x_mac, y_mac);
    const double muladd = run<MulAdd>(blocks, threads, sink, x_mac, y_mac);
    const double fmul   = run<Fmul>(blocks, threads, sink, x_mul, 0.f);
    const double fadd   = run<Fadd>(blocks, threads, sink, 0.f, y_add);
    const double imad   = run<Imad>(blocks, threads, sink, 1664525u, 1013904223u);
    const double hfma2  = run<Hfma2>(blocks, threads, sink, __float2half2_rn(0.999f),
                                     __float2half2_rn(0.001f));
    const double bfma2  = run<Bfma2>(blocks, threads, sink, __float2bfloat162_rn(0.999f),
                                     __float2bfloat162_rn(0.001f));

    // Achieved operations per SM per clock is the number that transfers across parts.
    const double per_sm_clk = 1e12 / (static_cast<double>(p.multiProcessorCount) * ghz * 1e9);
    auto row = [&](const char* name, double tops, const char* unit) {
        std::printf("  %-34s %8.2f %-7s %7.1f ops/SM/clk\n", name, tops, unit, tops * per_sm_clk);
    };
    row("FP32 FFMA  (__fmaf_rn)", ffma, "TFLOPS");
    row("FP32 mul+add (separate)", muladd, "TFLOPS");
    row("FP32 FMUL", fmul, "TFLOPS");
    row("FP32 FADD", fadd, "TFLOPS");
    row("INT32 IMAD (a*b+c)", imad, "TOPS");
    row("FP16 HFMA2", hfma2, "TFLOPS");
    row("BF16 HFMA2", bfma2, "TFLOPS");

    const int lanes = fp32_lanes_per_sm(p.major, p.minor);
    if (lanes > 0) {
        const double expected = 2.0 * lanes * p.multiProcessorCount * ghz / 1e3;
        std::printf("\n  unthrottled FFMA peak for this part: %.1f TFLOPS (%d lanes/SM x2 x SMs x "
                    "boost clock)\n",
                    expected, lanes);
    }

    // The community numbers for a locked card are ~0.39 TFLOPS FFMA against ~6.2 non-FMA, a ratio
    // near 1/16. Anything below a quarter is unmistakably a throttle, not measurement noise.
    std::printf("\n");
    if (muladd > 0.0 && ffma < 0.25 * muladd) {
        std::printf("VERDICT fma: THROTTLED   (FFMA %.2f vs mul+add %.2f TFLOPS, ratio %.3f)\n",
                    ffma, muladd, ffma / muladd);
    } else {
        std::printf("VERDICT fma: not throttled   (FFMA %.2f vs mul+add %.2f TFLOPS, ratio %.3f)\n",
                    ffma, muladd, ffma / muladd);
    }
    // IMAD has no separate reference rate, so compare it with FFMA: on an unthrottled Ampere part
    // they issue at similar rates, and a locked one drops it by an order of magnitude.
    if (ffma > 0.0 && imad < 0.25 * ffma) {
        std::printf("VERDICT imad: LOW   (IMAD %.2f TOPS is under a quarter of FFMA %.2f TFLOPS)\n",
                    imad, ffma);
    } else {
        std::printf("VERDICT imad: ok   (IMAD %.2f TOPS vs FFMA %.2f TFLOPS)\n", imad, ffma);
    }

    CHECK(cudaFree(sink));
    return 0;
}
