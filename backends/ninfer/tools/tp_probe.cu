// Go/no-go probe for tensor-parallel all-reduce transports (multi-GPU plan, phase P0).
//
// Question it answers: for a two-rank all-reduce of a [cols x hidden] FP32 partial, what does each
// transport cost in a CUDA graph, and is it correct under load? The decode step it has to fit inside
// is ~22 ms on the 27B with 128 all-reduces per token, so the number that decides TP is the
// per-collective latency at cols = 1..32, not bandwidth.
//
// Transports:
//   peer    kernel pushes the partial into the peer's VRAM inbox (needs cudaDeviceCanAccessPeer),
//           then raises a per-CTA flag with st.release.sys; the receiver polls its own VRAM.
//   host    the same kernel, but the inboxes live in mapped, portable pinned host memory.
//   staged  D2H into a pinned ring, event, H2D, local add: what stage_cross_rank_copy does today.
//           Timed eagerly and inside one joint multi-device capture (the current graph strategy).
//   nccl    ncclAllReduce, eager only. Built with -DTP_PROBE_NCCL and -lnccl.
//
// Kernel transports use a two-slot ring and an epoch counter kept in device memory, so a captured
// graph replays without re-parameterising. Two ranks on one device (--devices 0,0) exercise the
// protocol but say nothing about interconnect cost or wrong-device bugs.
//
// Build (Linux):  nvcc -O3 -arch=sm_86 tools/tp_probe.cu -o tp_probe [-DTP_PROBE_NCCL -lnccl]
// Run:            ./tp_probe --devices 0,1 [--soak 1000000] [--s2]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifdef TP_PROBE_NCCL
#include <nccl.h>
#endif

#define CK(expr)                                                                                  \
    do {                                                                                          \
        cudaError_t e_ = (expr);                                                                  \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__,  \
                         __LINE__, cudaGetErrorString(e_));                                       \
            std::exit(2);                                                                         \
        }                                                                                         \
    } while (0)

namespace {

constexpr int kMaxCtas    = 64;
constexpr int kBlock      = 256;
constexpr int kGraphOps   = 64;
constexpr int kGraphReps  = 20;
constexpr int kEagerOps   = 400;

// ---- device side -----------------------------------------------------------------------------

__device__ __forceinline__ std::uint32_t ld_acquire_sys(const std::uint32_t* p) {
    std::uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys(std::uint32_t* p, std::uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

struct XArgs {
    const float* mine;
    float* out;
    float* peer_data;               // remote inbox, slot-major [2][n]
    std::uint32_t* peer_flag;       // remote flags [2][kMaxCtas]
    const float* my_data;           // local inbox
    const std::uint32_t* my_flag;
    const std::uint32_t* base;      // epoch base, bumped after each batch of ops
    volatile std::uint32_t* err;    // mapped host word: nonzero = a spin timed out
    unsigned long long timeout_ns;
    int n;                          // floats, multiple of 4
    std::uint32_t k;                // index of this op inside its batch
    int rank;
};

// Push my partial to the peer, publish per-CTA flags, wait for the peer's, then reduce in a fixed
// rank order so both ranks produce bit-identical results.
__global__ void exchange_reduce(XArgs a) {
    const std::uint32_t epoch = *a.base + a.k + 1u;
    const std::uint32_t slot  = epoch & 1u;
    const int n4              = a.n >> 2;
    const int chunk4          = (n4 + gridDim.x - 1) / gridDim.x;
    const int lo              = blockIdx.x * chunk4;
    const int hi              = min(n4, lo + chunk4);
    const float4* mine        = reinterpret_cast<const float4*>(a.mine);
    float4* peer              = reinterpret_cast<float4*>(a.peer_data) + static_cast<size_t>(slot) * n4;

    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) { peer[i] = mine[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        st_release_sys(a.peer_flag + slot * kMaxCtas + blockIdx.x, epoch);
        const std::uint32_t* flag = a.my_flag + slot * kMaxCtas + blockIdx.x;
        const unsigned long long t0 = globaltimer();
        std::uint32_t spins         = 0;
        while (ld_acquire_sys(flag) != epoch) {
            if ((++spins & 1023u) == 0 && globaltimer() - t0 > a.timeout_ns) {
                *a.err = 1u;
                break;
            }
        }
    }
    __syncthreads();
    __threadfence();

    const float4* theirs = reinterpret_cast<const float4*>(a.my_data) + static_cast<size_t>(slot) * n4;
    float4* out          = reinterpret_cast<float4*>(a.out);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float4 p = __ldcv(theirs + i);
        const float4 m = mine[i];
        const float4 x = a.rank == 0 ? m : p;
        const float4 y = a.rank == 0 ? p : m;
        out[i]         = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
}

__global__ void bump_epoch(std::uint32_t* base, std::uint32_t by) { *base += by; }

__device__ __forceinline__ float pattern(int i, int rank, std::uint32_t iter) {
    return static_cast<float>((static_cast<std::uint32_t>(i) * 3u + rank * 5u + iter * 7u) & 255u);
}

__global__ void fill_partial(float* p, int n, int rank, const std::uint32_t* base, std::uint32_t k) {
    const std::uint32_t iter = *base + k;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        p[i] = pattern(i, rank, iter);
    }
}

__global__ void verify_sum(const float* out, int n, const std::uint32_t* base, std::uint32_t k,
                           unsigned long long* mismatches) {
    const std::uint32_t iter = *base + k;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (out[i] != pattern(i, 0, iter) + pattern(i, 1, iter)) { atomicAdd(mismatches, 1ULL); }
    }
}

__global__ void local_add(const float* mine, const float* theirs, float* out, int n, int rank) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float x = rank == 0 ? mine[i] : theirs[i];
        const float y = rank == 0 ? theirs[i] : mine[i];
        out[i]        = x + y;
    }
}

// ---- host side -------------------------------------------------------------------------------

enum class Transport { Peer, Host, Staged, Nccl };

const char* name_of(Transport t) {
    switch (t) {
    case Transport::Peer: return "peer";
    case Transport::Host: return "host";
    case Transport::Staged: return "staged";
    case Transport::Nccl: return "nccl";
    }
    return "?";
}

struct Rank {
    int dev = 0;
    cudaStream_t stream{};
    float* mine = nullptr;
    float* out = nullptr;
    float* inbox = nullptr;             // VRAM inbox for the peer transport (and staged H2D target)
    std::uint32_t* flags = nullptr;     // VRAM flags
    std::uint32_t* base = nullptr;
    unsigned long long* mismatches = nullptr;
    // staged ring
    float* ring[2]{};                   // pinned host slots this rank writes
    cudaEvent_t filled[2]{};
    cudaEvent_t read[2]{};              // recorded after this rank's H2D from the peer's slot
};

struct Probe {
    Rank rank[2];
    int max_floats = 0;
    // mapped host inboxes, one per receiving rank (device-visible pointers)
    float* host_data[2]{};
    std::uint32_t* host_flags[2]{};
    volatile std::uint32_t* err = nullptr;
    unsigned long long timeout_ns = 3'000'000'000ULL;
    bool same_device = false;
    bool peer_ok = false;
};

double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

void sync_all(Probe& p) {
    for (auto& r : p.rank) { CK(cudaSetDevice(r.dev)); CK(cudaStreamSynchronize(r.stream)); }
}

void reset_protocol(Probe& p) {
    for (int r = 0; r < 2; ++r) {
        CK(cudaSetDevice(p.rank[r].dev));
        CK(cudaMemsetAsync(p.rank[r].flags, 0, 2 * kMaxCtas * sizeof(std::uint32_t), p.rank[r].stream));
        CK(cudaMemsetAsync(p.rank[r].base, 0, sizeof(std::uint32_t), p.rank[r].stream));
        CK(cudaMemsetAsync(p.rank[r].mismatches, 0, sizeof(unsigned long long), p.rank[r].stream));
        std::memset(p.host_flags[r], 0, 2 * kMaxCtas * sizeof(std::uint32_t));
    }
    sync_all(p);
    *p.err = 0;
}

int ctas_for(int n) { return std::min(kMaxCtas, std::max(1, n / 4 / 128)); }

void launch_exchange(Probe& p, Transport t, int r, int n, std::uint32_t k) {
    Rank& me = p.rank[r];
    Rank& other = p.rank[1 - r];
    XArgs a{};
    a.mine = me.mine;
    a.out = me.out;
    a.base = me.base;
    a.err = p.err;
    a.timeout_ns = p.timeout_ns;
    a.n = n;
    a.k = k;
    a.rank = r;
    if (t == Transport::Peer) {
        a.peer_data = other.inbox;
        a.peer_flag = other.flags;
        a.my_data = me.inbox;
        a.my_flag = me.flags;
    } else {
        a.peer_data = p.host_data[1 - r];
        a.peer_flag = p.host_flags[1 - r];
        a.my_data = p.host_data[r];
        a.my_flag = p.host_flags[r];
    }
    exchange_reduce<<<ctas_for(n), kBlock, 0, me.stream>>>(a);
}

// One staged step on both ranks. Slot alternates so a writer never overwrites a slot the reader has
// not finished with; `k` is the step index within the current graph or eager batch.
void staged_step(Probe& p, int n, int k) {
    const int slot = k & 1;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    for (int r = 0; r < 2; ++r) {
        Rank& me = p.rank[r];
        CK(cudaSetDevice(me.dev));
        if (k >= 2) { CK(cudaStreamWaitEvent(me.stream, p.rank[1 - r].read[slot], 0)); }
        CK(cudaMemcpyAsync(me.ring[slot], me.mine, bytes, cudaMemcpyDeviceToHost, me.stream));
        CK(cudaEventRecord(me.filled[slot], me.stream));
    }
    for (int r = 0; r < 2; ++r) {
        Rank& me = p.rank[r];
        CK(cudaSetDevice(me.dev));
        CK(cudaStreamWaitEvent(me.stream, p.rank[1 - r].filled[slot], 0));
        CK(cudaMemcpyAsync(me.inbox, p.rank[1 - r].ring[slot], bytes, cudaMemcpyHostToDevice, me.stream));
        CK(cudaEventRecord(me.read[slot], me.stream));
        local_add<<<64, kBlock, 0, me.stream>>>(me.mine, me.inbox, me.out, n, r);
    }
}

struct Timing {
    bool ok = false;
    double median_us = 0, min_us = 0;
    std::string note;
};

Timing summarize(std::vector<double>& v, double scale) {
    Timing t;
    if (v.empty()) return t;
    std::sort(v.begin(), v.end());
    t.ok = true;
    t.median_us = v[v.size() / 2] / scale;
    t.min_us = v.front() / scale;
    return t;
}

// Eager: back-to-back launches, one host thread. Launch-bound for tiny sizes, so it is a ceiling,
// not the number that decides anything.
Timing time_eager(Probe& p, Transport t, int n) {
    std::vector<double> samples;
    for (int rep = 0; rep < 5; ++rep) {
        reset_protocol(p);
        const double t0 = now_us();
        for (int k = 0; k < kEagerOps; ++k) {
            if (t == Transport::Staged) {
                staged_step(p, n, k);
            } else {
                for (int r = 0; r < 2; ++r) {
                    CK(cudaSetDevice(p.rank[r].dev));
                    launch_exchange(p, t, r, n, 0);
                    bump_epoch<<<1, 1, 0, p.rank[r].stream>>>(p.rank[r].base, 1);
                }
            }
        }
        sync_all(p);
        samples.push_back((now_us() - t0) / kEagerOps);
        if (*p.err) { Timing bad; bad.note = "spin timeout"; return bad; }
    }
    return summarize(samples, 1.0);
}

// Graph: each rank captures kGraphOps back-to-back all-reduces on its own stream, no cross-rank
// events (the kernel protocol needs none). Rank 0 is launched first and spins until rank 1 joins.
Timing time_graph(Probe& p, Transport t, int n) {
    cudaGraphExec_t exec[2]{};
    cudaGraph_t graph[2]{};
    for (int r = 0; r < 2; ++r) {
        CK(cudaSetDevice(p.rank[r].dev));
        CK(cudaStreamBeginCapture(p.rank[r].stream, cudaStreamCaptureModeThreadLocal));
        for (int k = 0; k < kGraphOps; ++k) { launch_exchange(p, t, r, n, static_cast<std::uint32_t>(k)); }
        bump_epoch<<<1, 1, 0, p.rank[r].stream>>>(p.rank[r].base, kGraphOps);
        CK(cudaStreamEndCapture(p.rank[r].stream, &graph[r]));
        CK(cudaGraphInstantiate(&exec[r], graph[r], 0));
    }
    std::vector<double> samples;
    for (int rep = 0; rep < kGraphReps; ++rep) {
        reset_protocol(p);
        const double t0 = now_us();
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(p.rank[r].dev));
            CK(cudaGraphLaunch(exec[r], p.rank[r].stream));
        }
        sync_all(p);
        samples.push_back((now_us() - t0) / kGraphOps);
        if (*p.err) {
            Timing bad;
            bad.note = "spin timeout";
            return bad;
        }
    }
    for (int r = 0; r < 2; ++r) {
        CK(cudaSetDevice(p.rank[r].dev));
        CK(cudaGraphExecDestroy(exec[r]));
        CK(cudaGraphDestroy(graph[r]));
    }
    return summarize(samples, 1.0);
}

// Staged transport inside one joint capture: rank 0's stream is the origin, rank 1 joins by event.
// This is the strategy the tree uses today, so failing here answers the "is it viable" question.
Timing time_staged_joint_graph(Probe& p, int n) {
    Timing t;
    cudaEvent_t fork{}, join{};
    CK(cudaSetDevice(p.rank[0].dev));
    CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    CK(cudaSetDevice(p.rank[1].dev));
    CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CK(cudaSetDevice(p.rank[0].dev));
    cudaError_t e = cudaStreamBeginCapture(p.rank[0].stream, cudaStreamCaptureModeGlobal);
    if (e == cudaSuccess) {
        CK(cudaEventRecord(fork, p.rank[0].stream));
        CK(cudaSetDevice(p.rank[1].dev));
        CK(cudaStreamWaitEvent(p.rank[1].stream, fork, 0));
        for (int k = 0; k < kGraphOps; ++k) { staged_step(p, n, k); }
        CK(cudaSetDevice(p.rank[1].dev));
        CK(cudaEventRecord(join, p.rank[1].stream));
        CK(cudaSetDevice(p.rank[0].dev));
        CK(cudaStreamWaitEvent(p.rank[0].stream, join, 0));
        e = cudaStreamEndCapture(p.rank[0].stream, &graph);
    }
    if (e == cudaSuccess) { e = cudaGraphInstantiate(&exec, graph, 0); }
    if (e != cudaSuccess) {
        (void)cudaGetLastError();
        t.note = std::string("capture failed: ") + cudaGetErrorString(e);
        return t;
    }
    std::vector<double> samples;
    for (int rep = 0; rep < kGraphReps; ++rep) {
        sync_all(p);
        const double t0 = now_us();
        CK(cudaSetDevice(p.rank[0].dev));
        CK(cudaGraphLaunch(exec, p.rank[0].stream));
        sync_all(p);
        samples.push_back((now_us() - t0) / kGraphOps);
    }
    CK(cudaSetDevice(p.rank[0].dev));
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    return summarize(samples, 1.0);
}

#ifdef TP_PROBE_NCCL
Timing time_nccl_eager(Probe& p, int n) {
    Timing t;
    if (p.same_device) {
        t.note = "nccl needs two distinct devices";
        return t;
    }
    ncclComm_t comms[2];
    int devs[2] = {p.rank[0].dev, p.rank[1].dev};
    if (ncclCommInitAll(comms, 2, devs) != ncclSuccess) {
        t.note = "ncclCommInitAll failed";
        return t;
    }
    std::vector<double> samples;
    for (int rep = 0; rep < 5; ++rep) {
        sync_all(p);
        const double t0 = now_us();
        for (int k = 0; k < kEagerOps; ++k) {
            ncclGroupStart();
            for (int r = 0; r < 2; ++r) {
                CK(cudaSetDevice(p.rank[r].dev));
                ncclAllReduce(p.rank[r].mine, p.rank[r].out, n, ncclFloat, ncclSum, comms[r], p.rank[r].stream);
            }
            ncclGroupEnd();
        }
        sync_all(p);
        samples.push_back((now_us() - t0) / kEagerOps);
    }
    for (auto& c : comms) ncclCommDestroy(c);
    return summarize(samples, 1.0);
}
#endif

// Soak: fill, exchange and verify in one graph, replayed until `total` exchanges have run. Every
// element is an exactly-representable integer so any lost or stale word is a hard mismatch.
bool soak(Probe& p, Transport t, int n, long long total) {
    cudaGraphExec_t exec[2]{};
    cudaGraph_t graph[2]{};
    for (int r = 0; r < 2; ++r) {
        CK(cudaSetDevice(p.rank[r].dev));
        CK(cudaStreamBeginCapture(p.rank[r].stream, cudaStreamCaptureModeThreadLocal));
        for (int k = 0; k < kGraphOps; ++k) {
            fill_partial<<<32, kBlock, 0, p.rank[r].stream>>>(p.rank[r].mine, n, r, p.rank[r].base, k);
            launch_exchange(p, t, r, n, static_cast<std::uint32_t>(k));
            verify_sum<<<32, kBlock, 0, p.rank[r].stream>>>(p.rank[r].out, n, p.rank[r].base, k,
                                                            p.rank[r].mismatches);
        }
        bump_epoch<<<1, 1, 0, p.rank[r].stream>>>(p.rank[r].base, kGraphOps);
        CK(cudaStreamEndCapture(p.rank[r].stream, &graph[r]));
        CK(cudaGraphInstantiate(&exec[r], graph[r], 0));
    }
    reset_protocol(p);
    const long long launches = (total + kGraphOps - 1) / kGraphOps;
    for (long long i = 0; i < launches; ++i) {
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(p.rank[r].dev));
            CK(cudaGraphLaunch(exec[r], p.rank[r].stream));
        }
        if ((i & 255) == 255) {
            sync_all(p);
            if (*p.err) break;
        }
    }
    sync_all(p);
    unsigned long long bad[2]{};
    for (int r = 0; r < 2; ++r) {
        CK(cudaSetDevice(p.rank[r].dev));
        CK(cudaMemcpy(&bad[r], p.rank[r].mismatches, sizeof(bad[r]), cudaMemcpyDeviceToHost));
    }
    const bool ok = bad[0] == 0 && bad[1] == 0 && *p.err == 0;
    std::printf("soak %-6s n=%d exchanges=%lld mismatches=%llu/%llu timeout=%u -> %s\n", name_of(t), n,
                launches * kGraphOps, bad[0], bad[1], static_cast<unsigned>(*p.err), ok ? "PASS" : "FAIL");
    for (int r = 0; r < 2; ++r) {
        CK(cudaSetDevice(p.rank[r].dev));
        CK(cudaGraphExecDestroy(exec[r]));
        CK(cudaGraphDestroy(graph[r]));
    }
    return ok;
}

// S2: does pinned memory allocated under device A behave as pinned for a copy on device B? Without
// cudaHostAllocPortable it should not, and a graph node built from it may not capture.
void s2(int d0, int d1) {
    const size_t bytes = 16u << 20;
    void *plain = nullptr, *portable = nullptr;
    CK(cudaSetDevice(d0));
    CK(cudaHostAlloc(&plain, bytes, cudaHostAllocDefault));
    CK(cudaHostAlloc(&portable, bytes, cudaHostAllocPortable));
    CK(cudaSetDevice(d1));
    void* dev = nullptr;
    CK(cudaMalloc(&dev, bytes));
    cudaStream_t s{};
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (int which = 0; which < 2; ++which) {
        void* host = which == 0 ? plain : portable;
        const char* label = which == 0 ? "non-portable" : "portable    ";
        std::vector<double> v;
        for (int i = 0; i < 20; ++i) {
            const double t0 = now_us();
            CK(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s));
            CK(cudaStreamSynchronize(s));
            v.push_back(now_us() - t0);
        }
        std::sort(v.begin(), v.end());
        cudaGraph_t g{};
        cudaGraphExec_t ge{};
        cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        if (e == cudaSuccess) {
            e = cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s);
            cudaError_t end = cudaStreamEndCapture(s, &g);
            if (e == cudaSuccess) e = end;
        }
        if (e == cudaSuccess) e = cudaGraphInstantiate(&ge, g, 0);
        if (e == cudaSuccess) {
            e = cudaGraphLaunch(ge, s);
            if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        }
        std::printf("s2 H2D %s on device %d: %.2f GB/s  graph=%s\n", label, d1,
                    bytes / v[v.size() / 2] / 1e3, e == cudaSuccess ? "OK" : cudaGetErrorString(e));
        (void)cudaGetLastError();
    }
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) { out.push_back(std::atoi(item.c_str())); }
    return out;
}

std::vector<std::string> parse_names(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) { out.push_back(item); }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> devices{0, 1};
    std::vector<int> cols{1, 4, 16, 32, 128, 1024};
    std::vector<std::string> transports{"peer", "host", "staged", "nccl"};
    int hidden = 5120;
    long long soak_total = 0;
    bool do_s2 = false;
    long long timeout_ms = 3000;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--devices") devices = parse_list(value());
        else if (a == "--cols") cols = parse_list(value());
        else if (a == "--hidden") hidden = std::atoi(value().c_str());
        else if (a == "--transports") transports = parse_names(value());
        else if (a == "--soak") soak_total = std::atoll(value().c_str());
        else if (a == "--s2") do_s2 = true;
        else if (a == "--timeout-ms") timeout_ms = std::atoll(value().c_str());
        else {
            std::fprintf(stderr,
                         "usage: tp_probe [--devices a,b] [--cols 1,4,...] [--hidden N] "
                         "[--transports peer,host,staged,nccl] [--soak N] [--s2] [--timeout-ms N]\n");
            return 1;
        }
    }
    if (devices.size() != 2 || hidden % 4 != 0) { std::fprintf(stderr, "need two devices, hidden %% 4 == 0\n"); return 1; }

    int count = 0;
    CK(cudaGetDeviceCount(&count));
    for (int d : devices) {
        if (d < 0 || d >= count) { std::fprintf(stderr, "device %d not present (%d visible)\n", d, count); return 1; }
        cudaDeviceProp props{};
        CK(cudaGetDeviceProperties(&props, d));
        int async_engines = 0;
        CK(cudaDeviceGetAttribute(&async_engines, cudaDevAttrAsyncEngineCount, d));
        char bus[32]{};
        CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), d));
        std::printf("device %d: %s sm_%d%d sms=%d vram=%zu MiB copy_engines=%d bus=%s\n", d, props.name,
                    props.major, props.minor, props.multiProcessorCount,
                    static_cast<size_t>(props.totalGlobalMem >> 20), async_engines, bus);
    }

    Probe p;
    // Keep this under the Windows display driver's ~2 s kernel watchdog when running on a display GPU.
    p.timeout_ns    = static_cast<unsigned long long>(timeout_ms) * 1'000'000ULL;
    p.same_device = devices[0] == devices[1];
    p.max_floats = 0;
    for (int c : cols) p.max_floats = std::max(p.max_floats, c * hidden);
    const size_t max_bytes = static_cast<size_t>(p.max_floats) * sizeof(float);

    if (!p.same_device) {
        int a01 = 0, a10 = 0;
        CK(cudaDeviceCanAccessPeer(&a01, devices[0], devices[1]));
        CK(cudaDeviceCanAccessPeer(&a10, devices[1], devices[0]));
        std::printf("peer access: %d->%d=%d %d->%d=%d\n", devices[0], devices[1], a01, devices[1], devices[0], a10);
        p.peer_ok = a01 && a10;
        if (p.peer_ok) {
            for (int r = 0; r < 2; ++r) {
                CK(cudaSetDevice(devices[r]));
                cudaError_t e = cudaDeviceEnablePeerAccess(devices[1 - r], 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    std::printf("enable peer %d->%d failed: %s\n", devices[r], devices[1 - r], cudaGetErrorString(e));
                    p.peer_ok = false;
                }
                (void)cudaGetLastError();
            }
        }
    } else {
        p.peer_ok = true;
    }

    CK(cudaSetDevice(devices[0]));
    std::uint32_t* err_host = nullptr;
    CK(cudaHostAlloc(reinterpret_cast<void**>(&err_host), sizeof(std::uint32_t), cudaHostAllocMapped | cudaHostAllocPortable));
    p.err = err_host;
    for (int r = 0; r < 2; ++r) {
        Rank& k = p.rank[r];
        k.dev = devices[r];
        CK(cudaSetDevice(k.dev));
        CK(cudaStreamCreateWithFlags(&k.stream, cudaStreamNonBlocking));
        CK(cudaMalloc(&k.mine, max_bytes));
        CK(cudaMalloc(&k.out, max_bytes));
        CK(cudaMalloc(&k.inbox, 2 * max_bytes));
        CK(cudaMalloc(&k.flags, 2 * kMaxCtas * sizeof(std::uint32_t)));
        CK(cudaMalloc(&k.base, sizeof(std::uint32_t)));
        CK(cudaMalloc(&k.mismatches, sizeof(unsigned long long)));
        CK(cudaMemset(k.mine, 0, max_bytes));
        for (int s = 0; s < 2; ++s) {
            CK(cudaHostAlloc(reinterpret_cast<void**>(&k.ring[s]), max_bytes, cudaHostAllocPortable));
            CK(cudaEventCreateWithFlags(&k.filled[s], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&k.read[s], cudaEventDisableTiming));
        }
        void* hd = nullptr;
        void* hf = nullptr;
        CK(cudaHostAlloc(&hd, 2 * max_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostAlloc(&hf, 2 * kMaxCtas * sizeof(std::uint32_t), cudaHostAllocMapped | cudaHostAllocPortable));
        p.host_data[r] = static_cast<float*>(hd);
        p.host_flags[r] = static_cast<std::uint32_t*>(hf);
    }

    std::printf("\n%-7s %-6s %5s %11s %10s %10s\n", "path", "mode", "cols", "fp32_bytes", "median_us", "min_us");
    for (const std::string& name : transports) {
        Transport t;
        if (name == "peer") t = Transport::Peer;
        else if (name == "host") t = Transport::Host;
        else if (name == "staged") t = Transport::Staged;
        else if (name == "nccl") t = Transport::Nccl;
        else { std::fprintf(stderr, "unknown transport %s\n", name.c_str()); return 1; }
        if (t == Transport::Peer && !p.peer_ok) {
            std::printf("%-7s skipped: no peer access between the devices\n", name_of(t));
            continue;
        }
        for (int c : cols) {
            const int n = c * hidden;
            auto report = [&](const char* mode, const Timing& tm) {
                if (tm.ok) {
                    std::printf("%-7s %-6s %5d %11zu %10.2f %10.2f\n", name_of(t), mode, c,
                                static_cast<size_t>(n) * 4, tm.median_us, tm.min_us);
                } else {
                    std::printf("%-7s %-6s %5d %11zu   %s\n", name_of(t), mode, c, static_cast<size_t>(n) * 4,
                                tm.note.c_str());
                }
            };
            if (t == Transport::Nccl) {
#ifdef TP_PROBE_NCCL
                report("eager", time_nccl_eager(p, n));
#else
                std::printf("%-7s not built (-DTP_PROBE_NCCL -lnccl)\n", name_of(t));
                break;
#endif
            } else if (t == Transport::Staged) {
                report("eager", time_eager(p, t, n));
                report("joint", time_staged_joint_graph(p, n));
            } else {
                report("eager", time_eager(p, t, n));
                report("graph", time_graph(p, t, n));
            }
        }
    }

    bool all_ok = true;
    if (soak_total > 0) {
        std::printf("\n");
        for (const std::string& name : transports) {
            if (name == "peer" && p.peer_ok) all_ok = soak(p, Transport::Peer, hidden * 4, soak_total) && all_ok;
            if (name == "host") all_ok = soak(p, Transport::Host, hidden * 4, soak_total) && all_ok;
        }
    }
    if (do_s2) {
        std::printf("\n");
        if (p.same_device) std::printf("s2: needs two distinct devices\n");
        else s2(devices[0], devices[1]);
    }
    return all_ok ? 0 : 4;
}
