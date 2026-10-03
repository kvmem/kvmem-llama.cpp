// Qualification of `StageLink`, the boundary transfer between pipeline stages.
//
// Two physical cards are not needed for the protocol: every hazard here is an ordering question
// between streams, and a stage boundary is a memcpy plus a fence whichever device the two ends live
// on. Three ranks are therefore placed on device 0, with `force_staged` selecting the real pinned
// host path rather than the same-device shortcut. What that cannot show is a wrong-device pointer,
// which shared-device ranks hide; that is checked on real hardware.
//
// The reuse ordering is asserted on the captured graph rather than by racing two streams: a race
// harness has no power on Windows, where holding one stream also stalls the driver's submission of
// another stream's copies whether or not the fence exists. The graph states the dependency outright.

#include "core/device.h"
#include "core/stage_link.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <iostream>
#include <span>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

template <class Fn>
int expect_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::exception&) {
        return 0;
    }
    std::cerr << "expectation failed: " << label << " (nothing was thrown)\n";
    return 1;
}

std::vector<std::uint8_t> pattern(std::size_t bytes, std::uint32_t seed) {
    std::vector<std::uint8_t> out(bytes);
    for (std::size_t index = 0; index < bytes; ++index) {
        out[index] = static_cast<std::uint8_t>((index * 31U + seed * 97U + 1U) & 0xFFU);
    }
    return out;
}

int expect_payload(const std::vector<std::uint8_t>& actual, const std::vector<std::uint8_t>& want,
                   const char* label) {
    if (actual == want) { return 0; }
    std::size_t first = 0;
    while (first < want.size() && actual[first] == want[first]) { ++first; }
    std::cerr << "expectation failed: " << label << " (first difference at byte " << first
              << ": got " << static_cast<int>(actual[first]) << ", wanted "
              << static_cast<int>(want[first]) << ")\n";
    return 1;
}

class DeviceBytes {
public:
    explicit DeviceBytes(std::size_t bytes) : bytes_(bytes) { CUDA_CHECK(cudaMalloc(&data_, bytes)); }
    ~DeviceBytes() {
        if (data_ != nullptr) { (void)cudaFree(data_); }
    }
    DeviceBytes(const DeviceBytes&)            = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;

    [[nodiscard]] void* get() const noexcept { return data_; }
    void upload(const std::vector<std::uint8_t>& bytes) const {
        CUDA_CHECK(cudaMemcpy(data_, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    }
    // cudaMemset is asynchronous with respect to the host and the rank streams are non-blocking, so
    // settle it before a transfer that would otherwise race the zeroing.
    void clear() const {
        CUDA_CHECK(cudaMemset(data_, 0, bytes_));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    [[nodiscard]] std::vector<std::uint8_t> download() const {
        std::vector<std::uint8_t> out(bytes_);
        CUDA_CHECK(cudaMemcpy(out.data(), data_, bytes_, cudaMemcpyDeviceToHost));
        return out;
    }

private:
    void* data_       = nullptr;
    std::size_t bytes_ = 0;
};

std::vector<cudaGraphNode_t> graph_nodes(cudaGraph_t graph) {
    std::size_t count = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    if (count != 0) { CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count)); }
    return nodes;
}

// The one memcpy node of `kind` that reads `source` or writes `destination`. Null if none matches
// or the identity is ambiguous.
cudaGraphNode_t find_memcpy(cudaGraph_t graph, cudaMemcpyKind kind, const void* source,
                            const void* destination) {
    cudaGraphNode_t found = nullptr;
    for (const cudaGraphNode_t node : graph_nodes(graph)) {
        cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemcpy) { continue; }
        cudaMemcpy3DParms params{};
        CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &params));
        if (params.kind != kind) { continue; }
        if (source != nullptr && params.srcPtr.ptr != source) { continue; }
        if (destination != nullptr && params.dstPtr.ptr != destination) { continue; }
        if (found != nullptr) { return nullptr; }
        found = node;
    }
    return found;
}

bool depends_on(cudaGraph_t graph, cudaGraphNode_t from, cudaGraphNode_t to) {
    std::size_t edges = 0;
    // CUDA 13 dropped the four-argument form: the edge-data array is passed explicitly.
#if CUDART_VERSION >= 13000
    CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges));
#else
    CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, &edges));
#endif
    std::vector<cudaGraphNode_t> sources(edges);
    std::vector<cudaGraphNode_t> destinations(edges);
    if (edges != 0) {
#if CUDART_VERSION >= 13000
        CUDA_CHECK(cudaGraphGetEdges(graph, sources.data(), destinations.data(), nullptr, &edges));
#else
        CUDA_CHECK(cudaGraphGetEdges(graph, sources.data(), destinations.data(), &edges));
#endif
    }
    std::deque<cudaGraphNode_t> pending{from};
    std::vector<cudaGraphNode_t> seen{from};
    while (!pending.empty()) {
        const cudaGraphNode_t node = pending.front();
        pending.pop_front();
        if (node == to) { return true; }
        for (std::size_t edge = 0; edge < edges; ++edge) {
            if (sources[edge] != node) { continue; }
            const cudaGraphNode_t next = destinations[edge];
            if (std::find(seen.begin(), seen.end(), next) != seen.end()) { continue; }
            seen.push_back(next);
            pending.push_back(next);
        }
    }
    return false;
}

// Payloads below, at and above the pipelining threshold, and one that does not divide evenly into
// pieces.
constexpr std::size_t kSizes[] = {4096, 100U << 10, 1U << 20, (3U << 20) + 37};
constexpr std::size_t kSlotBytes = (3U << 20) + 4096;

int run() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);
    CUDA_CHECK(cudaSetDevice(0));

    int failures = 0;

    // The context itself spans more than two ranks now.
    const int device_ids[3] = {0, 0, 0};
    ninfer::DeviceContext ctx(std::span<const int>(device_ids, 3));
    failures += expect(ctx.size() == 3, "a context holds three ranks");
    failures += expect(ctx.model_parallel(), "three ranks are model parallel");
    for (std::size_t rank = 0; rank < ctx.size(); ++rank) {
        failures += expect(ctx.rank(rank).index == rank, "a rank knows its own index");
        failures += expect(ctx.rank(rank).stream != nullptr, "each rank owns a compute stream");
    }
    failures += expect(ctx.rank(0).stream != ctx.rank(1).stream &&
                           ctx.rank(1).stream != ctx.rank(2).stream,
                       "ranks on one card still get distinct streams");
    failures += expect(ctx.same_physical_device(0, 2), "ranks sharing a card are grouped");
    failures += expect_throws([&] { (void)ctx.rank(3); }, "a rank past the end is rejected");

    // 1. Byte integrity, every transport and size, ring reused many times over.
    for (const bool force_staged : {true, false}) {
        ninfer::StageLink link(ctx, 0, 1,
                               {.slot_bytes = kSlotBytes, .slots = 2, .force_staged = force_staged});
        failures += expect(link.transport() == (force_staged ? ninfer::LinkTransport::StagedHost
                                                              : ninfer::LinkTransport::Local),
                           "transport follows the topology and the override");
        const cudaStream_t from = ctx.rank(0).stream;
        const cudaStream_t to   = ctx.rank(1).stream;
        for (const std::size_t bytes : kSizes) {
            DeviceBytes source(bytes);
            DeviceBytes destination(bytes);
            for (std::uint32_t round = 0; round < 8; ++round) {
                const auto sent = pattern(bytes, round + 1);
                source.upload(sent);
                destination.clear();
                link.send(source.get(), bytes, round % link.slots(), from);
                link.recv(destination.get(), bytes, round % link.slots(), to);
                ctx.synchronize();
                failures += expect_payload(destination.download(), sent,
                                           force_staged ? "staged transfer delivers its bytes"
                                                        : "local transfer delivers its bytes");
            }
        }
    }

    // 2. Two hops, 0 -> 1 -> 2, which is a three-stage pipeline's boundaries.
    {
        ninfer::StageLink first(ctx, 0, 1, {.slot_bytes = kSlotBytes, .force_staged = true});
        ninfer::StageLink second(ctx, 1, 2, {.slot_bytes = kSlotBytes, .force_staged = true});
        const std::size_t bytes = 1U << 20;
        DeviceBytes a(bytes), b(bytes), c(bytes);
        const auto sent = pattern(bytes, 77);
        a.upload(sent);
        b.clear();
        c.clear();
        first.send(a.get(), bytes, 0, ctx.rank(0).stream);
        first.recv(b.get(), bytes, 0, ctx.rank(1).stream);
        second.send(b.get(), bytes, 0, ctx.rank(1).stream);
        second.recv(c.get(), bytes, 0, ctx.rank(2).stream);
        ctx.synchronize();
        failures += expect_payload(c.download(), sent, "a payload crosses two stage boundaries");
    }

    // 3. Payload halves captured into separate graphs, fences issued eagerly between launches. This
    //    is the shape of a per-stage graph pipeline: no event ever crosses a graph.
    {
        ninfer::StageLink link(ctx, 0, 1, {.slot_bytes = kSlotBytes, .force_staged = true});
        const std::size_t bytes = 1U << 20;
        DeviceBytes source(bytes);
        DeviceBytes destination(bytes);
        const cudaStream_t from = ctx.rank(0).stream;
        const cudaStream_t to   = ctx.rank(1).stream;

        cudaGraph_t send_graph = nullptr;
        cudaGraph_t recv_graph = nullptr;
        // One graph per slot: the slot index is baked into the memcpy node.
        cudaGraphExec_t send_exec[2]{};
        cudaGraphExec_t recv_exec[2]{};
        for (std::size_t slot = 0; slot < 2; ++slot) {
            CUDA_CHECK(cudaStreamBeginCapture(from, cudaStreamCaptureModeThreadLocal));
            link.send_payload(source.get(), bytes, slot, from);
            CUDA_CHECK(cudaStreamEndCapture(from, &send_graph));
            CUDA_CHECK(cudaGraphInstantiate(&send_exec[slot], send_graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(send_graph));

            CUDA_CHECK(cudaStreamBeginCapture(to, cudaStreamCaptureModeThreadLocal));
            link.recv_payload(destination.get(), bytes, slot, to);
            CUDA_CHECK(cudaStreamEndCapture(to, &recv_graph));
            CUDA_CHECK(cudaGraphInstantiate(&recv_exec[slot], recv_graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(recv_graph));
        }

        for (std::uint32_t round = 0; round < 8; ++round) {
            const std::size_t slot = round % 2;
            const auto sent        = pattern(bytes, 500 + round);
            source.upload(sent);
            destination.clear();
            link.wait_drained(slot, from);
            CUDA_CHECK(cudaGraphLaunch(send_exec[slot], from));
            link.mark_filled(slot, from);
            link.wait_filled(slot, to);
            CUDA_CHECK(cudaGraphLaunch(recv_exec[slot], to));
            link.mark_drained(slot, to);
            ctx.synchronize();
            failures += expect_payload(destination.download(), sent,
                                       "payload halves delivered through separate graphs");
        }
        for (std::size_t slot = 0; slot < 2; ++slot) {
            CUDA_CHECK(cudaGraphExecDestroy(send_exec[slot]));
            CUDA_CHECK(cudaGraphExecDestroy(recv_exec[slot]));
        }
    }

    // 4. Reuse ordering, read off the graph. Two sends through the same slot with one receive
    //    between them: the second send's D2H must depend on the receive's H2D, or it could overwrite
    //    staging the receiver is still reading.
    {
        ninfer::StageLink link(ctx, 0, 1, {.slot_bytes = kSlotBytes, .force_staged = true});
        const std::size_t bytes = 1U << 20;
        DeviceBytes first_source(bytes), second_source(bytes), first_destination(bytes),
            second_destination(bytes);
        const auto first  = pattern(bytes, 900);
        const auto second = pattern(bytes, 901);
        first_source.upload(first);
        second_source.upload(second);
        first_destination.clear();
        second_destination.clear();

        const cudaStream_t from = ctx.rank(0).stream;
        const cudaStream_t to   = ctx.rank(1).stream;
        cudaEvent_t join        = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));

        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(from, cudaStreamCaptureModeThreadLocal));
        link.send(first_source.get(), bytes, 0, from);
        link.recv(first_destination.get(), bytes, 0, to);
        link.send(second_source.get(), bytes, 0, from);
        link.recv(second_destination.get(), bytes, 0, to);
        // The receiving rank forked into the capture through the fence; join it back so the
        // capture can close on its origin stream.
        CUDA_CHECK(cudaEventRecord(join, to));
        CUDA_CHECK(cudaStreamWaitEvent(from, join, 0));
        CUDA_CHECK(cudaStreamEndCapture(from, &graph));
        failures += expect(graph != nullptr, "two eager transfers capture into one graph");

        const cudaGraphNode_t first_upload =
            find_memcpy(graph, cudaMemcpyHostToDevice, nullptr, first_destination.get());
        const cudaGraphNode_t second_download =
            find_memcpy(graph, cudaMemcpyDeviceToHost, second_source.get(), nullptr);
        if (first_upload == nullptr || second_download == nullptr) {
            failures += expect(false, "the graph carries both transfers' first pieces");
        } else {
            failures += expect(depends_on(graph, first_upload, second_download),
                               "reusing a slot waits for the receiver to drain it");
        }

        cudaGraphExec_t executable = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, from));
        ctx.synchronize();
        failures += expect_payload(first_destination.download(), first,
                                   "the first captured transfer delivers its payload");
        failures += expect_payload(second_destination.download(), second,
                                   "the second captured transfer delivers its payload");
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaEventDestroy(join));
    }

    // 6. One slot used eagerly, then inside a capture, then eagerly again: the capture must not wait
    //    on an event that was last recorded outside it (cudaErrorStreamCaptureIsolation), and the
    //    eager use after it must still be ordered.
    {
        ninfer::StageLink link(ctx, 0, 1, {.slot_bytes = kSlotBytes, .force_staged = true});
        const std::size_t bytes = 1U << 20;
        DeviceBytes source(bytes);
        DeviceBytes destination(bytes);
        const cudaStream_t from = ctx.rank(0).stream;
        const cudaStream_t to   = ctx.rank(1).stream;
        cudaEvent_t join        = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));

        const auto eager_round = [&](std::uint32_t seed, const char* label) {
            const auto sent = pattern(bytes, seed);
            source.upload(sent);
            destination.clear();
            link.send(source.get(), bytes, 0, from);
            link.recv(destination.get(), bytes, 0, to);
            ctx.synchronize();
            failures += expect_payload(destination.download(), sent, label);
        };
        eager_round(700, "eager use before a capture delivers its bytes");

        cudaGraph_t graph = nullptr;
        const cudaError_t begin = cudaStreamBeginCapture(from, cudaStreamCaptureModeThreadLocal);
        failures += expect(begin == cudaSuccess, "capture begins after eager use of the slot");
        link.send(source.get(), bytes, 0, from);
        link.recv(destination.get(), bytes, 0, to);
        CUDA_CHECK(cudaEventRecord(join, to));
        CUDA_CHECK(cudaStreamWaitEvent(from, join, 0));
        const cudaError_t end = cudaStreamEndCapture(from, &graph);
        failures += expect(end == cudaSuccess && graph != nullptr,
                           "a capture over a slot last used eagerly does not wait on the eager fence");
        if (end == cudaSuccess && graph != nullptr) {
            cudaGraphExec_t exec = nullptr;
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            const auto sent = pattern(bytes, 701);
            source.upload(sent);
            destination.clear();
            CUDA_CHECK(cudaGraphLaunch(exec, from));
            ctx.synchronize();
            failures += expect_payload(destination.download(), sent,
                                       "the captured use delivers its bytes on replay");
            CUDA_CHECK(cudaGraphExecDestroy(exec));
            CUDA_CHECK(cudaGraphDestroy(graph));
        } else {
            (void)cudaGetLastError();
        }
        eager_round(702, "eager use after a capture delivers its bytes");
        CUDA_CHECK(cudaEventDestroy(join));
    }

    // 5. Misuse is refused rather than corrupting a neighbouring slot.
    {
        ninfer::StageLink link(ctx, 0, 1, {.slot_bytes = 4096, .slots = 2, .force_staged = true});
        DeviceBytes buffer(8192);
        failures += expect_throws(
            [&] { link.send(buffer.get(), 8192, 0, ctx.rank(0).stream); },
            "a payload larger than the slot is rejected");
        failures += expect_throws([&] { link.send(buffer.get(), 16, 2, ctx.rank(0).stream); },
                                  "a slot past the ring is rejected");
        failures += expect_throws(
            [&] {
                ninfer::StageLink bad(ctx, 0, 5, {.slot_bytes = 16});
            },
            "a rank outside the context is rejected");
        failures += expect_throws(
            [&] {
                ninfer::StageLink bad(ctx, 0, 1, {.slot_bytes = 0});
            },
            "an empty slot is rejected");
        ctx.synchronize();
    }

    if (failures != 0) { return 1; }
    std::cout << "stage link protocol holds\n";
    return 0;
}

} // namespace

int run_stage_link_test() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
