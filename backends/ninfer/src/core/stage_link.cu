#include "core/stage_link.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

// Errors here are recoverable startup or usage failures (a slot that cannot be allocated, a payload
// larger than the ring), so they throw instead of aborting the process the way CUDA_CHECK does.
void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    throw std::runtime_error(std::string("StageLink ") + what + ": " + cudaGetErrorName(err) +
                             ": " + cudaGetErrorString(err));
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        (void)cudaEventDestroy(event);
        event = nullptr;
    }
}

// The capture `stream` is part of, or 0 outside any capture.
unsigned long long capture_id(cudaStream_t stream) {
    unsigned long long id           = 0;
    cudaStreamCaptureStatus status  = cudaStreamCaptureStatusNone;
    check(cudaStreamGetCaptureInfo(stream, &status, &id), "cudaStreamGetCaptureInfo");
    return status == cudaStreamCaptureStatusActive ? id : 0;
}

std::size_t pieces_for(LinkTransport transport, std::size_t bytes) {
    // A device-to-device slot has no second copy to overlap with.
    if (transport == LinkTransport::Local || bytes < kLinkMinimumPipelinedBytes) { return 1; }
    return kLinkPipelineDepth;
}

} // namespace

StageLink::StageLink(const DeviceContext& context, std::size_t from_rank, std::size_t to_rank,
                     StageLinkOptions options)
    : from_rank_(from_rank), to_rank_(to_rank), slots_(options.slots),
      slot_bytes_(options.slot_bytes) {
    if (from_rank >= context.size() || to_rank >= context.size()) {
        throw std::out_of_range("StageLink rank is out of range");
    }
    if (slots_ == 0 || slot_bytes_ == 0) {
        throw std::invalid_argument("StageLink needs at least one slot of nonzero size");
    }
    from_device_ = context.rank(from_rank).device;
    to_device_   = context.rank(to_rank).device;
    transport_   = (from_device_ == to_device_ && !options.force_staged)
                       ? LinkTransport::Local
                       : LinkTransport::StagedHost;
    ring_.resize(slots_);

    try {
        for (Slot& slot : ring_) {
            if (transport_ == LinkTransport::StagedHost) {
                // Portable, so either rank's context treats it as pinned regardless of which one
                // was current when it was allocated.
                check(cudaHostAlloc(&slot.data, slot_bytes_, cudaHostAllocPortable),
                      "cudaHostAlloc");
            } else {
                DeviceBinding bind(to_device_);
                check(cudaMalloc(&slot.data, slot_bytes_), "cudaMalloc");
            }
            // Events cannot be created while a stream is capturing, so every fence a slot will
            // ever need exists up front. `filled` belongs to the sender's device and `drained` to
            // the receiver's: an event is recorded by the device that owns it.
            {
                DeviceBinding bind(from_device_);
                for (cudaEvent_t& event : slot.filled) {
                    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                          "cudaEventCreate(filled)");
                }
            }
            {
                DeviceBinding bind(to_device_);
                for (cudaEvent_t& event : slot.drained) {
                    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                          "cudaEventCreate(drained)");
                }
            }
        }
    } catch (...) {
        release();
        throw;
    }
}

StageLink::~StageLink() { release(); }

void StageLink::release() noexcept {
    for (Slot& slot : ring_) {
        {
            DeviceBinding bind(from_device_);
            for (cudaEvent_t& event : slot.filled) { destroy_event(event); }
        }
        {
            DeviceBinding bind(to_device_);
            for (cudaEvent_t& event : slot.drained) { destroy_event(event); }
            if (slot.data != nullptr && transport_ == LinkTransport::Local) {
                (void)cudaFree(slot.data);
                slot.data = nullptr;
            }
        }
        if (slot.data != nullptr) {
            (void)cudaFreeHost(slot.data);
            slot.data = nullptr;
        }
    }
    ring_.clear();
}

StageLink::StageLink(StageLink&& other) noexcept
    : from_rank_(other.from_rank_), to_rank_(other.to_rank_), from_device_(other.from_device_),
      to_device_(other.to_device_), transport_(other.transport_), slots_(other.slots_),
      slot_bytes_(other.slot_bytes_), ring_(std::move(other.ring_)) {
    other.ring_.clear();
    other.slots_ = 0;
}

StageLink& StageLink::operator=(StageLink&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    from_rank_   = other.from_rank_;
    to_rank_     = other.to_rank_;
    from_device_ = other.from_device_;
    to_device_   = other.to_device_;
    transport_   = other.transport_;
    slots_       = other.slots_;
    slot_bytes_  = other.slot_bytes_;
    ring_        = std::move(other.ring_);
    other.ring_.clear();
    other.slots_ = 0;
    return *this;
}

void StageLink::check_slot(std::size_t slot, std::size_t bytes) const {
    if (slot >= ring_.size()) { throw std::out_of_range("StageLink slot is out of range"); }
    if (bytes > slot_bytes_) {
        throw std::invalid_argument("StageLink payload of " + std::to_string(bytes) +
                                    " bytes exceeds the slot size of " +
                                    std::to_string(slot_bytes_));
    }
}

void StageLink::send_payload(const void* source, std::size_t bytes, std::size_t slot,
                             cudaStream_t from_stream) {
    check_slot(slot, bytes);
    const cudaMemcpyKind kind = transport_ == LinkTransport::StagedHost
                                    ? cudaMemcpyDeviceToHost
                                    : cudaMemcpyDeviceToDevice;
    DeviceBinding bind(from_device_);
    check(cudaMemcpyAsync(ring_[slot].data, source, bytes, kind, from_stream), "send_payload");
}

void StageLink::recv_payload(void* destination, std::size_t bytes, std::size_t slot,
                             cudaStream_t to_stream) {
    check_slot(slot, bytes);
    const cudaMemcpyKind kind = transport_ == LinkTransport::StagedHost
                                    ? cudaMemcpyHostToDevice
                                    : cudaMemcpyDeviceToDevice;
    DeviceBinding bind(to_device_);
    check(cudaMemcpyAsync(destination, ring_[slot].data, bytes, kind, to_stream), "recv_payload");
}

void StageLink::wait_drained(std::size_t slot, cudaStream_t from_stream) {
    check_slot(slot, 0);
    DeviceBinding bind(from_device_);
    // An event that was never recorded is already satisfied, so the first use of a slot is free. One
    // recorded where this stream cannot legally wait on it (see the class comment) is skipped.
    const unsigned long long capture = capture_id(from_stream);
    for (std::size_t piece = 0; piece < ring_[slot].pieces; ++piece) {
        if (ring_[slot].drained_capture[piece] != capture) { continue; }
        check(cudaStreamWaitEvent(from_stream, ring_[slot].drained[piece], 0), "wait_drained");
    }
}

void StageLink::mark_filled(std::size_t slot, cudaStream_t from_stream) {
    check_slot(slot, 0);
    ring_[slot].pieces = 1;
    DeviceBinding bind(from_device_);
    check(cudaEventRecord(ring_[slot].filled[0], from_stream), "mark_filled");
}

void StageLink::wait_filled(std::size_t slot, cudaStream_t to_stream) {
    check_slot(slot, 0);
    DeviceBinding bind(to_device_);
    for (std::size_t piece = 0; piece < ring_[slot].pieces; ++piece) {
        check(cudaStreamWaitEvent(to_stream, ring_[slot].filled[piece], 0), "wait_filled");
    }
}

void StageLink::mark_drained(std::size_t slot, cudaStream_t to_stream) {
    check_slot(slot, 0);
    DeviceBinding bind(to_device_);
    // Every piece the sender used is recorded, so the next `wait_drained` covers all of them.
    const unsigned long long capture = capture_id(to_stream);
    for (std::size_t piece = 0; piece < ring_[slot].pieces; ++piece) {
        check(cudaEventRecord(ring_[slot].drained[piece], to_stream), "mark_drained");
        ring_[slot].drained_capture[piece] = capture;
    }
}

void StageLink::send(const void* source, std::size_t bytes, std::size_t slot,
                     cudaStream_t from_stream) {
    check_slot(slot, bytes);
    wait_drained(slot, from_stream);

    Slot& target                  = ring_[slot];
    const std::size_t pieces      = pieces_for(transport_, bytes);
    target.pieces                 = pieces;
    const std::size_t piece_bytes = (bytes + pieces - 1) / pieces;
    const cudaMemcpyKind kind     = transport_ == LinkTransport::StagedHost
                                        ? cudaMemcpyDeviceToHost
                                        : cudaMemcpyDeviceToDevice;
    DeviceBinding bind(from_device_);
    for (std::size_t piece = 0; piece < pieces; ++piece) {
        const std::size_t offset = std::min(piece * piece_bytes, bytes);
        const std::size_t length = std::min(piece_bytes, bytes - offset);
        if (length != 0) {
            check(cudaMemcpyAsync(static_cast<std::byte*>(target.data) + offset,
                                  static_cast<const std::byte*>(source) + offset, length, kind,
                                  from_stream),
                  "send");
        }
        check(cudaEventRecord(target.filled[piece], from_stream), "send fence");
    }
}

void StageLink::recv(void* destination, std::size_t bytes, std::size_t slot,
                     cudaStream_t to_stream) {
    check_slot(slot, bytes);

    Slot& source                  = ring_[slot];
    const std::size_t pieces      = source.pieces;
    const std::size_t piece_bytes = (bytes + pieces - 1) / pieces;
    const cudaMemcpyKind kind     = transport_ == LinkTransport::StagedHost
                                        ? cudaMemcpyHostToDevice
                                        : cudaMemcpyDeviceToDevice;
    DeviceBinding bind(to_device_);
    for (std::size_t piece = 0; piece < pieces; ++piece) {
        const std::size_t offset = std::min(piece * piece_bytes, bytes);
        const std::size_t length = std::min(piece_bytes, bytes - offset);
        check(cudaStreamWaitEvent(to_stream, source.filled[piece], 0), "recv wait");
        // Read after the wait: only then is `to_stream` part of the capture the send was in.
        const unsigned long long capture = capture_id(to_stream);
        if (length != 0) {
            check(cudaMemcpyAsync(static_cast<std::byte*>(destination) + offset,
                                  static_cast<const std::byte*>(source.data) + offset, length,
                                  kind, to_stream),
                  "recv");
        }
        // Released piece by piece, so the sender's next use of this slot can start as soon as the
        // last piece has been read rather than when the whole transfer is done.
        check(cudaEventRecord(source.drained[piece], to_stream), "recv fence");
        source.drained_capture[piece] = capture;
    }
}

} // namespace ninfer
