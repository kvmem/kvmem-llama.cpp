#pragma once

// The Vision tower on CPU threads, for `--vision-residency cpu`. The encoder mirrors the device one
// op by op (vision.cpp) in FP32 over weights decoded at load; only its merged embeddings reach the
// device, staged per prefill chunk the way overlay results are. One item encodes at a time across
// the process, since each encode already spreads over every core.

#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load/vision_cpu.h"
#include "models/qwen3_5/program/vision_control.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// Encodes one media item: `patches` is the processor's BF16 [patches, patch_width] row-major
// input and the result the merged BF16 embeddings [merged, output_hidden] row-major, which is the
// device handoff's [output_hidden, merged] layout. `threads` bounds the CPU threads (0: all).
// `cancelled`, when given, is polled between layers; a cancelled encode returns an empty vector.
[[nodiscard]] std::vector<std::uint16_t>
encode_vision_on_cpu(const CpuVisionWeights& weights, std::span<const std::uint16_t> patches,
                     const VisionItemControl& control, unsigned threads = 0,
                     const std::atomic<bool>* cancelled = nullptr);

// One Vision prefill session's host encoder: at most one item in flight, encoded on a worker
// thread beside other lanes' decode and handed over as BF16 host embeddings. Two result buffers
// alternate, so the next item can encode while prefill still reads the previous one.
class CpuVisionSession {
public:
    explicit CpuVisionSession(std::shared_ptr<const CpuVisionWeights> weights);
    ~CpuVisionSession();

    CpuVisionSession(const CpuVisionSession&)            = delete;
    CpuVisionSession& operator=(const CpuVisionSession&) = delete;

    // Starts the item's encode on a worker. The payload is shared so the patches outlive a prompt
    // that releases its media first.
    void submit_item(std::shared_ptr<const PreparedMediaPayload> payload,
                     const VisionItemControl& control);

    [[nodiscard]] bool pending() const noexcept { return worker_.joinable(); }

    [[nodiscard]] bool item_ready() const noexcept {
        return pending() && finished_.load(std::memory_order_acquire);
    }

    // Waits for the submitted item and returns its embeddings, valid until the item after next
    // completes.
    [[nodiscard]] std::span<const std::byte> complete_item();
    // Synchronous form inside the caller's prefill unit.
    [[nodiscard]] std::span<const std::byte>
    encode_item(std::shared_ptr<const PreparedMediaPayload> payload,
                const VisionItemControl& control);

    [[nodiscard]] double encode_seconds() const noexcept { return encode_seconds_; }

private:
    std::shared_ptr<const CpuVisionWeights> weights_;
    std::thread worker_;
    std::atomic<bool> finished_{false};
    std::atomic<bool> cancelled_{false};
    std::vector<std::uint16_t> results_[2];
    unsigned slot_ = 0;
    std::exception_ptr error_;
    double worker_seconds_ = 0.0;
    double encode_seconds_ = 0.0;
};

} // namespace ninfer::models::qwen3_5::execution
