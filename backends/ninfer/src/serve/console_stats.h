#pragma once

// Session statistics panel pinned beneath the scrolling operational console log. It aggregates the
// same completed-request measurements the `req#N done` records print, over the whole session and
// over the most recent requests, and redraws product::TerminalPanel after every change.

#include "serve/request_events.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::product {
class TerminalPanel;
}

namespace ninfer::serve {

// One completed request's contribution. Speculative totals from the Engine include n-gram rounds;
// the model fields here are the remainder (MTP or DFlash drafting), so the two sources never double
// count.
struct ConsoleRequestSample {
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    double ttft_seconds                    = 0.0;
    std::uint64_t prompt_tokens            = 0;
    std::uint64_t cache_hit_tokens         = 0;
    std::uint64_t computed_prefill_tokens  = 0;
    double prefill_seconds                 = 0.0;
    std::uint64_t decode_tokens            = 0;
    double decode_seconds                  = 0.0;
    std::uint64_t model_rounds             = 0;
    std::uint64_t model_drafted_tokens     = 0;
    std::uint64_t model_accepted_tokens    = 0;
    std::uint64_t ngram_rounds             = 0;
    std::uint64_t ngram_drafted_tokens     = 0;
    std::uint64_t ngram_accepted_tokens    = 0;
    std::uint64_t archive_drafted_tokens   = 0;
    std::uint64_t archive_accepted_tokens  = 0;
};

[[nodiscard]] ConsoleRequestSample make_console_request_sample(const GenerationOutcome& outcome);

// Sums over a set of completed requests. Rates and ratios are formed from the sums, so each request
// weighs by its tokens and time rather than counting as one equal vote.
struct ConsoleStatsTotals {
    std::uint64_t requests = 0;
    ConsoleRequestSample sum;

    void add(const ConsoleRequestSample& sample) noexcept;
};

struct ConsoleStatsSnapshot {
    std::uint64_t completed = 0;
    std::uint64_t failed    = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t rejected  = 0;
    std::optional<std::uint32_t> running;
    std::optional<std::uint32_t> waiting;
    std::size_t recent_window = 0;
    // Model drafter that produced the speculative columns; None until a request used one.
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    ConsoleStatsTotals session;
    ConsoleStatsTotals recent;
};

// Pure rendering of the panel rows; the terminal fits them to its width.
[[nodiscard]] std::vector<std::string>
render_console_stats_panel(const ConsoleStatsSnapshot& snapshot);

class ConsoleStatsPanel {
public:
    static constexpr std::size_t kRecentRequests = 10;

    explicit ConsoleStatsPanel(std::shared_ptr<product::TerminalPanel> panel);

    void request_done(const GenerationOutcome& outcome);
    void request_failure(const RequestFailure& failure);
    void request_rejected(const RequestFailure& failure);
    void runtime(const ninfer::RuntimeStats& current);
    void show();

    [[nodiscard]] ConsoleStatsSnapshot snapshot() const;

private:
    void publish_locked();

    std::shared_ptr<product::TerminalPanel> panel_;
    mutable std::mutex mutex_;
    ConsoleStatsSnapshot state_;
    std::deque<ConsoleRequestSample> recent_;
};

} // namespace ninfer::serve
