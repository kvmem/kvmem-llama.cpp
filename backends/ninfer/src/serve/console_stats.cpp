#include "serve/console_stats.h"

#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::string_view kBold  = "\x1b[1m";
constexpr std::string_view kDim   = "\x1b[2m";
constexpr std::string_view kReset = "\x1b[0m";
constexpr std::string_view kNone  = "-";

std::uint64_t saturating_sub(std::uint64_t total, std::uint64_t part) noexcept {
    return total >= part ? total - part : 0;
}

std::string ratio_cell(std::uint64_t part, std::uint64_t whole) {
    if (whole == 0) { return std::string(kNone); }
    return product::format_pretty_percent(static_cast<double>(part) / static_cast<double>(whole));
}

// The title names the tok/s unit once, so a rate cell keeps only its scaled number ("7.46k").
std::string rate_cell(std::uint64_t tokens, double seconds) {
    if (seconds <= 0.0) { return std::string(kNone); }
    std::string text = product::format_pretty_rate(static_cast<double>(tokens) / seconds, "tok");
    constexpr std::string_view kUnit = " tok/s";
    if (std::string_view(text).ends_with(kUnit)) { text.resize(text.size() - kUnit.size()); }
    return text;
}

std::string per_round_cell(std::uint64_t accepted, std::uint64_t rounds) {
    if (rounds == 0) { return std::string(kNone); }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f",
                  static_cast<double>(accepted) / static_cast<double>(rounds));
    return buffer;
}

struct Column {
    std::string_view heading;
    std::size_t width;
};

// The row label is left-aligned; every value column is right-aligned under its heading. The
// table is 77 columns wide (86 with the archive column), so it fits a console window snapped to
// half of a 1920-pixel screen.
constexpr std::size_t kLabelWidth   = 7;
constexpr std::size_t kArchiveWidth = 9;
constexpr Column kColumns[]         = {
    {"TTFT", 8},  {"cached", 8},  {"prefill", 9}, {"decode", 8},
    {"draft", 9}, {"acc/rnd", 9}, {"ngram", 8},   {"ng rnds", 10},
};
constexpr std::size_t kColumnCount   = std::size(kColumns);
constexpr std::size_t kDrafterColumn = 4; // heading names the configured model drafter

void append_right(std::string& out, std::string_view text, std::size_t width) {
    if (text.size() < width) { out.append(width - text.size(), ' '); }
    out.append(text);
}

void append_left(std::string& out, std::string_view text, std::size_t width) {
    out.append(text);
    if (text.size() < width) { out.append(width - text.size(), ' '); }
}

std::string drafter_heading(SpeculativeBackend backend) {
    if (backend == SpeculativeBackend::None) { return "draft"; }
    std::string heading = product::speculative_backend_name(backend);
    for (char& ch : heading) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return heading;
}

std::string render_row(std::string_view label, const ConsoleStatsTotals& totals,
                       bool show_archive) {
    const ConsoleRequestSample& sum       = totals.sum;
    const std::string cells[kColumnCount] = {
        totals.requests == 0 ? std::string(kNone)
                             : product::format_pretty_duration(
                                   sum.ttft_seconds / static_cast<double>(totals.requests)),
        ratio_cell(sum.cache_hit_tokens, sum.prompt_tokens),
        rate_cell(sum.computed_prefill_tokens, sum.prefill_seconds),
        rate_cell(sum.decode_tokens, sum.decode_seconds),
        ratio_cell(sum.model_accepted_tokens, sum.model_drafted_tokens),
        per_round_cell(sum.model_accepted_tokens, sum.model_rounds),
        ratio_cell(sum.ngram_accepted_tokens, sum.ngram_drafted_tokens),
        sum.ngram_rounds == 0 ? std::string(kNone) : product::format_pretty_count(sum.ngram_rounds),
    };
    std::string row = " ";
    append_left(row, label, kLabelWidth);
    for (std::size_t index = 0; index < kColumnCount; ++index) {
        append_right(row, cells[index], kColumns[index].width);
    }
    if (show_archive) {
        append_right(row, ratio_cell(sum.archive_accepted_tokens, sum.archive_drafted_tokens),
                     kArchiveWidth);
    }
    return row;
}

} // namespace

ConsoleRequestSample make_console_request_sample(const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const auto prompt_tokens = static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0));
    const std::uint64_t cache_hit_tokens = metrics.prefix_cache_hit_tokens;
    const SpeculativeBackend backend =
        metrics.speculative_draft_tokens > metrics.ngram_drafted_tokens
            ? metrics.speculative_backend
            : SpeculativeBackend::None;
    return ConsoleRequestSample{
        .speculative_backend     = backend,
        .ttft_seconds            = metrics.ttft_seconds,
        .prompt_tokens           = prompt_tokens,
        .cache_hit_tokens        = cache_hit_tokens,
        .computed_prefill_tokens = saturating_sub(prompt_tokens, cache_hit_tokens),
        .prefill_seconds         = metrics.prefill_seconds,
        // The first output token comes from prefill; the decode rate covers the rest, as the
        // request-done record does.
        .decode_tokens  = outcome.completion_tokens > 0
                              ? static_cast<std::uint64_t>(outcome.completion_tokens - 1)
                              : 0,
        .decode_seconds = metrics.decode_seconds,
        .model_rounds   = saturating_sub(metrics.speculative_rounds, metrics.ngram_rounds),
        .model_drafted_tokens =
            saturating_sub(metrics.speculative_draft_tokens, metrics.ngram_drafted_tokens),
        .model_accepted_tokens =
            saturating_sub(metrics.speculative_accepted_tokens, metrics.ngram_accepted_tokens),
        .ngram_rounds            = metrics.ngram_rounds,
        .ngram_drafted_tokens    = metrics.ngram_drafted_tokens,
        .ngram_accepted_tokens   = metrics.ngram_accepted_tokens,
        .archive_drafted_tokens  = metrics.ngram_archive_drafted_tokens,
        .archive_accepted_tokens = metrics.ngram_archive_accepted_tokens,
    };
}

void ConsoleStatsTotals::add(const ConsoleRequestSample& sample) noexcept {
    ++requests;
    sum.ttft_seconds += sample.ttft_seconds;
    sum.prompt_tokens += sample.prompt_tokens;
    sum.cache_hit_tokens += sample.cache_hit_tokens;
    // Rates pair tokens with time only where the phase ran, so a request that reused its whole
    // prompt or stopped after one token does not dilute the rate with zero-duration phases.
    if (sample.prefill_seconds > 0.0) {
        sum.computed_prefill_tokens += sample.computed_prefill_tokens;
        sum.prefill_seconds += sample.prefill_seconds;
    }
    if (sample.decode_seconds > 0.0) {
        sum.decode_tokens += sample.decode_tokens;
        sum.decode_seconds += sample.decode_seconds;
    }
    sum.model_rounds += sample.model_rounds;
    sum.model_drafted_tokens += sample.model_drafted_tokens;
    sum.model_accepted_tokens += sample.model_accepted_tokens;
    sum.ngram_rounds += sample.ngram_rounds;
    sum.ngram_drafted_tokens += sample.ngram_drafted_tokens;
    sum.ngram_accepted_tokens += sample.ngram_accepted_tokens;
    sum.archive_drafted_tokens += sample.archive_drafted_tokens;
    sum.archive_accepted_tokens += sample.archive_accepted_tokens;
}

std::vector<std::string> render_console_stats_panel(const ConsoleStatsSnapshot& snapshot) {
    const bool show_archive = snapshot.session.sum.archive_drafted_tokens != 0;
    std::size_t table_width = 1 + kLabelWidth + (show_archive ? kArchiveWidth : 0);
    for (const Column& column : kColumns) { table_width += column.width; }

    std::string title = "-- session stats, rates in tok/s | " +
                        product::format_pretty_count(snapshot.completed) + " done";
    if (snapshot.failed != 0) {
        title += ", " + product::format_pretty_count(snapshot.failed) + " failed";
    }
    if (snapshot.cancelled != 0) {
        title += ", " + product::format_pretty_count(snapshot.cancelled) + " cancelled";
    }
    if (snapshot.rejected != 0) {
        title += ", " + product::format_pretty_count(snapshot.rejected) + " rejected";
    }
    if (snapshot.running) { title += " | running " + std::to_string(*snapshot.running); }
    if (snapshot.waiting) { title += " | waiting " + std::to_string(*snapshot.waiting); }
    title += ' ';
    if (title.size() < table_width) { title.append(table_width - title.size(), '-'); }

    std::string headings = " ";
    append_left(headings, "", kLabelWidth);
    for (std::size_t index = 0; index < kColumnCount; ++index) {
        const Column& column = kColumns[index];
        append_right(headings,
                     index == kDrafterColumn ? drafter_heading(snapshot.speculative_backend)
                                             : std::string(column.heading),
                     column.width);
    }
    if (show_archive) { append_right(headings, "archive", kArchiveWidth); }

    std::vector<std::string> lines;
    lines.reserve(4);
    lines.push_back(std::string(kBold) + title + std::string(kReset));
    lines.push_back(std::string(kDim) + headings + std::string(kReset));
    lines.push_back(render_row("session", snapshot.session, show_archive));
    // Until the session outgrows the window, the recent row would repeat the session row.
    if (snapshot.completed > snapshot.recent_window) {
        lines.push_back(render_row("last " + std::to_string(snapshot.recent.requests),
                                   snapshot.recent, show_archive));
    }
    return lines;
}

ConsoleStatsPanel::ConsoleStatsPanel(std::shared_ptr<product::TerminalPanel> panel)
    : panel_(std::move(panel)) {
    state_.recent_window = kRecentRequests;
}

void ConsoleStatsPanel::request_done(const GenerationOutcome& outcome) {
    const ConsoleRequestSample sample = make_console_request_sample(outcome);
    std::lock_guard lock(mutex_);
    ++state_.completed;
    if (sample.speculative_backend != SpeculativeBackend::None) {
        state_.speculative_backend = sample.speculative_backend;
    }
    state_.session.add(sample);
    recent_.push_back(sample);
    if (recent_.size() > kRecentRequests) { recent_.pop_front(); }
    state_.recent = {};
    for (const ConsoleRequestSample& entry : recent_) { state_.recent.add(entry); }
    publish_locked();
}

void ConsoleStatsPanel::request_failure(const RequestFailure& failure) {
    std::lock_guard lock(mutex_);
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        ++state_.cancelled;
    } else {
        ++state_.failed;
    }
    publish_locked();
}

void ConsoleStatsPanel::request_rejected(const RequestFailure& failure) {
    std::lock_guard lock(mutex_);
    if (failure.classification == RequestFailureClass::ClientDisconnected) {
        ++state_.cancelled;
    } else {
        ++state_.rejected;
    }
    publish_locked();
}

void ConsoleStatsPanel::runtime(const ninfer::RuntimeStats& current) {
    std::lock_guard lock(mutex_);
    if (state_.running == current.running_requests && state_.waiting == current.waiting_requests) {
        return;
    }
    state_.running = current.running_requests;
    state_.waiting = current.waiting_requests;
    publish_locked();
}

void ConsoleStatsPanel::show() {
    std::lock_guard lock(mutex_);
    publish_locked();
}

ConsoleStatsSnapshot ConsoleStatsPanel::snapshot() const {
    std::lock_guard lock(mutex_);
    return state_;
}

void ConsoleStatsPanel::publish_locked() {
    // Publishing under the statistics lock keeps concurrent updates from drawing out of order.
    panel_->update(render_console_stats_panel(state_));
}

} // namespace ninfer::serve
