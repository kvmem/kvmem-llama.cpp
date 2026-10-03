#include "product/logging/logging.h"
#include "serve/console_stats.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

GenerationOutcome outcome(int prompt, std::uint32_t cache_hit, int completion, double ttft,
                          double prefill_seconds, double decode_seconds) {
    GenerationOutcome result;
    result.prompt_tokens                   = prompt;
    result.completion_tokens               = completion;
    result.metrics.prefix_cache_hit_tokens = cache_hit;
    result.metrics.ttft_seconds            = ttft;
    result.metrics.prefill_seconds         = prefill_seconds;
    result.metrics.decode_seconds          = decode_seconds;
    return result;
}

} // namespace

int main() {
    int failures = 0;

    // Engine speculative totals include n-gram rounds; the model drafter gets only the remainder.
    GenerationOutcome mixed                     = outcome(1000, 600, 101, 0.25, 0.2, 2.0);
    mixed.metrics.speculative_backend           = ninfer::SpeculativeBackend::DFlash2;
    mixed.metrics.speculative_rounds            = 30;
    mixed.metrics.speculative_draft_tokens      = 300;
    mixed.metrics.speculative_accepted_tokens   = 150;
    mixed.metrics.ngram_rounds                  = 10;
    mixed.metrics.ngram_drafted_tokens          = 100;
    mixed.metrics.ngram_accepted_tokens         = 70;
    mixed.metrics.ngram_archive_drafted_tokens  = 40;
    mixed.metrics.ngram_archive_accepted_tokens = 10;
    const ConsoleRequestSample sample           = make_console_request_sample(mixed);
    failures += check(sample.computed_prefill_tokens == 400,
                      "computed prefill must exclude cache-hit prompt tokens");
    failures += check(sample.decode_tokens == 100,
                      "decode tokens must exclude the first token emitted by prefill");
    failures += check(sample.model_rounds == 20 && sample.model_drafted_tokens == 200 &&
                          sample.model_accepted_tokens == 80,
                      "model-drafter totals must exclude n-gram rounds");
    failures += check(sample.ngram_rounds == 10 && sample.ngram_drafted_tokens == 100 &&
                          sample.ngram_accepted_tokens == 70,
                      "n-gram totals must be carried unchanged");
    failures += check(sample.speculative_backend == ninfer::SpeculativeBackend::DFlash2,
                      "a request with model drafts must name its drafter");

    GenerationOutcome ngram_only                = outcome(10, 0, 5, 0.1, 0.01, 0.1);
    ngram_only.metrics.speculative_backend      = ninfer::SpeculativeBackend::Mtp;
    ngram_only.metrics.speculative_rounds       = 2;
    ngram_only.metrics.speculative_draft_tokens = 8;
    ngram_only.metrics.ngram_rounds             = 2;
    ngram_only.metrics.ngram_drafted_tokens     = 8;
    failures += check(make_console_request_sample(ngram_only).speculative_backend ==
                          ninfer::SpeculativeBackend::None,
                      "a request without model drafts must not name a drafter");

    // Ratios and rates weigh requests by tokens and time; zero-duration phases do not dilute them.
    ConsoleStatsTotals totals;
    totals.add(make_console_request_sample(outcome(1000, 900, 11, 0.1, 0.05, 0.5)));
    totals.add(make_console_request_sample(outcome(3000, 0, 1, 0.3, 1.0, 0.0)));
    totals.add(make_console_request_sample(outcome(500, 500, 21, 0.2, 0.0, 1.0)));
    failures += check(totals.requests == 3, "totals must count every completed request");
    failures += check(totals.sum.prompt_tokens == 4500 && totals.sum.cache_hit_tokens == 1400,
                      "cache ratio must be token weighted");
    failures += check(totals.sum.computed_prefill_tokens == 3100 &&
                          std::abs(totals.sum.prefill_seconds - 1.05) < 1e-12,
                      "prefill rate must pair tokens only with requests that ran prefill");
    failures += check(totals.sum.decode_tokens == 30 && totals.sum.decode_seconds == 1.5,
                      "decode rate must pair tokens only with requests that ran decode");

    ConsoleStatsSnapshot snapshot;
    snapshot.completed           = 1;
    snapshot.failed              = 2;
    snapshot.running             = 1;
    snapshot.waiting             = 0;
    snapshot.recent_window       = 10;
    snapshot.speculative_backend = ninfer::SpeculativeBackend::DFlash2;
    snapshot.session.add(sample);
    snapshot.recent                      = snapshot.session;
    const std::vector<std::string> lines = render_console_stats_panel(snapshot);
    failures += check(lines.size() == 3, "a session inside the recent window shows one data row");
    if (lines.size() == 3) {
        failures += check(contains(lines[0], "1 done, 2 failed") &&
                              contains(lines[0], "running 1 | waiting 0"),
                          "title must carry request outcomes and live occupancy");
        failures += check(contains(lines[0], "rates in tok/s"),
                          "title must name the unit the rate cells omit");
        failures += check(contains(lines[1], "DFLASH2") && contains(lines[1], "archive"),
                          "headings must name the drafter and show the archive when used");
        const std::string& row = lines[2];
        failures += check(contains(row, "250 ms"), "row must show the mean TTFT");
        failures += check(contains(row, "60.0%"), "row must show the cache-hit ratio");
        failures += check(contains(row, " 2.00k ") && !contains(row, "tok/s"),
                          "row must show the prefill rate without repeating its unit");
        failures += check(contains(row, " 50.0 "), "row must show the decode rate");
        failures += check(contains(row, "40.0%") && contains(row, "4.00"),
                          "row must show model-drafter acceptance and accepted per round");
        failures += check(contains(row, "70.0%") && contains(row, "25.0%"),
                          "row must show n-gram and archive acceptance");
        // Every column shown, the table still fits a console snapped to half of a 1920-pixel
        // screen.
        for (std::size_t index = 1; index < lines.size(); ++index) {
            failures += check(ninfer::product::terminal_display_width(lines[index]) <= 86,
                              "panel table must fit a half-width console");
        }
    }

    // The recent row appears once the session outgrows the window and covers only its requests.
    const auto logging = std::make_unique<ninfer::product::LoggingRuntime>(
        ninfer::product::LoggingOptions{.logger_name = "console-stats-test"});
    ConsoleStatsPanel panel(logging->terminal_panel());
    for (int index = 0; index < 12; ++index) {
        panel.request_done(outcome(100, 0, 2, index < 2 ? 5.0 : 0.5, 0.1, 0.1));
    }
    panel.request_failure(
        RequestFailure{.classification = RequestFailureClass::ClientDisconnected});
    panel.request_rejected(RequestFailure{.classification = RequestFailureClass::Overload});
    const ConsoleStatsSnapshot rolled = panel.snapshot();
    failures += check(rolled.completed == 12 && rolled.session.requests == 12,
                      "session totals must keep every request");
    failures += check(rolled.recent.requests == 10 && rolled.recent.sum.ttft_seconds == 5.0,
                      "recent totals must hold only the last requests");
    failures += check(rolled.cancelled == 1 && rolled.rejected == 1 && rolled.failed == 0,
                      "disconnects count as cancelled and admission refusals as rejected");
    const std::vector<std::string> rolled_lines = render_console_stats_panel(rolled);
    failures += check(rolled_lines.size() == 4 && contains(rolled_lines[3], "last 10"),
                      "a session beyond the window shows the recent row");

    // Terminal fitting counts visible columns only and never splits an escape sequence.
    using ninfer::product::fit_terminal_line;
    using ninfer::product::terminal_display_width;
    const std::string styled = "\x1b[1mabcdef\x1b[0m";
    failures += check(terminal_display_width(styled) == 6, "escapes must occupy no columns");
    failures += check(fit_terminal_line(styled, 3) == "\x1b[1mabc\x1b[0m",
                      "a cut styled line must keep its escape and reset attributes");
    failures += check(fit_terminal_line(styled, 6) == styled, "a fitting line must be unchanged");
    failures += check(terminal_display_width("a\xe2\x94\x80z") == 3,
                      "a UTF-8 code point must occupy one column");
    failures += check(fit_terminal_line("a\xe2\x94\x80z", 2) == "a\xe2\x94\x80",
                      "fitting must not split a UTF-8 code point");

    if (failures == 0) { std::cout << "console stats tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
