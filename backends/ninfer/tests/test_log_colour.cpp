#include "product/log_colour/log_colour.h"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

using ninfer::product::log_colour::LogFamily;
using ninfer::product::log_colour::colourise_stats_line;
using ninfer::product::log_colour::family_colour_slot;
using ninfer::product::log_colour::family_for;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool is_plain(const std::string& text) { return text.find('\033') == std::string::npos; }

// The SGR spec (between ESC[ and the trailing m) of the colour wrap immediately
// preceding token, or nullopt when the token is not wrapped. Deliberately
// palette-agnostic: the spec's value depends on the console the test runs in
// (256-colour vs basic 16-colour), but its presence, equality and inequality do
// not.
std::optional<std::string> sgr_spec(const std::string& output, std::string_view token) {
    const std::size_t token_at = output.find(token);
    if (token_at == std::string::npos || token_at == 0 || output[token_at - 1] != 'm') {
        return std::nullopt;
    }
    const std::size_t open = output.rfind("\033[", token_at - 1);
    if (open == std::string::npos) { return std::nullopt; }
    return output.substr(open + 2, token_at - 1 - (open + 2));
}

} // namespace

int main() {
    int failures = 0;

    // Family classification of the current operational line prefixes.
    failures += check(
        family_for("throughput | 5.0s | prefill 2,400.0 tok/s") == LogFamily::General,
        "throughput line is not General");
    failures += check(
        family_for("capacity | KV 1,000 tokens, int8, auto") == LogFamily::General,
        "capacity line is not General");
    failures += check(
        family_for("req#42 done | openai-chat | stop token | prompt 1,200") == LogFamily::Done,
        "request done line is not Done");
    failures += check(
        family_for("req#42 started | openai-chat stream | 1 message") == LogFamily::None,
        "request started line is not None");
    failures += check(
        family_for("req#42 rejected during prepare | openai-chat stream | HTTP 413") ==
            LogFamily::General,
        "request rejected line is not General");
    failures += check(
        family_for("req#42 failed during generation | anthropic | HTTP 500") == LogFamily::General,
        "request failed line is not General");
    failures += check(
        family_for("req#42 response failed during transport") == LogFamily::General,
        "response failure line is not General");
    failures += check(family_for("host-cache 2 states") == LogFamily::Cache,
                      "host-cache line is not Cache");

    // Disabled colouring returns the line untouched.
    const std::string_view throughput =
        "throughput | 5.0s | prefill 2,400.0 tok/s (1,200 tok) | decode 45.2 tok/s (200 tok) | "
        "running 1 | host 12% (600ms)";
    failures += check(colourise_stats_line(throughput, false) == std::string(throughput),
                      "disabled colouring modified the line");

    // A request-start (settings) line stays plain even when colouring is enabled.
    const std::string_view started =
        "req#7 started | openai-chat stream | 3 messages | max output 1,200 | thinking off";
    failures += check(colourise_stats_line(started, true) == std::string(started),
                      "request started line was coloured");

    const std::string general = colourise_stats_line(throughput, true);
    // The line header stays plain; the first statistic after it is coloured.
    const std::size_t first_clause = general.find(" | ");
    failures += check(first_clause != std::string::npos && is_plain(general.substr(0, first_clause)),
                      "throughput line header was coloured");
    const auto interval      = sgr_spec(general, "5.0s");
    const auto prefill       = sgr_spec(general, "prefill");
    const auto prefill_value = sgr_spec(general, "2,400.0");
    failures += check(interval.has_value(), "throughput interval is not coloured");
    failures += check(prefill.has_value(), "throughput prefill statistic is not coloured");
    failures += check(prefill_value.has_value() && prefill.has_value() &&
                          *prefill_value == *prefill,
                      "prefill value does not inherit the prefill colour");

    // A done line: every clause is coloured by its clause-opening name, and the
    // legacy key=value vision-offload tail keeps its own colours.
    const std::string_view done =
        "req#42 done | openai-chat | stop token | prompt 1,200 | output 350 | cache 900 (75%) | "
        "TTFT 200ms | total 5.0s | prefill 6,000.0 tok/s | decode 45.2 tok/s | "
        "vision_offload_ms=123.4 vision_offload_evicted_mib=5";
    const std::string done_coloured = colourise_stats_line(done, true);
    const auto done_prefill         = sgr_spec(done_coloured, "prefill");
    const auto done_prompt          = sgr_spec(done_coloured, "prompt");
    const auto offload              = sgr_spec(done_coloured, "vision_offload_ms=123.4");
    failures += check(sgr_spec(done_coloured, "req#42").has_value() == false,
                      "done line header was coloured");
    failures += check(done_prefill.has_value(), "done line prefill statistic is not coloured");
    failures += check(done_prompt.has_value(), "done line prompt statistic is not coloured");
    failures += check(offload.has_value(), "legacy key=value tail of a done line is not coloured");

    // Families own disjoint kPalette blocks, so the same statistic name uses a
    // different colour region on a done line than on a throughput line.
    const std::size_t general_slot = family_colour_slot("prefill", LogFamily::General);
    const std::size_t done_slot    = family_colour_slot("prefill", LogFamily::Done);
    failures += check(general_slot < 16 && done_slot >= 32 && general_slot != done_slot,
                      "family colour blocks are not disjoint");

    // Stability: the same statistic name keeps its colour across lines.
    const auto prefill_again = sgr_spec(colourise_stats_line(throughput, true), "prefill");
    failures += check(prefill_again.has_value() && prefill.has_value() &&
                          *prefill_again == *prefill,
                      "statistic colour is not stable across lines");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
