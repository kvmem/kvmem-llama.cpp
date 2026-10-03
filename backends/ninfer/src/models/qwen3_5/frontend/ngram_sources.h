#pragma once

#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Called only for tool-result text. Recognize consecutive numbered runs, never
// rewrite target messages. Preserve code indentation after exactly one separator.
inline std::vector<std::string> ngram_numbered_sources(std::string_view text) {
    struct Line {
        std::uint64_t number;
        std::string_view body;
        std::string_view style;
    };

    const auto parse = [](std::string_view line) -> std::optional<Line> {
        while (!line.empty() && line.front() == ' ') { line.remove_prefix(1); }
        if (!line.empty() && line.front() == 'L') { line.remove_prefix(1); }
        std::uint64_t number = 0;
        const auto parsed    = std::from_chars(line.data(), line.data() + line.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr == line.data()) { return std::nullopt; }
        auto rest = line.substr(static_cast<std::size_t>(parsed.ptr - line.data()));
        for (std::string_view delimiter : {": ", "\t", "\xe2\x86\x92", " | ", "| "}) {
            if (rest.starts_with(delimiter)) {
                return Line{number, rest.substr(delimiter.size()), delimiter};
            }
        }
        return std::nullopt;
    };
    std::vector<std::string> sources;
    std::string run;
    std::uint64_t previous = 0;
    std::string_view style;
    std::size_t count = 0;
    const auto flush  = [&] {
        if (count >= 3) { sources.push_back(std::move(run)); }
        run.clear();
        count = 0;
    };
    while (!text.empty()) {
        auto length        = text.find('\n');
        const bool newline = length != std::string_view::npos;
        if (!newline) { length = text.size(); }
        const auto parsed = parse(text.substr(0, length));
        if (!parsed) {
            flush();
        } else {
            if (count != 0 && (previous == std::numeric_limits<std::uint64_t>::max() ||
                               parsed->number != previous + 1 || parsed->style != style)) {
                flush();
            }
            run.append(parsed->body);
            if (newline) { run.push_back('\n'); }
            previous = parsed->number;
            style    = parsed->style;
            ++count;
        }
        text.remove_prefix(length + (newline ? 1 : 0));
    }
    flush();
    return sources;
}

} // namespace ninfer::models::qwen3_5::frontend
