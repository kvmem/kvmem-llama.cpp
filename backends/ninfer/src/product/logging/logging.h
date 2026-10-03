#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace spdlog {
class logger;
}

namespace ninfer::product {

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    Off,
};

enum class LogColorMode {
    Auto,
    Always,
    Never,
};

enum class LogPresentation {
    Service,
    Tool,
};

struct LoggingOptions {
    std::string logger_name;
    LogLevel level               = LogLevel::Info;
    LogColorMode color           = LogColorMode::Auto;
    LogPresentation presentation = LogPresentation::Service;
};

[[nodiscard]] LogLevel parse_log_level(std::string_view value);

// One transient terminal line coordinated with the operational stderr sink. It is enabled only
// when stderr is a terminal; redirected output remains persistent spdlog records only.
class TerminalProgress {
public:
    ~TerminalProgress();

    TerminalProgress(const TerminalProgress&)            = delete;
    TerminalProgress& operator=(const TerminalProgress&) = delete;
    TerminalProgress(TerminalProgress&&)                 = delete;
    TerminalProgress& operator=(TerminalProgress&&)      = delete;

    [[nodiscard]] bool enabled() const noexcept;
    void update(std::string line);
    void clear() noexcept;

private:
    friend class LoggingRuntime;
    struct Impl;
    explicit TerminalProgress(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

// A multi-line panel pinned beneath the scrolling operational log. Persistent records scroll above
// it; the panel is erased and redrawn around each record, so it stays at the bottom of the console.
// It is enabled only when stderr is an interactive terminal with VT cursor control and the logger
// shows info records; redirected output never contains it.
class TerminalPanel {
public:
    ~TerminalPanel();

    TerminalPanel(const TerminalPanel&)            = delete;
    TerminalPanel& operator=(const TerminalPanel&) = delete;
    TerminalPanel(TerminalPanel&&)                 = delete;
    TerminalPanel& operator=(TerminalPanel&&)      = delete;

    [[nodiscard]] bool enabled() const noexcept;
    void update(std::vector<std::string> lines);
    void clear() noexcept;

private:
    friend class LoggingRuntime;
    struct Impl;
    explicit TerminalPanel(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

// Visible terminal columns of a line: ANSI CSI sequences occupy none and each UTF-8 code point one.
[[nodiscard]] std::size_t terminal_display_width(std::string_view line) noexcept;
// Truncate a line to at most `columns` visible columns, keeping its escape sequences intact and
// resetting SGR attributes when a styled line is cut.
[[nodiscard]] std::string fit_terminal_line(std::string_view line, std::size_t columns);

// Application-owned operational logger lifetime. Construction does not mutate spdlog's global
// default logger or registry; producers receive and retain the returned explicit shared handle.
class LoggingRuntime {
public:
    explicit LoggingRuntime(LoggingOptions options);
    ~LoggingRuntime();

    LoggingRuntime(const LoggingRuntime&)            = delete;
    LoggingRuntime& operator=(const LoggingRuntime&) = delete;
    LoggingRuntime(LoggingRuntime&&)                 = delete;
    LoggingRuntime& operator=(LoggingRuntime&&)      = delete;

    [[nodiscard]] std::shared_ptr<spdlog::logger> logger() const noexcept;
    [[nodiscard]] std::shared_ptr<TerminalProgress> terminal_progress() const noexcept;
    [[nodiscard]] std::shared_ptr<TerminalPanel> terminal_panel() const noexcept;
    void flush() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::product
