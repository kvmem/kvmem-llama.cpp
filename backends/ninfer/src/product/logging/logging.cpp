#include "product/logging/logging.h"

#include "product/log_colour/log_colour.h"

#include <spdlog/formatter.h>
#include <spdlog/logger.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#ifndef _WIN32
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::product {
namespace {

spdlog::level::level_enum to_spdlog_level(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Critical:
        return spdlog::level::critical;
    case LogLevel::Off:
        return spdlog::level::off;
    }
    throw std::invalid_argument("LoggingOptions level is invalid");
}

std::string_view service_level_name(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "TRACE";
    case spdlog::level::debug:
        return "DEBUG";
    case spdlog::level::info:
        return "INFO ";
    case spdlog::level::warn:
        return "WARN ";
    case spdlog::level::err:
        return "ERROR";
    case spdlog::level::critical:
        return "FATAL";
    case spdlog::level::off:
        return "OFF  ";
    case spdlog::level::n_levels:
        break;
    }
    return "UNKWN";
}

std::string_view tool_level_name(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "trace: ";
    case spdlog::level::debug:
        return "debug: ";
    case spdlog::level::info:
        return {};
    case spdlog::level::warn:
        return "warning: ";
    case spdlog::level::err:
        return "error: ";
    case spdlog::level::critical:
        return "fatal: ";
    case spdlog::level::off:
    case spdlog::level::n_levels:
        break;
    }
    return {};
}

class PrettyLogFormatter final : public spdlog::formatter {
public:
    explicit PrettyLogFormatter(LogPresentation presentation) : presentation_(presentation) {}

    void format(const spdlog::details::log_msg& message,
                spdlog::memory_buf_t& destination) override {
        if (presentation_ == LogPresentation::Service) {
            const auto since_epoch = message.time.time_since_epoch();
            const auto whole_seconds =
                std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
            const auto milliseconds =
                std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch - whole_seconds)
                    .count();
            const std::time_t wall_seconds = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::time_point(whole_seconds));
            std::tm local{};
#ifdef _WIN32
            localtime_s(&local, &wall_seconds);
#else
            localtime_r(&wall_seconds, &local);
#endif
            fmt::format_to(std::back_inserter(destination),
                           "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}  ", local.tm_year + 1900,
                           local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min,
                           local.tm_sec, milliseconds);
            message.color_range_start    = destination.size();
            const std::string_view level = service_level_name(message.level);
            destination.append(level.data(), level.data() + level.size());
            message.color_range_end = destination.size();
            destination.push_back(' ');
        } else {
            const std::string_view level = tool_level_name(message.level);
            message.color_range_start    = destination.size();
            destination.append(level.data(), level.data() + level.size());
            message.color_range_end = destination.size();
        }
        destination.append(message.payload.data(), message.payload.data() + message.payload.size());
        destination.push_back('\n');
    }

    [[nodiscard]] std::unique_ptr<spdlog::formatter> clone() const override {
        return std::make_unique<PrettyLogFormatter>(presentation_);
    }

private:
    LogPresentation presentation_;
};

spdlog::color_mode to_spdlog_color_mode(LogColorMode mode) {
    switch (mode) {
    case LogColorMode::Auto:
        return spdlog::color_mode::automatic;
    case LogColorMode::Always:
        return spdlog::color_mode::always;
    case LogColorMode::Never:
        return spdlog::color_mode::never;
    }
    throw std::invalid_argument("LoggingOptions color mode is invalid");
}

void report_logging_error(const std::string& message) noexcept {
    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
    if (reported.test_and_set(std::memory_order_relaxed)) { return; }
    std::fprintf(stderr, "ninfer logging failure: %s\n", message.c_str());
    std::fflush(stderr);
}

struct TerminalSize {
    std::size_t columns = 0;
    std::size_t rows    = 0;
};

TerminalSize stderr_terminal_size() noexcept {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (::GetConsoleScreenBufferInfo(::GetStdHandle(STD_ERROR_HANDLE), &info) == 0) { return {}; }
    return {
        .columns = static_cast<std::size_t>(info.srWindow.Right - info.srWindow.Left + 1),
        .rows    = static_cast<std::size_t>(info.srWindow.Bottom - info.srWindow.Top + 1),
    };
#else
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) != 0) { return {}; }
    return {.columns = size.ws_col, .rows = size.ws_row};
#endif
}

// SGR sequences matching spdlog's console level colours.
std::string_view level_colour(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "\x1b[37m";
    case spdlog::level::debug:
        return "\x1b[36m";
    case spdlog::level::info:
        return "\x1b[32m";
    case spdlog::level::warn:
        return "\x1b[33m\x1b[1m";
    case spdlog::level::err:
        return "\x1b[31m\x1b[1m";
    case spdlog::level::critical:
        return "\x1b[1m\x1b[41m";
    case spdlog::level::off:
    case spdlog::level::n_levels:
        break;
    }
    return {};
}

// Owns the transient footer beneath the scrolling records: an optional pinned panel followed by an
// optional progress line. Every persistent record erases the footer, writes itself where the footer
// began, and redraws the footer below it, so the footer stays at the bottom of the console while
// the scrollback keeps every record.
//
// With VT cursor control, each erase-record-redraw is one write bracketed as a synchronized update,
// so a terminal never paints a frame with the footer missing or half drawn. Without it the footer
// can only be the single progress line, erased with spaces around spdlog's own console writes.
class ProgressAwareStderrSink final : public spdlog::sinks::sink {
public:
    // The Windows color sink writes messages that carry a colour range through
    // WriteConsoleA, which silently fails on a redirected handle; with
    // color_mode::always every such message would vanish from file/pipe logs.
    // The formatter colours every service line's level, so any non-console
    // stderr must fall back to plain (uncoloured) writes.
    explicit ProgressAwareStderrSink(spdlog::color_mode color)
        : sink_(log_colour::stderr_is_console() ? color : spdlog::color_mode::never),
          interactive_(log_colour::stderr_is_console()),
          cursor_control_(interactive_ && log_colour::enable_stderr_vt_processing()),
          colour_(interactive_ && color != spdlog::color_mode::never),
          formatter_(std::make_unique<spdlog::pattern_formatter>()) {}

    ~ProgressAwareStderrSink() override { clear(); }

    void log(const spdlog::details::log_msg& message) override {
        std::lock_guard lock(mutex_);
        if (cursor_control_) {
            spdlog::memory_buf_t formatted;
            formatter_->format(message, formatted);
            const std::string_view text(formatted.data(), formatted.size());
            const std::string_view colour = colour_ ? level_colour(message.level) : "";
            std::string record;
            if (!colour.empty() && message.color_range_end > message.color_range_start) {
                record.append(text.substr(0, message.color_range_start));
                record.append(colour);
                record.append(text.substr(message.color_range_start,
                                          message.color_range_end - message.color_range_start));
                record.append("\x1b[m");
                record.append(text.substr(message.color_range_end));
            } else {
                record.append(text);
            }
            repaint_unlocked(record);
            return;
        }
        erase_legacy_unlocked();
        try {
            sink_.log(message);
        } catch (...) {
            draw_legacy_unlocked();
            throw;
        }
        draw_legacy_unlocked();
    }

    void flush() override {
        std::lock_guard lock(mutex_);
        sink_.flush();
        if (interactive_) { std::fflush(stderr); }
    }

    void set_pattern(const std::string& pattern) override {
        std::lock_guard lock(mutex_);
        formatter_ = std::make_unique<spdlog::pattern_formatter>(pattern);
        sink_.set_pattern(pattern);
    }

    void set_formatter(std::unique_ptr<spdlog::formatter> formatter) override {
        std::lock_guard lock(mutex_);
        formatter_ = formatter->clone();
        sink_.set_formatter(std::move(formatter));
    }

    [[nodiscard]] bool interactive() const noexcept { return interactive_; }

    [[nodiscard]] bool cursor_control() const noexcept { return cursor_control_; }

    void update(std::string line) {
        if (!interactive_) { return; }
        std::lock_guard lock(mutex_);
        if (cursor_control_) {
            progress_line_ = std::move(line);
            repaint_unlocked({});
            return;
        }
        erase_legacy_unlocked();
        progress_line_ = std::move(line);
        draw_legacy_unlocked();
    }

    void update_panel(std::vector<std::string> lines) {
        if (!cursor_control_) { return; }
        std::lock_guard lock(mutex_);
        panel_lines_ = std::move(lines);
        repaint_unlocked({});
    }

    void clear() noexcept {
        if (!interactive_) { return; }
        try {
            std::lock_guard lock(mutex_);
            progress_line_.clear();
            if (cursor_control_) {
                repaint_unlocked({});
            } else {
                erase_legacy_unlocked();
                std::fflush(stderr);
            }
        } catch (...) {}
    }

    // Leaves the last panel on screen as ordinary output, so the final statistics survive exit.
    void release_panel() noexcept {
        if (!cursor_control_) { return; }
        try {
            std::lock_guard lock(mutex_);
            const std::size_t columns = stderr_terminal_size().columns;
            std::string record;
            for (const std::string& line : panel_lines_) {
                record += columns > 1 ? fit_terminal_line(line, columns - 1) : line;
                record.push_back('\n');
            }
            panel_lines_.clear();
            repaint_unlocked(record);
        } catch (...) {}
    }

    void clear_panel() noexcept {
        if (!cursor_control_) { return; }
        try {
            std::lock_guard lock(mutex_);
            panel_lines_.clear();
            repaint_unlocked({});
        } catch (...) {}
    }

private:
    // Erases the drawn footer, writes `record` where it began, and draws the current footer, all as
    // one synchronized write.
    void repaint_unlocked(std::string_view record) {
        std::string out = "\x1b[?2026h";
        append_erase_unlocked(out);
        out.append(record);
        append_draw_unlocked(out);
        out += "\x1b[?2026l";
        std::fwrite(out.data(), 1, out.size(), stderr);
        std::fflush(stderr);
    }

    void append_erase_unlocked(std::string& out) {
        if (rendered_widths_.empty()) { return; }
        // A resize may have rewrapped the drawn lines; count the rows they occupy now.
        const std::size_t columns = stderr_terminal_size().columns;
        std::size_t rows          = 0;
        for (const std::size_t width : rendered_widths_) {
            rows += columns == 0 || width <= columns ? 1 : (width + columns - 1) / columns;
        }
        out += '\r';
        if (rows > 1) { out += "\x1b[" + std::to_string(rows - 1) + 'A'; }
        out += "\x1b[J";
        rendered_widths_.clear();
    }

    void append_draw_unlocked(std::string& out) {
        const TerminalSize size = stderr_terminal_size();
        // Keep one column free so no footer line reaches the auto-wrap margin, and keep at least
        // one scrolling row above the footer; a console too short for the panel shows only the
        // progress line.
        const std::size_t columns     = size.columns > 1 ? size.columns - 1 : 0;
        const std::size_t footer_rows = panel_lines_.size() + (progress_line_.empty() ? 0 : 1);
        const bool show_panel         = size.rows == 0 || footer_rows < size.rows;
        const auto append_line        = [&](std::string_view line) {
            if (!rendered_widths_.empty()) { out.push_back('\n'); }
            const std::string fitted =
                columns == 0 ? std::string(line) : fit_terminal_line(line, columns);
            out += fitted;
            rendered_widths_.push_back(terminal_display_width(fitted));
        };
        if (show_panel) {
            for (const std::string& line : panel_lines_) { append_line(line); }
        }
        if (!progress_line_.empty()) { append_line(progress_line_); }
    }

    void erase_legacy_unlocked() {
        if (rendered_widths_.empty()) { return; }
        std::fputc('\r', stderr);
        for (std::size_t index = 0; index < rendered_widths_.front(); ++index) {
            std::fputc(' ', stderr);
        }
        std::fputc('\r', stderr);
        rendered_widths_.clear();
    }

    void draw_legacy_unlocked() {
        if (!interactive_ || progress_line_.empty()) { return; }
        std::fwrite(progress_line_.data(), 1, progress_line_.size(), stderr);
        std::fflush(stderr);
        rendered_widths_.push_back(progress_line_.size());
    }

    std::mutex mutex_;
    spdlog::sinks::stderr_color_sink_st sink_;
    bool interactive_    = false;
    bool cursor_control_ = false;
    bool colour_         = false;
    std::unique_ptr<spdlog::formatter> formatter_;
    std::string progress_line_;
    std::vector<std::string> panel_lines_;
    // Visible width of each footer line currently on screen, top to bottom.
    std::vector<std::size_t> rendered_widths_;
};

} // namespace

LogLevel parse_log_level(std::string_view value) {
    if (value == "trace") { return LogLevel::Trace; }
    if (value == "debug") { return LogLevel::Debug; }
    if (value == "info") { return LogLevel::Info; }
    if (value == "warning" || value == "warn") { return LogLevel::Warning; }
    if (value == "error") { return LogLevel::Error; }
    if (value == "critical" || value == "fatal") { return LogLevel::Critical; }
    if (value == "off") { return LogLevel::Off; }
    throw std::invalid_argument("invalid log level: " + std::string(value));
}

namespace {

// Length of the ANSI CSI sequence starting at `index`, or 0 when none starts there.
std::size_t csi_length(std::string_view line, std::size_t index) noexcept {
    if (line[index] != '\x1b' || index + 1 >= line.size() || line[index + 1] != '[') { return 0; }
    std::size_t end = index + 2;
    while (end < line.size() && !(line[end] >= 0x40 && line[end] <= 0x7e)) { ++end; }
    return std::min(end + 1, line.size()) - index;
}

std::size_t utf8_length(std::string_view line, std::size_t index) noexcept {
    const auto byte    = static_cast<unsigned char>(line[index]);
    std::size_t length = byte >= 0xf0 ? 4 : byte >= 0xe0 ? 3 : byte >= 0xc0 ? 2 : 1;
    return std::min(length, line.size() - index);
}

} // namespace

std::size_t terminal_display_width(std::string_view line) noexcept {
    std::size_t width = 0;
    for (std::size_t index = 0; index < line.size();) {
        if (const std::size_t escape = csi_length(line, index); escape != 0) {
            index += escape;
            continue;
        }
        index += utf8_length(line, index);
        ++width;
    }
    return width;
}

std::string fit_terminal_line(std::string_view line, std::size_t columns) {
    std::string out;
    out.reserve(line.size());
    std::size_t width = 0;
    bool styled       = false;
    for (std::size_t index = 0; index < line.size();) {
        if (const std::size_t escape = csi_length(line, index); escape != 0) {
            out.append(line.substr(index, escape));
            styled = true;
            index += escape;
            continue;
        }
        if (width == columns) {
            if (styled) { out += "\x1b[0m"; }
            return out;
        }
        const std::size_t length = utf8_length(line, index);
        out.append(line.substr(index, length));
        index += length;
        ++width;
    }
    return out;
}

struct TerminalProgress::Impl {
    std::shared_ptr<ProgressAwareStderrSink> sink;
    bool info_enabled = true;
};

TerminalProgress::TerminalProgress(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TerminalProgress::~TerminalProgress() { clear(); }

bool TerminalProgress::enabled() const noexcept {
    return impl_->info_enabled && impl_->sink->interactive();
}

void TerminalProgress::update(std::string line) {
    if (impl_->info_enabled) { impl_->sink->update(std::move(line)); }
}

void TerminalProgress::clear() noexcept { impl_->sink->clear(); }

struct TerminalPanel::Impl {
    std::shared_ptr<ProgressAwareStderrSink> sink;
    bool info_enabled = true;
};

TerminalPanel::TerminalPanel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TerminalPanel::~TerminalPanel() { impl_->sink->release_panel(); }

bool TerminalPanel::enabled() const noexcept {
    return impl_->info_enabled && impl_->sink->cursor_control();
}

void TerminalPanel::update(std::vector<std::string> lines) {
    if (enabled()) { impl_->sink->update_panel(std::move(lines)); }
}

void TerminalPanel::clear() noexcept { impl_->sink->clear_panel(); }

struct LoggingRuntime::Impl {
    explicit Impl(LoggingOptions options) {
        const spdlog::level::level_enum level = to_spdlog_level(options.level);
        auto sink = std::make_shared<ProgressAwareStderrSink>(to_spdlog_color_mode(options.color));
        progress  = std::shared_ptr<TerminalProgress>(new TerminalProgress(
            std::make_unique<TerminalProgress::Impl>(sink, level <= spdlog::level::info)));
        panel     = std::shared_ptr<TerminalPanel>(new TerminalPanel(
            std::make_unique<TerminalPanel::Impl>(sink, level <= spdlog::level::info)));
        logger    = std::make_shared<spdlog::logger>(std::move(options.logger_name), sink);
        logger->set_formatter(std::make_unique<PrettyLogFormatter>(options.presentation));
        logger->set_level(level);
        logger->flush_on(spdlog::level::warn);
        logger->set_error_handler([sink](const std::string& message) noexcept {
            sink->clear();
            report_logging_error(message);
        });
    }

    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<TerminalProgress> progress;
    std::shared_ptr<TerminalPanel> panel;
};

LoggingRuntime::LoggingRuntime(LoggingOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

LoggingRuntime::~LoggingRuntime() { flush(); }

std::shared_ptr<spdlog::logger> LoggingRuntime::logger() const noexcept { return impl_->logger; }

std::shared_ptr<TerminalProgress> LoggingRuntime::terminal_progress() const noexcept {
    return impl_->progress;
}

std::shared_ptr<TerminalPanel> LoggingRuntime::terminal_panel() const noexcept {
    return impl_->panel;
}

void LoggingRuntime::flush() noexcept {
    try {
        impl_->logger->flush();
    } catch (const std::exception& exception) {
        impl_->progress->clear();
        report_logging_error(exception.what());
    } catch (...) {
        impl_->progress->clear();
        report_logging_error("unknown flush error");
    }
}

} // namespace ninfer::product
