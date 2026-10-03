#pragma once

#include "ninfer/types.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <utility>

namespace ninfer::runtime {

// Publishes one Engine runtime diagnostic through the product's observer. Without an observer,
// Info and above are written to stderr. It never throws: a logging failure must not disturb the
// worker or a destructor.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 3, 4)))
#endif
inline void publish_diagnostic(const DiagnosticObserver& observer, DiagnosticLevel level,
                               const char* format, ...) noexcept {
    std::va_list args;
    va_start(args, format);
    const int length = std::vsnprintf(nullptr, 0, format, args);
    va_end(args);
    if (length < 0) { return; }
    std::string message;
    try {
        message.resize(static_cast<std::size_t>(length));
    } catch (...) { return; }
    va_start(args, format);
    std::vsnprintf(message.data(), message.size() + 1, format, args);
    va_end(args);
    if (!observer.callback) {
        if (level != DiagnosticLevel::Debug) {
            std::fprintf(stderr, "[engine] %s\n", message.c_str());
        }
        return;
    }
    try {
        observer.callback(Diagnostic{.level = level, .message = std::move(message)});
    } catch (...) {}
}

} // namespace ninfer::runtime
