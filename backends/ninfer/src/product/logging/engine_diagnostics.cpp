#include "product/logging/engine_diagnostics.h"

#include "product/logging/pretty_format.h"

#include <spdlog/logger.h>

#include <utility>

namespace ninfer::product {

DiagnosticObserver engine_diagnostic_observer(std::shared_ptr<spdlog::logger> logger) {
    return DiagnosticObserver{
        .callback = [logger = std::move(logger)](const Diagnostic& diagnostic) {
            const std::string message = format_pretty_text(diagnostic.message);
            switch (diagnostic.level) {
            case DiagnosticLevel::Debug:
                logger->debug("engine | {}", message);
                break;
            case DiagnosticLevel::Info:
                logger->info("engine | {}", message);
                break;
            case DiagnosticLevel::Warning:
                logger->warn("engine | {}", message);
                break;
            case DiagnosticLevel::Error:
                logger->error("engine | {}", message);
                break;
            }
        }};
}

} // namespace ninfer::product
