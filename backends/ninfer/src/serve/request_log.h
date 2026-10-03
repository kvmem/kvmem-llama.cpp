#pragma once

// Optional full-precision JSONL measurement log. Human operational rendering is owned separately
// by operational_log.* and never consumes these serialized records.

#include "serve/request_events.h"
#include "serve/serve_options.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace spdlog {
class logger;
}

namespace ninfer::serve {

inline constexpr int kRequestLogSchemaVersion        = 27;
inline constexpr const char* kRequestLogArtifactType = "ninfer_serve_request_log";

struct ServerLogEnvironment {
    int device = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::uint64_t total_device_memory_bytes = 0;
    int compute_capability_major            = 0;
    int compute_capability_minor            = 0;
    std::string cuda_compile_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
};

// Pure JSON formatters are public to repository tests. Each return value is one complete JSON
// object without a trailing newline.
std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp_unix_ms,
    const ServeOptions& options, const ninfer::EngineOptions& engine_options,
    const ninfer::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
    const ServerLogEnvironment& environment, std::optional<std::uint64_t> artifact_size_bytes);
std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context);
std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp_unix_ms,
                                         const RequestRejectionLogContext& context);
std::string format_request_done_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome);
std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context, const std::string& message);
std::string format_throughput_json(const std::string& server_instance_id,
                                   std::uint64_t timestamp_unix_ms, const ThroughputReport& report);

ServerLogEnvironment query_server_log_environment(int device);

// Opens in append mode so one campaign file can contain multiple independently started MTP/model
// blocks. Every line carries server_instance_id because request ids restart at one per process.
class JsonlRequestLog {
public:
    // max_bytes = 0 keeps one unbounded file. Otherwise the active file is renamed to <path>.1
    // once it reaches max_bytes, older copies shift up to <path>.<keep> (the oldest is dropped),
    // and a fresh file starts with the server_start record so every file is self-describing.
    explicit JsonlRequestLog(const std::string& path,
                             const std::string& protected_artifact_path = {},
                             std::shared_ptr<spdlog::logger> logger     = {},
                             std::uint64_t max_bytes = 0, std::uint32_t keep = 4);

    JsonlRequestLog(const JsonlRequestLog&)            = delete;
    JsonlRequestLog& operator=(const JsonlRequestLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open(); }

    [[nodiscard]] const std::string& server_instance_id() const noexcept {
        return server_instance_id_;
    }

    void write_server_start(const ServeOptions& options,
                            const ninfer::EngineOptions& engine_options,
                            const ninfer::ModelSamplingDefaults& sampling_defaults,
                            const std::string& public_model_id, const ninfer::LoadSummary& load,
                            const ninfer::MemorySummary& memory);
    void write_request_start(const RequestLogContext& context);
    void write_request_rejected(const RequestRejectionLogContext& context);
    void write_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void write_request_error(const RequestLogContext& context, const std::string& message);
    void write_throughput(const ThroughputReport& report);

private:
    void append(std::string record);
    // Caller holds mutex_.
    void rotate_locked();

    std::string path_;
    std::string server_instance_id_;
    std::ofstream output_;
    std::mutex mutex_;
    std::shared_ptr<spdlog::logger> logger_;
    bool failed_                 = false;
    std::uint64_t max_bytes_     = 0;
    std::uint32_t keep_          = 4;
    std::uint64_t written_bytes_ = 0;
    std::string server_start_record_;
};

} // namespace ninfer::serve
