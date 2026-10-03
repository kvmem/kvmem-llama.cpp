#include "serve/http_server.h"

#include "product/logging/logging.h"
#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/mcp_proxy.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"
#include "serve/webui.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::string_view kCorsAllowedHeaders =
    "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
    "anthropic-user-profile-id";

// API routes keep their own 404s. Every other GET path belongs to the WebUI, whose client-side
// router owns paths the server has no file for.
bool is_api_path(std::string_view path) {
    return path == "/v1" || path.starts_with("/v1/") || path == "/health" || path == "/metrics" ||
           path == "/stats" || path == "/slots" || path == "/props" || path == kMcpProxyPath;
}


void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_demoted !=
               report.previous.pressure_private_owners_demoted ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_main_kv_lease_pages !=
               report.previous.device_main_kv_lease_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.device_backend_kv_lease_pages !=
               report.previous.device_backend_kv_lease_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    if (path == "/v1/load") { return "load"; }
    if (path == "/stats") { return "stats"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger,
                       std::shared_ptr<product::TerminalPanel> panel)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger),
                     static_cast<std::uint64_t>(options_.request_log_max_mib) << 20U,
                     options_.request_log_keep) {
    if (options_.log_stats_panel && panel != nullptr && panel->enabled()) {
        console_stats_ = std::make_unique<ConsoleStatsPanel>(std::move(panel));
    }
    // cpp-httplib is thread-per-connection: a worker is held for a connection's whole
    // life, including the idle keep-alive window between requests. Sizing the pool to the
    // request-lifetime capacity alone lets idle pooled connections occupy every worker, at
    // which point a C8 server accepts and runs requests strictly one at a time. Base
    // workers cover the admissible request capacity; the headroom is grown on demand for
    // connections that are merely open, and those dynamic workers retire once idle.
    constexpr std::size_t kKeepAliveWorkerHeadroom = 64;
    const std::size_t request_workers =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests + 1;
    const std::size_t worker_limit = request_workers + kKeepAliveWorkerHeadroom;
    server_.new_task_queue         = [request_workers, worker_limit] {
        return new httplib::ThreadPool(request_workers, worker_limit, worker_limit);
    };
    server_.set_tcp_nodelay(true);
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
    if (options_.stats_port != 0) { register_stats_routes(); }
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
    if (console_stats_) {
        console_stats_->request_rejected(
            make_request_failure(RequestFailurePhase::Prepare, context.error));
    }
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
    metrics_.record(outcome);
    if (console_stats_) { console_stats_->request_done(outcome); }
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
    if (console_stats_) { console_stats_->request_failure(failure); }
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        if (console_stats_) { console_stats_->runtime(current); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

httplib::Server::HandlerResponse HttpServer::pre_route(const httplib::Request& req,
                                                       httplib::Response& res) const {
    ensure_openai_request_id(req, res);
    if (!ready_.load(std::memory_order_acquire)) {
        // Runs for every route, including /health and OPTIONS, so a caller cannot tell "not
        // ready" apart from "unauthenticated" -- and skips the API-key check below, since a
        // loading-status response carries nothing worth protecting.
        ApiError error;
        error.status  = 503;
        error.type    = "service_unavailable";
        error.code    = "model_loading";
        error.message = "The model is still loading. Retry shortly.";
        // Weight load plus warmup measured ~10s on the 27B and longer on the 35B. Two seconds
        // is a polite poll interval rather than a promise about when readiness arrives.
        res.set_header("Retry-After", "2");
        if (req.path.rfind("/v1/messages", 0) == 0) {
            write_anthropic_error(res, error, new_anthropic_request_id());
        } else {
            write_openai_error(res, error);
        }
        return httplib::Server::HandlerResponse::Handled;
    }
    // The MCP relay carries no API key: the WebUI prefixes every header it means for the MCP
    // server, its own Authorization included, so requiring the key would break the relay
    // rather than protect it. It is opt-in, and the bind address is its boundary.
    if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS" ||
        (options_.webui_mcp_proxy && req.path == kMcpProxyPath) ||
        (webui_enabled() && req.method == "GET" && !is_api_path(req.path))) {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    // Accept both the OpenAI-style bearer token and the Anthropic-style
    // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
    // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
    const bool bearer_ok =
        matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
    const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
    if (!bearer_ok && !x_api_key_ok) {
        ApiError error;
        error.status  = 401;
        error.type    = "invalid_request_error";
        error.code    = "invalid_api_key";
        error.message = "missing or invalid API key";
        // Render the 401 in the shape the target endpoint speaks.
        if (req.path.rfind("/v1/messages", 0) == 0) {
            write_anthropic_error(res, error, new_anthropic_request_id());
        } else {
            write_openai_error(res, error);
        }
        return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers", std::string(kCorsAllowedHeaders)},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key. Headers a
        // client asks for beyond the fixed list (a WebUI's x-conversation-id, say) are allowed
        // too, since the preflight only gates which headers the browser may send.
        server_.Options(R"(.*)", [](const httplib::Request& req, httplib::Response& res) {
            res.status            = 204;
            const auto& requested = req.get_header_value("Access-Control-Request-Headers");
            if (!requested.empty()) {
                res.headers.erase("Access-Control-Allow-Headers");
                res.set_header("Access-Control-Allow-Headers",
                               std::string(kCorsAllowedHeaders) + ", " + requested);
            }
        });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        return pre_route(req, res);
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health",
                [this](const httplib::Request&, httplib::Response& res) { handle_health(res); });
    server_.Get("/v1/load", [this](const httplib::Request& req, httplib::Response& res) {
        handle_load(req, res);
    });
    server_.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res);
    });
    server_.Get("/stats", [this](const httplib::Request& req, httplib::Response& res) {
        handle_stats(req, res);
    });
    server_.Get("/slots", [this](const httplib::Request& req, httplib::Response& res) {
        handle_slots(req, res);
    });
    server_.Get("/props", [this](const httplib::Request& req, httplib::Response& res) {
        handle_props(req, res);
    });
    for (const char* base : {"/v1", "/v1/"}) {
        server_.Get(base, [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(make_api_index(public_model_id_).dump(), "application/json");
        });
    }
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
    if (options_.webui_mcp_proxy) {
        const auto relay = [](const httplib::Request& req, httplib::Response& res) {
            relay_mcp_proxy(req, res);
        };
        server_.Get(kMcpProxyPath, relay);
        server_.Post(kMcpProxyPath, relay);
        server_.Delete(kMcpProxyPath, relay);
    }
    // Registered last: httplib tries routes in order, so every API route above wins its path.
    if (webui_enabled()) {
        server_.Get(R"(/.*)", [this](const httplib::Request& req, httplib::Response& res) {
            handle_webui(req, res);
        });
    }
}

bool HttpServer::webui_enabled() const noexcept {
    return options_.enable_webui && !webui_assets().empty();
}

void HttpServer::handle_webui(const httplib::Request& req, httplib::Response& res) const {
    if (is_api_path(req.path)) {
        res.status = 404;
        return;
    }
    const bool gzip         = req.get_header_value("Accept-Encoding").find("gzip") != std::string::npos;
    const WebUiAsset* asset = req.path.size() > 1
                                  ? find_webui_asset(std::string_view(req.path).substr(1), gzip)
                                  : nullptr;
    if (asset == nullptr) { asset = find_webui_asset("index.html", gzip); }
    if (asset == nullptr) {
        res.status = 404;
        return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_header("ETag", std::string(asset->etag));
    res.set_header("Vary", "Accept-Encoding");
    if (!asset->encoding.empty()) {
        res.set_header("Content-Encoding", std::string(asset->encoding));
    }
    if (req.get_header_value("If-None-Match") == asset->etag) {
        res.status = 304;
        return;
    }
    res.set_content(reinterpret_cast<const char*>(asset->bytes.data()), asset->bytes.size(),
                    std::string(asset->content_type));
}

LoadSample HttpServer::load_sample() const {
    LoadSample sample;
    sample.uptime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - attached_at_).count();
    sample.admitted_requests      = service_->admitted_requests();
    sample.peak_admitted_requests = service_->peak_admitted_requests();
    sample.stats             = service_->runtime_stats();
    return sample;
}

void HttpServer::handle_load(const httplib::Request&, httplib::Response& res) const {
    res.set_header("Cache-Control", "no-store");
    res.set_content(make_load_report(load_capacity_, load_sample()), "application/json");
}

void HttpServer::handle_stats(const httplib::Request&, httplib::Response& res) const {
    res.set_header("Cache-Control", "no-store");
    res.set_content(make_stats_report(load_capacity_, load_sample()), "application/json");
}

// The pollers' own listener: one worker and no generation routes, so a dashboard or watchdog is
// never queued behind the connections the request pool is serving. It applies the same readiness
// and API-key rules as the main listener.
void HttpServer::register_stats_routes() {
    stats_server_.new_task_queue = [] { return new httplib::ThreadPool(1, 1, 64); };
    stats_server_.set_socket_options(configure_http_server_socket);
    stats_server_.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
            return pre_route(req, res);
        });
    stats_server_.Get(
        "/health", [this](const httplib::Request&, httplib::Response& res) { handle_health(res); });
    stats_server_.Get("/stats", [this](const httplib::Request& req, httplib::Response& res) {
        handle_stats(req, res);
    });
    stats_server_.Get("/v1/load", [this](const httplib::Request& req, httplib::Response& res) {
        handle_load(req, res);
    });
    stats_server_.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res);
    });
}

void HttpServer::handle_health(httplib::Response& res) const {
    const bool available = service_ != nullptr && service_->is_available();
    res.status           = available ? 200 : 503;
    res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                    "application/json");
}

void HttpServer::handle_metrics(const httplib::Request&, httplib::Response& res) const {
    res.set_header("Cache-Control", "no-store");
    res.set_content(metrics_.render(load_capacity_, load_sample()),
                    "text/plain; version=0.0.4; charset=utf-8");
}

// llama.cpp-shaped lane table. The Engine publishes how many lanes are running, not which request
// holds which lane, so the first `running` entries read as processing.
void HttpServer::handle_slots(const httplib::Request&, httplib::Response& res) const {
    const std::uint32_t running = service_->runtime_stats().running_requests;
    const bool speculative      = options_.speculative.backend != ninfer::SpeculativeBackend::None;
    nlohmann::json slots        = nlohmann::json::array();
    for (std::uint32_t lane = 0; lane < load_capacity_.max_concurrency; ++lane) {
        slots.push_back({{"id", lane},
                         {"n_ctx", load_capacity_.max_context},
                         {"speculative", speculative},
                         {"is_processing", lane < running}});
    }
    res.set_header("Cache-Control", "no-store");
    res.set_content(slots.dump(), "application/json");
}

void HttpServer::handle_props(const httplib::Request&, httplib::Response& res) const {
    const ninfer::SamplingPreset preset = service_->sampling_defaults().for_mode(
        options_.enable_thinking == false ? ninfer::SamplingMode::NonThinking
                                          : ninfer::SamplingMode::Thinking);
    const ninfer::SamplingOverrides& overrides = options_.sampling_overrides;
    nlohmann::json params = {
        {"n_predict", options_.default_max_tokens == kUnboundedOutputTokens
                          ? -1
                          : options_.default_max_tokens},
        {"temperature", options_.greedy ? 0.0F : overrides.temperature.value_or(preset.temperature)},
        {"top_k", overrides.top_k.value_or(preset.top_k)},
        {"top_p", overrides.top_p.value_or(preset.top_p)},
        {"min_p", overrides.min_p.value_or(preset.min_p)},
        {"presence_penalty", overrides.presence_penalty.value_or(preset.presence_penalty)},
        {"frequency_penalty", overrides.frequency_penalty.value_or(preset.frequency_penalty)},
    };
    params["seed"] = overrides.seed ? nlohmann::json(*overrides.seed) : nlohmann::json(-1);
    const nlohmann::json props = {
        {"default_generation_settings",
         {{"n_ctx", load_capacity_.max_context},
          {"speculative", options_.speculative.backend != ninfer::SpeculativeBackend::None},
          {"params", std::move(params)}}},
        {"total_slots", load_capacity_.max_concurrency},
        {"model_alias", public_model_id_},
        {"model_path", options_.artifact_path},
        {"modalities", {{"vision", options_.enable_vision}, {"audio", false}}},
        {"endpoint_slots", true},
        {"endpoint_props", true},
        {"endpoint_metrics", true},
        // The WebUI ungreys its "Use llama-server proxy" option from this flag alone.
        {"cors_proxy_enabled", options_.webui_mcp_proxy},
    };
    res.set_content(props.dump(), "application/json");
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context,
                                     options_.enable_vision,
                                     model_metadata_),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now(), options_.max_context,
                                      options_.enable_vision,
                                      model_metadata_),
                    "application/json");
}

bool HttpServer::bind() {
    return server_.bind_to_port(options_.host, options_.port) &&
           (options_.stats_port == 0 ||
            stats_server_.bind_to_port(options_.host, options_.stats_port));
}

void HttpServer::start_stats_listener() {
    if (options_.stats_port == 0 || stats_listener_.joinable()) { return; }
    stats_listener_ = std::thread([this] {
        try {
            (void)stats_server_.listen_after_bind();
        } catch (const std::exception&) {}
    });
}

void HttpServer::stop_stats_listener() {
    if (!stats_listener_.joinable()) { return; }
    stats_server_.stop();
    stats_listener_.join();
}

void HttpServer::start_serving_during_startup() {
    if (startup_listener_.joinable()) {
        throw std::logic_error("HTTP startup listener is already running");
    }

    // The readiness gate lives in register_routes()'s single pre-routing handler, ahead of the
    // API-key check -- not here, so starting this listener never replaces (and thereby drops) the
    // auth/request-ID middleware for the remainder of the process's life.
    start_stats_listener();
    startup_listener_ = std::thread([this] {
        // listen_after_bind() blocks here for the whole life of the server, spanning the switch
        // from 503 to serving. stop() is what ends it. It can also throw before ever reaching
        // that loop -- the task queue's thread pool spawns its worker threads here, and
        // std::thread's constructor throws std::system_error under resource exhaustion. An
        // exception escaping a thread function is std::terminate, so it is caught and folded into
        // the same false result a synchronous listen() failure already produces.
        bool result = false;
        try {
            result = server_.listen_after_bind();
        } catch (const std::exception&) {
        }
        startup_listener_result_.store(result, std::memory_order_release);
    });
}

bool HttpServer::await_startup_listener() {
    if (startup_listener_.joinable()) { startup_listener_.join(); }
    return startup_listener_result_.load(std::memory_order_acquire);
}

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_name);
    model_metadata_                = service.model_metadata();
    service_                       = &service;
    // memory_summary() takes the Engine execution lock; read it once here, never per /v1/load poll.
    const ninfer::MemorySummary memory = service.memory_summary();
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load, memory);
    load_capacity_ = make_load_capacity(public_model_id_, service.engine_options(), memory);
    attached_at_   = std::chrono::steady_clock::now();
    // Release: everything above must be visible to a handler that observes ready_ as true. This is
    // the only write, and handlers acquire it in the pre-routing guard before touching service_.
    ready_.store(true, std::memory_order_release);
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (console_stats_) { console_stats_->show(); }
    try {
        start_stats_listener();
        if (options_.log_stats_interval_ms != 0) {
            stats_stopping_ = false;
            stats_thread_   = std::thread([this] { run_stats_reporter(); });
        }
        // When the startup listener is running, the accept loop is already live on its thread and
        // has been since bind(); calling listen_after_bind() again would try to accept on the same
        // socket from two threads. Wait for that loop instead.
        const bool result =
            startup_listener_.joinable() ? await_startup_listener() : server_.listen_after_bind();
        stop_stats_listener();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_listener();
        stop_stats_reporter();
        // Stop and join the startup listener before this exception unwinds past us. attach() has
        // already run by the time listen() can be called, so that thread is live against
        // service_; the caller (main.cpp) destroys the attached GenerationService, constructed
        // after this HttpServer, before this object -- leaving that thread dereferencing a
        // dangling pointer for however long stack unwinding takes if it is still running then.
        if (startup_listener_.joinable()) {
            server_.stop();
            startup_listener_.join();
        }
        throw;
    }
}

void HttpServer::stop() {
    stats_server_.stop();
    server_.stop();
}

HttpServer::~HttpServer() {
    stop_stats_listener();
    if (startup_listener_.joinable()) {
        server_.stop();
        startup_listener_.join();
    }
    stop_stats_reporter();
}

} // namespace ninfer::serve
