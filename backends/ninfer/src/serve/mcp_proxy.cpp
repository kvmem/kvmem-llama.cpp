#include "serve/mcp_proxy.h"

#include "serve/http_server.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace ninfer::serve {
namespace {

std::string to_lower(std::string_view text) {
    std::string lowered(text);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

bool starts_with_ci(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && to_lower(text.substr(0, prefix.size())) == to_lower(prefix);
}

McpProxyRequest reject(std::string message) {
    McpProxyRequest request;
    request.error = std::move(message);
    return request;
}

// State shared between the handler, the chunked provider and the upstream thread. The upstream
// thread holds only this, never the thread handle, so it can never be the one to join itself.
struct McpProxyExchange {
    explicit McpProxyExchange(const McpProxyTarget& target) : client(target.host, target.port) {}

    httplib::Client client;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> chunks;
    bool head_ready = false;
    bool finished   = false;
    bool abandoned  = false;
    int status      = 0;
    httplib::Headers response_headers;
    std::string error;
};

// Owns the upstream thread. Dropping it (the handler's error path, or httplib releasing the
// provider when the browser hangs up or the stream ends) shuts the upstream socket down and joins:
// a silent MCP GET channel would otherwise hold the thread until the hour-long read timeout.
struct McpProxyRelay {
    std::shared_ptr<McpProxyExchange> exchange;
    std::thread worker;

    McpProxyRelay() = default;
    McpProxyRelay(const McpProxyRelay&)            = delete;
    McpProxyRelay& operator=(const McpProxyRelay&) = delete;

    ~McpProxyRelay() {
        if (!exchange || !worker.joinable()) { return; }
        std::unique_lock<std::mutex> lock(exchange->mutex);
        exchange->abandoned = true;
        // stop() only reaches a socket that exists, so repeat it until the send has returned: one
        // issued before the connection opened would otherwise miss it.
        while (!exchange->finished) {
            lock.unlock();
            exchange->client.stop();
            lock.lock();
            exchange->cv.wait_for(lock, std::chrono::milliseconds(50));
        }
        lock.unlock();
        worker.join();
    }
};

void run_upstream(const std::shared_ptr<McpProxyExchange>& exchange, httplib::Request upstream) {
    upstream.response_handler = [exchange](const httplib::Response& response) {
        {
            std::lock_guard<std::mutex> lock(exchange->mutex);
            if (exchange->abandoned) { return false; }
            exchange->status           = response.status;
            exchange->response_headers = response.headers;
            exchange->head_ready       = true;
        }
        exchange->cv.notify_all();
        return true;
    };
    upstream.content_receiver = [exchange](const char* data, std::size_t length, std::size_t,
                                           std::size_t) {
        {
            std::lock_guard<std::mutex> lock(exchange->mutex);
            if (exchange->abandoned) { return false; }
            exchange->chunks.emplace_back(data, length);
        }
        exchange->cv.notify_all();
        return true;
    };
    const httplib::Result result = exchange->client.send(upstream);
    {
        std::lock_guard<std::mutex> lock(exchange->mutex);
        // A transport failure after the head arrived is a truncated stream, not a failed request:
        // the status is already on its way to the browser.
        if (!result && !exchange->head_ready) { exchange->error = httplib::to_string(result.error()); }
        exchange->finished = true;
    }
    exchange->cv.notify_all();
}

void write_relay_error(httplib::Response& response, int status, const char* type, const char* code,
                       const std::string& message) {
    ApiError error;
    error.status  = status;
    error.type    = type;
    error.code    = code;
    error.message = "mcp proxy: " + message;
    write_openai_error(response, error);
}

} // namespace

McpProxyRequest parse_mcp_proxy_target(const std::string& url) {
    if (url.empty()) { return reject("proxy target url is empty"); }
    if (starts_with_ci(url, "https://")) {
        return reject("https proxy targets are unsupported: this build has no TLS client");
    }
    if (!starts_with_ci(url, "http://")) {
        return reject("proxy target url must be an absolute http:// url");
    }
    const std::string_view rest = std::string_view(url).substr(std::string_view("http://").size());
    const std::size_t authority_end = rest.find_first_of("/?#");
    const std::string_view authority =
        authority_end == std::string_view::npos ? rest : rest.substr(0, authority_end);
    if (authority.empty()) { return reject("proxy target url has no host"); }

    // A bracketed IPv6 literal keeps its colons; only a colon after the closing bracket, or the one
    // colon of a plain host, introduces the port.
    std::string_view host_part = authority;
    std::string_view port_part;
    const bool bracketed         = authority.front() == '[';
    const std::size_t port_colon = bracketed ? authority.find(':', authority.find(']') + 1)
                                             : authority.rfind(':');
    if (!bracketed && port_colon != std::string_view::npos && authority.find(':') != port_colon) {
        return reject("proxy target host is malformed");
    }
    if (port_colon != std::string_view::npos) {
        host_part = authority.substr(0, port_colon);
        port_part = authority.substr(port_colon + 1);
    }
    if (host_part.size() >= 2 && host_part.front() == '[' && host_part.back() == ']') {
        host_part = host_part.substr(1, host_part.size() - 2);
    }
    if (host_part.empty()) { return reject("proxy target url has no host"); }

    McpProxyRequest request;
    request.ok = true;
    if (!port_part.empty()) {
        if (!std::all_of(port_part.begin(), port_part.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; }) ||
            port_part.size() > 5) {
            return reject("proxy target port is not a number");
        }
        const long port = std::strtol(std::string(port_part).c_str(), nullptr, 10);
        if (port <= 0 || port > 65535) { return reject("proxy target port is out of range"); }
        request.target.port = static_cast<int>(port);
    }
    request.target.host = std::string(host_part);
    // The fragment is client-side only and never goes upstream.
    std::string_view path =
        authority_end == std::string_view::npos ? std::string_view() : rest.substr(authority_end);
    if (const std::size_t fragment = path.find('#'); fragment != std::string_view::npos) {
        path = path.substr(0, fragment);
    }
    request.target.path = path.empty() ? "/" : std::string(path);
    return request;
}

McpProxyRequest parse_mcp_proxy_request(const httplib::Request& request) {
    if (!request.has_param(kMcpProxyUrlParam)) {
        return reject(std::string("missing ") + kMcpProxyUrlParam + " query parameter");
    }
    McpProxyRequest parsed = parse_mcp_proxy_target(request.get_param_value(kMcpProxyUrlParam));
    if (!parsed.ok) { return parsed; }
    const std::size_t prefix_length = std::string_view(kMcpProxyHeaderPrefix).size();
    for (const auto& [name, value] : request.headers) {
        if (!starts_with_ci(name, kMcpProxyHeaderPrefix)) { continue; }
        std::string forwarded = name.substr(prefix_length);
        if (forwarded.empty()) { continue; }
        // The upstream client owns Host and the request framing.
        const std::string lowered = to_lower(forwarded);
        if (lowered == "host" || lowered == "content-length" || lowered == "connection" ||
            lowered == "transfer-encoding") {
            continue;
        }
        parsed.headers.emplace(std::move(forwarded), value);
    }
    return parsed;
}

bool header_name_is(const std::string& name, std::string_view expected) {
    return to_lower(name) == expected;
}

bool is_hop_by_hop_response_header(const std::string& name) {
    const std::string lowered = to_lower(name);
    if (lowered.starts_with("access-control-")) { return true; }
    return lowered == "connection" || lowered == "keep-alive" || lowered == "transfer-encoding" ||
           lowered == "content-length" || lowered == "upgrade" || lowered == "proxy-authenticate" ||
           lowered == "proxy-authorization" || lowered == "te" || lowered == "trailer";
}

void relay_mcp_proxy(const httplib::Request& request, httplib::Response& response) {
    McpProxyRequest parsed = parse_mcp_proxy_request(request);
    if (!parsed.ok) {
        write_relay_error(response, 400, "invalid_request_error", "invalid_proxy_target",
                          parsed.error);
        return;
    }

    auto relay      = std::make_shared<McpProxyRelay>();
    relay->exchange = std::make_shared<McpProxyExchange>(parsed.target);
    auto& client    = relay->exchange->client;
    client.set_connection_timeout(std::chrono::seconds(5));
    // The MCP GET channel stays open for the session and is silent between events; a short read
    // timeout would tear it down mid-session.
    client.set_read_timeout(std::chrono::hours(1));
    client.set_write_timeout(std::chrono::seconds(30));
    client.set_keep_alive(false);

    httplib::Request upstream;
    upstream.method  = request.method;
    upstream.path    = parsed.target.path;
    upstream.headers = std::move(parsed.headers);
    upstream.body    = request.body;
    relay->worker    = std::thread(run_upstream, relay->exchange, std::move(upstream));

    McpProxyExchange& exchange = *relay->exchange;
    std::unique_lock<std::mutex> lock(exchange.mutex);
    exchange.cv.wait(lock, [&exchange] { return exchange.head_ready || exchange.finished; });
    if (!exchange.head_ready) {
        const std::string reason = exchange.error.empty() ? "upstream request failed" : exchange.error;
        lock.unlock();
        write_relay_error(response, 502, "upstream_error", "proxy_target_unreachable", reason);
        return;
    }
    response.status                         = exchange.status;
    const httplib::Headers upstream_headers = exchange.response_headers;
    lock.unlock();

    std::string content_type = "application/octet-stream";
    for (const auto& [name, value] : upstream_headers) {
        if (is_hop_by_hop_response_header(name)) { continue; }
        // The chunked provider owns Content-Type; setting it here too would emit it twice.
        if (header_name_is(name, "content-type")) {
            content_type = value;
            continue;
        }
        response.set_header(name, value);
    }

    response.set_chunked_content_provider(
        content_type, [relay](std::size_t, httplib::DataSink& sink) -> bool {
            McpProxyExchange& shared = *relay->exchange;
            std::deque<std::string> ready;
            bool drained = false;
            {
                std::unique_lock<std::mutex> lock(shared.mutex);
                // Wake once a second to notice a browser that hung up on a silent channel.
                while (shared.chunks.empty() && !shared.finished) {
                    if (shared.cv.wait_for(lock, std::chrono::seconds(1)) ==
                            std::cv_status::timeout &&
                        !sink.is_writable()) {
                        return false;
                    }
                }
                ready.swap(shared.chunks);
                drained = shared.finished && ready.empty();
            }
            for (const std::string& chunk : ready) {
                if (!sink.write(chunk.data(), chunk.size())) { return false; }
            }
            if (drained) { sink.done(); }
            return true;
        });
}

} // namespace ninfer::serve
