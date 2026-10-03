#pragma once

#include <httplib.h>

#include <string>
#include <string_view>

namespace ninfer::serve {

// Same-origin relay for the WebUI's MCP client. A browser page cannot talk to a third-party MCP
// server directly: the MCP transports answer POST from a raw response writer, so the reply carries
// no Access-Control-Allow-Origin even when the preflight succeeds. The WebUI therefore sends the
// request to its own origin under kMcpProxyPath and the server forwards it. These three constants
// are the WebUI's wire contract ("Use llama-server proxy") and must match it exactly.
inline constexpr const char* kMcpProxyPath         = "/cors-proxy";
inline constexpr const char* kMcpProxyUrlParam     = "url";
inline constexpr const char* kMcpProxyHeaderPrefix = "x-llama-server-proxy-header-";

struct McpProxyTarget {
    std::string host;
    int port = 80;
    std::string path; // path plus query
};

// Parsed relay request, or the reason it must be rejected.
struct McpProxyRequest {
    bool ok = false;
    std::string error; // non-empty exactly when !ok
    McpProxyTarget target;
    httplib::Headers headers; // forwarded upstream, prefix already stripped
};

// Splits an absolute http:// URL. https is refused rather than downgraded: the vendored httplib has
// no TLS client, and a downgrade would put the caller's bearer token on the wire in the clear.
McpProxyRequest parse_mcp_proxy_target(const std::string& url);

// Target from the url query parameter, forwarded headers from the prefixed ones. Unprefixed headers
// belong to the hop between the browser and this server and are dropped.
McpProxyRequest parse_mcp_proxy_request(const httplib::Request& request);

// Case-insensitive header-name match; `expected` must already be lowercase.
bool header_name_is(const std::string& name, std::string_view expected);

// Headers not copied from the upstream response: hop-by-hop framing belongs to the upstream
// connection, and upstream CORS headers next to the ones --cors emits make a browser see
// "Access-Control-Allow-Origin: *, *" and reject the reply.
bool is_hop_by_hop_response_header(const std::string& name);

// Forwards one relay request and streams the upstream reply back. The upstream call runs on its
// own thread because httplib sends this response's status and headers when the handler returns,
// while the upstream status is known only once that request is in flight; the body then streams
// through a chunked provider, which keeps the long-lived MCP GET channel live.
void relay_mcp_proxy(const httplib::Request& request, httplib::Response& response);

} // namespace ninfer::serve
