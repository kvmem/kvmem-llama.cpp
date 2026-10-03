#include "serve/mcp_proxy.h"

#include <iostream>
#include <string>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

httplib::Request proxy_request(const std::string& url, const httplib::Headers& headers) {
    httplib::Request request;
    request.method  = "POST";
    request.path    = kMcpProxyPath;
    request.headers = headers;
    request.params.emplace(kMcpProxyUrlParam, url);
    return request;
}

} // namespace

int main() {
    int failures = 0;

    const McpProxyRequest plain = parse_mcp_proxy_target("http://127.0.0.1:12315/mcp");
    failures += check(plain.ok && plain.target.host == "127.0.0.1" && plain.target.port == 12315 &&
                          plain.target.path == "/mcp",
                      "host, port and path were not split out of the target url");

    const McpProxyRequest default_port = parse_mcp_proxy_target("http://example.test/mcp");
    failures += check(default_port.ok && default_port.target.port == 80,
                      "a target url without a port did not default to 80");

    const McpProxyRequest with_query =
        parse_mcp_proxy_target("http://127.0.0.1:8931/sse?session=abc&x=1");
    failures += check(with_query.ok && with_query.target.path == "/sse?session=abc&x=1",
                      "the query string was dropped from the forwarded path");

    const McpProxyRequest rooted = parse_mcp_proxy_target("http://127.0.0.1:12315");
    failures += check(rooted.ok && rooted.target.path == "/",
                      "a target url without a path did not forward to /");

    const McpProxyRequest fragment = parse_mcp_proxy_target("http://127.0.0.1:9/mcp#frag");
    failures += check(fragment.ok && fragment.target.path == "/mcp",
                      "the fragment was sent upstream instead of being dropped");

    const McpProxyRequest ipv6 = parse_mcp_proxy_target("http://[::1]:12315/mcp");
    failures += check(ipv6.ok && ipv6.target.host == "::1" && ipv6.target.port == 12315,
                      "a bracketed IPv6 literal was not unwrapped into host and port");

    const McpProxyRequest ipv6_default = parse_mcp_proxy_target("http://[::1]/mcp");
    failures += check(ipv6_default.ok && ipv6_default.target.host == "::1" &&
                          ipv6_default.target.port == 80,
                      "a bracketed IPv6 literal without a port was misparsed");

    // https must be refused outright: silently downgrading to http would send the
    // caller's Authorization header over the wire in the clear.
    failures += check(!parse_mcp_proxy_target("https://example.test/mcp").ok,
                      "an https target was accepted despite there being no TLS client");
    failures += check(!parse_mcp_proxy_target("example.test/mcp").ok,
                      "a schemeless target was accepted");
    failures += check(!parse_mcp_proxy_target("http://").ok, "a hostless target was accepted");
    failures += check(!parse_mcp_proxy_target("http://host:70000/mcp").ok,
                      "an out-of-range port was accepted");
    failures += check(!parse_mcp_proxy_target("http://host:abc/mcp").ok,
                      "a non-numeric port was accepted");
    failures += check(!parse_mcp_proxy_target("http://::1:12315/mcp").ok,
                      "an unbracketed IPv6 literal was accepted");
    failures += check(!parse_mcp_proxy_target("").ok, "an empty target was accepted");

    const httplib::Headers headers = {
        {std::string(kMcpProxyHeaderPrefix) + "Authorization", "Bearer secret"},
        {std::string(kMcpProxyHeaderPrefix) + "Accept", "text/event-stream"},
        // Framing belongs to the upstream hop; httplib::Client owns it.
        {std::string(kMcpProxyHeaderPrefix) + "Host", "wrong.test"},
        {std::string(kMcpProxyHeaderPrefix) + "Content-Length", "17"},
        // Unprefixed headers belong to the browser-to-server hop only.
        {"Authorization", "Bearer server-api-key"},
        {"User-Agent", "browser"},
    };
    const McpProxyRequest forwarded =
        parse_mcp_proxy_request(proxy_request("http://127.0.0.1:12315/mcp", headers));
    failures += check(forwarded.ok, "a well-formed proxy request was rejected");
    failures += check(forwarded.headers.size() == 2,
                      "the forwarded header set was not limited to the prefixed, non-framing ones");
    failures += check(forwarded.headers.find("Authorization") != forwarded.headers.end() &&
                          forwarded.headers.find("Authorization")->second == "Bearer secret",
                      "the target's Authorization header was not un-prefixed and forwarded");
    failures += check(forwarded.headers.find("Accept") != forwarded.headers.end(),
                      "a prefixed Accept header was not forwarded");
    failures += check(forwarded.headers.find("Host") == forwarded.headers.end() &&
                          forwarded.headers.find("Content-Length") == forwarded.headers.end(),
                      "upstream framing headers were forwarded");
    failures += check(forwarded.headers.find("User-Agent") == forwarded.headers.end(),
                      "an unprefixed browser header leaked upstream");

    httplib::Request no_url;
    no_url.method = "POST";
    no_url.path   = kMcpProxyPath;
    failures += check(!parse_mcp_proxy_request(no_url).ok,
                      "a proxy request without a url parameter was accepted");

    // The bug this whole path exists to avoid: relaying an upstream CORS header
    // next to this server's own makes the browser see "*, *" and reject the reply.
    failures += check(is_hop_by_hop_response_header("Access-Control-Allow-Origin") &&
                          is_hop_by_hop_response_header("access-control-expose-headers"),
                      "upstream CORS headers would have been copied onto the relayed response");
    failures += check(is_hop_by_hop_response_header("Content-Length") &&
                          is_hop_by_hop_response_header("Transfer-Encoding") &&
                          is_hop_by_hop_response_header("Connection"),
                      "upstream framing headers would have been copied onto a chunked response");
    failures += check(!is_hop_by_hop_response_header("Mcp-Session-Id") &&
                          !is_hop_by_hop_response_header("Content-Type") &&
                          !is_hop_by_hop_response_header("Cache-Control"),
                      "a header the MCP client needs was dropped from the relayed response");

    failures += check(header_name_is("Content-Type", "content-type") &&
                          header_name_is("content-TYPE", "content-type") &&
                          !header_name_is("Content-Length", "content-type"),
                      "case-insensitive header matching is wrong");

    if (failures == 0) { std::cout << "mcp proxy ok\n"; }
    return failures == 0 ? 0 : 1;
}
