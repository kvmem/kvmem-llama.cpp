#include "serve/http_transport.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#if defined(__linux__)
#    include <arpa/inet.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <unistd.h>
#endif

namespace {

using ninfer::serve::ClientDisconnected;
using ninfer::serve::SseTransport;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int test_sse_transport() {
    using namespace std::chrono_literals;
    int failures = 0;

    std::vector<std::string> writes;
    bool writable = true;
    bool write_ok = true;
    httplib::DataSink sink;
    sink.write = [&](const char* data, std::size_t size) {
        if (!write_ok) { return false; }
        writes.emplace_back(data, size);
        return true;
    };
    sink.is_writable = [&] { return writable; };

    std::atomic<bool> cancelled{false};
    const SseTransport::Clock::time_point start{100s};
    SseTransport transport(sink, cancelled, 5s, start);

    failures += check(!transport.poll(start + 4999ms) && writes.empty() && !cancelled,
                      "SSE transport emitted a heartbeat before the quiet interval");
    failures +=
        check(!transport.poll(start + 5s) &&
                  writes == std::vector<std::string>{std::string(SseTransport::kHeartbeatComment)},
              "SSE transport did not emit the standard comment heartbeat on time");
    failures += check(!transport.poll(start + 9999ms) && writes.size() == 1,
                      "SSE heartbeat did not reset the quiet interval");

    transport.write("data: token\n\n", start + 10s);
    failures += check(!transport.poll(start + 14999ms) && writes.size() == 2,
                      "normal SSE output did not reset the heartbeat interval");
    failures += check(!transport.poll(start + 15s) && writes.size() == 3 &&
                          writes.back() == SseTransport::kHeartbeatComment,
                      "quiet SSE output did not resume heartbeats after a data event");

    transport.write(std::vector<std::string>{"data: one\n\n", "data: two\n\n"}, start + 16s);
    failures +=
        check(writes.size() == 5 && writes[3] == "data: one\n\n" && writes[4] == "data: two\n\n",
              "SSE transport changed ordered multi-event output");

    writable = false;
    failures += check(transport.poll(start + 16s) && cancelled,
                      "unwritable SSE transport did not cancel its request");

    bool cancelled_write_threw = false;
    try {
        transport.write("data: unreachable\n\n", start + 16s);
    } catch (const ClientDisconnected&) { cancelled_write_threw = true; }
    failures += check(cancelled_write_threw && writes.size() == 5,
                      "cancelled SSE transport accepted another event");

    httplib::DataSink failed_sink;
    failed_sink.write       = [](const char*, std::size_t) { return false; };
    failed_sink.is_writable = [] { return true; };
    std::atomic<bool> heartbeat_cancelled{false};
    SseTransport failed_heartbeat(failed_sink, heartbeat_cancelled, 5s, start);
    failures += check(failed_heartbeat.poll(start + 5s) && heartbeat_cancelled,
                      "failed SSE heartbeat did not cancel its request");

    write_ok = false;
    writable = true;
    std::atomic<bool> event_cancelled{false};
    SseTransport failed_event(sink, event_cancelled, 5s, start);
    bool failed_event_threw = false;
    try {
        failed_event.write("data: failed\n\n", start);
    } catch (const ClientDisconnected&) { failed_event_threw = true; }
    failures += check(failed_event_threw && event_cancelled,
                      "failed SSE event write did not cancel its request");

    std::atomic<bool> releaser_cancelled{true};
    SseTransport released(sink, releaser_cancelled, 5s, start);
    failures += check(released.poll(start),
                      "response resource cancellation was not observed by SSE transport");

    std::vector<std::string> pings;
    httplib::DataSink ping_sink;
    ping_sink.write = [&](const char* data, std::size_t size) {
        pings.emplace_back(data, size);
        return true;
    };
    ping_sink.is_writable = [] { return true; };
    std::atomic<bool> anthropic_cancelled{false};
    SseTransport anthropic(ping_sink, anthropic_cancelled, 5s, start,
                           SseTransport::kHeartbeatAnthropic);
    failures += check(!anthropic.poll(start + 5s) && pings.size() == 1 &&
                          pings[0].starts_with(SseTransport::kHeartbeatComment) &&
                          pings[0].ends_with("event: ping\ndata: {\"type\":\"ping\"}\n\n"),
                      "the Anthropic heartbeat did not carry the comment and a ping event");
    return failures;
}

int test_ngram_identity() {
    using namespace ninfer::serve;
    int failures = 0;
    ServeOptions options;
    httplib::Request request;
    request.set_header("x-session-affinity", "conversation-a");
    const RequestJson empty = {{"store", false}};
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "disabled archive bound identity");
    options.speculative.ngram_archive_bytes = 512ULL << 20;
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "native identity retained without explicit opt-in");
    options.ngram_native_sessions = true;
    failures += check(resolve_ngram_session(request, empty, options).key == "kilo:conversation-a",
                      "native affinity was not bound");
    request.set_header("x-ninfer-draft-reset", "1");
    failures +=
        check(resolve_ngram_session(request, empty, options).reset, "native reset was ignored");
    request.headers.clear();
    RequestJson codex = {{"store", false},
                         {"client_metadata", {{"thread_id", "task-a"}, {"session_id", "task-a"}}}};
    failures += check(resolve_ngram_session(request, codex, options).key == "codex:task-a",
                      "Codex conversation was not bound");
    codex["client_metadata"]["session_id"] = "task-b";
    failures += check(resolve_ngram_session(request, codex, options).key.empty(),
                      "contradictory Codex identity accepted");
    const RequestJson claude = {
        {"metadata", {{"user_id", R"({"device_id":"shared-device","session_id":"chat-a"})"}}}};
    failures += check(resolve_ngram_session(request, claude, options).key == "claude:chat-a",
                      "Claude conversation was not bound");
    request.set_header("x-session-affinity", "other-task");
    failures += check(resolve_ngram_session(request, claude, options).key.empty(),
                      "ambiguous native identities merged");
    request.headers.clear();
    for (const RequestJson& body :
         {RequestJson{{"prompt_cache_key", "shared"}},
          RequestJson{{"metadata", {{"user_id", "shared-user"}}}},
          RequestJson{{"metadata", {{"user_id", R"({"device_id":"shared-device"})"}}}}}) {
        failures += check(resolve_ngram_session(request, body, options).key.empty(),
                          "non-conversation metadata authorized retention");
    }
    options.ngram_native_sessions = false;
    request.set_header("x-ninfer-draft-session", "fork");
    request.set_header("x-ninfer-draft-parent", "root");
    request.set_header("x-ninfer-draft-generation", "3");
    auto hints = resolve_ngram_session(request, empty, options);
    failures += check(hints.key == "explicit:fork" && hints.parent == "explicit:root" &&
                          hints.parent_generation == 3,
                      "explicit ancestor snapshot was not preserved");
    request.set_header("x-ninfer-draft-session", "different");
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "duplicate session header accepted");
    request.headers.clear();
    request.set_header("x-ninfer-draft-session", std::string(241, 'x'));
    failures += check(resolve_ngram_session(request, empty, options).key.empty(),
                      "unbounded session header accepted");
    return failures;
}

int test_ngram_generation() {
    using namespace ninfer;
    using namespace ninfer::serve;
    int failures = 0;
    const NgramArchiveStats complete{
        .enabled = true, .bound = true, .published = true, .generation = 9};
    httplib::Response response;
    set_ngram_generation_header(response, complete);
    failures += check(response.get_header_value("X-NInfer-Draft-Generation") == "9",
                      "completed archive generation header absent");
    failures += check(ngram_generation_comment(complete) == ": ninfer-draft-generation: 9\n\n",
                      "stream generation is not a standard SSE comment");
    for (int missing = 0; missing < 4; ++missing) {
        auto stats = complete;
        if (missing == 0) { stats.enabled = false; }
        if (missing == 1) { stats.bound = false; }
        if (missing == 2) { stats.published = false; }
        if (missing == 3) { stats.generation = 0; }
        httplib::Response absent;
        set_ngram_generation_header(absent, stats);
        failures += check(!absent.has_header("X-NInfer-Draft-Generation") &&
                              ngram_generation_comment(stats).empty(),
                          "unpublished archive advertised a completed generation");
    }
    return failures;
}

int test_sse_response_headers() {
    httplib::Response response;
    ninfer::serve::prepare_sse_response(response);
    return check(response.get_header_value("Cache-Control") == "no-cache" &&
                     response.get_header_value("X-Accel-Buffering") == "no",
                 "SSE response lost its no-cache or anti-buffering header");
}

int test_prompt_json_member_order() {
    httplib::Request request;
    request.body =
        R"({"model":"claude-local","tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"probe","input":{"zeta":"last","alpha":{"yankee":2,"bravo":true}}}]}],"max_tokens":32})";

    const auto parsed = ninfer::serve::parse_json_body(request);
    return check(
        parsed.at("tools").at(0).at("input_schema").at("properties").dump() ==
                R"({"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}})" &&
            parsed.at("messages").at(0).at("content").at(0).at("input").dump() ==
                R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})",
        "HTTP request JSON changed prompt-bearing object member order");
}

#if defined(__linux__)
class Socket final {
public:
    explicit Socket(int descriptor = -1) : descriptor_(descriptor) {}

    ~Socket() {
        if (descriptor_ >= 0) { ::close(descriptor_); }
    }

    Socket(const Socket&)            = delete;
    Socket& operator=(const Socket&) = delete;

    [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
    int descriptor_ = -1;
};

template <class T>
bool socket_option_equals(int socket, int level, int option, T expected) {
    T actual{};
    socklen_t size = sizeof(actual);
    return ::getsockopt(socket, level, option, &actual, &size) == 0 && size == sizeof(actual) &&
           actual == expected;
}

int test_inherited_socket_liveness() {
    int failures = 0;
    Socket listener(::socket(AF_INET, SOCK_STREAM, 0));
    if (listener.get() < 0) { return check(false, "failed to create HTTP listener test socket"); }
    ninfer::serve::configure_http_server_socket(listener.get());

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
    if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener.get(), 1) != 0) {
        return check(false, "failed to bind HTTP listener test socket");
    }
    socklen_t address_size = sizeof(address);
    if (::getsockname(listener.get(), reinterpret_cast<sockaddr*>(&address), &address_size) != 0) {
        return check(false, "failed to inspect HTTP listener test address");
    }

    Socket client(::socket(AF_INET, SOCK_STREAM, 0));
    if (client.get() < 0 || ::connect(client.get(), reinterpret_cast<const sockaddr*>(&address),
                                      sizeof(address)) != 0) {
        return check(false, "failed to connect HTTP liveness test socket");
    }
    Socket accepted(::accept(listener.get(), nullptr, nullptr));
    if (accepted.get() < 0) { return check(false, "failed to accept HTTP liveness test socket"); }

    failures += check(socket_option_equals(accepted.get(), SOL_SOCKET, SO_KEEPALIVE, 1),
                      "accepted HTTP socket did not inherit SO_KEEPALIVE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPIDLE, 10),
                      "accepted HTTP socket did not inherit TCP_KEEPIDLE");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPINTVL, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPINTVL");
    failures += check(socket_option_equals(accepted.get(), IPPROTO_TCP, TCP_KEEPCNT, 3),
                      "accepted HTTP socket did not inherit TCP_KEEPCNT");
    failures += check(
        socket_option_equals<unsigned int>(accepted.get(), IPPROTO_TCP, TCP_USER_TIMEOUT, 15000U),
        "accepted HTTP socket did not inherit TCP_USER_TIMEOUT");
    return failures;
}
#endif

} // namespace

int main() {
    int failures = test_ngram_identity() + test_ngram_generation() + test_sse_transport() +
                   test_sse_response_headers() + test_prompt_json_member_order();
#if defined(__linux__)
    failures += test_inherited_socket_liveness();
#endif
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
