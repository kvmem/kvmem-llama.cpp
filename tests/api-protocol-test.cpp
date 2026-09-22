#include "kvmem-api-openai.h"
#include "kvmem-api-anthropic.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x);                                                \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace anth = kvmem_anthropic;

static json body() {
    return {
        {"model", "local"}, {"max_tokens", 128}, {"messages", json::array({{{"role", "user"}, {"content", "Hello"}}})}};
}
static json tool(const char *name) {
    return {{"name", name},
            {"description", "Return a value"},
            {"input_schema",
             {{"type", "object"}, {"properties", {{"value", {{"type", "string"}}}}}, {"required", {"value"}}}}};
}
static anth::request parse(const json &value, bool count = false) {
    return anth::parse(value, "local", 4096, {}, json::object(), count);
}
static bool rejects(const json &value) {
    try {
        parse(value);
        return false;
    } catch (const std::exception &) {
        return true;
    }
}
static json payload(const std::string &frame) {
    return json::parse(frame.substr(frame.find("data: ") + 6));
}

static void requests() {
    auto value = body();
    value["system"] = json::array({{{"type", "text"}, {"text", "Rules"}, {"cache_control", {{"type", "ephemeral"}}}}});
    value["tools"] = {tool("echo"), tool("other")};
    value["tool_choice"] = {{"type", "tool"}, {"name", "echo"}, {"disable_parallel_tool_use", true}};
    auto parsed = parse(value);
    CHECK(!parsed.chat.enable_thinking && parsed.chat.template_kwargs.at("enable_thinking") == "false");
    CHECK(parsed.chat.msgs[0].role == "system" && parsed.chat.msgs[0].content == "Rules");
    CHECK(parsed.chat.tools.size() == 1 && parsed.chat.tools[0].name == "echo");
    CHECK(parsed.required_tool && parsed.single_tool && !parsed.chat.parallel_tool_calls);
    value["messages"].push_back(
        {{"role", "assistant"},
         {"content",
          {{{"type", "text"}, {"text", "Calling"}},
           {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "echo"}, {"input", {{"value", "hi"}}}}}}});
    value["messages"].push_back({{"role", "user"},
                                 {"content",
                                  {{{"type", "tool_result"}, {"tool_use_id", "toolu_1"}, {"content", "hi"}},
                                   {{"type", "text"}, {"text", "Continue"}}}}});
    parsed = parse(value);
    CHECK(parsed.chat.msgs.size() == 5);
    CHECK(parsed.chat.msgs[2].tool_calls[0].arguments == "{\"value\":\"hi\"}");
    CHECK(parsed.chat.msgs[3].role == "tool" && parsed.chat.msgs[3].tool_call_id == "toolu_1");
    CHECK(parsed.chat.last_user == "Continue");
    auto broken = value;
    broken["messages"][2]["content"][0]["tool_use_id"] = "unknown";
    CHECK(rejects(broken));
    broken = value;
    broken["messages"][2]["content"].erase(0);
    CHECK(rejects(broken));
    broken = body();
    broken["messages"][0]["content"] = {{{"type", "image"}, {"source", json::object()}}};
    CHECK(rejects(broken));
    for (const auto &limit : {json(0), json(-1), json(true), json(1.5), json(nullptr), json(UINT64_MAX), json(4097)}) {
        broken = body();
        broken["max_tokens"] = limit;
        CHECK(rejects(broken));
    }
    for (const auto &patch :
         {json{{"model", "unknown"}}, json{{"stream", "true"}}, json{{"temperature", 1.1}},
          json{{"thinking", {{"type", "enabled"}, {"budget_tokens", 100}}}}, json{{"tool_choice", {{"type", "any"}}}},
          json{{"tool_choice", {{"type", "auto"}, {"name", "echo"}}}}, json{{"stop_sequences", {""}}},
          json{{"top_k", 1.5}}, json{{"temperature", nullptr}}, json{{"output_config", json::object()}}}) {
        broken = body();
        broken.update(patch);
        CHECK(rejects(broken));
    }
    broken = body();
    broken.erase("max_tokens");
    CHECK(parse(broken, true).chat.msgs.size() == 1);
    httplib::Request req;
    try {
        anth::version(req);
        CHECK(false);
    } catch (const anth::request_error &) {
    }
    req.set_header("anthropic-version", "2023-06-01");
    anth::version(req);
    CHECK(anth::error_body("bad", 401)["error"]["type"] == "authentication_error");
}

static void stops() {
    anth::stop_buffer buffer;
    buffer.user = {"<END>"};
    buffer.all = buffer.user;
    CHECK(buffer.visible("Hello<EN", true) == "Hello");
    CHECK(buffer.visible("Hello<EX", true) == "Hello<EX");
    CHECK(buffer.visible("Hello<END>more", true) == "Hello");
    CHECK(buffer.user_stop() && buffer.matched == "<END>");
    CHECK(buffer.visible("Hello<END>more text", false) == "Hello");
    anth::stop_buffer utf8;
    CHECK(utf8.visible("a\xe4\xb8", true) == "a");
    CHECK(utf8.visible("a\xe4\xb8\xad", true) == "a\xe4\xb8\xad");
}

static void streaming() {
    auto input = body();
    input["tools"] = {tool("echo"), tool("other")};
    anth::response encoder(parse(input));
    common_chat_params params;
    encoder.init(params, true, "test", "local", 12, 0);
    CHECK(payload(encoder.start()[0])["message"]["usage"]["input_tokens"] == 12);
    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = "Hello";
    auto frames = encoder.update_message(msg);
    CHECK(frames.size() == 2 && payload(frames[0])["content_block"]["type"] == "text");
    msg.tool_calls.push_back({"echo", "{\"value\":", "ignored-model-id"});
    frames = encoder.update_message(msg);
    CHECK(frames.size() == 3 && payload(frames[0])["type"] == "content_block_stop");
    CHECK(payload(frames[1])["content_block"]["id"] == "toolu_test_1");
    CHECK(payload(frames[2])["delta"]["partial_json"] == "{\"value\":");
    msg.tool_calls[0].arguments += "\"hi\"}";
    frames = encoder.update_message(msg);
    CHECK(frames.size() == 1 && payload(frames[0])["delta"]["partial_json"] == "\"hi\"}");
    msg.tool_calls.push_back({"other", "{}", ""});
    frames = encoder.update_message(msg);
    CHECK(frames.size() == 3 && payload(frames[1])["index"] == 2);
    frames = encoder.finish(false, 9, 0, json::object());
    CHECK(frames.size() == 3 && payload(frames[1])["delta"]["stop_reason"] == "tool_use");
    CHECK(payload(frames.back())["type"] == "message_stop");
    CHECK(frames.back().find("[DONE]") == std::string::npos);
    anth::response fresh(parse(body()));
    fresh.init(params, false, "other", "local", 3, 0);
    fresh.start();
    frames = fresh.update("hi", true, json::object());
    CHECK(payload(frames[0])["index"] == 0);
    CHECK(payload(fresh.finish(true, 1, 0, json::object())[1])["delta"]["stop_reason"] == "max_tokens");
    CHECK(anth::response::error_end().empty());
}

static void openai_contract() {
    auto value = body();
    value.erase("model");
    ChatRequest parsed;
    std::string error;
    CHECK(parse_chat_request(value, parsed, error) && parsed.msgs[0].content == "Hello");
    CHECK(usage_json(10, 2, 20) == json({{"prompt_tokens", 10},
                                         {"completion_tokens", 2},
                                         {"total_tokens", 12},
                                         {"prompt_cache_hit_tokens", 10},
                                         {"prompt_cache_miss_tokens", 0}}));
    common_chat_msg_diff diff;
    diff.tool_call_index = 1;
    diff.tool_call_delta = {"echo", "{", "call_1"};
    CHECK(chat_diff_to_delta(diff)["tool_calls"][0]["function"]["arguments"] == "{");
    kvmem_openai_response encoder;
    encoder.init(common_chat_params{}, false, "test", "local", 10, 1);
    const auto first = payload(encoder.start()[0]);
    CHECK(first["object"] == "chat.completion.chunk" && first["choices"][0]["delta"]["content"].is_null());
    encoder.update("Hello", false, json::object());
    const auto frames = encoder.finish(false, 1, 0, json::object());
    CHECK(frames.size() == 3 && frames.back() == "data: [DONE]\n\n");
    CHECK(payload(frames[0])["choices"][0]["finish_reason"] == "stop");
    CHECK(payload(frames[1])["choices"].empty());
    httplib::Response res;
    encoder.result(res, "Hello", 1, false, 0, json::object());
    CHECK(json::parse(res.body)["choices"][0]["message"]["content"] == "Hello");
    kvmem_openai_response::error(res, 400, "bad");
    CHECK(res.body == "{\"error\":\"bad\"}");
}

int main() {
    requests();
    stops();
    streaming();
    openai_contract();
    std::puts("PASS: request validation, tool history, stop/UTF-8 buffering, SSE lifecycle, OpenAI wire contract");
}
