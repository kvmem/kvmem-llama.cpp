// GPU-free tests for the OpenAI Responses -> Chat Completions request bridge.
// The response body shape is exercised end to end by scripts/test_server_api.py
// against a live server; here we only check the conversion the server performs
// before it reaches the chat pipeline.

#include "kvmem-responses.h"
#include "kvmem-responses-stream.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using nlohmann::json;

void test_image_media_parser();

static json convert(const std::string & body) {
    return json::parse(kvmem_responses_to_chatcmpl(body));
}

static bool throws(const std::string & body) {
    try {
        kvmem_responses_to_chatcmpl(body);
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

static void test_string_input() {
    const auto out = convert(R"({"input": "hello"})");
    CHECK(!out.contains("input"));
    CHECK(out["messages"].size() == 1);
    CHECK(out["messages"][0]["role"] == "user");
    CHECK(out["messages"][0]["content"] == "hello");
}

static void test_instructions_become_system() {
    const auto out = convert(R"({"input": "hello", "instructions": "be brief"})");
    CHECK(!out.contains("instructions"));
    CHECK(out["messages"].size() == 2);
    CHECK(out["messages"][0]["role"] == "system");
    CHECK(out["messages"][0]["content"] == "be brief");
    CHECK(out["messages"][1]["role"] == "user");
}

static void test_input_item_list() {
    const auto out = convert(R"({"input": [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]})");
    CHECK(out["messages"].size() == 1);
    CHECK(out["messages"][0]["content"][0]["type"] == "text");
    CHECK(out["messages"][0]["content"][0]["text"] == "hi");
}

static void test_image_input_converts_in_order() {
    const auto out = convert(R"({"input": [{"role": "user", "content": [
        {"type": "input_text", "text": "Compare these images"},
        {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
        {"type": "input_image", "image_url": "https://example.com/second.png"}
    ]}]})");
    const auto & content = out["messages"][0]["content"];
    CHECK(content.size() == 3);
    CHECK(content[0]["type"] == "text");
    CHECK(content[0]["text"] == "Compare these images");
    CHECK(content[1]["type"] == "image_url");
    CHECK(content[1]["image_url"]["url"] == "data:image/png;base64,aGVsbG8=");
    CHECK(content[2]["type"] == "image_url");
    CHECK(content[2]["image_url"]["url"] == "https://example.com/second.png");
}

static void test_function_call_output() {
    const auto out = convert(R"({"input": [
        {"role": "user", "content": "go"},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"}
    ]})");
    CHECK(out["messages"].size() == 2);
    CHECK(out["messages"][1]["role"] == "tool");
    CHECK(out["messages"][1]["tool_call_id"] == "call_1");
    CHECK(out["messages"][1]["content"] == "42");
}

static void test_reasoning_item_summary_folds_into_content() {
    // @ai-sdk/openai keeps only the reasoning `summary` and replays it without a
    // `content` array. Upstream reads `content[0].text` and rejects the item
    // otherwise, which used to fail the whole second turn of a tool loop.
    const auto out = convert(R"({"input": [
        {"role": "user", "content": "go"},
        {"type": "reasoning", "encrypted_content": "", "summary": [
            {"type": "summary_text", "text": "I should list the files."}
        ]},
        {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"}
    ]})");
    // Three messages, not four: upstream merges the function_call into the
    // assistant turn the reasoning item opened.
    CHECK(out["messages"].size() == 3);
    CHECK(out["messages"][1]["role"] == "assistant");
    CHECK(out["messages"][1]["reasoning_content"] == "I should list the files.");
    // The function_call still merges into that same assistant turn.
    CHECK(out["messages"][1]["tool_calls"].size() == 1);
    CHECK(out["messages"][1]["tool_calls"][0]["function"]["name"] == "bash");
    CHECK(out["messages"][2]["role"] == "tool");
    CHECK(out["messages"][2]["content"] == "a.txt");
}

static void test_reasoning_item_with_content_is_left_alone() {
    // OpenAI's own clients send the raw `content` array; it must pass through
    // rather than be overwritten by the (possibly different) summary.
    const auto out = convert(R"({"input": [
        {"role": "user", "content": "go"},
        {"type": "reasoning", "encrypted_content": "", "summary": [
            {"type": "summary_text", "text": "the summary"}
        ], "content": [{"type": "reasoning_text", "text": "the raw reasoning"}]}
    ]})");
    CHECK(out["messages"][1]["reasoning_content"] == "the raw reasoning");
}

static void test_assistant_message_without_type_is_typed() {
    // @ai-sdk/openai replays the assistant turn as {role, content} with no
    // `type`; upstream only recognises an output message by type == "message"
    // and otherwise dies on "Cannot determine type of 'item'".
    const auto out = convert(R"({"input": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "the answer"},
        {"role": "user", "content": "again"}
    ]})");
    CHECK(out["messages"].size() == 3);
    CHECK(out["messages"][1]["role"] == "assistant");
    CHECK(out["messages"][1]["content"][0]["text"] == "the answer");
    CHECK(out["messages"][1]["content"][0]["type"] == "text");
}

static void test_tools_max_tokens_and_reasoning() {
    const auto out = convert(R"({
        "input": "go",
        "max_output_tokens": 32,
        "reasoning": {"effort": "low"},
        "tools": [{"type": "function", "name": "f", "description": "d", "parameters": {}}]
    })");
    CHECK(!out.contains("max_output_tokens"));
    CHECK(out["max_tokens"] == 32);
    CHECK(out["reasoning_effort"] == "low");
    CHECK(!out.contains("reasoning"));
    CHECK(out["tools"].size() == 1);
    CHECK(out["tools"][0]["type"] == "function");
    CHECK(out["tools"][0]["function"]["name"] == "f");
}

static void test_stream_passthrough() {
    // The server rejects streaming Responses itself; the bridge must not drop it.
    const auto out = convert(R"({"input": "hi", "stream": true})");
    CHECK(out["stream"] == true);
}

static void test_rejects_bad_input() {
    CHECK(throws(R"({"model": "m"})"));                                  // no input
    CHECK(throws(R"({"input": "hi", "previous_response_id": "resp_1"})")); // unsupported
    CHECK(throws(R"({"input": 5})"));                                    // wrong input type
    CHECK(throws("not json"));
    CHECK(throws(R"({"input": [{"role": "user", "content": [
        {"type": "input_image"}
    ]}]})"));
    CHECK(throws(R"({"input": [{"role": "user", "content": [
        {"type": "input_image", "file_id": "file_123"}
    ]}]})"));
}

// ---------------------------------------------------------------------------
// Streaming event sequence
// ---------------------------------------------------------------------------

// "event: <type>\ndata: <json>\n\n" -> the parsed data object.
static json decode_event(const std::string & line) {
    CHECK(line.compare(0, 7, "event: ") == 0);
    const size_t nl = line.find('\n');
    CHECK(nl != std::string::npos);
    CHECK(line.compare(nl + 1, 6, "data: ") == 0);
    CHECK(line.substr(line.size() - 2) == "\n\n");
    const std::string type = line.substr(7, nl - 7);
    json data = json::parse(line.substr(nl + 7));
    CHECK(data["type"] == type);   // the event: line and the type field must agree
    return data;
}

static std::vector<std::string> types_of(const std::vector<std::string> & raw) {
    std::vector<std::string> types;
    for (const std::string & line : raw) {
        types.push_back(decode_event(line)["type"]);
    }
    return types;
}

// The diff fields the server actually sees, built by hand: the emitter is a
// pure function of these, so no parser is needed to test it.
static common_chat_msg_diff text_diff(const std::string & content, const std::string & reasoning) {
    common_chat_msg_diff d;
    d.content_delta = content;
    d.reasoning_content_delta = reasoning;
    return d;
}

static common_chat_msg_diff call_name_diff(const std::string & id, const std::string & name, size_t index = 0) {
    common_chat_msg_diff d;
    d.tool_call_index = index;
    d.tool_call_delta.id = id;
    d.tool_call_delta.name = name;
    return d;
}

static common_chat_msg_diff call_args_diff(const std::string & args, size_t index = 0) {
    common_chat_msg_diff d;
    d.tool_call_index = index;
    d.tool_call_delta.arguments = args;
    return d;
}

static void test_stream_created() {
    KvMemResponsesStreamState st;
    const auto events = kvmem_responses_stream_created(st, "abc", "some-model");
    CHECK(events.size() == 2);
    const auto types = types_of(events);
    CHECK(types[0] == "response.created");
    CHECK(types[1] == "response.in_progress");
    const json r = decode_event(events[0])["response"];
    CHECK(r["id"] == "resp_abc");
    CHECK(r["object"] == "response");
    CHECK(r["status"] == "in_progress");
    CHECK(r["model"] == "some-model");
}

static void test_stream_reasoning_block_opens_once() {
    KvMemResponsesStreamState st;
    const auto first_raw = kvmem_responses_stream_events(st, text_diff("", "hmm"), "abc");
    const auto first = types_of(first_raw);
    // The item, its summary block, then the same text on both channels: the raw
    // `reasoning_text` upstream clients read, and the `reasoning_summary_text`
    // the OpenAI SDKs display and echo back.
    CHECK(first.size() == 4);
    CHECK(first[0] == "response.output_item.added");
    CHECK(first[1] == "response.reasoning_summary_part.added");
    CHECK(first[2] == "response.reasoning_text.delta");
    CHECK(first[3] == "response.reasoning_summary_text.delta");

    const json item = decode_event(first_raw[0])["item"];
    CHECK(item["id"] == "rs_abc");
    CHECK(item["type"] == "reasoning");
    CHECK(item["status"] == "in_progress");
    // The item opens with both channels present but empty; the text itself
    // arrives in the two delta events above.
    CHECK(item["summary"].size() == 1);
    CHECK(item["summary"][0]["type"] == "summary_text");
    CHECK(item["summary"][0]["text"] == "");
    CHECK(item["content"].size() == 1);
    CHECK(item["content"][0]["type"] == "reasoning_text");
    CHECK(item["content"][0]["text"] == "");
    CHECK(decode_event(first_raw[2])["content_index"] == 0);
    CHECK(decode_event(first_raw[3])["summary_index"] == 0);

    // The call above must have recorded the open itself; the caller never
    // writes the flag. A test that set it here would pass against an emitter
    // that re-opens the block on every token, which is the bug this covers.
    const auto second_raw = kvmem_responses_stream_events(st, text_diff("", " more"), "abc");
    const auto second = types_of(second_raw);
    CHECK(second.size() == 2);
    CHECK(second[0] == "response.reasoning_text.delta");
    CHECK(second[1] == "response.reasoning_summary_text.delta");
    const json d = decode_event(second_raw[0]);
    CHECK(d["delta"] == " more");
    CHECK(d["item_id"] == "rs_abc");
}

static void test_stream_text_block_opens_item_and_part() {
    KvMemResponsesStreamState st;
    const auto events = kvmem_responses_stream_events(st, text_diff("hi", ""), "abc");
    const auto types = types_of(events);
    CHECK(types.size() == 3);
    CHECK(types[0] == "response.output_item.added");
    CHECK(types[1] == "response.content_part.added");
    CHECK(types[2] == "response.output_text.delta");

    const json item = decode_event(events[0])["item"];
    CHECK(item["id"] == "msg_abc");
    CHECK(item["type"] == "message");
    CHECK(item["role"] == "assistant");
    const json part = decode_event(events[1])["part"];
    CHECK(part["type"] == "output_text");
    CHECK(part["text"] == "");
    CHECK(decode_event(events[2])["item_id"] == "msg_abc");
    CHECK(decode_event(events[2])["delta"] == "hi");
}

static void test_stream_reasoning_then_text_sequence() {
    // A hybrid model emits reasoning first, then the answer; both blocks open.
    KvMemResponsesStreamState st;
    std::vector<std::string> all;
    for (const auto & raw : kvmem_responses_stream_events(st, text_diff("", "think"), "abc")) {
        all.push_back(raw);
    }
    for (const auto & raw : kvmem_responses_stream_events(st, text_diff("ans", ""), "abc")) {
        all.push_back(raw);
    }
    const auto types = types_of(all);
    const std::vector<std::string> want = {
        "response.output_item.added",     // reasoning item
        "response.reasoning_summary_part.added",
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
        "response.output_item.added",     // message item
        "response.content_part.added",
        "response.output_text.delta",
    };
    CHECK(types == want);
}

static void test_stream_function_call_ids() {
    KvMemResponsesStreamState st;
    const auto first_raw = kvmem_responses_stream_events(st, call_name_diff("call_1", "get_weather"), "abc");
    const auto added = types_of(first_raw);
    CHECK(added.size() == 1);
    CHECK(added[0] == "response.output_item.added");
    // The incremental parser can repeat or extend a name. Neither reopens it.
    CHECK(kvmem_responses_stream_events(st, call_name_diff("call_1", "get_weather"), "abc").empty());
    CHECK(kvmem_responses_stream_events(st, call_name_diff("call_1", "get_weather_v2"), "abc").empty());

    const json item = decode_event(first_raw[0])["item"];
    CHECK(item["id"] == "fc_call_1");
    CHECK(item["call_id"] == "call_1");      // echoed verbatim, as the non-streaming path does
    CHECK(item["name"] == "get_weather");
    CHECK(item["arguments"] == "");

    // The name must be captured so argument deltas can address the item.
    const json d = decode_event(kvmem_responses_stream_events(st, call_args_diff("{\"c\""), "abc")[0]);
    CHECK(d["type"] == "response.function_call_arguments.delta");
    CHECK(d["item_id"] == "fc_call_1");
    CHECK(d["output_index"] == 0);
    CHECK(d["delta"] == "{\"c\"");
}

static void test_stream_empty_delta_is_silent() {
    // Partial parsing yields empty diffs before a block settles; emit nothing
    // rather than an event the client's state machine cannot consume.
    KvMemResponsesStreamState st;
    CHECK(kvmem_responses_stream_events(st, text_diff("", ""), "abc").empty());
}

static void test_stream_done_text_only() {
    KvMemResponsesStreamState st;
    st.text_started = true;
    common_chat_msg msg;
    msg.role = "assistant";
    msg.content = "hello";

    const auto events = kvmem_responses_stream_done(st, msg, "abc", "some-model", 7, 3, 2);
    const auto types = types_of(events);
    const std::vector<std::string> want = {
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    };
    CHECK(types == want);

    CHECK(decode_event(events[0])["text"] == "hello");
    CHECK(decode_event(events[0])["item_id"] == "msg_abc");
    const json part = decode_event(events[1])["part"];
    CHECK(part["type"] == "output_text");
    CHECK(part["text"] == "hello");
    CHECK(part["annotations"].is_array());
    const json item = decode_event(events[2])["item"];
    CHECK(item["type"] == "message");
    CHECK(item["status"] == "completed");
    CHECK(item["id"] == "msg_abc");
    CHECK(item["content"][0]["text"] == "hello");

    const json r = decode_event(events[3])["response"];
    CHECK(r["id"] == "resp_abc");
    CHECK(r["object"] == "response");
    CHECK(r["status"] == "completed");
    CHECK(r["model"] == "some-model");
    CHECK(r["output"].size() == 1);
    CHECK(r["usage"]["input_tokens"] == 7);
    CHECK(r["usage"]["output_tokens"] == 3);
    CHECK(r["usage"]["total_tokens"] == 10);
    CHECK(r["usage"]["input_tokens_details"]["cached_tokens"] == 2);
}

static void test_stream_done_reasoning_and_message() {
    KvMemResponsesStreamState st;
    // Feed both blocks through the delta path so the terminator reads state the
    // emitter produced rather than state the test hand-wrote.
    kvmem_responses_stream_events(st, text_diff("", "think"), "abc");
    kvmem_responses_stream_events(st, text_diff("answer", ""), "abc");
    common_chat_msg msg;
    msg.role = "assistant";
    msg.reasoning_content = "think";
    msg.content = "answer";

    const auto events = kvmem_responses_stream_done(st, msg, "abc", "m", 1, 2, 0);
    const auto types = types_of(events);
    // The summary channel closes itself; the raw reasoning_text channel has no
    // *.done upstream, so the item close is all that ends it.
    const std::vector<std::string> want = {
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.output_item.done",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    };
    CHECK(types == want);

    const json reasoning = decode_event(events[2])["item"];
    CHECK(reasoning["id"] == "rs_abc");
    CHECK(reasoning["type"] == "reasoning");
    CHECK(reasoning["content"][0]["type"] == "reasoning_text");
    CHECK(reasoning["content"][0]["text"] == "think");
    // The same text comes back on the summary channel, which is the one the
    // OpenAI SDKs keep: it is what a client echoes to us on the next turn.
    CHECK(reasoning["summary"][0]["type"] == "summary_text");
    CHECK(reasoning["summary"][0]["text"] == "think");
    CHECK(decode_event(events[0])["text"] == "think");
    CHECK(decode_event(events[0])["item_id"] == "rs_abc");
    CHECK(decode_event(events[1])["part"]["text"] == "think");

    const json r = decode_event(events[6])["response"];
    CHECK(r["output"].size() == 2);
    CHECK(r["output"][0]["type"] == "reasoning");
    CHECK(r["output"][1]["type"] == "message");
}

static void test_stream_done_tool_call() {
    KvMemResponsesStreamState st;
    // Drive the delta path so the terminator reads state the emitter recorded:
    // the tool call names itself once, which is what opens the item.
    kvmem_responses_stream_events(st, text_diff("ok", ""), "abc");
    kvmem_responses_stream_events(st, call_name_diff("call_1", "get_weather"), "abc");
    CHECK(st.function_calls.size() == 1);
    CHECK(st.function_calls[0].item_id == "fc_call_1");
    common_chat_msg msg;
    msg.role = "assistant";
    common_chat_tool_call tc;
    tc.id = "call_1";
    tc.name = "get_weather";
    tc.arguments = "{\"city\":\"SF\"}";
    msg.tool_calls.push_back(tc);

    const auto events = kvmem_responses_stream_done(st, msg, "abc", "m", 5, 4, 0);
    const auto types = types_of(events);
    const std::vector<std::string> want = {
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    };
    CHECK(types == want);

    const json arg_done = decode_event(events[0]);
    CHECK(arg_done["item_id"] == "fc_call_1");
    CHECK(arg_done["arguments"] == "{\"city\":\"SF\"}");
    const json item = decode_event(events[1])["item"];
    CHECK(item["id"] == "fc_call_1");
    CHECK(item["call_id"] == "call_1");
    CHECK(item["name"] == "get_weather");
    CHECK(item["status"] == "completed");
    const json r = decode_event(events[2])["response"];
    CHECK(r["output"].size() == 1);
    CHECK(r["output"][0]["type"] == "function_call");
}

static void test_stream_two_tool_calls_keep_their_ids_and_indices() {
    KvMemResponsesStreamState st;
    const auto first = kvmem_responses_stream_events(st, call_name_diff("call_1", "first_tool", 0), "abc");
    const auto second = kvmem_responses_stream_events(st, call_name_diff("call_2", "second_tool", 1), "abc");
    CHECK(decode_event(first[0])["output_index"] == 0);
    CHECK(decode_event(second[0])["output_index"] == 1);
    const auto first_args = kvmem_responses_stream_events(st, call_args_diff("{\"a\":1}", 0), "abc");
    const auto second_args = kvmem_responses_stream_events(st, call_args_diff("{\"b\":2}", 1), "abc");
    CHECK(decode_event(first_args[0])["item_id"] == "fc_call_1");
    CHECK(decode_event(first_args[0])["output_index"] == 0);
    CHECK(decode_event(second_args[0])["item_id"] == "fc_call_2");
    CHECK(decode_event(second_args[0])["output_index"] == 1);

    common_chat_msg msg;
    msg.role = "assistant";
    common_chat_tool_call one, two;
    one.id = "call_1"; one.name = "first_tool"; one.arguments = "{\"a\":1}";
    two.id = "call_2"; two.name = "second_tool"; two.arguments = "{\"b\":2}";
    msg.tool_calls = {one, two};
    const auto done = kvmem_responses_stream_done(st, msg, "abc", "m", 5, 4, 0);
    CHECK(types_of(done) == std::vector<std::string>({
        "response.function_call_arguments.done", "response.output_item.done",
        "response.function_call_arguments.done", "response.output_item.done", "response.completed"}));
    CHECK(decode_event(done[0])["item_id"] == "fc_call_1");
    CHECK(decode_event(done[0])["output_index"] == 0);
    CHECK(decode_event(done[1])["item"]["id"] == "fc_call_1");
    CHECK(decode_event(done[2])["item_id"] == "fc_call_2");
    CHECK(decode_event(done[2])["output_index"] == 1);
    CHECK(decode_event(done[3])["item"]["id"] == "fc_call_2");
    CHECK(decode_event(done[4])["response"]["output"].size() == 2);
}

static void test_stream_done_empty_content_still_completes() {
    // max_tokens hit before any text: the stream must still terminate with
    // response.completed, with an empty output array.
    KvMemResponsesStreamState st;
    common_chat_msg msg;
    msg.role = "assistant";
    const auto events = kvmem_responses_stream_done(st, msg, "abc", "m", 3, 0, 3);
    const auto types = types_of(events);
    CHECK(types.size() == 1);
    CHECK(types[0] == "response.completed");
    const json r = decode_event(events[0])["response"];
    CHECK(r["output"].is_array());
    CHECK(r["output"].empty());
}

int main() {
    test_string_input();
    test_instructions_become_system();
    test_input_item_list();
    test_image_input_converts_in_order();
    test_image_media_parser();
    test_function_call_output();
    test_reasoning_item_summary_folds_into_content();
    test_reasoning_item_with_content_is_left_alone();
    test_assistant_message_without_type_is_typed();
    test_tools_max_tokens_and_reasoning();
    test_stream_passthrough();
    test_rejects_bad_input();
    test_stream_created();
    test_stream_reasoning_block_opens_once();
    test_stream_text_block_opens_item_and_part();
    test_stream_reasoning_then_text_sequence();
    test_stream_function_call_ids();
    test_stream_empty_delta_is_silent();
    test_stream_done_text_only();
    test_stream_done_reasoning_and_message();
    test_stream_done_tool_call();
    test_stream_two_tool_calls_keep_their_ids_and_indices();
    test_stream_done_empty_content_still_completes();
    std::puts("responses request bridge tests PASS");
}
