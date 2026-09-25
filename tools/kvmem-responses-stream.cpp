#include "kvmem-responses-stream.h"

#include "nlohmann/json.hpp"

#include <ctime>
#include <utility>

using json = nlohmann::json;

// Item ids follow the non-streaming shape in llama-kvmem-server.cpp
// (responses_output_items): "rs_"/"msg_" carry the request id verbatim while
// "fc_" carries the tool call id.
static std::string kvmem_responses_reasoning_id(const std::string & response_id) {
    return "rs_" + response_id;
}

static std::string kvmem_responses_message_id(const std::string & response_id) {
    return "msg_" + response_id;
}

// A completed reasoning item, identical to the one responses_output_items()
// emits for the non-streaming path.
//
// The item carries the same text twice, in the two fields OpenAI defines:
// `summary` (the client-facing "reasoning_summary_text" the SDKs display and
// echo back) and `content` (the raw "reasoning_text", which is what upstream
// llama.cpp reads when a client sends this item back as input). Emitting only
// `content` leaves `summary` empty, and a client that keeps just the summary --
// @ai-sdk/openai does -- then returns an item with no `content` at all, which
// server_chat_convert_responses_to_chatcmpl rejects outright.
static json kvmem_responses_reasoning_item(const std::string & reasoning_text) {
    json summary = json::array();
    json content = json::array();
    if (!reasoning_text.empty()) {
        summary.push_back(json{{"text", reasoning_text}, {"type", "summary_text"}});
        content.push_back(json{{"text", reasoning_text}, {"type", "reasoning_text"}});
    }
    return json{
        {"summary", summary},
        {"type", "reasoning"},
        {"content", content},
        {"encrypted_content", ""},
    };
}

static json kvmem_responses_content_part(const std::string & text) {
    return json{
        {"type", "output_text"},
        {"annotations", json::array()},
        {"logprobs", json::array()},
        {"text", text},
    };
}

// A completed message item. `status` is included to match the non-streaming
// path; the newer upstream emitter drops it from the streamed item.
static json kvmem_responses_message_item(const common_chat_msg & msg, const std::string & message_id) {
    return json{
        {"content", json::array({kvmem_responses_content_part(msg.content)})},
        {"id", message_id},
        {"role", msg.role.empty() ? std::string("assistant") : msg.role},
        {"status", "completed"},
        {"type", "message"},
    };
}

static json kvmem_responses_function_call_item(const common_chat_tool_call & tool_call) {
    return json{
        {"id", "fc_" + tool_call.id},
        {"type", "function_call"},
        {"status", "completed"},
        {"arguments", tool_call.arguments},
        // Round-trips the client's function_call_output: the id the request
        // carried is the one echoed back, matching responses_output_items.
        {"call_id", tool_call.id},
        {"name", tool_call.name},
    };
}

// One SSE event. Upstream writes both an `event:` line and a `type` field; the
// Python SDK dispatches on either, so emit both.
//
// Every event carries a `sequence_number`, and every event that names an item
// carries its `output_index`. Both are required by the OpenAI wire format and
// are what a client uses to place deltas into its own item array; an event
// without them is well-formed JSON that a strict client will not apply.
static std::string kvmem_responses_event(
        KvMemResponsesStreamState * state, const std::string & type, json data) {
    data["type"] = type;
    if (state != nullptr) {
        data["sequence_number"] = state->next_sequence++;
    }
    return "event: " + type + "\ndata: " + data.dump() + "\n\n";
}

std::vector<std::string> kvmem_responses_stream_created(
        KvMemResponsesStreamState & state,
        const std::string & response_id,
        const std::string & model) {
    const std::time_t now = std::time(nullptr);
    const json response{
        {"id", "resp_" + response_id},
        {"object", "response"},
        {"status", "in_progress"},
        {"created_at", now},
        {"model", model},
        {"output", json::array()},
    };
    return {
        kvmem_responses_event(&state, "response.created", json{{"response", response}}),
        kvmem_responses_event(&state, "response.in_progress", json{{"response", response}}),
    };
}

std::vector<std::string> kvmem_responses_stream_events(
        KvMemResponsesStreamState & state,
        const common_chat_msg_diff & diff,
        const std::string & response_id) {
    std::vector<std::string> events;
    const std::string reasoning_id = kvmem_responses_reasoning_id(response_id);
    const std::string message_id   = kvmem_responses_message_id(response_id);

    if (!diff.reasoning_content_delta.empty()) {
        if (!state.reasoning_started) {
            state.reasoning_index  = state.next_output_index++;
            state.reasoning_started = true;
            events.push_back(kvmem_responses_event(&state, "response.output_item.added", json{
                {"output_index", state.reasoning_index},
                {"item", json{
                    {"id", reasoning_id},
                    // Both channels open with an empty block; the text arrives in
                    // the matching *.delta events below.
                    {"summary", json::array({json{{"text", ""}, {"type", "summary_text"}}})},
                    {"type", "reasoning"},
                    {"content", json::array({json{{"text", ""}, {"type", "reasoning_text"}}})},
                    {"encrypted_content", ""},
                    {"status", "in_progress"},
                }},
            }));
            events.push_back(kvmem_responses_event(&state, "response.reasoning_summary_part.added", json{
                {"item_id", reasoning_id},
                {"output_index", state.reasoning_index},
                {"summary_index", 0},
                {"part", json{{"text", ""}, {"type", "summary_text"}}},
            }));
            // Recorded here, not by the caller: a client that sees the item
            // opened twice would append a duplicate entry to its output array
            // and then index the wrong one for later deltas.
        }
        events.push_back(kvmem_responses_event(&state, "response.reasoning_text.delta", json{
            {"item_id", reasoning_id},
            {"output_index", state.reasoning_index},
            {"content_index", 0},
            {"delta", diff.reasoning_content_delta},
        }));
        // The same text on the summary channel, as a second content block. The
        // OpenAI SDKs surface thinking from `response.reasoning_summary_text.delta`
        // and store it into the item's `summary`; a client that is fed only the
        // raw `reasoning_text` events ends up with an empty summary and hands an
        // unusable reasoning item back on the next turn. `summary_index` is
        // required on the summary events and is a separate counter from
        // `content_index`, which the raw reasoning events use.
        events.push_back(kvmem_responses_event(&state, "response.reasoning_summary_text.delta", json{
            {"item_id", reasoning_id},
            {"output_index", state.reasoning_index},
            {"summary_index", 0},
            {"delta", diff.reasoning_content_delta},
        }));
    }

    if (!diff.content_delta.empty()) {
        if (!state.text_started) {
            state.text_index   = state.next_output_index++;
            state.text_started = true;
            events.push_back(kvmem_responses_event(&state, "response.output_item.added", json{
                {"output_index", state.text_index},
                {"item", json{
                    {"content", json::array()},
                    {"id", message_id},
                    {"role", "assistant"},
                    {"status", "in_progress"},
                    {"type", "message"},
                }},
            }));
            events.push_back(kvmem_responses_event(&state, "response.content_part.added", json{
                {"item_id", message_id},
                {"output_index", state.text_index},
                {"content_index", 0},
                {"part", json{
                    {"type", "output_text"},
                    {"text", ""},
                    {"annotations", json::array()},
                    {"logprobs", json::array()},
                }},
            }));
        }
        events.push_back(kvmem_responses_event(&state, "response.output_text.delta", json{
            {"item_id", message_id},
            {"output_index", state.text_index},
            {"content_index", 0},
            {"delta", diff.content_delta},
            {"logprobs", json::array()},
        }));
    }

    if (!diff.tool_call_delta.name.empty()) {
        // The function_call item id must persist for the argument deltas that
        // follow: the tool call id is only known once its name has streamed.
        // The name arrives once per tool call, so this opens the item once.
        state.fc_item_id   = "fc_" + (diff.tool_call_delta.id.empty() ? response_id : diff.tool_call_delta.id);
        state.fc_index     = state.next_output_index++;
        state.fc_started   = true;
        events.push_back(kvmem_responses_event(&state, "response.output_item.added", json{
            {"output_index", state.fc_index},
            {"item", json{
                {"id", state.fc_item_id},
                {"arguments", ""},
                {"call_id", diff.tool_call_delta.id},
                {"name", diff.tool_call_delta.name},
                {"type", "function_call"},
                {"status", "in_progress"},
            }},
        }));
    }

    if (!diff.tool_call_delta.arguments.empty()) {
        events.push_back(kvmem_responses_event(&state, "response.function_call_arguments.delta", json{
            {"item_id", state.fc_item_id.empty() ? "fc_" + response_id : state.fc_item_id},
            {"output_index", state.fc_index < 0 ? 0 : state.fc_index},
            {"delta", diff.tool_call_delta.arguments},
        }));
    }

    return events;
}

std::vector<std::string> kvmem_responses_stream_done(
        KvMemResponsesStreamState & state,
        const common_chat_msg & msg,
        const std::string & response_id,
        const std::string & model,
        int n_prompt_tokens,
        int n_gen_tokens,
        int n_cache_hit) {
    std::vector<std::string> events;
    const std::string reasoning_id = kvmem_responses_reasoning_id(response_id);
    const std::string message_id   = kvmem_responses_message_id(response_id);
    json output = json::array();
    int output_index = 0;

    if (!msg.reasoning_content.empty()) {
        events.push_back(kvmem_responses_event(&state, "response.reasoning_summary_text.done", json{
            {"item_id", reasoning_id},
            {"output_index", output_index},
            {"summary_index", 0},
            {"text", msg.reasoning_content},
        }));
        events.push_back(kvmem_responses_event(&state, "response.reasoning_summary_part.done", json{
            {"item_id", reasoning_id},
            {"output_index", output_index},
            {"summary_index", 0},
            {"part", json{{"text", msg.reasoning_content}, {"type", "summary_text"}}},
        }));
        json item = kvmem_responses_reasoning_item(msg.reasoning_content);
        item["id"] = reasoning_id;
        events.push_back(kvmem_responses_event(&state, "response.output_item.done", json{
            {"output_index", output_index},
            {"item", item},
        }));
        output.push_back(std::move(item));
        output_index++;
    }

    if (!msg.content.empty()) {
        events.push_back(kvmem_responses_event(&state, "response.output_text.done", json{
            {"item_id", message_id},
            {"output_index", output_index},
            {"content_index", 0},
            {"text", msg.content},
            {"logprobs", json::array()},
        }));
        const json part = kvmem_responses_content_part(msg.content);
        events.push_back(kvmem_responses_event(&state, "response.content_part.done", json{
            {"item_id", message_id},
            {"output_index", output_index},
            {"content_index", 0},
            {"part", part},
        }));
        json item = kvmem_responses_message_item(msg, message_id);
        events.push_back(kvmem_responses_event(&state, "response.output_item.done", json{
            {"output_index", output_index},
            {"item", item},
        }));
        output.push_back(std::move(item));
        output_index++;
    }

    for (const common_chat_tool_call & tool_call : msg.tool_calls) {
        if (!state.fc_item_id.empty()) {
            events.push_back(kvmem_responses_event(&state, "response.function_call_arguments.done", json{
                {"item_id", state.fc_item_id},
                {"output_index", output_index},
                {"arguments", tool_call.arguments},
            }));
        }
        json item = kvmem_responses_function_call_item(tool_call);
        events.push_back(kvmem_responses_event(&state, "response.output_item.done", json{
            {"output_index", output_index},
            {"item", item},
        }));
        output.push_back(std::move(item));
        output_index++;
    }

    const std::time_t now = std::time(nullptr);
    json response{
        {"id", "resp_" + response_id},
        {"object", "response"},
        {"created_at", now},
        {"completed_at", now},
        {"status", "completed"},
        {"model", model},
        {"output", std::move(output)},
        {"usage", json{
            {"input_tokens", n_prompt_tokens},
            {"output_tokens", n_gen_tokens},
            {"total_tokens", n_gen_tokens + n_prompt_tokens},
            {"input_tokens_details", json{{"cached_tokens", n_cache_hit}}},
        }},
    };
    events.push_back(kvmem_responses_event(&state, "response.completed", json{{"response", response}}));

    return events;
}
