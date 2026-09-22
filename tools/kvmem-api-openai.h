#pragma once

#include "kvmem-chat-request.h"
#include "kvmem-chat-id.h"
#include "kvmem-chat-template.h"
#include "kvmem-chat-sampling.h"
#include "build-info.h"
#include "httplib.h"
#include "llama-kvmem-diag.h"

#include <memory>

static common_json nlohmann_to_common(const json & j) {
    return common_json::parse(j.dump());
}

static json message_to_nlohmann(const common_chat_msg & msg) {
    return json::parse(msg.to_json_oaicompat().dump());
}

static json chat_diff_to_delta(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({std::move(tool_call)});
    }
    return delta;
}

struct StreamChatOut {
    common_chat_parser_params pp;
    common_chat_msg prev;
    std::string acc;
    std::vector<std::string> tc_ids;
    std::string request_id;
    int n_id = 0;
    int n_tc_delta = 0;

    StreamChatOut(const common_chat_params & chat, bool parse_tools, const std::string & id) : request_id(id) {
        pp = common_chat_parser_params(chat);
        pp.parse_tool_calls = parse_tools;
        if (!chat.parser.empty()) {
            pp.parser.load(chat.parser);
        }
    }

    std::vector<json> set_text(const std::string & text, bool partial) {
        acc = text;
        std::vector<json> chunks;
        try {
            common_chat_msg msg = common_chat_parse(acc, partial, pp);
            if (msg.empty() && partial) {
                return chunks;
            }
            if (msg.role.empty()) {
                msg.role = "assistant";
            }
            msg.set_tool_call_ids(tc_ids, [this]() {
                return kvmem_chat_tool_id(request_id, ++n_id);
            });
            const auto diffs = common_chat_msg_diff::compute_diffs(prev, msg);
            prev = std::move(msg);
            for (const auto & d : diffs) {
                json delta = chat_diff_to_delta(d);
                if (delta.empty()) {
                    continue;
                }
                if (d.tool_call_index != std::string::npos) {
                    n_tc_delta++;
                }
                chunks.push_back(std::move(delta));
            }
        } catch (const std::exception & e) {
            if (!partial) {
                LOG_WRN("srv    KVMEM_TRACE chat_stream_parse_fail %s\n", e.what());
            }
        }
        return chunks;
    }

    const char * finish_reason(bool hit_limit) const {
        if (!prev.tool_calls.empty()) {
            return "tool_calls";
        }
        if (hit_limit) {
            return "length";
        }
        return "stop";
    }
};

static json stream_choice_chunk(const std::string & cid, const std::string & model, std::time_t created, const json & delta, const char * finish, const json * timings = nullptr) {
    json choice = {
        {"index", 0},
        {"delta", delta},
        {"finish_reason", finish ? json(finish) : json(nullptr)},
    };
    // 1:1 upstream: {choices, created, id, model, system_fingerprint, object} (server-task.cpp to_json_oaicompat_chat)
    // 中文：逐字段对齐上游 OpenAI 流式 chunk 结构，客户端可无差别解析
    json chunk = {
        {"choices", json::array({std::move(choice)})},
        {"created", created},
        {"id", cid},
        {"model", model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object", "chat.completion.chunk"},
    };
    if (timings != nullptr) {
        chunk["timings"] = *timings;
    }
    return chunk;
}

// DeepSeek Chat Completions usage: prompt_tokens = hit + miss.
// Hit = reused prefix (n_past / LCP). Do not also emit OpenAI
// prompt_tokens_details.cached_tokens — OpenCode isOverflow would
// double-count cache.read against the context window.
static json usage_json(int n_prompt, int n_gen, int n_cache_hit) {
    if (n_prompt < 0) {
        n_prompt = 0;
    }
    if (n_gen < 0) {
        n_gen = 0;
    }
    if (n_cache_hit < 0) {
        n_cache_hit = 0;
    }
    if (n_cache_hit > n_prompt) {
        n_cache_hit = n_prompt;
    }
    return json{
        {"prompt_tokens", n_prompt},
        {"completion_tokens", n_gen},
        {"total_tokens", n_prompt + n_gen},
        {"prompt_cache_hit_tokens", n_cache_hit},
        {"prompt_cache_miss_tokens", n_prompt - n_cache_hit},
    };
}

// OpenAI: last stream chunk has empty choices + usage, no finish_reason.
// 1:1 upstream final usage chunk: {choices, created, id, model, system_fingerprint, object, usage}
// 中文：流式结尾的 usage 块——choices 为空、无 finish_reason，携带 usage 统计
static json stream_usage_chunk(const std::string & cid, const std::string & model, std::time_t created, int n_prompt, int n_gen, int n_cache_hit) {
    return json{
        {"created", created},
        {"id", cid},
        {"object", "chat.completion.chunk"},
        {"system_fingerprint", std::string(llama_build_info())},
        {"model", model},
        {"choices", json::array()},
        {"usage", usage_json(n_prompt, n_gen, n_cache_hit)},
    };
}

static bool strip_stop(std::string & content, const std::vector<std::string> & stops) {
    for (const auto & s : stops) {
        if (s.empty() || content.size() < s.size()) {
            continue;
        }
        if (content.compare(content.size() - s.size(), s.size(), s) == 0) {
            content.resize(content.size() - s.size());
            return true;
        }
    }
    return false;
}

static bool parse_chat_request(const json & body, ChatRequest & out, std::string & err) {
    if (!body.is_object()) {
        err = "request must be a JSON object";
        return false;
    }
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        err = "messages array required";
        return false;
    }
    try {
        out.msgs = common_chat_msgs_parse_oaicompat(nlohmann_to_common(body.at("messages")));
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    if (out.msgs.empty()) {
        err = "messages array required";
        return false;
    }
    for (auto it = out.msgs.rbegin(); it != out.msgs.rend(); ++it) {
        if (it->role == "user") {
            out.last_user = it->content.empty() ? it->render_content() : it->content;
            break;
        }
    }
    if (body.contains("tools") && !body["tools"].is_null()) {
        try {
            out.tools = common_chat_tools_parse_oaicompat(nlohmann_to_common(body.at("tools")));
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    }
    if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
        const auto & tc = body.at("tool_choice");
        try {
            if (tc.is_string()) {
                out.tool_choice = common_chat_tool_choice_parse_oaicompat(tc.get<std::string>());
            } else if (tc.is_object()) {
                // OpenAI named-function form → required (template/grammar in T2).
                out.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            } else {
                err = "tool_choice must be a string or object";
                return false;
            }
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
    }
    if (body.contains("parallel_tool_calls") && body["parallel_tool_calls"].is_boolean()) {
        out.parallel_tool_calls = body["parallel_tool_calls"].get<bool>();
        out.parallel_tool_calls_set = true;
    }
    if (body.contains("grammar") && body["grammar"].is_string()) {
        out.grammar = body["grammar"].get<std::string>();
    }
    if (body.contains("json_schema") && !body["json_schema"].is_null()) {
        out.json_schema = body["json_schema"].dump();
    }
    if (body.contains("response_format") && body["response_format"].is_object()) {
        const auto & rf = body["response_format"];
        const std::string rtype = rf.value("type", "");
        if (rtype == "json_object") {
            if (out.json_schema.empty()) {
                out.json_schema = rf.contains("schema") ? rf["schema"].dump() : "{}";
            }
        } else if (rtype == "json_schema") {
            const auto schema_wrapper = rf.value("json_schema", json::object());
            if (schema_wrapper.contains("schema")) {
                out.json_schema = schema_wrapper["schema"].dump();
            }
        } else if (!rtype.empty() && rtype != "text") {
            err = "response_format type must be text, json_object, or json_schema";
            return false;
        }
    }
    if (!out.tools.empty() && !out.grammar.empty()) {
        err = "Cannot use custom grammar constraints with tools.";
        return false;
    }
    if (body.contains("stop")) {
        const auto & stop = body.at("stop");
        if (stop.is_string()) {
            out.stop.push_back(stop.get<std::string>());
        } else if (stop.is_array()) {
            for (const auto & s : stop) {
                if (s.is_string()) {
                    out.stop.push_back(s.get<std::string>());
                }
            }
        }
    }
    out.stream = body.value("stream", false);
    if (body.contains("kvmem") && body["kvmem"].is_object()) {
        const auto & k = body["kvmem"];
        if (k.contains("query_begin")) {
            out.query_begin = k["query_begin"].get<int>();
        }
        if (k.contains("query_end")) {
            out.query_end = k["query_end"].get<int>();
        }
        if (k.contains("force_substr") && k["force_substr"].is_string()) {
            out.force_substr = k["force_substr"].get<std::string>();
        }
        if (k.contains("pin")) {
            if (k["pin"].is_string()) {
                out.force_substr = k["pin"].get<std::string>();
            } else if (k["pin"].is_array() && !k["pin"].empty() && k["pin"][0].is_string()) {
                out.force_substr = k["pin"][0].get<std::string>();
            }
        }
    }
    if (!kvmem_chat_template_override(body, out.enable_thinking, out.template_kwargs, err)) {
        return false;
    }
    if (!kvmem_chat_reasoning_budget_override(body, out.reasoning_budget_tokens, err)) {
        return false;
    }
    if (body.contains("reasoning_budget_message") && body["reasoning_budget_message"].is_string()) {
        out.reasoning_budget_message = body["reasoning_budget_message"].get<std::string>();
    }
    return true;
}

// Each HTTP request owns its response encoder and incremental parser.
struct kvmem_openai_response {
    std::string request_id, cid, model;
    std::time_t created = 0;
    int n_prompt = 0;
    bool parse_tools = false;
    common_chat_params formatted;
    std::shared_ptr<StreamChatOut> stream;

    void init(const common_chat_params & chat, bool tools, const std::string & id,
              const std::string & model_name, int prompt_tokens, std::time_t timestamp) {
        formatted = chat; parse_tools = tools; request_id = id; cid = "chatcmpl-" + id;
        model = model_name; n_prompt = prompt_tokens; created = timestamp;
    }
    static void error(httplib::Response & res, int status, const std::string & message) {
        res.status = status;
        res.set_content(json{{"error", message}}.dump(), "application/json");
    }
    static std::string frame(const json & data) { return "data: " + data.dump() + "\n\n"; }
    static std::vector<std::string> error_event(const std::string & message) {
        return {frame(json{{"error", message}})};
    }
    static std::vector<std::string> error_end() { return {"data: [DONE]\n\n"}; }
    static bool trim_stop(std::string & content, const std::vector<std::string> & stops) {
        return strip_stop(content, stops);
    }
    static void observe_spec(std::string &, const std::vector<std::string> &) {}
    static bool stop_requested() { return false; }
    std::vector<std::string> start() {
        stream = std::make_shared<StreamChatOut>(formatted, parse_tools, request_id);
        return {frame(stream_choice_chunk(cid, model, created,
                    json{{"role", "assistant"}, {"content", nullptr}}, nullptr))};
    }
    std::vector<std::string> update(const std::string & text, bool partial, const json & timings) {
        const auto deltas = stream->set_text(text, partial);
        std::vector<std::string> frames;
        for (size_t i = 0; i < deltas.size(); ++i) {
            const json * ts = i + 1 == deltas.size() ? &timings : nullptr;
            frames.push_back(frame(stream_choice_chunk(cid, model, created, deltas[i], nullptr, ts)));
        }
        return frames;
    }
    void trace(bool hit_limit) const {
        kvmem_diag("KVMEM_TRACE chat_stream n_tc_delta=%d finish=%s content_chars=%zu reasoning_chars=%zu\n",
                stream->n_tc_delta, stream->finish_reason(hit_limit),
                stream->prev.content.size(), stream->prev.reasoning_content.size());
    }
    std::vector<std::string> finish(bool hit_limit, int n_gen, int n_cache_hit, const json & timings) const {
        auto usage = stream_usage_chunk(cid, model, created, n_prompt, n_gen, n_cache_hit);
        usage["timings"] = timings;
        return {frame(stream_choice_chunk(cid, model, created, json::object(), stream->finish_reason(hit_limit))),
                frame(usage), "data: [DONE]\n\n"};
    }
    void result(httplib::Response & res, const std::string & content,
                int n_gen, bool hit_limit, int n_cache_hit, const json & timings) const {
            common_chat_msg msg = parse_assistant_output(content, formatted, parse_tools);
            std::vector<std::string> tc_ids;
            int n_id = 0;
            msg.set_tool_call_ids(tc_ids, [&n_id, this]() {
                return kvmem_chat_tool_id(request_id, ++n_id);
            });
            std::string finish = "stop";
            if (!msg.tool_calls.empty()) {
                finish = "tool_calls";
            } else if (hit_limit) {
                finish = "length";
            }
            json message;
            try {
                message = message_to_nlohmann(msg);
            } catch (const std::exception &) {
                message = json{{"role", "assistant"}, {"content", content}};
            }
            kvmem_diag("KVMEM_TRACE chat_out n_tool_calls=%zu finish=%s content_chars=%zu reasoning_chars=%zu\n",
                    msg.tool_calls.size(), finish.c_str(),
                    msg.content.size(), msg.reasoning_content.size());
            json out = {
                {"id", cid},
                {"object", "chat.completion"},
                {"created", created},
                {"model", model},
                {"system_fingerprint", std::string(llama_build_info())},
                {"choices", json::array({json{
                    {"index", 0},
                    {"message", message},
                    {"finish_reason", finish},
                }})},
                {"usage", usage_json(n_prompt, n_gen, n_cache_hit)},
            };
            out["timings"] = timings;
            res.set_content(out.dump(), "application/json");
    }
};
