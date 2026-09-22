#pragma once

#include "kvmem-chat-request.h"
#include "kvmem-chat-sampling.h"
#include "httplib.h"

#include <set>
#include <stdexcept>
#include <utility>

namespace kvmem_anthropic {

struct request_error : std::runtime_error {
    int status;
    request_error(const std::string &message, int code = 400) : std::runtime_error(message), status(code) {}
};

inline void require(bool condition, const std::string &message) {
    if (!condition) throw request_error(message);
}

inline void fields(const json &value, std::initializer_list<const char *> allowed, const std::string &where) {
    require(value.is_object(), where + " must be an object");
    for (const auto &item : value.items()) {
        bool found = false;
        for (const char *key : allowed)
            found |= item.key() == key;
        require(found, where + "." + item.key() + " is not supported");
    }
}

inline std::string string_field(const json &value, const char *key, bool empty = false) {
    require(value.contains(key) && value[key].is_string(), std::string(key) + " must be a string");
    const auto text = value[key].get<std::string>();
    require(empty || !text.empty(), std::string(key) + " must not be empty");
    return text;
}

inline bool identifier(const std::string &name, size_t max_length) {
    if (name.empty() || name.size() > max_length) return false;
    return std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}

inline void cache_hint(const json &value) {
    if (!value.contains("cache_control") || value["cache_control"].is_null()) return;
    const auto &hint = value["cache_control"];
    fields(hint, {"type", "ttl"}, "cache_control");
    require(string_field(hint, "type") == "ephemeral", "cache_control.type must be ephemeral");
    if (hint.contains("ttl")) {
        const auto ttl = string_field(hint, "ttl");
        require(ttl == "5m" || ttl == "1h", "cache_control.ttl must be 5m or 1h");
    }
    // A recognized hint is accepted; local prefix reuse does not implement Anthropic TTL/billing.
}

inline std::string text_content(const json &content) {
    if (content.is_string()) return content.get<std::string>();
    require(content.is_array(), "text content must be a string or text-block array");
    std::string text;
    for (const auto &block : content) {
        fields(block, {"type", "text", "cache_control"}, "text block");
        require(string_field(block, "type") == "text", "only text content is supported here");
        cache_hint(block);
        text += string_field(block, "text", true);
    }
    return text;
}

inline void version(const httplib::Request &req) {
    require(req.get_header_value("anthropic-version") == "2023-06-01", "anthropic-version must be 2023-06-01");
    require(req.get_header_value("anthropic-beta").empty(), "Anthropic beta features are not supported");
}

struct request {
    ChatRequest chat;
    std::set<std::string> tool_names;
    bool required_tool = false;
    bool single_tool = false;
};

inline request parse(const json &body, const std::string &model, int generation_limit,
                     const std::map<std::string, std::string> &template_defaults, const json &sampling_defaults,
                     bool count_only = false) {
    fields(body,
           {"model", "messages", "max_tokens", "system", "tools", "tool_choice", "stream", "stop_sequences",
            "temperature", "top_p", "top_k", "metadata", "thinking", "cache_control"},
           "request");
    if (string_field(body, "model") != model) throw request_error("unknown model; use the loaded model alias", 404);
    request result;
    auto &cr = result.chat;
    cr.enable_thinking = false;
    cr.template_kwargs = template_defaults;
    cr.template_kwargs["enable_thinking"] = "false";
    cr.template_kwargs.erase("reasoning_effort");
    cr.reasoning_budget_tokens = -1;
    cache_hint(body);
    if (body.contains("thinking")) {
        fields(body["thinking"], {"type"}, "thinking");
        require(string_field(body["thinking"], "type") == "disabled", "only thinking.type=disabled is supported");
    }
    if (body.contains("metadata")) {
        fields(body["metadata"], {"user_id"}, "metadata");
        if (body["metadata"].contains("user_id")) string_field(body["metadata"], "user_id", true);
    }
    if (!count_only) {
        require(body.contains("max_tokens") && body["max_tokens"].is_number_integer(),
                "max_tokens must be a positive integer");
        const double limit = body["max_tokens"].get<double>();
        require(limit >= 1 && limit <= generation_limit,
                "max_tokens must be in 1.." + std::to_string(generation_limit));
        cr.max_tokens = body["max_tokens"].get<int>();
    }
    if (body.contains("stream")) {
        require(body["stream"].is_boolean(), "stream must be a boolean");
        cr.stream = body["stream"].get<bool>();
    }
    if (body.contains("stop_sequences")) {
        require(body["stop_sequences"].is_array(), "stop_sequences must be an array");
        for (const auto &stop : body["stop_sequences"]) {
            require(stop.is_string() && !stop.get_ref<const std::string &>().empty(),
                    "stop_sequences must contain nonempty strings");
            cr.stop.push_back(stop.get<std::string>());
        }
    }
    cr.sampling = kvmem_chat_sampling_defaults(false);
    std::string error;
    require(kvmem_chat_sampling_override(sampling_defaults, cr.sampling, error), error);
    for (const char *key : {"temperature", "top_p", "top_k"}) {
        if (!body.contains(key)) continue;
        require(body[key].is_number(), std::string(key) + " must be a number");
        if (std::string(key) == "temperature") require(body[key].get<double>() <= 1, "temperature must be in 0..1");
    }
    require(kvmem_chat_sampling_override(body, cr.sampling, error), error);
    if (body.contains("tools")) {
        require(body["tools"].is_array(), "tools must be an array");
        for (const auto &tool : body["tools"]) {
            fields(tool, {"name", "description", "input_schema", "cache_control", "type"}, "tool");
            if (tool.contains("type"))
                require(string_field(tool, "type") == "custom", "only custom client tools are supported");
            common_chat_tool parsed;
            parsed.name = string_field(tool, "name");
            require(identifier(parsed.name, 64), "invalid tool name");
            require(result.tool_names.insert(parsed.name).second, "duplicate tool name");
            if (tool.contains("description")) parsed.description = string_field(tool, "description", true);
            require(tool.contains("input_schema") && tool["input_schema"].is_object(),
                    "input_schema must be an object");
            require(tool["input_schema"].value("type", "") == "object", "input_schema.type must be object");
            parsed.parameters = tool["input_schema"].dump();
            cache_hint(tool);
            cr.tools.push_back(std::move(parsed));
        }
    }
    if (body.contains("tool_choice")) {
        const auto &choice = body["tool_choice"];
        fields(choice, {"type", "name", "disable_parallel_tool_use"}, "tool_choice");
        const auto type = string_field(choice, "type");
        require(type == "auto" || type == "any" || type == "none" || type == "tool", "unsupported tool_choice.type");
        cr.tool_choice = type == "none"   ? COMMON_CHAT_TOOL_CHOICE_NONE
                         : type == "auto" ? COMMON_CHAT_TOOL_CHOICE_AUTO
                                          : COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        result.required_tool = type == "any" || type == "tool";
        require(!result.required_tool || !cr.tools.empty(), "tool_choice requires tools");
        if (type == "tool") {
            const auto name = string_field(choice, "name");
            require(result.tool_names.count(name) != 0, "tool_choice.name is not a declared tool");
            cr.tools.erase(std::remove_if(cr.tools.begin(), cr.tools.end(),
                                          [&](const common_chat_tool &t) { return t.name != name; }),
                           cr.tools.end());
            result.tool_names = {name};
        } else
            require(!choice.contains("name"), "tool_choice.name requires type=tool");
        if (choice.contains("disable_parallel_tool_use")) {
            require(choice["disable_parallel_tool_use"].is_boolean(), "disable_parallel_tool_use must be a boolean");
            result.single_tool = choice["disable_parallel_tool_use"].get<bool>();
            cr.parallel_tool_calls_set = true;
            cr.parallel_tool_calls = !result.single_tool;
        }
    }
    if (body.contains("system")) {
        common_chat_msg system;
        system.role = "system";
        system.content = text_content(body["system"]);
        cr.msgs.push_back(std::move(system));
    }
    require(body.contains("messages") && body["messages"].is_array() && !body["messages"].empty(),
            "messages must be a nonempty array");
    std::set<std::string> pending, seen;
    bool first = true;
    std::string last_role;
    for (const auto &message : body["messages"]) {
        fields(message, {"role", "content"}, "message");
        const auto role = string_field(message, "role");
        require(role == "user" || role == "assistant", "messages roles must be user or assistant");
        require(!first || role == "user", "the first message must have role=user");
        require(pending.empty() || role == "user", "tool_use must be followed by user tool_result blocks");
        first = false;
        last_role = role;
        require(message.contains("content"), "message.content is required");
        const auto &content = message["content"];
        common_chat_msg msg;
        msg.role = role;
        bool has_text = false, has_tools = false;
        if (content.is_string()) {
            require(pending.empty(), "missing tool_result blocks");
            msg.content = content.get<std::string>();
            has_text = true;
        } else {
            require(content.is_array() && !content.empty(), "content must be a string or nonempty block array");
            for (const auto &block : content) {
                require(block.is_object(), "content block must be an object");
                const auto type = string_field(block, "type");
                if (type == "text") {
                    fields(block, {"type", "text", "cache_control"}, "text block");
                    require(!has_tools, "assistant text must precede tool_use blocks");
                    require(role != "user" || pending.empty(), "tool_result blocks must precede user text");
                    msg.content += string_field(block, "text", true);
                    has_text = true;
                } else if (type == "tool_use") {
                    fields(block, {"type", "id", "name", "input", "cache_control"}, "tool_use");
                    require(role == "assistant", "tool_use requires role=assistant");
                    common_chat_tool_call call;
                    call.id = string_field(block, "id");
                    call.name = string_field(block, "name");
                    require(identifier(call.id, 256) && identifier(call.name, 64), "invalid tool_use id or name");
                    require(seen.insert(call.id).second, "duplicate tool_use id");
                    require(block.contains("input") && block["input"].is_object(), "tool_use.input must be an object");
                    call.arguments = block["input"].dump();
                    pending.insert(call.id);
                    msg.tool_calls.push_back(std::move(call));
                    has_tools = true;
                } else if (type == "tool_result") {
                    fields(block, {"type", "tool_use_id", "content", "is_error", "cache_control"}, "tool_result");
                    require(role == "user" && !has_text, "tool_result must precede text in a user message");
                    common_chat_msg tool;
                    tool.role = "tool";
                    tool.tool_call_id = string_field(block, "tool_use_id");
                    require(pending.erase(tool.tool_call_id) != 0, "tool_result has no matching pending tool_use");
                    if (block.contains("content")) tool.content = text_content(block["content"]);
                    if (block.contains("is_error")) {
                        require(block["is_error"].is_boolean(), "tool_result.is_error must be a boolean");
                        if (block["is_error"].get<bool>()) tool.content = "Tool error: " + tool.content;
                    }
                    cr.msgs.push_back(std::move(tool));
                } else
                    throw request_error("unsupported content block: " + type);
                cache_hint(block);
            }
        }
        if (role == "user") {
            require(pending.empty(), "missing tool_result blocks");
            cr.last_user = msg.content;
            if (!has_text && !cr.msgs.empty()) cr.last_user = cr.msgs.back().content;
        }
        if (has_text || has_tools) {
            if (!cr.msgs.empty() && cr.msgs.back().role == role && cr.msgs.back().tool_calls.empty() &&
                msg.tool_calls.empty()) {
                cr.msgs.back().content += "\n\n" + msg.content;
                if (role == "user") cr.last_user = cr.msgs.back().content;
            } else
                cr.msgs.push_back(std::move(msg));
        }
    }
    require(count_only || (last_role == "user" && pending.empty()),
            "generation requires a final user message with all tool results supplied");
    for (auto it = cr.msgs.rbegin(); it != cr.msgs.rend(); ++it) {
        if (it->role == "user") {
            cr.last_user = it->content;
            break;
        }
    }
    return result;
}

inline json error_body(const std::string &message, int status) {
    const char *type = status == 401   ? "authentication_error"
                       : status == 404 ? "not_found_error"
                       : status == 413 ? "request_too_large"
                       : status >= 500 ? "api_error"
                                       : "invalid_request_error";
    return {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
}

inline std::string event(const json &data) {
    return "event: " + data.at("type").get<std::string>() + "\ndata: " + data.dump() + "\n\n";
}

// Keep a possible stop prefix out of the stream until it is resolved.
struct stop_buffer {
    std::vector<std::string> user, all;
    size_t end = std::string::npos;
    std::string matched;

    void observe(const std::string &text) {
        if (end != std::string::npos) return;
        for (const auto &stop : all) {
            if (stop.empty()) continue;
            const auto pos = text.find(stop);
            if (pos < end) {
                end = pos;
                matched = stop;
            }
        }
    }
    std::string visible(const std::string &text, bool partial) {
        observe(text);
        size_t length = std::min(end, text.size());
        if (end == std::string::npos && partial) {
            size_t held = 0;
            for (const auto &stop : all) {
                for (size_t n = 1; n < stop.size() && n <= text.size(); ++n) {
                    if (text.compare(text.size() - n, n, stop, 0, n) == 0) held = std::max(held, n);
                }
            }
            length -= held;
        }
        // A token can end in the middle of a UTF-8 code point.
        if (length) {
            size_t start = length - 1;
            while (start && (static_cast<unsigned char>(text[start]) & 0xc0) == 0x80)
                --start;
            const auto lead = static_cast<unsigned char>(text[start]);
            const size_t bytes = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
            if (start + bytes > length) length = start;
        }
        return text.substr(0, length);
    }
    bool user_stop() const {
        return end != std::string::npos && std::find(user.begin(), user.end(), matched) != user.end();
    }
};

struct response {
    std::set<std::string> tool_names;
    bool required_tool = false, single_tool = false;
    stop_buffer stops;
    std::string id, model;
    int n_prompt = 0;
    common_chat_parser_params parser;
    common_chat_msg previous;
    int block = -1;
    bool text_open = false;
    int active_tool = -1;
    std::vector<std::string> arguments;
    std::vector<std::string> names;

    explicit response(const request &value)
        : tool_names(value.tool_names), required_tool(value.required_tool), single_tool(value.single_tool) {
        stops.user = value.chat.stop;
    }

    void init(const common_chat_params &chat, bool tools, const std::string &request_id, const std::string &model_name,
              int prompt_tokens, std::time_t) {
        id = "msg_" + request_id;
        model = model_name;
        n_prompt = prompt_tokens;
        parser = common_chat_parser_params(chat);
        parser.parse_tool_calls = tools;
        if (!chat.parser.empty()) parser.parser.load(chat.parser);
        stops.all = stops.user;
        stops.all.insert(stops.all.end(), chat.additional_stops.begin(), chat.additional_stops.end());
    }
    static void error(httplib::Response &res, int status, const std::string &message) {
        res.status = status;
        res.set_content(error_body(message, status).dump(), "application/json");
    }
    static std::vector<std::string> error_event(const std::string &message) {
        return {event(error_body(message, 500))};
    }
    static std::vector<std::string> error_end() { return {}; }
    bool trim_stop(std::string &text, const std::vector<std::string> &) {
        stops.observe(text);
        return stop_requested();
    }
    void observe_spec(std::string &text, const std::vector<std::string> &) { stops.observe(text); }
    bool stop_requested() const { return stops.end != std::string::npos; }
    static void trace(bool) {}

    std::string tool_id(size_t index) const { return "toolu_" + id.substr(4) + "_" + std::to_string(index + 1); }
    json usage(int n_gen) const { return {{"input_tokens", n_prompt}, {"output_tokens", n_gen}}; }
    std::string finish_reason(bool hit_limit) const {
        if (stops.user_stop()) return "stop_sequence";
        if (hit_limit && !stop_requested()) return "max_tokens";
        if (!previous.tool_calls.empty()) return "tool_use";
        return "end_turn";
    }
    json stop_sequence() const { return stops.user_stop() ? json(stops.matched) : json(nullptr); }
    json message(const json &content, int n_gen, bool done, bool hit_limit) const {
        return {{"id", id},
                {"type", "message"},
                {"role", "assistant"},
                {"model", model},
                {"content", content},
                {"stop_reason", done ? json(finish_reason(hit_limit)) : json(nullptr)},
                {"stop_sequence", done ? stop_sequence() : json(nullptr)},
                {"usage", usage(n_gen)}};
    }
    void validate(const common_chat_msg &msg, bool final, bool hit_limit = false) const {
        if (single_tool && msg.tool_calls.size() > 1)
            throw std::runtime_error("model emitted multiple tools with parallel tool use disabled");
        if (final && required_tool && msg.tool_calls.empty() && !hit_limit && !stops.user_stop())
            throw std::runtime_error("model did not emit the required tool call");
        for (const auto &call : msg.tool_calls) {
            if (!final && (call.name.empty() || call.arguments.empty())) continue;
            if (!tool_names.count(call.name)) throw std::runtime_error("model emitted an undeclared tool");
            if (final && !json::parse(call.arguments, nullptr, false).is_object())
                throw std::runtime_error("model emitted invalid tool input JSON");
        }
    }
    std::vector<std::string> start() {
        return {event({{"type", "message_start"}, {"message", message(json::array(), 0, false, false)}})};
    }

    void close(std::vector<std::string> &frames) {
        if (text_open || active_tool >= 0) frames.push_back(event({{"type", "content_block_stop"}, {"index", block}}));
        text_open = false;
        active_tool = -1;
    }
    std::vector<std::string> update(const std::string &raw, bool partial, const json &) {
        return update_message(common_chat_parse(stops.visible(raw, partial), partial, parser));
    }
    std::vector<std::string> update_message(common_chat_msg msg) {
        validate(msg, false);
        std::vector<std::string> frames;
        if (msg.content.compare(0, previous.content.size(), previous.content) != 0)
            throw std::runtime_error("model revised already streamed text");
        const auto delta = msg.content.substr(previous.content.size());
        if (!delta.empty()) {
            if (!arguments.empty()) throw std::runtime_error("text after tool output is not supported");
            if (!text_open) {
                ++block;
                text_open = true;
                frames.push_back(event({{"type", "content_block_start"},
                                        {"index", block},
                                        {"content_block", {{"type", "text"}, {"text", ""}}}}));
            }
            frames.push_back(event({{"type", "content_block_delta"},
                                    {"index", block},
                                    {"delta", {{"type", "text_delta"}, {"text", delta}}}}));
        }
        for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
            const auto &call = msg.tool_calls[i];
            if (call.name.empty() || call.arguments.empty()) break;
            if (i + 1 < arguments.size()) {
                if (arguments[i] != call.arguments || names[i] != call.name)
                    throw std::runtime_error("model revised a closed tool block");
                continue;
            }
            if (i == arguments.size()) {
                if (!arguments.empty() && !json::parse(arguments.back(), nullptr, false).is_object())
                    throw std::runtime_error("model emitted invalid tool input JSON");
                close(frames);
                ++block;
                active_tool = static_cast<int>(i);
                arguments.emplace_back();
                names.push_back(call.name);
                frames.push_back(event(
                    {{"type", "content_block_start"},
                     {"index", block},
                     {"content_block",
                      {{"type", "tool_use"}, {"id", tool_id(i)}, {"name", call.name}, {"input", json::object()}}}}));
            }
            if (call.name != names[i] || call.arguments.compare(0, arguments[i].size(), arguments[i]) != 0)
                throw std::runtime_error("model revised already streamed tool input");
            const auto suffix = call.arguments.substr(arguments[i].size());
            if (!suffix.empty())
                frames.push_back(event({{"type", "content_block_delta"},
                                        {"index", block},
                                        {"delta", {{"type", "input_json_delta"}, {"partial_json", suffix}}}}));
            arguments[i] = call.arguments;
        }
        previous = std::move(msg);
        return frames;
    }
    std::vector<std::string> finish(bool hit_limit, int n_gen, int, const json &) {
        validate(previous, true, hit_limit);
        std::vector<std::string> frames;
        close(frames);
        frames.push_back(
            event({{"type", "message_delta"},
                   {"delta", {{"stop_reason", finish_reason(hit_limit)}, {"stop_sequence", stop_sequence()}}},
                   {"usage", {{"output_tokens", n_gen}}}}));
        frames.push_back(event({{"type", "message_stop"}}));
        return frames;
    }
    void result(httplib::Response &res, const std::string &raw, int n_gen, bool hit_limit, int, const json &) {
        previous = common_chat_parse(stops.visible(raw, false), false, parser);
        validate(previous, true, hit_limit);
        json content = json::array();
        if (!previous.content.empty()) content.push_back({{"type", "text"}, {"text", previous.content}});
        for (size_t i = 0; i < previous.tool_calls.size(); ++i) {
            const auto &call = previous.tool_calls[i];
            content.push_back({{"type", "tool_use"},
                               {"id", tool_id(i)},
                               {"name", call.name},
                               {"input", json::parse(call.arguments)}});
        }
        res.set_content(message(content, n_gen, true, hit_limit).dump(), "application/json");
    }
};

} // namespace kvmem_anthropic
