#pragma once

#include "chat.h"
#include "common.h"
#include "log.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

struct ChatRequest {
    std::vector<common_chat_msg> msgs;
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool parallel_tool_calls = false;
    bool parallel_tool_calls_set = false;
    std::string json_schema;
    std::string grammar;
    std::vector<std::string> stop;
    std::string last_user;
    int max_tokens = 128;
    common_params_sampling sampling;
    bool stream = false;
    int query_begin = -1;
    int query_end = -1;
    std::string force_substr;
    bool enable_thinking = false;
    std::map<std::string, std::string> template_kwargs;
    int reasoning_budget_tokens = -1;
    std::string reasoning_budget_message;
};

static common_chat_msg parse_assistant_output(const std::string &content, const common_chat_params &chat,
                                              bool parse_tools) {
    try {
        common_chat_parser_params pp(chat);
        pp.parse_tool_calls = parse_tools;
        if (!chat.parser.empty()) {
            pp.parser.load(chat.parser);
        }
        common_chat_msg msg = common_chat_parse(content, false, pp);
        if (msg.role.empty()) {
            msg.role = "assistant";
        }
        return msg;
    } catch (const std::exception &e) {
        LOG_WRN("srv    KVMEM_TRACE chat_out_parse_fail %s\n", e.what());
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = content;
        return msg;
    }
}

inline common_chat_templates_inputs kvmem_chat_inputs(const ChatRequest &cr, const common_chat_templates *tmpls) {
    common_chat_templates_inputs inputs;
    inputs.messages = cr.msgs;
    inputs.tools = cr.tools;
    inputs.tool_choice = cr.tool_choice;
    inputs.grammar = cr.grammar;
    inputs.json_schema = cr.json_schema;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = cr.enable_thinking;
    inputs.chat_template_kwargs = cr.template_kwargs;
    // llama-server default: extract <think> into delta.reasoning_content.
    // NONE leaves thinking in content, so OpenCode TUI never classifies it.
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    if (cr.parallel_tool_calls_set) {
        inputs.parallel_tool_calls = cr.parallel_tool_calls;
    } else {
        const auto caps = common_chat_templates_get_caps(tmpls);
        const auto it = caps.find("supports_parallel_tool_calls");
        inputs.parallel_tool_calls = it != caps.end() && it->second;
    }
    return inputs;
}
