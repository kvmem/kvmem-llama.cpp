#include "guarded_main.h"
#include "models/qwen3_5/frontend/chat_template.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// Context reuse serves a request from the cached render of the conversation it extends, which
// holds only if rendering a conversation with its last message removed yields a byte-prefix of
// the full render, at the same message boundaries. Every maintained template is checked on an
// agentic conversation (instructions, tool calls, tool results, a second user turn, a thinking
// switch in a late message, two calls in one turn). Without preserved reasoning a new user query
// drops the reasoning of the closed turns, so that step is checked only with it preserved; a base
// ending between two results of one call group is no request shape, since the templates render
// consecutive results as one block.
namespace {

namespace fi = ninfer::models::qwen3_5::frontend;

fi::ChatMessage message(ninfer::ChatRole role, std::string text, std::string reasoning = {},
                        std::vector<fi::ToolCall> calls = {}) {
    fi::ChatMessage out;
    out.role = role;
    out.parts.push_back(fi::ChatPart::text_part(std::move(text)));
    out.reasoning_content = std::move(reasoning);
    out.tool_calls        = std::move(calls);
    return out;
}

using Conversation = std::vector<fi::ChatMessage>;
using ninfer::ChatRole;

Conversation first_turn() {
    return {
        message(ChatRole::System, "You are a coding agent. Use the tools to inspect files."),
        message(ChatRole::User, "Find the definition of plan_admission."),
        message(ChatRole::Assistant, "", "I need to locate the symbol in the sources.",
                {{"call_1", "grep", R"({"pattern": "plan_admission"})"}}),
        message(ChatRole::Tool, "src/engine/admission.cpp:42:Plan plan_admission("),
        message(ChatRole::Assistant, "It is defined at admission.cpp:42.",
                "The symbol is a free function in admission.cpp."),
    };
}

Conversation second_turn() {
    Conversation messages = first_turn();
    messages.push_back(message(ChatRole::User, "Now show me its callers."));
    messages.push_back(message(ChatRole::Assistant, "", "I need to search for call sites.",
                               {{"call_2", "grep", R"({"pattern": "plan_admission\\("})"}}));
    messages.push_back(message(ChatRole::Tool, "src/engine/core.cpp:1472: plan_admission(state);"));
    messages.push_back(
        message(ChatRole::Assistant, "It is called at core.cpp:1472.", "There is one call site."));
    return messages;
}

Conversation thinking_switch() {
    Conversation messages = first_turn();
    messages.push_back(message(ChatRole::User, "Answer briefly from now on. /no_think"));
    messages.push_back(message(ChatRole::Assistant, "Understood."));
    return messages;
}

Conversation parallel_calls() {
    return {
        message(ChatRole::System, "You are a coding agent."),
        message(ChatRole::User, "Check both files."),
        message(ChatRole::Assistant, "", "Two independent lookups.",
                {{"call_1", "read", R"({"path": "a.cpp"})"},
                 {"call_2", "read", R"({"path": "b.cpp"})"}}),
        message(ChatRole::Tool, "a.cpp contents"),
        message(ChatRole::Tool, "b.cpp contents"),
        message(ChatRole::Assistant, "Both files checked."),
    };
}

Conversation head(const Conversation& messages, std::size_t count) {
    return Conversation(messages.begin(), messages.begin() + static_cast<std::ptrdiff_t>(count));
}

struct Case {
    const char* label;
    bool preserve_thinking;
    bool tools;
};

int check_pair(const fi::CompiledChatTemplate& compiled, const std::string& label, const Case& mode,
               const Conversation& base, const Conversation& extended) {
    fi::ChatRenderOptions options;
    // The generation prompt is a per-request suffix after the last message; reuse ends at the
    // message frontiers, so the prefix property is checked without it.
    options.add_generation_prompt = false;
    options.enable_thinking       = true;
    options.preserve_thinking     = mode.preserve_thinking;
    if (mode.tools) {
        options.tool_jsons = {
            R"({"type": "function", "function": {"name": "grep", "description": "Search files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}})",
            R"({"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}})"};
    }
    const ninfer::ChatRole appended = extended.back().role;
    if (appended == ChatRole::Tool && base.back().role == ChatRole::Tool) { return 0; }
    if (appended == ChatRole::User && !mode.preserve_thinking &&
        std::any_of(base.begin(), base.end(), [](const fi::ChatMessage& message) {
            return !message.reasoning_content.empty();
        })) {
        return 0;
    }
    const std::string name       = label + " [" + mode.label + "]";
    const fi::RenderedChat full  = compiled.render(extended, options);
    const fi::RenderedChat again = compiled.render(extended, options);
    if (full.text != again.text) {
        std::cerr << name << ": the render is not deterministic\n";
        return 1;
    }
    const fi::RenderedChat shorter = compiled.render(base, options);
    const std::string& a           = shorter.text;
    const std::string& b           = full.text;
    if (a.size() > b.size() || b.compare(0, a.size(), a) != 0) {
        const std::size_t limit = std::min(a.size(), b.size());
        std::size_t at          = 0;
        while (at < limit && a[at] == b[at]) { ++at; }
        const auto window = [](const std::string& text, std::size_t at) {
            const std::size_t from = at > 60 ? at - 60 : 0;
            return text.substr(from, std::min(text.size(), at + 60) - from);
        };
        std::cerr << name << ": the shorter render is not a prefix (diverges at byte " << at
                  << ")\n  shorter: " << window(a, at) << "\n  longer:  " << window(b, at) << '\n';
        return 1;
    }
    if (shorter.message_boundaries.size() != base.size() + 1 ||
        full.message_boundaries.size() != extended.size() + 1) {
        std::cerr << name << ": message boundary count mismatch\n";
        return 1;
    }
    for (std::size_t i = 0; i < shorter.message_boundaries.size(); ++i) {
        if (shorter.message_boundaries[i] != full.message_boundaries[i]) {
            std::cerr << name << ": message boundary " << i << " moved between the renders\n";
            return 1;
        }
    }
    return 0;
}

int check_template(const char* file, const std::vector<Case>& modes) {
    std::ifstream in(std::string(NINFER_SOURCE_DIR "/tools/chat_templates/") + file,
                     std::ios::binary);
    if (!in) {
        std::cerr << "missing template " << file << '\n';
        return 1;
    }
    const std::string source((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
    const auto compiled         = fi::CompiledChatTemplate::resolve(source, file);
    const Conversation turn1    = first_turn();
    const Conversation turn2    = second_turn();
    const Conversation thinking = thinking_switch();
    const Conversation parallel = parallel_calls();
    int failures                = 0;
    for (const Case& mode : modes) {
        const std::string prefix = std::string(file) + " ";
        // A render needs the user query, so the shortest base is the instruction and the query.
        for (std::size_t i = 3; i <= turn1.size(); ++i) {
            failures +=
                check_pair(compiled, prefix + "turn1", mode, head(turn1, i - 1), head(turn1, i));
        }
        for (std::size_t i = turn1.size() + 1; i <= turn2.size(); ++i) {
            failures +=
                check_pair(compiled, prefix + "turn2", mode, head(turn2, i - 1), head(turn2, i));
        }
        for (std::size_t i = turn1.size() + 1; i <= thinking.size(); ++i) {
            failures += check_pair(compiled, prefix + "thinking switch", mode,
                                   head(thinking, i - 1), head(thinking, i));
        }
        for (std::size_t i = 3; i <= parallel.size(); ++i) {
            failures += check_pair(compiled, prefix + "parallel calls", mode, head(parallel, i - 1),
                                   head(parallel, i));
        }
    }
    return failures;
}

int run() {
    const std::vector<Case> modes = {{"preserve_thinking+tools", true, true},
                                     {"preserve_thinking", true, false},
                                     {"no-preserve+tools", false, true},
                                     {"no-preserve", false, false}};
    int failures                  = 0;
    for (const char* file : {"qwen3_6.jinja", "qwen3_8.jinja", "froggeric_v22_5.jinja"}) {
        failures += check_template(file, modes);
    }
    if (failures == 0) { std::cout << "ok (prefix stability)\n"; }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
