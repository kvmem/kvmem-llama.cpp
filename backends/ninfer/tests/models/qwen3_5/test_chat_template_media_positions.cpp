#include "guarded_main.h"
#include "models/qwen3_5/frontend/chat_template.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// Every maintained template must place a media placeholder wherever the image sits: in the first
// message, after a turn, in a tool result, after a tool result, and in two turns at once. A render
// that loses the placeholder of a later message rejects the request as a media count mismatch.
namespace {

namespace fi = ninfer::models::qwen3_5::frontend;
using ninfer::ChatRole;

fi::ChatMessage message(ChatRole role, std::string text, bool image = false) {
    fi::ChatMessage out;
    out.role = role;
    if (image) {
        fi::MediaData media;
        media.bytes      = {0x89, 0x50, 0x4e, 0x47};
        media.media_type = "image/png";
        out.parts.push_back(fi::ChatPart::image(std::move(media)));
    }
    out.parts.push_back(fi::ChatPart::text_part(std::move(text)));
    return out;
}

int run() {
    fi::ChatMessage call = message(ChatRole::Assistant, "");
    call.tool_calls.push_back({"call_1", "screenshot", "{}"});

    struct Case {
        const char* label;
        std::vector<fi::ChatMessage> messages;
        std::size_t placeholders;
    };

    const std::vector<Case> cases = {
        {"first message", {message(ChatRole::User, "what is this?", true)}, 1},
        {"after a turn",
         {message(ChatRole::User, "hello"), message(ChatRole::Assistant, "hi"),
          message(ChatRole::User, "what is this?", true)},
         1},
        {"in a tool result",
         {message(ChatRole::User, "look"), call, message(ChatRole::Tool, "captured", true)},
         1},
        {"after a tool result",
         {message(ChatRole::User, "look"), call, message(ChatRole::Tool, "captured"),
          message(ChatRole::Assistant, "done"), message(ChatRole::User, "and this?", true)},
         1},
        {"in two turns",
         {message(ChatRole::User, "this?", true), message(ChatRole::Assistant, "a cat"),
          message(ChatRole::User, "and this?", true)},
         2},
    };
    int failures = 0;
    for (const char* file : {"qwen3_6.jinja", "qwen3_8.jinja", "froggeric_v22_5.jinja"}) {
        std::ifstream in(std::string(NINFER_SOURCE_DIR "/tools/chat_templates/") + file,
                         std::ios::binary);
        const std::string source((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
        if (source.empty()) {
            std::cerr << "missing template " << file << '\n';
            ++failures;
            continue;
        }
        const auto compiled = fi::CompiledChatTemplate::resolve(source, file);
        for (const Case& test : cases) {
            const fi::RenderedChat rendered = compiled.render(test.messages);
            if (rendered.media_placeholders.size() != test.placeholders) {
                std::cerr << file << " " << test.label << ": " << rendered.media_placeholders.size()
                          << " placeholders, expected " << test.placeholders << '\n';
                ++failures;
            }
        }
    }
    if (failures == 0) { std::cout << "ok (media positions)\n"; }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
