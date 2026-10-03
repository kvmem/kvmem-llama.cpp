#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Qwen's tool syntax carries each argument as untyped text. This terminal contract records only
// the supported top-level JSON Schema types needed to normalize that text. A type mismatch remains
// a structured call for consumer validation; recursive validation is outside this non-strict
// contract.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        Legacy,
        DeclaredTypes,
    };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::Legacy;
        TypeSet types;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
    bool enforce_declared_names = false;
    // Function whose call opener the rendered prompt already carries, empty when the model chooses
    // freely. It lives here because the prompt that carries the opener is the only thing that knows
    // it: a second, independently supplied copy could name a different tool and the continuation
    // would be attributed to that one.
    std::string forced_tool_name;
};

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled,
                                std::string_view forced_tool_name = {});

// Native Qwen tool-call envelope for a structured final-response request. Parameters remain
// non-strict values, with each declared parameter emitted at most once in declaration order.
[[nodiscard]] std::string structured_tool_call_format(const ToolCallOutputContract& contract);

[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract);

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. At terminal time, valid calls are retained structurally. A region that opens a function
// but fails the strict reader goes through recovery: readable calls are kept and an unreadable one
// becomes a call to the reserved error tool. Only markup that never opens a function is restored
// verbatim. When the prompt already carries the opener of a forced call, the decoder is seeded
// with that opener, because the model never emits it.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    // A '<' and what follows while it can still grow into a tool-call region opener.
    std::string pending_marker_;
    // Bytes of tool_region_ that came from the prompt's opener rather than from the model. The
    // fallback path returns the region as ordinary content, and these must not appear there.
    std::size_t seeded_prefix_bytes_  = 0;
    std::size_t max_tool_name_length_ = 0;
    bool saw_tool_marker_             = false;
    bool forced_                      = false;
    bool finished_                    = false;
};

} // namespace ninfer::models::qwen3_5::frontend
