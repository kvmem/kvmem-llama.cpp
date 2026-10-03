#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

using Json                = nlohmann::ordered_json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

constexpr std::string_view kToolOpen      = "<tool_call>";
constexpr std::string_view kToolClose     = "</tool_call>";
constexpr std::string_view kFunctionOpen  = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParamOpen     = "<parameter=";
constexpr std::string_view kParamClose    = "</parameter>";
constexpr std::string_view kFunctionCallsOpen  = "<function_calls>";
constexpr std::string_view kFunctionCallsClose = "</function_calls>";

// Reserved name for a call the recovery pass could not read. No client declares it, so the client
// answers with an unknown-tool error and nothing runs; the arguments tell the model what went wrong.
constexpr std::string_view kMalformedCallTool = "malformed_tool_call";

// Markers tried as the start of the structured turn, the first one included.
constexpr std::size_t kMaxMarkerAttempts = 16;

struct RawParameter {
    std::string_view name;
    std::string_view value;
};

struct RawToolCall {
    std::string_view name;
    std::vector<RawParameter> parameters;
};

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

enum class ParameterNormalization : std::uint8_t {
    Emitted,
    Omitted,
    SchemaMismatch,
};

struct NormalizedParameter {
    ParameterNormalization disposition = ParameterNormalization::Emitted;
    std::string json_value;
};

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }

constexpr bool is_ascii_alphanumeric(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || is_ascii_digit(byte);
}

std::string_view trim_format_whitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && is_format_whitespace(text[begin])) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && is_format_whitespace(text[end - 1])) { --end; }
    return text.substr(begin, end - begin);
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

void skip_format_whitespace(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && is_format_whitespace(text[pos])) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

// A function or parameter tag. Beside the Qwen forms (<function=NAME>, <parameter=NAME>), agent
// harnesses write <function name="NAME">, <invoke name="NAME"> and <param name="NAME">. Every
// opening tag is closed by its own kind.
struct TagForm {
    std::string_view open; // followed by '=' or whitespace
    std::string_view close;
};

constexpr TagForm kFunctionForms[]  = {{"<function", "</function>"}, {"<invoke", "</invoke>"}};
constexpr TagForm kParameterForms[] = {{"<parameter", "</parameter>"}, {"<param", "</param>"}};

bool opens_tag(std::string_view text, std::size_t pos, const TagForm& form) {
    if (!starts_with_at(text, pos, form.open)) { return false; }
    const std::size_t next = pos + form.open.size();
    return next < text.size() && (text[next] == '=' || is_format_whitespace(text[next]));
}

template <std::size_t N>
const TagForm* tag_form_at(std::string_view text, std::size_t pos, const TagForm (&forms)[N]) {
    for (const TagForm& form : forms) {
        if (opens_tag(text, pos, form)) { return &form; }
    }
    return nullptr;
}

std::string_view unquote(std::string_view text) {
    text = trim_format_whitespace(text);
    if (text.size() >= 2 && (text.front() == '"' || text.front() == '\'') &&
        text.back() == text.front()) {
        return trim_format_whitespace(text.substr(1, text.size() - 2));
    }
    return text;
}

// The name in a tag header: `=NAME`, or a `name` attribute, quoted or not. Other attributes are
// skipped whole, so `filename="x"` never supplies the name. Empty when there is none.
std::string_view tag_name(std::string_view header) {
    header = trim_format_whitespace(header);
    if (header.starts_with('=')) { return unquote(header.substr(1)); }
    std::size_t pos = 0;
    while (pos < header.size()) {
        skip_format_whitespace(header, pos);
        const std::size_t attribute_begin = pos;
        while (pos < header.size() && header[pos] != '=' && !is_format_whitespace(header[pos])) {
            ++pos;
        }
        const std::string_view attribute = header.substr(attribute_begin, pos - attribute_begin);
        skip_format_whitespace(header, pos);
        if (pos == header.size() || header[pos] != '=') { continue; }
        ++pos;
        skip_format_whitespace(header, pos);
        std::string_view value;
        if (pos < header.size() && (header[pos] == '"' || header[pos] == '\'')) {
            const std::size_t close = header.find(header[pos], pos + 1);
            const std::size_t end   = close == std::string_view::npos ? header.size() : close;
            value                   = header.substr(pos + 1, end - pos - 1);
            pos                     = end == header.size() ? end : end + 1;
        } else {
            const std::size_t value_begin = pos;
            while (pos < header.size() && !is_format_whitespace(header[pos]) &&
                   header[pos] != '/') {
                ++pos;
            }
            value = header.substr(value_begin, pos - value_begin);
        }
        if (attribute == "name") { return trim_format_whitespace(value); }
    }
    return {};
}

// The header between an opening tag's name and its '>'.
std::string_view tag_header(std::string_view text, std::size_t pos, const TagForm& form,
                            std::size_t& tag_end) {
    const std::size_t header_begin = pos + form.open.size();
    tag_end                        = text.find('>', header_begin);
    if (tag_end == std::string_view::npos) { return {}; }
    return text.substr(header_begin, tag_end - header_begin);
}

// Openers of a tool-call region: the wrapper, the container, and a bare function of any form.
constexpr std::string_view kToolMarkers[] = {
    kToolOpen, kFunctionCallsOpen, "<function=", "<function ", "<invoke=", "<invoke "};

bool could_open_marker(std::string_view prefix) {
    return std::any_of(std::begin(kToolMarkers), std::end(kToolMarkers),
                       [&](std::string_view marker) { return marker.starts_with(prefix); });
}

bool opens_marker(std::string_view text) {
    return std::any_of(std::begin(kToolMarkers), std::end(kToolMarkers),
                       [&](std::string_view marker) { return text.starts_with(marker); });
}

// A bare function inside an unclosed <tool_call> or <function_calls> belongs to that opener's
// region, so it never starts a region of its own.
bool inside_open_wrapper(std::string_view text, std::size_t pos) {
    const std::string_view before = text.substr(0, pos);
    const auto unclosed           = [&](std::string_view open, std::string_view close) {
        const std::size_t opened = before.rfind(open);
        if (opened == std::string_view::npos) { return false; }
        const std::size_t closed = before.rfind(close);
        return closed == std::string_view::npos || closed < opened;
    };
    return unclosed(kToolOpen, kToolClose) || unclosed(kFunctionCallsOpen, kFunctionCallsClose);
}

// The first region opener at or after `from`.
std::size_t find_tool_marker(std::string_view text, std::size_t from) {
    for (std::size_t pos = text.find('<', from); pos != std::string_view::npos;
         pos             = text.find('<', pos + 1)) {
        if (!opens_marker(text.substr(pos))) { continue; }
        if (starts_with_at(text, pos, kToolOpen) || starts_with_at(text, pos, kFunctionCallsOpen) ||
            !inside_open_wrapper(text, pos)) {
            return pos;
        }
    }
    return std::string_view::npos;
}

bool opens_call(std::string_view text, std::size_t pos) {
    return starts_with_at(text, pos, kToolOpen) ||
           tag_form_at(text, pos, kFunctionForms) != nullptr;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return is_ascii_alphanumeric(byte) || byte == '_' || byte == '-';
    });
}

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

constexpr bool admits_type(TypeSet types, SchemaType type) {
    return (types.bits & type_bit(type)) != 0;
}

bool schema_type(std::string_view name, SchemaType& type) {
    if (name == "null") {
        type = SchemaType::Null;
    } else if (name == "boolean") {
        type = SchemaType::Boolean;
    } else if (name == "integer") {
        type = SchemaType::Integer;
    } else if (name == "number") {
        type = SchemaType::Number;
    } else if (name == "string") {
        type = SchemaType::String;
    } else if (name == "object") {
        type = SchemaType::Object;
    } else if (name == "array") {
        type = SchemaType::Array;
    } else {
        return false;
    }
    return true;
}

bool compile_direct_types(const Json& type_definition, TypeSet& types) {
    types = {};
    if (type_definition.is_string()) {
        SchemaType type;
        if (!schema_type(type_definition.get_ref<const std::string&>(), type)) { return false; }
        types.bits = type_bit(type);
        return true;
    }
    if (!type_definition.is_array() || type_definition.empty()) { return false; }
    for (const Json& member : type_definition) {
        if (!member.is_string()) { return false; }
        SchemaType type;
        if (!schema_type(member.get_ref<const std::string&>(), type)) { return false; }
        types.bits |= type_bit(type);
    }
    return types.bits != 0;
}

bool compile_schema_types(const Json& schema, TypeSet& types) {
    if (!schema.is_object()) { return false; }
    const auto direct = schema.find("type");
    if (direct != schema.end()) { return compile_direct_types(*direct, types); }

    const auto any_of     = schema.find("anyOf");
    const auto one_of     = schema.find("oneOf");
    const bool has_any_of = any_of != schema.end();
    const bool has_one_of = one_of != schema.end();
    if (has_any_of == has_one_of) { return false; }

    const Json& alternatives = has_any_of ? *any_of : *one_of;
    if (!alternatives.is_array() || alternatives.empty()) { return false; }

    TypeSet combined;
    for (const Json& alternative : alternatives) {
        TypeSet branch;
        if (!compile_schema_types(alternative, branch)) { return false; }
        combined.bits |= branch.bits;
    }
    if (combined.bits == 0) { return false; }
    types = combined;
    return true;
}

Contract::Tool compile_tool_contract(const Json& definition) {
    Contract::Tool contract;
    if (!definition.is_object()) { return contract; }
    const auto function = definition.find("function");
    if (function == definition.end() || !function->is_object()) { return contract; }
    const auto name = function->find("name");
    if (name == function->end() || !name->is_string()) { return contract; }
    contract.name = name->get<std::string>();

    const auto schema = function->find("parameters");
    if (schema == function->end() || !schema->is_object()) { return contract; }
    const auto properties = schema->find("properties");
    if (properties == schema->end() || !properties->is_object()) { return contract; }

    contract.parameters.reserve(properties->size());
    for (const auto& [parameter_name, property] : properties->items()) {
        Contract::Parameter parameter;
        parameter.name = parameter_name;
        if (compile_schema_types(property, parameter.types)) {
            parameter.policy = NormalizationPolicy::DeclaredTypes;
        }
        contract.parameters.push_back(std::move(parameter));
    }
    return contract;
}

bool same_contract(const Contract::Tool& lhs, const Contract::Tool& rhs) {
    if (lhs.parameters.size() != rhs.parameters.size()) { return false; }
    for (std::size_t i = 0; i < lhs.parameters.size(); ++i) {
        const Contract::Parameter& left  = lhs.parameters[i];
        const Contract::Parameter& right = rhs.parameters[i];
        if (left.name != right.name || left.policy != right.policy ||
            left.types.bits != right.types.bits) {
            return false;
        }
    }
    return true;
}

void append_tool_contract(Contract& contracts, const Json& definition) {
    Contract::Tool compiled = compile_tool_contract(definition);
    if (compiled.name.empty()) { return; }
    const auto existing =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& tool) { return tool.name == compiled.name; });
    if (existing == contracts.tools.end()) {
        contracts.tools.push_back(std::move(compiled));
        return;
    }
    if (existing->unambiguous && !same_contract(*existing, compiled)) {
        existing->parameters.clear();
        existing->unambiguous = false;
    }
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

const Contract::Parameter* find_parameter_contract(const Contract::Tool& tool,
                                                   std::string_view parameter_name) {
    const auto parameter =
        std::find_if(tool.parameters.begin(), tool.parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool.parameters.end() ? nullptr : &*parameter;
}

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ascii_case_equal(std::string_view text, std::string_view lowercase) {
    if (text.size() != lowercase.size()) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char byte = text[i];
        if (byte >= 'A' && byte <= 'Z') { byte = static_cast<char>(byte + ('a' - 'A')); }
        if (byte != lowercase[i]) { return false; }
    }
    return true;
}

bool json_number_is_integer(std::string_view number) {
    std::size_t pos = number.starts_with('-') ? 1 : 0;
    if (pos >= number.size()) { return false; }

    const std::size_t integer_begin = pos;
    while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
    const std::size_t integer_end = pos;

    std::size_t fraction_begin = pos;
    std::size_t fraction_end   = pos;
    if (pos < number.size() && number[pos] == '.') {
        fraction_begin = ++pos;
        while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
        fraction_end = pos;
    }

    bool exponent_negative     = false;
    std::size_t exponent_value = 0;
    if (pos < number.size() && (number[pos] == 'e' || number[pos] == 'E')) {
        ++pos;
        if (pos < number.size() && (number[pos] == '+' || number[pos] == '-')) {
            exponent_negative = number[pos] == '-';
            ++pos;
        }
        const std::size_t cap = number.size();
        while (pos < number.size() && is_ascii_digit(number[pos])) {
            const std::size_t digit = static_cast<std::size_t>(number[pos] - '0');
            if (exponent_value != cap) {
                if (exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10)) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }
            ++pos;
        }
    }
    if (integer_begin == integer_end || pos != number.size()) { return false; }

    bool coefficient_is_zero   = true;
    std::size_t trailing_zeros = 0;
    const auto observe_digit   = [&](char digit) {
        if (digit == '0') {
            ++trailing_zeros;
        } else {
            coefficient_is_zero = false;
            trailing_zeros      = 0;
        }
    };
    for (std::size_t i = integer_begin; i < integer_end; ++i) { observe_digit(number[i]); }
    for (std::size_t i = fraction_begin; i < fraction_end; ++i) { observe_digit(number[i]); }
    if (coefficient_is_zero) { return true; }

    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (!exponent_negative) {
        if (exponent_value >= fraction_digits) { return true; }
        return fraction_digits - exponent_value <= trailing_zeros;
    }
    if (exponent_value > trailing_zeros) { return false; }
    return fraction_digits <= trailing_zeros - exponent_value;
}

bool classify_json_value(std::string_view value, JsonValueKind& kind) {
    if (value.empty() || !Json::accept(value.begin(), value.end())) { return false; }
    switch (value.front()) {
    case 'n':
        kind = JsonValueKind::Null;
        return true;
    case 't':
    case 'f':
        kind = JsonValueKind::Boolean;
        return true;
    case '"':
        kind = JsonValueKind::String;
        return true;
    case '{':
        kind = JsonValueKind::Object;
        return true;
    case '[':
        kind = JsonValueKind::Array;
        return true;
    default:
        if (value.front() == '-' || is_ascii_digit(value.front())) {
            kind = json_number_is_integer(value) ? JsonValueKind::Integer : JsonValueKind::Number;
            return true;
        }
        return false;
    }
}

bool admits_value(TypeSet types, JsonValueKind kind) {
    switch (kind) {
    case JsonValueKind::Null:
        return admits_type(types, SchemaType::Null);
    case JsonValueKind::Boolean:
        return admits_type(types, SchemaType::Boolean);
    case JsonValueKind::Integer:
        return admits_type(types, SchemaType::Integer) || admits_type(types, SchemaType::Number);
    case JsonValueKind::Number:
        return admits_type(types, SchemaType::Number);
    case JsonValueKind::String:
        return admits_type(types, SchemaType::String);
    case JsonValueKind::Object:
        return admits_type(types, SchemaType::Object);
    case JsonValueKind::Array:
        return admits_type(types, SchemaType::Array);
    }
    return false;
}

std::string encode_json_string(std::string_view value) { return Json(std::string(value)).dump(); }

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types) {
    const std::string_view framed = remove_parameter_framing_newlines(encoded_value);
    if (admits_type(types, SchemaType::String)) {
        return {.json_value = encode_json_string(framed)};
    }

    const std::string_view value = trim_format_whitespace(framed);
    if (value.empty()) { return {.disposition = ParameterNormalization::Omitted}; }

    JsonValueKind kind;
    if (classify_json_value(value, kind)) {
        return {.disposition = admits_value(types, kind) ? ParameterNormalization::Emitted
                                                         : ParameterNormalization::SchemaMismatch,
                .json_value  = std::string(value)};
    }

    if (admits_type(types, SchemaType::Boolean)) {
        if (ascii_case_equal(value, "true")) { return {.json_value = "true"}; }
        if (ascii_case_equal(value, "false")) { return {.json_value = "false"}; }
    }
    return {.disposition = ParameterNormalization::SchemaMismatch,
            .json_value  = encode_json_string(framed)};
}

NormalizedParameter normalize_parameter(std::string_view encoded_value,
                                        const Contract::Parameter* parameter) {
    if (parameter != nullptr && parameter->policy == NormalizationPolicy::DeclaredTypes) {
        return normalize_declared_parameter(encoded_value, parameter->types);
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    if (Json::accept(value.begin(), value.end())) { return {.json_value = std::string(value)}; }
    return {.json_value = encode_json_string(value)};
}

// The strict reader accepts only complete, declared calls. The lenient one, used by the recovery
// pass on a call the strict reader rejected, also takes an undeclared name (the client answers
// with its own unknown-tool error), a parameter whose close tag is missing right before its
// function closes, and a call missing only its outer close tag.
class QwenToolRegionParser {
public:
    QwenToolRegionParser(std::string_view text, std::size_t max_name_length,
                         const Contract& contract, bool lenient = false)
        : text_(text), max_name_length_(max_name_length), contract_(contract), lenient_(lenient) {}

    // One call starting at pos: a <tool_call> or a bare function. On failure pos is left inside
    // the call.
    FallbackReason parse_one(std::size_t& pos, RawToolCall& call) const {
        if (starts_with_at(text_, pos, kToolOpen)) { return parse_tool_call(pos, call); }
        return parse_function(pos, call);
    }

    [[nodiscard]] std::uint32_t duplicate_parameters_repaired() const noexcept {
        return duplicate_parameters_repaired_;
    }

    FallbackReason parse(std::vector<RawToolCall>& calls) const {
        std::size_t pos = 0;
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (pos == text_.size()) {
                return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
            }
            if (consume(pos, kFunctionCallsOpen)) {
                const std::size_t before = calls.size();
                for (;;) {
                    skip_format_whitespace(text_, pos);
                    if (consume(pos, kFunctionCallsClose)) { break; }
                    RawToolCall call;
                    const FallbackReason failure = parse_function(pos, call);
                    if (failure != FallbackReason::None) { return failure; }
                    calls.push_back(std::move(call));
                }
                if (calls.size() == before) { return FallbackReason::MalformedStructure; }
                continue;
            }
            if (!opens_call(text_, pos)) {
                return calls.empty() ? FallbackReason::MalformedStructure
                                     : FallbackReason::TrailingContent;
            }

            RawToolCall call;
            const FallbackReason failure = parse_one(pos, call);
            if (failure != FallbackReason::None) { return failure; }
            calls.push_back(std::move(call));
        }
    }

private:
    bool consume(std::size_t& pos, std::string_view token) const {
        if (!starts_with_at(text_, pos, token)) { return false; }
        pos += token.size();
        return true;
    }

    FallbackReason parse_tool_call(std::size_t& pos, RawToolCall& call) const {
        if (!consume(pos, kToolOpen)) { return FallbackReason::MalformedStructure; }
        skip_format_whitespace(text_, pos);
        const FallbackReason failure = parse_function(pos, call);
        if (failure != FallbackReason::None) { return failure; }
        skip_format_whitespace(text_, pos);
        if (consume(pos, kToolClose)) { return FallbackReason::None; }
        if (lenient_ && (pos == text_.size() || starts_with_at(text_, pos, kToolOpen))) {
            return FallbackReason::None;
        }
        return FallbackReason::MalformedStructure;
    }

    FallbackReason parse_function(std::size_t& pos, RawToolCall& call) const {
        const TagForm* form = tag_form_at(text_, pos, kFunctionForms);
        if (form == nullptr) { return FallbackReason::MalformedStructure; }
        std::size_t tag_end = 0;
        call.name           = tag_name(tag_header(text_, pos, *form, tag_end));
        if (tag_end == std::string_view::npos ||
            !valid_function_name(call.name, max_name_length_)) {
            return FallbackReason::InvalidToolName;
        }
        if (!lenient_ && contract_.enforce_declared_names &&
            find_tool_contract(contract_, call.name) == nullptr) {
            return FallbackReason::UndeclaredTool;
        }
        pos = tag_end + 1;

        for (;;) {
            skip_format_whitespace(text_, pos);
            if (consume(pos, form->close)) { return FallbackReason::None; }
            const FallbackReason failure = parse_parameter(pos, form->close, call);
            if (failure != FallbackReason::None) { return failure; }
        }
    }

    FallbackReason parse_parameter(std::size_t& pos, std::string_view function_close,
                                   RawToolCall& call) const {
        const TagForm* form = tag_form_at(text_, pos, kParameterForms);
        if (form == nullptr) { return FallbackReason::MalformedStructure; }
        std::size_t tag_end         = 0;
        const std::string_view name = tag_name(tag_header(text_, pos, *form, tag_end));
        if (tag_end == std::string_view::npos || name.empty()) {
            return FallbackReason::MalformedStructure;
        }

        const std::size_t value_begin = tag_end + 1;
        std::size_t value_end         = 0;
        std::size_t next              = 0;
        if (lenient_ && find_unclosed_parameter_end(value_begin, function_close, value_end)) {
            next = value_end;
        } else if (find_parameter_close(value_begin, *form, value_end)) {
            next = value_end + form->close.size();
        } else {
            return FallbackReason::MalformedStructure;
        }
        const std::string_view value = text_.substr(value_begin, value_end - value_begin);
        // A repeated parameter keeps its last value, as JSON object syntax would, rather than
        // discarding an otherwise well-formed call.
        const auto existing = std::find_if(call.parameters.begin(), call.parameters.end(),
                                           [&](const RawParameter& p) { return p.name == name; });
        if (existing != call.parameters.end()) {
            existing->value = value;
            ++duplicate_parameters_repaired_;
        } else {
            call.parameters.push_back(RawParameter{.name = name, .value = value});
        }
        pos = next;
        return FallbackReason::None;
    }

    // A value that runs straight into its function's close tag, with no parameter tag of any
    // kind on the way. Any parameter markup in between makes the boundary a guess, and a guessed
    // boundary could cut a command short, so that stays unreadable.
    bool find_unclosed_parameter_end(std::size_t value_begin, std::string_view function_close,
                                     std::size_t& value_end) const {
        const std::size_t close = text_.find(function_close, value_begin);
        if (close == std::string_view::npos) { return false; }
        const std::string_view value = text_.substr(value_begin, close - value_begin);
        if (value.find("</param") != std::string_view::npos ||
            value.find("<param") != std::string_view::npos) {
            return false;
        }
        value_end = close;
        return true;
    }

    // A nested opening tag of the same kind, complete before `limit`.
    bool find_parameter_open_before(std::size_t scan, std::size_t limit, const TagForm& form,
                                    std::size_t& open_end) const {
        std::size_t candidate = text_.find(form.open, scan);
        while (candidate != std::string_view::npos && candidate < limit) {
            std::size_t tag_end = 0;
            if (opens_tag(text_, candidate, form) &&
                !tag_name(tag_header(text_, candidate, form, tag_end)).empty() && tag_end < limit) {
                open_end = tag_end + 1;
                return true;
            }
            candidate = text_.find(form.open, candidate + 1);
        }
        return false;
    }

    // The format has no escape, so a value that quotes its own closing tag is ambiguous. A closer
    // that runs on into ordinary text is value text; one followed, after whitespace, by markup or
    // by the end of the output ends the value. A malformed tag after a real closer is still
    // markup, so it is reported rather than folded into the value.
    bool ends_value(std::size_t pos) const {
        skip_format_whitespace(text_, pos);
        return pos == text_.size() || text_[pos] == '<';
    }

    bool find_parameter_close(std::size_t value_begin, const TagForm& form,
                              std::size_t& value_end) const {
        std::size_t depth = 1;
        std::size_t scan  = value_begin;
        for (;;) {
            const std::size_t close = text_.find(form.close, scan);
            if (close == std::string_view::npos) { return false; }

            std::size_t nested_open_end = 0;
            if (find_parameter_open_before(scan, close, form, nested_open_end)) {
                ++depth;
                scan = nested_open_end;
                continue;
            }

            if (depth != 1 || ends_value(close + form.close.size())) { --depth; }
            if (depth == 0) {
                value_end = close;
                return true;
            }
            scan = close + form.close.size();
        }
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
    bool lenient_;
    mutable std::uint32_t duplicate_parameters_repaired_ = 0;
};

GeneratedToolCall normalize_raw_tool_call(const RawToolCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);
    if (tool != nullptr && !tool->unambiguous) { tool = nullptr; }

    std::string arguments = "{";
    bool first            = true;
    for (const RawParameter& raw_parameter : raw.parameters) {
        const Contract::Parameter* parameter =
            tool == nullptr ? nullptr : find_parameter_contract(*tool, raw_parameter.name);
        NormalizedParameter normalized = normalize_parameter(raw_parameter.value, parameter);
        if (tool != nullptr && parameter == nullptr) {
            normalized.disposition = ParameterNormalization::SchemaMismatch;
        }
        if (normalized.disposition == ParameterNormalization::Omitted) {
            ++diagnostics.empty_arguments_omitted;
            continue;
        }
        if (normalized.disposition == ParameterNormalization::SchemaMismatch) {
            ++diagnostics.schema_mismatch_arguments;
        }

        if (!first) { arguments.push_back(','); }
        first = false;
        arguments += encode_json_string(raw_parameter.name);
        arguments.push_back(':');
        arguments += normalized.json_value;
    }
    arguments.push_back('}');

    return GeneratedToolCall{.name = std::string(raw.name), .arguments_json = std::move(arguments)};
}

ParsedToolCallOutput fallback(const std::string& text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = text;
    out.diagnostics = diagnostics;
    return out;
}

// The function a region or call opens, after its wrapper or container opener, or null.
const TagForm* opened_function(std::string_view text, std::size_t& pos) {
    if (starts_with_at(text, pos, kToolOpen)) {
        pos += kToolOpen.size();
    } else if (starts_with_at(text, pos, kFunctionCallsOpen)) {
        pos += kFunctionCallsOpen.size();
    }
    skip_format_whitespace(text, pos);
    return tag_form_at(text, pos, kFunctionForms);
}

// Markup that never opens a function is prose about tool calls, not an attempt at one. The Qwen
// form counts from its opener; another form only with a name, so prose about a <function ...> tag
// stays prose.
bool region_opens_function(std::string_view region) {
    std::size_t pos         = 0;
    const TagForm* function = opened_function(region, pos);
    if (function == nullptr) { return false; }
    if (starts_with_at(region, pos, kFunctionOpen)) { return true; }
    std::size_t tag_end = 0;
    return !tag_name(tag_header(region, pos, *function, tag_end)).empty() &&
           tag_end != std::string_view::npos;
}

std::string malformed_call_arguments(std::string_view call_text, std::size_t max_name_length) {
    Json arguments          = Json::object();
    std::size_t pos         = 0;
    const TagForm* function = opened_function(call_text, pos);
    if (function != nullptr) {
        std::size_t tag_end         = 0;
        const std::string_view name = tag_name(tag_header(call_text, pos, *function, tag_end));
        if (tag_end != std::string_view::npos && valid_function_name(name, max_name_length)) {
            arguments["intended_function"] = std::string(name);
        }
    }
    const std::string_view function_close = function != nullptr ? function->close : kFunctionClose;
    arguments["error"] =
        call_text.find(function_close) == std::string_view::npos
            ? "The output ended before the tool call was closed, so nothing was executed. Issue "
              "the call again; if it was long, split the work into smaller calls."
            : "The tool call markup was malformed, so nothing was executed. Issue the call again "
              "with every parameter opened and closed by its own tag.";
    return arguments.dump();
}

// The strict pass rejected the region. Calls it can read still stand; a call it cannot read gets
// the lenient reader, and a call neither can read is reported as a call to the reserved error
// tool, so the model sees an error and retries instead of its turn ending on raw markup. Nothing
// after that point is trusted. Text after the last call is a tool result the model went on to
// imagine, and it is dropped.
void recover_tool_calls(std::string_view region, std::size_t max_name_length,
                        const Contract& contract, ParsedToolCallOutput& out) {
    const QwenToolRegionParser strict(region, max_name_length, contract);
    const QwenToolRegionParser lenient(region, max_name_length, contract, true);
    std::size_t pos = 0;
    for (;;) {
        skip_format_whitespace(region, pos);
        if (pos == region.size()) { break; }
        if (starts_with_at(region, pos, kFunctionCallsOpen)) {
            pos += kFunctionCallsOpen.size();
            continue;
        }
        if (starts_with_at(region, pos, kFunctionCallsClose)) {
            pos += kFunctionCallsClose.size();
            continue;
        }
        if (!opens_call(region, pos)) {
            out.diagnostics.trailing_content_dropped = true;
            break;
        }
        const std::size_t start = pos;
        RawToolCall call;
        const std::uint32_t strict_repaired = strict.duplicate_parameters_repaired();
        if (strict.parse_one(pos, call) == FallbackReason::None) {
            out.diagnostics.duplicate_parameters_repaired +=
                strict.duplicate_parameters_repaired() - strict_repaired;
            out.tool_calls.push_back(normalize_raw_tool_call(call, contract, out.diagnostics));
            continue;
        }
        pos  = start;
        call = {};
        const std::uint32_t lenient_repaired = lenient.duplicate_parameters_repaired();
        if (lenient.parse_one(pos, call) == FallbackReason::None) {
            out.diagnostics.duplicate_parameters_repaired +=
                lenient.duplicate_parameters_repaired() - lenient_repaired;
            ++out.diagnostics.recovered_call_count;
            out.tool_calls.push_back(normalize_raw_tool_call(call, contract, out.diagnostics));
            continue;
        }
        out.tool_calls.push_back(GeneratedToolCall{
            .name           = std::string(kMalformedCallTool),
            .arguments_json = malformed_call_arguments(region.substr(start), max_name_length)});
        out.diagnostics.malformed_call_reported = true;
        break;
    }
    out.diagnostics.recovered = true;
}

} // namespace

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled,
                                std::string_view forced_tool_name) {
    if (!enabled) { return {}; }
    auto contract                    = std::make_shared<ToolCallOutputContract>();
    contract->enforce_declared_names = true;
    contract->forced_tool_name.assign(forced_tool_name);
    contract->tools.reserve(tool_jsons.size());
    for (const std::string& tool_json : tool_jsons) {
        const Json definition = Json::parse(tool_json, nullptr, false);
        if (!definition.is_discarded()) { append_tool_contract(*contract, definition); }
    }
    return contract;
}

std::string structured_tool_call_format(const ToolCallOutputContract& contract) {
    const Json whitespace{{"type", "regex"}, {"pattern", "[ \\t\\r\\n]{0,8}"}};
    const auto literal = [](const std::string& text) {
        return Json{{"type", "const_string"}, {"value", text}};
    };
    Json alternatives = Json::array();
    for (const auto& tool : contract.tools) {
        if (!tool.unambiguous) {
            throw std::invalid_argument("structured output requires unambiguous tool definitions");
        }
        Json elements =
            Json::array({literal(std::string(kToolOpen)), whitespace,
                         literal(std::string(kFunctionOpen) + tool.name + ">"), whitespace});
        for (const auto& parameter : tool.parameters) {
            if (parameter.name.empty() || parameter.name.find('>') != std::string::npos) {
                throw std::invalid_argument("tool parameter cannot be represented in Qwen XML");
            }
            Json value{{"type", "any_text"},
                       {"excludes", Json::array({kParamClose, kParamOpen, kFunctionClose, kToolOpen,
                                                 kToolClose})}};
            Json tagged{{"type", "tag"},
                        {"begin", std::string(kParamOpen) + parameter.name + ">"},
                        {"content", value},
                        {"end", kParamClose}};
            elements.push_back(
                Json{{"type", "optional"},
                     {"content", Json{{"type", "sequence"},
                                      {"elements", Json::array({tagged, whitespace})}}}});
        }
        elements.push_back(literal(std::string(kFunctionClose)));
        elements.push_back(whitespace);
        elements.push_back(literal(std::string(kToolClose)));
        elements.push_back(whitespace);
        alternatives.push_back(Json{{"type", "sequence"}, {"elements", elements}});
    }
    if (alternatives.empty()) { return {}; }
    return Json{{"type", "sequence"},
                {"elements",
                 Json::array({whitespace, Json{{"type", "plus"},
                                               {"content", Json{{"type", "or"},
                                                                {"elements", alternatives}}}}})}}
        .dump();
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract) {
    std::size_t candidate = find_tool_marker(text, 0);
    if (candidate == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.diagnostics.marker_seen = true;

    // Generated prose can quote a tool-call marker before the real turn. Try each marker in order
    // and accept the first region that consumes the response to its end; earlier markers stay
    // ordinary content. Every attempt can scan to the end of the output, so an output that keeps
    // repeating the marker is tried only up to a bound. The bound counts from the first marker,
    // not the last, so no call of a multi-call turn is ever left behind as text.
    const std::string_view source(text);
    const std::size_t first = candidate;
    std::vector<RawToolCall> raw_calls;
    std::size_t accepted         = std::string::npos;
    FallbackReason first_failure = FallbackReason::MalformedStructure;
    bool first_failure_recorded  = false;
    for (std::size_t attempt = 0; candidate != std::string::npos && attempt < kMaxMarkerAttempts;
         ++attempt) {
        std::vector<RawToolCall> calls;
        const QwenToolRegionParser parser(source.substr(candidate), max_tool_name_length, contract);
        const FallbackReason failure = parser.parse(calls);
        if (failure == FallbackReason::None) {
            accepted  = candidate;
            raw_calls = std::move(calls);
            out.diagnostics.duplicate_parameters_repaired = parser.duplicate_parameters_repaired();
            break;
        }
        if (!first_failure_recorded) {
            first_failure          = failure;
            first_failure_recorded = true;
            // A first region that opens with a complete call is the real turn, broken later on:
            // it goes to recovery rather than yielding to a later marker.
            if (!calls.empty()) { break; }
        }
        candidate = find_tool_marker(text, candidate + 1);
    }
    if (accepted == std::string::npos) {
        // No marker opens a region that parses: recover what the first one's region holds.
        out.diagnostics.fallback_reason    = first_failure;
        const std::string_view tool_region = source.substr(first);
        if (!region_opens_function(tool_region)) { return fallback(text, out.diagnostics); }
        out.content = rtrim_format_whitespace(source.substr(0, first));
        recover_tool_calls(tool_region, max_tool_name_length, contract, out);
    } else {
        out.content = rtrim_format_whitespace(source.substr(0, accepted));
        out.tool_calls.reserve(raw_calls.size());
        for (const RawToolCall& raw : raw_calls) {
            out.tool_calls.push_back(normalize_raw_tool_call(raw, contract, out.diagnostics));
        }
    }

    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length) {
    if (contract_ == nullptr || contract_->forced_tool_name.empty()) { return; }
    const std::string& forced_tool_name = contract_->forced_tool_name;
    // The generation prompt ends with exactly this opener, so generation resumes inside the call
    // and the region the parser sees has to begin where the prompt left off.
    tool_region_.append(kToolOpen);
    tool_region_.push_back('\n');
    tool_region_.append(kFunctionOpen);
    tool_region_.append(forced_tool_name);
    tool_region_.append(">\n");
    seeded_prefix_bytes_ = tool_region_.size();
    saw_tool_marker_     = true;
    forced_              = true;
}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    std::string visible;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char byte = text[index];
        if (!pending_marker_.empty()) {
            pending_marker_.push_back(byte);
            if (opens_marker(pending_marker_)) {
                tool_region_ = std::move(trailing_whitespace_);
                trailing_whitespace_.clear();
                tool_region_.append(pending_marker_);
                tool_region_.append(text.substr(index + 1));
                pending_marker_.clear();
                saw_tool_marker_ = true;
                break;
            }
            if (could_open_marker(pending_marker_)) { continue; }
            // The byte that ended the candidate can start the next one, so it is read again.
            pending_marker_.pop_back();
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.append(pending_marker_);
            pending_marker_.clear();
        }

        if (byte == '<') {
            pending_marker_.push_back(byte);
        } else if (is_format_whitespace(byte)) {
            trailing_whitespace_.push_back(byte);
        } else {
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.push_back(byte);
        }
    }
    return visible;
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish() {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    ParsedToolCallOutput parsed =
        parse_qwen_tool_call_output(tool_region_, max_tool_name_length_, *contract_);
    if (forced_ && parsed.diagnostics.fallback_reason != FallbackReason::None) {
        // A turn that ends after the function closed is a complete call missing only its outer
        // tag. Supplying that tag invents no argument byte; anything less complete keeps what
        // the recovery pass made of it.
        std::string completed = rtrim_format_whitespace(tool_region_);
        if (std::string_view(completed).ends_with(kFunctionClose)) {
            completed.push_back('\n');
            completed.append(kToolClose);
            ParsedToolCallOutput closed =
                parse_qwen_tool_call_output(completed, max_tool_name_length_, *contract_);
            if (closed.diagnostics.fallback_reason == FallbackReason::None) {
                closed.diagnostics.forced_call_closed = true;
                parsed                                = std::move(closed);
                tool_region_                          = std::move(completed);
            }
        }
    }
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        // The parser reports the held bytes before the accepted structured region, which are the
        // bytes after an earlier quoted marker that this decoder has not published yet.
        std::string content = std::move(parsed.content);
        trailing_whitespace_.clear();
        tool_region_.clear();
        pending_marker_.clear();
        return Terminal{.content     = std::move(content),
                        .tool_calls  = std::move(parsed.tool_calls),
                        .diagnostics = parsed.diagnostics};
    }

    std::string tail = std::move(trailing_whitespace_);
    tail.append(pending_marker_);
    pending_marker_.clear();
    // Only what the model produced. A forced turn that never formed a closing region still
    // carries the prompt's opener at the head of the region, and returning it would put bytes the
    // model never generated into the response. saw_tool_marker_ is set in the constructor for a
    // forced turn, so feed() never rebuilds the region and the prefix stays at its head.
    tail.append(std::string_view(tool_region_).substr(seeded_prefix_bytes_));
    seeded_prefix_bytes_ = 0;
    tool_region_.clear();
    return Terminal{
        .content = std::move(tail), .tool_calls = {}, .diagnostics = parsed.diagnostics};
}

} // namespace ninfer::models::qwen3_5::frontend
