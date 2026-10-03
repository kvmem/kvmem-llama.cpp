#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/structured_output.h"

#include <nlohmann/json.hpp>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::models::qwen3_5::frontend;

const fi::ToolCallOutputContract kLegacyContract;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string tool_definition(const std::string& tool_name, Json properties,
                            Json required = Json::array()) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) { parameters["required"] = std::move(required); }
    return Json{{"type", "function"},
                {"function", Json{{"name", tool_name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

std::shared_ptr<const fi::ToolCallOutputContract>
contract_from_definitions(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(
        std::span<const std::string>(definitions.data(), definitions.size()), true);
}

std::shared_ptr<const fi::ToolCallOutputContract> output_contract_for(const std::string& tool_name,
                                                                      Json properties) {
    const std::vector<std::string> definitions = {
        tool_definition(tool_name, std::move(properties))};
    return contract_from_definitions(definitions);
}

fi::ToolCallOutputContract contract_for(const std::string& tool_name, Json properties) {
    return *output_contract_for(tool_name, std::move(properties));
}

std::string
tool_call(std::string_view tool_name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=";
    text.append(tool_name);
    text += ">\n";
    for (const auto& [name, value] : parameters) {
        text += "<parameter=";
        text.append(name);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int check_rejected(const std::string& text, const fi::ToolCallOutputContract& contract,
                   ninfer::ToolCallParseFallbackReason reason, std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    return check(!parsed.is_tool_call_response && parsed.content == text &&
                     parsed.tool_calls.empty() && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

// The region failed the strict reader and nothing lenient could keep either, so the last call is
// the reserved error tool; intended_function is what the model tried to call, empty when it gave
// no usable name.
int check_reported(const fi::ParsedToolCallOutput& parsed, ninfer::ToolCallParseFallbackReason reason,
                   std::string_view intended_function, std::string_view message) {
    if (!parsed.is_tool_call_response || parsed.tool_calls.empty()) {
        return fail(std::string(message) + " (no calls)");
    }
    const auto& reported = parsed.tool_calls.back();
    const Json args      = Json::parse(reported.arguments_json);
    const bool intended_matches =
        intended_function.empty() ? !args.contains("intended_function")
                                  : args.value("intended_function", "") == intended_function;
    return check(reported.name == "malformed_tool_call" && args.at("error").is_string() &&
                     intended_matches && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.recovered && parsed.diagnostics.malformed_call_reported &&
                     parsed.diagnostics.structured_call_count == parsed.tool_calls.size() &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

int check_reported(const std::string& text, const fi::ToolCallOutputContract& contract,
                   ninfer::ToolCallParseFallbackReason reason, std::string_view intended_function,
                   std::string_view message) {
    return check_reported(fi::parse_qwen_tool_call_output(text, 64, contract), reason,
                          intended_function, message);
}

int check_parameter_schema_mismatch(const fi::ToolCallOutputContract& contract,
                                    std::string_view parameter_name, std::string_view value,
                                    std::string_view expected_json_value,
                                    std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{parameter_name, value}}), 64, contract);
    const std::string expected_arguments = "{" + Json(std::string(parameter_name)).dump() + ":" +
                                           std::string(expected_json_value) + "}";
    return check(
        parsed.is_tool_call_response && parsed.content.empty() && parsed.tool_calls.size() == 1 &&
            parsed.tool_calls.front().arguments_json == expected_arguments &&
            parsed.diagnostics.marker_seen && parsed.diagnostics.structured_call_count == 1 &&
            parsed.diagnostics.schema_mismatch_arguments == 1 &&
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
        std::string(message));
}

int test_basic_legacy_parsing() {
    const auto parsed = fi::parse_qwen_tool_call_output("Calling weather.\n"
                                                        "<tool_call>\n"
                                                        "<function=get_weather>\n"
                                                        "<parameter=city>\nParis\n</parameter>\n"
                                                        "<parameter=days>\n2\n</parameter>\n"
                                                        "</function>\n"
                                                        "</tool_call>",
                                                        64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "legacy call was not parsed");
    failures += check(parsed.content == "Calling weather.", "content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 1, "legacy call count changed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls.front().name == "get_weather", "function name changed");
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("city") == "Paris", "legacy string inference changed");
    failures += check(args.at("days") == 2, "legacy JSON inference changed");
    return failures;
}

int test_multiple_calls() {
    const std::string text = tool_call("first", {{"payload", "{\"ok\":true,\"items\":[1,2]}"}}) +
                             "\n" + tool_call("second", {{"value", "plain text"}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                      "multiple complete calls were not parsed");
    if (parsed.tool_calls.size() != 2) { return failures; }
    const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures +=
        check(first.at("payload").at("ok") == true && first.at("payload").at("items").at(1) == 2,
              "legacy object value changed");
    failures += check(second.at("value") == "plain text", "legacy plain text value changed");
    return failures;
}

int test_declared_strings_preserve_text() {
    const auto contract =
        contract_for("TaskUpdate",
                     Json{{"taskId", Json{{"type", "string"}}},
                          {"content", Json{{"type", "string"}}},
                          {"truthy", Json{{"type", "string"}}},
                          {"nullish", Json{{"type", "string"}}},
                          {"quoted", Json{{"type", "string"}}},
                          {"windows", Json{{"type", "string"}}},
                          {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared string call was rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("taskId") == "1", "numeric-shaped string was promoted");
    failures +=
        check(args.at("content") == "  {\"x\":1}\n", "string whitespace or content changed");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped string was promoted");
    failures +=
        check(args.at("quoted") == "\"literal\"", "quoted string was reinterpreted as JSON");
    failures += check(args.at("windows") == "  value  ", "CRLF framing changed string content");
    failures +=
        check(args.at("string_or_number") == "7", "string-admitting union did not preserve text");
    return failures;
}

int test_string_values_preserve_embedded_tool_markup() {
    const auto contract       = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                          {"timeout", Json{{"type", "integer"}}}});
    const std::string command = "python3 - <<'PY'\n"
                                "import re\n"
                                "pattern = r'<parameter=edits>\\n(.*?)\\n</parameter>'\n"
                                "print(pattern)\n"
                                "PY";
    const std::string text    = tool_call("bash", {{"command", command}, {"timeout", "30"}});
    const auto parsed         = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "balanced parameter markup inside a string broke the tool call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("command") == command,
                      "embedded parameter markup was removed from the string value");
    failures +=
        check(args.at("timeout") == 30, "sibling parameter after embedded markup was not parsed");

    const std::string nested_markup =
        "literal closes: </function> and </tool_call>\n"
        "<function=fake>body</function>\n"
        "<tool_call>body</tool_call>\n"
        "<parameter=outer>before<parameter=inner>value</parameter>after</parameter>";
    const std::string nested_text = tool_call("bash", {{"command", nested_markup}});
    const auto nested             = fi::parse_qwen_tool_call_output(nested_text, 64, contract);
    failures += check(nested.is_tool_call_response && nested.tool_calls.size() == 1,
                      "nested tool markup inside a string broke outer structure");
    if (nested.tool_calls.size() == 1) {
        const Json nested_args = Json::parse(nested.tool_calls.front().arguments_json);
        failures += check(nested_args.at("command") == nested_markup,
                          "nested function/tool/parameter markup was not preserved exactly");
    }
    return failures;
}

int test_unrepresentable_parameter_delimiters_are_reported() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string unmatched_open =
        tool_call("bash", {{"command", "echo '<parameter=unterminated>'"}});
    // Only a closer that runs on into ordinary text can be value text.
    const std::string closer_before_structure =
        tool_call("bash", {{"command", "echo\n</parameter>\n</function>\ntail"}});
    const std::string closer_before_other_markup =
        tool_call("bash", {{"command", "git status\n</parameter>\n<think>\nNo output."}});

    int failures = 0;
    failures += check_reported(unmatched_open, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure, "bash",
                               "unbalanced nested parameter open was silently repaired");
    failures += check_reported(closer_before_structure, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure, "bash",
                               "structurally ambiguous parameter close was guessed");
    failures += check_reported(closer_before_other_markup, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure, "bash",
                               "markup after a parameter close was folded into the value");
    return failures;
}

int test_quoted_parameter_close_stays_in_value() {
    const auto contract = contract_for(
        "agent", Json{{"task", Json{{"type", "string"}}}, {"mode", Json{{"type", "string"}}}});
    // Shape of an observed advisor call: prose quotes the closer sequence with escaped newlines.
    const std::string task =
        "Failures:\n"
        "   2. `</parameter>\\n</function>\\n</tool_call>` - stopped mid value\n"
        "   4. `>\\n</function>` - stopped mid `</parameter>`\n"
        "echo '</parameter>'";
    const std::string text = tool_call("agent", {{"task", task}, {"mode", "read"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(
        parsed.is_tool_call_response && parsed.tool_calls.size() == 1 && parsed.content.empty() &&
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
        "quoted parameter close demoted a valid call");
    if (parsed.tool_calls.size() == 1 && parsed.tool_calls.front().name == "agent") {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("task") == task && args.at("mode") == "read",
                          "quoted parameter close changed parameter values");
    }

    const std::string truncated = "<tool_call>\n<function=agent>\n<parameter=task>\n" + task;
    failures +=
        check_reported(truncated, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "agent", "truncated value with quoted closers was accepted");
    return failures;
}

int test_declared_json_types() {
    const auto contract = contract_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"unset", Json{{"type", "null"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "7"},
                                                     {"total", "8"},
                                                     {"ratio", "1.5"},
                                                     {"enabled", "true"},
                                                     {"payload", "{\"x\":1}"},
                                                     {"items", "[\"a\",2]"},
                                                     {"unset", "null"},
                                                     {"optional", "null"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid declared JSON values were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("count") == 7, "integer was not decoded");
    failures += check(args.at("total") == 8, "integer did not satisfy number");
    failures += check(args.at("ratio") == 1.5, "fractional number was not decoded");
    failures += check(args.at("enabled") == true, "JSON boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object was not decoded");
    failures +=
        check(args.at("items").is_array() && args.at("items").at(1) == 2, "array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null was not decoded");
    return failures;
}

int test_boolean_boundary() {
    const auto contract =
        contract_for("configure", Json{{"lower_true", Json{{"type", "boolean"}}},
                                       {"title_true", Json{{"type", "boolean"}}},
                                       {"upper_true", Json{{"type", "boolean"}}},
                                       {"mixed_false", Json{{"type", "boolean"}}},
                                       {"spaced_true", Json{{"type", "boolean"}}},
                                       {"windows_false", Json{{"type", "boolean"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=lower_true>\ntrue\n</parameter>\n"
                                        "<parameter=title_true>\nTrue\n</parameter>\n"
                                        "<parameter=upper_true>\nTRUE\n</parameter>\n"
                                        "<parameter=mixed_false>\nfAlSe\n</parameter>\n"
                                        "<parameter=spaced_true>\n \tTrUe \n</parameter>\n"
                                        "<parameter=windows_false>\r\nFaLsE\r\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "case-insensitive booleans were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("lower_true") == true && args.at("title_true") == true &&
                          args.at("upper_true") == true && args.at("spaced_true") == true,
                      "true variants were not canonicalized");
    failures += check(args.at("mixed_false") == false && args.at("windows_false") == false,
                      "false variants were not canonicalized");

    const auto one_flag = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    failures += check_parameter_schema_mismatch(one_flag, "flag", "1", "1",
                                                "integer boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "0", "0",
                                                "zero boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "\"true\"", "\"true\"",
                                                "string boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "yes", "\"yes\"",
                                                "plain boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "None", "\"None\"",
                                                "Python null mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "null", "null",
                                                "null boolean mismatch was not structured");
    return failures;
}

int test_exact_integer_boundary() {
    const auto integer_contract =
        contract_for("configure", Json{{"decimal", Json{{"type", "integer"}}},
                                       {"exponent", Json{{"type", "integer"}}},
                                       {"scaled", Json{{"type", "integer"}}},
                                       {"negative_zero", Json{{"type", "integer"}}},
                                       {"large", Json{{"type", "integer"}}}});
    const std::string valid = tool_call("configure", {{"decimal", "7.0"},
                                                      {"exponent", "1e2"},
                                                      {"scaled", "100e-2"},
                                                      {"negative_zero", "-0.0"},
                                                      {"large", "9007199254740992.0"}});
    const auto parsed       = fi::parse_qwen_tool_call_output(valid, 64, integer_contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "mathematically integral JSON numbers were rejected");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto one_integer = contract_for("configure", Json{{"value", Json{{"type", "integer"}}}});
    failures += check_parameter_schema_mismatch(one_integer, "value", "7.5", "7.5",
                                                "fractional integer mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_integer, "value", "1e-1", "1e-1",
                                                "fractional exponent mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        one_integer, "value", "9007199254740992.5", "9007199254740992.5",
        "large fractional integer mismatch lost its exact lexeme");

    const auto one_number = contract_for("configure", Json{{"value", Json{{"type", "number"}}}});
    const std::string large_fraction = tool_call("configure", {{"value", "9007199254740992.5"}});
    const auto number_parsed = fi::parse_qwen_tool_call_output(large_fraction, 64, one_number);
    failures += check(number_parsed.is_tool_call_response && number_parsed.tool_calls.size() == 1 &&
                          number_parsed.tool_calls.front().arguments_json ==
                              "{\"value\":9007199254740992.5}",
                      "valid number was rejected or lost its original precision");
    return failures;
}

int test_composed_schema_types() {
    const auto contract = contract_for(
        "configure",
        Json{{"flag",
              Json{{"anyOf", Json::array({Json{{"type", "boolean"}}, Json{{"type", "null"}}})}}},
             {"unset",
              Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "boolean"}}})}}},
             {"count",
              Json{{"anyOf", Json::array({Json{{"type", "integer"}}, Json{{"type", "null"}}})}}},
             {"nested",
              Json{{"anyOf", Json::array({Json{{"oneOf", Json::array({Json{{"type", "boolean"}},
                                                                      Json{{"type", "null"}}})}},
                                          Json{{"type", "integer"}}})}}},
             {"string_or_number",
              Json{{"oneOf", Json::array({Json{{"type", "string"}}, Json{{"type", "number"}}})}}}});
    const std::string text = tool_call("configure", {{"flag", "False"},
                                                     {"unset", "null"},
                                                     {"count", "7.0"},
                                                     {"nested", "TRUE"},
                                                     {"string_or_number", "7"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "explicit anyOf/oneOf primitive union was rejected");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("flag") == false && args.at("unset").is_null(),
                          "nullable boolean composition was decoded incorrectly");
        failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                          "nested primitive composition was decoded incorrectly");
        failures += check(args.at("string_or_number") == "7",
                          "string-admitting composition did not preserve text");
    }
    failures += check_parameter_schema_mismatch(
        contract, "count", "7.5", "7.5",
        "fractional anyOf integer/null mismatch was not structured");
    return failures;
}

int test_empty_declared_non_string_is_omitted() {
    const Json properties = {
        {"file_path", Json{{"type", "string"}}},
        {"new_string", Json{{"type", "string"}}},
        {"old_string", Json{{"type", "string"}}},
        {"replace_all", Json{{"type", "boolean"}}},
    };
    const std::string text = "I need one more check.\n\n"
                             "<tool_call>\n"
                             "<function=Edit>\n"
                             "<parameter=file_path>\n/tmp/probe.cpp\n</parameter>\n"
                             "<parameter=new_string>\n"
                             "std::map<std::uint32_t, int> counts;\n"
                             "</parameter>\n"
                             "<parameter=old_string>\nold line\n</parameter>\n"
                             "<parameter=replace_all>\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const std::vector<std::string> definitions = {tool_definition(
        "Edit", properties, Json::array({"file_path", "new_string", "old_string"}))};
    const auto contract                        = contract_from_definitions(definitions);
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, *contract);

    int failures = 0;
    failures +=
        check(parsed.is_tool_call_response && parsed.content == "I need one more check." &&
                  parsed.tool_calls.size() == 1 && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.structured_call_count == 1 &&
                  parsed.diagnostics.empty_arguments_omitted == 1 &&
                  parsed.diagnostics.schema_mismatch_arguments == 0 &&
                  parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
              "empty optional boolean demoted a complete Edit call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 3 && args.at("file_path") == "/tmp/probe.cpp" &&
                              args.at("new_string") == "std::map<std::uint32_t, int> counts;" &&
                              args.at("old_string") == "old line" && !args.contains("replace_all"),
                          "empty optional boolean was not omitted from Edit arguments");
    }

    bool every_split_matches = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        auto terminal = decoder.finish();
        if (visible != "I need one more check." || !terminal.content.empty() ||
            terminal.tool_calls.size() != 1 ||
            terminal.tool_calls.front().arguments_json !=
                parsed.tool_calls.front().arguments_json ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
            break;
        }
    }
    failures += check(every_split_matches,
                      "incremental Edit parsing depends on the transport chunk boundary");

    fi::ToolCallOutputDecoder bytewise(contract, 128);
    std::string bytewise_visible;
    for (const char byte : text) { bytewise_visible += bytewise.feed(std::string_view(&byte, 1)); }
    auto bytewise_terminal = bytewise.finish();
    failures +=
        check(bytewise_visible == "I need one more check." && bytewise_terminal.content.empty() &&
                  bytewise_terminal.tool_calls.size() == 1 &&
                  bytewise_terminal.diagnostics == parsed.diagnostics,
              "bytewise Edit parsing changed the terminal tool-call semantics");

    const auto string_contract =
        contract_for("configure", Json{{"label", Json{{"type", "string"}}}});
    const auto empty_string = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"label", ""}}), 64, string_contract);
    failures +=
        check(empty_string.is_tool_call_response && empty_string.tool_calls.size() == 1 &&
                  Json::parse(empty_string.tool_calls.front().arguments_json).at("label") == "",
              "empty declared string was incorrectly omitted");
    return failures;
}

int test_schema_mismatches_remain_structured() {
    const auto contract =
        contract_for("configure", Json{{"integer_value", Json{{"type", "integer"}}},
                                       {"number_value", Json{{"type", "number"}}},
                                       {"boolean_value", Json{{"type", "boolean"}}},
                                       {"object_value", Json{{"type", "object"}}},
                                       {"array_value", Json{{"type", "array"}}},
                                       {"null_value", Json{{"type", "null"}}}});

    int failures = 0;
    failures += check_parameter_schema_mismatch(contract, "number_value", "\"1\"", "\"1\"",
                                                "string number mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "boolean_value", "[]", "[]",
                                                "array boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "object_value", "[]", "[]",
                                                "array object mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "array_value", "{}", "{}",
                                                "object array mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "null_value", "false", "false",
                                                "boolean null mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        contract, "object_value", "{'x': True}", "\"{'x': True}\"",
        "Python object mismatch was not preserved for client validation");
    failures += check_parameter_schema_mismatch(
        contract, "array_value", "['a', None]", "\"['a', None]\"",
        "Python array mismatch was not preserved for client validation");
    return failures;
}

int test_unsupported_schema_uses_legacy_policy() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"partial_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"enum", Json::array({1, 2})}}})}}},
             {"mixed_composition", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                        {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "unsupported schema did not retain legacy policy");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");
    return failures;
}

int test_recovery_of_strict_failures() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    const std::string malformed = "<tool_call>\n<function=configure>\n";
    const auto truncated        = fi::parse_qwen_tool_call_output(malformed, 64, contract);
    failures += check_reported(truncated, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "configure", "missing structural tags were accepted");
    if (!truncated.tool_calls.empty()) {
        const Json args = Json::parse(truncated.tool_calls.back().arguments_json);
        failures += check(args.at("error").get<std::string>().find("ended before") !=
                              std::string::npos,
                          "a cut-off call was not reported as cut off");
    }

    const std::string suffix = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    const auto suffixed      = fi::parse_qwen_tool_call_output(suffix, 64, contract);
    failures += check(suffixed.is_tool_call_response && suffixed.content.empty() &&
                          suffixed.tool_calls.size() == 1 &&
                          suffixed.tool_calls.front().name == "configure" &&
                          suffixed.diagnostics.recovered &&
                          suffixed.diagnostics.trailing_content_dropped &&
                          !suffixed.diagnostics.malformed_call_reported &&
                          suffixed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::TrailingContent,
                      "a complete call followed by text did not keep the call and drop the text");

    const std::string missing_parameter_close =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
        "</function>\n</tool_call>";
    const auto unclosed = fi::parse_qwen_tool_call_output(missing_parameter_close, 64, contract);
    failures += check(unclosed.is_tool_call_response && unclosed.tool_calls.size() == 1 &&
                          unclosed.tool_calls.front().arguments_json == "{\"value\":\"x\"}" &&
                          unclosed.diagnostics.recovered_call_count == 1 &&
                          !unclosed.diagnostics.malformed_call_reported,
                      "a parameter closed by its function was not recovered");

    const std::string unknown_tool = tool_call("other", {{"value", "x"}});
    const auto unknown             = fi::parse_qwen_tool_call_output(unknown_tool, 64, contract);
    failures += check(unknown.is_tool_call_response && unknown.tool_calls.size() == 1 &&
                          unknown.tool_calls.front().name == "other" &&
                          unknown.diagnostics.recovered_call_count == 1 &&
                          unknown.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                      "an undeclared tool name was not handed to the client");

    const std::string invalid_name = tool_call("bad.name", {{"value", "x"}});
    failures += check_reported(invalid_name, kLegacyContract,
                               ninfer::ToolCallParseFallbackReason::InvalidToolName, "",
                               "invalid function-name character was accepted");

    const std::string prose = "Qwen wraps calls in <tool_call> tags.";
    failures += check_rejected(prose, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "prose that names the tag was turned into a call");
    return failures;
}

// The three calls that ended agent sessions in the queue on 23.09, reduced to their shape.
int test_queue_failures_are_recovered() {
    const std::vector<std::string> definitions = {
        tool_definition("alerts", Json::object()),
        tool_definition("namespaces_list", Json::object()),
        tool_definition("Grep", Json{{"pattern", Json{{"type", "string"}}},
                                     {"-n", Json{{"type", "boolean"}}}}),
        tool_definition("Bash", Json{{"command", Json{{"type", "string"}}}})};
    const auto contract = contract_from_definitions(definitions);
    int failures        = 0;

    // A deferred tool the model had not loaded, called next to two declared ones.
    const std::string three =
        tool_call("alerts") + "\n" + tool_call("namespaces_list") + "\n" + tool_call("streams");
    const auto parallel = fi::parse_qwen_tool_call_output(three, 64, *contract);
    failures += check(parallel.is_tool_call_response && parallel.tool_calls.size() == 3 &&
                          parallel.tool_calls[2].name == "streams" &&
                          parallel.diagnostics.recovered_call_count == 1,
                      "an undeclared third call discarded the two declared ones");

    // A flag parameter with neither value nor close tag.
    const std::string flag = "<tool_call>\n<function=Grep>\n<parameter=pattern>\nTODO\n</parameter>\n"
                             "<parameter=-n>\n</function>\n</tool_call>";
    const auto grep = fi::parse_qwen_tool_call_output(flag, 64, *contract);
    failures += check(grep.is_tool_call_response && grep.tool_calls.size() == 1 &&
                          grep.tool_calls.front().arguments_json == "{\"pattern\":\"TODO\"}" &&
                          grep.diagnostics.recovered_call_count == 1 &&
                          grep.diagnostics.empty_arguments_omitted == 1,
                      "an unclosed empty flag was not recovered");

    // Reasoning leaking into a parameter tag, then an imagined result and another call.
    const std::string leak = tool_call("Bash", {{"command", "ls"}}) + "\n" +
                             "<tool_call>\n<function=Bash>\n<parameter=command>\ngit status\n"
                             "</parameter>\n<parameter<think>\nNo output.\n" +
                             tool_call("Bash", {{"command", "rm -rf build"}});
    const auto leaked = fi::parse_qwen_tool_call_output(leak, 64, *contract);
    failures += check_reported(leaked, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "Bash", "a leaked reasoning tag was not reported");
    failures += check(leaked.tool_calls.size() == 2 &&
                          leaked.tool_calls.front().arguments_json == "{\"command\":\"ls\"}",
                      "the complete call before the broken one was lost, or a later one ran");

    bool every_split_matches = true;
    for (const std::string* text : {&three, &flag, &leak}) {
        const auto parsed = fi::parse_qwen_tool_call_output(*text, 64, *contract);
        fi::ToolCallOutputDecoder bytewise(contract, 64);
        std::string visible;
        for (const char byte : *text) { visible += bytewise.feed(std::string_view(&byte, 1)); }
        const auto terminal = bytewise.finish();
        if (!visible.empty() || !terminal.content.empty() ||
            terminal.tool_calls.size() != parsed.tool_calls.size() ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
        }
    }
    failures += check(every_split_matches, "incremental recovery differs from whole-text recovery");
    return failures;
}

int test_name_limits_and_non_strict_omissions() {
    const std::string name(128, 'a');
    const std::string text          = tool_call(name);
    const auto anthropic            = fi::parse_qwen_tool_call_output(text, 128, kLegacyContract);
    const auto openai               = fi::parse_qwen_tool_call_output(text, 64, kLegacyContract);
    const std::string too_long_text = tool_call(std::string(129, 'a'));
    const auto too_long = fi::parse_qwen_tool_call_output(too_long_text, 128, kLegacyContract);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1,
                      "128-character Anthropic tool name was rejected");
    failures += check_reported(openai, ninfer::ToolCallParseFallbackReason::InvalidToolName, "",
                               "128-character OpenAI tool name was accepted");
    failures += check_reported(too_long, ninfer::ToolCallParseFallbackReason::InvalidToolName, "",
                               "129-character Anthropic tool name was accepted");

    const std::string definition = tool_definition(
        "optional", Json{{"value", Json{{"type", "string"}}}}, Json::array({"value"}));
    const std::vector<std::string> definitions = {definition};
    const auto contract                        = contract_from_definitions(definitions);
    const auto omitted = fi::parse_qwen_tool_call_output(tool_call("optional"), 64, *contract);
    failures += check(omitted.is_tool_call_response && omitted.tool_calls.size() == 1 &&
                          omitted.tool_calls.front().arguments_json == "{}",
                      "non-strict parser enforced required parameters");
    return failures;
}

int test_conflicting_duplicate_tool_contracts_use_legacy_normalization() {
    const std::string integer_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "integer"}}}});
    const std::string string_definition =
        tool_definition("configure", Json{{"value", Json{{"type", "string"}}}});

    const std::vector<std::string> identical_definitions = {integer_definition, integer_definition};
    const auto identical = contract_from_definitions(identical_definitions);
    const auto accepted =
        fi::parse_qwen_tool_call_output(tool_call("configure", {{"value", "7"}}), 64, *identical);

    const std::vector<std::string> conflicting_definitions = {integer_definition,
                                                              string_definition};
    const auto conflicting      = contract_from_definitions(conflicting_definitions);
    const std::string ambiguous = tool_call("configure", {{"value", "7"}});
    const auto ambiguous_parsed = fi::parse_qwen_tool_call_output(ambiguous, 64, *conflicting);

    int failures = 0;
    failures += check(accepted.is_tool_call_response && accepted.tool_calls.size() == 1,
                      "identical duplicate tool contracts became ambiguous");
    failures +=
        check(ambiguous_parsed.is_tool_call_response && ambiguous_parsed.tool_calls.size() == 1 &&
                  ambiguous_parsed.tool_calls.front().arguments_json == "{\"value\":7}" &&
                  ambiguous_parsed.diagnostics.schema_mismatch_arguments == 0,
              "conflicting duplicate tool contracts did not use legacy normalization");
    return failures;
}

int test_partial_region_keeps_complete_calls() {
    const auto contract    = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    const std::string text = tool_call("configure", {{"flag", "true"}}) +
                             "\n<tool_call>\n<function=configure>\n<parameter=flag>\nfalse\n";
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    int failures      = check_reported(parsed, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                       "configure", "a cut-off second call was not reported");
    failures += check(parsed.tool_calls.size() == 2 &&
                          parsed.tool_calls.front().arguments_json == "{\"flag\":true}",
                      "the complete first call was not kept");
    return failures;
}

int test_quoted_marker_before_real_call() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nprintf broken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.tool_calls.front().name == "bash",
                      "a quoted marker before the real call demoted the structured turn");
    failures += check(parsed.content == "explaining " + quoted + " then the real turn",
                      "quoted marker or intervening prose was not retained as content");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("command") == "echo ok", "recovered call arguments changed");
    }
    return failures;
}

int test_quoted_marker_recovery_keeps_the_whole_turn() {
    const auto contract = contract_for("todo_write", Json{{"todos", Json{{"type", "array"}}}});
    // Shape of an observed turn: reasoning quoted the opener, then the model made the real call.
    const std::string prose = "Avoid `<tool_call>\n<function=NAME>`; set continuation.\n"
                              "Design ready; recording the plan.";
    const std::string call  = tool_call("todo_write", {{"todos", "[{\"id\":\"a\"}]"}});
    const std::string text  = prose + "\n\n" + call;

    int failures      = 0;
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    failures +=
        check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                  parsed.content == prose && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
              "quoted marker swallowed the real call");

    // Every call after the quote is kept, however many there are.
    std::string many = prose;
    for (int index = 0; index < 6; ++index) { many += "\n" + call; }
    const auto all = fi::parse_qwen_tool_call_output(many, 64, contract);
    failures += check(all.tool_calls.size() == 6 && all.content == prose,
                      "recovery left an earlier call of a multi-call turn behind as text");

    // A marker quoted inside the real call's value does not hide the call itself.
    const auto bash = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted_value =
        prose + "\n" + tool_call("bash", {{"command", "grep '<tool_call>'"}});
    const auto in_value = fi::parse_qwen_tool_call_output(quoted_value, 64, bash);
    failures += check(in_value.tool_calls.size() == 1 && in_value.content == prose,
                      "marker quoted in the recovered call's value was chosen as the start");

    // A complete first call is not a quote, so a later call never runs on its own.
    const auto suffixed =
        fi::parse_qwen_tool_call_output(call + "\nNow also:\n" + call, 64, contract);
    failures +=
        check(suffixed.tool_calls.size() == 1 && suffixed.diagnostics.trailing_content_dropped &&
                  suffixed.diagnostics.fallback_reason ==
                      ninfer::ToolCallParseFallbackReason::TrailingContent,
              "a later call was recovered after a complete first call");

    // Every candidate fails: the first marker's reason is the one reported.
    failures += check_reported(prose + "\nthen <tool_call> again", contract,
                               ninfer::ToolCallParseFallbackReason::UndeclaredTool, "NAME",
                               "failed recovery lost the primary reason");

    // Streamed in uneven chunks, including a split inside the quoted marker.
    for (const std::size_t chunk : {std::size_t{1}, std::size_t{3}, std::size_t{7}}) {
        fi::ToolCallOutputDecoder decoder(
            output_contract_for("todo_write", Json{{"todos", Json{{"type", "array"}}}}), 64);
        std::string visible;
        for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
            visible += decoder.feed(std::string_view(text).substr(offset, chunk));
        }
        auto terminal = decoder.finish();
        failures += check(visible + terminal.content == prose && terminal.tool_calls.size() == 1,
                          "streamed quoted marker lost prose bytes or the real call");
    }
    return failures;
}

int test_marker_attempts_are_bounded() {
    const auto contract     = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string call  = tool_call("bash", {{"command", "echo ok"}});
    const auto quoted_times = [](int count) {
        std::string prose;
        for (int index = 0; index < count; ++index) { prose += "quote `<tool_call>` here. "; }
        return prose;
    };

    int failures = 0;
    // Sixteen markers are tried, the first one included, so fifteen quotes still leave the call.
    const std::string prose = quoted_times(15);
    const auto reached      = fi::parse_qwen_tool_call_output(prose + "\n" + call, 64, contract);
    failures += check(reached.tool_calls.size() == 1 &&
                          reached.content == prose.substr(0, prose.size() - 1),
                      "a call after fifteen quoted markers was not recovered");

    const std::string beyond = quoted_times(16) + "\n" + call;
    failures +=
        check_rejected(beyond, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "markers past the attempt bound were still tried");
    return failures;
}

int test_later_candidate_must_consume_the_end() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<parameter=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text =
        quoted + "\n" + tool_call("bash", {{"command", "echo ok"}}) + "\nstill explaining";
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty() &&
                          parsed.content == text && parsed.diagnostics.marker_seen &&
                          parsed.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "a quoted marker before a non-terminal call was partially committed");
    return failures;
}

int test_incremental_quoted_marker_preserves_bytes() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 5;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures += check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().name == "bash",
                      "incremental quoted marker hid the real tool call");
    failures += check(visible + terminal.content == "explaining " + quoted + " then the real turn",
                      "incremental quoted marker lost or duplicated bytes");
    failures +=
        check(terminal.diagnostics.marker_seen && terminal.diagnostics.structured_call_count == 1 &&
                  terminal.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
              "incremental quoted marker changed terminal diagnostics");
    return failures;
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += legacy.feed("Calling weather.  \n<tool_");
    visible += legacy.feed("call>\n<function=get_weather>");
    visible += legacy.feed("\n</function>\n</tool_call>");
    auto legacy_terminal = legacy.finish();
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    boolean_visible += boolean.feed("<tool_call>\n<function=configure>\n<parameter=enabled>\nT");
    boolean_visible += boolean.feed("r");
    boolean_visible += boolean.feed("ue\n</parameter>\n</function>\n</tool_call>");
    auto boolean_terminal = boolean.finish();

    int failures = 0;
    failures += check(visible == "Calling weather." && legacy_terminal.tool_calls.size() == 1,
                      "incremental valid call was not committed");
    failures += check(boolean_visible.empty() && boolean_terminal.content.empty() &&
                          boolean_terminal.tool_calls.size() == 1,
                      "incremental boolean call leaked as content");
    if (boolean_terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(boolean_terminal.tool_calls.front().arguments_json);
        failures += check(args.at("enabled") == true,
                          "split case-insensitive boolean was not canonicalized");
    }
    return failures;
}

int test_incremental_fallback_preserves_bytes() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    auto malformed_terminal = malformed.finish();
    restored += malformed_terminal.content;

    const std::string prose = "prefix  \n<tool_call> is the tag";
    fi::ToolCallOutputDecoder prose_decoder(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string prose_restored;
    prose_restored += prose_decoder.feed(prose.substr(0, 12));
    prose_restored += prose_decoder.feed(prose.substr(12));
    auto prose_terminal = prose_decoder.finish();
    prose_restored += prose_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += ordinary.feed("ordinary text  ");
    ordinary_text += ordinary.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish().content;

    int failures = 0;
    failures += check(restored == "prefix" && malformed_terminal.tool_calls.size() == 1 &&
                          malformed_terminal.tool_calls.front().name == "malformed_tool_call" &&
                          malformed_terminal.diagnostics.malformed_call_reported &&
                          malformed_terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "malformed incremental call was not reported to the client");
    failures += check(prose_restored == prose && prose_terminal.tool_calls.empty() &&
                          prose_terminal.diagnostics.marker_seen &&
                          !prose_terminal.diagnostics.recovered,
                      "prose naming the tag lost raw bytes");
    failures += check(ordinary_text == "ordinary text  ",
                      "ordinary incremental output lost trailing whitespace");
    failures +=
        check(partial_restored == partial_original, "partial marker mismatch lost raw bytes");
    return failures;
}

int test_incremental_embedded_parameter_markup() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string command = "pattern='<parameter=inner>value</parameter>'\n"
                                "printf '%s' \"$pattern\"";
    const std::string text    = tool_call("bash", {{"command", command}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures +=
        check(visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1,
              "chunked embedded parameter markup was not committed as a tool call");
    if (terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(terminal.tool_calls.front().arguments_json);
        failures += check(args.at("command") == command,
                          "chunked embedded parameter markup changed string bytes");
    }
    return failures;
}

int test_claude_code_xml_markup_variants() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string standard_xml =
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Initial setup\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_standard = fi::parse_qwen_tool_call_output(standard_xml, 128, contract);
    failures +=
        check(parsed_standard.is_tool_call_response && parsed_standard.tool_calls.size() == 1 &&
                  parsed_standard.tool_calls.front().name == "TaskCreate",
              "function name attribute syntax was not parsed");
    if (parsed_standard.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_standard.tool_calls.front().arguments_json);
        failures += check(args.at("description") == "Initial setup",
                          "function name attribute argument changed");
    }

    const std::string invoke_xml =
        "<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Create tasks\n</parameter>\n</invoke>\n</tool_call>";
    const auto parsed_invoke = fi::parse_qwen_tool_call_output(invoke_xml, 128, contract);
    failures += check(parsed_invoke.is_tool_call_response && parsed_invoke.tool_calls.size() == 1 &&
                          parsed_invoke.tool_calls.front().name == "TaskCreate",
                      "invoke tag syntax was not parsed");

    const std::string function_calls_xml =
        "<function_calls>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Function calls container\n</parameter>\n</invoke>\n</function_calls>";
    const auto parsed_function_calls =
        fi::parse_qwen_tool_call_output(function_calls_xml, 128, contract);
    failures += check(parsed_function_calls.is_tool_call_response &&
                          parsed_function_calls.tool_calls.size() == 1 &&
                          parsed_function_calls.tool_calls.front().name == "TaskCreate",
                      "function_calls container syntax was not parsed");

    const std::string standalone_invoke =
        "Plan is ready:\n<invoke name=\"TaskCreate\">\n<param name=\"description\">\n"
        "Standalone invoke\n</param>\n</invoke>";
    const auto parsed_standalone =
        fi::parse_qwen_tool_call_output(standalone_invoke, 128, contract);
    failures += check(parsed_standalone.is_tool_call_response &&
                          parsed_standalone.content == "Plan is ready:" &&
                          parsed_standalone.tool_calls.size() == 1 &&
                          parsed_standalone.tool_calls.front().name == "TaskCreate",
                      "standalone invoke after plan was not parsed");

    return failures;
}

int test_duplicate_parameters_keep_last_value() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    // A repeated identical parameter is the common agent-harness case: the second write leaves the
    // value alone, and the repair is still counted.
    const std::string identical_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nfirst\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_identical = fi::parse_qwen_tool_call_output(identical_dup, 64, contract);
    failures +=
        check(parsed_identical.is_tool_call_response && parsed_identical.tool_calls.size() == 1,
              "duplicate identical parameter was not accepted");
    if (parsed_identical.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_identical.tool_calls.front().arguments_json);
        failures +=
            check(args.at("value") == "first", "repeated identical parameter value changed");
    }
    failures += check(parsed_identical.diagnostics.duplicate_parameters_repaired == 1,
                      "identical duplicate parameter repair was not recorded");

    // A conflicting repeat keeps the last value, matching the JSON-object rule the `=<name>`
    // markup already follows.
    const std::string conflicting_dup =
        "<tool_call>\n<function=configure>\n<parameter=value>\nfirst\n</parameter>\n"
        "<parameter=value>\nsecond\n</parameter>\n</function>\n</tool_call>";
    const auto parsed_conflicting = fi::parse_qwen_tool_call_output(conflicting_dup, 64, contract);
    failures +=
        check(parsed_conflicting.is_tool_call_response && parsed_conflicting.tool_calls.size() == 1,
              "conflicting duplicate parameter fell back to text");
    if (parsed_conflicting.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed_conflicting.tool_calls.front().arguments_json);
        failures += check(args.at("value") == "second",
                          "conflicting duplicate parameter did not keep the last value");
    }
    failures += check(parsed_conflicting.diagnostics.duplicate_parameters_repaired == 1,
                      "conflicting duplicate parameter repair was not recorded");

    return failures;
}

int test_attribute_token_boundary() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    const std::string text =
        "<tool_call>\n<function filename=\"x\" name=\"TaskCreate\">\n"
        "<parameter filename=\"ignored\" name=\"description\">\nCreate task\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, contract);
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.tool_calls.front().name == "TaskCreate",
                      "attribute token boundary failed to extract correct name");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("description") == "Create task",
                          "parameter attribute token boundary failed");
    }
    return failures;
}

// A closing tag of the other form is a structural failure, never a silent misparse: the call is
// reported to the model as malformed.
int test_mismatched_closing_tags_reported() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    const std::vector<std::pair<std::string, std::string_view>> cases = {
        {"<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
         "Value\n</parameter>\n</invoke>\n</tool_call>",
         "function opening with invoke closing tag was accepted"},
        {"<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
         "Value\n</parameter>\n</function>\n</tool_call>",
         "invoke opening with function closing tag was accepted"},
        {"<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
         "Value\n</param>\n</function>\n</tool_call>",
         "parameter opening with param closing tag was accepted"},
        {"<tool_call>\n<function name=\"TaskCreate\">\n<param name=\"description\">\n"
         "Value\n</parameter>\n</function>\n</tool_call>",
         "param opening with parameter closing tag was accepted"},
    };
    int failures = 0;
    for (const auto& [text, message] : cases) {
        failures += check_reported(text, contract,
                                   ninfer::ToolCallParseFallbackReason::MalformedStructure,
                                   "TaskCreate", message);
    }
    return failures;
}

// Harness forms go through the same prose check, recovery and streaming paths as the Qwen form.
int test_harness_forms_recovery_and_streaming() {
    const auto contract_ptr =
        output_contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    const fi::ToolCallOutputContract& contract = *contract_ptr;
    int failures                               = 0;

    const std::string prose = "Wrap the name in a <function signature> tag, or in <invoke x>.";
    failures += check_rejected(prose, contract, ninfer::ToolCallParseFallbackReason::InvalidToolName,
                               "prose about a function tag was turned into a call");

    const std::string container =
        "<function_calls>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "first\n</parameter>\n</invoke>\n<invoke name=\"TaskCreate\">\n"
        "<parameter name=\"description\">\ncut";
    const auto cut = fi::parse_qwen_tool_call_output(container, 64, contract);
    failures += check_reported(cut, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "TaskCreate", "a cut-off second invoke was not reported");
    failures += check(cut.tool_calls.size() == 2 &&
                          cut.tool_calls.front().arguments_json == "{\"description\":\"first\"}",
                      "the complete invoke before the cut-off one was lost");

    const std::string nested = "<invoke name=\"TaskCreate\">\n<param name=\"description\">\n"
                               "write <param name=\"x\">y</param> here\n</param>\n</invoke>";
    const auto nested_parsed = fi::parse_qwen_tool_call_output(nested, 64, contract);
    failures += check(nested_parsed.is_tool_call_response &&
                          nested_parsed.tool_calls.size() == 1 &&
                          Json::parse(nested_parsed.tool_calls.front().arguments_json)
                                  .at("description") == "write <param name=\"x\">y</param> here",
                      "a nested param tag cut its enclosing value short");

    const std::string standalone = "Plan is ready:\n<invoke name=\"TaskCreate\">\n"
                                   "<param name=\"description\">\nStandalone\n</param>\n</invoke>";
    const std::string doubled    = "x <" + tool_call("TaskCreate", {{"description", "d"}});
    bool every_split_matches     = true;
    for (const std::string* text : {&container, &nested, &standalone, &doubled, &prose}) {
        const auto parsed = fi::parse_qwen_tool_call_output(*text, 64, contract);
        fi::ToolCallOutputDecoder bytewise(contract_ptr, 64);
        std::string visible;
        for (const char byte : *text) { visible += bytewise.feed(std::string_view(&byte, 1)); }
        const auto terminal = bytewise.finish();
        if (visible + terminal.content != parsed.content ||
            terminal.tool_calls.size() != parsed.tool_calls.size() ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
        }
    }
    failures += check(every_split_matches, "incremental harness-form parsing differs from whole-text");
    return failures;
}

int test_claude_code_plan_and_task_create_exact_repro() {
    const std::string task_create_def =
        tool_definition("TaskCreate", Json{{"description", Json{{"type", "string"}}},
                                           {"task_type", Json{{"type", "string"}}},
                                           {"priority", Json{{"type", "integer"}}}});
    const std::string task_update_def =
        tool_definition("TaskUpdate", Json{{"taskId", Json{{"type", "string"}}},
                                           {"status", Json{{"type", "string"}}}});
    const auto contract = contract_from_definitions({task_create_def, task_update_def});

    const std::string full_response =
        "I have analyzed the repository requirements. Here is the implementation plan:\n\n"
        "### Plan\n"
        "1. Inspect existing CUDA kernels in `src/ops/softmax_attention/`\n"
        "2. Add test coverage for long context attention splits\n"
        "3. Update frontend tool call decoder\n\n"
        "Let me create the first task in the tracking system now:\n\n"
        "<tool_call>\n"
        "<function name=\"TaskCreate\">\n"
        "<parameter name=\"description\">\n"
        "Implement split-KV page-safety and bounded loops\n"
        "</parameter>\n"
        "<parameter name=\"task_type\">\n"
        "feature\n"
        "</parameter>\n"
        "<parameter name=\"priority\">\n"
        "1\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>";

    const auto parsed = fi::parse_qwen_tool_call_output(full_response, 128, *contract);
    int failures      = 0;
    failures += check(parsed.is_tool_call_response,
                      "Claude Code Plan + TaskCreate failed to parse as tool call");
    failures += check(parsed.tool_calls.size() == 1, "tool call count != 1");
    failures += check(parsed.content.starts_with("I have analyzed"), "plan content prefix lost");
    failures += check(parsed.content.ends_with("tracking system now:"), "plan content tail lost");

    if (parsed.tool_calls.size() == 1) {
        const auto& call = parsed.tool_calls.front();
        failures += check(call.name == "TaskCreate", "tool name != TaskCreate");
        const Json args = Json::parse(call.arguments_json);
        failures +=
            check(args.at("description") == "Implement split-KV page-safety and bounded loops",
                  "TaskCreate description argument changed");
        failures +=
            check(args.at("task_type") == "feature", "TaskCreate task_type argument changed");
        failures += check(args.at("priority") == 1, "TaskCreate priority argument changed");
    }

    return failures;
}

} // namespace

// A forced tool choice writes the call opener into the generation prompt, so the decoder owns an
// opener the model never emits: it completes the model's call, closes a turn that ends on the
// closed function, and never returns the prompt's bytes as content.
int test_forced_call_decoder() {
    const std::vector<std::string> definitions = {
        tool_definition("TaskUpdate", Json{{"taskId", Json{{"type", "string"}}}})};
    const auto contract = fi::build_tool_call_output_contract(
        std::span<const std::string>(definitions.data(), definitions.size()), true, "TaskUpdate");
    int failures = check(contract != nullptr && contract->forced_tool_name == "TaskUpdate",
                         "forced tool name was not recorded on the contract");

    const auto run = [&](std::string_view continuation) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = decoder.feed(continuation);
        auto terminal       = decoder.finish();
        return std::pair<std::string, fi::ToolCallOutputDecoder::Terminal>{std::move(visible),
                                                                           std::move(terminal)};
    };

    {
        const auto [visible, terminal] =
            run("\n<parameter=taskId>\n1\n</parameter>\n</function>\n</tool_call>");
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().name == "TaskUpdate" &&
                              !terminal.diagnostics.forced_call_closed,
                          "the seeded opener did not complete the model's call");
    }
    {
        const auto [visible, terminal] = run("\n<parameter=taskId>\n1\n</parameter>\n</function>");
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().name == "TaskUpdate" &&
                              terminal.diagnostics.forced_call_closed,
                          "a turn ending on the closed function did not become the forced call");
    }
    {
        const auto [visible, terminal] = run("\n<parameter=taskId>\n1\n</par");
        failures += check(visible.empty() && terminal.content.empty() &&
                              terminal.tool_calls.size() == 1 &&
                              terminal.tool_calls.front().name == "malformed_tool_call" &&
                              Json::parse(terminal.tool_calls.front().arguments_json)
                                      .value("intended_function", "") == "TaskUpdate" &&
                              terminal.diagnostics.malformed_call_reported &&
                              !terminal.diagnostics.forced_call_closed,
                          "a cut-off forced call was not reported against its tool");
    }
    return failures;
}

// A repeated parameter keeps its last value, as JSON object syntax would, and the repair is counted.
int test_duplicate_parameter_keeps_last_value() {
    int failures = 0;
    const fi::ToolCallOutputContract contract =
        contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    const std::string duplicate = tool_call("configure", {{"value", "first"}, {"value", "second"}});
    const auto parsed = fi::parse_qwen_tool_call_output(duplicate, 64, contract);

    failures += check(parsed.is_tool_call_response, "duplicate parameter still fell back to text");
    failures += check(parsed.content.empty(), "duplicate parameter left prose behind");
    failures += check(parsed.tool_calls.size() == 1, "duplicate parameter did not yield one call");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json == R"({"value":"second"})",
                          "duplicate parameter did not keep the last value");
    }
    failures += check(parsed.diagnostics.fallback_reason ==
                          ninfer::ToolCallParseFallbackReason::None,
                      "duplicate parameter still reported a fallback reason");
    failures += check(parsed.diagnostics.duplicate_parameters_repaired == 1,
                      "duplicate parameter repair was not recorded in diagnostics");
    return failures;
}

int test_constrained_tool_envelope() {
    const std::vector<std::string> tools{
        tool_definition("weather", Json{{"city", Json{{"type", "string"}}},
                                        {"units", Json{{"type", "string"}}}}),
        tool_definition("clock", Json::object()),
        R"({"type":"function","function":{"name":"run_code","parameters":{"type":"object","properties":{"language":{"type":"string"},"code":{"type":"string"}}}}})"};
    auto contract = fi::build_tool_call_output_contract(tools, true);
    std::vector<std::string> vocab(257);
    for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
    ninfer::text::StructuredCompiler compiler(vocab, {256});
    const auto format = fi::structured_tool_call_format(*contract);
    auto grammar      = compiler.compile(
        {ninfer::StructuredOutputKind::JsonSchema,
         R"({"type":"object","properties":{"location":{"type":"string"},"temperature":{"type":"number","minimum":-100,"maximum":100}},"required":["location","temperature"],"additionalProperties":false})"},
        {.reasoning_close = "</think>", .alternative_format = format});
    const auto accepts = [&](const std::string& value) {
        auto trial = grammar->fork();
        std::vector<ninfer::TokenId> tokens;
        for (unsigned char ch : value) { tokens.push_back(ch); }
        tokens.push_back(256);
        try {
            trial->accept(tokens);
            return true;
        } catch (const std::logic_error&) { return false; }
    };
    int failures = check(accepts("plan</think>{\"location\":\"Tokyo\",\"temperature\":28}"),
                         "reasoning + final schema rejected with multiple tools enabled");
    failures += check(!accepts("plan</think>{\"location\":\"Tokyo\",\"temperature\":101}"),
                      "tool alternative weakened final schema");
    failures += check(accepts("</think><tool_call><function=run_code>"
                              "<parameter=language>python</parameter>"
                              "<parameter=code>print(1)</parameter></function></tool_call>"),
                      "tool grammar reordered prompt-declared parameters");
    const std::string call = "<tool_call>\n<function=weather>\n<parameter=city>Tokyo</parameter>\n"
                             "<parameter=units>celsius</parameter>\n</function>\n</tool_call>";
    const std::string parallel = call + "\n<tool_call><function=clock></function></tool_call>";
    failures += check(accepts("plan</think>\n" + parallel), "parallel native tools rejected");
    const auto parsed = fi::parse_qwen_tool_call_output(parallel, 128, *contract);
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2 &&
                          parsed.content.empty(),
                      "constrained native calls did not round-trip");
    failures += check(!accepts("</think>" + call + " prose"), "tool suffix allowed prose");
    failures += check(!accepts("</think><tool_call><function=unknown></function></tool_call>"),
                      "undeclared tool accepted");
    failures += check(!accepts("</think><tool_call><function=weather><parameter=city>x</parameter>"
                               "<parameter=city>y</parameter></function></tool_call>"),
                      "duplicate parameter accepted");
    failures +=
        check(!accepts("</think><tool_call><function=weather>"), "incomplete tool can stop");
    failures += check(!accepts("</think>not JSON"), "reasoning alternative escaped constraints");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_duplicate_parameter_keeps_last_value();
    failures += test_constrained_tool_envelope();
    failures += test_basic_legacy_parsing();
    failures += test_forced_call_decoder();
    failures += test_multiple_calls();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_unrepresentable_parameter_delimiters_are_reported();
    failures += test_quoted_parameter_close_stays_in_value();
    failures += test_declared_json_types();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_schema_mismatches_remain_structured();
    failures += test_unsupported_schema_uses_legacy_policy();
    failures += test_recovery_of_strict_failures();
    failures += test_queue_failures_are_recovered();
    failures += test_name_limits_and_non_strict_omissions();
    failures += test_conflicting_duplicate_tool_contracts_use_legacy_normalization();
    failures += test_partial_region_keeps_complete_calls();
    failures += test_quoted_marker_before_real_call();
    failures += test_quoted_marker_recovery_keeps_the_whole_turn();
    failures += test_marker_attempts_are_bounded();
    failures += test_later_candidate_must_consume_the_end();
    failures += test_incremental_quoted_marker_preserves_bytes();
    failures += test_incremental_valid_and_boolean();
    failures += test_incremental_fallback_preserves_bytes();
    failures += test_incremental_embedded_parameter_markup();
    failures += test_claude_code_xml_markup_variants();
    failures += test_duplicate_parameters_keep_last_value();
    failures += test_attribute_token_boundary();
    failures += test_mismatched_closing_tags_reported();
    failures += test_harness_forms_recovery_and_streaming();
    failures += test_claude_code_plan_and_task_create_exact_repro();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
