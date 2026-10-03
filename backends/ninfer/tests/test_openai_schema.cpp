#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

template <typename Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 512}; }

Json base_request() {
    return Json{{"model", "qwen"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
}

OpenAIChatRequest parse(Json body) { return parse_chat_completion_request(body, limits()); }

ResolvedPromptSemantics semantics(const GenerationRequest& request) {
    ServeOptions server;
    return resolve_prompt_semantics(request, server);
}

ninfer::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), {});
}

ninfer::RequestOptions options(const GenerationRequest& request) {
    ServeOptions server;
    return to_request_options(request, server, semantics(request), true);
}

Json parse_sse(const std::string& event) {
    constexpr std::string_view prefix = "data: ";
    if (!event.starts_with(prefix) || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    return Json::parse(event.substr(prefix.size(), event.size() - prefix.size() - 2));
}

int test_request_envelope_and_sampling() {
    int failures                  = 0;
    Json body                     = base_request();
    body["stream"]                = true;
    body["stream_options"]        = Json{{"include_usage", true}, {"include_obfuscation", false}};
    body["max_completion_tokens"] = 48;
    body["max_tokens"]            = 9;
    body["temperature"]           = 0.7;
    body["top_p"]                 = 0.8;
    body["presence_penalty"]      = 0.3;
    body["frequency_penalty"]     = -0.2;
    body["seed"]                  = -1;
    body["top_k"]                 = 17;
    body["min_p"]                 = 0.05;
    body["timings_per_token"]     = true;
    body["return_progress"]       = true;

    const OpenAIChatRequest request = parse(body);
    failures += check(request.model == "qwen", "model remains in OpenAI envelope");
    failures += check(request.stream && request.include_usage, "stream metadata parsed");
    failures += check(request.timings_per_token && request.return_progress,
                      "llama.cpp response observations remain in the protocol envelope");
    failures += check(request.output_tokens_explicit && request.generation.max_tokens == 48,
                      "max_completion_tokens wins and explicitness stays in envelope");
    failures += check(request.generation.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
                      "signed seed maps modulo 2^64");
    failures +=
        check(request.generation.sampling.top_k == 17 && request.generation.sampling.min_p == 0.05,
              "compatible sampler extensions parsed");
    const ninfer::RequestOptions translated = options(request.generation);
    failures +=
        check(translated.execution.sampling.top_k == 17, "top_k reaches Engine request options");
    failures +=
        check(translated.execution.sampling.min_p && *translated.execution.sampling.min_p == 0.05F,
              "min_p reaches Engine request options");
    failures +=
        check(translated.execution.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
              "signed seed reaches Engine request options");
    failures += check(!translated.execution.post_thinking_sampling,
                      "post-thinking sampling stays off without a post_thinking object");

    Json post_thinking                    = base_request();
    post_thinking["post_thinking"]        = Json{{"temperature", 0.1}, {"top_k", 4}};
    const ninfer::RequestOptions switched = options(parse(post_thinking).generation);
    failures += check(switched.execution.post_thinking_sampling &&
                          switched.execution.post_thinking_sampling->temperature == 0.1F &&
                          switched.execution.post_thinking_sampling->top_k == 4 &&
                          !switched.execution.post_thinking_sampling->top_p,
                      "post_thinking fields reach Engine request options");
    post_thinking["post_thinking"] = Json::object();
    failures +=
        check(options(parse(post_thinking).generation).execution.post_thinking_sampling.has_value(),
              "an empty post_thinking object selects the preset");
    for (const Json& bad : {Json(0.2), Json{{"temperature", 2.5}}, Json{{"top_k", 21}},
                            Json{{"stop", "x"}}, Json{{"seed", 1.5}}}) {
        post_thinking["post_thinking"] = bad;
        failures +=
            check(api_error([&] { (void)parse(post_thinking); }).param.starts_with("post_thinking"),
                  "invalid post_thinking rejected: " + bad.dump());
    }
    {
        ServeOptions server;
        server.post_thinking_overrides.emplace();
        server.post_thinking_overrides->temperature = 0.3F;
        server.post_thinking_overrides->top_k       = 8;
        server.greedy                               = true;
        post_thinking["post_thinking"]              = Json{{"top_k", 2}};
        const GenerationRequest request             = parse(post_thinking).generation;
        const ninfer::RequestOptions merged =
            to_request_options(request, server, semantics(request), true);
        failures += check(merged.execution.post_thinking_sampling &&
                              merged.execution.post_thinking_sampling->top_k == 2 &&
                              merged.execution.post_thinking_sampling->temperature == 0.0F,
                          "request post_thinking fields over server ones, --greedy over both");
        const ninfer::RequestOptions server_only =
            to_request_options(parse(base_request()).generation, server,
                               semantics(parse(base_request()).generation), true);
        failures += check(server_only.execution.post_thinking_sampling &&
                              server_only.execution.post_thinking_sampling->top_k == 8,
                          "--post-thinking applies to a request without its own object");
    }

    const OpenAIChatRequest defaults = parse(base_request());
    failures +=
        check(!defaults.stream && !defaults.include_usage && !defaults.output_tokens_explicit &&
                  !defaults.timings_per_token && !defaults.return_progress &&
                  defaults.generation.max_tokens == limits().default_max_tokens,
              "protocol defaults remain outside GenerationRequest");

    Json malformed              = base_request();
    malformed["stream_options"] = true;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "stream_options",
                      "malformed stream_options rejected");
    malformed                      = base_request();
    malformed["timings_per_token"] = "yes";
    failures += check(api_error([&] { (void)parse(malformed); }).param == "timings_per_token",
                      "non-boolean timings_per_token rejected");
    malformed                    = base_request();
    malformed["return_progress"] = 1;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "return_progress",
                      "non-boolean return_progress rejected");
    return failures;
}

int test_structured_output() {
    int failures            = 0;
    auto body               = base_request();
    body["response_format"] = Json{{"type", "json_object"}};
    auto generation         = parse(body).generation;
    failures += check(options(generation).execution.structured_output.kind ==
                          ninfer::StructuredOutputKind::JsonObject,
                      "JSON mode reaches Engine");
    failures += check(semantics(generation).enable_thinking == false,
                      "JSON mode disables thinking default");
    const auto json_prompt = prompt(generation);
    failures += check(json_prompt.messages.front().role == ninfer::ChatRole::System &&
                          json_prompt.messages.front().parts.back().text.find("raw JSON") !=
                              std::string::npos,
                      "JSON response contract is visible to the model");
    body["response_format"] = Json{
        {"type", "json_schema"},
        {"json_schema", Json{{"name", "answer"},
                             {"strict", true},
                             {"schema", Json{{"type", "object"},
                                             {"properties", Json{{"x", Json{{"type", "integer"}}}}},
                                             {"required", Json::array({"x"})},
                                             {"additionalProperties", false}}}}}};
    generation = parse(body).generation;
    failures += check(prompt(generation)
                              .messages.front()
                              .parts.back()
                              .text.find(generation.structured_output.schema) != std::string::npos,
                      "API-only response schema reaches the prompt");
    failures += check(options(generation).execution.structured_output.kind ==
                          ninfer::StructuredOutputKind::JsonSchema,
                      "JSON schema reaches Engine");
    for (const auto& extra :
         {Json{{"enable_thinking", true}}, Json{{"reasoning_effort", "high"}}}) {
        auto enabled = body;
        enabled.update(extra);
        failures += check(semantics(parse(enabled).generation).enable_thinking == true,
                          "structured output preserves explicit reasoning");
    }
    for (const auto& extra : {Json{{"stop", "}"}}}) {
        auto invalid = body;
        invalid.update(extra);
        failures +=
            check(api_error([&] { (void)semantics(parse(invalid).generation); }).status == 400,
                  "incompatible structured output rejected");
    }
    body["response_format"]["json_schema"]["schema"]["not"] = Json::object();
    failures += check(api_error([&] { (void)parse(body); }).status == 400,
                      "unsupported schema keyword rejected");
    return failures;
}

int test_standard_field_policy() {
    int failures  = 0;
    auto rejected = [&](const char* key, Json value, const char* code) {
        Json body            = base_request();
        body[key]            = std::move(value);
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.param == key && error.code == code,
                          std::string(key) + " non-neutral value rejected");
    };

    rejected("n", 2, "n_not_supported");
    rejected("logit_bias", Json{{"12", 1}}, "logit_bias_not_supported");
    rejected("logprobs", true, "logprobs_not_supported");
    rejected("top_logprobs", 2, "logprobs_not_supported");
    rejected("response_format", Json{{"type", "json_schema"}}, "invalid_response_format");
    rejected("modalities", Json::array({"text", "audio"}), "modality_not_supported");
    rejected("web_search_options", Json::object(), "web_search_not_supported");
    rejected("moderation", Json::object(), "moderation_not_supported");
    rejected("verbosity", "high", "verbosity_not_supported");
    rejected("store", true, "store_not_supported");
    rejected("functions", Json::array({Json{{"name", "legacy"}}}), "legacy_tools_not_supported");

    Json neutral                      = base_request();
    neutral["n"]                      = 1;
    neutral["logit_bias"]             = Json{{"12", 0}, {"13", 0.0}};
    neutral["logprobs"]               = false;
    neutral["top_logprobs"]           = 0;
    neutral["response_format"]        = Json{{"type", "text"}};
    neutral["modalities"]             = Json::array({"text"});
    neutral["audio"]                  = Json{{"voice", "alloy"}};
    neutral["prediction"]             = Json{{"type", "content"}, {"content", "expected"}};
    neutral["verbosity"]              = "medium";
    neutral["store"]                  = false;
    neutral["functions"]              = Json::array();
    neutral["function_call"]          = "auto";
    neutral["metadata"]               = Json{{"trace", "client"}};
    neutral["user"]                   = "user-1";
    neutral["safety_identifier"]      = "safe-1";
    neutral["prompt_cache_key"]       = "cache-1";
    neutral["prompt_cache_options"]   = Json{{"retention", "24h"}};
    neutral["prompt_cache_retention"] = "24h";
    neutral["service_tier"]           = "priority";
    neutral["future_unknown_field"]   = Json{{"value", 1}};
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral controls and advisory hints are accepted");

    RequestLimits with_logprobs        = limits();
    with_logprobs.first_token_logprobs = true;
    const auto parse_with_logprobs     = [&](Json body) {
        return parse_chat_completion_request(body, with_logprobs);
    };
    Json top                      = base_request();
    top["top_logprobs"]           = 5;
    const OpenAIChatRequest first = parse_with_logprobs(top);
    failures += check(first.generation.first_token_top_logprobs == 5 &&
                          options(first.generation).execution.first_token_top_logprobs == 5,
                      "top_logprobs reaches Engine with --first-token-logprobs");
    for (const auto& [key, value] :
         std::vector<std::pair<const char*, Json>>{{"top_logprobs", 21}, {"logprobs", true}}) {
        Json body = top;
        body[key] = value;
        failures += check(api_error([&] { (void)parse_with_logprobs(body); }).status == 400,
                          std::string(key) + " beyond the first-token export was accepted");
    }
    Json streamed      = top;
    streamed["stream"] = true;
    failures += check(api_error([&] { (void)parse_with_logprobs(streamed); }).code ==
                          "logprobs_not_supported",
                      "streaming top_logprobs was accepted");

    Json zero_limit                     = base_request();
    zero_limit["max_completion_tokens"] = 0;
    const OpenAIChatRequest zero        = parse(zero_limit);
    failures += check(zero.output_tokens_explicit && zero.generation.max_tokens == 0,
                      "an explicit zero output limit reaches Engine's no-generation path");
    return failures;
}

int test_constrained_decoding_extensions() {
    int failures                                           = 0;
    const std::vector<std::pair<const char*, Json>> active = {
        {"grammar", "root ::= \"yes\" | \"no\""},
        {"structured_outputs", Json{{"json", Json{{"type", "object"}}}}},
        {"guided_json", Json{{"type", "object"}}},
        {"guided_regex", "[a-z]+"},
        {"guided_choice", Json::array({"yes", "no"})},
        {"guided_grammar", "root ::= \"yes\" | \"no\""},
    };
    for (const auto& [field, value] : active) {
        Json body            = base_request();
        body[field]          = value;
        const ApiError error = api_error([&] { (void)parse(body); });
        failures +=
            check(error.param == field && error.code == "constrained_decoding_not_supported" &&
                      error.message.find(field) != std::string::npos,
                  std::string(field) + " constrained decoding is explicitly rejected");
    }

    Json neutral                  = base_request();
    neutral["grammar"]            = "";
    neutral["structured_outputs"] = nullptr;
    neutral["guided_json"]        = nullptr;
    neutral["guided_regex"]       = nullptr;
    neutral["guided_choice"]      = nullptr;
    neutral["guided_grammar"]     = nullptr;
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral constrained-decoding extension values are accepted");
    return failures;
}

Json function_tool(std::string name = "weather", bool strict = false) {
    return Json{{"type", "function"},
                {"function", Json{{"name", std::move(name)},
                                  {"description", "Get weather"},
                                  {"parameters", Json{{"type", "object"}}},
                                  {"strict", strict}}}};
}

int test_tool_name_diagnostics() {
    int failures = 0;

    Json body                                    = base_request();
    body["tools"]                               = Json::array({function_tool("weather")});
    body["tools"][0]["function"]["name"]        = "mcp.weather";
    const std::string limit_spec =
        "[A-Za-z0-9_-]{1," + std::to_string(kMaximumToolNameLength) + "}";
    const ApiError invalid_name                  = api_error([&] { (void)parse(body); });
    failures += check(invalid_name.status == 400 &&
                          invalid_name.param == "tools[0].function.name" &&
                          invalid_name.message.find("'mcp.weather'") != std::string::npos &&
                          invalid_name.message.find("11 bytes") != std::string::npos &&
                          invalid_name.message.find(limit_spec) != std::string::npos,
                      "invalid tool name rejection names the value, its length, and its location");

    body["tools"][0]["function"]["name"] = Json::array({"not", "a", "string"});
    const ApiError non_string             = api_error([&] { (void)parse(body); });
    failures += check(non_string.param == "tools[0].function.name" &&
                          non_string.message.find("must be a string") != std::string::npos &&
                          non_string.message.find("array") != std::string::npos,
                      "non-string tool name rejection reports the actual JSON type");

    body                                 = base_request();
    body["tools"]                        = Json::array({function_tool("weather")});
    body["tools"][0]["function"]["name"] = "bad\nname";
    const ApiError escaped               = api_error([&] { (void)parse(body); });
    failures += check(escaped.message.find("bad\\x0aname") != std::string::npos &&
                          escaped.message.find('\n') == std::string::npos,
                      "control characters in a rejected name are escaped, not embedded raw");

    // Regression for VS Code Copilot agent traffic: MCP activation wrappers are
    // synthesized as "activate_fallback_mcp_<server>_<tool>" and exceed the
    // OpenAI 64-byte limit; the Engine must accept them up to the shared cap.
    const std::string mcp_name =
        "activate_fallback_mcp_pgsql-tools_pgsql_get_dashboard_metric_data";
    body["tools"][0]["function"]["name"] = mcp_name;
    const OpenAIChatRequest mcp_ok        = parse(body);
    failures += check(mcp_name.size() == 65 && mcp_ok.generation.tools.size() == 1 &&
                          mcp_ok.generation.tools[0].name == mcp_name,
                      "MCP wrapper names longer than 64 bytes are accepted");

    Json history = base_request();
    history["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "hi"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_1"},
                                 {"type", "function"},
                                 {"function",
                                  Json{{"name", "mcp.weather"}, {"arguments", "{}"}}}}})}}});
    const ApiError history_error = api_error([&] { (void)parse(history); });
    failures += check(history_error.param == "messages[1].tool_calls[0].function.name" &&
                          history_error.message.find("'mcp.weather'") != std::string::npos,
                      "invalid historical tool_call name reports its message location");

    Json allowed        = base_request();
    allowed["tools"]    = Json::array({function_tool("weather")});
    allowed["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "bad.name"}}})}}}};
    const ApiError allowed_error = api_error([&] { (void)parse(allowed); });
    failures += check(allowed_error.param == "tool_choice.allowed_tools.tools[0].name",
                      "invalid allowed_tools name reports its location");

    return failures;
}

int test_tools() {
    int failures                      = 0;
    Json body                         = base_request();
    body["tools"]                     = Json::array({function_tool()});
    const OpenAIChatRequest automatic = parse(body);
    failures += check(automatic.generation.uses_tools(), "function tools default to auto");
    failures += check(prompt(automatic.generation).options.tool_jsons.size() == 1,
                      "auto tools reach PromptInput");

    body["tools"][0]["future_item_field"]                 = "ignored";
    body["tools"][0]["function"]["future_function_field"] = "ignored";
    const std::string normalized_definition = prompt(parse(body).generation).options.tool_jsons[0];
    failures += check(normalized_definition.find("future_item_field") == std::string::npos &&
                          normalized_definition.find("future_function_field") == std::string::npos,
                      "unknown tool fields do not silently alter the model prompt");

    body["tool_choice"]          = "none";
    body["parallel_tool_calls"]  = false;
    const OpenAIChatRequest none = parse(body);
    failures +=
        check(!none.generation.uses_tools() && prompt(none.generation).options.tool_jsons.empty(),
              "tool_choice none makes parallel_tool_calls neutral and removes executable tools");

    body["parallel_tool_calls"]             = true;
    body["enable_thinking"]                 = false;
    body["tool_choice"]                     = "required";
    const GenerationRequest required_single = parse(body).generation;
    failures += check(required_single.tool_choice.forced_name == "weather" &&
                          prompt(required_single).options.forced_tool_name == "weather",
                      "required over a single callable tool forces that function");
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "weather"}}}};
    failures += check(parse(body).generation.tool_choice.forced_name == "weather",
                      "named tool choice forces that function");
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "missing"}}}};
    failures += check(api_error([&] { (void)parse(body); }).param == "tool_choice",
                      "named tool choice rejects a function absent from tools");
    body["enable_thinking"] = true;
    body["tool_choice"]     = Json{{"type", "function"}, {"function", Json{{"name", "weather"}}}};
    failures += check(api_error([&] { (void)prompt(parse(body).generation); }).code ==
                          "tool_choice_not_supported",
                      "a forced choice with reasoning enabled is rejected");
    body.erase("enable_thinking");
    const GenerationRequest default_reasoning = parse(body).generation;
    failures += check(semantics(default_reasoning).enable_thinking == false &&
                          prompt(default_reasoning).options.enable_thinking == false &&
                          prompt(default_reasoning).options.forced_tool_name == "weather",
                      "a forced choice turns off reasoning that only a default enabled");
    ServeOptions thinking_server;
    thinking_server.enable_thinking = true;
    failures +=
        check(resolve_prompt_semantics(default_reasoning, thinking_server).enable_thinking == false,
              "a forced choice turns off reasoning that the server default enables");
    ServeOptions effort_server;
    effort_server.default_reasoning_effort = RequestedReasoningEffort::Low;
    const ResolvedPromptSemantics forced_effort =
        resolve_prompt_semantics(default_reasoning, effort_server);
    failures += check(forced_effort.enable_thinking == false && !forced_effort.reasoning_effort,
                      "a forced choice turns off reasoning that the server default effort enables");
    body["reasoning_effort"] = "high";
    failures += check(api_error([&] { (void)prompt(parse(body).generation); }).code ==
                          "tool_choice_not_supported",
                      "a forced choice with a requested reasoning effort is rejected");
    body.erase("reasoning_effort");

    {
        Json plain = base_request();
        const GenerationRequest omitted = parse(plain).generation;
        const ResolvedPromptSemantics defaulted = resolve_prompt_semantics(omitted, effort_server);
        failures += check(defaulted.enable_thinking == true &&
                              defaulted.reasoning_effort == ninfer::ReasoningEffort::Low,
                          "the server default effort applies when the request names none");
        plain["reasoning_effort"] = "high";
        const ResolvedPromptSemantics explicit_effort =
            resolve_prompt_semantics(parse(plain).generation, effort_server);
        failures += check(explicit_effort.reasoning_effort == ninfer::ReasoningEffort::XHigh,
                          "a request effort overrides the server default effort");
        plain.erase("reasoning_effort");
        plain["enable_thinking"] = false;
        const ResolvedPromptSemantics disabled =
            resolve_prompt_semantics(parse(plain).generation, effort_server);
        failures += check(disabled.enable_thinking == false && !disabled.reasoning_effort,
                          "a request that disables thinking ignores the server default effort");
    }

    body                        = base_request();
    body["tools"]               = Json::array({function_tool(), function_tool("search")});
    body["tool_choice"]         = "required";
    const ApiError over_several = api_error([&] { (void)parse(body); });
    failures += check(over_several.code == "tool_choice_not_supported" &&
                          over_several.message.find("several tools") != std::string::npos,
                      "required over several tools stays rejected");

    body          = base_request();
    body["tools"] = Json::array({function_tool(), function_tool("search")});
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "search"}}})}}}};
    const GenerationRequest allowed = parse(body).generation;
    failures += check(allowed.tools.size() == 1 && allowed.tools[0].name == "search" &&
                          prompt(allowed).options.tool_jsons.size() == 1,
                      "allowed_tools auto narrows the executable function set");

    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "weather"}}})}};
    const GenerationRequest direct_allowed = parse(body).generation;
    failures += check(direct_allowed.tools.size() == 1 && direct_allowed.tools[0].name == "weather",
                      "direct allowed_tools compatibility shape is accepted");
    body["tool_choice"]["mode"] = "required";
    failures += check(parse(body).generation.tool_choice.forced_name == "weather",
                      "allowed_tools narrowed to one function forces it");
    body["tool_choice"]["mode"]             = "auto";
    body["tool_choice"]["tools"][0]["name"] = "missing";
    failures += check(
        api_error([&] { (void)parse(body); }).param == "tool_choice.allowed_tools.tools[0].name",
        "allowed_tools rejects names absent from the declared tool set");

    body          = base_request();
    body["tools"] = Json::array({function_tool("weather", true)});
    const OpenAIChatRequest strict_tools = parse(body);
    failures += check(strict_tools.generation.tools.size() == 1 &&
                          prompt(strict_tools.generation).options.tool_jsons[0].find(
                              "\"strict\":false") != std::string::npos,
                      "strict tools are accepted as advisory without reaching the prompt");
    body["tools"] = Json::array({Json{{"type", "custom"},
                                      {"custom",
                                       Json{{"name", "shell"},
                                            {"description", "Run a shell command"},
                                            {"format", Json{{"type", "grammar"},
                                                             {"grammar", "start: /.+/"}}}}}}});
    const GenerationRequest custom_tools = parse(body).generation;
    failures += check(custom_tools.tools.size() == 1 && custom_tools.tools[0].name == "shell" &&
                          custom_tools.tools[0].input_schema_json.find("\"input\"") !=
                              std::string::npos &&
                          custom_tools.tools[0].input_schema_json.find("start: /.+/") !=
                              std::string::npos,
                      "custom tools are served as a single-string-input function");

    // A hosted tool is the server's to run. NInfer has no executor, and the caller is not waiting
    // on one either, so the declaration is dropped instead of failing a request the client cannot
    // change: agent harnesses declare web_search unconditionally.
    body          = base_request();
    body["tools"] = Json::array({Json{{"type", "web_search"}}, function_tool()});
    const GenerationRequest hosted = parse(body).generation;
    failures += check(hosted.tools.size() == 1 && hosted.tools[0].name == "weather",
                      "hosted web_search is dropped and the function tool survives");
    body["tools"] = Json::array({Json{{"type", "web_search_preview_2025_03_11"}}});
    failures += check(parse(body).generation.tools.empty(),
                      "dated hosted tool spellings are dropped by family");
    body["tools"] = Json::array({Json{{"type", "web_search"}}});
    failures += check(!parse(body).generation.uses_tools(),
                      "a request whose only tool is hosted still parses, with no callable tools");

    // parallel_tool_calls=false is honoured by trimming the response to one call rather than
    // refused; the request must survive parsing for that to be possible.
    body                        = base_request();
    body["tools"]               = Json::array({function_tool()});
    body["parallel_tool_calls"] = false;
    const GenerationRequest sequential = parse(body).generation;
    failures += check(!sequential.parallel_tool_calls && sequential.uses_tools(),
                      "parallel_tool_calls=false is accepted alongside tools");
    body["parallel_tool_calls"] = true;
    failures += check(parse(body).generation.parallel_tool_calls,
                      "parallel_tool_calls=true is the default contract");
    body["parallel_tool_calls"] = false;
    body.erase("tools");
    failures += check(parse(body).generation.tools.empty(),
                      "parallel_tool_calls=false is neutral without tools");
    body["tool_choice"] = "auto";
    failures +=
        check(parse(body).generation.tools.empty(), "tool_choice auto is neutral without tools");

    Json history = base_request();
    history["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "weather?"}},
                     Json{{"role", "assistant"},
                          {"content", nullptr},
                          {"tool_calls",
                           Json::array({Json{{"id", "call_1"},
                                             {"type", "function"},
                                             {"function", Json{{"name", "weather"},
                                                               {"arguments", "not-json-yet"}}}}})}},
                     Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "sunny"}}});
    failures += check(parse(history).generation.has_tool_history(),
                      "tool-call history follows wire types without inventing JSON validation");

    Json mixed_assistant        = base_request();
    mixed_assistant["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "inspect"}},
         Json{{"role", "assistant"},
              {"content", "I will inspect it"},
              {"tool_calls",
               Json::array({Json{
                   {"id", "call_2"},
                   {"type", "function"},
                   {"function", Json{{"name", "inspect"}, {"arguments", R"({"path":"a"})"}}}}})}}});
    const GenerationRequest mixed_request  = parse(mixed_assistant).generation;
    const ninfer::PromptInput mixed_prompt = prompt(mixed_request);
    failures += check(mixed_request.messages[1].cache_boundary_after &&
                          !mixed_request.messages[1].content[0].cache_boundary_after &&
                          !mixed_prompt.context_cache.markers.empty() &&
                          mixed_prompt.context_cache.markers.back().location ==
                              ninfer::PromptCacheMarkerLocation::MessageBoundary &&
                          mixed_prompt.context_cache.markers.back().after_message_count == 2,
                      "automatic caching stops after a complete assistant text/tool-call turn");

    const Json ordered = Json::parse(
        R"({"model":"qwen","messages":[{"role":"user","content":"probe"}],"tools":[{"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}}]})");
    const ninfer::PromptInput ordered_prompt = prompt(parse(ordered).generation);
    failures += check(
        ordered_prompt.options.tool_jsons.size() == 1 &&
            ordered_prompt.options.tool_jsons.front() ==
                R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}},"strict":false}})",
        "OpenAI Chat changed tool-schema member order before PromptInput");
    return failures;
}

int test_messages_and_media() {
    int failures                   = 0;
    Json body                      = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "text"}, {"text", "alpha"}}, Json{{"type", "text"}, {"text", "beta"}}});
    const ninfer::PromptInput translated = prompt(parse(body).generation);
    failures += check(translated.messages[0].parts.size() == 2 &&
                          translated.messages[0].parts[0].text == "alpha" &&
                          translated.messages[0].parts[1].text == "beta",
                      "adjacent text parts preserve exact text without inserted newline");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image_url"},
              {"image_url", Json{{"url", "https://example.test/a.png"}, {"detail", "auto"}}}},
         Json{{"type", "video_url"}, {"video_url", "https://example.test/a.mp4"}}});
    const GenerationRequest media = parse(body).generation;
    failures += check(media.media_item_count() == 2 &&
                          media.messages[0].content[0].kind == ContentKind::Image &&
                          media.messages[0].content[1].kind == ContentKind::Video,
                      "image and video compatibility inputs normalize to Engine media");

    body["messages"][0]["content"][0]["image_url"]["detail"] = "high";
    failures += check(api_error([&] { (void)parse(body); }).code == "image_detail_not_supported",
                      "explicit image preprocessing detail rejected");

    auto content_rejected = [&](const char* role, const char* type) {
        Json invalid                   = base_request();
        invalid["messages"][0]["role"] = role;
        invalid["messages"][0]["content"] =
            Json::array({Json{{"type", type}, {type, "https://example.test/x"}}});
        return api_error([&] { (void)parse(invalid); }).code == "modality_not_supported";
    };
    failures +=
        check(content_rejected("assistant", "image_url"), "assistant media history rejected");
    failures += check(content_rejected("system", "image_url"),
                      "system media rejected at protocol boundary");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "capture it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_capture"},
                                 {"type", "function"},
                                 {"function", Json{{"name", "capture"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"},
              {"tool_call_id", "call_capture"},
              {"content",
               Json::array({Json{{"type", "text"}, {"text", "captured"}},
                            Json{{"type", "image_url"},
                                 {"image_url", Json{{"url", "https://example.test/capture.png"},
                                                    {"detail", "auto"}}}}})}}});
    const GenerationRequest tool_image = parse(body).generation;
    failures += check(tool_image.messages.back().role == ninfer::ChatRole::Tool &&
                          tool_image.messages.back().tool_call_id == "call_capture" &&
                          tool_image.messages.back().content.size() == 2 &&
                          tool_image.messages.back().content[0].kind == ContentKind::Text &&
                          tool_image.messages.back().content[1].kind == ContentKind::Image,
                      "tool result text and image parts normalize to one tool turn");

    body["messages"].back()["content"] = Json::array(
        {Json{{"type", "video_url"}, {"video_url", "https://example.test/capture.mp4"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "tool result video remains outside the Chat compatibility extension");

    body = base_request();
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "input_audio"}, {"input_audio", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "input audio rejected");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "file"}, {"file", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "file input rejected");

    body                        = base_request();
    body["messages"][0]["name"] = "speaker";
    failures += check(api_error([&] { (void)parse(body); }).code == "message_name_not_supported",
                      "message name rejected");

    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls",
         Json::array({Json{{"id", "call_1"},
                           {"type", "function"},
                           {"function", Json{{"name", "get_status"}, {"arguments", "{}"}}}}})}});
    body["messages"].push_back(Json{
        {"role", "tool"}, {"name", "get_status"}, {"tool_call_id", "call_1"}, {"content", "ok"}});
    const GenerationRequest named_tool_history = parse(body).generation;
    const ChatTurn& named_tool                 = named_tool_history.messages.back();
    failures += check(named_tool.role == ninfer::ChatRole::Tool &&
                          named_tool.tool_call_id == "call_1" && !named_tool.tool_result_name &&
                          named_tool.content.size() == 1 && named_tool.content[0].text == "ok",
                      "tool message name is an ignored compatibility extension");

    body["messages"].back()["name"] = Json::array();
    failures +=
        check(api_error([&] { (void)parse(body); }).message == "message name must be a string",
              "tool message name remains type checked");

    body                           = base_request();
    body["messages"][0]["name"]    = "";
    body["messages"][0]["content"] = Json::array();
    failures += check(parse(body).generation.messages[0].content.empty(),
                      "empty names and empty content arrays remain neutral");

    body = base_request();
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({Json{{"type", "refusal"}, {"refusal", "part"}}})},
             {"refusal", "top-level"}});
    const GenerationRequest refusal_history = parse(body).generation;
    const ChatTurn& refusal                 = refusal_history.messages.back();
    failures += check(refusal.content.size() == 2 && refusal.content[0].text == "part" &&
                          refusal.content[1].text == "top-level",
                      "assistant refusal history is preserved as assistant text");

    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}});
    failures += check(parse(body).generation.messages.back().content.empty(),
                      "an empty assistant history turn is representable");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "run it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"function_call", Json{{"name", "legacy"}, {"arguments", R"({"value":1})"}}}},
         Json{{"role", "function"}, {"name", "legacy"}, {"content", "done"}}});
    const GenerationRequest legacy = parse(body).generation;
    failures += check(legacy.messages[1].tool_calls.size() == 1 &&
                          legacy.messages[1].tool_calls[0].name == "legacy" &&
                          legacy.messages[2].role == ninfer::ChatRole::Tool,
                      "legacy function-call history lowers to Engine tool history");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", ""},
                                 {"type", "function"},
                                 {"function", Json{{"name", "weather"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", ""}, {"content", "done"}}});
    failures += check(parse(body).generation.has_tool_history(),
                      "string tool-call identifiers may be empty without changing history");
    return failures;
}

int test_reasoning_and_extensions() {
    int failures = 0;
    Json body    = base_request();
    body["reasoning_effort"] = "default";
    failures += check(!parse(body).generation.reasoning_effort.has_value(),
                      "reasoning_effort default alias resolves to the server-configured level");
    body["reasoning_effort"] = "auto";
    failures += check(!parse(body).generation.reasoning_effort.has_value(),
                      "reasoning_effort auto alias resolves to the server-configured level");
    body["reasoning_effort"] = "xhigh";
    failures += check(parse(body).generation.reasoning_effort == RequestedReasoningEffort::XHigh,
                      "explicit reasoning_effort still reaches the Engine");
    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"},
                                    {"content", "answer"},
                                    {"reasoning_content", "thought"},
                                    {"reasoning", "thought"}});
    failures += check(parse(body).generation.messages.back().reasoning_content == "thought",
                      "assistant reasoning aliases normalize");
    body["messages"].back()["reasoning"] = "different";
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting assistant reasoning aliases rejected");
    body["messages"].back()["reasoning_content"] = "";
    failures += check(parse(body).generation.messages.back().reasoning_content == "different",
                      "an empty reasoning alias does not conflict with a meaningful alias");
    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"}, {"content", nullptr}, {"reasoning_content", "unfinished thought"}});
    failures +=
        check(parse(body).generation.messages.back().reasoning_content == "unfinished thought",
              "reasoning-only assistant history is preserved");

    body                         = base_request();
    body["enable_thinking"]      = true;
    body["preserve_thinking"]    = false;
    body["chat_template_kwargs"] = Json{{"enable_thinking", true}, {"preserve_thinking", false}};
    const GenerationRequest normalized = parse(body).generation;
    failures += check(normalized.enable_thinking == true && normalized.preserve_thinking == false,
                      "Qwen/vLLM template aliases normalize");
    body["chat_template_kwargs"]["enable_thinking"] = false;
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting thinking aliases rejected");
    body                         = base_request();
    body["chat_template_kwargs"] = Json{{"future", 1}};
    failures +=
        check(Json::parse(parse(body).generation.chat_template_kwargs_json).at("future") == 1,
              "custom template keyword did not survive protocol parsing");
    body["chat_template_kwargs"] = Json{{"future", nullptr}};
    failures += check(parse(body).generation.messages.size() == 1,
                      "null unknown template option is neutral");

    body                        = base_request();
    body["repetition_penalty"]  = 1.0;
    body["mm_processor_kwargs"] = Json{{"max_pixels", nullptr}};
    failures +=
        check(parse(body).generation.messages.size() == 1, "neutral ecosystem defaults accepted");
    body["repetition_penalty"] = 1.1;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "repetition_penalty_not_supported",
              "non-neutral repetition penalty rejected");
    body                        = base_request();
    body["mm_processor_kwargs"] = Json{{"max_pixels", 100}};
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "mm_processor_kwargs_not_supported",
              "non-empty media processor kwargs rejected");
    return failures;
}

int test_stops_and_ranges() {
    int failures                            = 0;
    Json body                               = base_request();
    body["stop"]                            = Json::array({"A", "B"});
    const ninfer::RequestOptions translated = options(parse(body).generation);
    failures += check(translated.stop.strings.size() == 4,
                      "each stop string applies to Content and Reasoning");
    failures += check(translated.stop.strings[0].channel == ninfer::OutputChannel::Content &&
                          translated.stop.strings[1].channel == ninfer::OutputChannel::Reasoning,
                      "stop channel ordering is explicit");
    failures += check(translated.stop.include_model_defaults,
                      "checkpoint stop tokens apply when ignore_eos is absent");

    body["ignore_eos"]                       = true;
    const ninfer::RequestOptions ignore_eos = options(parse(body).generation);
    failures += check(!ignore_eos.stop.include_model_defaults,
                      "ignore_eos suppresses the checkpoint's stop tokens");
    failures += check(ignore_eos.stop.strings.size() == 4,
                      "ignore_eos keeps caller-supplied stop strings");
    body.erase("stop");
    failures += check(parse(body).generation.ignore_eos, "ignore_eos parses without a stop field");
    body["ignore_eos"] = "true";
    failures += check(api_error([&] { (void)parse(body); }).param == "ignore_eos",
                      "non-boolean ignore_eos rejected");
    body.erase("ignore_eos");

    body["stop"] = Json::array({"1", "2", "3", "4", "5"});
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "more than four stop strings rejected");
    body["stop"] = "";
    failures +=
        check(api_error([&] { (void)parse(body); }).param == "stop", "empty stop string rejected");

    body = base_request();
    failures += check(options(parse(body).generation).stop.include_model_defaults,
                      "an omitted ignore_eos keeps the checkpoint's own stop tokens");
    body["ignore_eos"] = false;
    failures += check(options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos false keeps the checkpoint's own stop tokens");
    body["ignore_eos"] = true;
    failures += check(parse(body).generation.ignore_eos &&
                          !options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos suppresses the checkpoint's own stop tokens");
    body["stop"] = Json::array({"A"});
    failures += check(options(parse(body).generation).stop.strings.size() == 2 &&
                          !options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos leaves caller stop strings in place");
    body.erase("stop");
    body["ignore_eos"] = "true";
    failures += check(api_error([&] { (void)parse(body); }).param == "ignore_eos",
                      "a non-boolean ignore_eos is rejected");

    body                                  = base_request();
    body["top_k"]                         = 21;
    const GenerationRequest invalid_top_k = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_top_k); }).param == "top_k",
                      "Engine translator owns sampler value range");
    body["top_k"]                         = 5;
    body["min_p"]                         = 1.1;
    const GenerationRequest invalid_min_p = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_min_p); }).param == "min_p",
                      "min_p range enforced by common Engine translator");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                                = "answer";
    outcome.reasoning                           = "thought";
    outcome.prompt_tokens                       = 20;
    outcome.completion_tokens                   = 7;
    outcome.reasoning_tokens                    = 3;
    outcome.finish_reason                       = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens     = 12;
    outcome.metrics.prompt_wall_seconds         = 0.04;
    outcome.metrics.generation_wall_seconds     = 0.03;
    outcome.metrics.speculative_draft_tokens    = 9;
    outcome.metrics.speculative_accepted_tokens = 6;
    return outcome;
}

OpenAIChatResponseIdentity identity() {
    return OpenAIChatResponseIdentity{.id = "chatcmpl-test", .model = "qwen", .created = 42};
}

int test_aggregate_response() {
    int failures              = 0;
    GenerationOutcome outcome = sample_outcome();
    Json response             = Json::parse(make_chat_completion_response(identity(), outcome));
    failures += check(response["choices"][0]["message"]["content"] == "answer" &&
                          response["choices"][0]["message"]["reasoning_content"] == "thought" &&
                          response["choices"][0]["message"]["refusal"].is_null(),
                      "aggregate response separates reasoning and content");
    failures += check(response["choices"][0]["logprobs"].is_null(),
                      "aggregate choice carries nullable logprobs");
    GenerationOutcome with_logprobs    = outcome;
    with_logprobs.first_token_logprobs = FirstTokenLogprobsView{
        .selected = {.bytes = "\xE4\xBD", .logprob = -0.5F},
        .top      = {{.bytes = "\xE4\xBD", .logprob = -0.5F}, {.bytes = "a", .logprob = -1.5F}}};
    const Json first  = Json::parse(make_chat_completion_response(identity(), with_logprobs));
    const Json& entry = first["choices"][0]["logprobs"]["content"][0];
    failures +=
        check(first["choices"][0]["logprobs"]["content"].size() == 1 &&
                  entry["token"] == "\xEF\xBF\xBD\xEF\xBF\xBD" &&
                  entry["bytes"] == Json::array({0xE4, 0xBD}) && entry["logprob"] == -0.5 &&
                  entry["top_logprobs"].size() == 2 && entry["top_logprobs"][1]["token"] == "a",
              "first-token log probabilities have the OpenAI content shape");
    failures += check(response["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3,
                      "aggregate usage exposes cache hits and reasoning tokens");
    failures += check(
        response["timings"]["cache_n"] == 12 && response["timings"]["prompt_n"] == 8 &&
            response["timings"]["prompt_ms"] == 40.0 &&
            response["timings"]["prompt_per_second"] == 200.0 &&
            response["timings"]["predicted_n"] == 7 &&
            response["timings"]["predicted_ms"] == 30.0 &&
            response["timings"]["predicted_per_second"] == 200.0 &&
            response["timings"]["draft_n"] == 9 && response["timings"]["draft_n_accepted"] == 6,
        "aggregate timings use exact cache and N-1 generation intervals");

    outcome.text.clear();
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit",
        .arguments_json =
            R"({"file_path":"/tmp/probe.cpp","old_string":"old","new_string":"new"})"});
    response         = Json::parse(make_chat_completion_response(identity(), outcome));
    const Json& call = response["choices"][0]["message"]["tool_calls"][0];
    failures += check(response["choices"][0]["finish_reason"] == "tool_calls" &&
                          response["choices"][0]["message"]["content"].is_null(),
                      "aggregate tool call has OpenAI terminal shape");
    failures += check(
        call["id"].get<std::string>().starts_with("call_") && call["function"]["name"] == "Edit" &&
            !Json::parse(call["function"]["arguments"].get<std::string>()).contains("replace_all"),
        "OpenAI adapter owns wire tool-call identifiers");
    return failures;
}

int test_stream_response() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true);
    Json role = parse_sse(stream.start());
    failures += check(role["choices"][0]["delta"]["role"] == "assistant" &&
                          role["choices"][0]["logprobs"].is_null() && role["usage"].is_null(),
                      "stream starts with role, nullable logprobs, and null usage");
    Json reasoning = parse_sse(stream.reasoning_delta("thought"));
    Json content   = parse_sse(stream.content_delta("ans"));
    failures += check(reasoning["choices"][0]["delta"]["reasoning_content"] == "thought" &&
                          content["choices"][0]["delta"]["content"] == "ans",
                      "stream separates reasoning and content deltas");

    GenerationOutcome outcome             = sample_outcome();
    const std::vector<std::string> events = stream.finish(outcome);
    failures +=
        check(events.size() == 4, "finish emits buffered suffix, terminal, usage, and done");
    failures += check(parse_sse(events[0])["choices"][0]["delta"]["content"] == "wer",
                      "terminal content suffix is emitted exactly once");
    failures += check(parse_sse(events[1])["choices"][0]["finish_reason"] == "stop",
                      "stream terminal finish reason emitted");
    const Json usage = parse_sse(events[2]);
    failures += check(usage["choices"].empty() &&
                          usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          usage["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3 &&
                          usage["timings"]["predicted_n"] == 7,
                      "dedicated stream usage carries token accounting and terminal timings");
    failures += check(events.back() == "data: [DONE]\n\n", "stream ends with DONE sentinel");

    OpenAIChatStream choiced_stream(identity(), true, false, false, true);
    (void)choiced_stream.start();
    (void)choiced_stream.reasoning_delta("thought");
    (void)choiced_stream.content_delta("ans");
    const std::vector<std::string> choiced_events = choiced_stream.finish(outcome);
    const Json choiced_usage                      = parse_sse(choiced_events[2]);
    failures += check(choiced_usage["choices"].size() == 1 &&
                          choiced_usage["choices"][0]["delta"].empty() &&
                          choiced_usage["choices"][0]["finish_reason"].is_null() &&
                          choiced_usage["usage"]["completion_tokens"] ==
                              usage["usage"]["completion_tokens"],
                      "usage-chunk-choice keeps usage accounting with a zero-delta choice");

    OpenAIChatStream mismatch(identity(), false);
    (void)mismatch.start();
    (void)mismatch.content_delta("different");
    failures += check(throws_logic([&] { (void)mismatch.finish(outcome); }),
                      "stream encoder rejects terminal/content divergence");

    OpenAIChatStream tool_stream(identity(), false);
    (void)tool_stream.start();
    GenerationOutcome tool_outcome;
    tool_outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit", .arguments_json = R"({"file_path":"/tmp/probe.cpp"})"});
    tool_outcome.finish_reason                 = ninfer::FinishReason::StopToken;
    const std::vector<std::string> tool_events = tool_stream.finish(tool_outcome);
    const Json tool_delta                      = parse_sse(tool_events[0]);
    failures += check(
        tool_delta["choices"][0]["delta"]["tool_calls"][0]["id"].get<std::string>().starts_with(
            "call_") &&
            tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Edit" &&
            parse_sse(tool_events[1])["choices"][0]["finish_reason"] == "tool_calls",
        "stream encoder owns stable OpenAI tool-call shape");
    return failures;
}

int test_stream_observations() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true, true, true);
    const Json role = parse_sse(stream.start());
    failures += check(!role.contains("timings") && !role.contains("prompt_progress"),
                      "transport role chunk precedes Engine observations");

    stream.note_start(
        ninfer::GenerationStart{.prompt = {.prompt_tokens = 32}, .reused_prompt_tokens = 12});
    const Json initial = parse_sse(stream.initial_prompt_progress());
    failures +=
        check(initial["choices"][0]["delta"].empty() && initial["prompt_progress"]["total"] == 32 &&
                  initial["prompt_progress"]["cache"] == 12 &&
                  initial["prompt_progress"]["processed"] == 12 &&
                  initial["prompt_progress"]["time_ms"] == 0,
              "initial prompt progress begins at the admitted cache frontier");
    failures += check(initial["timings"]["prompt_n"] == 20 && initial["timings"]["prompt_ms"] == 0.0 &&
                          initial["timings"]["prompt_per_second"] == 0.0 &&
                          initial["timings"]["predicted_n"] == 0,
                      "initial prompt progress carries a zero prompt-only timing snapshot");

    const Json middle = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 20,
        .elapsed_ns              = 57000000,
    }));
    failures += check(middle["prompt_progress"]["processed"] == 20 &&
                          middle["prompt_progress"]["time_ms"] == 57,
                      "prompt progress exposes a cumulative completed frontier");
    failures += check(middle["timings"]["prompt_ms"] == 57.0 &&
                          middle["timings"]["prompt_per_token_ms"] == 7.125 &&
                          middle["timings"]["prompt_per_second"] == 1000.0 * 8.0 / 57.0 &&
                          middle["timings"]["predicted_n"] == 0,
                      "prompt progress timings observe the processed suffix so far");
    const Json complete = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 32,
        .elapsed_ns              = 100000000,
    }));
    failures +=
        check(complete["prompt_progress"]["processed"] == complete["prompt_progress"]["total"],
              "final prompt progress reaches the complete prompt");
    failures += check(complete["timings"]["prompt_per_token_ms"] == 5.0 &&
                          complete["timings"]["prompt_per_second"] == 200.0,
                      "final prompt progress timings span the complete prompt suffix");

    OpenAIChatStream plain(identity(), false, false, true);
    (void)plain.start();
    plain.note_start(
        ninfer::GenerationStart{.prompt = {.prompt_tokens = 8}, .reused_prompt_tokens = 0});
    const Json plain_progress = parse_sse(plain.initial_prompt_progress());
    failures += check(!plain_progress.contains("timings"),
                      "prompt progress omits timings without timings_per_token");

    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens = 1, .prompt_elapsed_ns = 110000000, .generation_elapsed_ns = 0});
    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens      = 3,
        .prompt_elapsed_ns     = 110000000,
        .generation_elapsed_ns = 20000000,
    });
    const Json content = parse_sse(stream.content_delta("answer"));
    failures +=
        check(content["timings"]["prompt_n"] == 20 && content["timings"]["predicted_n"] == 3 &&
                  content["timings"]["predicted_per_second"] == 100.0,
              "visible output uses the latest independent commit observation");

    GenerationOutcome outcome = sample_outcome();
    outcome.reasoning.clear();
    const std::vector<std::string> terminal = stream.finish(outcome);
    failures += check(parse_sse(terminal[1])["timings"]["predicted_n"] == 7,
                      "terminal usage replaces live timing with exact final accounting");
    return failures;
}

int test_common_objects() {
    int failures      = 0;
    Json unbounded          = base_request();
    unbounded["max_tokens"] = -1;
    failures += check(parse(unbounded).generation.max_tokens == kUnboundedOutputTokens,
                      "max_tokens -1 leaves the output bounded only by context");
    unbounded["max_tokens"] = -2;
    failures += check(api_error([&] { (void)parse(unbounded); }).param == "max_tokens",
                      "a negative max_tokens other than -1 is rejected");
    Json without_model = base_request();
    without_model.erase("model");
    failures += check(parse(without_model).model.empty(),
                      "Chat Completions accepts a request that omits the model");
    without_model["model"] = 42;
    failures += check(api_error([&] { (void)parse(without_model); }).param == "model",
                      "a model that is present must still be a non-empty string");
    const ninfer::ModelMetadata metadata{
        .model_id       = "qwen3.6-27b",
        .weights_id     = "groupwise-int",
        .vocab_size     = 248077,
        .embedding_size = 5120,
        .native_context = 262144,
        .parameters     = 27000000000ULL,
        .weight_bytes   = 17000000000ULL,
    };
    const Json models = Json::parse(make_models_list("qwen", 7, 240000, true, metadata));
    failures +=
        check(models["data"][0]["id"] == "qwen" && models["data"][0]["max_model_len"] == 240000 &&
                  models["data"][0]["context_window"] == 240000 &&
                  models["data"][0]["modalities"]["vision"] == true,
              "models list advertises the configured context limit and vision");
    const Json list_meta = models["data"][0]["meta"];
    failures += check(list_meta["n_vocab"] == 248077 && list_meta["n_ctx"] == 240000 &&
                          list_meta["n_ctx_train"] == 262144 && list_meta["n_embd"] == 5120 &&
                          list_meta["n_params"] == 27000000000ULL &&
                          list_meta["size"] == 17000000000ULL &&
                          list_meta["ftype"] == "groupwise-int",
                      "models list exposes the llama.cpp-compatible model meta");
    failures += check(models["data"][0]["status"]["value"] == "loaded",
                      "models list marks the resident model as loaded");
    const Json index = make_api_index("qwen");
    failures += check(index["object"] == "api_base" && index["model"] == "qwen",
                      "the API base index names the configured model alias");
    bool lists_chat = false;
    for (const auto& endpoint : index["endpoints"]) {
        lists_chat = lists_chat || (endpoint["method"] == "POST" &&
                                    endpoint["path"] == "/v1/chat/completions");
    }
    failures += check(lists_chat, "the API base index lists Chat Completions");
    const Json model = Json::parse(make_model_object("qwen", 7, 240000, false, metadata));
    failures += check(model["max_model_len"] == 240000 && model["context_window"] == 240000 &&
                          model["modalities"]["vision"] == false,
                      "model lookup advertises the configured context limit and vision");
    failures += check(model["meta"]["n_embd"] == 5120 && model["meta"]["n_ctx_train"] == 262144 &&
                          model["meta"]["ftype"] == "groupwise-int",
                      "model lookup exposes the llama.cpp-compatible model meta");
    const Json error = Json::parse(make_error_body(
        ApiError{.status = 400, .message = "bad", .param = "messages", .code = "invalid"}));
    failures += check(error["error"]["param"] == "messages" && error["error"]["code"] == "invalid",
                      "OpenAI common error shape remains stable");
    return failures;
}

} // namespace

int test_assistant_continuation_mode() {
    int failures = 0;
    const Json trailing_assistant = Json{
        {"model", "qwen"},
        {"messages", Json::array({Json{{"role", "system"}, {"content", "system prompt"}},
                                  Json{{"role", "user"}, {"content", "question"}},
                                  Json{{"role", "assistant"}, {"content", "partial answer..."}}})}};
    // Without --assistant-prefill a trailing assistant message is history, and a new turn opens.
    failures += check(parse(trailing_assistant).generation.continuation ==
                          ninfer::PromptContinuationMode::NewAssistantTurn,
                      "a trailing assistant message became a prefill without --assistant-prefill");

    RequestLimits prefill_limits    = limits();
    prefill_limits.assistant_prefill = true;
    const auto parse_prefill        = [&](const Json& body) {
        return parse_chat_completion_request(body, prefill_limits);
    };
    Json ending_with_user = trailing_assistant;
    ending_with_user["messages"].push_back(Json{{"role", "user"}, {"content", "follow-up"}});
    failures += check(parse_prefill(ending_with_user).generation.continuation ==
                          ninfer::PromptContinuationMode::NewAssistantTurn,
                      "a conversation ending with a user message became a prefill");

    // Thinking left to the server default resolves enabled, and a prefill cannot open a
    // thinking turn, so only an explicit disable continues.
    const auto continued = parse_prefill(trailing_assistant);
    failures += check(continued.generation.continuation ==
                          ninfer::PromptContinuationMode::ContinueFinalAssistant,
                      "--assistant-prefill did not continue a trailing assistant message");
    failures += check(api_error([&] { (void)semantics(continued.generation); }).code ==
                          "assistant_prefill_not_supported",
                      "a prefill with default thinking was not refused");
    Json with_call = trailing_assistant;
    with_call["messages"].back()["tool_calls"] = Json::array(
        {Json{{"id", "call_1"},
              {"type", "function"},
              {"function", Json{{"name", "lookup"}, {"arguments", "{}"}}}}});
    failures += check(api_error([&] { (void)parse_prefill(with_call); }).code ==
                          "assistant_prefill_not_supported",
                      "a prefill carrying a tool call was accepted");
    Json no_thinking               = trailing_assistant;
    no_thinking["enable_thinking"] = false;
    const auto disabled            = parse_prefill(no_thinking);
    failures += check(semantics(disabled.generation).enable_thinking == false &&
                          prompt(disabled.generation).options.continuation ==
                              ninfer::PromptContinuationMode::ContinueFinalAssistant,
                      "a prefill with thinking disabled did not reach the prompt as a continuation");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_request_envelope_and_sampling();
    failures += test_structured_output();
    failures += test_standard_field_policy();
    failures += test_constrained_decoding_extensions();
    failures += test_tools();
    failures += test_tool_name_diagnostics();
    failures += test_messages_and_media();
    failures += test_reasoning_and_extensions();
    failures += test_stops_and_ranges();
    failures += test_aggregate_response();
    failures += test_stream_response();
    failures += test_stream_observations();
    failures += test_common_objects();
    failures += test_assistant_continuation_mode();
    if (failures == 0) { std::cout << "OpenAI Chat protocol tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
