#include "models/qwen3_5/frontend/frontend.h"

#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/media_cache.h"
#include "models/qwen3_5/frontend/processor.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/ngram_sources.h"
#include "text/unicode.h"
#include "text/structured_output.h"
#include <mutex>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = frontend;

constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";
static_assert(kThinkingControl.ends_with(fi::kCanonicalReasoningCloseSerialization));
constexpr double kRescaleFactor = 1.0 / 255.0;
constexpr double kVideoFps      = 2.0;
constexpr int kVideoMinFrames   = 4;
constexpr int kVideoMaxFrames   = 768;

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::int64_t require_integer(const Json& object, std::string_view field,
                             std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an integer");
    }
    return object.at(key).get<std::int64_t>();
}

double require_number(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be a number");
    }
    return object.at(key).get<double>();
}

double number_or_default(const Json& object, std::string_view field, std::string_view resource,
                         double default_value) {
    const std::string key(field);
    return object.contains(key) ? require_number(object, field, resource) : default_value;
}

std::int64_t integer_or_default(const Json& object, std::string_view field,
                                std::string_view resource, std::int64_t default_value) {
    const std::string key(field);
    return object.contains(key) ? require_integer(object, field, resource) : default_value;
}

const Json& require_object(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an object");
    }
    return object.at(key);
}

std::uint64_t positive_u64(std::int64_t value, std::string_view field) {
    if (value <= 0) { throw std::invalid_argument(std::string(field) + " must be positive"); }
    return static_cast<std::uint64_t>(value);
}

void validate_pixel_pipeline(const Json& config, std::string_view resource) {
    for (const char* flag : {"do_resize", "do_rescale", "do_normalize", "do_convert_rgb"}) {
        if (config.contains(flag) && (!config[flag].is_boolean() || !config[flag].get<bool>())) {
            throw std::invalid_argument(std::string(resource) + "." + flag +
                                        " must be enabled for the compiled pixel pipeline");
        }
    }
    if (config.contains("resample") &&
        (!config["resample"].is_number_integer() || config["resample"].get<int>() != 3)) {
        throw std::invalid_argument(std::string(resource) +
                                    ".resample must select bicubic interpolation");
    }
    if (require_integer(config, "patch_size", resource) != 16 ||
        require_integer(config, "temporal_patch_size", resource) != 2 ||
        require_integer(config, "merge_size", resource) != 2) {
        throw std::invalid_argument(std::string(resource) +
                                    " does not match the compiled Vision patch geometry");
    }
    const auto require_half_triplet = [&](std::string_view field) {
        const std::string key(field);
        if (!config.contains(key) || !config.at(key).is_array() || config.at(key).size() != 3) {
            throw std::invalid_argument(std::string(resource) + "." + key +
                                        " must contain three values");
        }
        for (const Json& value : config.at(key)) {
            if (!value.is_number() || value.get<double>() != 0.5) {
                throw std::invalid_argument(std::string(resource) + "." + key +
                                            " does not match the compiled normalization");
            }
        }
    };
    require_half_triplet("image_mean");
    require_half_triplet("image_std");
    if (number_or_default(config, "rescale_factor", resource, kRescaleFactor) != kRescaleFactor) {
        throw std::invalid_argument(std::string(resource) +
                                    ".rescale_factor does not match the compiled normalization");
    }
}

fi::ProcessorOptions processor_options(const FrontendResources& resources) {
    const Json image =
        parse_resource_json(resources.preprocessor_config_json, "preprocessor_config.json");
    const Json video = parse_resource_json(resources.video_preprocessor_config_json,
                                           "video_preprocessor_config.json");
    validate_pixel_pipeline(image, "preprocessor_config.json");
    validate_pixel_pipeline(video, "video_preprocessor_config.json");

    const Json& image_size = require_object(image, "size", "preprocessor_config.json");
    const Json& video_size = require_object(video, "size", "video_preprocessor_config.json");

    fi::ProcessorOptions options;
    options.image_min_pixels =
        positive_u64(require_integer(image_size, "shortest_edge", "preprocessor_config.json.size"),
                     "image shortest_edge");
    options.image_max_pixels =
        positive_u64(require_integer(image_size, "longest_edge", "preprocessor_config.json.size"),
                     "image longest_edge");
    options.video_min_pixels = positive_u64(
        require_integer(video_size, "shortest_edge", "video_preprocessor_config.json.size"),
        "video shortest_edge");
    options.video_max_pixels = positive_u64(
        require_integer(video_size, "longest_edge", "video_preprocessor_config.json.size"),
        "video longest_edge");
    options.video_fps =
        number_or_default(video, "fps", "video_preprocessor_config.json", kVideoFps);
    options.video_min_frames = static_cast<int>(
        integer_or_default(video, "min_frames", "video_preprocessor_config.json", kVideoMinFrames));
    options.video_max_frames = static_cast<int>(
        integer_or_default(video, "max_frames", "video_preprocessor_config.json", kVideoMaxFrames));
    if (options.video_fps != kVideoFps || options.video_min_frames != kVideoMinFrames ||
        options.video_max_frames != kVideoMaxFrames) {
        throw std::invalid_argument(
            "video_preprocessor_config.json does not match registered sampling defaults");
    }

    return options;
}

void validate_tokenizer_config(const FrontendResources& resources) {
    const Json tokenizer_config =
        parse_resource_json(resources.tokenizer_config_json, "tokenizer_config.json");
    if (tokenizer_config.value("add_bos_token", true) ||
        tokenizer_config.value("add_prefix_space", true)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match Qwen3.5 tokenizer prefix semantics");
    }
    if (!tokenizer_config.contains("pad_token") || !tokenizer_config.at("pad_token").is_string() ||
        tokenizer_config.at("pad_token").get<std::string>() != "<|endoftext|>") {
        throw std::invalid_argument(
            "tokenizer_config.json does not use the official <|endoftext|> pad token");
    }
}

fi::CompiledChatTemplate compile_chat_template(const FrontendResources& resources,
                                               const std::filesystem::path& override_path) {
    validate_tokenizer_config(resources);
    nlohmann::ordered_json tokens = nlohmann::ordered_json::object();
    const auto config             = nlohmann::ordered_json::parse(resources.tokenizer_config_json);
    for (const char* key : {"bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                            "cls_token", "mask_token"}) {
        if (!config.contains(key) || config[key].is_null()) continue;
        const auto& value = config[key];
        if (value.is_string())
            tokens[key] = value;
        else if (value.is_object() && value.contains("content"))
            tokens[key] = value.at("content");
    }
    if (config.contains("additional_special_tokens"))
        tokens["additional_special_tokens"] = config["additional_special_tokens"];
    std::string source(resources.chat_template_jinja);
    std::string name = "artifact:chat_template.jinja";
    if (!override_path.empty()) {
        name = override_path.string();
        std::ifstream file(override_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::invalid_argument("cannot read chat template: " + name);
        const auto size = file.tellg();
        if (size <= 0 || size > 16 * 1024 * 1024) {
            throw std::invalid_argument(
                "chat template must be a nonempty UTF-8 source file up to 16 MiB: " + name);
        }
        source.resize(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(source.data(), static_cast<std::streamsize>(source.size()))) {
            throw std::invalid_argument("cannot read chat template: " + name);
        }
    }
    return fi::CompiledChatTemplate::resolve(source, std::move(name), std::move(tokens));
}

[[noreturn]] void throw_processor_error(const fi::ProcessorError& error) {
    switch (error.kind()) {
    case fi::ProcessorErrorKind::BudgetExceeded:
        throw RequestError(RequestErrorKind::MediaBudgetExceeded, error.what());
    case fi::ProcessorErrorKind::ContextLengthExceeded:
        throw RequestError(RequestErrorKind::ContextLengthExceeded, error.what());
    case fi::ProcessorErrorKind::InvalidMedia:
        throw RequestError(RequestErrorKind::InvalidMedia, error.what());
    }
    throw std::logic_error("unknown Qwen3.5 processor error kind");
}

[[noreturn]] void throw_context_length_exceeded(std::uint32_t max_context) {
    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                       "prepared prompt exceeds Engine max_context " + std::to_string(max_context));
}

std::vector<fi::ChatMessage> convert_messages(std::vector<ChatMessage> messages) {
    std::vector<fi::ChatMessage> result;
    result.reserve(messages.size());
    for (ChatMessage& source : messages) {
        fi::ChatMessage target;
        target.role              = source.role;
        target.reasoning_content = std::move(source.reasoning_content);
        target.tool_call_id      = std::move(source.tool_call_id);
        target.tool_calls.reserve(source.tool_calls.size());
        for (ToolCall& call : source.tool_calls) {
            target.tool_calls.push_back(
                fi::ToolCall{.id             = std::move(call.id),
                             .name           = std::move(call.name),
                             .arguments_json = std::move(call.arguments_json)});
        }
        target.parts.reserve(source.parts.size());
        for (MessagePart& part : source.parts) {
            switch (part.kind) {
            case MessagePartKind::Text:
                target.parts.push_back(fi::ChatPart::text_part(std::move(part.text)));
                break;
            case MessagePartKind::Media: {
                if (part.media.bytes.empty()) {
                    throw std::invalid_argument("frontend media input contains no owning bytes");
                }
                fi::MediaData media;
                media.source_name         = std::move(part.media.source_name);
                media.media_type          = std::move(part.media.media_type);
                media.bytes               = std::move(part.media.bytes);
                media.image_resize_policy = part.media.image_resize_policy;
                switch (part.media.kind) {
                case MediaKind::Image:
                    target.parts.push_back(fi::ChatPart::image(std::move(media)));
                    break;
                case MediaKind::Video:
                    target.parts.push_back(fi::ChatPart::video(std::move(media)));
                    break;
                default:
                    throw std::invalid_argument("frontend media kind is invalid");
                }
                break;
            }
            default:
                throw std::invalid_argument("frontend message-part kind is invalid");
            }
        }
        result.push_back(std::move(target));
    }
    return result;
}

fi::ChatRenderOptions render_options(const PromptOptions& options,
                                     std::span<const PromptCacheMarker> cache_markers = {}) {
    fi::ChatRenderOptions rendered{.continuation              = options.continuation,
                                   .enable_thinking           = options.enable_thinking,
                                   .reasoning_effort          = options.reasoning_effort,
                                   .preserve_thinking         = options.preserve_thinking,
                                   .chat_template_kwargs_json = options.chat_template_kwargs_json,
                                   .add_vision_id             = options.add_vision_id,
                                   .tool_jsons                = options.tool_jsons,
                                   .forced_tool_name          = options.forced_tool_name};
    rendered.cache_markers.assign(cache_markers.begin(), cache_markers.end());
    return rendered;
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("prepared prompt must contain at least one token");
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("prepared prompt token count exceeds the target domain");
    }
    return static_cast<std::uint32_t>(count);
}

void assign_text_positions(PreparedPromptData& prompt) {
    const std::size_t count = prompt.token_ids.size();
    prompt.token_types.assign(count, 0);
    prompt.positions.resize(count * 3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t index = 0; index < count; ++index) {
            if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("prepared prompt position exceeds int32 range");
            }
            prompt.positions[axis * count + index] = static_cast<std::int32_t>(index);
        }
    }
    prompt.rope_delta = 0;
}

VisionItem convert_vision_item(fi::VisionItem item) {
    VisionItem result;
    result.modality =
        item.modality == fi::Modality::Image ? PromptModality::Image : PromptModality::Video;
    result.grid = VisionGrid{.temporal = item.grid.t, .height = item.grid.h, .width = item.grid.w};
    result.patch_begin    = item.patch_begin;
    result.patch_count    = item.patch_count;
    result.content_digest = item.content_digest;
    result.timestamps     = std::move(item.timestamps);
    result.token_spans.reserve(item.token_spans.size());
    for (const fi::TokenSpan span : item.token_spans) {
        result.token_spans.push_back(TokenSpan{.begin = span.begin, .count = span.count});
    }
    return result;
}

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                        std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)ninfer::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

std::optional<std::uint32_t> leading_instruction_boundary(std::span<const ChatRole> roles) {
    std::uint32_t leading = 0;
    while (leading < roles.size() &&
           (roles[leading] == ChatRole::System || roles[leading] == ChatRole::Developer)) {
        ++leading;
    }
    return leading != 0 ? std::optional<std::uint32_t>(leading) : std::nullopt;
}

bool exact_vision_frontier(std::uint32_t frontier, std::span<const VisionItem> items) {
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (last.count > std::numeric_limits<std::size_t>::max() - last.begin) { return false; }
        const std::size_t end = last.begin + last.count;
        if (first.begin < frontier && frontier < end) { return false; }
    }
    return true;
}

PreparedContextCache prepare_context_cache(
    ContextCacheHints hints, std::size_t message_count,
    std::span<const std::optional<std::uint32_t>> message_boundaries,
    std::span<const PromptCacheMarker> rendered_markers,
    std::span<const std::optional<std::uint32_t>> cache_boundaries,
    std::span<const VisionItem> vision_items, std::optional<std::size_t> engine_tool_marker_index,
    std::optional<std::uint32_t> leading_boundary, std::uint32_t full_prompt_frontier,
    std::uint32_t automatic_long_anchors, std::uint32_t long_anchor_min_spacing) {
    if (hints.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    }
    if (cache_boundaries.size() != rendered_markers.size()) {
        throw std::logic_error("rendered cache marker count changed during preparation");
    }

    PreparedContextCache out;
    if (hints.session_key) {
        if (hints.session_key->empty() || hints.session_key->size() > kPreparedSessionKeyCapacity) {
            throw std::invalid_argument("context cache session_key must contain 1 to 256 bytes");
        }
        PreparedSessionKey key;
        key.size = static_cast<std::uint16_t>(hints.session_key->size());
        std::copy(hints.session_key->begin(), hints.session_key->end(), key.bytes.begin());
        out.session_key = key;
    }
    switch (hints.retention) {
    case CacheRetentionHint::Default:
        out.retention = out.session_key ? runtime::RetentionClass::LiveSession
                                        : runtime::RetentionClass::RecentPrivate;
        break;
    case CacheRetentionHint::LiveSession:
        if (!out.session_key) {
            throw std::invalid_argument("LiveSession retention requires a session_key");
        }
        out.retention = runtime::RetentionClass::LiveSession;
        break;
    case CacheRetentionHint::Disposable:
        out.retention = runtime::RetentionClass::Disposable;
        break;
    default:
        throw std::invalid_argument("context cache retention hint is invalid");
    }
    out.update_session_index = hints.update_session_index;

    for (const PromptCacheMarker marker : hints.markers) {
        switch (marker.kind) {
        case PromptCacheMarkerKind::SharedStablePrefix:
        case PromptCacheMarkerKind::PrivateLongAnchor:
            break;
        default:
            throw std::invalid_argument("context cache marker kind is invalid");
        }
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count > message_count ||
                marker.leading_instruction_bytes != 0 || marker.after_tool_count != 0 ||
                marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache message marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            if (marker.after_message_count == 0 || marker.after_message_count > message_count ||
                marker.after_message_part_count == 0 || marker.leading_instruction_bytes != 0 ||
                marker.after_tool_count != 0) {
                throw std::invalid_argument("context cache message-part marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (marker.after_message_count != 0 || marker.leading_instruction_bytes == 0 ||
                marker.after_tool_count != 0 || marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache leading-instruction marker is invalid");
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_message_count != 0 || marker.leading_instruction_bytes != 0 ||
                marker.after_tool_count == 0 || marker.after_message_part_count != 0) {
                throw std::invalid_argument("context cache tool marker is invalid");
            }
            break;
        default:
            throw std::invalid_argument("context cache marker location is invalid");
        }
        if (marker.kind == PromptCacheMarkerKind::SharedStablePrefix &&
            marker.evidence == SharedCandidateEvidence::None) {
            throw std::invalid_argument("shared context cache marker evidence is empty");
        }
    }

    out.opportunities.reserve(7U + (hints.allow_engine_prefix_grid ? kPrefixGridCandidates : 0U) +
                              automatic_long_anchors);
    const auto add_opportunity = [&](PromptCacheMarkerKind kind, SharedCandidateEvidence evidence,
                                     std::uint32_t frontier, std::uint32_t input_order) {
        if (frontier == 0 || !exact_vision_frontier(frontier, vision_items)) { return; }
        const auto duplicate = std::find_if(
            out.opportunities.begin(), out.opportunities.end(), [&](const auto& existing) {
                return existing.kind == kind && existing.frontier == frontier;
            });
        if (duplicate == out.opportunities.end()) {
            out.opportunities.push_back(PreparedCacheOpportunity{.kind        = kind,
                                                                 .evidence    = evidence,
                                                                 .frontier    = frontier,
                                                                 .input_order = input_order});
        } else if (kind == PromptCacheMarkerKind::SharedStablePrefix) {
            duplicate->evidence |= evidence;
        }
    };

    for (std::size_t index = 0; index < hints.markers.size(); ++index) {
        const PromptCacheMarker marker = hints.markers[index];
        std::optional<std::uint32_t> resolved;
        if (marker.location == PromptCacheMarkerLocation::MessageBoundary) {
            if (marker.after_message_count < message_boundaries.size()) {
                resolved = message_boundaries[marker.after_message_count];
            }
        } else {
            if (index < cache_boundaries.size()) { resolved = cache_boundaries[index]; }
        }
        if (!resolved) { continue; }
        add_opportunity(marker.kind, marker.evidence, *resolved, static_cast<std::uint32_t>(index));
    }

    std::uint32_t engine_order = static_cast<std::uint32_t>(hints.markers.size());
    if (hints.allow_engine_automatic_shared_prefixes) {
        if (engine_tool_marker_index && *engine_tool_marker_index < cache_boundaries.size() &&
            cache_boundaries[*engine_tool_marker_index]) {
            add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                            SharedCandidateEvidence::EngineStructural,
                            *cache_boundaries[*engine_tool_marker_index], engine_order++);
        }
        if (leading_boundary && *leading_boundary < message_boundaries.size() &&
            message_boundaries[*leading_boundary]) {
            add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                            SharedCandidateEvidence::EngineStructural,
                            *message_boundaries[*leading_boundary], engine_order++);
        }
        add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                        SharedCandidateEvidence::EngineObserved, full_prompt_frontier,
                        engine_order++);
    }
    if (hints.allow_engine_prefix_grid) {
        // Structural boundaries only expose a prefix where the prompt's own shape happens to put
        // one. Two requests that merely start with the same long span - the same pasted document,
        // the same few-shot preamble inside a single user message - share no boundary at all, so
        // neither ever proposes a frontier the other could match.
        //
        // The grid supplies that missing agreement. Frontiers are placed at absolute multiples of a
        // page-sized stride rather than at positions measured from the end of the prompt, so two
        // prompts of unrelated lengths still name the same frontier wherever they still agree. The
        // stride doubles until the grid fits in kPrefixGridCandidates points, which keeps the count
        // bounded and keeps a coarser grid a subset of a finer one - prompts that pick different
        // strides still meet on the multiples of the larger.
        //
        // Every grid point carries EngineObserved and nothing else. That evidence is neither
        // "declared" nor "surplus", so the resource manager will not spend a vacant shared slot on
        // one speculatively; a grid frontier is only ever materialized once two distinct reuse
        // domains have independently proposed it, which is exactly the signal that two callers
        // share that prefix.
        std::uint32_t stride = kPrefixGridPageTokens;
        while (full_prompt_frontier / stride > kPrefixGridCandidates) { stride *= 2U; }
        for (std::uint32_t frontier = stride; frontier <= full_prompt_frontier; frontier += stride) {
            add_opportunity(PromptCacheMarkerKind::SharedStablePrefix,
                            SharedCandidateEvidence::EngineObserved, frontier, engine_order++);
        }
    }
    // Automatic private long anchors: message boundaries are where history rewrites diverge, so a
    // request matching an earlier boundary resumes from the retained anchor instead of root.
    // Walking back from the prompt end, a boundary joins the grid only when it lies at least the
    // current spacing below the previous grid point, and the spacing doubles with every point:
    // short tool-loop turns would otherwise put every anchor next to the endpoint, each costing a
    // prefill split and a StateImage for almost no coverage, while deep history stayed unanchored.
    // A boundary already carried as an opportunity (an explicit marker or an engine shared
    // candidate) keeps it and still occupies its grid position; the full prompt frontier is the
    // session endpoint's.
    std::uint32_t grid_positions = 0;
    std::uint32_t grid_point     = full_prompt_frontier;
    std::uint64_t spacing        = long_anchor_min_spacing;
    for (std::size_t index = message_boundaries.size();
         index > 0 && grid_positions < automatic_long_anchors; --index) {
        const std::optional<std::uint32_t> boundary = message_boundaries[index - 1];
        if (!boundary || *boundary == 0 || *boundary >= full_prompt_frontier) { continue; }
        if (static_cast<std::uint64_t>(grid_point - *boundary) < spacing) { continue; }
        ++grid_positions;
        grid_point = *boundary;
        spacing = std::min<std::uint64_t>(spacing * 2U, std::numeric_limits<std::uint32_t>::max());
        if (std::any_of(out.opportunities.begin(), out.opportunities.end(),
                        [&](const auto& existing) { return existing.frontier == *boundary; })) {
            continue;
        }
        add_opportunity(PromptCacheMarkerKind::PrivateLongAnchor,
                        SharedCandidateEvidence::EngineStructural, *boundary, engine_order++);
    }
    return out;
}

// Boundary facts for hybrid prefix-cache taps. The Program floors each frontier to a page
// boundary and applies its own priority, spacing and budget rules; this only reports where the
// rendered prompt has reusable structure.
PreparedTapHints prepare_tap_hints(const ContextCacheHints& hints,
                                   std::span<const std::optional<std::uint32_t>> message_boundaries,
                                   std::span<const std::optional<std::uint32_t>> cache_boundaries,
                                   std::optional<std::size_t> engine_tool_marker_index,
                                   std::optional<std::uint32_t> leading_boundary,
                                   const std::optional<RewriteCheckpointSpec>& rewrite_checkpoint) {
    using runtime::prefix_cache::TapHint;
    using runtime::prefix_cache::TapHintKind;
    // Protocol write-policy hints (OpenAI explicit mode, Anthropic cache_control) govern Legacy
    // shared-prefix publication. Hybrid taps are cheap and content-deduplicated, so explicit
    // markers only add taps; automatic placement is never suppressed.
    PreparedTapHints out;
    out.hints.reserve(message_boundaries.size() + hints.markers.size() + 3U);
    const auto add = [&](std::optional<std::uint32_t> frontier, TapHintKind kind) {
        if (frontier && *frontier != 0) { out.hints.push_back(TapHint{*frontier, kind}); }
    };
    for (std::size_t index = 0; index < hints.markers.size(); ++index) {
        const PromptCacheMarker& marker = hints.markers[index];
        // Only a client-named breakpoint is honored unconditionally; protocol-automatic markers
        // (OpenAI default caching, Anthropic automatic cache_control) are ordinary structural
        // boundaries that compete with the Engine's own.
        const TapHintKind kind = has_shared_candidate_evidence(
                                     marker.evidence, SharedCandidateEvidence::ExplicitBoundary)
                                     ? TapHintKind::Explicit
                                     : TapHintKind::Structural;
        if (marker.location == PromptCacheMarkerLocation::MessageBoundary) {
            if (marker.after_message_count < message_boundaries.size()) {
                add(message_boundaries[marker.after_message_count], kind);
            }
        } else if (index < cache_boundaries.size()) {
            add(cache_boundaries[index], kind);
        }
    }
    if (engine_tool_marker_index && *engine_tool_marker_index < cache_boundaries.size()) {
        add(cache_boundaries[*engine_tool_marker_index], TapHintKind::Structural);
    }
    if (leading_boundary && *leading_boundary < message_boundaries.size()) {
        add(message_boundaries[*leading_boundary], TapHintKind::Structural);
    }
    if (rewrite_checkpoint) { add(rewrite_checkpoint->frontier, TapHintKind::GenerationOpener); }
    for (const std::optional<std::uint32_t> boundary : message_boundaries) {
        add(boundary, TapHintKind::MessageBoundary);
    }
    return out;
}

} // namespace

ModelSamplingDefaults default_sampling(Architecture architecture) {
    ModelSamplingDefaults sampling;
    sampling.thinking     = {.temperature      = 1.0F,
                             .top_k            = 20,
                             .top_p            = 0.95F,
                             .min_p            = 0.0F,
                             .presence_penalty = architecture == Architecture::Qwen3_5Moe ? 1.5F : 0.0F,
                             .frequency_penalty = 0.0F};
    sampling.non_thinking = {.temperature       = 0.7F,
                             .top_k             = 20,
                             .top_p             = 0.80F,
                             .min_p             = 0.0F,
                             .presence_penalty  = 1.5F,
                             .frequency_penalty = 0.0F};
    sampling.post_thinking             = sampling.thinking;
    sampling.post_thinking.temperature = 0.2F;
    return sampling;
}

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, FrontendOptions options)
        : chat_template(compile_chat_template(resources, options.chat_template_path)),
          tokenizer(resources.tokenizer),
          processor(options.vision_enabled ? processor_options(resources) : fi::ProcessorOptions{}),
          vision_enabled(options.vision_enabled), max_context(options.max_context),
          automatic_long_anchors(options.automatic_long_anchors),
          long_anchor_min_spacing_tokens(options.long_anchor_min_spacing_tokens) {
        fi::bound_merged_tokens(processor, options.vision_max_merged_tokens);
        if (options.max_context == 0) {
            throw std::invalid_argument("frontend max_context must be nonzero");
        }
        const std::uint64_t vision_tokens =
            std::min<std::uint64_t>(options.max_context, kMaximumPromptVisionTokens);
        processor.max_vision_tokens = vision_tokens;
        processor.max_raw_patches   = vision_tokens * kRawPatchesPerVisionToken;
        if (vision_enabled) {
            const std::uint64_t minimum_live =
                processor.max_raw_patches * kPreparedVisionPatchFeatures * sizeof(std::uint16_t);
            if (minimum_live > options.media_live_bytes) {
                throw std::invalid_argument(
                    "media live-byte capacity cannot hold the maximum supported Vision prompt");
            }
            media_cache = std::make_shared<fi::MediaPreprocessCache>(
                options.media_cache_bytes, options.media_live_bytes,
                options.media_preprocess_threads, static_cast<std::size_t>(minimum_live));
        }
        if (!tokenizer || resources.public_token_count != tokenizer->vocab_size()) {
            throw std::invalid_argument(
                "Frontend requires the parsed model tokenizer and public token domain");
        }
        ngram_sources_enabled = options.ngram_sources_enabled;
        ngram_archive_enabled = options.ngram_archive_enabled;
        if (ngram_archive_enabled) {
            ngram_think_open  = tokenizer->encode("<think>");
            ngram_think_close = tokenizer->encode("</think>");
        }
        if (ngram_sources_enabled) {
            for (int id = 0; id < static_cast<int>(tokenizer->vocab_size()); ++id) {
                if (tokenizer->is_special_token(id)) { ngram_boundaries.push_back(id); }
            }
        }
        sampling = default_sampling(options.architecture);
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
        std::string control_suffix(kThinkingControl);
        const char* control_name = "canonical thinking control suffix";
        if (!options.thinking_budget_message.empty()) {
            control_name = "thinking budget message";
            control_suffix = options.thinking_budget_message;
            if (!control_suffix.ends_with(fi::kCanonicalReasoningCloseSerialization)) {
                control_suffix += fi::kCanonicalReasoningCloseSerialization;
            }
        }
        std::vector<TokenId> encoded = tokenizer->encode(control_suffix);
        if (encoded.empty()) {
            throw std::invalid_argument(std::string("Qwen tokenizer cannot encode the ") +
                                        control_name);
        }
        const std::string exact =
            tokenizer->decode(encoded, fi::DecodeOptions{.skip_special_tokens = false});
        const std::string presented =
            tokenizer->decode(encoded, fi::DecodeOptions{.skip_special_tokens = true});
        if (exact != control_suffix || presented.find(kThinkClose) == std::string::npos) {
            throw std::invalid_argument(std::string("Qwen tokenizer cannot present the ") +
                                        control_name);
        }
        for (const TokenId token : encoded) {
            if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), token) !=
                defaults.token_ids.end()) {
                throw std::invalid_argument(std::string(control_name) +
                                            " contains a default terminal token");
            }
        }
        thinking_control_tokens = std::make_shared<const std::vector<TokenId>>(std::move(encoded));
    }

    mutable std::mutex grammar_mutex;
    mutable std::unique_ptr<text::StructuredCompiler> grammar_compiler;
    fi::CompiledChatTemplate chat_template;
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    fi::ProcessorOptions processor;
    std::shared_ptr<fi::MediaPreprocessCache> media_cache;
    StopPolicy defaults;
    ModelSamplingDefaults sampling;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool ngram_sources_enabled = false;
    bool ngram_archive_enabled = false;
    std::vector<TokenId> ngram_think_open, ngram_think_close;
    std::vector<TokenId> ngram_boundaries;
    bool vision_enabled                          = true;
    std::uint32_t max_context                    = 0;
    // Startup-fixed, published once through Frontend::publish_long_anchor_limit while the Engine
    // is still single-threaded; atomic so the request threads that read it see the published value
    // without a data race on a const shared implementation.
    mutable std::atomic<std::uint32_t> automatic_long_anchors{0};
    std::uint32_t long_anchor_min_spacing_tokens = 0;
};

std::span<const std::int32_t> PreparedPromptData::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != token_ids.size() * 3) {
        throw std::out_of_range("invalid prepared-prompt position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * token_ids.size(), token_ids.size());
}

PreparedPrompt::PreparedPrompt() noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr) { throw std::logic_error("prepared prompt is empty"); }
    return PromptSummary{.starts_in_reasoning = data_->starts_in_reasoning,
                         .prompt_tokens       = checked_token_count(data_->token_ids.size()),
                         .has_media           = data_->has_media()};
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    if (data_ == nullptr) { return {}; }
    const PrepareStats& stats = data_->prepare;
    return PromptPreparationStats{
        .seconds                       = stats.seconds,
        .media_preprocess_seconds      = stats.media_preprocess_seconds,
        .media_preprocess_work_seconds = stats.media_preprocess_work_seconds,
        .tokenize_seconds              = stats.tokenize_seconds,
        .media_items                   = stats.media_items,
        .media_bytes                   = stats.media_bytes,
        .raw_patches                   = stats.raw_patches,
        .vision_tokens                 = stats.vision_tokens,
        .patch_bytes                   = stats.patch_bytes,
        .media_cache_hits              = stats.media_cache_hits,
        .media_cache_misses            = stats.media_cache_misses,
        .media_singleflight_waits      = stats.media_singleflight_waits,
        .built_patch_bytes             = stats.built_patch_bytes,
        .reused_patch_bytes            = stats.reused_patch_bytes,
    };
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept {
    return impl_->sampling;
}

void Frontend::publish_long_anchor_limit(std::uint32_t anchors) noexcept {
    // Only a frontend that anchors automatically offers the grid the resolved count sizes.
    if (impl_->automatic_long_anchors.load(std::memory_order_relaxed) == 0) { return; }
    impl_->automatic_long_anchors.store(anchors, std::memory_order_relaxed);
}

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, options));
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

std::unique_ptr<NgramArchive::Request> PreparedPrompt::bind_ngram(NgramArchive& archive,
                                                                  const NgramSessionHints& hints) {
    if (!data_) { return {}; }
    data_->ngram_snapshot.reset();
    if (hints.key.empty()) { return {}; }
    if (hints.reset) { archive.clear(hints.key); }
    // Bounded proposal-only views. The archive owns copied/indexed spans; it never
    // changes this prompt, its token positions, or its prefix-cache identity.
    std::array<NgramSourceView, kNgramRequestSourceCapacity> sources;
    std::size_t count = 0;
    for (int priority = static_cast<int>(NgramSourceKind::Tool); priority >= 0; --priority) {
        if (priority == static_cast<int>(NgramSourceKind::Tool)) {
            for (const auto& source : data_->ngram_sources) {
                if (count == sources.size()) { break; }
                sources[count++] = {source, NgramSourceKind::Tool};
            }
        }
        for (const auto& source : data_->ngram_archive_sources) {
            if (count == sources.size()) { break; }
            if (static_cast<int>(source.kind) == priority) { sources[count++] = source; }
        }
    }
    auto request          = archive.begin(hints.key, std::span(sources).first(count),
                                          data_->ngram_boundaries, hints.parent, hints.parent_generation);
    data_->ngram_snapshot = request ? request->snapshot() : nullptr;
    return request;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data = std::move(prompt.data_);
    return std::move(*data);
}

const PreparedPromptData& FrontendTestAccess::inspect(const PreparedPrompt& prompt) {
    return PreparedPromptAccess::view(prompt);
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    fi::check_preparation_control(control);
    const auto start              = Clock::now();
    const PromptOptions options   = input.options;
    ContextCacheHints cache_hints = std::move(input.context_cache);
    if (cache_hints.markers.size() > kMaximumExplicitPromptCacheMarkers) {
        throw std::invalid_argument("PromptInput supports at most four explicit cache markers");
    }
    std::vector<ChatRole> message_roles;
    message_roles.reserve(input.messages.size());
    for (const ChatMessage& message : input.messages) { message_roles.push_back(message.role); }
    const auto tool_call_output = fi::build_tool_call_output_contract(
        options.tool_jsons, !options.tool_jsons.empty(), options.forced_tool_name);
    const std::optional<std::uint32_t> leading_boundary =
        leading_instruction_boundary(message_roles);
    std::vector<PromptCacheMarker> rendered_markers = cache_hints.markers;
    std::optional<std::size_t> engine_tool_marker_index;
    if (cache_hints.allow_engine_automatic_shared_prefixes && !options.tool_jsons.empty()) {
        engine_tool_marker_index = rendered_markers.size();
        rendered_markers.push_back(PromptCacheMarker{
            .kind             = PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = SharedCandidateEvidence::EngineStructural,
            .location         = PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(options.tool_jsons.size()),
        });
    }
    const std::size_t message_count       = input.messages.size();
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }

    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.tool_call_output    = tool_call_output;
    if (impl_->ngram_sources_enabled) {
        result.ngram_boundaries = impl_->ngram_boundaries;
        std::size_t remaining   = impl_->max_context;
        for (const auto& message : messages) {
            if (remaining == 0) { break; }
            if (message.role != ChatRole::Tool) { continue; }
            for (const auto& part : message.parts) {
                if (remaining == 0) { break; }
                if (part.kind != fi::ChatPartKind::Text) { continue; }
                fi::check_preparation_control(control, "ngram proposal sources");
                for (const auto& source : fi::ngram_numbered_sources(part.text)) {
                    fi::check_preparation_control(control, "ngram proposal sources");
                    if (remaining == 0) { break; }
                    auto tokens = impl_->tokenizer->encode(source, {.max_tokens = remaining});
                    fi::check_preparation_control(control, "ngram proposal sources");
                    remaining -= tokens.size();
                    result.ngram_sources.push_back(std::move(tokens));
                }
            }
        }
    }
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    if (has_media) {
        fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor,
                                impl_->media_cache);
        fi::ProcessedInput processed;
        try {
            processed =
                processor.process(std::move(messages), render_options(options, rendered_markers),
                                  control, impl_->max_context);
        } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
        result.token_ids.assign(processed.input_ids.begin(), processed.input_ids.end());
        result.starts_in_reasoning = processed.starts_in_reasoning;
        result.token_types         = std::move(processed.token_types);
        result.positions           = std::move(processed.positions);
        result.rope_delta          = processed.rope_delta;
        result.media_payloads      = std::move(processed.media_payloads);
        result.vision_items.reserve(processed.vision_items.size());
        for (fi::VisionItem& item : processed.vision_items) {
            result.vision_items.push_back(convert_vision_item(std::move(item)));
        }
        result.prepare.media_items              = processed.stats.media_items;
        result.prepare.media_bytes              = processed.stats.media_bytes;
        result.prepare.raw_patches              = processed.stats.raw_patches;
        result.prepare.vision_tokens            = processed.stats.vision_tokens;
        result.prepare.attention_pairs          = processed.stats.attention_pairs;
        result.prepare.patch_bytes              = processed.stats.patch_bytes;
        result.prepare.media_cache_hits         = processed.stats.media_cache_hits;
        result.prepare.media_cache_misses       = processed.stats.media_cache_misses;
        result.prepare.media_singleflight_waits = processed.stats.media_singleflight_waits;
        result.prepare.built_patch_bytes        = processed.stats.built_patch_bytes;
        result.prepare.reused_patch_bytes       = processed.stats.reused_patch_bytes;
        result.prepare.media_preprocess_seconds = processed.stats.media_preprocess_seconds;
        result.prepare.media_preprocess_work_seconds =
            processed.stats.media_preprocess_work_seconds;
        result.prepare.tokenize_seconds    = processed.stats.tokenize_seconds;
        result.identity.rewrite_checkpoint = processed.rewrite_checkpoint;
        result.identity.rewrite_execution_frontiers =
            std::move(processed.rewrite_execution_frontiers);
        message_boundaries = std::move(processed.message_boundaries);
        cache_boundaries   = std::move(processed.cache_boundaries);
    } else {
        const fi::RenderedChat rendered = impl_->chat_template.render(
            messages, render_options(options, rendered_markers), control);
        result.starts_in_reasoning  = rendered.starts_in_reasoning;
        const auto tokenize_started = Clock::now();
        fi::EncodedChat encoded     = fi::encode_rendered_chat(
            *impl_->tokenizer, rendered, static_cast<std::size_t>(impl_->max_context) + 1U);
        result.prepare.tokenize_seconds =
            std::chrono::duration<double>(Clock::now() - tokenize_started).count();
        fi::check_preparation_control(control, "tokenization");
        if (encoded.input_ids.size() > impl_->max_context) {
            throw_context_length_exceeded(impl_->max_context);
        }
        result.token_ids                   = std::move(encoded.input_ids);
        result.identity.rewrite_checkpoint = encoded.rewrite_checkpoint;
        result.identity.rewrite_execution_frontiers =
            std::move(encoded.rewrite_execution_frontiers);
        message_boundaries = std::move(encoded.message_boundaries);
        cache_boundaries   = std::move(encoded.cache_boundaries);
        assign_text_positions(result);
    }
    (void)checked_token_count(result.token_ids.size());
    if (impl_->ngram_archive_enabled) {
        const std::span<const TokenId> tokens(result.token_ids);
        auto add = [&](std::span<const TokenId> span, NgramSourceKind kind, int priority) {
            if (static_cast<int>(kind) == priority && span.size() >= 4 &&
                result.ngram_archive_sources.size() < kNgramRequestSourceCapacity) {
                result.ngram_archive_sources.push_back({span, kind});
            }
        };
        // Use already-encoded message frontiers, not re-tokenized rendered text.
        // Views move with the owning token vector and never participate in KV identity.
        for (int priority = static_cast<int>(NgramSourceKind::Tool); priority >= 0; --priority) {
            for (std::size_t i = 0; i < message_roles.size() && i + 1 < message_boundaries.size();
                 ++i) {
                fi::check_preparation_control(control, "ngram archive sources");
                const auto role = message_roles[i];
                const auto kind = role == ChatRole::Tool        ? NgramSourceKind::Tool
                                  : role == ChatRole::Assistant ? NgramSourceKind::Generated
                                                                : NgramSourceKind::Text;
                if ((static_cast<int>(kind) != priority &&
                     !(role == ChatRole::Assistant && priority == 0)) ||
                    !message_boundaries[i + 1] || (!message_boundaries[i] && i != 0)) {
                    continue;
                }
                auto begin     = message_boundaries[i].value_or(0);
                const auto end = *message_boundaries[i + 1];
                if (begin >= end || end > tokens.size()) { continue; }
                if (impl_->tokenizer->decode_token_bytes(tokens[begin]) == "<|im_start|>") {
                    while (begin < end) {
                        const auto bytes = impl_->tokenizer->decode_token_bytes(tokens[begin++]);
                        if (bytes.find('\n') != std::string_view::npos) { break; }
                    }
                }
                auto body = tokens.subspan(begin, end - begin);
                if (role == ChatRole::Assistant && !impl_->ngram_think_open.empty() &&
                    !impl_->ngram_think_close.empty()) {
                    const auto& open  = impl_->ngram_think_open;
                    const auto& close = impl_->ngram_think_close;
                    const auto marker =
                        std::search(body.begin(), body.end(), open.begin(), open.end());
                    if (marker != body.end() && marker - body.begin() <= 4) {
                        const auto reasoning = marker + open.size();
                        const auto closure =
                            std::search(reasoning, body.end(), close.begin(), close.end());
                        add({reasoning, closure}, NgramSourceKind::Reasoning, priority);
                        if (closure == body.end()) { continue; }
                        body = {closure + close.size(), body.end()};
                    }
                }
                add(body, kind, priority);
            }
        }
    }
    result.identity.reusable = true;
    result.tap_hints         = prepare_tap_hints(cache_hints, message_boundaries, cache_boundaries,
                                                 engine_tool_marker_index, leading_boundary,
                                                 result.identity.rewrite_checkpoint);
    result.context_cache     = prepare_context_cache(
        std::move(cache_hints), message_count, message_boundaries, rendered_markers,
        cache_boundaries, result.vision_items, engine_tool_marker_index, leading_boundary,
        checked_token_count(result.token_ids.size()),
        impl_->automatic_long_anchors.load(std::memory_order_relaxed),
        impl_->long_anchor_min_spacing_tokens);
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    fi::check_preparation_control(control);
    const PromptOptions options           = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    if (!has_media) {
        const fi::RenderedChat rendered =
            impl_->chat_template.render(messages, render_options(options), control);
        const std::uint32_t count = checked_token_count(
            fi::encode_rendered_chat(*impl_->tokenizer, rendered).input_ids.size());
        fi::check_preparation_control(control, "tokenization");
        return count;
    }

    fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor,
                            impl_->media_cache);
    try {
        return checked_token_count(
            processor.count_tokens(std::move(messages), render_options(options), control));
    } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
}

MediaCacheSummary Frontend::media_cache_summary() const {
    if (impl_ == nullptr || !impl_->media_cache) { return {}; }
    const fi::MediaCacheStats stats = impl_->media_cache->stats();
    return MediaCacheSummary{
        .capacity_bytes      = stats.capacity_bytes,
        .live_capacity_bytes = stats.live_capacity_bytes,
        .retained_bytes      = stats.retained_bytes,
        .live_bytes          = stats.live_bytes,
        .entries             = stats.entries,
        .inflight            = stats.inflight,
        .queued_tasks        = stats.queued_tasks,
        .active_tasks        = stats.active_tasks,
        .preprocess_threads  = stats.preprocess_threads,
        .hits                = stats.hits,
        .misses              = stats.misses,
        .singleflight_waits  = stats.singleflight_waits,
        .evictions           = stats.evictions,
        .oversize_bypasses   = stats.oversize_bypasses,
    };
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (token_ids.size() > impl_->max_context) {
        throw_context_length_exceeded(impl_->max_context);
    }
    (void)checked_token_count(token_ids.size());
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.token_ids           = std::move(token_ids);
    if (impl_->ngram_sources_enabled) { result.ngram_boundaries = impl_->ngram_boundaries; }
    if (impl_->ngram_archive_enabled) {
        result.ngram_archive_sources.push_back({result.token_ids, NgramSourceKind::Text});
    }
    assign_text_positions(result);
    result.identity.reusable                  = allow_prefix_identity;
    result.context_cache.retention            = runtime::RetentionClass::RecentPrivate;
    result.context_cache.update_session_index = false;
    result.prepare.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    return impl_->tokenizer->encode(text);
}

std::string Frontend::token_bytes(TokenId token) const {
    if (impl_ == nullptr) { throw std::logic_error("frontend is empty"); }
    if (token < 0 || static_cast<std::size_t>(token) >= impl_->tokenizer->vocab_size()) {
        throw std::out_of_range("token id is outside the tokenizer vocabulary");
    }
    return std::string(impl_->tokenizer->decode_token_bytes(token, false));
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output,
                                            const ThinkingControlOptions& thinking,
                                            const StructuredOutputOptions& structured) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    std::shared_ptr<text::GrammarState> grammar;
    text::validate_structured_output(structured);
    if (structured.kind != StructuredOutputKind::None) {
        if (!caller_stop.token_ids.empty() || !caller_stop.strings.empty() ||
            !caller_stop.include_model_defaults || output.raw || output.preserve_special_tokens ||
            caller_stop.publish_stop_token) {
            throw std::invalid_argument(
                "structured output requires default stops and decoded text output");
        }
        text::StructuredOutputEnvelope envelope;
        if (prompt.data_->starts_in_reasoning) { envelope.reasoning_close = "</think>"; }
        if (prompt.data_->tool_call_output) {
            envelope.alternative_format =
                fi::structured_tool_call_format(*prompt.data_->tool_call_output);
        }
        std::lock_guard lock(impl_->grammar_mutex);
        if (!impl_->grammar_compiler) {
            std::vector<std::string> vocab(impl_->tokenizer->vocab_size());
            for (std::size_t id = 0; id < vocab.size(); ++id) {
                if (impl_->tokenizer->is_valid_token(id) &&
                    (!impl_->tokenizer->is_special_token(id) ||
                     impl_->tokenizer->decode_token_bytes(id, false) == "</think>")) {
                    vocab[id] = impl_->tokenizer->decode_token_bytes(id, false);
                }
            }
            impl_->grammar_compiler = std::make_unique<text::StructuredCompiler>(
                std::move(vocab), impl_->tokenizer->default_stop_token_ids());
        }
        grammar = impl_->grammar_compiler->compile(structured, envelope);
    }
    if (output.raw) { policy.publish_stop_token = true; }
    return OutputSession(
        impl_->tokenizer, std::move(policy), output, prompt.data_->starts_in_reasoning, thinking,
        impl_->thinking_control_tokens, prompt.data_->tool_call_output, std::move(grammar));
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

} // namespace ninfer::models::qwen3_5
