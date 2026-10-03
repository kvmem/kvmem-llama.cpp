#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "ninfer/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <functional>

namespace ninfer::serve {

// Media acquisition is a product-layer concern. Translation preserves part order
// and asks the caller to turn each wire source into owning bytes before the
// target frontend sees it.
using MediaAcquirer = std::function<ninfer::OwnedMedia(const ContentPart&)>;

struct ResolvedPromptSemantics {
    std::optional<bool> enable_thinking;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::string chat_template_kwargs_json;
};

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& req,
                                                 const ServeOptions& server);

ninfer::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media);

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy.
ninfer::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse);

// A session key for a request that names none (--derive-session-keys): FNV-1a over the text of
// the instruction messages and of the first user message, which stay fixed for the life of a
// conversation. A client compaction rewrites the first user message and so starts a new key.
// A conversation without user text gets none.
std::optional<std::string> derived_session_key(const GenerationRequest& req);

} // namespace ninfer::serve
