#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/ngram.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/registry.h"
#include "runtime/contract/request.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5 {

[[nodiscard]] ModelSamplingDefaults default_sampling(Architecture architecture);

struct FrontendOptions {
    std::filesystem::path chat_template_path;
    Architecture architecture              = Architecture::Qwen3_5;
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    // Largest merged-token count one media item may occupy; larger media is downscaled at
    // preprocessing instead of being rejected. Zero leaves the artifact's pixel ceilings.
    std::uint32_t vision_max_merged_tokens = 16384;
    // End-of-thinking message injected when a request hits its thinking budget. Empty
    // preserves the built-in canonical control suffix; a message lacking the canonical
    // </think> close serialization gets it appended at startup.
    std::string thinking_budget_message;
    // Derive proposal-only ngram sources from tool results, and bind requests to the
    // cross-request archive.
    bool ngram_sources_enabled = false;
    bool ngram_archive_enabled = false;
    // Engine-automatic long anchors: when nonzero, preparation synthesizes PrivateLongAnchor
    // opportunities at up to this many message boundaries, walking back from the prompt end on a
    // geometrically widening grid, so a later history rewrite diverging there resumes from the
    // retained anchor instead of root. The Engine may raise it after startup through
    // Frontend::publish_long_anchor_limit.
    std::uint32_t automatic_long_anchors = 0;
    // Minimum token gap between consecutive automatic anchors (and between the prompt end and the
    // first one), doubling per anchor; 0 anchors every one of the last boundaries.
    std::uint32_t long_anchor_min_spacing_tokens = 0;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::unique_ptr<NgramArchive::Request> bind_ngram(NgramArchive& archive,
                                                                    const NgramSessionHints& hints);

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string token_bytes(TokenId token) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output               = {},
                        const ThinkingControlOptions& thinking    = {},
                        const StructuredOutputOptions& structured = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;
    // Publishes the resolved long-anchor count. Startup builds the frontend before the sequence
    // plan exists, so the Engine hands the host-cache-resolved value to the grid the capture path
    // will create checkpoints for, before any request is prepared.
    void publish_long_anchor_limit(std::uint32_t anchors) noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::models::qwen3_5
