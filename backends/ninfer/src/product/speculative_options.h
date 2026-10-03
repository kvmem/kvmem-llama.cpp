#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// Copy drafting rides along with every drafter unless the command line sets its width. Fifteen
// drafts is the widest a multi-request batch verifies, so the default holds at any concurrency.
inline constexpr std::uint32_t kDefaultNgramDraftTokens = 15;

inline void apply_default_ngram_draft_tokens(SpeculativeOptions& options,
                                             bool width_explicit) noexcept {
    if (!width_explicit && options.backend != SpeculativeBackend::None) {
        options.ngram_draft_tokens = kDefaultNgramDraftTokens;
    }
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    if (options.ngram_archive_bytes != 0 &&
        (options.ngram_draft_tokens == 0 || options.ngram_session_bytes < (1ULL << 20) ||
         options.ngram_session_bytes > options.ngram_archive_bytes)) {
        throw std::invalid_argument("ngram archive requires ngram drafting and session capacity "
                                    "between 1 MiB and total archive capacity");
    }
    if (options.ngram_draft_tokens != 0 &&
        ((options.backend != SpeculativeBackend::DFlash2 &&
          options.backend != SpeculativeBackend::DFlash &&
          options.backend != SpeculativeBackend::Mtp) ||
         options.ngram_draft_tokens > 63 || options.ngram_min_match < 4 ||
         options.ngram_min_match > 64)) {
        throw std::invalid_argument(
            "ngram requires --spec mtp|dflash|dflash2, drafts 1..63 and match 4..64");
    }
    if (options.mtp_attention_window != 0 &&
        (options.backend != SpeculativeBackend::Mtp ||
         options.mtp_attention_window < options.draft_tokens + 1)) {
        throw std::invalid_argument(
            "--mtp-attention-window requires --spec mtp and at least --draft-tokens + 1 keys");
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        if (options.mtp_policy != MtpDraftPolicy::Fixed) {
            throw std::invalid_argument("--adaptive-mtp requires --spec mtp");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        if (options.mtp_policy != MtpDraftPolicy::Fixed) {
            throw std::invalid_argument("--adaptive-mtp requires --spec mtp");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        if (options.mtp_policy != MtpDraftPolicy::Fixed) {
            throw std::invalid_argument("--adaptive-mtp requires --spec mtp");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
