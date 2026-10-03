// Context-lookup drafting is exact by construction -- the verify step rejects a wrong guess -- so
// what these checks defend is not correctness of output but the properties the round economics
// depend on: that a miss reports zero rather than a stale or partial draft, that the *most recent*
// occurrence is the one continued, and that a proposal never runs off the end of the ledger.

#include "models/qwen3_5/program/speculative/lookup_draft.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::qwen3_5;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void continues_the_earlier_occurrence() {
    const std::vector<TokenId> ledger{1, 2, 3, 4, 1, 2};
    TokenId out[8]{};
    // The earlier "1 2" was followed by "3 4", then by the tail itself.
    require(lookup_draft(ledger, 2, 4, out) == 4, "expected four drafts");
    require(out[0] == 3 && out[1] == 4 && out[2] == 1 && out[3] == 2, "wrong continuation");
}

void a_miss_reports_zero() {
    const std::vector<TokenId> ledger{1, 2, 3, 4, 5, 6};
    TokenId out[8]{};
    // No earlier "5 6": the round must see zero and fall back to a plain step.
    require(lookup_draft(ledger, 2, 4, out) == 0, "a miss must report zero");
}

void the_most_recent_occurrence_wins() {
    // Two earlier "9 9" runs. The nearer one was followed by 7, the older by 5; in repetitive text
    // the nearer copy is the better guess, so the scan runs backwards.
    const std::vector<TokenId> ledger{9, 9, 5, 9, 9, 7, 0, 9, 9};
    TokenId out[8]{};
    require(lookup_draft(ledger, 2, 2, out) == 2, "expected two drafts");
    require(out[0] == 7 && out[1] == 0, "took an older occurrence over a nearer one");
}

void a_proposal_never_runs_past_the_ledger() {
    const std::vector<TokenId> ledger{4, 5, 6, 4, 5};
    TokenId out[8]{};
    // Only three tokens followed the earlier "4 5", however many were asked for.
    require(lookup_draft(ledger, 2, 8, out) == 3, "clamp to what actually followed");
    require(out[0] == 6 && out[1] == 4 && out[2] == 5, "wrong clamped continuation");
}

void degenerate_arguments_are_refused() {
    const std::vector<TokenId> ledger{1, 2, 3};
    TokenId out[4]{};
    require(lookup_draft(ledger, 0, 4, out) == 0, "a zero n-gram matches everything; refuse it");
    require(lookup_draft(ledger, 2, 0, out) == 0, "no room for drafts");
    require(lookup_draft(ledger, 2, 4, nullptr) == 0, "no destination");
    const std::vector<TokenId> shorter{1, 2};
    require(lookup_draft(shorter, 3, 2, out) == 0, "ledger shorter than the n-gram");
    const std::vector<TokenId> exact{1, 2};
    require(lookup_draft(exact, 2, 2, out) == 0, "nothing precedes the tail to match against");
}

} // namespace

int main() {
    try {
        continues_the_earlier_occurrence();
        a_miss_reports_zero();
        the_most_recent_occurrence_wins();
        a_proposal_never_runs_past_the_ledger();
        degenerate_arguments_are_refused();
        std::printf("lookup draft checks passed\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
