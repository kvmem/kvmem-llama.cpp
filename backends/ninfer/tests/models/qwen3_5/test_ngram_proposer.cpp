#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen3_5/frontend/ngram_sources.h"

#include <iostream>
#include <numeric>
#include <random>

using ninfer::models::qwen3_5::detail::NgramProposer;
using ninfer::models::qwen3_5::frontend::ngram_numbered_sources;

static void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

int main() {
    try {
        std::vector<int> source(128);
        std::iota(source.begin(), source.end(), 100);
        NgramProposer proposer(256, 1024);
        proposer.ingest(source);
        require(proposer.propose(source, 0).tokens.empty(), "zero budget cannot propose tokens");
        for (const auto [capacity, buckets] :
             {std::pair<std::size_t, std::size_t>{63, 8}, {64, 0}, {64, 3}}) {
            bool rejected = false;
            try {
                NgramProposer invalid(capacity, buckets);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid index dimensions must be rejected");
        }
        for (unsigned n : {4, 8, 12, 16, 32, 64}) {
            const auto match = proposer.propose(std::span<const int>(source).first(n), 15, 4);
            require(match.matched == n, "backward match extension");
            require(match.tokens == std::vector<int>(source.begin() + n, source.begin() + n + 15),
                    "contiguous source proposal");
        }
        auto match = proposer.propose(std::span<const int>(source).first(126), 15, 12);
        require(match.tokens.size() == 2, "stop at source boundary");
        require(proposer.propose(source, 15, 12).tokens.empty(), "no future source tokens");
        require(proposer.propose(std::span<const int>(source).first(8), 15, 12).tokens.empty(),
                "admission threshold");

        std::uint32_t boundary_checks = 0;
        for (std::uint32_t available = 0; available <= 65; ++available) {
            std::vector<int> bounded_source(64 + available);
            std::iota(bounded_source.begin(), bounded_source.end(), 1000);
            NgramProposer bounded(256, 1024);
            bounded.ingest(bounded_source);
            const auto history = std::span<const int>(bounded_source).first(64);
            require(bounded.propose_for_round(history, 0, 5).tokens.empty(),
                    "zero round budget cannot admit a draft");
            for (std::uint32_t maximum = 1; maximum <= 63; ++maximum) {
                for (std::uint32_t neural = 1; neural <= 15; ++neural) {
                    const auto draft = bounded.propose_for_round(history, maximum, neural);
                    const auto count = draft.tokens.size();
                    require(count <= maximum && count <= available, "round admission budget");
                    if (available >= maximum + 2) {
                        require(count == maximum, "full copy window unnecessarily shortened");
                    } else if (count != 0) {
                        require(count > neural && count + 1 < available,
                                "target bonus reached the source boundary");
                    } else {
                        require(available <= neural + 2, "useful copy tail discarded");
                    }
                    require(std::equal(draft.tokens.begin(), draft.tokens.end(),
                                       bounded_source.begin() + 64),
                            "admission changed source tokens");
                    ++boundary_checks;
                }
            }
        }
        for (const auto maximum : {std::numeric_limits<std::uint32_t>::max() - 1U,
                                   std::numeric_limits<std::uint32_t>::max()}) {
            bool rejected = false;
            try {
                (void)proposer.propose_for_round(source, maximum, 5);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "round lookahead overflow must be rejected");
        }
        std::cout << "round boundary checks=" << boundary_checks << '\n';

        NgramProposer special(256, 1024);
        special.set_boundaries({120});
        special.ingest(source);
        require(special.propose(std::span<const int>(source).first(18), 15, 12).tokens.size() == 2,
                "stop before control token");

        // Pressure raw and derived sources with both general and patterned token IDs.
        for (const unsigned shift : {0U, 9U}) {
            std::vector<int> old_source(2048), pressure(240000);
            std::mt19937 pressure_random(91026);
            const auto next_token = [&] {
                return static_cast<int>((1U + pressure_random() % (150000U >> shift)) << shift);
            };
            for (auto& token : old_source) { token = next_token(); }
            for (auto& token : pressure) { token = next_token(); }
            for (bool derived : {false, true}) {
                NgramProposer retained;
                if (!derived) { retained.ingest(old_source); }
                retained.ingest(pressure);
                if (derived) { retained.ingest(old_source); }
                retained.ingest(pressure);
                unsigned hits = 0;
                for (std::size_t end = 64; end + 15 <= old_source.size(); ++end) {
                    const auto tail  = std::span<const int>(old_source).subspan(end - 64, 64);
                    const auto draft = retained.propose(tail, 15);
                    if (draft.tokens.empty()) { continue; }
                    require(std::equal(draft.tokens.begin(), draft.tokens.end(),
                                       old_source.begin() + end),
                            "pressure changed the exact copied span");
                    ++hits;
                }
                std::cout << "pressure shift=" << shift << " derived=" << derived
                          << " hits=" << hits << '\n';
                require(hits >= 1750, "default index lost old spans under long-context pressure");
            }
        }

        // Independent live-ring oracle with forced hash collisions and repeated wrap-around.
        NgramProposer ring(128, 8);
        std::vector<int> ledger;
        std::mt19937 random(72309);
        std::size_t proposals = 0;
        for (int step = 0; step < 100000; ++step) {
            const int token = random() % 37 == 0 ? -1 : static_cast<int>(random() % 9);
            ring.append(token);
            ledger.push_back(token);
            if (ledger.size() < 32) { continue; }
            const auto begin   = ledger.size() > 128 ? ledger.size() - 128 : 0;
            const auto end     = begin + 16 + random() % (ledger.size() - begin - 16);
            const auto history = std::span<const int>(ledger).subspan(end - 16, 16);
            const auto draft   = ring.propose(history, 15, 4);
            if (draft.tokens.empty()) { continue; }
            ++proposals;
            bool found = false;
            for (auto pos = begin + draft.matched; pos + draft.tokens.size() <= ledger.size();
                 ++pos) {
                const auto tail = history.last(draft.matched);
                if (std::equal(tail.begin(), tail.end(), ledger.begin() + pos - draft.matched) &&
                    std::equal(draft.tokens.begin(), draft.tokens.end(), ledger.begin() + pos)) {
                    found = true;
                    break;
                }
            }
            require(found, "collision/wrapped position proposed a nonexistent span");
            require(std::find(draft.tokens.begin(), draft.tokens.end(), -1) == draft.tokens.end(),
                    "proposal crossed a boundary");
        }
        require(proposals > 100, "oracle must exercise actual proposals");
        for (const auto& delimiter : {std::string(": "), std::string("\t"),
                                      std::string("\xe2\x86\x92"), std::string(" | ")}) {
            const auto inputs = "  10" + delimiter + "def f():\n  11" + delimiter +
                                "    x = 1\n  12" + delimiter + "    return x\n";
            const auto sources = ngram_numbered_sources(inputs);
            require(sources == std::vector<std::string>{"def f():\n    x = 1\n    return x\n"},
                    "numbered tool text must preserve code indentation");
        }
        require(ngram_numbered_sources("1: a\n2: b\n4: d\n").empty(), "truncated run");
        require(ngram_numbered_sources("a\nb\nc\n").empty(), "plain text stays raw");
        require(ngram_numbered_sources("1: a\n2: b\n3: c")[0] == "a\nb\nc",
                "no invented terminal newline");
        require(ngram_numbered_sources("L1: a\r\nL2: \r\nL3:     b\r\n") ==
                    std::vector<std::string>{"a\r\n\r\n    b\r\n"},
                "CRLF and empty lines preserved");
        require(ngram_numbered_sources("1: a\n2: b\n3: c\n9: d\n10: e\n11: f\n") ==
                    std::vector<std::string>{"a\nb\nc\n", "d\ne\nf\n"},
                "discontinuous runs remain separate");
        require(ngram_numbered_sources("1: a\n2\tb\n3: c\n").empty(),
                "mixed display styles rejected");
        require(ngram_numbered_sources("18446744073709551615: a\n0: b\n1: c\n").empty(),
                "line numbering cannot wrap uint64");
        require(ngram_numbered_sources("18446744073709551616: a\n0: b\n1: c\n").empty(),
                "overflowing line number rejected");
        require(ngram_numbered_sources("\n    \nL\n: x\n").empty(),
                "empty and incomplete prefixes rejected");
        require(ngram_numbered_sources("1: caf\xc3\xa9\n2: cafe\xcc\x81\n3: \tvalue\n") ==
                    std::vector<std::string>{"caf\xc3\xa9\ncafe\xcc\x81\n\tvalue\n"},
                "Unicode normalization and indentation bytes must not change");
        require(ngram_numbered_sources(std::string("1: a") + '\0' + "b\n2: c\n3: d") ==
                    std::vector<std::string>{std::string("a") + '\0' + "b\nc\nd"},
                "embedded NUL must not truncate a proposal source");
        std::cout << "ngram CPU tests passed; oracle proposals=" << proposals << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
