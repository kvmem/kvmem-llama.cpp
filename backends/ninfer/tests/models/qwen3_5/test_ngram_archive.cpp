#include "models/qwen3_5/ngram.h"
#include "models/qwen3_5/program/ngram_proposer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

using namespace ninfer::models::qwen3_5;
using ninfer::TokenId;

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<TokenId> text(std::size_t count, TokenId first) {
    std::vector<TokenId> out(count);
    std::iota(out.begin(), out.end(), first);
    return out;
}

std::unique_ptr<NgramArchive::Request> begin(NgramArchive& archive, std::string_view key,
                                             std::span<const TokenId> tokens,
                                             NgramSourceKind kind = NgramSourceKind::Tool) {
    const NgramSourceView source{tokens, kind};
    return archive.begin(key, std::span(&source, 1));
}

bool copied(const NgramSnapshot& snapshot, std::span<const TokenId> source, std::size_t offset = 64,
            std::uint32_t count = 63) {
    const auto history = source.subspan(offset - std::min<std::size_t>(64, offset),
                                        std::min<std::size_t>(64, offset));
    const auto match   = snapshot.propose(history, count);
    return match.archived && match.source != 0 && match.tokens.size() == count &&
           std::equal(match.tokens.begin(), match.tokens.end(), source.begin() + offset);
}

void lifecycle() {
    NgramArchive archive;
    const auto source = text(1000, 100), other = text(1000, 10000);
    auto seed = begin(archive, "root", source);
    require(seed && seed->snapshot()->source_count() == 0, "uncommitted input became visible");
    require(!archive.begin("root", {}), "overlapping owner accepted");
    require(archive.publish(std::move(seed)), "initial publication");
    auto resumed      = archive.begin("root", {});
    const auto frozen = resumed->snapshot();
    require(frozen->generation() == 1 && copied(*frozen, source), "compaction lost the source");
    require(archive.publish(std::move(resumed)), "empty continuation publication");

    auto fork = archive.begin("branch", {}, {}, "root", 2);
    require(fork && copied(*fork->snapshot(), source), "explicit ancestor was not inherited");
    const NgramSourceView addition{other, NgramSourceKind::Tool};
    require(archive.publish(std::move(fork), std::span(&addition, 1)), "fork publication");
    auto root = archive.begin("root", {});
    require(!copied(*root->snapshot(), other), "fork leaked into parent");
    root.reset();
    auto sibling = archive.begin("sibling", {}, {}, "root", 2);
    require(sibling && !copied(*sibling->snapshot(), other), "fork leaked into sibling");
    sibling.reset();
    require(!archive.begin("wrong-generation", {}, {}, "root", 1), "stale ancestor accepted");
    require(!archive.begin("branch", {}, {}, "root", 2), "existing branch silently merged");

    auto cancelled = begin(archive, "root", other);
    cancelled.reset();
    auto retry = archive.begin("root", {});
    require(!copied(*retry->snapshot(), other), "cancelled overlay published");
    archive.clear("root");
    require(!frozen->valid() && !copied(*frozen, source), "clear did not revoke old view");
    require(!archive.publish(std::move(retry)), "revoked request published");
    auto fresh = archive.begin("root", {});
    require(fresh && fresh->snapshot()->source_count() == 0, "clear resurrected source");
    fresh.reset();
    require(archive.publish(begin(archive, "root", other)), "reused identity publication");
    require(!archive.begin("stale-fork", {}, {}, "root", 1),
            "cleared identity recycled an old generation");
    auto current = archive.begin("current-fork", {}, {}, "root", archive.stats("root").generation);
    require(current && copied(*current->snapshot(), other) && !copied(*current->snapshot(), source),
            "new generation inherited the wrong incarnation");
    current.reset();
    auto branch = archive.begin("branch", {});
    require(copied(*branch->snapshot(), source) && copied(*branch->snapshot(), other),
            "parent clear invalidated independent branch");
    branch.reset();
    auto unrelated = archive.begin("unrelated", {});
    require(unrelated && !copied(*unrelated->snapshot(), source),
            "unrelated session shared corpus");
    require(!archive.begin("", {}) && !archive.begin(std::string(257, 'x'), {}),
            "unbounded identity");
}

void compaction_and_dedup() {
    NgramArchive archive;
    const auto source = text(2048, 1000);
    require(archive.publish(begin(archive, "coding", source)), "seed copy corpus");
    const auto stable = archive.bytes();
    for (int depth = 1; depth <= 25; ++depth) {
        auto turn = begin(archive, "coding", source);
        require(turn && copied(*turn->snapshot(), source), "source lost after compaction");
        require(archive.publish(std::move(turn)), "replayed request publication");
        require(archive.bytes() == stable, "repeated source inflated storage");
        auto compacted = archive.begin("coding", {});
        require(compacted && copied(*compacted->snapshot(), source),
                "source-absent request cannot copy");
        require(archive.publish(std::move(compacted)), "compacted request publication");
        require(archive.bytes() == stable, "empty compaction inflated storage");
    }
    NgramArchive restarted;
    auto request = restarted.begin("coding", {});
    require(request && !copied(*request->snapshot(), source), "archive survived engine restart");
}

void owner_lifetime() {
    std::shared_ptr<const NgramSnapshot> pinned;
    std::unique_ptr<NgramArchive::Request> pending;
    const auto source = text(1000, 100);
    {
        NgramArchive archive;
        require(archive.publish(begin(archive, "lifetime", source)), "owner lifetime seed");
        pending = archive.begin("lifetime", {});
        pinned  = pending->snapshot();
        require(copied(*pinned, source), "owner lifetime source missing");
    }
    require(!pinned->valid() && !copied(*pinned, source),
            "destroyed archive left a usable snapshot");
    pending.reset();
    pinned.reset();
}

void snapshot_readers() {
    NgramArchive archive;
    const auto source = text(4096, 1000);
    require(archive.publish(begin(archive, "readers", source)), "reader source seed");
    auto request  = archive.begin("readers", {});
    auto snapshot = request->snapshot();
    request.reset();
    std::barrier phase(5);
    std::atomic<bool> correct{true};
    std::vector<std::jthread> readers;
    for (int index = 0; index < 4; ++index) {
        readers.emplace_back([&, pinned = snapshot] {
            for (int offset = 64; offset < 256; ++offset) {
                if (!copied(*pinned, source, offset)) { correct = false; }
            }
            phase.arrive_and_wait();
            phase.arrive_and_wait();
            if (pinned->valid() || copied(*pinned, source)) { correct = false; }
        });
    }
    phase.arrive_and_wait();
    archive.clear("readers");
    snapshot.reset();
    phase.arrive_and_wait();
    readers.clear();
    require(correct, "immutable readers or cross-thread revocation failed");
    auto fresh = archive.begin("readers", {});
    require(fresh && fresh->snapshot()->source_count() == 0,
            "reader release resurrected cleared source");
}

void overlapping_clear() {
    const auto source = text(4096, 1000);
    for (int iteration = 0; iteration < 16; ++iteration) {
        NgramArchive archive;
        require(archive.publish(begin(archive, "overlap", source)), "overlap source seed");
        auto request  = archive.begin("overlap", {});
        auto snapshot = request->snapshot();
        request.reset();
        std::atomic<unsigned> started{0};
        std::atomic<bool> cleared{false}, correct{true};
        std::vector<std::jthread> readers;
        for (int index = 0; index < 4; ++index) {
            readers.emplace_back([&, pinned = snapshot] {
                if (!copied(*pinned, source)) { correct = false; }
                ++started;
                while (!cleared.load()) {
                    const auto match = pinned->propose(std::span(source).first(64), 63);
                    // A lookup racing revocation may complete its immutable read or miss.
                    if (!match.tokens.empty() &&
                        (match.tokens.size() != 63 || !match.archived ||
                         !std::equal(match.tokens.begin(), match.tokens.end(),
                                     source.begin() + 64))) {
                        correct = false;
                    }
                }
                if (pinned->valid() || copied(*pinned, source)) { correct = false; }
            });
        }
        while (started.load() != 4) { std::this_thread::yield(); }
        archive.clear("overlap");
        snapshot.reset();
        cleared = true;
        readers.clear();
        require(correct, "lookup overlapping clear or last-reader release failed");
        auto fresh = archive.begin("overlap", {});
        require(fresh && fresh->snapshot()->source_count() == 0,
                "overlapping readers resurrected cleared sources");
    }
}

void maximal_live_match() {
    using detail::NgramProposer;
    for (std::uint32_t history = 4; history <= 64; history += 4) {
        for (std::uint32_t maximum = 3; maximum <= 65; ++maximum) {
            NgramMatch live{std::vector<TokenId>(maximum, 1), history};
            require(NgramProposer::maximal(live, history, maximum), "full live match not maximal");
            for (std::uint32_t matched = 0; matched <= history; ++matched) {
                for (std::uint32_t count = 0; count <= maximum; ++count) {
                    require(!(matched > live.matched ||
                              (matched == live.matched && count > live.tokens.size())),
                            "archive could improve a skipped live match");
                }
            }
            --live.matched;
            require(!NgramProposer::maximal(live, history, maximum),
                    "full draft hid a longer archive match");
            ++live.matched;
            live.tokens.pop_back();
            require(!NgramProposer::maximal(live, history, maximum),
                    "short draft hid a longer archive continuation");
        }
    }
}

void token_boundaries() {
    NgramArchive archive;
    auto source = text(512, 100);
    source[128] = -1;
    source[256] = 99999;
    const std::array<TokenId, 1> boundaries{99999};
    const NgramSourceView input{source, NgramSourceKind::Text};
    auto seed = archive.begin("boundaries", std::span(&input, 1), boundaries);
    require(archive.publish(std::move(seed)), "segmented source publication");
    auto request        = archive.begin("boundaries", {});
    const auto snapshot = request->snapshot();
    require(snapshot->propose(std::span(source).first(120), 63).tokens.size() == 8,
            "proposal crossed negative boundary");
    require(snapshot->propose(std::span(source).subspan(192, 60), 63).tokens.size() == 4,
            "proposal crossed special token");
    require(snapshot->propose(std::span(source).first(8), 63).tokens.empty(),
            "minimum match bypassed");
    require(snapshot->propose(source, 0).tokens.empty(), "zero output budget");
    for (const auto count : {4U, 8U, 12U, 16U, 32U, 64U}) {
        const auto match = snapshot->propose(std::span(source).subspan(300, count), 31, 4);
        require(match.matched == count && match.kind == NgramSourceKind::Text &&
                    match.tokens == std::vector<TokenId>(source.begin() + 300 + count,
                                                         source.begin() + 331 + count),
                "exact contiguous lookup or provenance");
    }

    // Drafts around storage chunk boundaries must still have a full history and
    // NG63's two-token handoff lookahead. Chunking is not a semantic source end.
    const auto large = text(70000, 200000);
    require(archive.publish(begin(archive, "large", large)), "large corpus publication");
    auto large_request = archive.begin("large", {});
    for (std::size_t offset = 16000; offset < 16640; ++offset) {
        require(copied(*large_request->snapshot(), large, offset, 65),
                "chunk boundary lost copy window");
    }
}

void pressure() {
    const NgramArchiveLimits limits{512 * 1024, 2 * 1024 * 1024, 4, 64};
    NgramArchive archive(limits);
    const auto code = text(16384, 100000);
    require(archive.publish(begin(archive, "coding", code)), "pressure seed");
    auto request = archive.begin("coding", {});
    auto pinned  = request->snapshot();
    require(archive.publish(std::move(request)), "pin publication");
    for (int i = 0; i < 60; ++i) {
        const auto noise = text(16384, 200000 + i * 20000);
        auto turn        = begin(archive, "coding", noise, NgramSourceKind::Reasoning);
        if (turn) { (void)archive.publish(std::move(turn)); }
        require(archive.bytes() <= limits.total_bytes, "pinned memory escaped global bound");
    }
    auto retained = archive.begin("coding", {});
    require(retained && copied(*retained->snapshot(), code),
            "reasoning pressure evicted tool code");
    require(retained->snapshot()->session_bytes() <= limits.session_bytes,
            "pinned session exceeded budget");
    retained.reset();
    require(copied(*pinned, code), "eviction mutated pinned snapshot");
    archive.clear("coding");
    const auto held = archive.bytes();
    require(held > 16384 * sizeof(TokenId) && !pinned->valid(), "clear hid pinned allocation");
    pinned.reset();
    require(archive.bytes() < held, "released pin did not release accounted storage");

    for (int i = 0; i < 30; ++i) {
        auto turn = begin(archive, "session-" + std::to_string(i), code);
        if (turn) { (void)archive.publish(std::move(turn)); }
        require(archive.bytes() <= limits.total_bytes, "session churn escaped memory bound");
    }
    auto oldest = archive.begin("session-0", {});
    require(oldest && !copied(*oldest->snapshot(), code), "evicted session resurrected source");
}

void cursor_handoff() {
    NgramArchive archive;
    for (std::uint32_t available = 0; available <= 65; ++available) {
        const auto source = text(64 + available, 1000);
        archive.clear("cursor");
        require(archive.publish(begin(archive, "cursor", source)), "cursor source publication");
        auto request        = archive.begin("cursor", {});
        const auto snapshot = request->snapshot();
        const auto start    = snapshot->propose(std::span(source).first(32), 1);
        require(!start.tokens.empty(), "cursor seed missing");
        const auto history = std::span(source).first(64);
        for (std::uint32_t maximum = 1; maximum <= 63; ++maximum) {
            for (const std::uint32_t neural : {1U, 3U, 5U, 7U, 15U}) {
                auto indexed = snapshot->propose(history, maximum + 2);
                auto cursor  = snapshot->continue_copy(start.source, 64, history, maximum + 2);
                indexed = detail::NgramProposer::finish_round(std::move(indexed), maximum, neural);
                cursor  = detail::NgramProposer::finish_round(std::move(cursor), maximum, neural);
                require(indexed.tokens == cursor.tokens, "cursor bypassed source-ending handoff");
                if (!cursor.tokens.empty()) {
                    require(cursor.tokens.size() <= maximum && cursor.tokens.size() + 1 < available,
                            "cursor used source-ending bonus");
                }
            }
        }
        auto correction = std::vector<TokenId>(history.begin(), history.end());
        correction.back() += 10000;
        require(snapshot->continue_copy(start.source, 64, correction, 63).tokens.empty(),
                "cursor ignored committed correction token");
        require(snapshot->continue_copy(start.source + 9999, 64, history, 63).tokens.empty(),
                "unknown cursor source accepted");
        archive.clear("cursor");
        require(snapshot->continue_copy(start.source, 64, history, 63).tokens.empty(),
                "clear did not revoke cursor");
    }
}

void randomized_sources() {
    NgramArchive archive;
    std::mt19937 random(91326);
    std::vector<std::vector<TokenId>> sources(24, std::vector<TokenId>(2048));
    for (auto& source : sources) {
        for (auto& token : source) { token = 100 + random() % 200000; }
        require(archive.publish(begin(archive, "oracle", source)), "random corpus publication");
    }
    auto request       = archive.begin("oracle", {});
    std::size_t checks = 0;
    for (const auto& source : sources) {
        for (std::size_t offset = 64; offset < 1984; offset += 13) {
            const auto match =
                request->snapshot()->propose(std::span(source).subspan(offset - 64, 64), 63);
            // Bucket collisions may reduce recall, never invent a continuation.
            if (match.tokens.empty()) { continue; }
            require(match.matched == 64 &&
                        match.tokens == std::vector<TokenId>(source.begin() + offset,
                                                             source.begin() + offset + 63),
                    "indexed proposal disagrees with exact source oracle");
            ++checks;
        }
    }
    require(checks > 3000, "unexpectedly low lookup coverage");
    std::cout << "archive exact-source checks=" << checks << '\n';
}

} // namespace

int main() {
    try {
        lifecycle();
        compaction_and_dedup();
        owner_lifetime();
        snapshot_readers();
        overlapping_clear();
        maximal_live_match();
        token_boundaries();
        pressure();
        cursor_handoff();
        randomized_sources();
        std::cout << "ngram archive lifecycle, compaction, bounds and lookup passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
