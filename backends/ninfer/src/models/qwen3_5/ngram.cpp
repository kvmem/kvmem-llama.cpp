#include "models/qwen3_5/ngram.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::size_t allocation_overhead = 128;
constexpr std::size_t chunk_tokens        = 16'384;
constexpr std::size_t chunk_overlap       = 256;
constexpr std::array<std::uint32_t, 3> widths{16, 8, 4};

struct Budget {
    explicit Budget(std::size_t capacity) : capacity(capacity) {}

    std::size_t capacity;
    std::atomic<std::size_t> used{0};
};

// Charges follow shared ownership, including evicted sources still pinned by a
// request or fork. Local charges count a shared source once per owning session.
struct Charge {
    std::shared_ptr<Budget> global, local;
    std::size_t physical = 0, logical = 0;

    Charge() = default;

    Charge(Charge&& other) noexcept
        : global(std::move(other.global)), local(std::move(other.local)),
          physical(std::exchange(other.physical, 0)), logical(std::exchange(other.logical, 0)) {}

    Charge& operator=(Charge&&) = delete;

    ~Charge() {
        if (global) { global->used.fetch_sub(physical); }
        if (local) { local->used.fetch_sub(logical); }
    }

    static bool fits(const std::shared_ptr<Budget>& budget, std::size_t bytes) {
        if (!budget) { return true; }
        const auto used = budget->used.load();
        return used <= budget->capacity && bytes <= budget->capacity - used;
    }

    static std::optional<Charge> acquire(std::shared_ptr<Budget> global,
                                         std::shared_ptr<Budget> local, std::size_t physical,
                                         std::size_t logical) {
        if (!fits(global, physical) || !fits(local, logical)) { return std::nullopt; }
        Charge out;
        out.global   = std::move(global);
        out.local    = std::move(local);
        out.physical = physical;
        out.logical  = logical;
        if (out.global) { out.global->used.fetch_add(physical); }
        if (out.local) { out.local->used.fetch_add(logical); }
        return out;
    }
};

std::uint64_t hash(std::span<const TokenId> tokens, std::uint64_t salt) {
    std::uint64_t value = 1469598103934665603ULL ^ salt;
    for (const auto token : tokens) {
        value = (value ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
    }
    // The index masks low bits; avalanche patterned token sequences before doing so.
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

bool better(const NgramMatch& candidate, const NgramMatch& current) {
    return candidate.matched > current.matched || (candidate.matched == current.matched &&
                                                   candidate.tokens.size() > current.tokens.size());
}

struct Source {
    Charge charge;
    std::uint64_t id, digest;
    std::size_t size, buckets;
    std::unique_ptr<TokenId[]> tokens;
    std::unique_ptr<std::uint32_t[]> heads, next;
    std::unique_ptr<std::uint64_t[]> filter;

    static std::size_t filter_words(std::size_t size) {
        return std::bit_ceil(std::max<std::size_t>(1, size / 8));
    }

    static std::uint64_t filter_mask(std::uint64_t key) {
        return (1ULL << ((key >> 12) & 63)) | (1ULL << ((key >> 18) & 63)) |
               (1ULL << ((key >> 24) & 63)) | (1ULL << ((key >> 30) & 63));
    }

    static std::size_t bucket_count(std::size_t size) {
        return std::bit_ceil(std::max<std::size_t>(16, size / 2));
    }

    static std::size_t bytes(std::size_t size) {
        return sizeof(Source) + size * (sizeof(TokenId) + 3 * sizeof(std::uint32_t)) +
               bucket_count(size) * sizeof(std::uint32_t) +
               filter_words(size) * sizeof(std::uint64_t) + 5 * allocation_overhead;
    }

    Source(Charge charge, std::uint64_t id, std::span<const TokenId> input)
        : charge(std::move(charge)), id(id), digest(hash(input, input.size())), size(input.size()),
          buckets(bucket_count(size)), tokens(std::make_unique<TokenId[]>(size)),
          heads(std::make_unique<std::uint32_t[]>(buckets)),
          next(std::make_unique<std::uint32_t[]>(3 * size)),
          filter(std::make_unique<std::uint64_t[]>(filter_words(size))) {
        std::copy(input.begin(), input.end(), tokens.get());
        for (std::uint32_t end = 4; end <= size; ++end) {
            for (std::uint32_t w = 0; w < widths.size(); ++w) {
                const auto n = widths[w];
                if (end < n) { continue; }
                const auto key = hash(input.subspan(end - n, n), n);
                if (n == 8) { filter[key & (filter_words(size) - 1)] |= filter_mask(key); }
                auto& head      = heads[key & (buckets - 1)];
                const auto slot = 3 * (end - 1) + w;
                next[slot]      = head;
                head            = slot + 1;
            }
        }
    }

    NgramMatch propose(std::span<const TokenId> history, std::uint32_t maximum,
                       std::uint32_t minimum_match,
                       const std::array<std::uint64_t, 3>& hashes) const {
        NgramMatch best;
        // Any >=8-token exact match must contain this suffix. A blocked Bloom
        // lookup saves random index reads on misses; positive hits still compare tokens.
        if (minimum_match >= 8) {
            const auto mask = filter_mask(hashes[1]);
            if ((filter[hashes[1] & (filter_words(size) - 1)] & mask) != mask) { return best; }
        }
        const std::span<const TokenId> data(tokens.get(), size);
        for (std::uint32_t w = 0; w < widths.size(); ++w) {
            const auto n = widths[w];
            if (history.size() < n) { continue; }
            const auto tail      = history.last(n);
            auto entry           = heads[hashes[w] & (buckets - 1)];
            std::uint32_t probes = 0, candidates = 0;
            while (entry && probes++ < 64 && candidates < 4) {
                const auto slot         = entry - 1;
                entry                   = next[slot];
                const std::uint32_t end = slot / 3 + 1;
                if (slot % 3 != w || end < n || end >= size ||
                    !std::equal(tail.begin(), tail.end(), tokens.get() + end - n)) {
                    continue;
                }
                ++candidates;
                std::uint32_t matched = n;
                while (matched < history.size() && matched < end &&
                       tokens[end - matched - 1] == history[history.size() - matched - 1]) {
                    ++matched;
                }
                if (matched < minimum_match || matched < best.matched) { continue; }
                const auto draft = data.subspan(end, std::min<std::size_t>(maximum, size - end));
                if (matched > best.matched || draft.size() > best.tokens.size()) {
                    best = {{draft.begin(), draft.end()}, matched, id, end};
                }
            }
        }
        return best;
    }
};

struct Entry {
    std::shared_ptr<const Source> source;
    std::shared_ptr<const Charge> ownership;
    std::uint64_t seen   = 0;
    NgramSourceKind kind = NgramSourceKind::Text;
};

struct Session {
    Charge charge;
    std::shared_ptr<Budget> budget;
    std::array<char, 256> key{};
    std::size_t key_size = 0;
    std::vector<Entry> entries;
    std::atomic<std::uint64_t> epoch{1};
    std::uint64_t generation = 0, seen = 0;
    bool active = false;

    Session(Charge charge, std::shared_ptr<Budget> budget, std::string_view name,
            std::size_t capacity)
        : charge(std::move(charge)), budget(std::move(budget)), key_size(name.size()) {
        std::copy(name.begin(), name.end(), key.begin());
        entries.reserve(capacity);
    }

    bool named(std::string_view name) const {
        return std::string_view(key.data(), key_size) == name;
    }
};

auto weakest(std::vector<Entry>& entries) {
    return std::min_element(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        if (a.kind != b.kind) { return a.kind < b.kind; }
        return a.seen < b.seen;
    });
}

} // namespace

struct NgramSnapshot::Impl {
    Charge charge;
    std::shared_ptr<Session> session;
    std::uint64_t epoch, generation;
    std::vector<Entry> entries;

    Impl(Charge charge, std::shared_ptr<Session> session)
        : charge(std::move(charge)), session(std::move(session)), epoch(this->session->epoch),
          generation(this->session->generation), entries(this->session->entries) {}
};

NgramSnapshot::NgramSnapshot(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

NgramSnapshot::~NgramSnapshot() = default;

bool NgramSnapshot::valid() const noexcept { return impl_->session->epoch == impl_->epoch; }

std::size_t NgramSnapshot::source_count() const noexcept {
    return valid() ? impl_->entries.size() : 0;
}

std::uint64_t NgramSnapshot::generation() const noexcept { return impl_->generation; }

std::size_t NgramSnapshot::session_bytes() const noexcept { return impl_->session->budget->used; }

NgramMatch NgramSnapshot::continue_copy(std::uint64_t source_id, std::uint32_t offset,
                                        std::span<const TokenId> history, std::uint32_t maximum,
                                        std::uint32_t minimum_match) const {
    if (!valid() || maximum == 0) { return {}; }
    for (const auto& entry : impl_->entries) {
        const auto& source = *entry.source;
        if (source.id != source_id || offset >= source.size) { continue; }
        const auto matched = std::min<std::size_t>(offset, history.size());
        if (matched < minimum_match || !std::equal(history.end() - matched, history.end(),
                                                   source.tokens.get() + offset - matched)) {
            return {};
        }
        const auto count = std::min<std::size_t>(maximum, source.size - offset);
        return {{source.tokens.get() + offset, source.tokens.get() + offset + count},
                static_cast<std::uint32_t>(matched),
                source.id,
                offset,
                entry.kind,
                true};
    }
    return {};
}

NgramMatch NgramSnapshot::propose(std::span<const TokenId> history, std::uint32_t maximum,
                                  std::uint32_t minimum_match) const {
    NgramMatch best;
    if (!valid() || maximum == 0 || history.size() < minimum_match) { return best; }
    std::array<std::uint64_t, 3> hashes{};
    for (std::size_t w = 0; w < widths.size(); ++w) {
        if (history.size() >= widths[w]) { hashes[w] = hash(history.last(widths[w]), widths[w]); }
    }
    // Most recently observed sources win otherwise equal matches.
    for (auto it = impl_->entries.rbegin(); it != impl_->entries.rend(); ++it) {
        auto match = it->source->propose(history, maximum, minimum_match, hashes);
        if (better(match, best)) {
            match.kind     = it->kind;
            match.archived = true;
            best           = std::move(match);
        }
    }
    return best;
}

struct NgramArchive::Request::Impl {
    Charge charge;
    std::shared_ptr<Session> session;
    std::uint64_t epoch;
    std::shared_ptr<const NgramSnapshot> snapshot;
    std::vector<Entry> staged;
    std::vector<TokenId> boundaries;

    Impl(Charge charge, std::shared_ptr<Session> session, std::span<const TokenId> boundaries,
         std::size_t capacity)
        : charge(std::move(charge)), session(std::move(session)), epoch(this->session->epoch),
          boundaries(boundaries.begin(), boundaries.end()) {
        staged.reserve(capacity);
        std::sort(this->boundaries.begin(), this->boundaries.end());
    }

    ~Impl() {
        if (session->epoch == epoch) { session->active = false; }
    }
};

NgramArchive::Request::Request(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

NgramArchive::Request::~Request() = default;

std::shared_ptr<const NgramSnapshot> NgramArchive::Request::snapshot() const noexcept {
    return impl_->snapshot;
}

struct NgramArchive::Impl {
    NgramArchiveLimits limits;
    std::shared_ptr<Budget> budget;
    Charge charge;
    std::vector<std::shared_ptr<Session>> sessions;
    std::uint64_t clock = 0, next_source = 1, publication = 0;

    explicit Impl(NgramArchiveLimits limits)
        : limits(limits), budget(std::make_shared<Budget>(limits.total_bytes)) {
        if (!limits.sessions || !limits.sources || limits.sources > 65'536 ||
            limits.sessions > 1024 || limits.session_bytes > limits.total_bytes) {
            throw std::invalid_argument("invalid ngram archive bounds");
        }
        const auto base = sizeof(Impl) + sizeof(Budget) + limits.sessions * sizeof(sessions[0]) +
                          kNgramRequestSourceCapacity * sizeof(NgramSourceView) +
                          3 * allocation_overhead;
        auto reserved = Charge::acquire(budget, {}, base, 0);
        if (!reserved) { throw std::invalid_argument("ngram archive budget is too small"); }
        // Charge is move-only so allocation accounting cannot be duplicated.
        charge.global   = std::move(reserved->global);
        charge.physical = std::exchange(reserved->physical, 0);
        sessions.reserve(limits.sessions);
    }

    std::shared_ptr<Session> find(std::string_view key) const {
        for (const auto& session : sessions) {
            if (session->named(key)) { return session; }
        }
        return {};
    }

    bool room(const std::shared_ptr<Session>& owner, std::size_t physical, std::size_t logical,
              NgramSourceKind priority = NgramSourceKind::Tool) {
        while (owner && !Charge::fits(owner->budget, logical) && !owner->entries.empty()) {
            const auto entry = weakest(owner->entries);
            if (entry->kind > priority) { break; }
            owner->entries.erase(entry);
        }
        if (owner && !Charge::fits(owner->budget, logical)) { return false; }
        while (!Charge::fits(budget, physical)) {
            std::shared_ptr<Session> victim;
            for (const auto& session : sessions) {
                if (session->entries.empty() || weakest(session->entries)->kind > priority) {
                    continue;
                }
                if (!victim || session->seen < victim->seen) { victim = session; }
            }
            if (!victim) { return false; }
            victim->entries.erase(weakest(victim->entries));
        }
        return true;
    }

    std::shared_ptr<Session> create(std::string_view key) {
        if (sessions.size() == limits.sessions) {
            auto victim = sessions.end();
            for (auto it = sessions.begin(); it != sessions.end(); ++it) {
                if (!(*it)->active && (victim == sessions.end() || (*it)->seen < (*victim)->seen)) {
                    victim = it;
                }
            }
            if (victim == sessions.end()) { return {}; }
            ++(*victim)->epoch;
            sessions.erase(victim);
        }
        const auto bytes = sizeof(Session) + sizeof(Budget) + limits.sources * sizeof(Entry) +
                           4 * allocation_overhead;
        if (bytes > limits.session_bytes || !room({}, bytes, 0)) { return {}; }
        auto local    = std::make_shared<Budget>(limits.session_bytes);
        auto reserved = Charge::acquire(budget, local, bytes, bytes);
        if (!reserved) { return {}; }
        auto session =
            std::make_shared<Session>(std::move(*reserved), std::move(local), key, limits.sources);
        sessions.push_back(session);
        return session;
    }

    std::optional<Entry> own(const std::shared_ptr<Session>& session,
                             std::shared_ptr<const Source> source, NgramSourceKind kind) {
        const auto bytes = Source::bytes(source->size) + allocation_overhead;
        if (!room(session, allocation_overhead, bytes, kind)) { return std::nullopt; }
        auto reserved = Charge::acquire(budget, session->budget, allocation_overhead, bytes);
        if (!reserved) { return std::nullopt; }
        return Entry{std::move(source), std::make_shared<Charge>(std::move(*reserved)), ++clock,
                     kind};
    }

    void stage(Request::Impl& request, std::span<const TokenId> tokens, NgramSourceKind kind) {
        if (tokens.size() < 4 || request.staged.size() == limits.sources) { return; }
        const auto digest = hash(tokens, tokens.size());
        auto find_source  = [&](std::vector<Entry>& entries) {
            return std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) {
                return entry.source->digest == digest && entry.source->size == tokens.size() &&
                       std::equal(tokens.begin(), tokens.end(), entry.source->tokens.get());
            });
        };
        const auto pending = find_source(request.staged);
        if (pending != request.staged.end()) {
            pending->kind = std::max(pending->kind, kind);
            return;
        }
        const auto retained = find_source(request.session->entries);
        if (retained != request.session->entries.end()) {
            auto refreshed = *retained;
            refreshed.kind = std::max(refreshed.kind, kind);
            refreshed.seen = ++clock;
            request.staged.push_back(std::move(refreshed));
            return;
        }
        const auto bytes = Source::bytes(tokens.size());
        if (!room(request.session, bytes + allocation_overhead, bytes + allocation_overhead,
                  kind)) {
            return;
        }
        auto reserved = Charge::acquire(budget, {}, bytes, 0);
        if (!reserved) { return; }
        auto source = std::make_shared<Source>(std::move(*reserved), next_source++, tokens);
        if (auto entry = own(request.session, std::move(source), kind)) {
            request.staged.push_back(std::move(*entry));
        }
    }

    void stage(Request::Impl& request, std::span<const NgramSourceView> sources) {
        // Under pressure retain reusable tool/code before generated chatter.
        for (int priority = static_cast<int>(NgramSourceKind::Tool); priority >= 0; --priority) {
            for (const auto& source : sources) {
                if (static_cast<int>(source.kind) != priority) { continue; }
                std::size_t begin = 0;
                while (begin < source.tokens.size()) {
                    auto boundary = [&](TokenId token) {
                        return token < 0 || std::binary_search(request.boundaries.begin(),
                                                               request.boundaries.end(), token);
                    };
                    while (begin < source.tokens.size() && boundary(source.tokens[begin])) {
                        ++begin;
                    }
                    auto end = begin;
                    while (end < source.tokens.size() && !boundary(source.tokens[end])) { ++end; }
                    for (auto offset = begin; offset < end;) {
                        const auto count = std::min(chunk_tokens, end - offset);
                        stage(request, source.tokens.subspan(offset, count), source.kind);
                        if (count == end - offset) { break; }
                        offset += count - chunk_overlap;
                    }
                    begin = end + (end < source.tokens.size());
                }
            }
        }
    }
};

NgramArchive::NgramArchive(NgramArchiveLimits limits) : impl_(std::make_unique<Impl>(limits)) {}

NgramArchive::~NgramArchive() {
    for (const auto& session : impl_->sessions) {
        ++session->epoch;
        session->entries.clear();
    }
}

std::unique_ptr<NgramArchive::Request> NgramArchive::begin(std::string_view key,
                                                           std::span<const NgramSourceView> input,
                                                           std::span<const TokenId> boundaries,
                                                           std::string_view parent,
                                                           std::uint64_t parent_generation) {
    if (key.empty() || key.size() > 256 || parent.size() > 256 || parent == key ||
        (parent.empty() != (parent_generation == 0))) {
        return {};
    }
    try {
        auto session = impl_->find(key);
        if (session && (session->active || !parent.empty())) { return {}; }
        auto ancestor = impl_->find(parent);
        if (!parent.empty() && (!ancestor || ancestor->generation != parent_generation)) {
            return {};
        }
        if (!session) { session = impl_->create(key); }
        if (!session) { return {}; }
        if (ancestor) {
            // A fixed copy of references is sufficient: sources themselves are immutable.
            const auto temporary_bytes =
                ancestor->entries.size() * sizeof(Entry) + allocation_overhead;
            if (!impl_->room(session, temporary_bytes, temporary_bytes)) { return {}; }
            auto temporary_charge =
                Charge::acquire(impl_->budget, session->budget, temporary_bytes, temporary_bytes);
            if (!temporary_charge) { return {}; }
            const auto entries = ancestor->entries;
            for (const auto& entry : entries) {
                if (auto inherited = impl_->own(session, entry.source, entry.kind)) {
                    session->entries.push_back(std::move(*inherited));
                }
            }
        }
        const auto bytes = sizeof(Request) + sizeof(Request::Impl) +
                           impl_->limits.sources * sizeof(Entry) + boundaries.size_bytes() +
                           4 * allocation_overhead;
        if (!impl_->room(session, bytes, bytes)) { return {}; }
        auto charge = Charge::acquire(impl_->budget, session->budget, bytes, bytes);
        if (!charge) { return {}; }
        auto request    = std::unique_ptr<Request>(new Request(std::make_unique<Request::Impl>(
            std::move(*charge), session, boundaries, impl_->limits.sources)));
        session->active = true;
        session->seen   = ++impl_->clock;
        impl_->stage(*request->impl_, input);
        const auto view_bytes = sizeof(NgramSnapshot) + sizeof(NgramSnapshot::Impl) +
                                session->entries.size() * sizeof(Entry) + 4 * allocation_overhead;
        if (!impl_->room(session, view_bytes, view_bytes)) { return {}; }
        auto view_charge = Charge::acquire(impl_->budget, session->budget, view_bytes, view_bytes);
        if (!view_charge) { return {}; }
        request->impl_->snapshot = std::shared_ptr<const NgramSnapshot>(new NgramSnapshot(
            std::make_unique<NgramSnapshot::Impl>(std::move(*view_charge), session)));
        return request;
    } catch (const std::bad_alloc&) { return {}; }
}

bool NgramArchive::publish(std::unique_ptr<Request> request,
                           std::span<const NgramSourceView> output) {
    if (!request || impl_->publication == std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    auto& pending = *request->impl_;
    auto& session = *pending.session;
    if (session.epoch != pending.epoch ||
        impl_->find(std::string_view(session.key.data(), session.key_size)) != pending.session) {
        return false;
    }
    pending.snapshot.reset();
    try {
        impl_->stage(pending, output);
    } catch (const std::bad_alloc&) {
        // Retention is optional; already staged validated sources can still commit.
    }
    for (auto& entry : pending.staged) {
        const auto old = std::find_if(
            session.entries.begin(), session.entries.end(),
            [&](const Entry& existing) { return existing.source->id == entry.source->id; });
        if (old != session.entries.end()) { session.entries.erase(old); }
        if (session.entries.size() == impl_->limits.sources) {
            const auto victim = weakest(session.entries);
            if (victim->kind > entry.kind) { continue; }
            session.entries.erase(victim);
        }
        session.entries.push_back(std::move(entry));
    }
    std::sort(session.entries.begin(), session.entries.end(),
              [](const Entry& a, const Entry& b) { return a.seen < b.seen; });
    session.generation = ++impl_->publication;
    session.seen       = ++impl_->clock;
    return true;
}

void NgramArchive::clear(std::string_view key) noexcept {
    for (auto it = impl_->sessions.begin(); it != impl_->sessions.end(); ++it) {
        if ((*it)->named(key)) {
            ++(*it)->epoch;
            (*it)->entries.clear();
            impl_->sessions.erase(it);
            return;
        }
    }
}

bool NgramArchive::publish(std::unique_ptr<Request> request, std::span<const TokenId> output,
                           std::uint32_t reasoning_tokens) {
    const auto split = std::min<std::size_t>(reasoning_tokens, output.size());
    const std::array<NgramSourceView, 2> sources{
        {{output.first(split), NgramSourceKind::Reasoning},
         {output.subspan(split), NgramSourceKind::Generated}}};
    return publish(std::move(request), sources);
}

std::size_t NgramArchive::bytes() const noexcept { return impl_->budget->used.load(); }

NgramArchiveStats NgramArchive::stats(std::string_view key) const noexcept {
    NgramArchiveStats out;
    out.enabled     = true;
    out.total_bytes = bytes();
    if (const auto session = impl_->find(key)) {
        out.generation    = session->generation;
        out.sources       = session->entries.size();
        out.session_bytes = session->budget->used;
    }
    return out;
}

} // namespace ninfer::models::qwen3_5
