#pragma once

// Stable per-statistic console colouring for stats output.
//
// Each statistic name gets a distinct colour on first appearance, so it keeps
// the same colour on every subsequent line and can be tracked across the log as
// its value changes. Callers decide when to colour (typically: only when stderr
// is an interactive console, so logs captured to files stay plain).
//
// Colours are emitted as ANSI SGR sequences using one of two palettes:
//   - a rich 256-colour palette when the console supports it (Windows Terminal,
//     and all POSIX terminals);
//   - a basic 16-colour palette otherwise. The legacy Windows console (conhost,
//     the classic cmd window) renders only the basic 16 colours, not 256-colour
//     sequences, so that palette is what actually shows there.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <unistd.h>
#endif

namespace ninfer::product::log_colour {

// Rich 256-colour palette: 48 cube colours, ordered in three blocks of 16 so
// each colourable family owns a whole block of one hue group and the three line
// types can't be confused. The block hue groups are maximally distant:
//   General [0..15]  = red / orange / yellow / magenta (max-red, warm)
//   Cache   [16..31] = blue / teal / cyan (low-red, cool)
//   Done    [32..47] = green / bright blue
// (The earlier ordering mixed an olive green and an orange into the cache block,
// so a host-cache line shared warm tones with a throughput line; that is fixed.)
// The done block mixes greens with bright blues because a run of similar greens
// is hard to tell apart; every blue stays bright enough on a black background.
// All 48 stay bright enough on a black console background.
constexpr std::array<std::uint16_t, 48> kPalette = {
    196, 197, 199, 201, 203, 205, 207, 208, // red / salmon / orange (general)
    210, 211, 214, 216, 220, 226, 227, 229, // orange / amber / yellow (general)
    20, 21, 26, 30, 31, 36, 41, 44, // blue / teal (cache)
    49, 51, 56, 61, 66, 71, 76, 86, // teal / cyan (cache)
    46, 47, 82, 76, 121, 118, 154, 159, // green (done)
    33, 39, 45, 69, 75, 111, 117, 153, // bright blue (done)
};

// Basic 16-colour SGR foreground codes, one set per colourable family. On a
// 16-colour console (a legacy console that predates 256-colour VT output) there
// are only ~7 hues, so three families cannot all be fully hue-disjoint; the sets
// below are chosen to be as distinguishable as that limit allows (general = warm
// hues, cache = cool hues, done = the bright forms). 256-colour consoles — which
// include every modern cmd/Windows Terminal — use kPalette instead and are
// cleanly disjoint. Black (30) and dark gray (90) are excluded so nothing
// renders unreadable on a black background.
constexpr std::array<std::uint16_t, 8> kBasicGeneral = {
    31, 33, 35, 37, 91, 93, 95, 97, // red yellow magenta white (+ bright forms)
};
constexpr std::array<std::uint16_t, 6> kBasicCache = {
    32, 34, 36, 92, 94, 96, // green blue cyan (+ bright forms)
};
constexpr std::array<std::uint16_t, 6> kBasicDone = {
    91, 93, 95, 97, 94, 92, // bright red yellow magenta white blue green
};

// Family of a stats line. Three families are colourable, each owning a disjoint
// block of kPalette so the three line types read as visually distinct; None is
// a non-colourable marker for lines that must stay plain (request-start "settings"
// lines).
enum class LogFamily : std::size_t {
    General = 0,
    Cache = 1,
    Done = 2,
    None = 3,
};
inline constexpr std::size_t kColorableFamilies = 3;
inline constexpr std::size_t kFamilySize        = kPalette.size() / kColorableFamilies;

[[nodiscard]] inline bool line_starts_with(std::string_view line, std::string_view prefix) {
    return line.size() >= prefix.size() && line.substr(0, prefix.size()) == prefix;
}

// Classify a console line into a family. Operational request lines start with
// "req#<id> <status> ...": "done" (a request's completion statistics) is Done,
// so request lifecycles use the green block and read as a distinct line type
// from the warm General block (throughput, startup, KV capacity); "started" is
// a request-START line whose body is the request's settings and is returned as
// None (kept plain); every other status (rejected / failed / cancelled /
// response) is General diagnostics.
//   - "host-cache ..." lines are Cache.
//   - every other line (throughput, startup, KV capacity) is General.
[[nodiscard]] inline LogFamily family_for(std::string_view line) {
    if (line_starts_with(line, "host-cache")) { return LogFamily::Cache; }
    if (line_starts_with(line, "req#")) {
        std::size_t id_end = 4;
        while (id_end < line.size() && line[id_end] != ' ') { ++id_end; }
        std::string_view status;
        if (id_end < line.size()) {
            const std::size_t status_end = line.find(' ', id_end + 1);
            status = line.substr(
                id_end + 1,
                status_end == std::string_view::npos
                    ? std::string_view::npos
                    : status_end - id_end - 1);
        }
        if (status == "done") { return LogFamily::Done; }
        if (status == "started") { return LogFamily::None; } // settings line -> plain
    }
    return LogFamily::General;
}

// True when stderr is an interactive console (not redirected to a file/pipe).
// The check is on the Win32 console handle itself: GetConsoleMode succeeds only
// for a real console and fails for a redirected handle, so it is the ground
// truth for "this window will render the colour". (The C runtime's _isatty is
// not used: it has been observed to report 0 on a genuine console, which would
// silently suppress the colour.)
[[nodiscard]] inline bool stderr_is_console() noexcept {
#ifdef _WIN32
    const HANDLE handle = ::GetStdHandle(STD_ERROR_HANDLE);
    if (handle == INVALID_HANDLE_VALUE || handle == 0) { return false; }
    DWORD mode           = 0;
    return ::GetConsoleMode(handle, &mode) != 0;
#else
    return ::isatty(STDERR_FILENO) == 1;
#endif
}

// Windows: allow the console to interpret VT escape sequences on stderr, and
// report whether that succeeded (i.e. this is a real console whose build honours
// the flag). A successful enable means the console is Win10 1607+ (incl. Win11)
// or Windows Terminal — both render the full 256-colour palette. Returns false
// when stderr is not a console (GetConsoleMode fails) or predates VT support.
[[nodiscard]] inline bool enable_stderr_vt_processing() noexcept {
#ifdef _WIN32
    static const bool done = [] {
        const HANDLE handle = GetStdHandle(STD_ERROR_HANDLE);
        DWORD mode          = 0;
        if (handle == INVALID_HANDLE_VALUE || handle == 0) { return false; }
        if (::GetConsoleMode(handle, &mode) == 0) { return false; }
        if ((mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0) { return true; } // already on
        return ::SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
    }();
    return done;
#else
    return true; // POSIX terminals: VT + 256-colour assumed, nothing to enable
#endif
}

// True when the console can render the rich 256-colour palette. Windows Terminal
// sets WT_SESSION; a legacy conhost (Win10 1607+ / Win11) supports 256-colour
// once VT processing is enabled on stderr, which enable_stderr_vt_processing()
// reports. POSIX terminals always support 256 colours. (Older conhosts that
// reject the VT flag fall back to the basic 16-colour palette.)
[[nodiscard]] inline bool rich_colour_supported() noexcept {
#ifdef _WIN32
#    pragma warning(push)
#    pragma warning(disable : 4996) // MSVC flags std::getenv; only used for env lookup
    const bool rich = ::std::getenv("WT_SESSION") != nullptr || enable_stderr_vt_processing();
#    pragma warning(pop)
    return rich;
#else
    return true;
#endif
}

// Stable first-appearance colour slot for a statistic name (process-wide). The
// output sequence is fixed, so the assignment is stable across runs too.
[[nodiscard]] inline std::size_t colour_slot(std::string_view key) {
    static std::unordered_map<std::string, std::size_t> assigned;
    const std::string owned(key);
    const auto found = assigned.find(owned);
    if (found != assigned.end()) { return found->second; }
    const std::size_t slot = assigned.size() % kPalette.size();
    assigned.emplace(std::move(owned), slot);
    return slot;
}

// Stable first-appearance palette index for a key within a family. Each colourable
// family owns a disjoint block of kPalette, so a cache-line key can never share a
// colour with a general-line or done-line key. (LogFamily::None must not be passed
// here — colourise_stats_line keeps None lines plain before reaching this point.)
[[nodiscard]] inline std::size_t family_colour_slot(std::string_view key, LogFamily family) {
    static std::unordered_map<std::string, std::size_t> assigned[kColorableFamilies];
    const std::size_t index = static_cast<std::size_t>(family) % kColorableFamilies;
    auto& map               = assigned[index];
    const std::string owned(key);
    const auto found = map.find(owned);
    if (found != map.end()) { return found->second; }
    const std::size_t base   = index * kFamilySize;
    const std::size_t within = map.size() % kFamilySize;
    const std::size_t slot   = base + within;
    map.emplace(std::move(owned), slot);
    return slot;
}

// The 256-colour value for a statistic's slot (used by the rich palette).
[[nodiscard]] inline std::uint16_t colour_for(std::string_view key) {
    return kPalette[colour_slot(key)];
}

// The SGR foreground spec (without the ESC[ ... m wrapper) for a slot, using
// whichever palette the console supports. Rich consoles use the full 48-colour
// palette (already split into three 16-colour blocks by family); basic consoles
// map the slot into the family's own code set.
[[nodiscard]] inline std::string colour_spec(std::size_t slot, LogFamily family) {
    if (rich_colour_supported()) {
        return "38;5;" + std::to_string(kPalette[slot]);
    }
    if (family == LogFamily::Cache) {
        return std::to_string(kBasicCache[slot % kBasicCache.size()]);
    }
    if (family == LogFamily::Done) {
        return std::to_string(kBasicDone[slot % kBasicDone.size()]);
    }
    return std::to_string(kBasicGeneral[slot % kBasicGeneral.size()]);
}

// Wrap text in an SGR sequence for an explicit palette slot, or return it
// unchanged when disabled.
[[nodiscard]] inline std::string colourize_slot(std::string_view text, std::size_t slot,
                                                LogFamily family, bool enabled) {
    if (!enabled) { return std::string(text); }
    (void)enable_stderr_vt_processing();
    std::string output;
    output.reserve(8 + text.size() + 8);
    output += "\033[";
    output += colour_spec(slot, family);
    output += 'm';
    output += text;
    output += "\033[0m";
    return output;
}

// Wrap text in an SGR colour sequence keyed by key (global palette), or return
// it unchanged when enabled is false.
[[nodiscard]] inline std::string colourize(std::string_view text, std::string_view key,
                                           bool enabled) {
    return colourize_slot(text, colour_slot(key), LogFamily::General, enabled);
}

// Colour every statistic of a pretty stats line with the stable colour of its
// name, within the line's family (so throughput, host-cache and done lines each
// use their own disjoint colour region). Pretty stats lines separate statistics
// into " | "-delimited clauses: the first clause is the line header ("throughput",
// "req#42 done") and stays plain, and each later clause is "name value [unit] ..."
// — the clause-opening name colours the whole clause, so a statistic keeps one
// colour as its value changes. A key=value token anywhere names its own
// statistic (a bare token after one inherits its colour), which keeps legacy
// key=value tails such as the vision offload fields of a done line colourable.
// A line classified as LogFamily::None (a request-start "settings" line) is
// returned entirely unchanged.
[[nodiscard]] inline std::string colourise_stats_line(std::string_view line, bool enabled) {
    if (!enabled) { return std::string(line); }
    const LogFamily family = family_for(line);
    if (family == LogFamily::None) { return std::string(line); } // settings line: plain
    std::string out;
    out.reserve(line.size() + 64);
    std::optional<std::string_view> unit_owner; // most recent statistic name
    std::size_t position = 0;
    std::size_t clause_index = 0;
    for (;;) {
        const std::size_t separator = line.find(" | ", position);
        const std::size_t clause_end =
            separator == std::string_view::npos ? line.size() : separator;
        std::size_t start = position;
        std::size_t token_index = 0;
        for (std::size_t i = position; i <= clause_end; ++i) {
            if (i != clause_end && line[i] != ' ') { continue; }
            const std::string_view token = line.substr(start, i - start);
            start = i + 1;
            if (token.empty()) { continue; }
            if (token_index != 0) { out += ' '; }
            ++token_index;
            const std::size_t eq = token.find('=');
            if (eq != std::string_view::npos) {
                unit_owner = token.substr(0, eq); // key=value names its own statistic
            } else if (clause_index != 0 && token_index == 1) {
                unit_owner = token; // a bare clause-opening token is the statistic's name
            }
            if (clause_index != 0 && unit_owner.has_value()) {
                out += colourize_slot(token, family_colour_slot(*unit_owner, family), family,
                                       true);
            } else {
                out += std::string(token);
            }
        }
        if (separator == std::string_view::npos) { break; }
        out += " | ";
        position = separator + 3;
        ++clause_index;
    }
    return out;
}

} // namespace ninfer::product::log_colour