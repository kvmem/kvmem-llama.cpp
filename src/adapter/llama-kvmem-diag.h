#pragma once

// Opt-in KVMEM_* diagnostic output.
//
// The KVMEM_* lines are machine-readable records: the scripts in scripts/ parse
// them with regexes to build the speed/VRAM/MTP reports, and a dozen of those
// scripts already export KVMEM_TRACE=1 for exactly that reason. They were never
// meant to be unconditional, but most print sites were unguarded, so a normal
// interactive session was drowned in them (163 of 535 lines in one real capture).
//
// The gate lives here because it serves both sides: the KVMem tools (tools/) and
// the llama.cpp adapter (src/adapter/). Those are separate modules with separate
// copies of the static below, so the *environment variable* is the authoritative
// switch -- the tool also mirrors --kvmem-trace into it for exactly that reason.
//
//   KVMEM_TRACE=1        environment variable, as the benchmark scripts set it
//   --kvmem-trace        same thing, per invocation (the tool sets the env var)
//   --no-kvmem-trace     force off even when the environment sets it
//
// With the gate off, an interactive run prints only llama.cpp's own output.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// One shared flag per module: the static is function-local inside an inline
// function, so every translation unit that includes this header in the same
// module sees the same instance.
inline bool & kvmem_diag_state() {
    static bool on = [] {
        const char * e = std::getenv("KVMEM_TRACE");
        // Unset, empty and "0" all mean off.
        return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return on;
}

inline void kvmem_diag_set(bool on) {
    kvmem_diag_state() = on;
}

inline bool kvmem_diag_enabled() {
    return kvmem_diag_state();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline void kvmem_diag(const char * fmt, ...) {
    if (!kvmem_diag_enabled()) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}
