#pragma once

// A real-model test that throws should say what happened, and should not sit in the suite as a
// permanent red when the reason is this box rather than the code.
//
// Upstream's real-model tests have a bare `int main()` with nothing catching, so any exception --
// an artifact rejection, a CUDA failure during Engine construction, a std::filesystem error --
// unwinds out of main and Windows terminates the process with 0xC0000409, printing nothing at all.
// CTest then reports `Exit code 0xc0000409` with no line to go on, which reads like memory
// corruption and is not: e06d7363 in a debugger is a C++ throw. Recovering the message meant
// attaching cdb.
//
// Use NINFER_GUARDED_TEST_MAIN(run) in place of `int main()`, with the body in a function
// returning int. Exit codes are unchanged, including 77 for a skip.

#include <exception>
#include <iostream>
#include <string_view>

namespace ninfer::test {

// A device-memory shortfall is the machine being busy, not a defect: these models are ~16-21 GiB
// resident, so a desktop holding a couple of GB is enough to tip one over. Windows adds a second
// spelling of the same thing, because a pinned host allocation is mapped into the GPU's address
// space and competes with it.
[[nodiscard]] inline bool is_capacity_shortfall(std::string_view message) noexcept {
    return message.find("available for runtime capacity") != std::string_view::npos ||
           message.find("cudaMallocHost failed to pin") != std::string_view::npos ||
           message.find("failed to reserve the host KV cache") != std::string_view::npos;
}

} // namespace ninfer::test

#define NINFER_GUARDED_TEST_MAIN(run_function)                                                     \
    int main() {                                                                                   \
        try {                                                                                      \
            return (run_function)();                                                                \
        } catch (const std::exception& error) {                                                    \
            const std::string_view message(error.what());                                          \
            if (::ninfer::test::is_capacity_shortfall(message)) {                                  \
                std::cout << "skip: insufficient free device memory -- " << message << std::endl;  \
                return 77;                                                                         \
            }                                                                                      \
            std::cerr << "FATAL: " << message << std::endl;                                        \
            return 1;                                                                              \
        } catch (...) {                                                                            \
            std::cerr << "FATAL: unknown exception" << std::endl;                                  \
            return 1;                                                                              \
        }                                                                                          \
    }
