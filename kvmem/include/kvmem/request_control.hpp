#pragma once

#include <functional>

namespace kvmem {

// Request-scoped host callbacks. Engines never need to own a transport object.
// The caller keeps this object alive until the synchronous driver call returns.
struct RequestControl {
    std::function<bool()> keep_alive;
    std::function<void(int, int)> on_prefill;
    bool aborted = false;
};

inline bool poll_request(RequestControl * control) {
    if (!control) return true;
    if (control->aborted) return false;
    if (control->keep_alive && !control->keep_alive()) control->aborted = true;
    return !control->aborted;
}

} // namespace kvmem
