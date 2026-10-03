#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

namespace ninfer::runtime {

// Scoped claim on a Program's seal window. A planner claims it before sealing its selected target
// so a concurrent demote cannot bump a victim's slot generation between the final assess and the
// seal (which would fail the seal's revalidate). Claiming backs off briefly on contention; if the
// window cannot be claimed within the budget, `claimed` stays false and the caller takes its own
// fallback instead of sealing. Release as soon as the seal (and any fallback) is done: holding it
// through result bookkeeping would starve concurrent planners waiting to seal.
template <class Session>
class SealWindowClaim {
public:
    explicit SealWindowClaim(Session& session) noexcept : session_(session) {
        for (std::uint32_t attempt = 0; attempt < kAttempts; ++attempt) {
            if (session_.try_claim_seal_window()) {
                claimed_ = true;
                break;
            }
            std::this_thread::sleep_for(kBackoff);
        }
    }

    SealWindowClaim(const SealWindowClaim&)            = delete;
    SealWindowClaim& operator=(const SealWindowClaim&) = delete;

    ~SealWindowClaim() { release(); }

    [[nodiscard]] bool claimed() const noexcept { return claimed_; }

    void release() noexcept {
        if (claimed_) {
            session_.release_seal_window();
            claimed_ = false;
        }
    }

private:
    static constexpr std::uint32_t kAttempts = 32;
    static constexpr std::chrono::microseconds kBackoff{125};

    Session& session_;
    bool claimed_ = false;
};

} // namespace ninfer::runtime
