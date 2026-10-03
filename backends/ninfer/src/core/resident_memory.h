#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace ninfer::core {

struct ResidentMemoryConfig {
    int device = 0;
    std::size_t reserve_bytes = 64ULL * 1024 * 1024;
    std::size_t host_pool_bytes = 0;
    std::size_t probe_step_bytes = 128ULL * 1024 * 1024;
};

// Only capacity failures are recoverable by reducing a startup candidate. Invalid
// counters, CUDA execution faults and invalid state remain ordinary runtime errors.
class VramCapacityError : public std::runtime_error {
public:
    enum class Reason { Capacity, Allocation, Shared };
    explicit VramCapacityError(const std::string& message, Reason reason = Reason::Capacity)
        : std::runtime_error(message), reason_(reason) {}
    Reason reason() const noexcept { return reason_; }
private:
    Reason reason_;
};

struct ResidentMemoryStats {
    std::size_t cuda_free_bytes = 0;
    std::size_t cuda_total_bytes = 0;
    std::size_t dedicated_bytes = 0;
    std::size_t shared_bytes = 0;
    std::size_t shared_baseline_bytes = 0;
    // Session-routed allocations plus scan scratch. Driver/Graph internals and
    // tiny TMA descriptors allocated during capture are outside this registry;
    // their residency is checked after Graph execution using process counters.
    std::size_t device_allocated_bytes = 0;
    std::size_t host_pool_bytes = 0;
    std::size_t host_used_bytes = 0;
    std::size_t verified_reserve_bytes = 0;
    bool verified = false;
};

// One session belongs to one engine and GPU. It must be created before weights
// are uploaded. Host backing is fixed before the immutable Shared baseline.
class ResidentMemorySession {
public:
    explicit ResidentMemorySession(const ResidentMemoryConfig& config);
    ~ResidentMemorySession();
    ResidentMemorySession(const ResidentMemorySession&) = delete;
    ResidentMemorySession& operator=(const ResidentMemorySession&) = delete;

    void* allocate_device(std::size_t bytes);
    void free_device(void* pointer) noexcept;
    void* allocate_host(std::size_t bytes);
    void free_host(void* pointer) noexcept;
    void check(const char* stage);
    void verify_working_set(bool with_reserve = true);
    // Temporary blocks are always released. The returned value is only a planner
    // candidate; the final contiguous arenas still require joint verification.
    std::size_t probe_available_bytes(std::size_t max_bytes);
    ResidentMemoryStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Routing is thread-local; ownership is not. Every owning allocation retains the
// session that allocated it, even after this binding has been destroyed.
class ResidentMemoryBinding {
public:
    explicit ResidentMemoryBinding(std::shared_ptr<ResidentMemorySession> session);
    ~ResidentMemoryBinding();
    ResidentMemoryBinding(const ResidentMemoryBinding&) = delete;
    ResidentMemoryBinding& operator=(const ResidentMemoryBinding&) = delete;

private:
    std::shared_ptr<ResidentMemorySession> previous_;
};

std::shared_ptr<ResidentMemorySession> current_resident_memory();

} // namespace ninfer::core
