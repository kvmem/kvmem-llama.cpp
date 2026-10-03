#pragma once

// Small fixed-capacity request execution for every backend.

#include "core/device.h"
#include "core/resident_memory.h"
#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/generation_budget.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

// `Manager` selects the prefix-cache implementation: ResourceManager (Legacy) or
// HybridResourceManager (docs/maintainer/hybrid-prefix-cache-spec.md). Both expose the same
// admission, capture and terminal-settlement surface to this core.
template <class Instance, class Manager = ResourceManager<typename Instance::ModelContract>>
class EngineCore {

public:
    using ModelContract      = typename Instance::ModelContract;
    using Program            = typename ModelContract::Program;
    using BasePlan           = typename ModelContract::RequestBasePlan;
    using Plan               = typename ModelContract::AdmissionCandidate;
    using SequenceHandle     = typename ModelContract::SequenceHandle;
    using CaptureOffer       = typename ModelContract::CaptureOffer;
    using PendingBatch       = typename ModelContract::PendingBatch;
    using PreparedPrompt     = typename ModelContract::PreparedPrompt;
    using NgramArchive       = typename ModelContract::NgramArchive;
    using OutputSession      = typename ModelContract::OutputSession;
    using PublishedOutput    = typename ModelContract::PublishedOutput;
    using Request            = RequestRecord<ModelContract>;
    using Scheduling         = Scheduler<Request>;
    using FifoSnapshot       = typename Scheduling::FifoSnapshot;
    using RoundMembership    = typename Scheduling::RoundMembership;
    using ControlMembership  = typename Scheduling::ControlMembership;
    using ActiveAdmissionSet = typename Scheduling::ActiveAdmissionSet;
    using ExecutionAction    = typename Scheduling::ExecutionAction;
    using AdmissionGrant     = typename Scheduling::AdmissionGrant;
    using ResourceManagement = Manager;
    using ResourceInspection = typename ResourceManagement::Inspection;
    using Clock              = std::chrono::steady_clock;

    [[nodiscard]] static Manager make_resource_manager(const EngineOptions& options,
                                                       ContextMachineCostModel context_cost) {
        if constexpr (std::is_same_v<Manager, ResourceManager<ModelContract>>) {
            return Manager(
                options.max_concurrency, options.context_cache.max_private_continuations.value(),
                options.context_cache.max_shared_prefixes.value(), options.context_cache.enabled,
                options.context_cache.max_long_anchors_per_continuation.value_or(0),
                std::move(context_cost));
        } else {
            (void)context_cost;
            return Manager(options.max_concurrency);
        }
    }

    EngineCore(Instance& instance, DeviceContext& device, const EngineOptions& options,
               ContextMachineCostModel context_cost)
        : instance_(instance), device_(device), max_context_(options.max_context),
          structured_output_(options.structured_output), max_concurrency_(options.max_concurrency),
          thorough_admission_search_(options.context_cache.thorough_admission_search),
          recover_invariant_failures_(options.recover_invariant_failures),
          kv_lease_growth_(options.context_cache.kv_lease_growth),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          resources_(make_resource_manager(options, std::move(context_cost))) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        scheduler_.allow_concurrent_prefill(options.concurrent_prefill);
        if (!options.context_cache.max_private_continuations ||
            !options.context_cache.max_shared_prefixes) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        if constexpr (std::is_same_v<Manager, ResourceManager<ModelContract>>) {
            if (options.context_cache.rolling_retention) { resources_.enable_rolling_retention(); }
            if (options.context_cache.release_diverged_checkpoints) {
                resources_.enable_diverged_checkpoint_release();
            }
            if (options.context_cache.recency_eviction) { resources_.enable_recency_eviction(); }
        }
        if (options.speculative.ngram_archive_bytes != 0) {
            ngram_archive_ = std::make_unique<NgramArchive>(typename NgramArchive::Limits{
                .session_bytes = options.speculative.ngram_session_bytes,
                .total_bytes   = options.speculative.ngram_archive_bytes});
        }
        diagnostics_ = options.diagnostic_observer;
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~EngineCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCore* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(EngineCore& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class EngineCore;
    };

    Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (options.execution.structured_output.kind != StructuredOutputKind::None &&
            !structured_output_) {
            throw std::invalid_argument(
                "structured output requires an Engine started with structured_output");
        }
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id        = 0;
        std::uint64_t publication_order = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            if (next_request_id_ == 0 || next_publication_order_ == 0) {
                throw std::overflow_error("request identity space exhausted");
            }
            ++outstanding_;
            request_id        = next_request_id_++;
            publication_order = next_publication_order_++;
        }

        std::shared_ptr<Request> request;
        try {
            if (ngram_archive_ && !options.ngram_session.key.empty()) {
                std::random_device entropy;
                options.execution.sampling.seed ^=
                    (static_cast<std::uint64_t>(entropy()) << 32) ^ entropy();
            }
            auto output = instance_.frontend.make_output_session(
                prompt, options.stop, options.output, options.execution.thinking,
                options.execution.structured_output);
            options.execution.grammar = output.grammar_state();
            const std::uint32_t capacity_output =
                max_context_ - prompt_summary.prompt_tokens + static_cast<std::uint32_t>(1);
            try {
                output.validate_generation_capacity(
                    std::min(options.execution.requested_output_tokens, capacity_output));
            } catch (const std::invalid_argument& error) {
                throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                                   error.what());
            }
            request = std::make_shared<Request>(request_id, publication_order, std::move(prompt),
                                                std::move(output), prompt_summary, prepare_seconds,
                                                std::move(options), consumer_mode, observation,
                                                pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    // Three context-cache exhaustions in a row without a successful admission mean the retained
    // checkpoints have wedged the pools: every retry of the session fails the same way. Reporting
    // the Engine unavailable lets the healthcheck-driven restart clear the pools.
    static constexpr std::uint32_t kStuckContextCacheExhaustions = 3;

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_ &&
               consecutive_context_cache_exhaustions_.load(std::memory_order_relaxed) <
                   kStuckContextCacheExhaustions;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
        } catch (...) {}
    }

private:
    enum class HostWorkClass : std::uint8_t {
        Decode,
        Prefill,
        Control,
    };

    using EngineHostPhase = RequestEngineHostPhase;

    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept {
        switch (phase) {
        case EngineHostPhase::Boundary:
            return nvtx::Name::EngineBoundary;
        case EngineHostPhase::CommitOutput:
            return nvtx::Name::EngineCommitOutput;
        case EngineHostPhase::Maintenance:
            return nvtx::Name::EngineMaintenance;
        }
        return nvtx::Name::EngineBoundary;
    }

    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
    };

    struct ActiveExposureSet {
        std::array<ActiveExposure, kMaximumConcurrency> entries{};
        std::size_t size = 0;
    };

    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ActiveExposureSet exposed;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count();
        return count > 0 ? static_cast<std::uint64_t>(count) : 0;
    }

    [[nodiscard]] ActiveExposureSet active_exposure_set() const {
        ActiveExposureSet result;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            result.entries[result.size++] = ActiveExposure{.request = slots_[lane], .lane = lane};
        }
        return result;
    }

    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const {
        return HostPhaseMeasurement{
            .started          = Clock::now(),
            .accounted_before = worker_accounted_elapsed_ns_,
            .exposed          = active_exposure_set(),
        };
    }

    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {}) noexcept {
        current_host_work_class_   = work_class;
        current_decode_lane_count_ = decode_lanes.size();
        for (std::size_t i = 0; i < decode_lanes.size(); ++i) {
            current_decode_lanes_[i] = decode_lanes[i];
        }
    }

    [[nodiscard]] bool current_decode_contains(std::uint32_t lane) const noexcept {
        return std::find(current_decode_lanes_.begin(),
                         current_decode_lanes_.begin() +
                             static_cast<std::ptrdiff_t>(current_decode_lane_count_),
                         lane) != current_decode_lanes_.begin() +
                                      static_cast<std::ptrdiff_t>(current_decode_lane_count_);
    }

    void add_class_host_time(std::uint64_t host_ns, std::uint64_t device_wait_ns) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        switch (current_host_work_class_) {
        case HostWorkClass::Decode:
            stats.decode_host_ns += host_ns;
            stats.decode_device_wait_ns += device_wait_ns;
            break;
        case HostWorkClass::Prefill:
            stats.prefill_host_ns += host_ns;
            stats.prefill_device_wait_ns += device_wait_ns;
            break;
        case HostWorkClass::Control:
            stats.control_host_ns += host_ns;
            stats.control_device_wait_ns += device_wait_ns;
            break;
        }
    }

    void expose_engine_phase(const ActiveExposureSet& exposed, EngineHostPhase phase,
                             std::uint64_t elapsed) noexcept {
        for (std::size_t i = 0; i < exposed.size; ++i) {
            const ActiveExposure& exposure = exposed.entries[i];
            RequestHostTiming& timing      = exposure.request->host_timing;
            timing.expose_engine(phase, elapsed,
                                 current_host_work_class_ == HostWorkClass::Decode &&
                                     current_decode_contains(exposure.lane));
        }
    }

    void finish_engine_phase(const HostPhaseMeasurement& measurement,
                             EngineHostPhase phase) noexcept {
        const std::uint64_t wall    = elapsed_ns(measurement.started, Clock::now());
        const std::uint64_t nested  = worker_accounted_elapsed_ns_ - measurement.accounted_before;
        const std::uint64_t own     = wall > nested ? wall - nested : 0;
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        switch (phase) {
        case EngineHostPhase::Boundary:
            stats.engine_boundary_ns += own;
            break;
        case EngineHostPhase::CommitOutput:
            stats.engine_commit_output_ns += own;
            break;
        case EngineHostPhase::Maintenance:
            stats.engine_maintenance_ns += own;
            break;
        }
        add_class_host_time(own, 0);
        expose_engine_phase(measurement.exposed, phase, own);
        worker_accounted_elapsed_ns_ += own;
    }

    void record_program_timing(runtime::ExecutionTiming timing,
                               const ActiveExposureSet& exposed) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        stats.program_submit_ns += timing.submit_host_ns;
        stats.program_post_ns += timing.post_host_ns;
        stats.device_wait_ns += timing.device_wait_ns;
        add_class_host_time(timing.host_ns(), timing.device_wait_ns);
        for (std::size_t i = 0; i < exposed.size; ++i) {
            const ActiveExposure& exposure = exposed.entries[i];
            RequestHostTiming& request     = exposure.request->host_timing;
            request.expose_program(timing, current_host_work_class_ == HostWorkClass::Decode &&
                                               current_decode_contains(exposure.lane));
        }
        worker_accounted_elapsed_ns_ += timing.elapsed_ns();
    }

    void finish_program_call(const HostPhaseMeasurement& measurement,
                             runtime::ExecutionTiming timing) noexcept {
        const std::uint64_t wall     = elapsed_ns(measurement.started, Clock::now());
        const std::uint64_t nested   = worker_accounted_elapsed_ns_ - measurement.accounted_before;
        const std::uint64_t observed = timing.elapsed_ns() + nested;
        if (wall > observed) { timing.submit_host_ns += wall - observed; }
        record_program_timing(timing, measurement.exposed);
    }

    void record_detail(std::uint64_t RuntimeHostWorkStats::* elapsed_member,
                       std::uint64_t RuntimeHostWorkStats::* invocation_member,
                       Clock::time_point started) noexcept {
        RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
        stats.*elapsed_member += elapsed_ns(started, Clock::now());
        ++(stats.*invocation_member);
    }

    class DetailScope {
    public:
        DetailScope(EngineCore& owner, std::uint64_t RuntimeHostWorkStats::* elapsed_member,
                    std::uint64_t RuntimeHostWorkStats::* invocation_member,
                    nvtx::Name range_name) noexcept
            : owner_(owner), elapsed_member_(elapsed_member), invocation_member_(invocation_member),
              started_(Clock::now()) {
            range_.emplace(range_name, nvtx::Category::Control);
        }

        ~DetailScope() {
            range_.reset();
            owner_.record_detail(elapsed_member_, invocation_member_, started_);
        }

        DetailScope(const DetailScope&)            = delete;
        DetailScope& operator=(const DetailScope&) = delete;

    private:
        EngineCore& owner_;
        std::uint64_t RuntimeHostWorkStats::* elapsed_member_;
        std::uint64_t RuntimeHostWorkStats::* invocation_member_;
        Clock::time_point started_;
        std::optional<nvtx::ScopedRange> range_;
    };

    class EnginePhaseScope {
    public:
        EnginePhaseScope(EngineCore& owner, EngineHostPhase phase)
            : owner_(owner), phase_(phase), measurement_(owner.begin_host_phase()) {
            range_.emplace(phase_range_name(phase), nvtx::Category::Runtime);
        }

        ~EnginePhaseScope() { finish(); }

        EnginePhaseScope(const EnginePhaseScope&)            = delete;
        EnginePhaseScope& operator=(const EnginePhaseScope&) = delete;

        void pause_range() noexcept { range_.reset(); }

        void resume_range() noexcept {
            if (active_ && !range_) {
                range_.emplace(phase_range_name(phase_), nvtx::Category::Runtime);
            }
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ProgramCallScope {
    public:
        explicit ProgramCallScope(EngineCore& owner)
            : owner_(owner), measurement_(owner.begin_host_phase()) {}

        ~ProgramCallScope() noexcept { finish(failed_timing_); }

        ProgramCallScope(const ProgramCallScope&)            = delete;
        ProgramCallScope& operator=(const ProgramCallScope&) = delete;

        [[nodiscard]] runtime::ExecutionTiming& failed_timing() noexcept { return failed_timing_; }

        void finish(runtime::ExecutionTiming timing) noexcept {
            if (!active_) { return; }
            owner_.finish_program_call(measurement_, timing);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        HostPhaseMeasurement measurement_;
        runtime::ExecutionTiming failed_timing_;
        bool active_ = true;
    };

    void publish_runtime_stats() {
        HostPhaseMeasurement measurement = begin_host_phase();
        std::optional<nvtx::ScopedRange> phase_range;
        phase_range.emplace(nvtx::Name::EngineMaintenance, nvtx::Category::Runtime);
        const Clock::time_point detail_started = Clock::now();
        std::optional<nvtx::ScopedRange> detail_range;
        detail_range.emplace(nvtx::Name::StatsPublication, nvtx::Category::Control);
        RuntimeStats snapshot = cumulative_stats_;
        resources_.populate_runtime_stats(*instance_.program, snapshot);
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
            const Clock::time_point now = Clock::now();
            for (const auto& request : pending_) {
                if (snapshot.queue_entries == RuntimeStats::kQueueReportCap) { break; }
                snapshot.queue[snapshot.queue_entries++] = RuntimeStats::QueueEntry{
                    .request_id   = request->id,
                    .wait_seconds = std::chrono::duration<double>(now - request->submitted).count(),
                };
            }
            if (!pending_.empty()) { last_queue_publication_ = now; }
        }
        snapshot.prefilling_requests = 0;
        snapshot.materializing_requests = materializing_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->is_prefilling() && !slots_[lane]->capture_pending) {
                ++snapshot.prefilling_requests;
            }
            if (slots_[lane]->is_decode_ready()) { ++snapshot.decode_ready_requests; }
            if (slots_[lane]->capture_pending) { ++snapshot.capture_pending_requests; }
            if (slots_[lane]->terminal_reason) { ++snapshot.terminal_pending_requests; }
        }
        detail_range.reset();
        record_detail(&RuntimeHostWorkStats::stats_publication_ns,
                      &RuntimeHostWorkStats::stats_publication_invocations, detail_started);
        phase_range.reset();
        finish_engine_phase(measurement, EngineHostPhase::Maintenance);
        snapshot.host_work = cumulative_stats_.host_work;
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
    }

    void record_prefix_selection(const RequestPlanSummary& summary) noexcept {
        switch (summary.prefix_reuse_path) {
        case PrefixReusePath::Root:
            ++cumulative_stats_.root_selections;
            break;
        case PrefixReusePath::PrivateEndpoint:
            ++cumulative_stats_.private_endpoint_selections;
            break;
        case PrefixReusePath::PrivateTurnClosure:
            ++cumulative_stats_.private_turn_closure_selections;
            break;
        case PrefixReusePath::PrivateResponseReplay:
            ++cumulative_stats_.private_response_replay_selections;
            break;
        case PrefixReusePath::PrivateLongAnchor:
            ++cumulative_stats_.private_long_anchor_selections;
            break;
        case PrefixReusePath::SharedStablePrefix:
            ++cumulative_stats_.shared_stable_prefix_selections;
            break;
        }
        cumulative_stats_.reused_prompt_tokens += summary.reusable_prompt_tokens;
        cumulative_stats_.last_selected_frontier_tokens = summary.reusable_prompt_tokens;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            EngineCore* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<typename Request::StreamEvent> events;
        for (;;) {
            start.reset();
            progress.reset();
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->response_done || request->stream_start.has_value() ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                start = std::move(request->stream_start);
                request->stream_start.reset();
                progress = std::move(request->stream_progress);
                request->stream_progress.reset();
                events.swap(request->events);
                done = request->response_done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (start) { sink->start(std::move(*start)); }
                    if (progress) { sink->progress(std::move(*progress)); }
                    for (auto& event : events) {
                        if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                            sink->timing(std::move(*timing));
                        } else {
                            sink->publish(std::move(std::get<OutputDelta>(event)));
                        }
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    request_admission_check();
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        request_admission_check();
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    request_admission_check();
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
    };

    // Coalesces admission-visible queue/resource changes. Ordinary prefill/decode progress does not
    // re-arm a temporarily blocked inspection.
    void request_admission_check() noexcept {
        admission_check_pending_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool consume_admission_check() noexcept {
        return admission_check_pending_.exchange(false, std::memory_order_acq_rel);
    }

    struct MaterializingRequest {
        std::shared_ptr<Request> request;
        LaneId destination;
        GenerationBudget budget;
        RequestPlanSummary summary;
        BackfillClass backfill_class   = BackfillClass::None;
        std::uint64_t protection_epoch = 0;
        Clock::time_point started;
    };

    [[nodiscard]] std::optional<GenerationTimingObservation>
    record_committed_output(const std::shared_ptr<Request>& request,
                            std::uint32_t accepted_tokens) {
        if (accepted_tokens == 0) { return std::nullopt; }
        const bool observe_wall =
            request->observation.phase_timings || request->observation.live_timings;
        const bool need_now         = !request->first_token || observe_wall;
        const Clock::time_point now = need_now ? Clock::now() : Clock::time_point{};
        if (!request->first_token) { request->first_token = now; }
        if (!observe_wall) { return std::nullopt; }
        if (!request->admitted_at || !request->first_token) {
            throw std::logic_error("committed output has no observed admission boundary");
        }
        request->last_token = now;
        if (!request->observation.live_timings) { return std::nullopt; }
        if (request->generated.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("generated token count exceeds observation domain");
        }
        return GenerationTimingObservation{
            .generated_tokens      = static_cast<std::uint32_t>(request->generated.size()),
            .prompt_elapsed_ns     = elapsed_ns(*request->admitted_at, *request->first_token),
            .generation_elapsed_ns = elapsed_ns(*request->first_token, *request->last_token),
        };
    }

    void append_output(const std::shared_ptr<Request>& request, PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt) {
        apply_post_thinking_sampling(*request);
        if (output.empty() && !timing) { return; }
        const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
        {
            std::lock_guard lock(request->mutex);
            if (streaming && timing) { request->events.emplace_back(std::move(*timing)); }
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                if (streaming) { request->events.emplace_back(std::move(delta)); }
            }
        }
        if (streaming) { request->cv.notify_one(); }
    }

    // Every commit ends here, whether the model or a thinking-budget control closed the reasoning
    // block, so the lane switches before its next round samples.
    void apply_post_thinking_sampling(Request& request) {
        if (request.post_thinking_applied || !request.options.execution.post_thinking_sampling ||
            !request.sequence || !request.output.reasoning_closed()) {
            return;
        }
        instance_.program->update_sampling(*request.sequence,
                                           *request.options.execution.post_thinking_sampling);
        request.post_thinking_applied = true;
    }

    void publish_prompt_progress(const std::shared_ptr<Request>& request) {
        if (!request->observation.prompt_progress) { return; }
        if (!request->admitted_begin || !request->admitted_at) {
            throw std::logic_error("prompt progress has no admitted request boundary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens ||
            request->computed_prompt_tokens > begin.prompt_tokens - begin.reused_prompt_tokens) {
            throw std::logic_error("prompt progress exceeds the admitted prompt frontier");
        }
        const PromptProgress progress{
            .total_prompt_tokens     = begin.prompt_tokens,
            .reused_prompt_tokens    = begin.reused_prompt_tokens,
            .processed_prompt_tokens = begin.reused_prompt_tokens + request->computed_prompt_tokens,
            .elapsed_ns              = elapsed_ns(*request->admitted_at, Clock::now()),
        };
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->stream_progress = progress;
        }
        request->cv.notify_one();
    }

    void publish_generation_start(const std::shared_ptr<Request>& request, BeginSummary begin) {
        if (request->admitted_begin) {
            throw std::logic_error("request admission published generation start twice");
        }
        request->admitted_begin = begin;
        if (request->observation.phase_timings || request->observation.live_timings ||
            request->observation.prompt_progress) {
            request->admitted_at = Clock::now();
        }
        if (request->consumer_mode != OutputConsumerMode::Streaming) { return; }
        {
            std::lock_guard lock(request->mutex);
            if (request->stream_start || request->response_done) {
                throw std::logic_error("streaming request has an invalid generation-start state");
            }
            request->stream_start = GenerationStart{
                .prompt               = request->prompt_summary,
                .reused_prompt_tokens = begin.reused_prompt_tokens,
            };
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->response_done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        request_admission_check();
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->ngram_archive.reset();
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->error         = std::move(error);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    // Best-effort noexcept completion for OOM/fatal paths.  Clears request
    // state (prompt, base_plan, sequence, lane, budget, terminal_reason),
    // releases reserved capacity if the consumer already left, sets
    // response_done, and notifies the consumer.  Each step is individually
    // guarded so a throw on one step does not skip the rest.
    void force_complete_error(const std::shared_ptr<Request>& request,
                              const std::exception_ptr& error) noexcept {
        try { request->prompt = {}; } catch (...) {}
        try { request->base_plan.reset(); } catch (...) {}
        try {
            request->ngram_archive.reset();
        } catch (...) {}
        try { request->model_state = EngineRequestState::ModelFinished; } catch (...) {}
        try { request->sequence.reset(); } catch (...) {}
        try { request->lane.reset(); } catch (...) {}
        try { request->budget.reset(); } catch (...) {}
        try { request->terminal_reason.reset(); } catch (...) {}
        bool release_capacity = false;
        try {
            std::lock_guard lock(request->mutex);
            if (!request->response_done) {
                request->error         = error;
                request->response_done = true;
            }
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release_capacity            = true;
            }
        } catch (...) {}
        if (release_capacity) { try { release_reserved_capacity(); } catch (...) {} }
        try { request->cv.notify_one(); } catch (...) {}
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        HostPhaseMeasurement completion = begin_host_phase();
        double prompt_wall_seconds      = 0.0;
        double generation_wall_seconds  = 0.0;
        if (request->observation.phase_timings && request->first_token) {
            if (!request->admitted_at || !request->last_token) {
                throw std::logic_error("observed request completed without stable timing bounds");
            }
            prompt_wall_seconds =
                std::chrono::duration<double>(*request->first_token - *request->admitted_at)
                    .count();
            generation_wall_seconds =
                std::chrono::duration<double>(*request->last_token - *request->first_token).count();
        }
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        if (!request->queue_wait_recorded) {
            request->host_timing.queue_wait_ns = elapsed_ns(request->submitted, Clock::now());
            request->queue_wait_recorded       = true;
        }
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.tool_calls              = request->output.take_tool_calls();
        result.tool_call_parse         = request->output.tool_call_parse_diagnostics();
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.matched_stop_string     = request->output.matched_stop_string();
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        result.timings                 = request->generation_timings;
        result.timings.prepare_seconds = request->prepare_seconds;
        result.speculative             = std::move(request->speculative_stats);
        result.first_token_logprobs    = std::move(request->first_token_logprobs);
        result.thinking                = request->output.thinking_stats();
        result.thinking.post_thinking_sampling = request->post_thinking_applied;
        if (request->ngram_archive) {
            result.ngram_archive.bound = true;
            if (reason != FinishReason::Cancelled &&
                !request->cancelled.load(std::memory_order_acquire)) {
                result.ngram_archive.published =
                    ngram_archive_->publish(std::move(request->ngram_archive),
                                            result.generated_token_ids, result.reasoning_tokens);
            } else {
                request->ngram_archive.reset();
            }
        }
        if (ngram_archive_) {
            auto stats      = ngram_archive_->stats(request->options.ngram_session.key);
            stats.bound     = result.ngram_archive.bound;
            stats.published = result.ngram_archive.published;
            if (!request->options.ngram_session.key.empty()) {
                stats.sampling_seed = request->options.execution.sampling.seed;
            }
            result.ngram_archive = stats;
        }
        result.materialization = request->materialization_diagnostics;
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.prompt_wall_seconds     = prompt_wall_seconds;
        result.timings.generation_wall_seconds = generation_wall_seconds;
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        finish_engine_phase(completion, EngineHostPhase::CommitOutput);
        result.engine_timing = request->host_timing.public_snapshot();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->result        = std::move(result);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    void complete_detached_cancelled(const std::shared_ptr<Request>& request) {
        try {
            complete_cancelled(request);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            complete_error(request, error);
            throw;
        }
    }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        if (scheduler_.owns_prefill_lane(lane)) { scheduler_.clear_prefill_lane(lane); }
        request_admission_check();
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        // An already-issued active unit may finish while another row owns the global resource
        // transaction.  Its cancellation cannot release topology until that transaction reaches
        // a stable terminal state.
        if (instance_.program->has_context_transaction()) { return cancelled; }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    bool settle_terminal_requests(HostPhaseMeasurement& boundary) {
        const bool manager_transaction = resources_.context_transaction_kind().has_value();
        const bool program_transaction = instance_.program->has_context_transaction();
        if (manager_transaction != program_transaction) {
            throw std::logic_error("Engine and Program disagree before terminal settlement");
        }
        if (program_transaction) { return false; }

        bool changed = false;
        for (;;) {
            std::optional<std::uint32_t> selected;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request == nullptr || !request->terminal_reason) { continue; }
                if (!selected ||
                    request->publication_order < slots_[*selected]->publication_order) {
                    selected = lane;
                }
            }
            if (!selected) { break; }

            const std::uint32_t lane = *selected;
            const auto request       = slots_[lane];
            if (!request->is_model_finished() || request->capture_pending || !request->sequence ||
                !request->lane || request->lane->value != lane ||
                resources_.lane_state(LaneId{lane}) != LogicalLaneState::TerminalPending) {
                throw std::logic_error("terminal-pending request has invalid ownership");
            }
            const FinishReason reason = *request->terminal_reason;
            auto finished =
                resources_.finish(*instance_.program, *request->lane, *request->sequence);
            request->generation_timings = finished.timings;
            request->speculative_stats  = std::move(finished.speculative);
            if (finished.salvaged) { ++cumulative_stats_.salvaged_continuations; }
            request->terminal_reason.reset();

            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            // Free the slot and publish the post-release snapshot before waking the caller so
            // runtime_stats() read after generate() returns reflects the released lane.
            remove_completed_slot(lane);
            publish_runtime_stats();
            complete_success(request, reason);
            boundary = begin_host_phase();
            changed  = true;
        }
        return changed;
    }

    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return; }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            // A copy: remove_completed_slot below resets the slot, and complete_success must
            // still reach the request.
            const auto request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            if (request->capture_pending) { continue; }
            if (!request->sequence || !request->lane || request->lane->value != lane) {
                throw std::logic_error("active cancellation has no sequence binding");
            }
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            auto aborted = resources_.abort(*instance_.program, *request->lane, *request->sequence);
            request->generation_timings = aborted.timings;
            request->speculative_stats  = std::move(aborted.speculative);
            if (aborted.salvaged) { ++cumulative_stats_.salvaged_continuations; }
            if (scheduler_.owns_prefill_lane(lane)) { scheduler_.clear_prefill_lane(lane); }
            append_output(request, request->output.commit_preview());
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            // Free the slot and publish the post-release snapshot before waking the caller so
            // runtime_stats() read after generate() returns reflects the released lane.
            remove_completed_slot(lane);
            publish_runtime_stats();
            complete_success(request, FinishReason::Cancelled);
            boundary = begin_host_phase();
        }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        for (const auto& request : cancelled) { scheduler_.on_waiting_removed(request->id); }
        for (const auto& request : expired) { scheduler_.on_waiting_removed(request->id); }
        try {
            for (const auto& request : cancelled) { complete_detached_cancelled(request); }
            for (const auto& request : expired) {
                complete_error(request,
                               std::make_exception_ptr(RequestError(
                                   RequestErrorKind::QueueTimeout,
                                   "inference request expired while waiting for admission")));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (const auto& request : cancelled) { complete_error(request, error); }
            for (const auto& request : expired) { complete_error(request, error); }
            throw;
        }
        if (!cancelled.empty() || !expired.empty()) {
            request_admission_check();
            publish_runtime_stats();
        }
        return have_pending;
    }

    void commit_pending(PendingBatch&& pending, std::span<const std::uint32_t> lane_indices,
                        bool decode_round,
                        const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        const std::size_t row_count = lane_indices.size();
        if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
            (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
            pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            std::array<LaneId, kMaximumConcurrency> invalid_lanes{};
            for (std::size_t row = 0; row < row_count; ++row) {
                invalid_lanes[row] = LaneId{lane_indices[row]};
            }
            resources_.apply_discard(std::span<const LaneId>(invalid_lanes.data(), row_count),
                                     discarded);
            throw std::logic_error("pending batch returned an invalid ragged layout");
        }

        std::array<LaneId, kMaximumConcurrency> lanes{};
        std::array<CommitDecision, kMaximumConcurrency> decisions{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        std::array<ContinuationAction, kMaximumConcurrency> continuations{};
        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<bool, kMaximumConcurrency> cancelled{};
        bool generated_staged = false;
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> terminal_requests{};
        std::array<std::uint32_t, kMaximumConcurrency> terminal_lanes{};
        std::array<FinishReason, kMaximumConcurrency> terminal_reasons{};
        std::size_t terminal_count = 0;
        for (std::size_t row = 0; row < row_count; ++row) {
            lanes[row] = LaneId{lane_indices[row]};
        }
        const auto rollback_generated = [&]() noexcept {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < row_count; ++row) {
                const auto& request = slots_[lane_indices[row]];
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };
        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane = lane_indices[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->sequence || !request->lane ||
                    request->lane->value != lane || !request->budget) {
                    throw std::logic_error("pending row has no active Engine request");
                }
                if (decode_round) { ++request->host_timing.decode_rounds; }
                cancelled[row] = cancelled_at_unit_start[lane];
                const std::int32_t raw_count =
                    pending.row_counts().empty() ? 1 : pending.row_counts()[row];
                if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                    throw std::logic_error("pending row has an invalid licensed extent");
                }
                const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
                const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                     static_cast<std::size_t>(count));
                generated_sizes[row]      = request->generated.size();
                if (cancelled[row]) {
                    (void)request->output.preview_terminal(FinishReason::Cancelled);
                    decisions[row] = CommitDecision{
                        .accepted_tokens = 0,
                        .terminal        = true,
                        .cancelled       = true,
                    };
                    finish_reasons[row] = FinishReason::Cancelled;
                    continue;
                }
                const OutputDecision decision = request->output.preview_model(
                    row_tokens, request->budget->remaining(), request->budget->limit_reason());
                if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                    (!decision.finished() && decision.accepted_tokens != count) ||
                    (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("output policy returned an invalid licensed prefix");
                }
                decisions[row] = CommitDecision{
                    .accepted_tokens              = decision.accepted_tokens,
                    .terminal                     = decision.finished(),
                    .cancelled                    = false,
                    .prefix_execution_split_after = decision.prefix_execution_split_after,
                };
                finish_reasons[row] = decision.finish_reason;
                continuations[row]  = decision.continuation;
            }
            generated_staged = true;
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (accepted == 0) { continue; }
                const auto& request = slots_[lane_indices[row]];
                if (request->generated.size() > request->generated.capacity() ||
                    accepted > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error("admission did not reserve generated-token capacity");
                }
                const auto first = pending.tokens().begin() +
                                   static_cast<std::ptrdiff_t>(row * pending.row_stride());
                request->generated.insert(request->generated.end(), first,
                                          first + static_cast<std::ptrdiff_t>(accepted));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            rollback_generated();
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            if (discarded.status == ConsumeStatus::Consumed) {
                resources_.apply_discard(std::span<const LaneId>(lanes.data(), row_count),
                                         discarded);
            } else if (!instance_.program->has_context_transaction()) {
                throw std::logic_error("Program could not abort a failed pending batch");
            }
            std::rethrow_exception(error);
        }

        std::optional<typename ModelContract::CommitResult> committed_storage;
        try {
            phase.pause_range();
            ProgramCallScope program_call(*this);
            auto committed = instance_.program->commit(
                std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count),
                CommitObservation::ReleasedRowsOnly, &program_call.failed_timing());
            program_call.finish(committed.timing);
            committed_storage.emplace(std::move(committed));
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            if (!instance_.program->has_context_transaction()) {
                resources_.release_failed_commit(std::span<const LaneId>(lanes.data(), row_count));
            }
            throw;
        }
        generated_staged = false;
        auto& committed  = *committed_storage;
        if (committed.row_count != row_count) {
            throw std::logic_error("Runtime commit result is not row aligned");
        }
        for (std::size_t row = 0; row < row_count; ++row) {
            const CommitDisposition expected = cancelled[row] ? CommitDisposition::CancelledReleased
                                               : decisions[row].terminal
                                                   ? CommitDisposition::Finishable
                                                   : CommitDisposition::Active;
            if (committed.rows[row].disposition != expected) {
                throw std::logic_error("Runtime commit row disposition is invalid");
            }
            if (committed.captures[row].has_value() &&
                (decode_round || expected != CommitDisposition::Active)) {
                throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
            }
        }
        resources_.apply_commit(std::span<const LaneId>(lanes.data(), row_count), committed);
        const bool terminal_in_batch = std::any_of(
            decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
            [](const CommitDecision& decision) { return decision.terminal; });

        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request = slots_[lane_indices[row]];
            if (cancelled[row]) {
                request->generation_timings = committed.rows[row].timings;
                request->speculative_stats  = std::move(committed.rows[row].speculative);
            }
        }

        if (decode_round) {
            ++cumulative_stats_.decode_rounds;
            cumulative_stats_.decode_row_rounds += row_count;
            for (std::size_t row = 0; row < row_count; ++row) {
                if (!cancelled[row]) {
                    cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                }
            }
        }

        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane     = lane_indices[row];
                const auto& request          = slots_[lane];
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (!cancelled[row]) {
                    request->budget->commit(accepted);
                    if (decode_round) { Scheduling::consume_service_work(*request, accepted); }
                }
                auto published = request->output.commit_preview();
                auto timing    = record_committed_output(request, accepted);
                append_output(request, std::move(published), std::move(timing));
                if (decisions[row].terminal) {
                    if (cancelled[row]) {
                        terminal_requests[terminal_count] = request;
                        terminal_lanes[terminal_count]    = lane;
                        terminal_reasons[terminal_count]  = finish_reasons[row];
                        ++terminal_count;
                    } else {
                        request->model_state     = EngineRequestState::ModelFinished;
                        request->terminal_reason = finish_reasons[row];
                    }
                } else if (committed.captures[row]) {
                    if (!request->is_prefilling()) {
                        throw std::logic_error("prompt-frontier capture lost its prefill owner");
                    }
                    const EngineRequestState post_capture_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                    if (terminal_in_batch) {
                        instance_.program->skip_capture(std::move(*committed.captures[row]));
                        request->model_state = post_capture_state;
                    } else {
                        reserve_active_capture(request, std::move(*committed.captures[row]),
                                               post_capture_state);
                    }
                    committed.captures[row].reset();
                    if (!request->capture_pending) {
                        request->model_state =
                            continuations[row] == ContinuationAction::ApplyTargetControl
                                ? EngineRequestState::ControlReady
                                : EngineRequestState::DecodeReady;
                    }
                } else {
                    request->model_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                }
            }
        } catch (...) {
            phase.finish();
            for (std::size_t index = 0; index < terminal_count; ++index) {
                complete_success(terminal_requests[index], terminal_reasons[index]);
                remove_completed_slot(terminal_lanes[index]);
            }
            throw;
        }
        phase.finish();
        for (std::size_t index = 0; index < terminal_count; ++index) {
            complete_success(terminal_requests[index], terminal_reasons[index]);
            remove_completed_slot(terminal_lanes[index]);
        }
    }

    [[nodiscard]] std::shared_ptr<Request> active_capture_owner() const {
        std::shared_ptr<Request> request;
        for (std::uint32_t candidate = 0; candidate < max_concurrency_; ++candidate) {
            if (slots_[candidate] == nullptr || !slots_[candidate]->capture_pending) { continue; }
            if (request != nullptr) {
                throw std::logic_error("multiple requests own one active-capture transaction");
            }
            request = slots_[candidate];
            if (!request->lane || request->lane->value != candidate || !request->sequence) {
                throw std::logic_error("capture-pending request has no active sequence binding");
            }
        }
        return request;
    }

    void reserve_active_capture(const std::shared_ptr<Request>& request, CaptureOffer&& offer,
                                EngineRequestState post_capture_state) {
        if (!request->lane || !request->sequence || request->capture_pending ||
            post_capture_state == EngineRequestState::Materializing ||
            post_capture_state == EngineRequestState::Waiting ||
            post_capture_state == EngineRequestState::ModelFinished) {
            throw std::logic_error("committed capture offer has invalid Engine ownership");
        }
        std::uint64_t blocked = 0;
        {
            std::lock_guard lock(queue_mutex_);
            blocked = pending_.size();
        }
        for (const auto& active : slots_) {
            if (active != nullptr && active != request && !active->terminal_reason) { ++blocked; }
        }
        const std::uint32_t blocked_runnable_requests =
            blocked > std::numeric_limits<std::uint32_t>::max()
                ? std::numeric_limits<std::uint32_t>::max()
                : static_cast<std::uint32_t>(blocked);
        const auto reserved = resources_.reserve_active_capture(
            *instance_.program, *request->lane, std::move(offer), blocked_runnable_requests,
            CancellationFlagView{&request->cancelled});
        if (reserved == ResourceManagement::ActiveCaptureReserveResult::Skipped) {
            ++cumulative_stats_.active_captures_skipped;
            return;
        }
        request->capture_pending    = true;
        request->post_capture_state = post_capture_state;
        (void)progress_context_transaction(false);
    }

    void
    resolve_prefill_progress(const std::shared_ptr<Request>& request,
                             typename ModelContract::PrefillProgress&& progress,
                             const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        ++cumulative_stats_.host_work.prefill_units;
        ++request->host_timing.prefill_units;
        cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
        Scheduling::consume_service_work(*request, 1);
        if (!request->admitted_begin) {
            throw std::logic_error("prefill progress has no committed admission summary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens) {
            throw std::logic_error("admitted prefix exceeds its prompt");
        }
        const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
        if (request->computed_prompt_tokens > suffix_tokens ||
            progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
            throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
        }
        request->computed_prompt_tokens += progress.processed_prompt_tokens;
        if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
            throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
        }
        if (progress.processed_prompt_tokens != 0) { publish_prompt_progress(request); }
        if (progress.capture) {
            if (progress.complete || progress.pending) {
                throw std::logic_error("prefill capture offer overlaps prompt completion");
            }
            reserve_active_capture(request, std::move(*progress.capture),
                                   EngineRequestState::Prefill);
            progress.capture.reset();
            return;
        }
        if (!progress.complete) { return; }
        if (!request->lane || !progress.pending) {
            throw std::logic_error("completed prefill has no lane or pending token");
        }
        if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
            throw std::logic_error("runtime Begin summary differs from committed admission");
        }
        const std::uint32_t lane = request->lane->value;
        if (scheduler_.owns_prefill_lane(lane)) {
            scheduler_.clear_prefill_lane(lane);
            request_admission_check();
        }
        request->begin                = progress.summary;
        request->first_token_logprobs = std::move(progress.first_token_logprobs);
        const std::array<std::uint32_t, 1> lanes{lane};
        phase.finish();
        commit_pending(std::move(*progress.pending), lanes, false, cancelled_at_unit_start);
        progress.pending.reset();
    }

    // The lowest lane owning staged prefill that can advance now. A lane offering an active
    // capture, or whose media item still encodes in a concurrent overlay window, yields its unit.
    [[nodiscard]] std::optional<std::uint32_t> runnable_prefill_lane() const {
        return scheduler_.select_runnable_prefill_lane(max_concurrency_, [&](std::uint32_t lane) {
            const auto& request = slots_[lane];
            return request != nullptr && request->is_prefilling() && !request->capture_pending &&
                   !(request->sequence && instance_.program->vision_pending(*request->sequence));
        });
    }

    void run_prefill_step(const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange prefill_range(nvtx::Name::Prefill, nvtx::Category::Prefill);
        EnginePhaseScope setup(*this, EngineHostPhase::CommitOutput);
        const auto prefill_lane = runnable_prefill_lane();
        if (!prefill_lane) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane;
        const auto request       = slots_[lane];
        if (request == nullptr || !request->is_prefilling() || request->capture_pending) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        if (!request->sequence) {
            throw std::logic_error("prefill request has no sequence handle");
        }
        setup.finish();
        ProgramCallScope program_call(*this);
        auto progress =
            instance_.program->advance_prefill(*request->sequence, &program_call.failed_timing());
        program_call.finish(progress.timing);
        resolve_prefill_progress(request, std::move(progress), cancelled_at_unit_start);
        // A completed prefill already re-arms admission (owner cleared above). Re-arm again when
        // the request keeps prefilling: each prefill boundary is an admission point, so a waiting
        // request can be admitted to a free lane while another request is still prefilling.
        if (!progress.complete && request->is_prefilling()) { request_admission_check(); }
        publish_runtime_stats();
    }

    [[nodiscard]] FifoSnapshot pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return Scheduling::fifo_snapshot(pending_);
    }

    [[nodiscard]] bool has_pending_requests() const {
        std::lock_guard lock(queue_mutex_);
        return !pending_.empty();
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void on_waiting_removed(const std::shared_ptr<Request>& request) noexcept {
        scheduler_.on_waiting_removed(request->id);
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    [[nodiscard]] ResourceInspection inspect_admission(const std::shared_ptr<Request>& request,
                                                       PlanningAllowance allowance) {
        allowance.cancellation = &request->cancelled;
        allowance.control_deadline_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           request->deadline.time_since_epoch())
                                           .count());
        return resources_.inspect(*instance_.program, request->prompt, *request->base_plan,
                                  request->publication_order, allowance);
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        on_waiting_removed(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress progress_context_transaction(bool yield_requested) {
        const std::optional<ContextTransactionKind> kind = resources_.context_transaction_kind();
        if (kind.has_value() != instance_.program->has_context_transaction()) {
            throw std::logic_error("Engine and Program disagree on context-transaction ownership");
        }
        if (!kind) { return AdmissionProgress::None; }
        DetailScope detail(*this, &RuntimeHostWorkStats::context_progress_ns,
                           &RuntimeHostWorkStats::context_progress_invocations,
                           nvtx::Name::ContextProgress);
        ++cumulative_stats_.host_work.control_units;

        const std::shared_ptr<Request> capture = active_capture_owner();
        std::atomic<bool> yield{yield_requested};
        CancellationFlagView cancellation{&yield};
        switch (*kind) {
        case ContextTransactionKind::Materialization: {
            if (!materializing_ || capture) {
                throw std::logic_error("materialization has conflicting Engine ownership");
            }
            const MaterializingRequest& control = *materializing_;
            const std::uint32_t lane            = control.destination.value;
            const auto& request                 = control.request;
            if (request == nullptr || !request->is_materializing() || request->lane ||
                request->sequence || request->budget || lane >= max_concurrency_ ||
                slots_[lane] != nullptr) {
                throw std::logic_error("materializing request has invalid Engine ownership");
            }
            cancellation = CancellationFlagView{&request->cancelled};
            break;
        }
        case ContextTransactionKind::ActiveCapture:
            if (materializing_ || !capture) {
                throw std::logic_error("active capture has conflicting Engine ownership");
            }
            cancellation = CancellationFlagView{&capture->cancelled};
            break;
        }

        auto outcome = resources_.progress_context_transaction(*instance_.program, cancellation);
        return std::visit(
            [&](auto&& terminal) -> AdmissionProgress {
                using Outcome = std::decay_t<decltype(terminal)>;
                if constexpr (std::is_same_v<Outcome, ContextTransactionInProgress>) {
                    return AdmissionProgress::ControlProgress;
                } else if constexpr (std::is_same_v<
                                         Outcome,
                                         typename ResourceManagement::MaterializationOutcome>) {
                    if (*kind != ContextTransactionKind::Materialization || !materializing_) {
                        throw std::logic_error(
                            "materialization outcome has no Engine control record");
                    }
                    MaterializingRequest& control = *materializing_;
                    const std::uint32_t lane      = control.destination.value;
                    const auto request            = control.request;
                    if (terminal.status == ContextTransactionStatus::Aborted) {
                        if (terminal.activation) {
                            throw std::logic_error(
                                "aborted materialization retained an activation");
                        }
                        materializing_.reset();
                        if (!terminal.failure.empty()) {
                            ++cumulative_stats_.context_cache_exhausted_requests;
                            consecutive_context_cache_exhaustions_.fetch_add(
                                1, std::memory_order_relaxed);
                            complete_error(request, std::make_exception_ptr(RequestError(
                                                        RequestErrorKind::Overloaded,
                                                        "context cache exhausted: " +
                                                            terminal.failure)));
                        } else {
                            complete_detached_cancelled(request);
                        }
                        request_admission_check();
                        publish_runtime_stats();
                        return AdmissionProgress::ControlProgress;
                    }
                    if (terminal.status != ContextTransactionStatus::Published ||
                        !terminal.activation) {
                        throw std::logic_error("published materialization has no adoption token");
                    }
                    auto activation = std::move(*terminal.activation);
                    terminal.activation.reset();
                    const SequenceHandle sequence = activation.sequence();
                    resources_.adopt(*instance_.program, std::move(activation));
                    consecutive_context_cache_exhaustions_.store(0, std::memory_order_relaxed);
                    request->sequence.emplace(sequence);
                    request->budget.emplace(std::move(control.budget));
                    request->lane.emplace(control.destination);
                    request->remaining_service_work      = control.summary.service_work_quanta;
                    request->backfill_epoch              = control.protection_epoch;
                    request->backfill_class              = control.backfill_class;
                    request->materialization_diagnostics = terminal.diagnostics;
                    request->model_state                 = EngineRequestState::Prefill;
                    request->host_timing.queue_wait_ns =
                        elapsed_ns(request->submitted, Clock::now());
                    request->queue_wait_recorded = true;
                    slots_[lane]                 = request;
                    record_prefix_selection(control.summary);
                    materializing_.reset();
                    scheduler_.set_prefill_lane(lane);
                    request_admission_check();
                    publish_runtime_stats();
                    return AdmissionProgress::ControlProgress;
                } else if constexpr (std::is_same_v<
                                         Outcome,
                                         typename ResourceManagement::ActiveCaptureOutcome>) {
                    if (*kind != ContextTransactionKind::ActiveCapture || !capture) {
                        throw std::logic_error("active-capture outcome has no Engine owner");
                    }
                    if (terminal.status == ContextTransactionStatus::Published) {
                        ++cumulative_stats_.active_captures_completed;
                    } else if (terminal.status == ContextTransactionStatus::Aborted) {
                        ++cumulative_stats_.active_captures_aborted;
                    } else {
                        throw std::logic_error("active capture returned an invalid terminal state");
                    }
                    capture->capture_pending = false;
                    capture->model_state     = capture->post_capture_state;
                    request_admission_check();
                    publish_runtime_stats();
                    return AdmissionProgress::ControlProgress;
                }
                throw std::logic_error("unknown resource transaction outcome");
            },
            std::move(outcome));
    }

    [[nodiscard]] AdmissionProgress
    admit_planned_request(const std::shared_ptr<Request>& request,
                          typename ResourceManagement::Choice&& choice, AdmissionGrant grant) {
        if (Clock::now() >= request->deadline) {
            const AdmissionProgress progress = remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
            if (progress == AdmissionProgress::ControlProgress) { request_admission_check(); }
            return progress;
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            on_waiting_removed(request);
            complete_detached_cancelled(request);
            request_admission_check();
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const LaneId destination         = choice.destination();
        const std::uint32_t lane         = destination.value;
        const RequestPlanSummary summary = choice.summary();
        if (grant.request_id() != request->id ||
            grant.service_work_quanta() != summary.service_work_quanta ||
            !scheduler_.validate_grant(grant)) {
            throw std::logic_error("admission choice lost its Scheduler grant");
        }
        GenerationBudget prepared_budget(summary.effective_output_tokens,
                                         summary.effective_limit_reason);
        try {
            request->generated.reserve(summary.effective_output_tokens);
        } catch (...) {
            const AdmissionProgress progress =
                remove_pending_error(request, std::current_exception());
            if (progress == AdmissionProgress::ControlProgress) { request_admission_check(); }
            return progress;
        }
        MaterializingRequest control{
            .request          = request,
            .destination      = destination,
            .budget           = std::move(prepared_budget),
            .summary          = summary,
            .backfill_class   = grant.backfill_class(),
            .protection_epoch = grant.protection_epoch(),
            .started          = Clock::now(),
        };

        if (ngram_archive_ && !request->ngram_admission_checked) {
            request->ngram_admission_checked = true;
            request->ngram_archive =
                request->prompt.bind_ngram(*ngram_archive_, request->options.ngram_session);
        }
        typename ResourceManagement::MaterializationReserveResult reserved;
        try {
            reserved = resources_.reserve_materialization(
                *instance_.program, std::move(choice), std::move(request->prompt),
                CancellationFlagView{&request->cancelled});
        } catch (const std::bad_alloc& oom) {
            publish_diagnostic(diagnostics_, DiagnosticLevel::Warning,
                               "out of memory during materialization reserve: %s", oom.what());
            oom_backoff_ = kOomBackoffIterations;
            ++oom_recovery_count_;
            if (!erase_pending(request)) {
                // The prompt was already moved into reserve_materialization before the
                // throw, so the request is unrecoverable.  This is a logic error — the
                // request must have been in pending_ — but we cannot serve it with a
                // moved-from prompt.  Fail hard rather than silently producing garbage.
                throw std::logic_error("OOM admission lost its waiting request");
            }
            on_waiting_removed(request);
            force_complete_error(request, oom_fallback_error_);
            try { publish_runtime_stats(); } catch (...) {}
            request_admission_check();
            return AdmissionProgress::ControlProgress;
        }
        if (reserved == ResourceManagement::MaterializationReserveResult::Stale) {
            request_admission_check();
            return AdmissionProgress::ControlProgress;
        }
        if (reserved == ResourceManagement::MaterializationReserveResult::Aborted) {
            if (!erase_pending(request)) {
                throw std::logic_error("aborted materialization lost its waiting request");
            }
            on_waiting_removed(request);
            complete_detached_cancelled(request);
            request_admission_check();
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }
        if (!erase_pending(request)) {
            throw std::logic_error("admitted request disappeared from the FIFO queue");
        }
        release_planning_state(request);
        if (materializing_ || slots_[lane] != nullptr) {
            throw std::logic_error("reserved materialization destination is not empty");
        }
        request->model_state = EngineRequestState::Materializing;
        materializing_.emplace(std::move(control));
        scheduler_.commit_admission(std::move(grant));
        publish_generation_start(
            request, BeginSummary{.prompt_tokens        = summary.prompt_tokens,
                                  .reused_prompt_tokens = summary.reusable_prompt_tokens,
                                  .prefix_reuse_path    = summary.prefix_reuse_path});

        publish_runtime_stats();
        return progress_context_transaction(false);
    }

    AdmissionProgress try_admit_one() {
        const auto other_runnable = static_cast<std::uint32_t>(
            std::count_if(slots_.begin(), slots_.end(), [](const auto& request) {
                return request && !request->capture_pending &&
                       (request->is_decode_ready() || request->is_prefilling());
            }));
        const PlanningAllowance allowance = PlanningAllowance::boundary(
            other_runnable, planning_now_ns(), thorough_admission_search_);
        DetailScope detail(*this, &RuntimeHostWorkStats::admission_policy_ns,
                           &RuntimeHostWorkStats::admission_policy_invocations,
                           nvtx::Name::AdmissionPolicy);
        bool control_progress = false;
        for (;;) {
            const FifoSnapshot queued = pending_snapshot();
            if (queued.empty()) {
                scheduler_.observe_fifo_head(std::nullopt);
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.head();
            scheduler_.observe_fifo_head(head->id);
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    on_waiting_removed(head);
                    complete_detached_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            // Inspection is read-only: a throw here (e.g. a planning invariant or an arena
            // overflow) must fail only this waiting request, not the whole Engine. A single
            // bad admission must never accumulate worker recoveries into a global fail-all.
            // The inspection result is move-only, so it is emplaced in place rather than
            // assigned after a default construction.
            std::optional<ResourceInspection> head_inspection;
            try {
                head_inspection.emplace(inspect_admission(head, allowance));
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_inspection->readiness == Readiness::PermanentlyInfeasible) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }
            if (head_inspection->readiness == Readiness::Ready ||
                head_inspection->readiness == Readiness::NeedsTransfer) {
                if (!head_inspection->choice) {
                    throw std::logic_error("ready resource inspection has no admission choice");
                }
                AdmissionGrant grant = scheduler_.grant_head(
                    head->id, head_inspection->choice->summary().service_work_quanta);
                return admit_planned_request(head, std::move(*head_inspection->choice),
                                             std::move(grant));
            }

            const ActiveAdmissionSet active =
                scheduler_.active_admission_set(slots_, max_concurrency_);
            if (active.size == 0) {
                // No active lane will release anything, so the head cannot wait its way in. Fail
                // only this request: a throw would stop the worker and refuse every later request.
                ++cumulative_stats_.context_cache_exhausted_requests;
                consecutive_context_cache_exhaustions_.fetch_add(1, std::memory_order_relaxed);
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::Overloaded,
                              "context cache exhausted: an idle Engine found no placement")));
                control_progress = true;
                continue;
            }
            if (!scheduler_.protect_blocked_head(head->id, active.span(),
                                                 instance_.program->resource_revision())) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::optional<std::uint64_t> protection_epoch = scheduler_.protection_epoch();
            if (!protection_epoch) {
                throw std::logic_error("blocked FIFO head has no protection epoch");
            }

            std::array<SequenceHandle, kMaximumConcurrency> persistent_borrowers{};
            std::size_t persistent_borrower_count = 0;
            for (const auto& active_request : slots_) {
                if (active_request == nullptr ||
                    active_request->backfill_class != BackfillClass::Persistent ||
                    active_request->backfill_epoch != *protection_epoch) {
                    continue;
                }
                if (!active_request->sequence) {
                    throw std::logic_error("persistent borrower has no sequence reservation");
                }
                persistent_borrowers[persistent_borrower_count++] = *active_request->sequence;
            }

            for (const std::shared_ptr<Request>& candidate : queued.backfill_candidates()) {
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        on_waiting_removed(candidate);
                        complete_detached_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }
                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                // Same per-request isolation as the FIFO head: a throwing inspection rejects
                // only this backfill candidate and leaves the rest of the Engine running.
                // Move-only inspection result, so emplaced in place.
                std::optional<ResourceInspection> candidate_inspection;
                try {
                    candidate_inspection.emplace(inspect_admission(candidate, allowance));
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (candidate_inspection->readiness == Readiness::PermanentlyInfeasible) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }
                if ((candidate_inspection->readiness != Readiness::Ready &&
                     candidate_inspection->readiness != Readiness::NeedsTransfer) ||
                    !candidate_inspection->choice) {
                    continue;
                }
                const auto proof = resources_.prove_persistent_backfill(
                    *instance_.program, *head->base_plan, *candidate_inspection->choice,
                    std::span<const SequenceHandle>(persistent_borrowers.data(),
                                                    persistent_borrower_count));
                if (!proof) { continue; }
                const RequestPlanSummary& candidate_plan = candidate_inspection->choice->summary();
                auto grant =
                    scheduler_.qualify_backfill(candidate->id, candidate_plan.service_work_quanta,
                                                active.span(), proof->resource_revision());
                if (grant) {
                    return admit_planned_request(
                        candidate, std::move(*candidate_inspection->choice), std::move(*grant));
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    void run_decode_round(const RoundMembership& membership,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange decode_range(nvtx::Name::Decode, nvtx::Category::Decode,
                                       static_cast<std::uint64_t>(membership.size));
        ProgramCallScope program_call(*this);
        auto pending = instance_.program->decode(
            membership.sequence_span(), membership.budget_span(), &program_call.failed_timing());
        program_call.finish(pending.execution_timing());
        commit_pending(std::move(pending), membership.lane_span(), true, cancelled_at_unit_start);
        publish_runtime_stats();
    }

    // A sequence whose Device KV lease can no longer grow finishes at the frontier its lease
    // covers. When the pool ran out of space, retained cache is released first so the answer can
    // continue; only a lease that still cannot take its smallest step has its remaining budget
    // bounded, so that finish carries the request's generation limit reason instead of the
    // sequence running past its lease and failing a launch.
    void apply_device_kv_lease_settlements() {
        if (!kv_lease_growth_) { return; }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->is_decode_ready() || !request->sequence ||
                !request->budget) {
                continue;
            }
            if (const auto shortfall =
                    instance_.program->device_kv_lease_shortfall(*request->sequence)) {
                const std::uint32_t released = resources_.reclaim_device_kv_for_lease(
                    *instance_.program, shortfall->main_pages, shortfall->backend_pages);
                if (instance_.program->resume_device_kv_lease(*request->sequence)) {
                    if (released != 0) {
                        publish_diagnostic(diagnostics_, DiagnosticLevel::Info,
                                           "Device KV lease of lane %u extended: released %u "
                                           "retained cache owner(s)",
                                           lane, released);
                    }
                    continue;
                }
                if (!request->lease_shortfall_reported) {
                    request->lease_shortfall_reported = true;
                    publish_diagnostic(diagnostics_, DiagnosticLevel::Warning,
                                       "Device KV lease of lane %u cannot grow after releasing %u "
                                       "retained cache owner(s); still short %u main / %u backend "
                                       "pages for its smallest step, so the answer ends early "
                                       "with finish_reason length",
                                       lane, released, shortfall->main_pages,
                                       shortfall->backend_pages);
                }
            }
            const std::uint32_t control = request->output.control_suffix_tokens();
            const std::optional<std::uint32_t> limit =
                instance_.program->device_kv_lease_settlement_tokens(*request->sequence, control);
            if (limit) { request->budget->cap_remaining(*limit); }
        }
    }

    void run_control_batch(const ControlMembership& membership) {
        nvtx::ScopedRange control_range(nvtx::Name::ControlBatch, nvtx::Category::Control,
                                        static_cast<std::uint64_t>(membership.size));
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        if (membership.empty() || membership.row_stride == 0 ||
            membership.tokens.size() !=
                static_cast<std::size_t>(membership.row_stride) * membership.size) {
            throw std::logic_error("thinking control membership is invalid");
        }

        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        bool generated_staged         = false;
        const auto rollback_generated = [&]() noexcept {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < membership.size; ++row) {
                const auto& request = slots_[membership.lanes[row]];
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };

        for (std::size_t row = 0; row < membership.size; ++row) {
            const auto& request = slots_[membership.lanes[row]];
            if (request == nullptr) {
                throw std::logic_error("thinking control membership lost its request");
            }
            generated_sizes[row] = request->generated.size();
        }
        generated_staged = true;
        try {
            for (std::size_t row = 0; row < membership.size; ++row) {
                const std::uint32_t lane = membership.lanes[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->is_control_ready() ||
                    request->capture_pending || !request->budget || !request->sequence ||
                    !request->lane || request->lane->value != lane) {
                    throw std::logic_error("thinking control row lost its active request");
                }
                const std::span<const TokenId> tokens =
                    std::span<const TokenId>(membership.tokens)
                        .subspan(row * membership.row_stride, membership.row_stride);
                const OutputDecision decision =
                    request->output.preview_control(tokens, request->budget->remaining());
                if (decision.accepted_tokens != membership.row_stride || decision.finished() ||
                    decision.continuation != ContinuationAction::Decode ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("target control preview returned an invalid decision");
                }
                prefix_execution_splits[row] = decision.prefix_execution_split_after;
                if (request->generated.size() > request->generated.capacity() ||
                    tokens.size() > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error(
                        "admission did not reserve thinking-control token capacity");
                }
                request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
            }
            phase.pause_range();
            ProgramCallScope program_call(*this);
            const runtime::ExecutionTiming timing = instance_.program->append_forced_tokens(
                membership.sequence_span(), membership.tokens, membership.row_stride,
                std::span<const std::optional<std::uint32_t>>(prefix_execution_splits.data(),
                                                              membership.size),
                &program_call.failed_timing());
            program_call.finish(timing);
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            throw;
        }
        generated_staged = false;

        ++cumulative_stats_.host_work.control_units;
        for (const std::uint32_t lane : membership.lane_span()) {
            ++slots_[lane]->host_timing.control_units;
        }

        for (std::size_t row = 0; row < membership.size; ++row) {
            const std::uint32_t lane = membership.lanes[row];
            const auto& request      = slots_[lane];
            request->budget->commit(membership.row_stride);
            Scheduling::consume_service_work(*request, membership.row_stride);
            cumulative_stats_.committed_decode_tokens += membership.row_stride;
            auto timing = record_committed_output(request, membership.row_stride);
            append_output(request, request->output.commit_preview(), std::move(timing));
            request->model_state = EngineRequestState::DecodeReady;
        }
        publish_runtime_stats();
    }

    // Recover from any std::bad_alloc in the work loop (and, with recover_invariant_failures, any
    // std::logic_error) by clearing active state and continuing.  The most common trigger is
    // device-KV reservation failure, but host allocations can also trigger it.  Errors active and
    // materializing requests, resets the scheduler and program state, but leaves pending requests
    // in the FIFO so they can retry once memory is freed.  The worker loop continues after this.
    // The worker holds execution_mutex_ across the failing operation and this cleanup.
    void recover_from_oom_locked(std::exception_ptr error) noexcept {
        if (!error) { error = oom_fallback_error_; }
        try { scheduler_.reset(); } catch (...) {}
        const std::shared_ptr<Request> materializing_request =
            materializing_ ? materializing_->request : nullptr;
        try { materializing_.reset(); } catch (...) {}
        cleanup_program_locked();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                auto slot_request = std::move(slots_[lane]);
                slots_[lane].reset();
                force_complete_error(slot_request, error);
            }
        }
        if (materializing_request != nullptr) {
            force_complete_error(materializing_request, error);
        }
        // The failing unit may have consumed the admission check: re-arm it so the still-pending
        // FIFO requests are inspected again without waiting for a new submission.
        request_admission_check();
        try { publish_runtime_stats(); } catch (...) {}
    }

    enum class ProgramCleanup : std::uint8_t {
        Failure,
        // The orderly stop: the Program may persist what it caches before dropping it.
        Shutdown,
    };

    // Drops every Program owner and the Engine's catalog of them. Reports when the Program had to
    // rebuild its context stores because an owner could not be released cleanly.
    void cleanup_program_locked(ProgramCleanup cleanup = ProgramCleanup::Failure) noexcept {
        decltype(instance_.program->fail_all_cleanup()) leaked;
        try {
            leaked = cleanup == ProgramCleanup::Shutdown ? instance_.program->shutdown_cleanup()
                                                         : instance_.program->fail_all_cleanup();
        } catch (...) {}
        try { resources_.clear_after_program_cleanup(); } catch (...) {}
        if (leaked) {
            publish_diagnostic(diagnostics_, DiagnosticLevel::Warning,
                               "recovery rebuilt the context stores: cleanup left %u main / %u "
                               "backend Device KV pages, %u Device and %u Host StateImages and "
                               "%zu Host KV bytes without an owner",
                               leaked->device_main_kv_pages, leaked->device_backend_kv_pages,
                               leaked->device_state_slots, leaked->host_state_slots,
                               leaked->host_kv_bytes);
        }
    }

    // The worker holds execution_mutex_ across the failing operation and this cleanup, so no
    // Program introspection can observe a partially cleared physical state.
    void fail_all_locked(std::exception_ptr error,
                         ProgramCleanup cleanup = ProgramCleanup::Failure) noexcept {
        std::deque<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.swap(pending_);
        }
        try { scheduler_.reset(); } catch (...) {}
        const std::shared_ptr<Request> materializing_request =
            materializing_ ? materializing_->request : nullptr;
        try { materializing_.reset(); } catch (...) {}
        cleanup_program_locked(cleanup);
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                auto slot_request = std::move(slots_[lane]);
                slots_[lane].reset();
                force_complete_error(slot_request, error);
            }
        }
        if (materializing_request != nullptr) {
            force_complete_error(materializing_request, error);
        }
        for (const auto& request : pending) { force_complete_error(request, error); }
        try { publish_runtime_stats(); } catch (...) {}
    }

    void crash_locked(const std::exception_ptr& error) noexcept {
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& e) {
            publish_diagnostic(diagnostics_, DiagnosticLevel::Error, "worker crash: %s", e.what());
        } catch (...) {
            publish_diagnostic(diagnostics_, DiagnosticLevel::Error,
                               "worker crash: unknown exception");
        }
        HostPhaseMeasurement cleanup = begin_host_phase();
        fail_all_locked(error);
        finish_engine_phase(cleanup, EngineHostPhase::Maintenance);
        try {
            publish_runtime_stats();
        } catch (...) {}
    }

    [[nodiscard]] std::shared_ptr<core::ResidentMemorySession> residency_session() const {
        if constexpr (requires { instance_.residency; }) {
            return instance_.residency;
        } else {
            return {};
        }
    }

    void worker_loop() noexcept {
        const auto residency = residency_session();
        core::ResidentMemoryBinding resident_binding(residency);
        Clock::time_point last_residency_check{};
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = materializing_.has_value();
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        if (residency) {
                            queue_cv_.wait_for(lock, std::chrono::milliseconds(250),
                                               [&] { return stopping_ || !pending_.empty(); });
                        } else {
                            queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                        }
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    const auto error = std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down"));
                    std::scoped_lock execution_lock(execution_mutex_);
                    fail_all_locked(error, ProgramCleanup::Shutdown);
                    return;
                }
            }

            std::unique_lock execution_lock(execution_mutex_);
            try {
                if (residency && Clock::now() - last_residency_check >=
                                     std::chrono::milliseconds(250)) {
                    residency->check("runtime boundary");
                    last_residency_check = Clock::now();
                }
                set_host_work_class(HostWorkClass::Control);
                HostPhaseMeasurement boundary = begin_host_phase();
                const bool have_pending       = expire_pending_requests();
                // A pure queue wait changes no state that would publish, so the waits /stats
                // reports are refreshed here.
                if (have_pending &&
                    Clock::now() - last_queue_publication_ >= std::chrono::seconds(1)) {
                    publish_runtime_stats();
                }
                (void)progress_context_transaction(have_pending);
                (void)settle_terminal_requests(boundary);
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary, boundary);
                apply_device_kv_lease_settlements();
                RoundMembership membership =
                    scheduler_.build_round_membership(slots_, max_concurrency_);
                const bool admission_check_pending =
                    admission_check_pending_.load(std::memory_order_acquire);
                const bool skip_admission = oom_backoff_ > 0;
                if (skip_admission) { --oom_backoff_; }
                if (!skip_admission &&
                    scheduler_.should_attempt_admission(
                        have_pending, admission_check_pending, !membership.empty(),
                        previous_unit_was_decode,
                        instance_.program->has_context_transaction()) &&
                    consume_admission_check()) {
                    if (residency) { residency->check("request admission"); }
                    (void)try_admit_one();
                    membership = scheduler_.build_round_membership(slots_, max_concurrency_);
                }

                // Cancellation is sampled once for the execution unit. A request arriving while
                // the GPU unit is in flight is observed at the next worker boundary; commit does
                // not reinterpret an already-issued unit with a later atomic read.
                const auto cancelled_at_unit_start = snapshot_cancellations();
                cancel_active_requests(cancelled_at_unit_start, boundary);
                const ControlMembership control_membership =
                    scheduler_.build_control_membership(slots_, max_concurrency_);
                if (!control_membership.empty()) {
                    set_host_work_class(HostWorkClass::Control);
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    run_control_batch(control_membership);
                    previous_unit_was_decode = true;
                    oom_recovery_count_ = 0;
                    continue;
                }
                membership = scheduler_.build_round_membership(slots_, max_concurrency_);

                const std::uint64_t prefill_mask = scheduler_.prefill_lane_mask();
                for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                    if ((prefill_mask & (1ULL << lane)) != 0 &&
                        (slots_[lane] == nullptr || !slots_[lane]->is_prefilling())) {
                        throw std::logic_error("prefill owner has no active Engine request");
                    }
                }
                const bool prefill_runnable = runnable_prefill_lane().has_value();
                const ExecutionAction action = scheduler_.choose_execution(
                    !membership.empty(), prefill_runnable, previous_unit_was_decode);
                if (action == ExecutionAction::Prefill) {
                    set_host_work_class(HostWorkClass::Prefill);
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    run_prefill_step(cancelled_at_unit_start);
                    previous_unit_was_decode = false;
                    oom_recovery_count_ = 0;
                    continue;
                }
                if (action == ExecutionAction::Decode) {
                    set_host_work_class(HostWorkClass::Decode, membership.lane_span());
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    run_decode_round(membership, cancelled_at_unit_start);
                    previous_unit_was_decode = true;
                    oom_recovery_count_ = 0;
                    continue;
                }
                set_host_work_class(HostWorkClass::Control);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                // Do NOT reset oom_recovery_count_ here: idle iterations and
                // admission-only iterations (where admit_planned_request caught
                // an OOM internally) did not complete a real work unit.  Only
                // successful control/prefill/decode above clears the streak.
                //
                // With no unit to run, a context transfer in flight is what the next boundary
                // waits for. Waiting on it directly resumes admission when the copy lands; the
                // timed wait below lasts a whole timer tick (about 15.6 ms on Windows).
                if (instance_.program->wait_context_transfer()) { continue; }
            } catch (const std::bad_alloc& oom) {
                {
                    char buf[256];
                    buf[0] = '\0';
                    int n = std::snprintf(buf, sizeof(buf), "worker out of memory: %s - recovering",
                                          oom.what());
                    if (n < 0) { n = 0; buf[0] = '\0'; }
                    if (materializing_ && (size_t)n < sizeof(buf)) {
                        n += std::snprintf(buf + n, sizeof(buf) - n, " mat=%llu",
                                           (unsigned long long)materializing_->request->id);
                    }
                    for (std::uint32_t lane = 0; lane < max_concurrency_ && (size_t)n < sizeof(buf);
                         ++lane) {
                        if (slots_[lane]) {
                            n += std::snprintf(buf + n, sizeof(buf) - n, " lane%u=%llu", lane,
                                               (unsigned long long)slots_[lane]->id);
                        }
                    }
                    publish_diagnostic(diagnostics_, DiagnosticLevel::Warning, "%s", buf);
                }
                if (++oom_recovery_count_ > kOomMaxRecoveries) {
                    publish_diagnostic(diagnostics_, DiagnosticLevel::Error,
                                       "worker out of memory: %u consecutive recoveries - failing "
                                       "all pending",
                                       oom_recovery_count_ - 1);
                    const std::exception_ptr fatal_error = oom_fallback_error_;
                    fail_all_locked(fatal_error);
                    return;
                }
                std::exception_ptr oom_error;
                try { oom_error = std::current_exception(); } catch (...) {}
                if (!oom_error) { oom_error = oom_fallback_error_; }
                HostPhaseMeasurement cleanup = begin_host_phase();
                recover_from_oom_locked(oom_error);
                finish_engine_phase(cleanup, EngineHostPhase::Maintenance);
                oom_backoff_ = kOomBackoffIterations;
                // Scheduler state was cleared by recover_from_oom_locked; treat the next
                // iteration as a fresh scheduling boundary (no decode continuity).
                previous_unit_was_decode = false;
                continue;
            } catch (const std::logic_error& invariant) {
                if (!recover_invariant_failures_) {
                    crash_locked(std::current_exception());
                    return;
                }
                publish_diagnostic(diagnostics_, DiagnosticLevel::Warning,
                                   "worker invariant failure: %s - recovering", invariant.what());
                if (++oom_recovery_count_ > kOomMaxRecoveries) {
                    publish_diagnostic(diagnostics_, DiagnosticLevel::Error,
                                       "worker: %u consecutive recoveries - failing all pending",
                                       oom_recovery_count_ - 1);
                    fail_all_locked(std::current_exception());
                    return;
                }
                HostPhaseMeasurement cleanup = begin_host_phase();
                recover_from_oom_locked(std::current_exception());
                finish_engine_phase(cleanup, EngineHostPhase::Maintenance);
                oom_backoff_             = kOomBackoffIterations;
                previous_unit_was_decode = false;
                continue;
            } catch (...) {
                crash_locked(std::current_exception());
                return;
            }
            execution_lock.unlock();
            std::unique_lock wait_lock(queue_mutex_);
            queue_cv_.wait_for(wait_lock, std::chrono::milliseconds(1));
        }
    }

    Instance& instance_;
    DeviceContext& device_;
    const std::uint32_t max_context_;
    const bool structured_output_;
    const std::uint32_t max_concurrency_;
    const bool thorough_admission_search_;
    const bool recover_invariant_failures_;
    const bool kv_lease_growth_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    ResourceManagement resources_;
    std::unique_ptr<NgramArchive> ngram_archive_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_              = 0;
    std::uint64_t next_request_id_        = 1;
    std::uint64_t next_publication_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<MaterializingRequest> materializing_;
    std::atomic<std::uint32_t> consecutive_context_cache_exhaustions_{0};
    Scheduling scheduler_;
    std::atomic<bool> admission_check_pending_{false};
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::array<std::uint32_t, kMaximumConcurrency> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;
    // Worker-thread only: when a snapshot last carried a non-empty queue.
    Clock::time_point last_queue_publication_{};
    RuntimeStats published_stats_;
    bool stopping_ = false;
    bool failed_   = false;
    static constexpr std::uint32_t kOomBackoffIterations = 4;
    static constexpr std::uint32_t kOomMaxRecoveries = 8;  // before failing all pending
    std::uint32_t oom_backoff_                    = 0;  // iterations to skip admission after OOM
    std::uint32_t oom_recovery_count_ = 0;  // consecutive OOMs without a successful work unit
    const std::exception_ptr oom_fallback_error_ = std::make_exception_ptr(
        RequestError(RequestErrorKind::Overloaded, "engine out of memory during execution"));
    DiagnosticObserver diagnostics_;
    std::thread worker_;
};

} // namespace ninfer::runtime
