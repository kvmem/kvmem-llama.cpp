#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/program_impl.h"

#include "core/device.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::models::qwen3_5::detail {
namespace {

constexpr auto kPageTokens = static_cast<std::uint32_t>(kPagedKVPageSize);
// Pages staged per disk transfer, for each KV family.
constexpr std::uint32_t kDiskStagingPages = 32;
// The rewrite seam and at most this many of the earliest long anchors are written with an owner:
// a follow-up turn re-renders the conversation, so boundaries inside generated text rarely match
// and the early, stable boundaries are the ones a later prompt shares.
constexpr std::size_t kSpilledAnchors = 3;
// A restore that takes longer than recomputing would is abandoned for the recompute.
constexpr auto kRestoreBudget = std::chrono::seconds(8);
// An owner released outside a materialization is queued within this budget.
constexpr auto kReleaseSpillBudget = std::chrono::seconds(20);
// Writing the resident continuations at shutdown stops after this long.
constexpr auto kShutdownSpillBudget = std::chrono::seconds(60);

void synchronize(const RankStreams& streams) {
    for (std::size_t rank = 0; rank < streams.size(); ++rank) {
        CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }
}

std::string disk_profile_directory(KvCacheStorage storage, std::size_t main_stride,
                                   std::size_t backend_stride, std::size_t state_bytes,
                                   const ops::RopeYarn& yarn) {
    std::string name = "kv" + std::to_string(static_cast<int>(storage)) + "_main" +
                       std::to_string(main_stride) + "_backend" + std::to_string(backend_stride) +
                       "_state" + std::to_string(state_bytes);
    // YaRN changes every rotated key, so its KV must never meet an unscaled run's.
    if (yarn.factor > 1.0F) {
        name += "_yarn" + std::to_string(std::lround(yarn.factor * 1000.0F)) + "_" +
                std::to_string(yarn.native_context);
    }
    if (yarn.interpolation_factor > 1.0F) {
        name += "_pi" + std::to_string(std::lround(yarn.interpolation_factor * 1000.0F)) + "_" +
                std::to_string(yarn.interpolation_threshold);
    }
    return name;
}

} // namespace

void ProgramImpl::open_disk_tier(const SequencePlanImpl& plan) {
    if (plan.context_cache.disk_kv_path.empty() || !context_cache.enabled) { return; }
    const HostKVPageLayout main_layout =
        plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry());
    std::vector<HostKVPageLayout> layouts{main_layout};
    std::optional<HostKVPageLayout> backend_layout;
    if (backend_kv_pages) {
        backend_layout = plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry());
        if (*backend_layout != main_layout) { layouts.push_back(*backend_layout); }
    }
    const std::size_t staging_bytes =
        (main_layout.page_stride + (backend_layout ? backend_layout->page_stride : 0U)) *
        kDiskStagingPages;
    disk_staging_arena = std::make_unique<HostKVArena>(staging_bytes, layouts);
    disk_main_staging  = disk_staging_arena->allocate(main_layout, kDiskStagingPages);
    if (backend_layout) {
        disk_backend_staging = disk_staging_arena->allocate(*backend_layout, kDiskStagingPages);
    }
    disk_state_staging_pool =
        std::make_unique<qwen3_5::HostStatePool>(state_images->host_layout(), 1U);
    disk_state_staging = disk_state_staging_pool->allocate();
    if (!disk_main_staging || (backend_layout && !disk_backend_staging) || !disk_state_staging) {
        throw std::runtime_error("disk KV tier staging could not be reserved");
    }

    const std::size_t backend_stride = backend_layout ? backend_layout->page_stride : 0U;
    const std::size_t state_bytes    = state_images->host_layout().image_bytes;
    const std::uint64_t capacity     = plan.context_cache.disk_kv_capacity_bytes != 0
                                           ? plan.context_cache.disk_kv_capacity_bytes
                                           : 64ULL << 30U;
    DiskKVBridge::Options options{
        .base_path = (plan.context_cache.disk_kv_path /
                      disk_profile_directory(kv_storage, main_layout.page_stride, backend_stride,
                                             state_bytes, rope_yarn))
                         .string(),
        .main_page_stride    = main_layout.page_stride,
        .backend_page_stride = backend_stride,
        .state_page_stride   = state_bytes,
        .capacity_bytes      = static_cast<std::size_t>(capacity),
        .direct_storage      = plan.context_cache.disk_kv_directstorage,
    };
    try {
        disk_kv = std::make_unique<DiskKVBridge>(std::move(options));
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("disk KV tier failed to open: ") + error.what());
    }
    if (disk_kv->slot_count(DiskKVKind::StateImage) == 0) {
        throw std::runtime_error("disk KV tier budget holds no StateImage; raise --disk-kv-gib");
    }
    disk_kv_restore = plan.context_cache.disk_kv_restore;
}

DiskKVIdentity ProgramImpl::disk_identity(const qwen3_5::detail::PrefixShortlistDigests& digests,
                                          std::uint32_t frontier) const {
    const std::array<std::uint64_t, 2> digest = digests.at(frontier);
    return DiskKVIdentity{
        .lo = digest[0], .hi = digest[1], .tag = capture_identity_tag(), .frontier = frontier};
}

ProgramImpl::DiskOwnerSpill ProgramImpl::plan_owner_spill(const SequenceState& sequence) const {
    DiskOwnerSpill spill;
    if (!disk_kv || !sequence.kv) { return spill; }
    const auto digest_limit = static_cast<std::uint32_t>(sequence.prefix_digests.size());
    const std::uint32_t frontier =
        std::min({sequence.text_kv_valid, sequence.execution_frontier, digest_limit});
    if (frontier == 0) { return spill; }
    const auto add_chain = [&](DiskKVKind kind, const KVAddressSpaceStore& addresses,
                               KVAddressSpaceHandle address, std::uint32_t chain_frontier) {
        const std::uint32_t pages =
            std::min(kv_pages_for_frontier(chain_frontier), addresses.mapped_pages(address));
        for (std::uint32_t page = 0; page < pages; ++page) {
            spill.items.push_back(DiskSpillItem{
                .kind     = kind,
                .frontier = std::min((page + 1U) * kPageTokens, chain_frontier),
                .page     = page,
            });
        }
    };
    add_chain(DiskKVKind::MainKV, *text_kv_addresses, sequence.kv->text, frontier);
    const std::uint32_t backend_frontier =
        std::min(backend_frontier_at(speculative_backend, frontier), backend_kv_valid(sequence));
    const bool backend = sequence.kv->backend && backend_kv_addresses && backend_kv_pages;
    if (backend && backend_frontier != 0) {
        add_chain(DiskKVKind::BackendKV, *backend_kv_addresses, *sequence.kv->backend,
                  backend_frontier);
    }
    if (sequence.endpoint_valid && sequence.execution_frontier == frontier &&
        state_store->valid(sequence.state.read)) {
        spill.items.push_back(DiskSpillItem{
            .kind = DiskKVKind::StateImage, .frontier = frontier, .state = sequence.state.read});
    }

    std::vector<std::pair<std::uint32_t, StateImageHandle>> checkpoints;
    if (sequence.rewrite_checkpoint.valid && sequence.rewrite_state &&
        state_store->valid(*sequence.rewrite_state) && sequence.rewrite_checkpoint.frontier != 0 &&
        sequence.rewrite_checkpoint.frontier < frontier) {
        checkpoints.emplace_back(sequence.rewrite_checkpoint.frontier, *sequence.rewrite_state);
    }
    std::vector<std::pair<std::uint32_t, StateImageHandle>> anchors;
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.frontier != 0 && anchor.frontier < frontier &&
            state_store->valid(anchor.state)) {
            anchors.emplace_back(anchor.frontier, anchor.state);
        }
    }
    std::sort(anchors.begin(), anchors.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });
    anchors.resize(std::min(anchors.size(), kSpilledAnchors));
    checkpoints.insert(checkpoints.end(), anchors.begin(), anchors.end());
    for (const auto& [checkpoint, state] : checkpoints) {
        // The chain keys a page by its end, so a boundary inside a page needs that page again
        // under its own frontier.
        if (checkpoint % kPageTokens != 0) {
            spill.items.push_back(DiskSpillItem{.kind     = DiskKVKind::MainKV,
                                                .frontier = checkpoint,
                                                .page     = checkpoint / kPageTokens});
        }
        const std::uint32_t backend_checkpoint =
            backend_frontier_at(speculative_backend, checkpoint);
        if (backend && backend_checkpoint % kPageTokens != 0) {
            spill.items.push_back(DiskSpillItem{.kind     = DiskKVKind::BackendKV,
                                                .frontier = backend_checkpoint,
                                                .page     = backend_checkpoint / kPageTokens});
        }
        spill.items.push_back(
            DiskSpillItem{.kind = DiskKVKind::StateImage, .frontier = checkpoint, .state = state});
    }
    return spill;
}

bool ProgramImpl::owner_spill_in_flight(const DiskOwnerSpill& spill) noexcept {
    return !spill.tickets.empty();
}

bool ProgramImpl::progress_owner_spill(const SequenceState& sequence, DiskOwnerSpill& spill) {
    if (!disk_kv) { return true; }
    if (spill.stopped) {
        // A stopped spill queues nothing more; only what the bridge already holds must land.
        spill.batch_ids.resize(spill.batch_submitted);
        spill.batch_bytes.resize(spill.batch_submitted);
    }
    while (spill.batch_submitted < spill.batch_ids.size()) {
        SpillTicket ticket =
            disk_kv->try_submit(spill.batch_ids[spill.batch_submitted], spill.batch_kind,
                                spill.batch_bytes[spill.batch_submitted]);
        if (!ticket) { return false; }
        spill.tickets.push_back(std::move(ticket));
        ++spill.batch_submitted;
    }
    for (const SpillTicket& ticket : spill.tickets) {
        if (DiskKVBridge::poll(ticket) == SpillStatus::Pending) { return false; }
    }
    for (const SpillTicket& ticket : spill.tickets) {
        const SpillStatus status = DiskKVBridge::poll(ticket);
        if (status == SpillStatus::Failed) { spill.stopped = true; }
        if (status == SpillStatus::Written) { ++spill.written; }
        if (status == SpillStatus::Present) { ++spill.deduplicated; }
    }
    if (!spill.tickets.empty() || !spill.batch_ids.empty()) {
        spill.last_progress = std::chrono::steady_clock::now();
    }
    spill.tickets.clear();
    spill.batch_ids.clear();
    spill.batch_bytes.clear();
    spill.batch_submitted = 0;

    // Stage the next batch: consecutive KV items of one family (at most the staging depth), or one
    // StateImage. Items already on disk are skipped; a chain whose bytes are unavailable ends the
    // spill, since nothing after the gap could be restored.
    while (!spill.stopped && spill.next < spill.items.size()) {
        const DiskSpillItem& first    = spill.items[spill.next];
        const DiskKVIdentity first_id = disk_identity(sequence.prefix_digests, first.frontier);
        if (disk_kv->contains(first_id, first.kind)) {
            ++spill.deduplicated;
            ++spill.next;
            continue;
        }
        spill.batch_kind = first.kind;
        if (first.kind == DiskKVKind::StateImage) {
            const std::size_t bytes               = state_images->host_layout().image_bytes;
            const StateReplicaResidency residency = state_store->valid(first.state)
                                                        ? state_store->residency(first.state)
                                                        : StateReplicaResidency::None;
            if (residency == StateReplicaResidency::None) {
                ++spill.next;
                continue;
            }
            if (const auto host = state_store->host_view(first.state)) {
                spill.batch_bytes.emplace_back(host->data, bytes);
            } else {
                HostStateImageView staging =
                    disk_state_staging_pool->writable_view(*disk_state_staging);
                state_images->copy_to_host(state_store->physical_slot(first.state), staging,
                                           transfer_streams);
                synchronize(transfer_streams);
                spill.batch_bytes.emplace_back(staging.data, bytes);
            }
            spill.batch_ids.push_back(first_id);
            ++spill.next;
            break;
        }

        const bool backend                   = first.kind == DiskKVKind::BackendKV;
        const KVAddressSpaceStore& addresses = backend ? *backend_kv_addresses : *text_kv_addresses;
        LogicalKVPageStore& pages            = backend ? *backend_kv_pages : *text_kv_pages;
        const KVAddressSpaceHandle address   = backend ? *sequence.kv->backend : sequence.kv->text;
        const std::size_t stride = backend ? backend_host_kv_page_stride : text_host_kv_page_stride;
        HostKVAllocation& staging_allocation = backend ? *disk_backend_staging : *disk_main_staging;
        const HostKVAllocationView staging = disk_staging_arena->writable_view(staging_allocation);
        std::vector<DeviceKVPageHandle> device_pages;
        std::vector<std::size_t> device_items;
        while (spill.next < spill.items.size() && spill.batch_ids.size() < kDiskStagingPages) {
            const DiskSpillItem& item = spill.items[spill.next];
            if (item.kind != first.kind) { break; }
            const DiskKVIdentity id = disk_identity(sequence.prefix_digests, item.frontier);
            if (spill.batch_ids.empty() || !disk_kv->contains(id, item.kind)) {
                if (item.page >= addresses.mapped_pages(address)) {
                    spill.stopped = true;
                    break;
                }
                const LogicalKVPageHandle logical = addresses.logical_page(address, item.page);
                if (pages.committed_columns(logical) < item.frontier - item.page * kPageTokens) {
                    spill.stopped = true;
                    break;
                }
                if (pages.host_replica_current(logical)) {
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const auto view =
                        host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
                    spill.batch_bytes.emplace_back(view.data(), stride);
                } else if (pages.device_resident(logical)) {
                    device_items.push_back(spill.batch_bytes.size());
                    device_pages.push_back(pages.physical(logical));
                    spill.batch_bytes.emplace_back();
                } else {
                    spill.stopped = true;
                    break;
                }
                spill.batch_ids.push_back(id);
            } else {
                ++spill.deduplicated;
            }
            ++spill.next;
        }
        if (!device_pages.empty()) {
            pages.physical_pool().copy_to_host(
                device_pages, staging.subview(0, static_cast<std::uint32_t>(device_pages.size())),
                transfer_streams);
            synchronize(transfer_streams);
            for (std::size_t slot = 0; slot < device_items.size(); ++slot) {
                const auto view = staging.subview(static_cast<std::uint32_t>(slot), 1);
                spill.batch_bytes[device_items[slot]] = {view.data(), stride};
            }
        }
        break;
    }
    if (!spill.batch_ids.empty()) { return false; }
    return spill.stopped || spill.next >= spill.items.size();
}

void ProgramImpl::spill_owner_to_disk(const SequenceState& sequence,
                                      std::chrono::steady_clock::time_point deadline) noexcept {
    if (!disk_kv || !sequence.kv) { return; }
    try {
        DiskOwnerSpill spill = plan_owner_spill(sequence);
        while (!progress_owner_spill(sequence, spill)) {
            if (std::chrono::steady_clock::now() > deadline) { spill.stopped = true; }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    } catch (...) {
        // The owner's memory is released after this returns; nothing may still read it.
        disk_kv->wait_idle();
    }
}

void ProgramImpl::spill_released_owner(const SequenceState& sequence) noexcept {
    if (!disk_kv) { return; }
    spill_owner_to_disk(sequence, std::chrono::steady_clock::now() + kReleaseSpillBudget);
}

void ProgramImpl::flush_disk_tier() noexcept {
    if (!disk_kv) { return; }
    // Shutdown writes the catalogued continuations still resident, so a restart can restore the
    // conversations that were live; pages already on disk are skipped.
    const auto deadline = std::chrono::steady_clock::now() + kShutdownSpillBudget;
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Catalogued) { continue; }
        if (std::chrono::steady_clock::now() > deadline) { break; }
        spill_owner_to_disk(continuation_states[index], deadline);
    }
}

std::optional<std::uint32_t>
ProgramImpl::disk_restorable_frontier(const RequestBasePlanImpl& base,
                                      const PreparedPromptData& prompt,
                                      std::uint32_t prompt_tokens) const {
    if (!disk_kv || !disk_kv_restore || prompt.has_media() || prompt_tokens < 2 ||
        base.prefix_digests.size() < prompt_tokens) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> frontiers = disk_kv->live_state_frontiers();
    std::sort(frontiers.begin(), frontiers.end(), std::greater<>());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());
    const bool backend       = backend_kv_pages != nullptr;
    const auto chain_present = [&](DiskKVKind kind, std::uint32_t frontier) {
        for (std::uint32_t end = kPageTokens; end <= frontier; end += kPageTokens) {
            if (!disk_kv->contains(disk_identity(base.prefix_digests, end), kind)) { return false; }
        }
        return frontier % kPageTokens == 0 ||
               disk_kv->contains(disk_identity(base.prefix_digests, frontier), kind);
    };
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier >= prompt_tokens) { continue; }
        if (!disk_kv->contains(disk_identity(base.prefix_digests, frontier),
                               DiskKVKind::StateImage) ||
            !chain_present(DiskKVKind::MainKV, frontier)) {
            continue;
        }
        const std::uint32_t backend_frontier = backend_frontier_at(speculative_backend, frontier);
        if (backend && backend_frontier != 0 &&
            !chain_present(DiskKVKind::BackendKV, backend_frontier)) {
            continue;
        }
        return frontier;
    }
    return std::nullopt;
}

bool ProgramImpl::restore_prefix_from_disk(SequenceState& sequence, RequestControl::Prefill& staged,
                                           std::uint32_t frontier) {
    if (!disk_kv || !sequence.kv || frontier == 0 || frontier >= staged.prompt_tokens ||
        sequence.prefix_digests.size() < frontier) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + kRestoreBudget;
    // Work already issued for this sequence (page activation, the state reset) precedes the
    // copies below, which run on the transfer streams.
    synchronize(compute_streams);
    const auto restore_chain = [&](DiskKVKind kind, LogicalKVPageStore& pages,
                                   const KVAddressSpaceStore& addresses,
                                   KVAddressSpaceHandle address, HostKVAllocation& allocation,
                                   std::uint32_t chain_frontier) {
        const std::uint32_t page_count = kv_pages_for_frontier(chain_frontier);
        if (page_count > addresses.mapped_pages(address)) { return false; }
        const HostKVAllocationView staging = disk_staging_arena->writable_view(allocation);
        const std::size_t stride           = staging.layout().page_stride;
        std::vector<DiskKVIdentity> ids;
        std::vector<DeviceKVPageHandle> destinations;
        for (std::uint32_t begin = 0; begin < page_count; begin += kDiskStagingPages) {
            if (std::chrono::steady_clock::now() > deadline) { return false; }
            const std::uint32_t count = std::min(kDiskStagingPages, page_count - begin);
            ids.clear();
            destinations.clear();
            for (std::uint32_t page = begin; page < begin + count; ++page) {
                ids.push_back(disk_identity(sequence.prefix_digests,
                                            std::min((page + 1U) * kPageTokens, chain_frontier)));
                destinations.push_back(pages.physical(addresses.logical_page(address, page)));
            }
            if (!disk_kv->restore_pages(ids, kind,
                                        std::span<std::byte>(staging.data(), count * stride))) {
                return false;
            }
            pages.physical_pool().copy_from_host(staging.subview(0, count), destinations,
                                                 transfer_streams);
            synchronize(transfer_streams);
        }
        return true;
    };
    if (!restore_chain(DiskKVKind::MainKV, *text_kv_pages, *text_kv_addresses, sequence.kv->text,
                       *disk_main_staging, frontier)) {
        return false;
    }
    const std::uint32_t backend_frontier = backend_frontier_at(speculative_backend, frontier);
    const bool backend = sequence.kv->backend && backend_kv_pages && backend_kv_addresses;
    if (backend && backend_frontier != 0 &&
        !restore_chain(DiskKVKind::BackendKV, *backend_kv_pages, *backend_kv_addresses,
                       *sequence.kv->backend, *disk_backend_staging, backend_frontier)) {
        return false;
    }
    HostStateImageView state = disk_state_staging_pool->writable_view(*disk_state_staging);
    if (std::chrono::steady_clock::now() > deadline ||
        !disk_kv->restore_page(
            disk_identity(sequence.prefix_digests, frontier), DiskKVKind::StateImage,
            std::span<std::byte>(state.data, state_images->host_layout().image_bytes))) {
        return false;
    }
    // Every byte is on the host and verified; only now is the recurrent state overwritten.
    state_images->copy_from_host(disk_state_staging_pool->view(*disk_state_staging),
                                 state_selectors(sequence).destination, transfer_streams);
    synchronize(transfer_streams);

    const std::uint32_t backend_valid =
        speculative_backend == SpeculativeBackend::Mtp      ? backend_frontier
        : speculative_backend == SpeculativeBackend::DFlash ? frontier
                                                            : 0U;
    commit_sequence_kv(sequence, frontier, backend ? backend_valid : 0U);
    sequence.text_kv_valid = frontier;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        sequence.mtp_kv_valid = backend_frontier;
    }
    if (is_masked_draft_backend(speculative_backend)) {
        sequence.dflash_context_frontier = frontier;
    }
    refresh_state_views(sequence);
    // The StateImage carries the hidden of the token before the frontier, which the MTP bridge
    // needs to extend the draft KV to the frontier.
    sequence.tail_hidden_valid = true;
    staged.base                = frontier;
    staged.cursor              = frontier;
    if (staged.prepare_mtp) { staged.mtp_bridge = MtpBridgeMode::BeforeSuffix; }
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
