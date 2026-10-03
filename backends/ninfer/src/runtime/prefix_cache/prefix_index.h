#pragma once

#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/cost.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime::prefix_cache {

// Hybrid prefix cache index (docs/maintainer/hybrid-prefix-cache-spec.md §5, §6, §9).
//
// The index is host-only bookkeeping. It owns exact identity (the block tree and snapshot tails),
// residency state, pins, host slab ids, device snapshot slot ids and every eviction decision.
// Device page bundles are opaque ids supplied by the caller and returned through
// PrefixIndexBackend when the index stops referencing them. The index performs no transfers; the
// caller moves bytes and reports completion.
//
// Pins are path-wide: pinning a node pins it and every ancestor, so an unpinned node always has an
// unpinned subtree. Every active sequence pins its whole mapped path, every in-flight transfer
// pins the path of the node it touches.

inline constexpr std::uint32_t kNoId = std::numeric_limits<std::uint32_t>::max();

struct NodeRef {
    std::uint32_t index      = kNoId;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept { return index != kNoId; }

    [[nodiscard]] friend bool operator==(NodeRef, NodeRef) noexcept = default;
};

struct SnapshotRef {
    std::uint32_t index      = kNoId;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept { return index != kNoId; }

    [[nodiscard]] friend bool operator==(SnapshotRef, SnapshotRef) noexcept = default;
};

enum class CopyState : std::uint8_t {
    Absent,
    Filling,
    Resident,
};

enum class SnapshotKind : std::uint8_t {
    Tap,
    Endpoint,
    OutputBoundary,
};

struct PrefixIndexConfig {
    // Upper bound of simultaneously indexed nodes. Every indexed node holds a device page bundle
    // or a host slab, so device pages + host slabs is sufficient.
    std::uint32_t max_nodes     = 0;
    std::uint32_t max_snapshots = 0;
    std::uint32_t host_slabs    = 0;
    // Slabs per snapshot image; a snapshot with a tail needs one more slab for the tail bundle.
    std::uint32_t image_slabs           = 0;
    std::uint32_t device_snapshot_slots = 0;
    std::uint64_t block_bytes           = 0; // one block bundle (all enabled pools)
    std::uint64_t image_bytes           = 0; // one state image
    // Whether the host tier backs KV block bundles and snapshot tails. Without it the host tier
    // holds state images only: tails and blocks are unbacked and their loss removes dependants.
    bool host_blocks = true;
    CacheCostModel cost;
};

// Physical resources the index references but does not own.
class PrefixIndexBackend {
public:
    virtual ~PrefixIndexBackend() = default;
    // The index dropped its last reference to this device page bundle (block or snapshot tail).
    virtual void release_device_block(std::uint32_t device_id) noexcept = 0;

    // The snapshot is being removed: every copy of its image is released. Called before the
    // snapshot reference becomes stale.
    virtual void release_snapshot(SnapshotRef /*snapshot*/) noexcept {}

    // The snapshot keeps its host image but loses its device image copy.
    virtual void drop_snapshot_device_image(SnapshotRef /*snapshot*/) noexcept {}
};

struct MatchCandidate {
    SnapshotRef snapshot;
    std::uint32_t frontier         = 0;
    std::uint32_t path_blocks      = 0; // full blocks up to and including the anchor
    std::uint32_t host_only_blocks = 0; // path blocks without a device copy (needing H2D)
    std::uint32_t filling_blocks   = 0; // path blocks another admission is already restoring
    bool tail                      = false;
    bool tail_on_device            = false;
    bool image_on_device           = false;
    std::uint64_t restore_bytes    = 0;
};

struct MatchResult {
    std::vector<NodeRef> path;              // matched full blocks in prompt order
    std::vector<MatchCandidate> candidates; // valid snapshots, deepest first
};

struct AdmissionChoice {
    std::optional<std::size_t> candidate; // absent: root
    double predicted_seconds = 0.0;
};

struct InsertResult {
    NodeRef node;
    bool inserted = false;
    // The supplied device id now belongs to the index: a new node, or an existing host-only node
    // whose Device copy is re-established from the inserting sequence's page.
    bool device_attached = false;
};

struct PublishResult {
    SnapshotRef snapshot;
    bool created = false;
};

struct NodeView {
    std::uint32_t depth = 0;
    NodeRef parent;
    std::uint32_t device_id            = kNoId;
    CopyState device                   = CopyState::Absent;
    std::uint32_t host_slab            = kNoId;
    CopyState host                     = CopyState::Absent;
    std::uint32_t pins                 = 0;
    std::uint32_t live_snapshots_below = 0;
};

struct SnapshotView {
    NodeRef anchor;
    std::uint32_t frontier     = 0;
    std::uint32_t tail_len     = 0;
    std::uint32_t tail_device  = kNoId;
    CopyState tail_device_copy = CopyState::Absent;
    std::uint32_t device_slot  = kNoId;
    CopyState host             = CopyState::Absent;
    std::span<const std::uint32_t> host_slabs; // image slabs, then the tail slab when tail_len>0
    std::uint32_t pins = 0;
    SnapshotKind kind  = SnapshotKind::Tap;
    std::uint32_t hits = 0;
    std::span<const TokenId> tail;
};

// Exact identity of one block, for saving the Host tier.
struct BlockIdentity {
    NodeRef parent;
    std::uint64_t lookup_hash = 0;
    std::uint64_t extra       = 0;
    std::span<const TokenId> tokens;
};

struct RestoredBlock {
    NodeRef node;
    std::uint32_t slab = kNoId;
};

struct PrefixIndexStats {
    std::uint32_t nodes                   = 0;
    std::uint32_t snapshots               = 0;
    std::uint32_t device_resident_blocks  = 0;
    std::uint32_t device_evictable_blocks = 0;
    std::uint32_t host_free_slabs         = 0;
    std::uint32_t free_device_slots       = 0;
    std::uint64_t device_block_evictions  = 0;
    std::uint64_t unbacked_node_losses    = 0;
    std::uint64_t host_dead_reclaims      = 0;
    std::uint64_t host_snapshot_evictions = 0;
    std::uint64_t device_slot_evictions   = 0;
    std::uint64_t snapshot_hits           = 0;
    double gdsf_inflation                 = 0.0;
};

class PrefixCacheIndex {
public:
    PrefixCacheIndex(const PrefixIndexConfig& config, PrefixIndexBackend& backend);

    PrefixCacheIndex(const PrefixCacheIndex&)            = delete;
    PrefixCacheIndex& operator=(const PrefixCacheIndex&) = delete;

    [[nodiscard]] const PrefixIndexConfig& config() const noexcept { return config_; }

    // Replaces the machine model used by choose() and snapshot valuation. Existing priorities
    // are refreshed.
    void set_cost(const CacheCostModel& cost);

    // ---- lookup and choice (§6.1, §6.2) -------------------------------------------------------
    // `block_hashes[b]` is block_lookup_hashes(tokens)[b]; `block_extras` is empty or one extra key
    // per full block. Only snapshots with frontier <= prompt_tokens - 1 are returned.
    [[nodiscard]] MatchResult match(std::span<const TokenId> tokens,
                                    std::span<const std::uint64_t> block_hashes,
                                    std::span<const std::uint64_t> block_extras,
                                    std::uint32_t prompt_tokens) const;
    [[nodiscard]] AdmissionChoice choose(const MatchResult& match,
                                         std::uint32_t prompt_tokens) const;
    // Records a hit on the snapshot selected for an admission.
    void note_hit(SnapshotRef snapshot);

    // ---- pins -----------------------------------------------------------------------------
    // `path` is a root path (path[i] is the parent of path[i+1], path[0] a root child).
    void acquire_path(std::span<const NodeRef> path);
    // Releases pins taken by acquire_path/insert_block; nodes that become unpinned join the device
    // LRU deepest first, so deeper blocks are evicted before shallower ones.
    void release_path(std::span<const NodeRef> path);
    // Pins one node and its ancestors (in-flight transfer of that node).
    void pin_node(NodeRef node);
    void unpin_node(NodeRef node);
    void pin_snapshot(SnapshotRef snapshot);
    void unpin_snapshot(SnapshotRef snapshot);

    // ---- tree mutation (§7.6) ------------------------------------------------------------------
    [[nodiscard]] std::optional<NodeRef> find_child(NodeRef parent, std::uint64_t lookup_hash,
                                                    std::span<const TokenId> block_tokens,
                                                    std::uint64_t extra) const;
    // Inserts a committed full block owning `device_id`, device Resident, pinned once for the
    // inserting sequence. When an identical child already exists `inserted` is false; a host-only
    // existing child adopts `device_id` as its Device copy (device_attached), otherwise the caller
    // keeps its page private.
    [[nodiscard]] InsertResult insert_block(NodeRef parent, std::uint64_t lookup_hash,
                                            std::span<const TokenId> block_tokens,
                                            std::uint64_t extra, std::uint32_t device_id);

    // ---- device residency of blocks (§6.4, §9.1)
    // ------------------------------------------------- Host-only pinned node receives a device
    // page being filled from its host slab.
    void begin_device_fill(NodeRef node, std::uint32_t device_id);
    void complete_device_fill(NodeRef node);
    // Returns the device id to the backend.
    void abort_device_fill(NodeRef node);
    // Releases up to `blocks` unpinned device copies, host-backed ones first. Returns the number
    // released. An unbacked release loses the node and its subtree (§9.4).
    std::uint32_t evict_device_blocks(std::uint32_t blocks);
    [[nodiscard]] std::uint32_t device_evictable_blocks() const noexcept;

    // ---- host residency (§9.3)
    // ------------------------------------------------------------------- Allocates a slab for a
    // device-resident pinned node (write-through destination). Absent when no slab can be freed.
    [[nodiscard]] std::optional<std::uint32_t> begin_host_fill(NodeRef node);
    void complete_host_fill(NodeRef node);
    void abort_host_fill(NodeRef node);
    // Allocates image slabs (+1 tail slab) for a pinned snapshot. Absent when they cannot be freed.
    [[nodiscard]] bool begin_snapshot_host_fill(SnapshotRef snapshot);
    void complete_snapshot_host_fill(SnapshotRef snapshot);
    void abort_snapshot_host_fill(SnapshotRef snapshot);
    // A pinned host-backed snapshot whose tail lost its Device copy receives `device_id` being
    // filled from its tail slab.
    void begin_tail_device_fill(SnapshotRef snapshot, std::uint32_t device_id);
    void complete_tail_device_fill(SnapshotRef snapshot);
    // Returns the device id to the backend.
    void abort_tail_device_fill(SnapshotRef snapshot);

    // ---- snapshots (§5.3, §7.4, §7.7, §9.2)
    // ------------------------------------------------------- Returns a free device snapshot slot
    // for a tap/endpoint destination (staging), evicting the least recently hit host-backed slot.
    // With `allow_unbacked`, an unpinned slot whose snapshot has no host copy may be evicted when
    // no backed slot exists (that snapshot is lost).
    [[nodiscard]] std::optional<std::uint32_t> acquire_device_slot(bool allow_unbacked);
    void release_device_slot(std::uint32_t slot);
    // Publishes a snapshot whose image is in staging slot `device_slot`. frontier must equal
    // 64 * (anchor depth + 1) + tail.size() (tail.size() for the root anchor). A tail requires
    // `tail_device_id`. For a duplicate (same anchor and tail) the staging slot is freed, the tail
    // device id is released and the existing snapshot is returned with created=false.
    [[nodiscard]] PublishResult publish_snapshot(NodeRef anchor, std::uint32_t frontier,
                                                 std::span<const TokenId> tail,
                                                 std::optional<std::uint32_t> tail_device_id,
                                                 std::uint32_t device_slot, SnapshotKind kind);

    // ---- persistence ---------------------------------------------------------------------------
    // What a saved Host tier keeps: every Host-resident snapshot whose anchor path is Host-resident
    // throughout, and exactly those paths, parents before children. Other Host content is dead
    // (no snapshot can resume through it) or unreachable after a restart.
    void collect_persistable(std::vector<NodeRef>& nodes,
                             std::vector<SnapshotRef>& snapshots) const;
    [[nodiscard]] BlockIdentity block_identity(NodeRef node) const;
    // Rebuild a saved Host tier into a fresh index, parents before children and anchors before
    // their snapshots. Nothing is evicted to make room, so a smaller Host tier restores a prefix
    // of what was saved; absent means no slab, node or snapshot entry was free.
    [[nodiscard]] std::optional<RestoredBlock> restore_host_block(NodeRef parent,
                                                                  std::uint64_t lookup_hash,
                                                                  std::span<const TokenId> tokens,
                                                                  std::uint64_t extra);
    [[nodiscard]] std::optional<SnapshotRef>
    restore_host_snapshot(NodeRef anchor, std::uint32_t frontier, std::span<const TokenId> tail,
                          SnapshotKind kind, std::uint32_t hits);

    // ---- views
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool valid(NodeRef node) const noexcept;
    [[nodiscard]] bool valid(SnapshotRef snapshot) const noexcept;
    [[nodiscard]] NodeView node(NodeRef node) const;
    [[nodiscard]] SnapshotView snapshot(SnapshotRef snapshot) const;
    [[nodiscard]] std::optional<SnapshotRef> slot_owner(std::uint32_t slot) const;
    [[nodiscard]] PrefixIndexStats stats() const noexcept;
    // Structural self-check for tests: throws std::logic_error on any invariant violation.
    void check_invariants() const;

private:
    struct Node {
        std::uint32_t generation = 0;
        bool occupied            = false;
        std::uint32_t parent     = kNoId;
        std::uint32_t depth      = 0;
        std::uint64_t hash       = 0;
        std::uint64_t extra      = 0;
        std::array<TokenId, kBlockTokens> tokens{};
        std::vector<std::uint32_t> children;
        std::vector<std::uint32_t> snapshots;
        std::uint32_t device_id  = kNoId;
        CopyState device         = CopyState::Absent;
        std::uint32_t host_slab  = kNoId;
        CopyState host           = CopyState::Absent;
        std::uint32_t pins       = 0;
        std::uint32_t live_below = 0;
        std::uint32_t dead_prev  = kNoId;
        std::uint32_t dead_next  = kNoId;
        bool in_dead             = false;
    };

    struct Snapshot {
        std::uint32_t generation = 0;
        bool occupied            = false;
        std::uint32_t anchor     = kNoId; // node index, kNoId for the root
        std::uint32_t frontier   = 0;
        std::uint32_t tail_len   = 0;
        std::array<TokenId, kBlockTokens - 1> tail{};
        std::uint32_t tail_device  = kNoId;
        CopyState tail_device_copy = CopyState::Absent;
        std::uint32_t device_slot  = kNoId;
        std::vector<std::uint32_t> host_slabs;
        CopyState host              = CopyState::Absent;
        std::uint32_t pins          = 0;
        SnapshotKind kind           = SnapshotKind::Tap;
        std::uint32_t hits          = 0;
        double priority             = 0.0; // GDSF H
        std::uint64_t last_hit_tick = 0;
    };

    enum class SlotState : std::uint8_t { Free, Staging, Owned };

    // Device LRU entries: node i is entry i, snapshot tail j is entry max_nodes + j.
    struct LruLink {
        std::uint32_t prev = kNoId;
        std::uint32_t next = kNoId;
        std::uint8_t list  = 0xff; // 0 backed, 1 unbacked, 0xff absent
    };

    struct LruList {
        std::uint32_t head  = kNoId;
        std::uint32_t tail  = kNoId;
        std::uint32_t count = 0;
    };

    [[nodiscard]] Node& require(NodeRef node);
    [[nodiscard]] const Node& require(NodeRef node) const;
    [[nodiscard]] Snapshot& require(SnapshotRef snapshot);
    [[nodiscard]] const Snapshot& require(SnapshotRef snapshot) const;
    [[nodiscard]] NodeRef ref_of_node(std::uint32_t index) const noexcept;
    [[nodiscard]] SnapshotRef ref_of_snapshot(std::uint32_t index) const noexcept;
    [[nodiscard]] static std::uint64_t child_key(std::uint32_t parent,
                                                 std::uint64_t lookup_hash) noexcept;

    [[nodiscard]] bool snapshot_valid(const Snapshot& snapshot) const noexcept;
    [[nodiscard]] bool tail_matches(const Snapshot& snapshot,
                                    std::span<const TokenId> prompt) const noexcept;

    void pin_path_from(std::uint32_t node);
    void unpin_path_from(std::uint32_t node, bool deepest_first_order);
    void refresh_node(std::uint32_t node);
    void refresh_tail(std::uint32_t snapshot);

    void lru_remove(std::uint32_t entry);
    void lru_append(std::uint32_t entry, std::uint8_t list);
    [[nodiscard]] std::uint32_t lru_pop(std::uint8_t list);

    void dead_remove(std::uint32_t node);
    void dead_append(std::uint32_t node);

    [[nodiscard]] bool take_free_slabs(std::uint32_t count, std::vector<std::uint32_t>& out);
    [[nodiscard]] bool allocate_slabs(std::uint32_t count, std::vector<std::uint32_t>& out,
                                      std::uint32_t protect_snapshot);
    void free_slab(std::uint32_t slab);

    void remove_snapshot(std::uint32_t snapshot);
    void remove_subtree(std::uint32_t node);
    void release_node_storage(std::uint32_t node);
    void drop_node_device_copy(std::uint32_t node);
    void drop_node_host_copy(std::uint32_t node);
    void adjust_live(std::uint32_t anchor, int delta);
    // Every Device residency and slot-state change goes through these so the gauges stats()
    // reports stay O(1): the Engine publishes statistics after every decode round.
    void set_device(Node& node, CopyState state) noexcept;
    void set_slot(std::uint32_t slot, SlotState state) noexcept;
    void update_priority(std::uint32_t snapshot);

    PrefixIndexConfig config_;
    PrefixIndexBackend* backend_ = nullptr;

    std::vector<Node> nodes_;
    std::vector<std::uint32_t> free_nodes_;
    std::unordered_multimap<std::uint64_t, std::uint32_t> children_;
    std::vector<std::uint32_t> root_children_;
    std::vector<std::uint32_t> root_snapshots_;

    std::vector<Snapshot> snapshots_;
    std::vector<std::uint32_t> free_snapshots_;

    std::vector<LruLink> lru_links_;
    std::array<LruList, 2> lru_{};

    std::uint32_t dead_head_ = kNoId;
    std::uint32_t dead_tail_ = kNoId;

    std::vector<std::uint32_t> free_slabs_;
    std::vector<SlotState> slot_state_;
    std::vector<std::uint32_t> slot_owner_;

    std::uint32_t node_count_      = 0;
    std::uint32_t snapshot_count_  = 0;
    std::uint32_t device_resident_ = 0;
    std::uint32_t free_slots_      = 0;
    std::uint64_t tick_            = 0;
    double inflation_              = 0.0;
    PrefixIndexStats counters_;
};

} // namespace ninfer::runtime::prefix_cache
