#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::runtime::prefix_cache;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class F>
void require_throws(F&& f, const std::string& message) {
    try {
        f();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("expected an exception: " + message);
}

class RecordingBackend final : public PrefixIndexBackend {
public:
    void release_device_block(std::uint32_t id) noexcept override {
        released.push_back(id);
        live.erase(id);
    }

    std::uint32_t allocate() {
        const std::uint32_t id = next++;
        live.insert(id);
        return id;
    }

    std::vector<std::uint32_t> released;
    std::set<std::uint32_t> live;
    std::uint32_t next = 1;
};

PrefixIndexConfig small_config(std::uint32_t host_slabs = 64, std::uint32_t slots = 2) {
    PrefixIndexConfig config;
    config.max_nodes                 = 256;
    config.max_snapshots             = 32;
    config.host_slabs                = host_slabs;
    config.image_slabs               = 4;
    config.device_snapshot_slots     = slots;
    config.block_bytes               = 1U << 20U;
    config.image_bytes               = 4U << 20U;
    config.cost.token_seconds        = 1.0e-4;
    config.cost.h2d_bytes_per_second = 50.0e9;
    return config;
}

std::vector<TokenId> make_tokens(std::uint32_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<TokenId> tokens(count);
    for (TokenId& token : tokens) { token = static_cast<TokenId>(rng() % 50000U); }
    return tokens;
}

// Inserts every full block of `tokens` below the root as one sequence would, returning its path.
std::vector<NodeRef> insert_sequence(PrefixCacheIndex& index, RecordingBackend& backend,
                                     const std::vector<TokenId>& tokens) {
    const auto hashes = block_lookup_hashes(tokens, {});
    std::vector<NodeRef> path;
    NodeRef parent;
    for (std::size_t b = 0; b < hashes.size(); ++b) {
        const std::span<const TokenId> block(tokens.data() + b * kBlockTokens, kBlockTokens);
        const std::uint32_t id    = backend.allocate();
        const InsertResult result = index.insert_block(parent, hashes[b], block, 0, id);
        require(result.inserted == result.device_attached || !result.inserted,
                "a new node must own its device id");
        if (!result.device_attached) { backend.release_device_block(id); }
        path.push_back(result.node);
        parent = result.node;
    }
    index.check_invariants();
    return path;
}

SnapshotRef publish_tap(PrefixCacheIndex& index, NodeRef anchor) {
    const auto slot = index.acquire_device_slot(false);
    require(slot.has_value(), "no device slot for a tap");
    const std::uint32_t frontier = (index.node(anchor).depth + 1U) * kBlockTokens;
    const PublishResult result =
        index.publish_snapshot(anchor, frontier, {}, std::nullopt, *slot, SnapshotKind::Tap);
    require(result.created, "tap snapshot not created");
    index.check_invariants();
    return result.snapshot;
}

void backup_node(PrefixCacheIndex& index, NodeRef node) {
    index.pin_node(node);
    const auto slab = index.begin_host_fill(node);
    require(slab.has_value(), "no slab for a block backup");
    index.complete_host_fill(node);
    index.unpin_node(node);
    index.check_invariants();
}

void backup_snapshot(PrefixCacheIndex& index, SnapshotRef snapshot) {
    index.pin_snapshot(snapshot);
    require(index.begin_snapshot_host_fill(snapshot), "no slabs for a snapshot backup");
    index.complete_snapshot_host_fill(snapshot);
    index.unpin_snapshot(snapshot);
    index.check_invariants();
}

void test_hashes() {
    const auto tokens = make_tokens(256, 1);
    const auto a      = block_lookup_hashes(tokens, {});
    const auto b      = block_lookup_hashes(tokens, {});
    require(a.size() == 4 && a == b, "block hashes are not deterministic");
    auto changed = tokens;
    changed[130] ^= 1;
    const auto c = block_lookup_hashes(changed, {});
    require(c[0] == a[0] && c[1] == a[1] && c[2] != a[2] && c[3] != a[3],
            "a token change must change its block hash and every later chained hash");
    const std::vector<std::uint64_t> extras{0, 7, 0, 0};
    const auto d = block_lookup_hashes(tokens, extras);
    require(d[0] == a[0] && d[1] != a[1], "the extra key must change the block hash");
    require_throws([&] { (void)block_lookup_hashes(tokens, std::vector<std::uint64_t>{1}); },
                   "short extras");
}

void test_match_and_choice() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(), backend);
    const auto tokens = make_tokens(64 * 8 + 10, 2);
    const std::vector<TokenId> first(tokens.begin(), tokens.begin() + 64 * 6);
    const auto path        = insert_sequence(index, backend, first);
    const SnapshotRef tap2 = publish_tap(index, path[1]); // frontier 128
    const SnapshotRef tap5 = publish_tap(index, path[4]); // frontier 320
    index.release_path(path);
    index.check_invariants();

    const auto hashes = block_lookup_hashes(tokens, {});
    const MatchResult match =
        index.match(tokens, hashes, {}, static_cast<std::uint32_t>(tokens.size()));
    require(match.path.size() == 6, "match did not follow every cached block");
    require(match.candidates.size() == 2 && match.candidates[0].snapshot == tap5 &&
                match.candidates[1].snapshot == tap2,
            "candidates are not deepest first");
    require(match.candidates[0].restore_bytes == 0 && match.candidates[0].image_on_device,
            "device-resident candidate should need no restore");
    const AdmissionChoice choice = index.choose(match, static_cast<std::uint32_t>(tokens.size()));
    require(choice.candidate == 0U, "the deepest device-resident snapshot should be chosen");

    // A prompt that ends exactly at the tap frontier cannot use it (at least one token prefills).
    const MatchResult exact = index.match(tokens, hashes, {}, 320);
    require(exact.candidates.size() == 1 && exact.candidates[0].snapshot == tap2,
            "a snapshot at the prompt end must not be offered");
    require(index.match(tokens, hashes, {}, 321).candidates[0].snapshot == tap5,
            "a snapshot one token before the prompt end must be offered");

    // A different token in block 3 stops the path at block 2 and hides tap5.
    auto edited = tokens;
    edited[64 * 3 + 5] += 1;
    const auto edited_hashes = block_lookup_hashes(edited, {});
    const MatchResult partial =
        index.match(edited, edited_hashes, {}, static_cast<std::uint32_t>(edited.size()));
    require(partial.path.size() == 3 && partial.candidates.size() == 1 &&
                partial.candidates[0].snapshot == tap2,
            "an edit must fall back to the snapshot before it");

    // The extra key (media identity) distinguishes identical tokens.
    std::vector<std::uint64_t> extras(hashes.size(), 0);
    extras[0]               = 99;
    const auto media_hashes = block_lookup_hashes(tokens, extras);
    require(index.match(tokens, media_hashes, extras, static_cast<std::uint32_t>(tokens.size()))
                .path.empty(),
            "different media must not match");
}

void test_collisions_and_tails() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(), backend);
    const auto a = make_tokens(64, 3);
    const auto b = make_tokens(64, 4);
    // Force the same lookup hash for two different blocks: exact comparison must separate them.
    const InsertResult first  = index.insert_block({}, 42, a, 0, backend.allocate());
    const InsertResult second = index.insert_block({}, 42, b, 0, backend.allocate());
    require(first.inserted && second.inserted && !(first.node == second.node),
            "colliding blocks must be distinct nodes");
    require(index.find_child({}, 42, a, 0) == first.node &&
                index.find_child({}, 42, b, 0) == second.node,
            "collision lookup returned the wrong node");
    require(!index.find_child({}, 42, a, 1).has_value(), "extra key mismatch must miss");
    index.check_invariants();

    // Endpoint with a tail: frontier 64 + 10 anchored at `first`.
    std::vector<TokenId> prompt = a;
    const auto tail_tokens      = make_tokens(10, 5);
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(7);
    prompt.push_back(8);
    const auto slot             = index.acquire_device_slot(false);
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(first.node, 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    require(endpoint.created, "endpoint snapshot not created");
    index.check_invariants();

    std::vector<std::uint64_t> hashes{42};
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].frontier == 74 &&
                match.candidates[0].tail && match.candidates[0].tail_on_device,
            "tail snapshot did not match its exact tail");
    auto other_tail = prompt;
    other_tail[70] += 1;
    require(index.match(other_tail, hashes, {}, static_cast<std::uint32_t>(prompt.size()))
                .candidates.empty(),
            "a different tail token must not match");

    // Duplicate publication frees the new slot and tail and returns the existing snapshot.
    const auto slot2              = index.acquire_device_slot(false);
    const std::uint32_t tail_copy = backend.allocate();
    const PublishResult duplicate = index.publish_snapshot(first.node, 74, tail_tokens, tail_copy,
                                                           *slot2, SnapshotKind::Endpoint);
    require(!duplicate.created && duplicate.snapshot == endpoint.snapshot,
            "duplicate snapshot publication was not deduplicated");
    require(backend.released.back() == tail_copy, "duplicate tail was not released");
    require(index.stats().free_device_slots == 1, "duplicate staging slot was not freed");
    require_throws(
        [&] {
            const auto s = index.acquire_device_slot(false);
            (void)index.publish_snapshot(first.node, 75, tail_tokens, backend.allocate(), *s,
                                         SnapshotKind::Tap);
        },
        "frontier mismatch");
    index.check_invariants();
}

void test_device_lru_order() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(0), backend);
    const auto tokens = make_tokens(64 * 4, 6);
    const auto path   = insert_sequence(index, backend, tokens);
    (void)publish_tap(index, path[3]);
    require(index.device_evictable_blocks() == 0, "pinned blocks must not be evictable");
    index.release_path(path);
    index.check_invariants();
    require(index.device_evictable_blocks() == 4, "released blocks must become evictable");

    // Unbacked: evicting the deepest block first loses only that node and its snapshot.
    const std::uint32_t deepest = index.node(path[3]).device_id;
    require(index.evict_device_blocks(1) == 1, "one block should be released");
    require(backend.released.back() == deepest, "deepest block must be evicted first");
    require(!index.valid(path[3]) && index.valid(path[2]), "only the deepest node is lost");
    require(index.stats().snapshots == 0, "the snapshot on a lost node must be removed");
    index.check_invariants();

    // Re-pinning a path protects it.
    const std::vector<NodeRef> prefix(path.begin(), path.begin() + 3);
    index.acquire_path(prefix);
    require(index.evict_device_blocks(10) == 0, "pinned path must not be evicted");
    index.release_path(prefix);
    require(index.evict_device_blocks(10) == 3, "whole released path must be evictable");
    require(index.stats().nodes == 0 && backend.live.empty(), "every page must be released");
    index.check_invariants();
}

void test_backed_before_unbacked() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64), backend);
    const auto a             = make_tokens(64 * 2, 7);
    const auto b             = make_tokens(64 * 2, 8);
    const auto path_a        = insert_sequence(index, backend, a);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_a = publish_tap(index, path_a[1]);
    const SnapshotRef snap_b = publish_tap(index, path_b[1]);
    backup_node(index, path_b[0]);
    backup_node(index, path_b[1]);
    backup_snapshot(index, snap_b);
    index.release_path(path_a); // older: unbacked
    index.release_path(path_b); // newer: backed
    index.check_invariants();
    require(index.evict_device_blocks(2) == 2, "two backed blocks should be released");
    require(index.valid(path_b[0]) && index.node(path_b[0]).device == CopyState::Absent &&
                index.node(path_b[0]).host == CopyState::Resident,
            "backed block must survive as host-only");
    require(index.node(path_a[0]).device == CopyState::Resident,
            "unbacked blocks must be evicted only after backed ones");
    require(index.valid(snap_a) && index.valid(snap_b), "no snapshot should be lost");

    // The host-only path is still a candidate and reports its restore bytes.
    const auto hashes = block_lookup_hashes(b, {});
    auto prompt       = b;
    prompt.push_back(1);
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].host_only_blocks == 2,
            "host-only path blocks must be counted");
    require(match.candidates[0].restore_bytes == 2U * (1U << 20U),
            "restore bytes must cover host-only blocks (image still on device)");

    // Restore one block to device.
    index.acquire_path(match.path);
    const std::uint32_t id = backend.allocate();
    index.begin_device_fill(match.path[0], id);
    require(index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()))
                    .candidates[0]
                    .filling_blocks == 1,
            "a filling block must be reported");
    index.complete_device_fill(match.path[0]);
    index.release_path(match.path);
    index.check_invariants();
}

void test_host_dead_and_gdsf() {
    RecordingBackend backend;
    // 12 slabs: images need 4 slabs each.
    PrefixCacheIndex index(small_config(12, 3), backend);
    const auto dead_tokens = make_tokens(64 * 4, 9);
    const auto dead_path   = insert_sequence(index, backend, dead_tokens);
    for (const NodeRef node : dead_path) { backup_node(index, node); }
    index.release_path(dead_path); // no snapshot: dead KV
    index.check_invariants();

    const auto a             = make_tokens(64 * 1, 10);
    const auto path_a        = insert_sequence(index, backend, a);
    const SnapshotRef snap_a = publish_tap(index, path_a[0]);
    // Allocating 4 slabs (8 free) needs no reclaim; the next ones reclaim dead blocks first.
    backup_snapshot(index, snap_a);
    require(index.stats().host_dead_reclaims == 0, "no reclaim needed yet");
    index.release_path(path_a);

    const auto b             = make_tokens(64 * 1, 11);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_b = publish_tap(index, path_b[0]);
    backup_snapshot(index, snap_b);
    require(index.stats().host_dead_reclaims == 0, "4 slabs were still free");
    backup_node(index, path_b[0]);
    require(index.stats().host_dead_reclaims == 1, "dead KV must be reclaimed before snapshots");
    require(index.valid(snap_a) && index.valid(snap_b),
            "no snapshot should be evicted for dead KV");
    index.release_path(path_b);

    // Hit b so it is more valuable; a new snapshot must evict a (the GDSF minimum).
    index.note_hit(snap_b);
    const auto c             = make_tokens(64 * 1, 12);
    const auto path_c        = insert_sequence(index, backend, c);
    const SnapshotRef snap_c = publish_tap(index, path_c[0]);
    const double before      = index.stats().gdsf_inflation;
    // Consume the remaining dead nodes first, then a snapshot's host copy must go.
    backup_snapshot(index, snap_c);
    require(index.stats().host_dead_reclaims == 4, "all dead nodes must go before snapshots");
    require(index.stats().host_snapshot_evictions == 1, "one snapshot host copy must be evicted");
    // a is the GDSF minimum; it is still complete on the Device, so only its host copy goes.
    require(index.valid(snap_a) && index.snapshot(snap_a).host == CopyState::Absent &&
                index.snapshot(snap_a).device_slot != kNoId,
            "the least valuable snapshot must yield its host copy and keep its device image");
    require(index.valid(snap_b) && index.snapshot(snap_b).host == CopyState::Resident &&
                index.valid(snap_c) && index.snapshot(snap_c).host == CopyState::Resident,
            "more valuable snapshots keep their host copies");
    require(index.stats().gdsf_inflation >= before, "GDSF inflation must be monotonic");

    // A host-only snapshot chosen by host pressure is removed entirely.
    const auto slot = index.acquire_device_slot(false);
    require(slot.has_value() && index.valid(snap_b) &&
                (index.snapshot(snap_b).device_slot == kNoId ||
                 index.snapshot(snap_c).device_slot == kNoId),
            "a backed device slot must be reclaimable");
    index.release_device_slot(*slot);
    index.release_path(path_c);
    index.check_invariants();
}

void test_host_only_reattach() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 2), backend);
    const auto tokens      = make_tokens(64 * 3, 30);
    const auto path        = insert_sequence(index, backend, tokens);
    const SnapshotRef snap = publish_tap(index, path[2]);
    for (const NodeRef node : path) { backup_node(index, node); }
    backup_snapshot(index, snap);
    index.release_path(path);
    require(index.evict_device_blocks(3) == 3, "backed blocks must be released");
    for (const NodeRef node : path) {
        require(index.node(node).device == CopyState::Absent, "block must be host-only");
    }
    const std::size_t live_before = backend.live.size();

    // A sequence recomputing the same blocks re-establishes their Device copies.
    const auto again = insert_sequence(index, backend, tokens);
    require(again == path, "recomputed blocks must resolve to the existing nodes");
    require(backend.live.size() == live_before + 3, "the index must keep the recomputed pages");
    for (const NodeRef node : path) {
        const NodeView view = index.node(node);
        require(view.device == CopyState::Resident && view.host == CopyState::Resident,
                "a reattached block is resident on both tiers");
    }
    // Device-resident duplicates stay private to the inserting sequence.
    const auto twice = insert_sequence(index, backend, tokens);
    require(backend.live.size() == live_before + 3, "a resident duplicate must not be retained");
    index.release_path(twice);
    index.release_path(again);
    index.check_invariants();
    require(index.evict_device_blocks(3) == 3, "reattached backed blocks are evictable again");
    index.check_invariants();
}

// A saved Host tier keeps exactly the Host-backed snapshots whose paths are Host-resident, and a
// fresh index rebuilt from it matches the same prompts through host-only blocks.
void test_persistence_roundtrip() {
    RecordingBackend backend;
    PrefixCacheIndex source(small_config(64, 2), backend);
    const auto tokens = make_tokens(64 * 3, 40);
    const auto path   = insert_sequence(source, backend, tokens);
    for (const NodeRef node : path) { backup_node(source, node); }
    const SnapshotRef backed = publish_tap(source, path[1]);
    backup_snapshot(source, backed);
    const auto tail_tokens      = make_tokens(10, 41);
    const auto slot             = source.acquire_device_slot(false);
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        source.publish_snapshot(path[2], 202, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    backup_snapshot(source, endpoint.snapshot);
    // A Device-only snapshot on an unbacked path is not persistable.
    const auto other      = make_tokens(64, 42);
    const auto other_path = insert_sequence(source, backend, other);
    const auto other_slot = source.acquire_device_slot(true);
    require(other_slot.has_value(), "no slot for an unbacked snapshot");
    (void)source.publish_snapshot(other_path[0], 64, {}, std::nullopt, *other_slot,
                                  SnapshotKind::Tap);
    source.release_path(path);
    source.release_path(other_path);

    std::vector<NodeRef> nodes;
    std::vector<SnapshotRef> snapshots;
    source.collect_persistable(nodes, snapshots);
    require(nodes.size() == 3 && snapshots.size() == 2,
            "only Host-backed snapshots and their paths are persistable");

    PrefixCacheIndex restored(small_config(64, 2), backend);
    std::vector<std::pair<NodeRef, NodeRef>> mapping;
    const auto mapped = [&](NodeRef ref) {
        for (const auto& [from, to] : mapping) {
            if (from == ref) { return to; }
        }
        return NodeRef{};
    };
    for (const NodeRef node : nodes) {
        const BlockIdentity identity = source.block_identity(node);
        const auto block             = restored.restore_host_block(
            mapped(identity.parent), identity.lookup_hash, identity.tokens, identity.extra);
        require(block.has_value(), "a persisted block did not restore");
        mapping.emplace_back(node, block->node);
    }
    for (const SnapshotRef snapshot : snapshots) {
        const SnapshotView view = source.snapshot(snapshot);
        require(restored
                    .restore_host_snapshot(mapped(view.anchor), view.frontier, view.tail, view.kind,
                                           view.hits)
                    .has_value(),
                "a persisted snapshot did not restore");
    }
    restored.check_invariants();

    auto prompt = tokens;
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(3);
    const auto hashes = block_lookup_hashes(prompt, {});
    const MatchResult match =
        restored.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 2 && match.candidates[0].frontier == 202 &&
                match.candidates[0].host_only_blocks == 3 && !match.candidates[0].image_on_device,
            "a restored Host tier must match through host-only blocks and images");

    // A full Host tier restores what fits and nothing else.
    PrefixCacheIndex tiny(small_config(1, 2), backend);
    const BlockIdentity first = source.block_identity(nodes.front());
    require(tiny.restore_host_block(NodeRef{}, first.lookup_hash, first.tokens, first.extra)
                .has_value(),
            "one slab must hold one block");
    const BlockIdentity second = source.block_identity(nodes[1]);
    require(!tiny.restore_host_block(NodeRef{}, second.lookup_hash, second.tokens, second.extra)
                 .has_value(),
            "restoring must not evict to make room");
    tiny.check_invariants();
}

void test_tail_device_fill() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 2), backend);
    const auto tokens = make_tokens(64, 31);
    const auto path   = insert_sequence(index, backend, tokens);
    backup_node(index, path[0]);
    const auto tail_tokens      = make_tokens(10, 32);
    const auto slot             = index.acquire_device_slot(false);
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(path[0], 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    require(endpoint.created, "endpoint not created");
    backup_snapshot(index, endpoint.snapshot);
    require(index.snapshot(endpoint.snapshot).host_slabs.size() == 5,
            "a host-backed tail needs its own slab");
    index.release_path(path);
    // Evict the block and the tail; both are host-backed so the snapshot survives.
    require(index.evict_device_blocks(2) == 2, "block and tail must be evictable");
    require(backend.live.count(tail_id) == 0, "the tail's device page must be released");
    const SnapshotView evicted = index.snapshot(endpoint.snapshot);
    require(index.valid(endpoint.snapshot) && evicted.tail_device_copy == CopyState::Absent,
            "a backed tail survives host-only");

    auto prompt = tokens;
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(7);
    const auto hashes = block_lookup_hashes(prompt, {});
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].tail &&
                !match.candidates[0].tail_on_device,
            "a host-only tail must be reported");

    // Abort then complete a restore of the tail.
    index.pin_snapshot(endpoint.snapshot);
    require_throws([&] { index.complete_tail_device_fill(endpoint.snapshot); }, "no fill open");
    const std::uint32_t first = backend.allocate();
    index.begin_tail_device_fill(endpoint.snapshot, first);
    index.abort_tail_device_fill(endpoint.snapshot);
    require(backend.live.count(first) == 0, "an aborted tail fill must return its page");
    const std::uint32_t second = backend.allocate();
    index.begin_tail_device_fill(endpoint.snapshot, second);
    index.complete_tail_device_fill(endpoint.snapshot);
    index.unpin_snapshot(endpoint.snapshot);
    index.check_invariants();
    const SnapshotView restored = index.snapshot(endpoint.snapshot);
    require(restored.tail_device_copy == CopyState::Resident && restored.tail_device == second,
            "a completed tail fill is resident");
    require(index.evict_device_blocks(1) == 1 && backend.live.count(second) == 0,
            "a restored tail is evictable again");
    index.check_invariants();
}

void test_device_slots() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 1), backend);
    const auto a             = make_tokens(64, 13);
    const auto path_a        = insert_sequence(index, backend, a);
    const SnapshotRef snap_a = publish_tap(index, path_a[0]);
    require(!index.acquire_device_slot(false).has_value(),
            "an unbacked slot must not be taken without permission");
    backup_snapshot(index, snap_a);
    const auto slot = index.acquire_device_slot(false);
    require(slot.has_value(), "a backed slot must be reusable");
    require(index.valid(snap_a) && index.snapshot(snap_a).device_slot == kNoId,
            "the evicted slot's snapshot must survive host-only");
    index.release_device_slot(*slot);

    // Unbacked with permission: the owner is lost.
    const auto b             = make_tokens(64, 14);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_b = publish_tap(index, path_b[0]);
    const auto forced        = index.acquire_device_slot(true);
    require(forced.has_value() && !index.valid(snap_b), "unbacked slot eviction loses its owner");
    index.release_device_slot(*forced);
    index.release_path(path_a);
    index.release_path(path_b);
    index.check_invariants();
}

void test_image_only_host_tier() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config(8, 1);
    config.image_slabs       = 1;
    config.host_blocks       = false;
    PrefixCacheIndex index(config, backend);
    const auto a      = make_tokens(64, 15);
    const auto path_a = insert_sequence(index, backend, a);
    index.pin_node(path_a[0]);
    require(!index.begin_host_fill(path_a[0]).has_value(),
            "an image-only host tier must not back KV blocks");
    index.unpin_node(path_a[0]);

    // Endpoint with a tail; its host copy holds the image only.
    const auto tail_tokens      = make_tokens(10, 16);
    const auto slot             = index.acquire_device_slot(false);
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(path_a[0], 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    backup_snapshot(index, endpoint.snapshot);
    require(index.snapshot(endpoint.snapshot).host_slabs.size() == 1,
            "an image-only host copy must not reserve a tail slab");
    index.release_path(path_a);
    index.check_invariants();

    // The tail is unbacked: evicting it loses the snapshot even though its image is on host.
    std::uint32_t released = 0;
    while (index.valid(endpoint.snapshot) && released < 4) {
        released += index.evict_device_blocks(1);
        index.check_invariants();
    }
    require(!index.valid(endpoint.snapshot), "a snapshot without its tail KV must be removed");
    require(index.stats().host_free_slabs == 8, "the removed snapshot's image slab was leaked");
}

void test_random_stress() {
    std::mt19937 rng(1234);
    for (int round = 0; round < 20; ++round) {
        RecordingBackend backend;
        PrefixIndexConfig config = small_config(40, 3);
        config.max_nodes         = 128;
        config.max_snapshots     = 16;
        PrefixCacheIndex index(config, backend);
        // A small vocabulary of shared prefixes so paths overlap.
        std::vector<std::vector<TokenId>> prompts;
        for (int p = 0; p < 6; ++p) {
            auto tokens = make_tokens(64 * (2 + p % 4), 100 + round);
            auto suffix = make_tokens(64 * 3, 200 + p + round * 10);
            tokens.insert(tokens.end(), suffix.begin(), suffix.end());
            tokens.push_back(1);
            prompts.push_back(std::move(tokens));
        }
        std::vector<std::vector<NodeRef>> held;
        for (int step = 0; step < 300; ++step) {
            const int op = static_cast<int>(rng() % 6U);
            if (op <= 1 && held.size() < 3) {
                const auto& prompt           = prompts[rng() % prompts.size()];
                const auto hashes            = block_lookup_hashes(prompt, {});
                const auto n                 = static_cast<std::uint32_t>(prompt.size());
                MatchResult match            = index.match(prompt, hashes, {}, n);
                const AdmissionChoice choice = index.choose(match, n);
                std::uint32_t reuse_blocks   = 0;
                if (choice.candidate) {
                    const MatchCandidate& candidate = match.candidates[*choice.candidate];
                    reuse_blocks                    = candidate.path_blocks;
                    index.note_hit(candidate.snapshot);
                }
                std::vector<NodeRef> path(match.path.begin(), match.path.begin() + reuse_blocks);
                index.acquire_path(path);
                for (const NodeRef node : path) {
                    if (index.node(node).device == CopyState::Absent) {
                        index.begin_device_fill(node, backend.allocate());
                        index.complete_device_fill(node);
                    }
                }
                NodeRef parent = path.empty() ? NodeRef{} : path.back();
                for (std::uint32_t b = reuse_blocks; b < hashes.size(); ++b) {
                    if (index.stats().nodes + 2 >= config.max_nodes) {
                        (void)index.evict_device_blocks(4);
                    }
                    const std::span<const TokenId> block(prompt.data() + b * kBlockTokens,
                                                         kBlockTokens);
                    const std::uint32_t id    = backend.allocate();
                    const InsertResult result = index.insert_block(parent, hashes[b], block, 0, id);
                    if (!result.device_attached) { backend.release_device_block(id); }
                    path.push_back(result.node);
                    parent = result.node;
                    if (index.node(result.node).host == CopyState::Absent &&
                        index.node(result.node).device == CopyState::Resident && rng() % 2U) {
                        index.pin_node(result.node);
                        if (index.begin_host_fill(result.node)) {
                            index.complete_host_fill(result.node);
                        }
                        index.unpin_node(result.node);
                    }
                    if (rng() % 3U == 0) {
                        if (const auto slot = index.acquire_device_slot(rng() % 2U)) {
                            const PublishResult published = index.publish_snapshot(
                                result.node, (index.node(result.node).depth + 1U) * kBlockTokens,
                                {}, std::nullopt, *slot, SnapshotKind::Tap);
                            if (published.created && rng() % 2U) {
                                index.pin_snapshot(published.snapshot);
                                if (index.begin_snapshot_host_fill(published.snapshot)) {
                                    index.complete_snapshot_host_fill(published.snapshot);
                                }
                                index.unpin_snapshot(published.snapshot);
                            }
                        }
                    }
                    index.check_invariants();
                }
                held.push_back(std::move(path));
            } else if (op <= 3 && !held.empty()) {
                const std::size_t which = rng() % held.size();
                index.release_path(held[which]);
                held.erase(held.begin() + static_cast<std::ptrdiff_t>(which));
            } else {
                (void)index.evict_device_blocks(1 + rng() % 8U);
            }
            index.check_invariants();
        }
        for (const auto& path : held) { index.release_path(path); }
        (void)index.evict_device_blocks(1000);
        index.check_invariants();
        // Every device page not held by an index node was released exactly once.
        std::uint32_t resident = 0;
        for (std::uint32_t id : backend.live) {
            (void)id;
            ++resident;
        }
        require(resident == index.stats().device_resident_blocks +
                                0U /* tails are never used in this test */,
                "device pages leaked or double-released");
    }
}

void test_tap_planner() {
    TapPlannerConfig config;
    config.max_new_taps   = 8;
    config.ladder_tokens  = 4096;
    config.min_gap_tokens = 1024;
    const std::vector<TapHint> hints{
        {100, TapHintKind::Structural},        {5000, TapHintKind::MessageBoundary},
        {9000, TapHintKind::MessageBoundary},  {21800, TapHintKind::MessageBoundary},
        {29990, TapHintKind::MessageBoundary}, {30010, TapHintKind::GenerationOpener},
        {12345, TapHintKind::Explicit},
    };
    const auto taps = plan_taps(30020, 0, hints, {}, {}, config);
    require(std::is_sorted(
                taps.begin(), taps.end(),
                [](const PlannedTap& a, const PlannedTap& b) { return a.position < b.position; }),
            "taps must be sorted");
    for (const PlannedTap& tap : taps) {
        require(tap.position > 0 && tap.position <= 30019, "tap outside the legal domain");
    }
    auto find = [&](const std::vector<PlannedTap>& planned,
                    std::uint32_t position) -> std::optional<PlannedTap> {
        for (const PlannedTap& tap : planned) {
            if (tap.position == position) { return tap; }
        }
        return std::nullopt;
    };
    require(find(taps, 12345) && find(taps, 12345)->placement == TapPlacement::Exact,
            "an explicit hint must be tapped exactly");
    require(find(taps, 30010) && find(taps, 30010)->placement == TapPlacement::Exact,
            "the generation opener must be tapped exactly");
    require(find(taps, 100) && find(taps, 100)->placement == TapPlacement::Exact,
            "a structural hint must be tapped exactly");
    require(!find(taps, 30019),
            "the prompt tail next to the generation opener covers nothing and must be dropped");
    // n - 8192 = 21828: the boundary 28 tokens below is within the gap, so the ladder snaps.
    require(find(taps, 21800) && find(taps, 21800)->placement == TapPlacement::Flexible,
            "the ladder must snap to a nearby message boundary and stay flexible");
    // n - 4096 = 25924 has no boundary within the gap: the ladder keeps its target.
    require(find(taps, 25924) && find(taps, 25924)->placement == TapPlacement::Flexible,
            "the ladder must keep its target without a nearby boundary");

    // A short user turn puts the system-block end and the generation opener in one cluster: the
    // earlier boundary is kept because a new conversation sharing the system block diverges
    // there, and the prompt tail next to it is dropped.
    const std::vector<TapHint> short_turn{{1085, TapHintKind::Structural},
                                          {1098, TapHintKind::GenerationOpener}};
    const auto clustered = plan_taps(1104, 0, short_turn, {}, {}, config);
    require(clustered.size() == 1 && clustered[0].position == 1085 &&
                clustered[0].placement == TapPlacement::Exact,
            "a semantic cluster must keep its earliest boundary alone");

    // An explicit breakpoint right after the system-block end does not replace it: only the
    // earlier snapshot serves a new conversation diverging between the two.
    const std::vector<TapHint> marked{{2109, TapHintKind::Structural},
                                      {2115, TapHintKind::Explicit},
                                      {2117, TapHintKind::GenerationOpener}};
    const auto both = plan_taps(2124, 0, marked, {}, {}, config);
    require(both.size() == 2 && both[0].position == 2109 && both[1].position == 2115,
            "an explicit breakpoint must not displace an earlier semantic boundary");

    // Without an opener near the end the prompt tail is a flexible tap.
    const std::vector<TapHint> plain{{100, TapHintKind::Structural}};
    const auto tail = plan_taps(3000, 0, plain, {}, {}, config);
    require(find(tail, 2999) && find(tail, 2999)->placement == TapPlacement::Flexible,
            "the prompt tail must be a flexible tap");

    // Resume base and existing snapshots suppress taps at or near them.
    const std::vector<std::uint32_t> existing{12345};
    const auto resumed = plan_taps(30020, 10000, hints, existing, {}, config);
    for (const PlannedTap& tap : resumed) {
        require(tap.position > 10000 && tap.position != 12345,
                "tap below base or duplicating an existing snapshot");
    }
    const std::vector<std::uint32_t> near_opener{30000};
    require(!find(plan_taps(30020, 0, hints, near_opener, {}, config), 30010),
            "a tap within the minimum separation of an existing snapshot must be dropped");

    // Exclusions move taps to the start of the excluded span.
    const std::vector<TapExclusion> exclusions{{12000, 13000}};
    const auto excluded = plan_taps(30020, 0, hints, {}, exclusions, config);
    for (const PlannedTap& tap : excluded) {
        require(!(tap.position > 12000 && tap.position < 13000), "tap inside a Vision span");
    }
    require(find(excluded, 12000).has_value(), "an excluded explicit tap moves to the span start");

    // Budget: the highest-priority tap survives.
    TapPlannerConfig one = config;
    one.max_new_taps     = 1;
    const auto single    = plan_taps(30020, 0, hints, {}, {}, one);
    require(single.size() == 1 && single[0].position == 12345, "tap budget or priority ignored");

    // Ladder size is logarithmic for a long prompt without hints.
    const auto ladder = plan_taps(240000, 0, {}, {}, {}, config);
    require(ladder.size() <= 7, "ladder must be logarithmic");
    require(plan_taps(1, 0, hints, {}, {}, config).empty(), "no taps for a 1-token prompt");
}

} // namespace

int main() {
    try {
        test_hashes();
        test_match_and_choice();
        test_collisions_and_tails();
        test_device_lru_order();
        test_backed_before_unbacked();
        test_host_dead_and_gdsf();
        test_device_slots();
        test_host_only_reattach();
        test_tail_device_fill();
        test_persistence_roundtrip();
        test_image_only_host_tier();
        test_random_stress();
        test_tap_planner();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "prefix cache index tests passed\n";
    return 0;
}
