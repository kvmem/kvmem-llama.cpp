# Hybrid Prefix Cache (HPC): design and implementation specification

Status: **implementation in progress**. HPC is an *alternative* prefix-cache mode selected at
launch with `--use-alt-prefix-caching` (`ContextCacheOptions::mode = ContextCacheMode::Hybrid`).
The existing system ([Resource scheduling and context cache](resource-scheduling-and-context-cache.md))
remains the default (`ContextCacheMode::Legacy`) and is not modified or removed; the two modes
coexist by explicit owner request. This document is the authority for the Hybrid mode.

Scope: an alternative to NInfer's prefix-reuse, checkpoint-retention and cache-pressure
system for `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` on one RTX 5090 (`sm_120a`), with
`max_concurrency` 1..8, every KV profile (BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16, K8V4),
and every speculative backend (none, MTP, DFlash, DFlash2).

In this line the mode runs on one device: pipeline stages (`--devices` with more than one entry)
are rejected with it, and so are the checkpoint catalog's own policies (`--auto-long-anchors`,
`--auto-prefix-grid`, `--context-cache-policy`, `--release-diverged-checkpoints`,
`--thorough-admission-search`, `--recency-eviction`, `--value-aware-demote`, the disk tier). Host
restores and snapshot copies take the rank-stream form of the KV and StateImage copy routines, and a
Vision prefill uses the concurrent overlay window when one is configured. The measurements below are the original
author's on an RTX 5090.

Coexistence rules:

- The mode is fixed at Engine construction. Legacy-only options (`--device-state-slots`,
  `--host-state-slots`, `--host-kv-mib`, `--max-private-continuations`, `--max-shared-prefixes`,
  `--max-long-anchors-per-continuation`, `--long-anchor-spacing`) are rejected together with
  `--use-alt-prefix-caching`; Hybrid-only options are rejected without it. `--host-cache-mib`
  applies to both modes (in Hybrid mode it sizes the Host slab pool); `--no-prefix-reuse`
  contradicts the mode and is rejected with it.
- Hybrid mode reuses the active-execution machinery unchanged: `LogicalKVPageStore`,
  `KVAddressSpaceStore`, execution rows, the active `StateStore` images, prefill/decode/speculative
  paths. It replaces only admission-source selection, retention, capture and pressure. Tree nodes
  hold non-writer references to logical pages; active address spaces map those pages exactly as a
  Legacy shared-prefix fork does.
- Nothing in §14.1 is deleted while Legacy remains a supported mode.

---

## Implementation status (branch `feat/hybrid-prefix-cache`)

Hybrid mode is implemented and selectable with `--use-alt-prefix-caching`. It configures itself:
the only capacity a deployment chooses is `--host-cache-mib` (default 8192, 0 = Device only);
every other value is derived from the rest of the configuration (§14.2).

| area | state |
|---|---|
| §5 index, §9 eviction (device LRU, host GDSF, dead KV), §7.1 tap planner | done; host-only unit tests (`ninfer_prefix_cache_index_test`) |
| §5.4 automatic Device sizing | done: in Hybrid mode the Main pool is not clamped to `C·L`, and `ninfer-serve` defaults `--kv-capacity` to `auto`, so free VRAM becomes Device block cache |
| §5.4 unified Host slab pool | done: KV blocks, snapshot images (split over slabs) and snapshot tails share one pinned pool; GDSF and the dead-KV sweep decide the split at run time |
| §6 admission as the Engine's materialization transaction | done: staging reserves every Device page and the state slot, Host restores run on a dedicated restore stream, and activation forks the lane at once; its Device work queues behind the copies it reads, layer by layer (§6.4, §6.5, §12.1) |
| §7.1 taps | done: exact taps (explicit, generation opener, structural) split the chunk; flexible taps (prompt tail, ladder) are realized at chunk boundaries at no extra forward pass |
| §7.4 tap publication | done: deferred until the anchoring blocks commit (a frontier inside a block, or an MTP backend one token behind); exact-frontier tails are copied into cache-owned pages |
| §7.6 block publication, write-through | done: blocks join the tree at every commit; a lane's blocks are written to Host in one batch when it releases them |
| §7.7 endpoint snapshots at finish and consistent abort | done |
| §6.2 cost model | done: the Engine's calibrated context-cost coefficients (prefill and Host-to-Device transfer) rank sources and value snapshots |
| §5.5 persistence across restarts | done (opt-in `--prefix-cache-file`) |
| §12.2 in-flight prefix coalescing | done: requests sharing a new prefix with a lane still prefilling wait for its snapshot at the divergence instead of prefilling the prefix again; in this line a request is admitted while another prefills only with `--concurrent-prefill`, so coalescing needs it |
| reuse-loss attribution | done: the request log's `materialization` record carries `cached_prefix_tokens` (longest cached block prefix, reusable or not), and `restored_host_bytes` |
| §7.2–§7.3 zero-split GDN state tap and phase alignment | not implemented: an exact tap costs one prefill split (about 15 ms per turn on 27B); flexible taps avoid it, and requests resuming from an endpoint skip the opener tap (§7.1) |
| §11.2 KV transfer Op | copy-engine path only (`cudaMemcpy2DAsync` runs over consecutive pages and slabs) |
| §6.3 persistent backfill proof | not issued: a blocked FIFO head is never overtaken |
| §12 optional features other than 12.2, §13.2 Op qualification | not implemented |
| §13.3 real-artifact scenarios | `ninfer_qwen3_5_hybrid_prefix_real_test`: Host vs Device restore exactness (with and without MTP), generation-opener and system-block reuse, Device-only mode, protocol cache hints, Vision, persistence across a restart; `NINFER_HYBRID_KV_DTYPE` runs them for every KV storage (bf16, int8, fp8, nvfp4, k8v4 pass). `ninfer_ngram_concurrent_real` runs on Hybrid with `NINFER_NGRAM_TEST_CONTEXT_CACHE=hybrid` |

Measured on an RTX 5090 (Windows, CUDA 13.4) with Qwen3.8-27B NVFP4
(`qwen3_8_27b_nvfp4-nvidia.ninfer`), `ninfer-serve --max-context 32768 --max-concurrency 2
--kv-dtype int8 --kv-capacity auto --prefill-chunk 2048 --host-cache-mib 12000`, plus
`--use-alt-prefix-caching` for Hybrid, a fresh server per workload:

| workload | Legacy | Hybrid |
|---|---|---|
| 4 conversations × 6 turns, thinking, shared ~5.9K-token system prompt: prompt tokens from cache | 83.0% | 90.9% |
| same: TTFT p50 / p90 / mean | 0.177 / 0.962 / 0.331 s | 0.149 / 0.283 / 0.200 s |
| same: TTFT median of later turns | 0.173 s | 0.129 s |
| 8 distinct ~11.4K-token prompts overflowing the Device pool, revisited: tokens reused on revisit | 0 of 8 | ~11,400 of each |
| cold ~8.6K-token prompts: server prefill / decode tok/s (median of 6) | 10.1k / 84.9 | 10.2k / 84.6 |
| 64-token prompt, 2048 output tokens: decode tok/s | 86.3 | 86.0–86.1 (86.1–86.3 with `--kv-capacity 65536`) |

With `--spec dflash2 --draft-tokens 7` and `--spec mtp --draft-tokens 3` the conversation workload
served 90.8% and 90.9% of prompt tokens from cache with no errors. At the Legacy pool size decode is
within noise; the default automatic pool (free VRAM becomes Device cache, several times larger)
costs about 0.2–0.35% decode throughput, attributed by the capped runs to KV placement across the
larger pool, not to per-round cache work.

In-flight coalescing (§12.2), `--max-concurrency 4`: four requests arriving together with a new
~13.9K-token system prompt served 74.9% of prompt tokens from cache (the three followers reuse
13,888 tokens each). Without coalescing they served 11.0%. Mean TTFT fell from 3.52 s to 1.48 s
and the burst's wall time from 5.56 s to 1.80 s.

Host restores (§6.5): copies run at 24–27 GB/s. Revisits of 8 ~11.4K-token prompts under two
clients, each restoring its 147 MiB state image, have a median TTFT of 45.7 ms, against 60.0 ms
when the admission waited for the copy. Revisits of two alternating ~90K-token conversations, each
restoring 2.6–2.8 GiB, have a TTFT of 268 ms with a short question (was 301 ms) and 1186 ms with a
3.4K-token tool output (was 1208 ms).

An Nsight Systems trace of a 44K-token revisit (1.49 GiB restored in 60 ms) shows the pipelining
at work. The first prefill pass runs its linear-attention layers at once and waits ~2.2 ms at each
attention layer for its KV, finishing as the copy does. The rest of the TTFT is not the restore:
the 45 new tokens ran as four passes over the model, split at exact snapshot and template
boundaries (~12–15 ms each with little attention). The prompt-attention kernel took ~2.3 ms per
attention layer over the 44K context in the one pass that used it.

Both are now addressed. Hybrid prefill no longer splits at the template's rewrite frontiers (§7.1),
and single-row prefill passes of 17–64 tokens take the chunked small-T (split-KV) attention route
once the context holds 64 visible keys per new token (80 for 16 query heads). That route is
4–12× faster than the prompt kernels at 16K–180K keys; for example 32 new tokens at 180K keys take
1.06 ms per layer instead of 9.5 ms. With ~90K cached tokens:

| revisit, one client | before | after |
|---|---|---|
| Device-resident, short question | 204 ms | 91 ms |
| Device-resident, 3.4K-token tool output | 1101 ms | 1050 ms |
| Host restore (2.6–2.8 GiB), short question | 268 ms | 170 ms |
| Host restore, 3.4K-token tool output | 1186 ms | 1067 ms |

Under two clients, ~11.4K-token Host revisits went from a 45.7 ms to a 30.1 ms median TTFT. Cold
prefill is unchanged (10.1k vs 10.2k tok/s).

## 0. Summary

| | Current system | HPC |
|---|---|---|
| Unit of reuse | owner/checkpoint (private continuation, shared prefix, 5 checkpoint kinds) | content-addressed 64-token KV block + sparse state snapshot |
| Identity | per-owner token ledgers, session index, shortlist keys, markers | exact token path in one radix tree (hash only for lookup) |
| KV sharing across sessions | only through an optional shared-prefix publication that must beat a private baseline | unconditional: identical blocks are one physical page |
| State checkpoint capture | prefill chunk split at the frontier + copy | a "tap" output of the chunked GDN kernel at any 64-aligned position, no split |
| Admission | bounded heuristic search over complete targets; up to 250 ms planning allowance per admission boundary | O(prompt blocks) lookup + arithmetic capacity check; target < 1 ms p99 |
| Pressure | monotone degradation graph, joint post-state projection, ordered stage peaks, one global transaction | independent eviction per tier: device LRU and host GDSF (GreedyDual-Size-Frequency) |
| Host tier | variable-extent arena with allocator geometry and a separate StateImage slot count | one pinned slab pool; KV blocks and snapshots share it with no fixed split between them |
| Device ↔ host | demote on pressure (copy before release) | asynchronous write-through; device eviction never copies |
| Code | ≈25 k lines in `runtime/engine/context_cache`, `program/{planning,transactions,storage}`, host arena, resource contracts | estimated 5–6 k lines |

What the design changes for users:

- **More reuse.** Every committed full block of every request becomes reusable, including
  cancelled and partially prefilled requests. Snapshots are deduplicated by content, so one
  snapshot serves every session that shares its prefix. Tap density is limited only by host memory,
  not by per-continuation anchor counts or catalog slots.
- **Lower TTFT.** No planning search. Host restores run asynchronously while other lanes keep
  decoding, and cost about 3.8 ms per snapshot plus about 0.65 ms per 1 k INT8 tokens. Capture
  frontiers no longer produce short split chunks.
- **Throughput unchanged or better.** Kernels, page layout, block tables and CUDA Graph keys are
  unchanged. Taps add under 0.1 ms per 4 k-token chunk. Write-through uses about 0.3 % of PCIe
  bandwidth.
- **Better memory efficiency.** Shared prefixes are stored once. KV that no snapshot can reach
  ("dead KV") is reclaimed first. The host pool has no fragmentation and no split between KV and
  state.

---

## 1. Why the current system is replaced

Evidence is taken from the current sources and documents.

1. **Ownership instead of content.** Reuse depends on which *owner* holds a checkpoint: private
   continuation, shared prefix, SessionIndex binding, or one of five checkpoint kinds
   (`SessionEndpoint`, `TurnClosure`, `ResponseReplay`, `LongAnchor`, `SharedStablePrefix`). It is
   also bounded by catalog slots (`max_private_continuations`, `max_shared_prefixes`,
   `max_long_anchors_per_continuation`). Two sessions with the same 30 k-token system+tools prefix
   share KV only if a shared prefix was published. Publication must be *strictly better than the
   private-only baseline* (resource-scheduling §7.2, §8.3).
2. **Planning cost on the critical path.** Materialization runs a bounded heuristic search over
   complete pressure targets. The Program projects joint post-states, stage peaks and Host
   allocator geometry for each target (§7.3–§8.7). Commit `1a0c64bb` gives every admission
   boundary a 250 ms planning allowance. That time sits directly in TTFT for the head request and
   stalls the worker for everyone else.
3. **Split chunks.** A checkpoint at an arbitrary frontier splits a prefill chunk (`text.cpp`
   `prefill_split_frontier_`), producing a short, inefficient chunk. `rewrite_execution_frontiers`
   adds more splits so replay matches the original decomposition.
4. **Host fragmentation and partitioning.** Host KV is a variable-extent arena whose feasibility
   depends on extent geometry, not free bytes (paged-kv §5.3). Host StateImages are a separate
   slot count. `--host-cache-mib` resolves a *static* split between them at startup.
5. **Fragility.** Correctness depends on invariants spread across the ResourceManager and Program
   boundary: capability generations, resource revisions, owner edges, claims, seal windows,
   borrowed-read ownership, and absolute ResourceResult adoption. Recent fix history
   (planning races, shared catalog saturation, lease reclaim) shows the maintenance cost.
6. **Size.** About 25 k lines: `context_cache/` 7.0 k, `planning/` pressure/recovery/request 6.6 k,
   `transactions/` 4.5 k, `storage/` 5.2 k, Host arena + prefix identity + resource contracts
   1.8 k.

The fundamental constraint does not change. Qwen3.5 is hybrid: 48/64 (27B) or 30/40 (35B-A3B)
layers are Gated DeltaNet with a recurrent state that cannot be rolled back. A prefix is therefore
reusable only at a position where the complete recurrent state exists. HPC keeps that rule and
builds everything else around making such states **cheap to create, cheap to store, and exactly
addressable**.

---

## 2. Prior art used

| Source | Idea adopted |
|---|---|
| vLLM automatic prefix caching | Full blocks keyed by (parent, tokens, extra keys). Reference-counted. Freed blocks go to an LRU queue in reverse order, so tail blocks are evicted first. Evictable blocks count as free capacity. |
| vLLM hybrid/Mamba prefix caching | Recurrent state is cached only at block-aligned positions. A hit is the deepest aligned position that has state. |
| SGLang RadixAttention / MambaRadixCache | Radix tree over token blocks. Recurrent-state entries attach to tree nodes with their own eviction. Leaf-first eviction. |
| SGLang HiCache | Host tier with write-through, batched or kernel-based non-contiguous transfers, and layer-ordered loading. |
| llama.cpp context checkpoints | A few full-state checkpoints per sequence for recurrent/SWA models. |
| LMCache | Layer-wise pipelined KV loading overlapped with compute. |
| GreedyDual-Size-Frequency (Cherkasova, 1998) | Size- and cost-aware eviction with O(1) aging via an inflation value `L`. |

NInfer-specific additions:

1. The GDN **state tap**: chunked GDN already produces the state at every 64-token intra-chunk
   boundary, so exposing it costs one extra state write.
2. **Phase alignment** by identity-transition padding, so taps land on absolute page boundaries.
3. One pinned **slab pool** for both KV and state.
4. **Dead-KV** accounting: KV that no snapshot can reach has zero reuse value.

---

## 3. Model facts that size the design

### 3.1 KV bytes

Per token, all full-attention layers, K+V including scales (from paged-kv §4.3):

| profile | per token/head | Qwen3.6/3.8-27B (16 FA × 4 KV heads) | page (64 tok) | 35B-A3B (10 × 2) | page |
|---|---:|---:|---:|---:|---:|
| BF16 | 1024 B | 64.0 KiB | 4.00 MiB | 20.0 KiB | 1.25 MiB |
| INT8-G64 | 528 B | 33.0 KiB | 2.06 MiB | 10.3 KiB | 0.64 MiB |
| FP8-row256 | 516 B | 32.3 KiB | 2.02 MiB | 10.1 KiB | 0.63 MiB |
| K8V4 | 402 B | 25.1 KiB | 1.57 MiB | 7.9 KiB | 0.49 MiB |
| NVFP4-G16 | 288 B | 18.0 KiB | 1.13 MiB | 5.6 KiB | 0.35 MiB |

MTP adds one layer's worth of pages to the bundle (+1/16 for 27B). A DFlash draft with
full-attention layers adds its own BF16 pool.

### 3.2 State image

Fixed per-sequence state (qwen3_5-model "State and numerical boundaries", dflash "Backend storage"):

| component | 27B | 35B-A3B |
|---|---:|---:|
| GDN recurrent, FP32 `[128,128,Hv]` × GDN layers | 144 MiB | 60 MiB |
| conv history, `(2·16·128 + Hv·128)·3` BF16 × GDN layers | 2.81 MiB | 1.41 MiB |
| continuation hidden | 10 KiB | 4 KiB |
| DFlash2 local rings (official config) | 40 MiB | config-derived |
| **total (DFlash2)** | **186.8 MiB** (matches the 195,897,344 B Host StateImage in README) | — |

One 27B snapshot equals about 91 INT8 pages (≈5.8 k tokens of KV), 166 NVFP4 pages, or 47 BF16
pages. **Snapshots dominate storage cost. Their placement is the main policy decision.**

### 3.3 RTX 5090 transfer and compute budget

These are planning values. The implementation measures them at startup (§11.4) and uses the
measurements.

| operation | value used for design |
|---|---|
| HBM D2D copy of one 186.8 MiB image (read+write) | ≈ 0.25 ms |
| PCIe 5.0 x16 pinned H2D / D2H (practical) | ≈ 50 GB/s each direction |
| H2D snapshot | ≈ 3.9 ms |
| H2D 100 k tokens INT8 KV (3.38 GB) | ≈ 68 ms |
| Prefill, 27B, ~121 k context (README median) | ≈ 4.2 k tok/s, so 100 k tokens ≈ 24 s |
| GDN tap write (144 MiB) inside a chunk | ≈ 0.09 ms |
| Write-through rate at 4.2 k tok/s INT8 prefill | ≈ 140 MB/s (0.3 % of PCIe) |

Consequences:

- A host-resident prefix is about 300× cheaper to restore than to recompute. **The host tier
  carries most of the reuse value.** The device tier is a latency optimisation.
- A state snapshot is cheap to create (a tap costs about 0.01 % of a chunk) and cheap to move
  (4 ms). **Snapshots can be dense. The limit is host bytes, not time.**

---

## 4. Architecture

```text
                 Frontend (gateway threads)
   tokens, media digests, boundary hints, per-block chained hashes
                          │ PreparedPrompt
                          ▼
┌──────────────────────── Engine worker (single mutation owner) ───────────────────────┐
│ Scheduler (unchanged FIFO/backfill)                                                  │
│      │ quote_admission / admit / advance_prefill / decode / finish / poll_transfers  │
│      ▼                                                                               │
│ Program                                                                              │
│  ├─ PrefixCache (model-agnostic core, src/runtime/prefix_cache/)                     │
│  │    BlockTree ─ nodes: 64 tokens + extra key, device page ids, host slab, refs     │
│  │    SnapshotIndex ─ snapshots attached to nodes (+ optional tail)                  │
│  │    DeviceEvictor (LRU, tail-first)     HostEvictor (GDSF + dead-KV sweep)         │
│  │    TapPlanner                           CostModel (calibrated prefill/PCIe)       │
│  ├─ Qwen3.5 binding (src/models/qwen3_5/program/prefix/)                             │
│  │    StateImageLayout serialisation, tap wiring, MTP/DFlash coverage rules          │
│  ├─ Device typed KV pools (unchanged layout) + active block tables (unchanged)       │
│  ├─ Device state pool: C active slots + D snapshot slots                             │
│  └─ TransferEngine: h2d stream (high prio), d2h stream (low prio), events            │
│                                                                                      │
│ HostSlabPool (src/core/host_slab_pool.*): one pinned allocation, fixed-size slabs    │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

Ownership against AGENTS.md boundaries:

- **Core** owns physical primitives: `HostSlabPool`, device page pools, transfer launch helpers,
  and the KV transfer Op.
- **Ops** own the GDN tap/phase extension, conv/ring/hidden tap gathers and the KV page transfer
  kernel. Each has its own contract and oracle.
- **Program** owns the whole cache: exact identity, physical residency, refs, eviction, taps.
  There is no cross-layer logical/physical split, because the cache's policy decisions are
  local physical facts. In Hybrid mode `ResourceManager`, `MaterializationPlanner`,
  `SharedCapturePlanner`, portfolio value and the target/pressure machinery are not constructed
  or called; they remain the Legacy-mode implementation.
- **Runtime** keeps the scheduler, the common request contracts and a small admission-quote
  contract (§8.1).
- **Frontend** supplies tokens, media digests, boundary hints and block hashes. It no longer
  produces cache "candidates", evidence flags or session keys for the cache.

---

## 5. Data model

### 5.1 Block key and exact identity

Block `b` of a prompt covers absolute positions `[64b, 64b+64)`.

```text
BlockKey(b) = ( parent NodeId,
                tokens[64b .. 64b+63],               // TokenId, exact
                extra = media_key(b) )               // 0 when no Vision item overlaps block b
media_key(b) = xxh3_64 over, for each Vision item whose token span overlaps block b and every
               earlier item (Vision changes later MRoPE positions via rope_delta):
               (content_digest[32], modality, grid, span begin/length)
lookup_hash(b) = xxh3_64(lookup_hash(b-1) || tokens || extra), lookup_hash(-1) = seed
```

- `lookup_hash` only selects a child in the parent's child map. A hit always compares
  `tokens` and `extra` exactly. A forced hash collision must not produce a false hit (tested,
  §13.1).
- Everything else that affects KV or state is fixed for the Engine lifetime: model artifact,
  KV profile, speculative backend, RoPE/YaRN options. None of it is part of the key. Positions are
  a function of tokens and media, so they are covered by the key.
- Thinking mode, templates, tools and roles are all rendered into tokens, so they need no key
  field.
- The Frontend computes `lookup_hash` for every full block during preparation, off the worker, and
  stores it in `PreparedPrompt::block_hashes`.

### 5.2 Nodes

```cpp
struct BlockNode {                      // host-side; stored in a slab-allocated arena
    NodeId   id; uint32_t generation;   // generation-checked handles
    NodeId   parent;
    uint32_t depth;                     // block index b; the node covers [64b, 64b+64)
    uint64_t lookup_hash, extra;
    TokenId  tokens[64];                // exact tokens (256 B)
    ChildMap children;                  // small open-addressed map keyed by lookup_hash
    // Residency
    int32_t  device_page[kMaxPools];    // per enabled pool; -1 when not device resident
    int32_t  host_slab;                 // -1 when not host resident
    Residency device_state;             // Absent | Resident | Filling(event)
    Residency host_state;               // Absent | Resident | Filling(event)
    // Liveness
    uint32_t active_refs;               // active sequences whose table maps this node
    uint32_t restore_pins;              // in-flight transfers reading/writing this node
    uint32_t live_snapshots_below;      // snapshots whose path includes this node
    uint64_t last_access_tick;
    IntrusiveLink device_lru;           // present iff active_refs==0 && device Resident
    IntrusiveLink host_dead;            // present iff live_snapshots_below==0 && no refs && host Resident
    SnapshotId first_snapshot;          // snapshots anchored here (intrusive list)
};
```

A node is only inserted when **all enabled pools** have committed all 64 positions of the block
(§7.6). A node may be device-only, host-only or both. A node with neither copy is deleted, along
with its subtree (§9.4).

### 5.3 Snapshots

```cpp
struct Snapshot {
    SnapshotId id; uint32_t generation;
    NodeId   anchor;                    // node of the last full block below frontier (root if F<64)
    uint32_t frontier;                  // F: the snapshot state is after tokens [0,F)
    uint8_t  tail_len;                  // F - 64*(anchor.depth+1), 0..63
    TokenId  tail_tokens[63];           // exact; only tail_len are meaningful
    // Tail KV (only when tail_len>0): one private, immutable page bundle
    int32_t  tail_device_page[kMaxPools];  int32_t tail_host_slab;
    // State image
    int16_t  device_slot;               // index into the D device snapshot slots, or -1
    SlabList host_slabs;                // ceil(image_bytes / slab_bytes) slabs, or empty
    Residency device_state, host_state;
    uint32_t restore_pins;
    SnapshotKind kind;                  // Tap | Endpoint | OutputBoundary (telemetry only)
    // Host eviction (GDSF)
    double   gdsf_h; uint32_t freq;
    uint64_t last_hit_tick;
};
```

- **Tap snapshots** have `tail_len == 0`: frontier on a page boundary, KV fully in the tree.
- **Endpoint snapshots** are taken at request finish or cancel at an exact frontier. They own the
  partial last page, which moves from the finishing sequence without a copy.
- A snapshot is **valid** iff every node on `path(anchor)`, the tail bundle (if any) and the image
  each have at least one resident copy (device or host). Validity is maintained eagerly:
  removing the last copy of a node deletes its subtree's snapshots (§9.4).
- There is at most one snapshot per `(anchor, tail tokens)`. Publishing a duplicate is a no-op
  that refreshes recency.

### 5.4 Memory pools

- **Device KV pools**: unchanged geometry, plane order, block-table matrix and consumer views
  (paged-kv §3–§4, §6.4, §10–§11). A page group is in exactly one state: `Free`, `Active`
  (owned by one sequence, private), `Cached` (owned by a node or snapshot tail), or `Filling`
  (transfer destination). `available = free + evictable_cached`.
- **Main pool size**: automatic sizing keeps the capacity curve. In Hybrid mode the upper clamp
  `M_max = C·L` is removed (only the int32 token representation bounds it): pages beyond the active
  leases are cache-only device capacity. `ninfer-serve` defaults `--kv-capacity` to `auto` in
  Hybrid mode. Active admission still requires `M ≥ max(L, C)`.
- **Device state pool**: `C` active slots plus `D = device_snapshot_slots` snapshot slots in the
  existing slot-indexed layout (`ssm_states [128,128,Hv,C+D]` per layer, and so on). Snapshot slots
  are also the staging destination for taps (§7.4). Default `D = C + 1` with a Host tier (one
  resident snapshot per live conversation plus one staging slot) and `C + 2` without one.
- **HostSlabPool**: one pinned allocation of `host_cache_bytes`, divided into slabs. A slab holds
  one block bundle: the Main page packed with its Host page layout, then the backend page at the
  next 4 KiB boundary; `slab_bytes` is that record rounded up to 4 KiB. A snapshot image uses
  `image_slabs = ceil(image_bytes / slab_bytes)` slabs, not necessarily contiguous (the image's
  packed byte `o` lives in slab `o / slab_bytes`), and a snapshot with a tail one slab more. The
  free list is an index stack, O(1). There is no fragmentation and no geometry check: capacity is
  `free_slabs`. A nonzero budget smaller than one snapshot plus one block is rejected at startup.
  The pool is pinned in chunks of at most 4 GiB of whole slabs (one very large pinned allocation can
  fail or stall under WDDM); a slab never crosses chunks. A strided copy run joins consecutive
  pages only while their slab records share a chunk and advance by one pitch no larger than the
  device's maximum copy pitch. Any two records form a candidate run, so records in different
  chunks, or gigabytes apart in one, must be split explicitly. A 52 GB Host tier otherwise produced
  an invalid `cudaMemcpy2DAsync` that aborted the server.

### 5.5 Persistence (opt-in)

With `persistent_file` (`--prefix-cache-file`), the Host tier outlives the process. `ninfer-serve`
resolves the path to an absolute one at launch. It rejects a directory, a missing parent
directory, or `--host-cache-mib 0`, so an unusable location fails before any caching:

- **Save**, in the worker's orderly stop when the Engine is destroyed (clean shutdown: Ctrl+C,
  Ctrl+Break, or closing the console window on Windows, where `ninfer-serve`'s console handler
  stops the server and blocks while `main` unwinds; Windows ends the process about 5 s after a
  close, and an unfinished save leaves the previous file in place).
  `Program::shutdown_cleanup` releases every lane first, so requests still in flight write their
  committed blocks through, and saves before the cleanup drops the cache. Every Host write
  lands, then every Host-resident snapshot whose anchor path is Host-resident, and exactly those
  paths, are written parents first with their slab bytes (a temporary file renamed into place).
  Dead KV and Device-only content are not saved. A worker that ended on a fatal error saves
  nothing, so the previous file stays in place. A worker recovery (OOM or recoverable logic
  error) empties the cache like every cleanup, so a later save holds only what was cached after it.
- **Load**, at Engine construction before any request: the file is used only when its fingerprint
  (absolute artifact path, size and modification time, prefill signature, KV storage, speculative
  backend, RoPE scaling, and the product binary's identity — build id plus executable size and
  time, so any rebuild invalidates it) and its Host geometry equal the running Engine's. Entries
  are rebuilt Host-only, parents before children, without evicting anything: a smaller Host tier
  restores a prefix of the file, and its first requests restore blocks and images through the
  ordinary Host restore path. A damaged file loads nothing.
- The startup log reports what was restored or why nothing was; the shutdown log reports the save.

---

## 6. Admission (lookup, choice, capacity)

### 6.1 Lookup

`PrefixCache::match(prompt)` runs on the worker:

1. Walk from the root. For block `b < floor((n-1)/64)`, find the child by `block_hashes[b]`,
   then compare tokens and extra exactly. Stop at the first miss. This gives the matched path
   `N_0..N_{k-1}`, `k` full blocks.
2. Collect candidate snapshots: every snapshot anchored on the matched path (walk each node's
   list) whose tail tokens equal `prompt[64·(anchor.depth+1) .. F)`, with **`F ≤ n − 1`**. At
   least one prompt token is always prefilled, so logits, continuation hidden and MTP/DFlash
   bridges come from the normal path. No full-hit path exists.
3. Frontiers inside a Vision token span are not valid resume points. Tap placement never creates
   them, and match rejects them defensively.

Complexity is O(k + snapshots on path). A 240 k prompt is 3,750 hash probes and exact 256 B
compares: tens of microseconds.

### 6.2 Choice

Candidates: root, the deepest valid snapshot `s*`, and the deepest valid snapshot whose image and
path are fully device resident, `s_dev`. At most three. For each:

```text
cost(s) = restore_seconds(s) + prefill_seconds(base=F_s, tokens=n-F_s)
restore_seconds = (host_path_bytes + host_image_bytes + host_tail_bytes) / measured_h2d_bw
                  + per_batch_latency
```

`prefill_seconds` is the existing calibrated machine model (`context_cost.cpp`: linear tokens
plus `B·S + S(S+1)/2` attention pairs, keyed by `prefill_signature`). Pick the minimum. Ties
favour fewer transferred bytes. This is O(1) and deterministic, and the cost model only ranks
feasible choices.

### 6.3 Capacity

```text
need_pages(s) = host_only_path_blocks(s)          // H2D destinations (shared once restored)
              + (s.tail_len>0 ? 1 : 0)            // private COW copy of the tail
              + growth_pages(prompt n, base F_s)  // existing entitlement: remaining prompt +
                                                  // bounded output window + per-lane cushion
need_state_slot = 1 active slot (a lane implies a free active slot)
```

The request is admissible with `s` iff `need_pages(s) ≤ available_pages`, where evictable pages
exclude pages pinned by this admission's own path. If not, try the next cheaper-to-materialise
candidate: `s_dev`, then root.

- Root infeasible because active entitlements hold the pages: `TemporarilyBlocked`.
- Root infeasible even on an empty cache: `PermanentlyInfeasible`.

This is the same classification as today, computed by arithmetic.

**Backfill proof** (engine-architecture §5.2) uses the same arithmetic:
`need(borrower) + need(head root) ≤ free + all_evictable + Σ donor entitlements released`.

Every unpinned Device-resident cache entry (node or snapshot tail) owns its page bundle
exclusively, so evicting one returns exactly one page to each pool and the arithmetic is exact.
Exact-frontier tap tails are copied into their own pages for this reason (§7.4). Candidates with a
block another admission is still filling are skipped.

### 6.4 Admit

Admission is the Engine's materialization transaction (`hybrid_reserve_materialization`, then
`progress_context_transaction` until `Published`). The first progress step stages it
(`hybrid_stage`), in the same Engine call as the quote:

1. Re-validate the quoted snapshot and path. Pin every path node up to `F_s` (removed from the
   device LRU) and pin `s`.
2. Evict from the device LRU until the pools hold every page the admission needs, then **reserve**
   them: `entitlement − F_s/64` for the fork, plus one per host-only path block and host-only tail.
   Reserve the lane's StateImage destination slot.
3. If anything is host-only, open a restore batch on the dedicated restore stream (ordered after
   the compute stream's queued work, since destinations may have been read by earlier kernels):
   - host-only blocks: materialize destination pages from the reservation, adopt them as cache
     references, mark the nodes `Filling`, and copy the slabs in runs of consecutive pages and
     slabs;
   - host-only tail: the same into a cache-owned page (the snapshot's tail is Device-resident
     again afterwards);
   - host-only image: copy the image slabs straight into the reserved destination slot.
   The batch is enqueued in the order a forward pass reads it (§6.5).

Activation (`hybrid_activate`) then builds the lane at once, without waiting for the copies: the
reservations return to the pools and the page-prefix fork takes them back; path pages are shared; a
partial tail is D2D-copied into a private page; the state is reset, D2D-copied from the snapshot's
device slot, or restored in place. The lane takes over the path pins and the snapshot pin is
dropped; a restore batch holds its own pins on the nodes and snapshot it copies until it lands.

Admission never evicts anything that another sequence has pinned. Evictions come from the device
LRU (§9.1) and complete synchronously: dropping a device copy is bookkeeping, since the data is
already backed up or deliberately discarded.

### 6.5 Restoring

A restore batch never holds its admission. `submit_restore` enqueues it on the restore stream in
the order a forward pass reads it, recording an event after each part:

1. the **prelude**: the snapshot tail bundle, every restored block's backend page, and the image's
   continuation hidden and DFlash local state;
2. then, for each model layer in forward order, that layer's state: a full-attention layer's KV
   planes of every restored block, or a linear-attention layer's conv and recurrent state.

`land_restore` makes the compute stream wait for the prelude (activation copies the tail and the
MTP bridge reads the backend pages and hidden) and hands the lane a ticket for the per-layer
events. The lane's first prefill pass waits for each layer's event just before that layer
(`TextContext::run_layers`), so it computes the early layers while later ones are still arriving;
later passes are stream-ordered behind it. A lane released before its first pass makes the compute
stream wait for the whole batch, because the state slot it returns may still be a destination.

The index keeps the batch's nodes and tail `Filling`, and the batch keeps them and its snapshot
pinned, until `poll()` (every admission quote) or `drain()` sees the last event. So no eviction,
dead-KV reclaim or Host write can free a slab or page a copy still reads or writes. Other admissions
skip `Filling` candidates for that short window.

Consequences:

- A Host revisit no longer waits for the host to notice the copy finished: under concurrency that
  was the rest of the other lane's decode round (~12 ms measured), and with an idle Engine a timer
  tick.
- The copy overlaps the restored request's own first prefill chunk: for a long context, the prefill
  of new tokens hides most of a multi-GB restore.
- Other lanes' decode rounds queue behind that prefill chunk as for any prefill. When the chunk is
  much shorter than the copy (a few new tokens after a very long restored context), they can wait up
  to the copy's remaining time once.

Failure: a CUDA error in a restore job is Engine-wide, as today. A request cancelled while staged
(before activation) waits for the restore stream, returns every destination and reservation, and
releases its pins.

---

## 7. Prefill, taps, decode, finish

### 7.1 Tap planning

`plan_taps(n, base F_s, hints, existing frontiers, Vision exclusions)` returns taps sorted by
position, each `{p, placement}` with `F_s < p ≤ n − 1`, `p` not strictly inside a Vision token span
(a candidate inside one moves to the span start), and no existing or already-planned snapshot at
or near `p` on this path.

Candidates, in priority order:

| priority | source | placement | rationale |
|---|---|---|---|
| 1 | explicit markers (OpenAI breakpoints, Anthropic `cache_control`, C++ `PromptInput` markers) | exact | the client names reuse points |
| 2 | start of the final assistant generation opener | exact | the next turn re-renders the history after it (for example, stripped thinking) |
| 3 | end of tools, end of the leading System/Developer block | exact | compaction and new sessions sharing the preamble |
| 4 | `n − 1` ("prompt tail") | flexible | regenerate / retry of the same prompt |
| 5 | geometric ladder: for k = 0,1,…, `n − G·2^k` (G = `tap_ladder_tokens`), snapped back to a message boundary within the minimum gap | flexible | edits deep in history. Spacing grows with distance, so there are about log2(n/G) taps |

- **Exact** taps split the prefill chunk at `p` (one more forward pass each) and snapshot there.
  Semantic boundaries are where later prompts diverge, so the exact position is worth the split.
- **Flexible** taps never split: the snapshot is taken at the end of the chunk that reaches `p`, or
  at the start of the prompt's final chunk when `p` lies in it (the prompt tail always does). A
  chunk boundary costs nothing beyond the snapshot copy.
- Semantic boundaries (generation opener, structural) closer than 64 tokens form one cluster and
  only its earliest position is kept: a short user turn puts the system-block end next to the
  opener, and a snapshot at the system-block end serves new conversations sharing it while the
  next turn recomputes fewer than 64 tokens.
- A snapshot serves every prompt that diverges after it, so proximity only makes a *later*
  candidate redundant: a non-explicit candidate is dropped when an accepted tap, an existing
  snapshot on the path or the base lies at most 64 tokens before it. Ladder candidates keep
  `tap_min_gap` (default `max(1024, chunk)`) on both sides instead. The opener therefore absorbs
  the prompt tail in chat traffic, leaving one split per request.
- Only markers with explicit evidence (a client-named breakpoint) are priority 1. Protocol-automatic
  markers (OpenAI default caching, Anthropic automatic `cache_control`) are structural boundaries.
- A request that resumed from an endpoint snapshot proves its client echoes generated turns token
  for token (agent loops, preserved reasoning), so its own endpoint serves the next turn and the
  generation opener is not tapped. Measured on a tool-calling agent trace (27B NVFP4, INT8 KV,
  `--preserve-thinking`), the opener split cost about 15 ms of a ~70–130 ms turn TTFT. Requests
  that resumed from a tap keep the exact opener: their client re-renders history.
- Hints are the Frontend's existing `message_boundaries` plus the structural boundaries it already
  computes.
- At most `max_new_taps` (default 8 with a Host tier, 2 without) taps are kept per request.

Tap placement is a function of the prompt, so it is deterministic. Endpoint snapshots are exact
(§7.7).

Exact taps are the only prefill splits in Hybrid mode. The chat template's rewrite execution
frontiers (after each assistant header, after `<think>`, after the reasoning close) split Legacy
prefill for its rewrite checkpoints and execution provenance. Hybrid captures nothing there, so a
Hybrid lane ignores them. Each split is a whole extra pass over the model. A short chat turn used
to prefill as four passes (reuse point, opener, `<think>`, reasoning close). Now it takes two (to
the opener, then the rest), or one when the opener is dropped, unless a client breakpoint or
structural boundary falls inside the new tokens.

### 7.2 Phase-aligned chunked GDN

**Not implemented** (§7.2 and §7.3 together form the zero-split tap). The implemented tap copies
the lane's committed StateImage at a chunk boundary (≈0.25 ms D2D), so only exact taps pay a split.
Build this only if measurement shows the remaining exact-tap split matters.

The chunked GDN kernel's intra-chunk grid is relative to the call start. To make every internal
boundary an absolute multiple of 64, the Op gains a `phase = base % 64` parameter:

- The kernel treats the call as if it were left-padded by `phase` virtual tokens with
  `k = 0, v = 0, g = 0, β = 0`. For such a token, `α = exp(0) = 1` exactly and
  `delta = 0 · (v − αS·k) = 0` exactly, so `S ← 1·S + outer(0, k) = S` **bit-exactly**.
- Padded outputs are not written.
- Padding is by predicated loads, with no workspace copy.
- `phase = 0` is the current behaviour, bit for bit.

Prefill chunks remain `--prefill-chunk` tokens long. The **first** chunk after a non-aligned base
is `chunk − phase` tokens long, so every later chunk starts page-aligned. This is one boundary
change in `TextContext::prefill`. In Hybrid mode no `prefill_split_frontier_` or
`rewrite_execution_frontiers` split is requested (both remain for Legacy).

### 7.3 Tap Op contracts

For each prefill unit with taps `T_u = T ∩ (unit_begin, unit_end]`, `|T_u| ≤ taps_per_unit`
(default 4, bounded by free snapshot slots):

1. **GDN state tap**: `gated_delta_net(..., StateTapSet taps)`. For each tap `j`, the kernel writes
   the FP32 state after absolute position `p_j` for every value head into
   `snapshot_slot[j].ssm_state[layer]`.
   - Stores use `st.global.cs` (streaming) to avoid displacing prefill working sets in L2.
   - Oracle: the FP64 naive recurrence evaluated to `p_j` (op-development §numerical).
   - Exactness criterion: the tapped state is **bit-identical** to the final state of a separate
     call on the same inputs ending at `p_j` with the same phase. Same chunk grid, same
     arithmetic.
2. **Conv tap**: per GDN layer, copy the `kernel_width − 1 = 3` pre-conv projected columns
   `[p_j − 3, p_j)`, taken from the current unit's projection output or the previous conv history
   when `p_j − unit_begin < 3`, into the slot's conv history. Exact copy, exact oracle.
3. **Hidden tap**: the continuation hidden at column `p_j − 1`, stored for completeness. It is
   never used for a full hit (§6.1).
4. **DFlash/DFlash2 ring tap**: after the unit's context K/V materialisation, and *before* the
   ring is overwritten, assemble `slot.ring = ring(positions [p_j − S, p_j))`. Each position comes
   from the pre-unit ring if `< unit_begin`, otherwise from the unit's freshly materialised K/V.
   The materialiser writes this unit's context K/V to a unit staging buffer first; the ring commit
   and the tap gathers both read from it. Exact copy.
   - The DFlash Full (paged) pool is committed through the chunk at unit commit, so its pages
     belong to the tree like Main pages.
5. **MTP**: MTP KV is paged. Tap snapshots carry no MTP state. A resume performs the Program's
   existing MTP bridge from the resumed frontier, as for any Program-resumed frontier today.

If no snapshot slot is free when the unit starts, the tap is skipped and counted
(`taps_skipped_no_slot`). In practice D2H of a slot (≈4 ms) finishes long before the next unit
(≥0.3 s).

### 7.4 Tap publication

At the chunk boundary the tap takes a staging device snapshot slot, copies the lane's committed
StateImage into it (including the continuation hidden of column `p − 1`, which prefill keeps while
taps remain) and waits as a **pending tap** of the lane. It is published as soon as the blocks it
anchors on are committed tree nodes:

- the anchor `p/64 − 1`, which for an MTP backend (one token behind the text frontier) commits
  only after the next chunk even when `p` is page-aligned;
- for `p % 64 ≠ 0`, block `p/64` as well, whose first `p % 64` columns are copied into a
  cache-owned tail page, so every cache entry keeps an exclusive page (§6.3).

Publication creates a `Snapshot{Tap}` with `device_slot` = the staging slot and enqueues D2H
write-through of the image (and tail) into host slabs; `live_snapshots_below++` along the path.
Once write-through completes, the slot becomes an **evictable device snapshot** (§9.2). A finishing
lane publishes its remaining pending taps from its own last page, or drops them.

### 7.5 Recomputed region

When `F_s < 64k`, blocks `[F_s/64 .. k)` already exist in the tree but are recomputed, because the
recurrent state must be rebuilt. The sequence writes them into its own private pages. At block
commit (§7.6), each such block finds the existing node, so the private page stays private
(`Duplicate`) and is freed at finish. The tree is not modified. KV-write elision, which maps the
existing pages and discards those writes, is an optional optimisation (§12.4).

### 7.6 Block commit (prefill and decode)

After any commit that advances every enabled pool's committed frontier across a block end
`64(b+1)`:

- If `child(N_{b−1}, key(b))` exists with a Device copy: the sequence's page stays private
  (`Duplicate`). If it exists host-only, it adopts the sequence's page as its Device copy. Otherwise
  **insert** a node owning the sequence's page. The node stays pinned for this sequence.
- D2H write-through of the lane's Device-only nodes is enqueued in one batch when the lane releases
  its path (finish, abort or cancel), before the nodes become evictable. Batching coalesces runs of
  consecutive pages and slabs into one strided copy per plane, and pinned nodes need no backup yet.

The block-table row is unchanged, so no GPU work is needed.

Only committed tokens count. Speculative provisional lead, rejected columns and MTP/DFlash lag are
excluded by requiring every enabled pool's *committed* frontier (paged-kv §9.1).

Decode blocks become cacheable the moment they fill. A later request that shares a long generated
output, such as an agent re-sending the assistant turn, can resume from the endpoint snapshot
(§7.7).

### 7.7 Finish and cancel

At terminal settlement (Finish, or Cancel after at least one committed unit):

1. If the request is cache-enabled, every pool has committed the frontier (the backend up to its
   own frontier, `F_end − 1` for MTP) and `F_end − (deepest snapshot on path) ≥ 64`:
   - The active StateImage itself becomes the snapshot image (frozen, no copy) in a device snapshot
     slot; if none is free, one is evicted (§9.2), otherwise the endpoint is skipped.
   - The partial last page (`F_end % 64 ≠ 0`) moves to the snapshot as its tail. There is no COW,
     because the writer is terminating. The backend page keeps its own committed extent.
   - Publish `Snapshot{Endpoint}` and enqueue D2H write-through of the image and tail.
2. For every pinned node: `active_refs--`. A node that reaches 0 goes to the device LRU tail
   (§9.1), inserted in **reverse block order**, so deeper blocks are evicted before shallower
   ones.
3. Free `Duplicate` and growth pages. Release the active state slot.

`Disabled` requests (Serve warmup, `--no-prefix-reuse`) never insert nodes or snapshots.
Their pages are freed at terminal.

---

## 8. Engine and API

### 8.1 Program API (replaces the pressure, transaction and capture surface)

```cpp
struct AdmissionQuote {                     // pure; valid until the next cache mutation
    uint32_t reuse_tokens;                  // F_s, reported as cached_tokens
    PrefixSource source;                    // Root | DeviceSnapshot | HostSnapshot
    uint32_t need_pages, available_pages;
    uint64_t restore_bytes;
    double   predicted_restore_s, predicted_prefill_s;
    AdmissionReadiness readiness;           // Ready | NeedsTransfer | TemporarilyBlocked | PermanentlyInfeasible
    uint64_t cache_epoch;                   // quote is stale if the epoch moved
};
[[nodiscard]] AdmissionQuote quote_admission(const PreparedPrompt&, const RequestBasePlan&) const;
[[nodiscard]] AdmitResult    admit(const AdmissionQuote&, runtime::LaneId, PreparedPrompt&&,
                                   runtime::CancellationFlagView);        // §6.4
[[nodiscard]] bool           restore_complete(SequenceHandle) const;      // §6.5
void                         poll_transfers();                            // every worker boundary
[[nodiscard]] std::optional<PersistentBackfillProof>
                             prove_persistent_backfill(...) const;        // §6.3 arithmetic
// Unchanged: advance_prefill, decode, append_forced_tokens, commit, abort_pending, finish, abort,
// device_kv_lease_* (the lease reclaim path now simply evicts from the device LRU).
[[nodiscard]] PrefixCacheStats prefix_cache_stats() const;
```

Not called in Hybrid mode (retained for Legacy): `inspect_admission`, `seal_identity`,
`begin_pressure_planning`, `shared_capture_split_prefill_work`, `start_resource_transaction`,
`progress_context_transaction`, `finalize_context_transaction`, `inspect_capture`,
`checkpoint_recovery_work`, `begin_capture_pressure_planning`, `shared_capture_matches`,
`skip_capture`, `reserve_active_capture*`, `retained_device_kv_pages`, `release_continuation`,
`release_shared_prefix`, `resource_revision`. The Hybrid methods throw `std::logic_error` in
Legacy mode and vice versa.

### 8.2 Engine

The Engine selects one of two admission paths at construction from `ContextCacheOptions::mode`.
Legacy keeps `ResourceManager` and the planners. Hybrid does not construct them:

- `try_admit_one()`: `quote → (Ready|NeedsTransfer) → admit`. With `NeedsTransfer`, the lane is
  occupied in `Restoring` and the scheduler treats it like a staged-prefill lane that is not yet
  runnable.
- `ensure_base_plan` builds no candidates.
- No `CaptureOffer` is produced. Taps are internal to prefill.
- `apply_device_kv_lease_settlements` evicts from the device LRU instead of releasing owners.
- `PrefixCacheStats` feeds the existing `record_prefix_selection` and request-log fields.

### 8.3 Frontend and protocol mapping

- `PreparedPrompt` gains `block_hashes` and `tap_hints` (sorted `{position, priority}`), filled in
  both modes (cheap). Hybrid mode ignores `PreparedContextCache` session key, retention and
  opportunities, and `PromptIdentity` rewrite fields; Legacy ignores the new fields.
- `ContextCacheHints` is unchanged. In Hybrid mode:
  - explicit-evidence markers map to priority-1 tap hints, automatic protocol markers to
    structural hints;
  - `session_key`, `retention`, `allow_engine_automatic_shared_prefixes` and
    `update_session_index` are ignored. Content addressing finds session chains without them.
- External protocol behaviour is identical in both modes:
  - OpenAI `prompt_cache_*` and breakpoints, and Anthropic `cache_control` validation, limits
    (four distinct breakpoints) and error cases are unchanged.
  - In Hybrid mode, breakpoints mean "tap here" (priority 1). Protocol write-policy modes
    (`mode:"explicit"`, Anthropic `cache_control`-only requests) do not suppress automatic taps:
    taps are cheap and content-deduplicated, so markers can only add reuse.
  - `cached_tokens` and `cache_read_input_tokens` report `reuse_tokens`. The streaming
    `message_start` timing is unchanged, because the quote fixes the value at admission.

---

## 9. Eviction

### 9.1 Device KV (LRU, tail-first)

- One intrusive list of `Cached` device pages whose owner has no refs and no pins: nodes with
  `active_refs == 0 && restore_pins == 0`, and snapshot tails with `restore_pins == 0`.
- On a hit, touched nodes move to the MRU end.
- On release, nodes are appended in reverse block order.
- Allocation pops from the LRU head:
  - **Host-backed** entries (host `Resident`): drop the device copy. Free.
  - **Unbacked** entries, which occur only when the host is disabled, full, or write-through is
    still filling: the node loses its last copy, so apply §9.4.

Two lists are kept, *backed* and *unbacked*, and backed entries are always consumed first.

### 9.2 Device snapshot slots

Slots hold at most `D` images. A slot is reusable when its snapshot is host-backed; the victim is
the least recently hit such slot. A slot whose snapshot is not backed is evicted only when no
backed slot exists and a tap or endpoint needs it. The snapshot then loses its image and is
deleted, unless it is host-backed.

### 9.3 Host (GDSF over snapshots, dead-KV first)

Allocation of `k` slabs:

1. **Dead KV sweep**: pop host-resident nodes with `live_snapshots_below == 0`,
   `active_refs == 0` and no pins, leaves first. These can never produce a hit.
2. Otherwise, evict the snapshot with minimum `H`, then repeat step 1 (its exclusive path has just
   become dead). Update `L := H(victim)`. A victim still complete on the Device (image in a device
   slot and tail resident) only gives up its host copy and stays restorable.

```text
H(s)  = L + F(s) · C(s) / Z(s)
C(s)  = prefill_seconds(base=F_a, tokens=F_s−F_a) − restore_seconds(s)   // clamp ≥ 0
        where a = nearest valid ancestor snapshot on path(s) (or root, F_a = 0)
Z(s)  = image_bytes + tail_bytes + exclusive_path_bytes(s)
        exclusive_path_bytes: host bytes of nodes on path(s) whose live_snapshots_below == 1
F(s)  = 1 + hits(s)                          // hits since publication
H is recomputed when s is published or hit, and when a scan needs a fresh C/Z
```

- `C` uses the same calibrated model as §6.2, so eviction and admission agree about value.
- Aging comes from `L`, so snapshots that are no longer hit fall behind new ones without timers.
- The scan is O(snapshots × path depth) in the worst case: about 300 snapshots × 3,750 blocks
  gives about 1 M simple steps (≈1 ms). The typical case is microseconds, because `C` and `Z` are
  cached per snapshot and invalidated only along a changed path. Snapshot count is bounded by
  `host_cache_bytes / image_bytes`.
- Pinned objects (`restore_pins`, `active_refs`, in-flight fills) are never evicted.

### 9.4 Losing the last copy

A node whose device and host copies are both gone is deleted together with its subtree. Every
snapshot anchored in the subtree is deleted, and `live_snapshots_below` is decremented along the
paths. A node with `active_refs > 0` always has a device copy, so this never affects an active
sequence. The subtree walk runs only in the unbacked cases of §9.1 and §9.2, and is bounded by
the subtree size.

### 9.5 Active guarantee

- Active entitlement is unchanged: the bounded output window, extension at round boundaries, and
  bounded completion when the pool cannot extend (paged-kv §6.2).
- Cached pages never count against it: `available = free + evictable`, so active growth always
  wins.
- Nothing active is ever evicted.

---

## 10. Invariants

1. Each device page group and each host slab has exactly one owner state:
   Free/Active/Cached/Filling, or Free/Node/SnapshotImage/SnapshotTail/Filling.
2. A tree node is immutable: its tokens, extra key and payload never change after insertion.
3. Every node on an active sequence's mapped path has `active_refs > 0` and a device-resident
   copy until the sequence releases it.
4. A snapshot exists only while it is valid (§5.3). Match returns only valid snapshots, and only
   with `F ≤ n − 1`.
5. Exact identity: a match implies exact equality of tokens and extra keys for every block and
   tail token. Hashes never decide equality.
6. A Filling destination is never mapped into a block table, and its readers wait on its event.
   A Filling source is never freed.
7. Taps are bit-identical to an unsplit run ending at the tap, for the same phase.
   Identity-padding for phase alignment is bit-exact.
8. `Disabled` requests leave no nodes or snapshots behind.
9. The worker is the only mutator. Transfers change residency only in `poll_transfers()` or at
   commit.
10. Cost models only rank choices. Capacity and validity checks never depend on predictions.

---

## 11. RTX 5090 / sm_120a optimisation requirements

### 11.1 Streams and priorities

- `h2d_stream` is created with the highest priority. Restores are latency critical.
- `d2h_stream` is created with the lowest priority. Write-through is latency tolerant.
- Use full-duplex transfer when `asyncEngineCount ≥ 2`. Otherwise, write-through jobs pause while
  a restore is queued.

### 11.2 KV page transfer Op (`paged_kv_transfer`)

Device plane order is unchanged: one page is `planes` contiguous slices, 64 planes for 27B INT8
plus MTP. Copying a 100 k-token restore plane by plane is about 100 k small copies, so transfers go
through one of two paths. P2 selects one by measurement:

- **A: SM copy kernel.** Reads or writes pinned host slabs through their mapped device pointers.
  One CTA handles one (block, plane-group) work item.
  - 16 B vector accesses; `ld.global.nc` for H2D sources, `st.global.cs` for destinations.
  - The grid is capped at `transfer_ctas` (default 16 for H2D restore, 4 for D2H write-through),
    so decode of other lanes keeps its SMs.
  - Work items are ordered layer-major, so the first layers finish first.
- **B: `cudaMemcpyBatchAsync`** (CUDA ≥ 12.8; toolchain is 13.1). Copies are coalesced into runs
  where both device page IDs and slab offsets are consecutive. This uses the copy engines and no
  SMs.

Acceptance: the selected path reaches ≥ 85 % of the measured contiguous `cudaMemcpyAsync` peak
for 2 MiB copies in each direction, on all five KV profiles, with random page permutations. If
both paths qualify, B is preferred for D2H (no SM use) and the faster one is used for H2D.

The host slab layout is the device page's plane slices concatenated in plane order, per pool. The
bytes are identical to device, so transfer is a pure copy with no requantisation. Restore is
bit-exact for every KV profile.

### 11.3 Other rules

- Snapshot image copies use one batched D2D kernel or memcpy set for the ~110 image pieces
  (48 SSM + 48 conv + hidden + ring). Target ≥ 1.4 TB/s effective D2D.
- Tap stores use streaming (`.cs`) hints. Taps add no launches: they are extra outputs of existing
  kernels, plus one small ring/conv gather per unit.
- CUDA Graph keys, decode graphs and block-table publication are unchanged. Prefill (not graphed)
  only receives extra tap pointers.
- All cache bookkeeping is host-side O(1) or O(path). No allocation happens on the worker hot
  path: arenas are sized at startup from `M`, `D` and `host_cache_bytes`.

### 11.4 Startup calibration

Record in the `server_start` memory/transfer ledger:

- H2D/D2H bandwidth over 64 × 2 MiB copies;
- D2D bandwidth;
- per-batch latency.

These feed §6.2 and §9.3. The existing prefill coefficients stay as they are.

---

## 12. Optional features (each gated by its own measurement)

1. **Layer-pipelined restore** (implemented, §6.5). The first prefill pass's layer `l` waits on
   the restore's event for layer `l` (`cudaStreamWaitEvent`), so the restore overlaps compute; the
   admission itself never waits for the copies.
   - Gate: TTFT reduction ≥ 30 ms p50 on host-restore requests of ≥ 50 k tokens.
2. **In-flight prefix coalescing** (implemented, `hybrid_await_sibling`). An admissible FIFO head
   whose prompt shares more tokens with a lane still prefilling than the cache can serve now
   reports `TemporarilyBlocked`, holding no pins, until that lane's snapshot at the divergence
   publishes; it is then admitted from it.
   - Target: the block boundary `⌊d/64⌋·64` below the divergence point `d` (at most `n-1`),
     added to the sibling's remaining plan as an exact tap. A frontier inside a block would
     publish only once the sibling completes that block. The sibling's deepest planned exact tap
     at or below `d` serves instead when it is at least that deep, or when prefilling the tokens
     between them costs no more than one chunk's fixed cost.
   - A snapshot the sibling has captured in the shared prefix but not yet published (its block or
     MTP backend page is incomplete) is also waited for. It publishes within the sibling's next
     chunk.
   - It waits only when the prefill it saves exceeds the split it may add, and when twice the
     sibling's predicted prefill up to the target (other lanes' decode rounds interleave) is within
     half the Engine's queue timeout (`pending_timeout_ms`). A waiting head stays in the FIFO.
   - Vision prompts do not coalesce: placeholder tokens are equal for different media.
   - Every sibling prefill boundary re-arms admission, and the head stops waiting once the sibling
     passes the target, finishes or is cancelled. So a missing snapshot never blocks it for long.
   - Gate: total prefill tokens reduced on the concurrent-shared-prefix trace, with no TTFT
     regression for other requests.
3. **Output-boundary snapshots.** The Frontend's output parser signals structural boundaries
   (after `</think>`, before `<tool_call>`), and the Program snapshots the state at that exact
   committed frontier.
   - Without speculation: D2D of the active slot at the round boundary.
   - With speculation: a two-segment ReplaySSM Fold, `S0 → slot` over the first `j` records, then
     `slot → active` over the rest. This is bit-identical by the Fold contract. Conv and ring taps
     are handled as in §7.3.
   - Gate: reuse gain on thinking-stripped agent traces.
4. **KV-write elision** for the recomputed region (§7.5): map existing pages and send those KV
   writes to a scratch page.
   - Gate: measurable page savings under pressure.

---

## 13. Test specification

### 13.1 Host-only unit tests (`tests/runtime/prefix_cache/`)

- **BlockTree**
  - insert/match with exact tokens and extra keys;
  - a forced constant `lookup_hash` must not produce a false hit;
  - different media digest with the same tokens must miss;
  - tail snapshot matches on exact tail tokens only;
  - `F ≤ n − 1` is enforced;
  - duplicate snapshot publication is a no-op.
- **Liveness**
  - `live_snapshots_below` stays correct under interleaved publish, evict and subtree deletion;
  - losing the last copy deletes exactly the subtree's snapshots.
- **Device LRU**
  - tail-first order after release;
  - hits move the whole path to MRU;
  - backed entries are consumed before unbacked;
  - pinned entries are never returned.
- **Host GDSF**
  - hand-computed cases: two siblings, nested ancestor/descendant, frequency;
  - dead KV is reclaimed before any snapshot;
  - the victim's exclusive path is freed;
  - `L` is monotonic.
- **HostSlabPool**
  - exhaustion, reuse and chunked allocation;
  - a slab never crosses chunks.
- **TapPlanner** (property tests over random prompts and boundaries)
  - all taps 64-aligned, in `(F_s, n−1]`, outside Vision spans;
  - deduplicated against the path;
  - priority order and budget respected;
  - the ladder stays within about log2(n/G) + 1 taps;
  - deterministic.
- **Admission arithmetic**
  - candidate fallback order;
  - Blocked vs Infeasible classification;
  - backfill proof on constructed states;
  - quote staleness via `cache_epoch`.

### 13.2 Op qualification (op-development.md contract, real 27B and 35B-A3B shapes)

- **GDN tap and phase**
  - FP64 naive-recurrence oracle at every tap, phases 0..63, T in {1, 63, 64, 65, 2048, 4096};
  - **bitwise** equality of each tap with a truncated call on the same phase;
  - `phase > 0` with zero taps matches the unpadded oracle within the existing GDN criterion;
  - `phase = 0` is bit-identical to the pre-change kernel.
- **Conv, hidden and ring taps**: exact byte comparison against a host reference gather, including
  `p − unit_begin < 3` and ring wrap (`p > S`, `unit length > S`).
- **`paged_kv_transfer`**
  - exact bytes for BF16, INT8, FP8, NVFP4 and K8V4, with Main+MTP and Main+DraftFull bundles,
    random page IDs, both directions;
  - bandwidth ≥ 85 % of the contiguous peak;
  - a concurrent-decode interference measurement is recorded.

### 13.3 Real-artifact integration (`ninfer_qwen3_5_prefix_real_test` scenarios, replacing the current set)

Each scenario runs for `--kv-dtype` in {bf16, int8, fp8, nvfp4, k8v4} × backend in {none, mtp,
dflash2}. Scenarios marked "full matrix" run for every combination; the rest use int8 × {none,
dflash2}.

| scenario | assertion |
|---|---|
| resume-equivalence (full matrix) | resumed vs root: first-token logits within tolerance; greedy top-1 agreement over 256 tokens ≥ the chunk-decomposition baseline (root with a different `--prefill-chunk`); CausalScoring NLL difference within the same baseline |
| multi-turn exact endpoint | turn k+1 reuses `F_end(k)` exactly, and the tail was COW'd |
| re-rendered history (thinking stripped) | reuse ≥ `floor64(n_k − 1)` |
| edit deep message | reuse = the deepest ladder tap below the edit |
| shared preamble across 8 sessions | device pages for the preamble counted once; all 8 hit |
| concurrency 8, identical prompts arriving together | correct outputs; no double restore of a Filling node |
| host-only restore (D snapshot slots evicted) | same outputs as device resume; `Restoring` observed |
| cancellation mid-prefill | taps and completed blocks published; next identical request reuses them |
| Vision: same image vs different image, identical tokens | hit vs miss; no resume inside a span |
| pressure with active growth | active request extends by evicting cache; bounded completion unchanged |
| host cache 0 | device-only cache works; unbacked eviction deletes subtrees correctly |
| `Disabled` request | tree and snapshot counts unchanged |
| protocol usage | `cached_tokens` / `cache_read_input_tokens` equal `reuse_tokens`; streaming `message_start` exact |

Fault injection covers:

- a failed host allocation at startup, which is a clear startup error;
- a forced CUDA error in a transfer job, which gives Engine-wide failure and cleanup with no leaked
  pins (checked by `prefix_cache_stats` after restart);
- `compute-sanitizer --tool memcheck` on one concurrency-8 scenario, for the lifetime claims.

### 13.4 Performance acceptance (RTX 5090, CUDA 13.1, `out/qwen3_6_27b.ninfer` and the README's Qwen3.8-27B NVFP4 + DFlash2 deployment)

The baseline is the current master, run on identical traces and flags, with `--host-cache-mib`
equal in both runs.

| metric | requirement |
|---|---|
| prompt tokens served from cache, agent trace (README workload) | ≥ baseline, target ≥ +5 points |
| admission planning time (quote + admit, host) | p99 < 1 ms (baseline: up to 250 ms allowance) |
| TTFT p50 / p95 on the agent trace | lower than baseline; report the breakdown (planning, restore, prefill) |
| cold-request prefill tok/s | within ±1 % |
| decode tok/s at c = 1, 2, 8 (with and without write-through active) | within ±1 % |
| tap overhead per 4 k-token chunk | < 0.5 % of chunk time |
| restore bandwidth | ≥ 85 % of measured PCIe peak |
| device memory | shared-preamble trace: ≥ 7× fewer preamble pages at c = 8 |

Record the hardware, driver, command and summarised results in `docs/performance.md`, following
bench/README conventions.

---

## 14. Implementation plan

Each phase builds, passes its tests and is committed separately (Conventional Commits).

| phase | content | exit criteria |
|---|---|---|
| P0 | Baseline: run the agent trace and microbenchmarks on current master; add planning-time and TTFT-breakdown fields to the request log if missing | baseline recorded |
| P1 | `src/core/host_slab_pool.*`; `src/runtime/prefix_cache/{block_tree,snapshot_index,device_lru,host_gdsf,tap_planner,cost}.{h,cpp}`; Frontend `block_hashes` and `tap_hints` | §13.1 green |
| P2 | Ops: GDN phase+tap, conv, hidden and ring taps, `paged_kv_transfer` (A and B), startup calibration | §13.2 green; transfer path selected |
| P3 | Program integration in `src/models/qwen3_5/program/prefix/`: admit/restore/poll, prefill taps and unit publication, block commit, finish/cancel, eviction hooks, lease settlement; Engine dispatch on `ContextCacheMode`; `--use-alt-prefix-caching` and Hybrid options/CLI/serving mapping. Legacy code is untouched | builds; §13.3 green in Hybrid mode; Legacy tests unchanged and green |
| P4 | Performance acceptance (§13.4, Hybrid vs Legacy on identical traces); tune defaults (`D`, `max_new_taps`, `tap_ladder_tokens`, `tap_min_gap`, transfer CTAs) | §13.4 met |
| P5 | Documentation: link this document from docs/README.md, engine-architecture.md and resource-scheduling-and-context-cache.md as the Hybrid-mode authority; document the flag in serving.md, cli.md and README | links checked, `git diff --check` |
| P6 (optional) | §12 features, each with its own gate | per gate |

### 14.1 Code that Hybrid mode bypasses (retained for Legacy mode)

This list records what Hybrid mode does not use. It is deleted only if Legacy mode is later
retired by an explicit decision.

- `src/runtime/engine/context_cache/`: `resource_manager.h`, `materialization_planner.h`,
  `shared_capture_planner.h`, `resource_search.h`, `context_portfolio_value.h`,
  `materialization_budget.h`, `seal_window_claim.h`. `context_cost.*` is reduced to the machine
  model.
- `src/models/qwen3_5/program/planning/`: `pressure.cpp`, `pressure_planner.{h,cpp}`,
  `checkpoint_recovery.cpp`, `resource_projection.h`, `rebuild_work.h`. `request_plan.cpp` is
  reduced to base planning.
- `src/models/qwen3_5/program/transactions/`: `materialization.cpp`, `capture.cpp`, `commit.cpp`.
- `src/models/qwen3_5/program/storage/`: `kv_store.h` (logical pages, address spaces) and
  `host_kv_store.h` are replaced. `state_store.h` is reduced to active and snapshot slots.
  `context.cpp` is rewritten.
- `src/core/host_kv_arena.*` and `src/models/qwen3_5/program/prefix_identity.*`.
- `src/runtime/contract/resources.h`: the pressure, target, candidate, checkpoint-ref and
  revision types.
- Options: Legacy-only `device_state_slots`, `host_state_slots`, `host_kv_capacity_bytes`,
  `max_private_continuations`, `max_shared_prefixes`, `max_long_anchors_per_continuation`,
  `long_anchor_min_spacing_tokens` and their CLI flags are rejected in Hybrid mode.
- Tests asserting owner, catalog or pressure-target behaviour keep running in Legacy mode.

### 14.2 Configuration surface

```cpp
enum class ContextCacheMode : std::uint8_t { Legacy, Hybrid };

struct HybridPrefixCacheOptions {                           // used only when mode == Hybrid
    std::optional<std::uint32_t> device_snapshot_slots;     // --device-snapshot-slots
    std::optional<std::uint32_t> max_new_taps;              // --cache-taps-per-request
    std::optional<std::uint32_t> tap_ladder_tokens;         // --cache-tap-ladder (G)
    std::optional<std::uint32_t> tap_min_gap_tokens;        // --cache-tap-min-gap
    std::filesystem::path persistent_file;                  // --prefix-cache-file (§5.5)
    std::string persistent_identity;                        // set by the product binary
};

struct ContextCacheOptions {                  // existing struct, extended
    ContextCacheMode mode = ContextCacheMode::Legacy; // --use-alt-prefix-caching selects Hybrid
    HybridPrefixCacheOptions hybrid;
    // Hybrid mode: host_cache_budget_bytes (--host-cache-mib) sizes the one slab pool.
    // ... existing Legacy fields unchanged ...
};
```

Engine construction (`normalize_engine_options`) resolves every unset Hybrid value, and
`Engine::options()` reports the effective ones. With `C = max_concurrency`, `chunk =
prefill_chunk` and a Host tier present when the budget is nonzero:

| value | default | reason |
|---|---|---|
| `host_cache_budget_bytes` | 8 GiB (`kDefaultHybridHostCacheBytes`); 0 disables the Host tier | the Legacy Host KV default |
| `device_snapshot_slots` | `C + 1` with a Host tier, `C + 2` without | one resident snapshot per live conversation, plus staging; without Host the slots are the only snapshot storage |
| `max_new_taps` | 8 with a Host tier, 2 without | without Host, taps would evict other conversations' snapshots |
| `tap_ladder_tokens` | `max(4096, 2·chunk)` | ladder taps land on chunk boundaries; a finer ladder only duplicates them |
| `tap_min_gap_tokens` | `max(1024, chunk)` | the same |
| KV capacity (`ninfer-serve`) | `--kv-capacity auto` unless given | free VRAM becomes Device block cache |

Legacy capacity options (`device_state_slots`, catalogs, anchors) are rejected, and the Legacy
Host pools (`host_state_slots`, `host_kv_capacity_bytes`) are zero in Hybrid mode.

---

## 15. Risks and mitigations

| risk | mitigation |
|---|---|
| Exact taps cost one prefill split each | Only semantic boundaries are exact (explicit, generation opener, structural); the prompt tail and ladder are flexible and cost no split. The proximity rule leaves one split per chat request. The zero-split GDN tap (§7.2–§7.3) removes the rest if measurement shows it matters. |
| Snapshots crowd KV out of host memory | GDSF weighs bytes against recompute seconds on the same scale for both. Dead-KV sweep runs first. |
| Write-through competes with decode | Low-priority stream on copy engines (path B) or ≤ 4 CTAs (path A). Measured in §13.4. Write-through can be throttled when decode is active without changing correctness. |
| Unbacked device eviction deletes subtrees | Only happens with no or full host cache. Backed entries are always preferred. |
| Large pinned allocations on Windows | Chunked allocation; the startup ledger reports the resolved size. |
| DFlash ring tap complexity | Isolated in one gather Op with an exact oracle. Pending-feature catch-up rules are unchanged. |
| Hybrid mode ignores session keys | Content addressing reproduces chain reuse; the protocol surface is unchanged in both modes. |
