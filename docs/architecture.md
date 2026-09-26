# Architecture (P0)

KVMem is a block-sparse KV working-set manager. llama.cpp will own inference;
this library owns selection, tiering, and (later) window assembly.

```
kvmem/          host policy + CPU/NVMe  (this repo, no llama.cpp headers)
src/adapter/    llama_memory_i wrapper  (P1+)
llama.cpp/      submodule + thin patches (P1+)
```

GPU attention cache is a **bounded block-slot pool** of size
`budget + gen_reserve`. Product default GPU KV type is llama.cpp **q8_0**
for K and V (`--kv-dtype q8_0`; `f16` or `q4_0` to override). Cell `pos` is
the original monotonic token position (slot index is not a RoPE coordinate).
Restore is packed GPU-format memcpy at that orig pos — no unrotated raw-K
and no re-RoPE on the product path. Retrieval scores **mean-K** (F32,
pre-RoPE, captured at first write). Packed K/V for a full block are
copied to host asynchronously when the block fills, overlapping later
prefill; eviction then skips if the copy exists. After retrieval pin,
decode keeps a GPU running sum of pre-RoPE K and writes mean-K when a
block fills (accepted tokens only; MTP drafts are not counted). Cold
stage-in is `copy_k_gpu` / `copy_v_gpu` + slab H2D. `llama-kvmem-server`
reuses the GPU prefix across requests (token LCP); the retrieval query
is the last `role=user` span. Chat tools reuse llama.cpp `common/chat`
+ `common_sampler`; the server does not execute tools. Packed transfers use a **32 MiB**
GPU slab. GPU-format CPU/NVMe scratch is allocated only when those tiers
are on. Each logical block occupies one slot of `block_tokens` cells.
Reselect is a `KvMemPlan` diff: resident selected blocks stay in their
slot; only `stage_out` cells are `seq_rm`'d. Flash Attention is not
modified. P1 recency does not re-RoPE and does not resurrect dropped
blocks.

The host store is per conversation; the GPU working set and the `llama_context`
stay single. `--kvmem-conversations N` keeps N host stores alive and
time-multiplexes them, so a conversation that returns after another was served
does not have to be reprocessed. A switch drains the whole working set to host
and rebuilds the incoming store's through the drain-and-restage path that
already runs inside a conversation (the host fallback in
`layout_gpu_slots_by_orig_pos` and `write_block_to_gpu` in
`src/adapter/llama-memory-kvmem.cpp`), not a second implementation. Requests
stay serialized and `n_seq_max` stays 1; the recurrent half is still a
server-side byte snapshot restored per request.

Hardware split on this machine: RTX 5050 (GPU 0) for models < 27B;
RTX 5090 (GPU 1) for 27B. Details in `scripts/gpu.sh` and
`docs/modification-plan.md`.

## Known v1 limit: generation length vs `gen_reserve`

GPU pool = `budget` (selected working set) + `gen_reserve` (decode slack).
After retrieval the selected blocks are **pinned**: decode must not
recency-reselect and drop resurrected blocks. New tokens only take free
slots in `gen_reserve`.

If the last GPU block is full and there is no free slot left,
`prepare_working_set` fails with `no free GPU slot for block N`
(`llama_decode(gen) failed rc=1`). It does **not** evict pinned
retrieval blocks. One generation therefore cannot exceed
`--kvmem-gen-reserve` (16384 on IQ3, 12288 on IQ4, CLI default 256),
including thinking. README documents this and a system-prompt cap.
We are working on the follow-up below.

### Why not steal slots from the selected set

The pool is already two partitions. Selected KV occupies `budget` slots
and stays pinned. Generation occupies `gen_reserve` slots. When those
are full, the card is not “choose generation or retrieval”: only the
reserve partition is full. Unpinning or recency-evicting selected
blocks would drop the query’s retrieved facts. Growing `budget +
gen_reserve` on 16 GiB is also out: recipes already sit near 15.5 GiB.
Streaming the whole generation through VRAM would bring back the
adaptive-KV-streaming cost curve. NVMe is not implemented in this port;
host RAM is enough for spilled gen KV.

### Follow-up: ring buffer **inside** `gen_reserve`

No third pool and no extra VRAM. `gen_reserve` becomes “how much of
**this turn’s** output attention can still see”, not a hard max length.

On pin, record `gen_start_pos`. When `alloc_slot()` fails:

1. Evict only the **oldest completed generation block**
   (`orig_pos_start >= gen_start_pos`, block full, `gpu_slot >= 0`).
2. Never evict the in-progress block, and never evict selected /
   pinned blocks (those slots stay out of the free list).
3. The block is already harvested to the host store (decode mean-K /
   packed copy). Stage it out, `free_slot`, retry `alloc_slot`.
4. Spill **one** block per event. Grain is `--kvmem-block-tokens`
   (128 on the 16 GiB recipes). One q8 block is a few MiB; PCIe is
   cheap next to 128 decode steps. Larger batches (1K/2K) save almost
   no bandwidth and yank a long stretch of self-output off GPU at
   once. Smaller than `block_tokens` would change global slot
   geometry; do not do that for the ring.

After a spill, GPU still holds: the full selected window + the most
recent ~`gen_reserve` tokens of this generation. Keep at least the
in-progress block (do not attend to only the current 128 tokens by
dumping every completed gen block). MTP’s follower pool uses the same
slot indices; free the same slot on the draft cache. GDN / recurrent
state is updated every token and does not live in these attention
slots.

Spilled gen KV stays in the host store. The next turn’s retrieval can
select it. This turn’s Flash Attention does not see it.

**Tail size.** Keep the current recipe reserves: 16K (IQ3) / 12K
(IQ4). That VRAM is already paid. It covers the default 4096 thinking
budget plus a normal answer without spinning the ring. 4K would cover
thinking but drop long code; 32K would steal from `budget` or blow
16 GiB. After the ring exists, do not add slots.

**Quality.** The hard crash goes away. A 32K thinking dump still only
attends to the last 12K/16K of itself. The system-prompt cap remains
useful. Seeing earlier thoughts in the same turn would be a later
step (retrieve from spilled gen blocks using the current generation
as query), not part of this fix.

**First implementation cut.** Pin records `gen_start_pos`;
`alloc_slot` failure evicts one full gen block; generate longer than
`gen_reserve` without dropping a retrieved needle. Recency decode’s
`block_count() > budget` mis-trigger is the same class of pin bug
and stays out of this cut.

## P7: MTP shares the slot-pool (plan B)

Logically long context must not grow a full-length MTP KV (plan A). The
draft context (`LLAMA_CONTEXT_TYPE_MTP`) gets a **follower** slot-pool
the same size as the target attention cache, same block IDs and slot
indices, original `pos` on cells. Speculative decoding stays in
llama.cpp `draft-mtp`. Details: `docs/kvmem-mtp-plan.md`.
