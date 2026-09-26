# Host-memory RSS growth: diagnosis and mitigation

## Symptom

During the 33-round, 256K-context multimodal canary
(`scripts/multimodal_canary.py --long-context-benchmark`), the server
process RSS grows from ~13 GiB to **47.8+ GiB** and never comes back down.
On hosts with less RAM this ends in OOM.

## Root cause (two cooperating factors)

This is **not** a leak in the leak-checker sense — the allocations are freed.
It is high-frequency allocation interacting with glibc retention/fragmentation:

1. **A ~10 MiB scratch buffer is allocated and discarded every decode step.**
   gdb backtrace of the hot site:
   `llama_memory_kvmem_mtp::init_batch → llama_batch_allocr::ubatch_add` —
   a `std::vector<float>` of 512 ubatch × 5120 embedding × 4 bytes, fresh
   per step, hundreds of thousands of times per long session.
2. **glibc's per-thread arenas pin the heap top.** glibc creates up to
   8 arenas per core; freed blocks can only be returned to the OS at the
   top of each arena heap. Small longer-lived allocations interleave above
   the freed 10 MiB holes, so the holes can never be trimmed. With 128+
   arenas on a many-core host, each arena ratchets its own top upward.

Whether a given machine *shows* the growth is environment-dependent —
arena count scales with core count, and the interleaving is timing-sensitive
(close to a race). The same binary can hold ~13.5 GiB on one Linux host
and exceed 47.8 GiB on another (observed: WSL2 desktop vs 32-vCPU cloud
host, same workload). Windows is largely unaffected because its heap can
decommit pages in the middle of a segment.

A/B evidence: RSS grows in both `--kvmem-mtp-state snapshots` and `replay`,
so the cause is unrelated to the GDN state-management strategy.

## Mitigation

`MALLOC_ARENA_MAX=2` caps glibc arenas; fragmentation can no longer spread
across arenas and released memory returns to the OS. Same 33-round canary:
**47.8+ GiB → ~21 GiB peak and flat** (12.1 GiB of that is the resident
model; the rest is retrieval scoring buffers).

The default is now set in `scripts/start-server.py` (the shared Linux
launcher for the IQ3/IQ4 recipes) via `env.setdefault`, so:

- it applies to existing CUDA users of the standard launch path;
- users can override it by exporting their own value;
- it is a no-op on non-glibc platforms.

This is an allocator-level mitigation, not a source-level fix. Whether the
per-step ubatch buffer can be reused or right-sized is a separate question
(lifetime, concurrency, peak impact) and is deliberately out of scope here.

## Reproduction / evidence

- Workload: `scripts/multimodal_canary.py --long-context-benchmark`
  (33 requests, 262058/262144 tokens), Qwen3.8-27B IQ3_S-mtp,
  budget 36864, reserve 16384, ctx 262144.
- Without mitigation: `peak_runtime_rss_mib` > 47.8 GiB equivalent.
- With `MALLOC_ARENA_MAX=2`: `peak_runtime_rss_mib` ≈ 20965 MiB,
  `peak_process_swap_mib` = 0.
- gdb backtrace of the allocation site preserved (`gdb-bt2.log`),
  available on request.

## Drive-by fix

`scripts/multimodal_canary.py`: streamed chunks may carry
`delta.content = null` (e.g. finish-reason chunks), which crashed the
response join with `TypeError`. Made the join null-safe.
