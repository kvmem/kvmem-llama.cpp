# Layer-split MTP follow-up plan

Scope: two local CUDA GPUs, one inference sequence, Qwen3.8-27B with an
embedded `nextn` layer, and the existing layer-split KVMem slot pool. Tensor
split, P2P, and TurboQuant are separate work. This is an implementation plan;
multi-GPU MTP is still rejected by this branch.

## What PR #54 supplies

[PR #54](https://github.com/kvmem/kvmem-llama.cpp/pull/54) groups recurrent
GDN layers by their CUDA pointer's device, allocates one replay-descriptor
buffer and stream per device, launches a fold for every group, then waits for
all streams before `replay_finish`. It relaxes the hybrid ReplaySSM predicate
that currently requires every recurrent layer on one GPU. Those are the two
useful MTP-specific ideas. The PR has no changed test files and bundles this
work with unrelated TurboQuant codecs. Its `KVMEM_MG_SAFE=1` switch is also
unnecessary here: this branch already selects the device-aware K/V path
automatically for multiple GPUs.

## Current boundaries

- `tools/llama-kvmem-server.cpp` and `tools/llama-kvmem-cli.cpp` reject MTP
  when the explicit device list contains more than one GPU.
- `src/adapter/llama-memory-kvmem-hybrid.cpp::use_gdn_replay` requires all
  recurrent layers on the same CUDA device. `llama-memory-kvmem.cpp` owns only
  one descriptor buffer, stream, and fold device.
- The target attention cache already follows layer placement. The MTP
  follower uses the target's logical slots and generally reads/writes its
  packed K/V through the owning tensor backend. The target's multi-GPU
  layout path bypasses D2D, so the follower's single-device D2D layout is not
  called on that path. This needs an explicit integration test with MTP
  attached, especially for resident and restored blocks.
- `kvmem_compute_pool` bounds target attention K/V per owning GPU before the
  MTP follower cache and all speculative rollback/ReplaySSM allocations are
  made. A 53,248-token rc3 budget/reserve can therefore pass target planning
  and still exhaust the GPU holding the `nextn` layer or most GDN layers.

## Phase 1: enable multi-GPU MTP with snapshot rollback

1. Permit `--spec-type draft-mtp` on an explicit CUDA layer split only when
   `--kvmem-mtp-state snapshots` is selected. Leave the current ReplaySSM
   default rejected for multi-GPU, with an error naming the required mode.
   Require the embedded `nextn` draft in this first release; an independent
   `--spec-draft-model` has separate placement to solve. Keep single-GPU
   behavior and its default unchanged.
2. Load the embedded MTP layer and create its follower context. Check that
   the follower K/V buffer is on its expected model-layer GPU, that its cell
   count equals the target slot pool, and that its K/V dtype is the requested
   draft dtype. Log the owner and bytes. Do not assume the draft layer is on
   CUDA0 or that a CUDA current device matches its buffers.
3. Exercise the existing host-backed layout/retrieval path with an attached
   follower. Check that both target and follower select the same block IDs,
   preserve resident packed K/V, restore evicted packed K/V, and commit only
   accepted speculative tokens. Rejected tokens must not advance either the
   attention cache or GDN state.
4. Add per-device MTP headroom to pool planning or preflight. It must cover
   the follower cache on its owner, four GDN state planes for MTP3 snapshot
   rollback, and compute buffers, then fail clearly before CUDA OOM if the
   requested fixed budget cannot fit. Start Q4 server tests at 5:1, where
   the MTP-off rc3-profile run left 2,347 / 3,553 MiB free, before trying
   8:1 (1,251 / 4,647 MiB) or 12:1 (649 / 5,251 MiB).

Snapshot mode is a correctness milestone, not the final default: the extra
GDN state planes use more VRAM than ReplaySSM. Test IQ3 first, since it has
more margin, then Q4. Compare greedy target output with single-GPU snapshot
mode where it fits; also track draft acceptance, rejected-prefix lengths,
and exact GDN state hashes rather than relying only on final text.

## Phase 2: per-device ReplaySSM

1. Port PR #54's GDN device grouping into `GdnReplay`, but use RAII for each
   stream and descriptor buffer. Verify that each layer's committed state,
   convolution state, and replay tensors belong to the same CUDA device.
   The fold itself is device-local and does not need P2P on this machine.
2. After `llama_synchronize(ctx)`, launch one fold per device group. Wait for
   every group before `replay_finish(n_keep)` or exposing the new logical
   position. If a launch/sync fails, poison the entire replay transaction and
   terminate that request; do not continue with some GPUs folded and others
   unfurled.
3. Relax `use_gdn_replay` to require supported CUDA owners, rather than one
   identical owner, only after multi-device fold tests pass. Then allow
   `--kvmem-mtp-state replay` on the two-GPU path. Keep `auto` conservative
   until the complete integration matrix passes.
4. Extend capacity checks for ReplaySSM record buffers, which differ from
   snapshot-plane memory, and make the startup log show target K/V, follower
   K/V, GDN state/records, and residual headroom on each GPU.

## Release gates and measurements

- Synthetic GDN folds with layers on both GPUs: keep 0, 1, partial, and all
  across MTP widths 1–3; exact committed-state/conv-state comparison against
  snapshot rollback, logits within the existing tolerance, and no advance
  after a rejected or aborted transaction.
- IQ3/Q4 end-to-end MTP3: long prefill, forced old-block retrieval, query
  replay, repeated requests/prefix reuse, a rejection-heavy prompt, and
  cancellation during verification. Assert matching target/follower block
  selection and packed K/V after restoration on both GPUs.
- Run single-GPU MTP regression, multi-GPU layer without MTP regression, and
  the project test suite. Only then update the default launcher to opt into
  dual MTP; the first experimental flag should remain explicit.
- Benchmark the rc3 profile at 5:1 and 8:1 with the same requests used for
  MTP-off measurements. Report prompt/decode throughput, draft acceptance,
  fold time, load time, per-GPU free/peak VRAM, and retrieval latency. MTP may
  lose speed if the draft layer sits on the slower 5050 or acceptance is low;
  enable it by default only when it improves sustained throughput safely.
