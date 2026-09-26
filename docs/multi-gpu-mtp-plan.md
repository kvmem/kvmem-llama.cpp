# Dual-GPU MTP implementation and remaining validation

Scope: two local CUDA GPUs, one inference sequence, an embedded `nextn` draft
layer, and the existing layer-split KVMem slot pool. The snapshot and ReplaySSM
paths described below are implemented on this branch. Benchmark and retrieval
results are recorded in [multi-gpu-layer.md](multi-gpu-layer.md).

## Supported path

- Enable MTP with `--spec-type draft-mtp` and an explicit
  `--kvmem-mtp-state snapshots` or `replay` on the two-GPU layer-split server.
  Multi-GPU `auto` selection and independent `--spec-draft-model` are not
  supported.
- The embedded `nextn` follower KV must be on the GPU that owns the draft
  layer, and it must use the target's logical slot count. The implementation
  validates placement and capacity before serving requests.
- ReplaySSM groups recurrent layers by their CUDA device. Each group owns its
  descriptor buffer and stream; the adapter launches one fold per device and
  waits for every group before committing the accepted prefix. A launch or
  synchronization failure poisons the transaction rather than publishing a
  partially folded state.
- Per-device pool planning accounts for target attention KV, the MTP follower
  KV, snapshot planes or ReplaySSM records, and compute headroom.

## Validation still needed before changing defaults

The dual-GPU mode remains opt-in. Before making it a launcher default, expand
the short local measurements into repeated runs covering long prefill, forced
old-block retrieval, prefix reuse, rejection-heavy verification, and request
cancellation. Track output parity, accepted draft tokens, per-GPU peak VRAM,
retrieval latency, and fold time. Keep the existing single-GPU MTP and dual-GPU
layer-without-MTP regressions in the same run matrix.

The current evidence shows that both snapshot and ReplaySSM modes complete the
repository task-1 request with matching output on the tested Q4_K_M setup. The
observed decode rates are close enough that the available short runs do not
establish a stable speed ranking. See the benchmark tables in
[multi-gpu-layer.md](multi-gpu-layer.md).

## Relationship to PR #54

[PR #54](https://github.com/kvmem/kvmem-llama.cpp/pull/54) also proposes
CUDA layer split and per-device GDN replay, bundled with TurboQuant KV codecs.
This branch keeps the KVMem multi-GPU work isolated, adds unequal layer
proportions and per-device KV-pool validation, and records local MTP and
retrieval checks. The implementations overlap in the core layer-split and
ReplaySSM behavior, so reviewers should treat them as alternative integration
paths rather than merge both wholesale.
