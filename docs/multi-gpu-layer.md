# Experimental CUDA layer split

This branch adds an opt-in, synchronous layer-split path for one local inference
session on multiple CUDA GPUs. The first validation target is two GPUs with
different VRAM sizes. It keeps one logical KVMem block/slot selection and one
host history. llama.cpp owns weight placement, layer execution and activation
transfers; each attention layer's K/V remains on that layer's GPU.

Use an explicit device list, `--split-mode layer`, `--gpu-layers all`, and
`--spec-type none`. `--tensor-split` accepts one proportion per selected GPU,
which is useful when the cards have different free memory. After model weights
load, KVMem caps the common KV token window by the smallest per-device
attention-KV capacity. The allocated cache is checked against each layer's
expected GPU. Hybrid models reuse the already allocated attention-cache size
instead of planning against free memory a second time.

The multi-GPU path synchronizes graph capture before reading temporary tensors
and uses each tensor's ggml backend for host reads/writes and slot remapping.
It bypasses the existing single-device CUDA staging, D2H pipe, D2D layout
scratch and GPU mean-K accumulator. This is a correctness baseline, not a
throughput optimization. It works without CUDA peer access. Single-GPU paths
keep their existing fast behavior. MTP/ReplaySSM, row/tensor split and
cross-machine execution are outside this release.

## Validation on 2026-09-24

- Windows CUDA 12.9 build and all 11 project tests passed.
- RTX 5060 Ti 16 GiB + RTX 5050 Laptop 8 GiB: Qwen3.5-0.8B-Q8_0 loaded
  model, attention KV and recurrent buffers on both GPUs; short generation
  matched the single-GPU and stock-cache controls token for token.
- A 1,278-token prompt with a 160-token KVMem budget selected an archived
  block, reported `stage_in=1`, remapped the active window and completed
  generation. Diagnostic reads of retained and restored K/V matched their
  source bytes on both devices. Greedy output after this forced retrieval was
  not identical to single-GPU output. Switching the KV type from Q8_0 to F16
  reversed which result each configuration produced, so bitwise generation
  parity is not established; quality needs a larger evaluation set.
- The dual-GPU server answered an OpenAI-compatible chat request.
- Qwen3.8-27B-GSQ-RCO-IQ3_S loaded about 6.6 GiB of model weights on CUDA0
  and 4.2 GiB on CUDA1, placed attention KV and recurrent state on both, and
  generated tokens. This short run is a capacity/functional check, not a
  reliable speed benchmark.
- A deliberately tiny 128-token budget plus forced retrieval of an old block
  failed query replay at a position gap on both single and dual GPU. The
  160-token budget retained the replay start and passed. This is a pre-existing
  minimum-window limitation, not evidence of a dual-GPU transfer failure.

## Relation to [PR #54](https://github.com/kvmem/kvmem-llama.cpp/pull/54)

PR #54 does add two-GPU layer work, alongside a substantially larger
TurboQuant KV codec port. It relaxes the server's device guard and adds a
`KVMEM_MG_SAFE=1` switch to bypass several single-GPU fast paths. It also
groups GDN Replay descriptors and fold launches by GPU, aiming to support
MTP/ReplaySSM across cards. Its diff still rejects multiple
`--tensor-split` proportions and does not add per-device KV-pool sizing. The
present branch instead makes the safe path automatic for multi-device models,
supports unequal layer proportions, validates KV placement and tests a
device-bounded pool. It deliberately defers MTP and TurboQuant. The GDN
grouping in #54 is a useful starting point for the later MTP phase, subject
to independent long-history and rollback tests.

Future tensor split with a mirrored active KV window can retain the same host
archive and logical slot selection, while replacing each layer's single KV
destination with a set of required replicas. Completion must wait for every
replica before publishing a new window. Peer-to-peer transfers may optimize
that fan-out but are not a prerequisite for the interface.
