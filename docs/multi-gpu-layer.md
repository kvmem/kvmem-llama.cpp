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

## IQ3 speed on the 16 GiB + 8 GiB pair

Qwen3.8-27B-GSQ-RCO-IQ3_S was measured after the other GPU services were
stopped. CUDA0 is the RTX 5060 Ti (16 GiB), and CUDA1 is the RTX 5050 Laptop
(8 GiB). The CUDA Driver API reports no peer access in either direction.
Each row is a separate process, with no benchmark processes overlapping.
Common settings: Windows CUDA 12.9, MTP off, Q8_0 K/V, 1,024-token context,
512-token KVMem budget plus 128-token generation reserve, batch 128,
ubatch 64, the same 302-token input and 64 generated tokens. KVMem's 64-token
query replay makes the reported prompt count 366. Throughput excludes model
load; all rates below are from the 64-token runs.

| Layer proportion CUDA0:CUDA1 | Model weights MiB CUDA0/CUDA1 | Prompt tok/s | Decode tok/s | Decode vs single |
|---|---:|---:|---:|---:|
| Single CUDA0 | 10,827 / — | 230 | **28.83** | 100% |
| 1:1 | 4,746 / 6,081 | 277 | 15.30 | 53% |
| 2:1 | 6,629 / 4,198 | 313 | 19.81 | 69% |
| 3:1 | 7,637 / 3,190 | 341 | 23.05 | 80% |
| 8:1 | 9,257 / 1,570 | 364 | 25.94 | 90% |

The earlier serial 24-token runs showed the same ordering (single 27.19;
1:1 15.19; 2:1 19.39; 3:1 22.30 tok/s). Prefill varied more
between runs than decode, so the prompt column is one observed run, not a
stable ranking. All five runs generated 64 tokens. The 1:1, 2:1 and 8:1
outputs matched the single-GPU token IDs; 3:1 matched the first 24 and then
diverged. The matched no-KVMem control gave 29.02 tok/s on one GPU and
23.94 tok/s at 3:1. This is close to the KVMem result at each placement;
the decode penalty is mainly associated with layer placement and inter-device
work, rather than KVMem's synchronous history transfer, which is idle during
ordinary decode. The two cards differ in speed and upstream scheduled four
copies for the dual graph versus one for the single graph; this benchmark does
not isolate their individual contributions.

A separate 1,278-token input forced retrieval of an old block with a
512-token budget. At 3:1, dual-GPU prefill was 409 tok/s, decode 20.02 tok/s,
and retrieval 32.58 ms; the single-GPU measurements were 335 tok/s,
25.83 tok/s, and 29.07 ms. A trace of the dual run confirmed 12 archived
blocks staged in, with sampled K/V byte comparisons matching on layers from
both GPUs. Prompt throughput varies between runs, while the decode slowdown
persisted. These are short, single-session tests and do not validate long-run
quality or thermal stability.

IQ3 already fits the 16 GiB card, so single-GPU decode is faster here. For a
larger or higher-precision model that requires both cards, choose the least
work on CUDA1 that still leaves enough free memory on both cards. In this
measurement 3:1 moved about 3.1 GiB of weights off CUDA0 at a roughly 20%
decode-throughput cost; 1:1 moved about 5.9 GiB at roughly 47% cost.

An initial two-token dual run under concurrent GPU services measured only
0.25 tok/s. At that time the 5060 Ti had about 4 GiB free in `nvidia-smi`;
after both GPUs became free, the slowdown did not recur. This suggests a
severe sensitivity to VRAM pressure, but the short run cannot establish the
driver-level cause. Benchmark results should therefore report free VRAM and
other GPU processes, not just total card size.

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
