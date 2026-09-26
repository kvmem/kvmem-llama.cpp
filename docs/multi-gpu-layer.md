# Experimental CUDA layer split

This branch adds an opt-in, synchronous layer-split path for one local inference
session on multiple CUDA GPUs. The first validation target is two GPUs with
different VRAM sizes. It keeps one logical KVMem block/slot selection and one
host history. llama.cpp owns weight placement, layer execution and activation
transfers; each attention layer's K/V remains on that layer's GPU.

Use an explicit device list, `--split-mode layer`, and `--gpu-layers all`.
`--tensor-split` accepts one proportion per selected GPU,
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
keep their existing fast behavior. Embedded-nextn MTP with snapshot rollback
or per-device ReplaySSM is experimental and requires an explicit
`--kvmem-mtp-state snapshots|replay` on the dual-GPU server. Row/tensor split
and cross-machine execution are outside this release.

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

## Q4_K_M capacity and speed on the same pair

The downloaded `Qwen3.8-27B-UD-Q4_K_M.gguf` is 16,464,440,224 bytes
(15.33 GiB), and its SHA-256 matched the accompanying checksum. Under the
small 1K-context CLI configuration below, it actually loaded on the 16 GiB
card alone, with only narrow headroom. This does not reproduce the user's
reported single-card load failure, which may involve different context or
server settings. The split measurements use
CUDA0 = RTX 5060 Ti and CUDA1 = RTX 5050 Laptop. Both GPUs were idle before
and after the runs.
The CLI reported 15,174 / 7,123 MiB free when preparing the model. Every
run offloaded 66/66 layers and mapped another 682 MiB of model data on CPU.

The short runs used the same 302-token input, 64 generated tokens, Q8_0 K/V,
MTP off, context 1,024, KVMem budget 512 plus 128 generation reserve, batch
128 and ubatch 64 as the IQ3 test. The reported prompt count of 366 includes
64 tokens of KVMem query replay. Each row was a separate, non-overlapping
process. Times are llama.cpp's internal timings, which exclude process startup
and teardown; prefill and decode are single-run observations.

| Layer proportion CUDA0:CUDA1 | Model weights MiB CUDA0/CUDA1 | Load s | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Single CUDA0 | 14,674 / — | 10.15 | 200 | **23.59** |
| 2:1 | 9,103 / 5,571 | 9.05 | 356 | 17.63 |
| 3:1 | 10,312 / 4,363 | 8.96 | 381 | 19.29 |
| 5:1 | 11,455 / 3,220 | 8.89 | 410 | 20.28 |
| 8:1 | 12,428 / 2,246 | 9.38 | 327 | 20.91 |
| 12:1 | 12,917 / 1,757 | 8.91 | 428 | 21.58 |

A second 12:1 run gave 8.80 s load, 440 prompt tok/s and 21.38 decode
tok/s. A stock-cache control at 12:1, without KVMem, gave 21.87 decode
tok/s (302 prompt tokens because it does not replay the query). Thus the
ordinary-decode difference attributable to KVMem in these short runs is
small; most of the cost is layer execution across unlike GPUs. Prefill varies
more between runs and should not be ranked from this table alone. The 12:1
split delivered about 91% of the single-card decode rate in this constrained
setup; the single card's prompt throughput was lower despite faster decode.

At 12:1 with a 1,278-token input, context 2,048 and forced retrieval of an
older block, the run completed at 455 prompt tok/s and 20.09 decode tok/s;
retrieval took 59.88 ms. A separate diagnostic run showed `stage_in=12`,
selected the forced old block, and found zero mismatched packed K bytes in
sampled attention layers on both GPUs before and after restoration. This is a
functional spot check rather than an end-to-end quality evaluation.

For this model, 8:1 is a reasonable initial setting: it gives about 97% of
the observed 12:1 decode rate while leaving roughly 0.5 GiB more CUDA0
headroom after weight placement. The 12:1 split is faster in this 1K/2K
context test but leaves only about 2.2 GiB free on CUDA0 at KVMem pool
planning; longer contexts or other GPU processes may favor 8:1 or 5:1. The
single-card result is too close to its memory limit to generalize to the
user's normal server settings.

### rc3 default-launcher server profile

The packaged rc3 `start-iq3.ps1` calls `start-server.ps1` with context
262,144, KVMem budget 36,864, generation reserve 16,384, Q8_0 K/V, block
size 128, MTP3, CPU vision projector, and thinking enabled. Its default GPU
selection is the RTX 5060 Ti alone. The equivalent launcher in this branch
was dry-run with the Q4_K_M model and the local Q8_0 projector, then run
without changing those settings. It reached HTTP readiness at about 13.76 s,
but left only 39 MiB free on the 5060 Ti, dropping to 11 MiB during the
request. One uncached 354-prompt-token,
64-completion-token chat request measured 12.62 prompt tok/s and 3.04 decode
tok/s, with a 49.13 s server-reported request time. This one near-capacity
run does not prove the exact driver-level cause of the slowdown.

The rc3 launcher itself only selects one GPU. For the dual-GPU comparison, the
same server binary and settings were used with an explicit CUDA0/CUDA1 layer
split and MTP disabled, to match the single-card MTP-off control. The request,
Q4_K_M model, CPU projector, context, KVMem budget/reserve, Q8_0 cache, block
size and thinking mode were otherwise the same. These are not strictly
equivalent to the single-card MTP3 run.

| Server placement | Ready s | Free VRAM MiB, 5060 Ti / 5050 | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| rc3 launcher, single with MTP3 | 13.76 | 39 / — | 12.62 | 3.04 |
| Dual 5:1, MTP off | 12.20 | 2,347 / 3,553 | 268.81 | 19.62 |
| Dual 8:1, MTP off | 11.64 | 1,251 / 4,647 | 283.45 | 20.35 |
| Dual 12:1, MTP off | 10.96 | 649 / 5,251 | 253.69 | 20.79 |

All four requests completed 64 tokens. The VRAM figures are idle-after-load
`nvidia-smi` samples, not peak free memory during inference. A single-GPU
server with the same rc3 memory settings but MTP disabled failed during a
1,768 MiB CUDA KV-cache allocation, so it supplies no matched single-GPU
throughput control. At these settings, 8:1 offers nearly the measured 12:1
decode speed with roughly 0.6 GiB more free VRAM on the 5060 Ti. The test
servers were stopped after each run.

## Experimental dual-GPU MTP

The implementation followed [multi-gpu-mtp-plan.md](multi-gpu-mtp-plan.md):
snapshot rollback first, then ReplaySSM with one GDN descriptor buffer and
fold stream per owning CUDA device. Each recurrent layer's state, convolution
state and five record tensors must have the same CUDA owner. A commit waits
for all launched device groups before publishing the new position. The
two-GPU path uses one embedded `nextn` layer; the follower KV must live on
that layer's GPU and use the requested draft dtype and target slot count.
Per-device pool planning reserves the snapshot planes or ReplaySSM records,
plus a worst-case F32 follower KV and compute headroom. `auto` and sidecar
draft models remain unsupported for multi-GPU MTP. The rc3 launcher remains
single-GPU by default; the dual-GPU mode is explicit.

IQ3 validation used 5:1, Q8_0 target/draft KV, a 512+128-token pool and a
1,278-token prompt that forced an archived block back into the active window.
Target retrieval staged in 12 blocks and the MTP follower restored 16 blocks
from host history. Both rollback modes generated the same 32 token IDs as
single-GPU snapshots. On the same dual placement, the serialized 156,894,364-
byte GDN state had the same FNV-64 hash (`1117ddd0f1d9d4e2`) after snapshot
and replay generation. Single-GPU placement produced a different state hash
despite identical token IDs, so state equality is only established between
rollback modes on the same layer placement. One-run generation rates were
40.64 tok/s for dual snapshots, 39.41 for dual ReplaySSM and 48.14 for single
snapshots.

The same IQ3 forced-retrieval prompt was also run at MTP widths 1 and 2.
Both widths exercised rejected and partially accepted drafts. Snapshot and
ReplaySSM produced the same 32 token IDs and GDN state hash at each width:
`ee2125fc97fbff18` at width 1 and `5ec5a2cc23780abd` at width 2. The
width-3 long-history run above exercised partial and full acceptance.
Q4_K_M at 5:1 with the 512+128-token pool also forced 12 target blocks back
into the window; the follower restored 16 host blocks. Snapshot and ReplaySSM
again produced identical 32-token output and GDN hash
(`46c81e0151974061`). These are end-to-end checks, not fault-injection tests
of a failed fold on one GPU.

The Q4_K_M CLI test kept the rc3 context and slot-pool settings (262,144
context, 36,864 budget, 16,384 reserve, block 128), with a 450-token reported
prompt, 64 generated tokens, Q8_0 target/draft KV and no vision projector.
All four dual runs generated the same 64 token IDs, accepted 42/64 draft
tokens and had identical snapshot/ReplaySSM GDN state hashes on 8:1
(`1bbcd3103671c095`). These CLI rates are not comparable to the server
rows below, where the default draft KV is F16 and the chat prompt differs.

| Split | Snapshot gen tok/s | ReplaySSM gen tok/s |
|---|---:|---:|
| 5:1 | 28.91 | 28.58 |
| 8:1 | 31.48 | 31.25 |

The matched server test used the Q4_K_M model and CPU Q8_0 projector, rc3
context and KVMem defaults, explicit layer split, and the same 365-token
OpenAI chat prompt plus 64-token completion in every row. The UI was disabled
for the test; no image was sent. Draft KV used the server's F16 default.
Free VRAM is an idle-after-load sample in MiB, ordered 5060 Ti / 5050. These
are single runs, not stable throughput distributions.

| Split | MTP state | Free VRAM MiB | Prompt tok/s | Decode tok/s |
|---|---|---:|---:|---:|
| 5:1 | off | 2,347 / 3,553 | 276.81 | 16.46 |
| 5:1 | snapshots | 1,955 / 2,755 | 223.32 | 18.28 |
| 5:1 | ReplaySSM | 2,333 / 2,807 | 201.65 | 19.46 |
| 8:1 | off | 1,251 / 4,647 | 263.03 | 16.71 |
| 8:1 | snapshots | 831 / 3,879 | 234.15 | 20.04 |
| 8:1 | ReplaySSM | 1,237 / 3,903 | 252.00 | 19.83 |

ReplaySSM improved decode by about 18-19% over MTP-off for this request,
while leaving about 392-406 MiB more free on the 5060 Ti than snapshots.
Snapshot versus ReplaySSM speed is too close and variable to rank from one
run. After cancelling a streamed 8:1 ReplaySSM request, two subsequent
requests succeeded with the same output; the second reused 364 of 365 prompt
tokens. The two modes also produced identical output for each split. At 5:1,
MTP output differs from the MTP-off output by a small wording choice, and
that difference repeated. At 8:1, all three modes matched. This may be
floating-point sensitivity to layer placement and speculative batch shape;
the tests do not prove bitwise output parity across all placements or broad
quality parity. Keep dual MTP opt-in pending wider model, prompt and stability
testing.

## Relation to [PR #54](https://github.com/kvmem/kvmem-llama.cpp/pull/54)

PR #54 does add two-GPU layer work, alongside a substantially larger
TurboQuant KV codec port. It relaxes the server's device guard and adds a
`KVMEM_MG_SAFE=1` switch to bypass several single-GPU fast paths. It also
groups GDN Replay descriptors and fold launches by GPU, aiming to support
MTP/ReplaySSM across cards. Its diff still rejects multiple
`--tensor-split` proportions and does not add per-device KV-pool sizing. The
present branch instead makes the safe path automatic for multi-device models,
supports unequal layer proportions, validates KV placement and tests a
device-bounded pool. It adds independently validated per-device GDN Replay
without TurboQuant. The GDN grouping in #54 informed the MTP follow-up, but
its wider patch was not imported.

Future tensor split with a mirrored active KV window can retain the same host
archive and logical slot selection, while replacing each layer's single KV
destination with a set of required replicas. Completion must wait for every
replica before publishing a new window. Peer-to-peer transfers may optimize
that fan-out but are not a prerequisite for the interface.
