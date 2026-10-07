# Multi-backend validation status — 2026-10-03

This is a development candidate, not a final P6–P10 release qualification.
The repository consolidation changes source ownership and build entry points;
it does not establish new GPU performance numbers.

2026-10-06: KVMem cold snapshots in ninfer are temporarily disabled at the launcher,
serve CLI and Engine option boundaries by user request. Disk results below are prior
evidence, not currently enabled functionality. Host history reuse remains available.

2026-10-06: Legacy lookup drafting is explicitly disabled in KVMem ninfer by user
request. The launcher rejects `--lookup-ngram` in `WorkerArgs`; serving and Engine
reject nonzero lookup values with a message pointing to `--ngram-draft-tokens` and
`--ngram-min-match`, including without MTP or alongside new ngram drafting.
The existing new-ngram admission matrix remains passing. An eight-job VS2022 Release
build of the server, P1 option fixture and KVMem product test bundle passed; launcher,
Engine (24 added lookup rejection cases), serving (48 added cases covering option
order) and three actual server CLI rejections passed without loading a model or GPU.
Local logs are in `results/lookup-disabled-20261006/`. The native patch update includes
only the eight files changed by this task and passed cached reverse verification;
the real staging index is unchanged. Whole-worktree `prepare-backends.py --check`
still rejects other tasks' newer source changes outside this update, which have not
been imported into this patch. The local server is updated; existing ZIPs are not.

## Implementation and prior evidence

- P0–P5: portable memory contracts and both backend adapters implemented.
- P6: fixed MTP, Main/MTP KV archives, original-position restoration and graph integration implemented.
- P7/P8: bounded retained Host histories and text concurrency (up to four active requests) implemented; selected session, queue and isolation checks passed.
- P9: single-active-request image support implemented and selected media identity/residency checks passed.
- P10: versioned, checksummed disk snapshots with atomic writes, quotas and recompute fallback implemented; INT8/MTP3 snapshot fault checks passed.
- RK8V4 and ngram: single-active-text integration implemented; selected payload, restore and speculative verification checks passed. Fixed MTP1–4 is accepted, but all widths and combinations are not fully qualified.
- Short-history reuse: input-prefix checkpoints now cover prompts below B+R. Prior tests passed 16 RK8V4/Graph/MTP4 short cases, 16 INT8/eager/non-MTP short cases, 18 RK8V4/Graph/MTP4 long cases, and a short RK8V4/MTP4/ngram disk cold/restart case.
- Actual DSH adapter checks passed. In one 8K conversation follow-up, 8,106 of 8,172 input tokens were reused; TTFT fell from roughly 11.3 seconds on the first turn to 1.1 seconds on the follow-up. Different turns and workloads: this is functional evidence, not a controlled speed comparison.

Detailed local evidence is retained outside the source checkout in the experiment
workspace: `results/p789-followup-20261003/acceptance.json`,
`results/p10-disk-int8-mtp3-fixed-20261003/acceptance.json`,
`results/p610-followup-20261003/runs.json`, and
`results/short-history-fix-20261003/acceptance.json`.
These paths identify local records; raw logs, models and personal configuration
are deliberately not included in Git.

## Generated-endpoint cache completion

The ninfer adapter now retains the complete committed execution endpoint and an
exact Frontend reconstruction boundary alongside its input/pre-query fallback.
The input fallback and generated reconstruction now have independent slots;
generation cannot replace the earlier input state. The four slots share one Host
KV archive. Same-query tool continuations retain
the frozen Q and selected view; changed queries use the existing probe/replay
route. Rewinding revokes later checkpoints before shared rows are overwritten.
Cold snapshots use v5 and persist all complete slots and the query attachment;
v3/v4 snapshots are recomputed. The llama.cpp adapter was not changed.

Earlier three-slot endpoint validation used Windows VS2022/CUDA 13.2, RTX 5060 Ti 16GB and the
Qwen3.8-27B-GSQ-RCO-IQ3_S MTP artifact. Ordinary and MTP4/ngram31 endpoint/rewrite
cases, 32 short-history and 18 long-history cases, single-token initialization,
new queries, edits, cancellation, append/window pressure, concurrency publication
order, and cold save/restart/cancel/corruption fallback passed. StateImage and
Main/MTP transfers were checked byte-for-byte. The independent token-sums oracle
passed 600 cases on the available RTX 5050 Laptop. The final single-token and
rewrite-marker adjustments were retested with the full MTP4 endpoint sequence
and the HTTP/DSH cases below; the earlier full matrix was not rerun on the final
binary. Those earlier checks used the v4 snapshot schema; they do not qualify the
later four-slot change described below.

A matched continuation comparison used B36864/R16384/H6144MiB, RK8V4,
MTP4/ngram31 and the same model, template and sampling settings on RTX 5060 Ti:

| Worker | Input tokens | Reused tokens | Computed prefill | TTFT |
|---|---:|---:|---:|---:|
| Before endpoint completion | 41308 | 8498 | 32810 | 53.861s |
| With endpoint completion | 41308 | 41266 | 42 | 1.810s |

The fixture generated 32768 tokens with a schema-constrained array, then replayed
a constructed tool transcript. The old worker seeded its original input
checkpoint with one output token and processed the same continuation; it did not
repeat the 32K generation. Appending to 64771 tokens, beyond the 53248-token
physical window, retained 41339 tokens and computed only the 23432-token suffix.
This comparison measures continuation work/TTFT, not generation throughput or
model quality. Endpoint capture and Host transfer costs were not individually
timed in this HTTP run.

Separately, the installed DSH adapter passed a real streaming tool-call/result
roundtrip through the Responses API with xhigh/preserve-thinking. The second
input reused 8132/8164 tokens, computed 32 and had TTFT 0.570s. This sample emitted
no model reasoning tokens; canonical thinking-close is covered by the native
regression. The deployed service kept its existing B81920/R32768 configuration.
This was an adapter roundtrip, not a full user Agent benchmark.

Local evidence: `results/endpoint-cache-fix-20261003/report.md`, `acceptance.json`,
`http-acceptance.json` and `dsh-endpoint-acceptance.json`. Existing build-tree
sources were checked against the vendored changed files; this targeted fix does
not qualify the full P6-P10/MTP4 combination matrix.

## Canonical input checkpoint preservation — 2026-10-04

The three-slot adapter reused one rewrite slot for both canonical input fallback
and generated thinking-close reconstruction. A real-model regression reproduced
the overwrite: input frontier 1485 was replaced by generated frontier 1531, and
the next 1675-token synthetic tool history reused only the 57-token Base. Separate
input/reconstruction slots now preserve the earlier input. Restoring a slot also
keeps its complete state when prefill starts exactly at that capture boundary;
repeat and cancelled requests no longer consume the only reusable input state.

On Windows VS2022/CUDA 13.2 and RTX 5060 Ti, the IQ3_S artifact passed ordinary
generation and RK8V4/Graph/MTP4/ngram31 input-fallback cases (physical window 512,
B384/R128, logical context 16384, Host 2 GiB). The next turn reused 1485 tokens;
later appends reused 1670 and 1855, and repeat/cancel/recovery reused 2040. Thinking
budget 16 ensures a generated reconstruction boundary is exercised. The existing
MTP4 endpoint suite passed retained reasoning, edited answers, changed queries,
history edits and cancellation. Separate-process v5 save/read and damaged-record
recompute tests passed. Native Main/MTP KV and StateImage restores were checked
byte-for-byte, with Host payload and device-page retirement bounds enforced.

Local evidence: `results/windows50-task2-b764974-20261003/cache-fix-fixed-matrix-mtp4-final`
(first two input-fallback groups) and `cache-fix-fixed-matrix-rest-mtp4`
(endpoint and snapshot groups). These are focused correctness regressions at a
smaller window; the separate full 256K rerun is recorded below. No new
generation-throughput improvement is claimed.

The rebuilt SM120a precompiled worker also passed HTTP streaming with the DSH
settings (B81920/R32768, H6144 MiB, RK8V4, MTP4/ngram31/min-match12, FP16 GDN).
Three synthetic tool-history inputs contained 7205, 7532 and 7859 tokens; the
later turns reused 7204 and 7531. A diagnostic thinking budget of 16 was added to
force thinking-close, and greedy sampling made this a functional check. All 70
embedded b11165 chat-UI assets matched the official archive. The DSH service was
restored using this worker with its existing production arguments. Local HTTP
evidence is in `cache-fix-http-checked`; the full rerun is recorded below.

## Complete IQ3_S Task 2 256K rerun — 2026-10-04

The rebuilt precompiled worker passed 33 requests / 32 fixed read_file tool
rounds on RTX 5060 Ti. Final input was 261528 tokens and
final output was 512, under a logical context of 262144. Settings were
RK8V4, fixed MTP4/ngram31/min-match12, B81920/R32768, Host 8192 MiB,
prefill 256, FP16 GDN, CUDA Graphs (72 MiB allowance), xhigh and the DSH
sampling/template/profile configuration. No diagnostic thinking cap was used.

All 32 tool rounds reported prefix hits; final history counters were 32 hits,
2 misses and 0 evictions. The previously failing 163493-token request had an
identical JSON payload: reuse rose from 0 to 155318, and TTFT fell from 727.815s
to 23.268s. Maximum tool-round TTFT was 26.190s.

Weighted tool decode was 41.83 tok/s; final decode was
36.96 tok/s with 23.358s TTFT.
MTP accepted 6170/12942 drafts
(47.67%); ngram accepted 523/1510.
These speculation totals exclude the initial base request. MTP is separated
by subtracting the matching per-tool-request ngram counts from mixed counters.

At GPU utilization of at least 50%, whole-card power averaged
114.3W (peak 147.8W;
limit 180W), and SM clocks averaged
1782MHz. Peak VRAM was 15228 MiB;
minimum available system RAM was 3584 MiB.
All 70 embedded UI assets passed. DSH was restored with its original 128K
context / 6 GiB Host arguments and the fixed worker after the test.

Local evidence: `results/windows50-task2-b764974-20261003/task2-256k-fixed-20261004/summary.json`,
`cache-fix-comparison.json`, `telemetry.csv` and the per-round API records.
The old interrupted run remains separate. This is a fixed tool-history load
test; it does not grade the generated program or qualify the full P6-P10 matrix.

## IQ3_S Task 2 256K with ngram disabled — 2026-10-04

The same packaged worker (SHA256
`9c1d1a312dc25e8de82a3861bc8d15483f67faff4d790db5f70a3cefdb733987`)
passed another 33 requests / 32 tool rounds on RTX 5060 Ti. MTP4 remained
enabled; ngram changed from 31 to 0. Command and environment comparison
confirmed this was the only engine argument difference, apart from the run's
request-log path. All 33 request JSON payloads were identical to the preceding
ngram31 run. Final input/output were again 261528/512 tokens.

| Observation | MTP4 + ngram31 | MTP4 + ngram0 |
| --- | ---: | ---: |
| Weighted tool decode, tok/s | 41.83 | 42.14 |
| Final decode, tok/s | 36.96 | 37.60 |
| Final TTFT, seconds | 23.358 | 23.567 |
| MTP accepted / drafted | 6170 / 12942 (47.67%) | 6858 / 13716 (50.00%) |
| Ngram accepted / drafted | 523 / 1510 | 0 / 0 |
| Tool rounds with prefix hits | 32 / 32 | 32 / 32 |
| Generated tokens, including base request | 10409 | 10596 |
| Summed request wall time, seconds | 964.236 | 972.692 |
| Busy whole-card mean power, W | 114.3 | 114.5 |
| Busy mean SM clock, MHz | 1782 | 1782 |

History counters were again 32 hits, 2 misses and 0 evictions. Maximum tool
TTFT was 26.449s; there were no TTFT outliers above 60s. The earlier failing
163493-token request reused 155318 tokens with 23.320s TTFT. Runtime props
confirmed MTP4/ngram0, and the log recorded no ngram drafts. All 70 embedded
UI assets passed. Telemetry collected 903 samples without sampling errors.

This single paired workload showed similar overall decode throughput. Output
trajectories and lengths differed despite identical fixed input histories, so
the small +0.75% weighted / +1.73% final decode changes do not establish a
repeatable gain from disabling ngram. System available RAM also differed:
minimum 723 MiB in this run versus 3584 MiB previously. Peak VRAM was
15120 MiB; peak runtime RSS/private commit were 15587/31277 MiB.

DSH was restored with its original MTP4/ngram31, 128K context and 6 GiB Host
configuration. No binary rebuild or package replacement was needed. Evidence:
`results/windows50-task2-b764974-20261003/task2-256k-mtp4-no-ngram-20261004/summary.json`,
`ngram-comparison.json`, `configuration-comparison.json`, `telemetry.csv` and
the per-round request/response records. Earlier runs and qualified archives
remain separate.

## IQ3_S Task 2 256K, INT8 B56K/R32K — 2026-10-04

The same fixed packaged worker passed 33 requests / 32 tool rounds on RTX
5060 Ti with `--kv-dtype int8`, B57344/R32768, MTP4 and ngram0. GPU KV
capacity was 90112 tokens; the engine reported 194 MiB free device memory
at startup. Logical context remained 262144; final input/output were
261528/512 tokens. Host payload allowance was increased to 10240 MiB
because complete INT8 main+MTP 256K history requires approximately
8.77 GiB, exceeding the preceding run's 8192 MiB allowance. Test API port
18203 was used because 18202 was occupied by a separate service.

Configuration verification confirmed the same binary and process environment,
MTP4/ngram0, sampler, template and device profile. All 33 request JSON payloads
matched the preceding RK8V4 B80K/R32K run. KV type, B and Host allowance
changed together; this comparison does not isolate the cost of KV quantization.

| Observation | RK8V4 B80K/R32K | INT8 B56K/R32K |
| --- | ---: | ---: |
| Weighted tool decode, tok/s | 42.14 | 41.55 |
| Final decode, tok/s | 37.60 | 45.14 |
| Effective incremental prefill, tok/s | 371.10 | 407.73 |
| Final TTFT, seconds | 23.567 | 22.040 |
| MTP accepted / drafted | 6858 / 13716 (50.00%) | 5959 / 13109 (45.46%) |
| Tool rounds with prefix hits | 32 / 32 | 32 / 32 |
| Generated tokens, including base request | 10596 | 9418 |
| Summed request wall time, seconds | 972.692 | 884.958 |
| Peak whole-card VRAM, MiB | 15120 | 15172 |
| Busy whole-card mean power, W | 114.5 | 118.5 |
| Busy mean SM clock, MHz | 1782 | 1775 |

Final history counters were 32 hits, 2 misses and 0 evictions. Maximum tool
TTFT was 22.869s with no outliers above 60s. The formerly failing 163493-token
request reused 155318 tokens with 21.788s TTFT. Ngram drafted/accepted counts
were both zero. All 70 embedded UI assets passed, and 817 telemetry samples
were collected without errors. The test validates load completion and cache
reuse; it does not grade generated program correctness.

System memory pressure limits performance interpretation: available physical
RAM briefly reached 11.8 MiB during the 49113-token request, with 3 runtime
samples below 128 MiB and 41 below 1024 MiB. Peak runtime RSS/private commit
were 10068/31585 MiB. This is a system-wide observation; the preflight also
showed 6112 MiB in use on the separate RTX 5050 GPU. Output trajectories and
lengths differed, so the shorter total wall time and higher final decode are
not evidence of a repeatable intrinsic INT8 speed advantage.

The worker SHA256 remained
`9c1d1a312dc25e8de82a3861bc8d15483f67faff4d790db5f70a3cefdb733987`.
Package source provenance is the b764974 base plus the recorded cache-fix
patch; current checkout commit at launch was recorded separately as 5f59454.
No worker rebuild or package replacement was performed. DSH was restored
with its original RK8V4 B80K/R32K, MTP4/ngram31, 128K context and 6144 MiB Host
arguments; post-test health was `ok`.

Evidence: `results/windows50-task2-b764974-20261003/task2-256k-int8-b56k-r32k-mtp4-no-ngram-20261004/summary.json`,
`int8-comparison.json`, `configuration-comparison.json`, `telemetry.csv` and
the per-round API records. Earlier runs and qualified archives remain intact.

## Consolidated checkout checks

- Fresh Windows VS2022 Release portable-core build: passed.
- CTest core/Host/session suites: 14/14 passed.
- Backend launcher argument/capability checks: passed.
- Existing Web UI builder unit tests: 3/3 passed.
- Fresh vendored ninfer CMake configure: passed with CUDA 13.2, SM120a and existing vcpkg dependencies. The portable core resolves inside this same checkout.
- A fresh full CUDA build and model inference from the reorganized checkout were not repeated as part of this repository-only change. Earlier deployed binaries and their validation remain separate evidence.

## Referenced backend layout (2026-10-03)

The engines now reference existing upstream projects through pinned submodules at
`backends/llamacpp` and `backends/ninfer`. The versioned integration patches reconstruct
the source trees from the cache-fix commit `3c6048d503655588f730766e13eab6e11ceb6a1d`
exactly: llama.cpp tree `33b84ca6308bfe68fccdfca4b805991d2c220681`, ninfer tree
`46f37b010410bbd7e6cfa0722447a8e878f2b76e`. Those hashes describe that layout
check. The current integration patch and prepared trees are the later values in
`backends/versions.json`.

- Fresh independent Windows checkout: both fixed commits fetched from their public
  upstream repositories; patch preparation and strict whole-tree verification passed.
- Repeated preparation is idempotent. An additional developer edit was preserved;
  strict verification rejected it and passed after restoration.
- Fresh VS2022 Release Host/server build with CUDA/HIP disabled: 21/21 CTest suites passed.
- Existing portable-core rebuild: 14/14 suites passed. UI builder: 3/3; ROCm entry:
  13/13; backend launcher, Windows runtime/quantizer packaging, shell and PowerShell
  syntax checks passed. Python checks used local Python 3.14.
- ninfer build wrapper configured successfully with CUDA 13.2, SM120a and the existing
  compatible vcpkg installation; the core resolves inside this checkout.
- Source export includes 6441 manifest-verified files, including 3620 llama.cpp and
  2513 ninfer files. Preparation passed without Git metadata; modifying a bundled
  backend file was rejected.

Local evidence is retained in `backend-reference-20261003/layout-verification.json`,
`patch-reconstruction.json` and `host-tests.log` in the experiment workspace. This
layout check did not repeat full CUDA builds, UI npm builds, real-model inference
or performance measurements. The deployed port 18201 worker and configuration were
not changed by this source reorganization; the release qualification below remains
separate.

## Before release / merging to master

The selected source checks are not a finished release matrix. Still open: a
controlled performance comparison on the RTX 5060 Ti, long-context MTP
qualification across the admitted combinations, and a package whose report
matches this source. Cold snapshots stay disabled by request, so disk restart
is not a current release gate. Parameter acceptance does not prove every
combination. Video, live NVMe and cross-backend KV migration remain unsupported.
See [the capability documentation](multi-backend.md).

The earlier llama.cpp MTP cold performance difference (approximately 3% in one
comparison) remains an unresolved performance finding, not a confirmed logic bug.
The original mandatory_trim sink-page logging discrepancy was approved for a
logging-only fix; the selection algorithm was preserved. Detailed provenance and
other legacy findings remain in the experiment workspace's `legacy-findings.md`.

早期图片＋MTP＋ngram 补测开放 INT8/NVFP4；当时记录的 K8V4 换行差异仍待定位，BF16/RK8V4 在后续补测中放行。该早期范围与复制/replay证据见 [图片ngram补测](vision-ngram-validation-20261005.md)，当前准入见 [能力说明](multi-backend.md)。

ngram1–15 的准入已从单路扩为1–8路。该轮文本及 resident/CPU 图片为五种 KV；启动并发大于1时，配置的16–63自动降至15并提示，单路保留原宽度。实际八路eager/ngram15及Graph验证见 [ngram并发补测](ngram-concurrency-validation-20261006.md)。其后四种 KV 见文末当前状态。

图文活动请求上限已扩为八路，启动脚本默认视觉上限改为 1024。5060 Ti 上的 INT8 CPU视觉/自适应MTP3/Graph 和 resident视觉/ordinary/eager 八路 Engine 回归已通过，后者使用每图实际1024 tokens。CPU视觉实际1024-token图片的 HTTP 冷/缓存两轮共16请求也通过，缓存阶段峰值8路；范围见 [视觉并发调整](vision-concurrency-validation-20261006.md)。

2026-10-06 又以格式登记和准入改动放行 `fp8`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8`，当时共九种原生 KV。新增四种的 360 项稀疏 Attention 数值检查、页搬运和 140 次真实模型请求全部通过；图片＋ngram 包含 resident C2、CPU C8、固定 MTP3/15 及自适应 MTP3，恢复阶段达到满配并发。冷快照保持禁用。写该记录时，快速 prefill 仍仅允许 INT8/RK8V4 文本；该限制随后被放宽。实测范围、首次夹具失败及重跑见 [新增KV准入补测](other-kv-admission-validation-20261006.md)。

## Current source status — 2026-10-07

上面各段保留当时的证据。当前源码边界是：

- 九种 KV 用于文本和 resident/CPU 图片，C1–8，固定或自适应 MTP，以及带 C2–8 宽度钳制的 ngram。
- 快速 prefill 允许 `int8`、`rk8v4`、`rk4v4`、`rk4v4-e8`、`rk2v4-e8` 的文本和 resident/CPU 图片，C1–8。见 [快速 prefill 准入补测](fast-prefill-admission-validation-20261006.md)。
- ninfer 多卡使用 layer pipeline。5060 Ti 与 5050 Laptop 的验收见 [多卡验收](ninfer-multigpu-validation-20261006.md)。这是容量路径，该轮没有吞吐提升结论。`mixed`/`strict`、overlay Vision、DFlash、并行 prefill 和冷快照仍按 [能力说明](multi-backend.md) 限制。
- 冷快照在启动器、CLI 和 Engine 入口保持关闭。旧 `--lookup-ngram` 保持拒绝。
- 2026-10-07 的 `prepare-backends.py --check` 与 `backends/versions.json` 中的两棵准备后树一致。已发布 ZIP 没有按这棵源码重打。
