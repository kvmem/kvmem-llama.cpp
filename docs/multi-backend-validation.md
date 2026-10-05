# Multi-backend validation status — 2026-10-03

This is a development candidate, not a final P6–P10 release qualification.
The repository consolidation changes source ownership and build entry points;
it does not establish new GPU performance numbers.

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
`46f37b010410bbd7e6cfa0722447a8e878f2b76e`. Baseline pins and patch hashes are recorded
in `backends/versions.json`.

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

Complete the remaining ngram, RK8V4 disk/restart, long-context MTP and full
HTTP/concurrency/image/package matrices, then run the final controlled performance
comparison on RTX 5060 Ti. Include fixed MTP2/4 in qualification rather than
assuming parameter acceptance proves every combination. Produce the final report
and release package only after these checks pass.

Video, concurrent ngram, live NVMe
and cross-backend KV migration are not supported by the current integration.
See [the capability documentation](multi-backend.md) for constraints.

The earlier llama.cpp MTP cold performance difference (approximately 3% in one
comparison) remains an unresolved performance finding, not a confirmed logic bug.
The original mandatory_trim sink-page logging discrepancy was approved for a
logging-only fix; the selection algorithm was preserved. Detailed provenance and
other legacy findings remain in the experiment workspace's `legacy-findings.md`.

图片＋MTP＋ngram 已放行 INT8/NVFP4；K8V4换行差异待定位，BF16/RK8V4组合未验收。范围与实际复制/replay证据见 [图片ngram补测](vision-ngram-validation-20261005.md)。
