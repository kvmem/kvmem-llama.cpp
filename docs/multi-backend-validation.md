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
The three slots share one Host KV archive. Same-query tool continuations retain
the frozen Q and selected view; changed queries use the existing probe/replay
route. Rewinding revokes later checkpoints before shared rows are overwritten.
Cold snapshots use v4 and persist all complete slots and the query attachment;
v3 snapshots are recomputed. The llama.cpp adapter was not changed.

Targeted validation used Windows VS2022/CUDA 13.2, RTX 5060 Ti 16GB and the
Qwen3.8-27B-GSQ-RCO-IQ3_S MTP artifact. Ordinary and MTP4/ngram31 endpoint/rewrite
cases, 32 short-history and 18 long-history cases, single-token initialization,
new queries, edits, cancellation, append/window pressure, concurrency publication
order, and cold save/restart/cancel/corruption fallback passed. StateImage and
Main/MTP transfers were checked byte-for-byte. The independent token-sums oracle
passed 600 cases on the available RTX 5050 Laptop. The final single-token and
rewrite-marker adjustments were retested with the full MTP4 endpoint sequence
and the HTTP/DSH cases below; the earlier full matrix was not rerun on the final
binary.

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

Adaptive MTP, video, concurrent RK8V4, concurrent ngram, image+ngram, live NVMe
and cross-backend KV migration are not supported by the current integration.
See [the capability documentation](multi-backend.md) for constraints.

The earlier llama.cpp MTP cold performance difference (approximately 3% in one
comparison) remains an unresolved performance finding, not a confirmed logic bug.
The original mandatory_trim sink-page logging discrepancy was approved for a
logging-only fix; the selection algorithm was preserved. Detailed provenance and
other legacy findings remain in the experiment workspace's `legacy-findings.md`.
