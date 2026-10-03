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

## Consolidated checkout checks

- Fresh Windows VS2022 Release portable-core build: passed.
- CTest core/Host/session suites: 14/14 passed.
- Backend launcher argument/capability checks: passed.
- Existing Web UI builder unit tests: 3/3 passed.
- Fresh vendored ninfer CMake configure: passed with CUDA 13.2, SM120a and existing vcpkg dependencies. The portable core resolves inside this same checkout.
- A fresh full CUDA build and model inference from the reorganized checkout were not repeated as part of this repository-only change. Earlier deployed binaries and their validation remain separate evidence.

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
