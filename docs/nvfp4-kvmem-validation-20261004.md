# Dual NVFP4 KVMem qualification — 2026-10-04

The local source/build/model combination passed all **229 planned cases**, three
product CPU unit programs and three independent B32768/R16384 MTP4 service-start
checks. Both Main and MTP use NVFP4-G16 K/V. Ngram and lookup were disabled.
The results qualify this configuration; packaged release and formal performance
comparison remain separate work.

## Configuration

- GPU: RTX 5060 Ti, 16,311 MiB,
  `GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c`, driver 610.62.
- Artifact: `Qwen3.8-27B-GSQ-RCO-IQ3_S-vision-mtp.ninfer`, 12,790,639,360 bytes,
  with text, resident vision and MTP. IQ3S names the mixed GGUF weight recipe.
- Build: Release, CUDA 13.2, sm_120a, MSVC 14.44.35207, eight compilation jobs.
  The existing build has D3D12 support; these 5060 Ti runs used the default CUDA
  memory policy. The pinned NInfer revision remains `6e9189c1089f`.
- Most fixtures use FP32 GDN. Input-fallback fixtures and the large service use
  FP16 GDN. Native K/V code/scale planes and GDN state transfers are checked for
  exact byte preservation. Numerical Attention checks use independent decoding
  of stored codes/scales and an FP64 softmax/Attention oracle.
- Timings are diagnostic measurements, not a performance comparison.

## Completed matrix

| Phase | Cases | Checks |
|---|---:|---|
| Primitives | 7 | Codec, Attention, payload layout, options and MTP graph routing; 90 internal sparse/reordered/cross-page cases, widths 1–5, 13, 53, 63–65 and poisoned future scale bytes |
| History | 46 | MTP0–4 eager/Graph; reuse, edits, query and generated-endpoint rewinds, canonical-input fallback, cancellation, output limits 1–5, reserve crossing, native/window controls and Host admission rejection |
| Sessions | 10 | MTP0–4 eager/Graph; A/B/A, eviction, pressure, clear and cancellation recovery |
| Concurrency | 28 | MTP0–4 × C2/C4 × eager/Graph; queues, cancellation isolation, publication order and Host pressure |
| Pressure control | 1 | INT8 MTP0/C2/eager H96 MiB, compared with NVFP4 H52 MiB |
| Vision | 10 | MTP0–4 eager/Graph; red/blue/multiple images, media identity, long-query replay, admission rejection and cancellation |
| Vision control | 1 | INT8 MTP0/eager; rejected admission preserves the earlier short-image checkpoint |
| Disk | 114 | MTP0–4 eager/Graph restart/cancel/clear/endpoints; corruption, truncation, interrupted save, quota, write failure, INT8-to-NVFP4 mismatch and FP16 canonical-input fallback |
| Quality | 12 | NVFP4 MTP0/4 at 8K/32K/64K/128K targets; INT8 controls at 8K/32K; early primary/backup needles, reuse, fresh recompute and seeded sampling |

The quality fixture calibrates prompts to target−512 within 64 tokens. Its
access-code and replay criteria passed; this is a constructed retrieval fixture,
not a general language-model accuracy benchmark. All four 8K NVFP4/INT8 MTP0/4
profiles repeated sampled tokens exactly within each cold or cached route, and
changing the seed on the cached route changed the output. The additional sampling
diagnostic is separate from the 229 planned cases.

Product unit programs `ninfer_openai_schema_test`, `ninfer_load_report_test` and
`ninfer_resource_manager_test` passed.

## Large service capacity

Three independent service starts passed with IQ3S, NVFP4 K/V, fixed MTP4 Graph,
B32768/R16384, prefill256, logical context131072, C1, H6144 MiB, FP16 GDN and
default CUDA allocation. Observed physical memory peaks were 12,930, 12,932 and
13,008 MiB. The final idle snapshot reported 3,045 MiB free and 259 MiB reserved
by the driver; driver reservation is already excluded from the free value.
GPU Shared Usage counters were retained separately; these measurements do not
establish zero Host residency.

The final process passed a **55,538-token** prompt, exceeding the 49,152-token
physical KV window. It recovered the early pulsar/galaxy fact, spilled and restored
native pages exactly, and reused **55,531 tokens** on repetition with the same
answer. Output limits 1–5 and HTTP disconnect/cancellation recovery passed. MTP
proposals executed; ngram proposal counts remained zero.

The 16K reserve is configured capacity, not a completed 16K output run. Smaller
fixtures crossed their 128-token generation reserve explicitly.

The tested service was retained at `http://127.0.0.1:18201/v1`, PID25928, at the
end of qualification. The separate RTX 5050 Bonsai NVFP4 MTP0 B20480/R10240
service remained healthy on port18202. Bonsai Q8 MTP is too large for that 8GB
card under the default CUDA budget; its earlier small WDDM functional checks
remain distinct from this IQ3S/default-CUDA qualification.

## Investigated test failures

All failed attempts were retained and classified. No Engine, codec or Attention
implementation change was required during the full qualification run.

1. **Host-pressure configuration:** the inherited INT8 H96 MiB quota admits two
   NVFP4 requests, so it cannot test serialization. Native payload/state checks
   passed. The INT8 control serialized correctly; NVFP4 pressure cases passed
   with the equivalent actual H52 MiB quota. The original command/counters remain
   recorded, but its first raw log was accidentally overwritten before the runner
   gained separate attempt filenames.
2. **Vision recovery expectation:** after rejecting an oversized three-image
   prompt, the fixture expected the earlier short-image session to disappear.
   NVFP4 and the INT8 control both correctly restored its frontier289 checkpoint.
   The test now requires retained-history reuse and still checks the image answer,
   resource bounds, Engine availability and absence of active page leaks.
3. **Cold endpoint directory:** the runner reused a populated plain-history disk
   directory for a supposedly cold endpoint case. Correct prefix reuse violated
   its zero-reuse expectation. Each endpoint cold/hit pair now gets a separate
   fresh directory; snapshot and prefix logic were unchanged.
4. **INT8 MTP4 cold/cached sampling:** the original 8K criterion demanded identical
   sampled tokens across cold prefill and cached suffix execution. Two processes
   reproduced a difference from output token2. A diagnostic verified three exact
   cold repetitions and two exact cached repetitions, with different outputs
   between routes. The fixture now enforces the documented same-route determinism
   contract, checks reuse frontiers and changes the seed on that same route.
   Cold/cached equality remains logged: both NVFP4 profiles and INT8 MTP0 matched;
   INT8 MTP4 differed. Native transfers and semantic recovery passed. The precise
   arithmetic cause was not isolated; this is not a confirmed state-restoration
   or NVFP4 defect.
5. **Service corpus length:** the first long-archive construction produced only
   17,066 tokens. Its explicit beyond-window assertion rejected the test input;
   the HTTP request itself succeeded. That directory was retained, the corpus
   was lengthened, and the final 55,538-token request passed. The first two
   successful startup checks were reused rather than repeated.

## Reproduction and retained evidence

The source fixtures and exact parameter syntax are documented in
[`backends/ninfer/tests/README.md`](../backends/ninfer/tests/README.md#kvmem-nvfp4-integration).
Build their affected targets with `cmake --build <build-dir> --parallel 8`.
Select the artifact explicitly and the GPU by UUID; keep ngram/lookup disabled.
The large service arguments are:

```powershell
$env:CUDA_VISIBLE_DEVICES = 'GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c'
./build-backends/ninfer/apps/ninfer-serve.exe C:/models/model-vision-mtp.ninfer `
  --host 127.0.0.1 --port 18201 --model-id iq3s-nvfp4 `
  --max-context 131072 --max-concurrency 1 --prefill-chunk 256 `
  --kv-dtype nvfp4 --spec mtp --draft-tokens 4 `
  --ngram-draft-tokens 0 --lookup-ngram 0 --gdn-state-fp16 `
  --device-profile off --cuda-memory-policy default --cuda-graph-allowance-mib 72 `
  --kvmem-budget 32768 --kvmem-gen-reserve 16384 `
  --kvmem-host-mib 6144 --kvmem-sessions 1 --kvmem-verify-transfers
```

Raw local evidence remains in the integration workspace's sibling directory
`results/nvfp4-5060ti-iq3s-20261004/`: `README.md`, `qualification.json`,
`runs.json`, `findings.md`, runner scripts, unique retry logs, request/response
records, per-process GPU counters and `final-services.json`. The earlier 8GB
investigation is in `results/nvfp4-integration-20261004/`. Large model, cache and
build artifacts are excluded from the repository.

The source is maintained through `backends/patches/ninfer-kvmem.patch` and
`backends/versions.json`; submodule pins are unchanged. Backend preparation and
both whitespace checks passed after export.
