# GDN output norm and gate fusion experiment

2026-09-30, branch `feat/gdn-output-fusion`, KVMem base `cd1d5c7343d8fde9b43ac9d6bd01b18736d20801`, llama.cpp pin `7fe450e19305b828c199d602c23a8337aaa1f03b`.

This change does not include draft vocabulary subsets, batch calibration or the MTP GPU hidden prototype. No GGUF conversion is needed. Following the user's 2026-10-01 request, supported fusion is enabled by default; `KVMEM_GDN_OUT_FUSION=0` preserves the baseline path. The local speed measurements and their limitations are recorded below.

## Implementation

The current CUDA backend already combines RMS norm with its weight multiplication, and SiLU with the final multiplication. This patch replaces those two kernels with one `rms_norm_silu_gate_f32` kernel. The final kernel uses 128 threads for 128 columns. The reference's extra 128 threads only contribute zeros to the reduction; the active values keep the same order. The SiLU function and multiplication order are preserved. Both the initial 256-thread kernel and the final 128-thread kernel passed exact FP32 comparisons.

For Qwen35 dense, enabling the experiment expands the gate projection before the output norm. This puts `RMS_NORM -> MUL`, `SiLU`, and the final `MUL` next to each other in the graph. The CUDA matcher checks dependency edges, intermediate use counts, F32 types, 128-column heads, vector weights, contiguous gates and output, and memory overlap. Input rows may have strides. The allocator retains the norm input and gate until the final output. Unsupported patterns keep their existing kernels.

```powershell
$env:KVMEM_GDN_OUT_FUSION = '1'
```

An unset variable or `1` enables fusion. Use `0` to restore the baseline. Restart the server after changing it. `KVMEM_GDN_OUT_FUSION_TRACE=1` logs actual fusion dispatches for diagnostics; keep it off for timing.

Scope: CUDA F32 output norm with 128-column heads; the graph scheduling change targets Qwen35 dense. Other architectures and CPU/HIP/Vulkan graph execution do not gain this CUDA fusion. The recurrent GDN kernel, MTP policy, cache formats and state rollback are unchanged.

## Validation and reproduction

- `tests/gdn-output-fusion-test.cpp` compares against the existing two-kernel path by retaining the weighted norm as an observed intermediate in the reference graph. It checks exact FP32 output equality for 1/4/64/256/512 tokens, two sequences, strided norm input, and unsupported shapes, gate layout, row weights and observed intermediates.
- Fixed synthetic inputs are retained across repeated graph executions. Intermediate tensors still use the normal scheduler allocation and reuse rules.
- `scripts/test-gdn-output-fusion.py` checks real model output and usage equality, rejected drafts, inactive conversation restore, checkpoint rewind and streaming cancellation. It compares explicit `0` with the unset/default-enabled mode, including actual dispatch checks. Its benchmark uses ABBA server ordering, warmup, rotating prompt order, greedy sampling and explicit cache reset.
- `scripts/prepare-backends.py --backend llamacpp --check` verifies the pinned revision and complete integration tree, including this fusion. The integration patch supersedes the historical patch stack on this development branch.
- CUDA Compute Sanitizer checks the new kernel with `--kernel-name kns=rms_norm_silu_gate_f32`. Use `GGML_CUDA_DISABLE_GRAPHS=1` for this isolated check; the normal backend handles graph-update failure by re-instantiating the graph, and the sanitizer otherwise reports those expected API failures.

```powershell
python scripts/test-gdn-output-fusion.py --phase correctness --output artifacts/gdn-correctness-replay
python scripts/test-gdn-output-fusion.py --phase correctness --mtp-state snapshots --output artifacts/gdn-correctness-snapshots
python scripts/test-gdn-output-fusion.py --phase benchmark --output artifacts/gdn-benchmark
python scripts/prepare-backends.py --backend llamacpp --check
```

Windows builds also include the MSVC dependency-prefix probe from the previous experiment, so Ninja records includes correctly on localized Visual Studio installations. This is a build fix, not a runtime optimization.

## Measurements

Hardware: RTX 5060 Ti 16 GiB (`CUDA0`), Intel Ultra 7 255H, CUDA 13.2, MSVC 14.44. Model: `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, Qwen35 dense, IQ3, MTP draft length 3, replay state. Single GPU, context 4096, batch/ubatch 512, recency budget 4096, greedy sampling, 96 generated tokens per request. Neither shelved feature is enabled.

Four server runs use baseline/fused/fused/baseline ordering. Each server warms up all three cases and records three rotated repeats. Each table entry is the median of six samples per mode. All 36 measured outputs and token usage records match exactly, with no prefix cache reuse.

### Final 128-thread kernel

| Prompt tokens | Prefill baseline/fused (ms) | Prefill speed change | Decode baseline/fused (tok/s) | Decode speed change |
| --- | --- | --- | --- | --- |
| 23 | 347.56 / 340.10 | +2.19% | 46.00 / 46.23 | +0.51% |
| 363 | 707.61 / 690.23 | +2.52% | 39.96 / 40.30 | +0.86% |
| 1511 | 2094.19 / 2076.97 | +0.83% | 43.92 / 43.97 | +0.12% |

The decode measurements overlap considerably: baseline/fused ranges are 45.50-46.57 / 45.67-46.55, 39.72-40.65 / 39.86-41.13, and 43.13-44.33 / 43.62-44.18 tok/s. These sub-1% median differences are not evidence of a substantial speedup. Results are specific to this hardware and model; there is no automatic parameter selection.

At batch 1024 / ubatch 256 in the correctness run, the main model CUDA compute buffer remains 78.90 MiB and the draft compute buffer remains 63.63 MiB in both modes. This does not establish identical memory usage for every input shape.

A follow-up memory check uses the benchmark's exact context 4096, batch/ubatch 512, recency budget 4096 and MTP3 settings, with a 1511-token prompt. Main CUDA compute buffer is 126.53 MiB and draft compute buffer is 126.27 MiB in both modes. Model, KV, recurrent-state and host compute buffer sizes also match. Logs and comparison are in `artifacts/fusion-memory-512/`. This compares allocated tensor buffers, not every byte of CUDA driver/process memory.

The 128-thread choice itself adds no persistent CUDA allocation. Fusion retains the norm input and gate until the final output, which can reduce scratch reuse in other graph layouts. The 512-token synthetic graph uses 12 MiB more scheduler buffer than its observed-intermediate reference. That graph is not the real model, so the synthetic increase should not be added to the real model numbers above. The matcher and overlap checks add small CPU work during graph preparation, direct execution or capture; their isolated cost has not been measured.

Final raw data and logs: `artifacts/gdn-128-replay/`, including `summary.json`, `benchmark.json`, `correctness.json` and server logs. The harness records the patch SHA256 in `config.json`. The synthetic test's microsecond fields include host scheduling and are not isolated kernel timings; they are not used for the table.

### Initial 256-thread kernel

The initial variant retained the reference's full thread count. In the same ABBA protocol, prefill changes were -1.37%, +0.77%, -0.39%, and decode changes were -2.32%, -1.50%, -1.56%. Its raw data remains in `artifacts/gdn-benchmark/` and its source diff in `artifacts/gdn-output-fusion-256.patch`. The two variants were measured in separate runs, so compare each with its own baseline.

### Complete Task 1 comparison

The 128-thread kernel was also tested with the repository's full Task 1: 12000-word background, a 896x896 image, then HTML/SVG generation, with thinking budget 128 and a 512-token output cap. Four fresh servers in OFF/ON/ON/OFF order each ran one warmup and two measured tasks. The four-sample median task request time fell from 30.9429 s to 30.6980 s (-0.79%). Long-text prefill improved 0.87% and code decode improved 0.94%. All measured request payloads, output text, reasoning text and token usage matched.

Peak whole-device VRAM was 15596.91 MiB in both modes. Measured process RSS peaks were 13970.01 / 13979.52 MiB, within the larger variation between server runs. This remains a small local gain. Detailed settings, stage results, memory sampling and raw artifact locations are in [the complete Task 1 report](gdn-task1-comparison.md).

### 256K IQ3 Task 2 stress comparison

On 2026-10-01, one complete Task 2 per mode (OFF then ON) passed all 33 requests, reaching 262058/262144 tokens. Effective prefill was 307.00 / 317.41 tok/s (+3.39%), aggregate tool decode 33.80 / 34.50 tok/s (+2.09%), and final 512-token code decode 29.55 / 30.54 tok/s (+3.37%). All paired inputs, outputs, reasoning and usage matched. Whole-device VRAM peaks were 15688.91 / 15660.01 MiB.

Windows working-set behavior differed substantially between the two runs: runtime RSS peaks were 14136.39 / 17200.92 MiB, whereas peak private commit differed by only 54.54 MiB. The baseline also experienced a large working-set trim. These are single-run observations and do not isolate the entire speed difference to fusion. Detailed context bands, caveats and artifacts are in [the Task 2 report](gdn-task2-comparison.md).

## Completed checks

- 12 synthetic cases: bitwise FP32 equality; seven supported cases dispatched the new kernel, while reversed graph order and other unsupported patterns fell back.
- Real model replay and snapshots output, usage, inactive conversation restoration, checkpoint rewind, streaming cancellation and rejected drafts matched between modes.
- Actual replay fusion dispatches cover prefill and verification widths, including 2, 3, 4, 5 and 256 tokens.
- New kernel Compute Sanitizer memcheck: zero errors with CUDA Graphs disabled for the isolated check.
- All 15 project CTest tests passed.
- Both LF and Windows CRLF full patch replay matched all 47 affected files.

Snapshots state mode logs and output comparison are in `artifacts/gdn-128-snapshots/`.
