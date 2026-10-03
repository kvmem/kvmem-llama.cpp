# RTX 5070 Ti Windows guide

[简体中文](rtx-5070ti-windows.md) · **English** · [Binary download and installation](rtx-5070ti-windows-downloads.md)

This package targets **RTX 5070 Ti 16 GB, Windows x64, one model and one concurrent
request**. It combines the CUDA 13.4.2 Native SM120a Release engine with
**NInfer Manager 1.4.1**. Double-click `NInferManager.exe` for everyday use.
Model management and monitoring run in your browser; no separate Python,
CMD or PowerShell launcher is needed.

The default profiles are **XXS 160K and S 128K**, both using the `strict`
memory policy. K means 1024 tokens. This guide covers operation, file placement,
settings and the evidence behind those defaults.

## 1. Start the manager

1. [Download the complete runtime directory from Quark Drive](https://pan.quark.cn/s/28b896c4b0c0)
   and save it to a permanent location, such as `qwen27b`. Extract it first if downloaded as an archive.
   Do not copy the EXE alone.
2. The package includes converted Swift S / XXS `.ninfer` models, using a little over 20GB
   of disk space together; the complete runtime directory currently totals approximately 23.5GB.
   Preserve the complete files in `model/`, including all continuation volumes when adding your own models.
3. Double-click `NInferManager.exe`. It has no main window or terminal;
   look for its icon in the Windows notification area.
4. Right-click the icon and open Manage Models to check the configuration.
   Choose XXS or S from the Start Model submenu.
5. Once ready, open monitoring or copy the API base and model name from the tray.

| Model ready | Stopped or not ready |
|---|---|
| ![Green N tray icon while running](assets/ninfer-tray-running.png) | ![Gray N tray icon while stopped](assets/ninfer-tray-stopped.png) |
| Mint background and dark-green N | Gray; tooltip text distinguishes loading, stopping and errors |

The management site defaults to `http://127.0.0.1:8090`, and the inference
API base to `http://127.0.0.1:18081/v1`. Opening the site from the tray establishes
the local management session. Both profiles expose
**`swift-1.5-qwen3.8-27b`** to clients, with only one loaded at a time.
Start is disabled while loading, running or stopping.

Closing the browser does not stop inference. Stopping the model leaves management
available; exiting the manager stops its model and monitoring site. A Windows Job
owns the engine and its descendants; normal shutdown first sends CTRL_BREAK.
The manager does not adopt unrelated processes merely because a port or PID matches.

### How Windows startup works

“Start with Windows” uses the fixed `NInferManager` value in the current user's
registry key `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run`.
Windows launches the manager **after that user signs in**, without administrator
permission. It is not a system service that runs before sign-in. A computer left
at the sign-in screen will not load the model.

| Independent setting | Effect when enabled |
|---|---|
| Start with Windows | Launch the tray manager after user sign-in |
| Automatically start default model | Load the selected default profile when the manager starts; initially `xxs-160k` |

**Start with Windows is off by default** and must be enabled in the site or tray.
Automatic loading of the default XXS 160K profile remains enabled in the initial
configuration. Existing personal settings keep their values; upgrading does not
turn either setting back on. Effective login startup depends on both registration
and Windows Startup Apps state.

Several copies can exist, but the current user has one startup registration:
**the last copy that explicitly enables or saves startup settings owns the entry**
and replaces its executable path. A normal or `--autostart` launch does not reclaim
registration for an older copy. Disabling startup in an old copy does not delete
an entry already pointing to a different copy. After moving the program, explicitly
enable startup from the new location to update the path.

Windows Startup Apps can independently disable this entry, and the manager respects
that state. An enabled value in a settings file does not prove Windows will launch
it; restoring login startup also requires allowing the entry in Windows.

An earlier Manager 1.3.0 check executed its registered command, automatically loaded
XXS 160K and returned a correct short reply; executing it twice did not reload the
model. That verified the older command path, not the new 1.3.1 registration rules.
**A real reboot/sign-in has not been performed to verify Windows' trigger itself.**

## 2. Where files belong

Version 1.3.1 separates the **program directory** from the **personal data directory**.
The EXE, engine, models and website remain in the program directory (PackageRoot).
Active configuration, logs and runtime state live in the data directory (DataRoot).
An installation under `Program Files` normally needs no writes to its program
directory and no administrator permission for everyday operation.

The default data directory is
`%LOCALAPPDATA%\NInferManager\<installation ID>`. The ID is derived from the
program's location, so each installation path gets a separate subdirectory. On first use at that location,
the manager copies the package's `config/` and fills missing initial configuration
from embedded defaults. Later launches use the active data-directory configuration
without repeatedly overwriting it from the package.

Only failure to create or write the preferred data directory triggers a fallback
to the program directory. Invalid configuration JSON is reported and preserved;
it does not silently cause a directory switch or reset to defaults.

```text
Program directory / PackageRoot, such as qwen27b/ or an installation under Program Files
├─ NInferManager.exe               Tray manager, Release / Windows x64
├─ engine/
│  ├─ ninfer-serve.exe              CUDA 13.4.2 Native SM120a Release engine
│  └─ *.dll                        Runtime dependencies; keep beside the EXE
├─ model/
│  ├─ *.ninfer                     Model entry files
│  └─ …                            Every continuation volume, with original names
├─ config/                         First-use configuration seeds, not the active copy
├─ wwwroot/                        Production management and monitoring website
├─ docs/                           Chinese/English guides, download page and images
├─ LICENSE                         Project license
└─ licenses/                       NVIDIA EULA, dependency licenses and sources

Personal data directory / DataRoot
%LOCALAPPDATA%\NInferManager\<installation ID>/
├─ config/
│  ├─ settings.json                Language, startup, default profile, scan paths, web port
│  ├─ profiles/
│  │  ├─ xxs-160k.json              XXS paths, API model name and launch arguments
│  │  └─ s-128k.json                S paths, API model name and launch arguments
│  ├─ chat_template.jinja           Active conversation template
│  ├─ chat_template.LICENSE         Template license
│  ├─ device-profiles.json          Active GPU calibration
│  ├─ initialized.json             First-use initialization marker
│  └─ history/                     Previous revisions saved before configuration edits
├─ runtime/                        Local process state and short-lived session data
└─ logs/                           Engine output, errors and request metrics
```

Back up `config/` in the data directory: it contains your settings and template.
`runtime/` is transient local state, not configuration to share. Saved parameters
and `config/history/` are both in the data directory. Release downloads exclude personal history,
logs, session credentials and model weights. Old engines, test launchers,
compilers, original GGUFs and raw benchmark logs do not belong in the everyday package.

Saved launch parameters apply to the **next model start**. Writes are atomic and
previous files are archived under `config/history/`. Invalid JSON is reported
and preserved instead of being silently replaced with defaults.

Relative paths follow their purpose: configuration resources such as
`config/chat_template.jinja` and `config/device-profiles.json` resolve against
DataRoot. `engine/ninfer-serve.exe`, `model/...` and model scan directories resolve
against PackageRoot. Absolute paths retain their specified location.

Moving or copying the program to a new path creates a separate data directory,
initialized from that package's `config/`. The two locations subsequently save
settings independently. Moving program files does not automatically carry later
edits from the old data directory; back up its configuration if you need those edits.
Versioned defaults live in `apps/windows-manager/config/` in the source tree and
are also embedded in the manager. Package seeds and active data are not kept in
two-way synchronization.

### Add models and change parameters

The site edits scan directories, model paths, client-facing model IDs, launch
parameters and the default profile. It scans `model/` initially and every
15 seconds, with a manual refresh available. Scanning checks v3 headers,
directories and volume completeness without loading all weights or initializing
the GPU. GGUF, safetensors and individual continuation volumes are not launch entries.
A complete file does not mean that every context setting fits in VRAM.

Question marks beside parameters provide explanations on mouse hover or keyboard
focus. The top-right Simplified Chinese / English switch persists and updates the
tray menu without changing model settings or discarding unsaved edits.

## 3. Models and default settings

The models are `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer` and its
`IQ3_S-mtp.ninfer` counterpart. These are text/MTP v3 containers converted from GGUF files that use multiple tensor quantization types
**without a usable image-understanding component**. Original GGUF files and
conversion reports are not runtime requirements. Another artifact needs its own
capacity and performance check.

| Setting | XXS default | S default |
|---|---|---|
| Profile ID | `xxs-160k` | `s-128k` |
| Context and fixed KV capacity | 163840 / 160K | 131072 / 128K |
| Prefill chunk | 1024 | 256 |
| Concurrency | 1 | 1 |
| KV / GDN state | rk8v4 / FP16 | Same |
| Drafting | Maximum 4 MTP drafts, adaptive MTP enabled, ngram 31, full MTP attention window | Same |
| Graph allowance | 72 MiB | Same |
| `--lm-head-draft` | Off | Off |
| CPU context cache | 6144 MiB, one device snapshot | Same |
| CUDA memory policy | `strict`, equivalent to `strict-64-128` | Same |
| Default output cap | `--default-max-tokens 0` | Same |
| Sampling | temperature 1, top-p 0.95, top-k 20, min-p 0 | Same |
| Penalties and seed | presence/frequency 0, seed 42; neutral repetition penalty 1 | Same |
| Thinking | Enabled, xhigh, preserve thinking | Same |

The table records the current package's saved everyday settings as of 2026-09-30.
Adaptive MTP adjusts draft length at runtime: 4 is the maximum, not a fixed count
for every round. Active profiles are read from `config/profiles/` in the data directory.
The repository's initialization profiles are
`apps/windows-manager/config/profiles/xxs-160k.json` and
`apps/windows-manager/config/profiles/s-128k.json`. They seed the first launch,
do not overwrite existing personal settings, and may differ from settings saved later on this machine.

An output cap of 0 removes the fixed default cap; clients can still supply their
own limit. It does not expand the context window. Input, reasoning and final output
share the window, so leave generation space. xhigh is an effort level, not a fixed
token budget. Editing context in the form also updates explicit KV capacity;
automatic KV can be configured in the advanced JSON editor.

The data directory's `config/chat_template.jinja` handles mid-conversation system/developer messages,
avoiding the original “System message must be at the beginning” error.
Scanning does not overwrite it with the artifact's embedded template.
`config/device-profiles.json` retains the earlier calibration for this 70-SM
RTX 5070 Ti: separate CUDA 13 Native recalibration did not improve overall performance.
It is not universal. Use a separate calibration file for another GPU; engine auto
mode can update the active file if the device does not match.

`NINFER_PREFILL_ALIGN=0` preserves the actual chunk. Inherited
`NINFER_PROMPT_FAST` and `CUDA_LAUNCH_BLOCKING` overrides are cleared.
The device profile already enables the fast prompt kernel; a reported
`fast_prefill_kernel=false` only means no additional CLI force-enable flag was supplied.

### 3.1 Using the parameter editor

Common controls are visible directly: API/context, GPU memory/cache, speculative decoding,
vision/media, sampling/reasoning, and templates/device routes. Advanced options are collapsed
initially and grouped by device/execution, precision/kernels, positional encoding, context
cache, disk cache, post-thinking sampling, API behavior, networking/queues and logging.
Complete JSON and environment variables have a separate editor. Collapsing a group never
disables or removes its saved parameters. Save changes to apply them on the next model start.

The question mark beside each parameter explains its purpose, effects, default and dependencies.
A blank field usually omits the option and uses engine/model defaults. The manager supplies
163840 (160K) for an omitted context limit and 18081 for an omitted API port. A placeholder is not an active value.
For example, MTP and copy drafts need the matching backend, while cross-request copy archives
also need a RAM budget. The current `--lookup-ngram` execution path is used by MTP;
the setting can be saved without MTP but does not take effect then.

Vision requires a model containing visual components. The Swift XXS/S artifacts converted here
contain only text/MTP; enabling a switch does not add image understanding. Vision uses the
`default` memory policy and cannot be combined with this version's `mixed/strict` policies.
CPU vision defaults to 256 merged media tokens when no explicit limit is set. Multi-GPU pipeline
options are Linux-only and marked unavailable in the Windows editor. D3D12 and DirectStorage
also require an engine built with their respective support.

When an API key or separate stats port is configured, the manager uses that authentication and
port for monitoring. A custom request-log path is supported; leaving it blank keeps the manager's
generated path. Relative output paths are resolved under the personal data directory.

## 4. Choose a memory policy

Use one `--cuda-memory-policy` for everyday device-memory behavior.
“Shared system memory” means Windows borrowing system RAM to back CUDA device
data when VRAM is insufficient. It is separate from the engine's explicit CPU
context cache.

| Value | UI name | Meaning |
|---|---|---|
| `default` | Default policy | Plan against CUDA-reported free memory, without extra strict checks; dedicated VRAM residency is not guaranteed. |
| `mixed` | Allow borrowing system RAM | Try actual allocations beyond CUDA free and accept Shared growth; potentially more capacity, but potentially much slower. |
| `strict` | Dedicated VRAM only | Check device-data residency; verify 64 MiB spare capacity and use a 128 MiB probe step by default. |
| `strict-128-64` | Custom strict mode | Verify 128 MiB spare capacity with a 64 MiB probe step. Both numbers are MiB. |

`strict` equals `strict-64-128`. The form expands reserve and probe-step fields
only for strict mode. Reserve may be zero; the step is 1–16384 MiB.
Strict automatically selects the required Hybrid context cache.
`--host-cache-mib` independently controls CPU context-cache capacity.

### Mixed capacity is bounded

Mixed permits driver-managed borrowing; it does not force data into Shared.
The driver still decides physical placement:

- Explicit KV capacity is attempted as requested; allocation failure is reported.
- With `--kv-capacity auto`, the bound is one `--max-context` window per
  concurrent request, rounded up to KV pages. It does not keep growing idle
  prefix capacity or allocate until system RAM is exhausted.
- Mixed does not use strict reserve or probe settings. `--kv-headroom-mib`
  applies only to default auto sizing; strict auto uses the policy's first number.

Mixed and strict currently support Windows single-GPU text generation and exclude
`--wddm-evictable-budget`. Strict also conflicts with explicitly choosing the
original cache or disabling prefix reuse.

### What strict checks

Strict allows controlled real CUDA allocations instead of treating CUDA free as
the only hard ceiling. Temporary probes are released, and the final weight and
KV arenas remain contiguous. **The final KV pool is not split into many small
allocations.** Spare capacity is temporarily allocated and released at startup,
not permanently reserved.

The CPU cache is touched and transfer-warmed before a Shared baseline is established.
Checks then cover Dedicated/Shared counters, GPU accessibility, and the combined
weight/KV/Graph layout. Zero Shared growth does not mean the process has an absolute
Shared value of zero: the explicitly configured CPU cache can contribute to the baseline.

Probe OOM has fallback, and auto KV candidates have bounded retries; an explicit
context is not silently reduced. Candidates exceeding the Shared baseline are
rejected. Runtime residency failure makes health and new requests return 503.
The policy cannot permanently pin Windows/WDDM placement. One probe verified
640 MiB beyond the initial CUDA free report, then rejected the next block for
Shared growth; that does not promise an extra 640 MiB on every run.

Task Manager, NVML/`nvidia-smi` and CUDA free use different counters and sampling
times. Whole-card free memory is not a promise of an equally large contiguous CUDA
allocation. Observed Graph size is the Dedicated delta during preparation, possibly
including lazy allocations; it is not an exact Graph allocation ledger.

## 5. Built-in monitoring

No `monitor.py` is needed. The manager samples every two seconds and keeps up to
six hours of in-memory history for the current model run. Browser refresh or closure
does not clear it; manager exit or a new model run resets it.

| Metric | What it helps explain |
|---|---|
| Prefill, decode, TTFT | Prompt processing, generation speed and first-output delay |
| MTP acceptance, cache hits and reuse paths | How much drafting is accepted and whether earlier context is reused |
| KV occupancy, transfers, scheduling and pressure | Context-storage and transfer bottlenecks |
| GPU utilization, power, temperature and memory | GPU activity and capacity pressure |
| Successful, failed and rejected requests | Cumulative results, rather than the length of a recent-request list |

Prefill and Decode are sampled every two seconds and handled independently.
A zero token delta keeps that metric's last valid rate in both its card and chart;
zero-delta intervals are excluded from the average. A positive delta updates the rate
using non-zero intervals within the last ten seconds: their token deltas divided by
their combined duration. Startup uses the valid intervals collected so far. After
more than ten seconds of idle sampling, resumed calculations use only new intervals
inside the window. Missing data, stops and counter resets clear the held value;
no rate appears before a valid interval is available. Completed-request rates still use
each request's actual duration, with separate labels. Latency, throughput, MTP and cache-hit
aggregates have a last-hour window. Logs are read incrementally with bounded tails,
and catch-up is indicated for large existing logs. Unsupported counters remain
unknown instead of becoming zero. NVML whole-card free, CUDA free, and process
Shared baseline/growth are displayed separately.

## 6. Current package performance: 2026-09-30

The manager is version 1.4.1; the engine is CUDA 13.4.2 / Native SM120a Release, with D3D12 residency disabled.

**Test environment:** RTX 5070 Ti 16 GB (16303 MiB total reported by NVML), Ryzen 7 9800X3D, approximately 32 GB system RAM, Windows 11 build 26200, NVIDIA driver 617.14. Other applications' GPU allocations were released before this fresh run; NVML reported 4 MiB used and 15992 MiB free.

### Setup and how to read the results

- The four strict configurations test only inputs near their own context limits: S/XXS 64K, default S 128K, and default XXS 160K. S mixed 196K tests only approximately 1K input. K means 1024 tokens; context and fixed KV capacity are equal, so 196K = 200704 tokens. Context capacity is not the actual input length; the table lists both.
- Each configuration starts once, then receives a warmup request for an SVG of a penguin riding a bicycle, capped at 1024 output tokens and excluded from results. xhigh thinking stays enabled, so the cap includes reasoning and does not guarantee complete SVG markup. Each measured workload then runs three times with seeds 42/142/242 and a 512-token output cap. The engine is not reloaded between repetitions.
- Actual settings: one concurrent request, rk8v4 / FP16 GDN state, maximum 4 MTP drafts with adaptive MTP, ngram 31, 72 MiB graph allowance, 6144 MiB Host cache, one device snapshot, xhigh with preserved thinking, and lm-head-draft off. Sampling: temperature 1, top-p 0.95, top-k 20, min-p 0, presence/frequency penalties 0. S uses chunk 256 and XXS chunk 1024; the additional S mixed configuration uses chunk 1024. Mixed explicitly selects the same Hybrid context cache used by strict.
- Inputs contain synthetic English observation records, retrieval keys, and a final analysis instruction. Every request has a different prefix. Eligibility requires zero cached input, root reuse, a completed SSE stream, and matching HTTP and exactly associated engine request_done token counts. This measures capacity and speed, not coding ability or answer quality.
- **Best of three means the complete request with the shortest total wall time.** Throughput, TTFT and memory headroom in a row come from that same request, not independent per-column maxima. The second table shows all three ranges. A best result requires three valid repetitions; otherwise the row is incomplete.
- Prefill = uncached input tokens / engine prompt time. Decode = (output tokens − 1) / engine predicted time. Client TTFT runs from request dispatch to the first nonempty reasoning, content, or tool stream event. It includes prefill and is not time to the final answer.
- GPU headroom is the **lowest sampled whole-GPU NVML free value during the HTTP request**, sampled approximately every 0.5 seconds. Process WDDM Dedicated/Shared is sampled approximately every second; invalid counters are excluded. Brief peaks may be missed. NVML free is not guaranteed CUDA allocation capacity and is not calculated as total minus used.
- Each mixed HTTP request has a 300-second limit covering prefill and generation; strict has a 600-second protective deadline. These limits also apply to warmup. The first timeout stops the remaining requests for that configuration and unloads the engine. Incomplete requests do not yield an inferred full TTFT, decode result, or best-of-three score.

This run produced **15 valid measured requests, with all three repetitions valid for 5/5 workloads**; 5/5 SVG warmups completed. Among valid measured requests, 15/15 produced reasoning only, without final content or tool calls; 15/15 have reasoning-token counts. Every valid measured request generated 512 tokens, so these results do not measure delivery of a 512-token final answer.

[All measured requests](assets/rtx5070ti-benchmark-20260930-reduced-all.csv) · [Selected complete best requests](assets/rtx5070ti-benchmark-20260930-reduced-best.csv) · [Three-run ranges and medians](assets/rtx5070ti-benchmark-20260930-reduced-ranges.csv). The CSVs also include server TTFT, power, CUDA residency snapshots, WDDM counters, combined MTP/ngram speculative acceptance, and output classification.

### Best complete request

| Configuration / policy | Chunk | Actual input / output | Valid runs; selected repeat | Prefill tok/s | Decode tok/s | Client TTFT s | Total s | GPU headroom MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S 64K / strict | 256 | 62,439 / 512 | 3/3; 3 | 1,666.2 | 105.59 | 37.557 | 42.459 | 1,911 |
| S 128K / strict | 256 | 127,970 / 512 | 3/3; 2 | 1,356.3 | 97.82 | 94.525 | 100.018 | 177 |
| XXS 64K / strict | 1024 | 62,439 / 512 | 3/3; 1 | 1,695.4 | 99.36 | 36.918 | 42.064 | 2,761 |
| XXS 160K / strict | 1024 | 160,726 / 512 | 3/3; 1 | 1,340.8 | 91.08 | 120.130 | 125.751 | 161 |
| S mixed 196K / 1K | 1024 | 993 / 512 | 3/3; 2 | 54.5 | 4.65 | 18.244 | 128.166 | 27 |

S 64K and XXS 64K use equal-length input and can be compared as complete configurations. Other strict rows use different input lengths, so their TTFT is not a direct model-speed ranking. This reduced matrix has no matched input lengths between mixed and strict; it does not establish a controlled slowdown ratio or percentage.

### Ranges across the three measurements

Ranges show the minimum–maximum across the three repetitions. This small sample describes observed variation, not statistical significance or universal performance.

| Configuration / input | Prefill tok/s | Decode tok/s | Client TTFT s | Total s | GPU headroom MiB |
|---|---:|---:|---:|---:|---:|
| S 64K / 61k | 1,666.25–1,667.82 | 96.09–105.59 | 37.52–37.56 | 42.46–42.87 | 1,911.00–1,911.00 |
| S 128K / near | 1,356.31–1,356.77 | 91.88–97.82 | 94.48–94.53 | 100.02–100.68 | 177.00–177.00 |
| XXS 64K / 61k | 1,690.51–1,695.37 | 97.60–99.36 | 36.92–37.04 | 42.06–42.34 | 2,761.00–2,761.00 |
| XXS 160K / near | 1,339.69–1,340.77 | 82.81–91.08 | 120.13–120.20 | 125.75–126.89 | 161.00–161.00 |
| S mixed 196K / 1k | 54.36–54.54 | 4.46–4.65 | 18.23–18.27 | 128.17–132.68 | 27.00–27.00 |

### Memory headroom and the purpose of mixed

**This S mixed 196K configuration deliberately attempts a capacity beyond dedicated VRAM and allows Windows to back device data with system RAM. It trades speed for capacity; it is not an acceleration mode.** Moving device data between GPU and system memory can sharply reduce both prefill and decode throughput. The driver still controls whether and how data is placed in Shared; mixed is not an API that forces Shared placement.

The following memory table aggregates all valid measured requests for each configuration, rather than only the selected best request.

| Configuration | Min NVML free MiB | Min CUDA free check snapshot MiB | Peak process Dedicated MiB | Peak process Shared MiB | Strict Shared baseline MiB | Max delta from strict baseline MiB |
|---|---:|---:|---:|---:|---:|---:|
| S 64K | 1,911 | 1,149 | 14,094.3 | 6,538.0 | 6,538.0 | 0.0 |
| S 128K | 177 | 0 | 15,828.3 | 6,538.0 | 6,538.0 | 0.0 |
| XXS 64K | 2,761 | 1,999 | 13,244.3 | 6,538.0 | 6,538.0 | 0.0 |
| XXS 160K | 161 | 0 | 15,844.3 | 6,538.0 | 6,538.0 | 0.0 |
| S mixed 196K | 27 | — | 15,982.5 | 8,660.0 | — | — |

The CUDA column contains free-memory snapshots saved by the engine residency checks, not continuous once-per-second direct CUDA polling. For S 64K / XXS 64K, minimum NVML free was 1911 / 2761 MiB, while minimum CUDA check snapshots were 1149 / 1999 MiB. S 128K / XXS 160K still showed 177 / 161 MiB NVML free, but their saved CUDA check snapshots both reached 0 MiB. That NVML headroom is not evidence that more context or other device allocations can be added freely. The interfaces use different accounting and observation times.

Across the 12 valid measured requests in the four strict configurations, both the Shared baseline and sampled peak were 6538 MiB, with a maximum baseline-relative increase of 0 MiB. This means no additional Shared growth was observed in this run; it does not mean Windows permanently pins device data in dedicated VRAM.

The explicitly configured 6144 MiB CPU context cache also contributes to Shared. The entire Shared total must not be attributed to spilled CUDA device data. Strict baseline/delta fields distinguish that normal Host cache. Mixed disables strict residency checks, so its runtime CUDA residency snapshots and strict-baseline fields are blank. Mixed Shared growth alone cannot precisely separate active Host cache from driver-managed spill.

A smaller fixed KV capacity leaves more room for the desktop and other GPU applications; default S 128K and XXS 160K prioritize context capacity. A larger fixed KV pool does not automatically shrink for a short request. Use the mixed configuration only when the extra capacity is needed and the wait is acceptable.

Strict configurations with three valid near-limit requests in this run: S 64K, S 128K, XXS 64K, XXS 160K. Capacity validation is limited to these configurations and this desktop load; incomplete configurations are not counted as successes. Mixed tests only short 1K input; completing it does not validate performance near 196K input.

## 7. Building and supported scope

### Runtime requirements

Validated: **Windows x64, RTX 5070 Ti, driver 617.14**. This repository's prebuilt
engine and accompanying converted `.ninfer` model artifacts support **RTX 50-series GPUs only**;
the engine retains only SM120a machine code. Other RTX 50-series models are expected
to be architecture-compatible but have not been individually tested.

RTX 30/40-series GPUs with **at least 16GB of dedicated VRAM** may also be able to run
the project, but this branch has not tested them. They cannot use this RTX 50-series
package directly. Build the engine from source for the target architecture (`86` for
RTX 30-series, `89` for RTX 40-series), then select an appropriate conversion recipe
and prepare the model. Context capacity and speed need validation on the actual GPU.
You can give this repository, its build instructions, and the
[model conversion guide](rtx-5070ti-windows-downloads.md) to an AI assistant to help
prepare the environment, compile the engine, convert GGUF to `.ninfer`, and configure
startup. The commands below target RTX 50-series GPUs and must be adapted for other architectures.

R615 or newer is the deployment recommendation for CUDA 13.4; this package has not
validated compatibility limits on older drivers. References:
[NVIDIA GPU list](https://developer.nvidia.com/cuda/gpus),
[Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/),
[CUDA driver requirements](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-driver).
With dependencies present, runtime does not require the full CUDA Toolkit,
a compiler, Python, Node.js or a global .NET installation.
Recheck context, Graph allowance, reserve and calibration on another GPU;
parameter changes normally need no recompilation. 5070 Ti measurements are not
measurements for the 5090 or 3090.

### 7.1 Start with the complete source

Use the [`main` branch of Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco](https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco/tree/main).
It contains this guide, Manager 1.3.1 and the strict/mixed memory policies. The upstream default branch or an older commit may not include these changes.
For a first checkout, run this command in PowerShell; the destination directory should not already exist:

```powershell
git clone --branch main --single-branch https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco.git C:\src\ninfer-all
```

If you already have the complete source from this branch, reuse that directory and adjust the paths below.
The build steps use `C:\src\ninfer-all` and first check that required files exist.

Prepare these development tools; they do not belong in the final runtime directory:

| Tool | Requirement for this route |
|---|---|
| Git and PowerShell | Fetch dependencies and run the commands below |
| Visual Studio 2022 Build Tools | Desktop development with C++, x64 MSVC v143 **14.44.35207**, Windows SDK, C++ CMake/Ninja tools |
| CUDA Toolkit | **13.4.2**, including nvcc 13.4.92, nvprune and cuBLAS development/runtime files |
| CMake / Ninja | Local CMake **4.4.3**; use the same version for a new environment. The project's 3.28 minimum does not establish older-version validation with CUDA 13.4/120a. Ninja on PATH |
| Node.js | **22.12 or newer**, matching Vite's current lockfile requirement |
| .NET SDK | **10.x**, for the self-contained Windows x64 manager |
| Python | 3.11 for conversion only; install CPU PyTorch, NumPy and download tools as described in the conversion guide |

Install CUDA from the [official 13.4.2 archive](https://developer.nvidia.com/cuda-13-4-2-download-archive).
The commands assume common VS Build Tools and CUDA installation paths; adjust them
if necessary. Adding only `cl.exe` to PATH is insufficient: headers, libraries and
the Windows SDK also need the developer environment.

Run subsequent steps in **the same PowerShell session**. `$packageRoot` is a new
output directory, not the Downloads installation currently in use. Four build jobs
were used on this 32 GB RAM machine; adjust `$jobs` to available memory.

```powershell
$ErrorActionPreference = 'Stop'
$ninferRepo = 'C:\src\ninfer-all'
$buildRoot = Join-Path $ninferRepo 'build-5070ti-native'
$vcpkgRoot = 'C:\src\vcpkg'
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$publishRoot = Join-Path $ninferRepo 'dist\windows-manager'
$packageRoot = 'C:\build\ninfer-package\qwen27b'
$jobs = 4

Set-Location $ninferRepo
foreach ($required in @(
  'apps/windows-manager/NInfer.Manager.csproj',
  'apps/windows-manager/config/profiles/xxs-160k.json',
  'src/product/cuda_memory_options.h',
  'tools/convert/__main__.py'
)) {
  if (-not (Test-Path -LiteralPath (Join-Path $ninferRepo $required))) {
    throw "Incomplete source checkout: $required"
  }
}

$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" -vcvars_ver=14.44 >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
$clPath = (Get-Command cl.exe -ErrorAction Stop).Source
$env:CUDA_PATH = $cudaRoot
$env:PATH = "$cudaRoot\bin\x64;$cudaRoot\bin;$env:PATH"
(Get-Item -LiteralPath $clPath).VersionInfo.FileVersion
& "$cudaRoot\bin\nvcc.exe" --version
cmake --version
ninja --version
```

Confirm MSVC 14.44 (the compiler file version normally reads 19.44) and nvcc 13.4.92.
Do not reuse the older 3090 build tree. Incrementally build an already-correct
5070 Ti tree rather than reconfiguring it.

### 7.2 Acquire dependencies and build CUDA operators

This standard route uses the repository's `vcpkg.json`, with its pinned baseline
and curl, FFmpeg/zlib and pkgconf dependencies. On first configure, the vcpkg
CMake toolchain downloads and builds them; this can take substantial time.
It avoids manually assembling FFmpeg include/library paths.
[Microsoft's manifest integration documentation](https://learn.microsoft.com/en-us/vcpkg/users/buildsystems/cmake-integration)
describes this automatic installation.

Use a new build directory. Do not change toolchains inside a tree configured with
the alternative prebuilt-dependency bridge. Dependencies for this route appear
under `$buildRoot\vcpkg_installed\x64-windows`.

```powershell
if (-not (Test-Path -LiteralPath $vcpkgRoot)) {
  git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
  if ($LASTEXITCODE -ne 0) { throw 'vcpkg clone failed' }
}
& "$vcpkgRoot\bootstrap-vcpkg.bat" -disableMetrics
if ($LASTEXITCODE -ne 0) { throw 'vcpkg bootstrap failed' }

cmake -S $ninferRepo -B $buildRoot -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$vcpkgRoot/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_MANIFEST_MODE=ON `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DNINFER_SM120_NATIVE=ON -DNINFER_BUILD_APPS=ON `
  -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF `
  -DNINFER_DIRECTSTORAGE=OFF -DNINFER_D3D12_RESIDENCY=OFF `
  "-DCUDAToolkit_ROOT=$cudaRoot" `
  "-DCMAKE_CUDA_COMPILER=$cudaRoot/bin/nvcc.exe" `
  "-DCMAKE_CUDA_HOST_COMPILER=$clPath" `
  "-DCMAKE_C_COMPILER=$clPath" "-DCMAKE_CXX_COMPILER=$clPath"
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
cmake --build $buildRoot --target ninfer_ops -j $jobs
if ($LASTEXITCODE -ne 0) { throw 'CUDA operators build failed' }
```

The configuration is Release, `120a`, Native ON, D3D12 residency OFF and
DirectStorage OFF. Native enables PDL, and Windows uses staged TMA descriptors;
separate compatibility-route options being OFF does not disable these Native paths.
No extra unmeasured cuBLAS/A8 performance options are added.

### 7.3 Prune the archive and link the engine

The measured CUDA archive included redundant PTX and could exceed Windows' 2 GiB PE
limit at final linking. Back it up, use the same toolkit's `nvprune.exe` to retain
SM120a SASS, and replace the link input only after success. This remains a Release
build. Do not apply this single-architecture step to multi-architecture packages.

```powershell
$opsArchive = Join-Path $buildRoot 'src\ops\ninfer_ops.lib'
$prunedArchive = Join-Path $buildRoot 'src\ops\ninfer_ops.native.lib'
$backupRoot = Join-Path $buildRoot 'archive-backup'
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$backupName = 'ninfer_ops-' + (Get-Date -Format 'yyyyMMdd-HHmmssfff') + '.with-ptx.lib'
Copy-Item -LiteralPath $opsArchive -Destination (Join-Path $backupRoot $backupName)

& "$cudaRoot\bin\nvprune.exe" -arch sm_120a $opsArchive -o $prunedArchive
if ($LASTEXITCODE -ne 0) { throw 'nvprune failed; original archive was not replaced' }
Copy-Item -LiteralPath $prunedArchive -Destination $opsArchive -Force

cmake --build $buildRoot --target ninfer-serve -j $jobs
if ($LASTEXITCODE -ne 0) { throw 'Engine link failed' }
```

The executable is `$buildRoot\apps\ninfer-serve.exe`. After changing CUDA operators,
build `ninfer_ops`, prune it again, then link; do not skip pruning after an archive
rebuild. Many architecture-specific kernels keep the executable large even in
Release mode. File size alone does not indicate Debug/Release, and these end-to-end
checks do not numerically qualify every NVFP4/MoE route.

### 7.4 Convert GSQ/RCO models

Follow the [download and GSQ GGUF conversion tutorial](rtx-5070ti-windows-downloads.md)
for Swift or ISTA GGUFs and the matching metadata/tokenizer, using `tools.convert`
from the same source. That tutorial includes the Python environment, downloads,
`qwen3_8_27b_gguf`, `text,mtp`, CPU conversion and `--proposal` commands.

For both defaults in this guide, convert Swift `IQ3_XXS` and `IQ3_S` separately.
Keep the documented filenames under the source tree's `converted-models/`.
The original GGUFs, metadata and conversion reports are build inputs, not runtime
package contents. The converter is not shipped with the application. Conversion
does not compile the CUDA engine or requantize the selected GGUF.

### 7.5 Build the website and tray manager

Install npm dependencies from the lockfile and build the website before publishing
C#. Vite writes to `apps/windows-manager/wwwroot/`, which the C# project copies to
the publish output. Reversing the order can publish a stale or missing website.

```powershell
node --version
dotnet --list-sdks
Push-Location (Join-Path $ninferRepo 'apps\windows-manager\web')
try {
  npm.cmd ci
  if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
  npm.cmd run build
  if ($LASTEXITCODE -ne 0) { throw 'Website build failed' }
} finally { Pop-Location }

dotnet publish (Join-Path $ninferRepo 'apps\windows-manager\NInfer.Manager.csproj') `
  -c Release -r win-x64 --self-contained true `
  -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true `
  -o $publishRoot
if ($LASTEXITCODE -ne 0) { throw 'Manager publish failed' }
```

The EXE in `$publishRoot` includes the .NET runtime. Conversion Python, build-time
Node.js and the .NET SDK are not runtime-package requirements.
Manager-only changes also require no CUDA rebuild.

### 7.6 Assemble a fresh runtime directory

Copy the full vcpkg **Release bin** DLL set, including FFmpeg/curl transitive
dependencies, then add cuBLAS/Lt from the same CUDA toolkit and the x64 VC runtime.
Do not rely only on the few DLLs that happen to appear beside the CMake executable
or mix identically named files from unrelated FFmpeg/CUDA installations.
This Windows route statically links cudart, but the package below also carries
the toolkit runtime for dependent libraries. The VC files come from the installed
MSVC Redist directory; see the
[Microsoft VC Redistributable requirements](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

**Source seed profiles and on-disk runtime profiles use different outer formats.**
The embedded seeds at `apps/windows-manager/config/profiles/*.json` have `id`,
`modelPath` and other fields directly at the top level. Disk profiles require a
`schemaVersion / savedAt / profile` wrapper. The script creates that wrapper;
do not recursively copy the whole source config directory as active configuration.
Alternatively, omit package config completely and let first launch initialize
everything from embedded defaults.

```powershell
if (Test-Path -LiteralPath $packageRoot) {
  throw 'Choose a new, empty packageRoot; do not overwrite a running installation'
}
foreach ($directory in @('', 'engine', 'model', 'config\profiles', 'docs\assets', 'licenses')) {
  New-Item -ItemType Directory -Path (Join-Path $packageRoot $directory) -Force | Out-Null
}
Copy-Item -LiteralPath (Join-Path $publishRoot 'NInferManager.exe') -Destination $packageRoot
Copy-Item -LiteralPath (Join-Path $publishRoot 'wwwroot') -Destination $packageRoot -Recurse
$engineDir = Join-Path $packageRoot 'engine'
Copy-Item -LiteralPath (Join-Path $buildRoot 'apps\ninfer-serve.exe') -Destination $engineDir

$dependencyRoot = Join-Path $buildRoot 'vcpkg_installed\x64-windows'
Get-ChildItem -LiteralPath (Join-Path $dependencyRoot 'bin') -File -Filter '*.dll' |
  Copy-Item -Destination $engineDir

foreach ($dll in @('cublas64_13.dll', 'cublasLt64_13.dll', 'cudart64_13.dll')) {
  $source = Join-Path $cudaRoot "bin\$dll"
  if (-not (Test-Path -LiteralPath $source)) { $source = Join-Path $cudaRoot "bin\x64\$dll" }
  if (-not (Test-Path -LiteralPath $source)) { throw "Missing CUDA DLL: $dll" }
  Copy-Item -LiteralPath $source -Destination $engineDir
}
$crtRoot = Join-Path $env:VCToolsRedistDir 'x64\Microsoft.VC143.CRT'
if (-not (Test-Path -LiteralPath $crtRoot)) {
  throw 'Locate the installed MSVC x64 redistributable directory and set crtRoot'
}
Get-ChildItem -LiteralPath $crtRoot -File -Filter '*.dll' | Copy-Item -Destination $engineDir

$seedRoot = Join-Path $ninferRepo 'apps\windows-manager\config'
Get-ChildItem -LiteralPath $seedRoot -File | Copy-Item -Destination (Join-Path $packageRoot 'config')
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $seedRoot 'profiles') -File -Filter '*.json') {
  $profile = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
  $document = [ordered]@{
    schemaVersion = 1
    savedAt = [DateTimeOffset]::UtcNow.ToString('O')
    profile = $profile
  }
  $target = Join-Path $packageRoot "config\profiles\$($file.Name)"
  [IO.File]::WriteAllText($target, ($document | ConvertTo-Json -Depth 32), [Text.UTF8Encoding]::new($false))
}

foreach ($name in @('rtx-5070ti-windows.md', 'rtx-5070ti-windows.en.md', 'rtx-5070ti-windows-downloads.md')) {
  Copy-Item -LiteralPath (Join-Path $ninferRepo "docs\$name") -Destination (Join-Path $packageRoot 'docs')
}
foreach ($name in @('ninfer-tray-running.png', 'ninfer-tray-stopped.png', 'rtx5070ti-benchmark-20260929-all.csv', 'rtx5070ti-xxs-chunks-20260929.csv')) {
  Copy-Item -LiteralPath (Join-Path $ninferRepo "docs\assets\$name") -Destination (Join-Path $packageRoot 'docs\assets')
}
Copy-Item -LiteralPath (Join-Path $ninferRepo 'LICENSE') -Destination $packageRoot
foreach ($port in Get-ChildItem -LiteralPath (Join-Path $dependencyRoot 'share') -Directory) {
  $copyright = Join-Path $port.FullName 'copyright'
  if (Test-Path -LiteralPath $copyright) {
    $licenseDir = Join-Path $packageRoot "licenses\$($port.Name)"
    New-Item -ItemType Directory -Path $licenseDir -Force | Out-Null
    Copy-Item -LiteralPath $copyright -Destination $licenseDir
  }
}
Get-ChildItem -LiteralPath $cudaRoot -File |
  Where-Object { $_.Name -match 'LICENSE|EULA' } |
  Copy-Item -Destination (Join-Path $packageRoot 'licenses')
```

Now copy the converted models. Remove the S entry from `$entryNames` if you only
need XXS. For different filenames or ISTA weights, select the actual model in the
website and save a separate profile.
The commands use each `.conversion.json` report's `files` list for the entry and
all volumes, without guessing volume names or overwriting existing model files.
Keep reports and outputs at their conversion location until copying is complete;
reports do not need to be included in the runtime package.

```powershell
$convertedModelRoot = Join-Path $ninferRepo 'converted-models'
$entryNames = @(
  'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.ninfer',
  'Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer'
)
$modelDir = Join-Path $packageRoot 'model'
$filesToCopy = @(foreach ($entryName in $entryNames) {
  $reportPath = Join-Path $convertedModelRoot ($entryName + '.conversion.json')
  $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
  if (-not $report.files) { throw "Conversion report has no files: $reportPath" }
  foreach ($file in $report.files) {
    [pscustomobject]@{
      source = $file.path
      target = Join-Path $modelDir ([IO.Path]::GetFileName($file.path))
    }
  }
})
foreach ($file in $filesToCopy) {
  if (-not (Test-Path -LiteralPath $file.source -PathType Leaf)) { throw "Missing model volume: $($file.source)" }
  if (Test-Path -LiteralPath $file.target) { throw "Model file already exists: $($file.target)" }
}
foreach ($file in $filesToCopy) {
  [IO.File]::Copy($file.source, $file.target, $false)
}
```

This produces the PackageRoot structure in section 2: one manager, one engine
directory, models, website, initial configuration and documentation. It does not
include the converter, build cache, test launchers or another old engine.
When redistributing software, retain the applicable project, CUDA, VC runtime and
dependency licenses.

### 7.7 Check the standalone directory and launch

Temporarily remove development tools from PATH and run engine `--help`, checking
that it does not secretly load DLLs from a development directory. This establishes
basic loading, not qualification of every inference path. Error 0xC0000135 usually
means a missing direct or transitive DLL; fix `engine/` instead of hiding the
problem behind the developer PATH.

Then launch the manager with `--no-autostart --open` to inspect configuration before
manually loading a model.

```powershell
$savedPath = $env:PATH
try {
  $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
  & (Join-Path $packageRoot 'engine\ninfer-serve.exe') --help
  if ($LASTEXITCODE -ne 0) { throw 'Packaged engine failed to start; inspect runtime dependencies' }
} finally { $env:PATH = $savedPath }

Start-Process -FilePath (Join-Path $packageRoot 'NInferManager.exe') `
  -ArgumentList @('--root', "`"$packageRoot`"", '--no-autostart', '--open') `
  -WindowStyle Hidden
```

The manager initializes AppData for this installation location. Website edits go
to DataRoot, not package seeds; changing package config does not overwrite an
existing DataRoot. Templates and device profiles resolve from DataRoot, while
engine/models resolve from PackageRoot. Moving the complete verified package to
its final location, such as Downloads/qwen27b, produces another installation ID
and first-use initialization as described in section 2.

Confirm model readiness in the site, send a short request, inspect logs and memory,
then stop the model. Management should remain available after stopping inference;
exiting the manager closes monitoring as well. Login startup requires explicit
enabling and is not enabled by this test command.

**Audit scope:** these instructions were checked against repository CMake,
the vcpkg manifest, the successful local build scripts, manager resource formats
and completed deployment checks. The measured engine used MSVC 14.44, CUDA 13.4.2
and a prebuilt FFmpeg/curl bridge. This audit did not reinstall all of vcpkg/CUDA/VS
on a blank Windows system or repeat the complete build from that environment.
Different dependency versions are not claimed to produce identical binaries.
The standard vcpkg route still needs dependency building and basic launch checks
on the reader's own development machine.

### 7.8 Completed manager validation

Manager 1.3.1 built in Release mode with zero warnings and zero errors.
All 117 backend checks and 37 platform checks passed; the latter include 19
isolated startup checks. A real Windows ACL test denied writes to the program
directory while AppData storage continued to work. The original ACL was restored
after the test.

The deployed application was also verified: 18 configuration files were imported
into AppData without content changes, and XXS 160K started successfully. Its
template and device-profile arguments pointed to AppData's `config/`, while logs
were written to AppData's `logs/`. A short API request returned `OK` (16 input and
2 output tokens), after which the model was stopped. **A real reboot/sign-in has
not been performed**; isolated startup checks do not replace that test. This is
separate from the GPU benchmark in section 6.

Manager checks require no GPU. Run from the repository root:

```powershell
dotnet build tests/windows-manager-platform/PlatformTest.csproj -c Release
dotnet tests/windows-manager-platform/bin/Release/net10.0-windows/PlatformTest.dll
dotnet run --project tests/windows-manager-backend/BackendCheck.csproj -c Release
```

Use `dotnet <assembly>` for platform checks so fake children and signal helpers
share the host. Optional `NINFER_TEST_ARTIFACT` checks a real model's headers
without loading tensors or creating a GPU context.
