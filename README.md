# KVMem + llama.cpp

**Prebuilt downloads:** [Windows x64 CUDA 13 / 12 (rc3)](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3) · [Linux / WSL2 x86_64 CUDA 13 / 12 (rc3)](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3) · [Windows / Linux ROCm (beta 2)](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/rc3-rocm-beta2)

**QQ community / QQ 交流群：1040777853**

## Near-lossless Qwen3.8-27B at a full 256K workspace on 16 GiB VRAM

llama.cpp inference with tiered KV memory for long-running agents.

**KVMem** adds a bounded GPU KV working set, host-memory storage and query-based retrieval to [llama.cpp](https://github.com/ggml-org/llama.cpp). llama.cpp handles model loading, inference, quantization and MTP. The separate `llama-kvmem-server` provides OpenAI-compatible chat, tools and optional vision. **NVMe offload is not implemented.**

This port supports **Qwen3.8-27B GGUF quants**, including IQ3 and IQ4. The sibling [kvmem-qw3](https://github.com/kvmem/kvmem-qw3) is a CUDA-native runtime focused on Q8, primarily tested on RTX PRO 6000.

The logical workspace (`-c`) can extend beyond 256K using host RAM; quality at those lengths remains experimental.

The [KVMem paper](https://arxiv.org/abs/2609.04852) shows that, on queries up to 256K, keeping only a **32K GPU-resident active context** is essentially lossless versus the **full 256K** history: **LongMemEval-S** 85.6% vs 86.6% accuracy, **AgentLongBench** 60.9% vs 59.5% task success.

**KV streaming vs. KVMem.** Both methods support a full 256K context on a 16 GiB GPU by storing part of the KV cache in host RAM. [Raymond Huang’s adaptive KV-cache streaming](https://medium.com/@raymond860909/running-qwen-27b-on-16g-vram-with-full-context-length-building-adaptive-kv-cache-streaming-for-bf1e819116e9) keeps part of the KV cache in VRAM and stores the rest in host RAM. During decoding, it prefetches the offloaded KV layer by layer through reusable GPU buffers, overlapping transfers with computation. This preserves attention over the entire history, but longer contexts increase both attention work and PCIe traffic, eventually slowing decode.

KVMem retrieves relevant historical blocks into a bounded GPU window, limiting the KV used for attention. On RTX 5060 Ti, the current MTP3 256K tool benchmark achieves **32–33 token/s decode**, **437–463 token/s prefill for initial computation** and **242–253 token/s overall prefill**, including input reprocessing and cache management.

**Performance on faster GPUs.** Our measurements use the RTX 5060 Ti, the entry-level 16 GB option in the desktop RTX 50 series. The 16 GB RTX 5070 Ti and RTX 5080 offer substantially more compute and roughly twice the memory bandwidth ([NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)). We therefore expect substantially faster GPU prefill and decode on these cards. Actual gains depend on the workload, CPU and host-memory transfers; benchmarks on these GPUs are welcome.

Current milestone: [`v0.16.0-rc3`](docs/milestones/v0.16.0-rc3.md) (pre-release).

**Limitation:** one generation cannot exceed `--kvmem-gen-reserve` (16384 tokens on the IQ3 recipe, 12288 on IQ4), including thinking. Retrieval pins the GPU window; new tokens only use those reserved slots. We are working on fixing this. For agent use, add a line to the system prompt such as: *Keep each turn's output, including thinking, within 16384 tokens* (use 12288 on IQ4). That makes oversized single-turn replies much less likely.

Version: **0.16.0-rc3**. See the [English / 中文 release notes](docs/releases/v0.16.0-rc3.md) for CUDA build choices and measured results.

## How KVMem works

Completed KV blocks are stored in host RAM. For each agent step, KVMem retrieves relevant blocks using the current query and places them in chronological order in a bounded GPU working set. Previously computed KV is reused across turns.

![High-level KVMem flow](docs/assets/kvmem-flow.svg)

Core flags (what the 16 GiB recipes still pass):

| Flag | Meaning |
|---|---|
| `-c` | Logical workspace, including history stored off GPU. 256K is the tested default; larger is experimental. |
| `--kvmem-budget` | How many historical tokens retrieval may keep on GPU. |
| `--kvmem-sink-tokens N` | Server and CLI: always keep the prefix in the GPU working set. Default `0` keeps one block (not disabled). Positive values round down to whole blocks, with a minimum of one block. For example, with block size 128, `1024` keeps 1024 tokens and `129` keeps 128. These blocks count toward `--kvmem-budget`. |
| `--kvmem-gen-reserve` | GPU slots reserved for new tokens so retrieval cannot fill the pool. **One generation cannot exceed this length** (including thinking). |
| `--kv-dtype` | Sets the same cache type for **main** attention K and V (IQ3 q8_0, IQ4 q5_0). Use `-ctk q8_0 -ctv q4_0` for mixed precision. |
| `--spec-type draft-mtp` | Enable multi-token prediction. |
| `--mmproj` | Vision projector GGUF. Omit for text-only. |

KVMem retrieval is on by default, with 128-token blocks, query replay `auto`, query policy `user`, MTP draft length 3, F16 draft KV, and ReplaySSM. You do not need to pass those unless you are overriding them. GPU KV size is `budget + gen_reserve`. When history exceeds `--kvmem-budget`, retrieval picks blocks for the current last-user query. Clients should send the full `messages` history each turn.

## How KVMem attaches to llama.cpp

`kvmem/` holds the host store and retrieval logic; `src/adapter/` connects it through llama.cpp’s memory interface. Attention kernels and original positions stay unchanged. Reselection transfers only blocks that changed.

Do **not** commit a dirty `llama.cpp` working tree. The submodule pointer is the pin; `scripts/apply-patches.sh` replays `patches/`.

## Tested platform

- Ubuntu 22.04.5 on WSL2, x86-64.
- RTX 5060 Ti with 16 GiB VRAM; Intel Core Ultra 7 255H and 32 GiB RAM (19.53 GiB visible to WSL2).
- CMake 4.4.3 and CUDA 13.2.86.

The project builds on llama.cpp's CUDA backend, with the platform above used for our measurements. Reports of successful runs, benchmarks and issues on other NVIDIA GPUs and systems are welcome. AMD/ROCm and Metal backends would need integration work.

## Prebuilt downloads

| Platform | Download | Notes |
|---|---|---|
| Windows x64 — CUDA 13.2.86 | [v0.16.0-rc3](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3) | Recommended **runtime** ZIP; GPU targets 75/80/86/89/90/120a. Quantizer is a separate optional ZIP. |
| Windows x64 — CUDA 12.9.86 | [v0.16.0-rc3](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3) | Alternative **runtime** ZIP; GPU targets 70/75/80/86/89/90/120a, including Volta. Quantizer is a separate optional ZIP. |
| Linux / WSL2 x86_64 — CUDA 13.2.86 | [v0.16.0-rc3 tar.gz](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.16.0-rc3/kvmem-v0.16.0-rc3-linux-x86_64-cuda13.2.86.tar.gz) | Runtime with CUDA libraries and both UIs; GPU targets 75/80/86/89/90/120a. Requires glibc 2.35+ and AVX2/FMA/F16C/BMI2. |
| Linux / WSL2 x86_64 — CUDA 12.9.86 | [v0.16.0-rc3 tar.gz](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.16.0-rc3/kvmem-v0.16.0-rc3-linux-x86_64-cuda12.9.86.tar.gz) | Runtime with CUDA libraries and both UIs; GPU targets 70/75/80/86/89/90/120a, including Volta. Requires glibc 2.35+ and AVX2/FMA/F16C/BMI2. |
| Windows x64 — ROCm (beta 2) | [rc3-rocm-beta2](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/rc3-rocm-beta2) | Native HIP runtime ZIP for gfx1100/gfx1200/gfx1201 (RX 7900 / 9060 XT / 9070 series). |
| Linux / WSL2 x86_64 — ROCm (beta 2) | [rc3-rocm-beta2](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/rc3-rocm-beta2) | Runtime tar.gz built on Ubuntu 24.04 with ROCm 7.2.x; other distributions may need a source build. |

Linux rc3 packages use the same source as Windows and include independent `scripts/linux/start-iq3.sh` / `start-iq4.sh` launchers. CUDA Toolkit, Python and Node.js are not required to run these packages. See the [Linux / WSL2 quick start and validation notes](https://github.com/kvmem/kvmem-llama.cpp/releases/tag/v0.16.0-rc3) and verify downloads with [SHA256SUMS](https://github.com/kvmem/kvmem-llama.cpp/releases/download/v0.16.0-rc3/SHA256SUMS). The source-tree launcher commands below apply to source builds; use the packaged README for prebuilt launcher arguments.

No model weights are bundled. For a Windows text-only setup, download the
ready-made IQ3 `-mtp` model linked in the [Windows quick start](scripts/windows/README.md).
For vision, download **`mmproj-Qwen3.8-27B-Q5_K-MIX.gguf`** from
[HermiHg](https://huggingface.co/HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF) and pass its path with `-Mmproj`
(Windows launchers) or `--mmproj` (server). No local projector quantization is
needed. The performance tables below retain their original Q8/BF16 projectors.
The locally converted IQ4 MTP-Q4_0 main model does not yet have a project-provided
download link in this release; use your prepared file or the optional quantizer.
The recipes and conversion commands below document the historical tested setup.

## Clone, patch, build

Building uses a C++17 compiler, CMake and **CUDA Toolkit 13.2 Update 2 (nvcc 13.2.86) or newer**. The Linux startup scripts use Python 3.10+ and `ss` (iproute2).

**CUDA compiler version matters for correctness.** The validated baseline is nvcc **13.2.86** on Linux/WSL2 and native Windows. A Windows build made with nvcc 13.2.51 produced garbage output from Qwen3.8-27B IQ3_S even with KVMem and MTP disabled; rebuilding unchanged source with 13.2.86 restored correct output. A successful build, health check or small Q8 model test does not validate IQ3 inference. Newer toolchains still need correctness testing before release.

Check `nvcc --version` for the compiler selected by CMake; `release 13.2` alone is insufficient, and the CUDA version shown by `nvidia-smi` describes driver support. After upgrading the Toolkit, configure a **new build directory** and rebuild the binaries. Updating the driver or replacing CUDA DLLs does not fix CUDA kernels already compiled into an old binary.

An experimental [native Windows build](scripts/windows/README.md) is being validated. It disables NVMe storage and includes PowerShell launchers; the performance results below remain Linux/WSL2 measurements.

```bash
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
git checkout v0.16.0-rc3
git submodule update --init
scripts/apply-patches.sh
scripts/build-cuda.sh
```

The submodule is ggml-org/llama.cpp at pin `b81c99b`. `scripts/apply-patches.sh` applies `patches/llama-kvmem-current.patch` (or `multimodal-upgrade.patch` on an older KVMem tree). Running it twice is safe. Do **not** apply numbered `0001`–`0004` together with the cumulative patch. See [patches/README.md](patches/README.md).

`scripts/build-cuda.sh` sets `GGML_CUDA_FA_ALL_QUANTS=ON` (needed for `--kv-dtype q5_0` on hybrid models). Binaries: `build/bin/llama-kvmem-server`.

The build script defaults to `CMAKE_CUDA_ARCHITECTURES=120a-real` for the tested RTX 5060 Ti. For another GPU, set `CMAKE_CUDA_ARCHITECTURES` to its appropriate target when running the script; other GPU targets have not been tested here.

## Browser chat

The updated Windows rc3 runtime packages include both UIs: **full UI by default** at `share/kvmem/ui`, plus the lightweight UI at `share/kvmem/ui-lightweight`. Their independent `start-iq3.ps1` / `start-iq4.ps1` scripts accept only `-Model`, `-Mmproj` and optional `-Gpu` (default `0`, index or UUID). They directly invoke the server and no longer use shared launch helpers. To choose the lightweight UI, edit `$UiDir` in the script to end in `share\kvmem\ui-lightweight`; to disable UI, replace `--webui` with `--no-ui`. Edit `$Port = 18200` to change the port. Download the runtime ZIP again for these updated scripts. Full UI does not add server-side tool execution or stream resumption to the KVMem backend. See the [Windows runtime guide](scripts/windows/README.md) for a complete launch command.

The optional lightweight UI reuses llama.cpp's Markdown/code renderer, input components and browser-local history. It supports text and images, separate thinking effort/budget controls, stopping generation, and server-measured decode speed. It does not execute tools or manage model loading.

Build the static page once with Node.js 22 and npm:

```bash
python3 scripts/build-webui.py
```

Add `--full-ui` to build the full upstream UI, including its generated icons and PWA assets. Use separate `--output` directories when keeping both builds.

Then start the rebuilt server with the usual IQ3/IQ4 script and open `http://127.0.0.1:18200/`. The server automatically serves `build/share/kvmem/ui/` when present. Precompiled packages can include the page, so users do not need Node.js. `--ui-dir PATH` selects another static directory; `--no-ui` disables the page.

Chat histories stay in this browser. Switching histories can require recomputing an uncached prompt; normal continuation reuses the existing KV cache. Closing or reloading the page interrupts generation; stream resumption is not included.

## llama-server CLI compatibility

The rc3 version of `llama-kvmem-server` accepts the common flags below with
their llama.cpp meanings. Use the rc3 binaries or rebuild from source; rc2
binaries predate these additions. This is an independent, single-slot server, so it does not
yet accept every `llama-server` option.

| Options | Meaning |
| --- | --- |
| `-t`, `--threads`; `-tb`, `--threads-batch` | CPU generation/batch threads; values <= 0 select hardware concurrency. An explicit `-t` also sets batch threads unless `-tb` is given. |
| `-b`, `--batch-size`; `-ub`, `--ubatch-size` | Logical/physical batch sizes. Omitted `-ub` retains KVMem's existing default of the logical batch size. |
| `-fa`, `--flash-attn on\|off\|auto` | Flash Attention mode; also applied to the MTP draft context. Backend/model restrictions still apply. |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` | Nonnegative layer count or `all` (`-2`). Automatic GPU fitting (`auto`/`-1`) is not implemented and produces an error. |
| `-a`, `--alias` | Model name returned by `/v1/models`, `/props` and chat responses. |
| `--api-key`, `--api-key-file` | API authentication; details below. |
| `-lm`, `--load-mode` | `auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`; legacy `--mmap`, `--no-mmap`, `--mlock` map to the corresponding mode. Last loading-mode flag wins. |
| `-np`, `--parallel` | Only `1` is supported. Automatic or multiple slots produce an error. |
| `-to`, `--timeout` | HTTP read/write timeout in seconds; KVMem retains its 1800-second default. |
| `--threads-http` | HTTP worker count; <= 0 selects automatically. This does not enable parallel inference slots. |
| `-dev`, `--device`; `--list-devices` | Select one device or an explicit CUDA list such as `CUDA0,CUDA1`; `none` selects CPU. List devices without loading a model. |
| `-mg`, `--main-gpu`; `-sm`, `--split-mode` | Multi-GPU requires `layer` and `--gpu-layers all`. `none` remains available for one GPU. |
| `-ts`, `--tensor-split` | Layer proportions, with one value for each explicitly selected GPU. |

Additional upstream aliases: `--usage` = `--help`, `--predict` = `--n-predict`,
`-s` = `--seed`, `-mm` = `--mmproj`, `--no-webui` = `--no-ui`, and
`--path` = `--ui-dir`.

The experimental layer path on this branch is opt-in. For example:

```powershell
.\llama-kvmem-server.exe --model C:\models\model.gguf --device CUDA0,CUDA1 --split-mode layer --tensor-split 2,1 --gpu-layers all --spec-type none --kvmem-budget 32768
```

Use `--list-devices` to obtain device names; the example budget must fit every
owning GPU. A multi-GPU request requires CUDA devices, full layer offload and
`--split-mode layer`. KVMem capture and KV layout use a synchronous backend
path for correctness. On two GPUs, embedded `nextn` MTP is experimental and
requires `--spec-type draft-mtp` plus an explicit
`--kvmem-mtp-state snapshots` or `replay`; `auto` and independent draft models
are not supported for multi-GPU MTP. Tensor and row split remain unsupported.
Automatic discovery still requires an explicit choice when several GPUs are
present. Indices refer to the visible device list (and to the selected list
when `--device` is supplied). Single-GPU defaults remain unchanged.

Threads, physical batch size and Flash Attention settings propagate to MTP.
Existing model, host/port, context, sampling, chat-template, vision and KV-cache
flags remain available; run `--help` for the full supported list. `-n` / `--n-predict`
now defaults to `-1`, matching llama-server: no additional output-token cap.
Generation still stops at EOS/stop sequences and remains bounded by the available
context and KVMem generation reserve; a request may set `max_tokens` explicitly.
Other KVMem defaults
and `--kvmem-*` controls remain unchanged. Numeric arguments reject malformed and
out-of-range values. Context size must be positive; `--n-predict` accepts `-1`
or a positive number.

For example, append these options to an existing model/KVMem launch command:

```sh
--threads 8 --threads-batch 8 --batch-size 512 --ubatch-size 128 \
--flash-attn on --gpu-layers all --parallel 1 --alias kvmem-27b \
--api-key-file /path/to/api-keys.txt
```

`--api-key KEY1,KEY2` accepts comma-separated keys (CSV quoting is supported).
`--api-key-file PATH` reads one key per line, ignoring blank lines and lines
starting with `#`; Windows CRLF files are supported. Repeated key flags append
allowed keys. Keys must contain printable ASCII without whitespace; empty lists,
empty/unreadable key files and malformed quoting fail startup.

Clients send `Authorization: Bearer YOUR_KEY` (or `X-Api-Key: YOUR_KEY`). With
keys configured, API routes including `/props` and `/v1/models` require a matching
key and return HTTP 401 otherwise. Health checks, CORS preflights and mounted UI
assets remain public. Loading the UI does not grant access to authenticated APIs;
clients must supply the key. Without key flags, authentication remains disabled.

Regression checks: `kvmem-server-options-test` via CTest, and
`python scripts/test_server_compat.py --server /path/to/llama-kvmem-server`.
Add `--model PATH` for live auth/inference checks, `--mtp` for MTP, and
`--mmproj PATH --image PATH` for the optional vision fixture containing `6037`.

### Environment variables and startup diagnostics

Supported environment variables use the names from this project's pinned
llama.cpp version. Values pass through the same validation as CLI arguments.
For ordinary settings, precedence is **CLI > environment > default**. Environment
values are validated first, so an invalid environment value must be corrected
even when a CLI override is present. API keys are additive: environment keys,
environment key files, CLI keys and CLI key files all add allowed credentials.
A CLI key does not revoke an environment key.

| Environment variables | Corresponding settings |
| --- | --- |
| `LLAMA_ARG_MODEL`, `LLAMA_ARG_ALIAS` | Model path and API model name |
| `LLAMA_ARG_HOST`, `LLAMA_ARG_PORT`, `LLAMA_ARG_TIMEOUT`, `LLAMA_ARG_THREADS_HTTP` | HTTP server |
| `LLAMA_ARG_CTX_SIZE`, `LLAMA_ARG_N_PREDICT`, `LLAMA_ARG_BATCH`, `LLAMA_ARG_UBATCH`, `LLAMA_ARG_THREADS` | Context, output and CPU/batch configuration |
| `LLAMA_ARG_DEVICE`, `LLAMA_ARG_N_GPU_LAYERS`, `LLAMA_ARG_MAIN_GPU`, `LLAMA_ARG_SPLIT_MODE`, `LLAMA_ARG_TENSOR_SPLIT` | GPU selection; the same layer-only multi-GPU restrictions apply |
| `LLAMA_ARG_FLASH_ATTN`, `LLAMA_ARG_CACHE_TYPE_K`, `LLAMA_ARG_CACHE_TYPE_V`, `LLAMA_ARG_N_PARALLEL` | Attention, KV types and single-slot configuration |
| `LLAMA_ARG_LOAD_MODE`, `LLAMA_ARG_MMAP`, `LLAMA_ARG_MLOCK` | Model loading; legacy environment options apply before `LOAD_MODE` |
| `LLAMA_ARG_MMPROJ`, `LLAMA_ARG_MMPROJ_OFFLOAD`, `LLAMA_ARG_IMAGE_MIN_TOKENS`, `LLAMA_ARG_IMAGE_MAX_TOKENS` | Vision |
| `LLAMA_ARG_UI`, `LLAMA_ARG_STATIC_PATH` | UI enabled/disabled and static directory |
| `LLAMA_API_KEY`, `LLAMA_ARG_API_KEY_FILE` | Authentication; no secret values are logged |
| `LLAMA_ARG_JINJA`, `LLAMA_ARG_CHAT_TEMPLATE`, `LLAMA_ARG_CHAT_TEMPLATE_FILE`, `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` | Templates; disabling Jinja is unsupported |
| `LLAMA_ARG_REASONING_EFFORT`, `LLAMA_ARG_THINK_BUDGET`, `LLAMA_ARG_THINK_BUDGET_MESSAGE`, `LLAMA_ARG_TOP_K` | Reasoning and top-k sampling |
| `LLAMA_ARG_SPEC_TYPE`, `LLAMA_ARG_SPEC_DRAFT_N_MAX`, `LLAMA_ARG_SPEC_DRAFT_P_MIN` | Existing MTP settings; independent draft models remain unsupported |

Boolean environment values accept `1/0`, `true/false`, `on/off`, `yes/no`
(case-insensitive). `--ui` / `--webui` can override `LLAMA_ARG_UI=false`.
Unsupported `LLAMA_ARG_*` names produce a warning without printing their values.
Unsupported API-key/TLS variable names fail startup rather than silently leaving
authentication or native TLS unconfigured. Empty keys and empty/unreadable key
files also fail startup. Unset both key variables and omit both key flags to
disable authentication.

PowerShell example (the values apply to the current shell and its child processes):

```powershell
$env:LLAMA_ARG_MODEL = 'C:\models\model.gguf'
$env:LLAMA_ARG_DEVICE = 'CUDA0'
$env:LLAMA_ARG_CTX_SIZE = '32768'
$env:LLAMA_ARG_PORT = '18200'
$env:LLAMA_ARG_API_KEY_FILE = 'C:\config\kvmem-api-keys.txt'
.\llama-kvmem-server.exe --ctx-size 65536
```

Enable `--kvmem-trace` (or `KVMEM_TRACE=1`) to emit the structured startup records described below. Without tracing, normal startup messages and errors remain available.

Startup first validates configuration, model/projector/UI files and incompatible
settings before loading model weights. `KVMEM_STARTUP requested=...` records the
requested configuration. After initialization and a successful port bind,
`KVMEM_STARTUP ready=...` records actual context/batch/thread values, output limits,
vision/MTP state, authentication status/key count and parameter sources. GPU and
Flash Attention requests are labeled as requested; llama.cpp's backend logs show
the actual placement and attention selection. Unlisted sources use defaults;
inherited batch/thread settings are identified explicitly. No raw key values or
key-file contents are included. Bind failures identify the address/port and do
not print a successful listening message.

Run `python scripts/test_server_environment.py --server PATH --output DIR`
for environment validation; add `--model SMALL_GGUF` for live precedence,
authentication, inference and startup-summary checks.

The default bind address remains `127.0.0.1`. For LAN access, pass `--host 0.0.0.0 --port 18200`; clients use the host computer's LAN IP. Configure `--api-key` when authentication is needed and allow the port through the host firewall.

## Recommended settings (16 GiB)

Both recipes use a 256K workspace and a bounded GPU KV working set. The listings below match `scripts/start-iq3.sh` / `start-iq4.sh`: they only pass flags that are not already server defaults. Sampling follows the Qwen3.8-27B card and can be overridden per request; `temperature=0` is greedy.

| | Thinking (these recipes) | Non-thinking |
|---|---:|---:|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 1.5 |
| frequency_penalty | 0.0 | 0.0 |
| repetition_penalty | 1.0 | 1.0 |

```bash
# Recommended IQ3 recipe; select the downloaded vision projector explicitly.
MMPROJ=/path/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf scripts/start-iq3.sh

# Preview the recommended configuration.
MODEL=/path/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  MMPROJ=/path/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf scripts/start-iq3.sh --dry-run
```

GPU selection honors `CUDA_VISIBLE_DEVICES`; otherwise it chooses a 5060 Ti or the only GPU. Ambiguous multi-GPU setups require an explicit selection. `MODEL`, `MMPROJ`, `MMPROJ_DEVICE`, `HOST` and `PORT` can override recipe defaults. MTP3 and ReplaySSM are server defaults; override with `SPEC_DRAFT_N_MAX` and `KVMEM_MTP_STATE` if needed. CUDA libraries come from the build directory, caller environment or the toolkit recorded during compilation; use `CUDA_HOME` or `LD_LIBRARY_PATH` for a custom installation. An existing matching service is reused; switching configuration requires `--restart`, which only stops this project's server.

### llama.cpp-compatible KV cache flags

Both `llama-kvmem-server` and `llama-kvmem-cli` accept llama.cpp's main-model
KV cache flags. These are equivalent ways to select Q8 K and V:

```text
-ctk q8_0 -ctv q8_0
--cache-type-k q8_0 --cache-type-v q8_0
--kv-dtype q8_0
```

For the IQ4 recipe, use `-ctk q5_0 -ctv q5_0`. `--kv-dtype` remains a shorthand
that sets both types. Arguments apply from left to right; the last assignment
to each component wins. Setting only `-ctk` does not change V (both default to
`q8_0`), so specify both when changing precision.

Supported types are `f16`, `f32`, `q8_0`, `q5_0` and `q4_0`.
K and V may independently select `q8_0`, `q5_0` or `q4_0`: all nine
quantized pairs are accepted. Float/quantized pairs such as `q8_0/f16` remain
rejected before model loading. Models that require shared K/V types still
cannot use mixed precision; execution also depends on backend kernel support.

GPU validation covers the common `q8_0/q8_0`, `q5_0/q5_0`, `q4_0/q4_0`
pairs and mixed **`q8_0/q4_0`**. Q8/Q4 additionally passed cache save/restore,
MTP replay and long-context checks on CUDA. The other five mixed quantized
pairs are enabled with argument-parsing checks only; they have not received
full inference, quality or performance validation. ROCm/Vulkan combinations
have not been validated here.

```text
llama-kvmem-server -m model.gguf -ctk q8_0 -ctv q4_0
```

The Linux recipes accept `--cache-type-k q8_0 --cache-type-v q4_0`;
in the updated Windows rc3 runtime scripts, edit `-ctk q8_0 -ctv q4_0`
directly in the script. The older source launcher also accepts
`-CacheTypeK q8_0 -CacheTypeV q4_0`. Existing recipe defaults are unchanged.

Flag compatibility does not imply support for every llama.cpp cache type or
mixed K/V combination. These flags affect the main model; MTP cache precision
is configured separately with `--spec-kv-dtype`. The server and recipes default
to `f16`; the CLI inherits the main K/V types unless overridden. Inherited
mixed K/V types are preserved independently; an explicit `--spec-kv-dtype`
sets both draft types together.

### Thinking and chat templates

The launchers accept optional template settings, using llama.cpp's native Jinja renderer:

```bash
scripts/start-iq3.sh --reasoning-effort low
scripts/start-iq4.sh --chat-template-file /path/custom.jinja \
  --chat-template-kwargs '{"enable_thinking":true}'
```

Add `--restart` to change an existing service. Inline Jinja is accepted through `--chat-template`; Jinja is always enabled (`--jinja` is also accepted).

Requests to `/v1/chat/completions` can override the defaults:

```json
{
  "messages": [{"role": "user", "content": "What is 19 × 23?"}],
  "reasoning_effort": "low",
  "chat_template_kwargs": {"enable_thinking": true},
  "reasoning_budget_tokens": 128,
  "max_tokens": 512
}
```

In the current GSQ 27B template, `low` and `xhigh` inject instructions for brief or careful reasoning; `medium` adds neither instruction. The template defaults to `xhigh`. These are prompt preferences: `reasoning_budget_tokens` controls the thinking budget, while `max_tokens` limits the whole output. Other models may support different effort levels.

`reasoning_effort: "none"` disables thinking; `"default"` removes the effort override and uses the template's default. A positive effort does not turn thinking back on if it is disabled. Request kwargs override launcher defaults, and top-level `reasoning_effort` overrides the value in kwargs. For `enable_thinking`, kwargs take precedence over the top-level field; use JSON booleans, not strings. Template changes reuse the common rendered prefix where possible; changing instructions near the start of the history can require processing that history again.

### IQ3 27B — text + vision, with MTP

`scripts/start-iq3.sh`. ISTA GGUF **as published** (MTP head not requantized). **Main KV q8_0**, MTP KV F16.

- Text: [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) → `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (use the `-mtp` file)
- Vision: [HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF](https://huggingface.co/HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF) → `mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` (already quantized; no local conversion).

Pass the downloaded projector explicitly with `MMPROJ=/path/mmproj-Qwen3.8-27B-Q5_K-MIX.gguf` for the Linux launcher, `-Mmproj` for the Windows launcher, or `--mmproj` for the server. The historical performance results below used the original Q8 projector.

```text
-m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
--mmproj mmproj-Qwen3.8-27B-Q5_K-MIX.gguf --no-mmproj-offload --image-max-tokens 512
-c 262144 -n 16384
--kvmem-budget 36864 --kvmem-gen-reserve 16384
--kv-dtype q8_0
--spec-type draft-mtp
--enable-thinking --reasoning-budget 4096
```

IQ3 now defaults to CPU vision (`--no-mmproj-offload`) to leave more GPU memory for inference. Vision remains available. To explicitly use GPU vision, set `MMPROJ_DEVICE=gpu` on Linux/WSL; in the updated Windows rc3 runtime script, replace `--no-mmproj-offload` with `--mmproj-offload`. Historical performance tables below retain their original projector placement.

### IQ4 27B — optional experimental comparison

**IQ3 is the primary recommended model and download for this release.** IQ4 is
retained only as an optional test configuration and in the historical benchmark
tables below. It uses a separately prepared MTP-Q4_0 main model, q5_0 main KV,
budget 32768 and generation reserve 12288.
The existing IQ4 launchers remain available to testers who already have the
required files; IQ4 is not part of the primary download/setup instructions.

### 5060 Ti results

**Test hardware:** RTX 5060 Ti 16 GiB, Intel Core Ultra 7 255H, 32 GiB RAM. Ubuntu 22.04.5 on WSL2 exposes 16 logical CPUs and 19.53 GiB RAM.

Both tasks use MTP3 with ReplaySSM and thinking with a 128-token budget and at most 512 output tokens per request. That is the speed-test setting; the start scripts default to `--reasoning-budget 4096`. IQ3 uses GPU Q8_0 vision; IQ4 uses CPU BF16 vision. RAM is runtime process RSS, excluding loading; VRAM is whole-GPU usage.

Task 1: ~12K text, then one image, then code generation. One warmup run precedes two measured runs. This test uses `--image-max-tokens 1024`; measured repeats reuse cached image embeddings. The recipes retain a 512-token image limit.

| Metric | IQ3 | IQ4 |
|---|---:|---:|
| Prefill — initial computation | 574.49 token/s | 595.60 token/s |
| Prefill — overall | 544.65 token/s | 505.52 token/s |
| First image encode (warmup) | **0.41 s** | **21.79 s** |
| Aggregate decode | **38.55 token/s** | **44.30 token/s** |
| Image decode | 38.64 token/s | 45.83 token/s |
| Code decode (512 tokens) | 39.88 token/s | 44.35 token/s |
| MTP acceptance | 70.74% | 80.85% |
| Runtime host RAM peak | **4308.21 MiB** | **4846.82 MiB** |
| VRAM peak | **15591.10 MiB** | **15445.10 MiB** |
| Minimum free VRAM | 460.90 MiB | 606.90 MiB |

Task 2: 32 tool-result rounds plus a base request, reaching **262058 / 262144 tokens** including generation. Both recipes receive identical requests; projectors stay loaded, but no images are sent.

| Metric | IQ3 | IQ4 |
|---|---:|---:|
| Prefill — initial computation | 437.13 token/s | 463.18 token/s |
| Prefill — overall | 242.06 token/s | 253.41 token/s |
| Aggregate tool-round decode | 31.74 token/s | 33.31 token/s |
| Code decode (512 tokens) | 30.53 token/s | 38.15 token/s |
| MTP acceptance | 64.70% | 67.08% |
| Runtime host RAM peak | **13483.52 MiB** | **11244.75 MiB** |
| VRAM peak | **15617.10 MiB** | **15591.69 MiB** |
| Minimum free VRAM | 434.90 MiB | 460.31 MiB |

Initial computation measures the first processing of new input. Overall includes any repeated processing, cache management and image encoding. Both rates use **total new input divided by the corresponding total time across the task**, counting visual rows as input positions. Decode includes thinking tokens. Code decode refers to the final request.

[Full results and benchmark commands](docs/recommended-config-performance.md). Summarize saved logs with `python3 scripts/summarize_canary.py <artifact-directory>`.

### Historical configuration comparison

| | IQ3 | IQ4 |
|---|---|---|
| Images | GPU vision | CPU vision |
| Decode (Task 1 / Task 2) | ~39 / ~32 token/s | ~44 / ~33 token/s |
| GPU KV window | 36K retrieve / 16K generate | 32K / 12K |
| Main KV | q8_0 | q5_0 |
| MTP weights | Official ISTA `-mtp` | Local Q4_0 requant of Unsloth |
| Runtime host RAM (Task 1 / Task 2) | 4308.21 / 13483.52 MiB RSS | 4846.82 / 11244.75 MiB RSS |

**Use IQ3 for the recommended setup.** IQ4 is retained only as an optional experimental comparison; the figures above are historical Linux/WSL2 measurements.

### Server logging

The server uses llama.cpp's timestamped logger. Normal output shows startup,
prompt processing, generation progress and a final timing summary. Prompt rates
exclude cached tokens; replay work remains included in elapsed prefill time.
Warnings and errors remain visible without enabling KVMem diagnostics.

- `-lv N`, `--verbosity N`, `--log-verbosity N`: `0` silent, `1` errors,
  `2` warnings, `3` normal output (default), `4` llama.cpp trace, `5` debug.
  Argument validation errors are always printed.
- `--kvmem-trace` or `KVMEM_TRACE=1`: additionally emit the raw `KVMEM_*`
  diagnostic records on stderr, preserving their benchmark/script format.
  This switch is independent of `--verbosity`.
- `--no-kvmem-trace`: override the environment and disable those diagnostics.
  An unset, empty or `0` environment value also disables them.

Scripts that parse KVMem records must explicitly set `KVMEM_TRACE=1`.
The existing `KVMEM_PERF=1` performance counters remain independently available;
they do not require full tracing. Tracing adds overhead, so compare benchmark
results using the same diagnostic settings.
For a detailed bug report, use `--verbosity 4 --kvmem-trace` and capture both
stdout and stderr. This logging integration adapts the diagnostic gate and
progress-reporting approach from [PR #9](https://github.com/kvmem/kvmem-llama.cpp/pull/9).

## APIs

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (sampling, stream, tools, optional images)

No auth or TLS unless an API key is set. Binds `127.0.0.1` by default. To serve
on the LAN, pass `--host 0.0.0.0` to the server or to the launchers
(`scripts/start-iq3.sh --host 0.0.0.0`; Windows: `start-iq3.ps1 -ListenHost 0.0.0.0`),
or set `HOST` / `LLAMA_ARG_HOST` (e.g. `HOST=0.0.0.0 scripts/start-iq3.sh`). Open
firewall ports for LAN clients. To require a key, pass `--api-key sk-xxx` to the
server or Linux launchers (`start-iq3.sh --api-key sk-xxx`), or `-ApiKey 'sk-xxx'`
/ `-ApiKeyFile path` on Windows (mirroring llama-server). Protected routes then
need `Authorization: Bearer sk-xxx` (or `X-Api-Key: sk-xxx`); `/health`,
`/v1/health`, OPTIONS requests and mounted UI static assets remain public.
On Linux, relative `--api-key-file` paths are resolved from the caller's current
directory and checked for readability before an existing service is stopped.
Native TLS is not supported. Stream `usage` includes
`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`.

## Documentation

- [v0.16.0-rc3 milestone](docs/milestones/v0.16.0-rc3.md)
- [Modification plan](docs/modification-plan.md)
- [Architecture](docs/architecture.md)
- [Patch replay](patches/README.md)
- [Recommended 16 GiB performance](docs/recommended-config-performance.md)
- [256K tool benchmark](docs/long-context-benchmark-2026-09-14.md)
- [Query replay](docs/query-replay-implementation-report-2026-09-14.md)
- [Multimodal usage](docs/multimodal-implementation-report-2026-09-14.md)
- Native Qwen engine: [kvmem/kvmem-qw3](https://github.com/kvmem/kvmem-qw3)

## Project layout

```
kvmem/            Host KVMem library (no llama.cpp includes)
src/adapter/      llama_memory_i wrapper
tools/            llama-kvmem-cli, llama-kvmem-server, vision helpers
scripts/          apply-patches, CUDA build, GPU bind, start helpers
patches/          Diffs against the llama.cpp pin
docs/             Architecture, milestones, multimodal
llama.cpp/        Submodule (pin only; apply patches after clone)
models/           Local GGUFs (gitignored)
```

## Acknowledgments

Thanks to **melis** and **redsnow23** from Bilibili for testing the project and providing helpful feedback.

## License

Checkpoints are distributed separately and may use different terms. llama.cpp remains under its upstream license. KVMem-qw3 source is Apache-2.0; this port should be treated the same unless a `LICENSE` file is added to this tree.

## Paper and citation

[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)

Copy the BibTeX entry below and cite it with `\cite{chai2026kvmem}`.

```bibtex
@misc{chai2026kvmem,
  title         = {{KVMem}: Virtualizing Million-Token Agent Workspaces on a Consumer {GPU}},
  author        = {Di Chai and Leye Wang and Zeshen Su and Zhiguo Xia and Zhihang Yu},
  year          = {2026},
  eprint        = {2609.04852},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.04852}
}
```
