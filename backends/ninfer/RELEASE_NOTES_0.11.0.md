# NInfer-3090 v0.11.0

**One RTX 3090 now prefills Qwen3.8-27B at up to 2,989 tokens/s, decodes a single stream at 187 tokens/s,
and holds 130,000 tokens of context in the same 24 GB.** This is the release where prompt processing
stops being the slow part: long documents, whole repositories and multi-turn agent transcripts are
ingested roughly twice as fast as v0.10.0, and the recommended launch profile is now the fastest one
measured on this card rather than the most cautious.

Five pull requests since v0.10.0 (#102, #104-#107), plus a catch-up with upstream NInfer that moves
the engine onto the v3 model container and a rewritten model layer. Everything below was measured on
this fork's own hardware (RTX 3090, `sm_86`, CUDA 12.8) unless a line says otherwise, and every
quality cost is stated next to the speed it buys.

## Who this is for

- **Local and self-hosted users on a 3090.** Faster first token on long prompts, faster decode, and a
  launcher that asks which model you want and picks the flags for you. Double-click `run.bat`.
- **Coding-agent and long-context users** (Claude Code, Cline, Aider and anything that resends a
  growing transcript). Prefix reuse and the tuned context cache are on in the recommended profile,
  and prefill is the part of an agent loop that used to dominate.
- **Small teams and anyone serving several users from one card.** One process, up to eight concurrent
  streams, OpenAI- and Anthropic-compatible endpoints, and a new `GET /v1/load` endpoint that a load
  balancer or gateway can poll to route around a busy engine.
- **Image and video workloads.** Vision is on in the recommended profile at a cost of about 10 MiB.

## Prefill: roughly 2x faster, in two steps

Prompt processing is where a 3090 spent most of its time on a long input. Two changes, each measured
against the previous state on the same build and card in one session (Qwen3.8-27B, INT8 KV,
`ninfer_bench`, prefill tok/s at context length):

**On by default: integer activations for every prefill projection (#105).** The attention and GDN
projections now reach the same int8 tensor-core path the MLP already used, and the shared schedule
stopped re-streaming each weight matrix once per 128 tokens.

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| v0.10.0 route set (integer MLP only) | 1,303 | 1,278 | 1,195 | 1,023 |
| **v0.11.0 default** | **1,685** | **1,649** | **1,522** | **1,243** |
| change | +29% | +29% | +27% | +22% |

Perplexity moves by **+0.004%**, inside run-to-run noise. The two steps were measured in separate
sessions, so read them as compounding rather than as one table: about 1.9x at 1K and 2.3x at 4K
against the v0.10.0 route set.

**Opt in with `--prefill-cublas --prefill-chunk 4096` (#106).** Weights are materialised as int8 with
one scale per row and the GEMM is handed to cuBLAS, which runs this card's shapes about twice as fast
as a hand-written integer mainloop can.

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| integer-activation route | 1,667 | 1,634 | 1,504 | 1,241 |
| **`--prefill-cublas --prefill-chunk 4096`** | **2,380** | **2,989** | **2,609** | **1,904** |
| change | +43% | +83% | +74% | +54% |

The cost is stated plainly: **+0.156% perplexity** (4.343155 to 4.349944 on the 1M-token corpus), about
0.5 GiB of extra workspace (156 to 661 MiB at chunk 4096), and decode is untouched. It is off by default because it is a quality
trade, and it can be narrowed: `--no-prefill-cublas-projections` keeps the attention and GDN
projections on the integer route and drops the cost to **+0.072%** for less of the speedup. Unaligned
prompt lengths, which is nearly every real prompt, are where it is furthest ahead.

## Decode: DFlash2 is now the recommended backend at one to four streams

DFlash2 drafts up to seven tokens per round, against three for MTP. Aggregate decode tok/s through the
serving route, thinking off, greedy, timed from the server's own request log:

| concurrent streams | MTP3 | DFlash2 K=7 | change |
|---:|---:|---:|---:|
| 1 | 135.0 | **187.1** | +39% |
| 2 | 238.2 | **313.5** | +32% |
| 4 | 387.4 | **406.2** | +5% |
| 8 | **522.8** | does not fit | |

The lead shrinks with concurrency because batching already amortises the weight sweep that
speculation exploits, and at eight streams the draft weights leave too little memory for the KV. So:
DFlash2 for one to four users, MTP3 for a full eight-lane server. The best draft window depends on the
workload (7 is the best compromise, not a constant); the sweep is in `docs/performance.md`.

## A launcher that picks the right flags

The recommended 27B profile is now the measured fast set, and the longer-context set is one variable
away. From the archive, `run.bat qwen38-27b` (or `./run.sh qwen38-27b`) starts:

```
--spec dflash2 --draft-tokens 7 --lm-head-draft --prefill-cublas --prefill-chunk 4096
--kv-dtype rk8v4 --embedding-q4 --gdn-state-fp16 --vision --vision-residency overlay
```

About **1.7x the prefill and 1.39x the decode** of the previous defaults, for +0.156% (cuBLAS) and
+0.08% (`rk8v4`) perplexity. It tops out near 130K tokens of context. For the longest context, still
fast, set `NINFER_SPEC=mtp`:

```
--spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --prefill-chunk 2048
--kv-dtype rk8v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16 --vision --vision-residency overlay
```

That set was verified at **200,000 tokens** by loading it. The difference is a real trade rather than a
free upgrade: DFlash2's draft weights and its refusal of `--lm-head-q6` cost about 65K tokens of
context between them. Every flag is explained, with what it buys and what it costs, in
[`docs/performance.md`](docs/performance.md#recommended-configurations-rtx-3090-qwen38-27b).

The scripts themselves were consolidated: `scripts/` went from 47 files to 13. One
`run.{bat,sh} <model> [profile]`, one `download-model.{bat,sh} <model>`, one `package-release`.
Double-click `run.bat` or `download-model.bat` and it asks which model you want.

**It starts even when the card is busy.** A Windows desktop can hold 2-3 GiB of a 3090, and the
default profile needs more room than that leaves. If the server is refused at startup for lack of GPU
memory, the launcher steps down by itself, an eighth of the context at a time with a smaller prefill
chunk and fewer state slots, tells you what it did, and starts. On a desktop holding 2.8 GiB,
`run.bat qwen38-27b` was refused three times and started at 81,920 tokens, then answered a request.
Anything you set explicitly (`NINFER_CONTEXT` and friends) is honoured as given and fails loudly, and
`NINFER_FALLBACK=off` turns the step-down off.

## Serving

- **`GET /v1/load` (#102).** A cheap, pollable endpoint reporting admitted, running, prefilling,
  decoding and waiting requests, KV and state occupancy, and monotonic counters you can difference
  into tok/s. It takes no engine lock, so polling cannot stall decode. Built for gateways and load
  balancers that schedule across several servers; the contract is in `docs/serving.md`.
- **`--lookup-ngram N` (#106), opt-in.** Context-lookup drafting: match the last N tokens against the
  sequence so far and propose what followed. It is exact, since a wrong guess costs throughput and never
  changes a token. Measured honestly: N of 8 or more gains about 3% (4.6% at K=5) and short N *loses*,
  because this model's draft head already accepts 95-96%. It is here for copy-heavy workloads, not as a
  headline.
- **Custom Jinja chat templates**, using llama.cpp's Jinja, replace the compiled C++ template, so a
  model's own template works as shipped. Reasoning-effort values that upstream's bundled template
  rejects are collapsed so Claude Code keeps working.

## The engine underneath: upstream v3 catch-up (#104)

This fork caught up with 40 upstream commits, which changed the model file format and the whole model
layer, so most of it was a port rather than a merge: a single config-driven `Qwen3_5` implementation
replaces the per-model packages. Every fork feature was re-ported onto it (load-time transcoding,
overlay vision, the sm_86 kernels, rk8v4, the multi-GPU expert split, the context cache, the Windows
platform work). The full suite passed **142/142** on an RTX 3090, and real-model checks covered
Qwen3.8-27B, Qwen3.6-27B and Qwen3.6-35B-A3B across text, MTP3, DFlash2, DFlash and vision.

## What we tried and dropped, on purpose

Negative results are recorded so nobody repeats them: overlapping the weight dequantise with the GEMM
(worse than serial), a larger cuBLAS tile budget (inside noise, +450 MiB), inverting the prefill loops
(a window inherits the chunk's scheduling stall, so it is a memory play at best), INT8 attention
probabilities (priced, then dropped), and a panel-major weight layout that is committed but parked.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench` | throughput benchmark against the public Engine route |
| `run.bat` / `run.sh` | serving profiles: `run <model> [profile]` |
| `download-model.bat` / `.sh` | pinned, resumable, checksum-verified model downloads |
| `README.md`, `SHA256SUMS.txt`, `LICENSE`, `VERSION` | the guide, checksums for every file, licence |

The Windows archive also carries the DLLs it needs: FFmpeg, libcurl, zlib, and NVIDIA's cuBLAS runtime
(`cublas64_12.dll` and `cublasLt64_12.dll`, redistributed under the CUDA Toolkit EULA, whose text is
included as `NVIDIA-CUDA-EULA.txt`), so no CUDA Toolkit is required. **New requirement on Linux:** the binaries now link the cuBLAS runtime, so install
the CUDA 12.8 runtime libraries including cuBLAS (`sudo apt install cuda-libraries-12-8` from NVIDIA's
repository), plus FFmpeg 6 and libcurl. They are built on Ubuntu 24.04, so glibc 2.38 or newer. Model
artifacts are not included: `download-model` fetches the Qwen3.6-35B-A3B (21 GB), Qwen3.8-27B (19 GB, the DFlash2 bundle,
which also carries the MTP weights) or Qwen3.6-27B (16 GB) artifact.

## Known issues

- **DFlash2 does not fit at eight concurrent streams** on 24 GB; use MTP3 for a full eight-lane server.
- **`--prefill-cublas` is a quality trade** (+0.156% perplexity) and is off by default.
- **The default DFlash2 context (131,072 at one lane) needs about 5.2 GB after the weights**, so on a
  card shared with a busy desktop it is refused and the launcher steps down (see above); expect
  roughly 80K on a desktop holding 2.8 GiB, and up to 131K on a headless card. Each refused attempt
  costs about 12 seconds. The `mtp` profile's defaults are extrapolated rather than measured on the
  launcher's exact command line. `--kv-capacity auto` does not help here, because the engine still
  reserves room for one full `--max-context` sequence.
- **Upstream route tables not yet swept for `sm_86`.** The catch-up brought retuned shape tables that
  this card has not been swept against. Every end-to-end figure above is measured; the per-shape tables
  underneath are inherited and listed in `TODO.md`. This fork's history is that inherited tables were
  wrong by 12-41%, so there may be more to gain.
- **The `--devices N,M` multi-GPU split** has only been exercised with `--devices 0,0` on a single-GPU box.
- **NVFP4 and K8V4 KV require a Blackwell GPU** and are refused on `sm_86` with a clear message.
- **Speculative decoding is not bit-identical to width-1 greedy decode**, because verification evaluates
  several columns in one pass; a near-tie argmax can flip. MTP behaves the same way.

## Upgrading

**Breaking: model files must be v3.** This release reads only the v3 `.ninfer` container, and a v2 file
from an earlier release is refused at load. You do not need to re-download: the weight bytes are
unchanged, so upgrade in place in about two minutes with the standard-library-only tool:

```
python tools/upgrade_ninfer_v2_to_v3.py OLD.ninfer NEW.ninfer
```

**Breaking: launcher scripts were renamed.** If you have shortcuts or service files pointing at the
old scripts:

| before | now |
|---|---|
| `run-qwen38-c1-maxctx.*` | `run qwen38-27b` (`NINFER_SPEC=mtp` for the longest context) |
| `run-qwen36-35b-a3b-c1-maxctx.*` | `run qwen36-35b-a3b` |
| `run-qwen38-c1.*` / `run-qwen38-c8.*` | `run qwen38-27b int8` / `run qwen38-27b c8` |
| `run-qwen38-vision.*` / `run-qwen36-35b-vision.*` | the default profiles already serve vision |
| `download-qwen38-27b.*` and the other two | `download-model <model>` |

The recommended 27B profile now defaults to **one lane at 131,072 tokens with DFlash2** (the old Linux
launcher defaulted to two lanes); pass `NINFER_CONCURRENCY` or `NINFER_SPEC=mtp` to change that. The
Windows launcher now also reads `NINFER_MODEL_DIR`, like the Linux one.

Everything else is drop-in: every new flag is opt-in and off by default. **Worth turning on:**
`--prefill-cublas --prefill-chunk 4096` if you value prompt speed over the last 0.15% of perplexity,
and DFlash2 for one to four users.
