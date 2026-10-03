# NInfer CLI

`build/apps/ninfer` runs one request against one v3 `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

The examples use Qwen3.8-27B NVFP4 with FP8 KV storage.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype int8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting NInfer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort. `--greedy` selects exact argmax
decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be
combined with `--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as
reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw
control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only
  the selected speculative backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI uses root-only context mode, so it does not reserve an extra Device
  checkpoint StateImage or capture a continuation that no later request could consume.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and does not
accelerate Vision encode. The default speculative and Vision settings produce the smallest resident
profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, or the
35B-A3B DFlash or Qwen3.8-27B DFlash2 backend with one to fifteen. Both masked-draft backends
may be combined with `--vision`.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype int8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 4`, optionally with `--lm-head-draft` and `--vision`.
DFlash2 accepts every draft count from 1 through 15. Seven is the checkpoint recommendation and
what `docs/performance.md` was measured with, but **four is faster on this hardware** — measured
on the RTX 3090, 27B, INT8 KV, greedy, generating 256 tokens of ordinary prose, mean of three
runs:

| `--draft-tokens` | decode | vs no speculation |
|---:|---:|---:|
| (none) | 37.7 tok/s | — |
| 1 | 49.7 | +31.7% |
| 2 | 56.4 | +49.5% |
| 3 | 58.4 | +54.9% |
| **4** | **59.1** | **+56.5%** |
| 5 | 57.2 | +51.5% |
| 6 | 48.4 | +28.3% |
| 7 | 48.2 | +27.7% |
| 8 | 47.6 | +26.2% |
| 10 | 42.4 | +12.4% |
| 12 | 40.6 | +7.6% |

Seven costs 18.4% against four, which is the same fact as four being 22.6% faster than seven — only
the denominator differs. There is a distinct cliff between five and six — 57.2 to 48.4 —
which looks like a block-geometry boundary rather than an acceptance effect, since acceptance is
still rising there. `--lm-head-draft` is within noise of unset for DFlash2 at every count and can
be left off.

For reference on the same measurement, MTP3 with the draft head reaches 62.4 tok/s (+65.3%), so
MTP remains the faster backend on text — but by 5% at DFlash2's best draft count, not the 36% that
seven implies.
Both `groupwise-int` and `nvfp4` artifacts use the same Engine route, including CUDA Graph,
concurrent requests, sampling penalties, and prefix reuse. An artifact without the companion
weights reports a missing DFlash2 component when selected. Vision, MTP and DFlash follow the same
rule: their weights are required only when that component is enabled at startup.

Only one speculative backend can be enabled per Engine. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with
the optimized proposal head. DFlash accepts one to fifteen draft tokens; seven forms the measured
block length eight, while fifteen uses the maximum supported block length sixteen.

## Common options

Windows text generation can opt in to observed-residency admission with
`--cuda-memory-policy strict`, which selects Hybrid automatically. The CLI's Hybrid cache has no
persistent Host retention tier because an invocation serves one request. The default CLI route
and driver allocation policy are unchanged. See [strict CUDA residency](serving.md#strict-cuda-residency)
for probe semantics and limits.

The table lists executable defaults. The examples above select FP8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--rope-yarn` | past the model's native window, YaRN at factor `--max-context` / native instead of plain RoPE for the text and MTP layers | off |
| `--rope-yarn-factor F` | YaRN at a fixed factor in `[1,4]` for every position whatever `--max-context` is; it grows neither the default context nor the KV pool | `1` (as `--rope-yarn` decides) |
| `--rope-scaling-factor F` | instead of YaRN, linear position interpolation for the text and MTP layers: a position past `--rope-scaling-original-context` rotates at original + (position - original) / F, so positions up to it keep their exact angles; `[1,32]`, excludes `--rope-yarn` and `--rope-yarn-factor`, and raises neither the four-times-native window cap nor the KV pool. The DFlash adapter keeps its plain RoPE | `1` (off) |
| `--rope-scaling-original-context N` | the interpolation threshold | the native window |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--kv-headroom-mib N` | device memory in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto`. `--vram-headroom-mib` is accepted as an alias | `1024` |
| `--cuda-memory-policy default\|mixed\|strict\|strict-N-N` | choose reported-free sizing, allowed Windows system-RAM fallback, or verified VRAM residency; `strict` means reserve 64 MiB / probe step 128 MiB. [Policy details](rtx-5070ti-windows.en.md#4-choose-a-memory-policy) | `default` |
| `--use-alt-prefix-caching` | select Hybrid device context storage; automatic with `strict`, with no persistent Host retention tier in the CLI | off |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see the [README](../README.md#several-gpus-pipeline-stages---devices-ab)); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|rk4v4-e8\|nvfp4\|k8v4` | KV-cache storage. `rk8v4` is opt-in RotorQuant, `rk4v4` opt-in Lloyd-Max 4-bit keys and `rk4v4-e8` opt-in E8-lattice INT4 keys; all eight are accepted on this fork's sm_86/sm_89 targets | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--mtp-attention-window N` | MTP only: the draft head attends to the first 64 keys and the newest `N` before its query, verification keeps full attention (see [serving](serving.md#mtp-attention-window)) | `0` (whole history) |
| `--lookup-ngram N` | context-lookup drafting alongside `--spec`: the last `N` tokens are matched against the sequence so far and what followed is proposed; exact, since verification rejects a wrong guess | `0` (off) |
| `--ngram-draft-tokens N` | copy drafting alongside `--spec`: up to `N` tokens (1..63) copied from earlier prompt, tool-result or output text that the last `--ngram-min-match` tokens match, verified by the target; `0` disables it; see [Ngram copy proposals](ngram.md) | `15` with `--spec`, else `0` |
| `--ngram-min-match N` | shortest match a copy is drawn from, `4..64` | `12` |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay\|cpu` | `overlay` keeps the Vision tower host-pinned and borrows device memory per image from the evictable text weight tail (no resident Vision cost; needs CUDA VMM). `--vision-offload on\|off` is accepted as an alias for `overlay\|resident`. `cpu` encodes on CPU threads from host FP32 weights, with no device Vision memory and `--vision-max-merged` capped at 256 unless given | `resident` |
| `--vision-cpu` | `--vision` with `--vision-residency cpu` | off |
| `--vision-max-merged N` | merged-token budget of one media item; larger media downscales at preprocessing | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding | off |
| `--post-thinking` | sample the answer with the post-thinking preset (temperature `0.2`) from the token after the reasoning block closes | off |
| `--post-thinking-temperature F`, `--post-thinking-top-p F`, `--post-thinking-top-k N` | post-thinking overrides; each implies `--post-thinking` | preset |
| `--post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F]` | the same overrides in one flag | preset |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |
| `--log-colours on\|off` | `on` gives every statistic of the stderr summary a stable colour, even when stderr is redirected | off |

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture
and rendered prompt mode. The current official models use:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ninfer --help` for the exact option contract.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server, on every model-parallel rank. When unset, CUDA's own default applies (`auto`), a
heuristic that spins while the host has more cores than active CUDA contexts. `spin` always
spins, trading one busy core for the lowest synchronization latency. Use `blocking` to let the
waiting thread sleep; the decode performance cost depends on the host. `yield` yields the CPU
while waiting.

```bash
NINFER_CUDA_SYNC=blocking ./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. The practical allocation
on one RTX 3090 depends on the selected artifact, media workload, output budget, and KV-cache type.
Use `--kv-dtype int8` for the recommended large-context quality profile; BF16 is also available.
The artifact describes its model configuration and weight representations; `--kv-dtype`
independently selects runtime KV storage. All eight
formats — `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `k8v4`, `nvfp4` — are accepted on SM86; measured size,
decode speed and perplexity for each are in
[`docs/config-calculator.html`](config-calculator.html). The Blackwell-only
`mma.sync...kind::f8f6f4` restriction applies to FP8/NVFP4 *weights and activations*, not to KV
storage, which is why this paragraph used to say `fp8` KV was rejected here.

`rk8v4` is the best all-round choice: rotated INT8 keys with a packed signed int4 value plane,
about 23% smaller than INT8 for about 0.082% perplexity, and the flattest decode curve of any
format measured. `nvfp4` buys the most context — 45% smaller than INT8 — at about 13% of decode
speed at a 32K cache depth. `rk4v4` keeps `rk8v4`'s values and stores keys as 4-bit Lloyd-Max
indices: 31% smaller than `rk8v4`, within 3% of `nvfp4`'s size, better perplexity than `nvfp4`
(+0.21% against INT8) and `rk8v4`'s decode speed, so it is the choice when context is the limit.
`rk4v4-e8` has `rk4v4`'s size but snaps each octet of a scaled G64 key group to the nearest E8
lattice point before the codes are clamped to [-8, 7]; the coset bit is not stored, so per-value
error is no better than plain INT4. `fp8` and `k8v4` are each beaten by `rk8v4` on size, speed and
quality together, so neither has a niche. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.

## JSON and JSON Schema output

`--json` constrains output to a JSON object. `--json-schema FILE` constrains it to the supported
JSON Schema subset described in [serving](serving.md#structured-output). Thinking defaults off;
an explicit `--reasoning-effort` or `--thinking-budget` retains reasoning before the constrained answer.
The two format flags are mutually exclusive and reject raw output and custom stops. The CLI adds
the requested format/schema to the model's instructions before tokenization. They work with
ordinary, MTP, DFlash and DFlash2 execution.

```bash
./build/apps/ninfer model.ninfer --prompt 'Return the city as JSON' --json --max-new 128
./build/apps/ninfer model.ninfer --prompt 'Return a weather record' --json-schema weather.schema.json --max-new 128
```

Read the reported finish reason: an output/context limit or cancellation can truncate the JSON.
