# NInfer-3090 v0.8.0

This release makes shared-prefix reuse work without the client asking for it.
Three defects kept the context cache from publishing anything a *different*
request could match, so a fleet of requests sharing a long preamble recomputed
their whole prompt every time. It also adds an opt-in mode that lets unrelated
callers share a prefix with no protocol hint at all, and ships native Windows
and Linux x64 binaries on the same explicit SM86 runtime profile as prior
releases.

Every figure below is measured on this machine against the Qwen3.6-35B-A3B
artifact, direct to the server's bound port with no proxy in the path.

## Changes

- **OpenAI requests no longer disable every shared-prefix candidate.**
  `apply_openai_prompt_cache_policy` set `allow_engine_automatic_shared_prefixes`
  to false on every request, following the rule that a protocol carrying its own
  write policy should suppress the Engine's structural candidates. That rule does
  not hold for OpenAI: the only marker the policy places sits at the end of the
  request's own prompt, which no request with a different tail can ever match. So
  nothing was ever published. With a 1554-token prompt whose shared span is a
  leading system message, and no client hint of any kind, reuse goes from **0 of
  10 requests to 1529 of 1554 prompt tokens (98.4%)**.

  The equivalent line in the Anthropic path is guarded by an explicit client
  opt-in and is unchanged.

- **The protocol's end-of-prompt marker no longer takes shared catalog slots.**
  Both OpenAI and Anthropic place their automatic marker at the end of the
  request's own prompt, so every request's copy sits at a frontier no differing
  request can match. It was still allowed to occupy a spare slot, which filled
  the catalog with single-use checkpoints and starved the genuinely reusable
  structural candidate. A *larger* catalog made this worse, because it offered
  more vacancies to waste. `DefaultAutomatic` now has to earn its slot the way
  `EngineObserved` does — two distinct reuse domains asking for the same key —
  which is exactly the case where a conversational endpoint really has become a
  common prefix.

  Eight distinct ~1430-token preambles round-robined, reuse after the first
  round: at `--max-shared-prefixes 8`, **65.6% to 98.3%**; at 16, **32.8% to
  98.3%**; prefill median at 16 slots **0.305 s to 0.044 s**. The 16-slot catalog
  previously lost to the 8-slot one. Both now behave identically.

- **Adds `--auto-prefix-grid`** (off by default). Structural boundaries only
  expose a prefix where the prompt's own shape happens to put one, so two callers
  who merely start with the same long span — the same pasted document, the same
  few-shot preamble inside a single user message — share no boundary and never
  propose a frontier the other could match. The grid offers candidates at
  absolute multiples of a 256-token stride that doubles until the grid fits in
  eight points; absolute positions rather than offsets from the end, so prompts
  of unrelated lengths still name the same frontier wherever they still agree.
  Grid points carry `EngineObserved` only, so one is materialized solely once two
  independent callers have both proposed it and it can never speculatively
  consume a vacant slot.

  On a prompt whose shared span sits inside a single user message, with no client
  hint: reuse **0% to 82.6%** steady state, prefill **-71%**, TTFT **-64%**. The
  cold requests before it warms are unchanged (0.323 s vs 0.322 s prefill), so it
  costs nothing measurable before it pays off.

- **`scripts/run-qwen36-35b-a3b-c1-maxctx.bat` and its new `.sh` counterpart**
  ship tuned cache defaults: `--max-shared-prefixes 8` (was 4),
  `--host-state-slots 32` (was 16), and `--auto-prefix-grid` on. Four shared
  slots thrashed at **8.4%** reuse against an eight-preamble working set. This
  costs no device memory: verified at 114,688 context on the same profile, KV
  still resolves 1,792/1,792 pages, runtime reservation is unchanged at 1.22 GiB,
  and free-after-startup is byte-identical at 492.4 MiB. The price is pinned host
  memory, 982.6 MiB to 1.92 GiB.

## Sizing note

A context-cache checkpoint is a KV prefix plus a StateImage. On this model the
StateImage is **61.4 MiB flat** — 30 GDN layers of 128x128x32 FP32 recurrent
state plus conv — regardless of prefix length. Unlike KV pages, which several
checkpoints of one conversation share, it cannot be shared between two frontiers
at all, because the recurrent state at token N is a function of every token
before it. That fixed cost is why the catalog is small.

Size `--max-shared-prefixes` to the number of distinct preambles in play and
`--host-state-slots` to roughly twice (shared + private). Both cost pinned host
memory, not device memory.

## Long conversations

Long agentic sessions do not depend on any of the above; they run on the private
turn-closure path, sized by `--max-private-continuations`. Measured with a
194-turn coding-agent loop — tool calls, file pastes, whole history resent each
turn — growing to 64,668 tokens: **exactly one cache miss, on turn 1**. Computed
prefill stayed flat at ~340 tokens per turn regardless of history length, and
TTFT went 0.085 s at 3k tokens to 0.177 s at 64k. Two interleaved agents showed
the same picture. The limit on turn count is `--max-context`, not the cache.

## Validation

- `ctest` 110/110 on the release build.
- Windows: Visual Studio 2022 BuildTools and CUDA 12.8
- Linux: WSL2 Ubuntu 24.04, CUDA Toolkit 12.8, GCC 13, CMake, Ninja
- Both built for `CMAKE_CUDA_ARCHITECTURES=86`.
