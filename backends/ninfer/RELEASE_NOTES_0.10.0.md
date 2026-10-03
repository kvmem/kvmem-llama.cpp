# NInfer-3090 v0.10.0

14 pull requests since v0.9.1 (#88-#101). The headline is decode speed: a run of bandwidth and
occupancy fixes to the existing Q4/Q5 GEMVs, followed by new tensor-core kernels built specifically
for the 1-32 column range MTP verification and cohort decode run at, take Qwen3.8-27B chat decode
from 74.6 to **113.2 tok/s at C1** and 268.6 to **460.4 tok/s at C8** on this card. Alongside that,
five new opt-in flags trade a measured, documented amount of quality for VRAM and context length,
and the context-cache planner finally has RTX 3090 numbers instead of guessing from an RTX 5090
model.

Three PRs in this release are negative results recorded on purpose (#92, #93, and half of #100's
backstory) — a tensor-core MoE port and an 8-warp occupancy fold, both measured and reverted. That is
consistent with how this fork operates: a faster kernel is not a faster model until an end-to-end
paired A/B says so, and the ones that don't move the needle are worth writing down so nobody tries
them twice.

Everything below was measured on this fork's own hardware (RTX 3090, `sm_86`, CUDA 12.8) unless a
PR says otherwise.

---

## Decode throughput: tensor-core small-T kernels, up to 1.7x

Qwen3.8-27B MTP3 decode cost about 1.5x a plain decode step per round, against 1.14x for vLLM/Marlin
on the same card — the MTP verify round (4 columns) and the cohort round (up to 32 columns) fell
between the single-token GEMVs and the prefill GEMM tiles, with no kernel tuned for that width range.
New tensor-core kernels for exactly 1-32 columns, routed through every Q4/Q5 decode GEMM, bring the
round down to 1.15x.

Same card, before/after (fork v0.9.1 vs this release), greedy unless noted, INT8 KV, MTP3 with the
optimized draft head:

| workload | v0.9.1 | v0.10.0 |
|---|---:|---:|
| thinking-off chat, C1, mean TPOT over 8 prompts | 74.6 tok/s | **113.2 tok/s** |
| same, temperature 0.7 | 72.6 | **112.8** |
| same, C8 | 268.6 | **460.4** |
| reasoning cohort C1 / C2 / C4 / C8 | 63 / 89 / 137 / 221 | **93 / 168 / 271 / 376** |
| `ninfer_bench` plain / MTP3 | 39.98 / 53.96 | **47.15 / 85.29** |
| perplexity, quick corpus | 4.342425 | 4.342425 (bit-identical) |

For reference, the vLLM/Marlin stack this was checked against reports 111-124 tok/s at C1 and 407.3
at C8 on the same metric, at a lower 250 W power cap with 5-8% of its own run-to-run spread — parity
at C1, ahead at C8.

## The bandwidth work underneath it

Before the small-T kernels landed, six narrower changes moved dense 27B decode from 37.44 to
**40.11 tok/s** — 69.0% to **73.9%** of this card's measured 854.2 GB/s read ceiling — each verified
with an interleaved A/B against its own control:

| change | measured |
|---|---:|
| q5 GEMV: 8 weights per lane, not 2 | +3.90% |
| q5 GEMV: exact half2 dequant, per-group scale hoisted out of the loop | +2.67% |
| q4 SwiGLU GEMV, the same widening | +1.95% |
| branch-free Batcher top-8 replaces a divergent insertion sort (35B MoE routing) | +3.15% |
| GDN input projection: route width 8 to the split-K direct path | +2.85% at the C8 cohort |
| `w8_pair` medium tiles: the two sm_86-viable ones instantiated | 13-17% over chunking |

int8 perplexity held to 4.342425372232802 across the branch — identical to master to twelve figures
over 261,167 scored tokens, including the q5 change that reorders float accumulation.

The investigation behind it is the reusable part: a stall counter says where warps are waiting, not
that removing the wait makes the kernel faster. Widening the consume loop exposed a compute bound;
a half2 bit-trick exposed a *memory-latency* bound; and the pipelining fix that logic predicted next
(`kStages=3`) was then worth exactly nothing (40.12 vs 40.11 tok/s) — measured and discarded rather
than assumed.

The NVFP4 sm_86 load failure also now says what is wrong and what to do, instead of an opaque error.

## MoE selection: the branch-free win that was actually there, and two that weren't

**`sparse_moe_select_top8_warp`'s merge step, branch-free: +2.39% end to end on the 35B-A3B, output
bit-identical.** The eight-element *sort* was already made branch-free; its *merge* — three bitonic
restore stages consuming the sort's output — still ran 60 data-dependent `if (!better) swap` branches
across five steps, on a warp with nothing else scheduled on its SM to hide them. `TODO.md` had
already counted the merge as the dominant cost (100 ops vs the sort's 19) and predicted "roughly 2%"
from an 8-warp redesign; branch-freeing the existing merge in place delivered that whole predicted
gain with no new algorithm — three independent estimates (a sensitivity probe, an earlier branch, and
`master` itself) landed within half a point of each other: +2.85% predicted, +2.53% and +2.39%
measured.

**A tensor-core port of the same routing kernels was investigated and retired.** `sparse_moe` has no
`mma.sync` despite its Q4/Q5 experts running at 2-46 tokens — the shape the small-T decode kernels
were built for. Counters say why that reasoning doesn't transfer here: the FMA pipe never exceeds
~42% and DRAM is the high-water mark on every measured kernel — these routes are memory-bound by a
factor of three on the pipe they already use (5.7-6.5 MAC/byte against ~21 machine balance), and top-8
of 256 experts means each expert's weights are read once and used once — no reuse for an MMA to
exploit. int8 tensor cores would make the imbalance worse while adding quantization error. No code
shipped from this investigation; the finding is recorded so nobody rebuilds it.

**An 8-warp occupancy fold for the routed gather (`d3`) was built, measured, and reverted.** It
raised occupancy from 77.13% to 87.08% and cut the kernel's own duration by 8.6% exactly as designed
— but the four MoE kernels run in a PDL pipeline that overlaps, so `d3` finishing 0.8 µs earlier
doesn't let the next kernel start 0.8 µs earlier. End to end: -0.03% at C1, +0.11% at MTP3, both
noise. It also reordered the shared expert's FP32 accumulation for zero gain, so it was reverted. A
separate check confirmed the routed gather itself reads no extra bytes (`dram__sectors_read` × 32 B
equals `dram__bytes_read` exactly) — there was no coalescing or relayout win available either. Three
geometries on this gather have now been measured; all three are negative.

## New quality-trade flags, off by default

| flag | what it does | measured |
|---|---|---|
| `--gdn-state-fp16` | FP16 GDN recurrent-state storage | free within noise, +2.0% real-text C8, halves the host state image |
| `--lm-head-q4` | int4 vocabulary head, requantized at load | +3.2% real-text C8, ~0% C1, costs +0.69% perplexity and alters greedy output from ~char 210 |
| `--mlp-a8-decode` | int8 tensor-core MLP gate_up for 16-32 column cohort widths (27B) | +1.28% at C8 with MTP3; 0.008-0.037 relative L2 against an FP64 oracle; perplexity can't see it (scores at prefill widths) |

The first read on `--lm-head-q4` was +41% at MTP3 — a measurement artifact from `ninfer_bench`'s
default corpus, which is 98.4% repeated bigrams and inflates speculative acceptance. Real numbers
came from `tools/bench/run_chat_decode.py` on actual prompts, which gained an `env=K=V` arm field so
env-gated trades can be A/B'd on one binary without losing the interleaving this card's 3-5%
process-to-process drift requires. Full method and the per-trade verdict are in
`docs/maintainer/quality-trade-experiments.md`.

`--mlp-a8-decode` also corrected a documented premise: `docs/performance.md` and `TODO.md` called C8
tensor-rate bound at ~68% of bf16 MMA peak. Counters say otherwise — tensor pipe, L1/TEX and DRAM sit
within three points of each other at 31% occupancy on the bf16 kernel, which is latency-bound, not
compute-bound. The int8 route's own counters are the evidence: it halved tensor-pipe pressure as
designed and returned only 4-6%, because that pipe was never the constraint.

## VRAM savings: transcode weights at load, spend it on context

A new load-time transcode path (`src/artifact/transcode.*`) requantizes W8G32 vocabulary and expert
weights to a narrower group-64 width while the artifact materializes, so the weights arena actually
shrinks — the on-disk artifact is unchanged, and the overlay eviction mirror only ever sees the final
bytes. Quality was pre-measured without writing a kernel, by fake-quantizing W8-layout codes; the real
transcoded runs matched those perplexities to every printed digit.

New flags (`ninfer`, `ninfer-serve`, `ninfer-perplexity`; all opt-in, off by default; recorded in
`server_start`):

| flag | model | saves | perplexity (`--quick`, rk8v4) |
|---|---|---:|---:|
| `--embedding-q4` | qwen3.8-27b | 644 MiB, +24.3K tokens of context | 4.346413 → 4.343738 (−0.062%) |
| `--lm-head-q6` | qwen3.8-27b | +12.9K tokens | 4.346413 → 4.346931 (+0.012%) |
| `--embedding-q4 --lm-head-q6` together | qwen3.8-27b | **+37.2K tokens** (≈182K → ≈220K largest single-lane context) | 4.344221 (−0.050%) |
| `--embedding-q6` | qwen3.6-35b-a3b | 143 MiB, ≈18K tokens | 4.373904 → 4.375419 (+0.035%) |
| `--mtp-experts-q4` | qwen3.6-35b-a3b only | ≈350 MiB (draft layer's routed experts to the same Q4G64/Q6G64 pair every text layer already uses) | doesn't run under perplexity; decode −0.5% (296.29 → 294.93 tok/s), acceptance 61.17% → 60.55% |

`--embedding-q4` and `--embedding-q6` are mutually exclusive; pick Q6 where Q4's perplexity cost is
measurable (the 35B-A3B) and Q4 where it isn't (the 27B). `--lm-head-q4` and `--lm-head-q6` are
rejected outright, rather than silently ignored, on DFlash/DFlash2 (which read the W8 head directly)
and on the 35B-A3B (whose head is already Q6G64).

**`--lm-head-q4` was quietly broken before this release**, and is fixed here: it requantized on the
GPU after upload but left the original W8 planes allocated, so it froze no memory at all, and it was
silently skipped under overlay vision. It now goes through the same load-time path as the others,
frees 644 MiB, and works with overlay.

**Overlay vision has a real floor these flags can cross.** Its exclusive fallback borrows its encode
window from the evict-ranked weights (head, embedding, draft, MTP). On the 35B-A3B those total
about a GiB against a 960 MiB window at default `--vision-max-merged`, and a Q6 embedding leaves them
16 MiB short. Startup now refuses up front with a message naming the shortfall and every flag that
would fix it, instead of failing deep inside vision handling later.

**Launcher defaults moved up** now that the flags exist to back them: Windows' qwen3.8 maxctx
launcher default context rises **131,072 → 163,840 (+25%)** with `--embedding-q4 --gdn-state-fp16`,
and the 35B-A3B C1 maxctx launcher's default rises **114,688 → 147,456 (+28.6%)** with
`--mtp-experts-q4 --gdn-state-fp16`. Both changes were measured rung by rung with arms alternated to
control for desktop VRAM drift, and both READMEs are updated to match.

## RTX 3090 context-cost presets

The context-cache materialization planner priced every decision — keep, demote, or evict a cached
conversation, and how much search a decision is worth — from a compiled RTX 5090 model, because no
3090 entry existed. On this card that model was badly wrong: it predicted **26.0 s** for a
55,000-token Qwen3.8-27B prefill that actually takes **57 s**, and it assumed roughly 40 GB/s for
host↔device transfers against a measured ~23 GB/s on this card's PCIe path (device-to-device copies,
by contrast, measured about 4x *faster* than assumed).

A compiled `nvidia-geforce-rtx-3090-sm86` entry replaces the generic fallback, fit from
`ninfer_context_cost_bench` across all three groupwise-int artifacts at up to 65,536 tokens of
context (so the attention term, which dominates at long prefixes, isn't extrapolated). Every fit
passed its own held-out acceptance check; the worst held-out prefill point was 7.4% off, most within
0-2%. `ninfer-serve` startup now logs `transfer_source: compiled-default, prefill_source:
compiled-default` on this card instead of `generic-default`. An A/B against the six-session
55K-context rotation case showed identical decisions either way — on this card, with Windows capping
pinned host KV at a few hundred MiB, device capacity already decides that case — so the value here is
a correct cost model for platforms where host-KV restore is real (Linux, or a larger host tier), not
a behavior change on this one.

## Serving: `ignore_eos`, and an MTP graph topology fix

Ports two upstream Neroued/ninfer PRs (#197, #221), keeping their original authorship, plus what our
Windows build needed to accept them:

- **`ignore_eos` on Chat Completions.** `ninfer-serve` accepted the field and silently ignored it, so
  a request with `max_tokens=512` that would naturally stop at 267 tokens returned `finish_reason:
  "stop"` at 267 instead of running to 512 — which is what SGLang and vLLM do for the same request,
  and what every cross-engine throughput comparison assumes. `stop.include_model_defaults` is now set
  to `!ignore_eos`; caller-supplied stop strings and stop token ids are unaffected, the default stays
  `false`, and only `/v1/chat/completions` is touched (`/v1/messages` and `/v1/responses` are not).
- **MTP graph profiles now carry a topology class.** Upstream's fix keeps a CUDA graph executable from
  covering both the prompt route and the chunked small-T attention route at once at a draft window of
  6 or more, which fails startup with `cudaErrorGraphExecUpdateFailure`. This fork's MTP cap is 5, so
  the bug itself can't be hit today — but the fix is in ahead of it, and a new test builds against the
  pre-fix code to confirm it actually catches the failure at k=6.
- The upstream test as written links `CUDA::cudart`, which collides with the `cudart_static` target
  `ninfer_core` already links on MSVC (`LNK2005`); it now links `${NINFER_CUDART_TARGET}` instead.

## Upstream sync: materialization planner, adapted for MSVC

Caught up two commits with `neroued/master` (through `d4929686`): a fix for a missing `bf16` include
in the Q4 top-k kernel, and a rework of the context-cache materialization search's guidance
comparator and planning budgets. Upstream builds with GCC/Clang, so this needed two fork-specific
fixes to land clean: the new comparator used `__uint128_t`, which MSVC doesn't have, ported to this
fork's existing `core::wide_multiply`/`wide_less` 128-bit helpers with identical results on the same
`uint64_t` operands; and a test that modeled 1 ms per fake assessment with `sleep_for` failed on
Windows' ~15.6 ms timer tick, fixed by busy-waiting the fake instead. A performance A/B against master
across TTFT resource cases showed differences inside this card's 3-5% process-to-process drift, with
one exception worth naming: the new planner's admission policy stops earlier under pressure
(`insufficient_expected_gain` after a 5 ms initial grant, versus the old planner hitting its full time
budget every time), which is a real behavior change even though the two builds made the same
resume/evict decisions in this A/B. The one thing this exposed — the planner pricing decisions from
`generic-default` coefficients on this card — is what the context-cost presets above resolve in the
same release.

---

## Known issues

- **NVFP4 and K8V4 KV cache require an `sm_100a`/`sm_120a` GPU.** Unchanged from v0.9.1: recognised
  and parsed, rejected with that diagnostic on sm_86/sm_89 — the conversion needs Blackwell-only PTX.
- **A 0.019% perplexity drift against upstream remains open**, scoped to a bisect not yet run;
  `TODO.md` §3 has the constraints and the candidates already eliminated.
- **Real-model tests skip without `NINFER_*_WEIGHTS`**, and a skip is not a pass.
- **Speculative decoding is not bit-identical to width-1 greedy decode**, because verification
  evaluates k+1 columns in one pass where plain decode evaluates one, so a near-tie argmax can flip
  reduction order. MTP reproduces this identically, so it predates DFlash2 and is unrelated to any
  change in this release.
- **`--mlp-a8-decode` has no perplexity coverage.** Scoring runs at prefill widths, so
  `ninfer-perplexity` cannot see a decode-only trade; its usage text says so.

## Upgrading

Drop-in. No artifact re-conversion, no configuration changes, and every new flag in this release is
opt-in and off by default. Existing launcher scripts pick up the new, higher context defaults
automatically; pass the previous context explicitly if you need the old ceiling.

**Worth turning on:** `--gdn-state-fp16` is a clean win with no measured quality cost — free at C1,
+2.0% real-text at C8, and it halves the pinned host state image. `--embedding-q4`/`--embedding-q6`
and `--lm-head-q6` are similarly close to free; `--lm-head-q4` and `--mlp-a8-decode` trade a small,
documented amount of quality for more.
