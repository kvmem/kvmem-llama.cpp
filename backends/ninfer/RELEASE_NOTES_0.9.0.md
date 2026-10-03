# NInfer-3090 v0.9.0

The largest release this fork has cut: **166 commits, 390 files**. Two headline capabilities — a
second GPU that turns into KV cache, and DFlash2 speculative decoding that previously could not
start at all — plus a systematic re-measurement of the kernel route tables this fork had been
inheriting from RTX 5090 tuning.

Everything below was measured on this fork's own hardware (RTX 3090, `sm_86`, CUDA 12.8). Where a
number comes from upstream's RTX 5090, it says so.

---

## Two GPUs: 8.1x the KV cache

`--devices 0,1` materializes each layer's expert/MLP block on the second GPU. Rank 0 keeps
everything else — embeddings, attention, GDN, norms, the head — and therefore keeps the KV cache,
the GDN recurrent state and the context cache. **Every byte rank 0 sheds becomes KV.**

| | max total KV | |
|---|---|---|
| single GPU | 237,248 tokens | 262,144 context **fails to start at all** |
| `--devices 0,1`, C=8 @ 262k | **1,929,728 tokens** | 8 sessions at ~241k each, 1.08 GiB spare |

For Qwen3.6-35B-A3B, rank-0 weights drop from 19.6 GiB to **2.15 GiB**, and memory free for KV
rises from 3.71 GiB to **21.1 GiB**. Its MoE is ~88% of its weights, so it divides especially well;
the 27B divides less dramatically (15.9 → 6.79 GiB) because its attention and GDN projections at
hidden 5120 are large next to its dense MLP.

**It is also faster under load, not slower.** Decode at concurrency is bandwidth-bound, and the
split reads expert weights from card 1's memory while reading KV from card 0:

| tok/s aggregate | C=1 | C=4 | C=8 |
|---|---|---|---|
| single GPU | 140.7 | 292.9 | 348.2 |
| `--devices 0,1` | — | 300.1 | **404.8** |
| `--devices 0,1` + MTP3 | **174.8** | **336.6** | 356.0 |

At C=8 the split is **16% faster** than one card. Rule of thumb: `--devices 0,1` alone for
concurrent serving, add `--spec mtp --draft-tokens 3` for single-stream latency.

Measured on a rented bridgeless 2× RTX 3090 (no NVLink; `nvidia-smi topo -m` reports PHB), int8 KV.
**Greedy output is byte-identical to the single-GPU reference** in every split configuration, on
both models, with and without MTP.

---

## DFlash2 speculative decoding now works

`--spec dflash2` previously died at startup with `context_kv_materialize: invalid cache V`.

The cause is worth naming, because it is the same class of bug this fork keeps finding at merge
boundaries. `context_kv_materialize` arrived whole from upstream, so nothing conflicted and nothing
got reviewed. It required **FP16** for the cyclic cache's V plane while requiring BF16 for K. Its
consumer, sliding-window attention, demands BF16 for both — so the two could never have worked
together: either validation rejects it, as it did, or FP16 bits get silently decoded as BF16.

Measured on the real 27B once fixed:

| | acceptance | tok/round |
|---|---|---|
| DFlash2 + `--vision` | 85.7% | 7.00 |
| DFlash2 text | 20.0% | 2.38 |

### Also fixed: the 27B failing to start with `--spec mtp`

If your `qwen3_8_27b.ninfer` artifact carries a `dflash2/` bundle, v0.8.1 could not start it at
all — the artifact binder requires every object to be consumed by the selected target, and the 27B
target had no idea `dflash2/*` existed:

```
FATAL server failed during startup | artifact object was not consumed by the selected target: dflash2/feature_projection
```

That is why `dflash2` appeared in an error from a command line that never mentioned it, and why the
35B was unaffected. The DFlash2 weights are now bound optionally, so they are consumed whether or
not you select DFlash2. **Reported by @miscdebris; fixed in this release.**

---

## Route tables re-measured for sm_86

This fork had been running route boundaries tuned on an RTX 5090. **Every route table the upstream
catch-up changed was re-measured on this card, and all of them were wrong here**, by 12–41%:

| table | inherited failure | best gain |
|---|---|---|
| W8 linear-pair k=5120 (27B KV projection) | SIMT owned 85 columns where this fork gave it 4 | **41%** at T=85 |
| Q5 linear-add (35B residual) | `{49,192}` wrong above ~88 columns | **41%** at k=17408 T=192 |
| W8 attention-input DFlash2 (27B) | small-T routed to 48, losing 35% from T=17 | **34%** at T=32 |
| W8 SwiGLU DFlash2 (27B) | a c128 tile routed past 128 columns | **32%** at T=160 |
| Q4/Q5 attention input | the tuned table was **dead code** beside a hardcoded chain | 502.8 → 226.3 µs at T=16 |

Three failure shapes recurred, each of which looks perfectly reasonable on sm_120: a small-T kernel
routed past its useful range (its *last* tile was always its worst), a c128 column tile routed past
128 columns, and a kernel that never wins anywhere owning a band.

Pointing Q4/Q5 at its own table exposed **two `switch` fallthroughs** that upstream's boundaries had
kept dormant. A fallthrough here runs a second kernel over the first — both compute the same
projection, so the output stays correct and only the cost doubles. Invisible to every correctness
test. `-Werror=implicit-fallthrough` now guards non-MSVC builds so the next one fails to compile.

---

## Causal small-T attention: 13–25% faster decode

Upstream's causal small-T rework had been reverted here (12 files) because it produced
garbage-magnitude output for the entire INT8 family — and INT8 is this fork's default KV dtype.

Root cause: `small_t_i8.cuh` stages V for an **f16** PV MMA but filled that tile using the **bf16**
dequant helpers. Both types are sixteen bits, so it compiled, ran, and reinterpreted every value —
3–11× the reference with a ratio that varied per value. Both affected storages (int8-g64 and rk8v4)
go through exactly those two helpers, which is why the symptom was "the INT8 family" and nothing
else.

With that fixed, upstream's version is adopted and the revert is gone — **13–25% faster at every
shape measured**, on the decode path, for the default KV dtype:

| context | W=1 | W=2 | W=4 | W=6 |
|---|---|---|---|---|
| 512 | 23.6 → **18.4** | 25.6 → **21.5** | 29.7 → **24.6** | 36.9 → **27.6** |
| 2048 | 31.7 → **24.6** | 35.8 → **27.6** | 43.0 → **33.8** | 68.6 → **52.2** |
| 8192 | 68.6 → **52.2** | 77.8 → **59.4** | 95.2 → **82.9** | 112.6 → **84.0** |

(int8, cached decode, cold, B=1, d256-h24-kv4, median µs)

---

## Correctness hardening

- **Speculative rounds can no longer license a token chosen from non-finite logits.** A diverged
  forward pass gives the selected column an all-NaN row; both the dense and sparse commit paths now
  refuse it and return the non-finite sentinel instead of an arbitrary plausible-looking token. The
  check is a *required* argument, so a new call site cannot compile without stating what it checked.
- **One declaration of what a KV plane is.** Every plane cast — producer and consumer, for all six
  cache storages — now derives its C++ type from `d256_kv_cache_profile`. A producer/consumer
  disagreement is a compile error rather than a silent reinterpretation. This is the guard against
  the FP16-V asymmetry that has returned at three separate merges.
- **Published contracts corrected.** Four headers and two maintainer documents advertised FP16 V
  storage that the implementation had not used for some time, including `kv_cache_append`'s — the
  one subsystem whose *code* has always resisted this bug. A stale contract is how it kept coming
  back: a reader trusts the header, writes `__half`, and it is byte-compatible.
- **A test tolerance that was below its own dtype's floor.** The linear-pair gross-error limit
  demanded the single worst element round exactly as the FP32 oracle does — tighter than one BF16
  ULP. Over 330 sampled cases the distribution was bimodal with two outliers sitting on the
  threshold. Raised to a defensible bound; the relative-L2 check that actually constrains a kernel
  is untouched.

---

## Testing and tooling

- **Real-model tests actually run now.** They defaulted to maximum configurations that no 24 GB card
  can hold, so all seven failed or skipped permanently and nobody read them. `ninfer_add_test` gained
  `TEST_ARGS`; the DFlash2 27B test is registered with a configuration that fits and **passes**
  (`accepted=20/20`). With the artifacts present and an idle GPU, three real-model tests pass and the
  rest skip with an explicit reason rather than failing.
- **Capacity shortfalls skip instead of failing.** A busy desktop holding a couple of GB is enough to
  tip a 20.8 GiB model over; that is not a defect, and it no longer shows up as a red test.
- **Real-model failures say why.** These mains had no top-level `try`/`catch`, so an engine throw
  became an unhandled-exception fastfail with *no output at all* — which reads exactly like memory
  corruption and is not.
- **Shared benchmark harness for route boundaries** (`bench/ops/schedule_sweep.cuh`), so retuning an
  Op after a merge costs a ~40-line file. It carries the two traps that produced confidently wrong
  answers first: kernels do not police their column domains, and warm timings pick different winners
  than cold ones.
- **CodePulse reviews trigger from a workflow**, pinned to an immutable commit SHA — the job holds
  `id-token: write`, and a tag can be repointed.

---

## Upgrading

Drop-in. No artifact re-conversion, no configuration changes, no CLI changes to existing flags.
`--devices N,M` is new and optional; everything else behaves as before, faster.

If you hit the `dflash2/feature_projection` startup failure on v0.8.1, this release fixes it.
