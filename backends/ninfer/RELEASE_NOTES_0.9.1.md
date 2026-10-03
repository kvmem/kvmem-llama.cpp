# NInfer-3090 v0.9.1

**181 commits, 159 files** since v0.9.0. No new capabilities: this release is one week of
measuring what v0.9.0 shipped, and it moves numbers on the paths people actually run. The largest
single win is a **flag change** — DFlash2's recommended draft count was wrong by 22.6% — and the
largest kernel win is a route table that was **52.8%** off on this hardware at one width.

Everything below was measured on this fork's own hardware (RTX 3090, `sm_86`, CUDA 12.8). Where a
number comes from upstream's RTX 5090, it says so.

---

## DFlash2: use four draft tokens, not seven

v0.9.0 shipped DFlash2 and this repository recorded it as *costing* ~20% against no speculation on
text. Both the recommendation and the claim were wrong, and the reason is worth more than the
numbers: they were measured on `bench/fixtures/bench_corpus.ids`, which holds 65,536 tokens drawn
from **682 distinct ids with 98.4% of its bigrams repeated**, because it is a curated bank tiled to
length. The manifest asserted that repetition "fills length only and does not bias throughput" —
true for prefill and plain decode, and false for anything that drafts. Swept through that corpus,
DFlash2 reports **100% acceptance at every draft count** and decode rises to 159 tok/s on free
tokens. None of that is a statement about text.

Re-measured through the serving path on the model's own generated prose — 27B DFlash2 artifact,
INT8 KV, greedy, 256 generated tokens, mean of three runs, spread ≤0.2 tok/s except one row:

| `--draft-tokens` | decode | vs none |
|---:|---:|---:|
| (none) | 37.7 tok/s | — |
| 1 | 49.7 | +31.7% |
| 2 | 56.4 | +49.5% |
| 3 | 58.4 | +54.9% |
| **4** | **59.1** | **+56.5%** |
| 5 | 57.2 | +51.5% |
| 6 | 48.4 | +28.3% |
| 7 | 48.2 | +27.7% |
| 10 | 42.4 | +12.4% |
| 12 | 40.6 | +7.6% |

**DFlash2 is a win at every count from 1 to 12.** Four is 22.6% faster than the seven
`docs/cli.md` used to recommend -- `59.1 / 48.2 - 1`, the throughput speedup this repository's
methodology defines. Stated the other way round, seven gives up 18.4% against four; both describe
the same pair and the denominator is what differs, and `docs/cli.md` now says four and carries this table. MTP3 with the draft head
reaches 62.4 tok/s on the same measurement, so the gap to MTP is 5% at DFlash2's best count rather
than the 36% previously recorded. `--lm-head-draft` is within noise of unset for DFlash2 at every
count.

`scripts/sweeps/dflash2-draft-tokens-realtext.ps1` is the sweep that does not use the tiled corpus,
and the corpus manifest and generator docstring no longer make the claim.

---

## Route tables re-measured for `sm_86`, again

v0.9.0 re-measured four tables. This release does the rest, and the two largest were wrong by more
than anything found in v0.9.0.

**`w8_pair` at k=2048: up to 52.8%.** The largest route table in the tree — 37 routes on the 35B
DFlash pair — had never been measured here. Eight of those routes were not merely mistuned, they
were *structurally unable to win*: under `NINFER_SM8X_COMPAT`, defined on every sm_86/89 build,
`w8_pair_splitk_medium_launch` does `(void)schedule` and loops the exact kernel over ≤32-column
chunks, so all twelve `DualSplitKMediumC*` schedules are the same kernel here. Confirmed before
changing anything: four swept medium entries read exactly 24.576 µs each at T=33.

| T | 80 | 96 | 112 | 128 | 160 | 176 | 192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| before (µs) | 44.0 | 47.1 | 58.4 | 62.5 | 75.8 | 88.1 | 91.1 |
| after (µs) | 39.9 | 41.0 | 39.9 | 37.9 | 46.1 | 46.1 | 43.0 |
| gain | 9.3% | 13.0% | 31.6% | 39.3% | 39.2% | 47.7% | **52.8%** |

(cold, median of 9, public Op. Medium still wins at T=64 and loses from T=66 on.)

**Q4 SwiGLU's `Materialized` bands: up to 23.1%.** Upstream alternates `Materialized` and the c128
tile three times across 49..640, which is implausible on its face — a winner does not flip back and
forth over a contiguous range — and it does not survive measurement. `Materialized` lost at every
width in `{49,128}` and `{257,384}` in every run (23.1% at T=257, 15.2% at 320, 8.7% at 384). With
`{513,640}` also re-measured on an idle card at 31–51 repetitions, the table collapses from **ten
routes to five**: 49..∞ is now one route and the alternation is gone entirely.

**GDN input projection: two tiles the table did not have.** This was the only route table in the
repository with no schedule bench, while being implicated in two separate slowdowns. It has one
now (`bench/ops/q4_q5_gdn_input_schedule_bench.cu`). First result was a negative one worth having —
every existing boundary (6/7, 32/33, 64/65) sits exactly where the curves cross. Second was that
`R64C8` and `R64C16` were missing: c8 wins 7..8 by 10.7–11.5% and c16 wins 9..16 by 8.8–9.2%, each
collapsing one column past its own width because a second pass costs a whole extra weight read.
End to end that is **+5.7% on the C8 serving profile** (3 of 3 non-overlapping) and **+5.3% on
DFlash2 at k=6** (6 of 6 paired runs), lifting C8 decode from 132.9 to **140.8 tok/s** — 3.83× C1
rather than 3.62×.

**`SmallTMaximumSplits` removed rather than extended.** The device asked for the bump for every
quantized storage while the host granted it to `Fp8E4M3Row256` only, which read like the host
shortchanging nvfp4 and k8v4 at exactly the depths where they measure worst. Measured, it is the
other way round: extending the grant cost nvfp4 2.3–3.1% and k8v4 2.4–2.8% at 4K–32K depth, and
removing it from fp8 as well made fp8 *faster* (+0.4–0.5%), with an unchanged control holding to
0.1–0.3%. More splits at depth is worse.

One negative result is recorded beside its route table so it is not retried: a narrow-extent
`R64C8` was built for `q5_linear_add` — the tile both entries above asked for — and it loses at
every width, 6–32% behind c16. Narrowing a tile helps when padding is the cost and hurts when
bandwidth is; that kernel is already weight-read-bound.

---

## Correctness fixes in paths that produce wrong output rather than an error

- **A missing `__syncthreads()` between `cp_wait<0>()` and the shared-memory dequant** in five
  quantized causal-attention kernels. `cp_wait<0>()` retires only the *calling thread's* cp.async
  group, and `dequant_k_tile()` then walks the whole tile with every thread strided across it — so
  threads read tiles other threads had not yet landed. This is the race behind the decode
  non-determinism first noticed on the three KV formats that stage dequantized tiles in shared
  memory.
- **Speculative rounds validate the whole committed span, not just the terminal column.** Columns
  before the divergence point are accepted *because* the target argmax matched the draft; if such a
  column's logits are all NaN that argmax is arbitrary, so the match is meaningless and the round
  licenses a token the target never endorsed. All five greedy sites now check `[0, accepted_count]`
  — the sparse warp routes with a single `__ballot_sync`, the greedy+penalties path inside the
  column loop it already runs, and both dense multiblock routes through a shared helper. Reverting
  only the sparse path to terminal-only gives 40 failures, all 40 in the new matched-column cases.
- **`DeviceBuffer::copy_from_host` was invisible to every stream the engine owns.** For a
  host-to-device copy out of *pageable* memory, `cudaMemcpy` returns once the source is staged for
  DMA; that trailing DMA completes on the legacy stream, and every stream `DeviceContext` creates is
  `cudaStreamNonBlocking` and therefore exempt from legacy ordering. "Stage the inputs, then launch
  on the engine stream" was a race — and it is the sequence every Op test's setup uses. Host uploads
  now complete before returning.

---

## Serving and operational fixes

- **Serve answers 503 while the model loads** instead of accepting connections nothing replies to.
  The socket is bound before the Engine is constructed on purpose, so a port clash fails in
  milliseconds rather than after ten seconds of weight loading — but that left a window where, on
  the 27B, the first TCP connect was accepted at **0.27 s** and the first HTTP 200 arrived at
  **9.5 s**, with everything in between timing out. A TCP readiness probe called that ready.
- **The pinned host KV cache is sized against free VRAM, not host RAM.** Every real-model test on a
  24 GB card failed at startup with `cudaMallocHost failed to pin 8192 MiB of host memory (this is
  system RAM, not VRAM)` while 20 GiB of system RAM was free — because on Windows/WDDM a pinned host
  allocation is mapped into the GPU's address space and charged against the card. The parenthetical
  was the bug. `docs/` now states what `--host-kv-mib 8192` actually gets, which differs by platform.
- **Pending CUDA errors are no longer swallowed** on the pin's success path, and the startup
  `context cache` line prints what was actually pinned rather than the requested target.
- **Launchers are overridable and no longer bind the LAN by default**; the two `-maxctx` launchers
  parse on Linux now; and `ninfer-serve` says which knob to turn when a pinned allocation fails.
- **Downloaders**: revisions pinned and verified (including the 35B revision that actually carries
  DFlash, and a Qwen3.6-27B target), artifacts staged under the revision so a resumed fetch cannot
  splice two revisions together, and the final move is checked instead of assumed.

---

## Host-side performance

Two changes on the path every request takes before the GPU sees it, both measured on the product
tokenizer read out of a real artifact (248,044 tokens, 247,587 merge rules), arms interleaved in one
round with one process per point:

- **BPE merges live in a flat open-addressed table.** `Tokenizer::encode` looks a rule up once per
  adjacent symbol pair of every word, and every lookup used to indirect into a separately allocated
  node. The rules never change after load and their count is known before the first insert. Rank
  order is load-bearing and commented as such: with linear probing the first claimant keeps a slot,
  and a low rank is a merge the encoder performs often.
- **NFC normalisation is skipped when the text is already pure ASCII.** Every byte below 0x80 is a
  starter with combining class zero, so pure ASCII is already in NFC. The test reads raw bytes
  rather than decoded codepoints, which is what makes it safe — every byte of a multi-byte UTF-8
  sequence is ≥ 0x80, so no non-ASCII input can reach the fast path, well-formed or not, and the
  rejection contract for invalid UTF-8 is unchanged. `normalize_nfc` falls to 0.014–0.016 of its
  cost on a 32K-token chat (0.7948 ms → 0.0111 ms) and host encode to 0.773–0.866 over nine
  pure-ASCII fixtures.

---

## MoE prefill

The routed gate/up is the largest kernel of the MoE prefill operation and the one that waits: on the
35B-A3B routing at a 256-token chunk it holds 69.5% of the operation's kernel time while reaching
**38.5%** of this card's measured read ceiling, next to a routed down kernel launched over the same
jobs that reaches **85.5%**. Neither is near compute bound. The narrow routed gate/up goes from two
`cp_async` stages to six — 32 k-tiles against a 32-column B tile, so six stages fit the 48 KiB
static shared limit exactly at 49,152 bytes — and the routed grid is sized from the number of work
items instead of a fixed three blocks per SM, capped at 32 blocks per SM. Two L2 hints were added
alongside: the shared-expert down weights are prefetched from the D1 tail, and the next projection
is warmed from the MoE down tail with the hint issued before the block barrier.

---

## Measurement tooling

Most of this release's evidence needed instruments that did not exist, and getting a wrong answer
first is what produced several of them.

- `--spread` in `schedule_sweep.cuh` and the Q4 SwiGLU bench. `ColdTiming` had carried `min_us` and
  `p95_us` all along and the harness threw both away, printing the median alone — so the 19.6%
  spread that made one boundary undecidable was invisible in the table.
- A **decode-step profile** and a MoE **prefill pipeline-depth sweep**; the decode roofline got a
  denominator that is actually right, and the busy-time calculation unions kernel intervals instead
  of summing overlapping ones.
- Sweeps create their own output directory, no longer hardcode this checkout's path, capture the
  numeric KV capacity, and refuse to report a stale perplexity or trust a stale CSV.
- A **fit calculator** for KV format, context and speculation, cross-checked against what the engine
  actually reserves per mode.
- The serving benchmarks stop the server **by signal on Windows** rather than `TerminateProcess`,
  which is what previously aborted a campaign mid-run.
- CI runs the host-only checks, and the Linux script guard covers what it claims.

---

## Build and release

- **The build scripts can cut a release.** Both configured `NINFER_BUILD_BENCHMARKS` at its `OFF`
  default while every packager ships `bench/ninfer_bench`, so packaging failed *after* the whole tree
  had compiled. `--package`/`-Package` now implies the option, and `build.ps1` sets
  `NINFER_BUILD_ROOT` for the packager so `-BuildDir` and `-Package` agree.
- **Linux binaries no longer embed the builder's absolute paths in host code.**
  `-ffile-prefix-map` covers C, C++ and the host half of every CUDA translation unit, which is where
  the v0.9.0 Linux binaries carried `/home/ash/ninfer-rel/src/...` about 200 times. Two exclusions,
  both measured rather than assumed: **device-side `__FILE__` comes from nvcc's own frontend and has
  no remapping flag**, so `.cu` paths remain — 395 of them in this release's `ninfer-serve`, all
  device-side — and MSVC has no working equivalent (`/d1trimfile:` is undocumented and had no effect
  on `__FILE__` at 14.44.35207), so the Windows binaries keep theirs.
- The line-ending policy is enforced repository-wide rather than in `scripts/` only.
- **The archive now ships a README that describes the archive.** Both packagers previously copied a
  checkout-oriented document — the Windows one still titled v0.6.1, the Linux one a guide to
  *building from source* — and both point at `scripts/download-*`, while the packagers copy those
  scripts to the archive root. Anyone following the packaged README hit a missing-file error. There
  are now two archive READMEs written for the flat layout, with absolute links, since the archive
  ships no `docs/` directory.
- **`SHA256SUMS.txt` in the Windows archive is LF.** `Set-Content` wrote CRLF, so `sha256sum -c`
  looked for filenames ending in a carriage return and reported `FAILED open or read` for every
  line. The Linux list is written by `sha256sum` itself and was always fine. Both files now verify
  with the standard tool; `Get-FileHash` was never affected.

---

## Known issues

Unchanged from v0.9.0 and stated here so they are not rediscovered:

- **NVFP4 and K8V4 KV cache require an `sm_100a`/`sm_120a` GPU.** They are recognised and parsed,
  and rejected with that diagnostic on sm_86/sm_89 — the conversion needs Blackwell-only PTX.
- **A 0.019% perplexity drift against upstream is open** and scoped to a bisect, not yet run. It is
  a repository question rather than a hardware one; `TODO.md` §3 carries the constraints and the
  three candidates already eliminated.
- **Real-model tests skip without `NINFER_*_WEIGHTS`**, and a skip is not a pass. Capacity
  shortfalls also skip rather than fail: a desktop holding a couple of GB is enough to tip a
  20.8 GiB model over.
- **Speculative decoding is not bit-identical to width-1 greedy decode.** Verification evaluates
  k+1 columns in one pass where plain decode evaluates one, so the reductions run in a different
  order and a near-tie argmax can flip. MTP reproduces it identically, so it predates DFlash2.

---

## Upgrading

Drop-in. No artifact re-conversion, no configuration changes, no CLI changes to existing flags.

**One thing worth changing by hand:** if you run `--spec dflash2 --draft-tokens 7`, use
`--draft-tokens 4` instead. That is 22.6% more decode throughput on the same hardware and
artifact.

---

## Validation

`ctest` on this RTX 3090, `sm_86`, CUDA 12.8: **126 tests, 126 passed, 0 failed** in 946 s, with
seven real-model tests skipping for absent `NINFER_*_WEIGHTS`. `scripts/check-linux-scripts.sh`
passes, which covers the shipped Linux launchers and downloaders end to end against fixtures.

`ninfer_gdn_gating_proj_test`, which this repository recorded as a standing sm_86 failure on two
cases, **passes** — worst `relative_l2` ratio 0.474 against its limit on the case that previously
read 1.21, over three consecutive runs. The limit itself is unchanged, so the kernel moved, not the
tolerance; the note recording it as failing was stale rather than describing this release.
