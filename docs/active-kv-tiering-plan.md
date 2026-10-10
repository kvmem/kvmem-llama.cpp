# Active KV tiering implementation

Status: P1/P2 common RAM ownership and both backend acceptance gates passed.
P3-P8 portable SSD, active/idle/concurrent correctness, feature combinations,
fault/cancellation, patch reconstruction and Windows dependency packaging completed.
Llama 260K RAM long-prefill and physical cold-SSD/memory-pressure performance
qualification remain open; native Linux GPU verification is unavailable locally.
Measured results and qualifications are
in [the final regression record](active-kv-tiering-validation-20261010.md).
The dated checkpoints below retain earlier results and unresolved observations.

## Scope

Share native KV storage between llama.cpp and ninfer while preserving each backend's
native representation, GPU allocator and scheduler. Complete K/V payloads (including
scales and MTP) may leave RAM. Active mean-K statistics remain separately accounted.
Active tiering and idle session caching share ownership and quotas. Linux and Windows
use the same storage contract. Restart persistence and cross-process sharing are out
of scope.

## Ordered gates

1. P0: freeze source/binaries, fixtures and configuration; establish A/A variability.
2. P1: common ownership, capacity reservations, transfer lifetime and version rules.
3. P2a: llama RAM integration; exact payload/state checks and performance acceptance.
4. P2b: ninfer RAM integration; native planes, Graph/MTP and performance acceptance.
   Both RAM paths must pass before SSD implementation begins.
5. P3: bounded portable file I/O, cancellation, accounting and fault tests.
6. P4: single-request active SSD tiering, one backend at a time; repeat eviction and
   restore with history larger than RAM, and test SSD-disabled/enabled-unused costs.
7. P5: idle session integration without duplicate KV serialization, quota/lease tests.
8. P6: concurrent storage waits and engine-safe completion; hot/cold request fairness.
9. P7: individually measured optimizations only where evidence justifies complexity.
10. P8: remove replaced paths, document support, run final cumulative regression.

## Efficient verification

Every logical change gets an incremental build and affected behavioral tests. Changes
on inference paths also get a fixed performance screen. Backend-local changes primarily
test that backend; shared execution/storage changes test both. Long contexts and broad
combinations run at milestones, not after each edit. Small windows/quotas trigger daily
pressure tests; real long contexts verify scale at acceptance.

Initial fixtures: short C1, beyond-window C1, multi-turn continuation, fixed MTP.
Add concurrency, session exchange, vision and 128K/256K baselines before their stages.
Preserve initial binaries for cumulative comparisons and the last accepted candidate
for attribution. Three paired runs screen changes; expand to seven or more for a
suspected regression or stage acceptance. Alternate AB/BA and retain all samples.
Insufficient evidence does not pass a gate. Tail latency uses enough requests, not
the small number of paired runs.

Investigate sustained 2% throughput/median latency degradation; confirmed >3% blocks
acceptance. Confirmed >5% p95 degradation blocks concurrency acceptance. These are
investigation/acceptance limits, not an intended slowdown allowance. Keep prompt,
output length, budgets, speculation and cache conditions fixed. Report actual SSD
I/O cost separately from regressions of existing RAM tasks; freeze first correct SSD
implementation as the optimization reference.

Performance GPU: RTX 5060 Ti, GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c.
Correctness may use either free GPU. Compilation defaults to eight jobs.
Native Windows, native Linux and WSL evidence are labeled separately.

## Initial provenance

Working branch: feat/active-kv-tiering, based on eb98424e7654c5bdd054bf9ce46485d09a991f84.
Both backends are prepared from backends/versions.json with their recorded patches.
Original workspaces and service binaries are preserved. Local evidence is stored in
the parent active-kv-tiering-20261009 directory; it is not part of the source package.
Packed K/V tiering is already approved (legacy finding L010).

## Storage and scheduling boundaries for the next stages

The portable backing file and bounded I/O queue belong to the common library. A
record owns native bytes and its RAM/disk capacity charges; moving a record from
an active sequence to an idle history does not copy those bytes or create a second
disk authority. Session identity, content versions and the right to publish a view
remain with the existing engine catalog. Mean-K and recurrent checkpoints are
separate from KV payload accounting.

Reserve the destination before writing, retain the source until successful
completion, and expose failure to the transaction. An unfinished write is never a
readable disk copy. Read leases keep bytes alive through DMA, cancellation and
record deletion. In-flight buffers, queue depth and temporary disk destinations
need explicit bounds; neither asynchronous work nor idle retention may bypass
active-request admission. A full disk is an admission/transfer failure, not an
instruction to discard the only copy.

The common worker performs file operations only. It cannot call either backend's
GPU API or publish a session view. In particular, ninfer's Program-wide pinned
staging buffer cannot be held by an SSD task across scheduler turns. Pending reads
must finish into a bounded request-owned buffer before an engine-safe transfer
uses that staging area. Waiting requests retain their transaction and source
ownership while unrelated ready requests continue. Llama lane waits likewise must
not hold a global GPU or scheduler lock.

P3 first verifies portable I/O and its failure/lifetime rules independently. P4
then qualifies actual full K/V spills in one active request. P5/P6 integrate the
same record ownership with idle histories and nonblocking engine waits. These are
implementation requirements, not claims that SSD or concurrent waits already work.

## Current acceptance evidence (2026-10-10)

- Windows host suites: 15/15 passed. WSL/GCC common-storage tests also passed with
  address, leak and undefined-behavior checking. This is not native Linux GPU evidence.
- llama native KV/MTP GPU checks: 32 passed; 10 existing physical-OOM injection cases
  skipped on Windows/WDDM. Byte restoration and recurrent-state comparisons are exact.
- llama seven ordinary pairs with matched original/candidate build flags: outputs
  and token usage match; median total latency changes for short/window/continuation
  are -0.36/+0.14/+0.11 percent. The initial three-pair continuation prefill change
  of +3.67 percent became +0.41 percent after expansion. Total time, TTFT and decode
  one-sided 95% mean-ratio upper bounds are below 3 percent. Internal short and
  continuation prefill intervals remain wider (4.21/3.35 percent upper bounds), so
  this establishes ordinary end-to-end acceptance, not equivalence of every timer.
  One slow original run is retained; no speedup is claimed. Earlier unmatched
  architecture builds remain diagnostic and are excluded from this comparison.
- ninfer native archives and pending allocations now use one Program-owned RAM
  account. Native planes, INT8/eager sessions, RK8V4/MTP3/Graph sessions, eight-way
  adaptive MTP/ngram concurrency and generated endpoint restoration passed.
  The idle disk test is unavailable: both original and candidate reject the existing
  deliberately disabled ninfer cold-snapshot entry before execution. Preserve that
  failed attempt; qualify the replacement in P5. Seven ordinary pairs preserve exact
  output/usage, with median total changes -0.01/-0.08/-0.12 percent; all measured
  one-sided 95% mean-ratio upper bounds are below 3 percent.
- Fixed MTP: llama seven pairs preserve exact output/usage, with median total
  changes -0.48/-0.10/-0.51 percent and every measured upper bound below 3 percent.
  Ninfer initially had a wide short-TTFT interval after seven pairs (median +0.96,
  upper +3.68 percent); four additional pairs resolve this without dropping samples.
  Eleven-pair short TTFT is +0.60 percent, upper +1.95 percent. Total changes are
  +0.05/+0.10/-0.16 percent; all output/usage is exact and all measured upper bounds
  are below 3 percent. The RAM gate passes for these fixtures. Accepted binaries
  are frozen in `baseline/ram-accepted`; long-context/concurrent SSD acceptance is
  still pending and is not implied by RAM acceptance.
- Legacy finding L012 reproduced silent V-copy loss when the old asynchronous NVMe
  tier fills. On 2026-10-10 the user approved reserving destinations before releasing
  the only RAM copy, publishing only successful writes, and preserving valid bytes
  on failure. Apply this rule to the replacement common layer and old entry points;
  the old entry is now fixed in source and passes Linux/WSL capacity/write-failure
  regressions, address/undefined/leak checks, and Windows 15/15 host tests. Failed
  writes retain RAM bytes and report errors; clear drains and recovers. Inference
  benchmark binaries above predate this isolated legacy repair. The replacement
  common SSD layer still requires its own capacity and I/O failure qualification.
- P3 common primitives now pass Windows 17/17 host suites. `spill_io_test` checks
  arbitrary byte ranges, offsets beyond 4 GiB, independent files, concurrent ranges,
  extent reuse, queued cancellation, exception propagation and destruction drain.
  `tiered_kv_storage_test` checks histories larger than RAM, exact cold reads,
  partial rewrite, pinned/mutable exclusions, shared owners, SSD-full refusal and
  release after queued reads. WSL/GCC address/undefined/leak checks pass, including
  actual RLIMIT_FSIZE write failures that preserve the sole valid RAM copy and allow
  retry. Windows executes capacity/queue failures and short reads; actual OS
  write-limit injection is Linux-only. These are library tests, not GPU SSD evidence.
- P3 three-pair RAM screens retain exact output/usage. Median total changes for
  short/window/continuation are llama -0.42/+0.40/+0.35 percent and ninfer
  -0.60/-0.38/+0.02 percent. No sustained >2 percent slowdown appears; short TTFT
  remains noisy and these screens do not replace final acceptance. Evidence is in
  `measurements/p3-llama-ram` and `measurements/p3-ninfer-ram`.
  WSL ThreadSanitizer fails before test entry with an unexpected memory mapping
  (both default PIE and no-PIE); no race-sanitizer pass is claimed.
- P4 starts with a ninfer C1 native-payload adapter. Logical history admission is
  separated from physical RAM accounting, with one B+R transfer window per lane
  held back as transition headroom. Idle histories retain the same records;
  obsolete disabled cold-snapshot serialization is removed. A real-model test
  compares an ample-RAM oracle with forced SSD pressure, exact tokens, prefix
  reuse, native plane checks and I/O/quota counters. Both real-engine checks pass:
  INT8/eager retains 73,531,392 native bytes with H=21,626,880 and D=51,904,512;
  RK8V4/MTP3/Graph retains 60,370,944 bytes with H=17,756,160 and D=42,614,784.
  Repeated and rewritten queries match the RAM oracle exactly, including reuse;
  native payload/state checks, released GPU pages and temporary-file cleanup pass.
  These are correctness runs, not performance samples. Evidence is in
  `p4-ninfer-c1`; the first correct binary is in `baseline/ssd-first-correct/ninfer`.
  The llama packed-store adapter now passes Windows 17/17 host suites and WSL
  ASAN/UBSAN/leak checks, including cold packed-row rewrite, full snapshot roundtrip
  and full-disk refusal preserving earlier K/V. The full legacy RawKvStore suite
  also passes after these changes.
- P4 llama real-engine C1 checks pass with exact text and token/cache usage for
  initial generation, repeated request and edited final query. The 0.8B fixture
  retains 23,187,456 native bytes under a 16 MiB H quota; disk writes/read bytes are
  6,423,552/2,088,960. Enabling SSD with 256 MiB H produces zero disk I/O and identical
  output. A 27B fixed-MTP fixture retains 138,346,496 native bytes under shared
  64 MiB H; actual disk writes/read bytes are 71,258,112/11,206,656, with no quota
  overrun or I/O error. These are correctness runs made during compilation and
  are excluded from performance results. The first correct SSD binary is frozen
  in `baseline/ssd-first-correct/llama`; performance acceptance remains pending.
  Concurrent SSD scheduling and idle checkpoint/statistics accounting remain P5/P6.
- P4 three-pair screens preserve exact output/usage. Against accepted RAM binaries,
  median total changes (short/window/continuation) are llama +0.008/+0.072/+0.024
  percent and ninfer +0.179/-0.166/-0.404 percent. Ninfer short TTFT is +2.277
  percent (upper bound +5.593), despite internal prefill/decode not regressing;
  seven pairs retain a +2.277 percent median and +4.913 percent upper bound. This
  TTFT metric remains unqualified; total +0.179 and internal prefill +0.085 percent
  are stable. SSD-enabled but
  unused total changes are llama -0.257/-0.009/+0.192 and ninfer
  -0.146/-0.335/-0.036 percent. Small-sample TTFT intervals remain wide; these
  screens do not establish final equivalence. All samples are retained under
  `measurements/p4-*`.

## P5 integration under test

Native K/V records keep their ownership across active/idle transitions. Llama's
idle serializer omits shared native records and writes only state/statistics
attachments into the same SpillFile. Its existing session RAM soft cap controls
attachment placement; the common H cap remains specific to native payloads.
Ninfer may demote idle checkpoint state images once native SSD pressure exists,
bounded by remaining logical admission capacity and the same D account. Active
checkpoint images remain separately counted pinned RAM. Destination writes/read
allocations finish before their source is released. No GPU work runs on storage
workers. P5 passes Windows 17/17 host suites and WSL/GCC ASAN/UBSAN/leak checks.
Llama A/B/A exchange and edited-query checks preserve exact output/cache usage,
with H=16,763,904 and combined native/state D=92,840,292 bytes; two attachment spills
and one restore occur without extra session eviction or I/O error. Ninfer INT8/eager
and RK8V4/MTP3/Graph checks pass native payload/state, token and reuse equality.
Idle state RAM reaches zero, and combined D is respectively 513,767,424 and
504,477,696 bytes, within 512 MiB. Frozen binaries are in `baseline/p5-idle-correct`.
Ninfer's seven-pair RAM comparison against the accepted RAM binary has exact
outputs and upper 95% latency ratio bounds below 3% for every measured metric.
Median total changes (short/window/continuation) are -0.252/-0.021/-0.325 percent;
short TTFT is -4.244 percent with upper bound +0.498 percent. The earlier P4 short
TTFT concern did not recur in this P5 run; its earlier samples remain retained.
Llama's seven-pair total changes are +0.327/-0.069/+0.106 percent. Short TTFT
is +3.581 percent with upper bound +5.370 percent and is not accepted. A targeted
three-pair probe with six identical cold short requests per process has median
TTFT +3.091 percent, upper +11.184 percent. Neither probe establishes its cause.
The original samples are retained. On 2026-10-10 the user explicitly deferred
investigation of the reported occasional C1 delays and directed implementation
and remaining acceptance work to continue. These unresolved confidence bounds
are disclosed rather than relabeled as passing.

## P6 implementation checkpoints

Common deferred sources retain immutable payload identity without pinning RAM.
Bounded CPU jobs acquire short leases while copying and own no GPU object.
Ninfer window transfers now retain their transaction across scheduler rounds:
pending lanes yield; the engine alone publishes archive records and GPU views.
INT8/eager and RK8V4/MTP3/Graph real-model checks pass exact output, reuse, native
bytes/state, quota and cleanup through asynchronous append, query replay, history
state materialization and idle checkpoint writes. Workers own only CPU buffers;
the engine retains pinned checkpoint sources and publishes completed state.
Wait-only prefill does not consume compute service entitlement. Forced control
text is committed in bounded chunks, with exact RAM/SSD output in a real-model
fixture whose control suffix exceeds R. Focused frontend closure/accounting tests
also pass; the broader frontend suite stops in tool-call grammar cases outside
this changed control path and is not counted as passed.

Llama C2 active/idle A/B, C/D, A/B exchange passes RAM/SSD output and usage equality,
one shared H/D account, real I/O and cleanup. Ninfer C8 RK8V4/MTP3/Graph passes
eight active lanes, cancellation of one, peer continuation and late-publication
retention. An earlier publication fixture did not overlap decoding and was not
accepted; its retained log precedes the corrected longer older request.

Cancellation parks a lane until its started CPU I/O drains, leaving other lanes
runnable. Shared policy selection releases its lock around physical writes and
skips records already being relocated. A Windows test intercepts only its own
WriteFile import: one disk-full error fails the cold request while its hot peer
matches RAM output exactly and a later retry succeeds. Holding a write while
cancelling that request still lets the peer emit eight more tokens before the
write is released. No production fault hook or environment option is introduced.
Common Windows 17/17 and WSL ASAN/UBSAN/leak checks pass, including real Linux
write errors classified as request-local TransferFailed. Concurrent latency,
final RAM/MTP comparisons, long-context qualification and packaging remain.

## Final verification scope, 2026-10-10

Seven-pair cumulative ordinary comparisons against the original engines preserve
exact output and usage. Median short/window/continuation total latency changes
are llama -0.241/+0.242/+0.410 percent and ninfer -0.101/+0.083/-0.159 percent.
Llama continuation includes two first-token outliers (+240/+389 ms); its total
upper bound is +3.522 percent. Ninfer short/continuation first-token confidence
bounds also remain wider than the original gate. The user's deferral above
applies to these observations; the underlying samples remain available.

These changed-behavior tests have now run: blocked cold reads/rollback,
engine fault/cancellation regression, C2 hot/cold performance, fixed-MTP cumulative
screens, real 128K/256K milestones, CPU vision/fast-prefill/adaptive MTP/Graph SSD
combinations, launcher options, exact patch/tree reconstruction and clean-runtime
dependency packaging. Already accepted broad RAM/format matrices were not repeated.

Llama 260K RAM input expanded from three to seven pairs: total median +2.476%,
upper95 +4.150%; continuation median -1.553%, upper95 +3.410%. These wider bounds
do not establish equivalence within 3%. A single instrumented pair preserves exact
output/usage and identical GPU transfer bytes/calls; the difference lies mainly
in the first prefill phase, without an established root cause. Those instrumented
samples are excluded from acceptance. Follow-up should isolate this path before
more full-model repeats, and test only changes justified by that evidence.
Ninfer 128K is a single paired capacity milestone, not an equivalence claim.
Native Linux GPU/build qualification needs an appropriate Linux environment.
The SSD fixtures use buffered file I/O and do not evict the OS file cache or
measure physical disk reads. RAM-tier cold records may still hit the file cache.
Add a bounded isolated physical-cold-read/memory-pressure comparison before
claiming the observed SSD-path TTFT difference applies when physical RAM is scarce.
Actual SSD cost is reported separately from original-engine RAM regressions.
