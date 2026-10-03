# Research notes

Recorded by Ian Ranson in [Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom);
the kernels they describe are the ones in this line.

Records for implementation candidates that were built, measured, and then **not** routed, plus the
mechanism notes those measurements depend on. An entry exists so a later round can see what was already
tried and on what evidence, instead of rebuilding the same candidate. Every number says what its timing
boundary was: a complete public Op, a kernel inside it, or a one-side probe. The reports under
`docs/performance/` own published results; this file does not replace them.

## The two Q5 parent shapes

These two shapes sit next to each other in `ops/linear/q5/q5_rowsplit_gemm_simt.cuh` and are easy to
describe wrongly, which has already happened once in the original branch's own documents. Both are used by the two
Q4/Q5 input projections, and they are not the same shape.

**split4** (`q5_rowsplit_gemm_simt_split4_kernel`): **one CTA owns one output row** (`row = blockIdx.x`),
and **its four warps split the K dimension**: `chunk = threadIdx.x >> 5` selects the warp's K quarter, each
warp reduces its own partial sums with `warp_reduce_sum`, and the four partials meet in `s_part[4][kTt]`
behind `__syncthreads()`. It stages no weights and no activations in shared memory - both the quantized
weights and the activations are read from global memory. It is instantiated per exact column count
(`kTt`), which is why it covers exactly the live columns.

**c4 SIMT** (`q5_rowsplit_gemm_simt_kernel` with `kTt = 4`): **one output row per warp**, up to **four
columns** per column tile (`blockIdx.y` selects the tile), with the **quantized weight planes staged in
shared memory** (`s_nib`, `s_hi`, `s_sc`, filled by `q5_simt_issue_slab` through a cp.async pipeline) and
**activations read from the input tensor** (`q5_simt_consume_slab` reads `x0 + tt*k + xoff`). It does not
share an activation slab across rows; that was the removed row-block shape's mechanism, not this one.

## Q5 row-block small-T shape (`ops/linear/q5/q5_rowsplit_rowblock_small_t.cuh`, removed)

The shape staged one activation slab per block in shared memory and let `kRowsPerBlock` warps read it, so
one warp still owned one output row but the repeated activation loads were divided by the row count. It was
introduced to serve the Q5 parent of the two Q4/Q5 input projections at `T=7..9`. It is not routed at any
column count, and the file was removed from the production tree (git history keeps it, including the
`__syncwarp()` fence its pipeline needed).

What decided it, with the timing boundary of each number:

| Measurement, and its boundary | row-block | other shape | reading |
|---|---|---|---|
| GDN Q5 parent, `T=7/8/9`, **one-side probe** (us) | 62.7 / 62.7 / 79.1 | 52.5 / 60.6 / 70.9 (split4, one-side probe) | row-block loses at every count |
| GDN Q5 parent, `T=10/12`, **kernel duration inside the complete Op** (nsys per-instance minimum, us) | 91.4 / 102.3 | - | the decisive numbers below |
| attention complete projection, `T=7/8/9`, **complete public Op** | - | - | split4 is 18.8% / 14.6% / 5.4% faster than what was routed (row-block at 7/8, c4 at 9) |

Two caveats belong with that table, and both are the reason the row-block is not routed. Where each
number lives: the one-side probe rows are the R7 round's probe logs
(`profiles/bench/input_proj_r7/data/results_probe_*.txt`), the kernel-level row is
`profiles/bench/input_proj_r7/data/nsys/kernels.txt`, the attention row is
`profiles/bench/input_proj_r7/data/attn_ab.txt`, and this review round's complete-Op numbers are
`profiles/bench/input_proj_r7_review/data/`.

- A **one-side probe** ranked the row-block *ahead* of the c4 tile at `T=10` and `T=12`, and the
  row-block's own kernel-level numbers (the 91.4 / 102.3 us row) came out of the same attempt. What
  settled it was the complete public Op, which ranked the row-block behind at both counts, in both forms.
  A one-side or single-kernel ranking is therefore not used to move a route boundary in these Ops.
- The **narrow-tile counterparts and the complete-Op numbers of that reverted attempt are not in this
  bundle**: the complete-Op CSVs were written under the same file names as the runs that followed and were
  overwritten, and the "the projection op's own bench agrees" sentence that used to sit in the round report
  actually pointed at two-node **probe** sums (105.7 / 128.3 us), which is a different instrument and must
  not be quoted as the projection bench. The numbers are left as they were recorded rather than retimed;
  the decision they supported is unchanged, and its own evidence is listed below.

The current band ends were decided in the review round from a complete-Op comparison with its raw CSVs in
this bundle (`data/raw_s2/`, `data/raw_s2b/`), not from the numbers above.

## Q5 split4 band ends, per parent

The split4 shape is the Q5 parent mechanism for the low column counts of both fused projections, and the
two parents end their bands at different counts because their row counts differ (12288 against 7168):

- **GDN parent: split4 through `T=10`.** At `T=10` split4 beats the c4 tile in the complete Op for every
  organisation that exposes 10 aggregate columns - `B=1/W=10` (105.7 against 110.3 us cold), `B=2/W=5`
  (104.9 against 109.7) and `B=5/W=2` (104.2 against 108.1), in both forms and under both cache policies,
  and by more warm (about 93 against 104 us). At `T=11` and `T=12` it loses (118.0 against 111.9 and 140.5
  against 113.9 us cold), so the band ends at 10.
- **attention parent: split4 through `T=9`.** `T=10` ties with the c4 tile (77.06 us both) and `T=11`/`T=12`
  lose (85.0 against 79.1 and 95.5 against 81.2 us), so the band ends at 9.

Both ends are crossovers between two legal shapes, not limits of either shape: both shapes are correct at
every count in `[2,15]` for the GDN parent and `[2,12]` for the attention parent.

## Fused projection + convolution for the Q4/Q5 GDN input projections

The op either projects into a workspace and then runs the convolution as its own step, or fuses the
convolution into the projection epilogue (which is what the resolutions `T=1,2,3,5,6` already do). Three
attempts to widen the fused route were built and measured; all three were slower, so the materialized route
stays for the rest:

| Candidate | routed form (us, cold) | candidate (us, cold) | note |
|---|---|---|---|
| `T=4`, `B=1`, Snapshot / Record | 52.2 / 50.9 | 54.5 / 53.0 | Snapshot needs no workspace (81920 -> 0 B) and one fewer node (5 -> 4) |
| `T=7`, `B=1`, Snapshot | 69.4 | 81.2 | |
| `T=8`, `B=1`, Snapshot | 76.3 | 91.4 | |
| aggregate 8, `B=2/W=4`, Snapshot | 75.0 | 95.5 | batched fused organisation |
| aggregate 8, `B=8/W=1`, Snapshot | 75.0 | 106.2 | same |

Every number in that table is a **complete public Op** median from the trial and adopted builds
alternating in one window (`data/raw_a/`, `data/raw_b/`, `data/raw_c/`).

What can and cannot be concluded:

- The candidates above **did not win**, and that is all those measurements say. They rule out those
  specific implementations, not fusion in general.
- For `T=7`/`T=8` and for the batched organisation, the candidates drive the Q4 side through the row-split
  SIMT kernel while the routed form uses the K-split MMA kernel there (7..12), and they move the
  convolution into the projection epilogue. That is an implementation difference between those candidates
  and the routed form, and it is consistent with the sizes of the losses.
- For `T=4` it does **not** explain anything: the routed form at `T=4` is `launch_t4_pdl`, whose Q4 side is
  **also the row-split SIMT kernel** (`Q4GdnSimtR8C4Schedule`) with the dependent-launch pair. So "the
  candidate replaced the K-split Q4 with SIMT" is wrong for `T=4`; what is left there is the measurement:
  the fused candidate is 4-5% slower in both forms while being structurally simpler.
- The **Snapshot** `T=4` prototype removes the projection workspace (81920 -> 0 bytes). The **Record**
  form has no such saving to report: its materialized route writes the caller-owned `conv_record` through
  its own publish path, and its public temporary-workspace query is 0 in both builds. The two forms are
  not interchangeable here.

**Not implemented:** fusing the K-split MMA Q4 side with the Q5 split4 side and a sequence-collecting
convolution. Nothing measured above excludes it; it needs a different kernel rather than another
instantiation of the existing template.

The three candidates' minimal patches, their per-configuration measured table and their trial builds' test
logs are kept with the review bundle under `profiles/bench/input_proj_r7_review/data/prototypes/`. The
measured prototype sources were reverted before the final build and are not retained; that directory's
README marks the patches as reconstructions and lists which trial-build artifacts do still exist.

## Decode-round measurement setup (perf/decode-round-overheads)

The two entries below were measured on the RTX 5090 with the NVIDIA ModelOpt Qwen3.8-27B artifact
(`qwen3_8_27b_nvfp4-nvidia.ninfer`: NVFP4 MLP, FP8 attention/GDN projections), DFlash2 K=7 with the
optimized proposal head, INT8 KV, CUDA 13.4, Windows WDDM. The **controlled** numbers are
`ninfer_bench -pg 16384,256` and `-pg 60000,256` with greedy decoding, so every arm decodes the same
token stream (identical round and acceptance counts in every run); each value is the mean of 6-9
repetitions from arms interleaved in one window, standard deviation below 0.02 ms. The candidates were
selected per process from one binary (commit `f2c991d0` on branch `perf/decode-round-overheads`, not
merged; it holds every candidate below behind `NINFER_AB_ITEMS`) so arms differ only in the item. Kernel
attributions come from Nsight Systems `--cuda-graph-trace=node` captures of one 128-token decode at
16K context, compared kernel by kernel as the interval between consecutive kernel end times.

## Programmatic dependent launch with an entry-time trigger (replaced)

The first PDL form had every decode-path kernel call `griddepcontrol.launch_dependents` at entry,
then wait, and hinted the first four K phases of each FP8 SIMT warp's weight rows into L2 before the
wait. It was correct (byte-identical greedy output) and **10.4 % slower per round at 16K, 9.4 % at
60K**. The profile showed kernel lifetimes overlapping heavily (50 ms of summed kernel time in an
18.4 ms round) and the loss concentrated in bandwidth-bound kernels whose successors had launched
early: the 80-CTA NVFP4 down projection (+1.25 ms per round, 37 -> 57 us each), the FP8 out
projection (+0.32 ms), the GDN record kernel (+0.16 ms) and attention (+0.10 ms). With an entry
trigger the next kernels cascade onto the SMs the running grid leaves idle and issue their weight
loads and L2 prefetches while it is still streaming, so they compete for its bandwidth rather than
fill idle time.

The adopted form triggers dependents after the main loop of every streaming kernel (NVFP4/FP8/Q8/Q4
GEMMs, GDN record, drafter sliding-window attention and convolution prepare) and keeps the entry
trigger only for short kernels. It measured 2.5 % faster at 16K and 2.3 % at 60K. Two further
variants were measured on that form and not kept: launching KV-cache attention as a dependent (0.5 %
slower at 16K, equal at 60K; it has no weights to stage) and a two-phase L2 prefetch in the FP8 SIMT
kernel before the wait (no measurable change against removing it).

## Fused NVFP4 RMSNorm + SwiGLU MLP + down (not routed)

A fused Op computed `residual += down(SwiGLU(gate_up(RMSNorm(residual))))` for the NVFP4 AllowA4
route at T=5..128: one kernel for Offset RMSNorm plus NVFP4 activation quantization (the D=5120
RMSNorm launcher's reduction order, reused from the fused attention-input route), the W4A4 gate/up
MMA with an epilogue that wrote the down projection's NVFP4 codes and scales directly (16-row groups,
the same `quantize_nvfp4_k16` arithmetic), and the unchanged down MMA reading that plane. It was
byte-identical to the unfused composition at every tested T (5..128, eager and graph).

| Arm (controlled, ms/round) | 16K | 60K |
|---|---:|---:|
| without PDL: fusion vs master | -0.36 % | -0.32 % |
| with PDL: fusion + PDL vs PDL alone | +0.7 % | +0.7 % |

Without PDL the fusion removes two small launches per layer. With PDL those kernels already overlap
the neighbouring GEMMs' weight staging, and the fusion only lengthens the serial path: the fused
norm/quantize kernel (8 CTAs, 3.4 us) cost 218 us per round against 196 us for the separate norm and
quantize, and the quantizing epilogue added 103 us to the gate/up MMA against 80 us for the quantize
kernel it replaced, plus about 60 us across the following GEMMs. It was removed with PDL adopted; the
measured source is commit `f2c991d0`.

## DFlash2 proposal filtered by the request's top_k / min_p / top_p (not adopted)

The DFlash2 candidate selector draws each proposal from `softmax(edge / temperature)` over its 16
candidates. The candidate restricted that distribution exactly as the target sampler restricts its
own candidates (top_k, then min_p against the best weight, then the shortest top_p prefix of the
pre-truncation weight), renormalized it, and retained the restricted q for acceptance, so target
sampling stayed exact. It passed an FP64 oracle for the filtered q and draw.

Open-loop controlled test (production thinking sampling: temperature 1.0, top_k 20, top_p 0.95;
DFlash2 K=7 plus ngram 15/12, INT8 KV, one lane; 16 coding prompts x 2 seeds x 1536 output tokens
per arm, three rounds with the arm order alternated, 96 requests and about 140K tokens per arm):

| Arm | Output tok/s | Tokens per round | Neural acceptance |
|---|---:|---:|---:|
| master proposal | 208.1 | 3.358 | 33.19 % |
| filtered proposal | 206.5 | 3.331 | 32.91 % |

The per-round paired acceptance differences were +0.37, -0.26 and -0.99 points. The draft's own
top_p tail over 16 candidates carries little mass, so trimming it barely moves q toward the target
distribution. A closed-loop agentic pair had shown +2.0 points; its control arm produced 25 % less
text and three more cache-accounting request failures, and the controlled test attributes that
difference to content rather than the filter.

A proposal temperature scale (the draft softmax at `temperature * s`, retained q unchanged in
meaning) was measured the same way, on the build with the adopted round changes:

| Arm | Output tok/s | Neural acceptance |
|---|---:|---:|
| s = 1 (master) | 213.7 | 33.19 % |
| s = 0.7 | 213.0 | 33.04 % |
| s = 0.5 | 207.5 | 31.71 % |

Sharpening does not raise acceptance; at 0.5 it lowers it. Neither form was kept.
