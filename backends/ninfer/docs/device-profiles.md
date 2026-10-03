# Device route profiles

Many NInfer operations have several interchangeable schedules: CTA shapes and warp counts of the
small-T attention kernel, the column tiles of the ternary and groupwise projections, whether the
probability-times-value product accumulates in FP16, which prompt-attention kernel prefills. The
compiled tables that choose between them were measured on one card. The best choice depends on the
part: its SM count sets how many CTAs fill a wave, its memory system sets where a kernel stops being
latency-bound, and a schedule that is best on an RTX 5090 can be several times slower on an RTX 3090.

A device route profile records, for one GPU model, the schedule each operation takes per width band
where a measurement found a faster one than the compiled table. The engine installs it before any
operation runs; operations without an entry keep their compiled route.

## Where profiles come from

1. **Your profile file.** `$NINFER_DEVICE_PROFILES`, else `$XDG_CACHE_HOME/ninfer/device-profiles.json`,
   else `~/.cache/ninfer/device-profiles.json` (`%LOCALAPPDATA%\ninfer\device-profiles.json` on
   Windows). `--device-profile-path` names another file.
2. **The built-in table.** Profiles measured on the RTX 3090, RTX 4090, RTX 5090 and the RTX PRO 6000
   Blackwell ship inside the binary (`src/runtime/engine/device_profiles.json`). The PRO 6000's
   Workstation (600 W), Max-Q (300 W) and Server editions report different names, so each has its
   own entry.
3. **Calibration at first start.** A GPU with neither is calibrated once when the engine starts,
   before the weights are loaded. This takes 20 to 40 seconds, and the result is written to the
   profile file for later starts.

An entry applies only to the same hardware class (the GPU name and compute capability) and the same
SM count, so a laptop part that shares a desktop part's name but not its SM count is calibrated on its
own. With several GPUs (`--devices`), every distinct device gets its own profile.

`--device-profile auto` (the default) follows the order above, `--device-profile calibrate` measures
again and replaces the stored entry, and `--device-profile off` uses the compiled tables only.

## Calibrating by hand

```bash
ninfer-calibrate --print > my-gpu.json
```

`ninfer-calibrate` measures every route family on the current GPU (`--device N`) and stores the profile
where the engine looks for it (`--out PATH` elsewhere). `--detail` logs every candidate's time,
`--only PREFIX` limits the run to route keys with that prefix, and `--no-ternary`, `--no-groupwise`,
`--no-attention` and `--no-linear-attention` skip whole families. Close other GPU work first: the measurement assumes an idle
device.

Each candidate is timed through the same dispatch an inference call uses, on synthetic weights and
caches of the registered model shapes, with the L2 cache flushed before every sample, as the median
of eleven runs. A candidate replaces the compiled route only when it is at least 3 % faster, the win
survives a second interleaved measurement, and its output matches the compiled route's to within
5 %.

## What is calibrated

| route key | chooses | measured on |
|---|---|---|
| `t2_i8_route` | small-T kernel or prefill GEMM per width, ternary (Bonsai) projections | the GDN layer's projections at widths 16 to 192 |
| `t2_i8_small/<N>x<K>` | row and column tile of the ternary small-T kernel | every ternary projection shape, widths 1 to 32 |
| `q4_q5_attn_input/...`, `q4_q5_gdn_input/...`, `q5_linear_add/...`, `q4_linear_swiglu/...` | fused groupwise (Qwen3.6/3.8) projection schedules | the model shapes, widths 1 to 64 |
| `attn_i8_small/h24/<kv>/w<W>` | warps, CTAs per SM, key block, split QK (`q`) and early fetch (`e`) of the INT8-family small-T attention | query widths 1 to 8, KV windows 8K, 64K and 262K, for `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8` |
| `attn_pv_f16` | FP16 accumulation of the probability-times-value product per key tile | a 1024-token prompt chunk at 32K and a decode step at 131K |
| `attn_pack_gqa` | the standard INT8 prompt kernel with each KV head's query heads packed into its tiles (PackGQA) | a 1024-token prompt chunk at 32K and 131K |
| `attn_prompt_fast` | the fast prompt-attention kernel (rows kept in registers, FP16 PV per tile) | a wave-aligned prompt chunk at 32K and 131K |
| `unified/q4_q5_attn_input`, `unified/q4_q5_gdn_input`, `unified/q4_linear_swiglu`, `unified/q4_linear_add`, `unified/q5_linear_add/5120x<k>` | upstream's routes for the fused groupwise projections over the unified Linear templates (`unified`) instead of this line's own | the model shapes, widths 1 to 64 (the small-T launches only where those are all the switch covers) |
| `unified/q4_linear_topk`, `unified/q8_*`, `unified/fp8_*`, `unified/nvfp4_*`, `unified/bf16_*`, `unified/context_kv_materialize` | the same switch for the Q4 top-k head, the Q8 fused projections (attention and GDN inputs, LinearAdd, pair, SwiGLU, top-k, grouped convolution, context-KV materialization), the FP8 and NVFP4 fused projections (attention and GDN inputs with their conv forms, LinearAdd, SwiGLU, the FP8 top-k), and the BF16 attention input and LinearAdd | not calibrated: the built-in RTX 3090, 4090 and 5090 profiles carry the widths where each Op's benchmark ran faster on the unified routes over two rounds per table (at least 3 % over the geometric mean, no case more than 3 % slower); elsewhere a profile or `NINFER_LINEAR_ROUTES` sets them |
| `gdn_two_stage/h<value heads>` | the two-stage GDN prefill (fused Q/K normalization and control preparation, then one FP32-state recurrence that also writes the output) instead of the WY/state-passing/output pipeline | prompts of 16 to 8192 tokens, 48 and 32 value heads |

`NINFER_SMALLT_PV_F16`, `NINFER_PROMPT_PV_F16`, `NINFER_PROMPT_PACK_GQA`, `NINFER_PROMPT_FAST` and
`NINFER_GDN_TWO_STAGE` (`0` or `1`) override the profile for their route; `NINFER_LINEAR_ROUTES`
(`legacy` or `unified`) takes one table for every Linear shape and every `unified/*` switch. `--fast-prefill-kernel` turns the fast prompt kernel on regardless.

## Batch composition

A profile picks each width's fastest schedule, and schedules differ in the order in which they sum.
A step that serves four requests runs the projections at width four and splits the attention
differently from a step that serves one, so a request's numbers depend slightly on what shares its
steps, and a greedy answer can change where its two most likely tokens are nearly tied. In a check
on an RTX 5090 (Ternary Bonsai 2, four lanes), one request sent four times at once matched its lone
answer in one copy of four with the built-in profile, and in all four with `--device-profile off`
and `NINFER_PREFILL_ALIGN=0` (as on the previous master) or without the profile's ternary small-T
routes; four different documents at once matched their lone answers in one or two of four with the
profile and in three or four without it. The answers that differ part at such a word and stay
equivalent: four 60,000-token needle documents served at once returned all twelve codes in order,
with and without speculation, on `rk8v4` and `rk4v4`. Where answers should depend on batch
composition as little as possible, run with `--device-profile off` and `NINFER_PREFILL_ALIGN=0`.

## Sizes derived from the device

Some launch sizes follow the device directly and need no profile:

- **Prefill chunk.** `--prefill-chunk` is adjusted to the nearby multiple of 128 whose
  prompt-attention grid leaves the least of its last wave idle on this GPU (see
  `causal_softmax_attention_prompt_aligned_chunk`); `NINFER_PREFILL_ALIGN=0` keeps the requested
  chunk.
- **Wave counts.** The chunked GDN output kernel and the sparse-MoE prefill size their grids from the
  SM count and the kernel's occupancy, and the small-T attention holds its split count to whole waves
  of the device.
