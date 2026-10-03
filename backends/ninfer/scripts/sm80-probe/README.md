# sm_80 probe (CMP 170HX and other GA100 parts)

Decides, in a couple of rented minutes, whether a GA100-class card is worth tuning for. It answers
the questions a listing cannot:

| question | probe | why it decides |
|---|---|---|
| Is the community firmware unlock applied? | `nvidia-smi` memory / PCIe fields | 8 GB stock vs 64 GB (or 10 to 40 GB) changes which models fit at all |
| Is FP32 FMA / INT32 IMAD throttled? | `tools/fma_rate_probe.cu` | stock cards run FFMA at ~1/16 rate; every kernel uses it |
| Do the tensor cores run at rated speed? | `tools/tensor_core_rate_probe.cu` | the engine's prefill and decode GEMMs are all `mma.sync` |
| What does device memory deliver? | `tools/hbm_bandwidth_probe.cu` | decode is bandwidth-bound |
| What does the host link deliver? | `tools/pcie_bandwidth_probe.cu` | sets model load time and host state-slot cost |
| Is every byte of the (unlocked) VRAM good? | `tools/vram_pattern_probe.cu` | a defective region corrupts weights silently |

The unlock itself is applied by the host (patched `nvidia-open` kernel modules, Linux only). A
container cannot apply it, so a rented card is either unlocked or it is not. The first lines of the
summary say which.

## Running it

sm_80 support is on the `feat/sm80-cmp170hx-probe` branch. The script clones the branch, so it must
be pushed first.

Rent with a CUDA **devel** image on Ubuntu 24.04, for example
`nvidia/cuda:12.8.1-devel-ubuntu24.04` (`nvcc` is needed, and CMake 3.28 for `--engine`; 22.04
ships 3.22). If the card is unlocked to 64 GB, check the listing for host RAM, since the unlock
documentation asks for at least twice the VRAM. Then, from a machine with ssh access:

```bash
ssh -p PORT root@HOST 'bash -s' < scripts/sm80-probe/probe.sh
```

Or paste `probe.sh` as the instance on-start script and read `/root/probe-out/summary.txt` later.
A dead-man switch powers the box off after `MAX_RUNTIME_SECONDS` (default 3600, `0` disables), so a
forgotten instance does not run all night; destroying it is still up to you.

Probes only, about two minutes of which most is compiling. To also build the real engine for this
architecture and generate text (the check that the kernels give sensible output on this card), give
it a model:

```bash
ssh -p PORT root@HOST 'MODEL_URL=https://.../model.ninfer bash -s -- --engine' < scripts/sm80-probe/probe.sh
```

`--tests` builds the `ninfer_tests` bundle (every test, one executable) and runs the op correctness
suites from it on the card itself (they need no model), so it takes a full test build's time and
disk. The two flags combine: `bash -s -- --engine --tests`. For `--engine`, rent with at least 80 GB of disk (a
20 GB artifact, a build tree and the CUDA image), and pass `MODEL_SHA256` next to `MODEL_URL` to check
the download.

`bash probe.sh --summarize DIR` re-prints the summary of a finished run from its output files.

## Reading the summary

| line | what to look for |
|---|---|
| `capacity` | 64 GiB (or 40) means unlocked. 8 GiB means a stock card: stop, it is a different and much worse target. |
| `fma` / `imad` | `not throttled` / `ok`. `THROTTLED` means the unlock is not applied to compute, and the FMA-heavy kernels would need a mul+add rewrite. |
| `tensor` | `WORKING` is at least half the unthrottled GA100 rate for this SM count and clock. `CRIPPLED` is at or below the plain FP32 rate, and prefill would be worse than a 3090's. |
| `hbm` | The decode ceiling. A 3090 reaches 894 GB/s. |
| `pcie` / `transfer` | Gen2 x4 is ~1.7 GB/s: a 16 GiB artifact loads in about ten seconds, but a host state-slot swap is ~90 ms. |
| `vram` | `clean`. `BAD` lists the regions; do not put weights on this card. |

Reference numbers for an RTX 3090 (the fork host, 2026-09-21): memory read 894 GB/s, BF16
83 TFLOPS, INT8 337 TOPS, FFMA 35-40 TFLOPS depending on boost clock, pinned host to device
25 GB/s.

## What passing does and does not mean

A pass says the card is worth tuning for. It does not say the engine is tuned for it: every route
table in this repository was measured on a 3090 (82 SMs, 936 GB/s, 6 MB L2) and the GA100 part has
70 SMs, a different memory system and a 32 MB L2. Expect the sm_80 build to run correctly and
unoptimized, the same standing the sm_89 build had at first (`docs/rtx-4090-early.md`). Sweeping the
schedules on the card is the next step, and the largest one.

Only one GPU is tested (`PROBE_GPU`, default 0). Numbers taken on different rented boxes are not
comparable for anything compute-bound, so compare like with like on one box.
