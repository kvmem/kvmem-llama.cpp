# Multi-GPU test harness

Scripts for validating the layer pipeline split (`--devices N,M`, optionally `--stage-layers`) on
rented dual-GPU hardware. The tests that need a real second card are
`ninfer_qwen3_5_stages_real_test` (with `NINFER_TEST_ARTIFACT` set) and `ninfer-tp-probe`, which
measures the boundary-transfer cost the pipeline pays and decides whether tensor parallelism is
worth building on that box.
They assume a vast.ai instance, but nothing here is specific to that provider beyond the CLI calls
in `iterate.sh`.

| script | what it does |
|---|---|
| `onstart-pipeline.sh` | Instance bootstrap: install deps, clone the branch, start the model download, build, run the equivalence check. Runs once at instance creation. |
| `iterate.sh` | Pull, rebuild incrementally and re-run **on an already-rented box**. |
| `bench-concurrency.sh` | Context ceiling and aggregate throughput under concurrent load. Piped in over ssh. |

## The workflow that matters

The 21 GB model download is the long pole -- roughly 25 minutes, against ~10 minutes for a clean
build and seconds for an incremental one. Destroying the instance to test a one-line fix throws
that away every time.

So: **edit locally, commit, push, and let the box pull.** `iterate.sh` does exactly that
(`git fetch` + `reset --hard FETCH_HEAD`, then an incremental build). Nothing is ever edited on the
remote, so nothing can be lost there.

```bash
bash iterate.sh --build-only    # pull HEAD and rebuild
bash iterate.sh --run-only      # equivalence against the single-GPU reference
bash iterate.sh --capacity      # what KV each mode resolves with --kv-capacity auto
```

## Two things learned the expensive way

**Test by output, not by startup.** A split that starts cleanly and reports sensible memory can
still be reading the wrong device through a stale pointer. The check that matters is greedy text
from `--devices 0,1` against greedy text from `--device 0`, byte for byte.

**Most of it is testable on one GPU.** `--devices 0,0` puts both stages on the same card. It frees
no memory and is not a deployment, but it exercises per-stage placement and materialization,
per-stage workspaces, KV pools and state shards, and the boundary transfers (forced through pinned
host memory with `NINFER_FORCE_STAGED_LINKS=1`). What it cannot see is a pointer that is valid on
the wrong device, because both stages can read everything; that, the PCIe cost and the
compute-capability check are what the second card is for.

## Downloads

Use `aria2c -x16 -s16 -c`, not `curl`. A single-stream curl throttled to ~1 MB/s and then stalled
outright at 11 GB of 21; the same file over 16 parallel ranges sustains ~98 MB/s and resumes rather
than restarting.
