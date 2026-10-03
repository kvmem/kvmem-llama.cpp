# A/B serve benchmark: a control build vs this build

Black-box A/B benchmark of two `ninfer-serve` builds serving the **same artifact** over the
OpenAI chat completions API, replaying a synthesized agentic workload and comparing avg TTFT,
prefix-cache hit statistics, cold prefill tok/s and output tok/s. [`../agentic_ab/`](../agentic_ab/README.md)
is the fuller closed-loop suite; this rig is the shorter fixed-request one.

Metrics come from each serve's own request log (`--request-log-jsonl`) — no LLM is used at run
time and no client-side timing is trusted.

## Files

| File | Purpose |
|---|---|
| `ab_runner.py` | The runner: calibrates max-context on the control arm, launches each arm on the same port, replays the workload (concurrent groups), parses both request logs, writes `ab_report.md` with a paste-ready README block. Fails loudly (no `AB_DONE.txt`) if any request is rejected or errored. |
| `ab_workloads.py` | Deterministic (seed 42) workload generator: 14 requests in 12 groups shaped from a logged agent workload — multi-turn tool-agent sessions (~33 tools, thinking on), an ~82K-token shared base, one ~140K cold prefill, a concurrent pair, a strict-prefix multi-turn ramp, and a concurrent shared-prefix cache-contention pair. |
| `ab_watchdog.py` | Optional auto-trigger: polls the port, fires the runner ~20 s after the server on it closes, and never starts a server there itself. |
| `control_build.bat` | Windows build driver for the control arm. `control\src` must be a checkout/worktree of the control commit; run it as `control_build.bat configure` then `control_build.bat build`. |

Outputs (written next to this script): `ab_report.md`, `arm_control.jsonl`,
`arm_treatment.jsonl`, `runner.log`, `serve_control.log`, `serve_treatment.log`,
`AB_DONE.txt`.

## Replicating the test

1. **Build both arms**: the control from a checkout of the commit to compare with (on Windows,
   `control_build.bat` into `control\src`), the treatment from this checkout's normal build.
   The runner only needs the two `ninfer-serve` paths.
2. **Set the flags** to compare in `BAT_FLAGS` in `ab_runner.py` (the list the original
   published run used is committed as-is, with `--prefill-chunk 2048` and
   `--ngram-min-match 12`). Flag fairness is automatic: the runner queries the control's
   `--help` at run time and drops exactly the flags it does not support, logging what it drops.
3. **Run** `AB_MODEL=<artifact> python3 ab_runner.py`. The runner probes the largest
   `MAX_CONTEXT_CANDIDATES` value at which the control starts and runs **both arms at that same
   context**. Then it serves the control, replays the workload, serves the treatment, replays it
   again, and writes the report.
4. Optional: launch `python3 ab_watchdog.py` in the background beforehand and stop the server on
   the port; the watchdog fires the runner after the port has been closed ~20 s in a row.

| Variable | Default |
|---|---|
| `AB_MODEL` | required |
| `AB_CONTROL_EXE` | this directory's `control/build/apps/ninfer-serve` (or `apps/Release/`) |
| `AB_TREATMENT_EXE` | this checkout's `build/apps/ninfer-serve` (or `apps/Release/`) |
| `AB_WORKDIR` | the current directory; the serves run there |
| `AB_HOST` / `AB_PORT` | `127.0.0.1` / `8080` (same override in runner and watchdog) |

## Workload and metrics

The workload targets are **actual** token counts (`CHARS_PER_TOKEN = 2.325`, calibrated
on this content mix — a naive chars/3.8 estimate understates real Qwen tokens by ~1.64×
here). Scenario groups, in replay order:

1. **S1** sequential private continuations on the shared base, ~85K → 140K actual.
2. **S2** one cold ~140K prefill with a different root prefix (raw prefill compute).
3. **S3** two concurrent ~95K requests with long outputs.
4. **S4** multi-turn ramp ~90K → 135K, each turn re-sending the full history.
5. **S5** two concurrent ~130K requests sharing the base (~260K combined KV working set)
   — cache hits under contention.

Groups are replayed in order; requests within a group go out concurrently (the serve
runs `--max-concurrency 2`, as the logged workload did). A one-message smoke request primes each
arm and is excluded from metrics.

Reported metrics (all from the serve's request log):

- avg / median TTFT over all completed requests;
- prefix-cache hit rate = Σ hit tokens / Σ prompt tokens, plus per-reuse-path
  requests / hit-token counts (`root`, `private_response_replay`,
  `private_long_anchor`, …);
- cold (root) prefill tok/s — median over requests that did a full uncached prefill;
- output tok/s = Σ completion tokens / Σ decode wall time (per-request decode rates are
  the comparable view; at temperature 1.0 the completion totals differ by sampling).

Note: a "non-cached prefill tok/s" (computed tokens / prefill wall time) exists in the
full report's aggregates, but the README table reports the cold row because wall time
includes prefix-cache reads, so that ratio penalizes the arm that caches more.

## Published run

The original author's run (Wallawalla47/ninfer-custom `master` 9c79553c vs upstream with the
Windows port, f2406102), 2026-09-22, RTX 5090,
max-context 180000, flags as committed in `ab_runner.py` (`--prefill-chunk 2048`,
`--ngram-min-match 12`), seed 42, 14/14 requests completed on both arms, 0 errors:
TTFT 28.1 s → 14.6 s (−48.1%), prefix-cache hit rate 19.8% → 66.3%, cold prefill
4,159 → 4,220 tok/s (+1.5%), output 173 → 181 tok/s (+4.6%).
