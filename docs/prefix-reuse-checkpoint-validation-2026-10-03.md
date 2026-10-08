# Prefix-reuse checkpoint validation, 2026-10-03

Validated server source `31cf7f2` (recurrent checkpoint kept on the LCP boundary)
on Windows 11 with RTX 5060 Ti 16 GiB, NVIDIA driver 617.14, Ryzen 5 9600X,
48 GiB RAM, MSVC 14.44 and a Release CUDA 13.2.86 build.
Server SHA-256: `A1BB3CBF498C85EEADE07758EBA1BC82567774E0679FD634BCC1F8AD4B35550B`.

Compared against the official **v0.16.0-rc3 Windows CUDA 13.2.86** release build,
SHA-256 `48A4FDA230BB5FC49BBD48BA3FBD2E89ADA136D450CB1025F3FDDCE6023A59D4`.
Both ran the same model, the same flag set and the same request sequence on the
same machine; the binary is the only variable.
The `llama.cpp` checkout is `b81c99b`; it was not edited for this work.

## Results

`tests/prefix-reuse-regression.py` — three requests where the trailing user
message is rewritten each turn (its role flips to `assistant` on the next turn):

| build | turn 1 | turn 2 | turn 3 | `no_recurrent_checkpoint` | turn 3 `prefix_hit_rows` |
|---|---:|---:|---:|---:|---:|
| v0.16.0-rc3 (official) | 4.85 s | 4.71 s | 4.68 s | 2 | 0 |
| `31cf7f2` (this change) | 4.91 s | 4.95 s | 1.03 s | 0 | 3464 |

Unmodified, turn 3 is as slow as turn 1 and reports no reuse at all. With the
change, the third turn reuses 3464 rows and completes in about a fifth of the time.

Longer workload (`agentperf-local`, 168 turns, identical recorded requests, same
engine before and after):

| | wall | TTFT p50 | TTFT p95 | end-to-end output |
|---|---:|---:|---:|---:|
| before | 48.8 min | 9,160 ms | 21,254 ms | 8.14 t/s |
| after | 30.2 min | 1,549 ms | 7,189 ms | 38.59 t/s |

The three tasks that hit this pattern (SWE-bench style, a trailing nudge every
turn) went from TTFT p50 of 13,676 / 15,927 / 10,047 ms to 1,355 / 1,460 / 1,471 ms.

## Behavior checked

- Prefix reuse survives a trailing `user` message whose role flips to `assistant`
  on the next turn — the pattern agent harnesses produce with per-turn reminders.
- Reuse is unchanged when the trailing message is an append-only `tool` result
  (99.6–99.8% on the same workload).
- The `[query, eval_end)` replay origin, and therefore the replay and residency
  requirements, are unchanged.
- Cross-request shared-prefix reuse is **not** addressed here: a strictly shorter
  earlier request still recomputes. That is a separate question.
- The new split point participates in eviction under the 4-snapshot cap; no
  anomaly was seen in these runs, but it was not stress-tested beyond them.

## Notes

- The `agentperf-local` figures are supporting evidence only. KVMem does not
  report `meta.n_ctx`, so the tool marks those runs non-comparable; read them as
  before/after within the same engine, not as a comparable benchmark.
- The primary evidence is the server's own trace (`multimodal_reset ...
  reason=no_recurrent_checkpoint lcp=... oldest_checkpoint=...`) and the
  regression test shipped with the change. Neither depends on a third-party
  tool's scoring.
- Trace lines, token counts and timings only; no user data.
