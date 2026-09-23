# Auto-continue-thinking

When a reasoning turn hits the per-round output budget while the thinking block
is still open, the server keeps the chain in the context and starts another
round by itself, instead of returning a truncated answer.

Measured on RTX 3080 10G / Bonsai 2 27B PQ2_0 / 160K workspace, identical request
both times:

| | continuation off | continuation on |
|---|---|---|
| finish_reason | `length` | `stop` |
| completion_tokens | 1200 (capped) | 31038 (6 rounds) |
| reasoning | 5272 chars, cut mid-sentence | 64507 chars, complete |
| content | empty | 54402 chars |

Cross-round continuity shows up in the debug log - `prefix_hit_rows` advances by
exactly the committed chain length every round:

```
multimodal_prefill prefix_hit_rows = 143 -> 6143 -> 12144 -> 18145 -> 24146 -> 30147
```

## Flags

| flag | meaning |
|---|---|
| `--auto-continue-thinking` | enable the loop |
| `--act-max-rounds N` | safety cap on the number of rounds (default 8) |
| `--act-round-tokens N` | output budget per round; 0 = request `max_tokens` |
| `--act-total-tokens N` | budget across all rounds; 0 = unlimited |
| `--act-continue-cue STR` | text injected between rounds (default `\n`) |
| `--act-prefill-mode MODE` | `auto` \| `legacy` \| `restart` (default `auto`) |
| `--act-think-end STR` | thinking end tag (default `</think>`) |

Per request: `"auto_continue_thinking": true|false`.

## Why rounds instead of a bigger n_predict

A single generation may not exceed `--kvmem-gen-reserve`; going over it aborts
the request and throws the round away. Rounding keeps every generation under the
reserve, and the finished round becomes history that KVMem can spill to host
memory, so the GPU window always has room. The inter-round prefill is nearly
free: the LCP already covers the committed chain, so only the cue is decoded.

## Requirements

- `--kvmem-gen-reserve` must stay greater than `--act-round-tokens`.
- `--reasoning-budget -1`. A positive budget makes the model close its thinking
  block by itself, after which it considers the task finished and the chain
  cannot resume.
- `--act-prefill-mode auto`, together with the default `user` query policy.
  `run_prefill_multimodal()` takes no prompt argument and reads
  `st.active_prompt`, which is assigned once per request, so a continuation
  round has to advance it explicitly - otherwise the next round dies with
  "non-consecutive token position".

## Diagnostics

```
KVMEM_ACT_STATE act_on=1 thinking_on=1 max_rounds=8 round_tokens=6000 total_tokens=36000
KVMEM_ACT_ROUND k=0 n_gen=6000 total=6000 in_think=1
KVMEM_ACT_ROUND k=1 n_gen=6000 total=12000 in_think=1
```

`act_on=0` means the switch never reached the server; `thinking_on=0` means the
request disabled thinking, so there is no chain to continue and the round is
closed deliberately. Both lines need `KVMEM_TRACE=1`.