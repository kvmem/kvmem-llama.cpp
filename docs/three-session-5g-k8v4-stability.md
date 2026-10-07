# Three-session 5 GiB K8/V4 stability and cold-prefill comparison

This is an opt-in real-model service test of three independent IQ3 sessions.
Main attention KV uses K=`q8_0`, V=`q4_0` (`-ctk q8_0 -ctv q4_0`). Each session
grows to about 4.8 GiB, with a 5 GiB RAM soft limit and a **10 GiB NVMe quota**.
One session is active and two are stored on disk. The test then switches through
A/B/C twice, checks isolated answers and KV prefix hits, and compares the first
restore of each session with the exact same prompt on a fresh, empty server.

The script owns only the temporary servers it launches. Free the selected GPU
before running it. It uses a 256K context, IQ3 MTP3/replay, and synthetic
channel histories; it can take tens of minutes because it builds three roughly
140K-token histories and performs three cold prefills.

```powershell
python scripts/test_server_three_session_5g.py `
  --server .\build-session\bin\llama-kvmem-server.exe `
  --model models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf `
  --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c `
  --output artifacts\three-session-5g-k8v4-5060ti
```

The output directory contains `result.json`, the hot and cold server logs, the
exact cold request messages, and the runner log. Temporary KV snapshots are
removed after each server stops. All HTTP times exclude model startup. Hot and
cold times include prompt processing, generation, and HTTP overhead; the
`session_restore` log value isolates the hot session transfer. The fresh-server
cold prompt must have the same token count as its hot counterpart and zero
cache hits.

## RTX 5060 Ti result, 2026-09-27

All **163 checks passed** on branch `feat/session-nvme-cache`. Accounted active
KV was A=4.783 GiB, B=4.779 GiB, C=4.782 GiB. With two idle sessions, steady
disk use was about 9.45–9.50 GiB. The initial A restore selected the actual
`exchange` path and reached 10,735,573,492 bytes (9.998 GiB) of peak disk
occupancy against a 10,737,418,240-byte quota, leaving 1,844,748 bytes.
Later restores selected `ram_first` because the host had room for it.

| First restore | Hot transfer path | `session_restore` | Hot request | Cold prefill request | Hot speedup |
|---|---|---:|---:|---:|---:|
| A, 143,522 tokens | `exchange` | 25.19 s | 27.16 s | 175.53 s | 6.46× |
| B, 142,603 tokens | `ram_first` | 25.12 s | 27.00 s | 173.98 s | 6.44× |
| C, 141,809 tokens | `ram_first` | 27.27 s | 29.27 s | 172.95 s | 5.91× |

The second A/B/C cycle also passed: hot request times were 29.58, 30.57 and
31.40 seconds. Each restore reused over 140K prompt tokens. All answers retained
the correct channel code, the session count stayed at three, and no populated
session was evicted. One initial empty, zero-row slot was removed by LRU and
incremented the raw eviction counter; the test checks logged evicted row counts
to distinguish this from data loss. Disk errors and cache-clear fallbacks were
absent. The original RTX 5060 Ti service on port 18200 was restored afterward
and passed `/health`.

This measurement is one GPU, model, prompt shape and disk. The precise speedup
depends on NVMe bandwidth and model prefill throughput; the comparison here
uses exact matching requests for each session.
