# NVMe-disabled regression: IQ3 K8/V4

The session disk cache is disabled when `--kvmem-session-nvme-gb` is absent.
This check covers both the default single-session behavior (no conversation
flags) and RAM-only multi-session behavior (`--kvmem-conversations 3`, no disk
flags). It uses the same IQ3 GGUF, K=`q8_0`/V=`q4_0`, draft MTP, and RTX 5060 Ti as
the [three-session NVMe stress test](three-session-5g-k8v4-stability.md).

```powershell
python scripts/test_server_conversations.py `
  --server .\build-session\bin\llama-kvmem-server.exe `
  --model models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf `
  --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c `
  --kv-key-dtype q8_0 --kv-value-dtype q4_0 `
  --kvmem-budget 2048 --kvmem-gen-reserve 512 `
  --mtp --bytes-case --output artifacts\nvme-off-k8v4-5060ti\short-3ram

python scripts/test_server_nvme_off_long.py `
  --server .\build-session\bin\llama-kvmem-server.exe `
  --model models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf `
  --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c `
  --messages artifacts\three-session-5g-k8v4-5060ti-rerun\cold-messages-A.json `
  --expected-tokens 143522 `
  --output artifacts\nvme-off-k8v4-5060ti\long
```

The first script checks default interleaving misses, RAM-only multi-session
hits and answer isolation, three RAM-only sessions with zero disk bytes, LRU
eviction, byte-cap eviction, and that a rejected request leaves cache state
alone. The long check sends the exact 143,522-token A prompt from the NVMe
stress test to a fresh default single-session server. It then verifies a
same-session follow-up and confirms no session disk cache or transfers appear
in the log. Free the selected GPU before running either script; each script
owns the temporary servers it launches.

## RTX 5060 Ti result, 2026-09-27

- Short regression: **147/147 checks passed**. The default server advertised
  one slot and no conversation capability. With exactly three RAM-only
  sessions, A/B/C all answered with their own code; all second visits hit
  their prefix; `disk_bytes` and `disk_bytes_max` stayed zero. LRU, byte cap,
  and rejected-request state checks also passed.
- Long default single-session check: **10/10 checks passed**. The cold request
  had 143,522 tokens, zero cache hits, and took **176.81 s** wall time
  (**175.73 s** prompt evaluation). The follow-up hit 143,528 cached tokens
  and took **1.66 s**. No session disk cache or transfer was initialized.
- Matched cold request with NVMe enabled but empty in the stress test took
  **175.53 s** wall time (**174.64 s** prompt evaluation). Off was 0.73%
  slower in these single runs, a difference too small to infer a regression.

These checks found no functional side effect of leaving session NVMe disabled.
The three RAM-only histories in the short test are small; the long test checks
one approximately 4.8 GiB KV session. This does not assert that three 5 GiB
sessions will fit in RAM without NVMe. The original service on port 18200 was
restored after testing and passed `/health`.
