# 1:10 multi-session exchange stability test

This is a release-level service stability check for the RAM/NVMe session cache.
It scales the 10/15/20 GiB exchange case to approximately **1/1.5/2 GiB**:

| Resource | Starting state | Final state |
|---|---:|---:|
| Active session in host RAM | A about 1 GiB | B about 1.5 GiB |
| Inactive session on NVMe | B about 1.5 GiB | A about 1 GiB |
| NVMe quota | 2 GiB | 2 GiB |
| Session RAM soft limit | 1.2 GiB | 1.2 GiB; B may exceed it |

Both sessions must retain their exact KV across repeated A/B/A/B switches. A
temporary 2.5 GiB disk allocation must not be required. The active 1.5 GiB
session must not be rejected for exceeding the 1.2 GiB soft limit. The real
model test validates the server, prefix reuse, answer isolation, quota counters
and zero avoidable evictions. The transfer test forces the bounded-RAM
**exchange** path using real vector allocations and snapshot files. On a host
with ample free RAM, the server may correctly choose `ram_first` instead.

## Run on RTX 5060 Ti

Free the 5060 Ti before starting; the script launches and owns its own server
on an unused loopback port. It pins CUDA to the provided GPU UUID and writes
the service log plus machine-readable result under `artifacts/`.

```powershell
& .\build-session-host\bin\kvmem-session-transfer-test.exe --scale-1to10
python scripts/test_server_session_exchange_1to10.py `
  --server .\build-session\bin\llama-kvmem-server.exe `
  --model models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf `
  --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c `
  --output artifacts\session-exchange-1to10-5060ti
```

The transfer test's one unit is `(1 GiB / 10)` bytes. It uses 10 units of A,
15 units of B, a 20-unit disk quota and 15 units of total test RAM. The unit
test checks the measured peak after every chunk, both switch directions and
two complete cycles. This opt-in run allocates up to about 2.5 GiB while
constructing its fixture and writes several GiB; it is separate from the
default fast CTest run.

The server test starts with shorter prompts and grows each history until its
measured session storage is within 6% of 1 or 1.5 GiB. It uses a 1.2 GiB soft
limit, 2 GiB NVMe quota, three conversation slots, flash attention, q8_0 KV and
the model's MTP path. It verifies correct channel answers and more than 1024
cached prefix tokens after every restore, then checks no eviction occurred.
The test fails if a model/context setting cannot reach the target range. The
log's `session_transfer` events show which route was used. The output retains
`server.log` and `result.json`; private snapshot files are removed unless
`--keep-cache` is set.

## Local verification (Windows, 2026-09-27)

- The opt-in transfer test passed with A=`1,073,741,820` bytes,
  B=`1,610,612,730` bytes, a `2,147,483,640`-byte disk quota, a measured
  `1,610,612,730`-byte RAM peak and a `2,040,110,370`-byte disk peak. It used
  the chunk exchange route in both directions for two cycles.
- The RTX 5060 Ti real-model run reached A=0.977 GiB and B=1.453 GiB of
  accounted session RAM. All 105 checks passed across three A/B restore pairs,
  with correct answers, cached prefixes, no evictions and disk usage below
  2 GiB. The available physical RAM let the live server choose `ram_first`;
  the bounded-RAM exchange route was exercised by the transfer test above.
- The pre-existing 5060 Ti service was restored on port 18200 and `/health`
  returned HTTP 200 after the run.

Run the existing short [session disk regression](session-disk-cache.md) as a
separate gate for corruption, file locks, count LRU and small-quota behavior.
The 1:10 check is deliberately focused on large session exchange and repeated
service use.
