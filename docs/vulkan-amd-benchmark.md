# Vulkan / AMD benchmark — full experimental data

Hardware validation of the Vulkan integration on a physical AMD GPU.

## Environment

- Cloud host: degima.ai AMD instance (cloud server)
- OS: Ubuntu 24.04.5 LTS, kernel 6.8.0-138-generic (x86-64)
- GPU: AMD Radeon RX 7900 XTX 24 GB (RADV NAVI31), Mesa 25.2.8-0ubuntu0.24.04.2, Vulkan API 1.4.318
- CPU: AMD Ryzen 9 5950X 16-Core (32 vCPU, AVX2), 125 GB RAM
- Model: `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (12.1 GB) + `mmproj-Q5_K-MIX.gguf`
- Base: upstream `1734a28` (PR #27, quantized KV pairs) + this PR
- Build: `GGML_CUDA=OFF GGML_VULKAN=ON LLAMA_CURL=OFF`, Release
- Allocator: `MALLOC_ARENA_MAX=2` (defaulted by `scripts/start-vulkan.sh` and the canary)

## Regression: kvmem-gdn-replay-test on RADV

22/22 PASS on the physical device (`AMD Radeon RX 7900 XTX (RADV NAVI31)`),
including GGML record/fold widths 1–6, multilayer fold (4 layers), native
Vulkan shader fold, and a 1000-round native record/fold stress case.

## Mode A/B checks (server logs)

| Mode | Activation | Log evidence |
|---|---|---|
| default | native Vulkan replay | `KVMEM_GDN_MEMORY mode=replay-vulkan layers=48` |
| `KVMEM_GDN_FOLD=ggml` | generic GGML fold | `KVMEM_GDN_MEMORY mode=replay-ggml backend=Vulkan` |
| `--kvmem-mtp-state snapshots` | snapshot fallback | `KVMEM_GDN_ALLOCATION mode=snapshots`, `spec_ckpt tgt=RS` |

All three answered a smoke query correctly.

## 33-round long-context canary (same flow as README)

Command (run by `scripts/multimodal_canary_vulkan.py`):

```
python3 scripts/multimodal_canary_vulkan.py \
  --model Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --mmproj mmproj-Q5_K-MIX.gguf \
  --query-policy user --kv q8_0 --draft-kv f16 --image-max-tokens 512 \
  --budget 36864 --reserve 16384 --ctx 262144 --batch 512 \
  --thinking-budget 128 --long-context-benchmark --long-chunk-tokens 8192 \
  --folder logs/task2-v2
```

Result: **33/33 requests PASS**, 32 tool rounds, context filled to
262058/262144 (99.97% — identical fill to the README run).

| Metric | README IQ3 (RTX 5060 Ti · CUDA) | This PR (RX 7900 XTX · Vulkan) |
|---|---|---|
| Prefill — initial | 437.13 tok/s | 348.41 tok/s |
| Prefill — overall | 242.06 tok/s | 191.23 tok/s |
| Decode | 31.74 tok/s | 49.63 tok/s |
| MTP acceptance | 64.70% | 72.68% |
| RAM peak | 13,483.52 MiB | 20,965 MiB (VmHWM, measured) |
| VRAM | 15,617.10 MiB peak | 14,246.89 MiB (AMD DRM sysfs, 1,475 samples) |
| Context fill | 262058/262144 (99.97%) | 262058/262144 (99.97%) |

### Full summary.json (key fields)

```json
{
  "request_count": 33, "tool_rounds": 32,
  "final_prompt_tokens": 261546, "final_total_tokens": 262058,
  "tool_prefill_s": 1366.52, "tool_decode_s": 85.45,
  "first_pass_prefill_tps": 348.41, "effective_prefill_tps": 191.23,
  "aggregate_decode_tps": 49.63,
  "mtp_drafted": 3481, "mtp_accepted": 2530, "mtp_accept_pct": 72.68,
  "replay_rounds": 26, "total_replay_rows": 212177,
  "first_pass_s": 749.72, "replay_s": 554.63, "retrieval_s": 57.43,
  "peak_runtime_rss_mib": 20965.34, "peak_loading_rss_mib": 11738.13,
  "peak_vram_mib": 14246.89, "min_free_vram_mib": 10313.11,
  "peak_process_swap_mib": 0.0,
  "vram_measurement_source": "/sys/class/drm/card0/device",
  "vram_sample_count": 1475,
  "malloc_arena_max": 2, "trace_enabled": true
}
```

### Notes on comparability

1. Decode and MTP acceptance exceed the CUDA baseline mainly because the
   7900 XTX outclasses the 5060 Ti; this is not a claim that Vulkan beats
   CUDA per se.
2. RAM peak includes the 12.1 GB model resident plus retrieval scoring
   buffers, measured with the arena mitigation in place. Without
   `MALLOC_ARENA_MAX=2`, the same test grows past 47.8 GB on this host
   (see PR description, root-cause section).
3. VRAM is measured device-wide via AMD DRM sysfs
   (`mem_info_vram_used`, 1,475 samples). The README figure is an NVML
   process-level peak; the magnitudes are consistent and no OOM occurred
   at full 256K.
4. The previous estimator (summing logged buffer allocations) was removed:
   telemetry is now measured, or explicitly null — never an estimate
   presented as a measurement.

## Host-memory incident data (root cause evidence)

- Symptom: RSS grows from ~13 GB to 47.8+ GB over the 33-round canary,
  never returning (both `snapshots` and `replay` modes).
- gdb backtrace of the hot allocation: `llama_memory_kvmem_mtp::init_batch
  → llama_batch_allocr::ubatch_add` — a ~10 MB `std::vector<float>`
  (512 ubatch × 5120 embedding × 4 B) allocated and discarded every decode
  step (full backtrace: `gdb-bt2.log` in the evidence archive).
- With `MALLOC_ARENA_MAX=2`: same test peaks at ~21 GB and stays flat.

## Evidence archive

`logs/task2-v2/` (per-request responses, server stderr with trace, summary),
canary stdout, build log, and the A/B server logs are preserved and
available on request.
