# NInfer-3090 v0.7.0

This release ports three additional KV-cache storage formats to Ampere/Ada
(RTX 3090 / RTX 4090) and ships native Windows and Linux x64 binaries built
against the current Qwen3.6-35B-A3B artifact on the same explicit SM86
runtime profile used by prior releases.

## Changes

- Adds `--kv-dtype nvfp4`: NVFP4 (e2m1) key and value planes with a group-16
  scale, previously recognized but rejected on this fork. Both the decode
  (small-context) and prompt (long-context) attention routes are ported, with
  a new non-warp-specialized prompt kernel replacing NVFP4's original
  Hopper+-only implementation.
- Adds `--kv-dtype k8v4`: an FP8 E4M3 key plane paired with an NVFP4 value
  plane, decode + prompt + append all ported.
- Adds `--kv-dtype fp8`: plain FP8 E4M3 KV-cache, decode + prompt ported.
  sm_86/sm_89 have no FP8 tensor-core path at all (unlike INT8, which keeps
  a native path), so FP8, K8V4, and NVFP4 attention dequantize K (and, for
  K8V4/NVFP4, V) to BF16/FP16 up front and run QK on native BF16 MMA instead.
- Extends the KV-cache page-pool planner so `--kv-dtype nvfp4|k8v4|fp8` work
  end-to-end from the CLI/server down to the underlying paged KV-cache
  allocation, not just inside the attention/append kernels.
- Fixes a KV-cache append dispatch gap that silently overflowed device memory
  for the NVFP4/K8V4 profiles, and a test-harness bug that could mask
  already-computed decode-path results whenever a prompt-path case in the
  same test group was skipped for an unsupported architecture.

## Validation

The release build and smoke gate uses these components:

- Windows: Visual Studio 2022 BuildTools and CUDA 12.8
- Linux: WSL2 Ubuntu, CUDA Toolkit 12.8, GCC, CMake, Ninja

Each packaged `ninfer` binary completes a real RTX 3090 Qwen3.6-35B-A3B
generation under `--kv-dtype nvfp4`, `k8v4`, and `fp8` before publication,
in addition to the full `ctest` suite (110/110 passing).

## Limits

- NVFP4/K8V4/FP8 attention on sm_86/sm_89 dequantizes to BF16/FP16 before the
  QK product rather than using native low-precision tensor cores, so their
  throughput profile is closer to the BFloat16 KV-cache route than to INT8's.
- This release does not change the default KV-cache dtype; existing
  `--kv-dtype bf16|int8|rk8v4` deployments are unaffected.
