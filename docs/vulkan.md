# Vulkan backend (AMD Radeon, Intel Arc, …)

KVMem builds on llama.cpp's CUDA backend by default. This document describes the
generic-backend build, which targets llama.cpp's **Vulkan** backend. No CUDA
toolkit is required. Backend availability is not a guarantee that every GPU,
driver, model layout, or quantization is supported.

## What works, what changes

The implementation includes the bounded GPU working set, host-RAM block store,
query-based retrieval, MTP speculative decoding and the OpenAI-compatible
server. Because the CUDA staging kernels are GPU-specific, the generic build
routes staging through portable paths:

| Piece | CUDA build | Generic (Vulkan) build |
| --- | --- | --- |
| KV block H2D/D2H staging | async CUDA pipeline (`llama-kvmem-stagein.cu`) | synchronous `ggml_backend_tensor_set/get` |
| Layout compaction | batched device D2D via scratch | host-bounce payloads, then re-stage |
| RoPE + Hadamard + quant on stage-in | fused GPU kernel | host compute, then H2D |
| Decode mean-K accumulation | GPU kernel | host accumulation |
| GDN replay record | CUDA GDN kernel | Vulkan GDN kernel |
| GDN replay fold (ReplaySSM) | CUDA kernel | Native Vulkan state and convolution-history compute shaders |

The first four use portable staging: retrieval steps transfer KV
blocks synchronously. Their end-to-end performance needs measurement on the
target device. ReplaySSM (`--kvmem-mtp-state replay`) uses
Vulkan-native record and fold shaders. Native fold failure fails the replay
transaction; it does not silently switch to a GGML graph. Set
`--kvmem-mtp-state snapshots` to force the old rollback-snapshot path.
`KVMEM_MTP_STATE` is translated by `start-server.py`; it is not a substitute
for this flag when invoking the binary directly.

Replay fold dispatch follows the device that owns the recurrent tensors.
The fold tests also run in a binary containing both CUDA and Vulkan. However,
the full KVMem adapter still selects its KV staging implementation at build time.
Build with `GGML_CUDA=OFF` for Vulkan inference. The CUDA adapter rejects other
GPU backends at initialization instead of passing their buffers to CUDA.
Set `KVMEM_GDN_FOLD=ggml` to force the portable graph fold for A/B validation
or driver troubleshooting; leave it unset for the native shader path.

The Vulkan fold currently specializes the Qwen 27B GDN layout used by KVMem.
The explicitly selected GGML implementation uses the same shape checks and is also exercised on CPU.
It requires backend support for `GATED_DELTA_NET` and the copy/concat operations;
it does not imply support on every GGML backend.

## Build (Ubuntu 24.04)

```bash
sudo apt install build-essential cmake git \
    libvulkan-dev glslc spirv-headers          # shader toolchain
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp
scripts/apply-patches.sh                        # also applies patches/vulkan-support.patch
scripts/build-vulkan.sh                         # binaries under build-vulkan/bin/
```

For native Linux AMD testing, install the **RADV** driver (part of Mesa):

```bash
sudo apt install mesa-vulkan-drivers vulkan-tools
vulkaninfo --summary   # your GPU must appear here
```

Confirm the selected device is the physical GPU, not llvmpipe/lavapipe.
Do not infer WSL GPU support from a successful build: the Linux Vulkan loader
inside WSL must enumerate an accelerated device. This workspace currently
validates software Vulkan locally; native Linux AMD results do not establish
AMD acceleration inside WSL.

## Run

The `start-iq3.sh`/`start-iq4.sh` launchers do GPU selection through
`nvidia-smi`; on Vulkan machines use the NVIDIA-independent wrapper:

```bash
bash scripts/start-vulkan.sh \
  -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --mmproj mmproj-Qwen3.8-27B-Q5_K-MIX.gguf --mmproj-offload --image-max-tokens 512 \
  -c 262144 -n 16384 \
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 \
  --kv-dtype q8_0 \
  --spec-type draft-mtp \
  --kvmem-mtp-state replay \
  --enable-thinking --reasoning-budget 4096 \
  --port 18200
```

The wrapper defaults `MALLOC_ARENA_MAX=2` before process startup, preserving
an explicit user value. This is a glibc allocator-retention mitigation, not
proof that every allocation leak or long-context memory problem is fixed.
Direct binary invocations must set it themselves if desired.

`scripts/multimodal_canary_vulkan.py` reads AMD device-wide VRAM usage from
DRM sysfs, or an explicitly available `KVMEM_VRAMMON` helper. Select a device
with `KVMEM_DRM_DEVICE=/sys/class/drm/card0/device` when ambiguous. Missing
telemetry produces `peak_vram_mib: null`; log buffer sizes are only an allocation
estimate, not a measured peak. Use `--trace` when collecting trace-based timing.

Multi-GPU: llama.cpp's Vulkan backend enumerates all Vulkan devices; use the
standard llama.cpp device selection (`--device Vulkan0`, `--main-gpu`, etc.)
instead of `CUDA_VISIBLE_DEVICES`.

## Notes and limitations

- The 16 GiB recipes require per-device validation; a CUDA memory budget does
  not guarantee identical Vulkan allocations or headroom.
- `--kv-dtype q5_0` needs flash-attention support for that type in the Vulkan
  backend; if the backend refuses it, drop to `q8_0` or add `--no-flash-attn`.
- NVMe spill is not implemented (same as the CUDA build).
- ReplaySSM passes an independent FP64 recurrence check and same-backend snapshot
  comparisons, including production cache shapes and distinct multilayer states.
  Software-Vulkan tests do not establish that the reported 40 GB end-to-end
  host-memory growth is fixed; that requires the original multi-request workload.
- ReplaySSM shaders pass the CPU and software-Vulkan regression suite;
  AMD hardware validation is still required.
- Windows Vulkan builds and WSL AMD hardware acceleration have not been
  validated by this integration.
