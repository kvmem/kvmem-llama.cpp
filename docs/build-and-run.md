# Build and run (any configuration)

The outer repo is a thin layer around the `llama.cpp` submodule. The submodule
is the pin; all local modifications live in `patches/` and are replayed by
`scripts/apply-patches.sh`.

## 1. Get the tree

```bash
git clone --recurse-submodules https://github.com/<you>/kvmem-llama.cpp.git
cd kvmem-llama.cpp
./scripts/apply-patches.sh        # idempotent, safe to re-run
```

## 2. Build

`scripts/build-cuda.sh` is the reference path. Override these environment
variables for your machine - nothing is hardcoded:

| variable | default | notes |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | `120a-real` | **set this to your GPU**, e.g. `86-real` (RTX 30xx), `89-real` (RTX 40xx), `120a-real` (RTX 50xx) |
| `CMAKE_CUDA_COMPILER` | `nvcc` | path to nvcc if CUDA is not on PATH |
| `CMAKE_BUILD_TYPE` | `Release` | |
| `BUILD_DIR` | `<root>/build` | |
| `NPROC` | `nproc` | parallel jobs |

```bash
CMAKE_CUDA_ARCHITECTURES=86-real ./scripts/build-cuda.sh
# binaries land in build/bin
```

Windows: `powershell -File scripts/windows/build.ps1` (see
`scripts/windows/README.md`).

Other backends - no CUDA needed:

```bash
# CPU only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
      -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT="$PWD"
cmake --build build -j

# Apple Metal, ROCm, Vulkan, SYCL ... switch GGML_CUDA=OFF and enable the
# corresponding GGML_<BACKEND>=ON. See llama.cpp/docs/build.md.
```

`-DGGML_CUDA_FA_ALL_QUANTS=ON` is what `build-cuda.sh` uses; keep it if you want
`q5_0` KV. `q4_0` and `q8_0` work without it.

## 3. Run

```bash
build/bin/llama-kvmem-server \
  -m /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -c 163840 -ngl 99 -fa on -np 1 \
  -ctk q8_0 -ctv q8_0 \
  --kvmem-budget 24576 --kvmem-gen-reserve 12288 --kvmem-block-tokens 64 \
  --kvmem-query-policy user \
  --auto-continue-thinking --act-round-tokens 6000 --act-max-rounds 8 \
  --act-total-tokens 36000 --act-prefill-mode auto \
  --enable-thinking --reasoning-budget -1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

Sizing rules that are easy to get wrong:

- the GPU KV pool is `--kvmem-budget + --kvmem-gen-reserve` tokens;
- `--act-round-tokens` must be smaller than `--kvmem-gen-reserve`;
- rough VRAM cost per resident token, including the draft KV:
  `q4_0 ~ 22 KiB`, `q8_0 ~ 32 KiB`;
- **idle VRAM is misleading** - the staging buffers only appear under load, so
  fire one real request before trusting the headroom;
- reserve extra VRAM for the MTP block (~0.35 GB) plus its draft KV.

## 4. Verify the loop

Send the same request twice, once with `"auto_continue_thinking": false` and
once with `true`. The second one must report a much larger `completion_tokens`
and `finish_reason: "stop"`, while the server log shows one
`KVMEM_ACT_ROUND` per continuation with `in_think=1`.