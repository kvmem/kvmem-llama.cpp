# Build NInfer for RTX 3090 on Linux

This guide builds the `sm_86` runtime for one NVIDIA GeForce RTX 3090 or RTX 3090 Ti.
NInfer-3090 v0.6.1 publishes a Linux x64 archive built for this exact SM86 target.

Do not change `CMAKE_CUDA_ARCHITECTURES` to `89`.
The RTX 4090 fork uses Ada-specific schedules that do not apply to the RTX 3090.

## Container build

The repository Dockerfile gives the shortest build path on Bazzite and other Linux distributions.
It uses Ubuntu 24.04, CUDA 13.1, GCC 13, Ninja, FFmpeg, and curl.

Install Docker and the NVIDIA Container Toolkit first.
Then make sure that Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.1.2-runtime-ubuntu24.04 nvidia-smi
```

Build the image from the repository root:

```bash
docker build --tag ninfer-3090:sm86 .
```

Run the Qwen3.8 server with a model directory from the host:

```bash
docker run --rm --gpus all \
  --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-3090:sm86 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 65536 --kv-capacity 65536 \
  --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The API is available at `http://127.0.0.1:8080/v1`.

## Native Ubuntu 24.04 build

Install the CUDA Toolkit 12.8 or newer from NVIDIA.
Then install the host compiler and media dependencies:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential gcc-13 g++-13 cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

Select GCC 13 for host and CUDA compilation:

```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13
```

Configure and build the Linux applications:

```bash
cmake -S . -B build-sm86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=OFF

cmake --build build-sm86 --parallel 2
```

The build creates these applications:

```text
build-sm86/apps/ninfer
build-sm86/apps/ninfer-serve
```

## Optional vcpkg dependencies

Linux can use the pinned `vcpkg.json` manifest instead of system FFmpeg and curl packages.
Bootstrap the pinned vcpkg revision:

```bash
git clone https://github.com/microsoft/vcpkg.git "$HOME/.local/share/vcpkg"
git -C "$HOME/.local/share/vcpkg" checkout 4bca8fd8654e5ba76f92661db7bfe954768ad8ef
"$HOME/.local/share/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
```

Add these options to the native CMake command:

```text
-DCMAKE_TOOLCHAIN_FILE=$HOME/.local/share/vcpkg/scripts/buildsystems/vcpkg.cmake
-DVCPKG_TARGET_TRIPLET=x64-linux
```

## Build options

Every option below is off by default; a default build runs the tested routes. Pass them to the
CMake configure command as `-DNAME=VALUE`.

| option | effect |
|---|---|
| `NINFER_SFU_SIGMOID_SILU=ON` | sigmoid and SiLU evaluate `ex2.approx` and a correctly rounded reciprocal on the SFU instead of `expf` and a divide |
| `NINFER_SFU_SOFTPLUS=ON` | the GDN decay gate's softplus runs on the SFU as well, switching to a log1p series where e^x is below 1/16 |
| `NINFER_BF16_RESIDUAL_ADD=ON` | BF16 linear projections are added onto the residual in bf16 instead of fp32 |
| `NINFER_NVCC_SPLIT_COMPILE=N` | nvcc optimizes one translation unit on N threads (`--split-compile`); `0` is every core, which a parallel ninja multiplies by its job count |
| `NINFER_PTXAS_VERBOSE=ON` | ptxas reports each kernel's registers and local-memory spills |
| `NINFER_WEBUI_DIR=PATH` | compiles the WebUI in that directory (an `index.html` or `index.html.gz` at its root) into the server |
| `NINFER_SM120_NATIVE=ON` | on a `120a` build, compiles upstream's native routes instead of the `mma.sync` path (every `120a` build compiles the FP8 A8 and NVFP4 W4A4 units) |
| `NINFER_PDL=ON` | on a `120a` build of the `mma.sync` path, launches decode-graph kernels as programmatic dependents, so a kernel stages its weights while the one before it finishes (the native routes always do) |
| `NINFER_TMA_STAGING=ON` | passes the NVFP4 TMA descriptors through device memory with a capturable staging kernel, as Windows builds must, so that path can be tested on Linux (`120a`) |
| `NINFER_MEDIA_NATIVE_PNG=ON` | decodes PNG images natively instead of through FFmpeg, as Windows builds do by default, so that decoder can be tested on Linux |
| `NINFER_D3D12_RESIDENCY=ON` | Windows: offers `--wddm-evictable-budget`, device arenas from a D3D12 heap held resident |
| `NINFER_DIRECTSTORAGE=ON` | Windows: fetches the DirectStorage 1.3 runtime and offers `--disk-kv-directstorage` for disk-tier restores |

The Windows options compile against Windows headers but have not been run on this line.
At run time, `NINFER_CUDA_SYNC=spin|blocking|yield|auto` selects the CUDA synchronization schedule
(unset keeps CUDA's default) and `NINFER_T2_A8_TILE=off` returns the ternary prefill route to its
64-row kernel.

## Bash scripts

The `scripts/` directory contains Bash versions of each Windows download, launcher, and packaging
script. Download scripts save models under `scripts/models` by default:

```bash
./scripts/download-model.sh qwen38-27b
./scripts/download-model.sh qwen36-35b-a3b
./scripts/download-model.sh qwen36-27b
```

Set `NINFER_MODEL_DIR` to use another model directory. `run.sh` serves either model, and
`NINFER_MODEL` points it at a model path anywhere:

```bash
./scripts/run.sh qwen38-27b
./scripts/run.sh qwen36-35b-a3b
NINFER_MODEL=/path/to/qwen3_8_27b.ninfer ./scripts/run.sh qwen38-27b int8
```

The launcher uses `build-linux/apps/ninfer-serve` when the executable is not beside the script.
Set `NINFER_SERVER` to select another executable. Profiles and the measurements behind them are in
[launcher profiles](maintainer/launcher-profiles.md).

`scripts/package-release.sh` creates the Linux `tar.gz` archive for the release named in `VERSION`,
and `scripts/package-release.ps1` the Windows ZIP; `build.sh --package` and `build.ps1 -Package` run
them after a build.

## Validation

Make sure that the applications start:

```bash
./build-sm86/apps/ninfer --help
./build-sm86/apps/ninfer-serve --help
```

Run one short generation with the real Qwen3.8 artifact:

```bash
./build-sm86/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode in two sentences." \
  --max-context 8192 --max-new 32 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

A successful compile does not qualify Linux performance.
Record the GPU, driver, CUDA Toolkit, compiler, workload, and result before you publish Linux measurements.
