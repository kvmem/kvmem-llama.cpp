#!/usr/bin/env bash
# Configure + build kvmem tests, libllama (CUDA sm_120), and llama-kvmem-cli.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CU13="${CU13:-/home/leye/kvmem_qw3/.cu13-env}"
if [[ -x "$CU13/bin/nvcc" ]]; then
    export PATH="$CU13/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CU13/lib:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="$CU13"
    CMAKE_CUDA_COMPILER="$CU13/bin/nvcc"
else
    CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-nvcc}"
fi

CMAKE="${CMAKE:-$ROOT/.venv/bin/cmake}"
if [[ ! -x "$CMAKE" ]]; then
    CMAKE="$(command -v cmake)"
fi

BUILD="${BUILD_DIR:-$ROOT/build}"
TYPE="${CMAKE_BUILD_TYPE:-Release}"

"${PYTHON:-python3}" "$ROOT/scripts/prepare-backends.py" --backend llamacpp

"$CMAKE" -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE="$TYPE" \
    -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-120a-real}" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_QUANTS=all \
    -DKVMEM_BUILD_LLAMA=ON \
    -DLLAMA_KVMEM=ON \
    -DLLAMA_KVMEM_ROOT="$ROOT"

"$CMAKE" --build "$BUILD" -j"${NPROC:-$(nproc)}"
echo "binaries under $BUILD/bin"
