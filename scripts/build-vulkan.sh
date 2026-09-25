#!/usr/bin/env bash
# Configure + build libllama (Vulkan) and llama-kvmem-server.
# Target: AMD Radeon (RDNA3/RDNA4 e.g. RX 7900 XTX / 9060 XT), Intel Arc,
# and any other GPU with a Vulkan 1.2+ driver. No CUDA toolkit required.
#
# Requires: C++17 compiler, CMake, Vulkan headers (libvulkan-dev) and a GLSL
# compiler (glslc from shaderc).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CMAKE="${CMAKE:-$ROOT/.venv/bin/cmake}"
if [[ ! -x "$CMAKE" ]]; then
    CMAKE="$(command -v cmake)"
fi

BUILD="${BUILD_DIR:-$ROOT/build-vulkan}"
TYPE="${CMAKE_BUILD_TYPE:-Release}"

"$CMAKE" -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE="$TYPE" \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=ON \
    -DKVMEM_BUILD_LLAMA=ON \
    -DLLAMA_KVMEM=ON \
    -DLLAMA_KVMEM_ROOT="$ROOT" \
    "$@"

"$CMAKE" --build "$BUILD" -j"${NPROC:-$(nproc)}"
echo "binaries under $BUILD/bin"
