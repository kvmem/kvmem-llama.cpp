#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rocm="${ROCM:-${ROCM_PATH:-${HIP_PATH:-}}}"
build_dir="${BUILD_DIR:-${root}/build-hip-linux}"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

if [[ -z "${rocm}" ]]; then
    echo "[error] ROCm root not set. Set ROCM, ROCM_PATH or HIP_PATH." >&2
    exit 1
fi

rocm_bin="${rocm}/bin"
if [[ ! -x "${rocm_bin}/clang++" ]]; then
    rocm_bin="${rocm}/llvm/bin"
fi
if [[ ! -x "${rocm_bin}/clang++" ]]; then
    rocm_bin="${rocm}/lib/llvm/bin"
fi
if [[ ! -x "${rocm_bin}/clang++" ]]; then
    echo "[error] AMD clang not found under ${rocm}." >&2
    exit 1
fi

export PATH="${rocm_bin}:${PATH}"
export ROCM_PATH="${rocm}"

# Replay the cumulative integration and RDNA2 quantized-KV FA fix.
"${root}/scripts/apply-patches.sh"

configure_args=()
target_args=()
targets="${GPU_TARGETS:-${AMDGPU_TARGETS:-}}"
if [[ -n "${targets}" ]]; then
    target_args+=("-DGPU_TARGETS=${targets}" "-DCMAKE_HIP_ARCHITECTURES=${targets}")
fi
if [[ "${FRESH:-0}" == "1" ]]; then
    # CMake >= 3.24 resets its cache without deleting an arbitrary directory.
    configure_args+=(--fresh)
fi

cmake "${configure_args[@]}" -S "${root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC:-${rocm_bin}/clang}" \
    -DCMAKE_CXX_COMPILER="${CXX:-${rocm_bin}/clang++}" \
    -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER:-${rocm_bin}/clang++}" \
    -DGGML_HIP=ON \
    -DROCM_PATH="${rocm}" \
    -DGGML_HIP_UMA=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DKVMEM_ENABLE_NVME=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    '-DCMAKE_INSTALL_RPATH=$ORIGIN;$ORIGIN/../lib;$ORIGIN/../kvmem' \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF \
    -DGGML_VULKAN=OFF \
    -DKVMEM_BUILD_LLAMA=ON \
    -DLLAMA_KVMEM=ON \
    -DLLAMA_KVMEM_ROOT="${root}" \
    "${target_args[@]}" \
    "$@"

if [[ "${CONFIGURE_ONLY:-0}" != "1" ]]; then
    cmake --build "${build_dir}" --config Release -j"${jobs}"
fi
