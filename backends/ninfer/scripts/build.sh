#!/usr/bin/env bash
# Build NInfer-3090 on Linux, including WSL2, with the toolchain this project needs.
#
# The usual stumbling block is that the CUDA toolkit is installed but nvcc is not on PATH -
# /usr/local/cuda-12.8/bin is not added by default on Ubuntu. Configure then fails the CMakeLists
# version guard with a message about the compiler rather than about PATH. This script finds the
# toolkit itself and forces it through CUDACXX.
#
#   ./scripts/build.sh                 configure + build into build-linux
#   ./scripts/build.sh --test          ... then run the test suite
#   ./scripts/build.sh --package       ... then build the release archive for VERSION
#   ./scripts/build.sh --benchmarks    ... include bench/ (implied by --package)
#   ./scripts/build.sh --clean         delete the build directory first
#   ./scripts/build.sh --target ninfer-serve
#
# WSL note: building under /mnt/c goes through the 9p filesystem and is markedly slower than
# building inside the Linux filesystem. It works, and it keeps one dist/ for both platforms, but
# clone to ~/ instead if you are iterating.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${NINFER_BUILD_DIR:-$repo_root/build-linux}"
arch="${NINFER_CUDA_ARCH:-86}"
run_tests=0
clean=0
package=0
target=''
benchmarks=0

while (( $# )); do
  case "$1" in
    --test) run_tests=1; shift ;;
    --clean) clean=1; shift ;;
    --benchmarks) benchmarks=1; shift ;;
    --package) package=1; shift ;;
    --target) target="${2:-}"; shift 2 ;;
    --build-dir) build_dir="${2:-}"; shift 2 ;;
    --arch) arch="${2:-}"; shift 2 ;;
    # Print the header comment block, stopping at the first line that is not a comment. A pinned
    # line range drifts every time the block grows: it was '2,20p' against a block ending at 17,
    # so --help trailed `set -euo pipefail` and a blank line.
    -h|--help) sed -n '2,${/^#/!q;p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

# The packager ships bench/ninfer_bench alongside the CLI and the server, but benchmarks
# are an opt-in subdirectory (NINFER_BUILD_BENCHMARKS defaults to OFF). Configuring without them and
# then packaging fails late, after the whole tree has been built, with "Missing release product:
# .../bench/ninfer_bench" - so make --package imply the option rather than leaving the two settings
# to be kept consistent by hand.
if (( package )); then benchmarks=1; fi

case "$arch" in 80|86|89) ;; *) printf 'CUDA arch must be 80, 86 or 89, got %s\n' "$arch" >&2; exit 2 ;; esac

# --- locate the toolchain ---------------------------------------------------------------------

if [[ -z "${CUDACXX:-}" ]]; then
  for candidate in /usr/local/cuda-12.8/bin/nvcc /usr/local/cuda-12.9/bin/nvcc \
                   /usr/local/cuda/bin/nvcc "$(command -v nvcc 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then CUDACXX="$candidate"; break; fi
  done
fi
if [[ -z "${CUDACXX:-}" ]]; then
  cat >&2 <<'MISSING'
No nvcc found. CMakeLists requires CUDA >= 12.8. Looked in /usr/local/cuda-12.8/bin,
/usr/local/cuda-12.9/bin, /usr/local/cuda/bin and PATH. Install the toolkit, or set CUDACXX
to its nvcc if it lives elsewhere.
MISSING
  exit 1
fi
export CUDACXX
export PATH="$(dirname -- "$CUDACXX"):$PATH"

for tool in cmake ninja; do
  command -v "$tool" >/dev/null || { printf 'Missing %s. apt install cmake ninja-build\n' "$tool" >&2; exit 1; }
done

printf 'toolchain: %s (%s)\n' "$CUDACXX" "$("$CUDACXX" --version | tail -1)"
printf 'toolchain: %s\n' "$(command -v c++)"

# --- configure and build ----------------------------------------------------------------------

if (( clean )) && [[ -d "$build_dir" ]]; then
  printf 'removing %s\n' "$build_dir"
  rm -rf -- "$build_dir"
fi

cd -- "$repo_root"
benchmarks_option=OFF
if (( benchmarks )); then benchmarks_option=ON; fi

cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" \
      -DNINFER_BUILD_BENCHMARKS="$benchmarks_option"
if [[ -n "$target" ]]; then
  cmake --build "$build_dir" --target "$target"
else
  cmake --build "$build_dir"
fi

if (( run_tests )); then
  # One GPU, so keep the parallelism low: unrelated CUDA tests contend for memory and produce
  # failures that do not reproduce when the test is run on its own.
  ctest --test-dir "$build_dir" -j2 --output-on-failure
fi

if (( package )); then
  NINFER_BUILD_ROOT="$build_dir" "$repo_root/scripts/package-release.sh"
fi

printf '\nbuilt into %s\n' "$build_dir"
printf '  server : %s\n' "$build_dir/apps/ninfer-serve"
printf '  cli    : %s\n' "$build_dir/apps/ninfer"
