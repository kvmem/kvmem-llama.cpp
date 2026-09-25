#!/usr/bin/env bash
# Pass server CLI options verbatim; no NVIDIA device discovery.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-vulkan}"
BINARY="$BUILD/bin/llama-kvmem-server"
if [[ ! -x "$BINARY" ]]; then
    echo "Build the Vulkan server first: bash scripts/build-vulkan.sh" >&2
    exit 1
fi
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export LD_LIBRARY_PATH="$BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$BINARY" "$@"
