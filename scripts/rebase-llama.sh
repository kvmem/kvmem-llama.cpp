#!/usr/bin/env bash
# Bump llama.cpp submodule and replay patches/.
# Usage: scripts/rebase-llama.sh [commit-or-tag]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/llama.cpp"
TARGET="${1:-origin/kvmem/prism}"

cd "$LLAMA"
git fetch --tags origin
git checkout --detach "$TARGET"
echo "llama.cpp now at $(git rev-parse --short HEAD) $(git log -1 --oneline)"
cd "$ROOT"
./scripts/apply-patches.sh
echo "rebuild with: scripts/build-cuda.sh"
