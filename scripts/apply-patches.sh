#!/usr/bin/env bash
# Replay maintained diffs without commits or Git identity changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${KVMEM_LLAMA_DIR:-$ROOT/llama.cpp}"
PATCH="$ROOT/patches/llama-kvmem-current.patch"
RDNA2_FATTN="$ROOT/patches/0005-hip-rdna2-quantized-kv-fa-vec.patch"
BUDGET_UPGRADE="$ROOT/patches/reasoning-budget-upgrade.patch"
REPLAY_UPGRADE="$ROOT/patches/replayssm-upgrade.patch"
UPGRADE="$ROOT/patches/multimodal-upgrade.patch"
cd "$LLAMA"

# An incremental patch may also fit an older, incomplete tree. Check that
# its result contains the entire current patch before changing live files.
can_upgrade() {
    local upgrade="$1" check_dir added removed path rc=1
    git apply --check "$upgrade" 2>/dev/null || return 1
    check_dir="$(mktemp -d)"
    while IFS=$'\t' read -r added removed path; do
        if [[ -f "$path" ]]; then
            mkdir -p "$check_dir/$(dirname "$path")"
            cp -p -- "$path" "$check_dir/$path"
        fi
    done < <(git apply --numstat "$PATCH")
    if (cd "$check_dir" && git apply "$upgrade" && git apply --reverse --check "$PATCH") 2>/dev/null; then
        rc=0
    fi
    rm -rf -- "$check_dir"
    return "$rc"
}

if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "KVMem patches already applied"
elif git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "applied current KVMem patch to pinned llama.cpp"
elif can_upgrade "$BUDGET_UPGRADE"; then
    git apply "$BUDGET_UPGRADE"
    echo "upgraded existing KVMem tree with reasoning budget fix"
elif can_upgrade "$REPLAY_UPGRADE"; then
    git apply "$REPLAY_UPGRADE"
    echo "upgraded existing KVMem tree with ReplaySSM support"
elif can_upgrade "$UPGRADE"; then
    git apply "$UPGRADE"
    echo "upgraded existing KVMem tree with multimodal and ReplaySSM support"
else
    echo "llama.cpp differs from the supported pin or KVMem baseline; no files changed" >&2
    echo "inspect local changes before replaying $PATCH" >&2
    exit 1
fi

if git apply --reverse --check "$RDNA2_FATTN" 2>/dev/null; then
    echo "gfx1030 RDNA2 quantized-KV Flash Attention patch already applied"
else
    git apply --check "$RDNA2_FATTN"
    git apply "$RDNA2_FATTN"
    echo "applied gfx1030 RDNA2 quantized-KV Flash Attention patch"
fi
