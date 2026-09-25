#!/usr/bin/env bash
# Replay maintained diffs without commits or Git identity changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${KVMEM_LLAMA_DIR:-$ROOT/llama.cpp}"
PATCH="$ROOT/patches/llama-kvmem-current.patch"
BUDGET_UPGRADE="$ROOT/patches/reasoning-budget-upgrade.patch"
REPLAY_UPGRADE="$ROOT/patches/replayssm-upgrade.patch"
UPGRADE="$ROOT/patches/multimodal-upgrade.patch"
VULKAN="$ROOT/patches/vulkan-support.patch"
VULKAN_REPLAY="$ROOT/patches/vulkan-replay.patch"
cd "$LLAMA"

all_patches_applied() {
    local check_dir patch added removed path rc=1
    git apply --reverse --check "$VULKAN_REPLAY" 2>/dev/null || return 1
    check_dir="$(mktemp -d)"
    for patch in "$PATCH" "$VULKAN" "$VULKAN_REPLAY"; do
        while IFS=$'\t' read -r added removed path; do
            if [[ -f "$path" ]]; then
                mkdir -p "$check_dir/$(dirname "$path")"
                cp -p -- "$path" "$check_dir/$path"
            fi
        done < <(git apply --numstat "$patch")
    done
    if (cd "$check_dir" &&
            git apply --reverse "$VULKAN_REPLAY" &&
            git apply --reverse "$VULKAN" &&
            git apply --reverse --check "$PATCH") 2>/dev/null; then
        rc=0
    fi
    rm -rf -- "$check_dir"
    return "$rc"
}

base_patches_applied() {
    local check_dir patch added removed path rc=1
    git apply --reverse --check "$VULKAN" 2>/dev/null || return 1
    check_dir="$(mktemp -d)"
    for patch in "$PATCH" "$VULKAN"; do
        while IFS=$'\t' read -r added removed path; do
            if [[ -f "$path" ]]; then
                mkdir -p "$check_dir/$(dirname "$path")"
                cp -p -- "$path" "$check_dir/$path"
            fi
        done < <(git apply --numstat "$patch")
    done
    if (cd "$check_dir" &&
            git apply --reverse "$VULKAN" &&
            git apply --reverse --check "$PATCH") 2>/dev/null; then
        rc=0
    fi
    rm -rf -- "$check_dir"
    return "$rc"
}

if all_patches_applied; then
    echo "KVMem patches already applied"
    echo "Vulkan ReplaySSM patch already applied"
    exit 0
fi

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

VULKAN_READY=0
if base_patches_applied; then
    echo "KVMem patches already applied"
    echo "Vulkan/generic backend patch already applied"
    VULKAN_READY=1
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
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

# Generic-backend (Vulkan) support: adds the non-CUDA stage-in source and the
# LLAMA_KVMEM_CUDA compile definition. Independent of the main patch.
if [ -f "$VULKAN" ]; then
    if (( VULKAN_READY )); then
        :
    elif git apply --reverse --check "$VULKAN" 2>/dev/null; then
        echo "Vulkan/generic backend patch already applied"
    elif git apply --check "$VULKAN" 2>/dev/null; then
        git apply "$VULKAN"
        echo "applied Vulkan/generic backend patch"
    else
        echo "warning: $VULKAN does not apply to this tree; skipping" >&2
    fi
fi

if [ -f "$VULKAN_REPLAY" ]; then
    if git apply --reverse --check "$VULKAN_REPLAY" 2>/dev/null; then
        echo "Vulkan ReplaySSM patch already applied"
    elif git apply --check "$VULKAN_REPLAY" 2>/dev/null; then
        git apply "$VULKAN_REPLAY"
        echo "applied Vulkan ReplaySSM patch"
    else
        echo "warning: $VULKAN_REPLAY does not apply to this tree; skipping" >&2
    fi
fi
