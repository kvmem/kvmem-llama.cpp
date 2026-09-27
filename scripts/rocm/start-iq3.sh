#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 MODEL.gguf MMPROJ.gguf [auto|ROCmN[,ROCmN...]]" >&2
    exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${BUILD_DIR:-${root}}"
if [[ ! -x "${build}/bin/llama-kvmem-server" && -z "${BUILD_DIR:-}" ]]; then
    build="${root}/build-hip-linux"
fi
model="$1"; mmproj="$2"; gpu="${3:-auto}"
budget="${KVMEM_GPU_KV_BUDGET:-28672}"
port=18200
ui="${build}/share/kvmem/ui"
[[ "$gpu" == auto || "$gpu" =~ ^ROCm[0-9]+(,ROCm[0-9]+)*$ ]] || { echo 'Select auto or comma-separated devices from --list-devices.' >&2; exit 2; }
[[ "$budget" =~ ^[0-9]+$ ]] && (( budget >= 8192 && budget <= 262144 )) || { echo 'KVMEM_GPU_KV_BUDGET must be 8192..262144 tokens.' >&2; exit 2; }
for file in "$model" "$mmproj" "${build}/bin/llama-kvmem-server" "${ui}/index.html"; do
    [[ -f "$file" ]] || { echo "Missing file: $file" >&2; exit 2; }
done
export LD_LIBRARY_PATH="${build}/lib:${build}/bin:${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if grep -qi microsoft /proc/sys/kernel/osrelease; then
    export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
fi
available_devices="$("${build}/bin/llama-kvmem-server" --list-devices)"
if [[ "$gpu" == auto ]]; then
    gpu="$(printf '%s\n' "$available_devices" |
        sed -nE 's/^[[:space:]]*(ROCm[0-9]+):.*\(([0-9]+) MiB,.*/\2 \1/p' |
        sort -nr | head -n 1 | awk '{print $2}')"
    [[ -n "$gpu" ]] || { echo "No ROCm GPU is available: $available_devices" >&2; exit 2; }
else
    IFS=',' read -ra selected <<< "$gpu"
    for device in "${selected[@]}"; do
        grep -q "^[[:space:]]*${device}:" <<< "$available_devices" || { echo "Unavailable GPU: $device" >&2; exit 2; }
    done
fi
split="${KVMEM_SPLIT_MODE:-layer}"
[[ "$split" =~ ^(none|layer|row)$ ]] || { echo 'Invalid KVMEM_SPLIT_MODE' >&2; exit 2; }
extra=(--split-mode "$split")
if [[ -n "${KVMEM_TENSOR_SPLIT:-}" ]]; then extra+=(--tensor-split "$KVMEM_TENSOR_SPLIT"); fi
echo "IQ3 / HIP / ${gpu}: http://127.0.0.1:${port}/ (Ctrl+C to stop)"
exec "${build}/bin/llama-kvmem-server" \
    -m "$model" --mmproj "$mmproj" --no-mmproj-offload \
    --device "$gpu" "${extra[@]}" -ngl 99 --load-mode auto \
    --host 127.0.0.1 --port "$port" --webui --ui-dir "$ui" \
    -c 262144 -b 512 -n 16384 \
    --kvmem --kvmem-budget "$budget" --kvmem-gen-reserve 16384 \
    --kvmem-block-tokens 128 --kvmem-query-policy user --kv-dtype q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 2 --spec-kv-dtype "${KVMEM_DRAFT_KV:-f16}" \
    --kvmem-mtp-state replay --image-max-tokens 512 \
    --enable-thinking --reasoning-budget 4096
