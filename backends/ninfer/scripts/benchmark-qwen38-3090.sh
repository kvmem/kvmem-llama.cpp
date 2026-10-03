#!/usr/bin/env bash
# Linux counterpart of benchmark-qwen38-3090.bat: the Qwen3.8-27B cohort sweep on one RTX 3090.
set -euo pipefail

# ======================== EDITABLE SETTINGS ========================
MAX_CONTEXT="${NINFER_BENCH_MAX_CONTEXT:-131072}"
OUTPUT_TOKENS="${NINFER_BENCH_OUTPUT_TOKENS:-1024}"
PREFILL_PROMPT_CHARACTERS="${NINFER_BENCH_PREFILL_CHARS:-28000}"
COHORTS="${NINFER_BENCH_COHORTS:-1,2,4,8}"
KV_DTYPE="${NINFER_BENCH_KV_DTYPE:-rk8v4}"
START_DELAY_SECONDS="${NINFER_BENCH_START_DELAY:-10}"
# Hands wide prefill GEMMs to cuBLAS: about 1.73x prefill for +0.156% perplexity (4.343155 ->
# 4.349944 on the 1M corpus). Set to 0 to measure the default-quality engine instead. The chunk
# follows it, because the route only amortises its weight-sized dequantise over a call's tokens,
# and at this sweep's usual 512 it is a loss.
# This sweep stays on MTP3 for memory, not speed. DFlash2 at K=7 is faster at every concurrency it
# can run -- aggregate decode tok/s C1/C2/C4 of 187.1/313.5/406.2 against MTP3's 135.0/238.2/387.4,
# so +39%/+32%/+5% -- the lead shrinking because batching already amortises the weight sweep that
# speculation exploits. What stops it is that at C8 it does not fit: the draft model's weights are
# 18.3 GiB against 16.7, and only a 2048-token KV leaves room, which is too little for eight
# streams. This sweep includes C8. A C1-C4 deployment should set NINFER_BENCH_SPEC=dflash2.
SPEC="${NINFER_BENCH_SPEC:-mtp}"
DRAFT_TOKENS="${NINFER_BENCH_DRAFT_TOKENS:-}"
PREFILL_CUBLAS="${NINFER_BENCH_PREFILL_CUBLAS:-1}"
PREFILL_CHUNK="${NINFER_BENCH_PREFILL_CHUNK:-}"
# ==================================================================

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# download-model.sh qwen38-27b fetches the DFlash2 bundle as qwen3_8_27b.ninfer, and it carries the
# MTP weights too, so that one file runs every spec. A maintainer who keeps the bundle as a separate
# qwen3_8_27b_dflash2.ninfer beside a dense qwen3_8_27b.ninfer gets it preferred for DFlash2.
models_dir="${NINFER_MODEL_DIR:-$repo/..}"
default_model='qwen3_8_27b.ninfer'
if [[ "$SPEC" == 'dflash2' && -f "$models_dir/qwen3_8_27b_dflash2.ninfer" ]]; then
  default_model='qwen3_8_27b_dflash2.ninfer'
fi
model="${NINFER_BENCH_MODEL:-$models_dir/$default_model}"
server="${NINFER_BENCH_SERVER:-$repo/build-linux/apps/ninfer-serve}"

if [[ ! -x "$server" ]]; then
  printf 'ERROR: Server not found: %s\n' "$server" >&2
  printf 'Build it first:  ./scripts/build.sh\n' >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  printf 'ERROR: Model not found: %s\n' "$model" >&2
  printf 'Download it first:  ./scripts/download-model.sh qwen38-27b  (that file carries the DFlash2 weights)\n' >&2
  exit 1
fi
command -v uv >/dev/null || { printf 'ERROR: uv is not available in PATH.\n' >&2; exit 1; }

printf '\nRTX 3090 Qwen3.8 benchmark\n'
printf '  Shared context : %s tokens (C1 full; C8 capped at 8K per request)\n' "$MAX_CONTEXT"
printf '  Decode output  : %s tokens\n' "$OUTPUT_TOKENS"
printf '  Cohorts        : %s\n' "$COHORTS"
printf '  KV cache       : %s\n' "$KV_DTYPE"
printf '  Speculation    : %s%s\n' "$SPEC" "${DRAFT_TOKENS:+ K=$DRAFT_TOKENS}"
if [[ "$PREFILL_CUBLAS" != '0' ]]; then
  printf '  Prefill route  : cuBLAS, +0.156%% perplexity (NINFER_BENCH_PREFILL_CUBLAS=0 for the default engine)\n'
else
  printf '  Prefill route  : default integer-activation\n'
fi
printf '  Results        : %s/benchmark_results/linux_3090_*\n' "$repo"
if [[ "${KV_DTYPE,,}" == 'int8' && "$MAX_CONTEXT" -gt 65536 ]]; then
  printf 'WARNING: This high-context INT8 profile is not the recommended 3090 benchmark setting.\n'
fi
printf '\nStarting in %s seconds. Press Ctrl+C to cancel.\n' "$START_DELAY_SECONDS"
sleep "$START_DELAY_SECONDS"

cd -- "$repo"
# The draft window and the prefill chunk default inside the Python script (they depend on the
# backend and on the cuBLAS route), so they are passed only when set. A conditional NAME=value
# cannot sit in the command prefix: bash recognises assignment words before expansion, so the
# expanded word would be run as a command name.
env_overrides=()
if [[ -n "$DRAFT_TOKENS" ]]; then env_overrides+=("NINFER_BENCH_DRAFT_TOKENS=$DRAFT_TOKENS"); fi
if [[ -n "$PREFILL_CHUNK" ]]; then env_overrides+=("NINFER_BENCH_PREFILL_CHUNK=$PREFILL_CHUNK"); fi
env \
  NINFER_BENCH_SERVER="$server" \
  NINFER_BENCH_MODEL="$model" \
  NINFER_BENCH_MAX_CONTEXT="$MAX_CONTEXT" \
  NINFER_BENCH_OUTPUT_TOKENS="$OUTPUT_TOKENS" \
  NINFER_BENCH_PREFILL_CHARS="$PREFILL_PROMPT_CHARACTERS" \
  NINFER_BENCH_COHORTS="$COHORTS" \
  NINFER_BENCH_KV_DTYPE="$KV_DTYPE" \
  NINFER_BENCH_SPEC="$SPEC" \
  NINFER_BENCH_PREFILL_CUBLAS="$PREFILL_CUBLAS" \
  ${env_overrides[@]+"${env_overrides[@]}"} \
  uv run tools/bench/run_qwen38_windows_3090_benchmarks.py

printf '\nBENCHMARK COMPLETE. Open the results directory printed above.\n'
