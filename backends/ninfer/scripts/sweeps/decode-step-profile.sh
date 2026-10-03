#!/usr/bin/env bash
# Where does a decode step's time actually go? Linux port of decode-step-profile.ps1, plus the
# MTP3 round, which is the question the .ps1 did not ask: a speculative round verifies K+1
# columns in one pass, and on this card that pass costs ~1.47x a plain decode step where
# vLLM's Marlin path costs ~1.14x. Profiling both on the same binary, card and cache depth shows
# which kernels carry the difference.
#
#   scripts/sweeps/decode-step-profile.sh                       # 27b-decode and 27b-decode-mtp3
#   scripts/sweeps/decode-step-profile.sh 27b-decode-mtp3 27b-prefill
#
# Configurations (all INT8 KV, --max-ctx 8192, one measured repetition after one warmup):
#   27b-decode        -n 128                                         plain decode, T=1 per round
#   27b-decode-mtp3   -n 128 --spec mtp --draft-tokens 3 --lm-head-draft   verify T=4 per round
#   27b-prefill       -pg 4096,128
#   35b-decode        -n 128   (qwen3_6_35b_a3b.ninfer)
#   35b-prefill       -pg 4096,128
#
# The bench corpus is a tiled bank with 98.4% repeated bigrams, so MTP acceptance there is ~100%
# and says nothing about text (RELEASE_NOTES_0.9.1.md). That does not matter here: a round's
# kernel work is fixed by the draft count, not by what gets accepted, and this script reports
# per-kernel time and launch counts, from which ms per round follows.
#
# **--cuda-graph-trace node is not optional.** Decode replays captured CUDA graphs and nsys's
# default records a replay as one opaque entity; without node tracing the per-kernel trace holds
# only the few kernels launched outside the graph and the window reads as ~99% idle.
#
# NINFER_MODEL_DIR   artifact directory (verbatim); otherwise models/ then scripts/models/
# NINFER_BUILD_DIR   build tree holding bench/ninfer_bench (default build-sm86, then build-linux)
# NINFER_SWEEP_OUT   output directory (default profiles/sweeps)
# NINFER_NSYS        nsys binary (default: nsys on PATH)
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

config_args() {
  case "$1" in
    27b-decode)      echo "qwen3_8_27b.ninfer -n 128" ;;
    27b-decode-mtp3) echo "qwen3_8_27b.ninfer -n 128 --spec mtp --draft-tokens 3 --lm-head-draft" ;;
    27b-prefill)     echo "qwen3_8_27b.ninfer -pg 4096,128" ;;
    35b-decode)      echo "qwen3_6_35b_a3b.ninfer -n 128" ;;
    35b-prefill)     echo "qwen3_6_35b_a3b.ninfer -pg 4096,128" ;;
    *) return 1 ;;
  esac
}

configs=("$@")
(( ${#configs[@]} == 0 )) && configs=(27b-decode 27b-decode-mtp3)

# The artifacts this run actually needs, deduplicated. The probe below matches on these rather than
# on "any .ninfer": a directory holding one unrelated artifact would otherwise be selected and every
# requested configuration reported MISSING, while the candidate that holds them sits unexamined.
required=()
for key in "${configs[@]}"; do
  spec="$(config_args "$key")" || continue
  read -r artifact _ <<<"$spec"
  for seen in ${required[@]+"${required[@]}"}; do
    [[ "$seen" == "$artifact" ]] && artifact="" && break
  done
  [[ -n "$artifact" ]] && required+=("$artifact")
done

if [[ -n "${NINFER_MODEL_DIR:-}" ]]; then
  model_dir="$NINFER_MODEL_DIR"
else
  model_dir=""
  fallback=""
  # Prefer a candidate holding every required artifact; remember the first holding any, so a
  # partial checkout still reaches the per-configuration MISSING report rather than a bare exit.
  for candidate in "$repo/models" "$repo/scripts/models"; do
    compgen -G "$candidate/*.ninfer" >/dev/null || continue
    [[ -z "$fallback" ]] && fallback="$candidate"
    complete=1
    for artifact in ${required[@]+"${required[@]}"}; do
      [[ -f "$candidate/$artifact" ]] || { complete=0; break; }
    done
    if (( complete )); then
      model_dir="$candidate"
      break
    fi
  done
  [[ -z "$model_dir" ]] && model_dir="$fallback"
  if [[ -z "$model_dir" ]]; then
    echo "No .ninfer artifact under models/ or scripts/models/; set NINFER_MODEL_DIR." >&2
    exit 1
  fi
fi

bench=""
for candidate in "${NINFER_BUILD_DIR:-}" build-sm86 build-linux build; do
  [[ -n "$candidate" && -x "$candidate/bench/ninfer_bench" ]] && { bench="$candidate/bench/ninfer_bench"; break; }
done
if [[ -z "$bench" ]]; then
  echo "ninfer_bench not found; build it (-DNINFER_BUILD_BENCHMARKS=ON) or set NINFER_BUILD_DIR." >&2
  exit 1
fi

nsys="${NINFER_NSYS:-$(command -v nsys || true)}"
if [[ -z "$nsys" || ! -x "$nsys" ]]; then
  echo "nsys not found; install Nsight Systems or set NINFER_NSYS." >&2
  exit 1
fi

out="${NINFER_SWEEP_OUT:-profiles/sweeps}"
mkdir -p "$out"

for key in "${configs[@]}"; do
  if ! spec="$(config_args "$key")"; then
    echo "$key: unknown configuration" >&2
    continue
  fi
  read -r artifact args <<<"$spec"
  weights="$model_dir/$artifact"
  if [[ ! -f "$weights" ]]; then
    echo "$key: MISSING $weights"
    continue
  fi
  stem="$out/prof_$key"
  # Clear everything nsys can write for this stem. A later `nsys stats` failure otherwise leaves the
  # previous run's CSVs in place, and this iteration reports someone else's numbers as its own.
  rm -f "$stem.nsys-rep" "$stem.sqlite" "${stem}_cuda_gpu_trace.csv" "${stem}_cuda_gpu_kern_sum.csv"

  # The report file, not nsys's exit status, decides success -- the same test the .ps1 uses.
  # shellcheck disable=SC2086  # $args is a deliberate word list
  "$nsys" profile --force-overwrite true --output "$stem" \
      --trace cuda --cuda-graph-trace node --capture-range cudaProfilerApi --capture-range-end stop \
      "$bench" --weights "$weights" --kv-dtype int8 --max-ctx 8192 \
      $args -r 1 --warmup 1 --profile-measured > "$stem.log" 2>&1 || true
  if [[ ! -f "$stem.nsys-rep" ]]; then
    echo "$key: profile FAILED"
    tail -n 5 "$stem.log" | sed 's/^/    /'
    continue
  fi
  if ! "$nsys" stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv \
      --output "$stem" "$stem.nsys-rep" > "$stem.stats.log" 2>&1; then
    echo "$key: nsys stats FAILED"
    tail -n 5 "$stem.stats.log" | sed 's/^/    /'
    continue
  fi

  python3 - "$key" "${stem}_cuda_gpu_trace.csv" "${stem}_cuda_gpu_kern_sum.csv" <<'PY'
import csv, sys

key, trace_path, kern_path = sys.argv[1:4]
intervals = []
with open(trace_path, newline="") as f:
    for row in csv.DictReader(f):
        start, dur = row.get("Start (ns)"), row.get("Duration (ns)")
        if start and dur:
            s = float(start)
            intervals.append((s, s + float(dur)))
if not intervals:
    print(f"{key}: trace CSV had no kernel rows")
    sys.exit(0)

# Busy time is the union of kernel intervals, not the sum of durations: decode launches on more
# than one stream and concurrent kernels overlap, so summing double-counts exactly when
# concurrency is working (TODO.md #60).
intervals.sort()
busy, run_start, run_end = 0.0, *intervals[0]
for s, e in intervals[1:]:
    if s <= run_end:
        run_end = max(run_end, e)
    else:
        busy += run_end - run_start
        run_start, run_end = s, e
busy += run_end - run_start
wall = max(e for _, e in intervals) - min(s for s, _ in intervals)

print()
print(f"== {key}, int8, one measured repetition ==")
print(f"  kernel launches        : {len(intervals):,}")
print(f"  GPU busy               : {busy / 1e6:,.3f} ms")
print(f"  window wall            : {wall / 1e6:,.3f} ms")
print(f"  idle between kernels   : {(wall - busy) / 1e6:,.3f} ms  ({100 * (wall - busy) / wall:.1f}% of the window)")
try:
    with open(kern_path, newline="") as f:
        rows = list(csv.DictReader(f))
except FileNotFoundError:
    rows = []
if rows:
    print("  top kernels by total time:")
    for row in rows[:12]:
        total = float(row["Total Time (ns)"]) / 1e6
        count = int(row["Instances"])
        avg = float(row["Avg (ns)"]) / 1e3
        print(f"    {total:9.2f} ms  {count:6d} x  {avg:8.1f} us  {row['Name'][:110]}")
PY
done

echo "== done =="
