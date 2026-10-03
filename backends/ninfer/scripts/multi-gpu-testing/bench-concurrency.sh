#!/usr/bin/env bash
# Runs ON the rented box, piped in over ssh. Nothing is edited there; this is a local file.
#
# Two questions:
#   1. How much context can we actually hold at a given concurrency, in each mode?
#   2. Under that load, does it still generate at a usable rate, or did the crossings cripple it?
#
# Aggregate tok/s under concurrent load is the number that matters -- single-stream decode is the
# worst case for an offload design, because the per-forward-pass crossing cost is paid by one
# token instead of being shared across a batch.
set -uo pipefail

MODEL=${MODEL:-/root/models/qwen3_6_35b_a3b.ninfer}
MODEL_ID=${MODEL_ID:-qwen3.6-35b-a3b}
SERVE=/root/build/apps/ninfer-serve
PORT=18080

start_server() {   # devices concurrency context [extra serve flags...]
  local devices=$1 conc=$2 ctx=$3
  shift 3
  timeout 900 "$SERVE" "$MODEL" --devices "$devices" --max-context "$ctx" --kv-capacity auto \
    --kv-dtype int8 --max-concurrency "$conc" --host 127.0.0.1 --port "$PORT" "$@" \
    > /root/s.log 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    grep -qE "listening on|startup failed|FATAL|^usage:" /root/s.log && break
    sleep 2
  done
  grep -q "listening on" /root/s.log
}

stop_server() {
  # SIGKILL, and never a bare `wait`. ninfer-serve does not exit on SIGTERM, so the `timeout 900`
  # wrapper keeps it alive for the full fifteen minutes and an unqualified `wait` blocks on that --
  # which turned a 30 minute sweep into a three hour one, with each config idling after its
  # requests had already finished.
  pkill -9 -f "$SERVE" 2>/dev/null
  local waited=0
  while pgrep -f "$SERVE" >/dev/null && [ "$waited" -lt 20 ]; do
    sleep 1
    waited=$(( waited + 1 ))
  done
  sleep 2
}

capacity_line() { grep -E "capacity \| KV" /root/s.log | head -1 | sed 's/.*capacity | //'; }
fail_line()     { grep -m1 -E "startup failed|FATAL|^error:" /root/s.log | cut -c1-110; }

probe() {          # devices concurrency context
  if start_server "$@"; then
    printf "  devices=%-4s C=%-3s ctx=%-7s -> %s\n" "$1" "$2" "$3" "$(capacity_line)"
    stop_server
    return 0
  fi
  printf "  devices=%-4s C=%-3s ctx=%-7s -> FAILED: %s\n" "$1" "$2" "$3" "$(fail_line)"
  stop_server
  return 1
}

# One request. Prints "<completion_tokens> <seconds>" so both aggregate throughput and per-request
# latency can be derived; under concurrent load the second number is what users actually feel.
fire() {
  local body start end
  start=$(date +%s.%N)
  body=$(curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions"     -H 'Content-Type: application/json'     -d "{\"model\":\"$MODEL_ID\",\"max_tokens\":$2,\"temperature\":0,
         \"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}")
  end=$(date +%s.%N)
  local toks
  toks=$(printf '%s' "$body" | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))
except Exception: print(0)
")
  python3 -c "print(f'{$toks} {$end - $start:.3f}')"
}

bench() {          # label devices concurrency context tokens_each [extra serve flags...]
  local label=$1 devices=$2 conc=$3 ctx=$4 tokens=$5
  shift 5
  if ! start_server "$devices" "$conc" "$ctx" "$@"; then
    printf "  %-22s FAILED: %s
" "$label" "$(fail_line)"
    stop_server
    return 1
  fi

  # Warm once: the first request pays graph capture and allocation, which is not what we measure.
  fire "Say OK." 4 >/dev/null

  local start end
  start=$(date +%s.%N)
  local i
  for i in $(seq 1 "$conc"); do
    fire "Write several detailed paragraphs about the number $i, its mathematical properties, and where it appears in nature."       "$tokens" > "/root/r$i.txt" &
  done
  wait
  end=$(date +%s.%N)

  local total=0 slowest=0
  for i in $(seq 1 "$conc"); do
    local t e
    t=$(awk '{print $1}' "/root/r$i.txt" 2>/dev/null || echo 0)
    e=$(awk '{print $2}' "/root/r$i.txt" 2>/dev/null || echo 0)
    total=$(( total + t ))
    slowest=$(python3 -c "print(max($slowest, $e))")
  done

  # Reported with awk to keep the quoting simple; mixing shell, python and f-strings here was a
  # syntax error waiting to happen.
  awk -v label="$label" -v conc="$conc" -v total="$total" -v start="$start" -v end="$end"       -v slowest="$slowest" 'BEGIN {
        wall = end - start; if (wall < 0.001) wall = 0.001;
        agg = total / wall;
        printf "  %-22s C=%-2s tokens=%-6s wall=%6.1fs  aggregate=%7.1f tok/s  per-request=%6.1f tok/s  slowest=%6.1fs
",
               label, conc, total, wall, agg, agg / conc, slowest;
      }'
  stop_server
}

case "${1:-all}" in
ceiling)
  echo "=== context ceiling, single GPU ==="
  for c in "1 262144" "1 131072" "2 131072" "4 65536"; do probe 0 $c; done
  # 8 is kMaximumConcurrency; higher is refused at argument parsing, not for want of memory.
  echo "=== context ceiling, pipeline stages ==="
  for c in "8 262144" "8 131072" "8 65536" "4 262144" "2 262144"; do probe 0,1 $c; done
  ;;
throughput)
  # Does concurrency amortise the crossings, as the MTP result implied it should? Single-stream
  # decode is the worst case for an offload design; the per-forward-pass crossing cost is shared
  # across a batch, so this is where it should recover.
  echo "=== aggregate throughput, no speculation ==="
  for c in 1 4 8; do bench "single GPU"  0   "$c" 16384 200; done
  for c in 1 4 8; do bench "split 0,1"   0,1 "$c" 16384 200; done
  echo "=== aggregate throughput, MTP3 ==="
  for c in 1 4 8; do bench "single + MTP" 0   "$c" 16384 200 --spec mtp --draft-tokens 3; done
  for c in 1 4 8; do bench "split + MTP"  0,1 "$c" 16384 200 --spec mtp --draft-tokens 3; done
  ;;
*)
  bash "$0" ceiling
  bash "$0" throughput
  ;;
esac
