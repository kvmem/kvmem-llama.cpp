#!/usr/bin/env bash
# Pull, rebuild and re-run on an already-rented box, without paying for the model download again.
#
# The 21 GB model is the long pole -- roughly 25 minutes against a 10 minute clean build and a few
# seconds for an incremental one. Destroying the instance to test a one-line fix throws that away
# every time, so this keeps the box and iterates on it over SSH.
#
#   bash iterate.sh                 # pull HEAD, rebuild, run the equivalence check
#   bash iterate.sh --build-only    # pull and rebuild, skip the run
#   bash iterate.sh --run-only      # run against whatever is already built
#   bash iterate.sh --capacity      # how much KV each mode can actually resolve
#   bash iterate.sh --concurrency   # several large sessions at once, which is the actual goal
#
# Reads the instance id from .vast_instance_id.
set -uo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
vastai="$root/.venv/Scripts/vastai.exe"
[[ -x "$vastai" ]] || vastai="$root/.venv/bin/vastai"

[[ -f .vast_instance_id ]] || { echo "no .vast_instance_id" >&2; exit 1; }
instance="$(cat .vast_instance_id)"

url="$("$vastai" ssh-url "$instance" 2>/dev/null | sed 's|ssh://root@||')"
[[ -n "$url" ]] || { echo "could not resolve ssh url for $instance" >&2; exit 1; }
port="${url##*:}"
addr="${url%%:*}"

run() {
  ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=25 -o BatchMode=yes -p "$port" "root@$addr" "$@"
}

BRANCH="${NINFER_BRANCH:-feat/dual-gpu-graph-mode}"
MODE="${1:-}"

if [[ "$MODE" != "--run-only" ]]; then
  echo "==> pulling $BRANCH and rebuilding"
  run "set -e -o pipefail
    cd /root/src
    git fetch --depth 1 origin '$BRANCH'
    git reset --hard FETCH_HEAD
    echo \"at \$(git rev-parse --short HEAD)\"
    # Incremental: only the touched translation units recompile, which is seconds rather than the
    # ten minutes a clean build costs.
    ram_gb=\$(awk '/MemTotal/ {printf \"%d\", \$2/1024/1024}' /proc/meminfo)
    jobs=\$(( ram_gb / 2 )); [ \$jobs -lt 4 ] && jobs=4
    [ \$jobs -gt \$(nproc) ] && jobs=\$(nproc)
    cmake --build /root/build --target ninfer -j \$jobs 2>&1 | tail -5"
  rc=$?
  [[ $rc -eq 0 ]] || { echo "build failed (rc=$rc)" >&2; exit 1; }
fi

[[ "$MODE" == "--build-only" ]] && exit 0

if [[ "$MODE" == "--capacity" ]]; then
  # The headline number. Ask each mode to resolve the largest KV it can and report what it got;
  # rank 0 serves attention, so its free memory is exactly what is available for KV.
  echo "==> KV capacity with --kv-capacity auto"
  run 'MODEL=/root/models/qwen3_6_35b_a3b.ninfer
    for mode in "--device 0" "--devices 0,1"; do
      echo "--- $mode ---"
      timeout 1200 /root/build/apps/ninfer "$MODEL" --prompt hi --max-new 1 --greedy         --kv-dtype int8 --max-context 262144 --kv-capacity auto $mode 2>&1         | grep -E "KV capacity|gpu weights used|free after weights|free after startup|error" | head -8
    done'
  exit $?
fi

if [[ "$MODE" == "--concurrency" ]]; then
  # Several large sessions at once is what the split is for, so check the engine actually resolves
  # that shape rather than inferring it from free bytes.
  conc="${CONCURRENCY:-4}"
  ctx="${CTX:-131072}"
  echo "==> concurrency $conc at context $ctx"
  run "MODEL=/root/models/qwen3_6_35b_a3b.ninfer
    for mode in '--device 0' '--devices 0,1'; do
      echo \"--- \$mode ---\"
      timeout 900 /root/build/apps/ninfer-serve \"\$MODEL\" \$mode --max-context $ctx         --kv-capacity auto --kv-dtype int8 --max-concurrency $conc --host 127.0.0.1 --port 18080         > /root/serve.log 2>&1 &
      pid=\$!
      for i in \$(seq 1 120); do
        grep -qE 'engine ready|error' /root/serve.log && break
        sleep 5
      done
      grep -E 'engine ready|capacity|error' /root/serve.log | head -5
      kill \$pid 2>/dev/null
      wait 2>/dev/null
    done"
  exit $?
fi

echo "==> waiting for the model"
run 'while [ ! -f /root/model.done ]; do sleep 10; done; cat /root/model.done; ls -la /root/models/'

PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog. }"
REPEATS="${REPEATS:-300}"
MAX_NEW="${MAX_NEW:-60}"

echo "==> equivalence: single GPU against pipeline stages"
run "set -u
  MODEL=/root/models/qwen3_6_35b_a3b.ninfer
  PROMPT=\$(python3 -c \"print('$PROMPT' * $REPEATS)\")
  COMMON=\"--max-new $MAX_NEW --greedy --kv-dtype int8 --no-thinking --max-context 8192 --kv-capacity 8192\"

  echo '--- single GPU ---'
  /root/build/apps/ninfer \"\$MODEL\" --prompt \"\$PROMPT\" \$COMMON --device 0 \
    >/root/a.txt 2>/root/a.err; echo \"exit=\$?\"
  grep -E 'prefill speed|decode speed|gpu weights used|free after weights' /root/a.err || tail -3 /root/a.err

  echo '--- pipeline stages (0,1) ---'
  /root/build/apps/ninfer \"\$MODEL\" --prompt \"\$PROMPT\" \$COMMON --devices 0,1 \
    >/root/b.txt 2>/root/b.err; echo \"exit=\$?\"
  grep -E 'prefill speed|decode speed|gpu weights used|free after weights' /root/b.err || tail -3 /root/b.err

  echo '--- equivalence ---'
  if diff -q /root/a.txt /root/b.txt >/dev/null 2>&1; then
    echo 'OUTPUTS_IDENTICAL=yes'
    head -c 200 /root/a.txt
  else
    echo 'OUTPUTS_IDENTICAL=no'
    diff /root/a.txt /root/b.txt | head -20
  fi"
