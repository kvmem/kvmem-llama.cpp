#!/usr/bin/env bash
# Does the pipeline split actually run, and does it produce the right tokens?
#
# The test that matters is equivalence: greedy decoding on one GPU and greedy decoding with the
# the layers split into one pipeline stage per GPU must produce identical text. Anything else -- it
# starts, it is fast, memory looks right -- can be true of a build that is silently reading the
# wrong device's memory.
#
# It also records what each card holds, because the point of the split is the weight and KV room it
# frees on each card, and compares the KV capacity the engine resolves in each mode.
set -uo pipefail
exec 2>&1
echo "=== pipeline execution test $(date -u +%FT%TZ) ==="

BRANCH="${NINFER_BRANCH:-feat/dual-gpu-graph-mode}"
REPO="${NINFER_REPO:-https://github.com/ashalliants/ninfer-3090.git}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6/qwen3_6_35b_a3b.ninfer}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}"
PROMPT="${PROMPT:-List the first eight prime numbers, then explain briefly why 1 is not prime.}"
MAX_NEW="${MAX_NEW:-120}"

if [[ "$MAX_RUNTIME_SECONDS" -gt 0 ]]; then
  ( sleep "$MAX_RUNTIME_SECONDS"; echo "=== MAX_RUNTIME reached, halting ==="; poweroff || halt -f ) \
    >/dev/null 2>&1 &
  echo "dead-man switch armed: ${MAX_RUNTIME_SECONDS}s"
fi

nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
nvidia-smi topo -m 2>&1 | head -6

echo "--- installing build dependencies ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake ninja-build build-essential pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev \
  libswresample-dev >/dev/null 2>&1

echo "--- cloning ${BRANCH} ---"
git clone --depth 1 --branch "$BRANCH" "$REPO" /root/src || { echo "CLONE_FAILED"; exit 1; }
echo "at $(git -C /root/src rev-parse --short HEAD)"

echo "--- starting model download in background ---"
# aria2c over 16 connections, not curl. A single-stream curl throttled to ~1 MB/s and then stalled
# outright at 11 GB of 21 on one run, costing most of an hour; the same file over 16 parallel
# ranges sustains ~98 MB/s, and -c resumes instead of restarting. HuggingFace serves ranges fine.
mkdir -p /root/models
apt-get install -y -qq aria2 >/dev/null 2>&1
( aria2c -x16 -s16 -c --file-allocation=none -d /root/models \
      -o qwen3_6_35b_a3b.ninfer "$MODEL_URL" > /root/aria.log 2>&1 \
    && echo OK > /root/model.done || echo FAILED > /root/model.done ) &

echo "--- configuring ---"
cmake -S /root/src -B /root/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${NINFER_ARCH:-86}" \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)" \
  2>&1 | tail -3 || { echo "CONFIGURE_FAILED"; exit 1; }

# Cap parallelism by RAM: these hosts pair high core counts with variable memory and each nvcc job
# wants on the order of a gigabyte.
ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
jobs=$(( ram_gb / 2 )); [[ "$jobs" -lt 4 ]] && jobs=4
[[ "$jobs" -gt $(nproc) ]] && jobs=$(nproc)
echo "--- building ninfer + ninfer-serve with -j${jobs} (${ram_gb} GB RAM, $(nproc) cores) ---"
if ! cmake --build /root/build --target ninfer ninfer-serve -j "$jobs" 2>&1 | tail -12; then
  echo "BUILD_FAILED"; exit 1
fi

echo "--- waiting for the model ---"
while [[ ! -f /root/model.done ]]; do sleep 10; done
grep -q OK /root/model.done || { echo "MODEL_MISSING"; exit 1; }
ls -la /root/models/

MODEL=/root/models/qwen3_6_35b_a3b.ninfer
COMMON=(--prompt "$PROMPT" --max-new "$MAX_NEW" --greedy --kv-dtype int8 --no-thinking
        --max-context 8192 --kv-capacity 8192)

echo "=== A: SINGLE GPU (reference) ==="
/root/build/apps/ninfer "$MODEL" "${COMMON[@]}" --device 0 >/root/single.txt 2>/root/single.err
echo "SINGLE_EXIT=$?"
grep -E "gpu weights used|free after weights|free after startup|decode speed" /root/single.err || true
echo "--- output ---"; cat /root/single.txt

echo "=== B: EXPERT OFFLOAD ACROSS TWO GPUs ==="
/root/build/apps/ninfer "$MODEL" "${COMMON[@]}" --devices 0,1 >/root/split.txt 2>/root/split.err
echo "SPLIT_EXIT=$?"
grep -E "gpu weights used|free after weights|free after startup|decode speed" /root/split.err || true
echo "--- output ---"; cat /root/split.txt
echo "--- last stderr on failure ---"; tail -20 /root/split.err

echo "=== EQUIVALENCE ==="
if diff -q /root/single.txt /root/split.txt >/dev/null 2>&1; then
  echo "OUTPUTS_IDENTICAL=yes"
else
  echo "OUTPUTS_IDENTICAL=no"
  echo "--- diff ---"; diff /root/single.txt /root/split.txt | head -30
fi

# The headline number: how much KV the split buys on the card that serves attention.
echo "=== KV CAPACITY, AUTOMATIC, EACH MODE ==="
for mode in "--device 0" "--devices 0,1"; do
  echo "--- $mode ---"
  timeout 900 /root/build/apps/ninfer "$MODEL" --prompt hi --max-new 1 --greedy \
    --kv-dtype int8 --max-context 262144 --kv-capacity auto $mode 2>&1 \
    | grep -E "KV capacity|free after weights|gpu weights used|error" | head -6
done

echo "=== PIPELINE TEST END ==="
sleep 60
poweroff || halt -f || exit 0
