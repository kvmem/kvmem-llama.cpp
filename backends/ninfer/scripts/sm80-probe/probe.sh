#!/usr/bin/env bash
# Is this rented GPU worth porting to? Identify it, run the five probes that decide, print a verdict.
#
# Written for a CMP 170HX (GA100, sm_80) on vast.ai, where three things are not knowable from the
# listing and each one decides whether the port is worth doing:
#
#   * is the community firmware unlock applied (VRAM, FMA/IMAD throttle, PCIe generation)?
#   * do the tensor cores (mma.sync) run at their rated speed?
#   * is every byte of the unlocked HBM actually good?
#
# It runs unattended as an instance on-start script, or piped over ssh onto a box you already have:
#
#   ssh -p PORT root@HOST 'bash -s' < scripts/sm80-probe/probe.sh
#   ssh -p PORT root@HOST 'bash -s -- --engine --tests' < scripts/sm80-probe/probe.sh
#   bash scripts/sm80-probe/probe.sh --summarize DIR       # re-read a finished run
#
# The default run is probes only and takes a couple of minutes, most of it compiling. Add --engine
# to also build the real engine for this architecture and generate text with a model (MODEL_PATH, or
# MODEL_URL to download one); that is the check that the kernels produce sensible output here. Add
# --tests to build and run the op correctness suites on the card itself.
#
# Environment:
#   NINFER_BRANCH   branch to test (default feat/sm80-cmp170hx-probe)
#   NINFER_REPO     git URL
#   NINFER_ARCH     CUDA arch number (default: from nvidia-smi, else 80)
#   PROBE_GPU       which GPU index to test (default 0)
#   MAX_RUNTIME_SECONDS  dead-man switch that powers the box off (default 3600; 0 disables)
#   MODEL_PATH / MODEL_URL   the .ninfer artifact for --engine
#   MODEL_SHA256    if set, the downloaded artifact is checked against it
set -uo pipefail
exec 2>&1

BRANCH="${NINFER_BRANCH:-feat/sm80-cmp170hx-probe}"
REPO="${NINFER_REPO:-https://github.com/ashalliants/ninfer-3090.git}"
GPU="${PROBE_GPU:-0}"
SRC="${SRC:-/root/src}"
OUT="${OUT:-/root/probe-out}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-3600}"

# ---------------------------------------------------------------------------------------------
# Summary. Reads only the files the run wrote, so a finished run can be re-read on any machine.
# ---------------------------------------------------------------------------------------------

# First number that precedes the given unit on the first line matching the pattern.
rate() { awk -v pat="$2" -v unit="$3" '$0 ~ pat { for (i = 2; i <= NF; i++) if ($i == unit) { print $(i-1); exit } }' "$1" 2>/dev/null; }
verdict() { grep -h "^VERDICT $2" "$1" 2>/dev/null | head -1; }

summarize() {
  local d="$1"
  echo
  echo "=================================================================="
  echo " SUMMARY  ($(date -u +%FT%TZ))"
  echo "=================================================================="

  local name mem_mib cc
  name="$(sed -n '2p' "$d/gpu.csv" | cut -d, -f2 | sed 's/^ *//')"
  mem_mib="$(sed -n '2p' "$d/gpu.csv" | cut -d, -f3 | tr -dc '0-9')"
  cc="$(sed -n '2p' "$d/gpu.csv" | cut -d, -f4 | tr -d ' ')"
  echo " card:      $name   (compute capability ${cc:-?})"

  # Capacity. The 8 GB stock card unlocks to 64 GB; the 10 GB one to 40 GB.
  if [[ -n "${mem_mib:-}" ]]; then
    if [[ "$name" != *CMP* ]]; then echo " capacity:  $(( mem_mib / 1024 )) GiB"
    elif (( mem_mib >= 30000 )); then echo " capacity:  $(( mem_mib / 1024 )) GiB   -> unlock applied"
    else echo " capacity:  $(( mem_mib / 1024 )) GiB   -> NOT unlocked (stock capacity)"; fi
  fi
  local gen_cur gen_max w_cur w_max idle=""
  IFS=, read -r gen_cur gen_max w_cur w_max < <(sed -n '2p' "$d/gpu.csv" | cut -d, -f6-9 | tr -d ' ')
  # A link idle-clocks below its maximum until it carries traffic, and this is read before any.
  [[ "${gen_cur:-0}" =~ ^[0-9]+$ && "${gen_max:-0}" =~ ^[0-9]+$ ]] && (( gen_cur < gen_max )) \
    && idle="  [idle-clocked reading; the copy rate below is authoritative]"
  echo " pcie:      Gen${gen_cur:-?} x${w_cur:-?} now (card maximum Gen${gen_max:-?} x${w_max:-?})$idle"
  local h2d
  h2d="$(rate "$d/pcie.txt" 'host->device, pinned' 'GB/s')"
  [[ -n "$h2d" ]] && echo " transfer:  ${h2d} GB/s pinned host->device   ($(awk '/artifact upload/ {print $(NF-1), $NF; exit}' "$d/pcie.txt" 2>/dev/null) for a 16 GiB artifact)"

  # FMA / IMAD throttle.
  echo " fma:       $(verdict "$d/fma.txt" fma | sed 's/^VERDICT fma: //')"
  echo " imad:      $(verdict "$d/fma.txt" imad | sed 's/^VERDICT imad: //')"

  # Tensor cores against what an unthrottled GA100 does for this SM count and clock. Dense BF16 is
  # 2048 flops/SM/clk and INT8 is 4096 on sm_80; the probe prints both the rates and its SM count
  # and clock. sm_86/89 parts are half rate on f32 accumulate, so the comparison only means
  # something on 8.0.
  local sms mhz bf16 s8
  read -r sms mhz < <(sed -n 's/^GPU: .* \([0-9][0-9]*\) SMs  *\([0-9][0-9]*\) MHz.*/\1 \2/p' "$d/tensor.txt" | head -1)
  bf16="$(rate "$d/tensor.txt" 'BF16 m16n8k16' 'TFLOPS')"
  s8="$(rate "$d/tensor.txt" 'INT8 m16n8k32' 'TOPS')"
  if [[ -n "${bf16:-}" && -n "${sms:-}" ]]; then
    awk -v bf="$bf16" -v s8="${s8:-0}" -v sms="$sms" -v mhz="$mhz" -v cc="${cc:-}" 'BEGIN {
      ebf = sms * mhz * 2048 / 1e6; es8 = sms * mhz * 4096 / 1e6
      printf " tensor:    BF16 %.1f TFLOPS, INT8 %.1f TOPS", bf, s8
      if (cc != "8.0") { printf "   (no unthrottled reference for cc %s)\n", cc; exit }
      printf "   vs unthrottled %.0f / %.0f  (%.0f%% / %.0f%%)\n", ebf, es8, 100 * bf / ebf, 100 * s8 / es8
      r = bf / ebf
      if (r >= 0.5)      v = "WORKING     tensor cores run near rated speed"
      else if (r >= 0.1) v = "PARTIAL     tensor cores run but well below rated speed"
      else               v = "CRIPPLED    at or below the non-tensor FP32 rate; prefill would be slow"
      printf " tensor:    %s\n", v
    }'
  else
    echo " tensor:    (no result)"
  fi

  local hbm best
  best="$(grep -h '^Best sustained bus rate' "$d/hbm.txt" 2>/dev/null)"
  [[ -n "$best" ]] && echo " hbm:       $best"
  echo " vram:      $(verdict "$d/vram.txt" vram | sed 's/^VERDICT vram: //')"

  echo
  echo " Reference, RTX 3090 on the fork host (2026-09-21): memory read 894 GB/s, BF16 83 TFLOPS,"
  echo " INT8 337 TOPS, FFMA 35-40 TFLOPS (boost-clock dependent), pinned host->device 25 GB/s."
}

if [[ "${1:-}" == "--summarize" ]]; then
  [[ -n "${2:-}" && -d "$2" ]] || { echo "usage: $0 --summarize DIR" >&2; exit 2; }
  summarize "$2"
  exit 0
fi

ENGINE=0
TESTS=0
for arg in "$@"; do
  case "$arg" in
    --engine) ENGINE=1 ;;
    --tests) TESTS=1 ;;
    *) echo "unknown argument: $arg (expected --engine, --tests or --summarize DIR)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------------------------

echo "=== sm80 probe $(date -u +%FT%TZ)  branch=$BRANCH engine=$ENGINE tests=$TESTS ==="

if [[ "$MAX_RUNTIME_SECONDS" -gt 0 ]]; then
  ( sleep "$MAX_RUNTIME_SECONDS"; echo "=== MAX_RUNTIME reached, halting ==="; poweroff || halt -f ) \
    >/dev/null 2>&1 &
  echo "dead-man switch armed: ${MAX_RUNTIME_SECONDS}s"
fi

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "--- device ---"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,compute_cap,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.sm,clocks.max.mem,power.limit --format=csv \
  > "$OUT/gpu.csv" 2>/dev/null
if [[ ! -s "$OUT/gpu.csv" ]]; then
  # Older drivers reject the compute_cap field; ask for everything else and assume 8.0.
  nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.sm,clocks.max.mem,power.limit --format=csv \
    | awk -F, -v OFS=, 'NR == 1 { $3 = $3 ",compute_cap" } NR > 1 { $3 = $3 ",8.0" } { print }' \
    > "$OUT/gpu.csv"
fi
cat "$OUT/gpu.csv"
nvidia-smi topo -m 2>&1 | head -6
echo "host: $(nproc) cores, $(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo) GiB RAM, $(df -h /root | awk 'NR==2 {print $4}') free in /root"
echo "os: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"

CC="$(sed -n '2p' "$OUT/gpu.csv" | cut -d, -f4 | tr -d ' ')"
ARCH="${NINFER_ARCH:-${CC//./}}"
ARCH="${ARCH:-80}"
echo "arch: sm_$ARCH"
case "$(sed -n '2p' "$OUT/gpu.csv" | cut -d, -f2)" in
  *CMP*) ;;
  *) echo "NOTE: this is not a CMP card; the summary's thresholds are written for one." ;;
esac

echo "--- toolchain ---"
NVCC="$(command -v nvcc || true)"
[[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]] && NVCC=/usr/local/cuda/bin/nvcc
if [[ -z "$NVCC" ]]; then
  echo "NO_NVCC: this image has no CUDA compiler. Rent with a CUDA *devel* image (nvidia/cuda:12.8.1-devel-ubuntu24.04)."
  exit 1
fi
"$NVCC" --version | tail -2
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }

echo "--- source: $BRANCH ---"
if [[ -d "$SRC/.git" ]]; then
  # A failed refresh must stop the run: probing and testing stale code would report on the wrong tree.
  git -C "$SRC" fetch --depth 1 origin "$BRANCH" && git -C "$SRC" reset --hard FETCH_HEAD \
    || { echo "SOURCE_REFRESH_FAILED (fetch or reset of $BRANCH)"; exit 1; }
else
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$SRC" || { echo "CLONE_FAILED (is the branch pushed?)"; exit 1; }
fi
echo "at $(git -C "$SRC" rev-parse --short HEAD)"

echo "--- building probes for sm_$ARCH ---"
mkdir -p "$OUT/bin"
for p in fma_rate_probe tensor_core_rate_probe hbm_bandwidth_probe pcie_bandwidth_probe vram_pattern_probe; do
  rm -f "$OUT/bin/$p" # a stale binary from an earlier run must not pass for this build
  "$NVCC" -O3 -std=c++17 -arch="sm_$ARCH" "$SRC/tools/$p.cu" -o "$OUT/bin/$p" 2>&1 | tail -5 \
    || true
  [[ -x "$OUT/bin/$p" ]] || { echo "BUILD_FAILED: $p"; exit 1; }
done

# Order matters a little: a long compute probe first heats the card, so the bandwidth numbers that
# follow are honest for a sustained load rather than an idle P-state.
# One failure policy for every stage: keep going so all logs are produced, remember what failed, and
# exit non-zero at the end so automation cannot accept incomplete or invalid results.
FAILED_STAGES=()
run() {
  local label="$1" file="$2"; shift 2
  echo "--- $label ---"
  local rc=0
  timeout 600 "$@" > "$OUT/$file" 2>&1 || rc=$?
  cat "$OUT/$file"
  if (( rc != 0 )); then
    echo "($label exited $rc)"
    FAILED_STAGES+=("$label (exit $rc)")
  fi
}
run "FP32 / INT32 / packed-half rates" fma.txt    "$OUT/bin/fma_rate_probe"
run "tensor core rates"                tensor.txt "$OUT/bin/tensor_core_rate_probe"
run "device memory bandwidth"          hbm.txt    "$OUT/bin/hbm_bandwidth_probe"
run "host<->device transfer"           pcie.txt   "$OUT/bin/pcie_bandwidth_probe" --mib 256
run "VRAM pattern test"                vram.txt   "$OUT/bin/vram_pattern_probe"

summarize "$OUT" | tee "$OUT/summary.txt"

# ---------------------------------------------------------------------------------------------
# Optional: does the real engine run correctly on this architecture?
# ---------------------------------------------------------------------------------------------
if (( ENGINE || TESTS )); then
  echo
  echo "=== BUILD for sm_$ARCH ==="
  # libcurl and Python are configure-time requirements of the server target and the tests.
  apt-get update -qq
  apt-get install -y -qq cmake ninja-build build-essential pkg-config aria2 python3 \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev \
    libswresample-dev libcurl4-openssl-dev >/dev/null 2>&1

  # CMakeLists needs 3.28; Ubuntu 22.04 ships 3.22 and 24.04 ships 3.28.
  cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
  if [[ "$(printf '3.28\n%s\n' "$cmake_version" | sort -V | head -1)" != "3.28" ]]; then
    echo "CMAKE_TOO_OLD: have $cmake_version, need 3.28. Use an ubuntu24.04 image."; exit 1
  fi

  testing=OFF; (( TESTS )) && testing=ON
  cmake -S "$SRC" -B /root/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING="$testing" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER="$NVCC" 2>&1 | tail -4
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "CONFIGURE_FAILED"; exit 1; }
  # Each nvcc job wants on the order of a gigabyte, so cap parallelism by RAM as well as cores.
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  jobs=$(( ram_gb / 2 )); (( jobs < 4 )) && jobs=4; (( jobs > $(nproc) )) && jobs=$(nproc)
  echo "building with -j$jobs ($ram_gb GiB RAM)"
fi

if (( ENGINE )); then
  echo
  echo "=== ENGINE: build and generate ==="
  MODEL="${MODEL_PATH:-}"
  if [[ -z "$MODEL" && -n "${MODEL_URL:-}" ]]; then
    mkdir -p /root/models
    MODEL="/root/models/$(basename "${MODEL_URL%%\?*}")"
    # aria2c over 16 ranges, not curl: a single stream stalled outright on this provider before.
    aria2c -x16 -s16 -c --file-allocation=none -d /root/models -o "$(basename "$MODEL")" "$MODEL_URL" \
      > /root/aria.log 2>&1 || { echo "MODEL_DOWNLOAD_FAILED"; exit 1; }
  fi
  [[ -n "$MODEL" && -f "$MODEL" ]] || { echo "NO_MODEL: set MODEL_PATH or MODEL_URL"; exit 1; }
  ls -l "$MODEL"
  if [[ -n "${MODEL_SHA256:-}" ]]; then
    echo "$MODEL_SHA256  $MODEL" | sha256sum -c - || { echo "MODEL_CHECKSUM_MISMATCH"; exit 1; }
  fi

  cmake --build /root/build --target ninfer -j "$jobs" 2>&1 | tail -8
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "ENGINE_BUILD_FAILED"; exit 1; }

  echo "--- greedy generation, single GPU ---"
  /root/build/apps/ninfer "$MODEL" \
    --prompt "List the first eight prime numbers, then explain briefly why 1 is not prime." \
    --max-new 200 --greedy --kv-dtype int8 --no-thinking --max-context 8192 --kv-capacity 8192 \
    --device 0 > "$OUT/engine-output.txt" 2> "$OUT/engine-stderr.txt"
  engine_rc=$?
  echo "ENGINE_EXIT=$engine_rc"
  (( engine_rc == 0 )) || FAILED_STAGES+=("engine generation (exit $engine_rc)")
  grep -E "gpu weights used|free after weights|free after startup|decode speed|error" "$OUT/engine-stderr.txt" | head
  echo "--- output (read it: sensible text means the kernels are right here) ---"
  cat "$OUT/engine-output.txt"
fi

# The op correctness suites that exercise the mma.sync / cp.async kernels, run on the real card. The
# The list selects CTest entries by name; they all run from the ninfer_tests bundle built below.
if (( TESTS )); then
  echo
  echo "=== TESTS: op correctness on this card ==="
  TEST_TARGETS="ninfer_linear_q4_a16_test ninfer_linear_q8_a16_test ninfer_linear_bf16_a16_test \
ninfer_linear_topk_test ninfer_gated_delta_net_test ninfer_softmax_attention_test \
ninfer_sparse_moe_test ninfer_attn_input_proj_test ninfer_gdn_input_proj_test \
ninfer_rmsnorm_test ninfer_kv_cache_test ninfer_gdn_gating_proj_test"
  # The listed names are CTest entries. Each is an OBJECT library linked into the single
  # ninfer_tests executable that CTest actually runs (`ninfer_tests <name>`), so build that
  # bundle; building only the object targets leaves no binary to run. It compiles every test.
  cmake --build /root/build --target ninfer_tests -j "$jobs" 2>&1 | tail -6
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "TESTS_BUILD_FAILED"; exit 1; }
  pattern="^($(echo $TEST_TARGETS | tr ' ' '|'))\$"
  ctest --test-dir /root/build -j1 --output-on-failure -R "$pattern" 2>&1 | tee "$OUT/tests.txt" | tail -30
  tests_rc="${PIPESTATUS[0]}"
  (( tests_rc == 0 )) || FAILED_STAGES+=("ctest (exit $tests_rc; see $OUT/tests.txt)")
fi

if (( ${#FAILED_STAGES[@]} )); then
  echo
  echo "=== FAILED: ${#FAILED_STAGES[@]} stage(s); results in $OUT are incomplete or invalid ==="
  printf '  - %s\n' "${FAILED_STAGES[@]}"
  exit 1
fi

echo
echo "=== DONE. Everything is in $OUT (summary.txt, per-probe .txt files) ==="
