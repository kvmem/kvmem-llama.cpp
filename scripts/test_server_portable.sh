#!/usr/bin/env bash
# Compile the changed server translation units and exercise standalone compatibility
# tests on Linux without requiring CUDA or downloading a model. Full server linking
# and inference are validated separately on a CUDA host.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-build/portable}
mkdir -p "$out"
includes=(-Itools -Ikvmem/include -Isrc/adapter -Ibackends/llamacpp/include -Ibackends/llamacpp/ggml/include
          -Ibackends/llamacpp/common -Ibackends/llamacpp/vendor -Ibackends/llamacpp/vendor/cpp-httplib
          -Ibackends/llamacpp/tools/server -Ibackends/llamacpp/tools/mtmd)
for test in server-options-test server-progress-test output-limit-test lane-pool-test; do
  "${CXX:-g++}" -std=c++17 -O2 -pthread "${includes[@]}" "tests/$test.cpp" -o "$out/$test"
  "$out/$test"
done
for source in llama-kvmem-server kvmem-spec; do
  "${CXX:-g++}" -std=c++17 -O1 -pthread -DKVMEM_ENABLE_NVME=0 -DLLAMA_SUBPROCESS \
    "${includes[@]}" -c "tools/$source.cpp" -o "$out/$source.o"
done
