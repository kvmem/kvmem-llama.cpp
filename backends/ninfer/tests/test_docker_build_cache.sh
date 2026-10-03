#!/usr/bin/env bash
# Exercise the real Dockerfile with a tiny CMake project; never run NInfer.
set -euo pipefail

builder=${1:-docker}
export BUILDKIT_PROGRESS=plain
repo=$(cd "$(dirname "$0")/.." && pwd)
test_dir=$(mktemp -d "$repo/build-cache-test.XXXXXX")
context="$test_dir/context"
mkdir "$context"
cache_id="ninfer-$(basename "$test_dir")"
cache_id=${cache_id,,}
tag="localhost/$cache_id:check"
printf 'Build-cache checks: %s\n' "$test_dir"

# Isolate both caches from production builds and other test runs. Keep the build
# instruction otherwise intact: the assertions below exercise its actual behavior.
sed -e "s/id=ninfer-build,/id=$cache_id,/" \
    -e "s/type=cache,target=\/ccache/type=cache,id=$cache_id-ccache,target=\/ccache/" \
    "$repo/Dockerfile" > "$test_dir/Dockerfile"
cp "$test_dir/Dockerfile" "$context/Dockerfile"

cat > "$context/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.28)
project(build_cache_check LANGUAGES CXX)
include(defaults.cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps")
add_executable(ninfer main.cpp)
target_compile_definitions(ninfer PRIVATE CACHED_DEFAULT=${CACHED_DEFAULT})
add_executable(ninfer-serve other.cpp)
EOF
printf 'set(CACHED_DEFAULT 0 CACHE STRING "Fixture default")\n' > "$context/defaults.cmake"
cat > "$context/main.cpp" <<'EOF'
#include "expected.h"
#ifndef RECIPE_FLAG
#define RECIPE_FLAG 0
#endif
static_assert(RECIPE_FLAG == EXPECTED_FLAG, "stale compiler flags");
static_assert(CACHED_DEFAULT == EXPECTED_DEFAULT, "stale CMake default");
int main() { return 0; }
EOF
printf 'int main() { return 0; }\n' > "$context/other.cpp"

expect() {
    printf '#define EXPECTED_FLAG %s\n#define EXPECTED_DEFAULT %s\n' "$1" "$2" \
        > "$context/expected.h"
}

build() {
    local label=$1
    # A changed context forces RUN to execute rather than reuse an image layer.
    printf '%s\n' "$label" > "$context/build-check.txt"
    if ! "$builder" build --target build --tag "$tag" "$context" \
        > "$test_dir/$label.log" 2>&1; then
        cat "$test_dir/$label.log"
        return 1
    fi
    printf 'PASS %s\n' "$label"
}

expect 0 0
build cold

sed 's/-DCMAKE_BUILD_TYPE=Release/-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DRECIPE_FLAG=1/' \
    "$test_dir/Dockerfile" > "$context/Dockerfile"
expect 1 0
build flag-added

cp "$test_dir/Dockerfile" "$context/Dockerfile"
expect 0 0
# Exercise restored headers with old timestamps as well as flag removal.
touch -t 200001010000 "$context/expected.h"
build flag-removed

build non-code-edit
grep -q 'ninja: no work to do' "$test_dir/non-code-edit.log"

printf '\nstatic_assert(true, "ordinary source edit");\n' >> "$context/main.cpp"
build source-edit
# Exactly one translation unit should rebuild; the other executable is unchanged.
test "$(grep -c 'Building CXX object' "$test_dir/source-edit.log")" -eq 1

printf 'set(CACHED_DEFAULT 1 CACHE STRING "Fixture default")\n' > "$context/defaults.cmake"
expect 0 1
build default-changed

printf 'set(CACHED_DEFAULT 0 CACHE STRING "Fixture default")\n' > "$context/defaults.cmake"
expect 0 0
build default-restored

sed '/^WORKDIR \/src/i ENV CXXFLAGS=-DRECIPE_FLAG=1' \
    "$test_dir/Dockerfile" > "$context/Dockerfile"
expect 1 0
build environment-added

cp "$test_dir/Dockerfile" "$context/Dockerfile"
expect 0 0
build environment-removed

printf 'All checks passed. Logs and fixture: %s\n' "$test_dir"
printf 'Test image: %s (cache mounts use prefix %s)\n' "$tag" "$cache_id"
