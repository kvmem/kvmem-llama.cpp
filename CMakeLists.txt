cmake_minimum_required(VERSION 3.16)
project(kvmem_llamacpp LANGUAGES CXX)

# Guard against a poisoned build directory. If the very first configure cannot
# identify the compiler (clang++ not yet on PATH, for example), CMake never
# loads Platform/Windows-Clang.cmake and caches CMAKE_CXX_FLAGS_RELEASE as an
# empty string. Every later configure in that directory then inherits it, the
# build still succeeds, and inference runs roughly 400x slower because nothing
# -- including the HIP device code -- is compiled with -O3. Fail loudly instead.
if (CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_FLAGS_RELEASE STREQUAL ""
        AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR
        "CMAKE_BUILD_TYPE=Release but CMAKE_CXX_FLAGS_RELEASE is empty, so this "
        "build would be compiled without optimisation. This happens when a build "
        "directory was first configured before the C++ compiler could be "
        "identified. Delete the build directory (e.g. build-hip) and configure "
        "again with clang/clang++ on PATH.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()
if (WIN32 AND NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()
add_subdirectory(kvmem)

option(KVMEM_BUILD_LLAMA "Build llama.cpp with the KVMem adapter" ON)

# GPU backend selection.
#   (default)      -> CUDA, the upstream KVMem target.
#   -DGGML_HIP=ON  -> ROCm/HIP on AMD.
# llama.cpp's HIP backend does not hipify: it compiles the very same
# ggml/src/ggml-cuda sources with AMD clang, so every KVMem hook that lives in
# ggml-cuda/llama.cpp carries over unchanged. Only the adapter's own runtime
# shim (src/adapter/llama-kvmem-gpu.h) and the stage-in kernel need a backend
# switch, which is what the KVMEM_GPU_BACKEND block below wires up.
if (KVMEM_BUILD_LLAMA AND GGML_HIP)
    set(KVMEM_GPU_BACKEND "HIP")
else()
    set(KVMEM_GPU_BACKEND "CUDA")
endif()

if (KVMEM_BUILD_LLAMA)
    if (KVMEM_GPU_BACKEND STREQUAL "CUDA")
        if (NOT CMAKE_CUDA_COMPILER AND EXISTS "/home/leye/kvmem_qw3/.cu13-env/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "/home/leye/kvmem_qw3/.cu13-env/bin/nvcc" CACHE FILEPATH "nvcc")
        endif()
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "120a-real" CACHE STRING "Blackwell sm_120")
        endif()

        option(GGML_CUDA "ggml: use CUDA" ON)
    else()
        # ROCm: the CUDA backend must stay off and no CUDA arch may be forced.
        set(GGML_CUDA OFF CACHE BOOL "ggml: use CUDA" FORCE)
        if (NOT GPU_TARGETS AND AMDGPU_TARGETS)
            set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "AMD GPU targets for the HIP build")
        endif()
        if (NOT GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            # Same default the rest of this machine's builds use (RX 6900 XT).
            set(GPU_TARGETS "gfx1030" CACHE STRING "AMD GPU targets for the HIP build")
        endif()
        if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")
        endif()

        # ggml-hip calls find_package(hip) in its own directory scope, so
        # hip::host / hip::device are not visible up here. Import them again for
        # the root scope (imported targets are not global by default).
        if (NOT ROCM_PATH)
            if (DEFINED ENV{ROCM_PATH})
                set(ROCM_PATH "$ENV{ROCM_PATH}")
            elseif (DEFINED ENV{HIP_PATH})
                set(ROCM_PATH "$ENV{HIP_PATH}")
            else()
                message(FATAL_ERROR
                    "GGML_HIP=ON but ROCM_PATH/HIP_PATH is not set. Point one of "
                    "them at your ROCm install, or pass -DROCM_PATH=<dir>.")
            endif()
        endif()
        file(TO_CMAKE_PATH "${ROCM_PATH}" ROCM_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
        find_package(hip REQUIRED)

        message(STATUS "KVMem: ROCm/HIP backend, GPU_TARGETS=${GPU_TARGETS}, ROCM_PATH=${ROCM_PATH}")
    endif()
    # Default llama.cpp FA vec kernels are only F16/Q4_0/Q8_0/BF16. q5_0 KV
    # (and q4_1/q5_1) need this or FlashAttention refuses those types and
    # Qwen3.8 hybrid prefill collapses (~75 t/s vs ~500).
    set(GGML_CUDA_FA_ALL_QUANTS ON CACHE BOOL
        "compile all FlashAttention KV quants (needed for --kv-dtype q5_0)" FORCE)
    set(LLAMA_KVMEM ON CACHE BOOL "" FORCE)
    set(LLAMA_KVMEM_ROOT "${CMAKE_SOURCE_DIR}" CACHE PATH "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "")
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "")

    add_subdirectory(llama.cpp)

    if (KVMEM_GPU_BACKEND STREQUAL "HIP")
        # The stage-in kernels must be built by AMD clang with `-x hip`, exactly
        # how ggml-hip builds ggml-cuda/*.cu on Windows (CMake does not support
        # the HIP language there). They live in their own object library so
        # `-x hip` is confined to this single file: linking hip::host straight
        # into `llama` would turn every libllama translation unit into a HIP
        # device compilation unit.
        add_library(kvmem-stagein-hip OBJECT src/adapter/llama-kvmem-stagein.cu)
        set_source_files_properties(src/adapter/llama-kvmem-stagein.cu
            PROPERTIES LANGUAGE CXX)
        target_include_directories(kvmem-stagein-hip PRIVATE
            src/adapter
            llama.cpp/include
            llama.cpp/ggml/include)
        target_compile_definitions(kvmem-stagein-hip PRIVATE
            LLAMA_KVMEM=1
            GGML_USE_HIP
            __HIP_PLATFORM_AMD__=1)
        # HIP declares the runtime API [[nodiscard]] where CUDA does not, so the
        # adapter's intentional fire-and-forget cudaFree()/cudaEventDestroy()/
        # cudaStreamWaitEvent() statements warn only on this backend. clang tags
        # these -Wunused-value (not -Wunused-result); suppress precisely that,
        # only for the translation units that pull in the runtime shim, rather
        # than weakening diagnostics for all of libllama.
        target_compile_options(kvmem-stagein-hip PRIVATE
            -Wno-unused-result -Wno-unused-value)
        # No explicit -x hip / --offload-arch here: the hip::device interface
        # already contributes "-x hip --offload-arch=<CMAKE_HIP_ARCHITECTURES>",
        # which is how ggml-hip gets them too. Adding them a second time makes
        # CMake's option de-duplication emit a malformed "-x --offload-arch=...".
        target_link_libraries(kvmem-stagein-hip PRIVATE hip::device)
        target_sources(llama PRIVATE $<TARGET_OBJECTS:kvmem-stagein-hip>)

        # The adapter's host-side .cpp files call the HIP runtime directly, so
        # libllama needs the runtime too. hip::host is a bare INTERFACE target
        # (lib/cmake/hip/hip-targets.cmake) carrying only the amdhip64 link, so
        # this pulls in the import library without adding any compile option.
        target_link_libraries(llama PRIVATE hip::host)

        # Same [[nodiscard]] suppression as above, for the two adapter TUs that
        # include the runtime shim.
        set_source_files_properties(
            ${CMAKE_SOURCE_DIR}/src/adapter/llama-memory-kvmem.cpp
            ${CMAKE_SOURCE_DIR}/src/adapter/llama-memory-kvmem-mtp.cpp
            TARGET_DIRECTORY llama
            PROPERTIES COMPILE_OPTIONS "-Wno-unused-result;-Wno-unused-value")
    endif()

    add_executable(llama-kvmem-cli tools/llama-kvmem-cli.cpp tools/kvmem-spec.cpp)
    target_include_directories(llama-kvmem-cli PRIVATE
        src/adapter
        llama.cpp/include
        llama.cpp/common)
    target_link_libraries(llama-kvmem-cli PRIVATE llama llama-common)
    target_compile_features(llama-kvmem-cli PRIVATE cxx_std_17)

    add_executable(llama-kvmem-server tools/llama-kvmem-server.cpp tools/kvmem-spec.cpp
        tools/kvmem-vision.cpp llama.cpp/tools/server/server-common.cpp)
    target_include_directories(llama-kvmem-server PRIVATE
        src/adapter
        llama.cpp/include
        llama.cpp/common
        llama.cpp/tools/server
        llama.cpp/tools/mtmd
        llama.cpp/vendor
        llama.cpp/vendor/cpp-httplib)
    target_link_libraries(llama-kvmem-server PRIVATE llama llama-common cpp-httplib mtmd)
    target_compile_features(llama-kvmem-server PRIVATE cxx_std_17)

    # std::getenv is standard C++ and portable; MSVC marks it deprecated in
    # favour of the Windows-only _dupenv_s, and warns on the POSIX-ish helpers
    # the server uses for the diagnostics toggle (_putenv_s). Make the same
    # choice llama.cpp makes for its own sources. Set per target so it reaches
    # exactly the KVMem tools that read the environment (KVMEM_TRACE, ...).
    # target_compile_definitions takes a single target per call.
    if (WIN32)
        target_compile_definitions(llama-kvmem-cli    PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_definitions(llama-kvmem-server PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()

    foreach(tool llama-kvmem-cli llama-kvmem-server)
        target_compile_definitions(${tool} PRIVATE KVMEM_ENABLE_NVME=$<BOOL:${KVMEM_ENABLE_NVME}>)
    endforeach()

    # Run manually with an MTP GGUF path; no model is downloaded by CTest.
    # Windows-only exclusion: this test drives llama_kv_cache / llama_batch_allocr
    # internals directly. On ELF every symbol of a shared libllama is exported, so
    # it links; PE only exports the LLAMA_API surface, and exporting all of
    # libllama's internals just to link one dev test is not a trade worth making.
    # The test also skips (exit 77) unless an MTP GGUF is supplied explicitly.
    if (NOT WIN32)
        add_executable(kvmem-mtp-kv-test tests/mtp-kv-test.cpp tools/kvmem-spec.cpp)
        target_include_directories(kvmem-mtp-kv-test PRIVATE src/adapter tools llama.cpp/src)
        target_link_libraries(kvmem-mtp-kv-test PRIVATE llama llama-common kvmem)
        add_test(NAME kvmem-mtp-kv-test COMMAND kvmem-mtp-kv-test)
        set_tests_properties(kvmem-mtp-kv-test PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(kvmem-gdn-replay-test tests/gdn-replay-test.cpp)
    target_link_libraries(kvmem-gdn-replay-test PRIVATE ggml)
    add_test(NAME kvmem-gdn-replay-test COMMAND kvmem-gdn-replay-test)
    set_tests_properties(kvmem-gdn-replay-test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(kvmem-reasoning-budget-test tests/reasoning-budget-test.cpp)
    target_include_directories(kvmem-reasoning-budget-test PRIVATE tools)
    target_link_libraries(kvmem-reasoning-budget-test PRIVATE llama-common)
    add_test(NAME kvmem-reasoning-budget-test COMMAND kvmem-reasoning-budget-test)

    add_executable(kvmem-chat-template-test tests/chat-template-test.cpp)
    target_include_directories(kvmem-chat-template-test PRIVATE tools)
    target_link_libraries(kvmem-chat-template-test PRIVATE llama-common)
    add_test(NAME kvmem-chat-template-test COMMAND kvmem-chat-template-test)

    add_executable(kvmem-chat-id-test tests/chat-id-test.cpp)
    target_include_directories(kvmem-chat-id-test PRIVATE tools)
    target_link_libraries(kvmem-chat-id-test PRIVATE llama-common)
    add_test(NAME kvmem-chat-id-test COMMAND kvmem-chat-id-test)
endif()
