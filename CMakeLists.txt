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

# Conversation selection and eviction policy: header-only and free of llama.cpp
# and CUDA, so it stays testable in the portable KVMEM_BUILD_LLAMA=OFF build.
add_executable(kvmem-conversation-store-test tests/conversation-store-test.cpp)
target_include_directories(kvmem-conversation-store-test PRIVATE tools)
add_test(NAME kvmem-conversation-store-test COMMAND kvmem-conversation-store-test)

option(KVMEM_BUILD_LLAMA "Build llama.cpp with the KVMem adapter" ON)
option(KVMEM_ROCM "Build KVMem with the HIP/ROCm backend" OFF)
if (KVMEM_ROCM)
    set(GGML_HIP ON CACHE BOOL "ggml: use HIP" FORCE)
endif()

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
    if (GGML_VULKAN)
        message(FATAL_ERROR "This HIP integration does not include the Vulkan adapter. Use the Vulkan worktree; do not combine its backend with CUDA/HIP staging.")
    endif()
    if (KVMEM_GPU_BACKEND STREQUAL "CUDA")
        # Let CMake discover nvcc (or honor CUDACXX/CMAKE_CUDA_COMPILER) and
        # let ggml select the local GPU architecture. This keeps the CUDA path
        # portable without changing its backend semantics.
        option(GGML_CUDA "ggml: use CUDA" ON)
    else()
        # ROCm: the CUDA backend must stay off and no CUDA arch may be forced.
        set(GGML_CUDA OFF CACHE BOOL "ggml: use CUDA" FORCE)
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

        # Respect an explicit target, but otherwise ask the installed ROCm
        # toolchain which AMD devices are present. A repository default such as
        # gfx1030/gfx1100 produces binaries that silently fail or run on the
        # wrong ISA when the checkout is moved to another machine.
        if (NOT GPU_TARGETS AND DEFINED ENV{GPU_TARGETS})
            set(GPU_TARGETS "$ENV{GPU_TARGETS}" CACHE STRING "AMD GPU targets for the HIP build")
        elseif (NOT GPU_TARGETS AND AMDGPU_TARGETS)
            set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "AMD GPU targets for the HIP build")
        elseif (NOT GPU_TARGETS AND DEFINED ENV{AMDGPU_TARGETS})
            set(GPU_TARGETS "$ENV{AMDGPU_TARGETS}" CACHE STRING "AMD GPU targets for the HIP build")
        endif()
        if (NOT GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            find_program(KVMEM_AMDGPU_ARCH
                NAMES amdgpu-arch amdgpu-arch.exe
                HINTS "${ROCM_PATH}/bin" "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/lib/llvm/bin")
            if (KVMEM_AMDGPU_ARCH)
                execute_process(
                    COMMAND "${KVMEM_AMDGPU_ARCH}"
                    RESULT_VARIABLE KVMEM_AMDGPU_ARCH_RESULT
                    OUTPUT_VARIABLE KVMEM_DETECTED_GPU_TARGETS
                    ERROR_VARIABLE KVMEM_AMDGPU_ARCH_ERROR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                string(REPLACE "\r\n" ";" KVMEM_DETECTED_GPU_TARGETS "${KVMEM_DETECTED_GPU_TARGETS}")
                string(REPLACE "\n" ";" KVMEM_DETECTED_GPU_TARGETS "${KVMEM_DETECTED_GPU_TARGETS}")
                list(FILTER KVMEM_DETECTED_GPU_TARGETS EXCLUDE REGEX "^$")
                list(REMOVE_DUPLICATES KVMEM_DETECTED_GPU_TARGETS)
            endif()
            # WSL can expose HIP devices while LLVM's PCI-based probe is empty.
            if (NOT KVMEM_DETECTED_GPU_TARGETS AND NOT WIN32)
                find_program(KVMEM_ROCM_ENUM NAMES rocm_agent_enumerator
                    HINTS "${ROCM_PATH}/bin" NO_DEFAULT_PATH)
                if (KVMEM_ROCM_ENUM)
                    execute_process(COMMAND "${KVMEM_ROCM_ENUM}"
                        RESULT_VARIABLE KVMEM_AMDGPU_ARCH_RESULT
                        OUTPUT_VARIABLE KVMEM_ENUM_OUTPUT
                        ERROR_VARIABLE KVMEM_AMDGPU_ARCH_ERROR
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
                    string(REGEX MATCHALL "gfx[0-9a-f]+" KVMEM_DETECTED_GPU_TARGETS "${KVMEM_ENUM_OUTPUT}")
                    list(FILTER KVMEM_DETECTED_GPU_TARGETS EXCLUDE REGEX "^gfx000$")
                    list(REMOVE_DUPLICATES KVMEM_DETECTED_GPU_TARGETS)
                endif()
            endif()
            if (KVMEM_AMDGPU_ARCH_RESULT EQUAL 0 AND KVMEM_DETECTED_GPU_TARGETS)
                set(GPU_TARGETS "${KVMEM_DETECTED_GPU_TARGETS}" CACHE STRING
                    "AMD GPU targets detected by amdgpu-arch")
            else()
                message(FATAL_ERROR
                    "Could not detect an AMD GPU architecture with amdgpu-arch. "
                    "Connect a supported GPU, or cross-compile explicitly with "
                    "-DGPU_TARGETS=<gfx...> (or set GPU_TARGETS in the environment). "
                    "amdgpu-arch reported: ${KVMEM_AMDGPU_ARCH_ERROR}")
            endif()
        endif()
        if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")
        endif()

        # PR #33 enables HIP at the root; keep Windows on the CXX HIP path.
        if (NOT WIN32)
            cmake_minimum_required(VERSION 3.21)
            if (NOT CMAKE_HIP_COMPILER)
                find_program(KVMEM_HIP_CLANG NAMES clang++
                    HINTS "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/lib/llvm/bin" "${ROCM_PATH}/bin"
                    NO_DEFAULT_PATH REQUIRED)
                set(CMAKE_HIP_COMPILER "${KVMEM_HIP_CLANG}" CACHE FILEPATH "HIP compiler")
            endif()
            enable_language(HIP)
            set(CMAKE_HIP_STANDARD 17)
            set(CMAKE_HIP_STANDARD_REQUIRED ON)
        endif()

        # ggml-hip calls find_package(hip) in its own directory scope, so
        # hip::host / hip::device are not visible up here. Import them again for
        # the root scope (imported targets are not global by default).
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
        if (WIN32 AND EXISTS "${ROCM_PATH}/lib/llvm/amdgcn/bitcode/ocml.bc")
            set(KVMEM_ROCM_DEVICE_LIBS "--rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
            target_compile_options(ggml-hip PRIVATE "${KVMEM_ROCM_DEVICE_LIBS}")
        endif()
        # The stage-in kernels must be built by AMD clang with `-x hip`, exactly
        # how ggml-hip builds ggml-cuda/*.cu on Windows (CMake does not support
        # the HIP language there). They live in their own object library so
        # `-x hip` is confined to this single file: linking hip::host straight
        # into `llama` would turn every libllama translation unit into a HIP
        # device compilation unit.
        add_library(kvmem-stagein-hip OBJECT src/adapter/llama-kvmem-stagein.cu)
        if (WIN32)
            set_source_files_properties(src/adapter/llama-kvmem-stagein.cu PROPERTIES LANGUAGE CXX)
        else()
            set_source_files_properties(src/adapter/llama-kvmem-stagein.cu PROPERTIES LANGUAGE HIP)
            set_property(TARGET kvmem-stagein-hip PROPERTY HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
        endif()
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
        if (KVMEM_ROCM_DEVICE_LIBS)
            target_compile_options(kvmem-stagein-hip PRIVATE "${KVMEM_ROCM_DEVICE_LIBS}")
        endif()
        # No explicit -x hip / --offload-arch here: the hip::device interface
        # already contributes "-x hip --offload-arch=<CMAKE_HIP_ARCHITECTURES>",
        # which is how ggml-hip gets them too. Adding them a second time makes
        # CMake's option de-duplication emit a malformed "-x --offload-arch=...".
        if (WIN32)
            target_link_libraries(kvmem-stagein-hip PRIVATE hip::device)
        else()
            target_link_libraries(kvmem-stagein-hip PRIVATE hip::host)
        endif()
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
        if (GGML_CUDA)
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(kvmem-mtp-kv-test PRIVATE CUDA::cudart)
            target_compile_definitions(kvmem-mtp-kv-test PRIVATE KVMEM_TEST_CUDA)
        endif()
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

    add_executable(kvmem-server-options-test tests/server-options-test.cpp)
    target_include_directories(kvmem-server-options-test PRIVATE tools llama.cpp/include llama.cpp/ggml/include)
    add_test(NAME kvmem-server-options-test COMMAND kvmem-server-options-test)

    find_package(Threads REQUIRED)
    add_executable(kvmem-server-progress-test tests/server-progress-test.cpp)
    target_include_directories(kvmem-server-progress-test PRIVATE tools llama.cpp/vendor)
    target_link_libraries(kvmem-server-progress-test PRIVATE Threads::Threads)
    add_test(NAME kvmem-server-progress-test COMMAND kvmem-server-progress-test)

    add_executable(kvmem-output-limit-test tests/output-limit-test.cpp)
    target_include_directories(kvmem-output-limit-test PRIVATE tools llama.cpp/vendor/cpp-httplib)
    target_link_libraries(kvmem-output-limit-test PRIVATE llama-common cpp-httplib)
    add_test(NAME kvmem-output-limit-test COMMAND kvmem-output-limit-test)
endif()
