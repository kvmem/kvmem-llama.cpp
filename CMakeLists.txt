cmake_minimum_required(VERSION 3.16)
project(kvmem_llamacpp LANGUAGES CXX)

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
if (KVMEM_BUILD_LLAMA)
    if (NOT CMAKE_CUDA_COMPILER AND EXISTS "/home/leye/kvmem_qw3/.cu13-env/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/home/leye/kvmem_qw3/.cu13-env/bin/nvcc" CACHE FILEPATH "nvcc")
    endif()
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "120a-real" CACHE STRING "Blackwell sm_120")
    endif()

    option(GGML_CUDA "ggml: use CUDA" ON)
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

    # MSVC warns on the POSIX-ish helpers the server uses for the diagnostics
    # toggle (_putenv_s) and treats those warnings as noise next to real ones.
    if (MSVC)
        target_compile_definitions(llama-kvmem-cli    PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_definitions(llama-kvmem-server PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()

    foreach(tool llama-kvmem-cli llama-kvmem-server)
        target_compile_definitions(${tool} PRIVATE KVMEM_ENABLE_NVME=$<BOOL:${KVMEM_ENABLE_NVME}>)
    endforeach()

    # Run manually with an MTP GGUF path; no model is downloaded by CTest.
    add_executable(kvmem-mtp-kv-test tests/mtp-kv-test.cpp tools/kvmem-spec.cpp)
    target_include_directories(kvmem-mtp-kv-test PRIVATE src/adapter tools llama.cpp/src)
    target_link_libraries(kvmem-mtp-kv-test PRIVATE llama llama-common kvmem)
    add_test(NAME kvmem-mtp-kv-test COMMAND kvmem-mtp-kv-test)
    set_tests_properties(kvmem-mtp-kv-test PROPERTIES SKIP_RETURN_CODE 77)

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
