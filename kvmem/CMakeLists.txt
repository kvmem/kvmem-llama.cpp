if (WIN32)
    set(KVMEM_NVME_DEFAULT OFF)
else()
    set(KVMEM_NVME_DEFAULT ON)
endif()
option(KVMEM_ENABLE_NVME "Build the POSIX NVMe KV storage tier" ${KVMEM_NVME_DEFAULT})
if (WIN32 AND KVMEM_ENABLE_NVME)
    message(FATAL_ERROR "NVMe storage is not supported on Windows; use KVMEM_ENABLE_NVME=OFF")
endif()

add_library(kvmem
    src/host/kvmem_store.cpp
    src/host/kvmem_runtime.cpp
    src/host/raw_kv_store.cpp
    src/host/mean_key_retrieval.cpp
    src/host/memory_contract.cpp
    src/host/kv_content.cpp
    src/host/native_kv_payload.cpp
    src/host/session_memory.cpp
    src/host/rope.cpp
)
set_target_properties(kvmem PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kvmem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvmem PUBLIC cxx_std_17)
target_compile_definitions(kvmem PUBLIC KVMEM_ENABLE_NVME=$<BOOL:${KVMEM_ENABLE_NVME}>)

if (WIN32)
    # std::getenv is standard C++ and portable; MSVC marks it deprecated in
    # favour of the Windows-only _dupenv_s. Make the same choice llama.cpp makes
    # for its own sources rather than edit portable code.
    target_compile_definitions(kvmem PRIVATE _CRT_SECURE_NO_WARNINGS)
    # On ELF a shared library exports every symbol, which is what
    # `target_link_libraries(llama PRIVATE kvmem)` relies on. PE has no such
    # default, so without this libllama fails to link with undefined
    # kvmem::* symbols. Let CMake generate the export list.
    set_target_properties(kvmem PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

function(kvmem_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE kvmem)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

kvmem_add_test(backend_rebind_test)
kvmem_add_test(kvmem_store_test)
kvmem_add_test(pinned_kv_tier_test)
kvmem_add_test(memory_contract_test)
kvmem_add_test(native_kv_payload_test)
kvmem_add_test(native_residency_test)
kvmem_add_test(kv_content_test)
kvmem_add_test(mean_key_retrieval_test)
if (NOT WIN32 OR NOT KVMEM_ENABLE_NVME)
    # The runtime suite already guards its only NVMe case with this option.
    kvmem_add_test(kvmem_runtime_test)
endif()
if (WIN32)
    # NVMe-specific suites require the POSIX implementation.
    message(STATUS "kvmem: skipping POSIX/NVMe suites on Windows")
else()
    if (KVMEM_ENABLE_NVME)
        kvmem_add_test(nvme_kv_tier_test)
    else()
        kvmem_add_test(nvme_disabled_test)
    endif()
    kvmem_add_test(raw_kv_store_test)
endif()
