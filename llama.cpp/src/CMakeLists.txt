llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_KVMEM)
    if (LLAMA_KVMEM_ROOT STREQUAL "")
        message(FATAL_ERROR "LLAMA_KVMEM=ON requires LLAMA_KVMEM_ROOT")
    endif()
    target_sources(llama PRIVATE
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-memory-kvmem.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-memory-kvmem-hybrid.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-memory-kvmem-mtp.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-batch.cpp
        ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-capture.cpp
    )
    if (GGML_CUDA)
        enable_language(CUDA)
        target_sources(llama PRIVATE
            ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-stagein.cu)
        if (DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET llama PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
        endif()
    endif()
    target_include_directories(llama PRIVATE
        ${LLAMA_KVMEM_ROOT}/kvmem/include
        ${LLAMA_KVMEM_ROOT}/src/adapter
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src
    )
    target_compile_definitions(llama PRIVATE LLAMA_KVMEM=1)
    if (NOT TARGET kvmem)
        add_subdirectory(${LLAMA_KVMEM_ROOT}/kvmem ${CMAKE_BINARY_DIR}/kvmem-from-llama)
    endif()
    target_link_libraries(llama PRIVATE kvmem)
    if (GGML_CUDA)
        find_package(CUDAToolkit QUIET)
        if (CUDAToolkit_FOUND)
            target_link_libraries(llama PRIVATE CUDA::cudart)
        endif()
    endif()
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
