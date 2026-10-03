find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Windows links the static CUDA runtime so shipped binaries do not depend on a cudart DLL.
if(WIN32)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()

# vcpkg (the Windows build) exposes FFmpeg and libcurl through CMake packages; Linux uses
# pkg-config. Both paths produce the same two consumer targets. With vcpkg's CMake toolchain its
# own FFmpeg find wrapper serves find_package; build_native.bat instead exports VCPKG_ROOT and
# VCPKG_TARGET_TRIPLET for a prebuilt triplet tree, which cmake/FindFFMPEG.cmake reads (it also
# names that tree FFMPEG_VCPKG_TREE, for staging its DLLs beside the executables).
if(WIN32 AND NOT VCPKG_TOOLCHAIN AND DEFINED ENV{VCPKG_ROOT} AND
   DEFINED ENV{VCPKG_TARGET_TRIPLET})
  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{VCPKG_ROOT}/installed/$ENV{VCPKG_TARGET_TRIPLET}")
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
else()
  find_package(PkgConfig REQUIRED)
  # Keep the floors the previous root build required: src/media/decode/decode.cpp is written
  # against these ABIs, so an older installation must fail at configure rather than at link.
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
    # CURL::libcurl is an ALIAS of the concrete imported target, and the DLLs its import library
    # references are staged only when declared on that target.
    if(DEFINED FFMPEG_VCPKG_TREE)
      if(TARGET CURL::libcurl_shared)
        set(_curl_real CURL::libcurl_shared)
      else()
        set(_curl_real CURL::libcurl_static)
      endif()
      if(TARGET ${_curl_real})
        file(GLOB _curl_dlls "${FFMPEG_VCPKG_TREE}/bin/*.dll")
        if(_curl_dlls)
          set_target_properties(${_curl_real} PROPERTIES
            IMPORTED_LINK_DEPENDENT_LIBRARIES "${_curl_dlls}")
        endif()
      endif()
    endif()
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
