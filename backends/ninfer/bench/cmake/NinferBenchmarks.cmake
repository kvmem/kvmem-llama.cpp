# Microbenchmarks are programs in the ninfer_benches bundle (cmake/NinferBundles.cmake): one
# executable instead of one per benchmark, each carrying its own copy of the kernel image. Run one
# with `ninfer_benches <name> [args...]`; `ninfer_benches --list` names them. The product benchmark
# ninfer_bench stays its own executable.
include(${PROJECT_SOURCE_DIR}/cmake/NinferBundles.cmake)

function(ninfer_add_bench name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LIBRARIES")
  ninfer_bundle_program(ninfer_benches ${name} SOURCES ${arg_SOURCES} LIBRARIES ${arg_LIBRARIES})
  ninfer_internal_includes(${name})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

function(ninfer_add_op_bench name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES")
  ninfer_add_bench(${name} SOURCES ${arg_SOURCES} LIBRARIES ninfer_ops)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ops)
endfunction()
