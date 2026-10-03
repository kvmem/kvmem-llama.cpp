# Executable bundles: many test or benchmark programs linked into one executable.
#
# Every program that links ninfer_ops carries the whole kernel image (~450 MB on sm_86). Linked
# one executable per program, the ~150 tests and benchmarks took tens of GB for copies of the same
# device code. A bundle compiles each program as an OBJECT library whose `main` is renamed to a
# unique entry, and links all of them once, behind a dispatcher:
#
#   ninfer_tests <program> [args...]
#
# The program sees argv[0] == <program>, so usage text built from argv[0] reads as before. Each
# program still runs in its own process, one per CTest entry, so exit codes (77 = skip), global
# state and CUDA context are exactly as they were for a standalone executable.
#
# A program's sources must not define non-internal symbols that another program in the same bundle
# also defines; keep helpers in an anonymous namespace (the link fails loudly if not).

# Declare `name` as a program in `bundle`. The OBJECT library `name` accepts ordinary target_*
# calls afterwards; its link libraries are collected when the bundle is finalized.
function(ninfer_bundle_program bundle name)
  cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "SOURCES;LIBRARIES")
  add_library(${name} OBJECT ${arg_SOURCES})
  target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  target_compile_definitions(${name} PRIVATE main=ninfer_bundle_entry_${name})
  # A renamed main loses main's implicit `return 0`, so falling off its end returns garbage and a
  # passing program reports failure. Make that a compile error instead.
  if(MSVC)
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/we4715 /we4716>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/we4715,/we4716>)
  else()
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror=return-type>)
  endif()
  set_property(GLOBAL APPEND PROPERTY NINFER_BUNDLE_${bundle}_PROGRAMS ${name})
endfunction()

# The command that runs `name` from `bundle`, for add_test(COMMAND ...).
function(ninfer_bundle_command out bundle name)
  set(${out} ${bundle} ${name} PARENT_SCOPE)
endfunction()

# Create the `bundle` executable from every program declared so far.
function(ninfer_finalize_bundle bundle)
  get_property(programs GLOBAL PROPERTY NINFER_BUNDLE_${bundle}_PROGRAMS)
  if(NOT programs)
    return()
  endif()
  list(SORT programs)

  set(declarations "")
  set(entries "")
  set(libraries "")
  foreach(program IN LISTS programs)
    # Detect which of the two standard signatures the renamed main has.
    set(signature "int, char**")
    set(call "ninfer_bundle_entry_${program}")
    get_target_property(sources ${program} SOURCES)
    get_target_property(source_dir ${program} SOURCE_DIR)
    set(found_main OFF)
    foreach(source IN LISTS sources)
      if(NOT IS_ABSOLUTE "${source}")
        set(source "${source_dir}/${source}")
      endif()
      file(READ "${source}" text)
      if(text MATCHES "int[ \t\r\n]+main[ \t\r\n]*\\([ \t\r\n]*int")
        set(found_main ON)
      elseif(text MATCHES "int[ \t\r\n]+main[ \t\r\n]*\\([ \t\r\n]*\\)" OR
             text MATCHES "NINFER_GUARDED_TEST_MAIN\\(")
        set(found_main ON)
        set(signature "")
        set(call "[](int, char**) { return ninfer_bundle_entry_${program}(); }")
      endif()
    endforeach()
    if(NOT found_main)
      message(FATAL_ERROR "bundle program ${program} defines no recognizable main")
    endif()
    string(APPEND declarations "int ninfer_bundle_entry_${program}(${signature});\n")
    string(APPEND entries "    {\"${program}\", ${call}},\n")

    get_target_property(program_libraries ${program} LINK_LIBRARIES)
    if(program_libraries)
      list(APPEND libraries ${program_libraries})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES libraries)

  set(dispatcher "${CMAKE_CURRENT_BINARY_DIR}/${bundle}_dispatch.cpp")
  set(NINFER_BUNDLE_NAME "${bundle}")
  set(NINFER_BUNDLE_DECLARATIONS "${declarations}")
  set(NINFER_BUNDLE_ENTRIES "${entries}")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/bundle_dispatch.cpp.in" "${dispatcher}" @ONLY)

  add_executable(${bundle} "${dispatcher}")
  target_link_libraries(${bundle} PRIVATE ${programs} ${libraries})
endfunction()
