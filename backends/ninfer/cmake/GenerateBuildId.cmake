# GenerateBuildId.cmake -- write the NInfer build id to a C header at BUILD time.
#
# Run as:  cmake -DSRC_DIR=<src> -DOUT=<out header> -P cmake/GenerateBuildId.cmake
#
# The id is `<base>[-dirty]`:
#
#   <base>   $NINFER_BUILD_ID if that env var is set (the one-off restamp build_*.bat
#            scripts use it), otherwise `git describe --always --tags` from the source
#            tree.
#   -dirty   appended when the product sources differ from HEAD -- see below.
#
# Generating it at build time -- instead of configure time, as apps/CMakeLists.txt did -- means
# the baked id always reflects the tree that is actually being compiled, not a stale value from
# the last reconfigure. The header is only rewritten when its content changes, so main.cpp
# recompiles only when the id actually changes (no needless churn on every build).
#
# Why the dirty test is scoped instead of plain `git describe --dirty`: --dirty marks the whole
# worktree, so an uncommitted edit *outside* the compiled sources -- a converter script, a test, a
# README, a local log or watcher, an experiment directory -- stamps a release built from an
# otherwise pristine commit as dirty. That is what shipped on 2026-09-22: binaries built from
# eb5396bf reported eb5396bf-dirty only because tests/convert/test_sources.py and
# tools/convert/sources/compressed_tensors.py happened to be modified in the working tree.
#
# So the dirty test asks the narrower question that the id is meant to answer: do the files that
# are actually compiled into these binaries match HEAD? A tracked modification, or a newly added
# file, under one of the product source roots below means they do not; anything elsewhere cannot
# change the binary and must not taint its identity. Keep NINFER_PRODUCT_ROOTS in step with the
# layout -- a new top-level directory whose sources get compiled in belongs in this list, or
# changes to it would go unreported.

set(NINFER_PRODUCT_ROOTS src include apps cmake third_party)

if(ENV{NINFER_BUILD_ID})
  set(_id "$ENV{NINFER_BUILD_ID}")
else()
  set(_id "")
  set(_dirty "")
  find_program(_git_executable git)
  if(_git_executable)
    execute_process(
      COMMAND "${_git_executable}" describe --always --tags
      WORKING_DIRECTORY "${SRC_DIR}"
      OUTPUT_VARIABLE _id
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(_id)
      # Tracked modifications and new files under the compiled roots, relative to HEAD. Untracked
      # files elsewhere (local logs, watchers, experiment output) are invisible here by design.
      execute_process(
        COMMAND "${_git_executable}" status --porcelain --untracked-files=normal --
                ${NINFER_PRODUCT_ROOTS} CMakeLists.txt CMakePresets.json
        WORKING_DIRECTORY "${SRC_DIR}"
        OUTPUT_VARIABLE _product_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _status_result
        ERROR_QUIET)
      if(NOT _status_result EQUAL 0)
        # Cannot tell whether the sources match HEAD, so do not claim a clean identity the build
        # has not earned.
        set(_dirty "-dirty")
        message(WARNING "git status failed (${_status_result}); assuming the product sources "
                        "are dirty for the NINFER build id.")
      elseif(_product_status)
        set(_dirty "-dirty")
        message(STATUS "NINFER build id will be marked dirty -- product sources differ "
                       "from HEAD:\n${_product_status}")
      endif()
    endif()
  endif()
  if(NOT _id)
    set(_id "unknown")
    # A build that cannot record its real identity must not pass silently: the startup log would
    # otherwise show "build unknown" for a real binary.
    message(WARNING "NINFER build id unavailable (git missing or not a repository) -- "
                    "the startup log will record 'build unknown'. Rebuild from a git "
                    "checkout (or set NINFER_BUILD_ID) to record the real id.")
  else()
    set(_id "${_id}${_dirty}")
  endif()
endif()

string(REPLACE "\"" "\\\"" _id_escaped "${_id}")
# Single string (not a list) so file(WRITE) does not inject a ';' list separator.
set(_content "// Auto-generated at build time by cmake/GenerateBuildId.cmake -- do not edit.\n#define NINFER_BUILD_ID \"${_id_escaped}\"\n")

if(EXISTS "${OUT}")
  file(READ "${OUT}" _existing)
  if(_existing STREQUAL _content)
    return()
  endif()
endif()

get_filename_component(_out_dir "${OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")
file(WRITE "${OUT}" "${_content}")
message(STATUS "NINFER build id: ${_id}")