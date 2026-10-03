find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(ops/tests.cmake)

ninfer_finalize_bundle(ninfer_tests)

# A focused regression bundle for the opt-in CUDA memory policy. Reuse the same
# test objects; developers need not link all numerical kernel tests to exercise
# the allocator's CLI, accounting and capacity contracts.
set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_memory_policy_tests_PROGRAMS
  ninfer_cli_options_test ninfer_serve_options_test ninfer_load_report_test
  ninfer_request_log_test ninfer_kv_capacity_test ninfer_context_cache_defaults_test
  ninfer_admission_policy_test ninfer_resource_manager_test
  ninfer_arena_test ninfer_arena_ranks_test ninfer_device_buffer_visibility_test
  ninfer_decode_graph_test)
ninfer_finalize_bundle(ninfer_memory_policy_tests)
ninfer_stage_runtime_dlls(ninfer_memory_policy_tests)

# Every test executable shares one per-config runtime directory; stage the FFmpeg/curl DLL closure
# there once so they resolve their imports (the Windows loader only searches the executable's own
# directory, the system directories and PATH).
ninfer_stage_test_runtime_dlls(ninfer_tests)
