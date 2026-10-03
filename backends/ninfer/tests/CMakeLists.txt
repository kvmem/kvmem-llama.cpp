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

# P1 feasibility probes: opt-in targets in the isolated checkout, no production CLI.
add_executable(ninfer_p1_engine experimental/p1_engine.cpp)
target_link_libraries(ninfer_p1_engine PRIVATE ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p1_engine)
ninfer_add_test(ninfer_p1_sparse STANDALONE
  SOURCES experimental/p1_sparse.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_p1_sparse)
ninfer_add_test(ninfer_p1_payload STANDALONE
  SOURCES experimental/p1_payload.cpp LIBRARIES ninfer_core)
ninfer_add_test(ninfer_p1_options STANDALONE
  SOURCES experimental/p1_options.cpp LIBRARIES ninfer::engine ninfer_core ninfer::json)
ninfer_stage_runtime_dlls(ninfer_p1_options)
ninfer_add_test(ninfer_p1_rope STANDALONE
  SOURCES experimental/p1_rope.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_p1_rope)

# The query provenance tests are also runnable without linking unrelated GPU test families.
ninfer_add_test(ninfer_memory_frontend STANDALONE NEEDS_SOURCE_DIR
  SOURCES models/qwen3_5/test_frontend.cpp LIBRARIES ninfer_engine ninfer_core ninfer::json)
ninfer_stage_runtime_dlls(ninfer_memory_frontend)
ninfer_add_test(ninfer_token_sums STANDALONE
  SOURCES experimental/token_sums.cpp LIBRARIES ninfer_ops)
ninfer_op_oracle_options(ninfer_token_sums)
ninfer_add_test(ninfer_p4_retrieval STANDALONE
  SOURCES experimental/p4_retrieval.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p4_retrieval)
ninfer_add_test(ninfer_p4_history STANDALONE
  SOURCES experimental/p4_history.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p4_history)
ninfer_add_test(ninfer_memory_sessions STANDALONE
  SOURCES experimental/memory_sessions.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_sessions)
ninfer_add_test(ninfer_memory_concurrency STANDALONE
  SOURCES experimental/memory_concurrency.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_concurrency)
ninfer_add_test(ninfer_memory_vision STANDALONE
  SOURCES experimental/memory_vision.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_vision)
ninfer_add_test(ninfer_memory_disk STANDALONE
  SOURCES experimental/memory_disk.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_disk)
ninfer_add_test(ninfer_memory_disk_final STANDALONE
  SOURCES experimental/memory_disk.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_disk_final)
# Independently linked so the preceding P6-P9 qualification keeps its binaries.
ninfer_add_test(ninfer_memory_formats STANDALONE
  SOURCES experimental/p4_history.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_formats)
ninfer_add_test(ninfer_memory_ngram STANDALONE
  SOURCES experimental/memory_ngram.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_memory_ngram)
ninfer_add_test(ninfer_p4_quality STANDALONE
  SOURCES experimental/p4_quality.cpp LIBRARIES ninfer::engine)
ninfer_stage_runtime_dlls(ninfer_p4_quality)

set_property(GLOBAL PROPERTY NINFER_BUNDLE_ninfer_kvmem_product_tests_PROGRAMS
  ninfer_serve_options_test ninfer_openai_schema_test ninfer_load_report_test
  ninfer_resource_manager_test)
ninfer_finalize_bundle(ninfer_kvmem_product_tests)
ninfer_stage_runtime_dlls(ninfer_kvmem_product_tests)
