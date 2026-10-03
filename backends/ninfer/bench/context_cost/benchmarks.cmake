# Static context-cost calibrator. Transfer measurements call production Paged-KV copy primitives,
# then release their synthetic fixtures before the public Engine optionally loads one real artifact
# for text/Vision prefill measurements.
ninfer_add_bench(ninfer_context_cost_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_cost_bench.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/context_cost_measure.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/model_context_fixture.cpp"
  LIBRARIES ninfer_engine ninfer_core ${NINFER_CUDART_TARGET} ninfer::json)
target_include_directories(ninfer_context_cost_bench PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(ninfer_context_cost_bench PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
