target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_gemm_concat.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_gemm_concat_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_gemm_mma_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_gemm_splitk.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_gemm_splitk_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_pair_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/linear_pair.cpp"
)
