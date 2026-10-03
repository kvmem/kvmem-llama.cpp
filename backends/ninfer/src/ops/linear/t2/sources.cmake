target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/t2_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/t2_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/t2_rowsplit_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/t2_rowsplit_gemm_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/t2_rowsplit_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/t2_small_t_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/t2_small_t_v2.cu"
)
