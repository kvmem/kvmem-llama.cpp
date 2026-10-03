add_library(ninfer_linear_swiglu_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_swiglu_test_common.cpp")
ninfer_test_includes(ninfer_linear_swiglu_test_support)
ninfer_op_oracle_options(ninfer_linear_swiglu_test_support)
target_link_libraries(ninfer_linear_swiglu_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_swiglu_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_nvfp4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

# Integer-activation (s8) routes selected by LinearPolicy::AllowA8Int / AllowA8IntDecode.
ninfer_add_op_test(ninfer_linear_swiglu_q4_a8int_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a8int.cpp"
  LIBRARIES ninfer_linear_swiglu_test_support)

ninfer_add_op_test(ninfer_linear_swiglu_q4a8_int_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4a8_int.cpp"
  LIBRARIES ninfer_ops)

# The cuBLAS prefill route (LinearPolicy::AllowPrefillCublas) across linear_swiglu, linear_add and
# the attention/GDN split projections, against an FP64 oracle.
ninfer_add_op_test(ninfer_w4_cublas_prefill_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_w4_cublas_prefill.cpp"
  LIBRARIES ninfer_ops)
