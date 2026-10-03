set(ninfer_op_tests
  add_bias
  gelu
  hadamard_transform
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  rmsnorm_pack_tail
  gated_rmsnorm
  l2norm
  gated_delta_net
  gated_delta_net_two_stage
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  prepare_ragged_prefix
  scatter
  scatter_bf16_batch
  target_logprobs
  paged_kv_window
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES ninfer_ops)
endforeach()

ninfer_add_op_test(ninfer_linear_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp"
  LIBRARIES ninfer_ops)

add_test(NAME ninfer_softmax_attention_nvfp4_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --nvfp4-only)

add_test(NAME ninfer_softmax_attention_k8v4_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --k8v4-only)

add_test(NAME ninfer_softmax_attention_rk4v4_e8_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --rk4v4-e8-only)
add_test(NAME ninfer_softmax_attention_rk2v4_e8_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --rk2v4-e8-only)

# Upstream's DFlash2 verification sweep. ~460 s: six storage families (five upstream, plus this
# fork's rk8v4) over every narrow width, batch and base offset, so it gets a timeout well clear of
# the default 1500 s only if the machine is slow -- left at the default here because it finishes in
# under a third of it.
add_test(NAME ninfer_softmax_attention_dflash2_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --dflash2-only)
add_test(NAME ninfer_softmax_attention_int8_prompt_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --int8-prompt-only)
add_test(NAME ninfer_softmax_attention_pack_gqa_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --pack-gqa-only)

set_tests_properties(
  ninfer_softmax_attention_nvfp4_test
  ninfer_softmax_attention_k8v4_test
  ninfer_softmax_attention_rk4v4_e8_test
  ninfer_softmax_attention_rk2v4_e8_test
  ninfer_softmax_attention_dflash2_test
  ninfer_softmax_attention_int8_prompt_test
  ninfer_softmax_attention_pack_gqa_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_kv_plane_types_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_plane_types.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_rmsnorm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp"
  LIBRARIES ninfer_ops)

add_test(NAME ninfer_kv_cache_append_nvfp4_test
  COMMAND ninfer_tests ninfer_kv_cache_append_test --nvfp4-only)

add_test(NAME ninfer_kv_cache_append_k8v4_test
  COMMAND ninfer_tests ninfer_kv_cache_append_test --k8v4-only)

set_tests_properties(
  ninfer_kv_cache_append_nvfp4_test
  ninfer_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe.cpp"
  LIBRARIES ninfer_ops)

# The routing sort on its own, exhaustively. The Op test above covers SparseMoe end to end and
# would not localise a network that mis-sorts a minority of score permutations; this one runs the
# device function over all 8! orderings and all 3^8 tie patterns.
ninfer_add_op_test(ninfer_sparse_moe_route_network_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe_route_network.cu"
  LIBRARIES ninfer_ops)

# Table decode of the E8 root codec against its arithmetic reference, over every code pair.
ninfer_add_op_test(ninfer_e8_root_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_e8_root_decode.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_proj_fused_rmsnorm_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj_fused_rmsnorm.cpp"
  LIBRARIES ninfer_ops)
set_tests_properties(ninfer_attn_input_proj_fused_rmsnorm_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_small_t_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_small_t.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_small_t_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_small_t.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_dynamic_grouped_conv_add_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_state_fp16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_state_fp16.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_vocabulary_transcode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vocabulary_transcode.cpp"
  LIBRARIES ninfer_ops ninfer_artifact)

# Pure host resolution -- no device work, so it runs on a machine with no GPU at all.
ninfer_add_op_test(ninfer_route_coverage_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_coverage.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")

add_test(NAME ninfer_softmax_attention_wide_test
  COMMAND ninfer_tests ninfer_softmax_attention_test --wide-only)
set_tests_properties(ninfer_softmax_attention_wide_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 RUN_SERIAL TRUE)

add_test(NAME ninfer_sparse_moe_wide_test
  COMMAND ninfer_tests ninfer_sparse_moe_test --wide-only)
set_tests_properties(ninfer_sparse_moe_wide_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)

foreach(mode IN ITEMS ngram-only onehot-distribution mtp-onehot mtp-distribution ngram-negative-penalties wide-accept wide-distribution)
  string(REPLACE "-" "_" test_suffix "${mode}")
  add_test(NAME ninfer_speculative_${test_suffix}_test
    COMMAND ninfer_tests ninfer_speculative_round_test --${mode})
  set_tests_properties(ninfer_speculative_${test_suffix}_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
endforeach()
set_tests_properties(ninfer_speculative_wide_accept_test PROPERTIES TIMEOUT 600)
set_tests_properties(ninfer_speculative_wide_distribution_test PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

add_test(NAME ninfer_gdn_state_fp16_two_stage_test
  COMMAND ninfer_tests ninfer_gdn_state_fp16_test 64 --two-stage)
set_tests_properties(ninfer_gdn_state_fp16_two_stage_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

add_test(NAME ninfer_gdn_replay_fold_wide_test
  COMMAND ninfer_tests ninfer_gdn_replay_fold_test --wide-only)
set_tests_properties(ninfer_gdn_replay_fold_wide_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)

# The fused projections once more with every width on each route table: their own routes, and
# upstream's over the unified templates that a device profile may take per width.
foreach(table IN ITEMS legacy unified)
  foreach(test IN ITEMS attn_input_proj attn_input_small_t gdn_input_proj gdn_input_small_t
                        gdn_input_proj_conv_snapshot gdn_input_proj_conv_record linear_add_q4_a16
                        linear_add_q5_a16 linear_add_q5_small_t linear_add_q8_a16
                        linear_pair_q8_a16 linear_swiglu_q4_a16 linear_swiglu_q8_a16 linear_topk
                        context_kv_materialize linear_dynamic_grouped_conv_add)
    add_test(NAME ninfer_${test}_${table}_routes_test COMMAND ninfer_tests ninfer_${test}_test)
    set_tests_properties(ninfer_${test}_${table}_routes_test
      PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 ENVIRONMENT NINFER_LINEAR_ROUTES=${table})
  endforeach()
endforeach()

# The FP8 Linear and fused suites on each table too.
foreach(table IN ITEMS legacy unified)
  foreach(test IN ITEMS linear_fp8_a16 linear_fp8_a8 linear_add_fp8 linear_swiglu_fp8)
    add_test(NAME ninfer_${test}_${table}_routes_test COMMAND ninfer_tests ninfer_${test}_test)
    set_tests_properties(ninfer_${test}_${table}_routes_test
      PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 ENVIRONMENT NINFER_LINEAR_ROUTES=${table})
  endforeach()
endforeach()

# The NVFP4 Linear and fused suites on each table too.
foreach(table IN ITEMS legacy unified)
  foreach(test IN ITEMS linear_nvfp4_a16 linear_nvfp4_a4 linear_add_nvfp4 linear_swiglu_nvfp4)
    add_test(NAME ninfer_${test}_${table}_routes_test COMMAND ninfer_tests ninfer_${test}_test)
    set_tests_properties(ninfer_${test}_${table}_routes_test
      PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 ENVIRONMENT NINFER_LINEAR_ROUTES=${table})
  endforeach()
endforeach()

# The BF16 Linear suite on each table too.
foreach(table IN ITEMS legacy unified)
  add_test(NAME ninfer_linear_bf16_a16_${table}_routes_test
    COMMAND ninfer_tests ninfer_linear_bf16_a16_test)
  add_test(NAME ninfer_linear_add_bf16_a16_${table}_routes_test
    COMMAND ninfer_tests ninfer_linear_add_bf16_a16_test)
  set_tests_properties(ninfer_linear_bf16_a16_${table}_routes_test
    ninfer_linear_add_bf16_a16_${table}_routes_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 ENVIRONMENT NINFER_LINEAR_ROUTES=${table})
endforeach()
