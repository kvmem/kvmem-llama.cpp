add_library(ninfer_core STATIC
  arena.cu
  resident_memory.cpp
  resident_scan.cu
  device.cu
  stage_link.cu
  dtype.cpp
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_kv_arena.cpp
  disk_kv_store.cpp
  disk_kv_bridge.cpp
  direct_storage_reader.cpp
  token_logprobs.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
  evictable_kv_pool.cu
  evictable_weight_pool.cu
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ${NINFER_CUDART_TARGET} CUDA::nvtx3 Threads::Threads CUDA::cuda_driver)
if(WIN32)
  target_link_libraries(ninfer_core PUBLIC pdh dxgi)
endif()
if(WIN32 AND NINFER_D3D12_RESIDENCY)
  target_link_libraries(ninfer_core PUBLIC d3d12 dxgi)
endif()
if(WIN32 AND NINFER_DIRECTSTORAGE)
  target_link_libraries(ninfer_core PUBLIC ninfer_directstorage)
endif()
