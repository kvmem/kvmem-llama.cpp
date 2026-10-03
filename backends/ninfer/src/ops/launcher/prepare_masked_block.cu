#include "ops/launcher/prepare_masked_block.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/kernel/prepare_masked_block.cuh"

namespace ninfer::ops::detail {

void prepare_masked_block_launch(const Tensor& anchors, const Tensor& lengths,
                                 const Tensor& valid_columns, std::int32_t mask_id, Tensor& ids,
                                 Tensor& positions, std::int32_t block_size, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_consumer({dim3(ids.ne[1]), dim3(32), 0, stream},
                                    prepare_masked_block_kernel,
                                    static_cast<const std::int32_t*>(anchors.data),
                                    static_cast<const std::int32_t*>(lengths.data),
                                    static_cast<const std::int32_t*>(valid_columns.data), mask_id,
                                    static_cast<std::int32_t*>(ids.data),
                                    static_cast<std::int32_t*>(positions.data), block_size));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
