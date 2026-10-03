// Width-4 H16 cached instantiation of the INT8-family small-T partial launch; one unit per
// width, geometry and input keeps the long ptxas runs parallel.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

namespace ninfer::ops::detail {

template void launch_small_t_i8<CausalD256H16Kv2, 4, CausalCachedInput>(bool, bool, const Tensor&, CausalCachedInput, const Tensor&, float,
    PagedKVBatchLayerView, const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t,
    std::int32_t, Tensor&, Tensor&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
