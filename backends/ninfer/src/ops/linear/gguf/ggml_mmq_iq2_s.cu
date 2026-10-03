// The integer tensor-core product for one block type, in its own unit so the heavy template builds in
// parallel with the others.
#include "ggml_bridge_mmq.cuh"

NINFER_GGUF_MATRIX_INSTANCE(GGML_TYPE_IQ2_S);
