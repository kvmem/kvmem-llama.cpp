# Replay implementation review

## Confirmed defects and fixes

- Generic fold passed `cache_r_l` directly to concat. Production allocates this
  tensor as `[30720, 1]`, while the old tests used `[3, 10240]`. Replacing the test
  tensor with the production shape reproduced a `ggml_concat` assertion before
  the fix. Fold now creates a `[3, 10240]` view before concatenation and copies
  the result back into the original cache tensor.
- Generic fold checked that all buffers had the same type but did not check
  whether its execution backend could access that type. It now checks
  `ggml_backend_supports_buft` before constructing or executing the graph.
- The prior claim of full CUDA+Vulkan adapter compatibility was incorrect.
  Replay dispatch is dynamic, but KV staging, capture streams, and device-pointer
  helpers are still selected by the CUDA compile definition. The CUDA adapter
  now rejects non-CUDA GPU placement with instructions to build `GGML_CUDA=OFF`.
  Full mixed-backend staging remains unimplemented.

## Test changes

- Single-layer tests use production-shaped recurrent state and convolution caches.
- Multilayer tests use distinct keys, initial states, and convolution histories.
  Each layer has its own reference output. The accepted prefix retains an old
  history row, so both old history and accepted input history are checked.
- The multilayer test now exercises the native Vulkan fold as well as GGML.
- Single-round prefix tests compare state against a separate scalar FP64
  recurrence with a floating-point tolerance. Same-backend snapshot comparisons
  remain byte-exact. The FP64 calculation does not call the GGML GDN operator.

## What the implementation actually does

The native Vulkan shader computes decay, the state/key dot product, the gated
delta, and the state update for each accepted token. A second shader updates
convolution history. It is a real GPU replay implementation.

The native follow-up aligns the scalar-gated recurrence with CUDA: accumulate
`dot(state, key)`, reduce it, multiply by decay, then compute delta and update.
Both Vulkan forward and fold include `gdn_replay_math.glsl`, corresponding to
CUDA's shared `gdn_delta_f32` and `gdn_update_f32` helpers. This changes Vulkan
floating-point operation ordering from its earlier implementation. Cross-device
bitwise equivalence is not promised because subgroup reductions and exp differ.
The vector-gated KDA branch retains its per-row decay before reduction.

| CUDA component | Vulkan counterpart |
| --- | --- |
| Record forward with K=0 | GDN record pipeline; no committed-state writes |
| `gdn_delta_f32`, `gdn_update_f32` | Shared GLSL recurrence helpers |
| `gdn_fold_f32` | `gdn_replay_fold.comp`, accepted tokens only |
| `gdn_conv_fold_f32` | `gdn_replay_conv.comp`, three-row history |
| Stream synchronization before commit | Backend submission and synchronization before `replay_finish` |

Native failure no longer falls back to GGML. The tests now copy all five forward
inputs into separate replay buffers through graph nodes, matching the model's
capture path. They check partially filled capacity-6 buffers and poison the
rejected suffix of every record. Invalid last-layer descriptors are rejected
before earlier layers are modified.

The portable graph calls the existing `GATED_DELTA_NET` operator with one final
state output and discards its attention output. It avoids per-token snapshots,
but does extra attention work and builds/allocates a graph per fold. It requires
GDN support in the selected backend; it is not a basic-operator implementation
that automatically enables every GGML backend.

Replay and query replay are different paths. A small fold stress test does not
exercise multi-request retrieval, KV staging, or full-model scheduler memory.
Its RSS cannot establish that the original roughly 40 GB memory issue is fixed.
The original workload still needs an end-to-end memory trace on AMD hardware.

The earlier CTest summaries said 10/10, but each suite actually executed nine
tests successfully and skipped the model-dependent MTP test. No model-level
acceptance, end-to-end memory, or AMD subgroup result should be inferred from
those summaries. Software Vulkan exercises llvmpipe's subgroup configuration;
SPIR-V validation alone does not validate other devices' arithmetic behavior.
