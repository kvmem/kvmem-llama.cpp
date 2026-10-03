# The consolidated line

`master` of this repository is the `master` of
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090) (v0.11.0 and the multi-GPU
pipeline stages, most of both written by Warlax (WarlaxZ); that line continues
[Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090), a fork of Neroued's NInfer) plus
this fork's production behaviours and ternary work, the patches of
[TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090), ideas from
[UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090), pull requests to
[Neroued/ninfer](https://github.com/Neroued/ninfer) and work from the forks listed below. When the
base moves, these commits are applied again on its new head one by one, each adapted to what changed
underneath rather than merged, so every commit stands on its own and carries its rationale and its
authors; this note is the map.

## What the line carries over its base

| area | behaviour | where it lives in the tree |
|---|---|---|
| context cache | a private capture that cannot be placed reclaims the oldest eligible private resident (publication order), so every conversation regains reuse from its second turn | `runtime/engine/context_cache/resource_manager.h`, `shared_capture_planner.h`, the capture gate in `models/qwen3_5/program/transactions/capture.cpp` |
| context cache | an infeasible shared capture takes the same reclamation route, so the shared reuse frontier keeps advancing under pressure | `resource_manager.h` |
| context cache | replacement scenarios are offered only when no catalog slot is vacant, so an exact repeat does not displace the leading-instruction prefix the next question needs | `resource_manager.h` |
| context cache | the demand window behind the committed mask holds 64 requests, beyond the replay distance of a workload that replays a few dozen prefixes twice | `context_portfolio_value.h`, `materialization_planner.h`, `resource_manager.h`, `shared_capture_planner.h` |
| context cache | a private-only capture with no committed demand and no shared credit goes straight to that reclamation: the capture search could not produce a plan for it, yet spent its whole 4,096-target budget on the engine thread first (80 ms to 0.5 s before the first token of every new conversation once the private catalog filled) | `resource_manager.h` |
| engine | store exhaustion during placement (`ContextCacheExhausted`, a `std::bad_alloc` naming the store) fails only the request with HTTP 429 and is counted, instead of failing the worker | the reservation sites in `models/qwen3_5/program/{program_impl,graphs,storage/context,transactions/materialization}.cpp`, the catches in `progress_materialization_transaction`, `runtime/engine/engine_core.h` |
| engine | a Paged KV exhaustion names its page numbers; three consecutive exhaustions without a successful admission mark the Engine unavailable for the healthcheck | `core/paged_kv_cache.cpp`, `engine_core.h` |
| tests | the materialization fault test also counts `cudaHostAlloc`, which `PinnedHostBuffer` pins through since the multi-GPU stages (the base's test fails on every run) | `tests/artifact/materialization_cuda_errors.cpp`, `tests/artifact/tests.cmake` |
| serve | a named or single-tool forced `tool_choice` is executed by opening the call in the generation prompt (v0.11 rejects it with `tool_choice_not_supported`) | `models/qwen3_5/frontend/chat_template.cpp` (opener after the generation prompt), `tool_call_parser.{h,cpp}` (seeded decoder), `serve/translate.cpp` (reasoning rule), the three request parsers, `serve/request_log.*` (schema v22) |

The line also carries ternary checkpoints (Ternary Bonsai 2 27B):

| area | behaviour | where it lives in the tree |
|---|---|---|
| format | `t2_g128_fp16`: ternary codes as 2-bit two's complement with one binary16 scale per 128 columns, row-split only | `core/weight.h`, `core/weight_view.cpp`, `artifact/formats.cpp`, `tools/artifact/` |
| ops | T2 linear routes (small-T tensor-core kernel, narrow MMA tiles for prefill), composed T2 attention/GDN input projections, `linear_add`/`linear_swiglu`, the T2 full head and the indexed T2 proposal head in `linear_topk` (keys carry the shortlist's global token ids) | `ops/linear/t2/`, `ops/wrapper/`, `ops/linear_topk/t2.cu`, `ops/linear_topk/grouped_ksplit_topk.cuh` |
| ops | T2 int8 activations at every width (`--prefill-a8`): a small-T s8 kernel for decode, speculative verify and prompts up to 192 tokens (one launch over both parents of an attention or GDN pair), and above it the int8-activation GEMM of the dense formats, padding T to its cheapest tile width, whose ternary codes decode straight to s8 in the shared mainloop | `ops/linear/t2/t2_small_t_i8.cuh`, `ops/common/rowsplit_a8_mma.cuh` (`T2Codec`), `ops/linear/t2/t2_a8.{h,cu}`, `ops/linear/linear.cpp`, the T2 paths of `ops/wrapper/{linear_add,attn_input_proj,gdn_input_proj}.cpp` |
| ops | the T2 pair's GDN record and snapshot projections take the pair's policy (their two-parent forms were A16 only); the record kernel stages its window in shared memory before the recurrence | `ops/wrapper/gdn_input_proj.cpp`, `models/qwen3_5/execution/gdn.cpp`, `ops/linear_attention/gated_delta_net/recurrent.cuh` |
| ops | `linear_dynamic_grouped_conv_add` takes any row-split projection Linear registers at `[5120, 4096\|17408]`: Linear writes the BF16 plane the materialised Q8 route fills and the shared finish kernel adds the convolution (Q8 keeps its fused routes); Q4 shapes `n5120_k4096`, `n5120_k17408` and `n5120_k25600` for a Q4 DFlash2 adapter | `ops/wrapper/dynamic_grouped_conv.cpp`, `ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_{plan.cpp,materialized.cu}`, `ops/linear/q4/shapes/` |
| ops | `hadamard_transform` and `silu_mul_hadamard`; producers that write their output rotated (`rmsnorm_hadamard`, `gated_rmsnorm_hadamard`, `sigmoid_mul_hadamard`, `gdn_norm_gating_proj_rotated`), bit-identical to the op followed by the transform, each running four warps per 1024-point block (`hadamard_quarter_forward_store`); `embedding_rotated` gathers a T2 token row and applies the inverse transform | `ops/kernel/hadamard_transform.cuh`, `ops/kernel/hadamard_producers.cuh`, `ops/wrapper/hadamard_transform.cpp`, `ops/gdn_gating_proj/bf16/` |
| model | Hadamard-rotated Uses (`hadamard_signs` auxiliary): the norms and gates hand rotated inputs to rotated projections (`InputBasis`), other shapes rotate in place; a T2 token table is restored at the gather with the output head's hidden-width signs; the residual stream stays primal; the ternary output, MTP, DFlash2 and proposal heads take the text projections' integer route | `models/qwen3_5/execution/rotation.h`, `parameters.cpp`, `load/prepare.cpp`, `model.cpp`, the attention/GDN/FFN/head sites |
| attention | a single row of the INT8-family small-T attention runs its splits in whole waves of the device's SMs: the host passes the partial kernel's splits per wave to the partial kernel and the reducer, whose per-window policy rounds down to whole waves within the 64 staged page IDs (the split tiers fill one wave of a 170-SM part; on 82 SMs their counts left a nearly empty last wave) | `ops/softmax_attention/dense/causal_cache/small_t.{cu,cuh}`, `small_t_i8.cuh` |
| vision | an overlay encode window may also borrow the DFlash adapter (ranked below MTP), which a ternary table and head alone cannot cover | `models/qwen3_5/load/vision_overlay.cpp`, `load.cpp` |
| convert | `bonsai2_27b_ternary` builds a Qwen3.8-27B artifact whose text tower, head and token table come from PrismML's PQ2_0 GGUF, with the DFlash2 adapter's feature, output and MLP projections in Q4 (its fused query/key/value projection stays Q8); `--source mtp` takes the MTP head from a separately trained file; `--proposal` gathers the proposal head's rows from the imported ternary head in its own encoding | `tools/convert/ternary.py`, `tools/convert/sources/gguf.py`, `tools/convert/qwen3_5.py`, `tools/convert/proposal.py` |

Patches taken from TertiumOrganum1's fork of this line, with their authorship:

| area | behaviour | where it lives in the tree |
|---|---|---|
| KV cache | `rk4v4-e8`: keys rotated by H256 as in `rk8v4`, scaled per G64 group by absmax/7 and rounded per octet to the nearest E8 point, stored as int4 nibbles (the coset bit is not stored); values keep the packed int4 G32 plane of `rk8v4`. 280 bytes per token and KV head against 408. The E8-lattice KV codecs first appeared in UDPSendToFailed's NInfer-4090 (2026-08-16), hardened there by Daniel Parker | `core/paged_kv_storage.h`, `ops/kv_cache/int8_g64_codec.cuh`, `ops/kv_cache/append/`, the packed-key paths of `causal_cache/prompt_i8.cuh` and `small_t_i8.cuh`, `serve/serve_options.cpp` |
| ops | the T2 integer prefill route runs a 128x64 int8 tile: activations quantised per token and 128-K group, the int32 sum converted once per four MMAs; `NINFER_T2_A8_TILE=off` restores the 64-row kernel | `ops/linear/t2/t2_prefill_i8.cuh`, `ops/linear/t2/t2_a8.{h,cu}` |
| frontend | a region that opens a function but fails the strict reader goes through a recovery pass; readable calls are kept and the counters reach the parse diagnostics and both request logs | `models/qwen3_5/frontend/tool_call_parser.{h,cpp}`, `serve/{operational,request}_log.cpp` |
| context cache | a shared capture whose replacement releases more than assessed keeps the surplus; one that releases less is aborted like any other capture instead of failing the engine | `models/qwen3_5/program/transactions/capture.cpp` |
| build | `sm_120a` on the `mma.sync` compatibility path | the CUDA architecture lists |

Their fix fitting the GDN gating split-k to the real SM count is not carried: the base solved the
same launch failure in its own gating planner.

Ideas taken from UDPSendToFailed's NInfer-4090, re-implemented on this tree:

| area | behaviour | where it lives in the tree |
|---|---|---|
| build | `ninfer_core` and `ninfer_ops` compile whole-program: no relocatable device code and no device link (1,155 kernels with a stack frame become 927; the server binary grows from 498 to 637 MB); Matt Anderson's change in NInfer-4090 | `cmake/NinferTargets.cmake`, `docs/maintainer/build-system.md` |
| attention | the INT8-family small-T and prompt kernels read the query, key and value scales each lane needs from shared memory; the shuffles from a computed lane they replace compiled to out-of-line calls | `ops/softmax_attention/dense/causal_cache/{small_t_i8,prompt_i8}.cuh` |
| serve | the listening socket sets `TCP_NODELAY`, which accepted connections inherit | `serve/http_server.cpp` |
| ops | sigmoid and silu evaluate `ex2.approx` and a correctly rounded reciprocal; softplus keeps `expf`/`log1pf` unless the build sets `NINFER_SFU_SOFTPLUS`, which evaluates ln 2 · lg2(1 + 2^(x log2 e)) and takes the log1p series where e^x < 1/16 | `ops/common/math.cuh`, `CMakeLists.txt`, `tests/ops/test_gdn_gating.cpp` |
| attention | a small-T split whose keys span more than the 64 page IDs it stages in shared memory reads each page's physical index from the block table; the visible-key limit rises from 262,144 to 1,048,576 | `ops/softmax_attention/dense/causal_cache/small_t{,_bf16,_fp8,_i8,_k8v4,_nvfp4}.cuh`, `small_t.cu`, `include/ninfer/ops/softmax_attention.h` |
| model | positions up to four times `max_position_embeddings`; `--rope-yarn` applies YaRN at factor `max_context` / native (`--rope-yarn-factor` fixes it in `[1,4]` instead) as Hugging Face computes it (correction range from beta_fast 32 and beta_slow 1, attention factor 0.1 ln s + 1 on cos and sin) to the text and MTP layers; the DFlash adapter keeps plain RoPE | `ops/launcher/rope.cu`, `ops/kernel/rope.cuh`, `ops/wrapper/rope.cpp`, `models/qwen3_5/program/planning/startup.cpp`, `models/qwen3_5/execution/{attention,text}.cpp` |
| KV cache | `rk2v4-e8`: keys rotated and scaled per G64 group as in `rk4v4-e8`, each 8-dimension block stored as the nearest of E8's 240 roots plus a byte of 4-bit log-radius and signed residual axis (64-byte key rows); values keep `rk8v4`'s plane; 216 bytes per token and KV head. The read side expands a block to int8 codes in the INT8 layout, so QK stays the same s8 MMA | `ops/kv_cache/e8_root_codec.cuh`, `ops/kv_cache/int8_g64_codec.cuh` (`KvKeyCoding`), `ops/kv_cache/append/`, `causal_cache/{prompt_i8,small_t_i8}.cuh`, `core/paged_kv_storage.h` |
| core | Windows builds with `NINFER_D3D12_RESIDENCY`: `--wddm-evictable-budget` allocates owning device arenas from a shared D3D12 heap made resident at maximum priority (over-budget denied) and imported into CUDA, checked by a read-back, released with its arena; KV is planned against the adapter's memory less the weights and a 512 MiB desktop floor; keylimesoda's opt-in and round-trip check are part of it | `core/arena.{h,cu}`, `runtime/engine/model_instance.cpp` |
| core | `--disk-kv-directstorage` (Windows builds with `NINFER_DIRECTSTORAGE`): the disk tier's restores read a staging batch of claimed pages through one DirectStorage queue into host memory, check each CRC, and fall back to mapped reads when a batch fails or does not complete in 10 s; `dstorage.dll` is delay-loaded | `core/direct_storage_reader.{h,cpp}`, `core/disk_kv_bridge.cpp` (`restore_pages_direct`), `core/disk_kv_store.cpp` (`claim_read`), `CMakeLists.txt`, `apps/CMakeLists.txt` |
| serve | a server default reasoning effort for requests that name none | `serve/serve_options.cpp`, `serve/translate.cpp` |
| build | `NINFER_NVCC_SPLIT_COMPILE` (`--split-compile N`) and `NINFER_PTXAS_VERBOSE` | `CMakeLists.txt` |
| speculative | MTP draft windows up to 15: the frame domain, the envelope array, the round transition and every guard take K up to 15 | `models/qwen3_5/program/round_buffers.h`, `internal.h`, `product/speculative_options.h` |
| serve | `/metrics`, `/slots` and `/props` and llama.cpp model discovery fields (`/metrics` and `/slots` by Sergiusz Michalik) | `serve/http_server.cpp`, `serve/openai_common.cpp` |
| serve | a WebUI compiled in from `NINFER_WEBUI_DIR`, served at `/` with the API base announced | `serve/webui/`, `serve/CMakeLists.txt`, `serve/http_server.cpp` |
| serve | output limits bounded only by the context when a request names none | `serve/serve_options.cpp`, `serve/openai_*_request.cpp` |
| ops | the block sampler keeps each thread's candidates in shared memory | `ops/kernel/sampling_device.cuh` |
| ops | `NINFER_BF16_RESIDUAL_ADD`: BF16 linear projections add onto the residual in bf16 | `ops/linear_add/bf16/` |
| ops | the chunked GDN prefill zeroes and writes its tiles with vector stores and skips the decay exponentials of strictly upper-triangular tiles; results are unchanged bit for bit | `ops/linear_attention/gated_delta_net/chunked/{output,prepare_wy_wu,state_passing}.cuh` |

Pull requests to [Neroued/ninfer](https://github.com/Neroued/ninfer), ported onto this tree with
their authorship:

| PR | behaviour | author |
|---|---|---|
| #54 | the exception that terminates the process is named | Aleksandr Iakimov |
| #97 | container builds keep incremental native compilation | Duncan Betts |
| #162, #163 | llama.cpp-compatible model metadata on `/v1/models`; timings during prompt progress | Héctor Ramón Jiménez |
| #197 | `ignore_eos` documented and pinned in the schema test | Thireus |
| #199 | sparse MoE keeps Q4 group quads in flight in the routed gate/up dot product and on the T = 1 path | Mykhailo Dementii |
| #264, #273 | a partial last M tile in the fused NVFP4 SwiGLU TMA route; the text profile of `rmsnorm_rope` | Mykhailo Dementii |
| #268 | the attention output gate applied by the small-T reduce epilogue on BF16 and INT8 caches, bit-identical to the separate multiply | Mykhailo Dementii |
| #286-#290 | NVFP4 sparse-MoE expert banks: one divisor per stacked source matrix in the artifact, an NVFP4 profile for decode and the small-token route (CUDA cores, every build) and for prefill (W4A4, every sm_120a build; sm_8x builds refuse the banks at bind), and the `qwen3_6_35b_a3b_nvfp4` recipe. The decode codec sits beside the row-split codecs instead of coming through #286's refactor, so the groupwise profiles compile to the same kernels | Mykhailo Dementii |
| #282 | GGUF (any ggml quant level) as a conversion source | giveen |
| #284 | the `qwen3_8_27b_q6` recipe with a fused Q6 gate/up shape and tuned Q6 dispatch | bingchengcc |
| #294 | structured output through xgrammar, speculative decoding included, opt-in with `--structured-output` | Andrey Shvartsman |
| #295 | `reasoning.summary` and `include: reasoning.encrypted_content` | Mac |
| #297 | layout state survives an overflow | Duncan Betts |
| #299 | a repeated tool-call parameter keeps its last value | adubkov |
| #304, #305, #307 | the inference speed-of-light estimator; attention RMSNorm fused with NVFP4 quantization; single-pass logsumexp for target log probabilities | Duncan Betts |
| #309 | quoted reasoning closes and later tool-call markers stay content; a first region that opens with a complete call goes to the recovery pass instead of yielding to a later marker | Fedor Suchkov |
| #311 | adjacent Q4/Q5 groups pipelined at T32 for the attention input | MOVIBALE |
| #316 | Copilot tool shapes, agent-host tool names up to 256 bytes, located tool-name rejections, an opt-in usage-chunk choice | Damian Sromek |
| #300 (via Wallawalla47) | the agent-harness XML tool-call forms: `<function name=>`, `<invoke>`, `<function_calls>`, `<param name=>` | pkochubey, Ian Ranson |

From other forks and upstream `master`:

| source | behaviour | where it lives in the tree |
|---|---|---|
| [IMGillusion/ninfer-disk-kv](https://github.com/IMGillusion/ninfer-disk-kv) | a disk (L3) tier under the Host tier: an evicted private continuation writes its KV prefix chain and its endpoint, rewrite and early anchor StateImages to per-family files keyed by the prefix digest (CRC-checked, LRU, surviving restarts); with `--disk-kv-restore` a request with no resident prefix is seeded from the longest restorable frontier | `core/disk_kv_{store,bridge}.{h,cpp}`, `models/qwen3_5/program/storage/disk_tier.cpp`, the spill sites in `transactions/materialization.cpp` and `storage/context.cpp`, the probe in `planning/request_plan.cpp`, the seed in `prefill.cpp` |
| IMGillusion | `--context-cache-policy rolling`: a capture that extends a resident the request matched exactly inherits that resident's demand within its cache session | `runtime/engine/context_cache/resource_manager.h` |
| IMGillusion | `--first-token-logprobs`: Chat Completions `top_logprobs` report the first generated token's log probability and alternatives from the logits copied before sampling | `core/token_logprobs.{h,cpp}`, `models/qwen3_5/execution/text.cpp`, `program/prefill.cpp`, `serve/openai_chat_{request,response}.cpp` |
| [MirkoCovizzi/ninfer-rtx5090-mobile](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile) | `--adaptive-mtp`: each MTP round verifies the width its controller picks from measured draft survival and measured round cost, with CUDA Graphs per width | `models/qwen3_5/program/speculative/mtp_adaptive.h`, `decode.cpp`, `graphs.cpp`, `round_buffers.cpp` (`verification_view`), `planning/graph_profiles.cpp` |
| [Wallawalla47/ninfer-custom](https://github.com/Wallawalla47/ninfer-custom) (Ian Ranson) | `--fast-prefill-kernel`: an INT8-G64 prompt attention kernel with FP16 PV per 64-key tile (this line extends it to `rk8v4` and the packed key codings and lets the device profile turn it on), and prefill chunks rounded to whole attention waves | `ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh`, `prompt.cu`, `planning/startup.cpp` |
| Wallawalla47 (Ian Ranson) | the engine worker recovers from an out-of-memory failure (David Oelfke's change in gzenz/ninfer, ported by Ian Ranson); `--kv-headroom-mib`, `--cuda-graph-allowance-mib`, `--thinking-budget-message` | `runtime/engine/engine_core.h`, `serve/serve_options.cpp` |
| [tmark00/ninfer](https://github.com/tmark00/ninfer) | `rk2v4-e8` root codes decoded from tables; the gated RMSNorm prefetch cutoff from the device's SM count; the WebUI's MCP traffic relayed behind `--webui-mcp-proxy`; the GDN projection test bound; LF chat templates and fixtures | `ops/kv_cache/e8_root_codec.cuh`, `ops/launcher/rmsnorm.cu`, `serve/mcp_proxy.{h,cpp}`, `.gitattributes` |
| pelebel, natpate | the server announces openable URLs, indexes the API base and echoes preflight headers | `serve/operational_log.cpp`, `serve/http_server.cpp`, `serve/openai_common.cpp` |
| Matt Anderson | registers for the wide Q5 split tiles and a two-row kernel from eight columns; an idle Engine fails only the request it cannot place | `ops/linear/q5/`, `runtime/engine/engine_core.h` |
| [Neroued/ninfer](https://github.com/Neroued/ninfer) `master` | Minnnn's column-band routing of the q4/q5 A16 input projections with one split4 Q5 parent; `NINFER_CUDA_SYNC` for the CUDA synchronization schedule (adubkov, Neroued), which stays at CUDA's default when unset; the accurate SiLU restored in the NVFP4 SwiGLU | `ops/attn_input_proj/q4_q5/`, `ops/gdn_input_proj/q4_q5/`, `core/device.cu`, `ops/linear_swiglu/nvfp4/` |

The September 27 sweep of the maintained forks (Wallawalla47, gzenz, IMGillusion, upstream pull
requests and `master`), each change re-applied on this tree and opt-in where it moves numbers:

| source | behaviour | where it lives in the tree |
|---|---|---|
| remesis, Ian Ranson (Wallawalla47) | n-gram copy drafting beside MTP, DFlash and DFlash2, on with 15 copied tokens whenever `--spec` is set: the copy family keeps its own verification width, neural rounds keep theirs; an optional RAM archive per `X-NInfer-Draft-Session` | `models/qwen3_5/program/ngram_proposer.h`, `speculative/lookup_draft.h`, `product/speculative_options.h`, `models/qwen3_5/ngram.{h,cpp}` |
| Ian Ranson (Wallawalla47) | the hybrid prefix cache (`--use-alt-prefix-caching`): content-addressed 64-token KV blocks shared across requests plus sparse state snapshots, one Host slab pool sized by `--host-cache-mib`; `--use-original-prefix-caching` names the default | `runtime/engine/context_cache/hybrid_resource_manager.h`, `models/qwen3_5/program/prefix/hybrid_*`, `docs/maintainer/hybrid-prefix-cache-spec.md` |
| Ian Ranson (Wallawalla47), Gideon Zenz | around the checkpoint catalog: `--recency-eviction` (recency ladder, demotion to Host first), `--kv-lease-growth` (an answer's Device KV lease grows on demand), `--host-cache-mib` (one Host budget), `--auto-long-anchors`, a seal-window claim for materialization and shared captures, least-recently-used replacement of automatic shared prefixes, the shared catalog sized for one request's candidates, salvage of what an aborted request prefilled | `runtime/engine/context_cache/{resource_manager,materialization_planner,seal_window_claim}.h`, `models/qwen3_5/program/transactions/`, `planning/startup.cpp` |
| Gideon Zenz | `--thorough-admission-search` (up to 250 ms and every candidate for a new request's plan), `--value-aware-demote` (eviction ranked by rebuild cost), `NINFER_MAT_DEBUG` traces | `runtime/engine/context_cache/{materialization_budget,materialization_planner}.h`, `models/qwen3_5/program/planning/pressure_value_ranking.h` |
| David Oelfke, Ian Ranson | `--concurrent-prefill`; worker recovery that re-arms admission, `--recover-invariant-failures` | `runtime/engine/engine_core.h` |
| Gideon Zenz | `--mtp-attention-window`: the MTP draft attends to its sink page and newest window through a rebound page table (`paged_kv_window_rows`), verification keeps full attention | `ops/{kernel,launcher,wrapper}/paged_kv_window*`, `models/qwen3_5/execution/text.cpp`, `planning/startup.cpp` |
| Gideon Zenz | post-thinking sampling: a thinking request's lane switches to its own preset once reasoning closes (`--post-thinking*`, a `post_thinking` request object) | `runtime/engine/{engine.cpp,engine_core.h}`, `models/qwen3_5/program/decode.cpp`, `models/qwen3_5/frontend/output_session.cpp`, `product/post_thinking_options.h`, `serve/` |
| Gideon Zenz | `GET /stats` with the waiting queue and `--stats-port`; `tools/monitor` dashboard and wedge watchdog; request-log rotation; `--derive-session-keys`; Anthropic `ping` events with each heartbeat; `--lenient-assistant-history`; a developer message retried as system when a template refuses the role; `tools/longctx_recall_probe.py` | `serve/{load_report,http_server,http_transport,translate,openai_responses_request}.cpp`, `models/qwen3_5/frontend/chat_template.cpp`, `tools/` |
| David Oelfke | CPU Vision (`--vision-residency cpu`): the tower in FP32 on host threads with packed token panels and key-blocked attention, no device Vision memory; `--rope-scaling-factor` and `--rope-scaling-original-context` interpolate positions past a threshold in the RoPE kernel | `models/qwen3_5/{load,execution}/vision_cpu.*`, `ops/kernel/rope.cuh`, `product/rope_yarn_options.h` |
| Ian Ranson (Wallawalla47) | `--assistant-prefill`, `--unconstrained-response-format`, grouped `--help`, the build id, `--log-colours`, `--log-stats-panel`, levelled diagnostics records, a native Windows build | `serve/`, `apps/`, `product/logging/`, `CMakeLists.txt` |
| Ian Ranson (Wallawalla47) | decode-graph kernels launched as programmatic dependents (`NINFER_PDL` on compatibility builds, always on native ones), split-KV attention for short prefill steps over long contexts, a runtime-shape BF16 GEMM fallback, MTP banks of mixed formats, measured CUDA Graph memory in `server_start` | `ops/`, `models/qwen3_5/program/graphs.cpp`, `core/device.cu` |
| Ian Ranson (Wallawalla47) | ModelOpt NVFP4/FP8 and Quasar NVFP4 conversion, the `grouped_mse` scale search | `tools/convert/` |
| MGS Creativa, IMGillusion, Alexey Dubkov, Duncan Betts, giveen | a Vision loan takes only pages no reservation needs; LRU disk-tier eviction and positioned I/O; a quoted parameter closer stays inside its value; the fused RMSNorm and NVFP4 attention input at every width; the shared catalog default | `runtime/engine/context_cache/`, `core/disk_kv_*`, `models/qwen3_5/frontend/tool_call_parser.cpp`, `ops/attn_input_proj/nvfp4/` |
| [Neroued/ninfer](https://github.com/Neroued/ninfer) `master` | the unified Q4, Q5, Q6 and Q8 A16 Linear templates with sliced-K schedules, beside this line's routes and kernels; each card class takes them only in the width bands where two sweeps on an RTX 3090, 4090 and 5090 measured them faster, and `NINFER_LINEAR_ROUTES=legacy\|unified` forces one table | `ops/linear/common/route_table.{h,cpp}`, `ops/linear/q{4,5,6,8}/` |
| [Neroued/ninfer](https://github.com/Neroued/ninfer) `master` | the unified FP8, NVFP4 and BF16 Linear templates, and upstream's moves of the Q4, Q5, Q8, FP8, NVFP4 and BF16 fused projections onto them (attention and GDN inputs with their conv forms, LinearAdd, SwiGLU, the Q8 pair, the top-k heads, the Q8 grouped convolution and context-KV materialization), compiled beside this line's routes in `ops::detail::unified`; the FP8, NVFP4 and BF16 Linear shapes take them per width through `LinearRouteFamily` bands and each fused Op through its `unified/<op>` device-profile key, only where measured faster on the card | `ops/linear/common/route_table.{h,cpp}`, `ops/linear/{fp8,nvfp4,bf16}/`, `ops/{attn_input_proj,gdn_input_proj,linear_add,linear_swiglu,linear_pair,linear_topk,context_kv_materialize,dynamic_grouped_conv}/` |
| [Neroued/ninfer](https://github.com/Neroued/ninfer) `master` | the two-stage GDN prefill: one pass normalizes Q and K and prepares the gates, then one FP32-state recurrence also writes the output, in place of the WY, state-passing and output kernels, from 16 tokens where the profile's `gdn_two_stage/h<value heads>` says so or `NINFER_GDN_TWO_STAGE=1` | `ops/linear_attention/gated_delta_net/two_stage/` |
| Gideon Zenz | PackGQA: each KV head's query heads packed into the INT8 prompt kernel's tiles, where the profile's `attn_pack_gqa` says so or `NINFER_PROMPT_PACK_GQA=1` | `ops/softmax_attention/dense/causal_cache/prompt.cu` |
| this line, found in the port | kernels set their dynamic shared memory once per device rather than once per process, so a second GPU no longer launches without it; a forwarded call stays inside upstream's domain (the FP8 GDN record through 16 columns, the NVFP4 snapshot's A16 schedules through 16) and reads the integer-A8 policies as A16, as this line's plans do; the NVFP4 record's small-T launch follows its snapshot's table; a workspace query sizes each width for the table that serves it; on sm_8x the BF16 TMA route falls back to the cp.async MMA of the same tile | `core/device.h`, `ops/linear/common/route_table.h`, `ops/*/fp8/`, `ops/*/nvfp4/`, `ops/linear/bf16/bf16_a16_tma_mma.cuh` |
| this line, found in the sweep | the FP8 small-T attention reduces its partials through the shared reducer (its own copy faulted on sm_120a, also on `master`); the GDN record kernel runs window by window past 16 columns; the stale private reclaim follows `--recency-eviction`'s order and counts the checkpoints it drops; the issue #251 reclaim spares the shared prefix a capture extends | `ops/softmax_attention/dense/causal_cache/small_t_fp8.cu`, `ops/linear_attention/gated_delta_net/recurrent.cuh`, `runtime/engine/context_cache/resource_manager.h` |

Assessed in this sweep and not taken: the per-search planning window restart, whose search the
thorough mode already bounds; the gzenz v2 engine's host-KV safety net, units, fit gate and state
leases, which this line's Host and disk tiers and eviction options cover; moving the turn-closure
checkpoint to the last stripped turn, which the next request can no longer reuse byte for byte;
an edit of the froggeric template's instructions; the QUASAR binding overrides and the weights
profile switch, which v3 artifacts make unnecessary; the small-T page-ID enlargement, covered by
the block-table fallback; Kimi Delta Attention, which has no model here.

The disk tier differs from IMGillusion's in one place: its abandon path released borrowed memory
while write tickets were still pending, which could store a page whose bytes changed under a valid
CRC; here every submitted ticket is awaited before an owner's memory goes. The Windows DirectStorage
reader is this line's own: NInfer-4090's copy-on-write snapshot store is built into its v0.6 runtime.

NInfer-4090 anchors its YaRN at 1,048,576 tokens, so everything shorter runs unscaled RoPE; this
line applies it at Qwen's documented factor instead.

Assessed and not taken:

- Wallawalla47's `--tolerant-tool-calls` recovery overlaps the recovery pass taken from Tertium,
  and its report of truncation ahead of a tool call would stop an agent harness from retrying the
  malformed call that recovery hands it.
- Wallawalla47's split-KV page floor, covered here by the block-table fallback; its lease work is
  taken above behind options.
- #292's Q5 K-split MMA and Q4 staging ring: the base carries its own small-T Q5 MMA and a
  multi-stage Q4 ring, with route boundaries re-measured on an RTX 3090; #292's are tuned on an
  RTX 5090.
- #167 (an FP8 A8 GEMM staged through TMA) predates upstream's move of the FP8 A8 launches into
  per-shape launchers, which the base carries; its schedules and cost model would have to be
  re-derived there, and the route exists only on native Blackwell builds.
- Already in the base or in this line under another name: #61's per-image Vision budget
  (`--vision-max-merged`), #152's shared-prefix candidate at the system/developer frontier (the
  Engine's structural candidate, kept enabled for OpenAI requests), #221's MTP topology classes (Mykhailo Dementii),
  #235's lower CUDA floor (12.8 here), the Windows builds of #59, #84 and #233, and #173's
  `rk2v4-e8` (Daniel Parker's upstream PR of the E8-root codec that UDPSendToFailed's NInfer-4090 carried first).
- #274 raises a context-cache default; `--max-shared-prefixes 7` gives the same capacity. #300 is
  an RFC bag whose items are in the base, taken above, or declined.
- The LRU catalog policy, the host-state byte budget, the context-trace diagnostics and the
  prefix-cache scenario battery of the earlier v0.10-based line. Measured in production, the
  branching policy did not help and sometimes hurt; the context-cache fixes above are what keep
  prefills from being triggered.

The ternary path is measured on one GPU; a ternary artifact split across devices by the base's
pipeline stages (`--devices A,B,...`) has not been run. The Windows-only options
(`NINFER_D3D12_RESIDENCY`, `NINFER_DIRECTSTORAGE`) pass a syntax check against Windows headers
and have not been run.

## Device profiles and the September 2026 performance round

| area | behaviour | where it lives in the tree |
|---|---|---|
| ops | device route profiles: per hardware class and SM count, the schedule each route key takes per width band, installed per CUDA device before any Op runs; `DeviceRouteForce` for calibration | `ops/common/device_route.{h,cpp}` |
| calibration | `calibrate_device_routes` times every route family through the inference dispatch on synthetic weights and caches, L2 flushed per sample; a candidate wins only 3 % ahead, again in a second interleaved round, and with output within 5 % of the compiled route | `calibration/device_calibration.{h,cu}`, `apps/calibrate/main.cpp` (`ninfer-calibrate`) |
| runtime | profiles come from the user's file, then the built-in table (`device_profiles.json`, embedded at configure time: RTX 3090, 4090, 5090, and the RTX PRO 6000 Blackwell Workstation, Max-Q and Server editions, each its own hardware class at 188 SMs), else a calibration at first start that is saved; `--device-profile auto\|off\|calibrate`, `--device-profile-path` | `runtime/engine/device_profile.{h,cpp}`, `device_profiles_builtin.cpp.in`, `model_instance.cpp`, `serve/serve_options.cpp` |
| runtime | compiled context-cost presets for the RTX PRO 6000 Workstation Edition: its transfer fit and the prefill fits of the groupwise-int 27B, groupwise-int 35B-A3B and NVFP4/FP8 27B signatures | `runtime/engine/context_cache/context_cost_defaults.cpp` |
| attention | INT8-family small-T tiers routed by profile: warps, CTAs per SM, key block, split QK across producer warps (`q`), the next tile's codes and scales staged in registers a whole iteration ahead (`e`), both (`qe`) | `ops/softmax_attention/dense/causal_cache/small_t_i8{.cuh,_launch.cuh}`, `small_t.cu` |
| attention | FP16 accumulation of P·V per key tile (small-T and INT8 prompt kernels), by profile (`attn_pv_f16`) or `NINFER_SMALLT_PV_F16` / `NINFER_PROMPT_PV_F16` | `small_t_i8.cuh`, `prompt_i8.cuh`, `ops/common/mma.cuh` |
| attention | the fast prompt kernel also serves `rk8v4` (packed int4 values decoded from byte-pair `ldmatrix.trans`) and the packed key codings (`rk4v4`, `rk4v4-e8`, `rk2v4-e8`, expanded into the stage's INT8 tile); on by profile (`attn_prompt_fast`), `NINFER_PROMPT_FAST`, or `--fast-prefill-kernel` | `prompt_i8_fast.cuh`, `prompt.cu` |
| planning | the prefill chunk is the multiple of 128 near the request whose prompt-attention grid leaves the least of a last wave idle on the device, for either prompt kernel (Ian Ranson's fast kernel rounded its own chunks down to whole waves); `NINFER_PREFILL_ALIGN=0` keeps the request | `causal_softmax_attention.cpp`, `models/qwen3_5/program/planning/startup.cpp` |
| ops | grids sized from the device: the chunked GDN output wave from occupancy, the sparse-MoE prefill cap from the SM count | `linear_attention/gated_delta_net/chunked/output.cu`, `sparse_moe/prefill/sparse_moe_prefill_kernels.cu` |
| ops | ternary small-T band up to 64 columns, with 16-row 32-column and 32-row 16-column K8 schedules | `ops/linear/t2/t2_a8.{h,cu}` |
| graphs | MTP draft windows past eight verify columns build one CUDA Graph executable per profile (Mykhailo Dementii's topology classes, #221, cover up to eight) | `models/qwen3_5/program/planning/graph_profiles.cpp` |
| graphs | MTP and one-token decode profiles share an executable only when every attention call they capture takes the same route, as `causal_softmax_attention_route_family` reports for the model's geometry and KV storage (BF16 takes the prompt kernel up to 128 keys); MTP class ids get a per-width stride of 64 | `models/qwen3_5/program/planning/graph_profiles.cpp`, `graphs.cpp`, `startup.cpp`, `ops/.../causal_softmax_attention.cpp` |
| build | the INT8 small-T launch compiles in 28 units (width, geometry, input) | `ops/softmax_attention/sources.cmake` |
| build | every `120a` build compiles the FP8 A8 and NVFP4 W4A4 units (`NINFER_SM120_FP8`, `NINFER_SM120_NVFP4`), the default compatibility path included, and FP8 and NVFP4 weights keep their stored A8 and A4 permissions there; before, an NVFP4 artifact failed at runtime planning on that path because the A16 SwiGLU route stops at 16 columns, and FP8 weights prefilled through their dequantizing A16 route | `CMakeLists.txt`, `ops/CMakeLists.txt`, `models/qwen3_5/load/prepare.cpp`, `ops/linear/fp8/fp8_launch.cuh`, `ops/linear/nvfp4/nvfp4_launch.cuh`, `ops/weight_input.cpp`, `ops/wrapper/sparse_moe.cpp` |

## Verification

- `ninfer_resource_manager_test` after every context-cache commit (it builds on a Mac without
  CUDA: Homebrew clang, `-include exception`, the Xcode SDK sysroot).
- `ninfer_tool_call_parser_test`, `ninfer_qwen3_5_frontend_test` (fixture tokenizer), the OpenAI,
  Responses and Anthropic schema tests cover the forced tool call.
- `ninfer_hadamard_transform_test`, `ninfer_linear_t2_a16_test`, `ninfer_linear_t2_a8_test` and the
  T2 full and indexed cases of `ninfer_linear_topk_test` check the ternary ops against FP64 oracles;
  the T2 cases of `ninfer_attn_input_proj_test`, `ninfer_gdn_input_proj_test` and
  `ninfer_gdn_input_proj_conv_record_test` cover the pairs under both policies;
  `ninfer_linear_dynamic_grouped_conv_add_test` runs Q8, Q4 and Q5 projections and
  `ninfer_linear_q4_a16_test` the adapter's Q4 shapes; `ninfer_softmax_attention_test` runs the
  INT8-family small-T cases at the capped split counts; `tests/convert/test_ternary.py` covers the
  GGUF mapping and `tests/convert/test_proposal.py` the exact proposal rows.
- `ninfer_softmax_attention_test --rk4v4-e8-only`, the exact key encoding in
  `ninfer_kv_cache_append_test` and `ninfer_tool_call_parser_test` cover the taken patches.
- `ninfer_disk_kv_store_test` and `ninfer_disk_kv_bridge_test` (sanitizer-clean on a Mac without
  CUDA), `ninfer_softmax_attention_test --int8-prompt-only` (both INT8 prompt kernels),
  `ninfer_qwen3_5_mtp_adaptive_test`, `ninfer_token_logprobs_test`, `ninfer_e8_root_decode_test`
  and the serve option and schema tests cover the fork ports.
- `ninfer_sparse_moe_test` walks the NVFP4 profile at T = 1, 2 and 12 on sm_8x and from T = 1 to
  4097 on an sm_120a build; `ninfer_gdn_replay_fold_test` folds records packed at a narrower
  width than planned; `ninfer_mtp_round_test` prepares proposals wider than the verification; the
  batch cases of `ninfer_softmax_attention_test` compare the gated output byte for byte with the
  separate multiply.
- On an RTX 3090 (sm_86) before publishing: the full build and `ctest` (169 tests, eight of them
  real-model tests that skip without an artifact), the artifact and convert pytest suites, the
  real-weight loading tests with the Bonsai and Huihui artifacts under MTP and DFlash2, both
  artifacts served with the sanity and acceptance suites (Bonsai under MTP, DFlash2 and without
  speculation, Huihui under MTP; Huihui with DFlash2 does not fit a 198,400-token cache on 24 GB),
  structured output under MTP and DFlash2, adaptive MTP against fixed windows, and the disk tier
  end to end under MTP, DFlash2 and without speculation (a restored prefix answers exactly as the
  resident one did).
- This round, on the final code: the full `ctest` (170 tests with `ninfer_device_profile_test`) on
  an RTX 3090 (`sm_86`), RTX 4090 (`sm_89`) and RTX 5090 (`sm_120a`); the real-model tests with the
  Bonsai and Qwen3.8 artifacts (the MoE and DFlash v1 tests need other artifacts, and the prefix test
  has no prompt golden for the Bonsai template); each card started with no profile file, took its
  built-in profile and wrote none; `ninfer_qwen3_5_mtp_graph_profiles_test` over both geometries,
  every storage and the one-token profiles, and 21 server starts per card across KV storages, MTP
  widths 3 to 15 (adaptive included), DFlash2, four lanes and small BF16 windows; four 60,000-token
  needle documents at once returned all twelve codes on an RTX 5090.
- The September 27 sweep, on the final code: the full `ctest` on an RTX 3090 (held at 220 W), RTX
  4090 and RTX 5090 with the Qwen3.8 artifact, where only the MoE and DFlash v1 real-model tests
  fail, for want of their artifacts; on the RTX 5090, greedy answers byte-identical to `master` for
  prose, code, a repeated list and Chinese, with and without MTP, and the same 33-token natural stop
  in the stop-chat copy; n-gram copy drafting left prose decode where it was (48.9 against 49.1
  tok/s on the 3090, 96.7 against 96.6 on the 4090, 159.6 against 159.5 on the 5090, under MTP;
  unchanged under DFlash2) and copied code about three times as fast under MTP and twice as fast
  under DFlash2 on every card; `--mtp-attention-window` 1024 and 256 answered as full attention
  did; five serve checks per card; CUDA Graphs stayed within their allowance (3.5 to 17% of it on
  the RTX 5090, 9 to 67% on the 3090 and 4090); the `NINFER_PDL` build answered as the default
  build at the same speed on the RTX 5090. The Linear route bands come from two sweeps per card of
  every shape with both tables, and a third sweep of the default routes.
- The rest of Neroued's wave and PackGQA, on the final code: the full `ctest` (257 tests) on an RTX
  3090, 4090 and 5090 with the Qwen3.8 artifact, where again only the MoE and DFlash v1 real-model
  tests fail, for want of their artifacts; before the bands, the suites the ports touch under both
  tables on each card, which is where the domain, policy, workspace and record issues above showed;
  every fused Op benchmark and every FP8, NVFP4 and BF16 shape under both tables in two rounds of
  opposite order per card (the 4090's host held it at 300 to 450 W, the 5090's at 450 W); the
  calibration of the unified Q4/Q5 switches, the two-stage GDN prefill and PackGQA; and Engine
  prefill of the Qwen3.8 artifact on the 3090 with the two-stage GDN off and on, 1,604 and 1,607
  tok/s at 4K and 1,556 and 1,576 at 16K, with MTP decode unchanged at 35.4 tok/s. Greedy answers
  with each card's built-in routes against the same build forced onto the legacy tables with the
  two-stage prefill off matched byte for byte in 5, 3 and 4 of 8 cases on the 3090, 4090 and 5090
  (the repeated file on every card) and elsewhere parted at a near tie into equally coherent text.
- sm_120a: on an RTX 5090 both the default compatibility build and the native build
  (`NINFER_SM120_NATIVE`) pass `ctest` (169 tests), and the native build converts and serves the
  RedHatAI Qwen3.6-35B-A3B NVFP4 checkpoint.
- RTX PRO 6000 Blackwell (188 SMs), default `120a` build: the full `ctest` (171 tests; the FP8 A8
  and NVFP4 A4 cases run, and only two cases that need an artifact path skip); the real-model tests
  with the Qwen3.8 groupwise and NVFP4/FP8 artifacts and the 35B-A3B groupwise and NVFP4
  artifacts; greedy answers of the Qwen3.8, Ternary Bonsai 2, GSQ-RCO and 35B-A3B artifacts
  without speculation, under MTP and DFlash2 and with Vision, byte-identical to the build before
  the FP8 and NVFP4 changes, while the Qwen3.8 NVFP4/FP8 artifact, which did not start before,
  answers in all four modes and prefills 4,096 tokens at 11,822 tok/s, within 1.4% of a native
  build of the same commit. The RedHatAI Qwen3.6-35B-A3B
  NVFP4 checkpoint converts and serves on this build: quick-corpus perplexity 4.410 against 4.364
  for the groupwise artifact, scored at 15,080 against 11,266 tok/s. Each edition (Workstation at
  600 W, Max-Q at 300 W, Server at 600 W) was calibrated twice, and on the Workstation edition a
  start with no profile file took the built-in profile and wrote none, and a server resolves the
  compiled context-cost presets.

## Deployment

Deployment files are not part of this public line. The production checkout adds them on a
private branch on top of `master`.
