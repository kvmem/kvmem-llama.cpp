# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype int8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone, and two requests run concurrently when their complete reservations fit.

With `C=2` and two extra Device checkpoint slots, the process owns two active StateImage guarantees
plus a global pool of two Device-resident checkpoints. Eight pinned Host State slots and 8 GiB of
pinned Host KV retain inactive continuations under Device pressure. Active request capacity is two.

`--host-state-slots` and `--host-kv-mib` are independent allocations, so their sum — not
`--host-kv-mib` alone — is the Host RAM the process pins: a Host StateImage costs one
`memory.host_state_image_bytes` regardless of the prefix depth it resumes. `--host-cache-mib N`
replaces both with a single ceiling: the engine sizes the Host state pool from the checkpoint
inventory the capture path creates, `(2 + max-long-anchors-per-continuation) * private
continuations + shared prefixes` images, then, with `--auto-long-anchors` and a nonzero anchor
count, spends the remaining state headroom under the half-budget cap on **more long anchors per
continuation** — up to the count whose re-prefill gap still outweighs one StateImage — re-sizes the
pool for the grown count, gives Host KV the remaining bytes, and refuses to start if that state
footprint would exceed half the budget. A budget therefore never lowers the configured anchor
count, and the count it resolves is the one the frontend grid, the ResourceManager and the Program
all use; without automatic anchoring only client markers create anchors, so the count stays as
configured. Both unit costs and the derived split are reported in the `server_start` memory
ledger.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `--vision`; each accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.

### Adaptive MTP

With `--adaptive-mtp`, `--draft-tokens K` is the most drafts an MTP round verifies; every round
still proposes `K` for the next one. Each request tracks how far its drafts survive, position by
position, with a fast and a slow moving average; the Engine prices each verification width from
the round times it has measured at that width and batch size (a width not yet run is priced from
a relative cost shape scaled by the widths that have) and picks the width with the most expected
committed tokens per second over this round and the next. A width changes only after it has won
several rounds, and a width is probed upward when every row's drafts have kept surviving. While
three drafts are ready the width stays at three or more, so the Engine captures CUDA Graphs for
widths `min(3, K)..K`: the graph memory, and the startup capture time, grow with the number of
widths. Committed tokens are still the target's greedy choices, but attention splits its
reduction by the verify width, so a near-tie can resolve differently at another width, as it does
between two fixed `--draft-tokens` settings. The per-request log reports the rounds run at each
width and the width changes.

### MTP attention window

Each MTP draft step attends over the whole history of its cache, so at a long context the draft
reads as many keys as one target attention layer, once per draft. With `--mtp-attention-window N`
the draft head attends to the first 64 keys and the newest `N` (at least `--draft-tokens + 1`,
rounded out to whole 64-key pages) before its first query, which keeps its read constant as the
context grows. Only the drafts see the window: target verification attends to everything and
alone decides which tokens are committed, so greedy output is unchanged and only the acceptance
rate can move. The Engine rebinds the draft's page table to those pages and shifts its positions,
so the attention kernels are the unwindowed ones. `server_start` records the window.

### Vision residency

`--vision` keeps the Vision tower, its encode workspace and the item handoff resident for the
process lifetime. `--vision-residency overlay` removes that cost on memory-tight cards: the tower
lives in pinned host memory, the sequence plan reserves nothing for Vision, and each image is
encoded inside a bounded window whose device memory is borrowed for the duration of the encode.

A window is funded from free KV pages when they cover it. Those pages hold nothing, so nothing is
copied, the text weights stay mapped, and the encode runs on its own stream while other lanes keep
decoding: the lane that owns the image simply yields its prefill units until the encode completes.
The pages are out of circulation while the loan is open, so the admission capacity shrinks with it
and no request is ever admitted into memory that has been lent away.

When free pages cannot cover the window, it falls back to the evict-ranked tail of the text weights
(lm_head, token embedding, draft head, MTP head), restored from a pinned mirror before the prefill
unit ends. That window is exclusive: the borrowed weights are unmapped, so nothing else runs until
it closes. The fallback is always available, so a vision request never fails for lack of memory,
and a KV cache smaller than one window simply uses it for every image.

Either way the window streams the tower through two layer slots, lands the embeddings in pinned
memory, and the prefill then uploads only each chunk's embedding columns. A follow-up turn over the
same image reuses the prefix and opens no window. Embeddings are produced by the same kernels on the
same bytes, so completions are identical between residencies. `--vision-max-merged N` bounds one
item's merged tokens (media above it is downscaled at preprocessing) and sizes the window. The
request log line reports `overlay=<windows>x<conc|excl|mixed> <ms> (evict <MiB> <ms>, restore <ms>,
staged <MiB>)` and the JSON record carries `vision_overlay`, including `exclusive_windows`.

`--vision-residency cpu` (or `--vision-cpu`, which also sets `--vision`) keeps no Vision memory on
the device at all: the tower is decoded to FP32 host memory at load (about 1.7 GiB for the Qwen3.6
tower) and each image is encoded on CPU threads, then uploaded a chunk's columns at a time like an
overlay result. The encode of the next image runs beside other lanes' decode, and the lane that
owns it yields its prefill units until it completes, but it is slow: on a 12-core desktop CPU a
512x512 image takes seconds, and the cost grows with the square of the patches, so this residency
caps `--vision-max-merged` at 256 unless it is given. One encode runs at a time across the process.
The CPU encoder computes in FP32 where the device encoder rounds activations to BF16, so embeddings
agree closely but not bit for bit. It reads the tower's weights in the official artifacts' grouped
formats as well as BF16, FP8 and NVFP4; a projection stored with a Hadamard rotation or an input
gather is refused at load.

## Structured output

NInfer uses the vendored XGrammar v0.2.7 C++ compiler and token matcher to constrain final
response content. This requires no model conversion or additional weights. Ordinary decoding,
MTP, DFlash, and DFlash2 share the same target sampling contract, including CUDA Graphs,
streaming, prefix reuse, and mixed constrained/unconstrained concurrent requests.

It is opt-in: start the server with `--structured-output` (the CLI's `--json` and `--json-schema`
enable it for their run). That reserves one grammar mask plane per draft position and lane, and a
DFlash or DFlash2 round then carries a host grammar stage even when no request is constrained.
Without it the server keeps the default path unchanged and refuses a constrained request with HTTP
400 `response_format_not_supported`; with `--unconstrained-response-format` it generates such a
request unconstrained instead, for clients that send a JSON format on every request and parse the
answer themselves.

Chat Completions accepts:

```json
{
  "model": "your-model-id",
  "messages": [{"role": "user", "content": "Give the city and temperature as JSON."}],
  "max_tokens": 128,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "temperature": {"type": "number"}
        },
        "required": ["city", "temperature"],
        "additionalProperties": false
      }
    }
  }
}
```

Use `response_format: {"type":"json_object"}` for any JSON object. JSON Schema mode can also
constrain other root types. Responses uses the flat form
`text.format: {"type":"json_schema","name":"weather","strict":true,"schema":{...}}` and
reports that format in its Response object. Anthropic Messages accepts
`output_config.format: {"type":"json_schema","schema":{...}}`.

Supported schema constraints are `type`, `properties`, `required`, `additionalProperties`,
`items`, `prefixItems`, `minItems`, `maxItems`, `minLength`, `maxLength`, `enum`, `const`,
`anyOf`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `$defs`, `definitions`,
and local fragment `$ref` (including recursive schemas).
Annotations `$schema`, `title`, `description`, `default`, `examples`, and `$comment`
do not impose generation constraints. Other keywords are rejected with HTTP 400: this includes
`multipleOf`, `pattern`, `format`, `oneOf`, `allOf`, `uniqueItems`, and conditionals.
Numeric bounds require an explicit `integer` or `number` type and finite values within
`+/- (2^53-1)`. Integer bounds must be whole numbers. Bounded numbers use ordinary decimal
notation with at most six fractional digits; a range with no representable value is rejected.
`$id` and external references are rejected. An explicit `$schema` must be JSON Schema 2020-12
or draft-07. Local references use `#` or literal object paths such as `#/$defs/node`;
escaped or empty path segments are rejected. Bounded strings use unescaped Unicode characters; escaped quotes, backslashes,
and control characters are excluded from that generated subset. Nonnegative length/item bounds
must fit a signed 32-bit integer. `$ref` and `anyOf` cannot have sibling constraints; `const`
and `enum` allow a matching single `type` declaration but no other sibling constraints. Move
constraints into the referenced schema or each union branch. Missing `additionalProperties`
and `items` retain JSON Schema defaults. Properties are emitted in schema declaration order;
this is a valid subset of the requested schema. Whitespace between JSON elements is bounded to
eight characters per run to prevent whitespace-only generation loops. Additional-property key spellings are restricted
where necessary to prevent escaped aliases from overwriting declared typed properties.
`strict:false` does not disable enforcement.

Structured responses default to thinking disabled when no reasoning mode or budget is requested.
Explicit thinking and thinking budgets are supported: reasoning is returned in its normal channel,
and the first reasoning-close delimiter transitions to constrained final content. Active tools are
also supported. After reasoning, a response can emit native Qwen tool calls or the final JSON value.
The schema applies to final content, not reasoning or tool arguments. Tool responses have their normal
tool-call finish reason and need not contain a JSON content value. Tool definitions and thinking
settings may remain enabled on later requests after tool results are supplied.

Serving also exposes the requested format and schema to the model as a leading instruction,
asking for raw JSON without Markdown fences and retaining tools for information gathering.
This helps the model choose the intended final-answer branch when tools remain enabled; the
token grammar enforces validity independently. Tool selection and termination remain model
decisions under `tool_choice:"auto"`; use `tool_choice:"none"` when no further tools are needed.
Requests without an explicit response format receive no format instruction or grammar.

The native tool envelope restricts calls to declared names and emits each declared parameter at most
once, in the declaration order shown in the prompt. Parameter values remain non-strict; delimiter spellings that would make
Qwen XML ambiguous are excluded. Strict tool argument generation and arbitrary grammar/regex aliases
remain unsupported. Custom stops and assistant-prefill continuation are rejected with structured
output. Public C++ callers select `ExecutionOptions::structured_output` and use default stops and
decoded text output.

Only normal completed final-content responses guarantee a complete JSON document satisfying the supported
schema. Token/context limits, cancellation, transport failure, or generation errors can leave a
partial document; inspect the finish reason or Responses status before parsing it as complete.
SSE content deltas are ordinary partial JSON bytes; concatenate them before parsing. Constraints
guarantee format, not factual accuracy or semantic task success.

Masks are applied before target top-k/top-p/min-p filtering at **every** speculative position,
including correction and bonus tokens. Unconstrained draft distributions remain unchanged;
accept/reject and residual sampling use the constrained target distribution. Per-request grammar
state advances only with Engine output commit, and is never restored from a prompt/KV cache.
DFlash/DFlash2 CUDA Graphs contain a host matcher node between draft generation and verification;
this adds a CPU synchronization point and mask transfers per round. No speedup claim is implied.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/load` | serving capacity, current load, and monotonic token counters (see [Load](#load)) |
| `GET /metrics` | Prometheus text with llama.cpp's `--metrics` series plus NInfer's (see [Metrics](#metrics)) |
| `GET /stats` | the `/v1/load` snapshot plus the ingress peak and every Engine counter since startup (see [Stats](#stats)) |
| `GET /slots` | llama.cpp-shaped lane table: one entry per lane, the first `running` marked processing |
| `GET /props` | llama.cpp-shaped server properties: default sampling, context, lanes, modalities |
| `GET /v1` | endpoint index for the announced API base: the model alias and this table |
| `GET /v1/models` | configured OpenAI model alias, effective `max_model_len` (also as `context_window`), whether it accepts images, and a llama.cpp-compatible `meta` object (see [Models](#models)) |
| `GET /v1/models/{id}` | lookup of the configured alias with the same fields |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. Anthropic Messages
follows each comment with the protocol's `ping` event, which clients skip: a client that abandons a
request whose first event is late does not count comments, so a queued or long-prefilling request
would otherwise be dropped by it. On Linux, accepted
connections also use TCP keepalive and a 15-second `TCP_USER_TIMEOUT`; together with the heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

The HTTP transport is thread-per-connection, and a worker stays with a connection for its whole
life, including the idle window between keep-alive requests. The pool therefore reserves one base
worker per admissible request (`max_concurrency + max_pending_requests + 1`) and grows on demand up
to 64 further workers for connections that are merely open; the extra workers retire once idle.
Without that headroom a client-side connection pool of otherwise idle sockets occupies every worker
and the server accepts real requests strictly one at a time.

### Startup readiness

The server binds its port and starts accepting connections before the Engine has loaded weights
and finished warmup, so a port clash is reported in milliseconds rather than after loading. Every
route -- including `/health`, `OPTIONS`, and requests that would otherwise be unauthenticated --
answers `503` with `Retry-After: 2` until warmup completes, in the target's own error shape:

```json
{"error":{"message":"The model is still loading. Retry shortly.","type":"service_unavailable","code":"model_loading"}}
```

`POST /v1/messages` and `/v1/messages/count_tokens` receive the Anthropic envelope instead, with a
`request_id` field and a `request-id` header, matching every other error response on those
endpoints:

```json
{"type":"error","error":{"type":"api_error","message":"The model is still loading. Retry shortly."},"request_id":"req_..."}
```

A readiness probe should poll `GET /health` (or any endpoint) and expect `503` until the model is
ready rather than treating an accepted TCP connection as a signal of readiness.

### Load

`GET /v1/load` is a cheap, pollable snapshot for load balancers and gateways that schedule across
several servers. It requires the API key when one is configured, answers `503 model_loading` until
warmup completes like every other route, and reads only the Engine's already-published runtime
counters and the ingress count, so polling it does not wait on or delay the GPU executor.

```bash
curl http://127.0.0.1:8080/v1/load -H 'Authorization: Bearer local-secret'
```

```json
{
  "object": "ninfer.load",
  "model": "qwen3.8-27b",
  "uptime_seconds": 812.4,
  "capacity": {"max_concurrency": 4, "max_pending_requests": 32, "max_admitted_requests": 36,
               "max_context": 65536, "kv_capacity_tokens": 131072, "kv_capacity_pages": 2048,
               "kv_page_tokens": 64, "device_state_slots": 8, "host_state_slots": 24,
               "host_kv_bytes": 17179869184},
  "requests": {"admitted": 6, "running": 4, "prefilling": 1, "decode_ready": 3, "waiting": 2,
               "materializing": 0},
  "occupancy": {"device_main_kv_pages": 1500, "device_main_kv_tokens": 96000,
                "device_state_slots": 5, "host_state_slots": 7, "host_kv_bytes": 2147483648},
  "counters": {"computed_prefill_tokens": 48200113, "committed_decode_tokens": 6120452,
               "reused_prompt_tokens": 30911840, "decode_rounds": 861307,
               "decode_row_rounds": 2448180}
}
```

- `uptime_seconds` counts from the moment the server became ready, not from process start.
- `capacity` is fixed once the Engine is ready. `max_admitted_requests` is
  `max_concurrency + max_pending_requests`, the ingress bound described in
  [Execution behavior](#execution-behavior); `kv_capacity_tokens` is the resolved page-aligned Main
  KV pool and `kv_page_tokens` its page size.
- `requests.admitted` counts requests holding ingress capacity, from preparation until the response
  is released; a new generation request is rejected with `server_overloaded` (HTTP 429, or 529 on
  Anthropic endpoints) when it would exceed `max_admitted_requests`. `running` counts occupied execution lanes (at most
  `max_concurrency`), of which `prefilling` is the lane that owns the staged prefill and
  `decode_ready` the lanes in the decode batch. `waiting` counts requests submitted to the Engine
  FIFO that have not been admitted to a lane, including those held back until their KV entitlement
  fits.
- `occupancy` reports current Main KV pages (and tokens), Device and Host StateImage slots, and Host
  KV bytes in use. Main KV occupancy includes retained reusable prefixes, which the resource planner
  may evict under pressure, so a full pool does not by itself mean new requests will wait.
- `counters` are monotonic since startup; derive rates by differencing two polls.
  `computed_prefill_tokens` excludes prefix-reused prompt tokens (reported separately in
  `reused_prompt_tokens`). `committed_decode_tokens` counts tokens committed by decode rounds and
  excludes each request's first token, which prefill emits; with speculative decoding a round commits
  several tokens per row. `decode_row_rounds` is the sum of decode batch sizes over `decode_rounds`.
- Gauges and counters come from the snapshot the Engine publishes at execution boundaries, so they
  can trail the instant of the poll by up to one boundary.

### Stats

`GET /stats` is the `/v1/load` report (`object` `ninfer.stats`) with `requests.peak_admitted`, the
most requests admitted at once since startup, and the Engine's counters since startup in the shape of
the request log's `throughput` record (see [Structured request log](#structured-request-log)):
`tokens`, `throughput_tokens_per_second` averaged over the uptime, `scheduler`, `decode_batch`,
`host_work` and the `context_cache` counters and gauges. `queue` holds `depth`, the number of
requests waiting for admission, and `entries`: the first 16 of them in submission order, each with
its Engine `request_id`, `position` and `wait_seconds`, refreshed at least once a second while
requests wait. It needs the API key like `/v1/load` and, like it, reads only published snapshots.
Dashboards poll it, or the same route on `--stats-port`.

`memory.cuda_residency` contains the Engine worker's cached residency observation. Reading
`/stats` does not query CUDA or Windows performance counters and does not acquire the execution
lock. It has `enabled` and `verified` booleans and these byte counters:
`cuda_free_bytes`, `cuda_total_bytes`, `dedicated_bytes`, `shared_bytes`,
`shared_baseline_bytes`, `device_allocated_bytes`, `host_pool_bytes`,
`host_used_bytes`, and `verified_reserve_bytes`. The startup JSONL record includes the
same object under `memory.cuda_residency`, plus the selected policy, reserve and probe step in
its `engine` object. When `enabled` is false, zero counters mean monitoring is disabled;
they do not prove zero memory use. `verified` reports the latest admission/guard state,
not permanent physical pinning.

### Metrics

`GET /metrics` renders the same snapshot as `/v1/load`, plus totals accumulated from completed
requests, in the Prometheus text format. The `llamacpp:` series keep the names and meanings of
llama.cpp's `--metrics`, so dashboards and autoscalers built for it read this server unchanged:
`prompt_tokens_total` counts prompt tokens evaluated by prefill (reused prefixes excluded),
`prompt_seconds_total` and `tokens_predicted_seconds_total` the prefill and decode time of completed
requests, `tokens_predicted_total` their generated tokens, `n_decode_total` and
`n_busy_slots_per_decode` the decode rounds and their average batch, `kv_cache_usage_ratio` and
`kv_cache_tokens` the device KV occupancy, and `requests_processing` / `requests_deferred` the
running and waiting requests. The `ninfer:` series add completed requests, admitted requests,
prefix-cache hits, context-cache reuse, speculative drafts and acceptances, context-cache
exhaustions and uptime.

### WebUI

A build configured with `-DNINFER_WEBUI_DIR=<dir>` compiles the files of a built WebUI into
`ninfer-serve` and serves them on `GET /`; any other `GET` path outside the API routes returns the
entry page, for the UI's client-side router. The static files need no API key, the API routes it
calls still do. `--no-webui` turns it off at startup. The build never downloads anything:
`scripts/fetch-webui.sh [tag]` unpacks one of llama.cpp's released WebUIs, which speaks this
server's `/props`, `/slots`, `/v1/models` and streamed Chat Completions, and prints the directory to
configure with.

```bash
./scripts/fetch-webui.sh
cmake -S . -B build -DNINFER_WEBUI_DIR=<the directory it printed>
```

The WebUI's MCP client cannot reach an MCP server from the browser: the Streamable HTTP transports
answer POST from a raw writer without CORS headers. `--webui-mcp-proxy` serves the `/cors-proxy`
relay that the WebUI's "Use llama-server proxy" option targets (the target in the `url` query
parameter, its headers prefixed with `x-llama-server-proxy-header-`) and reports
`cors_proxy_enabled` in `/props`. The relay streams, forwards to any `http://` host it is given
(`https://` is refused: there is no TLS client), and carries no API key, because the WebUI prefixes
its own `Authorization` along with every other header; enable it only on a trusted bind address.

### Models

`GET /v1/models` and `GET /v1/models/{id}` return the configured public OpenAI model alias
(defaults to the artifact `metadata.name`, overridable with `--model-id`) together with the
effective `max_model_len` (the `--max-context` ceiling, also as `context_window`), `modalities.vision`
(whether `--vision` accepts images), and a `meta` object in the shape exposed by `llama.cpp`. The `meta` facts describe the registered artifact behind the alias:

| Field | Meaning |
|---|---|
| `n_vocab` | tokenizer token domain |
| `n_ctx` | configured per-request context ceiling (equal to `max_model_len`) |
| `n_ctx_train` | model native/training context |
| `n_embd` | model embedding width |
| `n_params` | total logical weight elements across the registered artifact tensors |
| `size` | encoded weight payload bytes of the registered artifact |
| `ftype` | encoded tensor formats of the artifact, joined with `+` when mixed |

`GET /v1/models/{id}` returns the same object for the single configured alias and a `404` for any
other id.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image detail is omitted or `auto`;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation, and llama.cpp's `-1` generates until the context runs out;
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` (`0..20`) and `min_p` (`0..1`) sampler extensions;
- the `post_thinking` extension object (see [post-thinking sampling](#post-thinking-sampling));
- up to four non-empty stop strings, applied to both reasoning and answer output;
- the `ignore_eos` benchmarking extension shared with vLLM, SGLang and llama.cpp: a boolean,
  default `false`, that drops the checkpoint's own stop tokens so generation runs to the output
  budget (`finish_reason:"length"`) unless a caller-supplied stop string or context capacity ends
  it first. Caller stop strings still apply; any other value type is rejected with 400. Output past
  the model's natural end is not meaningful text. `/v1/responses` and `/v1/messages` do not
  honor it;
- `n:1`, text-only `modalities`, and `response_format` with `text`, `json_object`, or `json_schema`;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`, optionally shaped by `--usage-chunk-choice` for strict client
  parsers;
- function tools and free-form `custom` tools, the latter served to the model as a
  single-string-input function under the caller's own tool name so callers that dispatch by name
  keep working;
- `tool_choice` `auto`, `none`, a named function, `required` over a single callable tool, or
  `allowed_tools` in either mode, which force that call; a `custom` selection is accepted as
  advisory narrowing;
- `strict:true` as an advisory flag, since decoding is not constrained to the schema, and
  `parallel_tool_calls:false`, honored by keeping the first call;
- assistant tool-call history, tool-result messages, and legacy function-call history;
- the top-level `reasoning_effort` field, where the `default` and `auto` aliases resolve to the
  server-configured level;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes JSON constrained output, nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, `required` tool choice over several callable tools, explicit
low/high image detail, web search, moderation, low/high verbosity, stored Chat Completions, and
non-empty legacy `functions`.

behavior. This includes nonzero `logit_bias`, requested log probabilities,
audio/file input or audio output, tool `strict:true`, required or named tool choice,
`parallel_tool_calls:false` with enabled tools, explicit low/high image detail, web search,
moderation, low/high verbosity, stored Chat Completions, and non-empty legacy `functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.
Known constrained-decoding aliases (`grammar`, `structured_outputs`, `guided_json`, `guided_regex`,
`guided_choice`, and `guided_grammar`) receive the same explicit rejection instead of being treated
as unknown hints.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `top_logprobs:0`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on a `tool` message is accepted as an ignored, output-neutral compatibility
extension for clients that mirror the function name onto tool results. It does not participate in
tool identity, prompt rendering, or output. Non-string values are malformed; non-empty names on
other message roles remain unsupported because they carry participant identity that the loaded chat
template cannot represent.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

Across Chat Completions, Responses, and Anthropic Messages, an explicit top-level tool-parameter
type controls conversion of Qwen's untyped parameter text. String-admitting values remain strings;
other explicitly typed values are decoded as JSON without coercion. NInfer does not validate
generated arguments against the full JSON Schema.

Tool calls are read in the Qwen form (`<tool_call>` around `<function=NAME>` with
`<parameter=NAME>` values) and in the forms agent harnesses write: `<function name="NAME">`,
`<invoke name="NAME">` and `<param name="NAME">`, bare or inside a `<function_calls>` container.
Every opening tag must be closed by its own kind.

String values keep tool-call markup and balanced nested parameter tags as text. The format has no
escape, so a closing parameter tag inside a value is read by what follows it: ordinary text makes it
value text, while `<` or the end of the output, after optional whitespace, makes it the end of the
value. A value therefore cannot contain a closing parameter tag followed by markup, nor an unmatched
nested parameter opener; either makes the call malformed.

Generated prose can quote tool-call markup before the real call. Unless the first marker's region
opens with a complete call, each marker is tried in order, and the first complete region that
consumes the rest of the response becomes the structured turn; quoted markup before it stays
ordinary content. Generated reasoning closes only at a `</think>` followed by format whitespace or
the end of the turn, so a marker the model quotes while reasoning (followed by punctuation or an
escaped newline) stays in the reasoning channel. At most sixteen markers are tried, the first one
included, so an output that keeps repeating the marker costs a bounded amount of parsing. A region that opens a function but does not parse is recovered: complete calls
before the failure are kept, a call that cannot be read is returned as a call to the reserved
`malformed_tool_call` tool, whose arguments tell the model what went wrong so that it retries, and
text after it is dropped.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions. With `--assistant-prefill` a final assistant
message is an assistant prefill, as on `/v1/messages`: generation continues that turn in place
instead of opening a new assistant turn after it. The prefill must be text only, and a prefill whose
thinking is not explicitly disabled (top-level or `chat_template_kwargs` `enable_thinking`, or
`reasoning_effort: "none"`) is refused with `assistant_prefill_not_supported`.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Clients send a wider effort vocabulary than the maintained Qwen templates accept (`low`, `medium`,
`xhigh`). This fork collapses `minimal` onto `low` and `high`/`max` onto `xhigh` before rendering,
so OpenAI and Claude Code requests that send `high` do not fail inside the template.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. Anthropic
`thinking:{"type":"enabled","budget_tokens":N}` overrides this default for that request.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget. If the effective output
capacity extends past the cap but cannot fit the complete tokenizer-derived control suffix plus one
post-close model token, preparation is rejected with HTTP 400 code
`thinking_budget_capacity_insufficient` rather than partially inserting control. The server does
not promise that the model will emit nonempty content or a tool call after the marker.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The selected template
interprets the other standard values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
Conflicting explicit `enable_thinking` and effort values return `conflicting_template_option`.

`--default-reasoning-effort` sets the effort for requests that name none. It yields to everything
the request decides: a request effort replaces it, and a request that disables thinking, forces a
tool call or prefills the assistant turn is served as if it were unset. `none` turns thinking off
by default; any other value conflicts with `--no-thinking` and fails startup.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

With `--first-token-logprobs`, a non-streaming Chat Completions request with `top_logprobs: N`
(`1..20`) gets `choices[0].logprobs.content` with one entry: the first generated token, its log
probability and the `N` most likely tokens at that position, under the raw next-token distribution
before temperature, penalties and filters. `token` is the token's text with any partial UTF-8
sequence replaced and `bytes` its exact bytes. Later tokens carry none, and `logprobs: true` stays
unsupported.

### llama.cpp-compatible request observations

Every successful Chat Completions response includes a top-level `timings` object. This is a
llama.cpp-compatible response extension, not an OpenAI field. In a stream it is attached to the
last JSON chunk before `[DONE]`: the empty `choices` usage chunk when
`stream_options.include_usage` is true, otherwise the finish-reason chunk.

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```

`cache_n` is the exact Engine-proven reused prompt prefix and `prompt_n` is the remaining prompt
suffix, so `cache_n + prompt_n` equals `usage.prompt_tokens`. Prompt time starts when admission
commits that exact reuse choice and ends when the first output token is committed. Generation time
starts at that first token and ends at the last committed output token. Accordingly, generation
speed uses `max(predicted_n - 1, 0)` token intervals; the first token belongs to prompt latency and
is not counted again as a decode interval. Zero-token, one-token, zero-duration, and exact-cache-hit
cases report finite zero rates rather than `NaN` or infinity. Speculative requests additionally
include terminal `draft_n` and `draft_n_accepted` when draft work occurred.

Set top-level `timings_per_token: true` on a streaming request to attach the latest cumulative
timing snapshot to each visible reasoning or content chunk and, when `return_progress` is also
enabled, to each prompt-processing chunk. During prompt processing that snapshot is prompt-only:
`predicted_n`, `predicted_ms`, `predicted_per_token_ms`, and `predicted_per_second` remain zero
until the first output commit, `prompt_ms` is the elapsed wall time since committed admission, and
the prompt rates span the processed suffix (`processed - cache`) tokens rather than the complete
prompt. This does not enable terminal timings, which are always present. A model commit that is
temporarily hidden by UTF-8, stop-string, reasoning, or tool-call buffering still advances the
cumulative token count; the next visible chunk observes that committed frontier. The option
increases response serialization and transport volume and is off by default.

Set top-level `return_progress: true` together with `stream: true` to receive prompt-processing
chunks:

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  },
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 41.5,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 0,
    "predicted_ms": 0.0,
    "predicted_per_token_ms": 0.0,
    "predicted_per_second": 0.0
  }
}
```

The `timings` object appears only when `timings_per_token: true` is also set. It reports the same
prompt-only live snapshot described above: `time_ms` is integral milliseconds while
`timings.prompt_ms` keeps sub-millisecond precision, and the prompt rates cover the 2,048
processed suffix tokens so far.

The initial event has `processed == cache`. Later cumulative events are published only after the
corresponding prefill unit commits, may be coalesced when the consumer is slower than prefill, and
never move backwards. The final event has `processed == total` and precedes the first output delta.
For an exact full-prefix hit, the initial event already has `cache == processed == total` and no
synthetic prompt work is reported. `time_ms` is elapsed wall time since committed admission;
clients may calculate actual suffix progress as `(processed-cache)/(total-cache)` when the
denominator is nonzero.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with `--default-max-tokens` but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values pass to the selected template |
| `reasoning.summary` | omitted, `null`, or any string; every string requests the same fixed protocol placeholder without changing model execution, and the original value is echoed in the response |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `text.format` | `text`, `json_object`, or flat `json_schema`; see [structured output](#structured-output) |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, a named function, `required` over a single callable tool, or function-only `allowed_tools` in either mode; a namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` is accepted only when no effective tool is callable |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted, empty, or `["reasoning.encrypted_content"]`; the supported value requests the local raw-reasoning mirror described below |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | valid `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, `safety_identifier`, and `user` values are accepted without being mapped to Engine session identity |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored. When a rejection is caused by a
specific value, the error names that value (ASCII-escaped and truncated), its size where relevant,
and its location in the request, both in the error `param` (for example `tools[0].function.name`
or `messages[2].tool_calls[0].function.name`) and in the message text.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Up to four such values become shared stable-prefix
boundaries; they affect reuse opportunities, not prompt identity or output semantics. String
message status/phase metadata is accepted but has no Qwen prompt representation.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Only functions in the current effective tool set can become structured calls;
undeclared model output remains ordinary text. `allowed_tools` filters that set without changing
declaration order, while `tool_choice:"none"` disables structured tool output even when the history
contains earlier calls.

A forced selection names one function: `tool_choice` naming it directly, `required`, or
`allowed_tools` with mode `required` over a single callable tool. NInfer executes it by ending the
generation prompt with that call's opener, so the answer can only continue inside the call. It
therefore requires a fresh answer position and a prompt without a thinking block: reasoning that
the request itself enables is rejected with `tool_choice_not_supported`, while reasoning that only a
default turns on, the server's or the chat template's, is switched off for that request. `required`
over several callable tools is rejected too, because the function itself would still be sampled. The forced call is present and first in the output; a model that keeps writing may
add further calls, exactly as under `auto` with parallel calls enabled.

NInfer does not execute functions or enforce JSON Schema through constrained decoding, so
`strict:true`, hosted tools, remote MCP tools, and custom free-form tools are rejected. Deferred
loading, output schemas, and caller restrictions that exclude direct invocation are also rejected
because their semantics cannot be honored.

NInfer does not execute functions or constrain tool arguments with JSON Schema, so
`strict:true`, required or named tool choice, hosted tools, remote MCP tools, and custom free-form
tools are rejected. Deferred loading, output schemas, and caller restrictions that exclude direct
invocation are also rejected because their semantics cannot be honored.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` item containing raw `reasoning_text`; it returns a placeholder summary if
  `reasoning.summary` is requested;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

When `include:["reasoning.encrypted_content"]` is requested, reasoning Items also
carries en `encrypted_content` equal to its raw `reasoning_text`. This field is **not**
**encrypted** and provides no confidentiality.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

For a reasoning Item requested with any string-valued `reasoning.summary`, its
`response.output_item.added` and `.done` payloads carry the same placeholder summary. Immediately
after the Item is added, the stream emits `response.reasoning_summary_part.added`,
`response.reasoning_summary_text.delta`, `response.reasoning_summary_text.done`, and
`response.reasoning_summary_part.done` with `summary_index:0`, then continues with the raw
`reasoning_text` content lifecycle. Omitted or `null` summary requests emit none of these summary
events and retain an empty Item `summary` array.

For `include:["reasoning.encrypted_content"]`, the in-progress
`response.output_item.added` Item omits `encrypted_content` because the complete reasoning text is
not available yet. `response.output_item.done` and the terminal Response output contain the same
complete raw mirror. A response with no reasoning Item emits no encrypted placeholder.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous tool-call marker prefix or the structured tool region is
held. Tool markup that never opens a function is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, `include` values other than
`reasoning.encrypted_content`, background execution, compaction, files/audio, and
OpenAI-hosted/MCP/custom tools. Except for the two explicitly documented placeholders,
these are compatibility boundaries rather than silently accepted approximations.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
Mid-conversation System messages retain their input position. A final text-only Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media, Thinking, or tool calls and cannot start with Thinking
enabled; Thinking left to the server default counts as enabled, so a prefill needs it disabled
explicitly.

`max_tokens` is optional for local clients and otherwise uses `--default-max-tokens`; a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k`, and
`stop_sequences` enter Engine execution. A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
`budget_tokens >= 1024` and less than `max_tokens`, and that budget is passed to Engine. Visible
Thinking is returned with an opaque local signature; SSE emits its `signature_delta` before
closing the block. Assistant Thinking blocks must be passed back unmodified with that signature;
signatures belong to the current serve process and are invalid after it restarts.
`display:"omitted"` is rejected because NInfer cannot provide Anthropic's
encrypted hidden-reasoning restore semantics. `preserve_thinking` remains a NInfer extension for
closed-turn reasoning history. `output_config.effort` passes its protocol-validated value to the
selected template.

User-defined, non-strict tools support `name`, `description`, object `input_schema`, and
`input_examples`. `tool_choice` `auto`, `none`, `tool` naming a declared tool, and `any` over a
single callable tool are executable; the last two require reasoning to be disabled for the request.
`any` over several tools, `strict:true`, active single-call enforcement, deferred tools, tools that
exclude direct model calls, Anthropic-provided/server tools, toolsets, MCP, and containers are
rejected because their required constraint or executor is absent. `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Ephemeral `cache_control` on the request, tools, System blocks, and User text/image frontiers is a
best-effort retention hint. NInfer maps representable breakpoints to exact prompt frontiers, keeps
the latest markers allowed by the Engine configuration, and ignores TTL and unrepresentable cache
hints rather than rejecting generation. Reuse still requires exact rendered-token compatibility;
aggregate usage reports verified reused tokens in `cache_read_input_tokens` and leaves cache
creation unknown. Streaming emits `message_start` after Engine admission commits the prefix
selection and before transfer/prefill output, so its uncached/cache-read split is already exact;
terminal cumulative usage matches the aggregate response.

Documents, Search Results, Files, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated; `GET /v1/load`,
`GET /stats` and `GET /metrics` require the key, on `--stats-port` too.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default. A preflight's
`Access-Control-Request-Headers` are allowed on top of the fixed list, so browser clients that send
their own headers (a WebUI's `x-conversation-id`) pass the preflight.

After the listening line, startup prints the URLs to open: the API base (`/v1`) and the WebUI when
one is served. A wildcard bind address (`0.0.0.0`, `::`) is announced through loopback, since a
browser does not accept it as a destination.

## Server options

The table lists executable defaults. The startup example selects a long-context FP8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--stats-port N` | also serve `GET /health`, `/stats`, `/v1/load` and `/metrics` on port `N` of the same address, with one worker of their own, so a dashboard or watchdog is never queued behind the connections the request pool serves; the readiness and API-key rules are the main listener's | off |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--rope-yarn` | past the model's native window, YaRN at factor `--max-context` / native instead of plain RoPE for the text and MTP layers | off |
| `--rope-yarn-factor F` | YaRN at a fixed factor in `[1,4]` for every position whatever `--max-context` is, as Qwen documents one factor per deployment; it grows neither the default context nor the KV pool, and long-context extrapolation is not a quality guarantee | `1` (as `--rope-yarn` decides) |
| `--rope-scaling-factor F` | instead of YaRN, linear position interpolation for the text and MTP layers: a position past `--rope-scaling-original-context` rotates at original + (position - original) / F, so positions up to it keep their exact angles; `[1,32]`, excludes `--rope-yarn` and `--rope-yarn-factor`, and raises neither the four-times-native window cap nor the KV pool. The DFlash adapter keeps its plain RoPE | `1` (off) |
| `--rope-scaling-original-context N` | the interpolation threshold | the native window |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context`, or `auto` with `--use-alt-prefix-caching` | `8192` |
| `--kv-headroom-mib N` | device memory in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto`. `--vram-headroom-mib` is accepted as an alias | `1024` |
| `--cuda-memory-policy default\|mixed\|strict\|strict-N-N` | choose reported-free sizing, allowed Windows system-RAM fallback, or verified VRAM residency; `strict` means reserve 64 MiB / probe step 128 MiB. [Policy details](rtx-5070ti-windows.en.md#4-choose-a-memory-policy) | `default` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `600000` |
| `--recover-invariant-failures` | a broken internal invariant in the Engine worker fails the active and materializing requests and leaves the waiting ones queued, as recovery from out of memory does, instead of failing the Engine; eight consecutive recoveries without a completed unit still fail it | off |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--fast-prefill-kernel` | prefill an `int8` KV cache with the fast prompt-attention kernel (FP16 PV accumulation per 64-key tile) and round `--prefill-chunk` down to whole attention waves; a small perplexity cost (see [perplexity](perplexity.md)) | off |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-colours on\|off` | `on` colours the console log's levels and gives every statistic of the operational lines a stable colour; `off` keeps the log plain; a redirected stderr is always plain | levels coloured on a console |
| `--log-stats-panel on\|off` | pin the session statistics panel beneath the console log on an interactive terminal | off |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see the [README](../README.md#several-gpus-pipeline-stages---devices-ab)); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--request-log-max-mib N` | rotate the request log once it reaches N MiB: the file is renamed to `FILE.1`, older copies move up to `FILE.<keep>` and the oldest is dropped, and the new file starts with the `server_start` record; requires `--request-log-jsonl` | `0` (one unbounded file) |
| `--request-log-keep N` | rotated request logs kept; `0` keeps none | `4` |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|rk4v4-e8\|nvfp4\|k8v4` | KV-cache storage. `rk8v4` is opt-in RotorQuant, `rk4v4` opt-in Lloyd-Max 4-bit keys and `rk4v4-e8` opt-in E8-lattice INT4 keys; all eight are accepted on this fork's sm_86/sm_89 targets | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--adaptive-mtp` | MTP only: each round verifies 3..`--draft-tokens` drafts, the width favored by the drafts' measured survival and the measured round cost; see [Adaptive MTP](#adaptive-mtp) | off |
| `--mtp-attention-window N` | MTP only: the draft head attends to the first 64 keys and the newest `N` before its query; verification keeps full attention; see [MTP attention window](#mtp-attention-window) | `0` (whole history) |
| `--lookup-ngram N` | context-lookup drafting alongside `--spec`: the last `N` tokens are matched against the sequence so far and what followed is proposed; exact, since verification rejects a wrong guess | `0` (off) |
| `--ngram-draft-tokens N` | copy drafting alongside `--spec`: up to `N` tokens (1..63; above 15 only at `--max-concurrency 1`) copied from earlier prompt, tool-result or output text that the last `--ngram-min-match` tokens match, verified by the target; `0` disables it; see [Ngram copy proposals](ngram.md) | `15` with `--spec`, else `0` |
| `--ngram-min-match N` | shortest match a copy is drawn from, `4..64` | `12` |
| `--ngram-archive-mib N` | RAM archive that keeps finished requests' copy sources for later requests naming the same `X-NInfer-Draft-Session` | `0` (off) |
| `--ngram-session-mib N` | per-session share of that archive, 1 MiB up to the archive size | `128` |
| `--ngram-native-sessions` | with the archive, also recognize the session identities Kilo, Codex and Claude send | off |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--default-max-tokens N` | output limit when omitted by a request; `0` generates until the context runs out | `8192` |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--no-webui` | stop serving a WebUI built in with `NINFER_WEBUI_DIR` | served when built in |
| `--webui-mcp-proxy` | relay the WebUI's MCP traffic at `/cors-proxy` (http targets, no API key) | off |
| `--structured-output` | accept JSON and JSON Schema response formats (see [Structured output](#structured-output)) | off |
| `--unconstrained-response-format` | without `--structured-output`, generate a JSON or JSON Schema request unconstrained instead of refusing it | off |
| `--assistant-prefill` | continue a Chat Completions request's trailing assistant message in place, as `/v1/messages` does | off |
| `--lenient-assistant-history` | accept Responses input whose assistant message content or reasoning follows `function_call` Items in one run: it joins that run's assistant turn, as Messages flattens content blocks, and the template renders it before the calls. Without it such input fails with `invalid_assistant_history` rather than being silently reordered | off |
| `--thinking-budget-message TEXT` | message a thinking-enabled request receives at its thinking budget instead of the built-in notice; the canonical `</think>` close is appended when missing | built-in |
| `--default-reasoning-effort E` | effort for requests that name none: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` | unset |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay\|cpu` | `overlay` keeps the Vision tower in pinned host memory and encodes each image inside a window borrowed from the evict-ranked text weight tail, so `--vision` no longer reserves device memory and `--kv-capacity auto` resolves the no-vision capacity; requires `--vision` and CUDA virtual memory management. `--vision-offload on\|off` is accepted as an alias for `overlay\|resident`. `cpu` decodes the tower to host FP32 and encodes on CPU threads with no device Vision memory (see [Vision residency](#vision-residency)); it caps `--vision-max-merged` at 256 unless given | `resident` |
| `--vision-cpu` | `--vision` with `--vision-residency cpu` | off |
| `--vision-max-merged N` | merged-token budget of one media item, `[64, 16384]`; larger images and video frame pairs are downscaled at preprocessing instead of being rejected, and the overlay window is sized for it | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--cuda-graph-allowance-mib N` | total CUDA Graph driver-state allowance in MiB, subtracted from the KV sizing budget | computed |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--use-alt-prefix-caching` | select the hybrid prefix cache ([spec](maintainer/hybrid-prefix-cache-spec.md)): content-addressed 64-token KV blocks shared across requests plus sparse state snapshots. It configures itself: `--kv-capacity` defaults to `auto` (free VRAM becomes Device block cache) and `--host-cache-mib` sizes the one pinned Host pool that blocks and snapshots share. The checkpoint-catalog flags are rejected with it: its capacities, `--auto-long-anchors`, `--auto-prefix-grid`, `--derive-session-keys`, `--context-cache-policy`, `--release-diverged-checkpoints`, `--thorough-admission-search`, `--recency-eviction`, `--value-aware-demote` and the disk tier, as are pipeline `--devices`. Requests that share a new prefix with one still prefilling wait for its snapshot instead of prefilling the prefix again, which needs `--concurrent-prefill` to admit them meanwhile. | off |
| `--use-original-prefix-caching` | select the checkpoint catalog explicitly; it is already the default, and the flag is accepted for command lines written for builds where the hybrid cache is. Rejected together with `--use-alt-prefix-caching` | default |
| `--device-snapshot-slots N` | hybrid: Device state snapshot slots (`1..64`) | `max-concurrency + 1`; `+ 2` without a Host tier |
| `--cache-taps-per-request N` | hybrid: new prefill state snapshots per request (`0..64`) | `8`; `2` without a Host tier |
| `--cache-tap-ladder N` | hybrid: history-snapshot ladder base G; flexible taps at `prompt − G·2^k` | `max(4096, 2 * prefill-chunk)` |
| `--cache-tap-min-gap N` | hybrid: minimum tokens between ladder snapshots | `max(1024, prefill-chunk)` |
| `--prefix-cache-file PATH` | hybrid: at startup, restore the Host tier from `PATH` if the file exists; on clean shutdown (Ctrl+C, Ctrl+Break, or closing the console window), save it there (every Host-backed snapshot and the block path it resumes through). Windows ends a closing console's process about 5 s after the close; a save still running then is abandoned and the previous file kept, so stop large caches with Ctrl+C. `PATH` may be relative (resolved against the launch directory) or absolute, e.g. `--prefix-cache-file "e:\NInfer-Deploy-V3\file.cache"`. Its directory must exist, and the flag needs a Host tier (not `--host-cache-mib 0`). A file written for another artifact, KV format, speculative backend, RoPE scaling or `ninfer-serve` binary is ignored and replaced at shutdown. The startup log reports what was restored. Saving writes up to `--host-cache-mib` of data. | off: nothing is saved or restored |
| `--device-state-slots N` | extra Device checkpoint StateImages beyond the active-lane guarantee | `max-concurrency` |
| `--host-state-slots N` | pinned Host StateImage capacity | `8` |
| `--host-kv-mib N` | shared pinned Host Main/Backend KV byte capacity in MiB | `8192` |
| `--host-cache-mib N` | single pinned Host RAM ceiling for the whole retention tier in MiB. Hybrid mode: the slab pool KV blocks and state snapshots share, split at run time by eviction value; `0` keeps the cache on the Device only, and a nonzero budget below one snapshot is rejected. Default cache: the engine derives the Host StateImage slot count from the checkpoint inventory the capture path creates, spends the remaining state headroom on more long anchors per continuation when `--auto-long-anchors` is on, and gives Host KV the remainder. Replaces `--host-state-slots` and `--host-kv-mib`, which are rejected alongside it. | hybrid `8192`; otherwise unset (component flags used) |
| `--max-private-continuations N` | private continuation descriptor capacity | `2 * max-concurrency` |
| `--max-shared-prefixes N` | Engine-wide shared stable-prefix descriptor capacity | `max(max-concurrency, 7)` |
| `--max-long-anchors-per-continuation N` | private long-anchor limit per continuation; with `--auto-long-anchors`, `--host-cache-mib` raises it within the state inventory it funds and never lowers it | `2`, `4` with `--auto-long-anchors` |
| `--derive-session-keys` | a request that names no session -- Chat Completions and Messages carry none -- gets a key derived from its instruction messages and first user message, which stay fixed for the conversation's life. The conversation then keeps a session lineage and live-session retention for its checkpoints, as a Responses conversation does. A client compaction rewrites the first user message and starts a new key | off |
| `--auto-long-anchors` | the engine anchors up to that many message boundaries of every request itself, so a later request that rewrites earlier history -- a compacted or edited transcript -- resumes from the nearest retained anchor instead of root; each anchor costs a prefill split and a StateImage whether or not the client ever rewrites | off |
| `--long-anchor-spacing N` | with `--auto-long-anchors`, the minimum token gap between automatic anchors, doubling per anchor walking back from the prompt end (anchor k sits at least `N * 2^k` tokens below the previous grid point), so short tool-loop turns do not each cost an anchor and deep history stays covered; `0` anchors every one of the last boundaries | `1024` |
| `--max-cache-markers-per-request N` | caller marker input-complexity bound | `4` |
| `--disk-kv-path DIR` | disk tier under the Host tier; see [Disk tier](#disk-tier) | off |
| `--disk-kv-gib N` | disk tier budget in GiB | `64` |
| `--disk-kv-restore` | seed a new request's matching prefix from the disk tier | write-only |
| `--disk-kv-directstorage` | read restores through Microsoft DirectStorage; Windows builds with `-DNINFER_DIRECTSTORAGE=ON` only, untested | mapped reads |
| `--first-token-logprobs` | accept Chat Completions `top_logprobs` (`1..20`, non-streaming) and report the first generated token's log probability with that many alternatives under the raw next-token distribution; `logprobs: true` stays unsupported | off |
| `--context-cache-policy default\|rolling` | `rolling`: within one cache session (a Responses `prompt_cache_key`), a capture that extends a resident checkpoint the request matched exactly inherits that resident's demand, so a conversation whose prompt only grows keeps rolling its frontier forward; with conversations sharing a prefix, one conversation's extension can evict the prefix the others use | `default` |
| `--concurrent-prefill` | admit waiting requests to free lanes while other requests prefill, instead of holding admission until the staged prefill finishes | off |
| `--kv-lease-growth` | admission reserves the prompt plus a 4096-token output window (or `--prefill-chunk` when larger) instead of the whole `max_tokens` budget and extends it at decode-round boundaries, releasing idle retained cache owners least recently used first when the pool is short; an answer whose smallest step still does not fit ends with `finish_reason=length` before `max_tokens` | off |
| `--recency-eviction` | pressure ranks private conversations and shared prefixes in one least-recently-used order (latest hit or publication): a request that does not fit gives up the fewest oldest owners that make it fit and spares those it does not need, instead of clearing every owner it cannot reuse; kept owners, shared prefixes included, are demoted to Host where it has room, and the search may fully evict only inside that tail and only an owner Host cannot take. A private checkpoint that finds the state pools full reclaims the idle conversation that is oldest in this order, not the oldest publication | off |
| `--value-aware-demote` | pressure search charges evicting a private conversation its rank by rebuild cost among the request's victims, so among otherwise equal plans it keeps the conversations most expensive to rebuild, demoted to Host, and evicts the cheapest; it only reorders a bounded search, which can then settle on a different plan | off |
| `--thorough-admission-search` | a new request's admission searches up to 250 ms for its reuse plan at every boundary, even while other requests decode (otherwise 50 ms idle, 10 ms busy), and a request whose plan saves more earns a longer base grant; every eligible option with an uncertain forecast gets its own short discovery grant (otherwise one per search), an option already generated for a candidate is always assessed, and the gain a search may spend against is the request's own re-prefill rather than a share of it per runnable request; a running decode can then pause up to that long when a request arrives | off |
| `--release-diverged-checkpoints` | a private checkpoint of one cache session whose next request diverges from it at the checkpoint's frontier keeps no retention value, so it is the first to go when the cache needs room; a checkpoint that merely cannot serve the request, another session's or a shared prefix, keeps its value. A client that switches back to an earlier branch of the conversation loses that branch's cache | off |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--cors` | permissive browser CORS headers | off |
| `--usage-chunk-choice` | give the streamed usage chunk a zero-delta choice, for strict client parsers that reject the OpenAI-conformant empty `choices` array | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |
| `--post-thinking` | sample the answer of every thinking request with the post-thinking preset once its reasoning closes | off |
| `--post-thinking-temperature F`, `--post-thinking-top-p F`, `--post-thinking-top-k N` | post-thinking overrides; each implies `--post-thinking` | preset |
| `--post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F]` | the same overrides in one flag | preset |

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B and Qwen3.8-27B use `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

### Post-thinking sampling

A thinking request samples its reasoning and its answer with one preset. Post-thinking sampling
switches the device sampler once, at the first decode round after the model closes its reasoning
block, to a second preset: the thinking preset with temperature `0.2`. Tokens that a speculative
round accepts together with the close keep the reasoning sampler. It is off by default. The server enables
it for every thinking request with `--post-thinking` or any `--post-thinking-*` override; a single
request enables it with a `post_thinking` object on Chat Completions, Responses or Messages:

```json
{"post_thinking": {"temperature": 0.1, "top_k": 10}}
```

The object accepts `temperature` (`0..2`; `0..1` on Messages), `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty` and `seed`; an empty object selects the preset. Request
fields override the server's post-thinking flags, omitted fields come from the preset, an omitted
seed continues the request's seed, and `--greedy` forces temperature `0` in both phases. A request
without thinking, or one whose reasoning never closes, keeps its original sampler. The JSONL
`request_done` record reports `post_thinking_sampling` when the switch happened.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`:
`C` slots guarantee active requests and `H` is a global checkpoint pool. Host State and Host KV are
independent startup-fixed pinned-memory capacities; Host KV is shared by Main and the selected
Backend pool and is consumed in physical page extents. `--no-prefix-reuse` selects root-only Engine
mode and cannot be combined with any of the seven explicit context-cache capacity flags, including
zero-valued flags.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

Serve writes human-readable operational records to stderr using
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`. Normal output covers material startup milestones,
readiness, request lifecycle, fixed-interval throughput, and shutdown; `--log-level debug` exposes
internal startup and resource-planning detail. A terminal may use one transient line during startup,
but Serve throughput is always a persistent record. Redirected stderr contains no terminal control
sequences.

With `--log-stats-panel on`, on an interactive terminal that accepts VT cursor control and at `info`
verbosity or more, Serve pins a session statistics panel beneath the scrolling records.
Records scroll above it and remain in the scrollback; the panel is redrawn after each record and
left on screen as ordinary output at exit. It has one row over every completed request and, once
more than ten have completed, one over the last ten:

| Column | Aggregate |
|---|---|
| TTFT | mean time to first token |
| cached | prefix-cache hit tokens / prompt tokens |
| prefill | computed (non-cached) prompt tokens / prefill seconds, in tok/s |
| decode | output tokens after the first / decode seconds, in tok/s (per-request rate, not batch throughput) |
| `<DRAFTER>`, acc/rnd | model-drafter (MTP or DFlash) accepted / drafted tokens, and accepted tokens per model-drafted round; n-gram rounds are excluded |
| ngram, ng rnds | n-gram accepted / drafted tokens and verification rounds |
| archive | n-gram archive accepted / drafted tokens, shown once the archive has drafted |

The table is 77 columns wide, 86 with the archive column, so it fits a console window snapped to
half of a 1920-pixel screen; a narrower window cuts the rows at its edge.

Ratios and rates divide summed tokens by summed seconds, so each request weighs by its size. The
title counts completed, failed, cancelled, and rejected requests and, while throughput reporting is
enabled, the current running and waiting requests. The panel only reads the same outcomes as the
`req#N done` records; it changes no request behavior.

Pretty values use readable units and rounded rates; use the independent request JSONL for complete
fields and full precision. Operational records never contain prompts, generated text,
request bodies, credentials, or arbitrary client error messages.
If a tool marker is returned to text because its structure or tool identity cannot be represented,
Serve emits one warning with only the failure classification, never the generated markup.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Add `--request-log-jsonl profiles/bench/run/server.requests.jsonl` to the startup command to write
the log at that path.

Every line is one `ninfer_serve_request_log` schema-v22 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities and switches, n-gram drafting options, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, Host State/KV capacity and occupancy, KV sizing ledger, CUDA Graph allowance and measured cost, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, request-owned materialization cost/search diagnostics, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters including n-gram and draft-archive counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.materialization` is the context-cache decision committed for that request: predicted
immediate, future-loss and total nanoseconds (`predicted_now_ns`, `predicted_future_loss_ns`,
`predicted_total_ns`, and `initial_predicted_total_ns` before search); `targets_evaluated`,
`projection_work`, `planning_elapsed_ns` and `search_elapsed_ns`; `stop_reason`; `budget_exhausted`;
`selected_degradation_units` and `selected_maximal_fallback`; and the optional-search accounting
`first_improvement_ns` (or `null`), `incumbent_improvements`, `search_work`, `search_granted_ns`,
`search_renewals`, `search_discovery_used`, `search_overshoot_ns`, `search_stop_phase` and
`search_boundary_limited`. Stop reasons are `no_pressure`, `queue_exhausted`, `target_budget`,
`expansion_capacity`, `time_budget`, `insufficient_expected_gain` and `work_budget`; `time_budget`
means the wall or control allowance ran out, while a request too cheap to justify optional search
reports `insufficient_expected_gain`. Search phases are `none`, `setup`, `construction`,
`assessment`, `expansion` and `refinement`. Search is bounded and heuristic; these diagnostics do not
claim model or global optimality, and aborted planning attempts are not published. With the
environment variable `NINFER_MAT_DEBUG` set, admission also writes its raw inputs to stderr: a
`[candgen]` line for each catalogued checkpoint it did or did not turn into a candidate, and
`[mat-debug]` lines for each identity cost, fast selection, assessed target and denied search step,
the last with the gain and economic bound behind the denial.
`cached_prefix_tokens` and `restored_host_bytes` describe a hybrid prefix-cache admission and are `0`
with the checkpoint catalog. `cached_prefix_tokens` is the longest prompt prefix held as cached KV
blocks, whether or not it was reusable: reuse also needs a state snapshot inside it, so a gap to
`prefix_cache_hit_tokens` is prefix lost to snapshot placement. `restored_host_bytes` is what the
admission copied back from the Host tier; those copies overlap the request's first prefill pass, so
their time is part of its prefill.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, `duplicate_parameters_repaired`, and a stable fallback reason. A
parameter named more than once in one call keeps its last value, as in JSON object syntax, and
counts once in `duplicate_parameters_repaired` for each repeat instead of demoting the call to text.
`forced_call_closed` reports a forced call whose closing tag the decoder supplied. When the strict
pass rejects a region, `recovered` reports that a recovery pass still produced calls from it (with
`recovered_call_count` the calls only that pass kept), `malformed_call_reported` that an unreadable
call became a call to the reserved error tool, and `trailing_content_dropped` that text after the
last call was dropped. Fallback reasons are `none`, `malformed_structure`, `invalid_tool_name`,
`undeclared_tool`, and `trailing_content`. These counters contain no tool arguments or generated
text.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`, and under
`--adaptive-mtp` also `window_transitions` and `rounds_per_window`. Rates can be derived downstream
from raw token counts and seconds instead of rounded stderr strings. `ngram_rounds`,
`ngram_drafted_tokens`, and `ngram_accepted_tokens` are the part of `rounds`, `drafted_tokens`, and
`accepted_tokens` whose proposal came from n-gram copy drafting, and `ngram_archive_rounds`,
`ngram_archive_drafted_tokens`, and `ngram_archive_accepted_tokens` the part of those whose copy
source was the retained draft archive (see [n-gram drafting](ngram.md)). The
`speculative.ngram_archive` object reports that archive at the end of the request: `enabled` when
the server has one (`--ngram-archive-mib`), `bound` when the request was bound to a draft session,
`published` when its input and output were published into it, `generation` as the session's latest
completed generation, `sources` and `session_bytes` as the bound session's retained source count and
bytes, `total_bytes` as the whole archive's bytes, and `sampling_seed` as the effective seed after
request-domain separation for a request that named a session, otherwise `null`.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the `residency` (`resident`, `overlay` or `cpu`),
the aggregate prompt and maximum-item token bounds plus encode peak and handoff layout/usage within
that same allocation; these bytes must not be added to `workspace.capacity_bytes`. The field is
`null` when Vision is disabled.
`cuda_graph_allowance_bytes` is the CUDA Graph memory the KV sizing reserved, and
`cuda_graph_measured_bytes` the Device memory graph preparation actually took at startup (`0`
without CUDA Graphs); the startup log warns when the second exceeds the first.
In `strict` mode this measurement uses the matching process/GPU Dedicated counter delta
across Graph preparation. CUDA's free-memory delta can otherwise report zero once WDDM clamps
free memory to zero. The observed preparation delta may include lazy driver/kernel allocations
in that interval; it is not a per-Graph allocation ledger.

`server_start.engine` records `ngram_draft_window` and `ngram_min_match` (`--ngram-draft-tokens`,
`0` when n-gram drafting is off, and `--ngram-min-match`), the draft-archive budgets
`ngram_archive_bytes` and `ngram_session_bytes` (`--ngram-archive-mib`, `0` keeps drafting
request-local, and `--ngram-session-mib`), and `ngram_native_sessions`. Its `context_cache` object
records the resolved capacities and the `rolling_retention`, `release_diverged_checkpoints`,
`thorough_admission_search`, `recency_eviction`, `value_aware_demote`, `kv_lease_growth`,
`automatic_long_anchors` and `long_anchor_min_spacing_tokens` settings, the cache `mode` (`legacy` or `hybrid`), the
`host_cache_budget_bytes`, and under `--use-alt-prefix-caching` a `hybrid` object with the resolved
snapshot slots, taps per request, tap ladder and minimum gap and whether the Host tier persists.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Console lines use local
`[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. Engine runtime diagnostics are ordinary records
prefixed `engine |` that `--log-level` filters: the device route profile the Engine installs and a
calibration it runs are `info`; an ignored or unsaved profile, recovery from out of memory or,
with `--recover-invariant-failures`, from a broken invariant, and context stores rebuilt after it
are warnings; a worker crash or repeated recoveries that fail every pending request are errors. OpenAI Responses, OpenAI Chat, and Anthropic
generation requests receive a request ID when they enter synchronous preparation. Successful
preparation produces `request_start`; a preparation failure produces `request_rejected` without a
matching start. Later generation failures produce `request_error`. Schema/model validation
rejections before preparation and token-count-only calls are not measurement requests and do not
receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `materializing`, `capture_pending`, and
`terminal_pending` fields are the Engine scheduler snapshot at the end of the interval. The JSONL
`context_cache` object reports selection, capture, transfer, COW, pressure spill, private/shared
owner degradation and eviction, private owner demotion to Host (`pressure.private_owners_demoted`, a
subset of `private_owners_degraded`), checkpoint drop, pressure search, budget exhaustion, maximal fallback, and historical-fork
counters as interval deltas; `occupancy` and `last_selection` are end-of-interval gauges. Materialization predictions are
request-owned and appear only on the corresponding `request_done` event.
`pressure.searches` counts plans accepted into Program resource transactions, including a transaction that later ends in
request-local abort; committed victim counters likewise report the resulting stable cache changes.
`salvage.published` counts cancelled requests whose live state was published as a continuation
endpoint at its last committed frontier.
`captures.skipped` counts capture offers the Program declined because they were not physically
feasible: unlike `captures.aborted` it has no other trace, so it is the counter for silent retention
loss. `occupancy.device_main_kv_lease_pages` and `device_backend_kv_lease_pages` are the part of
`device_main_kv_pages` and `device_backend_kv_pages` that active requests hold as reservation but
have not yet written.

With `--use-alt-prefix-caching`, `context_cache.hybrid` is present once the cache has inserted a
block or holds a snapshot. `device_blocks` (Device-resident 64-token KV blocks), `evictable_blocks`
(those Device eviction may drop now), `tree_blocks` (blocks on the Device or Host), `snapshots`,
`host_capacity_bytes` and `host_used_bytes` are end-of-interval gauges; the rest are interval
deltas. `snapshot_hits` counts admissions that resumed from a snapshot and `reused_tokens` the
prompt tokens they reused. `blocks_inserted` counts new tree blocks, `blocks_reattached` blocks whose
existing tree entry took a request's Device pages, and `blocks_duplicate` committed blocks the tree
already held on the Device, whose pages were released. `taps_created` and `taps_skipped` count
planned prefill snapshots published and dropped, and `endpoints_created` end-of-answer snapshots.
`host_image_writes`, `host_block_writes`, `host_image_restores`, `host_block_restores`,
`host_write_bytes` and `host_restore_bytes` count Host-tier write-through and restores.
`evicted_blocks` counts Device block evictions, `host_snapshot_evictions` snapshots evicted from the
Host tier, `host_dead_reclaims` Host slabs reclaimed from KV that no snapshot can reach, and
`unbacked_node_losses` Device evictions of blocks with no Host copy, which remove them and the
blocks after them from the cache.

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose admission, context-transaction, replica, and
stats-publication slow paths; these detail values are already contained in a top-level Host phase
and must not be added to `total`. Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. The stderr interval line shows only total Host milliseconds,
decode Host/device-wait microseconds per round, boundary, and maintenance; use JSONL for analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

### Disk tier

`--disk-kv-path DIR` adds a disk tier under the Host tier. When a private continuation is evicted,
by pressure or by the catalog limit, it is written there before its memory is released: its KV pages
as a prefix chain, the StateImage at its endpoint, and the tail page and StateImage of its rewrite
seam and of its three earliest long anchors. The continuations still resident at shutdown are
written the same way. Pages are keyed by the prefix digest at their last token and by the execution
profile, so an identical prefix is stored once, and the files survive restarts. Each artifact gets
its own subdirectory and each KV format, stride and YaRN setting its own files, so one model or
profile never restores another's KV. `--disk-kv-gib` (default 64) is the total budget, split
65/25/10 across Main KV, Backend KV and StateImages (85/15 without a backend); the least recently
used page goes first. The files are sparse until written, and a torn page fails its CRC and reads as
a miss.

With `--disk-kv-restore`, a request with no resident prefix looks for the longest frontier `E` on
disk whose StateImage and KV chain match its own prompt, is admitted as reusing `E`, and has `[0, E)`
copied into its sequence before prefill starts at `E`; under MTP the StateImage's hidden state
extends the draft KV to `E`. A restore is abandoned after 8 seconds, and any miss or failure falls
back to recomputing the prompt, so a result never depends on the disk contents. Restores read with
eight threads and run on the engine thread, so other lanes pause during one; media prompts are never
restored. Without the flag the tier only writes. Writes of an evicted owner during admission proceed
a batch at a time between decode rounds; an owner released elsewhere is written within 20 seconds,
and shutdown writes for at most 60.

On Windows, a build configured with `-DNINFER_DIRECTSTORAGE=ON` fetches the DirectStorage 1.3
runtime and offers `--disk-kv-directstorage`: a restore then reads its pages into host memory
through a DirectStorage queue, every page of a staging batch in flight at once, instead of copying
from the mapped files. Each page is still checked against its CRC, and a batch that fails or does
not complete within 10 seconds falls back to the mapped reads. This path has been checked to compile
against Windows headers but has not been run.

## Strict CUDA residency

On Windows, `--cuda-memory-policy strict` automatically selects the Hybrid cache and enables controlled
allocation above the CUDA-reported free-memory estimate. It currently requires one GPU, text
generation and the Hybrid context cache. Vision, pipeline devices and
`--wddm-evictable-budget` are rejected. Ordinary CUDA allocation remains the default; selecting
this policy does not enable D3D12 or change inference kernels.

The strict policy prepares intentional pinned Host resources before recording its process Shared
baseline. Host cache remains system RAM and is accounted separately; its configured size must not
simply be subtracted from Windows Shared usage. The startup probe estimates capacity with
temporary allocations, then releases them and builds the actual contiguous weight, KV/state and
workspace layout. `cudaMemGetInfo` supplies an estimate, not a hard rejection boundary.
Successful `cudaMalloc` calls alone are insufficient: device data must be touched and the complete
working set must pass the residency guard after initialization and Graph preparation.

The second number in `strict-64-128` controls the temporary probe step in MiB. It does not split tensors,
alter their pointer arithmetic or replace continuous device arenas with independent pointers.
The first number requests an additional touched spare allocation at final admission;
the allocation is released after validation. This default is an experimental margin, not a claim
that 64 MiB is sufficient for every desktop workload or model configuration. Graph allowance and
the spare allocation are separate resources. Both numbers apply only to strict mode.

For `--kv-capacity auto`, strict mode uses the reserve from its policy value, counted once.
`--kv-headroom-mib` is accepted only with `default + auto`. Mixed auto capacity is bounded
to one full page-rounded context window per concurrent request; it does not fill system RAM.
Mixed explicit capacity bypasses the CUDA-free planning rejection and attempts the requested
allocation without rejecting driver Shared growth. Default retains the old free-based planning
and also does not guarantee permanent dedicated residency. Neither mixed nor strict enables
D3D12; they currently require Windows single-GPU text generation. See the English
[memory-policy reference](rtx-5070ti-windows.en.md#4-choose-a-memory-policy)
for exact values, Host-cache separation, limits and the manager form.

An automatic KV candidate may fall back to a smaller legal capacity after an allocation or
residency rejection, with at most eight startup attempts, but the Engine must still back the
requested `--max-context`. It does not
silently shorten the advertised context. Invalid data or CUDA execution errors are not treated
as capacity failures. A candidate with unexplained Shared growth is not published as ready.

This is an observational Windows guard, not an interface that permanently pins all CUDA device
pages in physical VRAM. Counter sampling can detect and reject a spill after its first occurrence;
it cannot prove that no transient migration ever happened. Other GPU workloads can also change
residency after startup. Consider NVIDIA Control Panel's per-application **CUDA — Sysmem Fallback
Policy → Prefer No Sysmem Fallback** if available, and recalibrate after changing that driver
setting. The Engine does not modify driver profiles. See the [NVIDIA setting documentation](https://nvidia.custhelp.com/app/answers/detail/a_id/5490/).

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay.
A request joins that batch only after its staged prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row. A staged prefill holds
admission back until it finishes. With `--concurrent-prefill` waiting requests are still admitted
to free lanes while others prefill, so a new request's admission and prefill overlap the prefill
and decode of the others (each prefill unit advances one chunk of the lowest staged lane per
worker boundary).

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue. Because there is no preemption, a queued
request waits out the generations ahead of it, so the deadline has to be scaled to the longest
response the deployment allows rather than to a connection timeout: at C1 on an RTX 3090 a single
6,500-token response occupies the engine for about 106 seconds. The 600,000 ms default admits a
queued caller behind roughly ten such responses; lower it only to fail fast on purpose.

One request owns the staged prefill at a time, and the executor alternates a single prefill chunk
with a single decode round, so `--prefill-chunk` sets the worst-case pause every active stream sees
while a new prompt is ingested. On an RTX 3090 ingesting a 4,900-token prompt behind four active
streams, the largest inter-token gap measured 1,043 ms at chunk 1024, 515 ms at 512, and 312 ms at
256, against an 82 ms median decode interval; the ingesting request's own prefill rate fell only
from 1,135 to 1,130 to 1,110 tok/s. Prefill is not batched across requests at any chunk size, so a
smaller chunk trades almost no ingestion throughput for a proportionally smaller stall. The shipped
concurrent launcher uses 512.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom; omitting the option makes it follow `--max-context`. The
CUDA Graph driver-state allowance reserved against that budget is computed from the enabled
graph profiles and concurrency unless `--cuda-graph-allowance-mib` supplies an explicit total; a
too-small value risks CUDA out-of-memory at graph capture and a too-large one shrinks the KV pool.
Capacity resolves once at startup.

Admission reserves the full prompt-plus-effective-output page entitlement through request
completion. A request remains queued until a legal resource plan can satisfy that entitlement.
With `--kv-lease-growth` admission instead reserves the prompt plus a bounded Device KV window over
the remaining output (4096 tokens, or `--prefill-chunk` when larger) and extends it at each
decode-round boundary, so a client that asks for far more output than it generates does not hold
the prefix cache out of the pool. When the pool cannot extend the window, idle retained cache
owners are released least recently used first; if the smallest step still does not fit, the
request completes at the frontier its window covers and reports `finish_reason=length` before
reaching `max_tokens`.

Each reusable checkpoint contains KV and complete continuation state. At admission, capture, and
finish boundaries, resource pressure may keep it on Device, move its StateImage and/or KV replicas
to pinned Host memory, or evict it. The planner compares incoming-request work with the later
recovery cost imposed on retained checkpoints. Active requests retain their state and completion
reservations, and placement choices preserve model semantics. The full policy and invariants are
defined in [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The completion log reports the
reused token count as `cache=`.

The completion log reports one of six reuse paths: `root`, `private_endpoint`,
`private_turn_closure`, `private_response_replay`, `private_long_anchor`, or
`shared_stable_prefix`. Reuse validation covers KV, recurrent state, hidden state, selected-backend
state, and the exact prompt frontier. With stable `preserve_thinking=true`, the auxiliary checkpoint
rolls to the message frontier immediately before the current response's deterministic generation
prologue. A normalized response, compact-summary instruction, or replacement user suffix therefore
replays the small generation prologue and only the changed suffix while retaining the complete
stable conversation prefix. Stable `false` places the turn-closure checkpoint before the first
assistant opener in the open turn, so closing that turn can recompute its opener and omit its
reasoning without discarding the preceding conversation.

`preserve_thinking` selects the capture frontier for newly created checkpoints. Existing exact
checkpoints remain reusable across a mode change. If the desired boundary is behind the selected
reuse frontier and has no snapshot, the Engine keeps the valid hit and defers the new checkpoint. A
later request that diverges before every retained checkpoint starts from root. The JSONL completion
record exposes the restored checkpoint as `prefix_reuse_path`. Reasoning-effort changes participate
in rendered-token identity and exact-prefix selection.

An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior
history remains eligible for `private_endpoint`. If the client modifies, removes, or moves a
historical system message, the token prefix genuinely differs and a miss/reset is correct.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools or constrain their arguments. Final JSON responses can
use [structured output](#structured-output).

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
