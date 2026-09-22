# Anthropic-compatible Messages API (experimental)

KVMem can expose a subset of the Anthropic Messages protocol alongside its existing
OpenAI-compatible API. Enable it with `--anthropic`; it is **disabled by default**.
`--no-anthropic` disables it again. These flags affect only the additional routes.

中文说明：这是独立的 Anthropic 协议适配模块，默认关闭。新增接口复用现有推理流程，
OpenAI 的请求解析、错误格式和流式输出保留在独立模块中。两种接口共享单个模型槽位，
会正常排队；协议隔离不代表算力隔离。首版支持文本、流式、客户端工具调用和本地 token
计数，尚未宣称与 Claude Code 或全部 Anthropic 功能兼容。

## Enable and call

Add `--anthropic --alias local-model` to the server's usual command. For example:

```sh
llama-kvmem-server -m model.gguf --alias local-model --anthropic --port 18210 --api-key example-key

curl http://127.0.0.1:18210/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: example-key' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"local-model","max_tokens":128,"messages":[{"role":"user","content":"Hello"}]}'
```

Use the actual loaded model name or `--alias`; unknown names return 404. The existing
`GET /v1/models` remains OpenAI-shaped and lists that name. Messages requests require
`anthropic-version: 2023-06-01`. Authentication uses the existing configured keys;
`X-Api-Key` and `Authorization: Bearer ...` follow the same validation and precedence
as the other APIs. Anthropic beta headers are not supported.

Official Python SDK example:

```python
from anthropic import Anthropic

client = Anthropic(api_key="example-key", base_url="http://127.0.0.1:18210")
with client.messages.stream(
    model="local-model",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

The base URL is the server root. The SDK supplies the API version header. API keys
above are placeholders. Configuration and model capabilities remain local to KVMem.

## Supported subset

| Feature | Behavior |
| --- | --- |
| Messages | Text strings or text blocks; top-level string/text-block `system`; multi-turn user/assistant history |
| Generation | Required positive `max_tokens`, optional `stream`, `stop_sequences`, `temperature`, `top_p`, `top_k` |
| Client tools | `name`, `description`, object `input_schema`; `tool_choice` auto/any/none/tool; optional `disable_parallel_tool_use` |
| Tool history | Assistant `tool_use` becomes an internal tool call; matching user `tool_result` becomes a tool message |
| Responses | Message content blocks, stable tool IDs, `stop_reason`, `stop_sequence`, and local input/output token usage |
| Streaming | Named message/content-block events, incremental text and tool JSON, terminal message delta and stop |
| Counting | `POST /v1/messages/count_tokens`, using the same local chat template and tokenizer as generation |
| Errors | Anthropic error envelope on these two routes, including authentication errors and in-stream errors |

Named tool choice restricts the available tool definitions to the selected tool.
Tool arguments must form an object at completion. A model that emits an undeclared
tool or malformed final tool input produces an error instead of an invalid success
response. The model/template still determines its tool-use capabilities.

Assistant text must precede its tool-use blocks. User tool-result blocks must precede
any new user text, with every pending tool call answered exactly once. Text-only
tool results are supported; `is_error=true` adds a `Tool error: ` prefix for the model.
Generation requires a final user turn; assistant prefills are not supported.

Stop sequences are excluded from output, including sequences that cross token
boundaries or occur inside one token. Streaming holds a possible stop prefix until
it is resolved. A speculative batch can already have processed extra tokens when
a stop is detected; usage counts processed generation tokens.

`max_tokens` must fit the server's generation limit and available context. Messages
rejects invalid/oversized limits rather than adopting the OpenAI route's existing
clamping and sentinel rules. `max_tokens=0` cache warming is not supported.

## Counting, caching, and thinking

For counting, send `model`, `messages`, and optional `system`/`tools`/`tool_choice` to
`/v1/messages/count_tokens`; no `max_tokens` is required. Counts include the rendered
local chat template and tool definitions. They are not Claude tokenizer estimates
and do not represent KVMem's compressed GPU working set.

Counting acquires the model lock for template/tokenizer access, but does not run
prefill or generation, change model cache state, or update `/slots` task counters.

Recognized `cache_control: {"type":"ephemeral"}` hints, optionally with `ttl` of
`5m` or `1h`, are accepted as hints only. Existing local prefix reuse continues;
Anthropic TTL, cache creation/read accounting, and billing semantics are not
implemented. `usage.input_tokens` reports the full rendered local prompt count;
`usage.output_tokens` reports generated tokens. No Anthropic cache billing counts
are invented.

Messages disables thinking for this initial subset regardless of the process's
OpenAI thinking defaults. An explicit `thinking: {"type":"disabled"}` is accepted.
Extended/adaptive thinking, signatures, images/documents, citations, server tools,
MCP connectors, structured output configuration, beta features and unknown request
fields/blocks return explicit errors. OpenAI vision and thinking behavior is
unchanged by this adapter.

## Isolation and validation

- `tools/kvmem-api-openai.h` owns the existing OpenAI parser and JSON/SSE encoding.
- `tools/kvmem-api-anthropic.h` owns Messages validation, content blocks, stop buffering
  and encoding. Its route registration lives in `kvmem-api-anthropic-routes.h`.
- `tools/kvmem-chat-request.h` contains the internal request and shared template inputs.
- `tools/kvmem-server-chat.h` runs the common generation path with a request-local
  response policy. It does not construct either protocol's wire-format JSON.

The Messages adapter does not call the OpenAI HTTP handler or transform OpenAI SSE.
No request-global protocol switch or shared content-block counter is used. The
existing authentication gate accepts an optional route-specific failure formatter;
OpenAI authentication errors keep their previous representation.

The CMake target `kvmem-api-protocol-test` checks parser validation, tool history,
stop/UTF-8 buffering, content-block state and the OpenAI wire contract without model
weights. The Windows build script includes this target and its CTest entry.

The integration test compares a supplied unmodified baseline binary with the
candidate, both with Messages disabled and enabled, then repeats the comparison
after mixed-protocol traffic. Only dynamic IDs, timestamps and timing measurements
are normalized; cache/token counters and event structure remain compared.

```sh
python scripts/test_api_protocols.py \
  --baseline /path/to/baseline/llama-kvmem-server \
  --server /path/to/candidate/llama-kvmem-server \
  --model /path/to/test-model.gguf --output /path/to/results --require-sdk
```

Install the official `anthropic` SDK in the test environment for `--require-sdk`.
The script defaults to CPU model placement, uses temporary loopback ports, and only
stops servers it starts. For GPU/KVMem/MTP coverage, add the appropriate
`--gpu-layers all --device CUDA0 --kvmem --mtp`; the supplied model must include an
MTP head, and the test checks that speculative decoding actually became active.
MTP tests default to `--mtp-state snapshots`; use `--mtp-state replay` for a
supported Qwen 27B model to exercise ReplaySSM. Add `--mmproj /path/to/projector.gguf
--image /path/to/image.png` for OpenAI image JSON/SSE baseline comparisons. Thinking
JSON/SSE comparisons are included automatically. `--benchmark` records five
128-token OpenAI samples after two warmups for each binary/flag combination.

References: [Messages API](https://platform.claude.com/docs/en/api/messages/create),
[streaming lifecycle](https://platform.claude.com/docs/en/build-with-claude/streaming),
[token counting](https://platform.claude.com/docs/en/api/messages/count_tokens).
