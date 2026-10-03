# NInfer-3090 v0.8.1

A compatibility release. v0.8.0's request validation rejected two shapes that agent CLIs send
unconditionally, which stopped codex-cli connecting at all. Neither rejection was enforcing
anything the client could act on. No engine or kernel changes; the v0.8.0 caching work is
unchanged and carried forward.

## Changes

- **A declared hosted tool no longer fails the request.** codex-cli puts
  `{"type": "web_search"}` in `tools` on every request whether or not search is wanted, and strict
  validation rejected the whole request over that one entry — the client cannot remove it, so the
  only workaround was `-c web_search="disabled"`.

  A hosted tool is the *server's* to execute. NInfer has no executor for one, and the caller is not
  waiting on it either, so the declaration is now dropped: the model is never told the tool exists,
  which is the same outcome as not declaring it. Nothing is silently misreported — the tool cannot
  be called because it was never offered.

  Client-executed types such as `custom` stay rejected. Dropping one of those would leave the
  caller waiting for a call that can never arrive, which is worse than a clear error. The line is
  "would the server have run it", not "do we recognise the name". Matching is by family prefix
  (`web_search`, `code_interpreter`, `file_search`, `image_generation`, `computer_use`, `mcp`) so
  dated spellings like `web_search_preview_2025_03_11` are covered. Applies to both
  `/v1/chat/completions` and `/v1/responses`.

- **`parallel_tool_calls=false` is honoured instead of refused.** codex-cli has no built-in
  metadata for a custom local model, falls back to conservative defaults, and that path sends
  `parallel_tool_calls=false` whenever any tool is available — which in an agentic run with a shell
  executor is always.

  The old error said the guarantee could not be met while tools were enabled. That is true of
  *decoding*, which remains unconstrained, but the part of the contract a client can observe is
  what the response contains, and that much is enforceable: keep the first tool call and drop the
  rest. Measured against a live server on the same prompt and seed, with a tool the model wants to
  call twice:

  | | tool calls | finish |
  |---|---|---|
  | `parallel_tool_calls=true` (default) | 2 — `pwd`, `whoami` | `tool_calls` |
  | `parallel_tool_calls=false` | 1 — `pwd` | `tool_calls` |

  The call kept is the first, so a sequential executor runs it and the model asks for the next on
  the following turn.

`web_search_options` still returns `web_search_not_supported`. That field is an explicit request
for hosted search with citations rather than a passive declaration the client cannot remove, so
failing loudly stays the honest answer.

## Upgrading from v0.8.0

Drop-in. Same launchers, same flags, same model artifacts. If you were passing
`-c web_search="disabled"` to codex-cli to work around the first issue, you no longer need to.

## Validation

- `ctest` 110/110 on the release build.
- All four codex-shaped requests return 200, where three of them returned 400 on v0.8.0.
- Windows: Visual Studio 2022 BuildTools and CUDA 12.8
- Linux: WSL2 Ubuntu 24.04, CUDA Toolkit 12.8, GCC 13, CMake, Ninja
- Both built for `CMAKE_CUDA_ARCHITECTURES=86`.
