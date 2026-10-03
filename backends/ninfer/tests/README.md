# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization and Python-writer/C++-reader interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- root C++ tests — core storage, runtime admission/resource policy, public API, serving protocols,
  logging, benchmark reports and causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, `models/qwen3_5/`
and `ops/`. Registration helpers live in `cmake/NinferTests.cmake`; included manifests keep
executables and CTest working directories under `build/tests/`.

Tests link into one executable, `ninfer_tests`, rather than one each: every test that links the
Op library would otherwise carry its own copy of the whole kernel image (~450 MB on sm_86), which
added up to tens of GB. Each test keeps its name as a program in the bundle and still runs in its
own process under CTest. Run one by hand with `build/tests/ninfer_tests <name> [args...]`;
`ninfer_tests --list` prints the names. `ninfer_jinja_test` and
`ninfer_artifact_materialization_test` stay standalone because Python tests invoke them by path.
A test's helpers and entry function must be internal (anonymous namespace) so that programs do not
collide at link time. The mechanism is `cmake/NinferBundles.cmake`.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/Q8, FP8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Alternatively, `cmake --preset dev` enables products, tests and benchmarks together.
After building, `ctest --preset dev` runs the same CTest suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_tests
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

## Running a test under compute-sanitizer

`memcheck` and `initcheck` both work on these binaries, including the large statically linked ones.
Invoke it unqualified so `PATH` resolves it:

```bash
compute-sanitizer --tool initcheck --error-exitcode 9 \
  build/tests/ninfer_tests ninfer_softmax_attention_test
```

**Do not call the copy under `CUDA/v12.4/compute-sanitizer/` by absolute path.** Two copies are
installed on a typical toolkit layout, and the 2024.1.0 one in v12.4 prints its banner and
`ERROR SUMMARY: 0 errors`, exits 0, and **never executes the binary** — a clean report having
checked nothing. `PATH` resolves `compute-sanitizer` to the v12.8 launcher, which is the working
one, so the unqualified form above is safe.

Two messages that look like failures and are not:

- *"Target application terminated before first instrumented API call"* — the test made no CUDA
  calls, usually because it skipped for a missing `NINFER_*_WEIGHTS` environment variable. Set the
  variable, or pick a test that does not need one.
- A single `-k 'regex:a|b'`-style argument disappearing in PowerShell: `|` is parsed as a pipeline
  before the tool sees it. One pattern per invocation.

`ncu` needs one thing more: GPU performance counters are administrator-only by default on Windows
and it fails with `ERR_NVGPUCTRPERM` otherwise. Run it from an elevated shell — that satisfies the
permission per-run, with nothing persistent and no reboot.
`scripts/sweeps/admin-profile.ps1` does this for the profiles the maintainer notes reference.

The variable-width DFlash2 target-attention subset can be run with
`./build/tests/ninfer_tests ninfer_softmax_attention_test --dflash2-only`. It covers D256/Q24/KV4 across all five
cache codecs, W=2..16, B=1..8, request-local prefixes, cache effects, and Graph metadata/input
updates. The default executable also runs the existing attention geometries and prefill tests.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target ninfer_tests
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

The Linear, LinearAdd and LinearSwiGLU common `.cpp` implementations each compile once into a
test support library. Both those libraries and the Op test executables receive the oracle's
`-fno-fast-math` and `-ffp-contract=off` options on GNU/Clang C++ compilers.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert \
  tests/test_serve_corpus.py tests/test_speed_of_light.py
```

The Python suites exercise conversion and encoded output, without running model inference.
The maintained environment uses Python 3.11 with the dependencies for those suites. C++ binding and Engine tests
cover consumption of their resulting representation.

The real loading test accepts an explicit artifact path and optional component selection:

```bash
./build/tests/ninfer_tests ninfer_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine. It accepts Qwen3.6 and Qwen3.8 artifacts; its prompt-token goldens are kept
per model generation because each registers its own chat template:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_score_real_test --output-on-failure
```

Run the 35B-A3B MoE route independently:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_moe_real_test --output-on-failure
```

Without `NINFER_TEST_ARTIFACT`, CTest marks these real Engine tests as skipped. Run GPU integration
tests serially. `NINFER_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as `vision`,
`pressure-resume` or `concurrent`; the default is `all`. These integration checks
use behavior and state accounting rather than another numerical path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares stripped and preserved closed-turn prompt lengths, and verifies compatible
prefix reuse, speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## DFlash2 Engine integration

The real test uses an artifact containing DFlash2 and checks output budgets, speculative activity,
penalty-enabled sampling, compact batches with unequal budgets, same-route same-seed replay,
retained/fresh prefix behavior and absence of a full backend KV pool. A shared DFlash/DFlash2 fixture starts decode at token 63, verifies across the page
boundary, stops after one target column at token 64, and checks the exact retained frontier and
subsequent generation with and without reuse.
The KV Store test checks exact mapping and reservation accounting for the same transition.
K>=7 also exercises a stop inside a licensed block; K=15 additionally checks oversized prefill,
local ring wrap, and the logical context-capacity tail. Optional Vision runs image/video capture
and prefix restore. Zero extra Device StateImage slots exercise Host snapshot/restore.

```bash
cmake --build build -j --target ninfer_tests
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 15 1 1 8
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16` or `int8`),
Vision enabled, and extra Device StateImage slots. Defaults are `15 1 1 8 bf16 0 3`. Run GPU
integration tests serially. The individual Op suites remain the numerical/state-transition oracle;
the fixed Engine fixture does not define bit parity across arbitrary floating-point routes.

## Structured output

The CPU grammar and protocol tests and the GPU sampling/speculative tests qualify this path:

```bash
ctest --test-dir build --output-on-failure -R 'ninfer_(structured_output|sampling|speculative_round|openai_schema|openai_responses|anthropic_schema|cli_options|prompt_input|tool_call_parser|qwen3_5_frontend|qwen3_5_structured_round)_test$'
```

`tests/test_structured_output_live.py` runs a temporary loopback server and validates completed
outputs with the independent Python `jsonschema` validator. Use Python 3.11 with `requests` and
`jsonschema` installed in a test environment. Supply an explicit v3 artifact containing the
selected speculative backends; the test never downloads or converts weights.

```bash
python tests/test_structured_output_live.py \
  --server build/apps/ninfer-serve --artifact /absolute/path/model.ninfer \
  --output-dir work/structured-live --modes none mtp dflash2 dflash2-eager
```

It checks conflicting prompts, greedy and stochastic generation, schema versus JSON object mode,
concurrent and mixed traffic, prefix reuse, SSE, token limits, disconnect cleanup, compile errors,
Responses, and Anthropic Messages. It also runs a tool-result-to-bounded-JSON exchange while
reasoning and multiple tools remain enabled, including a Markdown-formatted example in the prompt.
CPU tests cover numeric endpoints, schema enforcement with tool alternatives, split and mixed-token
reasoning transitions, speculative mask previews, injected thinking-budget closure, and prompt cache
boundary preservation. `--concurrency 8 --draft-tokens 15 --modes dflash2` exercises
the largest draft and batch dimensions. A separate DFlash-capable artifact can use `--modes dflash`.
The server is terminated on success or failure. An occupied test port causes the test to stop.

The old/new Qwen3.8 binding matrix uses `out/qwen3_8_27b_old.ninfer` and
`out/qwen3_8_27b_nvfp4_old.ninfer` for the legacy inventories, and the canonical filenames above
for the companion artifacts. The legacy paths may be overridden with
`NINFER_QWEN3_8_27B_OLD_WEIGHTS` and `NINFER_QWEN3_8_27B_NVFP4_OLD_WEIGHTS`.

## Container build cache

Check the Dockerfile's incremental-build and configuration-invalidation behavior without compiling
or running NInfer:

```bash
bash tests/test_docker_build_cache.sh docker
```

This uses the real build stage with a tiny CMake fixture and isolated cache mounts. It checks added
and removed compiler flags, changed and restored cached defaults, environment flags, old header
timestamps, non-code edits, and one-file incremental compilation. Docker BuildKit is required;
`podman` can be passed instead to check that builder. Logs and the fixture remain in an ignored
`build-cache-test.*` directory; the test image and its small build caches remain in the builder.
The check needs the Dockerfile's build dependencies but no GPU or model weights.
