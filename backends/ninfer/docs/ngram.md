# Ngram copy proposals

Ngram proposals accelerate repeated text, such as copying source code from a tool result into
an edit. They supplement the selected neural drafter; they are not an external knowledge store,
a model quantization mode, or a change to thinking settings.

## Defaults

Copy drafting is on whenever a drafter is: with `--spec mtp|dflash|dflash2` the CLI, the server
and `ninfer_bench` default to `--ngram-draft-tokens 15 --ngram-min-match 12`. Fifteen is the
widest copy a multi-request batch verifies, so the default holds at every `--max-concurrency`.
`--ngram-draft-tokens 0` turns it off; without `--spec` it stays off.

```sh
build/apps/ninfer-serve /path/to/model.ninfer \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

runs MTP with three neural drafts and fifteen-token copies, and

```sh
build/apps/ninfer-serve /path/to/model.ninfer \
  --max-concurrency 1 --spec mtp --draft-tokens 3 --lm-head-draft --ngram-draft-tokens 31
```

widens the copies for a single-request server.

| Selected backend | Neural draft tokens | Ngram draft tokens (C=1) | Ngram draft tokens (C>1) |
|---|---:|---:|---:|
| MTP | 1..15 | 1..63 | 1..15 |
| DFlash | 1..15 | 1..63 | 1..15 |
| DFlash2 | 1..15 | 1..63 | 1..15 |

The artifact must contain the selected drafter; ngram does not convert one drafter into
another. The minimum match defaults to 12 and its supported range is 4..64.

At `--max-concurrency > 1` each round is a batch of up to `--max-concurrency` active requests.
The GDN conv-record workspace behind the recurrent state admits at most 16 verification columns
once the batch holds more than one request, so a verify width above 15 is admitted only at
`--max-concurrency 1`. This caps the ngram width the same way for every backend.

`--lookup-ngram N` is a different, older source: it replaces an MTP round's neural drafts with
what followed the last `N` tokens earlier in the sequence, inside the neural window. Copy
proposals match at least `--ngram-min-match` tokens, reach past the neural window and draw on
tool results and retained sessions as well; the two can be combined.

## Round shapes

A round in which at least one row has a copy proposal is a copy round and verifies at the ngram
window; every other round keeps the neural drafter's own width, including the narrower widths
adaptive MTP selects. In a multi-request copy round a row without a match verifies its neural
proposal inside the wider round. The decode frame, replay records and target workspace are
planned for the wider of the two windows, and a round packs its rows at the width it verified.
DFlash and DFlash2 run the drafter in a multi-request copy round too, so a row without a match
keeps its neural proposal, and retain an append buffer sized for the widest round: a neural round
catches up the target features a preceding copy round left behind.

Copy rounds capture their own CUDA Graph family at the ngram window, next to the neural
families; the planned graph allowance and workspace cover both, so enabling copies costs device
memory that the KV budget no longer gets. Larger widths can reduce target rounds on long copy
spans but increase per-round attention, projection, replay and workspace costs, so measure both
short and long contexts; the longest supported width need not be fastest.

The mixed-FP8 27B target retains 16-bit activations for FP8 residual projections
during 17..64-column single-request verification. Its ordinary narrow path uses
that precision already; crossing the core's 22/25-column A8 thresholds otherwise
adds another quantization change. Weight and KV formats and prefill are unchanged.
This does not promise identical floating-point results between different widths.

## Operation

Each request owns a bounded CPU index of source positions. Lookups use 16-, 8- and 4-token
suffixes, compare hits exactly and extend matches backward through up to 64 committed
tokens to select a source. A proposal
copies a contiguous span from that source. Hash collisions and overwritten positions cannot
license target output. Short or absent matches fall back to the selected neural drafter.

Source endings can need different surrounding text, such as tool-call framing.
Admission looks ahead by two extra CPU tokens, leaving the last source token beyond
the target's bonus for a neural round. Useful partial spans remain eligible; a short
tail offering no more drafts than the neural width falls back to that provider.
This policy does not inspect whitespace, token identities or client names, and adds
no GPU graph family. It is not a guarantee of correct file contents: every output
still needs the same caller-side validation as ordinary generation.

The index includes the prepared prompt and committed output, never rejected or pending
verification columns. Known special tokens end source spans. The default index has about
36 MiB of token and bucket storage per active request, plus bounded proposal-only sources
in the prepared prompt. This index is released with the request. Optional session retention
uses a separate bounded CPU archive, described below; unrelated clients never share it.
The prompt is indexed before derived tool sources. The larger index retains useful old
spans under diverse long-context pressure; bounded storage still does not guarantee a hit
for every previous span.

Tool results containing at least three consecutively numbered lines can provide an additional
de-numbered source. Recognized separators include colon-space, tab, an arrow and pipe-space;
an optional `L` prefix is supported. Indentation, CRLF and terminal newlines are preserved.
The original tool message, target tokens, positions and prefix-cache identity are unchanged.
Plain tool results and earlier generated edits can match directly without normalization.
This is protocol-neutral and does not require Kilo-, Codex- or Claude-specific engine paths.
Derived-source tokenization uses the engine's context budget and stops when that
budget is exhausted. This only limits optional proposal sources: it never truncates
the target prompt to make an oversized request valid. Preparation cancellation
is checked around source tokenization.

Every proposed token is verified by the target. MTP and DFlash retain their deterministic-draft
acceptance paths; DFlash2 represents the proposal as a one-hot distribution in its existing
sparse verifier. Existing target commit, stop, cancellation and cache-restore ownership remain
authoritative. Ordinary neural rounds keep their original verification width.

This implementation is position-indexed, not a literal port of llama.cpp's hashed next-token
chain. It does not promise bit-identical output across different verification widths: floating
point execution and quantized projection choices can depend on width, just as in other batched
or speculative decoding configurations. This caveat does not permit using unverified output
or publishing speculative state before commit.

## Retain Sources Through Compaction

For a local single-owner server, opt in to a RAM-only archive:

```sh
--ngram-archive-mib 512 --ngram-session-mib 128
```

Requests identify a conversation with `X-NInfer-Draft-Session`. Without an identity,
drafting remains request-local. Add `--ngram-native-sessions` to recognize Kilo's
`x-session-affinity`, Codex's `client_metadata.thread_id`, and Claude's JSON-encoded
`metadata.user_id.session_id`. This separate local opt-in permits draft retention
even when the client sends `store:false`; it does not enable Responses-object storage.
Conflicting or invalid identities fall back to request-local drafting. Session names
are routing identifiers inside the existing local trust boundary, not authentication.

The Engine fixes an immutable view at admission and publishes validated input and
committed output after successful generation. Cancellation and failure discard the
private overlay. Retention is independent of GPU/host prefix-cache survival. Retained
tokens never enter the target prompt or KV: they can accelerate a continuation the
target already considers likely, not recover knowledge absent from its context.

Retention works at any `--max-concurrency`. Each request takes its own immutable snapshot at
admission, and archive mutations stay serialized on the engine worker, so a concurrent request
never observes another request's partial publication and unrelated identities never share sources.
A single session admits one binding at a time: if two requests name the same conversation identity
concurrently, one binds and the other falls back to request-local drafting; both still generate
correctly.

`X-NInfer-Draft-Reset: 1` clears the identified archive before a request and revokes
its old views/cursors. Restarting the Engine clears all archives. A new identity starts
empty. Explicit-header clients can fork the parent's latest completed generation using
`X-NInfer-Draft-Parent` and `X-NInfer-Draft-Generation`; a missing/stale parent or an
already existing destination does not merge histories. Native client forks receive
their own identity; inheritance is not guessed from similar prompts. Deleting a stored
Responses object is separate from clearing a draft session.
Generation numbers are opaque, Engine-lifetime identifiers, not per-session turn
counts. Clearing or evicting a session never recycles its number; clients must
discard generation references when the server restarts.

A successful publication reports its completed generation in the non-streaming
`X-NInfer-Draft-Generation` response header. Streaming responses put the standard SSE
comment `: ninfer-draft-generation: N` before their terminal event. Ordinary SSE
clients ignore this comment; explicit-fork clients can read it without access to
server logs. No generation is advertised for unbound or unpublished requests.

Immutable spans use exact-checked 16/8/4-token position indexes. Unchanged sources
deduplicate, and a validated copy cursor avoids repeat lookup while a span continues.
The cursor rechecks committed correction/bonus/control tokens and uses the same
source-ending handoff as request-local drafting. The archive is bounded by session
and Engine capacities, including indexes, metadata, staging and still-pinned sources;
pressure can evict sources or bypass retention without failing inference. It allocates
no VRAM and creates no archive files. The existing per-request index is additional.

Retention-enabled requests mix the supplied sampling seed with fresh request entropy.
This prevents retained generated proposals from depending on the same random draws
used to accept them during a retry. A seed alone does not reproduce an archived
conversation; disable retention for deterministic same-seed replay. Greedy target
selection is unaffected by this random-domain separation. The generation result
and structured request log expose the effective seed as `ngram_archive.sampling_seed`;
the ordinary request seed remains the caller's input. Reconstructing the same
proposal path additionally requires the same archive snapshot and model geometry.

## Evaluate

Compare ngram off (`--ngram-draft-tokens 0`) and on with the same binary, artifact, precision, sampling, prompt, output
budget and context-cache state. Verify output correctness before reporting speed. Measure raw
file copies, numbered tool-file copies, edits with insertions/deletions, and non-copying reasoning
separately. Include short and near-capacity contexts and both cold and cached prompts.

Report computed prefill tokens separately from reused tokens. Decode throughput excludes the
first prefill-sampled output token; whole-request latency includes prefill and cache restoration.
Building the CPU index and derived sources adds host work, especially visible on short prefills.
Faster copies do not imply faster prefill or the same improvement across an entire coding turn.
Reasoning, novel code and short matches can leave little reusable text.

Request statistics include `ngram_rounds`, `ngram_drafted_tokens` and
`ngram_accepted_tokens` in addition to aggregate speculative statistics. A useful performance
test must demonstrate accepted ngram proposals, not just successfully enabling an option.
The product `ninfer_bench` accepts the same ngram flags and preserves these
counters in table, CSV and JSON reports. Its ordinary one-token decode seed is
not a copy workload; see [benchmark guidance](../bench/README.md).

The [scope proposal](https://github.com/Neroued/ninfer/issues/234) links RTX 5090
copy, edit and compaction measurements with their artifacts, toolchain, commands
and correctness checks. Those pre-v3 results belong to the source revisions in
the [evidence release](https://github.com/remesis/ninfer/releases/tag/ngram-mod-evidence-20260912),
not to later rebases. Remeasure the relevant workload before attributing the same
gain to another revision or configuration.

## Real-Artifact Checks

These programs run from the test bundle, for example
`build/tests/ninfer_tests ninfer_ngram_lifecycle_real 63 5 1 1 nvfp4 12 0 mtp`.

`ninfer_ngram_lifecycle_real` uses `NINFER_NGRAM_TEST_WEIGHTS`. For example,
`63 5 1 1 nvfp4 12 0 mtp` selects NG63, neural K5, CUDA Graphs, optimized proposal
head, NVFP4 KV, minimum match 12, no additional soak and MTP. It checks retained
device versus host-restored state, cancellation, queued isolation, output budgets,
partial pages and context tails. Append `--strict-fresh` to additionally require
bit-identical fresh-prefill and cached output. That separate diagnostic may return
2 on quantized paths, including upstream without ngram; all retained-state checks
remain mandatory. It skips without the artifact environment variable.

`ninfer_ngram_archive_real` uses `NINFER_NGRAM_TEST_WEIGHTS` and takes the backend
(`mtp`, `dflash` or `dflash2`), optional graph mode (`0` or `1`) and optional
concurrency (`1..8`). It exercises NG63/K5 source-absent regeneration, isolated and
explicitly forked sessions, reset, cancellation/retry and publication generations with
prefix reuse disabled. With concurrency above one it also runs concurrent requests on
distinct sessions, two concurrent forks of one source, a same-identity concurrent pair
(one binds, one falls back) and a multi-round concurrent soak, asserting exact output
on every lane. This greedy, thinking-disabled contract fixture is not an
agent-performance benchmark. It skips without the artifact environment variable.

`ninfer_ngram_thinking_real` uses `NINFER_NGRAM_TEST_WEIGHTS` and accepts a backend,
neural draft size and ngram draft size, for example `mtp 5 15`. It checks output and
thinking budgets, forced control-token transitions, exact source-prefix answers,
positive ngram acceptance, and token-identical consecutive cached repeats.
It skips when the artifact environment variable is absent.

The optional fourth argument `--strict-fresh` additionally requires bit-identical
fresh-versus-cached generation and returns 2 when that diagnostic differs. Quantized
prefill/cache paths can already differ on upstream without ngram, including in
reasoning text; this is distinct from cached-repeat determinism and budget correctness.
Keep that diagnostic separate when comparing upstream and patched engines.

`ninfer_ngram_stop_chat_real` uses the same artifact environment variable and accepts
a backend and ngram draft size, for example `dflash2 15`. Neural draft size is five.
It uses canonical assistant-prefilled copy requests, stops inside speculative rounds,
and compares repeated retained continuations with equivalent output-limited ones.
It requires exact source-prefix text, complete prefix reuse and valid output-budget
and finish-reason accounting. An optional `--strict-fresh` argument retains
the separate fresh-prefill identity diagnostic; natural stop tokens are allowed to
end an answer before its maximum output budget. `--no-cuda-graph` selects eager
execution. This test also skips without an artifact.

`ninfer_ngram_concurrent_real` uses `NINFER_NGRAM_TEST_WEIGHTS` and takes a backend
(`mtp`, `dflash` or `dflash2`), an ngram draft width, and a concurrency. For example,
`dflash2 15 5` matches the shipped DFlash2 serving configuration. It checks that a
single-lane run on a concurrency>1 engine is token-identical to a graph-mode
single-request reference, that two concurrent copy lanes each reproduce an exact
source prefix over a long soak, and that two concurrent free-form lanes stay
non-degenerate over hundreds of tokens. Passing width `0` runs the free-form pair with
ngram disabled as a baseline. Cross-lane token identity is deliberately not required:
batched lanes are prefetched independently and need not share every round, so a
batch-size change alone can move a greedy near-tie (the ngram-disabled baseline
diverges the same way). `NINFER_NGRAM_TEST_MAX_CONTEXT` (default 4096) and
`NINFER_NGRAM_TEST_NO_GRAPH` adjust the shared-GPU footprint and graph mode.
