# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype int8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `nvfp4`, and `k8v4`.

All seven have been measured on this corpus; the results, alongside each format's size and decode
speed, are in [`docs/config-calculator.html`](config-calculator.html).
`--fast-prefill-kernel` scores `int8` with the fast prompt-attention kernel (as
`ninfer-serve --fast-prefill-kernel` prefills); `report.json` records it as `fast_prefill_kernel`.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

`--disjoint` replaces the sliding windows with back-to-back windows of `--context` tokens, each
scored from its second token on, and drops a partial window at the end. That is the protocol of
the WikiText-2 perplexities quantization papers and model cards quote (GPTQ lineage: the test rows
joined by blank lines, 2,048-token windows). With the test split written to a file:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_gsq_rco_iq3_s.ninfer \
  --text wiki.test.joined.txt --context 2048 --disjoint --kv-dtype bf16
```

Past the model's native window `--rope-yarn` applies YaRN at factor `--context` / native, and
`--rope-yarn-factor F` at a fixed factor in `[1,4]` whatever the window is (default `1`, native
RoPE). `--rope-scaling-factor F` instead interpolates positions past
`--rope-scaling-original-context` (default: the native window) linearly by `F` in `[1,32]`, keeping
the angles of every position up to it. None changes the artifact or the default 4,096-token window;
long-context extrapolation is not a quality guarantee, so keep the factor fixed while comparing other
numerical settings. The report records all four settings.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v4 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results; its execution configuration
records `rope_yarn`, `rope_yarn_factor`, `rope_scaling_factor` and `rope_scaling_original_context`.
