"""Synthesized agentic workloads for the NInfer A/B benchmark.

Mirrors the SHAPE and SCALE of a logged agent request log (775 requests): long
multi-turn OpenAI chat-completions sessions with ~33 tools, thinking on, streaming; median prompt ~100K tokens (p25 ~57K, p75 ~154K);
completion median ~530 / p75 ~1800 tokens; overall prefix-cache hit rate ~91%.

Deterministic (seeded) so both A/B arms replay byte-identical requests.

generate_workloads() returns a list of *groups*; each group is a list of request
specs sent concurrently (bounded by the serve's --max-concurrency 2). Groups are
processed in order — a later group only starts after the earlier one completes —
which is how prefix-reuse and multi-turn scenarios build the cache the way real
agentic sessions do. Every request shares one ~82K-token session base (system +
tool definitions + long agentic history), so the prefix cache actually fires.

Token targets below are ACTUAL token counts. CHARS_PER_TOKEN was calibrated on the
2026-09-22 A/B run: the real Qwen tokenizer + chat template + 33 tool definitions
pack ~1.64x denser than the original chars/3.8 guess (est 86805 -> actual 141557;
ratio stable 1.627-1.645 across 7 requests), i.e. ~2.3 chars per real token. The
calibrated max-context for both arms is 160000, so the largest target is ~140K
actual, leaving >=12K headroom for output tokens (the earlier run's failure was
context_length_exceeded on requests whose true size was ~1.64x the estimate).

Scenarios:
  S1 sequential private continuation — 5 sequential requests on the shared base,
     ~85K -> 140K actual (the logged workload's dominant reuse path).
  S2 cold long prefill — one ~140K request with a different root prefix (measures
     raw prefill compute; reuse_path "root").
  S3 concurrent decode — 2 simultaneous requests (~95K actual each) with long outputs.
  S4 multi-turn ramp — 4 sequential requests growing ~90K -> 135K actual, each
     re-sending the full history (strict prefix extension per turn).
  S5 cache contention under concurrency — 2 simultaneous ~130K-actual requests
     (shared base + ~48K unique middles); combined working set (~260K KV tokens)
     exceeds the KV working set so cache state competes while both hit the base.
"""
import json
import random

N_TOOLS = 33
CHARS_PER_TOKEN = 2.325  # calibrated 2026-09-22: chars per REAL token for this mix.

BASE_TOKENS = 82000  # shared session base in ACTUAL tokens (~50K by the old guess).


SYSTEM_PROMPT = (
    "You are an expert software-engineering agent working in a local repository.\n"
    "You use tools to read, write, search, and execute code. Follow the user's "
    "instructions precisely and make small, correct, incremental changes. Before "
    "reporting done, verify changes with the project's build and test commands. "
    "Reason carefully about edge cases, correctness, and performance.\n\n"
    "Ground rules:\n"
    "- Always read a file before editing it; do not guess its contents.\n"
    "- Do not invent command output or file contents; use the tools to confirm.\n"
    "- Prefer idiomatic, minimal diffs that match the project's conventions.\n"
    "- When a change is non-trivial, explain the approach before implementing.\n"
    "- Keep the user informed of progress and stop to clarify on ambiguity.\n"
    "- Security first: never log or commit secrets or sensitive values.\n"
    "- Target correctness and clarity; benchmark only where performance matters.\n"
    "- Final answers are concise: lead with the outcome, then supporting detail.\n"
)

COLD_SYSTEM_PROMPT = (
    "You are a data-pipeline debugging agent for a distributed ETL platform.\n"
    "You inspect job manifests, partition layouts, and backfill logs to diagnose "
    "failures. Cite the exact manifest fields and partition keys that support each "
    "conclusion. Prefer the smallest safe remediation and validate it against the "
    "historical partition checksums before proposing a re-run.\n\n"
    "Ground rules:\n"
    "- Never modify a live partition; stage fixes on the shadow namespace first.\n"
    "- Cross-check every claim against the manifest and the partition index.\n"
    "- Report the blast radius (partitions, downstream jobs) before any action.\n"
    "- Keep remediation steps reversible and idempotent.\n"
)

_TOOL_DEFS = [
    ("read_file", "Read the contents of a file at an absolute path.",
     {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
    ("write_file", "Write text content to a file, creating or overwriting it.",
     {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    ("edit_file", "Replace an exact text span within a file.",
     {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
     ["old_string", "new_string"]),
    ("glob", "Find files matching a glob pattern.",
     {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    ("grep_search", "Search file contents using a regular expression.",
     {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}, ["pattern"]),
    ("run_shell_command", "Run a shell command and capture its output.",
     {"command": {"type": "string"}, "timeout": {"type": "integer"}, "directory": {"type": "string"}}, ["command"]),
    ("list_directory", "List the entries of a directory.", {"path": {"type": "string"}}, ["path"]),
    ("web_fetch", "Fetch a URL and extract the requested information.",
     {"url": {"type": "string"}, "prompt": {"type": "string"}}, ["url"]),
    ("web_search", "Search the web for current information.", {"query": {"type": "string"}}, ["query"]),
    ("manage_todos", "Create or update the session task list.",
     {"items": {"type": "array", "items": {"type": "string"}}}, ["items"]),
]
_FILLER_TOOLS = [
    "analyze_code", "refactor_module", "run_tests", "git_status", "git_diff", "git_commit",
    "create_branch", "apply_patch", "explain_code", "debug_error", "profile_perf", "profile_memory",
    "lint_check", "type_check", "format_code", "generate_docs", "audit_dependencies", "scan_secrets",
    "report_coverage", "build_project", "run_container", "check_environment", "tail_logs", "trace_stack",
    "run_benchmark", "compare_artifacts", "read_config", "write_config", "add_note", "read_notes",
    "search_notes", "archive_file",
]

_PROSE = [
    "the", "function", "returns", "a", "value", "computed", "from", "each", "element", "of", "the",
    "input", "array", "which", "should", "be", "validated", "before", "it", "is", "used", "in", "order",
    "to", "ensure", "correctness", "and", "reasonable", "performance", "across", "the", "whole", "module",
    "the", "implementation", "handles", "the", "relevant", "edge", "cases", "by", "checking", "the",
    "bounds", "first", "then", "applying", "the", "transform", "and", "finally", "returning", "the",
    "accumulated", "result", "so", "that", "callers", "can", "rely", "on", "a", "well", "defined",
    "contract", "with", "no", "hidden", "state", "mutations",
]


def build_tools():
    tools = []
    for name, desc, props, req in _TOOL_DEFS:
        tools.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    for name in _FILLER_TOOLS:
        if len(tools) >= N_TOOLS:
            break
        tools.append({"type": "function", "function": {
            "name": name, "description": "Agent capability helper: " + name.replace("_", " ") + ".",
            "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]}}})
    return tools[:N_TOOLS]


def _code(rng, n_lines):
    out = []
    for i in range(n_lines):
        out.append("def op_%d(x: int, y: int, acc: list) -> int:" % i)
        out.append("    v = x * %d + y // (%d)" % (i + 1, i + 1))
        out.append("    if v > %d:" % (i * 97))
        out.append("        acc.append(normalize(v, %d))" % i)
        out.append("    elif v < -%d:" % (i * 13))
        out.append("        acc.extend(clamp(v, %d, %d))" % (i, i + 2))
        out.append("    return reduce_step(v, acc, k=%d)" % i)
        out.append("")
    return "\n".join(out)


def _file(rng, n_lines):
    out = []
    for i in range(n_lines):
        tag = rng.choice(["alpha", "beta", "gamma", "delta", "epsilon"])
        out.append("// %s module, unit %d: wiring and bookkeeping" % (tag, i))
        out.append('const value_%d = compute_%s(%d, %d, "cfg_%d");' % (i, tag, i, i * 2 + 1, i))
        out.append("if (value_%d > THRESHOLD_%d) { handle_%s(%d); }" % (i, i, tag, i))
        out.append("else if (value_%d < 0) { recover_%s(%d); }" % (i, tag, i))
        out.append("")
    return "\n".join(out)


def _prose(rng, n_words):
    return " ".join(rng.choice(_PROSE) for _ in range(n_words))


def _assistant_toolcall(rng, i, tools):
    fn = rng.choice(tools)["function"]["name"]
    return {"role": "assistant",
            "content": "I'll use %s to inspect module_%d and confirm the current state." % (fn, i),
            "tool_calls": [{"id": "call_%04d" % i, "type": "function",
                            "function": {"name": fn,
                                         "arguments": json.dumps({"arg": "task_%d" % i,
                                                                  "path": "/repo/src/module_%d.py" % i,
                                                                  "limit": 200})}}]}


def _tool_result(rng, i):
    body = _code(rng, 50) if i % 2 == 0 else _file(rng, 50)
    return {"role": "tool", "tool_call_id": "call_%04d" % i, "content": body}


def _content_len(msgs):
    return sum(len(m.get("content") or "") for m in msgs)


def _fill(rng, tools, msgs, target_chars, start_idx):
    """Append agentic turns (assistant tool-call, tool result, assistant prose) until target_chars."""
    i = start_idx
    while _content_len(msgs) < target_chars:
        msgs.append(_assistant_toolcall(rng, i, tools))
        msgs.append(_tool_result(rng, i))
        msgs.append({"role": "assistant", "content": _prose(rng, 70)})
        i += 1
    return msgs


def build_base(rng, tools, target_chars):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Implement and verify the batched attention KV-cache reclaim path. Read the existing "
            "module first, then make a minimal correct change and run the affected tests. Keep the "
            "diff small and idiomatic, and do not change unrelated behavior.")},
    ]
    _fill(rng, tools, msgs, target_chars, 0)
    return msgs


def build_cold_base(rng, tools, target_chars):
    """A ~100K-token context with a DIFFERENT root prefix than the shared base (cold prefill)."""
    msgs = [
        {"role": "system", "content": COLD_SYSTEM_PROMPT},
        {"role": "user", "content": (
            "The nightly backfill for the orders pipeline failed on partition 2026-09-19 after the "
            "schema migration. Diagnose the root cause from the manifests and partition checksums, "
            "then propose the smallest safe remediation. %s" % _prose(rng, 120))},
    ]
    _fill(rng, tools, msgs, target_chars, 4000)
    return msgs


def make_request(rid, msgs, tools, max_tokens, temperature=1.0, top_p=0.95):
    return {"id": rid, "messages": msgs, "tools": tools,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "stream": True, "tool_choice": "auto"}


def _extend(rng, tools, base, target_tokens, start_idx, query):
    """base + initial user query, extended with agentic turns to ~target_tokens, ending with
    a final user query (the turn that elicits the response)."""
    chars = target_tokens * CHARS_PER_TOKEN
    msgs = list(base)
    msgs.append({"role": "user", "content": query})
    _fill(rng, tools, msgs, chars, start_idx)
    return msgs


def generate_workloads(seed=42):
    """Return a list of groups; each group is a list of request specs sent concurrently."""
    rng = random.Random(seed)
    tools = build_tools()
    base = build_base(rng, tools, int(BASE_TOKENS * CHARS_PER_TOKEN))  # ~82K-actual shared base
    groups = []

    # S1: sequential private continuation — 5 sequential requests on the shared base,
    # 85K -> 140K actual. Only the base (~82K actual) is cacheable across requests.
    s1_targets = [85000, 100000, 115000, 128000, 140000]
    for k, target in enumerate(s1_targets):
        msgs = _extend(rng, tools, base, target, 1000 + k * 500,
                       "Follow-up %d: reconcile the reclaim path against the %d-way concurrent "
                       "case and report the exact diff plus the affected tests." % (k, k + 1))
        groups.append([make_request(100 + k, msgs, tools, max_tokens=8192)])

    # S2: cold long prefill — one ~140K-actual request with a different root prefix (no reuse).
    cold = build_cold_base(rng, tools, int(140000 * CHARS_PER_TOKEN))
    cold.append({"role": "user", "content":
                 "Final step: confirm the remediation order and the partition keys to re-run, "
                 "with the checksum evidence for each."})
    groups.append([make_request(200, cold, tools, max_tokens=8192)])

    # S3: concurrent decode — 2 simultaneous ~95K-actual requests (distinct middles), long outputs.
    s3 = []
    for k in range(2):
        msgs = _extend(rng, tools, base, 95000, 3000 + k * 1000,
                       "Concurrent task %d: write a detailed end-to-end walkthrough of the "
                       "reclaim path and the failing tests, then the fix." % k)
        s3.append(make_request(300 + k, msgs, tools, max_tokens=16384))
    groups.append(s3)

    # S4: multi-turn ramp — 4 sequential requests; context grows 90K -> 135K actual and each
    # turn re-sends the full history (strict prefix extension: turn k is a prefix of turn k+1).
    running = list(base)
    for k, target in enumerate([90000, 105000, 120000, 135000]):
        running = _fill(rng, tools, running, int(target * CHARS_PER_TOKEN), 5000 + k * 1000)
        running.append({"role": "assistant", "content": _prose(rng, 120)})  # model's prior reply
        running.append({"role": "user", "content":
                        "Turn %d: continue and verify the %d-stage integration against the "
                        "live rig, then list the open risks." % (k, k + 1)})
        groups.append([make_request(400 + k, list(running), tools, max_tokens=8192)])

    # S5: cache contention under concurrency — 2 simultaneous ~130K-actual requests: shared
    # base + ~48K unique middles each. Combined working set (~260K KV tokens) exceeds the
    # working cache so the two middles compete for cache state while both hit the base.
    s5 = []
    for k in range(2):
        msgs = _extend(rng, tools, base, 130000, 7000 + k * 2000,
                       "Contended task %d: audit the interaction between the two concurrent "
                       "reclaim pipelines and the shared eviction policy, then recommend the "
                       "safest isolation change." % k)
        s5.append(make_request(500 + k, msgs, tools, max_tokens=8192))
    groups.append(s5)

    return groups


def summarize(groups):
    total = 0
    lines = []
    for g_i, group in enumerate(groups):
        for r in group:
            clen = _content_len(r["messages"])
            est = int(clen / CHARS_PER_TOKEN)
            total += est
            lines.append("group %2d: req %d  est_prompt_tokens~%-7d messages=%-4d max_tokens=%d"
                         % (g_i, r["id"], est, len(r["messages"]), r["max_tokens"]))
    lines.append("TOTAL: %d groups, %d requests, est_prompt_tokens~%d"
                 % (len(groups), sum(len(g) for g in groups), total))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(generate_workloads()))
