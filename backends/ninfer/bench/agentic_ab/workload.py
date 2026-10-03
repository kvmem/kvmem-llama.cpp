"""Deterministic agentic coding workload for the NInfer serve A/B suite.

The workload models what logged coding-agent traffic to one ninfer-serve looks like (~3,300
requests): long multi-turn tool loops whose
turns add a few hundred to a few thousand tokens to a 30K-200K conversation, periodic
context compaction, parallel subagent fan-outs that share a system-and-tools prefix,
cold side calls over the whole history, client retries and aborted requests, a
second and third session interleaved with the first, and `max_tokens: 64000` on every
agent turn, as the real clients send it.

Every observation the client sends (tool results, user messages, subagent reports,
compaction summaries) is generated here, ahead of time, from real source files, real
diffs and real commit history of this repository at a pinned commit, so both arms
receive byte-identical observations. The assistant turns are *not* scripted: the runner
feeds each arm's own model output back into the conversation, exactly as an agent client
does, so the engine's private-endpoint reuse (the dominant path for agent traffic) happens
the way it does for a real client. See README.md for the scenario and its rationale.

`build_plan()` returns a JSON-serialisable plan; `python workload.py` prints a summary.
"""
import hashlib
import json
import os
import random
import re
import subprocess

# ---------------------------------------------------------------------------------------
# Corpus: real files, diffs and history of this repository at a pinned commit.
# ---------------------------------------------------------------------------------------

REPO = os.environ.get("AB_CORPUS_REPO",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Pinned so the observations never change when the working tree does.
CORPUS_COMMIT = os.environ.get("AB_CORPUS_COMMIT", "34c53fac")

# Planning-only size estimate (the report uses the server's real token counts).
# Calibrated on this corpus with the Qwen3.8 tokenizer + chat template.
CHARS_PER_TOKEN = 3.1

AREAS = {
    # Session A: C++ engine work on the context cache.
    "engine": (r"^src/(runtime|core|serve|models/qwen3_5/program)/.*\.(cpp|h)$",
               ["src/runtime", "src/core", "src/serve", "src/models/qwen3_5/program"]),
    # Session B: Python tooling (the eval harness and artifact tools).
    "python": (r"^(eval|tools/artifact|tools/convert|tools/bench)/.*\.py$",
               ["eval", "tools/artifact", "tools/convert", "tools/bench"]),
    # Session C: tests and documentation.
    "tests": (r"^(tests/.*\.(cpp|h|py)|docs/.*\.md)$", ["tests", "docs"]),
}


def _git(*args):
    out = subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, check=True)
    return out.stdout.decode("utf-8", "replace")


class Corpus:
    def __init__(self, commit=CORPUS_COMMIT):
        self.commit = commit
        self._files = {}
        self._commits = {}
        names = _git("ls-tree", "-r", "--name-only", commit).splitlines()
        self.by_area = {a: sorted(n for n in names if re.match(rx, n))
                        for a, (rx, _) in AREAS.items()}
        for area, files in self.by_area.items():
            if not files:
                raise SystemExit("corpus area %r is empty at %s" % (area, commit))

    def text(self, path):
        if path not in self._files:
            self._files[path] = _git("show", "%s:%s" % (self.commit, path)).replace("\r\n", "\n")
        return self._files[path]

    def lines(self, path):
        return self.text(path).split("\n")

    def commits(self, area):
        if area not in self._commits:
            self._commits[area] = _git("log", "--format=%H", "-n", "300", self.commit, "--",
                                       *AREAS[area][1]).split()
        return self._commits[area]

    def diff(self, area, sha):
        return _git("show", "--stat", "--patch", "--format=commit %H%nAuthor: %an%nDate:   %ad%n%n    %s%n",
                    sha, "--", *AREAS[area][1]).replace("\r\n", "\n")

    def pick_file(self, rng, area, min_lines=40, max_lines=None):
        files = self.by_area[area]
        for _ in range(200):
            p = rng.choice(files)
            n = len(self.lines(p))
            if n >= min_lines and (max_lines is None or n <= max_lines):
                return p
        return rng.choice(files)


def est_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------------------
# Observations (tool results as an agent client would return them).
# Each generator returns (tool_name, arguments, text, execution_seconds).
# ---------------------------------------------------------------------------------------

def obs_read(c, rng, area, target_tok, path=None, whole=False):
    path = path or c.pick_file(rng, area, min_lines=60)
    lines = c.lines(path)
    budget = int(target_tok * CHARS_PER_TOKEN)
    start = 0 if whole or len(lines) < 80 else rng.randrange(0, max(1, len(lines) // 2))
    out, used, i = [], 0, start
    while i < len(lines) and (whole or used < budget):
        row = "%6d\t%s" % (i + 1, lines[i])
        out.append(row)
        used += len(row) + 1
        i += 1
    head = "Showing lines %d-%d of %d total lines in %s\n" % (start + 1, i, len(lines), path)
    return ("read_file", {"path": "/work/ninfer/" + path, "offset": start + 1, "limit": i - start},
            head + "\n".join(out), rng.uniform(0.15, 0.5))


def _identifiers(text):
    return [w for w in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{6,}\b", text)
            if not w.isupper() and w not in ("include", "return", "static_cast", "noexcept")]


def obs_grep(c, rng, area, target_tok):
    src = c.pick_file(rng, area, min_lines=30)
    idents = _identifiers(c.text(src)) or ["context"]
    budget = int(target_tok * CHARS_PER_TOKEN)
    out, used, pats = [], 0, []
    for _ in range(4):
        pat = rng.choice(idents)
        pats.append(pat)
        for path in c.by_area[area]:
            for n, line in enumerate(c.lines(path), 1):
                if pat in line:
                    row = "%s:%d:%s" % (path, n, line.rstrip()[:220])
                    out.append(row)
                    used += len(row) + 1
                    if used >= budget:
                        break
            if used >= budget:
                break
        if used >= budget * 0.6:
            break
    text = "\n".join(out) if out else "No matches found"
    return ("grep", {"pattern": "|".join(pats), "path": "/work/ninfer"}, text,
            rng.uniform(0.2, 0.8))


def obs_glob(c, rng, area, target_tok):
    files = c.by_area[area]
    k = max(5, min(len(files), int(target_tok * CHARS_PER_TOKEN / 48)))
    start = rng.randrange(0, max(1, len(files) - k))
    text = "\n".join("/work/ninfer/" + f for f in files[start:start + k])
    return ("glob", {"pattern": AREAS[area][1][0] + "/**/*"}, text, rng.uniform(0.1, 0.3))


def obs_diff(c, rng, area, target_tok):
    budget = int(target_tok * CHARS_PER_TOKEN)
    for _ in range(20):
        sha = rng.choice(c.commits(area))
        text = c.diff(area, sha)
        if len(text) > budget // 3:
            break
    return ("run_shell_command", {"command": "git show %s" % sha[:10]}, text[:budget],
            rng.uniform(0.3, 0.9))


def obs_edit_ack(c, rng, area, path=None):
    path = path or c.pick_file(rng, area, min_lines=40)
    lines = c.lines(path)
    s = rng.randrange(0, max(1, len(lines) - 25))
    snippet = "\n".join("%6d\t%s" % (i + 1, lines[i]) for i in range(s, min(len(lines), s + 22)))
    text = ("The file /work/ninfer/%s has been updated. Here's the result of running `cat -n` on "
            "a snippet of the edited file:\n%s" % (path, snippet))
    return ("edit_file", {"path": "/work/ninfer/" + path, "old_string": lines[s][:80],
                          "new_string": lines[s][:80]}, text, rng.uniform(0.1, 0.3))


def obs_build(c, rng, area, target_tok, fail=None):
    """A build/test run: real target and file names, a real code excerpt for a failure."""
    budget = int(target_tok * CHARS_PER_TOKEN)
    files = c.by_area[area]
    fail = rng.random() < 0.35 if fail is None else fail
    out = []
    if area == "python":
        out.append("$ python -m pytest -q eval/tests")
        out.append("============================= test session starts =============================")
        out.append("platform linux -- Python 3.11.9, pytest-8.3.2, pluggy-1.5.0")
        out.append("rootdir: /work/ninfer")
        tests = [f for f in files if "/tests/" in f] or files
        for i in range(10_000):
            f = tests[i % len(tests)]
            row = "%s %s" % (f, "".join(rng.choice("......F." if fail else "........")
                                        for _ in range(rng.randrange(4, 30))))
            out.append(row)
            if sum(len(x) for x in out) > budget * 0.55:
                break
    else:
        out.append("$ cmake --build build -j --target ninfer_runtime_tests ninfer-serve && "
                   "ctest --test-dir build -R context_cache --output-on-failure")
        cpp = [f for f in files if f.endswith((".cpp", ".cu"))] or files
        for i in range(10_000):
            f = cpp[(i * 7) % len(cpp)]
            out.append("[%3d%%] Building CXX object %s.o" % (min(99, i * 3), f.replace("src/", "src/CMakeFiles/")))
            if sum(len(x) for x in out) > budget * 0.45:
                break
        out.append("[100%] Linking CXX executable tests/ninfer_runtime_tests")
        out.append("Test project /work/ninfer/build")
        for i in range(rng.randrange(8, 30)):
            out.append("      Start %2d: context_cache.case_%02d" % (i + 1, i))
            out.append("%2d/%2d Test #%2d: context_cache.case_%02d ..........   Passed    %.2f sec"
                       % (i + 1, 30, i + 1, i, rng.uniform(0.01, 2.5)))
    if fail:
        path = c.pick_file(rng, area, min_lines=60)
        lines = c.lines(path)
        s = rng.randrange(0, max(1, len(lines) - 30))
        out.append("")
        out.append("FAILED %s:%d - AssertionError: expected the retained checkpoint to survive "
                   "host demotion" % (path, s + 12))
        out.append("---------- captured context ----------")
        for i in range(s, min(len(lines), s + 30)):
            out.append("%6d  %s" % (i + 1, lines[i]))
        out.append("1 failed, %d passed in %.2fs" % (rng.randrange(40, 300), rng.uniform(4, 60)))
    else:
        out.append("100%% tests passed, 0 tests failed out of %d" % rng.randrange(20, 300))
    text = "\n".join(out)[:budget + 4000]
    return ("run_shell_command", {"command": out[0][2:]}, text, rng.uniform(2.0, 6.0))


def obs_big_log(c, rng, area, target_tok):
    """A long pasted log (the user pastes CI output)."""
    parts, used = [], 0
    budget = int(target_tok * CHARS_PER_TOKEN)
    while used < budget:
        _, _, t, _ = obs_build(c, rng, area, 2500, fail=rng.random() < 0.5)
        parts.append(t)
        used += len(t)
    return "\n".join(parts)[:budget]


# Observation mix per area: (generator, weight, token target range)
MIX = {
    "engine": [("read", 34, (700, 5200)), ("grep", 18, (250, 1500)), ("build", 14, (500, 2600)),
               ("diff", 10, (800, 3000)), ("edit", 18, None), ("glob", 6, (120, 400))],
    "python": [("read", 38, (600, 4200)), ("grep", 16, (200, 1200)), ("build", 18, (400, 2000)),
               ("diff", 6, (600, 2200)), ("edit", 17, None), ("glob", 5, (100, 350))],
    "tests":  [("read", 36, (700, 4800)), ("build", 22, (500, 2400)), ("grep", 14, (250, 1200)),
               ("edit", 22, None), ("glob", 6, (100, 350))],
}


def draw_obs(c, rng, area, kind=None):
    mix = MIX[area]
    if kind is None:
        kind = rng.choices([m[0] for m in mix], weights=[m[1] for m in mix])[0]
    rng_tok = dict((m[0], m[2]) for m in mix).get(kind) or (500, 2000)
    # log-uniform size within the range: many small results, a few large ones
    lo, hi = rng_tok
    target = int(lo * (hi / lo) ** rng.random())
    if kind == "read":
        return obs_read(c, rng, area, target)
    if kind == "grep":
        return obs_grep(c, rng, area, target)
    if kind == "build":
        return obs_build(c, rng, area, target)
    if kind == "diff":
        return obs_diff(c, rng, area, target)
    if kind == "glob":
        return obs_glob(c, rng, area, target)
    return obs_edit_ack(c, rng, area)


# ---------------------------------------------------------------------------------------
# Personas: system prompts and tool schemas of the agent clients seen in the logs.
# ---------------------------------------------------------------------------------------

def _tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_S = {"type": "string"}
_I = {"type": "integer"}
_B = {"type": "boolean"}

_CORE_TOOLS = {
    "read_file": _tool(
        "read_file",
        "Reads a file from the local filesystem. The path must be absolute. By default up to 2000 "
        "lines are returned from the start of the file; use offset and limit to page through long "
        "files. Output uses cat -n formatting with 1-based line numbers. Always read a file before "
        "editing it and prefer reading several related files in one turn when you already know "
        "which ones you need. Binary files are reported as such and not decoded.",
        {"path": dict(_S, description="Absolute path of the file to read."),
         "offset": dict(_I, description="1-based line to start reading from."),
         "limit": dict(_I, description="Maximum number of lines to return.")}, ["path"]),
    "write_file": _tool(
        "write_file",
        "Writes a file to the local filesystem, replacing any existing content. Read the file "
        "first when it already exists. Prefer edit_file for small changes; use write_file when "
        "most of the file changes or when creating a new file. Never create documentation files "
        "unless explicitly asked.",
        {"path": dict(_S, description="Absolute path of the file to write."),
         "content": dict(_S, description="Complete new content of the file.")},
        ["path", "content"]),
    "edit_file": _tool(
        "edit_file",
        "Performs an exact string replacement in a file. old_string must match the file exactly, "
        "including indentation, and must be unique unless replace_all is set. Include enough "
        "surrounding context to make the match unique. The file must have been read in this "
        "conversation first.",
        {"path": _S, "old_string": dict(_S, description="Exact text to replace."),
         "new_string": dict(_S, description="Replacement text."),
         "replace_all": dict(_B, description="Replace every occurrence.")},
        ["path", "old_string", "new_string"]),
    "glob": _tool(
        "glob",
        "Fast file pattern matching that works with any codebase size. Supports patterns such as "
        "**/*.cpp or src/**/*.h and returns matching paths sorted by modification time.",
        {"pattern": _S, "path": _S}, ["pattern"]),
    "grep": _tool(
        "grep",
        "Searches file contents with ripgrep-compatible regular expressions. Filter files with "
        "glob or type, choose output_mode content, files_with_matches or count, and use context "
        "lines to see surrounding code. Prefer this over running grep through the shell.",
        {"pattern": _S, "path": _S, "glob": _S, "output_mode": _S, "context": _I}, ["pattern"]),
    "list_directory": _tool(
        "list_directory", "Lists files and directories in a given absolute path.",
        {"path": _S, "ignore": {"type": "array", "items": _S}}, ["path"]),
    "run_shell_command": _tool(
        "run_shell_command",
        "Executes a shell command in a persistent session with an optional timeout. Use it for "
        "builds, tests, git and other terminal operations, not for reading or searching files. "
        "Quote paths that contain spaces, avoid interactive commands, and prefer absolute paths "
        "over changing directory. Output beyond 30000 characters is truncated. Long-running "
        "commands can be started in the background and polled with bash_output.",
        {"command": _S, "description": _S, "timeout": _I, "run_in_background": _B},
        ["command"]),
    "web_fetch": _tool(
        "web_fetch", "Fetches a URL, converts the page to markdown and answers the prompt about it.",
        {"url": _S, "prompt": _S}, ["url", "prompt"]),
}

_CC_EXTRA = [
    _tool("multi_edit", "Applies several exact string replacements to one file atomically; every "
          "edit must succeed or none is applied.",
          {"path": _S, "edits": {"type": "array", "items": {"type": "object", "properties": {
              "old_string": _S, "new_string": _S, "replace_all": _B}}}}, ["path", "edits"]),
    _tool("bash_output", "Retrieves new output from a background shell started with "
          "run_shell_command.", {"shell_id": _S, "filter": _S}, ["shell_id"]),
    _tool("kill_shell", "Terminates a background shell by id.", {"shell_id": _S}, ["shell_id"]),
    _tool("web_search", "Searches the web and returns result snippets with source links.",
          {"query": _S, "allowed_domains": {"type": "array", "items": _S}}, ["query"]),
    _tool("todo_write", "Creates and updates the structured task list for the current session. "
          "Use it for multi-step work so progress stays visible; keep exactly one item in "
          "progress and mark items completed as soon as they are done.",
          {"todos": {"type": "array", "items": {"type": "object", "properties": {
              "content": _S, "status": _S, "active_form": _S}}}}, ["todos"]),
    _tool("task", "Launches a subagent to handle a complex, multi-step search or analysis task "
          "autonomously. Several subagents can run concurrently; each returns one final report. "
          "Give each a detailed, self-contained prompt and say whether it should write code.",
          {"description": _S, "prompt": _S, "subagent_type": _S}, ["description", "prompt"]),
    _tool("notebook_edit", "Replaces, inserts or deletes a cell in a Jupyter notebook.",
          {"notebook_path": _S, "cell_id": _S, "new_source": _S, "edit_mode": _S},
          ["notebook_path", "new_source"]),
    _tool("ask_user", "Asks the user a multiple-choice question when a decision is genuinely "
          "theirs to make.", {"question": _S, "options": {"type": "array", "items": _S}},
          ["question", "options"]),
    _tool("save_memory", "Saves a durable fact about the user or project for future sessions.",
          {"fact": _S}, ["fact"]),
    _tool("exit_plan_mode", "Presents the implementation plan for approval and leaves plan "
          "mode.", {"plan": _S}, ["plan"]),
    _tool("git_status", "Shows the working tree status and the current branch.", {}, []),
    _tool("lsp_diagnostics", "Returns compiler and language-server diagnostics for a file.",
          {"path": _S}, ["path"]),
]

_QC_EXTRA = [
    _tool("read_many_files", "Reads several files or glob patterns at once and concatenates them "
          "with separators.", {"paths": {"type": "array", "items": _S}}, ["paths"]),
    _tool("google_web_search", "Performs a web search and returns a summary with citations.",
          {"query": _S}, ["query"]),
    _tool("save_memory", "Remembers a specific fact across sessions.", {"fact": _S}, ["fact"]),
    _tool("todo_write", "Maintains the session task list.",
          {"todos": {"type": "array", "items": {"type": "object"}}}, ["todos"]),
    _tool("task", "Delegates a self-contained task to a subagent and returns its final report.",
          {"description": _S, "prompt": _S, "subagent_type": _S}, ["description", "prompt"]),
    _tool("exit_plan_mode", "Presents a plan for approval.", {"plan": _S}, ["plan"]),
    _tool("skill", "Loads a named skill's instructions into the conversation.", {"name": _S},
          ["name"]),
    _tool("lsp", "Queries the language server: definition, references, hover, symbols.",
          {"operation": _S, "path": _S, "line": _I, "character": _I}, ["operation", "path"]),
    _tool("ask_user", "Asks the user a clarifying question.", {"question": _S}, ["question"]),
]

CC_INSTRUCTIONS = """You are Forge, an interactive command-line coding agent. You help the user with
software-engineering work in their local repository: fixing bugs, adding features, refactoring,
explaining code and running builds and tests. Use the tools available to you; never invent file
contents, command output or test results.

# Tone
Be concise and direct. Lead with the outcome, then the supporting detail. Do not narrate routine
tool use. When you reference code, use `path:line` so the user can navigate to it. Do not add
praise, apologies or filler.

# Doing the work
- Understand before you change anything: read the relevant files and search for the callers,
  tests and documentation that the change affects.
- Plan multi-step work with todo_write and keep the list current.
- Match the surrounding code: naming, comment density, error handling and formatting.
- Keep diffs focused on the task. Do not refactor unrelated code or add speculative abstractions.
- After a change, build the affected targets and run the affected tests. Report failures
  faithfully with the relevant output; never claim success you have not verified.
- Prefer edit_file for targeted changes and write_file when most of a file changes.
- When a task needs broad exploration, launch several task subagents in parallel with
  self-contained prompts, then synthesise their reports.

# Safety
- Never commit, push or rewrite history unless the user asks.
- Never print, log or commit secrets.
- Ask before destructive operations such as deleting files or resetting branches.

# Tool use
- Batch independent tool calls in one turn.
- Use grep and glob for searching; do not run grep, find or cat through the shell.
- Long builds may be run in the background and polled with bash_output.

# Environment
Working directory: /work/ninfer
Platform: linux (x86_64), CUDA 13.1, GCC 14
Today's date: 2026-09-24
Git: branch fix/context-cache-demotion, 3 files modified
"""

QC_INSTRUCTIONS = """You are an interactive CLI agent specializing in software engineering tasks. Your
primary goal is to help users safely and efficiently, adhering strictly to the following
instructions and utilizing your available tools.

# Core Mandates
- Conventions: rigorously adhere to existing project conventions. Analyze surrounding code,
  tests and configuration first.
- Libraries: never assume a library is available; verify its usage in the project first.
- Style and structure: mimic the style, structure, typing and architectural patterns of the
  existing code.
- Comments: add comments sparingly, focusing on why rather than what.
- Proactiveness: fulfil the request thoroughly, including directly implied follow-up actions.
- Explaining changes: do not summarize changes unless asked.

# Primary Workflow
1. Understand: use search_file_content and glob extensively, then read_file to validate
   assumptions.
2. Plan: build a coherent, grounded plan and share a concise version with the user.
3. Implement: use the available tools, strictly adhering to the core mandates.
4. Verify (tests): identify the project's testing procedures and run them.
5. Verify (standards): run the build, linting and type-checking commands for the project.

# Operational Guidelines
- Tone: concise and direct; fewer than three lines of text per response when practical.
- Use tools for actions and text only for communication.
- Explain critical shell commands that modify the file system before running them.
- Run independent searches in parallel.

# Environment
Working directory: /work/ninfer (Python 3.11 virtualenv active)
Today's date: 2026-09-24
"""

SUB_INSTRUCTIONS = """You are a read-only exploration subagent launched by a coding agent. Your job is
to answer one self-contained research question about the repository as completely and precisely
as you can, then return a single final report.

Rules:
- You cannot modify files. Use read_file, grep, glob and list_directory to investigate, and
  run_shell_command only for read-only commands such as git log or git show.
- Search broadly first, then read the most relevant files in full.
- Cite every claim with `path:line`.
- Your final message is the only thing the parent agent sees: include every finding, the exact
  locations, and any uncertainty. Do not ask questions; make reasonable assumptions and state them.
"""

REV_INSTRUCTIONS = """You are a code-review subagent. Review the change described by the parent agent for
correctness bugs, numerical issues, lifetime and concurrency hazards, and violations of the
project's contracts. Only report findings you can support with code evidence (`path:line`), rank
them by severity, and say for each what input or state makes it fail. Do not propose style
changes. Return one final report.
"""

COMPACT_SYSTEM = """You are a helpful AI assistant tasked with summarizing conversations between a user
and a coding agent so the agent can continue the work in a fresh context."""

CHECK_SYSTEM = """You analyze the recent history of an autonomous coding agent to decide whether it is
stuck in an unproductive loop (repeating the same tool calls, oscillating edits, or making no
progress) and who should speak next. Answer with a single JSON object:
{"reasoning": string, "is_loop": boolean, "next_speaker": "user" | "model"}."""


def personas(c):
    """Return {name: {"system": str, "tools": [..]}} built from the pinned corpus."""
    agents = c.text("AGENTS.md")
    arch = c.text("docs/maintainer/engine-architecture.md")
    contributing = c.text("CONTRIBUTING.md")
    cli = c.text("docs/cli.md")
    docmap = c.text("docs/README.md")
    opdev = c.text("docs/maintainer/op-development.md")
    core = _CORE_TOOLS
    cc_tools = [core[k] for k in ("read_file", "write_file", "edit_file", "glob", "grep",
                                  "list_directory", "run_shell_command", "web_fetch")] + _CC_EXTRA
    qc_tools = [core[k] for k in ("read_file", "write_file", "edit_file", "glob", "grep",
                                  "list_directory", "run_shell_command", "web_fetch")] + _QC_EXTRA
    sub_tools = [core[k] for k in ("read_file", "glob", "grep", "list_directory",
                                   "run_shell_command", "web_fetch")]
    rev_tools = sub_tools[:5] + [_tool("report_finding", "Records one review finding.",
                                       {"path": _S, "line": _I, "severity": _S, "summary": _S},
                                       ["path", "summary"])]
    return {
        "cc": {"system": CC_INSTRUCTIONS + "\n# Project instructions (AGENTS.md)\n\n" + agents +
               "\n# Architecture notes (docs/maintainer/engine-architecture.md)\n\n" + arch,
               "tools": cc_tools},
        "qc": {"system": QC_INSTRUCTIONS + "\n# Project context (CONTRIBUTING.md)\n\n" +
               contributing + "\n# CLI reference (docs/cli.md)\n\n" + cli,
               "tools": qc_tools},
        "sub": {"system": SUB_INSTRUCTIONS + "\n# Project instructions (AGENTS.md)\n\n" + agents +
                "\n# Documentation map\n\n" + docmap +
                "\n# Architecture notes\n\n" + arch,
                "tools": sub_tools},
        "rev": {"system": REV_INSTRUCTIONS + "\n# Project instructions (AGENTS.md)\n\n" + agents +
                "\n# Op development contract\n\n" + opdev,
                "tools": rev_tools},
        "compact": {"system": COMPACT_SYSTEM, "tools": []},
        "check": {"system": CHECK_SYSTEM, "tools": []},
    }


# ---------------------------------------------------------------------------------------
# History building (resumed sessions start from a saved conversation, as after a restart).
# ---------------------------------------------------------------------------------------

_THOUGHTS = [
    "I need to see how %s is handled before changing anything.",
    "Let me check the callers of %s and the tests that cover it.",
    "The failure points at %s; reading the surrounding code.",
    "Before editing I want to confirm how %s interacts with eviction.",
    "Checking whether %s already has coverage in the test suite.",
]


def prior_turn(c, rng, area, idx, tool_names):
    """One completed tool round trip of a resumed session: assistant call + tool result."""
    name, args, text, _ = draw_obs(c, rng, area)
    if name not in tool_names:
        name = "run_shell_command" if "run_shell_command" in tool_names else tool_names[0]
    subject = args.get("path") or args.get("pattern") or args.get("command") or "this module"
    call_id = "call_%s" % hashlib.sha1(("%s:%d" % (area, idx)).encode()).hexdigest()[:16]
    assistant = {"role": "assistant", "content": "",
                 "reasoning_content": rng.choice(_THOUGHTS) % str(subject)[:120],
                 "tool_calls": [{"id": call_id, "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}}]}
    return [assistant, {"role": "tool", "tool_call_id": call_id, "content": text}]


def messages_tokens(msgs):
    n = 0
    for m in msgs:
        n += est_tokens(m.get("content") or "") + est_tokens(m.get("reasoning_content") or "")
        for tc in m.get("tool_calls") or []:
            n += est_tokens(tc["function"]["arguments"]) + 8
        n += 6
    return n


def tools_tokens(tools):
    return est_tokens(json.dumps(tools))


def build_resumed(c, rng, persona, area, task, target_tokens, tag):
    msgs = [{"role": "system", "content": persona["system"]}, {"role": "user", "content": task}]
    names = [t["function"]["name"] for t in persona["tools"]]
    base = tools_tokens(persona["tools"])
    i = 0
    while base + messages_tokens(msgs) < target_tokens:
        msgs += prior_turn(c, rng, area, _seed(tag) % 100000 + i, names)
        if i % 9 == 8:
            msgs.append({"role": "assistant", "content":
                         "Progress so far: the reclaim path is narrowed down; next I will verify the "
                         "remaining callers and run the affected tests."})
            msgs.append({"role": "user", "content": "ok, keep going"})
        i += 1
    return msgs


# ---------------------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------------------

def _seed(*parts):
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def _loop(c, rng, area, n, start, copy_at=(), edit_at=(), extra=None):
    """n ordinary agent-loop turns; copy_at/edit_at insert copy-heavy file-writing turns."""
    steps = []
    for k in range(n):
        idx = start + k
        if k in copy_at:
            path = c.pick_file(rng, area, min_lines=90, max_lines=230)
            _, _, text, delay = obs_read(c, rng, area, 0, path=path, whole=True)
            fn = rng.choice(_identifiers(c.text(path)) or ["the main function"])
            steps.append({"op": "request", "cls": "loop", "copy": True, "obs": text, "delay": delay,
                          "user": ("Looks right. Apply it now: call write_file with the COMPLETE "
                                   "updated contents of /work/ninfer/%s. Keep every other line "
                                   "byte-identical; only add a short comment above the first use "
                                   "of %s explaining its ownership." % (path, fn)),
                          "user_delay": rng.uniform(6, 14)})
            continue
        if k in edit_at:
            path = c.pick_file(rng, area, min_lines=60, max_lines=400)
            _, _, text, delay = obs_read(c, rng, area, 2600, path=path)
            steps.append({"op": "request", "cls": "loop", "copy": True, "obs": text, "delay": delay,
                          "user": ("Use edit_file on /work/ninfer/%s to wrap the body of the second "
                                   "function shown above in a scope guard. old_string must contain "
                                   "that function's entire body exactly as shown." % path),
                          "user_delay": rng.uniform(5, 12)})
            continue
        _, _, text, delay = draw_obs(c, rng, area)
        steps.append({"op": "request", "cls": "loop", "obs": text, "delay": delay})
        if extra and k in extra:
            steps += extra[k]
    return steps


def _subagent(c, rng, P, name, persona_key, area, task, turns):
    steps = [{"op": "request", "cls": "subagent_first", "obs": None, "delay": 0.0}]
    for k in range(turns - 1):
        kind = rng.choice(["read", "read", "grep", "read", "diff", "glob"])
        _, _, text, delay = draw_obs(c, rng, area, kind=kind)
        last = k == turns - 2
        steps.append({"op": "request", "cls": "subagent_loop", "obs": text, "delay": delay,
                      "user": ("You have enough information now. Write your final report."
                               if last else None)})
    return {"name": name, "persona": persona_key, "area": area,
            "initial": [{"role": "system", "content": P[persona_key]["system"]},
                        {"role": "user", "content": task}], "steps": steps}


def _report_obs(c, rng, area, names):
    """What the parent agent receives back from its subagents (scripted, so both arms match)."""
    parts = []
    for n in names:
        path = c.pick_file(rng, area, min_lines=40)
        lines = c.lines(path)
        s = rng.randrange(0, max(1, len(lines) - 12))
        cites = "\n".join("- %s:%d: `%s`" % (path, s + i + 1, lines[s + i].strip()[:140])
                          for i in range(min(10, len(lines) - s)) if lines[s + i].strip())
        parts.append("## Report from %s\n\nThe relevant logic lives in %s. Findings:\n%s\n\n%s"
                     % (n, path, cites, _identifiers(c.text(path))[:40]))
    return "\n\n".join(parts)


def _summary(c, rng, area, words=1400):
    files = [c.pick_file(rng, area) for _ in range(14)]
    body = ["# Session summary", "", "## Task",
            "Fix the context-cache demotion path so a private owner evicted under device pressure "
            "keeps its host copy, add a regression test, and keep the context-cache tests green.",
            "", "## Files read or changed"]
    body += ["- /work/ninfer/%s" % f for f in files]
    body += ["", "## Key decisions"]
    idents = []
    for f in files:
        idents += _identifiers(c.text(f))[:12]
    for i in range(0, min(len(idents), 120), 6):
        body.append("- %s interacts with %s; %s must run before %s releases its pages."
                    % tuple(idents[i:i + 4]) if len(idents[i:i + 4]) == 4 else "- (none)")
    body += ["", "## Current state",
             "The fix is implemented in the eviction planner; two tests still fail on the "
             "host-restore scenario. Next: re-run the context-cache tests, inspect the failure, "
             "and finish the regression test.", ""]
    text = "\n".join(body)
    return text[: int(words * 6.2)]


def build_plan(seed=42, scale=1.0):
    """The full scenario. `scale` stretches/shrinks the loop lengths (1.0 = the published size)."""
    c = Corpus()
    P = personas(c)
    rng = random.Random(seed)

    def n(x):
        return max(2, int(round(x * scale)))

    actors = []
    # ---- Session A: main agent (Claude-Code-like persona), C++ engine work --------------
    a_task = ("The context-cache demotion path drops the host copy when a private owner is evicted "
              "under device pressure, so the next turn re-prefills the whole conversation. Find "
              "where the device-to-host demotion decision is made, add a regression test, fix it, "
              "and build and run the affected tests before you report back.")
    a_hist = build_resumed(c, rng, P["cc"], "engine", a_task, 85_000, "A")
    a_hist.append({"role": "user", "content":
                   "I'm back. Continue from where we left off: finish the fix in the eviction "
                   "planner and get the context-cache tests green."})
    fan1 = ["explore-1", "explore-2", "explore-3", "explore-4"]
    fan2 = ["explore-5", "explore-6", "explore-7"]
    pair = ["survey-1", "survey-2"]
    a_steps = [{"op": "request", "cls": "cold_resume", "obs": None, "delay": 0.0},
               {"op": "signal", "name": "A_resumed"},
               {"op": "wait", "signal": "C_resumed"},
               {"op": "lockstep"}]
    a_steps += _loop(c, rng, "engine", n(7), 0, copy_at=(), edit_at=(4,))
    a_steps.append({"op": "spawn", "actors": fan1})
    a_steps.append({"op": "request", "cls": "loop", "obs": _report_obs(c, rng, "engine", fan1),
                    "delay": 0.2})
    a_steps += _loop(c, rng, "engine", n(9), 100, copy_at=(5,), edit_at=())
    a_steps.append({"op": "compact", "cls": "compaction", "max_tokens": 8192,
                    "summary": _summary(c, rng, "engine")})
    a_steps.append({"op": "request", "cls": "restart", "obs": None, "delay": 0.5})
    a_steps.append({"op": "signal", "name": "A_restarted"})
    a_steps += _loop(c, rng, "engine", n(7), 200)
    a_steps.append({"op": "spawn", "actors": fan2})
    a_steps.append({"op": "request", "cls": "loop", "obs": _report_obs(c, rng, "engine", fan2),
                    "delay": 0.2})
    # The other terminals have finished: session A wraps up alone (last loop, a final review
    # subagent, the pull-request write-up), so every arm has sustained single-request decoding.
    a_steps += [{"op": "wait", "signal": "B_done"}, {"op": "wait", "signal": "C_done"}]
    a_steps += _loop(c, rng, "engine", n(6), 300, copy_at=(3,))
    a_steps.append({"op": "spawn", "actors": ["review-2"]})
    a_steps.append({"op": "request", "cls": "loop",
                    "obs": _report_obs(c, rng, "engine", ["review-2"]), "delay": 0.2})
    a_steps.append({"op": "request", "cls": "loop", "obs": None, "delay": 0.0,
                    "user": ("Tests are green. Write the pull-request description now: summary, "
                             "root cause, the fix and why it is correct, the regression test, "
                             "tests run, and remaining risks. Be thorough (about 1,500 words) "
                             "and do not call tools."),
                    "user_delay": rng.uniform(8, 15)})
    actors.append({"name": "A", "persona": "cc", "area": "engine", "initial": a_hist,
                   "steps": a_steps, "top": True})

    for i, name in enumerate(fan1):
        actors.append(_subagent(c, rng, P, name, "sub", "engine",
                                "Research question %d: in /work/ninfer, trace every path by which a "
                                "retained private checkpoint can lose its host copy (%s). Report the "
                                "exact functions, their callers, and the tests that cover them."
                                % (i + 1, ["eviction", "demotion", "lease extension",
                                           "admission planning"][i]), n(5)))
    for i, name in enumerate(fan2):
        actors.append(_subagent(c, rng, P, name, "sub", "engine",
                                "Research question %d: verify the fix against %s and report any "
                                "remaining path that still drops a host copy."
                                % (i + 5, ["the shared-prefix catalog", "the long-anchor set",
                                           "partial-prefill salvage"][i]), n(4)))

    # ---- Session B: second agent (Qwen-Code-like persona), Python tooling ----------------
    b_task = ("Add a --resume option to the eval coordinator so an interrupted evaluation run "
              "continues from its last completed item instead of starting over. Keep the result "
              "format unchanged and add tests.")
    b_hist = build_resumed(c, rng, P["qc"], "python", b_task, 60_000, "B")
    b_hist.append({"role": "user", "content": "Continue. The resume test is still failing."})
    b_extra = {3: [{"op": "retry", "cls": "retry", "delay": 4.0}],
               5: [{"op": "side_call", "persona": "check", "cls": "check", "max_tokens": 256}]}
    b_steps = [{"op": "wait", "signal": "A_resumed"},
               {"op": "request", "cls": "cold_resume", "obs": None, "delay": 0.0},
               {"op": "signal", "name": "B_resumed"},
               {"op": "wait", "signal": "C_resumed"},
               {"op": "lockstep"}]
    b_steps += _loop(c, rng, "python", n(11), 0, copy_at=(8,), extra=b_extra)
    # Two research subagents at once: their long final reports decode together, so every arm
    # has sustained two-request decoding for analyze.py to measure.
    b_steps.append({"op": "spawn", "actors": pair})
    b_steps.append({"op": "request", "cls": "loop", "obs": _report_obs(c, rng, "python", pair),
                    "delay": 0.2})
    # The user leaves this terminal while session A compacts and restarts; the session must
    # still be cached when they come back.
    b_steps.append({"op": "wait", "signal": "A_restarted"})
    _, _, text, delay = draw_obs(c, rng, "python", kind="build")
    b_steps.append({"op": "request", "cls": "after_idle", "obs": text, "delay": 8.0,
                    "user": "Back now. Where are we? Continue with the failing resume test."})
    b_steps += _loop(c, rng, "python", n(4), 100)
    b_steps.append({"op": "side_call", "persona": "check", "cls": "check", "max_tokens": 256})
    # Client-side context management: old tool outputs are cleared from the history, which
    # changes the prompt from an early message onwards.
    b_steps.append({"op": "clear_old_tool_results", "keep_last": 20})
    _, _, text, delay = draw_obs(c, rng, "python", kind="read")
    b_steps.append({"op": "request", "cls": "history_edit", "obs": text, "delay": delay})
    b_steps += _loop(c, rng, "python", n(5), 200)
    b_steps.append({"op": "signal", "name": "B_done"})
    actors.append({"name": "B", "persona": "qc", "area": "python", "initial": b_hist,
                   "steps": b_steps, "top": True})
    for i, name in enumerate(pair):
        actors.append(_subagent(c, rng, P, name, "sub", "python",
                                "Research question %d: in /work/ninfer, find every place the eval "
                                "tooling %s. Report the exact functions, the file formats involved, "
                                "and what a resumed run would have to reconstruct."
                                % (i + 1, ["writes per-item results", "tracks run progress"][i]),
                                n(4)))

    # ---- Session C: third agent (Claude-Code-like persona), tests and docs ----------------
    c_task = ("Write request-log schema tests for the new context-cache fields and document them "
              "in docs/serving.md. Follow the existing test layout.")
    c_hist = build_resumed(c, rng, P["cc"], "tests", c_task, 25_000, "C")
    c_hist.append({"role": "user", "content": "Continue with the schema tests."})
    big = obs_big_log(c, rng, "tests", 16_000)
    c_extra = {8: [{"op": "request", "cls": "loop", "obs": None, "delay": 12.0,
                    "user": "CI failed on the Windows runner, full log below:\n\n" + big,
                    "abort_after": 1.5, "retry_delay": 4.0}],
               11: [{"op": "spawn", "actors": ["review-1"]},
                    {"op": "request", "cls": "loop", "obs": _report_obs(c, rng, "tests", ["review-1"]),
                     "delay": 0.2}]}
    c_steps = [{"op": "wait", "signal": "B_resumed"},
               {"op": "request", "cls": "cold_resume", "obs": None, "delay": 0.0},
               {"op": "signal", "name": "C_resumed"},
               {"op": "lockstep"}]
    c_steps += _loop(c, rng, "tests", n(16), 0, copy_at=(4, 13), extra=c_extra)
    c_steps.append({"op": "signal", "name": "C_done"})
    actors.append({"name": "C", "persona": "cc", "area": "tests", "initial": c_hist,
                   "steps": c_steps, "top": True})
    actors.append(_subagent(c, rng, P, "review-1", "rev", "tests",
                            "Review the new request-log schema tests for missing boundary cases "
                            "and incorrect assumptions about field presence.", n(4)))
    actors.append(_subagent(c, rng, P, "review-2", "rev", "engine",
                            "Review the eviction-planner fix for the dropped host copy: lifetime "
                            "and ownership of the demoted pages, accounting of the host replica, "
                            "and whether the regression test would have caught the original bug.",
                            n(4)))

    # Seeds: one per request attempt, identical in both arms; they tag requests in the log.
    seen = set()
    for a in actors:
        for i, s in enumerate(a["steps"]):
            if s["op"] in ("request", "side_call", "compact", "retry"):
                s["seed"] = _seed(seed, a["name"], i)
                s["tag"] = "%s#%d" % (a["name"], i)
                if s.get("abort_after"):
                    s["abort_seed"] = _seed(seed, a["name"], i, "aborted")
                for k in ("seed", "abort_seed"):
                    if k in s:
                        assert s[k] not in seen, "seed collision"
                        seen.add(s[k])
    return {"seed": seed, "scale": scale, "corpus_commit": c.commit, "personas": P,
            "actors": actors}


def summarize(plan):
    lines = []
    P = plan["personas"]
    for a in plan["actors"]:
        base = tools_tokens(P[a["persona"]]["tools"]) + messages_tokens(a["initial"][:1])
        hist = messages_tokens(a["initial"]) + tools_tokens(P[a["persona"]]["tools"])
        reqs = [s for s in a["steps"] if s["op"] in ("request", "side_call", "compact", "retry")]
        obs = sum(est_tokens(s.get("obs") or "") + est_tokens(s.get("user") or "") for s in reqs)
        copy = sum(1 for s in reqs if s.get("copy"))
        lines.append("%-10s persona=%-4s prefix~%6d start~%7d requests=%3d observations~%7d copy=%d"
                     % (a["name"], a["persona"], base, hist, len(reqs), obs, copy))
    total = sum(1 for a in plan["actors"] for s in a["steps"]
                if s["op"] in ("request", "side_call", "compact", "retry"))
    lines.append("TOTAL %d actors, %d requests (seed %d, scale %.2f, corpus %s)"
                 % (len(plan["actors"]), total, plan["seed"], plan["scale"], plan["corpus_commit"]))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(summarize(build_plan(scale=scale)))
