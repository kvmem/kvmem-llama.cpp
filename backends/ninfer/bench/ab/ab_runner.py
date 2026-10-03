#!/usr/bin/env python3
"""Autonomous A/B benchmark: a control ninfer-serve build vs this line's build (treatment).

Runs both arms sequentially against the SAME artifact and the SAME synthesized agentic
workloads (ab_workloads.py), using the serve's own request log (--request-log-jsonl) as
the source of truth for metrics. Collects avg TTFT, prefix-cache hit stats, cold prefill
tok/s and output tok/s, and writes a comparison report (with a paste-ready README block)
next to this script.

See README.md for how to point the rig at your builds and flags; every path and the port
come from AB_* environment variables. No LLM is used at run time. The port must be free:
the runner never starts or stops another server on it.
"""
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import ab_workloads

# Outputs (report, arm logs, done marker) are written next to this script.
AB_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths come from AB_* environment variables:
#   AB_MODEL (required), AB_WORKDIR, AB_CONTROL_EXE, AB_TREATMENT_EXE, AB_HOST, AB_PORT
REPO = os.path.abspath(os.path.join(AB_DIR, "..", ".."))
EXE = ".exe" if os.name == "nt" else ""
# Working directory the serves run in (relative paths in the flags resolve against it).
DEPLOY = os.environ.get("AB_WORKDIR") or None
MODEL = os.environ.get("AB_MODEL")


def default_serve(build_dir):
    """ninfer-serve in a build tree: apps/ (single-config) or apps/Release/ (multi-config)."""
    for path in (os.path.join(build_dir, "apps", "ninfer-serve" + EXE),
                 os.path.join(build_dir, "apps", "Release", "ninfer-serve" + EXE)):
        if os.path.exists(path):
            return path
    return os.path.join(build_dir, "apps", "ninfer-serve" + EXE)


# Control = the build this line is compared with, built from a control worktree
# (control_build.bat on Windows); treatment = this checkout's build.
CONTROL_EXE = os.environ.get("AB_CONTROL_EXE", default_serve(os.path.join(AB_DIR, "control", "build")))
TREATMENT_EXE = os.environ.get("AB_TREATMENT_EXE", default_serve(os.path.join(REPO, "build")))
REPORT = os.path.join(AB_DIR, "ab_report.md")
DONE = os.path.join(AB_DIR, "AB_DONE.txt")
RUNNER_LOG = os.path.join(AB_DIR, "runner.log")
CREATE_NEW_PROCESS_GROUP = 0x00000200


def serve_process_kwargs():
    """Starts a serve in its own process group, so stopping it stops its children too."""
    if os.name == "nt":
        return {"creationflags": CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_process_tree(proc, timeout=90):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=timeout)
    except Exception:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.kill()

HOST = os.environ.get("AB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AB_PORT", "8080"))
WORKLOAD_SEED = 42

# The flags use --max-context 220000, but a control that bakes in a 1 GiB automatic KV headroom
# with no flag to lower it may not start there: its minimum KV reservation scales with
# max-context. We probe the largest context at which the control starts and run BOTH arms at
# that context
# (the workloads need at most ~140K actual tokens, so anything >= 150000 keeps them intact).
MAX_CONTEXT_CANDIDATES = [180000, 170000, 160000, 150000]
BAT_MAX_CONTEXT = "220000"

# The launch flags used for the published benchmark (name, value); value None = flag
# with no value. They match the launcher of the original published run with the tested variant
# --prefill-chunk 2048 / --ngram-min-match 12; edit here to benchmark your own
# configuration. The treatment uses all of them; the control uses the subset its
# --help supports (flags it does not know are dropped automatically). --request-log-jsonl
# is appended by build_args(). --use-original-prefix-caching selects the checkpoint catalog,
# this line's default, explicitly.
BAT_FLAGS = [
    ("--host", HOST), ("--port", str(PORT)),
    ("--use-original-prefix-caching", None),
    ("--max-context", "220000"), ("--max-concurrency", "2"),
    ("--spec", "dflash2"), ("--draft-tokens", "7"),
    ("--lm-head-draft", None),
    ("--ngram-draft-tokens", "15"), ("--ngram-min-match", "12"),
    ("--kv-dtype", "int8"),
    ("--preserve-thinking", None),
    ("--host-kv-mib", "24000"),
    ("--pending-timeout-ms", "900000"),
    ("--prefill-chunk", "2048"),
    ("--kv-capacity", "auto"), ("--vram-headroom-mib", "0"),
    ("--log-colours", "on"),
    ("--host-state-slots", "64"),
    ("--max-private-continuations", "32"),
    ("--max-long-anchors-per-continuation", "8"),
    ("--max-shared-prefixes", "32"),
    ("--ngram-archive-mib", "2048"),
    ("--ngram-session-mib", "256"),
    ("--ngram-native-sessions", None),
    ("--default-thinking-budget", "16384"),
    ("--thinking-budget-message", "Considering the limited time available to the user, "
     "I must stop thinking now. Time to act:"),
]


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(RUNNER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def port_open(port):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect((HOST, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def model_ready():
    """True once the serve has finished loading the model (/v1/models answers with a model)."""
    try:
        with urllib.request.urlopen("http://%s:%d/v1/models" % (HOST, PORT), timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
            return len(data.get("data", [])) > 0
    except Exception:
        return False


def gpu_memory_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_gpu_idle(timeout=240, threshold=6000, interval=5):
    """Wait until no model is loaded on the GPU (memory below threshold). Used after another
    stops so the next serve does not start on a GPU still holding the previous model's memory."""
    mib = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        mib = gpu_memory_mib()
        if mib is not None and mib < threshold:
            log("GPU idle (%d MiB in use)" % mib)
            return
        log("waiting for GPU to release (%s MiB in use)..." % mib)
        time.sleep(interval)
    raise SystemExit("GPU still busy after %ds (%s MiB in use); not safe to start a serve" % (timeout, mib))


def settle_gpu(timeout=90, threshold=6000):
    """Soft variant of wait_gpu_idle for use between calibration probes (never aborts)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        mib = gpu_memory_mib()
        if mib is not None and mib < threshold:
            return
        time.sleep(3)
    log("WARNING: GPU not idle after %ds; continuing anyway" % timeout)


def with_max_context(flags, ctx):
    return [f if f[0] != "--max-context" else ("--max-context", str(ctx)) for f in flags]


def help_flags(exe):
    """Return the set of --flag names that `exe --help` advertises."""
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, cwd=DEPLOY, timeout=60)
    text = (out.stdout or "") + (out.stderr or "")
    return set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", text))


def build_args(exe, flags, log_path):
    args = [exe, MODEL]
    for name, val in flags:
        args.append(name)
        if val is not None:
            args.append(val)
    args += ["--request-log-jsonl", log_path]
    return args


def start_serve(exe, flags, log_path, serve_log):
    if os.path.exists(log_path):
        os.remove(log_path)
    args = build_args(exe, flags, log_path)
    log("LAUNCH: " + " ".join(args) + "   [serve stdout/stderr -> " + os.path.basename(serve_log) + "]")
    sf = open(serve_log, "w", encoding="utf-8")
    proc = subprocess.Popen(args, cwd=DEPLOY, stdout=sf, stderr=subprocess.STDOUT,
                            **serve_process_kwargs())
    try:
        time.sleep(3)
        deadline = time.time() + 420
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("serve exited early (rc=%s); see %s"
                                   % (proc.returncode, os.path.basename(serve_log)))
            if port_open(PORT):
                break
            time.sleep(2)
        else:
            raise RuntimeError("serve did not bind :%d within 420s; see %s"
                               % (PORT, os.path.basename(serve_log)))
        # The port binds before the model finishes loading; wait until /v1/models actually answers.
        ready_deadline = time.time() + 360
        while time.time() < ready_deadline:
            if proc.poll() is not None:
                raise RuntimeError("serve exited during model load (rc=%s); see %s"
                                   % (proc.returncode, os.path.basename(serve_log)))
            if model_ready():
                log("serve ready on :%d (pid %s, model loaded)" % (PORT, proc.pid))
                return proc
            time.sleep(3)
        raise RuntimeError("serve model did not become ready within 360s; see %s"
                           % os.path.basename(serve_log))
    except Exception:
        # Never leave a half-started serve holding :8080 or the GPU (a later probe or arm
        # would otherwise deadlock on port binding).
        if proc.poll() is None:
            log("killing un-started serve (pid %s)" % proc.pid)
            stop_process_tree(proc, timeout=30)
        raise
    finally:
        sf.close()


def stop_serve(proc):
    log("stopping serve (pid %s) and its process group" % proc.pid)
    stop_process_tree(proc)
    wait_gpu_idle()  # let the GPU release its memory before the next arm loads


def get_model_id():
    try:
        with urllib.request.urlopen("http://%s:%d/v1/models" % (HOST, PORT), timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        mid = data["data"][0]["id"]
        log("model id from /v1/models: %s" % mid)
        return mid
    except Exception as e:
        log("/v1/models unavailable (%s); using default model name" % e)
        return "qwen3.8-27b"


def http_chat(model_id, request):
    body = {
        "model": model_id,
        "messages": request["messages"],
        "tools": request["tools"],
        "tool_choice": request.get("tool_choice", "auto"),
        "max_tokens": request["max_tokens"],
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "stream": request.get("stream", True),
    }
    req = urllib.request.Request(
        "http://%s:%d/v1/chat/completions" % (HOST, PORT),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and "[DONE]" in line:
                break
    return time.time() - t0


def smoke(model_id):
    body = {"model": model_id, "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 8, "temperature": 0.0, "stream": False}
    req = urllib.request.Request("http://%s:%d/v1/chat/completions" % (HOST, PORT),
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    log("smoke ok: finish=%s" % data.get("choices", [{}])[0].get("finish_reason"))


def replay(model_id, groups):
    for g_i, group in enumerate(groups):
        log("group %d: %d request(s) -> %s" % (g_i, len(group), [r["id"] for r in group]))
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {}
            for r in group:
                futs[r["id"]] = ex.submit(http_chat, model_id, r)
            for rid in group_ids(group):
                t0 = time.time()
                try:
                    wall = futs[rid].result()
                    log("  req %d done in %.1fs (client wall)" % (rid, wall))
                except Exception as e:
                    log("  req %d ERROR: %s" % (rid, e))
        time.sleep(2)


def group_ids(group):
    return [r["id"] for r in group]


def _median_mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.median(vals), (sum(vals) / len(vals))


def parse_arm_log(path):
    """Parse a serve request-log.jsonl (same schema as log.json). Return per-request rows + aggregates."""
    rows = []
    errors = []
    if not os.path.exists(path):
        return {"requests": rows, "errors": errors, "aggregates": {}}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            ev = o.get("event")
            if ev in ("request_error", "request_rejected"):
                errors.append(o.get("error") or o.get("request"))
                continue
            if ev != "request_done":
                continue
            t = o.get("timings_seconds") or {}
            res = o.get("result") or {}
            req = o.get("request") or {}
            if (req.get("message_count") or 0) <= 1:
                continue  # the runner's smoke request is not part of the workload
            prefill_s = t.get("prefill")
            decode_s = t.get("decode")
            prompt_tok = res.get("prompt_tokens")
            comp_tok = res.get("completion_tokens")
            ttft = t.get("ttft")
            prefill_tps = (prompt_tok / prefill_s) if (prefill_s and prompt_tok) else None
            output_tps = (comp_tok / decode_s) if (decode_s and comp_tok) else None
            rows.append({
                "request_id": req.get("request_id"),
                "ttft_s": ttft, "prefill_s": prefill_s, "decode_s": decode_s,
                "total_s": t.get("total"),
                "prompt_tokens": prompt_tok, "completion_tokens": comp_tok,
                "computed_prefill_tokens": res.get("computed_prefill_tokens"),
                "prefill_tps": prefill_tps, "output_tps": output_tps,
                "cache_hit_tokens": res.get("prefix_cache_hit_tokens", 0),
                "reuse_path": res.get("prefix_reuse_path"),
                "finish": res.get("finish_reason"),
            })
    rows.sort(key=lambda r: (r.get("total_s") or 0))
    ttfts, ptp, otp = [r["ttft_s"] for r in rows], [r["prefill_tps"] for r in rows], [r["output_tps"] for r in rows]
    ttft_med, ttft_mean = _median_mean(ttfts)
    p_med, p_mean = _median_mean(ptp)
    o_med, o_mean = _median_mean(otp)
    # "Cold" prefill = a root (uncached) request that did real prefill compute. A fully
    # prefix-cached request's prefill_s is a tiny host/device cache read, so prompt/prefill_s
    # overstates prefill compute speed; the fair prefill-throughput metric is the cold median.
    cold_ptp = [r["prefill_tps"] for r in rows if r.get("reuse_path") == "root"]
    cold_med, cold_mean = _median_mean(cold_ptp)
    total_prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    total_comp = sum(r["completion_tokens"] or 0 for r in rows)
    total_cache = sum(r["cache_hit_tokens"] or 0 for r in rows)
    cache_rate = (total_cache / total_prompt) if total_prompt else 0.0
    # Headline throughput metrics (workload-wide, wall-time based):
    #  prefill_compute_tps = total computed (NON-cached) prefill tokens / total prefill wall time
    #  output_tps_overall  = total completion tokens / total decode wall time
    total_prefill_s = sum(r["prefill_s"] or 0 for r in rows)
    total_decode_s = sum(r["decode_s"] or 0 for r in rows)
    total_computed = sum(r["computed_prefill_tokens"] or 0 for r in rows)
    prefill_compute_tps = total_computed / total_prefill_s if total_prefill_s > 0 else None
    output_tps_overall = total_comp / total_decode_s if total_decode_s > 0 else None
    # Cache-hit stats by reuse path (e.g. root / private_endpoint / shared_stable_prefix / ...).
    path_stats = {}
    for r in rows:
        p = r["reuse_path"] or "unknown"
        d = path_stats.setdefault(p, {"requests": 0, "hit_tokens": 0, "prompt_tokens": 0})
        d["requests"] += 1
        d["hit_tokens"] += r["cache_hit_tokens"] or 0
        d["prompt_tokens"] += r["prompt_tokens"] or 0
    aggregates = {
        "n_requests": len(rows), "n_errors": len(errors),
        "ttft_s_median": ttft_med, "ttft_s_mean": ttft_mean,
        "prefill_tps_cold_median": cold_med, "prefill_tps_cold_mean": cold_mean,
        "n_cold_requests": len(cold_ptp),
        "prefill_tps_median": p_med, "prefill_tps_mean": p_mean,
        "output_tps_median": o_med, "output_tps_mean": o_mean,
        "total_prompt_tokens": total_prompt, "total_completion_tokens": total_comp,
        "total_cache_hit_tokens": total_cache, "cache_hit_rate": cache_rate,
        "total_prefill_s": total_prefill_s, "total_decode_s": total_decode_s,
        "total_computed_prefill_tokens": total_computed,
        "prefill_compute_tps": prefill_compute_tps,
        "output_tps_overall": output_tps_overall,
        "cache_path_stats": path_stats,
    }
    return {"requests": rows, "errors": errors, "aggregates": aggregates}


def _fmt(v, unit=""):
    if v is None:
        return "n/a"
    return ("%.3g" % v) + unit


def _sec(v):
    """Seconds with 1 decimal (README-friendly)."""
    return "n/a" if v is None else ("%.1f" % v)


def _tok(v):
    """Integer token counts with thousands separators (README-friendly)."""
    if v is None:
        return "n/a"
    return format(int(round(v)), ",d")


def _rate(v):
    """Tokens/sec as a rounded integer with separators."""
    return "n/a" if v is None else format(int(round(v)), ",d")


def _pct(v):
    return "n/a" if v is None else ("%.1f%%" % (v * 100.0))


def _delta(c, t, points=False):
    """Signed Δ (treatment vs control); direction of goodness is in the metric name."""
    if c in (None, 0) or t is None or c is None:
        return "n/a"
    if points:
        return "%+.1f pp" % ((t - c) * 100.0)
    return "%+.1f%%" % ((t - c) / c * 100.0)


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown GPU"


def _flag_str(flags):
    return " ".join(n + (" " + v if v else "") for n, v in flags)


def readme_block(ctrl, treat, ctrl_flags, treat_flags, dropped, ctx):
    """The paste-ready README section: headline metrics table + configuration used."""
    ca, ta = ctrl["aggregates"], treat["aggregates"]
    n = ca.get("n_requests")
    med_prompt = None
    prompts = sorted(r["prompt_tokens"] or 0 for r in ctrl["requests"] if r["prompt_tokens"])
    if prompts:
        med_prompt = statistics.median(prompts)
    L = []
    L.append("## Inference performance: this build vs the control (A/B benchmark)\n")
    L.append("Both arms serve the same official NInfer Qwen3.8-27B NVFP4 artifact (%s) on a "
             "%s, run at the same max-context (%d — the largest context at which the control "
             "arm starts on this card; see below) and replay the same synthesized agentic "
             "workload: %d requests shaped from a logged agent workload (multi-turn "
             "tool-agent sessions, ~%s-token shared prefix, median prompt ≈ %sK tokens, two "
             "concurrent-request pairs including a shared-prefix cache-contention pair, "
             "max-concurrency 2).\n"
             % (os.path.basename(MODEL), gpu_name(), ctx, n, _tok(ab_workloads.BASE_TOKENS),
                _fmt((med_prompt or 0) / 1000.0)))
    L.append("| Metric | Control | This build | Δ |\n")
    L.append("|---|---|---|---|\n")
    L.append("| Avg TTFT (s, lower better) | %s | %s | %s |\n"
             % (_sec(ca.get("ttft_s_mean")), _sec(ta.get("ttft_s_mean")),
                _delta(ca.get("ttft_s_mean"), ta.get("ttft_s_mean"))))
    L.append("| TTFT median (s, lower better) | %s | %s | %s |\n"
             % (_sec(ca.get("ttft_s_median")), _sec(ta.get("ttft_s_median")),
                _delta(ca.get("ttft_s_median"), ta.get("ttft_s_median"))))
    L.append("| Prefix-cache hit rate (hit/prompt tokens) | %s | %s | %s |\n"
             % (_pct(ca.get("cache_hit_rate")), _pct(ta.get("cache_hit_rate")),
                _delta(ca.get("cache_hit_rate"), ta.get("cache_hit_rate"), points=True)))
    L.append("| Total cache-hit tokens (of %s prompt) | %s | %s | %s |\n"
             % (_tok(ca.get("total_prompt_tokens")), _tok(ca.get("total_cache_hit_tokens")),
                _tok(ta.get("total_cache_hit_tokens")),
                _delta(ca.get("total_cache_hit_tokens"), ta.get("total_cache_hit_tokens"))))
    L.append("| Prefill tok/s, cold (root) requests, median | %s | %s | %s |\n"
             % (_rate(ca.get("prefill_tps_cold_median")), _rate(ta.get("prefill_tps_cold_median")),
                _delta(ca.get("prefill_tps_cold_median"), ta.get("prefill_tps_cold_median"))))
    L.append("| Output tok/s | %s | %s | %s |\n"
             % (_rate(ca.get("output_tps_overall")), _rate(ta.get("output_tps_overall")),
                _delta(ca.get("output_tps_overall"), ta.get("output_tps_overall"))))
    L.append("")
    L.append("- Avg/median TTFT over all %d completed requests per arm.\n" % n)
    L.append("- Cold (root) prefill = requests that did a full uncached prefill "
             "(n: control %s, this build %s).\n"
             % (ca.get("n_cold_requests"), ta.get("n_cold_requests")))
    L.append("- Output tok/s = total completion tokens / total decode wall time (completion "
             "tokens: control %s, this build %s; thinking on for both arms; per-request decode "
             "rates are comparable — see the per-request tables in the full report).\n"
             % (_tok(ca.get("total_completion_tokens")), _tok(ta.get("total_completion_tokens"))))
    # Cache-hit stats by reuse path (union of paths seen by either arm).
    paths = sorted(set(ca.get("cache_path_stats", {})) | set(ta.get("cache_path_stats", {})))
    if paths:
        L.append("- Cache hits by reuse path (requests / hit tokens):\n")
        L.append("  \n")
        L.append("  | Path | Control | This build |\n")
        L.append("  |---|---|---|\n")
        for p in paths:
            c = ca.get("cache_path_stats", {}).get(p, {"requests": 0, "hit_tokens": 0})
            t = ta.get("cache_path_stats", {}).get(p, {"requests": 0, "hit_tokens": 0})
            L.append("  | %s | %d / %s | %d / %s |\n"
                     % (p, c["requests"], _tok(c["hit_tokens"]), t["requests"], _tok(t["hit_tokens"])))
        L.append("")
    L.append("**Configuration and launch parameters.** Model: the official NInfer Qwen3.8-27B "
             "NVFP4 artifact `%s`; GPU: %s; max-context %d for both arms; max-concurrency 2; "
             "`--kv-dtype int8`.\n" % (os.path.basename(MODEL), gpu_name(), ctx))
    L.append("Launch parameters, this build (all flags):\n\n```\n%s\n```\n" % _flag_str(treat_flags))
    L.append("Launch parameters, the control (same list minus the flags it does not "
             "support, which are dropped: %s):\n\n```\n%s\n```\n"
             % (", ".join("`%s`" % d for d in dropped), _flag_str(ctrl_flags)))
    L.append("A control that keeps an automatic KV headroom under `--kv-capacity auto` it cannot "
             "lower (this build's `--kv-headroom-mib 0`) may not start at the launcher's "
             "max-context %s on this card; both arms run at the same "
             "calibrated %d so the arms stay comparable. Thinking is on for both arms "
             "(`--default-thinking-budget` + `--preserve-thinking`).\n"
             % (BAT_MAX_CONTEXT, ctx))
    L.append("Workload: seed %d; groups replayed in order, requests within a group sent "
             "concurrently. The full rig (workload generator, runner, watchdog, control build "
             "driver) lives in this directory so the test can be replicated with your own "
             "builds and flags.\n" % WORKLOAD_SEED)
    return "".join(L)


def compute_report(ctrl, treat, ctrl_flags, treat_flags, dropped, ctx):
    ca, ta = ctrl["aggregates"], treat["aggregates"]
    L = []
    L.append("# NInfer A/B: control vs. this build (treatment)\n")
    L.append("Generated: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    L.append("Control: %s\n" % CONTROL_EXE)
    L.append("Treatment: %s\n" % TREATMENT_EXE)
    L.append("\n**Paste-ready README block** (copy what is between the markers):\n")
    L.append("<!-- BEGIN README BLOCK -->\n")
    L.append("\n")
    L.append(readme_block(ctrl, treat, ctrl_flags, treat_flags, dropped, ctx))
    L.append("\n<!-- END README BLOCK -->\n")
    L.append("\n## Settings\n")
    L.append("- Control flags: %s\n" % _flag_str(ctrl_flags))
    L.append("- Treatment flags: %s\n" % _flag_str(treat_flags))
    L.append("- **max-context deviation:** the launcher uses %s; both arms run at the "
             "calibrated value %d. A control that cannot start at %s on this card keeps KV "
             "headroom under `--kv-capacity auto` that it cannot lower (this build's "
             "`--kv-headroom-mib 0`), so its minimum KV reservation plus that headroom exceeds "
             "the memory free after the weights. Shrinking max-context shrinks the "
             "reservation; the treatment matches the control's context so the arms stay "
             "comparable. Workload prompts are capped at ~140K tokens + output headroom so "
             "they fit the smallest candidate context.\n"
             % (BAT_MAX_CONTEXT, ctx, BAT_MAX_CONTEXT))
    L.append("\n## Headline detail (workload-wide)\n")
    L.append("| Aggregates | Control | Treatment |\n")
    L.append("|---|---|---|\n")
    L.append("| Requests completed | %s | %s |\n" % (ca.get("n_requests"), ta.get("n_requests")))
    L.append("| Requests errored | %s | %s |\n" % (ca.get("n_errors"), ta.get("n_errors")))
    L.append("| Total prompt tokens | %s | %s |\n" % (_tok(ca.get("total_prompt_tokens")), _tok(ta.get("total_prompt_tokens"))))
    L.append("| Total completion tokens | %s | %s |\n" % (_tok(ca.get("total_completion_tokens")), _tok(ta.get("total_completion_tokens"))))
    L.append("| Total computed (non-cached) prefill tokens | %s | %s |\n"
             % (_tok(ca.get("total_computed_prefill_tokens")), _tok(ta.get("total_computed_prefill_tokens"))))
    L.append("| Total prefill wall time (s) | %s | %s |\n" % (_sec(ca.get("total_prefill_s")), _sec(ta.get("total_prefill_s"))))
    L.append("| Total decode wall time (s) | %s | %s |\n" % (_sec(ca.get("total_decode_s")), _sec(ta.get("total_decode_s"))))
    L.append("| Total cache-hit tokens | %s | %s |\n" % (_tok(ca.get("total_cache_hit_tokens")), _tok(ta.get("total_cache_hit_tokens"))))
    L.append("| Prefill tok/s (cold/root median) | %s | %s |\n"
             % (_rate(ca.get("prefill_tps_cold_median")), _rate(ta.get("prefill_tps_cold_median"))))
    L.append("| Cold/root requests (n) | %s | %s |\n" % (ca.get("n_cold_requests"), ta.get("n_cold_requests")))
    L.append("\n## Control per-request (completion order)\n")
    L.append(_table(ctrl["requests"]))
    L.append("\n## Treatment per-request (completion order)\n")
    L.append(_table(treat["requests"]))
    L.append("\n## Errors\n")
    L.append("- Control: %d\n- Treatment: %d\n" % (len(ctrl["errors"]), len(treat["errors"])))
    for e in (ctrl["errors"] + treat["errors"])[:10]:
        L.append("  - %s\n" % json.dumps(e)[:300])
    return "".join(L)


def _table(rows):
    L = ["| id | TTFT s | prefill s | prompt tok | prefill tok/s | decode s | comp tok | out tok/s | cache hit tok | path | finish |"]
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["request_id"], _sec(r["ttft_s"]), _sec(r["prefill_s"]), _tok(r["prompt_tokens"]),
            _rate(r["prefill_tps"]), _sec(r["decode_s"]), _tok(r["completion_tokens"]),
            _rate(r["output_tps"]), _tok(r["cache_hit_tokens"]), r["reuse_path"], r["finish"]))
    return "\n".join(L)


def calibrate_control(ctrl_flags):
    """Probe the largest --max-context at which the CONTROL serve starts.

    A control that keeps a fixed automatic KV headroom under `--kv-capacity auto` may not start
    at the flags' context, since the minimum KV reservation grows with max-context (see
    serve_control.log). A smaller context shrinks the reservation; the treatment runs at
    the same context so the arms stay comparable. Returns (ctx, proc, flags) for the arm to
    reuse the probe's already-loaded serve.
    """
    for ctx in MAX_CONTEXT_CANDIDATES:
        flags = with_max_context(ctrl_flags, ctx)
        probe_log = os.path.join(AB_DIR, "serve_control.log")
        log("control calibration probe: --max-context %d" % ctx)
        try:
            proc = start_serve(CONTROL_EXE, flags, os.path.join(AB_DIR, "arm_control.jsonl"),
                               probe_log)
        except RuntimeError as e:
            log("  probe %d failed: %s" % (ctx, e))
            settle_gpu()
            continue
        log("  control starts at --max-context %d; running BOTH arms at this context" % ctx)
        return ctx, proc, flags
    raise SystemExit(
        "control serve failed to start at every candidate max-context %s; "
        "see serve_control.log" % MAX_CONTEXT_CANDIDATES)


def run_arm(name, exe, flags, log_path, groups, proc=None):
    log("=== ARM: %s ===" % name)
    serve_log = os.path.join(AB_DIR, "serve_%s.log" % name)
    if proc is None:
        proc = start_serve(exe, flags, log_path, serve_log)
    else:
        log("reusing the calibrated serve (pid %s) as the %s arm" % (proc.pid, name))
    try:
        mid = get_model_id()
        smoke(mid)
        replay(mid, groups)
    finally:
        stop_serve(proc)
    log("=== ARM %s: parsing %s ===" % (name, os.path.basename(log_path)))
    return parse_arm_log(log_path)


def report_window_end():
    # The runner never starts another server on the port; whoever stopped one restarts it.
    log("A/B window finished; the port is free (nothing was started on it)")


def wait_for_exes(exes, timeout=14400, interval=30):
    """Block until every serve exe exists (the builds are in progress and will produce them)."""
    deadline = time.time() + timeout
    missing = [e for e in exes if not os.path.exists(e)]
    if not missing:
        log("all serve exes present")
        return
    while missing and time.time() < deadline:
        log("waiting for build artifacts: " + ", ".join(os.path.basename(e) for e in missing))
        time.sleep(interval)
        missing = [e for e in exes if not os.path.exists(e)]
    if missing:
        raise SystemExit("missing serve exes after %ds: %s" % (timeout, ", ".join(missing)))
    log("all serve exes present")


def main():
    if not MODEL:
        raise SystemExit("set AB_MODEL to the .ninfer artifact both arms serve")
    dry = "--dry-run" in sys.argv
    log("A/B runner starting (dry-run=%s)" % dry)
    groups = ab_workloads.generate_workloads(WORKLOAD_SEED)
    log("workloads: %s" % ab_workloads.summarize(groups).splitlines()[-1])

    treat_flags = list(BAT_FLAGS)
    if not dry:
        wait_for_exes([CONTROL_EXE, TREATMENT_EXE])
    # Derive the control's supported flags from its --help; drop anything unsupported.
    supported = help_flags(CONTROL_EXE) if os.path.exists(CONTROL_EXE) else set()
    ctrl_flags = [f for f in BAT_FLAGS if f[0] in supported]
    dropped = [f[0] for f in BAT_FLAGS if f[0] not in supported]
    log("control supported %d/%d flags; dropped (unsupported by the control): %s"
        % (len(ctrl_flags), len(BAT_FLAGS), ", ".join(dropped)))

    if dry:
        log("DRY RUN: control flags -> %s" % " ".join(n + (" " + v if v else "") for n, v in ctrl_flags))
        log("DRY RUN: treatment flags -> %d flags" % len(treat_flags))
        return

    # Another server may just have stopped; make sure its GPU memory is fully released before
    # loading the control model (otherwise the control serve can OOM on a GPU still holding it).
    failure = None
    try:
        wait_gpu_idle()
        # The control may not start at the flags' 220000 context (fixed KV headroom);
        # calibrate the largest workable context and apply it to both arms.
        ctx, ctrl_proc, ctrl_flags = calibrate_control(ctrl_flags)
        treat_flags = with_max_context(treat_flags, ctx)
        ctrl = run_arm("control", CONTROL_EXE, ctrl_flags,
                       os.path.join(AB_DIR, "arm_control.jsonl"), groups, proc=ctrl_proc)
        treat = run_arm("treatment", TREATMENT_EXE, treat_flags,
                        os.path.join(AB_DIR, "arm_treatment.jsonl"), groups)
        report = compute_report(ctrl, treat, ctrl_flags, treat_flags, dropped, ctx)
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(report)
        log("report written: %s" % REPORT)
        # A rejected/errored request (e.g. context_length_exceeded) means the workload did not
        # apply fully to an arm — the report is diagnostic only, never an "OK" result.
        n_bad = ctrl["aggregates"].get("n_errors", 0) + treat["aggregates"].get("n_errors", 0)
        if n_bad:
            raise SystemExit(
                "A/B run INVALID: %d rejected/errored request(s) "
                "(ctrl=%d, treat=%d); see %s for details and the arm logs for the rejection "
                "reasons" % (n_bad, ctrl["aggregates"].get("n_errors", 0),
                             treat["aggregates"].get("n_errors", 0), REPORT))
        write_done("OK")
        log("A/B complete")
    except BaseException as e:
        failure = e
        log("A/B FAILED: %s" % e)
        log(traceback.format_exc())
        log("startup diagnostics in: serve_control.log / serve_treatment.log (same dir)")
        # No AB_DONE.txt on failure: a later :8080 close re-triggers the watchdog.
    finally:
        try:
            report_window_end()
        except Exception as e:
            log("window-end report failed: %s" % e)
    if failure is not None:
        raise SystemExit(1)


def write_done(status):
    with open(DONE, "w", encoding="utf-8") as f:
        f.write(status + " at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
