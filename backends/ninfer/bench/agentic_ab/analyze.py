#!/usr/bin/env python3
"""Analyze an agentic A/B run directory and write report.md + summary.json into it.

Usage: python analyze.py <run_dir>
       python analyze.py --aggregate <out_dir> <run_dir> <run_dir> [...]

Each arm directory (control, treatment and optionally alt) holds the client's own record of
every request (client.jsonl, with the per-request seed and the workload class) and the serve's
request log (request_log.jsonl). They are joined on the request seed, which is identical in
every arm, so every metric compares the same logical requests. All timings and token counts
come from the serve's request log; every arm is compared with the control.
"""
import json
import os
import random
import re
import statistics
import sys

# Every arm is compared with the control; `alt` is the treatment build with AB_ALT_EXTRA_FLAGS.
ARMS = ("control", "treatment", "alt")
LABEL = {"control": "Control build", "treatment": "This build",
         "alt": "This build, hybrid prefix cache"}
SHORT = {"control": "Control", "treatment": "This", "alt": "Hybrid"}
CONTINUING = {"loop", "after_idle", "history_edit", "retry", "abort_retry", "subagent_loop"}
NEW_LONG = {"cold_resume", "compaction", "check"}
CLASS_DOC = [
    ("cold_resume", "a saved session resumed after a restart (whole prompt is new)"),
    ("loop", "an agent tool-loop turn continuing its own conversation"),
    ("subagent_first", "a subagent's first turn (shares its system+tools prefix with siblings)"),
    ("subagent_loop", "a subagent tool-loop turn"),
    ("compaction", "the whole conversation sent for summarisation (new prompt)"),
    ("restart", "the fresh context after compaction (system prompt + summary)"),
    ("check", "a loop-detection side call over the whole history (new prompt)"),
    ("after_idle", "a session resumed after sitting idle while other sessions ran"),
    ("history_edit", "the first turn after the client cleared old tool results"),
    ("retry", "the user retrying the previous turn with the identical prompt"),
    ("abort_retry", "the retry after the client aborted a long request"),
]


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except ValueError:
                        pass
    return rows


def load_arm(run_dir, arm):
    d = os.path.join(run_dir, arm)
    client = load_jsonl(os.path.join(d, "client.jsonl"))
    server = load_jsonl(os.path.join(d, "request_log.jsonl"))
    meta = {}
    if os.path.exists(os.path.join(d, "arm.json")):
        with open(os.path.join(d, "arm.json"), encoding="utf-8") as f:
            meta = json.load(f)
    done, errs, start, thr = {}, {}, None, []
    for o in server:
        ev = o.get("event")
        seed = ((o.get("request") or {}).get("sampling") or {}).get("seed")
        if ev == "server_start":
            start = o
        elif ev == "request_done" and seed is not None:
            done[seed] = o
        elif ev in ("request_error", "request_rejected") and seed is not None:
            errs[seed] = o
        elif ev == "throughput":
            thr.append(o)
    reqs, events = [], []
    for c in client:
        if "event" in c:
            events.append(c)
            continue
        s = done.get(c["seed"])
        r = dict(c)
        r["server"] = s
        if s:
            res, t = s.get("result") or {}, s.get("timings_seconds") or {}
            spec = s.get("speculative") or {}
            r.update({
                "prompt": res.get("prompt_tokens") or 0,
                "hit": res.get("prefix_cache_hit_tokens") or 0,
                "computed": res.get("computed_prefill_tokens") or 0,
                "completion": res.get("completion_tokens") or 0,
                "thinking": res.get("model_thinking_tokens") or 0,
                "path": res.get("prefix_reuse_path") or "unknown",
                "finish_server": res.get("finish_reason"),
                "ttft": t.get("ttft"), "prefill_s": t.get("prefill") or 0.0,
                "decode_s": t.get("decode") or 0.0, "total_s": t.get("total") or 0.0,
                "queue_s": (s.get("engine_timing") or {}).get("queue_wait_seconds") or 0.0,
                "end": s["timestamp_unix_ms"] / 1000.0,
                "spec_accepted": spec.get("accepted_tokens") or 0,
                "spec_drafted": spec.get("drafted_tokens") or 0,
                "ngram_accepted": spec.get("ngram_accepted_tokens") or 0,
                "ngram_drafted": spec.get("ngram_drafted_tokens") or 0,
                "rounds": spec.get("rounds") or 0,
            })
        reqs.append(r)
    build = "?"
    serve_log = os.path.join(d, "serve.log")
    if os.path.exists(serve_log):
        with open(serve_log, encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = re.search(r"build ([0-9a-f]{7,40})", ln)
                if m:
                    build = m.group(1)
                    break
    return {"arm": arm, "requests": reqs, "events": events, "start": start,
            "throughput": thr, "meta": meta, "errors": errs, "build": build}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def median(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def decode_windows(A):
    """The serve's throughput records as decode samples.

    Each record covers ~5 s and carries the decode tokens committed in it, the engine's own
    decode time (device wait + host work of decode rounds, which excludes prefill chunks and
    idle time), the decode rounds, and the request-rows those rounds ran. A record whose rows
    equal `b` x its rounds decoded `b` requests in every round."""
    out = []
    for o in A["throughput"]:
        batch = o.get("decode_batch") or {}
        work = (o.get("host_work") or {}).get("work_class_seconds") or {}
        rounds, rows = batch.get("rounds") or 0, batch.get("row_rounds") or 0
        busy = (work.get("decode_device_wait") or 0.0) + (work.get("decode_host") or 0.0)
        if rounds <= 0 or busy <= 0.0:
            continue
        out.append({"t": o.get("timestamp_unix_ms", 0) / 1000.0, "rounds": rounds, "rows": rows,
                    "tokens": (o.get("tokens") or {}).get("committed_decode", 0), "busy": busy})
    return sorted(out, key=lambda w: w["t"])


BOOTSTRAP_BLOCK = 6        # consecutive 5 s records resampled together (~30 s of decoding)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_MIN_BLOCKS = 5   # fewer blocks than this give no meaningful interval


def decode_rate(windows):
    """Output tok/s of engine decode time, split into engine speed and speculative acceptance.

    tokens / decode time = decode rounds per second (the engine's speed at this batch size)
    x tokens per round. Rounds per second is what a build's kernels and host work set; tokens
    per request-round is how many drafted tokens were accepted, which moves with the text being
    written (a file copy accepts far more than fresh reasoning) as much as with the drafters.
    Each is a ratio of sums with a 95 % block-bootstrap interval: consecutive records usually
    decode the same requests, so they are resampled in blocks, and the interval reflects how much
    the ratio moves with which turns happened to decode. With fewer than BOOTSTRAP_MIN_BLOCKS
    blocks an estimate has no interval (`lo`/`hi` are None)."""
    blocks = [windows[i:i + BOOTSTRAP_BLOCK] for i in range(0, len(windows), BOOTSTRAP_BLOCK)]
    sums = [{k: sum(w[k] for w in b) for k in ("tokens", "busy", "rounds", "rows")}
            for b in blocks]
    ratios = {"tps": ("tokens", "busy"), "rounds_per_s": ("rounds", "busy"),
              "tokens_per_row_round": ("tokens", "rows")}
    samples = {k: [] for k in ratios}
    rng = random.Random(0)
    for _ in range(BOOTSTRAP_RESAMPLES if len(sums) >= BOOTSTRAP_MIN_BLOCKS else 0):
        drawn = [sums[rng.randrange(len(sums))] for _ in sums]
        for k, (num, den) in ratios.items():
            samples[k].append(sum(b[num] for b in drawn) / sum(b[den] for b in drawn))
    out = {"seconds": sum(b["busy"] for b in sums), "records": len(windows)}
    for k, (num, den) in ratios.items():
        total = sum(b[den] for b in sums)
        out[k] = {"value": sum(b[num] for b in sums) / total if total else None,
                  "lo": pct(samples[k], 0.025), "hi": pct(samples[k], 0.975)}
    return out


def decode_metrics(A):
    """Decode rates by how many requests decoded together, from engine decode time.

    `one` and `two` use only records in which every round decoded one (two) requests, so a
    build's batching mix cannot move them; `all` is every decode record at the batching the
    run produced. `two_request_rounds` is the share of decode rounds that ran two requests."""
    windows = decode_windows(A)
    rounds = sum(w["rounds"] for w in windows)
    return {
        "one": decode_rate([w for w in windows if w["rows"] == w["rounds"]]),
        "two": decode_rate([w for w in windows if w["rows"] == 2 * w["rounds"]]),
        "all": decode_rate(windows),
        "two_request_rounds": (sum(w["rows"] - w["rounds"] for w in windows) / rounds
                               if rounds else None),
        "mean_decode_batch": sum(w["rows"] for w in windows) / rounds if rounds else None,
    }


REQUEST_BLOCK = 10         # consecutive requests resampled together for request-metric intervals


def request_intervals(rows):
    """95 % block-bootstrap intervals for the TTFT and cache rows of one run.

    Requests are taken in send order and resampled in blocks of REQUEST_BLOCK: nearby requests
    share queueing and cache state, so a block keeps them together. The interval shows how much
    a metric moves with which stretches of the run it happened to contain; it cannot show how
    differently another run would interleave, which is what repeated seeds are for."""
    rows = sorted(rows, key=lambda r: r.get("t_send") or 0)
    blocks = [rows[i:i + REQUEST_BLOCK] for i in range(0, len(rows), REQUEST_BLOCK)]
    if len(blocks) < BOOTSTRAP_MIN_BLOCKS:
        return {}
    stats = {
        "ttft_mean": lambda rs: mean([r["ttft"] for r in rs]),
        "ttft_median": lambda rs: median([r["ttft"] for r in rs]),
        "ttft_p90": lambda rs: pct([r["ttft"] for r in rs], 0.9),
        "ttft_cont_mean": lambda rs: mean([r["ttft"] for r in rs if r["cls"] in CONTINUING]),
        "hit_rate": lambda rs: (sum(r["hit"] for r in rs) / sum(r["prompt"] for r in rs)
                                if sum(r["prompt"] for r in rs) else None),
        "computed_tokens": lambda rs: sum(r["computed"] for r in rs),
        "cont_full_prefill_main": lambda rs: sum(
            1 for r in rs if r["cls"] in CONTINUING and r["cls"] != "subagent_loop"
            and r["hit"] == 0),
    }
    samples = {k: [] for k in stats}
    rng = random.Random(1)
    for _ in range(BOOTSTRAP_RESAMPLES):
        drawn = [r for _ in blocks for r in blocks[rng.randrange(len(blocks))]]
        for k, f in stats.items():
            v = f(drawn)
            if v is not None:
                samples[k].append(v)
    return {k: (pct(v, 0.025), pct(v, 0.975)) for k, v in samples.items() if v}


def metrics(A, cold_seeds):
    rows = [r for r in A["requests"] if r.get("server") and r["cls"] != "aborted"]
    cont = [r for r in rows if r["cls"] in CONTINUING]
    newl = [r for r in rows if r["cls"] in NEW_LONG]
    cold = [r for r in rows if r["seed"] in cold_seeds]
    dec = [r for r in rows if r["decode_s"] > 0 and r["completion"] > 1]
    copy = [r for r in dec if r["copy"]]
    big = [r for r in cold if r["computed"] >= 32768]
    budget = ((A["start"] or {}).get("server") or {}).get("default_thinking_budget")
    tp = sum(r["prompt"] for r in rows)
    th = sum(r["hit"] for r in rows)
    m = {
        "n": len(rows),
        "n_failed": sum(1 for r in A["requests"] if r["cls"] != "aborted" and not r.get("server")),
        "ttft_mean": mean([r["ttft"] for r in rows]),
        "ttft_median": median([r["ttft"] for r in rows]),
        "ttft_p90": pct([r["ttft"] for r in rows], 0.9),
        "service_ttft_mean": mean([r["ttft"] - r["queue_s"] for r in rows]),
        "queue_mean": mean([r["queue_s"] for r in rows]),
        "ttft_cont_mean": mean([r["ttft"] for r in cont]),
        "ttft_cont_median": median([r["ttft"] for r in cont]),
        "ttft_new_mean": mean([r["ttft"] for r in newl]),
        "n_cont": len(cont), "n_new": len(newl),
        "prompt_tokens": tp, "hit_tokens": th, "hit_rate": th / tp if tp else None,
        "computed_tokens": sum(r["computed"] for r in rows),
        "cont_full_prefill": sum(1 for r in cont if r["hit"] == 0),
        "cont_full_prefill_main": sum(1 for r in cont if r["hit"] == 0 and r["cls"] != "subagent_loop"),
        "cont_full_prefill_sub": sum(1 for r in cont if r["hit"] == 0 and r["cls"] == "subagent_loop"),
        "n_cont_main": sum(1 for r in cont if r["cls"] != "subagent_loop"),
        "n_cont_sub": sum(1 for r in cont if r["cls"] == "subagent_loop"),
        "cold_prefill_big": (sum(r["computed"] for r in big) /
                             sum(r["prefill_s"] for r in big)) if big else None,
        "n_cold_big": len(big),
        "decode": decode_metrics(A),
        "thinking_budget_hits": sum(1 for r in rows if budget and r["thinking"] >= budget),
        "budget_hit_ngram": sum(r["ngram_accepted"] for r in rows
                                if budget and r["thinking"] >= budget),
        "budget_hit_tokens": sum(r["completion"] for r in rows
                                 if budget and r["thinking"] >= budget),
        "cont_hit_rate": (sum(r["hit"] for r in cont) / max(1, sum(r["prompt"] for r in cont))),
        "cold_prefill_median": median([r["computed"] / r["prefill_s"] for r in cold if r["prefill_s"]]),
        "cold_prefill_aggregate": (sum(r["computed"] for r in cold) /
                                   max(1e-9, sum(r["prefill_s"] for r in cold))) if cold else None,
        "n_cold": len(cold),
        "output_tps": (sum(r["completion"] for r in dec) / sum(r["decode_s"] for r in dec)) if dec else None,
        "output_tps_copy": (sum(r["completion"] for r in copy) / sum(r["decode_s"] for r in copy)) if copy else None,
        "n_copy": len(copy),
        "completion_tokens": sum(r["completion"] for r in rows),
        "thinking_tokens": sum(r["thinking"] for r in rows),
        "decode_seconds": sum(r["decode_s"] for r in rows),
        "spec_accept": (sum(r["spec_accepted"] for r in rows) /
                        max(1, sum(r["spec_drafted"] for r in rows))),
        "ngram_accepted": sum(r["ngram_accepted"] for r in rows),
        "ngram_drafted": sum(r["ngram_drafted"] for r in rows),
        "wall_seconds": A["meta"].get("wall_seconds"),
        "context_guards": sum(1 for e in A["events"] if e.get("event") == "context_guard"),
        "intervals": request_intervals(rows),
    }
    paths = {}
    for r in rows:
        d = paths.setdefault(r["path"], [0, 0])
        d[0] += 1
        d[1] += r["hit"]
    m["paths"] = paths
    pressure = {}
    for o in A["throughput"]:
        cc = o.get("context_cache") or {}
        for k, v in (cc.get("pressure") or {}).items():
            if isinstance(v, (int, float)):
                pressure[k] = pressure.get(k, 0) + v
        for kind in ("main_kv_transfers",):
            for d, t in ((cc.get(kind) or {}).items()):
                pressure["%s_%s_bytes" % (kind, d)] = pressure.get("%s_%s_bytes" % (kind, d), 0) + \
                    (t.get("bytes") or 0)
    m["pressure"] = pressure
    return m


# ---------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------

def f1(v):
    return "n/a" if v is None else "%.1f" % v


def f2(v):
    return "n/a" if v is None else "%.2f" % v


def f0(v):
    return "n/a" if v is None else "%.0f" % v


def ntok(v):
    return "n/a" if v is None else format(int(round(v)), ",d")


def ppct(v):
    return "n/a" if v is None else "%.1f %%" % (v * 100)


def with_ci(fmt, seconds_key=None):
    """Formats a decode estimate with its 95 % interval, e.g. `187 (184-190)`."""
    def render(d):
        e, seconds = d
        if e["value"] is None:
            return "n/a"
        if e["lo"] is None:
            return "%s (%.0f s, too little for an interval)" % (fmt(e["value"]), seconds)
        return "%s (%s-%s)" % (fmt(e["value"]), fmt(e["lo"]), fmt(e["hi"]))
    return render


def chg(c, t):
    if c in (None, 0) or t is None:
        return "n/a"
    d = (t - c) / c * 100
    return "%s%.1f %%" % ("+" if d >= 0 else "−", abs(d))


def chg_pp(c, t):
    if c is None or t is None:
        return "n/a"
    d = (t - c) * 100
    return "%s%.1f points" % ("+" if d >= 0 else "−", abs(d))


def chg_abs(c, t):
    if c is None or t is None:
        return "n/a"
    d = t - c
    return "%s%d" % ("+" if d >= 0 else "−", abs(d))


def headline_specs(c):
    """The comparison rows: (label, metric path, format, change kind, divisor).

    A path ending in a decode estimate renders with its interval; a request metric renders with
    its interval from `intervals` when it has one. Change kinds: `rel` percent, `pp` points,
    `abs` count difference."""
    return [
        ("Average time to first token (s, lower is better)", ("ttft_mean",), f1, "rel", None),
        ("Median time to first token (s, lower is better)", ("ttft_median",), f2, "rel", None),
        ("90th-percentile time to first token (s, lower is better)", ("ttft_p90",), f1, "rel",
         None),
        ("Average TTFT, continuing-session turns (s)", ("ttft_cont_mean",), f2, "rel", None),
        ("Average TTFT, new long prompts (s)", ("ttft_new_mean",), f1, "rel", None),
        ("Prompt tokens served from cache", ("hit_rate",), ppct, "pp", None),
        ("Prompt tokens prefilled (lower is better)", ("computed_tokens",), ntok, "rel", None),
        ("Main-session turns that re-prefilled the whole prompt (of %d)" % c["n_cont_main"],
         ("cont_full_prefill_main",), f0, "abs", None),
        ("Subagent turns that re-prefilled the whole prompt (of %d)" % c["n_cont_sub"],
         ("cont_full_prefill_sub",), f0, "abs", None),
        ("Prefill tok/s, requests with no cache hit (all sizes)", ("cold_prefill_aggregate",),
         ntok, "rel", None),
        ("Prefill tok/s, requests with no cache hit, 32K+ tokens", ("cold_prefill_big",), ntok,
         "rel", None),
        ("Output tok/s, one request decoding", ("decode", "one", "tps"), ntok, "rel", None),
        ("Output tok/s, two requests decoding (combined)", ("decode", "two", "tps"), ntok, "rel",
         None),
        ("Output tok/s, all decoding at the run's own batching", ("decode", "all", "tps"), ntok,
         "rel", None),
        ("Decode rounds/s, one request decoding (engine speed)",
         ("decode", "one", "rounds_per_s"), f1, "rel", None),
        ("Decode rounds/s, two requests decoding (engine speed)",
         ("decode", "two", "rounds_per_s"), f1, "rel", None),
        ("Tokens per round, one request decoding (speculative acceptance)",
         ("decode", "one", "tokens_per_row_round"), f2, "rel", None),
        ("Decode rounds that ran two requests", ("decode", "two_request_rounds"), ppct, "pp",
         None),
        ("Workload wall time (min, lower is better)", ("wall_seconds",), f1, "rel", 60.0),
    ]


def resolve(m, path, divisor=None):
    """The scalar at `path` (a decode estimate's value), or None."""
    v = m
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    if isinstance(v, dict):
        v = v.get("value")
    return v / divisor if v is not None and divisor else v


def change_value(kind, c, t):
    if c is None or t is None:
        return None
    if kind == "rel":
        return (t - c) / c * 100 if c else None
    return (t - c) * 100 if kind == "pp" else t - c


def change_text(kind, d, spread=None):
    """`−12.3 %`, `+4.1 points` or `−7`, with an optional `(lo to hi)` range."""
    if d is None:
        return "n/a"
    unit = {"rel": " %", "pp": " points", "abs": ""}[kind]
    digits = "%.0f" if kind == "abs" and spread is None else "%.1f"

    def signed(x):
        return ("+" if x >= 0 else "−") + (digits % abs(x)) + unit

    if spread is None:
        return signed(d)
    return "%s (%s to %s)" % (signed(d), bare(signed(spread[0])), bare(signed(spread[1])))


def bare(text):
    """A formatted number without its unit, for the bounds of a range."""
    return text.replace(" points", "").replace(" %", "")


def cell(m, spec):
    """One arm's value in the single-run table, with its interval when it has one."""
    _, path, fmt, _, divisor = spec
    v = m
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    if isinstance(v, dict):   # decode estimate
        seconds = resolve(m, path[:-1] + ("seconds",)) or 0.0
        return with_ci(fmt)((v, seconds))
    value = resolve(m, path, divisor)
    ci = (m.get("intervals") or {}).get(path[-1]) if len(path) == 1 else None
    if value is None or not ci:
        return fmt(value)
    return "%s (%s-%s)" % (fmt(value), bare(fmt(ci[0])), bare(fmt(ci[1])))


def table_header(arms, change_label):
    others = arms[1:]
    return ["| Metric | %s | %s |" % (" | ".join(LABEL[a] for a in arms),
                                      " | ".join(change_label % (SHORT[a], SHORT[arms[0]])
                                                 for a in others)),
            "|---|" + "---|" * (len(arms) + len(others))]


def headline(arms, ms):
    """The comparison table: one column per arm, then each arm's change against the control."""
    L = table_header(arms, "%s vs %s")
    for spec in headline_specs(ms[arms[0]]):
        _, path, _, kind, divisor = spec
        c = resolve(ms[arms[0]], path, divisor)
        L.append("| %s | %s | %s |" % (
            spec[0], " | ".join(cell(ms[a], spec) for a in arms),
            " | ".join(change_text(kind, change_value(kind, c, resolve(ms[a], path, divisor)))
                       for a in arms[1:])))
    return "\n".join(L)


def each(arms, value, fmt=lambda v: v):
    """`control 1.2, this 3.4, hybrid 5.6`, for the notes."""
    return ", ".join("%s %s" % (SHORT[a].lower(), fmt(value(a))) for a in arms)


def cold_table(arms, A, cold_seeds):
    served = {a: {r["seed"]: r for r in A[a]["requests"] if r.get("server")} for a in arms}
    others = arms[1:]
    L = ["| Request | Class | Prompt tokens | %s | %s |"
         % (" | ".join("%s prefill tok/s" % SHORT[a] for a in arms),
            " | ".join("%s vs %s" % (SHORT[a], SHORT[arms[0]]) for a in others)),
         "|---|---|---|" + "---|" * (len(arms) + len(others))]
    base = served[arms[0]]
    for s in sorted(cold_seeds, key=lambda s: base[s]["computed"]):
        rates = [served[a][s]["computed"] / served[a][s]["prefill_s"]
                 if served[a][s]["prefill_s"] else None for a in arms]
        L.append("| %s | %s | %s | %s | %s |" % (
            base[s]["tag"], base[s]["cls"], ntok(base[s]["computed"]),
            " | ".join(ntok(r) for r in rates), " | ".join(chg(rates[0], r) for r in rates[1:])))
    return "\n".join(L)


def class_table(arms, A):
    L = ["| Class | n | %s | %s |" % (" | ".join("%s avg TTFT (s)" % SHORT[a] for a in arms),
                                     " | ".join("%s cached" % SHORT[a] for a in arms)),
         "|---|---|" + "---|" * (2 * len(arms))]
    for cls, _ in CLASS_DOC:
        rs = [[r for r in A[a]["requests"] if r["cls"] == cls and r.get("server")] for a in arms]
        if not any(rs):
            continue
        L.append("| %s | %d | %s | %s |" % (
            cls, max(len(r) for r in rs), " | ".join(f2(mean([x["ttft"] for x in r])) for r in rs),
            " | ".join(ppct(sum(x["hit"] for x in r) / max(1, sum(x["prompt"] for x in r)))
                       for r in rs)))
    return "\n".join(L)


def path_table(arms, ms):
    L = ["| Reuse path | %s |" % " | ".join("%s (requests / cached tokens)" % SHORT[a] for a in arms),
         "|---|" + "---|" * len(arms)]
    for p in sorted(set().union(*(ms[a]["paths"] for a in arms))):
        cells = [ms[a]["paths"].get(p, [0, 0]) for a in arms]
        L.append("| %s | %s |" % (p, " | ".join("%d / %s" % (n, ntok(t)) for n, t in cells)))
    return "\n".join(L)


def pressure_table(arms, ms):
    keys = ["private_owners_evicted", "private_owners_degraded", "shared_owners_evicted",
            "shared_owners_degraded", "checkpoints_dropped", "maximal_fallback_selections",
            "search_budget_exhaustions", "spill_pages", "main_kv_transfers_d2h_bytes",
            "main_kv_transfers_h2d_bytes"]
    L = ["| Engine counter (summed over the run) | %s |" % " | ".join(SHORT[a] for a in arms),
         "|---|" + "---|" * len(arms)]
    for k in keys:
        v = [ms[a]["pressure"].get(k) for a in arms]
        if all(x is None for x in v):
            continue
        fmt = (lambda v: "n/a" if v is None else "%.1f GiB" % (v / 2 ** 30)) if k.endswith("bytes") \
            else (lambda v: "n/a" if v is None else format(int(v), ",d"))
        L.append("| %s | %s |" % (k, " | ".join(fmt(x) for x in v)))
    return "\n".join(L)


def per_request(A):
    L = ["| Tag | Class | Prompt | Cached | Prefilled | Path | Queue s | TTFT s | Out | Decode tok/s | Finish |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(A["requests"], key=lambda r: r.get("t_send") or 0):
        if not r.get("server"):
            L.append("| %s | %s | | | | | | | | | %s |" % (r["tag"], r["cls"], r["status"]))
            continue
        L.append("| %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["cls"], " (copy)" if r["copy"] else "", ntok(r["prompt"]), ntok(r["hit"]),
            ntok(r["computed"]), r["path"], f1(r["queue_s"]), f2(r["ttft"]), ntok(r["completion"]),
            ntok(r["completion"] / r["decode_s"]) if r["decode_s"] else "n/a", r["finish_server"]))
    return "\n".join(L)


def server_row(A):
    s = A["start"] or {}
    e, mem = s.get("engine") or {}, s.get("memory") or {}
    cc = e.get("context_cache") or {}
    return ("KV capacity %s tokens; host state slots %s; host KV %.1f GiB; private "
            "continuations %s; long anchors per continuation %s; shared prefixes %s; build %s"
            % (ntok(e.get("kv_capacity")), cc.get("host_state_slots"),
               (mem.get("host_kv_capacity_bytes") or 0) / 2 ** 30, cc.get("max_private_continuations"),
               cc.get("max_long_anchors_per_continuation"), cc.get("max_shared_prefixes"),
               A["build"]))


def flag_str(flags):
    return " ".join(n + ("" if v is None else " " + (('"%s"' % v) if " " in v else v))
                    for n, v in flags)


def aggregate(out_dir, run_dirs):
    """One report over runs of the same arms with different workload seeds.

    Each cell is the mean over seeds with the min-max range; each change is the per-seed change
    against that seed's control, averaged, with its min-max range. A seed changes every
    observation and every sampled answer, so the ranges show how much a result depends on the
    particular session rather than on the build."""
    runs = []
    for d in run_dirs:
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
            runs.append((d, json.load(f)))
    arms = [a for a in ARMS if all(a in s for _, s in runs)]
    if arms[:1] != ["control"] or len(arms) < 2:
        raise SystemExit("aggregate needs a control arm and one other arm in every run")
    seeds = [s["config"]["seed"] for _, s in runs]
    L = ["# Agentic A/B over %d workload seeds: %s\n" % (len(runs), " vs ".join(LABEL[a] for a in arms)),
         "Runs: %s\n" % ", ".join("seed %d `%s`" % (seed, d) for seed, (d, _) in zip(seeds, runs))]
    first = runs[0][1]
    L.append("- Model: `%s`, `--max-context %s`; %d requests per arm per seed; every arm of a "
             "seed replays the same observations, and each seed replays different ones.\n"
             % (os.path.basename(first["config"]["model"]),
                ntok(first["config"].get("max_context")), first["control"]["n"]))
    problems = ["seed %d %s: %d failed request(s)" % (seed, a, s[a]["n_failed"])
                for seed, (_, s) in zip(seeds, runs) for a in arms if s[a]["n_failed"]]
    if problems:
        L.append("**Validity warnings:** " + "; ".join(problems) + "\n")
    L.append("## Headline (mean over seeds, min-max in brackets)\n")
    L += table_header(arms, "%s vs %s, per seed")
    for label, path, fmt, kind, divisor in headline_specs(first["control"]):
        fmt = f1 if kind == "abs" else fmt
        cells = []
        for a in arms:
            v = [resolve(s[a], path, divisor) for _, s in runs]
            v = [x for x in v if x is not None]
            cells.append("n/a" if not v else fmt(sum(v) / len(v)) if len(v) == 1 else
                         "%s (%s-%s)" % (fmt(sum(v) / len(v)), bare(fmt(min(v))),
                                         bare(fmt(max(v)))))
        changes = []
        for a in arms[1:]:
            d = [change_value(kind, resolve(s["control"], path, divisor),
                              resolve(s[a], path, divisor)) for _, s in runs]
            d = [x for x in d if x is not None]
            changes.append("n/a" if not d else
                           change_text(kind, sum(d) / len(d),
                                       (min(d), max(d)) if len(d) > 1 else None))
        L.append("| %s | %s | %s |" % (label, " | ".join(cells), " | ".join(changes)))
    L.append("")
    L.append("## Per seed\n")
    keys = [("Prompt tokens served from cache", ("hit_rate",), ppct, None),
            ("Average TTFT (s)", ("ttft_mean",), f1, None),
            ("Output tok/s, one request", ("decode", "one", "tps"), ntok, None),
            ("Decode rounds/s, one request", ("decode", "one", "rounds_per_s"), f1, None),
            ("Completion tokens", ("completion_tokens",), ntok, None),
            ("Wall time (min)", ("wall_seconds",), f1, 60.0)]
    L.append("| Seed | Arm | %s |" % " | ".join(k[0] for k in keys))
    L.append("|---|---|" + "---|" * len(keys))
    for seed, (_, s) in zip(seeds, runs):
        for a in arms:
            L.append("| %d | %s | %s |" % (seed, SHORT[a], " | ".join(
                fmt(resolve(s[a], path, divisor)) for _, path, fmt, divisor in keys)))
    L.append("")
    L.append("Each seed's own report (`report.md` in its run directory) has the intervals, notes "
             "and per-request detail for that seed.")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("report: %s" % os.path.join(out_dir, "report.md"))


def main(argv):
    if argv[:1] == ["--aggregate"]:
        aggregate(argv[1], argv[2:])
        return
    run_dir = argv[0]
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    arms = [a for a in ARMS if os.path.exists(os.path.join(run_dir, a, "client.jsonl"))]
    if arms[:1] != ["control"] or len(arms) < 2:
        raise SystemExit("%s needs a control arm and at least one other arm" % run_dir)
    A = {a: load_arm(run_dir, a) for a in arms}
    # Matched cold set: requests with no cache hit in EVERY arm and a real prefill.
    served = [{r["seed"]: r for r in A[a]["requests"] if r.get("server")} for a in arms]
    cold = {s for s in served[0] if all(s in d and d[s]["hit"] == 0 and d[s]["computed"] >= 4096
                                        for d in served)}
    ms = {a: metrics(A[a], cold) for a in arms}
    summary = dict({"config": cfg, "cold_seeds": sorted(cold)}, **ms)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    c = ms["control"]
    ctx = cfg.get("max_context")
    gpu = next((A[a]["start"].get("environment", {}).get("gpu_name")
                for a in reversed(arms) if A[a]["start"]), None) or "GPU"
    L = []
    L.append("# Agentic A/B: %s\n" % " vs ".join(LABEL[a] for a in arms))
    L.append("Run directory: `%s`\n" % run_dir)
    L.append("- Model: `%s` on an %s, `--max-context %s` for every arm (launch bat: %s)."
             % (os.path.basename(cfg["model"]), gpu, ntok(ctx), ntok(cfg.get("bat_max_context"))))
    L.append("- Workload: %d requests per arm (seed %d, scale %.2f, corpus commit `%s`), "
             "`max_tokens: %d` on agent turns, thinking on."
             % (c["n"], cfg["seed"], cfg["scale"], cfg["corpus_commit"], cfg["agent_max_tokens"]))
    exes = {"control": cfg["control_exe"], "treatment": cfg["treatment_exe"],
            "alt": cfg["treatment_exe"]}
    for a in arms:
        extra = (" with `%s`" % " ".join(cfg.get("alt_extra_flags") or [])) if a == "alt" else ""
        L.append("- %s: `%s`%s - %s" % (LABEL[a], exes[a], extra, server_row(A[a])))
    L.append("- Total benchmark time including model loads: %.1f min.\n"
             % (cfg.get("total_seconds", 0) / 60))
    problems = []
    for a in arms:
        if ms[a]["n_failed"]:
            problems.append("%s: %d workload request(s) have no request_done record"
                            % (a, ms[a]["n_failed"]))
        if ms[a]["context_guards"]:
            problems.append("%s: the client context guard fired %d time(s)"
                            % (a, ms[a]["context_guards"]))
    if problems:
        L.append("**Validity warnings:** " + "; ".join(problems) + "\n")

    def per_arm(value, fmt=str):
        return each(arms, lambda a: value(ms[a]), fmt)

    L.append("## Headline\n")
    L.append(headline(arms, ms) + "\n")
    L.append("How to read it:\n")
    L.append("- Every row compares the same logical requests: the client tags each one with a "
             "seed that is identical in every arm. Observations (tool results, user messages, "
             "summaries) are byte-identical; assistant turns are each arm's own output fed back, "
             "as an agent client does, so prompt totals differ slightly (tokens: %s)."
             % per_arm(lambda m: m["prompt_tokens"], ntok))
    L.append("- TTFT includes queueing behind other sessions (the serve runs 2 lanes and up to "
             "7 requests are in flight). Average queue wait (s): %s; average TTFT without queue "
             "wait (s): %s."
             % (per_arm(lambda m: m["queue_mean"], f1),
                per_arm(lambda m: m["service_ttft_mean"], f2)))
    L.append("- Continuing-session turns (%d) are tool-loop turns, retries, the post-idle and "
             "post-history-edit turns: cache retention decides their TTFT. New long prompts (%d) "
             "are resumed sessions, compaction and loop-check calls: prefill speed decides theirs."
             % (c["n_cont"], c["n_new"]))
    L.append("- \"No cache hit\" prefill rates are token-weighted (total prefilled tokens / "
             "total prefill time) over the %d requests that had no cache hit in *every* arm and "
             "prefilled at least 4,096 tokens (%d of them 32K+; per-request table below). The "
             "per-request median (tok/s) is %s: most of those requests are ~11-17K subagent "
             "prompts, where the prompt-attention kernel matters least."
             % (c["n_cold"], c["n_cold_big"], per_arm(lambda m: m["cold_prefill_median"], ntok)))
    L.append("- Output tok/s is decode tokens the server committed per second of the engine's "
             "own decode time (device wait plus host work of decode rounds, from the serve's "
             "~5 s throughput records), so prefill chunks and idle time do not dilute it. The "
             "one- and two-request rows use only the records in which every decode round ran "
             "that many requests (seconds of decode time, one / two requests: %s), so a build's "
             "batching mix cannot move them; the *all* row is every decode record at the batching "
             "the run produced (mean decode batch: %s). Brackets are 95 %% block-bootstrap "
             "intervals over ~30 s stretches of decoding."
             % (per_arm(lambda m: "%s / %s" % (f1(m["decode"]["one"]["seconds"]),
                                               f1(m["decode"]["two"]["seconds"]))),
                per_arm(lambda m: m["decode"]["mean_decode_batch"], f2)))
    L.append("- Output tok/s = decode rounds/s x tokens per round. Rounds/s is the engine's own "
             "speed and barely moves with the text; tokens per round is speculative acceptance, "
             "which moves with what the model happened to write (file copies accept far more than "
             "fresh reasoning), so it carries most of the interval on output tok/s. Compare "
             "rounds/s for kernel and host speed, and tokens per round for drafting; each arm "
             "samples its own text, so acceptance differs between runs as well as builds. Tokens "
             "per request-round with two requests decoding: %s. Per-request completion / decode "
             "wall time (tok/s: %s; file-writing turns: %s) is what one stream saw, including "
             "other lanes' batching and prefill chunks."
             % (per_arm(lambda m: m["decode"]["two"]["tokens_per_row_round"]["value"], f2),
                per_arm(lambda m: m["output_tps"], ntok),
                per_arm(lambda m: m["output_tps_copy"], ntok)))
    L.append("- Output volume differs because the sampled text does: completion tokens %s "
             "(thinking: %s); turns that used the whole thinking budget: %s (ngram copies "
             "supplied %s of those turns' tokens; a high share signals repetitive thinking). "
             "Longer outputs hold more KV and add batching, so they also shift cache pressure "
             "and queueing; compare repeated runs before attributing a thinking-length "
             "difference to a build."
             % (per_arm(lambda m: m["completion_tokens"], ntok),
                per_arm(lambda m: m["thinking_tokens"], ntok),
                per_arm(lambda m: m["thinking_budget_hits"]),
                per_arm(lambda m: "%s of %s" % (ntok(m["budget_hit_ngram"]),
                                                ntok(m["budget_hit_tokens"])))))
    L.append("- Speculative acceptance of drafted tokens: %s. Ngram drafting accepted %s."
             % (per_arm(lambda m: m["spec_accept"], ppct),
                per_arm(lambda m: "%s of %s drafted tokens (%s of output)"
                        % (ntok(m["ngram_accepted"]), ntok(m["ngram_drafted"]),
                           ppct(m["ngram_accepted"] / max(1, m["completion_tokens"]))))))
    L.append("")
    L.append("## Prefill on requests with no cache hit\n")
    L.append(cold_table(arms, A, cold) + "\n")
    L.append("## By request class\n")
    L.append(class_table(arms, A) + "\n")
    L.append("Classes: " + "; ".join("`%s` %s" % c for c in CLASS_DOC) + ".\n")
    L.append("## Where cache hits came from\n")
    L.append(path_table(arms, ms) + "\n")
    L.append("## Cache pressure\n")
    L.append(pressure_table(arms, ms) + "\n")
    L.append("## Launch parameters\n")
    for a in arms:
        if a == "control":
            L.append("%s (flags it does not know dropped: %s; a `--host-cache-mib` it does not "
                     "know is replaced by the explicit host-cache flags it resolved "
                     "to):\n\n```text\n%s\n```\n"
                     % (LABEL[a], ", ".join("`%s`" % d for d in cfg.get("dropped_for_control", [])),
                        flag_str(cfg.get("control_flags", []))))
        else:
            L.append("%s:\n\n```text\n%s\n```\n" % (LABEL[a], flag_str(cfg.get(a + "_flags", []))))
    L.append("## Per-request detail\n")
    for a in arms:
        L.append("### %s\n\n%s\n" % (LABEL[a], per_request(A[a])))
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(headline(arms, ms))
    print("report: %s" % os.path.join(run_dir, "report.md"))


if __name__ == "__main__":
    main(sys.argv[1:])
