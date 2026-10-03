#!/usr/bin/env python3
"""NInfer monitoring sidecar.

Polls the server's GET /stats, tails its JSONL request log (--request-log-jsonl), and samples
nvidia-smi, /proc and an optional temperature sensor command, then serves:
  - a self-contained HTML dashboard at /
  - a JSON snapshot at /api/samples
  - a liveness probe at /healthz

Stdlib only. One sampler thread ticks every --interval seconds and appends a sample to a bounded
ring buffer; the HTTP handlers read that state. The sidecar survives the inference server being
down or restarting: failed polls degrade to a "stale" banner, and the request-log tail follows
rotation by rename or truncation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# /stats reads published snapshots, so a slow poll means an unreachable server rather than a busy
# one. The server counts as down once the last successful poll is this old, so a single failed
# poll does not flap the banner.
POLL_TIMEOUT_S = 5
STALE_AFTER_MS = 60_000
SENSOR_TIMEOUT_S = 15


class Config:
    def __init__(self, args: argparse.Namespace) -> None:
        self.port = args.port
        self.bind = args.bind
        self.stats_url = args.stats_url.rstrip("/")
        self.api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        self.jsonl = args.jsonl
        self.interval = args.interval
        self.samples = args.samples
        self.events_cap = 500
        self.kv_log_cap = 200
        self.window_seconds = 3600
        self.sensor_command = shlex.split(args.sensor_command) if args.sensor_command else None
        self.sensor_label = args.sensor_label
        self.sensor_warn_c = args.sensor_warn_c
        self.sensor_critical_c = args.sensor_critical_c
        self.stop_pid_file = args.stop_server_pid_file
        self.stop_after_s = args.stop_after_s
        self.pidfile = args.pidfile


class Tail:
    """Incrementally tail a file, surviving rotation (rename) and truncation.

    On first open it seeks to the end (no backfill). Each read_lines() call returns the complete
    lines appended since the last call. If the file's inode changes (renamed away) or its size
    shrinks below our position (truncated), it reopens at the start of the new file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.f = None
        self.ino = None
        self.pos = 0
        self.buf = ""
        self._open(at_end=True)

    def _open(self, at_end: bool) -> None:
        try:
            self.f = open(self.path, "rb")
            st = os.fstat(self.f.fileno())
            self.ino = st.st_ino
            self.f.seek(0, 2 if at_end else 0)
            self.pos = self.f.tell()
        except OSError:
            self.f = None
            self.ino = None
            self.pos = 0
        self.buf = ""

    def read_lines(self) -> list[str]:
        lines: list[str] = []
        if self.f is None:
            self._open(at_end=False)
            if self.f is None:
                return lines
        try:
            st = os.stat(self.path)
        except OSError:
            self.f = None
            return lines
        if st.st_ino != self.ino or st.st_size < self.pos:
            # A rotation starts a new file that the server writes from its beginning.
            self._open(at_end=False)
        self.f.seek(self.pos)
        data = self.f.read()
        if not data:
            return lines
        self.pos += len(data)
        text = self.buf + data.decode("utf-8", "replace")
        *complete, self.buf = text.split("\n")
        for ln in complete:
            if ln.strip():
                lines.append(ln)
        return lines


def flatten_stats(stats: dict, memory: dict) -> dict:
    """Maps the server's /stats body onto the flat fields the dashboard draws."""
    capacity = stats.get("capacity", {})
    occupancy = stats.get("occupancy", {})
    requests = stats.get("requests", {})
    cache = stats.get("context_cache", {})
    pressure = cache.get("pressure", {})
    cache_occupancy = cache.get("occupancy", {})
    selections = cache.get("selections", {})

    def transfers(block: str, unit: str) -> dict:
        out = {}
        for direction in ("d2h", "h2d", "d2d"):
            leg = cache.get(block, {}).get(direction, {})
            out[f"{direction}_{unit}"] = leg.get(unit, 0)
            out[f"{direction}_bytes"] = leg.get("bytes", 0)
            out[f"{direction}_seconds"] = leg.get("seconds", 0)
        return out

    main_kv = transfers("main_kv_transfers", "pages")
    backend_kv = transfers("backend_kv_transfers", "pages")
    state = transfers("state_transfers", "count")
    return {
        "http": {"in_flight": requests.get("admitted"),
                 "max_in_flight": requests.get("peak_admitted")},
        "scheduler": stats.get("scheduler", {}),
        "counters": stats.get("counters", {}),
        "memory": {
            "kv_capacity_page_groups": capacity.get("kv_capacity_pages", 0),
            "host_kv_capacity_bytes": capacity.get("host_kv_bytes", 0),
            "host_kv_occupied_bytes": occupancy.get("host_kv_bytes", 0),
            "host_state_capacity_slots": capacity.get("host_state_slots", 0),
            "host_state_occupied_slots": occupancy.get("host_state_slots", 0),
            "kv_payload_bytes": memory.get("kv_payload_bytes"),
        },
        "kv_transfers": {**{f"main_kv_{k}": v for k, v in main_kv.items()},
                         **{f"backend_kv_{k}": v for k, v in backend_kv.items()}},
        "state_transfers": {f"state_{k}": v for k, v in state.items()},
        "pressure": {
            **pressure,
            "device_main_kv_occupied_pages": occupancy.get("device_main_kv_pages", 0),
            "device_backend_kv_occupied_pages": cache_occupancy.get("device_backend_kv_pages", 0),
            "device_state_occupied_slots": occupancy.get("device_state_slots", 0),
        },
        "cache_reuse": {**{f"{k}_selections": v for k, v in selections.items()
                           if k != "reused_prompt_tokens"},
                        "reused_prompt_tokens": stats.get("counters", {}).get(
                            "reused_prompt_tokens", 0)},
    }


class Monitor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.samples: deque[dict] = deque(maxlen=cfg.samples)
        self.events: deque[dict] = deque(maxlen=cfg.events_cap)
        self.kv_log: deque[dict] = deque(maxlen=cfg.kv_log_cap)
        self.server_info: dict = {
            "up": False,
            "instance_id": None,
            "model": None,
            "memory": {},
            "last_stats_unix_ms": None,
            "last_good_stats": None,
        }
        self.last_counters: dict | None = None
        self.last_rates: dict | None = None
        self.last_cpu_idle = None
        self.last_cpu_total = None
        self.jsonl_tail = Tail(cfg.jsonl) if cfg.jsonl else None
        self._kv_totals: dict[str, int] = {}
        # The speculative backend in use (last seen in a request log record), e.g. "dflash2".
        # The dashboard labels the accept-rate tile, chart and log with it.
        self.spec_backend: str | None = None
        self._last_kv_counters: dict | None = None
        # When the sensor reading first crossed critical (epoch seconds); None while at or below.
        self._sensor_over_since: float | None = None
        self._last_stop_outcome: str | None = None
        self._lock = threading.Lock()

    def bootstrap_server_info(self) -> None:
        # The request-log tail starts at the end, so the server_start record written before this
        # sidecar started is found by one scan of the current file.
        if not self.cfg.jsonl:
            return
        try:
            with open(self.cfg.jsonl, "rb") as f:
                last = None
                for raw in f:
                    if b'"server_start"' in raw:
                        last = raw
            if last is not None:
                self._server_start(json.loads(last.decode("utf-8", "replace")))
        except (OSError, json.JSONDecodeError):
            pass

    def _server_start(self, rec: dict) -> None:
        self.server_info["instance_id"] = rec.get("server_instance_id")
        self.server_info["model"] = rec.get("server", {}).get("public_model_id")
        self.server_info["memory"] = rec.get("memory", {})

    # -- request log --------------------------------------------------------

    def handle_jsonl_line(self, line: str) -> None:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return
        event = rec.get("event")
        ts = rec.get("timestamp_unix_ms")
        if event == "server_start":
            self._server_start(rec)
        elif event == "request_done":
            self._record_done(rec, ts)
        elif event == "request_rejected":
            self._record_event("rejected", rec, ts)
        elif event == "request_error":
            self._record_event("error", rec, ts)

    def _record_done(self, rec: dict, ts) -> None:
        req = rec.get("request", {})
        res = rec.get("result", {})
        spec = rec.get("speculative", {})
        timings = rec.get("timings_seconds", {})
        ttft = timings.get("ttft")
        queue = rec.get("engine_timing", {}).get("queue_wait_seconds")
        backend = spec.get("backend")
        if backend and backend != "none":
            self.spec_backend = backend
        # proc = processing time = ttft - queue (restore and prefill, without the queue wait).
        proc = None
        if ttft is not None and queue is not None and 0 <= queue <= ttft:
            proc = ttft - queue
        decode = timings.get("decode") or 0
        self.events.append(
            {
                "t": ts,
                "kind": "done",
                "id": req.get("request_id"),
                "protocol": req.get("protocol"),
                "prompt_tokens": res.get("prompt_tokens"),
                "completion_tokens": res.get("completion_tokens"),
                "ttft": ttft,
                "queue": queue,
                "proc": proc,
                "finish_reason": res.get("finish_reason"),
                "prefix_hit": res.get("prefix_cache_hit_tokens", 0),
                "prefix_path": res.get("prefix_reuse_path"),
                "mtp_accepted": spec.get("accepted_tokens", 0),
                "mtp_drafted": spec.get("drafted_tokens", 0),
                "mtp_rounds": spec.get("rounds", 0),
                "decode_tps": (res.get("completion_tokens", 0) / decode) if decode > 0 else None,
            }
        )
        drafted = spec.get("drafted_tokens", 0)
        if drafted:
            self.kv_log.append({
                "t": ts,
                "line": (f"req#{req.get('request_id')} {backend} accepted "
                         f"{spec.get('accepted_tokens', 0)}/{drafted} in "
                         f"{spec.get('rounds', 0)} rounds"),
            })

    def _record_event(self, kind: str, rec: dict, ts) -> None:
        err = rec.get("error", {})
        self.events.append(
            {
                "t": ts,
                "kind": kind,
                "id": rec.get("request", {}).get("request_id"),
                "protocol": rec.get("request", {}).get("protocol"),
                "message": err.get("message") or err.get("code") or kind,
            }
        )

    # -- /stats poll --------------------------------------------------------

    def poll_stats(self) -> None:
        request = urllib.request.Request(self.cfg.stats_url + "/stats")
        if self.cfg.api_key:
            request.add_header("Authorization", "Bearer " + self.cfg.api_key)
        try:
            with urllib.request.urlopen(request, timeout=POLL_TIMEOUT_S) as r:
                stats = json.loads(r.read().decode("utf-8", "replace"))
        except (OSError, ValueError, urllib.error.URLError):
            stats = None
        now_ms = int(time.time() * 1000)
        if stats is not None:
            self.server_info["last_stats_unix_ms"] = now_ms
            self.server_info["last_good_stats"] = stats
            if stats.get("model"):
                self.server_info["model"] = stats["model"]
            counters = stats.get("counters", {})
            cur_decode = counters.get("committed_decode_tokens", 0)
            cur_prefill = counters.get("computed_prefill_tokens", 0)
            now = time.time()
            if self.last_counters is not None:
                dt = now - self.last_counters["t"]
                dd = cur_decode - self.last_counters["decode"]
                dp = cur_prefill - self.last_counters["prefill"]
                # A restart zeroes the counters: a negative delta skips the interval.
                if dt > 0 and dd >= 0 and dp >= 0:
                    self.last_rates = {"decode_tps": dd / dt, "prefill_tps": dp / dt}
                else:
                    self.last_rates = None
            self.last_counters = {"t": now, "decode": cur_decode, "prefill": cur_prefill}
        last = self.server_info.get("last_stats_unix_ms")
        self.server_info["up"] = last is not None and now_ms - last <= STALE_AFTER_MS

    # -- system pollers -----------------------------------------------------

    def read_gpu(self) -> dict | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,"
                    "temperature.gpu,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            p = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
            return {
                "mem_used_mb": float(p[0]),
                "mem_total_mb": float(p[1]),
                "util_pct": float(p[2]),
                "power_w": float(p[3]),
                "temp_c": float(p[4]),
                "power_limit_w": float(p[5]),
            }
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def read_cpu(self) -> float | None:
        try:
            with open("/proc/stat") as f:
                first = f.readline().split()
            vals = [int(x) for x in first[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
            total = sum(vals)
            pct = 0.0
            if self.last_cpu_total is not None and self.last_cpu_idle is not None:
                dt = total - self.last_cpu_total
                di = idle - self.last_cpu_idle
                if dt > 0:
                    pct = 100.0 * (1.0 - di / dt)
            self.last_cpu_idle = idle
            self.last_cpu_total = total
            return max(0.0, min(100.0, pct))
        except (OSError, ValueError, IndexError):
            return None

    def read_ram(self) -> dict | None:
        try:
            info: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    parts = rest.strip().split()
                    if parts:
                        info[key] = int(parts[0])  # kB
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", 0)
            return {"used_mb": (total - avail) / 1024.0, "total_mb": total / 1024.0}
        except (OSError, ValueError):
            return None

    def read_sensor(self) -> float | None:
        # The command prints a bare Celsius value. A failed read yields None (the tile shows "–")
        # and never stops the tick.
        if not self.cfg.sensor_command:
            return None
        try:
            out = subprocess.run(self.cfg.sensor_command, capture_output=True, text=True,
                                 timeout=SENSOR_TIMEOUT_S)
            m = re.search(r"-?\d+(?:\.\d+)?", out.stdout)
            return float(m.group(0)) if m else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _stop_server(self, temp_c: float) -> None:
        """SIGKILLs the exact server PID the operator's launcher wrote, never a name pattern."""
        try:
            with open(self.cfg.stop_pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGKILL)
            outcome = f"SIGKILL sent to pid {pid}"
        except (OSError, ValueError) as error:
            outcome = f"stop failed: {error}"
        # Every tick retries, but the request table records each distinct outcome once.
        if outcome == self._last_stop_outcome:
            return
        self._last_stop_outcome = outcome
        self.events.append(
            {
                "t": int(time.time() * 1000),
                "kind": "sensor_stop",
                "id": None,
                "protocol": None,
                "message": (f"{self.cfg.sensor_label} {temp_c:.1f}C > "
                            f"{self.cfg.sensor_critical_c:.0f}C - {outcome}"),
            }
        )

    # -- sampler tick -------------------------------------------------------

    def _rolling(self) -> dict:
        # Rolling acceptance, TTFT and throughput percentiles over the latest requests and
        # samples, so the dashboard can plot them as time series.
        done = [e for e in list(self.events)[-60:] if e["kind"] == "done"]
        acc = sum(e.get("mtp_accepted", 0) for e in done)
        dr = sum(e.get("mtp_drafted", 0) for e in done)
        ttfts = sorted(e["ttft"] for e in done if e.get("ttft") is not None)
        procs = sorted(e["proc"] for e in done if e.get("proc") is not None)
        recent = list(self.samples)[-60:]
        req_dec = sorted(e["decode_tps"] for e in done if e.get("decode_tps") is not None)
        pre = sorted(s["rates"]["prefill_tps"] for s in recent
                     if (s.get("rates") or {}).get("prefill_tps") is not None)
        return {
            "mtp_accept_rate": (acc / dr) if dr else None,
            "ttft_p50": self._pct(ttfts, 50),
            "ttft_p95": self._pct(ttfts, 95),
            "proc_p50": self._pct(procs, 50),
            "proc_p95": self._pct(procs, 95),
            "decode_tps_p50": self._pct(req_dec, 50),
            "decode_tps_p95": self._pct(req_dec, 95),
            "prefill_tps_p50": self._pct(pre, 50),
            "prefill_tps_p95": self._pct(pre, 95),
        }

    def tick(self) -> None:
        if self.jsonl_tail is not None:
            for line in self.jsonl_tail.read_lines():
                self.handle_jsonl_line(line)
        self.poll_stats()
        gpu = self.read_gpu()
        cpu = self.read_cpu()
        ram = self.read_ram()
        sensor = self.read_sensor()
        if sensor is not None and sensor > self.cfg.sensor_critical_c:
            # Stop the server once the reading has stayed above critical for --stop-after-s, then
            # again every tick until it cools, so the hardware cools before drawing power again.
            if self._sensor_over_since is None:
                self._sensor_over_since = time.time()
            elif (self.cfg.stop_pid_file and
                  time.time() - self._sensor_over_since >= self.cfg.stop_after_s):
                self._stop_server(sensor)
        elif sensor is not None:
            self._sensor_over_since = None
            self._last_stop_outcome = None
        stats = self.server_info.get("last_good_stats")
        slim = None
        kv_delta = {}
        if stats:
            slim = flatten_stats(stats, self.server_info.get("memory") or {})
            kt, pr = slim["kv_transfers"], slim["pressure"]
            cur = {
                "d2h_pages": kt["main_kv_d2h_pages"] + kt["backend_kv_d2h_pages"],
                "h2d_pages": kt["main_kv_h2d_pages"] + kt["backend_kv_h2d_pages"],
                "d2h_bytes": kt["main_kv_d2h_bytes"] + kt["backend_kv_d2h_bytes"],
                "h2d_bytes": kt["main_kv_h2d_bytes"] + kt["backend_kv_h2d_bytes"],
                "spill_pages": pr.get("spill_pages", 0),
                "owners_evicted": pr.get("private_owners_evicted", 0) +
                                  pr.get("shared_owners_evicted", 0),
                "owners_demoted": pr.get("private_owners_demoted", 0),
                "owners_degraded": pr.get("private_owners_degraded", 0) +
                                   pr.get("shared_owners_degraded", 0),
                "checkpoints_dropped": pr.get("checkpoints_dropped", 0),
                "maximal_fallbacks": pr.get("maximal_fallback_selections", 0),
                "searches": pr.get("searches", 0),
                "search_budget_exhaustions": pr.get("search_budget_exhaustions", 0),
            }
            if self._last_kv_counters is not None:
                for key, val in cur.items():
                    delta = val - self._last_kv_counters.get(key, 0)
                    if delta >= 0:
                        kv_delta[key] = delta
                        self._kv_totals[key] = self._kv_totals.get(key, 0) + delta
            self._last_kv_counters = cur
        sample = {
            "t": int(time.time() * 1000),
            "server_up": self.server_info["up"],
            "stats": slim,
            "rates": self.last_rates,
            "kv_events": kv_delta,
            "rolling": self._rolling(),
            "gpu": gpu,
            "cpu_pct": cpu,
            "ram": ram,
            "hpwr_c": sensor,
        }
        with self._lock:
            self.samples.append(sample)

    def start(self) -> None:
        t = threading.Thread(target=self._loop, name="sampler", daemon=True)
        t.start()

    def _loop(self) -> None:
        while True:
            try:
                self.tick()
            except Exception as error:  # a bad tick must never stop the sampler
                print(f"sampler tick failed: {error!r}", flush=True)
            time.sleep(self.cfg.interval)

    # -- aggregates ---------------------------------------------------------

    @staticmethod
    def _pct(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        k = (len(vals) - 1) * p / 100.0
        lo = int(k)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    def compute_aggregates(self) -> dict:
        now_ms = int(time.time() * 1000)
        window_ms = self.cfg.window_seconds * 1000
        evs = [e for e in self.events if now_ms - (e.get("t") or 0) <= window_ms]
        done = [e for e in evs if e["kind"] == "done"]
        errors = [e for e in evs if e["kind"] == "error"]
        rejected = [e for e in evs if e["kind"] == "rejected"]
        ttfts = sorted(e["ttft"] for e in done if e.get("ttft") is not None)
        mtp_acc = sum(e.get("mtp_accepted", 0) for e in done)
        mtp_dr = sum(e.get("mtp_drafted", 0) for e in done)
        mtp_rd = sum(e.get("mtp_rounds", 0) for e in done)
        prefix_hits = sum(1 for e in done if (e.get("prefix_hit") or 0) > 0)
        by_path: dict[str, int] = {}
        for e in done:
            p = e.get("prefix_path")
            if p:
                by_path[p] = by_path.get(p, 0) + 1
        smp = [s for s in self.samples if now_ms - s["t"] <= window_ms and s.get("rates")]
        dec = [s["rates"]["decode_tps"] for s in smp]
        pre = [s["rates"]["prefill_tps"] for s in smp]
        return {
            "window_seconds": self.cfg.window_seconds,
            "requests": {"done": len(done), "errors": len(errors), "rejected": len(rejected)},
            "ttft": {
                "p50": self._pct(ttfts, 50),
                "p95": self._pct(ttfts, 95),
                "max": ttfts[-1] if ttfts else None,
            },
            "mtp": {
                "accept_rate": (mtp_acc / mtp_dr) if mtp_dr else None,
                "tokens_per_round": (mtp_acc / mtp_rd) if mtp_rd else None,
                "requests_with_mtp": sum(1 for e in done if (e.get("mtp_drafted") or 0) > 0),
            },
            "prefix": {
                "hit_rate": (prefix_hits / len(done)) if done else None,
                "by_path": by_path,
            },
            "throughput": {
                "decode_tps_avg": (sum(dec) / len(dec)) if dec else None,
                "prefill_tps_avg": (sum(pre) / len(pre)) if pre else None,
            },
        }

    # -- HTTP payloads ------------------------------------------------------

    def api_samples(self) -> dict:
        with self._lock:
            samples = list(self.samples)
            events = list(self.events)
            kv_log = list(self.kv_log)
            spec_backend = self.spec_backend
        sensor = None
        if self.cfg.sensor_command:
            sensor = {"label": self.cfg.sensor_label, "warn_c": self.cfg.sensor_warn_c,
                      "critical_c": self.cfg.sensor_critical_c}
        return {
            "generated_unix_ms": int(time.time() * 1000),
            "server": {
                "up": self.server_info["up"],
                "last_stats_unix_ms": self.server_info.get("last_stats_unix_ms"),
                "instance_id": self.server_info.get("instance_id"),
                "model": self.server_info.get("model"),
                "spec_backend": spec_backend,
            },
            "sensor": sensor,
            "latest": samples[-1] if samples else None,
            "samples": samples,
            "events": events,
            "kv_log": kv_log,
            "aggregates": self.compute_aggregates(),
        }


# ---------------------------------------------------------------------------
# HTML dashboard (self-contained, no CDN)
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NInfer Monitor</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCIgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0Ij4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2NCwwKSBzY2FsZSgtMSwxKSI+CiAgICA8cmVjdCB3aWR0aD0iNjQiIGhlaWdodD0iNjQiIHJ4PSIxNCIgZmlsbD0iIzBGMTcyQSIvPgogICAgPCEtLSBMZWZ0IHBvc3QgLS0+CiAgICA8cmVjdCB4PSIxMiIgeT0iMTQiIHdpZHRoPSI4IiBoZWlnaHQ9IjM2IiByeD0iNCIgZmlsbD0iIzM4QkRGOCIvPgogICAgPCEtLSBSaWdodCBwb3N0IC0tPgogICAgPHJlY3QgeD0iNDQiIHk9IjE0IiB3aWR0aD0iOCIgaGVpZ2h0PSIzNiIgcng9IjQiIGZpbGw9IiMzOEJERjgiLz4KICAgIDwhLS0gQ2VudHJhbCBQdWxzZSAvIERpYWdvbmFsIC0tPgogICAgPHBvbHlsaW5lIHBvaW50cz0iMTYsMzIgMjYsNDQgMzgsMjAgNDgsMzIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzIyQzU1RSIgc3Ryb2tlLXdpZHRoPSI3IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KICA8L2c+Cjwvc3ZnPg==">
<style>
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2333; --border:#2b3444;
  --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --good:#3fb950;
  --warn:#d29922; --bad:#f85149; --grid:#21262d;
  --c1:#58a6ff; --c2:#3fb950; --c3:#d29922; --c4:#bc8cff; --c5:#f778ba;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:14px 20px;
  border-bottom:1px solid var(--border);background:var(--panel);
  position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0;font-weight:600}
.pill{padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600}
.pill.up{background:#12291c;color:var(--good);border:1px solid #238639}
.pill.stale{background:#2a230c;color:var(--warn);border:1px solid #9e6a03}
.pill.down{background:#2d1215;color:var(--bad);border:1px solid #b62324}
.meta{color:var(--muted);font-size:12px;margin-left:auto}
.banner{display:none;background:#2d1215;color:var(--bad);padding:8px 20px;
  font-size:13px;border-bottom:1px solid #b62324}
main{padding:16px 20px 40px;max-width:1400px;margin:0 auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px}
.tile .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:22px;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
.tile .s{color:var(--muted);font-size:11px;margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;margin-top:14px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.card h2{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.04em}
canvas{width:100%;height:180px;display:block;cursor:crosshair}
#tooltip{position:fixed;pointer-events:none;background:#161b22;border:1px solid #30363d;
  border-radius:6px;padding:4px 8px;font:11px/1.4 ui-monospace,monospace;
  color:var(--text);z-index:100;display:none;white-space:nowrap}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase}
td.msg{white-space:normal;max-width:420px;color:var(--muted)}
.kvlog{font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);
  max-height:260px;overflow:auto}
.kvlog div{padding:2px 0;border-bottom:1px solid var(--grid)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:6px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block}
.bar{display:flex;height:14px;border-radius:4px;overflow:hidden;margin:6px 0 2px;
  background:var(--panel2)}
.bar div{height:100%}
.barlabel{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.empty{color:var(--muted);font-size:12px;padding:20px;text-align:center}
</style>
</head>
<body>
<header>
  <h1>NInfer Monitor</h1>
  <span id="status" class="pill down">…</span>
  <span class="meta" id="meta"></span>
</header>
<div class="banner" id="banner">Inference server unreachable — showing last good data.</div>
<div id="tooltip"></div>
<main>
  <div class="tiles" id="tiles"></div>
  <div class="grid">
    <div class="card"><h2>Scheduler occupancy</h2>
      <canvas id="c-sched"></canvas><div class="legend" id="l-sched"></div></div>
    <div class="card"><h2>In-flight queries</h2>
      <canvas id="c-conc"></canvas><div class="legend" id="l-conc"></div></div>
    <div class="card"><h2>Admitted requests</h2>
      <canvas id="c-http"></canvas><div class="legend" id="l-http"></div></div>
    <div class="card"><h2>Decode throughput (tok/s)</h2>
      <canvas id="c-decode"></canvas><div class="legend" id="l-decode"></div></div>
    <div class="card"><h2>Prefill throughput (tok/s)</h2>
      <canvas id="c-prefill"></canvas><div class="legend" id="l-prefill"></div></div>
    <div class="card"><h2>Host KV usage</h2>
      <canvas id="c-kv"></canvas><div class="legend" id="l-kv"></div></div>
    <div class="card"><h2>Device KV occupancy</h2>
      <canvas id="c-devkv"></canvas><div class="legend" id="l-devkv"></div></div>
    <div class="card"><h2 id="h-mtp-acc">MTP acceptance</h2>
      <canvas id="c-mtp"></canvas><div class="legend" id="l-mtp"></div></div>
    <div class="card"><h2>TTFT (s)</h2>
      <canvas id="c-ttft"></canvas><div class="legend" id="l-ttft"></div></div>
    <div class="card"><h2>Processing time (proc = ttft - queue)</h2>
      <canvas id="c-proc"></canvas><div class="legend" id="l-proc"></div></div>
    <div class="card"><h2>GPU</h2>
      <canvas id="c-gpu"></canvas><div class="legend" id="l-gpu"></div></div>
    <div class="card"><h2>GPU power (W)</h2>
      <canvas id="c-gpupwr"></canvas><div class="legend" id="l-gpupwr"></div></div>
    <div class="card" id="card-sensor"><h2 id="h-sensor">Sensor (°C)</h2>
      <canvas id="c-hpwr"></canvas><div class="legend" id="l-hpwr"></div></div>
    <div class="card"><h2>CPU &amp; RAM</h2>
      <canvas id="c-cpura"></canvas><div class="legend" id="l-cpura"></div></div>
    <div class="card"><h2>KV Cache</h2><div id="kvbars"></div></div>
    <div class="card"><h2>Transfers (pages/interval)</h2>
      <canvas id="c-kvev"></canvas><div class="legend" id="l-kvev"></div>
      <div id="xferstats" style="margin-top:8px"></div></div>
    <div class="card"><h2>Pressure events (per interval)</h2>
      <canvas id="c-press"></canvas><div class="legend" id="l-press"></div></div>
    <div class="card"><h2>Pressure &amp; cache reuse</h2><div id="pressure"></div></div>
  </div>
  <div class="grid">
    <div class="card"><h2>Recent requests</h2>
      <div style="overflow:auto;max-height:260px"><table id="reqs"></table></div></div>
    <div class="card"><h2 id="h-mtp-log">MTP log</h2><div class="kvlog" id="kvlog"></div></div>
  </div>
</main>
<script>
const $=id=>document.getElementById(id);
const C={sched:['#58a6ff','#3fb950','#d29922','#bc8cff','#f778ba','#f85149','#3fb950'],
         concurrent:['#58a6ff'],
         decode:['#58a6ff','#d29922'],prefill:['#3fb950','#d29922'],
         kv:['#58a6ff','#3fb950','#d29922'],
         kvev:['#f85149','#d29922','#58a6ff','#3fb950','#bc8cff'],
         mtp:['#3fb950','#bc8cff'],ttft:['#58a6ff','#d29922'],proc:['#bc8cff','#f778ba'],
         gpu:['#58a6ff','#3fb950'],cpura:['#58a6ff','#d29922'],
         hpwr:['#f0883e']};
const chartMeta={};
function decimate(pts,max){if(pts.length<=max)return pts;
  const step=Math.ceil(pts.length/max),out=[];
  for(let i=0;i<pts.length;i+=step)out.push(pts[i]);
  if(out[out.length-1]!==pts[pts.length-1])out.push(pts[pts.length-1]);
  return out;}
function lineChart(cv,series,opt={}){
  const dpr=window.devicePixelRatio||1;
  const w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
  const pad={l:42,r:10,t:8,b:18};
  const all=[];series.forEach(s=>s.points.forEach(p=>all.push(p)));
  if(!all.length){ctx.fillStyle='#8b949e';ctx.font='12px sans-serif';
    ctx.textAlign='center';ctx.fillText('no data',w/2,h/2);return;}
  let tmin=Math.min(...all.map(p=>p[0])),tmax=Math.max(...all.map(p=>p[0]));
  if(tmax===tmin)tmax=tmin+1;
  let vmin=0,vmax=0;
  series.forEach(s=>s.points.forEach(p=>{vmin=Math.min(vmin,p[1]);vmax=Math.max(vmax,p[1]);}));
  if(opt.vmax!=null)vmax=opt.vmax;
  if(opt.vmin!=null)vmin=Math.max(vmin,opt.vmin);
  if(vmin<0)vmin=0;
  if(vmax<=vmin)vmax=vmin+1;
  if(opt.integers){const step=Math.max(1,Math.ceil((vmax-vmin)/4));vmax=vmin+step*4;}
  const X=t=>pad.l+(t-tmin)/(tmax-tmin)*(w-pad.l-pad.r);
  const Y=v=>h-pad.b-(v-vmin)/(vmax-vmin)*(h-pad.t-pad.b);
  ctx.strokeStyle='#21262d';ctx.fillStyle='#8b949e';ctx.font='10px sans-serif';
  ctx.lineWidth=1;ctx.textAlign='right';
  for(let g=0;g<=4;g++){const v=vmin+(vmax-vmin)*g/4,y=Y(v);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillText(opt.integers?String(Math.round(v)):fmtNum(v),pad.l-5,y+3);}
  ctx.textAlign='center';
  for(let g=0;g<=3;g++){const t=tmin+(tmax-tmin)*g/3;
    ctx.fillText(fmtT(t),X(t),h-4);}
  series.forEach((s,si)=>{
    const pts=decimate(s.points,720);
    ctx.strokeStyle=s.color;ctx.lineWidth=1.6;ctx.beginPath();
    pts.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]);
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
    ctx.stroke();});
  (opt.thresholds||[]).forEach(th=>{
    const y=Y(th.v);
    ctx.strokeStyle=th.color;ctx.lineWidth=1;ctx.setLineDash([5,4]);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=th.color;ctx.font='10px sans-serif';ctx.textAlign='left';
    ctx.fillText(th.label||String(th.v),pad.l+4,y-4);
  });
  // Register chart for hover tooltips
  if(cv.id){
    chartMeta[cv.id]={series:series,pad:pad,w:w,h:h,tmin:tmin,tmax:tmax};
  }
}
function fmtNum(v){if(v==null)return'–';
  if(v>=1000)return(v/1000).toFixed(1)+'k';
  if(v>=100)return v.toFixed(0);
  if(v>=10)return v.toFixed(1);
  return v.toFixed(2);}
function fmtT(ms){const d=new Date(ms);
  return d.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'});}
function fmtAgo(ms){const s=Math.max(0,Math.round((Date.now()-ms)/1000));
  if(s<60)return s+'s ago';if(s<3600)return Math.round(s/60)+'m ago';
  return Math.round(s/3600)+'h ago';}
function seriesFrom(samples,fn,color,label){
  const pts=[];samples.forEach(s=>{const v=fn(s);if(v!=null)pts.push([s.t,v]);});
  return{label,color,points:pts};}
function setLegend(id,series){$(id).innerHTML=series.map(s=>
  '<span><i class="dot" style="background:'+s.color+'"></i>'+s.label+'</span>').join('');}
function draw(cvId,legendId,series,opt){lineChart($(cvId),series,opt||{});
  setLegend(legendId,series);}
function tile(k,v,s,color){return '<div class="tile"><div class="k">'+k+'</div>'+
  '<div class="v"'+(color?' style="color:'+color+'"':'')+'>'+(v==null?'–':v)+'</div>'+(s?'<div class="s">'+s+'</div>':'')+'</div>';}
function bar(segments){ // segments: [{pct,color,label}]
  let h='<div class="bar">';
  segments.forEach(sg=>{h+='<div style="width:'+sg.pct+'%;background:'+sg.color+'"></div>';});
  h+='</div><div class="barlabel">';
  h+=segments.map(sg=>sg.label+' '+(sg.pct==null?'–':sg.pct.toFixed(0)+'%')).join(' &nbsp; ');
  h+='</div>';return h;}
// Chart hover system
function setupHover(){
  document.querySelectorAll('canvas').forEach(cv=>{
    if(cv.id&&chartMeta[cv.id]){
      cv.onmousemove=function(ev){
        const m=chartMeta[cv.id];
        if(!m||!m.series||!m.series.length){return;}
        const rect=cv.getBoundingClientRect();
        const x=ev.clientX-rect.left;
        // Find nearest sample
        const plotW=m.w-m.pad.l-m.pad.r;
        const tFrac=(x-m.pad.l)/(m.w-m.pad.l-m.pad.r);
        const t=m.tmin+tFrac*(m.tmax-m.tmin);
        const lines=[fmtT(t)];
        for(const ser of m.series){
          if(!ser.points||!ser.points.length)continue;
          // Find nearest point by time
          let best=null,bestDist=Infinity;
          for(const p of ser.points){
            const d=Math.abs(p[0]-t);
            if(d<bestDist){bestDist=d;best=p;}
          }
          if(best!=null){
            lines.push(ser.label+': '+best[1].toFixed(1));
          }
        }
        const tt=$('tooltip');
        tt.style.display='block';
        tt.textContent=lines.join(' | ');
        const tw=tt.offsetWidth;
        let lx=ev.clientX+12;
        if(lx+tw>window.innerWidth-8)lx=ev.clientX-tw-12;
        if(lx<8)lx=8;
        tt.style.left=lx+'px';
        tt.style.top=Math.max(ev.clientY-40,0)+'px';
        tt.textContent=lines.join(' | ');
      };
      cv.onmouseleave=function(){$('tooltip').style.display='none';};
    }
  });
}

function render(d){
  const srv=d.server,latest=d.latest||{},agg=d.aggregates;
  // status pill
  const st=$('status');
  if(srv.up){st.className='pill up';st.textContent='LIVE';}
  else if(srv.last_stats_unix_ms){st.className='pill stale';st.textContent='STALE';}
  else{st.className='pill down';st.textContent='DOWN';}
  $('banner').style.display=srv.up?'none':'block';
  $('meta').textContent=(srv.model||'?')+' · '+(srv.instance_id||'no instance')+
    ' · updated '+fmtAgo(d.generated_unix_ms);
  // tiles
  const rates=latest.rates||{};
  const gpu=latest.gpu||{};const ram=latest.ram||{};
  const sensor=d.sensor,hpwr=latest.hpwr_c;
  let hpwrSub='no data',hpwrColor=null;
  if(sensor&&hpwr!=null){
    if(hpwr>sensor.critical_c){hpwrSub='critical';hpwrColor='#f85149';}
    else if(hpwr>sensor.warn_c){hpwrSub='warm';hpwrColor='#d29922';}
    else hpwrSub='nominal';
  }
  $('card-sensor').style.display=sensor?'block':'none';
  if(sensor)$('h-sensor').textContent=sensor.label+' (°C)';
  const specName=(srv.spec_backend||'MTP');
  $('h-mtp-acc').textContent=specName+' acceptance';
  $('h-mtp-log').textContent=specName+' log';
  const tiles=[
    tile('Decode p50',latest.rolling&&latest.rolling.decode_tps_p50!=null?fmt0(latest.rolling.decode_tps_p50)+' tok/s':'–','p95 '+(latest.rolling&&latest.rolling.decode_tps_p95!=null?fmt0(latest.rolling.decode_tps_p95)+' tok/s':'–')),
    tile('Prefill p50',latest.rolling&&latest.rolling.prefill_tps_p50!=null?fmt0(latest.rolling.prefill_tps_p50)+' tok/s':'–','p95 '+(latest.rolling&&latest.rolling.prefill_tps_p95!=null?fmt0(latest.rolling.prefill_tps_p95)+' tok/s':'–')),
    tile(specName+' accept',pct(agg.mtp.accept_rate),'tok/round '+(agg.mtp.tokens_per_round?agg.mtp.tokens_per_round.toFixed(2):'–')),
    tile('Prefix hit',pct(agg.prefix.hit_rate),Object.keys(agg.prefix.by_path||{}).length+' paths'),
    tile('TTFT p50',agg.ttft.p50?agg.ttft.p50.toFixed(2)+'s':'–','p95 '+(agg.ttft.p95?agg.ttft.p95.toFixed(1)+'s':'–')),
    tile('GPU',gpu.util_pct!=null?gpu.util_pct.toFixed(0)+'%':'–',
      gpu.mem_used_mb?(gpu.mem_used_mb/1024).toFixed(1)+' / '+(gpu.mem_total_mb/1024).toFixed(0)+' GB':''),
    tile('GPU pwr',gpu.power_w!=null?gpu.power_w.toFixed(0)+' W':'–','of '+(gpu.power_limit_w!=null?gpu.power_limit_w.toFixed(0):'?')+' W'),
    ...(sensor?[tile(sensor.label,hpwr!=null?hpwr.toFixed(1)+'°C':'–',hpwrSub,hpwrColor)]:[]),
    tile('CPU',latest.cpu_pct!=null?latest.cpu_pct.toFixed(0)+'%':'–',ram.used_mb?(ram.used_mb/1024).toFixed(1)+' / '+(ram.total_mb/1024).toFixed(0)+' GB':''),
    tile('Admitted',(latest.stats?.http?.in_flight!=null?latest.stats.http.in_flight:'–')+' / '+(latest.stats?.http?.max_in_flight!=null?latest.stats.http.max_in_flight:'–'),'now / peak'),
  ];
  $('tiles').innerHTML=tiles.join('');
  // charts
  const S=d.samples;
  draw('c-sched','l-sched',[
    seriesFrom(S,s=>s.stats?.scheduler?.running,C.sched[0],'running'),
    seriesFrom(S,s=>s.stats?.scheduler?.prefilling,C.sched[1],'prefilling'),
    seriesFrom(S,s=>s.stats?.scheduler?.decode_ready,C.sched[2],'decode_ready'),
    seriesFrom(S,s=>s.stats?.scheduler?.waiting,C.sched[3],'waiting'),
    seriesFrom(S,s=>s.stats?.scheduler?.materializing,C.sched[4],'materializing'),
    seriesFrom(S,s=>s.stats?.scheduler?.capture_pending,C.sched[5],'capture_pending'),
    seriesFrom(S,s=>s.stats?.scheduler?.terminal_pending,C.sched[6],'terminal_pending'),
  ],{vmax:8});
  draw('c-conc','l-conc',[
    seriesFrom(S,s=>{const sc=s.stats?.scheduler;
      return sc?(sc.running||0)+(sc.prefilling||0):null;},C.concurrent[0],'in flight'),
  ],{integers:true});
  draw('c-decode','l-decode',[
    seriesFrom(S,s=>s.rolling?.decode_tps_p50!=null?s.rolling.decode_tps_p50:null,C.decode[0],'p50'),
    seriesFrom(S,s=>s.rolling?.decode_tps_p95!=null?s.rolling.decode_tps_p95:null,C.decode[1],'p95'),
  ]);
  draw('c-prefill','l-prefill',[
    seriesFrom(S,s=>s.rolling?.prefill_tps_p50>1?s.rolling.prefill_tps_p50:null,C.prefill[0],'p50'),
    seriesFrom(S,s=>s.rolling?.prefill_tps_p95>1?s.rolling.prefill_tps_p95:null,C.prefill[1],'p95'),
  ]);
  draw('c-kvev','l-kvev',[
    seriesFrom(S,s=>s.kv_events?.d2h_pages,C.kvev[0],'d2h (park)'),
    seriesFrom(S,s=>s.kv_events?.h2d_pages,C.kvev[3],'h2d (restore)'),
  ],{integers:true});
  // Device KV occupancy time-series
  draw('c-devkv','l-devkv',[
    seriesFrom(S,s=>{
      const m=s.stats?.memory,p=s.stats?.pressure;
      if(!m||!p)return null;
      const cap=m.kv_capacity_page_groups;
      if(!cap)return null;
      return(p.device_main_kv_occupied_pages||0)/cap*100;
    },C.kv[0],'device KV %'),
  ],{vmax:100});
  // Requests holding ingress capacity
  draw('c-http','l-http',[
    seriesFrom(S,s=>s.stats?.http?.in_flight,C.concurrent[0],'admitted'),
  ],{integers:true});
  // Pressure events per interval
  draw('c-press','l-press',[
    seriesFrom(S,s=>s.kv_events?.spill_pages||0,C.kvev[0],'spill'),
    seriesFrom(S,s=>s.kv_events?.owners_evicted||0,C.kvev[1],'evict'),
    seriesFrom(S,s=>s.kv_events?.owners_demoted||0,C.kvev[4],'demote'),
    seriesFrom(S,s=>s.kv_events?.owners_degraded||0,C.kvev[2],'degrade'),
    seriesFrom(S,s=>s.kv_events?.maximal_fallbacks||0,C.kvev[3],'fallback'),
  ],{integers:true});
  const pr=latest.stats?.pressure||{},cr=latest.stats?.cache_reuse||{};
  const sp=pr.spill_pages||0,ev=(pr.private_owners_evicted||0)+(pr.shared_owners_evicted||0);
  const dg=(pr.private_owners_degraded||0)+(pr.shared_owners_degraded||0);
  const fb=pr.maximal_fallback_selections||0,cd=pr.checkpoints_dropped||0;
  const selKeys=Object.keys(cr).filter(k=>k.endsWith('_selections'));
  const crTotal=selKeys.reduce((a,k)=>a+(cr[k]||0),0);
  const rpt=cr.reused_prompt_tokens||0;
  const crBar=crTotal?selKeys.map(k=>{
    const v=cr[k]||0;const p=v/crTotal*100;const c=k.includes('turn_closure')?'#3fb950':k.includes('root')||k.includes('full')?'#f85149':'#58a6ff';
    return bar([{pct:p,color:c,label:k.replace(/_selections/g,'').replace(/_/g,' ')}]).replace('<div class="bar">','<div class="bar" style="margin:1px 0">');
  }).join(''):'<div class="empty">no cache reuse data</div>';
  const sbe=pr.search_budget_exhaustions||0,srch=pr.searches||0;
  $('pressure').innerHTML=
    '<div style="font-size:12px;color:var(--muted);margin-bottom:4px">Pressure (cumulative)</div>'+
    '<div style="font-size:13px;margin-bottom:4px">'+
    '<span style="color:#f85149">spill:'+sp+'</span> · '+
    '<span style="color:#f85149">evict:'+ev+'</span> · '+
    '<span style="color:#d29922">degrade:'+dg+'</span> · '+
    '<span style="color:#f85149">fallback:'+fb+'</span> · '+
    '<span style="color:#d29922">ckpt_drop:'+cd+'</span></div>'+
    '<div style="font-size:13px;margin-bottom:8px">'+
    '<span style="color:#d29922">searches:'+srch+'</span> · '+
    '<span style="color:#f85149">budget_exhaust:'+sbe+'</span></div>'+
    '<div style="font-size:12px;color:var(--muted);margin-bottom:4px">Cache reuse paths</div>'+
    crBar+
    '<div style="font-size:12px;color:var(--muted);margin-top:6px">reused prompt tokens: '+rpt.toLocaleString()+'</div>';
  draw('c-kv','l-kv',[
    seriesFrom(S,s=>{const m=s.stats?.memory;return m&&m.host_kv_capacity_bytes?m.host_kv_occupied_bytes/m.host_kv_capacity_bytes*100:null;},C.kv[2],'host KV used %'),
  ]);
  draw('c-mtp','l-mtp',[
    seriesFrom(S,s=>s.rolling?.mtp_accept_rate!=null?s.rolling.mtp_accept_rate*100:null,C.mtp[0],'accept %'),
  ],{vmax:100});
  draw('c-ttft','l-ttft',[
    seriesFrom(S,s=>s.rolling?.ttft_p50,C.ttft[0],'p50'),
    seriesFrom(S,s=>s.rolling?.ttft_p95,C.ttft[1],'p95'),
  ]);
  draw('c-proc','l-proc',[
    seriesFrom(S,s=>s.rolling?.proc_p50,C.proc[0],'p50'),
    seriesFrom(S,s=>s.rolling?.proc_p95,C.proc[1],'p95'),
  ]);
  draw('c-gpu','l-gpu',[
    seriesFrom(S,s=>s.gpu?.util_pct,C.gpu[0],'GPU util %'),
    seriesFrom(S,s=>s.gpu?.mem_total_mb?s.gpu.mem_used_mb/s.gpu.mem_total_mb*100:null,C.gpu[1],'GPU mem %'),
  ],{vmax:100});
  draw('c-gpupwr','l-gpupwr',[
    seriesFrom(S,s=>s.gpu?.power_w,C.gpu[0],'power W'),
  ],{vmax:gpu.power_limit_w||450});
  if(sensor){
    const hpwrVals=S.map(s=>s.hpwr_c).filter(v=>v!=null);
    draw('c-hpwr','l-hpwr',[
      seriesFrom(S,s=>s.hpwr_c,C.hpwr[0],sensor.label+' °C'),
    ],{vmin:0,vmax:Math.max(sensor.critical_c+6,hpwrVals.length?Math.max(...hpwrVals)+4:0),
       thresholds:[{v:sensor.warn_c,color:'#d29922',label:'warn '+sensor.warn_c+'°'},
                   {v:sensor.critical_c,color:'#f85149',label:'crit '+sensor.critical_c+'°'}]});
  }
  draw('c-cpura','l-cpura',[
    seriesFrom(S,s=>s.cpu_pct,C.cpura[0],'CPU %'),
    seriesFrom(S,s=>s.ram?.total_mb?s.ram.used_mb/s.ram.total_mb*100:null,C.cpura[1],'RAM %'),
  ],{vmax:100});
  // KV bars
  renderKvBars(latest);
  // Transfer stats (KV + state combined)
  renderXferStats(latest);
  // recent requests
  renderReqs(d.events);
  // speculative decoding summaries of recent requests
  $('kvlog').innerHTML=d.kv_log.slice(-20).reverse().map(e=>
    '<div>'+esc(e.line)+'</div>').join('')||'<div class="empty">no speculative decoding yet</div>';
}
function fmt0(v){return v==null?'–':(v>=100?v.toFixed(0):v.toFixed(1));}
function pct(v){return v==null?'–':(v*100).toFixed(0)+'%';}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function timeCell(ms){return ms?'<td title="'+fmtAgo(ms)+'">'+fmtT(ms)+'</td>':'<td>–</td>';}
function renderKvBars(latest){
  const mem=latest.stats?.memory||{};
  const gb=v=>(v/1073741824).toFixed(1)+' GB';
  const pstats=latest.stats?.pressure||{};
  const pg=pstats.device_main_kv_occupied_pages||0;const bpg=pstats.device_backend_kv_occupied_pages||0;
  const mpg=mem.kv_capacity_page_groups||0;
  const pgPct=mpg?pg/mpg*100:0,bpgPct=mpg?bpg/mpg*100:0;
  const budget=mem.host_kv_capacity_bytes||0,used=mem.host_kv_occupied_bytes||0;
  const usedPct=budget?used/budget*100:0;
  const slots=mem.host_state_occupied_slots||0,maxSlots=mem.host_state_capacity_slots||0;
  const devSlots=pstats.device_state_occupied_slots||0;
  const evict=(pstats.private_owners_evicted||0)+(pstats.shared_owners_evicted||0);
  const demoted=pstats.private_owners_demoted||0;
  const spill=pstats.spill_pages||0,ckptDrop=pstats.checkpoints_dropped||0;
  let h='<div style="font-size:12px;color:var(--muted);margin-bottom:4px">device KV: '+pg+' / '+mpg+' pages ('+pgPct.toFixed(0)+'%)'+(mem.kv_payload_bytes?' · payload: '+gb(mem.kv_payload_bytes):'')+'</div>';
  h+=bar([{pct:pgPct,color:C.kv[0],label:'used'},{pct:100-pgPct,color:'#30363d',label:'free'}]);
  h+='<div style="font-size:12px;color:var(--muted);margin:8px 0 4px">host KV: '+gb(used)+' / '+gb(budget)+' ('+usedPct.toFixed(0)+'%) · state slots '+slots+'/'+maxSlots+' · device state '+devSlots+'</div>';
  if(budget>0){
    h+=bar([{pct:usedPct,color:C.kv[2],label:'used'},{pct:100-usedPct,color:'#30363d',label:'free'}]);
  } else {
    h+='<div class="empty">host KV cache disabled</div>';
  }
  h+='<div style="font-size:12px;color:var(--muted);margin:6px 0 2px">pressure (cum): evict '+evict+' · demote '+demoted+' · spill '+spill+' · ckpt_drop '+ckptDrop+'</div>';
  $('kvbars').innerHTML=h;
}
function renderXferStats(latest){
  const kt=latest.stats?.kv_transfers||{},st=latest.stats?.state_transfers||{};
  const gb=v=>(v/1073741824).toFixed(2)+' GB';
  const mb=v=>(v/1048576).toFixed(1)+' MB';
  $('xferstats').innerHTML=
    '<div style="font-size:12px;color:var(--muted);margin-bottom:2px">KV bytes (cum)</div>'+
    '<div style="font-size:12px;margin-bottom:4px">'+
    '<span style="color:#f85149">D2H '+gb(kt.main_kv_d2h_bytes||0)+'</span> · '+
    '<span style="color:#3fb950">H2D '+gb(kt.main_kv_h2d_bytes||0)+'</span> · '+
    '<span style="color:var(--muted)">'+(kt.main_kv_d2h_seconds||0).toFixed(1)+'s/'+(kt.main_kv_h2d_seconds||0).toFixed(1)+'s</span></div>'+
    '<div style="font-size:12px;color:var(--muted);margin-bottom:2px">State transfers (count)</div>'+
    '<div style="font-size:12px">'+
    '<span style="color:#f85149">D2H '+(st.state_d2h_count||0)+'</span> · '+
    '<span style="color:#3fb950">H2D '+(st.state_h2d_count||0)+'</span> · '+
    '<span style="color:var(--muted)">'+gb(st.state_d2h_bytes||0)+'/'+gb(st.state_h2d_bytes||0)+'</span> · '+
    '<span style="color:var(--muted)">'+(st.state_d2h_seconds||0).toFixed(1)+'s/'+(st.state_h2d_seconds||0).toFixed(1)+'s</span></div>';
}
function renderReqs(events){
  const rows=events.slice(-20).reverse();
  let h='<tr><th>time</th><th>id</th><th>proto</th><th>kind</th><th>prompt</th><th>gen</th><th>ttft</th><th>proc</th><th>prefix</th><th>reuse</th><th>finish</th><th></th></tr>';
  rows.forEach(e=>{
    if(e.kind==='done'){
      const pp=e.prefix_path||'';
      const prefixHit=(e.prefix_hit||0)>0;
      const green=pp&&(pp.includes('turn_closure')||pp.includes('append')||pp.includes('shared')||prefixHit);
      const hkCell=pp
        ?'<span style="color:'+(green?'#3fb950':'#f85149')+'" title="prefix_hit='+e.prefix_hit+'">'+esc(pp)+'</span>'
        :'<span style="color:#8b949e">–</span>';
      h+='<tr>'+timeCell(e.t)+'<td>'+e.id+'</td><td>'+esc(e.protocol||'')+'</td><td>done</td>'+
        '<td>'+e.prompt_tokens+'</td><td>'+e.completion_tokens+'</td>'+
        '<td>'+(e.ttft!=null?e.ttft.toFixed(1)+'s':'')+'</td>'+
        '<td>'+(e.proc!=null?e.proc.toFixed(1)+'s':'')+'</td>'+
        '<td>'+(e.prefix_hit||0)+'</td>'+
        '<td>'+hkCell+'</td>'+
        '<td>'+esc(e.finish_reason||'')+'</td><td class="msg"></td></tr>';
    }else{
      h+='<tr>'+timeCell(e.t)+'<td>'+(e.id??'')+'</td><td>'+esc(e.protocol||'')+'</td><td>'+e.kind+'</td>'+
        '<td colspan="7"></td><td class="msg">'+esc(e.message||'')+'</td></tr>';
    }
  });
  $('reqs').innerHTML=h;
}
// per-sample derived series: KV occupancy comes from the embedded /stats
// snapshot; MTP/TTFT come from the rolling values computed in the sampler.
async function refresh(){
  try{
    const r=await fetch('/api/samples');
    const d=await r.json();
    render(d);
    setupHover();
  }catch(e){console.error(e);}
}
refresh();setInterval(refresh,5000);
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    server_version = "ninfer-monitor/1.0"

    def log_message(self, format: str, *args) -> None:
        # The dashboard polls every 5 s; per-request access lines would bury the sidecar log.
        del format, args

    def do_GET(self) -> None:  # noqa: N802
        monitor: Monitor = self.server.monitor  # type: ignore[attr-defined]
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/samples":
            self._send(json.dumps(monitor.api_samples()).encode("utf-8"), "application/json")
        elif path == "/healthz":
            self._send(b"ok", "text/plain")
        else:
            self._send(b"not found", "text/plain", 404)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NInfer monitoring sidecar")
    p.add_argument("--port", type=int, default=8090, help="dashboard port")
    p.add_argument("--bind", default="127.0.0.1", help="dashboard address")
    p.add_argument("--stats-url", default="http://127.0.0.1:8080",
                   help="base URL of the server's /stats, e.g. its --stats-port listener")
    p.add_argument("--api-key-env", default="NINFER_API_KEY",
                   help="environment variable holding the server's --api-key, if it has one")
    p.add_argument("--jsonl", default=None,
                   help="the server's --request-log-jsonl file; without it there is no request "
                        "table, TTFT or acceptance")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    p.add_argument("--samples", type=int, default=4320, help="ring-buffer length (6 h at 5 s)")
    p.add_argument("--sensor-command", default=None,
                   help="command that prints one temperature in Celsius, sampled every tick")
    p.add_argument("--sensor-label", default="Sensor", help="dashboard name of that reading")
    p.add_argument("--sensor-warn-c", type=float, default=52.0)
    p.add_argument("--sensor-critical-c", type=float, default=62.0)
    p.add_argument("--stop-server-pid-file", default=None,
                   help="file holding the server's PID: SIGKILL it once the sensor stays above "
                        "critical for --stop-after-s, and every tick after until it cools")
    p.add_argument("--stop-after-s", type=float, default=60.0)
    p.add_argument("--pidfile", default=None, help="write the sidecar's own PID here")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(args)
    monitor = Monitor(cfg)
    monitor.bootstrap_server_info()
    monitor.start()
    if cfg.pidfile:
        with open(cfg.pidfile, "w") as f:
            f.write(str(os.getpid()))
    httpd = ThreadingHTTPServer((cfg.bind, cfg.port), Handler)
    httpd.monitor = monitor  # type: ignore[attr-defined]
    print(f"ninfer-monitor listening on {cfg.bind}:{cfg.port} "
          f"(stats={cfg.stats_url}, jsonl={cfg.jsonl})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.pidfile and os.path.exists(cfg.pidfile):
            os.remove(cfg.pidfile)


if __name__ == "__main__":
    main()
