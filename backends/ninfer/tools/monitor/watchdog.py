#!/usr/bin/env python3
"""Restarts a wedged NInfer server: work outstanding and no engine progress.

Polls GET /stats every --interval seconds. The server is wedged when its scheduler reports work
(running, prefilling, decode-ready, waiting, materializing, capture-pending or terminal-pending
requests) and none of its cumulative engine counters -- tokens, decode rounds, prefill and control
units, Host work, context transfers and pressure searches -- has advanced for --stall-seconds. An
unreachable server with work outstanding at the last poll counts as no progress: a frozen process
can stop answering entirely. The prefill counter advances per prefill chunk, so a long prefill is
progress, not a stall.

On a wedge it runs --restart-command once, then gives the server --grace-seconds to come back. A
server that restarted by itself (its uptime went down) or is still loading (503) resets the
detector and gets the same grace. After --max-restarts restarts within --restart-window-seconds it
stops restarting and only reports, since a wedge that keeps coming back needs a person.

Stdlib only; one status line per poll on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OUTSTANDING = ("running", "prefilling", "decode_ready", "waiting", "materializing",
               "capture_pending", "terminal_pending")


@dataclass
class Poll:
    """One /stats poll: `loading` for a 503, `stats` on success, neither when unreachable."""
    stats: dict | None = None
    loading: bool = False
    error: str = ""


def poll(url: str, api_key: str | None, timeout: float) -> Poll:
    request = urllib.request.Request(url.rstrip("/") + "/stats")
    if api_key:
        request.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Poll(stats=json.loads(response.read().decode("utf-8", "replace")))
    except urllib.error.HTTPError as error:
        return Poll(loading=error.code == 503, error=f"HTTP {error.code}")
    except (OSError, ValueError) as error:
        return Poll(error=type(error).__name__)


def outstanding_work(stats: dict) -> int:
    scheduler = stats.get("scheduler", {})
    return sum(int(scheduler.get(name, 0) or 0) for name in OUTSTANDING)


def progress_fingerprint(stats: dict) -> tuple:
    """Every cumulative counter that advances while the engine works."""
    tokens = stats.get("tokens", {})
    decode = stats.get("decode_batch", {})
    host = stats.get("host_work", {})
    cache = stats.get("context_cache", {})

    def moved(block: str) -> tuple:
        legs = cache.get(block, {})
        return tuple(legs.get(direction, {}).get("bytes", 0) for direction in ("d2h", "h2d", "d2d"))

    return (
        tokens.get("computed_prefill", 0),
        tokens.get("committed_decode", 0),
        decode.get("rounds", 0),
        host.get("units", {}).get("prefill", 0),
        host.get("units", {}).get("control", 0),
        host.get("elapsed_seconds", {}).get("total", 0.0),
        host.get("device_wait_seconds", 0.0),
        moved("main_kv_transfers"),
        moved("backend_kv_transfers"),
        moved("state_transfers"),
        cache.get("pressure", {}).get("searches", 0),
        cache.get("captures", {}).get("completed", 0),
        cache.get("captures", {}).get("aborted", 0),
    )


@dataclass
class Detector:
    stall_seconds: float
    grace_seconds: float
    max_restarts: int
    window_seconds: float
    fingerprint: tuple | None = None
    last_progress: float | None = None
    work_outstanding: bool = False
    last_uptime: float | None = None
    grace_until: float = 0.0
    restarts: list[float] = field(default_factory=list)
    stopped: bool = False

    def reset(self, now: float) -> None:
        self.fingerprint = None
        self.last_progress = None
        self.work_outstanding = False
        self.grace_until = now + self.grace_seconds

    def observe(self, result: Poll, now: float) -> str:
        """Updates the state from one poll and returns what it saw."""
        if result.loading:
            self.reset(now)
            self.last_uptime = None
            return "loading"
        if result.stats is None:
            # Silence keeps the last known work and lets its stall timer run.
            return f"unreachable ({result.error})"
        stats = result.stats
        uptime = float(stats.get("uptime_seconds", 0.0) or 0.0)
        if self.last_uptime is not None and uptime < self.last_uptime:
            self.reset(now)
        self.last_uptime = uptime
        work = outstanding_work(stats)
        self.work_outstanding = work > 0
        fingerprint = progress_fingerprint(stats)
        if fingerprint != self.fingerprint or not self.work_outstanding:
            self.fingerprint = fingerprint
            self.last_progress = now
        return f"work={work}"

    def stalled_for(self, now: float) -> float:
        if not self.work_outstanding or self.last_progress is None:
            return 0.0
        return now - self.last_progress

    def restart_due(self, now: float) -> bool:
        return (not self.stopped and now >= self.grace_until and
                self.stalled_for(now) >= self.stall_seconds)

    def record_restart(self, now: float) -> bool:
        """Counts a restart; False once the budget of the window is spent."""
        self.restarts = [t for t in self.restarts if now - t < self.window_seconds]
        if len(self.restarts) >= self.max_restarts:
            self.stopped = True
            return False
        self.restarts.append(now)
        self.reset(now)
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stats-url", default="http://127.0.0.1:8080",
                        help="base URL of the server's /stats, e.g. its --stats-port listener")
    parser.add_argument("--api-key-env", default="NINFER_API_KEY",
                        help="environment variable holding the server's --api-key, if it has one")
    parser.add_argument("--restart-command", required=True,
                        help="command that restarts the server, e.g. 'systemctl restart ninfer'")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--stall-seconds", type=float, default=150.0,
                        help="work outstanding with no engine progress this long is a wedge")
    parser.add_argument("--grace-seconds", type=float, default=90.0,
                        help="no restart this long after a start, a restart or a load")
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--restart-window-seconds", type=float, default=1800.0)
    parser.add_argument("--timeout", type=float, default=5.0, help="per-poll HTTP timeout")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    restart = shlex.split(args.restart_command)
    detector = Detector(args.stall_seconds, args.grace_seconds, args.max_restarts,
                        args.restart_window_seconds)
    detector.reset(time.monotonic())
    while True:
        now = time.monotonic()
        seen = detector.observe(poll(args.stats_url, api_key, args.timeout), now)
        stalled = detector.stalled_for(now)
        state = "stopped" if detector.stopped else (
            "grace" if now < detector.grace_until else "watching")
        print(f"{time.strftime('%H:%M:%S')} {seen} stalled={stalled:.0f}s {state} "
              f"restarts={len(detector.restarts)}", flush=True)
        if detector.restart_due(now):
            if detector.record_restart(now):
                print(f"WEDGE: work outstanding and no engine progress for {stalled:.0f}s; "
                      f"running the restart command", flush=True)
                result = subprocess.run(restart, check=False)
                print(f"restart command exited {result.returncode}", flush=True)
            else:
                print(f"WEDGE REPEAT: {args.max_restarts} restarts within "
                      f"{args.restart_window_seconds:.0f}s; restarts stopped, needs a person",
                      flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
