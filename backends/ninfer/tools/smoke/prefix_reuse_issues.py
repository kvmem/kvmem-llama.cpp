"""Run the real-model prefix-caching scenarios against a .ninfer artifact.

The scenarios live in ninfer_qwen3_5_prefix_real_test (NINFER_PREFIX_REAL_SCENARIO selects one);
this runner executes them one process each, plus the host-only resource-manager suite, and
prints one summary. The GPU should be free: with --wait-port the runner first waits for a server
on that port to stop and for Device memory to drop.

Covered changes:
  1. recency-ordered retention (--recency-eviction)   -> "recency-retention",
     "review-publication-recency", "review-shared-survives-ladder"
  2. cost-scaled admission search budget               -> ninfer_resource_manager_test (host)
  3. shared-catalog saturation reclaim (issue #251)     -> "shared-saturation-reclaim",
     "shared-replacement", "review-shared-reclaim-lru"
  4. spaced automatic long anchors                     -> "review-anchor-spacing"
  5. agent traffic under contention: a shared system + tools prefix, a growing multi-turn
     conversation with tool calls, thinking on and preserved -> the "agent-*" scenarios, which
     assert shared-prefix reuse, pressure evictions and no leaked shared references.

Usage:
  python3 tools/smoke/prefix_reuse_issues.py --artifact model.ninfer [--build-dir build]
      [--build-config Release] [--wait-port 8080] [--log smoke.log]
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXE_SUFFIX = ".exe" if os.name == "nt" else ""
E2E_BINARY_NAME = "ninfer_qwen3_5_prefix_real_test" + EXE_SUFFIX
UNIT_BINARY_NAME = "ninfer_resource_manager_test" + EXE_SUFFIX

# A binary built before the newest commit touching these sources may predate a cache fix.
STALENESS_WATCHED_SOURCES = [
    "src/runtime/engine/context_cache/resource_manager.h",
    "src/runtime/engine/context_cache/materialization_planner.h",
    "src/models/qwen3_5/program/planning/pressure_planner.cpp",
    "src/models/qwen3_5/program/transactions/commit.cpp",
    "src/models/qwen3_5/program/transactions/capture.cpp",
]


@dataclass
class Scenario:
    label: str
    kind: str  # "e2e" or "unit"
    name: str  # NINFER_PREFIX_REAL_SCENARIO value, or "all" for the host suite


SCENARIOS = [
    Scenario("recency-ordered prefix retention", "e2e", "recency-retention"),
    Scenario("publication counts as recency", "e2e", "review-publication-recency"),
    Scenario("a recent shared prefix survives the ladder", "e2e", "review-shared-survives-ladder"),
    Scenario("issue #251 shared-catalog saturation reclaim", "e2e", "shared-saturation-reclaim"),
    Scenario("issue #251 shared reuse at full capacity", "e2e", "shared-replacement"),
    Scenario("issue #251 LRU reclaim order", "e2e", "review-shared-reclaim-lru"),
    Scenario("spaced automatic long anchors reach deep history", "e2e", "review-anchor-spacing"),
    Scenario("agent: shared system+tools prefix across tool-call turns", "e2e", "agent-multi-turn"),
    Scenario("agent: private-continuation saturation keeps the most recent", "e2e",
             "agent-private-continuations"),
    Scenario("agent: concurrent requests settle without leaked references", "e2e",
             "agent-concurrent"),
    Scenario("agent: long conversations overflow the KV capacity", "e2e", "agent-kv-pressure"),
    Scenario("agent: automatic prefixes reclaim a saturated shared catalog", "e2e",
             "agent-shared-catalog"),
    Scenario("resource manager suite (cost-scaled search budget, reclaim)", "unit", "all"),
]


class Tee:
    """Writes to stdout and an append-mode log file, flushing both on every write."""

    def __init__(self, stream, path: Path):
        self._stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def newest_source_commit_unix_time() -> int | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%ct", "--", *STALENESS_WATCHED_SOURCES],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def find_binary(tests_dir: Path, name: str, build_config: str | None) -> Path | None:
    """The binary in tests/ (single-config generators) or tests/<config>/ (multi-config)."""
    if build_config is not None:
        path = tests_dir / build_config / name
        return path if path.is_file() else None
    candidates = [tests_dir / name]
    if tests_dir.is_dir():
        candidates += [child / name for child in tests_dir.iterdir() if child.is_dir()]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def device_memory_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for_free_gpu(host: str, port: int, timeout_s: float, threshold_mib: int = 2048) -> None:
    """Waits for the server on `port` to stop and then for it to release Device memory: a
    lingering process can still hold the model after its port closes."""
    deadline = time.monotonic() + timeout_s
    while port_in_use(host, port):
        if time.monotonic() >= deadline:
            raise SystemExit(f"port {port} still in use after {timeout_s:.0f}s")
        print(f"  port {port} in use; waiting for the server to stop")
        time.sleep(5)
    while True:
        used = device_memory_used_mib()
        if used is None or used < threshold_mib:
            return
        if time.monotonic() >= deadline:
            print(f"  warning: Device memory still at {used} MiB after {timeout_s:.0f}s")
            return
        print(f"  Device memory at {used} MiB; waiting for it to be released")
        time.sleep(5)


def run_command(cmd: list[str], env: dict[str, str], timeout_s: float,
                heartbeat_s: float = 30.0) -> int:
    """Runs cmd with its output in a temporary file (a chatty binary cannot fill a pipe), prints a
    heartbeat while it runs and the tail of its output when it exits."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    started = time.monotonic()
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False,
                                     encoding="utf-8") as log:
        log_path = log.name
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        timed_out = False
        while proc.poll() is None:
            remaining = started + timeout_s - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                timed_out = True
                break
            time.sleep(min(heartbeat_s, remaining))
            if proc.poll() is None:
                print(f"  still running ({time.monotonic() - started:.0f}s)", flush=True)
    with open(log_path, encoding="utf-8", errors="replace") as log:
        content = log.read()
    os.unlink(log_path)
    tail = "\n".join(content.strip().splitlines()[-40:])
    if tail:
        print(tail, flush=True)
    print(f"  -> exit {proc.returncode} in {time.monotonic() - started:.1f}s"
          + (" (timeout)" if timed_out else ""), flush=True)
    return proc.returncode


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, help="the .ninfer artifact")
    parser.add_argument("--build-dir", default=str(REPO / "build"), help="CMake build directory")
    parser.add_argument("--build-config", default=None,
                        help="multi-config build directory name; default: the newest binary")
    parser.add_argument("--scenario", action="append", default=None,
                        help="run only these scenario names (repeatable)")
    parser.add_argument("--wait-port", type=int, default=None,
                        help="wait for a server on this port to stop and free the GPU first")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--wait-timeout", type=float, default=3600)
    parser.add_argument("--e2e-timeout", type=float, default=1800,
                        help="per-scenario timeout in seconds")
    parser.add_argument("--unit-timeout", type=float, default=300)
    parser.add_argument("--log", default=None, help="also append the output to this file")
    args = parser.parse_args()

    if args.log:
        sys.stdout = Tee(sys.stdout, Path(args.log))
    if not Path(args.artifact).is_file():
        raise SystemExit(f"artifact not found: {args.artifact}")
    tests_dir = Path(args.build_dir) / "tests"
    e2e_binary = find_binary(tests_dir, E2E_BINARY_NAME, args.build_config)
    unit_binary = find_binary(tests_dir, UNIT_BINARY_NAME, args.build_config)
    if e2e_binary is None or unit_binary is None:
        raise SystemExit(f"test binaries not found under {tests_dir}; build the tests first")

    built = e2e_binary.stat().st_mtime
    print(f"e2e binary: {e2e_binary} (built {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(built))})")
    newest = newest_source_commit_unix_time()
    if newest is not None and built < newest:
        print("warning: the e2e binary predates the newest commit touching the cache sources; "
              "rebuild it before trusting a result", file=sys.stderr)

    if args.wait_port is not None:
        wait_for_free_gpu(args.host, args.wait_port, args.wait_timeout)

    selected = [scenario for scenario in SCENARIOS
                if args.scenario is None or scenario.name in args.scenario]
    env = dict(os.environ, NINFER_TEST_ARTIFACT=str(Path(args.artifact).resolve()))
    results: list[tuple[Scenario, int]] = []
    for scenario in selected:
        if scenario.kind == "e2e":
            code = run_command([str(e2e_binary)],
                               dict(env, NINFER_PREFIX_REAL_SCENARIO=scenario.name),
                               args.e2e_timeout)
        else:
            code = run_command([str(unit_binary)], dict(os.environ), args.unit_timeout)
        results.append((scenario, code))

    print("\n=== summary ===")
    failed = 0
    for scenario, code in results:
        # A scenario that skips (exit 77) did not run, which is a failure for this runner.
        passed = code == 0
        failed += 0 if passed else 1
        print(f"  [{'PASS' if passed else 'FAIL'}] {scenario.label}  (exit {code})")
    if failed:
        print(f"\n{failed} of {len(results)} failed")
        return 1
    print(f"\nall {len(results)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
