#!/usr/bin/env python3
"""Compare bounded KVMem KV memory with native KV memory on one ROCm GPU.

The sampler records whole-device VRAM from amd-smi and process RSS.  It first
runs native KV at the long prompt length, then KVMem at a short and long
length.  KVMem is accepted only when its reported slot-pool bytes/cells are
identical at both lengths and its sampled long-run VRAM peak is no higher than
the native long run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KV_RE = re.compile(r"KVMEM_KV_BYTES bytes=(\d+) cells=(\d+).*")
MTP_RE = re.compile(r"KVMEM_TRACE mtp_pool cells=(\d+).* bytes=(\d+).*")
PROMPT_RE = re.compile(r"n_prompt=(\d+)")


def find_cli() -> Path:
    for path in (ROOT / "build-hip-linux/bin/llama-kvmem-cli", ROOT / "build-rocm/bin/llama-kvmem-cli", ROOT / "build/bin/llama-kvmem-cli"):
        if path.is_file():
            return path
    raise SystemExit("llama-kvmem-cli not found; run scripts/build-rocm.sh")


def vram_mib(gpu: int) -> float | None:
    try:
        result = subprocess.run(
            ["amd-smi", "metric", "--mem-usage", "--gpu", str(gpu), "--json"],
            capture_output=True, text=True, timeout=5, check=True)
        return float(result and json.loads(result.stdout)["gpu_data"][0]["mem_usage"]["used_vram"]["value"])
    except (KeyError, IndexError, ValueError, json.JSONDecodeError, OSError, subprocess.SubprocessError):
        return None


def rss_mib(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def run_case(cli: Path, model: Path, words: int, gpu: int, kvmem: bool,
             budget: int, reserve: int, batch: int, mtp_draft_n_max: int) -> dict:
    prompt = " memory" * words
    with tempfile.NamedTemporaryFile("w", prefix="kvmem_rocm_", suffix=".txt", delete=False) as handle:
        handle.write(prompt)
        prompt_file = handle.name
    try:
        ctx = words + reserve + 128
        command = [str(cli), "-m", str(model), "-f", prompt_file, "-n", "1", "-c", str(ctx),
                   "-b", str(batch), "-ngl", "99", "--temp", "0", "--no-prompt", "--kv-dtype", "q8_0"]
        if mtp_draft_n_max:
            command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp_draft_n_max)]
        if kvmem:
            command += ["--kvmem", "--kvmem-method", "recency", "--kvmem-budget", str(budget),
                        "--kvmem-gen-reserve", str(reserve), "--kvmem-block-tokens", "32"]
        env = os.environ.copy()
        env["KVMEM_TRACE"] = "1"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        peaks = {"vram": vram_mib(gpu), "rss": 0.0}
        done = threading.Event()

        def sample() -> None:
            while not done.is_set():
                measured = vram_mib(gpu)
                if measured is not None:
                    peaks["vram"] = max(peaks["vram"] or 0.0, measured)
                peaks["rss"] = max(peaks["rss"], rss_mib(process.pid) or 0.0)
                time.sleep(0.15)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        stdout, stderr = process.communicate()
        done.set()
        sampler.join(timeout=6)
        info = {"rc": process.returncode, "vram_mib": peaks["vram"], "rss_mib": peaks["rss"],
                "stderr": stderr, "stdout": stdout, "words": words}
        if match := KV_RE.search(stderr):
            info["kv_bytes"] = int(match.group(1))
            info["kv_cells"] = int(match.group(2))
        if match := MTP_RE.search(stderr):
            info["mtp_cells"] = int(match.group(1))
            info["mtp_bytes"] = int(match.group(2))
        if match := PROMPT_RE.search(stderr):
            info["n_prompt"] = int(match.group(1))
        return info
    finally:
        Path(prompt_file).unlink(missing_ok=True)


def summary(name: str, row: dict) -> None:
    vram = f"{row['vram_mib']:.1f}" if row['vram_mib'] is not None else 'unavailable'
    print(f"{name}: rc={row['rc']} prompt={row.get('n_prompt')} vram_peak={vram} MiB "
          f"rss_peak={row['rss_mib']:.1f} MiB kv_bytes={row.get('kv_bytes')} cells={row.get('kv_cells')} "
          f"mtp_bytes={row.get('mtp_bytes')} mtp_cells={row.get('mtp_cells')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--short-words", type=int, default=1024)
    parser.add_argument("--long-words", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--reserve", type=int, default=128)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--mtp-draft-n-max", type=int, default=0,
                        help="enable model-internal MTP speculative decoding with this many draft tokens")
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.long_words <= args.short_words:
        raise SystemExit("--long-words must exceed --short-words")

    cli = find_cli()
    native = run_case(cli, args.model, args.long_words, args.gpu, False, args.budget, args.reserve, args.batch,
                      args.mtp_draft_n_max)
    short = run_case(cli, args.model, args.short_words, args.gpu, True, args.budget, args.reserve, args.batch,
                     args.mtp_draft_n_max)
    long = run_case(cli, args.model, args.long_words, args.gpu, True, args.budget, args.reserve, args.batch,
                    args.mtp_draft_n_max)
    summary("native-long", native)
    summary("kvmem-short", short)
    summary("kvmem-long", long)

    failures = []
    if native['rc'] != 0:
        failures.append(f"native-long exited {native['rc']}")
    for name, row, requested in (("kvmem-short", short, args.short_words),
                                 ("kvmem-long", long, args.long_words)):
        if row["rc"] != 0:
            failures.append(f"{name} exited {row['rc']}")
        if row.get("n_prompt", 0) < requested // 2:
            failures.append(f"{name} did not process its requested prompt")
    if short.get("kv_bytes") is None or long.get("kv_bytes") is None:
        failures.append("KVMEM_KV_BYTES log missing")
    elif (short["kv_bytes"], short["kv_cells"]) != (long["kv_bytes"], long["kv_cells"]):
        failures.append("KVMem GPU KV bytes/cells grew with prompt length")
    if args.mtp_draft_n_max:
        if short.get("mtp_bytes") is None or long.get("mtp_bytes") is None:
            failures.append("KVMem MTP pool log missing")
        elif (short["mtp_bytes"], short["mtp_cells"]) != (long["mtp_bytes"], long["mtp_cells"]):
            failures.append("KVMem MTP GPU KV bytes/cells grew with prompt length")
    if long['vram_mib'] is None or native['vram_mib'] is None:
        failures.append('VRAM sampling unavailable; check amd-smi access before comparing peaks')
    elif long["vram_mib"] > native["vram_mib"]:
        failures.append("KVMem long-run VRAM peak exceeded native KV peak")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS: KVMem GPU KV stays bounded and its long-run VRAM peak is no higher than native KV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
