"""How many bytes does one decoded token actually read?

Decode is memory-bound: the step time is set by how much of the card's bandwidth the path can
use, so "% of peak" is the one number that says whether there is headroom left. That needs a
denominator, and this repository did not have one.

Two things were wrong with the arithmetic used before.

**Resident is not read.** Dividing throughput into `weights_capacity_bytes` counts weights the
decode path never streams: the vision tower, the DFlash and MTP bundles when they are not
selected, the draft head with speculation off, and all but one row of the token embedding. On
the Qwen3.8-27B that is 1.12 GB of the 16.86 GB resident -- reporting against the resident
figure overstates the achieved bandwidth by about six points.

**An MoE never reads its resident weights at all.** Applying the dense formula to the
Qwen3.6-35B-A3B returns 391% of peak, which is not a result, it is a proof that the formula
does not apply: 18.56 GB of its weights are routed experts, and one token touches 8 of 256 per
layer. Counting the routed tensors at top_k/experts is what makes the MoE comparable to the
dense model at all.

Both counts come from the artifact's own object directory -- names, shapes and byte lengths --
so this is arithmetic over what was actually shipped rather than a model card. Only the header
is read; the payload is never touched, so it runs in milliseconds on a 21 GB file and needs no
GPU and no torch.

    python tools/decode_byte_accounting.py models/qwen3_6_35b_a3b.ninfer\
        --kv-bytes-per-token 10560 --depth 4096 --tok-s 173.83

Pass --tok-s to get the roofline; without it you get the byte counts alone.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path

PREFIX = struct.Struct("<8sQ")
MAGIC = b"NINFER\x00\x02"

# src/ops/sparse_moe/sparse_moe_route.cuh. Every shipped MoE target uses these; if a future one
# does not, this is the constant to plumb through rather than to edit.
SPARSE_MOE_EXPERTS = 256
SPARSE_MOE_TOP_K = 8

# Advertised 384-bit GDDR6X at 19.5 Gbps, and the sustained read rate measured on this card with
# tools/hbm_bandwidth_probe.cu (4 GiB working set, best of five). Decode streams weights and KV
# *in*, so the read rate is its ceiling; the advertised number is printed beside it because it is
# the one people quote.
ADVERTISED_GB_S = 936.1
ACHIEVABLE_READ_GB_S = 854.2

# Towers a plain text decode never touches. `mtp` and `dflash` become live when the matching
# --spec is selected, which is why they are named rather than pattern-matched.
UNREAD_TOWERS = ("vision", "dflash", "mtp", "frontend")

# Read only when the optimized proposal head is selected.
UNREAD_TEXT_OBJECTS = ("text/draft_head", "text/draft_head_token_ids")


def read_directory(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        magic, json_bytes = PREFIX.unpack(handle.read(PREFIX.size))
        if magic != MAGIC:
            raise SystemExit(f"{path} is not a v2 .ninfer artifact")
        directory = json.loads(handle.read(json_bytes))
    return directory["objects"] if isinstance(directory, dict) else directory


def account(objects: list[dict]) -> dict[str, int | float]:
    dense = 0
    routed_total = 0
    embedding_row = 0
    unread: dict[str, int] = defaultdict(int)

    for obj in objects:
        name = str(obj.get("name", ""))
        size = int(obj.get("bytes", 0))
        tower = name.split("/", 1)[0]
        if tower in UNREAD_TOWERS or name in UNREAD_TEXT_OBJECTS:
            unread[tower] += size
            continue
        if name.endswith("token_embedding"):
            # One row per token, not the table.
            rows = int(obj["shape"][0])
            embedding_row = size // rows
            unread[tower] += size - embedding_row
            continue
        if "/moe/routed_" in name:
            routed_total += size
            continue
        dense += size

    routed_read = routed_total * SPARSE_MOE_TOP_K / SPARSE_MOE_EXPERTS
    return {
        "dense": dense,
        "embedding_row": embedding_row,
        "routed_total": routed_total,
        "routed_read": routed_read,
        "per_token": dense + embedding_row + routed_read,
        "unread": dict(unread),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--kv-bytes-per-token", type=int, default=0,
                        help="from ninfer_bench's kv_payload_bytes / max_context")
    parser.add_argument("--depth", type=int, action="append", default=[],
                        help="cache depth in tokens; repeatable")
    parser.add_argument("--tok-s", type=float, action="append", default=[],
                        help="measured decode tok/s at the matching --depth; repeatable")
    parser.add_argument("--json", action="store_true",
                        help="emit the byte counts as JSON, for scripts that need the denominator")
    args = parser.parse_args()

    if len(args.tok_s) != len(args.depth):
        raise SystemExit("--depth and --tok-s must be given the same number of times")

    result = account(read_directory(args.artifact))
    per_token = float(result["per_token"])

    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "unread"}))
        return

    print(f"{args.artifact.name}: bytes read per decoded token, one lane, no speculation")
    print(f"  dense weights, every token          : {result['dense']:>15,} B")
    print(f"  one token_embedding row             : {result['embedding_row']:>15,} B")
    if result["routed_total"]:
        print(f"  routed experts, all {SPARSE_MOE_EXPERTS}             : "
              f"{result['routed_total']:>15,} B")
        print(f"  routed experts read ({SPARSE_MOE_TOP_K}/{SPARSE_MOE_EXPERTS})        : "
              f"{result['routed_read']:>15,.0f} B")
    print(f"  {'-' * 56}")
    print(f"  per token                           : {per_token:>15,.0f} B "
          f"({per_token / 1e9:.3f} GB)")
    if result["unread"]:
        print("  resident but not read by a text decode:")
        for tower, size in sorted(result["unread"].items(), key=lambda kv: -kv[1]):
            print(f"    {tower:<10} {size:>15,} B")

    if not args.depth:
        return
    print()
    print(f"  {'depth':>8} {'tok/s':>8} {'KV GB':>8} {'GB/s':>8} {'advertised':>11} "
          f"{'achievable':>11}")
    for depth, tok_s in zip(args.depth, args.tok_s):
        kv = args.kv_bytes_per_token * depth
        achieved = (per_token + kv) * tok_s
        print(f"  {depth:>8} {tok_s:>8.2f} {kv / 1e9:>8.3f} {achieved / 1e9:>8.1f} "
              f"{100 * achieved / (ADVERTISED_GB_S * 1e9):>10.1f}% "
              f"{100 * achieved / (ACHIEVABLE_READ_GB_S * 1e9):>10.1f}%")


if __name__ == "__main__":
    main()
