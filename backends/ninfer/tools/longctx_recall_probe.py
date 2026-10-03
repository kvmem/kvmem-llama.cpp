#!/usr/bin/env python3
"""Long-context recall probe: needle recall, cold prefill against a cached prefix.

Plants a unique recall code at four depths of one long prompt and asks the model for all four.
The identical prompt runs twice against a running ninfer-serve:

  cold  the first request prefills the whole prompt;
  warm  the second request is served from the cached prefix (the report shows how many prompt
        tokens the server reports as cached).

Reading the result:
  - recall fails in both runs  -> long-context quality at this length (position interpolation,
    KV precision, the model itself);
  - recall fails only warm     -> the cache restore path corrupts deep positions;
  - recall passes in both      -> attention recall is intact at this length.

The default depths, about 3%, 37%, 70% and 97% of the target, straddle a 262144-token native
window at a 300k target, so a run with --rope-scaling-factor covers both sides of it. A prefill
this long takes minutes and occupies the server; run it on an idle server.

Stdlib only. Usage:
  python3 tools/longctx_recall_probe.py [--url http://127.0.0.1:8080] [--target-tokens 300000]
      [--model ID] [--max-tokens 512] [--timeout 1800] [--json out.json]
The API key, if the server has one, is read from $NINFER_API_KEY.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request

CHARS_PER_TOKEN = 4.2
DEPTHS = (0.033, 0.367, 0.700, 0.967)


def request(url, api_key, path, payload=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def make_codes(rng):
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return ["-".join("".join(rng.choice(alphabet) for _ in range(4)) for _ in range(2))
            for _ in DEPTHS]


def build_prompt(target_tokens, codes, rng):
    """Filler text with one marker line per depth; returns the prompt and each marker's offset."""
    total_chars = int(target_tokens * CHARS_PER_TOKEN)
    vocabulary = ("the quick brown fox jumps over lazy dog near river under bridge past mill "
                  "beside orchard along coast within valley beyond ridge across plain through "
                  "forest around harbor near station").split()
    doc = " ".join(rng.choices(vocabulary, k=total_chars // 6 + 64))[:total_chars]
    markers = []
    for zone, (depth, code) in enumerate(zip(DEPTHS, codes), start=1):
        line = (f"MARKER {zone}: The recall code for zone {zone} is {code}. "
                f"Remember the exact code {code} for zone {zone}.")
        at = min(int(total_chars * depth), len(doc) - len(line) - 16)
        doc = doc[:at] + "\n" + line + "\n" + doc[at:]
        markers.append((zone, code, at))
    question = ("\n\nFour MARKER lines in the text above each define a recall code for a zone "
                "(zones 1 to 4). Reply with exactly the four recall codes, one per line, in zone "
                "order. Nothing else.")
    return doc + question, markers


def run_once(args, api_key, model, prompt):
    payload = {
        "model": model,
        "instructions": "You are a precise recall assistant.",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": args.max_tokens,
        "store": False,
        "stream": False,
    }
    started = time.monotonic()
    out = request(args.url, api_key, "/v1/responses", payload, args.timeout)
    wall = time.monotonic() - started
    text = "".join(part.get("text", "")
                   for item in out.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")
    return text, out.get("usage") or {}, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", help="model id (default: the server's only model)")
    parser.add_argument("--target-tokens", type=int, default=300000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1800, help="per-request seconds")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    api_key = os.environ.get("NINFER_API_KEY")
    model = args.model or request(args.url, api_key, "/v1/models")["data"][0]["id"]
    rng = random.Random(args.seed)
    codes = make_codes(rng)
    prompt, markers = build_prompt(args.target_tokens, codes, rng)
    estimated = len(prompt) / CHARS_PER_TOKEN
    print(f"prompt: {len(prompt)} chars, about {int(estimated)} tokens", flush=True)

    results = {}
    for mode in ("cold", "warm"):
        text, usage, wall = run_once(args, api_key, model, prompt)
        prompt_tokens = usage.get("input_tokens")
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
        scale = prompt_tokens / estimated if prompt_tokens else 1.0
        zones = [{"zone": zone, "code": code,
                  "tokens": int(at / CHARS_PER_TOKEN * scale), "recalled": code in text}
                 for zone, code, at in markers]
        recalled = sum(zone["recalled"] for zone in zones)
        results[mode] = {"prompt_tokens": prompt_tokens, "cached_tokens": cached,
                         "wall_seconds": round(wall, 1), "recalled": recalled, "of": len(codes),
                         "zones": zones, "response": text[:2000]}
        print(f"\n{mode}: prompt_tokens={prompt_tokens} cached_tokens={cached} "
              f"wall={wall:.1f}s recalled={recalled}/{len(codes)}", flush=True)
        for zone in zones:
            print(f"  {'ok  ' if zone['recalled'] else 'MISS'} zone {zone['zone']} at about "
                  f"{zone['tokens']} tokens: {zone['code']}", flush=True)

    cold, warm = results["cold"]["recalled"], results["warm"]["recalled"]
    full = len(codes)
    if cold == full and warm == full:
        verdict = "recall intact at this length, cold and warm"
    elif cold < full and warm < full:
        verdict = "recall fails cold and warm: long-context quality at this length"
    elif warm < full:
        verdict = "recall fails only warm: the cache restore path loses deep positions"
    else:
        verdict = "warm recalls more than cold: inspect the zones"
    print(f"\nverdict: {verdict}")
    if not results["warm"]["cached_tokens"]:
        print("note: the warm run reports no cached tokens, so it did not exercise the cache")
    if args.json_out:
        with open(args.json_out, "w") as out:
            json.dump({"target_tokens": args.target_tokens, "seed": args.seed,
                       "results": results, "verdict": verdict}, out, indent=2)
    return 0 if cold == full and warm == full else 1


if __name__ == "__main__":
    sys.exit(main())
