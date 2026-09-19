#!/usr/bin/env python3
"""KVMem long-context needle test.

KVMem's whole point is keeping a logical workspace far larger than the GPU KV
budget. This builds a prompt longer than that budget, hides one unique fact
early in it, and checks whether the server still retrieves it.

    python scripts/kvmem_needle_test.py --target-tokens 48000
    python scripts/kvmem_needle_test.py --port 18200 --budget 32768

Exit codes: 0 retrieved, 1 not retrieved, 2 the request failed.
"""

import argparse
import json
import sys
import time
import urllib.request

FILLER = (
    "The maintenance log for the northern relay station records routine checks, "
    "calibration drift, cable inspection, battery voltage and the ambient "
    "temperature at the time of each visit. "
)

NEEDLE_CODE = "8341-QX"
NEEDLE = (
    "IMPORTANT RECORD: the vault access code for project ZEPHYR is "
    + NEEDLE_CODE
    + " and it must be quoted exactly. "
)

QUESTION = (
    "Earlier in this conversation a record states the vault access code for "
    "project ZEPHYR. Reply with that code exactly and nothing else."
)


def post(url, payload, timeout, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--target-tokens", type=int, default=48000,
                    help="approximate prompt length to build")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--budget", type=int, default=0,
                    help="GPU KV budget in tokens, for the report only")
    ap.add_argument("--needle-at", type=float, default=0.20,
                    help="where to hide the needle, as a fraction of the prompt")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--api-key", default=None,
                    help="send Authorization: Bearer <key> (needed when the "
                         "server was started with --api-key)")
    ap.add_argument("--no-think", action="store_true",
                    help="disable the reasoning block for this request "
                         "(server accepts reasoning_effort=none per request)")
    args = ap.parse_args()

    target_chars = int(args.target_tokens * args.chars_per_token)
    body_reps = max(1, target_chars // len(FILLER))

    # Put the needle at the requested position, then pad the rest so the total
    # length still matches. Everything before the needle is "old" context that a
    # bounded GPU working set has to evict.
    head_reps = max(1, int(body_reps * args.needle_at))
    tail_reps = max(1, body_reps - head_reps)
    prompt = FILLER * head_reps + NEEDLE + FILLER * tail_reps

    print("prompt: %d chars (~%d tokens by %.1f chars/token)"
          % (len(prompt), len(prompt) / args.chars_per_token, args.chars_per_token))
    print("needle: %r at %.0f%% of the prompt, %d chars in"
          % (NEEDLE_CODE, args.needle_at * 100, len(FILLER) * head_reps))
    if args.budget:
        print("GPU budget: %d tokens -> the needle is %s the budget"
              % (args.budget,
                 "INSIDE" if len(prompt) / args.chars_per_token <= args.budget else "OUTSIDE (retrieval required)"))

    url = "http://%s:%d/v1/chat/completions" % (args.host, args.port)
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt},
                     {"role": "user", "content": QUESTION}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    if args.no_think:
        payload["reasoning_effort"] = "none"

    print("\nsending...")
    started = time.time()
    try:
        body = post(url, payload, args.timeout, args.api_key)
    except Exception as exc:
        print("FAIL: request error: %s" % exc)
        return 2
    wall = time.time() - started

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or "").strip()
    usage = body.get("usage") or {}

    print("wall            : %.1fs" % wall)
    print("finish_reason   : %s" % choice.get("finish_reason"))
    print("usage           : %s" % json.dumps(usage, ensure_ascii=False))
    if usage.get("completion_tokens") and wall > 0:
        print("gen rate        : %.2f tok/s"
              % (usage["completion_tokens"] / wall))
    if reasoning:
        print("reasoning       : %s" % reasoning[:200])
    print("content         : %r" % content[:300])

    # Search the answer and the thinking block: a thinking model may answer
    # correctly inside its reasoning even when the visible tail is truncated.
    found = NEEDLE_CODE in content or NEEDLE_CODE in reasoning
    print("\nverdict: needle %s" % ("RETRIEVED" if found else "NOT FOUND"))
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
