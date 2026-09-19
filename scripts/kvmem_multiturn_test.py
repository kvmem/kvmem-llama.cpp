#!/usr/bin/env python3
"""Multi-turn growth test for llama-kvmem-server.

A single large prompt and a long conversation exercise different paths: each new
turn re-prefills, KVMem commits the previous turn into its host store, and the
checkpoint/capture bookkeeping accumulates. This walks the context up in steps
and reports where it breaks, if anywhere.

    python scripts/kvmem_multiturn_test.py --port 18200 --api-key <KEY> \
        --turns 4 --tokens-per-turn 20000

Exit codes: 0 all turns served, 1 a turn failed, 2 the server was unreachable.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "The maintenance log for the northern relay station records routine checks, "
    "calibration drift, cable inspection, battery voltage and the ambient "
    "temperature at the time of each visit. "
)


def post(url, payload, api_key, timeout):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw[:300]}
    except Exception as exc:
        return -1, {"error": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--tokens-per-turn", type=int, default=20000)
    ap.add_argument("--chars-per-token", type=float, default=5.9)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    url = "http://%s:%d/v1/chat/completions" % (args.host, args.port)
    chars = int(args.tokens_per_turn * args.chars_per_token)
    body = FILLER * (chars // len(FILLER))

    # A growing conversation: every turn carries the whole history.
    messages = []
    failures = []
    for turn in range(1, args.turns + 1):
        messages.append({"role": "user", "content": "Section %d. %s" % (turn, body)})
        messages.append({"role": "assistant", "content": "Noted section %d." % turn})

        payload = {
            "model": "local",
            "messages": messages + [{"role": "user", "content": "Say OK"}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
        approx = (len(messages) + 1) * args.tokens_per_turn
        print("turn %d: ~%d tokens of history -> " % (turn, approx), end="", flush=True)
        started = time.time()
        status, out = post(url, payload, args.api_key, args.timeout)
        wall = time.time() - started

        if status != 200:
            print("FAIL status=%s (%s)" % (status, json.dumps(out)[:160]))
            failures.append(turn)
            break

        usage = out.get("usage") or {}
        print("ok  prompt=%s  %s  %.1fs"
              % (usage.get("prompt_tokens"), json.dumps(usage)[:90], wall))

    print()
    if failures:
        print("FAILED at turn(s): %s" % failures)
        return 1
    print("all %d turns served" % args.turns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
