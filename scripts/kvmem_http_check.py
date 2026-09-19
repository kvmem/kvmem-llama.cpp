#!/usr/bin/env python3
"""Smoke-check a running llama-kvmem-server over its OpenAI-compatible API.

No third-party dependencies: this machine has no curl/requests guarantee, so
everything goes through urllib.

    python scripts/kvmem_http_check.py                    # default question
    python scripts/kvmem_http_check.py --wait 1800        # wait for a load
    python scripts/kvmem_http_check.py --prompt "..." --max-tokens 256

Exit codes: 0 all checks passed, 1 a request failed, 2 the server never became
healthy within --wait seconds.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Optional API key, set from --api-key. Sent as `Authorization: Bearer <key>`,
# which is what an OpenAI-compatible client uses.
API_KEY = None


def request(url, payload=None, timeout=600):
    data = None
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def wait_healthy(base, limit):
    """Poll /health until it reports ok. Returns elapsed seconds, or None."""
    started = time.time()
    last = None
    while time.time() - started < limit:
        try:
            status, body = request(base + "/health", timeout=10)
            last = body.get("status", status)
            if body.get("status") == "ok":
                return time.time() - started
        except Exception as exc:            # connection refused while loading
            last = type(exc).__name__
        time.sleep(5)
    print("  last /health state: %s" % last)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--wait", type=int, default=0,
                    help="seconds to wait for /health (0 = single probe)")
    ap.add_argument("--prompt", default="What is 2+3? Answer with the number only.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--api-key", default=None,
                    help="send Authorization: Bearer <key>; also asserts that a "
                         "request without it is rejected with 401")
    args = ap.parse_args()

    global API_KEY
    API_KEY = args.api_key

    base = "http://%s:%d" % (args.host, args.port)
    failures = []

    print("[1/5] /health")
    if args.wait > 0:
        elapsed = wait_healthy(base, args.wait)
        if elapsed is None:
            print("  FAIL: not healthy within %ss" % args.wait)
            return 2
        print("  ok, healthy after %.1fs" % elapsed)
    else:
        try:
            _, body = request(base + "/health", timeout=15)
            print("  %s" % body)
        except Exception as exc:
            print("  FAIL: %s" % exc)
            return 2

    print("[2/5] /v1/models")
    try:
        _, body = request(base + "/v1/models", timeout=30)
        for m in body.get("data", []):
            print("  %s" % m.get("id"))
    except Exception as exc:
        failures.append("/v1/models: %s" % exc)
        print("  FAIL: %s" % exc)

    print("[3/5] POST /v1/chat/completions")
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": False,
    }
    started = time.time()
    try:
        _, body = request(base + "/v1/chat/completions", payload, timeout=1800)
        wall = time.time() - started
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            print("  reasoning (%d chars): %s" % (len(reasoning), reasoning[:200]))
        print("  content: %r" % content[:400])
        print("  finish_reason: %s" % choice.get("finish_reason"))
        usage = body.get("usage") or {}
        print("  usage: %s" % json.dumps(usage, ensure_ascii=False))
        print("  wall: %.2fs" % wall)
        comp = usage.get("completion_tokens") or 0
        if content.strip():
            pass
        elif reasoning.strip() and choice.get("finish_reason") == "length":
            # A thinking model that exhausted max_tokens inside its reasoning
            # block is working correctly, not failing.
            print("  note: max_tokens exhausted inside the reasoning block; "
                  "raise --max-tokens to see a final answer")
        else:
            failures.append("empty content")
            print("  FAIL: empty content")
        if comp and wall > 0:
            print("  approx generation rate: %.2f tok/s" % (comp / wall))
    except Exception as exc:
        failures.append("chat: %s" % exc)
        print("  FAIL: %s" % exc)

    print("[4/5] auth enforcement")
    if args.api_key:
        saved = API_KEY
        API_KEY = None
        try:
            request(base + "/v1/models", timeout=30)
            failures.append("unauthenticated /v1/models was accepted")
            print("  FAIL: request without a key was accepted")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                print("  ok: no key -> 401")
            else:
                failures.append("unauthenticated request returned %s" % exc.code)
                print("  FAIL: expected 401, got %s" % exc.code)
        except Exception as exc:
            failures.append("auth probe: %s" % exc)
            print("  FAIL: %s" % exc)
        finally:
            API_KEY = saved
    else:
        print("  skipped (no --api-key given, so the server is unauthenticated)")

    print("[5/5] verdict")
    if failures:
        for f in failures:
            print("  FAIL: %s" % f)
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
