#!/usr/bin/env python3
"""Regression test for the KVMem server's output-length request handling.

Clients differ in how they ask for output length: some send `max_tokens`, some
`max_completion_tokens`, some a value larger than the deployment can serve, and
some JSON `null` or `0` for "let the server decide". An OpenAI-compatible
endpoint should accept all of those and clamp, rather than rejecting with 400.

    python scripts/kvmem_output_limit_test.py --port 18200 --api-key <KEY>

Exit codes: 0 all cases behaved, 1 a case misbehaved, 2 the server was unreachable.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# A prompt whose answer is short, so a huge requested limit is never actually
# reached and each case returns quickly.
PROMPT = "Answer with just the number: 6*7?"


def post(url, payload, api_key, timeout):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw[:200]}


CASES = [
    # (label, extra body fields, expected status)
    ("no length field",                  {},                                    200),
    ("max_completion_tokens = 64",       {"max_completion_tokens": 64},         200),
    ("max_completion_tokens = 12288",    {"max_completion_tokens": 12288},      200),
    ("max_completion_tokens = 65536",    {"max_completion_tokens": 65536},      200),
    ("max_completion_tokens = 2000000",  {"max_completion_tokens": 2000000},    200),
    ("max_completion_tokens = -1",       {"max_completion_tokens": -1},         200),
    ("max_completion_tokens = 0",        {"max_completion_tokens": 0},          200),
    ("max_completion_tokens = null",     {"max_completion_tokens": None},       200),
    ("max_completion_tokens = 4096.0",   {"max_completion_tokens": 4096.0},     200),
    ("max_tokens = 200000",              {"max_tokens": 200000},               200),
    ("max_completion_tokens = 'abc'",    {"max_completion_tokens": "abc"},      400),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    url = "http://%s:%d/v1/chat/completions" % (args.host, args.port)
    failures = []

    print("%-34s %-8s %-8s %s" % ("case", "expect", "got", "verdict"))
    print("-" * 66)
    for label, extra, expected in CASES:
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0.0,
        }
        payload.update(extra)
        try:
            status, body = post(url, payload, args.api_key, args.timeout)
        except Exception as exc:
            status, body = -1, {"error": str(exc)}

        ok = status == expected
        if not ok:
            failures.append(label)
        detail = ""
        if status == 200:
            content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            detail = repr(content.strip()[:24])
        else:
            detail = json.dumps(body)[:80]
        print("%-34s %-8d %-8s %s  %s"
              % (label, expected, status, "OK" if ok else "MISMATCH", detail))

    print()
    if failures:
        print("FAILED cases: %s" % ", ".join(failures))
        return 1
    print("all %d cases behaved as expected" % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
