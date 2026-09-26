#!/usr/bin/env python3
"""Exercise /v1/responses through the official OpenAI Python SDK.

The SDK drives the Responses stream with its own state machine, so this catches
wire-shape problems the in-repo compat suite cannot: an unknown event, a delta
addressed to the wrong item id, or a missing content_part all raise before the
final text is assembled.

Usage:
    python scripts/test_responses_sdk.py --server PATH --model PATH
    ... --mmproj PATH --mtp                 # MTP and vision fixtures

Exit code 0 when every check passes.
"""

import argparse
import base64
import os
import subprocess
import sys
import time
import urllib.request

import openai

FAILURES = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" -- " + str(detail)[:300]) if not ok else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def wait_healthy(proc, port, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode} before becoming healthy")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not become healthy")


def text_of(response):
    return "".join(part.text or ""
                   for item in response.output if item.type == "message"
                   for part in item.content if part.type == "output_text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--port", type=int, default=18270)
    ap.add_argument("--log", default=os.path.join(os.environ.get("TEMP", "/tmp"),
                                                  "responses-sdk-server.log"))
    args = ap.parse_args()

    flags = ["-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
             "-c", "8192", "-np", "1", "--no-webui", "--predict", "512"]
    if args.mmproj:
        flags += ["--mmproj", args.mmproj]
    if args.mtp:
        # The same MTP configuration the compat suite uses.
        flags += ["--kv-dtype", "q8_0", "--spec-type", "draft-mtp",
                  "--spec-draft-n-max", "3", "--spec-kv-dtype", "f16",
                  "--kvmem-mtp-state", "replay"]

    log = open(args.log, "wb")
    proc = subprocess.Popen([args.server] + flags, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_healthy(proc, args.port)
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="sk-test")
        tools = [{
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }]

        # --- non-streaming text --------------------------------------------
        r = client.responses.create(model="test", input="What is 2+3? Answer with the number only.",
                                    max_output_tokens=32, temperature=0)
        check("non-stream text", text_of(r).strip() == "5", text_of(r))
        check("non-stream shape",
              r.object == "response" and r.status == "completed" and r.id.startswith("resp_"),
              r.model_dump())
        check("non-stream usage",
              r.usage.input_tokens > 0 and r.usage.output_tokens > 0
              and r.usage.total_tokens == r.usage.input_tokens + r.usage.output_tokens
              and r.usage.input_tokens_details is not None,
              r.usage)

        # --- streaming text -------------------------------------------------
        events, text = [], ""
        for event in client.responses.create(model="test", input="What is 2+3? Answer with the number only.",
                                             max_output_tokens=32, temperature=0, stream=True):
            events.append(event.type)
            if event.type == "response.output_text.delta":
                text += event.delta
        check("stream text", text.strip() == "5", text)
        check("stream text matches non-stream", text.strip() == text_of(r).strip(),
              (text, text_of(r)))
        check("stream lifecycle",
              events.count("response.created") == 1
              and events.count("response.in_progress") == 1
              and events.count("response.output_item.added") >= 1
              and events.count("response.output_text.done") == 1
              and events.count("response.content_part.done") == 1
              and events.count("response.output_item.done") >= 1
              and events.count("response.completed") == 1
              and events[-1] == "response.completed", events)

        # --- streaming with reasoning (each item type must parse) -----------
        events, text, reasoning = [], "", ""
        for event in client.responses.create(model="test", input="What is 12*12? Answer with the number only.",
                                             max_output_tokens=256, temperature=0,
                                             reasoning={"effort": "low"}, stream=True):
            events.append(event.type)
            if event.type == "response.output_text.delta":
                text += event.delta
            elif event.type == "response.reasoning_text.delta":
                reasoning += event.delta
        check("reasoning stream text", "144" in text, text)
        check("reasoning stream completes",
              events.count("response.completed") == 1 and events[-1] == "response.completed", events)
        check("reasoning item streamed",
              (not reasoning) or "response.output_item.added" in events, (reasoning[:40], events))

        r = client.responses.create(model="test", input="What is 12*12? Answer with the number only.",
                                    max_output_tokens=256, temperature=0,
                                    reasoning={"effort": "low"})
        kinds = [item.type for item in r.output]
        check("non-stream reasoning text", "144" in text_of(r), text_of(r))
        check("non-stream item types",
              all(k in ("reasoning", "message", "function_call") for k in kinds), kinds)
        for item in r.output:
            item_data = item.model_dump()
            if item.type == "reasoning" and item_data.get("content"):
                summary = item_data.get("summary") or []
                check("non-stream reasoning summary",
                      bool(summary) and summary[0].get("text") == item_data["content"][0].get("text"),
                      item_data)

        # --- streaming tool call --------------------------------------------
        events, calls = [], []
        for event in client.responses.create(model="test", input="Call get_weather for San Francisco.",
                                             tools=tools, max_output_tokens=256, temperature=0,
                                             stream=True):
            events.append(event.type)
            if event.type == "response.output_item.done" and event.item.type == "function_call":
                calls.append(event.item)
        check("stream tool call emitted",
              len(calls) == 1 and calls[0].name == "get_weather"
              and "San Francisco" in (calls[0].arguments or ""), calls)
        check("stream tool call events",
              "response.function_call_arguments.delta" in events
              and "response.function_call_arguments.done" in events, events)
        check("stream tool call completes",
              events.count("response.completed") == 1 and events[-1] == "response.completed", events)

        # --- call_id round-trip via function_call_output ---------------------
        if calls:
            r = client.responses.create(
                model="test",
                input=[{"role": "user", "content": "Call get_weather for San Francisco."},
                       {"type": "function_call", "call_id": calls[0].call_id,
                        "name": calls[0].name, "arguments": calls[0].arguments},
                       {"type": "function_call_output", "call_id": calls[0].call_id,
                        "output": "18 C and sunny"}],
                max_output_tokens=64, temperature=0)
            check("function_call_output accepted", r.status == "completed", r.model_dump())

        # --- non-streaming tool call ----------------------------------------
        r = client.responses.create(model="test", input="Call get_weather for San Francisco.",
                                    tools=tools, max_output_tokens=256, temperature=0)
        fc = [item for item in r.output if item.type == "function_call"]
        check("non-stream tool call",
              len(fc) == 1 and fc[0].name == "get_weather"
              and "San Francisco" in (fc[0].arguments or ""), fc)

        # --- instructions -> system, on both paths ---------------------------
        r = client.responses.create(model="test", instructions="Answer with the single word BANANA.",
                                    input="Say the fruit named in your instructions.",
                                    max_output_tokens=16, temperature=0)
        check("instructions honored (non-stream)", "BANANA" in text_of(r).upper(), text_of(r))

        text = ""
        for event in client.responses.create(
                model="test", instructions="Answer with the single word BANANA.",
                input="Say the fruit named in your instructions.",
                max_output_tokens=16, temperature=0, stream=True):
            if event.type == "response.output_text.delta":
                text += event.delta
        check("instructions honored (stream)", "BANANA" in text.upper(), text)

        # --- vision on both paths --------------------------------------------
        if args.mmproj:
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
            url = "data:image/png;base64," + base64.b64encode(png).decode()
            vision_input = [{"role": "user", "content": [
                {"type": "input_text", "text": "Describe this image briefly."},
                {"type": "input_image", "image_url": url}]}]
            r = client.responses.create(model="test", input=vision_input,
                                        max_output_tokens=128, reasoning={"effort": "none"})
            check("vision non-stream", r.status == "completed" and bool(text_of(r).strip()),
                  r.model_dump())
            events, text = [], ""
            for event in client.responses.create(model="test", input=vision_input,
                                                 max_output_tokens=128,
                                                 reasoning={"effort": "none"}, stream=True):
                events.append(event.type)
                if event.type == "response.output_text.delta":
                    text += event.delta
            check("vision stream", bool(text.strip()) and events.count("response.completed") == 1,
                  (events, text))
        else:
            try:
                client.responses.create(model="test", input=[{"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}]}])
                check("vision requires mmproj", False, "accepted an image")
            except openai.BadRequestError as e:
                check("vision requires mmproj", "mmproj" in str(e), str(e))

        print()
        if FAILURES:
            print(f"{len(FAILURES)} FAILURES: " + ", ".join(FAILURES))
            return 1
        print("ALL PASS")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
