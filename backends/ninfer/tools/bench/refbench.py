#!/usr/bin/env python3
"""refbench.py: reference measurements against one running ninfer-serve.

Every request is greedy, non-streaming, thinking off unless a suite says otherwise, and starts with
a nonce so no prefix is reused unless the suite asks for it. Device memory is sampled every 0.2 s
through NVML (nvidia-smi as a fallback); each row carries the peak seen while it ran.

Suites:
  idle      device memory of the idle server.
  depth     one cold request per depth: a document of D tokens and a question, G new tokens.
            TTFT, prefill tok/s, decode tok/s at depth, acceptance, peak memory.
  accept    short prompts with long answers (prose, code, reasoning) for draft acceptance.
  conc      C concurrent cold requests: a document of --conc-depth tokens each, or with --conc-depth 0
            the short prompts of accept; aggregate tok/s over the wall time, per-request decode tok/s.
  niah      three codes at 33/66/90 % of a document of each length; found / order.
Rows go to <out>/<suite>.jsonl; the server's request log gives the server-side TTFT.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid


def post(url, body, timeout):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Memory:
    """Device memory sampler: current and peak MiB of one GPU."""

    def __init__(self, index):
        self.index = index
        self.peak = 0
        self.current = 0
        self.stop = False
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001 - nvidia-smi fallback
            self.nvml = None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def read(self):
        if self.handle is not None:
            return self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used // (1024 * 1024)
        out = subprocess.run(["nvidia-smi", "-i", str(self.index), "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        return int(out.strip() or 0)

    def loop(self):
        while not self.stop:
            try:
                self.current = self.read()
                self.peak = max(self.peak, self.current)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)

    def reset_peak(self):
        self.peak = self.current


class Corpus:
    def __init__(self, tokenizer_path, files):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        text = "\n\n".join(open(path, encoding="utf-8").read() for path in files)
        self.ids = self.tokenizer.encode(text, add_special_tokens=False).ids

    def text(self, start, count):
        if count <= 0:
            return ""
        ids = []
        while len(ids) < start + count:
            ids.extend(self.ids)
        return self.tokenizer.decode(ids[start:start + count])


class Client:
    def __init__(self, base_url, model, timeout, memory):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.memory = memory

    def chat(self, messages, max_tokens, thinking=False, extra=None):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        if extra:
            body.update(extra)
        self.memory.reset_peak()
        started = time.time()
        response, error = None, None
        try:
            response = post(self.base_url + "/v1/chat/completions", body, self.timeout)
        except urllib.error.HTTPError as exc:
            error = "HTTP %d: %s" % (exc.code, exc.read()[:600].decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001 - a failed request is a result
            error = "%s: %s" % (type(exc).__name__, exc)
        wall = time.time() - started
        return normalize(response, wall, error, self.memory.peak)


def normalize(response, wall, error, peak_mib):
    row = {"wall_s": round(wall, 3), "error": error, "peak_mib": peak_mib}
    if response is None:
        return row
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    produced = usage.get("completion_tokens") or 0
    prompt = usage.get("prompt_tokens") or 0
    prefill = (timings.get("prompt_ms") or 0.0) / 1000.0
    decode = (timings.get("predicted_ms") or 0.0) / 1000.0
    cached = timings.get("cache_n") or 0
    drafted = timings.get("draft_n") or 0
    accepted = timings.get("draft_n_accepted") or 0
    row.update(
        content=message.get("content") or "",
        finish=choice.get("finish_reason"),
        prompt_tokens=prompt, completion_tokens=produced, cached_tokens=cached,
        prefill_s=round(prefill, 4), decode_s=round(decode, 4), drafted=drafted, accepted=accepted,
        prefill_tps=round((prompt - cached) / prefill, 1) if prefill > 0 and prompt > cached else None,
        decode_tps=round((produced - 1) / decode, 2) if decode > 0 and produced > 1 else None,
        acceptance=round(accepted / drafted, 4) if drafted else None,
        tokens_per_round=round(produced / (produced - accepted), 3) if produced > accepted else None,
    )
    return row


class Sink:
    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        self.done.add(json.loads(line)["key"])
                    except (ValueError, KeyError):
                        pass

    def write(self, row):
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.done.add(row["key"])


def show(suite, row, extra=""):
    print("%s [%s] %s prompt=%s prefill=%ss (%s tok/s) out=%s decode=%s tok/s acc=%s tpr=%s peak=%sMiB %s%s" % (
        time.strftime("%H:%M:%SZ", time.gmtime()), suite, row["key"], row.get("prompt_tokens"), row.get("prefill_s"),
        row.get("prefill_tps"), row.get("completion_tokens"), row.get("decode_tps"), row.get("acceptance"),
        row.get("tokens_per_round"), row.get("peak_mib"), extra,
        (" ERROR " + row["error"][:300]) if row.get("error") else ""), flush=True)


def doc_messages(document, question):
    return [{"role": "user", "content": document},
            {"role": "assistant", "content": "I have read the whole document."},
            {"role": "user", "content": question}]


DEPTH_QUESTION = "In 3-4 sentences, summarize what happens in the final paragraphs of the document."
DEPTH_QUESTIONS = {
    "short": DEPTH_QUESTION,
    "long": "Retell the final part of the document in detail, paragraph by paragraph, in about 400 words.",
}


def suite_idle(client, sink, args, ctx):
    time.sleep(1.0)
    row = {"key": "idle", "suite": "idle", "used_mib": client.memory.current}
    sink.write(row)
    print("%s [idle] used=%sMiB" % (time.strftime("%H:%M:%SZ", time.gmtime()), row["used_mib"]), flush=True)


def suite_warmup(client, sink, args, ctx):
    corpus = ctx["corpus"]
    for size in (256, 4096):
        document = "Warm-up %s.\n\n%s" % (uuid.uuid4(), corpus.text(0, size))
        row = client.chat(doc_messages(document, DEPTH_QUESTION), 64)
        row.update(key="warmup/%d/%s" % (size, time.time()), suite="warmup")
        show("warmup", row)


def suite_depth(client, sink, args, ctx):
    corpus = ctx["corpus"]
    for depth in args.depths:
        for rep in range(args.reps):
            key = "depth/%d/%d" % (depth, rep)
            if key in sink.done:
                continue
            document = "Reading session %s. Here is a long document.\n\n%s" % (
                uuid.uuid4(), corpus.text(rep * 997, max(0, depth - 120)))
            row = client.chat(doc_messages(document, DEPTH_QUESTIONS[args.depth_question]), args.gen)
            row.update(key=key, suite="depth", depth=depth, rep=rep, question=args.depth_question)
            row.pop("content", None) if not args.keep_text else None
            sink.write(row)
            show("depth", row)
            if row.get("error"):
                return


ACCEPT_PROMPTS = [
    ("prose", "Write a detailed, well-structured essay of about 600 words on the history of the printing press "
              "and its effect on European science."),
    ("code", "Write a complete, well-commented Python module implementing a thread-safe LRU cache with TTL expiry, "
             "including unit tests using unittest."),
    ("reason", "A train leaves city A at 09:00 at 80 km/h; another leaves city B, 440 km away, at 10:00 at 100 "
               "km/h towards A. Work out step by step when and where they meet, then verify the answer."),
    ("ru", "Подробно, в 500 слов, объясни, как работает спекулятивное декодирование в больших языковых моделях "
           "и почему оно ускоряет генерацию."),
    ("list", "List 40 distinct everyday kitchen items, one per line, each with a one-sentence description."),
]


def suite_accept(client, sink, args, ctx):
    for name, prompt in ACCEPT_PROMPTS:
        key = "accept/" + name
        if key in sink.done:
            continue
        row = client.chat([{"role": "user", "content": "[%s] %s" % (uuid.uuid4().hex[:8], prompt)}], args.accept_gen)
        row.update(key=key, suite="accept", prompt_name=name)
        if not args.keep_text:
            row.pop("content", None)
        sink.write(row)
        show("accept", row)


def suite_conc(client, sink, args, ctx):
    corpus = ctx["corpus"]
    for level in args.conc:
        key = "conc/%d" % level
        if key in sink.done:
            continue
        rows = [None] * level
        client.memory.reset_peak()

        def run(index):
            if args.conc_depth > 0:
                document = "Session %s.\n\n%s" % (uuid.uuid4(), corpus.text(index * 4099, max(0, args.conc_depth - 120)))
                rows[index] = client.chat(doc_messages(document, DEPTH_QUESTION), args.gen)
            else:
                _, prompt = ACCEPT_PROMPTS[index % len(ACCEPT_PROMPTS)]
                rows[index] = client.chat([{"role": "user", "content": "[%s] %s" % (uuid.uuid4().hex[:8], prompt)}], args.gen)

        started = time.time()
        threads = [threading.Thread(target=run, args=(i,)) for i in range(level)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.time() - started
        produced = sum((r or {}).get("completion_tokens") or 0 for r in rows)
        errors = [r.get("error") for r in rows if r and r.get("error")]
        row = {"key": key, "suite": "conc", "level": level, "depth": args.conc_depth, "wall_s": round(wall, 3),
               "completion_tokens": produced, "aggregate_tps": round(produced / wall, 2) if wall > 0 else None,
               "per_request_decode_tps": [r.get("decode_tps") for r in rows if r],
               "peak_mib": client.memory.peak, "errors": errors}
        sink.write(row)
        print("%s [conc] level=%d depth=%d aggregate=%s tok/s per=%s peak=%sMiB errors=%d" % (
            time.strftime("%H:%M:%SZ", time.gmtime()), level, args.conc_depth, row["aggregate_tps"],
            row["per_request_decode_tps"], row["peak_mib"], len(errors)), flush=True)


CODE_WORDS = ["AMBER", "COBALT", "HARBOR", "LANTERN", "MAPLE", "ORBIT", "QUARTZ", "SIERRA", "TUNDRA", "VELVET"]


def suite_niah(client, sink, args, ctx):
    corpus = ctx["corpus"]
    for length in args.niah:
        key = "niah/%d" % length
        if key in sink.done:
            continue
        rng = random.Random(length)
        codes = ["%s-%04d" % (rng.choice(CODE_WORDS), rng.randint(1000, 9999)) for _ in range(3)]
        body = max(0, length - 250)
        cuts = [int(body * f) for f in (0.33, 0.66, 0.90)]
        parts, previous = [], 0
        for cut, code in zip(cuts, codes):
            parts.append(corpus.text(previous, cut - previous))
            parts.append("\nIMPORTANT SECRET VERIFICATION CODE: %s\n" % code)
            previous = cut
        parts.append(corpus.text(previous, body - previous))
        document = "Archival record %s. Secret verification codes appear inside it.\n\n%s" % (uuid.uuid4(), "".join(parts))
        question = ("List every secret verification code that appears in the document, in order of appearance, "
                    "separated by commas. Reply with only the codes.")
        row = client.chat(doc_messages(document, question), 96)
        text = row.get("content") or ""
        positions = [text.find(code) for code in codes]
        row.update(key=key, suite="niah", length=length, codes=codes, found=[c for c in codes if c in text],
                   order_ok=all(p >= 0 for p in positions) and positions == sorted(positions))
        sink.write(row)
        show("niah", row, "| found=%d/3 order=%s | %s" % (len(row["found"]), row["order_ok"], text[:60].replace("\n", " ")))


SUITES = {"idle": suite_idle, "warmup": suite_warmup, "depth": suite_depth, "accept": suite_accept,
          "conc": suite_conc, "niah": suite_niah}


def ints(text):
    return [int(v) for v in text.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suites", default="idle,warmup,depth,accept")
    parser.add_argument("--depths", type=ints, default=ints("1024,8192,32768,65536,131072,196608,261120"))
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--gen", type=int, default=256)
    parser.add_argument("--depth-question", choices=sorted(DEPTH_QUESTIONS), default="short")
    parser.add_argument("--accept-gen", type=int, default=512)
    parser.add_argument("--conc", type=ints, default=ints("1,2,4"))
    parser.add_argument("--conc-depth", type=int, default=8192)
    parser.add_argument("--niah", type=ints, default=ints("32768,131072"))
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus-dir", required=True, help="comma list of directories, walked for .txt")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--keep-text", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = []
    for directory in args.corpus_dir.split(","):
        for root, _, names in sorted(os.walk(directory)):
            files.extend(os.path.join(root, name) for name in sorted(names) if name.endswith(".txt"))
    memory = Memory(args.gpu_index)
    ctx = {"corpus": Corpus(args.tokenizer, files)}
    client = Client(args.base_url, args.model, args.timeout, memory)
    for suite in args.suites.split(","):
        sink = Sink(os.path.join(args.out, suite + ".jsonl"))
        SUITES[suite](client, sink, args, ctx)
    memory.stop = True


if __name__ == "__main__":
    sys.exit(main())
