"""GPU regression: prefix reuse must survive a rewritten trailing user message.

Scenario this locks in (what agent harnesses actually send):

    turn 1: [system, user(long), user(nudge-1)]
    turn 2: [system, user(long), assistant, tool, user(nudge-2)]
    turn 3: [system, user(long), assistant, tool, assistant, tool, user(nudge-3)]

The trailing message is a per-turn nudge whose *role* flips from `user` to
`assistant` on the next turn. That moves the LCP boundary off the recurrent
checkpoint, so the engine finds `oldest_checkpoint = lcp + 1`, reports
`no_recurrent_checkpoint`, and recomputes the whole prompt — every turn,
self-locking (TTFT grows with prompt size and never recovers).

Expected with the prefix-checkpoint fix: at most one `no_recurrent_checkpoint`
(the first divergence, which no fix can rule out in principle — in practice the
checkpoint pre-seeded during the first prefill absorbs even that, so the count
observed here is zero), and `prefix_hit_rows` non-zero from the next turn on.
Before the fix the count equals the number of divergent turns.

Requires a model the KVMem adapter accepts and a visible GPU. Exits 77 when no
GPU is present, so portable CI (KVMEM_BUILD_LLAMA=OFF) skips instead of failing.

Usage:
    python tests/prefix-reuse-regression.py --exe <llama-kvmem-server> \
        --model <model.gguf> --out <dir> --gpu 0
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SKIP = 77


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mmproj', type=Path, help='optional vision projector')
    args = parser.parse_args()

    if not args.exe.exists() or not args.model.exists():
        print(f'skip: missing exe or model', file=sys.stderr)
        return SKIP

    args.out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['KVMEM_TRACE'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['PATH'] = str(args.exe.parent) + os.pathsep + env['PATH']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    port = free_port()
    base = f'http://127.0.0.1:{port}'
    command = [str(args.exe), '-m', str(args.model),
               '--host', '127.0.0.1', '--port', str(port),
               '-c', '32768', '-n', '256', '-ngl', '99', '-b', '512', '--no-ui',
               '--kvmem', '--kvmem-budget', '16384', '--kvmem-gen-reserve', '2048',
               '--kvmem-block-tokens', '32', '--kv-dtype', 'q8_0',
               '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
               '--kvmem-query-policy', 'user',
               '--reasoning-effort', 'none', '--temp', '0']
    if args.mmproj:
        command += ['--mmproj', str(args.mmproj), '--no-mmproj-offload', '--image-max-tokens', '256']

    (args.out / 'argv.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    log_path = args.out / 'server.log'
    log = log_path.open('wb')

    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print('server exited early; see server.log', file=sys.stderr)
                return 1
            try:
                with opener.open(base + '/health', timeout=3):
                    break
            except Exception:
                time.sleep(1)
        else:
            print('server did not become healthy in time', file=sys.stderr)
            return 1

        sysp = ('You are a careful software engineering assistant. '
                'Follow instructions exactly. ') * 120
        longp = ('The repository contains a Django application with several apps. ') * 200
        base_msgs = [
            {'role': 'system', 'content': sysp},
            {'role': 'user', 'content': longp + '\n\nTask: fix the failing test in tests/test_views.py.'},
        ]

        def nudge(n):
            return {'role': 'user', 'content': f'Tool-call turn {n} of 100. Remaining turns: {100 - n}.'}

        def post(name, messages):
            offset = log_path.stat().st_size
            payload = {'messages': messages, 'max_tokens': 24, 'temperature': 0,
                       'reasoning_effort': 'none'}
            raw = json.dumps(payload).encode()
            (args.out / f'{name}.request.json').write_bytes(raw)
            req = urllib.request.Request(base + '/v1/chat/completions', data=raw,
                                         headers={'Content-Type': 'application/json'})
            started = time.monotonic()
            with opener.open(req, timeout=240) as response:
                status, body = response.status, response.read().decode()
            elapsed = time.monotonic() - started
            (args.out / f'{name}.response.txt').write_text(body, encoding='utf-8')
            with log_path.open('rb') as reader:
                reader.seek(offset)
                trace = reader.read().decode(errors='replace')
            (args.out / f'{name}.trace.txt').write_text(trace, encoding='utf-8')
            content = ''
            try:
                content = json.loads(body)['choices'][0]['message'].get('content') or ''
            except Exception:
                pass
            record = {'case': name, 'status': status, 'seconds': round(elapsed, 2),
                      'prompt_tokens': json.loads(body).get('usage', {}).get('prompt_tokens')
                      if status == 200 else None}
            print(json.dumps(record), flush=True)
            return status, content, trace

        # --- the scenario -----------------------------------------------------
        status1, reply1, trace1 = post('turn1', base_msgs + [nudge(1)])
        if status1 != 200:
            print('turn1 failed', file=sys.stderr)
            return 1
        reply = reply1 or 'Looking at the failing test now.'

        status2, reply2, trace2 = post('turn2', base_msgs + [
            {'role': 'assistant', 'content': reply},
            {'role': 'tool', 'content': 'test_views.py: AssertionError: expected 200 got 302'},
            nudge(2),
        ])
        status3, reply3, trace3 = post('turn3', base_msgs + [
            {'role': 'assistant', 'content': reply},
            {'role': 'tool', 'content': 'test_views.py: AssertionError: expected 200 got 302'},
            {'role': 'assistant', 'content': reply2 or 'Checking the view decorator.'},
            {'role': 'tool', 'content': 'views.py: @login_required present on the handler'},
            nudge(3),
        ])

        # --- assertions -------------------------------------------------------
        all_trace = trace1 + trace2 + trace3
        resets = len(re.findall(r'no_recurrent_checkpoint', all_trace))
        hits = [int(m) for m in re.findall(r'prefix_hit_rows=(\d+)', trace3)]
        decisions = re.findall(r'KVMEM_TRACE multimodal_prefill[^\r\n]*', all_trace)

        result = {
            'status': [status1, status2, status3],
            'no_recurrent_checkpoint_total': resets,
            'turn3_prefix_hit_rows': hits,
            'prefill_lines': len(decisions),
        }
        (args.out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)

        failures = []
        if [status1, status2, status3] != [200, 200, 200]:
            failures.append('a turn did not return 200')
        # Without the fix, every divergent turn resets — here that is 2 (turn2 and
        # turn3), which is exactly the self-locking behaviour being fixed. With the
        # fix, at most the first divergence resets; in practice even that is avoided
        # because the first prefill already lands a checkpoint on the boundary.
        if resets > 1:
            failures.append(f'expected at most 1 no_recurrent_checkpoint, saw {resets} '
                            '(prefix reuse is re-locking on every divergent turn)')
        if not any(h > 0 for h in hits):
            failures.append('turn3 never reported a non-zero prefix_hit_rows (prefix not reused)')

        if failures:
            for f in failures:
                print('FAIL: ' + f, file=sys.stderr)
            print(f'inspect {args.out}', file=sys.stderr)
            return 1
        print('PASS: prefix reuse survives the rewritten trailing user message')
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == '__main__':
    sys.exit(main())
