"""Interleaved multi-conversation KV regression. Uses a supplied model, never downloads one.

Three server launches against the same model:
  baseline   no new flag, asserting today's behavior (a returning conversation misses)
  multi      --kvmem-conversations 4, asserting hits survive an interleave and needles stay isolated
  eviction   --kvmem-conversations 2, asserting LRU eviction bounds the store count

Needle isolation is the load-bearing assertion: attaching to the wrong host
store shows up as the other conversation's code in the answer, which no byte
count or cache counter would catch.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--mtp', action='store_true')
p.add_argument('--bytes-case', action='store_true',
               help='also run the byte-cap case, whose eviction point depends on model geometry')
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['KVMEM_TRACE'] = '1'  # the trace assertions below read the server log
checks = []
headers = {'Authorization': 'Bearer api-test-only', 'Content-Type': 'application/json'}

SYSTEM = 'Read the notes and answer with the code only.'
CHANNELS = {
    'A': ('alpha', 'ALPHA-731'),
    'B': ('beta', 'BETA-118'),
    'C': ('gamma', 'GAMMA-905'),
}
ALL_CODES = [code for _, code in CHANNELS.values()]


def check(name, ok):
    checks.append({'name': name, 'pass': bool(ok)})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)


def notes(channel):
    """Long deterministic body, distinct per channel, with the needle at the end."""
    name, code = CHANNELS[channel]
    # 150 lines is a few thousand tokens: past 1024 by a wide margin so a cache
    # hit is unambiguous, and well inside the 16K context two turns need.
    lines = [f'Channel {name} note {i:03d}: routing entry {i} is stable and requires no action.'
             for i in range(150)]
    lines.append(f'The {name} code is {code}.')
    lines.append(f'What is the {name} code? Answer with the code only.')
    return '\n'.join(lines)


def followup(channel):
    name, _ = CHANNELS[channel]
    return f'Repeat the {name} code exactly, nothing else.'


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(base, path, data=None, auth=True, method=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers=headers if auth else {}, method=method)
    try:
        with opener.open(req, timeout=600) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def run_case(label, extra_args, script, probe=None):
    """Launch one server, run `script` (a list of channel names) and collect everything.

    `probe` runs after the scripted turns, against the live server, and its
    return value comes back as result['probe'].
    """
    port = free_port()
    base = f'http://127.0.0.1:{port}'
    args = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(port),
            '-c', '16384', '-n', '128', '--api-key', 'api-test-only', '--threads-http', '4',
            '--reasoning-effort', 'none', '--temp', '0', '--no-webui', '--kv-dtype', 'q8_0',
            '-fa', 'on', '--kvmem-budget', '4096', '--kvmem-gen-reserve', '2048']
    args += ['--spec-type', 'draft-mtp' if a.mtp else 'none']
    args += extra_args
    log_path = out / f'{label}-server.log'
    log = log_path.open('wb')
    proc = subprocess.Popen(args, stdout=log, stderr=log, env=env,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    history = {name: [{'role': 'system', 'content': SYSTEM}] for name in CHANNELS}
    turns = []
    try:
        deadline = time.monotonic() + 240
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'server exited; see {log_path.name}')
            try:
                if request(base, '/health', auth=False)[0] == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('startup')
            time.sleep(1)
        status, body, _ = request(base, '/props')
        props = json.loads(body)
        for index, channel in enumerate(script):
            messages = history[channel]
            first = len(messages) == 1
            messages.append({'role': 'user',
                             'content': notes(channel) if first else followup(channel)})
            status, raw, _ = request(base, '/v1/chat/completions', {
                'messages': messages, 'max_tokens': 32, 'temperature': 0,
                'reasoning_effort': 'none'})
            check(f'{label} turn {index} {channel} succeeds', status == 200)
            response = json.loads(raw)
            text = response['choices'][0]['message']['content'] or ''
            messages.append({'role': 'assistant', 'content': text})
            turns.append({'channel': channel, 'text': text,
                          'usage': response['usage'], 'timings': response['timings']})
        probed = probe(base, history) if probe else None
        slot = json.loads(request(base, '/slots')[1])[0]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
    text = log_path.read_text(encoding='utf-8', errors='replace')
    keep = ('store_', 'KVMEM_STORE_', 'multimodal_reset',
            'KVMEM_PREFILL_DECISION', 'cache_commit')
    trace = [line for line in text.splitlines() if any(k in line for k in keep)]
    (out / f'{label}-trace.txt').write_text('\n'.join(trace) + '\n', encoding='utf-8')
    (out / f'{label}-responses.json').write_text(json.dumps(turns, indent=2), encoding='utf-8')
    return {'turns': turns, 'log': text, 'trace': trace, 'props': props, 'slot': slot,
            'probe': probed}


def hit(turn):
    # base.row of the checkpoint the turn resumed from (kvmem-multimodal-server.h:362),
    # which is the resumed prefix and not the token LCP: a fork reports 0.
    return turn['usage']['prompt_cache_hit_tokens']


def answers(case, label):
    for index, turn in enumerate(case['turns']):
        name, code = CHANNELS[turn['channel']]
        others = [c for c in ALL_CODES if c != code]
        check(f'{label} turn {index} {turn["channel"]} keeps its own needle',
              code in turn['text'] and not any(o in turn['text'] for o in others))


def accounting(case, label):
    for index, turn in enumerate(case['turns']):
        timings, usage = turn['timings'], turn['usage']
        check(f'{label} turn {index} timings cache count matches usage',
              timings['cache_n'] == usage['prompt_cache_hit_tokens'])
        check(f'{label} turn {index} prompt accounting holds',
              timings['prompt_n'] + timings['cache_n'] == usage['prompt_tokens'])


def store_counts(case):
    return [int(m) for m in re.findall(r'store_select .*?stores=(\d+)', case['log'])]


def pool_counts(case):
    """Live host stores as the adapter counts them, from its own trace.

    The server's table and the adapter's pool must hold the same set. They
    diverge exactly when a table row is dropped without its host store being
    destroyed, which is a conversation's whole KV leaked with nothing able to
    reach or release it, so the cap reads as held while host RAM keeps growing.
    """
    return [int(m) for m in re.findall(r'KVMEM_STORE_\w+ .*?n_stores=(\d+)', case['log'])]


try:
    # Case 1: today's behavior. Every turn of a returning conversation is a
    # full miss because the previous conversation's store was destroyed.
    baseline = run_case('baseline', [], ['A', 'B', 'A', 'B'])
    check('baseline prompts are long enough to be unambiguous',
          baseline['turns'][0]['usage']['prompt_tokens'] > 1024)
    for index, turn in enumerate(baseline['turns']):
        check(f'baseline turn {index} is a miss today', hit(turn) == 0)
    check('baseline emits no store traces', 'KVMEM_TRACE store_' not in baseline['log'])
    check('baseline single slot', baseline['props']['total_slots'] == 1)
    check('baseline props has no conversation capability',
          'conversations' not in baseline['props'].get('kvmem', {}))
    check('baseline slot has no kvmem object', 'kvmem' not in baseline['slot'])
    accounting(baseline, 'baseline')
    answers(baseline, 'baseline')

    # Case 2: the feature. A and B keep their KV in host RAM across each
    # other's turns, so the second and third visits resume their own prefix.
    multi = run_case('multi', ['--kvmem-conversations', '4'], ['A', 'B', 'A', 'B', 'A'])
    a1, b1, a2, b2, a3 = multi['turns']
    check('multi conversations armed', 'store_select' in multi['log'])
    check('multi A1 is cold', hit(a1) == 0)
    check('multi B1 does not attach to A', hit(b1) == 0)
    check('multi A2 hits after B was served',
          hit(a2) >= 1024 and hit(a2) >= 0.9 * a1['usage']['prompt_tokens'])
    check('multi B2 hits after A2 was served',
          hit(b2) >= 0.9 * b1['usage']['prompt_tokens'])
    check('multi A3 still resident', hit(a3) >= hit(a2))
    check('multi extends the two live conversations three times',
          multi['log'].count('store_select action=extend') == 3)
    check('multi store count stays within the cap',
          store_counts(multi) and max(store_counts(multi)) <= 4)
    check('multi slot reports the conversations',
          multi['slot']['kvmem']['conversations']['max'] == 4 and
          multi['slot']['kvmem']['conversations']['count'] >= 2 and
          multi['slot']['kvmem']['conversations']['switches'] >= 2)
    check('multi props advertises the capability',
          multi['props']['kvmem']['conversations'] == 4)
    check('multi still one slot', multi['props']['total_slots'] == 1)
    # Every switch reported success and no rollback ran. A refusal names the
    # conv_active/adapter-active divergence, and store_wiped on a failed switch
    # means the outgoing store was destroyed before an attach threw; either one
    # is a healthy run turning into a degraded one.
    check('multi switches cleanly',
          'store switch refused' not in multi['log'] and
          'store switch failed' not in multi['log'] and
          'reason=switch_failed' not in multi['log'])
    accounting(multi, 'multi')
    answers(multi, 'multi')

    # Case 3: eviction. A third conversation cannot fit under a cap of two, so
    # the least recently used store goes; the survivor still hits and the
    # evicted conversation comes back as an ordinary miss.
    evict = run_case('eviction', ['--kvmem-conversations', '2'],
                     ['A', 'B', 'A', 'B', 'C', 'B', 'A'])
    e_a1, e_b1, e_a2, e_b2, e_c1, e_b3, e_a3 = evict['turns']
    check('eviction cold starts are cold', hit(e_a1) == 0 and hit(e_b1) == 0)
    check('eviction A2 hits before the cap bites', hit(e_a2) > 0)
    check('eviction B2 hits before the cap bites', hit(e_b2) > 0)
    check('eviction C1 is cold', hit(e_c1) == 0)
    check('eviction survivor still hits after C arrived', hit(e_b3) > 0)
    check('eviction victim comes back as a miss', hit(e_a3) == 0)
    check('eviction is least recently used',
          'store_evict' in evict['log'] and 'reason=lru' in evict['log'])
    check('eviction store count is bounded',
          store_counts(evict) and max(store_counts(evict)) <= 2)
    check('eviction leaves no orphaned host store',
          pool_counts(evict) and max(pool_counts(evict)) <= 2 and
          'cannot be released' not in evict['log'] and
          'refused destroy' not in evict['log'])
    accounting(evict, 'eviction')
    answers(evict, 'eviction')

    # Case 4: a request that answers 400 must not move conversation state. The
    # store switch runs below every validation of the request, so a rejected
    # request parks nothing, creates no store and evicts no victim.
    #
    # The trigger is the n_ctx check in handle_chat, "prompt + max_tokens
    # exceeds n_ctx": the prompt below is hundreds of KB against the 16K
    # context these servers are launched with, so it rejects whatever the
    # tokenizer does with it, and it needs no projector. It is also the last
    # 400 this server can be made to answer: the three checks below it need an
    # image group, a chat template with no thinking markers, and a grammar the
    # draft model cannot take, in that order.
    #
    # The cap is 2 with A and B resident, so the new conversation this prompt
    # describes could only be attached by evicting A, the least recently used.
    # A hitting afterwards is the assertion with teeth: it says no victim was
    # destroyed. The counters cover the rest, that nothing was parked, created
    # or forked.
    def reject_probe(base, history):
        def counters():
            return json.loads(request(base, '/slots')[1])[0]['kvmem']['conversations']
        before = counters()
        oversized = '\n'.join(
            f'Filler line {i:05d}: present only to push this prompt past the context window.'
            for i in range(4000))
        status, raw, _ = request(base, '/v1/chat/completions', {
            'messages': [{'role': 'user', 'content': oversized}],
            'max_tokens': 32, 'temperature': 0, 'reasoning_effort': 'none'})
        after = counters()
        messages = history['A'] + [{'role': 'user', 'content': followup('A')}]
        status_a, raw_a, _ = request(base, '/v1/chat/completions', {
            'messages': messages, 'max_tokens': 32, 'temperature': 0,
            'reasoning_effort': 'none'})
        survivor = json.loads(raw_a) if status_a == 200 else {}
        return {'status': status, 'error': raw.decode('utf-8', 'replace')[:200],
                'before': before, 'after': after,
                'survivor_status': status_a,
                'survivor_hit': survivor.get('usage', {}).get('prompt_cache_hit_tokens', 0),
                'survivor_text': (survivor.get('choices') or [{}])[0].get('message', {}).get('content') or ''}

    reject = run_case('reject', ['--kvmem-conversations', '2'],
                      ['A', 'B'], probe=reject_probe)
    probe = reject['probe']
    check('reject answers 400', probe['status'] == 400)
    check('reject is the n_ctx rejection', 'n_ctx' in probe['error'])
    for field in ('active', 'switches', 'count', 'forks', 'extends', 'evictions',
                  'resets', 'refusals'):
        check(f'reject leaves conversations.{field} alone',
              probe['after'][field] == probe['before'][field])
    # Two scripted turns plus the surviving conversation's turn below. The
    # rejected request contributes none: conversation_begin_request emits one
    # store_select per request it is called for, and it is not called for this
    # one.
    check('reject selects no conversation',
          reject['log'].count('store_select') == 3)
    check('reject victim survives', probe['survivor_status'] == 200 and probe['survivor_hit'] > 0)
    check('reject victim keeps its own needle',
          CHANNELS['A'][1] in probe['survivor_text'] and
          not any(o in probe['survivor_text'] for o in ALL_CODES if o != CHANNELS['A'][1]))

    if a.bytes_case:
        # Qualitative on purpose: the per-conversation byte size depends on
        # model geometry, so asserting an exact eviction point would flake.
        bytes_case = run_case('bytes', ['--kvmem-conversations', '4',
                                        '--kvmem-conversations-gb', '0.05'],
                              ['A', 'B', 'A', 'B'])
        check('byte cap evicts', 'reason=bytes' in bytes_case['log'])
        # The byte cap is the path that used to name the attached conversation
        # as a victim, so this is where an orphaned host store would show up.
        check('byte cap leaves no orphaned host store',
              pool_counts(bytes_case) and max(pool_counts(bytes_case)) <= 4 and
              'cannot be released' not in bytes_case['log'])
        answers(bytes_case, 'bytes')
finally:
    (out / 'results.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
