"""Compare OpenAI against a baseline binary and exercise opt-in Messages routes.

Uses supplied binaries/model, ephemeral loopback ports, and only terminates servers it starts.
Run with the official anthropic SDK installed to include SDK interoperability checks.
"""
import argparse
import base64
import concurrent.futures
import contextlib
import http.client
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--server', required=True)
parser.add_argument('--baseline', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--gpu-layers', default='0')
parser.add_argument('--device', default='none')
parser.add_argument('--mtp', action='store_true')
parser.add_argument('--mtp-state', choices=['snapshots', 'auto', 'replay'], default='snapshots')
parser.add_argument('--kvmem', action='store_true')
parser.add_argument('--require-sdk', action='store_true')
parser.add_argument('--benchmark', action='store_true', help='also measure five 128-token OpenAI samples after warmup')
parser.add_argument('--mmproj', help='optional projector for OpenAI vision regression')
parser.add_argument('--image', help='optional image for OpenAI vision regression (requires --mmproj)')
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
checks = []
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(name, condition):
    checks.append({'name': name, 'pass': bool(condition)})
    print(('PASS ' if condition else 'FAIL ') + name, flush=True)
    if not condition:
        raise AssertionError(name)


def request(port, path, data=None, *, raw=None, key=True, anthropic=False, version=True, method=None):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['X-Api-Key' if anthropic else 'Authorization'] = 'test-only' if anthropic else 'Bearer test-only'
    if anthropic and version:
        headers['anthropic-version'] = '2023-06-01'
    payload = raw if raw is not None else json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(f'http://127.0.0.1:{port}' + path, payload, headers, method=method)
    try:
        with opener.open(req, timeout=180) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


@contextlib.contextmanager
def server(binary, label, enabled=False):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [binary, '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
               '--alias', 'local-test', '--api-key', 'test-only', '--no-ui', '-c', '8192', '-n', '256',
               '-ngl', args.gpu_layers, '--device', args.device, '--threads', '4', '--threads-http', '4',
               '--reasoning-effort', 'none', '--temp', '0', '--presence-penalty', '0', '--seed', '1234',
               '--spec-type', 'draft-mtp' if args.mtp else 'none']
    command += ['--kvmem', '--kvmem-budget', '4096', '--kvmem-gen-reserve', '2048', '--kv-dtype', 'q8_0'] if args.kvmem else ['--no-kvmem']
    if enabled:
        command += ['--anthropic']
    if args.mtp:
        command += ['--kvmem-mtp-state', args.mtp_state]
    if args.mmproj:
        command += ['--mmproj', args.mmproj, '--image-max-tokens', '512']
    with (output / (label + '.log')).open('wb') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 240
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'{label} exited with {proc.returncode}; see its log')
                try:
                    if request(port, '/health', key=False)[0] == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(label + ' startup')
                time.sleep(.15)
            if args.mtp:
                check(label + ' MTP is actually active', json.loads(request(port, '/slots')[1])[0]['speculative'])
            yield port
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def normalized(value):
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {'id', 'tool_call_id', 'created', 'system_fingerprint'}:
                result[key] = '<dynamic>'
            elif key == 'timings':
                result[key] = {k: item[k] if k in {'cache_n', 'prompt_n', 'predicted_n'} else '<time>' for k in item}
            else:
                result[key] = normalized(item)
        return result
    return value


def events(raw, named=False):
    records = []
    name = None
    for line in raw.decode().splitlines():
        if line.startswith('event: '):
            name = line[7:]
        if line.startswith('data: '):
            data = line[6:]
            value = data if data == '[DONE]' else json.loads(data)
            if named:
                check('named SSE event matches JSON type', isinstance(value, dict) and name == value['type'])
            records.append(value)
            name = None
    return records


def oa_body(stream=False):
    return {'messages': [{'role': 'user', 'content': 'Reply with exactly the word hello.'}],
            'max_tokens': 16, 'temperature': 0, 'stream': stream, 'cache_reset': True}


def legacy_snapshot(port):
    snapshot = {}
    for path in ['/v1/models', '/health', '/props']:
        code, raw, _ = request(port, path)
        snapshot[path] = [code, json.loads(raw)]
    cases = [('/v1/chat/completions', oa_body()), ('/chat/completions', oa_body(True)),
             ('/v1/chat/completions', {'messages': []}),
             ('/v1/chat/completions', {**oa_body(), 'max_tokens': None})]
    tool = {'type': 'function', 'function': {'name': 'echo', 'description': 'Return the value',
            'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']}}}
    for stream in (False, True):
        cases.append(('/v1/chat/completions', {**oa_body(stream), 'max_tokens': 80, 'tools': [tool], 'tool_choice': 'required'}))
        cases.append(('/v1/chat/completions', {**oa_body(stream), 'max_tokens': 96,
                     'enable_thinking': True, 'reasoning_effort': 'low', 'reasoning_budget_tokens': 16}))
    if args.image:
        if not args.mmproj:
            raise ValueError('--image requires --mmproj')
        image_url = 'data:image/png;base64,' + base64.b64encode(Path(args.image).read_bytes()).decode()
        for stream in (False, True):
            cases.append(('/v1/chat/completions', {**oa_body(stream), 'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Read the four digits in the image. Output only the digits.'},
                {'type': 'image_url', 'image_url': {'url': image_url}}]}]}))
    for index, (path, body) in enumerate(cases):
        code, raw, headers = request(port, path, body)
        check(f'legacy case {index} status', code == (400 if index in (2, 3) else 200))
        content_type = headers.get('Content-Type', headers.get('content-type', ''))
        data = events(raw) if 'text/event-stream' in content_type else json.loads(raw)
        if body.get('enable_thinking'):
            reasoning = (''.join(chunk['choices'][0].get('delta', {}).get('reasoning_content', '') or ''
                                 for chunk in data if isinstance(chunk, dict) and chunk.get('choices'))
                         if body['stream'] else data['choices'][0]['message'].get('reasoning_content', ''))
            check('OpenAI thinking really emits reasoning content', bool(reasoning))
        snapshot[str(index)] = [code, content_type, normalized(data)]
    for key in (False, True):
        code, raw, _ = request(port, '/v1/chat/completions', raw=b'not-json', key=key)
        snapshot['malformed-' + str(key)] = [code, json.loads(raw)]
    # Repeat a prompt without reset so cache accounting is compared as well.
    warm = oa_body(); warm.pop('cache_reset')
    for index in range(2):
        code, raw, _ = request(port, '/v1/chat/completions', warm)
        snapshot['warm-' + str(index)] = [code, normalized(json.loads(raw))]
    return normalized(snapshot)


def anth_body(stream=False):
    return {'model': 'local-test', 'max_tokens': 32, 'stream': stream,
            'messages': [{'role': 'user', 'content': 'Reply with exactly the word hello.'}], 'temperature': 0}


def benchmark(port):
    samples = []
    for index in range(7):
        body = {**oa_body(), 'max_tokens': 128, 'messages': [{'role': 'user',
                'content': 'Write numbers from 1 through 1000, separated by spaces. Continue without explanation.'}]}
        started = time.monotonic()
        code, raw, _ = request(port, '/v1/chat/completions', body)
        elapsed = time.monotonic() - started
        result = json.loads(raw)
        check('benchmark generated 128 tokens', code == 200 and result['usage']['completion_tokens'] == 128)
        if index >= 2:
            samples.append({'wall_ms': elapsed * 1000, **result['timings']})
    return {'samples': samples, 'median_decode_ms': statistics.median(item['predicted_ms'] for item in samples),
            'median_wall_ms': statistics.median(item['wall_ms'] for item in samples)}


def anthropic_checks(port):
    code, raw, _ = request(port, '/v1/messages', raw=b'not-json', key=False, anthropic=True)
    check('Anthropic authentication before parsing', code == 401 and json.loads(raw)['type'] == 'error')
    code, raw, _ = request(port, '/v1/messages', anth_body(), anthropic=True, version=False)
    check('Anthropic version validation', code == 400 and json.loads(raw)['error']['type'] == 'invalid_request_error')
    before = json.loads(request(port, '/slots')[1])
    count_body = anth_body(); count_body.pop('max_tokens'); count_body.pop('stream')
    code, raw, _ = request(port, '/v1/messages/count_tokens', count_body, anthropic=True)
    count = json.loads(raw)['input_tokens']
    check('token count returns without generating', code == 200 and count > 0 and before == json.loads(request(port, '/slots')[1]))
    code, raw, _ = request(port, '/v1/messages', anth_body(), anthropic=True)
    message = json.loads(raw)
    check('Anthropic text response', code == 200 and message['type'] == 'message' and message['content'][0]['type'] == 'text')
    check('count agrees with generation usage', message['usage']['input_tokens'] == count)
    code, raw, _ = request(port, '/v1/messages', anth_body(True), anthropic=True)
    stream = events(raw, True)
    check('Anthropic SSE lifecycle', code == 200 and stream[0]['type'] == 'message_start' and stream[-1]['type'] == 'message_stop')
    check('Anthropic stream has no OpenAI frames', '[DONE]' not in raw.decode() and all('choices' not in event for event in stream))
    text = ''.join(event['delta']['text'] for event in stream if event['type'] == 'content_block_delta' and event['delta']['type'] == 'text_delta')
    check('Anthropic streamed text matches JSON', text == ''.join(block['text'] for block in message['content'] if block['type'] == 'text'))
    # Request a known substring as a stop, including a match inside a generated token.
    stop_body = anth_body(True); stop_body['stop_sequences'] = ['ello']
    code, raw, _ = request(port, '/v1/messages', stop_body, anthropic=True)
    stopped = events(raw, True)
    text = ''.join(event['delta'].get('text', '') for event in stopped if event['type'] == 'content_block_delta')
    terminal = next(event for event in stopped if event['type'] == 'message_delta')
    check('stop substring is not leaked', code == 200 and 'ello' not in text and terminal['delta'] == {'stop_reason': 'stop_sequence', 'stop_sequence': 'ello'})
    tool_body = anth_body()
    tool_body['max_tokens'] = 128
    tool_body['messages'][0]['content'] = 'Call echo with value hello.'
    tool_body['tools'] = [{'name': 'echo', 'description': 'Echo a value', 'input_schema': {
        'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']}}]
    tool_body['tool_choice'] = {'type': 'tool', 'name': 'echo', 'disable_parallel_tool_use': True}
    for stream_tools in (False, True):
        tool_body['stream'] = stream_tools
        code, raw, _ = request(port, '/v1/messages', tool_body, anthropic=True)
        if stream_tools:
            records = events(raw, True)
            starts = [item['content_block'] for item in records if item['type'] == 'content_block_start' and item['content_block']['type'] == 'tool_use']
            encoded = ''.join(item['delta']['partial_json'] for item in records if item['type'] == 'content_block_delta' and item['delta']['type'] == 'input_json_delta')
            check('streamed tool arguments decode to an object', code == 200 and len(starts) == 1 and isinstance(json.loads(encoded), dict))
            tool_call = {**starts[0], 'input': json.loads(encoded)}
        else:
            result = json.loads(raw)
            check('Anthropic named tool is enforced', code == 200 and result.get('stop_reason') == 'tool_use')
            tool_call = next(block for block in result['content'] if block['type'] == 'tool_use')
        check('tool name and stable ID', tool_call['name'] == 'echo' and tool_call['id'].startswith('toolu_'))
        followup = {**tool_body, 'stream': False, 'tool_choice': {'type': 'none'}, 'messages': [
            tool_body['messages'][0], {'role': 'assistant', 'content': [tool_call]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': tool_call['id'], 'content': 'hello'},
                                       {'type': 'text', 'text': 'Reply with the returned value.'}]}]}
        code, raw, _ = request(port, '/v1/messages', followup, anthropic=True)
        check('tool result round trip', code == 200 and json.loads(raw)['stop_reason'] in ('end_turn', 'max_tokens'))
    for patch in [{'thinking': {'type': 'enabled'}}, {'model': 'unknown'}, {'messages': []}, {'max_tokens': 0}]:
        code, raw, _ = request(port, '/v1/messages', {**anth_body(), **patch}, anthropic=True)
        check('unsupported/invalid request has Anthropic error', code in (400, 404) and json.loads(raw)['type'] == 'error')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(request, port, '/v1/messages', anth_body(), anthropic=True),
                pool.submit(request, port, '/v1/chat/completions', oa_body(True)),
                pool.submit(request, port, '/v1/messages', anth_body(True), anthropic=True)]
        replies = [job.result(timeout=180) for job in jobs]
    first = json.loads(replies[0][1])
    third = events(replies[2][1], True)
    check('concurrent mixed requests retain protocol formats', all(reply[0] == 200 for reply in replies)
          and first['type'] == 'message' and events(replies[1][1])[-1] == '[DONE]'
          and third[-1]['type'] == 'message_stop')
    check('concurrent Messages have independent IDs', first['id'] != third[0]['message']['id'])
    # Close immediately after message_start, while prefill may still be running.
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
    long_body = anth_body(True)
    long_body['messages'][0]['content'] = 'hello ' * 1000 + 'Reply with hello.'
    conn.request('POST', '/v1/messages', json.dumps(long_body), headers={
        'X-Api-Key': 'test-only', 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'})
    response = conn.getresponse()
    check('disconnect test starts streaming', response.status == 200 and response.readline().startswith(b'event: message_start'))
    response.close(); conn.close()
    deadline = time.monotonic() + 30
    while json.loads(request(port, '/slots')[1])[0]['is_processing'] and time.monotonic() < deadline:
        time.sleep(.1)
    check('disconnect releases model slot', not json.loads(request(port, '/slots')[1])[0]['is_processing'])
    code, raw, _ = request(port, '/v1/chat/completions', oa_body())
    check('OpenAI works after Anthropic error/disconnect', code == 200 and bool(json.loads(raw)['choices']))
    try:
        import anthropic
    except ImportError:
        if args.require_sdk:
            raise
        print('SKIP official SDK (install anthropic or use --require-sdk)', flush=True)
    else:
        with anthropic.Anthropic(api_key='test-only', base_url=f'http://127.0.0.1:{port}',
                max_retries=0, http_client=anthropic.DefaultHttpxClient(trust_env=False, timeout=180)) as client:
            payload = anth_body(); payload.pop('stream'); payload.pop('temperature')
            final = client.messages.create(**payload)
            check('official SDK nonstreaming', final.type == 'message' and bool(final.content))
            with client.messages.stream(**payload) as stream:
                final = stream.get_final_message()
                check('official SDK stream accumulation', final.type == 'message' and bool(final.content))
            count = client.messages.count_tokens(model='local-test', messages=payload['messages'])
            check('official SDK count_tokens', count.input_tokens > 0)
            sdk_tools = {**tool_body}; sdk_tools.pop('stream'); sdk_tools.pop('temperature')
            with client.messages.stream(**sdk_tools) as stream:
                final = stream.get_final_message()
                check('official SDK streamed tools', final.stop_reason == 'tool_use' and any(block.type == 'tool_use' and isinstance(block.input, dict) for block in final.content))


try:
    snapshots = {}
    measurements = {}
    with server(args.baseline, 'baseline') as port:
        snapshots['baseline'] = legacy_snapshot(port)
        (output / 'snapshot-baseline.json').write_text(json.dumps(snapshots['baseline'], indent=2), encoding='utf-8')
        if args.benchmark:
            measurements['baseline'] = benchmark(port)
    with server(args.server, 'candidate-off') as port:
        check('Messages disabled by default', request(port, '/v1/messages', anth_body(), anthropic=True)[0] == 404)
        snapshots['off'] = legacy_snapshot(port)
        if args.benchmark:
            measurements['off'] = benchmark(port)
    with server(args.server, 'candidate-on', True) as port:
        snapshots['on'] = legacy_snapshot(port)
        if args.benchmark:
            measurements['on'] = benchmark(port)
        for mode in ('off', 'on'):
            (output / ('snapshot-' + mode + '.json')).write_text(json.dumps(snapshots[mode], indent=2), encoding='utf-8')
            check('OpenAI matches baseline with Anthropic ' + mode, snapshots[mode] == snapshots['baseline'])
        anthropic_checks(port)
        snapshots['after'] = legacy_snapshot(port)
        (output / 'snapshot-after.json').write_text(json.dumps(snapshots['after'], indent=2), encoding='utf-8')
        check('OpenAI matches baseline after mixed protocol traffic', snapshots['after'] == snapshots['baseline'])
    if measurements:
        for mode in ('off', 'on'):
            measurements[mode]['decode_change_percent'] = 100 * (measurements[mode]['median_decode_ms'] / measurements['baseline']['median_decode_ms'] - 1)
        (output / 'benchmark.json').write_text(json.dumps(measurements, indent=2), encoding='utf-8')
finally:
    (output / 'results.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
