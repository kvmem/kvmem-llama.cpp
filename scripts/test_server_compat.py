"""CLI validation and optional real-model HTTP/auth/inference regression tests.

No downloads. Uses an ephemeral loopback port and terminates only its own server.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', required=True)
p.add_argument('--model')
p.add_argument('--mmproj')
p.add_argument('--image', help='optional image containing the digits 6037')
p.add_argument('--mtp', action='store_true')
p.add_argument('--output', default='server-compat-results')
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["KVMEM_TRACE"] = "1"
checks = []

def check(name, ok):
    checks.append({'name': name, 'pass': bool(ok)})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)

try:
    invalid = [
        ['--gpu-layers', 'auto'], ['-ngl', '-1'], ['-ngl', 'abc'],
        ['--parallel', '2'], ['-np', '0'], ['--threads', '4oops'],
        ['--threads-batch', '2147483648'], ['-ub', '0'], ['-ub', '-1'],
        ['--batch-size', 'abc'], ['--ctx-size', '-1'], ['--port', '65536'],
        ['--flash-attn', 'maybe'], ['--alias', ''], ['--api-key', ''],
        ['--api-key', ',,'], ['--api-key-file', str(out / 'missing-key-file')],
        ['--kvmem-gpu-ratio', 'nan'], ['--kvmem-cpu-gb', 'inf'],
        ['--load-mode', 'bad'], ['--api-key'],
        ['--timeout', '-1'], ['--threads-http', '2x'], ['--device', '__missing__'],
        ['--device', ''], ['--split-mode', 'bad'], ['--main-gpu', '-2'],
        ['--tensor-split', 'nan,1'], ['--tensor-split', '1,'],
        ['--split-mode', 'tensor'],
        ['--split-mode', 'row'], ['--device', 'CUDA0,CUDA1'],
        ['--tensor-split', '1,1'], ['--no-kvmem', '--device', 'CUDA0,CUDA1'],
        ['--no-kvmem', '--split-mode', 'row'],
        ['--kvmem-conversations', '0'], ['--kvmem-conversations', '-1'],
        ['--kvmem-conversations', '1.5'], ['--kvmem-conversations', '2x'],
        ['--kvmem-conversations'], ['--kvmem-conversations', '2147483648'],
        ['--kvmem-conversations-gb', 'nan'], ['--kvmem-conversations-gb', 'inf'],
        ['--kvmem-conversations-gb', '-1'], ['--kvmem-conversations-gb', '1x'],
        # Cross-checks, not value parsing: the cap needs N > 1 to cap, and
        # several host stores need KVMem. Both throw, so both print the same
        # "invalid arguments (source=...)" line the loop below asserts on.
        ['--kvmem-conversations-gb', '24'],
        ['--no-kvmem', '--kvmem-conversations', '2'],
    ]
    for flags in invalid:
        r = subprocess.run([a.server, '-m', '__nonexistent__.gguf', *flags], env=env,
                           capture_output=True, timeout=20)
        text = r.stderr.decode(errors='replace')
        check('reject ' + ' '.join(flags), r.returncode != 0 and
              'failed to load model' not in text and ('invalid' in text or 'missing value' in text))
    # Accepted aliases must get as far as the missing model, not silently become zero.
    for flags in [['-ngl', 'all'], ['--gpu-layers', '0'],
                  ['-t', '2', '-tb', '3', '-ub', '64', '-fa', 'on', '-np', '1'],
                  ['--alias', 'test-model', '--load-mode', 'none'],
                  ['--predict', '256', '-s', '123', '-mm', 'projector.gguf', '--no-webui'],
                  ['--timeout', '60', '--threads-http', '2', '--device', 'none'],
                  ['--kvmem-conversations', '1'],
                  ['--kvmem-conversations', '8', '--kvmem-conversations-gb', '24']]:
        r = subprocess.run([a.server, '-m', '__nonexistent__.gguf', *flags], env=env,
                           capture_output=True, timeout=20)
        check('accept ' + ' '.join(flags), b'failed to load model' in r.stderr)
    r = subprocess.run([a.server, '--usage'], env=env, capture_output=True, timeout=20)
    check('--usage alias', r.returncode == 0 and b'usage:' in r.stderr)
    r = subprocess.run([a.server, '--list-devices'], env=env, capture_output=True, timeout=20)
    check('--list-devices without model', r.returncode == 0 and b'Available devices:' in r.stdout)
    if a.model:
        with tempfile.TemporaryDirectory(prefix='kvmem-compat-') as tmp:
            tmp = Path(tmp)
            ui = tmp / 'ui'
            ui.mkdir()
            (ui / 'index.html').write_text('compat-ui', encoding='utf-8')
            (ui / 'asset.js').write_text('compat-asset', encoding='utf-8')
            keyfile = tmp / 'test-keys.txt'
            keyfile.write_bytes(b'# test-only synthetic credentials\r\n\r\nfile-key\r\n')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            args = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(port),
                    '-c', '8192', '--gpu-layers', 'all', '-b', '512',
                    '-ub', '128', '-t', '2', '-tb', '3', '-fa', 'on', '-np', '1',
                    '--alias', 'compat-model', '--api-key', 'first-key,second-key',
                    '--api-key-file', str(keyfile), '--path', str(ui),
                    '-s', '1234', '--temp', '0', '--reasoning-effort', 'none',
                    '--timeout', '2', '--threads-http', '2', '--device', 'CUDA0', '-sm', 'none', '-mg', '0']
            if a.mtp:
                args += ['--kvmem', '--kvmem-budget', '4096', '--kvmem-gen-reserve', '2048',
                         '--kv-dtype', 'q8_0', '--spec-type', 'draft-mtp',
                         '--spec-draft-n-max', '3', '--spec-kv-dtype', 'f16', '--kvmem-mtp-state', 'replay']
            else:
                args += ['--no-kvmem', '--kv-dtype', 'f16', '--spec-type', 'none']
            if a.mmproj:
                args += ['--mmproj', a.mmproj, '--image-max-tokens', '512']
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            base = f'http://127.0.0.1:{port}'

            def request(path, data=None, headers=None, method=None):
                req = urllib.request.Request(base + path, data=data, headers=headers or {}, method=method)
                try:
                    with opener.open(req, timeout=180) as r:
                        return r.status, r.read()
                except urllib.error.HTTPError as e:
                    return e.code, e.read()

            with (out / 'server.log').open('wb') as log:
                proc = subprocess.Popen(args, env=env, stdout=log, stderr=log,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    deadline = time.monotonic() + 240
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f'server exited {proc.returncode}; see server.log')
                        try:
                            if request('/health')[0] == 200:
                                break
                        except OSError:
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError('server startup')
                        time.sleep(1)
                    check('health public', request('/health')[0] == 200)
                    with socket.create_connection(('127.0.0.1', port), timeout=10) as slow:
                        slow.sendall(b'GET /health HTTP/1.1\r\nHost: localhost\r\n')
                        started = time.monotonic()
                        try:
                            timeout_reply = slow.recv(4096)
                        except ConnectionResetError:
                            timeout_reply = b''
                        elapsed = time.monotonic() - started
                        check('HTTP read timeout applied', elapsed < 8 and
                              (timeout_reply == b'' or b'408' in timeout_reply or b'400' in timeout_reply))
                    check('UI public', request('/')[1] == b'compat-ui')
                    check('UI asset public', request('/asset.js')[1] == b'compat-asset')
                    for route in ['/props', '/v1/models', '/v1/chat/completions', '/unknown']:
                        check('unauthorized ' + route, request(route)[0] == 401)
                    status, body = request('/v1/chat/completions', b'not-json')
                    check('auth before JSON parsing', status == 401 and json.loads(body)['error']['type'] == 'authentication_error')
                    check('preflight public', request('/v1/chat/completions', method='OPTIONS')[0] == 204)
                    check('wrong key rejected', request('/v1/models', headers={'Authorization': 'Bearer wrong'})[0] == 401)
                    for headers in [{'Authorization': 'Bearer first-key'}, {'Authorization': 'second-key'},
                                    {'X-Api-Key': 'file-key'}]:
                        status, body = request('/v1/models', headers=headers)
                        check('accepted key form ' + next(iter(headers)) + str(len(checks)),
                              status == 200 and json.loads(body)['data'][0]['id'] == 'compat-model')
                    headers = {'Authorization': 'Bearer first-key', 'Content-Type': 'application/json'}
                    status, body = request('/props', headers=headers)
                    props = json.loads(body)
                    check('default output has no 128-token cap', status == 200 and
                          props['default_generation_settings']['params']['n_predict'] == (2048 if a.mtp else 8192))
                    check('authorized malformed JSON', request('/v1/chat/completions', b'not-json', headers)[0] == 400)
                    questions = [('What is 2+3? Answer with the number only.', '5'),
                                 ('中国的首都是哪里？只回答城市名。', '北京')]
                    if a.mtp:
                        questions.append(('Write numbers 1 through 20 in ascending order, separated by commas. Output only the list.',
                                          ','.join(str(i) for i in range(1, 21))))
                    if a.image and a.mmproj:
                        dataurl = 'data:image/png;base64,' + base64.b64encode(Path(a.image).read_bytes()).decode()
                        questions.append(([{'type': 'text', 'text': 'Read the four digits in the image. Output only the digits.'},
                                           {'type': 'image_url', 'image_url': {'url': dataurl}}], '6037'))
                    answers = []
                    for content, expected in questions:
                        payload = {'messages': [{'role': 'user', 'content': content}], 'max_tokens': 128,
                                   'reasoning_effort': 'none', 'temperature': 0, 'stream': False}
                        status, body = request('/v1/chat/completions', json.dumps(payload).encode(), headers)
                        response = json.loads(body)
                        answers.append(response)
                        (out / 'answers.json').write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding='utf-8')
                        text = response.get('choices', [{}])[0].get('message', {}).get('content', '') or ''
                        normalized = ''.join(text.strip().strip('。.!').split())
                        check('inference ' + expected, status == 200 and normalized == expected)
                        check('response alias', response.get('model') == 'compat-model')
                    # kvmem.conversation_id was an unrecognized key before
                    # multi-conversation support, so a client that sends one
                    # must still be served on a default single-store server,
                    # whatever it sends. The value is dropped, never rejected.
                    for bad in [123, '', 'a' * 129, 'has space', None]:
                        payload = {'messages': [{'role': 'user', 'content': 'What is 2+3? Answer with the number only.'}],
                                   'max_tokens': 4, 'reasoning_effort': 'none', 'temperature': 0,
                                   'stream': False, 'kvmem': {'conversation_id': bad}}
                        status, body = request('/v1/chat/completions', json.dumps(payload).encode(), headers)
                        check(f'unusable conversation_id ignored {bad!r}', status == 200)
                    payload = {'messages': [{'role': 'user', 'content': 'What is 2+3? Answer with the number only.'}],
                               'max_tokens': 32, 'reasoning_effort': 'none', 'temperature': 0, 'stream': True}
                    status, body = request('/v1/chat/completions', json.dumps(payload).encode(), headers)
                    chunks = [json.loads(line[6:]) for line in body.decode().splitlines()
                              if line.startswith('data: ') and line != 'data: [DONE]']
                    content = ''.join(chunk['choices'][0].get('delta', {}).get('content', '') or ''
                                      for chunk in chunks if chunk.get('choices'))
                    check('authenticated SSE', status == 200 and content.strip() == '5' and b'data: [DONE]' in body)
                    payload = {'messages': [{'role': 'user', 'content':
                        'Write numbers 1 through 60 in ascending order, separated by commas. Output only the list.'}],
                        'reasoning_effort': 'none', 'temperature': 0, 'stream': False}
                    status, body = request('/v1/chat/completions', json.dumps(payload).encode(), headers)
                    response = json.loads(body)
                    content = response['choices'][0]['message']['content']
                    check('omitted n/max_tokens generates beyond 128', status == 200 and
                          response['usage']['completion_tokens'] > 128 and
                          ''.join(content.strip().strip('.').split()) == ','.join(str(i) for i in range(1, 61)))
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            logtext = (out / 'server.log').read_text(encoding='utf-8', errors='replace')
            check('target effective threads/ubatch', 'KVMEM_CONTEXT target threads=2 threads_batch=3 ubatch=128' in logtext)
            check('HTTP thread configuration applied', 'KVMEM_HTTP threads=2 timeout=2' in logtext)
            if a.mtp:
                check('draft effective threads/ubatch', 'KVMEM_CONTEXT draft threads=2 threads_batch=3 ubatch=128' in logtext)
            check('keys absent from logs', all(key not in logtext for key in ['first-key', 'second-key', 'file-key']))
            keyfile.unlink()
finally:
    (out / 'results.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
