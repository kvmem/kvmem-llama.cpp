"""Exercise a local two-GPU KVMem worker through its public HTTP interface."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--red', type=Path, required=True)
    parser.add_argument('--blue', type=Path, required=True)
    parser.add_argument('--residency', choices=['resident', 'cpu'], default='cpu')
    parser.add_argument('--stage-layers', default='40,24')
    parser.add_argument('--port', type=int, default=18310)
    parser.add_argument('--single-device-reference', action='store_true',
                        help='Diagnostic control on the first GPU with the same C2 HTTP workload')
    args = parser.parse_args()
    args.output = args.output.resolve()
    uuids = [item.strip() for item in args.gpu.split(',')]
    if len(uuids) != 2 or len(set(uuids)) != 2 or not all(item.startswith('GPU-') for item in uuids):
        parser.error('--gpu requires two distinct UUIDs')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
    cards = {row[0].strip(): int(row[1]) for row in csv.reader(inventory.splitlines())}
    if any(uuid not in cards or cards[uuid] > 1024 for uuid in uuids):
        parser.error('a requested GPU is unavailable or occupied')
    args.output.mkdir(parents=True, exist_ok=True)
    for port in [args.port, args.port + 1]:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                parser.error(f'test port is occupied: {port}')
    exe = args.build.resolve() / 'apps/ninfer-serve.exe'
    command = [str(exe), str(args.model.resolve()), '--host', '127.0.0.1',
               '--port', str(args.port), '--stats-port', str(args.port + 1),
               '--model-id', 'multigpu-qualification',
               *(['--device', '0'] if args.single_device_reference else
                 ['--devices', '0,1', '--stage-layers', args.stage_layers]), '--device-profile', 'off',
               '--max-context', '8192', '--max-concurrency', '2', '--prefill-chunk', '128',
               '--default-max-tokens', '128', '--kvmem-budget', '1536', '--kvmem-gen-reserve', '128',
               '--kvmem-host-mib', '1024', '--kvmem-sessions', '4', '--kvmem-verify-transfers',
               '--kv-dtype', 'int8', '--spec', 'mtp', '--draft-tokens', '3', '--adaptive-mtp',
               '--ngram-draft-tokens', '0', '--vision', '--vision-residency', args.residency,
               '--vision-max-merged', '1024', '--derive-session-keys',
               '--request-log-jsonl', str(args.output / 'http-requests.jsonl')]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NINFER_', 'KVMEM_', 'LLAMA_ARG_', 'GGML_'))}
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids), CUDA_DEVICE_ORDER='PCI_BUS_ID')
    if os.name == 'nt':
        system = Path(os.environ['SystemRoot'])
        env['PATH'] = str(system / 'System32') + os.pathsep + str(system)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{args.port}'

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

    def get(path):
        endpoint = f'http://127.0.0.1:{args.port + 1}' if path in ['/stats', '/health'] else base
        with opener.open(endpoint + path, timeout=15) as response:
            return json.load(response)

    bodies = []
    for lane, (color, path) in enumerate([('red', args.red), ('blue', args.blue)]):
        code = f'ORCHID-{9100 + lane}'
        content = [{'type': 'text', 'text': 'Start with the archive access code and dominant image color in English, then explain typical uses of this color in at least 150 words.'},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()}}]
        bodies.append(({'model': 'multigpu-qualification', 'enable_thinking': False,
                        'temperature': 0, 'max_tokens': 128, 'messages': [
                            {'role': 'system', 'content': f'This archive has access code {code}.'},
                            {'role': 'user', 'content': content}]}, code, color))

    def wave(label):
        barrier = threading.Barrier(2)
        before = get('/stats')

        def request(lane):
            body, code, color = bodies[lane]
            req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
            barrier.wait()
            with opener.open(req, timeout=600) as response:
                result = json.load(response)
            content = result['choices'][0]['message']['content']
            assert code in content and color in content.lower(), (label, lane, content)
            assert result['usage']['prompt_tokens'] >= 1024, result['usage']
            return result

        peak = 0
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request, lane) for lane in range(2)]
            while not all(f.done() for f in futures):
                stats = get('/stats')
                peak = max(peak, stats['requests']['running'])
                assert stats['kvmem']['host_payload_bytes'] <= 1024 * (1 << 20)
                assert stats['kvmem']['active_host_reservation_bytes'] <= 1024 * (1 << 20)
                time.sleep(0.1)
            results = [f.result() for f in futures]
        after = get('/stats')
        assert after['requests']['running'] == 0 and after['requests']['waiting'] == 0
        for key in ['resident_pages', 'mtp_resident_pages', 'active_host_reservation_bytes']:
            assert after['kvmem'][key] == 0, (key, after['kvmem'])
        if label == 'warm':
            assert peak == 2, peak
            counters = after['counters']
            assert counters['decode_row_rounds'] - before['counters']['decode_row_rounds'] > counters['decode_rounds'] - before['counters']['decode_rounds']
            assert counters['reused_prompt_tokens'] > before['counters']['reused_prompt_tokens']
        save(f'http-{label}.json', {'results': results, 'peak': peak, 'before': before, 'after': after})
        print(f'HTTP {label} PASS peak={peak}', flush=True)
        return results

    with (args.output / 'http-server.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=exe.parent, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        save('http-config.json', {'command': command, 'gpu': uuids, 'pid': child.pid,
                                  'runtime_path': env['PATH']})
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                assert child.poll() is None, f'server exited {child.returncode}'
                try:
                    if get('/health')['status'] == 'ok':
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('server did not become ready')
            props = get('/props')
            save('http-props.json', props)
            assert props['total_slots'] == 2 and props['modalities']['vision']
            assert props['execution']['mode'] == ('single' if args.single_device_reference else 'layer')
            assert props['execution']['devices'] == ([0] if args.single_device_reference else [0, 1])
            assert props['execution']['stage_layers'] == ([] if args.single_device_reference else
                                                        [int(item) for item in args.stage_layers.split(',')])
            assert props['kvmem']['capabilities']['shared_host_history']
            cold, warm = wave('cold'), wave('warm')
            # Free generation uses different verify batch shapes in serialized cold prefill
            # and warm compact decode. Record exact-text diagnostics separately from the
            # HTTP contract (identity, reuse, quotas and stream delivery). Exact source/token
            # restoration remains a required assertion in the memory_ngram model fixture.
            exact_text = [first['choices'][0]['message']['content'] == restored['choices'][0]['message']['content']
                          for first, restored in zip(cold, warm)]
            save('http-free-generation-comparison.json', {'cold_warm_exact_text': exact_text})
            print(f'HTTP free generation exact text diagnostic: {exact_text}', flush=True)
            body = {'model': 'multigpu-qualification', 'enable_thinking': False,
                    'temperature': 0, 'max_tokens': 64, 'stream': True,
                    'messages': [{'role': 'user', 'content': 'Start with ORCHID-STREAM, then explain why the sky is blue.'}]}
            req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
            chunks = []
            with opener.open(req, timeout=600) as response:
                for line in response:
                    if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                        chunks.append(json.loads(line[6:]))
            content = ''.join(item['choices'][0].get('delta', {}).get('content') or '' for item in chunks if item['choices'])
            assert 'ORCHID-STREAM' in content, content
            save('http-stream.json', chunks)
            save('http-summary.json', {'passed': True,
                                       'gpu': uuids[:1] if args.single_device_reference else uuids,
                                       'residency': args.residency,
                                       'single_device_reference': args.single_device_reference,
                                       'cold_warm_exact_text': exact_text,
                                       'images': '1024 x 1024 red/blue', 'requests': 5,
                                       'checks': ['props', 'image identity', 'history reuse', 'shared H budget', 'C2 warm batching', 'streaming text', 'retirement']})
            print('HTTP SINGLE REFERENCE PASS' if args.single_device_reference else 'HTTP MULTIGPU PASS', flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=30)


if __name__ == '__main__':
    main()
