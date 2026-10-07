"""Staged correctness qualification of native and KVMem layer pipelines on two GPUs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='Two GPU UUIDs in stage order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--red', type=Path, required=True)
    parser.add_argument('--blue', type=Path, required=True)
    parser.add_argument('--phase', choices=['core', 'native', 'text', 'vision', 'edge', 'formats'], required=True)
    parser.add_argument('--stage-layers', default='40,24', help='Uneven native oracle split')
    parser.add_argument('--case', action='append', help='Run only a named case; may be repeated')
    parser.add_argument('--split-reference', action='store_true', help='Use the first dual-card row when one card cannot hold the model')
    parser.add_argument('--same-placement-reference', action='store_true', help='Require exact Graph/eager/transport invariance per layer placement when heterogeneous SM counts change GGUF Stream-K rounding')
    args = parser.parse_args()
    uuids = [value.strip() for value in args.gpu.split(',')]
    if len(uuids) != 2 or len(set(uuids)) != 2 or not all(value.startswith('GPU-') for value in uuids):
        parser.error('--gpu requires two distinct UUIDs')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,memory.total',
                                         '--format=csv,noheader,nounits'], text=True)
    cards = {row[0].strip(): [item.strip() for item in row[1:]] for row in csv.reader(inventory.splitlines())}
    for uuid in uuids:
        if uuid not in cards:
            parser.error(f'GPU is unavailable: {uuid}')
        if int(cards[uuid][1]) > 1024:
            parser.error(f'GPU is occupied ({cards[uuid][1]} MiB): {uuid}')
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in list(env):
        if name.startswith('NINFER_TEST_'):
            del env[name]
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               NINFER_TEST_DEVICE_IDS='0,1')
    model = str(args.model.resolve())
    images = [str(args.red.resolve()), str(args.blue.resolve())]
    cases = []

    def add(name, target, values=(), extra=None):
        cases.append((name, target, list(values), extra or {}))

    if args.phase == 'core':
        for target in ['ninfer_multi_gpu_test', 'ninfer_kv_capacity_test', 'ninfer_serve_options_test',
                       'ninfer_p1_options', 'ninfer_qwen3_5_loading_test', 'ninfer_artifact_materialization_test']:
            add(target, target)
    elif args.phase == 'native':
        add('native-stages', 'ninfer_qwen3_5_stages_real_test', extra={
            'NINFER_TEST_ARTIFACT': model, 'NINFER_TEST_SPLIT_INVARIANCE': '1' if args.split_reference or args.same_placement_reference else '0',
            'NINFER_TEST_SAME_PLACEMENT_REFERENCE': '1' if args.same_placement_reference else '0',
            'NINFER_TEST_MAX_STAGES': '2', 'NINFER_TEST_STAGE_LAYERS': args.stage_layers})
    elif args.phase == 'text':
        add('history-int8-eager', 'ninfer_p4_history', [model, 'int8', 'eager', str(args.output / 'history.tsv'), '0'])
        add('endpoint-int8-graph-mtp3', 'ninfer_p4_history', [model, 'int8', 'graph', str(args.output / 'endpoint.tsv'), '3', 'endpoint'])
        add('sessions-int8-graph-mtp3', 'ninfer_memory_sessions', [model, '3', 'int8', 'graph'])
        add('concurrency2-int8-graph-mtp3', 'ninfer_memory_concurrency', [model, '3', '2', '512', 'int8', 'graph', 'fixed'])
        add('history-rk8v4-adaptive-fast', 'ninfer_p4_history', [model, 'rk8v4', 'graph', str(args.output / 'fast.tsv'), '3', 'all', '0', 'adaptive-fast'])
    elif args.phase == 'vision':
        for residency, execution, drafts in [('resident', 'eager', '0'), ('cpu', 'graph', '3')]:
            for route in ['native', 'kvmem']:
                add(f'vision-{route}-{residency}-{execution}', 'ninfer_memory_vision',
                    [model, drafts, route, *images, 'int8', execution, residency, 'all'])
        add('vision-copy-rk8v4-adaptive-ngram', 'ninfer_memory_vision',
            [model, '3', 'kvmem', *images, 'rk8v4', 'graph', 'resident', 'copy-compare', 'adaptive', '15'])
        add('vision-ngram2-rk8v4-fast', 'ninfer_memory_ngram',
            [model, 'rk8v4', 'graph', '31', '3', 'fixed', '2', *images, 'resident'],
            {'NINFER_TEST_FAST_PREFILL': '1'})
        add('vision-concurrency2-cpu', 'ninfer_memory_concurrency',
            [model, '3', '2', '1024', 'int8', 'graph', 'adaptive', *images, 'cpu', '1024', '3072'])
    elif args.phase == 'formats':
        for storage in ['bf16', 'fp8', 'nvfp4', 'k8v4', 'rk4v4', 'rk4v4-e8', 'rk2v4-e8']:
            add(f'sessions-{storage}', 'ninfer_memory_sessions', [model, '3', storage, 'graph'])
    else:
        # First stage has no full-attention layers: origin still belongs to rank 0, reductions to rank 1.
        add('no-attention-first-stage', 'ninfer_memory_sessions', [model, '3', 'int8', 'graph'],
            {'NINFER_TEST_DEVICE_IDS': '1,0', 'NINFER_TEST_STAGE_LAYERS': '3,61'})
    if args.case:
        missing = set(args.case) - {case[0] for case in cases}
        if missing:
            parser.error(f'unknown cases: {sorted(missing)}')
        cases = [case for case in cases if case[0] in args.case]
    summary_name = args.phase + ('-selected' if args.case else '') + '-summary.json'
    report = {'phase': args.phase, 'model': model, 'devices': {uuid: cards[uuid] for uuid in uuids}, 'cases': []}
    for name, target, values, extra in cases:
        executable = args.build.resolve() / 'tests' / (target + ('.exe' if os.name == 'nt' else ''))
        if not executable.is_file():
            executable = args.build.resolve() / 'tests' / ('ninfer_pipeline_tests' + ('.exe' if os.name == 'nt' else ''))
            values = [target, *values]
        command = [str(executable), *values]
        started = time.monotonic()
        print(f'RUN {name}', flush=True)
        with (args.output / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, env=dict(env, **extra), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        report['cases'].append({'name': name, 'command': command, 'environment': {
            'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'], 'NINFER_TEST_DEVICE_IDS': env['NINFER_TEST_DEVICE_IDS'], **extra},
            'exit_code': result.returncode, 'seconds': round(time.monotonic() - started, 2)})
        (args.output / summary_name).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'{name}: exit={result.returncode}', flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
