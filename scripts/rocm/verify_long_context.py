#!/usr/bin/env python3
"""Verify a completed 256K KVMem agent run from its saved evidence."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open(encoding='utf-8') as source:
        return json.load(source)


def verify(folder, expected_requests):
    requests = read_json(folder / 'requests.json')
    context = read_json(folder / 'long-context.json')
    summary = read_json(folder / 'summary.json')
    rounds = context['rounds']
    if len(requests) != expected_requests or len(rounds) != expected_requests - 1:
        raise ValueError(f'Expected {expected_requests} requests and {expected_requests - 1} '
                         f'tool rounds; got {len(requests)} and {len(rounds)}')
    expected_labels = ['long-base'] + [f'long-tool-{i:03d}' for i in range(len(rounds))]
    if [item['label'] for item in requests] != expected_labels:
        raise ValueError('Request labels are missing, duplicated or out of order')
    if any(item['status'] != 200 for item in requests):
        raise ValueError('At least one request did not return HTTP 200')
    if not rounds[-1]['final'] or any(item['final'] for item in rounds[:-1]):
        raise ValueError('Final source-file round was not reached exactly once')
    context_size = context['target_ctx']
    prompt_tokens = rounds[-1]['prompt_tokens']
    if not context_size - 1024 <= prompt_tokens < context_size - context['generation_limit']:
        raise ValueError(f'Final prompt did not nearly fill context: {prompt_tokens}/{context_size}')
    if any(item['usage']['prompt_cache_hit_tokens'] <= 0 for item in requests[2:]):
        raise ValueError('Retained prompt prefix was lost in a later round')
    if any(not (folder / (label + '.response.txt')).is_file() for label in expected_labels):
        raise ValueError('A saved response is missing')
    runtime_rss = summary.get('peak_runtime_rss_mib')
    if runtime_rss is None or runtime_rss <= 0:
        raise ValueError('Runtime RAM peak was not recorded')
    final_completion_tokens = requests[-1]['usage']['completion_tokens']
    expected_completion_tokens = context['generation_limit']
    length_pass = final_completion_tokens == expected_completion_tokens
    final_text = []
    final_finish_reason = None
    with (folder / (expected_labels[-1] + '.response.txt')).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.startswith('data: {'):
                continue
            chunk = json.loads(line[6:])
            for choice in chunk.get('choices', []):
                final_text.append(choice.get('delta', {}).get('content') or '')
                final_finish_reason = choice.get('finish_reason') or final_finish_reason
    literal_tool_call = '<tool_call>' in ''.join(final_text)
    if not length_pass:
        result = 'STRUCTURAL_PASS_FINAL_LENGTH_FAIL'
    elif literal_tool_call:
        result = 'STRUCTURAL_PASS_FINAL_TOOL_CALL_FAIL'
    else:
        result = 'STRUCTURAL_PASS_LENGTH_PASS_SEMANTIC_REVIEW_REQUIRED'
    return {
        'result': result,
        'requests': len(requests),
        'tool_rounds': len(rounds),
        'final_prompt_tokens': prompt_tokens,
        'final_completion_tokens': final_completion_tokens,
        'expected_completion_tokens': expected_completion_tokens,
        'final_finish_reason': final_finish_reason,
        'literal_tool_call': literal_tool_call,
        'context_size': context_size,
        'peak_runtime_rss_mib': round(runtime_rss, 2),
        'elapsed_request_seconds': round(sum(item['elapsed_s'] for item in requests), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--expected-requests', type=int, default=33)
    args = parser.parse_args()
    result = verify(args.folder, args.expected_requests)
    print(json.dumps(result, indent=2))
    if result['final_completion_tokens'] != result['expected_completion_tokens'] or result['literal_tool_call']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
