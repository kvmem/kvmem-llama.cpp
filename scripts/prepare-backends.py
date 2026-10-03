#!/usr/bin/env python3
"""Initialize pinned backend submodules and apply the repository's integration patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *args, check=True, env=None):
    return subprocess.run(['git', '-c', 'core.longpaths=true', '-c', 'core.autocrlf=false',
                           '-C', str(directory), *args], check=check,
                          capture_output=True, text=True, env=env)


def verify_tree(directory, expected):
    # A separate index includes patch-added files without changing the user's index.
    handle, filename = tempfile.mkstemp(prefix='kvmem-backend-', suffix='.index')
    os.close(handle)
    os.unlink(filename)
    env = dict(os.environ, GIT_INDEX_FILE=filename)
    try:
        git(directory, 'read-tree', 'HEAD', env=env)
        git(directory, 'add', '-A', env=env)
        actual = git(directory, 'write-tree', env=env).stdout.strip()
        if actual != expected:
            raise RuntimeError(f'{directory}: source differs from the pinned integration patch; '
                               'record backend edits in its patch before release validation')
    finally:
        if Path(filename).exists():
            os.unlink(filename)


def prepare(name, entry, check_only):
    directory = ROOT / entry['path']
    patch = ROOT / entry['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != entry['patch_sha256']:
        raise RuntimeError(f'{name}: update versions.json after changing the integration patch')
    # Exported source bundles contain prepared sources and a per-file manifest, without Git metadata.
    if not (ROOT / '.git').exists():
        manifest = json.loads((ROOT / 'SOURCE-MANIFEST.json').read_text(encoding='utf-8'))
        if manifest['backends'][name] != entry:
            raise RuntimeError(f'{name}: source bundle backend version differs from versions.json')
        files = {path: value for path, value in manifest['files'].items()
                 if path.startswith(entry['path'] + '/')}
        if not files:
            raise RuntimeError(f'{name}: source bundle has no backend files')
        for path, expected in files.items():
            source = (ROOT / path).resolve()
            if not source.is_relative_to(directory.resolve()):
                raise RuntimeError(f'{name}: invalid source manifest path: {path}')
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'{name}: source bundle file differs from manifest: {path}')
        print(f'{name}: verified prepared source bundle', flush=True)
        return
    if not check_only:
        git(ROOT, 'submodule', 'update', '--init', '--recursive', '--depth', '1', '--', entry['path'])
    if not (directory / '.git').exists():
        raise RuntimeError(f'{name}: run scripts/prepare-backends.py --backend {name}')
    head = git(directory, 'rev-parse', 'HEAD').stdout.strip()
    if head != entry['base_commit']:
        raise RuntimeError(f'{name}: expected {entry["base_commit"]}, got {head}')
    ready = git(directory, 'apply', '--reverse', '--check', str(patch), check=False)
    if ready.returncode != 0:
        if check_only:
            raise RuntimeError(f'{name}: integration patch is missing or modified')
        git(directory, 'apply', '--check', str(patch))
        git(directory, 'apply', '--whitespace=nowarn', str(patch))
    if check_only:
        verify_tree(directory, entry['patched_tree'])
    print(f'{name}: {head[:12]} + {patch.name} ready', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('llamacpp', 'ninfer', 'all'), default='all')
    parser.add_argument('--check', action='store_true',
                        help='verify exact prepared sources without fetching or applying patches')
    args = parser.parse_args()
    versions = json.loads((ROOT / 'backends/versions.json').read_text(encoding='utf-8'))
    names = ('llamacpp', 'ninfer') if args.backend == 'all' else (args.backend,)
    try:
        for name in names:
            prepare(name, versions[name], args.check)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        parser.exit(1, f'Backend preparation failed: {detail}\n')


if __name__ == '__main__':
    main()
