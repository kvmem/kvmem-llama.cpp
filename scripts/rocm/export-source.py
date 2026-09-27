#!/usr/bin/env python3
"""Export committed parent sources and the clean pinned submodule without private files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('Archive already exists.')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', '.', ':(exclude)llama.cpp'],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD:llama.cpp'], cwd=ROOT, text=True).strip()
    manifest = dict(source_commit=commit, llama_commit=pin, llama_patch_applied=False, files={})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'x:gz') as target:
        for repository, ref, prefix in [(ROOT, commit, ''), (ROOT / 'llama.cpp', pin, 'llama.cpp/')]:
            data = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'archive', ref], cwd=repository)
            with tarfile.open(fileobj=io.BytesIO(data)) as source:
                for member in source:
                    if not member.isfile():
                        continue
                    if '..' in Path(member.name).parts or member.name.startswith('/'):
                        raise RuntimeError('Unsafe archive member')
                    content = source.extractfile(member).read()
                    name = prefix + member.name
                    manifest['files'][name] = hashlib.sha256(content).hexdigest()
                    member.name = 'kvmem-rocm-source/' + name
                    target.addfile(member, io.BytesIO(content))
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        member = tarfile.TarInfo('kvmem-rocm-source/SOURCE-MANIFEST.json')
        member.size = len(data)
        target.addfile(member, io.BytesIO(data))
    h = hashlib.sha256()
    with output.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    Path(str(output) + '.sha256').write_text(f'{h.hexdigest()}  {output.name}\n')
    print(output)


if __name__ == '__main__':
    main()
