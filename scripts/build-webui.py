#!/usr/bin/env python3
"""Build the lightweight or full UI from pinned llama.cpp sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def node_command(name, platform=None):
    return name + '.cmd' if (platform or os.name) == 'nt' else name


def prepare_workspace(directory, root, full_ui):
    directory, root = directory.resolve(), root.resolve()
    if directory == root or directory in root.parents:
        raise ValueError('--build-dir must not be the repository or one of its parents')
    for name in ('backends', 'kvmem', 'src', 'tools', 'scripts', 'tests', 'ui', '.git'):
        source = root / name
        if directory == source or source in directory.parents:
            raise ValueError('--build-dir must not be inside source directories')
    # Separate modes: a full build never reuses lightweight routes or config.
    work = directory / ('full' if full_ui else 'lightweight')
    if work.is_symlink() or (work.exists() and work.resolve().parent != directory):
        raise ValueError('UI work directory must not be a link outside --build-dir')
    marker = work / '.kvmem-ui-workspace'
    if work.exists() and any(work.iterdir()) and not marker.is_file():
        raise ValueError(f'Refusing to overwrite unowned work directory: {work}')
    work.mkdir(parents=True, exist_ok=True)
    marker.write_text('kvmem-webui-v1\n', encoding='utf-8')
    return work


def remove_generated_directory(path, work):
    target = path.resolve()
    if target == work.resolve() or work.resolve() not in target.parents:
        raise ValueError(f'Refusing to remove directory outside UI workspace: {path}')
    if path.exists():
        shutil.rmtree(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir', type=Path, default=ROOT / 'build-ui')
    ap.add_argument('--output', type=Path, default=ROOT / 'build/share/kvmem/ui')
    ap.add_argument('--full-ui', action='store_true',
                    help='build the complete upstream UI instead of the lightweight entry points')
    # 中文：--full-ui 构建完整上游 UI（含 PWA/manifest）；默认仅构建轻量级入口页面
    args = ap.parse_args()
    work = prepare_workspace(args.build_dir, ROOT, args.full_ui)
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare-backends.py'),
                    '--backend', 'llamacpp'], check=True)
    backend = ROOT / 'backends/llamacpp'
    versions = json.loads((ROOT / 'backends/versions.json').read_text(encoding='utf-8'))
    pin = versions['llamacpp']['base_commit']
    # Copy prepared sources: git archive of the parent omits submodules and their patches.
    shutil.copytree(backend / 'tools/ui', work, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('node_modules', 'dist', '.svelte-kit', '.git'))
    if not args.full_ui:
        remove_generated_directory(work / 'src/routes', work)
        shutil.copytree(ROOT / 'ui/lightweight', work / 'src/routes')
        # The lightweight entry uses upstream components without the full application's lifecycle.
        config = (work / 'vite.config.ts').read_text(encoding='utf-8')
        config = config.replace('\t\t\tSvelteKitPWA(SVELTEKIT_PWA_OPTIONS),', '')
        (work / 'vite.config.ts').write_text(config, encoding='utf-8')
        for path in (work / 'src/lib/constants').glob('*.ts'):
            text = path.read_text(encoding='utf-8')
            if 'export const STORAGE_APP_NAME =' in text:
                import re
                path.write_text(re.sub(r"export const STORAGE_APP_NAME = .*;", "export const STORAGE_APP_NAME = 'kvmem-chat';", text), encoding='utf-8')
        hook = work / 'src/lib/hooks/use-keyboard-shortcuts.svelte.ts'
        hook.write_text(hook.read_text(encoding='utf-8').replace("page.route.id === '/(chat)'", "(page.route.id as string | null) === '/(chat)'"), encoding='utf-8')
        tsconfig = work / 'tsconfig.json'
        tsconfig.write_text('\n'.join(line for line in tsconfig.read_text(encoding='utf-8').splitlines()
                                      if '"tests/' not in line and '".storybook/' not in line), encoding='utf-8')
    lock_hash = hashlib.sha256((work / 'package-lock.json').read_bytes()).hexdigest()
    stamp = work / '.installed-lock'
    if not (work / 'node_modules').is_dir() or not stamp.is_file() or stamp.read_text() != lock_hash:
        # npm.cmd / npx.cmd resolves the batch wrapper on Windows (plain 'npm' only works in a shell/POSIX).
        # 中文：Windows 上必须用 npm.cmd/npx.cmd 才能解析批处理包装器（裸 'npm' 仅适用于 POSIX shell）
        subprocess.run([node_command('npm'), 'ci', '--no-audit', '--no-fund'], cwd=work, check=True)
        stamp.write_text(lock_hash)
    subprocess.run([node_command('npx'), 'svelte-kit', 'sync'], cwd=work, check=True)
    subprocess.run([node_command('npx'), 'svelte-check', '--threshold', 'error'], cwd=work, check=True)
    command = [node_command('npm'), 'run', 'build'] if args.full_ui else [node_command('npx'), 'vite', 'build']
    subprocess.run(command, cwd=work, check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(work / 'dist', output, dirs_exist_ok=True)
    shutil.copy2(ROOT / 'backends/llamacpp/LICENSE', output / 'llama.cpp-LICENSE')
    if args.full_ui:
        # Full UI: hash upstream src/lib entry files so the manifest records whether the page is genuinely upstream-built.
        # 中文：full-ui 模式下对上游 src/lib 的 TS 入口文件做哈希，写入 manifest 以证明产物非轻量构建
        entry_files = {p.relative_to(work).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((work / 'src').rglob('*')) if p.is_file()}
        mode = 'full-ui'
    else:
        entry_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / 'ui/lightweight').iterdir()) if p.is_file()}
        mode = 'lightweight'
    manifest = {'llama_commit': pin, 'backend_patch_sha256': versions['llamacpp']['patch_sha256'],
                'mode': mode, 'entry_files': entry_files}
    (output / 'ui-build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)

if __name__ == '__main__':
    main()
