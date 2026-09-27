#!/usr/bin/env python3
"""Package a native HIP build into a new local directory and archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def windows_runtime(build, sdk, output, targets):
    reader = next((path for path in (sdk / 'bin/llvm-readobj.exe',
                                     sdk / 'lib/llvm/bin/llvm-readobj.exe') if path.is_file()), None)
    if reader is None:
        raise RuntimeError('Missing llvm-readobj.exe in ROCm SDK: ' + str(sdk))
    search = [build / 'bin', sdk / 'bin', sdk / 'lib/llvm/bin']
    queue = list((output / 'bin').glob('*.exe'))
    external = set()
    seen = set()
    # HIP loads the compiler and RTC dynamically, so imports alone are insufficient.
    for pattern in ('amd_comgr*.dll', 'hiprtc*.dll', 'rocm_kpack*.dll'):
        for file in (sdk / 'bin').glob(pattern):
            dest = output / 'bin' / file.name
            copy(file, dest)
            queue.append(dest)
    while queue:
        file = queue.pop()
        text = subprocess.check_output([str(reader), '--coff-imports', str(file)], text=True)
        for name in re.findall(r'^\s+Name: (.+\.dll)\s*$', text, re.M | re.I):
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            if key.startswith('libomp140'):
                raise RuntimeError('Non-redistributable MSVC libomp dependency; build with GGML_OPENMP=OFF.')
            if key.startswith(('api-ms-', 'ext-ms-', 'msvcp', 'vcruntime', 'vcomp', 'concrt')):
                external.add(name)
                continue
            found = next((folder / name for folder in search if (folder / name).is_file()), None)
            if found:
                dest = output / 'bin' / name
                copy(found, dest)
                queue.append(dest)
            elif (Path(os.environ['SystemRoot']) / 'System32' / name).is_file():
                external.add(name)
            else:
                raise RuntimeError(f'Unresolved DLL {name} imported by {file.name}')
    packs = sdk / '.kpack'
    if packs.is_dir():
        if not targets:
            raise RuntimeError('Set GPU_TARGETS before packaging a multi-arch ROCm SDK.')
        for target in targets:
            kernels = list(packs.glob('*_' + target + '.kpack'))
            if not any(p.name.startswith('blas_lib_') for p in kernels):
                raise RuntimeError('Missing BLAS kernel pack for ' + target)
            for file in kernels:
                copy(file, output / '.kpack' / file.name)
    for name in ('rocblas', 'hipblaslt'):
        data = sdk / 'bin' / name / 'library'
        if not data.is_dir() and not packs.is_dir():
            raise RuntimeError('Missing ROCm kernel library: ' + str(data))
        if data.is_dir():
            shutil.copytree(data, output / 'bin' / name / 'library')
    return sorted(external)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--sdk', type=Path, help='Windows HIP SDK root')
    ap.add_argument('--runtime-licenses', type=Path, help='Collected component license directory')
    ap.add_argument('--validation', required=True, type=Path)
    ap.add_argument('--local-dirty-source', action='store_true',
                    help='Package an uncommitted local test build and mark its provenance')
    args = ap.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    windows = os.name == 'nt'
    if output.exists() or Path(str(output) + '.zip').exists() or Path(str(output) + '.tar.gz').exists():
        raise RuntimeError('Output already exists; choose a new package directory.')
    if build == output or output in build.parents or ROOT == output or output in ROOT.parents:
        raise RuntimeError('Output must not be a source/build directory or its ancestor.')
    cache = (build / 'CMakeCache.txt').read_text()
    source_status = subprocess.run(
        ['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=normal',
         '--', '.', ':(exclude)llama.cpp'], check=True, capture_output=True, text=True)
    source_dirty = bool(source_status.stdout.strip())
    if source_dirty and not args.local_dirty_source:
        raise RuntimeError('Source tree has uncommitted changes. Commit reviewed code before release packaging, or use --local-dirty-source for a local test package.')
    options = dict(re.findall(r'^([^/#\n][^:\n]*):[^=\n]+=(.*)$', cache, re.M))
    targets = [item for item in options.get('GPU_TARGETS', '').split(';') if item]
    for key, value in [('CMAKE_BUILD_TYPE', 'Release'), ('GGML_HIP', 'ON'),
                       ('GGML_CUDA', 'OFF'), ('GGML_NATIVE', 'OFF'), ('KVMEM_ENABLE_NVME', 'OFF')]:
        if options.get(key) != value:
            raise RuntimeError(f'{key} must be {value}, got {options.get(key)}')
    if windows and not args.sdk:
        ap.error('Windows packaging requires --sdk.')
    args.validation.read_text(encoding='utf-8')
    ui = build / 'share/kvmem'
    for mode in ('ui', 'ui-lightweight'):
        if not (ui / mode / 'index.html').is_file():
            raise RuntimeError('Missing built UI: ' + mode)
    suffix = '.exe' if windows else ''
    binaries = [build / 'bin' / (name + suffix)
                for name in ('llama-kvmem-server', 'llama-kvmem-cli', 'llama-bench')]
    missing = [str(path) for path in binaries if not path.is_file()]
    if missing:
        raise RuntimeError('Missing built executable(s): ' + ', '.join(missing))
    license_source = None
    if windows:
        license_source = args.runtime_licenses or args.sdk.resolve() / 'share/doc'
        if not license_source.is_dir():
            raise RuntimeError('Missing ROCm component licenses: ' + str(license_source))
    output.mkdir(parents=True)
    for binary in binaries:
        copy(binary, output / 'bin' / binary.name)
    external = []
    if windows:
        external = windows_runtime(build, args.sdk.resolve(), output, targets)
        shutil.copytree(license_source, output / 'licenses/ROCm')
    else:
        # Keep the OS/ROCm driver stack external; bundle only this build's libraries.
        for folder in (build / 'bin', build / 'kvmem'):
            for file in folder.glob('*.so*'):
                copy(file, output / 'lib' / file.name)
        external = ['compatible ROCm runtime', 'compatible AMD driver', 'glibc and libstdc++']
    for name in ('start-iq3.ps1', 'start-iq3.sh'):
        if (windows and name.endswith('.ps1')) or (not windows and name.endswith('.sh')):
            copy(ROOT / 'scripts/rocm' / name, output / 'scripts/rocm' / name)
    shutil.copytree(ui, output / 'share/kvmem')
    copy(ROOT / 'docs/rocm.md', output / 'README.md')
    copy(ROOT / 'docs/rocm-contributors.md', output / 'rocm-contributors.md')
    copy(args.validation, output / 'VALIDATION.md')
    copy(ROOT / 'llama.cpp/LICENSE', output / 'licenses/llama.cpp-MIT.txt')
    for tree in ('kvmem', 'llama.cpp/vendor'):
        for file in (ROOT / tree).rglob('*'):
            if file.is_file() and re.match(r'^(LICENSE|COPYING|NOTICE)', file.name, re.I):
                copy(file, output / 'licenses' / file.relative_to(ROOT))
    keys = ('GPU_TARGETS', 'CMAKE_HIP_ARCHITECTURES', 'GGML_HIP', 'GGML_CUDA',
            'GGML_NATIVE', 'GGML_OPENMP', 'CMAKE_BUILD_TYPE', 'BUILD_SHARED_LIBS')
    info = dict(source_commit=git('rev-parse', 'HEAD'),
                source_worktree_dirty=source_dirty,
                llama_commit=git('rev-parse', 'HEAD:llama.cpp'),
                patch_sha256=sha(ROOT / 'patches/llama-kvmem-current.patch'),
                platform='windows-x64' if windows else 'linux-x86_64',
                build_options={k: options.get(k) for k in keys},
                external_dependencies=external, models_included=False,
                cpu_requirement='AVX2, FMA, F16C, BMI2',
                validation='See VALIDATION.md; compiled architectures are not a hardware certification.')
    sdk_root = args.sdk.resolve() if args.sdk else Path(options.get('ROCM_PATH', '/missing-rocm-sdk'))
    version_file = sdk_root / '.info/version'
    info['rocm_version'] = version_file.read_text(encoding='utf-8').strip() if version_file.is_file() else None
    compiler = options.get('CMAKE_CXX_COMPILER')
    if compiler:
        info['compiler_version'] = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    info['source_files_sha256'] = {
        name: sha(ROOT / name) for name in git('ls-files').splitlines()
        if (ROOT / name).is_file()
    }
    (output / 'BUILD-INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(output).as_posix()}\n' for p in files))
    if windows:
        archive_path = Path(str(output) + '.zip')
        with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for p in sorted(output.rglob('*')):
                if p.is_file():
                    archive.write(p, p.relative_to(output.parent))
    else:
        archive_path = Path(str(output) + '.tar.gz')
        with tarfile.open(archive_path, 'x:gz') as archive:
            archive.add(output, arcname=output.name)
    Path(str(archive_path) + '.sha256').write_text(f'{sha(archive_path)}  {archive_path.name}\n')
    print(archive_path)


if __name__ == '__main__':
    main()
