#!/usr/bin/env python3
"""Package an isolated CUDA build; does not modify or restart running services."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--build', required=True, type=Path)
    ap.add_argument('--cuda-root', required=True, type=Path)
    ap.add_argument('--cudart-license', required=True, type=Path)
    ap.add_argument('--cublas-license', required=True, type=Path)
    ap.add_argument('--apache-license', type=Path, default=Path('/usr/share/common-licenses/Apache-2.0'))
    ap.add_argument('--source-manifest', required=True, type=Path)
    ap.add_argument('--source-archive', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--ui-dir', type=Path, help='optional built static chat UI')
    args = ap.parse_args()
    source, build = args.source.resolve(), args.build.resolve()
    manifest = json.loads(args.source_manifest.read_text())
    for name, expected in manifest['files'].items():
        if sha(source / name) != expected:
            raise RuntimeError(f'source differs from manifest: {name}')
    package = args.output.resolve()
    package.mkdir(parents=True, exist_ok=False)
    for name in ['bin', 'lib', 'scripts', 'licenses', 'build', 'provenance']:
        (package / name).mkdir()
    for name in ['llama-kvmem-server', 'llama-kvmem-cli', 'llama-quantize']:
        shutil.copy2(build / 'bin' / name, package / 'bin' / name)
    for folder in [build / 'bin', build / 'kvmem']:
        for lib in folder.glob('*.so*'):
            # Dereference build symlinks; preserve actual SONAMEs as relative links.
            target = package / 'lib' / lib.resolve().name
            if not target.exists():
                shutil.copy2(lib.resolve(), target)
            alias = package / 'lib' / lib.name
            if alias != target and not alias.exists():
                alias.symlink_to(target.name)
    cuda_lib = args.cuda_root.resolve() / 'targets/x86_64-linux/lib'
    for name in ['libcudart.so.13', 'libcublas.so.13', 'libcublasLt.so.13']:
        lib = cuda_lib / name
        shutil.copy2(lib.resolve(), package / 'lib' / lib.resolve().name)
        (package / 'lib' / name).symlink_to(lib.resolve().name)
    (package / 'build/bin').symlink_to('../bin')
    scripts_root = source / 'scripts'
    for name in ['start-iq3.sh', 'start-iq4.sh', 'start-server.py', 'stop-iq3.sh']:
        shutil.copy2(scripts_root / name, package / 'scripts' / name)
    (package / 'scripts/prebuilt').mkdir()
    shutil.copy2(scripts_root / 'prebuilt/diagnose.py', package / 'scripts/prebuilt/diagnose.py')
    shutil.copy2(scripts_root / 'prebuilt/README.md', package / 'README.md')
    ui = args.ui_dir if args.ui_dir is not None else build / 'share/kvmem/ui'
    if args.ui_dir is not None and not (ui / 'index.html').is_file():
        raise RuntimeError(f'UI directory has no index.html: {ui}')
    if (ui / 'index.html').is_file():
        shutil.copytree(ui, package / 'share/kvmem/ui')
    shutil.copy2(args.cudart_license, package / 'licenses/NVIDIA-CUDA.txt')
    shutil.copy2(args.cublas_license, package / 'licenses/NVIDIA-cuBLAS.txt')
    shutil.copy2(args.apache_license, package / 'licenses/Apache-2.0.txt')
    shutil.copy2(source / 'backends/llamacpp/LICENSE', package / 'licenses/llama.cpp-MIT.txt')
    # Retain vendored dependencies' license texts and embedded notices in the source archive.
    for p in (source / 'backends/llamacpp/vendor').rglob('*'):
        if p.is_file() and p.name.lower().startswith(('license', 'copying', 'notice')):
            dest = package / 'licenses/vendor' / p.relative_to(source / 'backends/llamacpp/vendor')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    (package / 'licenses/KVMem-NOTICE.txt').write_text(
        'KVMem port: source README declares Apache-2.0; source snapshot included separately.\n'
        'CUDA/cuBLAS libraries are NVIDIA components for use with this application under the enclosed terms.\n'
        'Host glibc, libstdc++, libgcc, libgomp and NVIDIA driver are not redistributed.\n')
    shutil.copy2(args.source_manifest, package / 'provenance/source-manifest.json')
    cache = (build / 'CMakeCache.txt').read_text()
    options = dict(re.findall(r'^((?:BUILD_SHARED_LIBS|CMAKE_(?:CUDA|CXX)_COMPILER|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|GGML_NATIVE|GGML_AVX\w*|GGML_FMA|GGML_F16C|GGML_BMI2|GGML_CUDA_FA_QUANTS)):[^=]+=(.*)$', cache, re.M))
    info = {'status': 'experimental-build', 'version': (source / 'VERSION').read_text().strip(),
            'base_commit': manifest['base_commit'],
            'llama_commit': manifest['llama_commit'], 'source_archive_sha256': sha(args.source_archive),
            'source_manifest_sha256': sha(args.source_manifest), 'build_options': options,
            'compiler': subprocess.check_output(['c++','--version'], text=True).splitlines()[0],
            'cuda_compiler': subprocess.check_output([str(args.cuda_root/'bin/nvcc'),'--version'], text=True),
            'os_release': Path('/etc/os-release').read_text(),
            'cpu_requirement': 'x86_64 with AVX2, FMA, F16C and BMI2',
            'gpu_target': 'sm_120a', 'system_baseline': 'Ubuntu 22.04 / glibc 2.35',
            'models_included': False}
    (package / 'BUILD-INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    # Every ELF must use only relative RPATHs; never ship driver stubs.
    for p in [* (package/'bin').iterdir(), * (package/'lib').iterdir()]:
        if p.is_symlink():
            continue
        dynamic = subprocess.check_output(['readelf','-d',str(p)], text=True)
        for value in re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', dynamic):
            if any(part and not part.startswith('$ORIGIN') for part in value.split(':')):
                raise RuntimeError(f'nonportable RPATH in {p.name}: {value}')
        if p.name.startswith('libcuda.'):
            raise RuntimeError('driver must not be bundled')
    sums = ''.join(f'{sha(p)}  {p.relative_to(package)}\n' for p in sorted(package.rglob('*'))
                   if p.is_file() and not p.is_symlink())
    (package / 'SHA256SUMS').write_text(sums)
    archive = package.with_name(package.name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(package, arcname=package.name)
    archive.with_name(archive.name + '.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print(archive)


if __name__ == '__main__':
    main()
