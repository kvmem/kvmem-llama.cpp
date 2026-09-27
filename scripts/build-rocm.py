#!/usr/bin/env python3
"""One native HIP build entry point; WSL uses the Linux implementation."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

COMMON_RADEON_TARGETS = (
    'gfx1030', 'gfx1031', 'gfx1032', 'gfx1036',  # RX 6000 and RDNA2 integrated GPUs
    'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',  # RX 7000 and RDNA3 integrated GPUs
    'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153',  # recent integrated Radeon GPUs
    'gfx1200', 'gfx1201',            # RX 9000 desktop families
)


def host_platform():
    system = platform.system()
    if system == 'Windows':
        return 'windows', False
    if system == 'Linux':
        release = platform.release().lower()
        wsl = 'microsoft' in release or 'wsl' in release or bool(os.environ.get('WSL_INTEROP'))
        return 'linux', wsl
    raise ValueError('Supported build hosts are Windows and Linux (including WSL2).')


def compiler_directory(root, target):
    name = 'clang++.exe' if target == 'windows' else 'clang++'
    return next((root / suffix for suffix in ('bin', 'llvm/bin', 'lib/llvm/bin')
                 if (root / suffix / name).is_file()), None)


def discover_rocm(explicit, env, target):
    selected = explicit or next((env[k] for k in ('ROCM', 'ROCM_PATH', 'HIP_PATH') if env.get(k)), None)
    if selected:
        if target == 'linux' and re.match(r'^[A-Za-z]:[\\/]', selected):
            raise ValueError('Linux/WSL needs a Linux ROCm SDK, not the inherited Windows SDK path. Set --rocm to the Linux installation.')
        root = Path(selected).expanduser().resolve()
        if not compiler_directory(root, target):
            raise ValueError('No AMD clang++ found under ROCm root: ' + str(root))
        return root
    # Resolve tool symlinks; never guess a machine-specific SDK root.
    for name in ('hipconfig', 'hipconfig.exe', 'hipcc', 'hipcc.exe', 'amdgpu-arch', 'amdgpu-arch.exe'):
        tool = shutil.which(name, path=env.get('PATH'))
        if not tool:
            continue
        for candidate in list(Path(tool).resolve().parents)[:4]:
            if compiler_directory(candidate, target) and (candidate / 'include/hip/hip_runtime.h').is_file():
                return candidate
    raise ValueError('ROCm SDK not found. Put its tools on PATH, set ROCM_PATH/HIP_PATH, or pass --rocm. No driver or SDK is installed automatically.')


def make_plan(args, root):
    host, wsl = host_platform()
    target = args.target or host
    if target != host:
        raise ValueError('--' + target + ' selects a native build, not a cross-compiler. Run Linux Python inside Linux/WSL, or Windows Python on Windows.')
    env = os.environ.copy()
    rocm = discover_rocm(args.rocm, env, target)
    default_build = 'build-hip-win' if target == 'windows' else 'build-hip-linux'
    build = Path(args.build_dir or env.get('BUILD_DIR') or root / default_build).expanduser().resolve()
    if build == root or build in root.parents or (build / 'CMakeLists.txt').exists() or (build / '.git').exists():
        raise ValueError('Use a dedicated output directory, not a source directory or its ancestor: ' + str(build))
    jobs = args.jobs if args.jobs is not None else int(env.get('JOBS') or os.cpu_count() or 1)
    if jobs < 1:
        raise ValueError('--jobs/JOBS must be positive.')
    if args.gpu_targets:
        targets = (';'.join(COMMON_RADEON_TARGETS) if args.gpu_targets == 'common'
                   else args.gpu_targets.replace(',', ';'))
        if any(not re.fullmatch(r'gfx[0-9a-f]+(?::(?:xnack|sramecc)[+-])*', part) for part in targets.split(';')):
            raise ValueError('--gpu-targets must be common or list gfx targets separated by semicolons or commas.')
        packs = rocm / '.kpack'
        if packs.is_dir():
            missing = [part for part in targets.split(';')
                       if not (packs / f'blas_lib_{part}.kpack').is_file()]
            if missing:
                raise ValueError('ROCm SDK lacks BLAS kernel packs for: ' + ', '.join(missing))
        env['GPU_TARGETS'] = targets
        # The explicit CLI override wins over an older environment alias.
        env.pop('AMDGPU_TARGETS', None)
    for key in ('GPU_TARGETS', 'AMDGPU_TARGETS'):
        if env.get(key) and any(not re.fullmatch(r'gfx[0-9a-f]+(?::(?:xnack|sramecc)[+-])*', part)
                                for part in env[key].split(';')):
            raise ValueError(key + ' must be a semicolon-separated list of gfx targets.')
    env.update(ROCM=str(rocm), ROCM_PATH=str(rocm), HIP_PATH=str(rocm),
               BUILD_DIR=str(build), JOBS=str(jobs))
    if args.fresh:
        env['FRESH'] = '1'
    if args.configure_only:
        env['CONFIGURE_ONLY'] = '1'
    warnings = []
    if wsl:
        env.setdefault('HSA_ENABLE_DXG_DETECTION', '1')
        warnings.append('WSL uses the Linux HIP build. Windows ROCm DLLs are not Linux runtime libraries.')
        if not Path('/dev/dxg').exists():
            warnings.append('/dev/dxg is absent: WSL GPU access is not established. Explicit targets can build without a visible GPU; inference still needs working GPU access.')
        if str(root).startswith('/mnt/'):
            warnings.append('For WSL, prefer a Linux-filesystem checkout and a separate build directory. Windows Git worktree metadata may not be usable by Linux Git.')
    if target == 'windows':
        # Pass values via the environment, not through a constructed shell command.
        # The retained batch launcher uses delayed expansion.
        for name, value in [('repository', str(root)), ('ROCM', str(rocm)), ('BUILD_DIR', str(build))]:
            if any(c in value for c in '!%"\r\n'):
                raise ValueError(name + ' contains characters unsupported by the Windows batch launcher.')
        command = [env.get('COMSPEC', 'cmd.exe'), '/d', '/c', 'scripts\\build-hip.bat']
    else:
        command = ['bash', str(root / 'scripts/build-rocm.sh')]
    visible = {key: env[key] for key in ('ROCM', 'BUILD_DIR', 'JOBS', 'GPU_TARGETS', 'AMDGPU_TARGETS', 'FRESH', 'CONFIGURE_ONLY') if key in env}
    return dict(platform=target, wsl=wsl, command=command, cwd=str(root),
                environment=visible, warnings=warnings), env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--windows', dest='target', action='store_const', const='windows')
    target.add_argument('--linux', dest='target', action='store_const', const='linux')
    parser.add_argument('--rocm', help='Native ROCm SDK root (CLI overrides environment)')
    parser.add_argument('--gpu-targets', help='Optional gfx targets or common Radeon release profile; default is automatic detection')
    parser.add_argument('--build-dir', help='Separate output directory, relative to the current directory')
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--fresh', action='store_true', help='Reset CMake cache, requires CMake >= 3.24')
    parser.add_argument('--configure-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved plan without patching, configuring or building')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        plan, env = make_plan(args, root)
        print(json.dumps(plan, indent=2), flush=True)
        if args.dry_run:
            return 0
        # Catch Windows-created worktree links before either launcher edits source.
        check = subprocess.run(['git', '-C', str(root / 'llama.cpp'), 'rev-parse', '--show-toplevel'],
                               capture_output=True, text=True, env=env)
        if check.returncode and (root / 'llama.cpp/.git').exists():
            raise ValueError('llama.cpp Git metadata is not usable in this OS. Initialize the submodule in a native checkout; do not reset or rewrite another worktree.\n' + check.stderr.strip())
        if not (root / 'llama.cpp/CMakeLists.txt').is_file():
            raise ValueError('llama.cpp sources are missing. Initialize submodules or extract the complete source bundle.')
        return subprocess.run(plan['command'], cwd=root, env=env).returncode
    except (ValueError, OSError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
