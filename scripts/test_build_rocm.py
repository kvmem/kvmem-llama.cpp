"""Model-free planning checks for the unified ROCm entry point."""
import argparse
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('rocm_build', Path(__file__).with_name('build-rocm.py'))
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class BuildPlanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'project with spaces'
        self.root.mkdir()
        self.sdk = Path(self.directory.name) / 'sdk'
        (self.sdk / 'llvm/bin').mkdir(parents=True)
        for name in ('clang++', 'clang++.exe'):
            (self.sdk / 'llvm/bin' / name).touch()
        self.args = argparse.Namespace(target=None, rocm=str(self.sdk), build_dir=None,
                                       jobs=2, gpu_targets=None, fresh=False, configure_only=False)
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def plan(self, system, wsl=False):
        with mock.patch.object(build, 'host_platform', return_value=(system, wsl)), \
                mock.patch.object(build, 'discover_rocm', return_value=self.sdk):
            return build.make_plan(self.args, self.root)

    def test_linux_rejects_windows_sdk_path(self):
        with self.assertRaisesRegex(ValueError, 'Linux ROCm SDK'):
            build.discover_rocm('D:/example/ROCm', {}, 'linux')

    def test_linux_and_wsl_share_command(self):
        linux, _ = self.plan('linux')
        wsl, env = self.plan('linux', True)
        self.assertEqual(linux['command'], wsl['command'])
        self.assertEqual(wsl['platform'], 'linux')
        self.assertTrue(wsl['wsl'])
        self.assertTrue(wsl['warnings'])
        self.assertEqual(env['HSA_ENABLE_DXG_DETECTION'], '1')

    def test_wsl_respects_explicit_dxg_setting(self):
        os.environ['HSA_ENABLE_DXG_DETECTION'] = '0'
        _, env = self.plan('linux', True)
        self.assertEqual(env['HSA_ENABLE_DXG_DETECTION'], '0')

    def test_windows_uses_fixed_batch_command(self):
        plan, env = self.plan('windows')
        self.assertEqual(plan['command'][-1], 'scripts\\build-hip.bat')
        self.assertEqual(env['ROCM_PATH'], str(self.sdk))
        self.assertTrue(env['BUILD_DIR'].endswith('build-hip-win'))

    def test_cross_os_rejected(self):
        self.args.target = 'windows'
        with self.assertRaisesRegex(ValueError, 'not a cross-compiler'):
            self.plan('linux', True)

    def test_cli_architecture_overrides_environment(self):
        os.environ['GPU_TARGETS'] = 'gfx900'
        os.environ['AMDGPU_TARGETS'] = 'gfx906'
        self.args.gpu_targets = 'gfx1030,gfx1100'
        _, env = self.plan('linux')
        self.assertEqual(env['GPU_TARGETS'], 'gfx1030;gfx1100')
        self.assertNotIn('AMDGPU_TARGETS', env)

    def test_common_radeon_release_profile_includes_gfx1030(self):
        self.args.gpu_targets = 'common'
        _, env = self.plan('windows')
        self.assertEqual(env['GPU_TARGETS'], ';'.join(build.COMMON_RADEON_TARGETS))
        self.assertIn('gfx1030', env['GPU_TARGETS'].split(';'))
        self.assertIn('gfx1200', env['GPU_TARGETS'].split(';'))

    def test_common_profile_checks_installed_kernel_packs(self):
        (self.sdk / '.kpack').mkdir()
        self.args.gpu_targets = 'common'
        with self.assertRaisesRegex(ValueError, 'lacks BLAS kernel packs'):
            self.plan('windows')

    def test_windows_sdk_rejected_on_linux(self):
        self.args.rocm = r'Z:\tools\rocm'
        with self.assertRaisesRegex(ValueError, 'Linux ROCm SDK'):
            build.discover_rocm(self.args.rocm, {}, 'linux')

    def test_source_directory_rejected(self):
        self.args.build_dir = str(self.root)
        with self.assertRaisesRegex(ValueError, 'dedicated output directory'):
            self.plan('linux')

    def test_automatic_target_is_not_hardcoded(self):
        _, env = self.plan('linux')
        self.assertNotIn('GPU_TARGETS', env)
        self.assertNotIn('AMDGPU_TARGETS', env)


if __name__ == '__main__':
    unittest.main()
