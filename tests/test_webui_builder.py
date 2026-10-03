import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('builder', Path(__file__).resolve().parents[1] / 'scripts/build-webui.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

class BuilderTest(unittest.TestCase):
    def test_platform_commands(self):
        self.assertEqual(builder.node_command('npm', 'posix'), 'npm')
        self.assertEqual(builder.node_command('npx', 'posix'), 'npx')
        self.assertEqual(builder.node_command('npm', 'nt'), 'npm.cmd')
        self.assertEqual(builder.node_command('npx', 'nt'), 'npx.cmd')

    def test_protect_sources_and_unowned_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo'
            root.mkdir()
            sentinel = root / 'keep.txt'
            sentinel.write_text('keep')
            for path in [root, root.parent, root / 'backends/llamacpp/tools/ui',
                         root / 'backends/ninfer', root / 'kvmem/include', root / 'scripts/cache']:
                with self.assertRaises(ValueError):
                    builder.prepare_workspace(path, root, True)
            unowned = root / 'build-ui/full'
            unowned.mkdir(parents=True)
            (unowned / 'keep.txt').write_text('keep')
            with self.assertRaises(ValueError):
                builder.prepare_workspace(unowned.parent, root, True)
            self.assertEqual(sentinel.read_text(), 'keep')
            self.assertEqual((unowned / 'keep.txt').read_text(), 'keep')

    def test_modes_do_not_share_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo'
            root.mkdir()
            a = builder.prepare_workspace(root / 'build-ui', root, False)
            b = builder.prepare_workspace(root / 'build-ui', root, True)
            self.assertNotEqual(a, b)
            (a / 'light-only.txt').write_text('keep')
            self.assertFalse((b / 'light-only.txt').exists())
            self.assertEqual(builder.prepare_workspace(root / 'build-ui', root, False), a)
            with self.assertRaises(ValueError):
                builder.remove_generated_directory(root, a)
            with self.assertRaises(ValueError):
                builder.remove_generated_directory(a, a)

if __name__ == '__main__':
    unittest.main()
