"""Model-free regression tests: unavailable VRAM must not become a fake peak."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from multimodal_canary_vulkan import VulkanSampler


class VulkanSamplerTests(unittest.TestCase):
    def test_missing_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'server.stderr.log').write_text('model buffer size = 123.00 MiB\n')
            sampler = VulkanSampler(folder)
            sampler.drm_device = None
            sampler.VRAMMON = str(folder / 'missing')
            self.assertIsNone(sampler._probe_vrammon())
            self.assertEqual(sampler._probe_static((folder / 'server.stderr.log').read_text()), 123)
            sampler.rows = [(0, 'loading', None, None), (1, 'decode', None, None)]
            with patch.object(sampler, 'join'):
                result = sampler.finish()
            self.assertIsNone(result['peak_vram_mib'])
            self.assertEqual(result['vram_sample_count'], 0)
            self.assertEqual(result['logged_allocation_estimate_mib'], 123)
            self.assertEqual(result['sampling_errors'], [])

    def test_drm_samples_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'mem_info_vram_total').write_text(str(16 * 1048576))
            (folder / 'mem_info_vram_used').write_text(str(5 * 1048576))
            with patch.dict(os.environ, {'KVMEM_DRM_DEVICE': directory}):
                sampler = VulkanSampler(folder)
            self.assertEqual(sampler._probe_vrammon(), (16, 5))
            sampler.rows = [(0, 'decode', 5, 11), (1, 'decode', None, None),
                            (2, 'decode', 7, 9)]
            with patch.object(sampler, 'join'):
                result = sampler.finish()
            self.assertEqual(result['peak_vram_mib'], 7)
            self.assertEqual(result['vram_sample_count'], 2)
            self.assertEqual(result['phase_metrics']['decode']['min_free_mib'], 9)
            self.assertEqual(result['vram_measurement_source'], directory)


if __name__ == '__main__':
    unittest.main()
