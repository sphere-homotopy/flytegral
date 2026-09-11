import unittest
import numpy as np

from brain_runtime.runtime import sample_retina, project_counts, decode_readout


class RuntimeMathTests(unittest.TestCase):
    def test_sample_retina_maps_uv_to_raster(self):
        raster = np.array([
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
        ], dtype=np.float32)
        uv = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        sampled = sample_retina(raster, uv)
        np.testing.assert_allclose(sampled, np.array([0.0, 0.8, 0.4], dtype=np.float32))

    def test_project_counts_is_deterministic_and_size_bounded(self):
        counts = np.array([1, 2, 3, 4], dtype=np.int32)
        ids = np.array([10, 11, 12, 13], dtype=np.int64)
        first = project_counts(counts, ids, bins=8)
        second = project_counts(counts, ids, bins=8)
        self.assertEqual(first.shape, (8,))
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.abs(first).sum()), 10.0)

    def test_decode_readout_clamps_to_slider_range(self):
        features = np.array([2.0, -1.0], dtype=np.float32)
        weights = np.array([2.0, 0.0], dtype=np.float32)
        self.assertEqual(decode_readout(features, weights, bias=0.2), 1.0)
        self.assertEqual(decode_readout(features, -weights, bias=-0.2), 0.0)


if __name__ == '__main__':
    unittest.main()
