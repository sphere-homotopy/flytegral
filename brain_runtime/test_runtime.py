import unittest
import numpy as np

from brain_runtime.runtime import sample_retina, project_counts, decode_readout, activity_features


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


    def test_activity_features_runs_reset_brain_and_returns_projected_rates(self):
        class FakeBrain:
            def __init__(self):
                self.n = 4
                self.ids = np.array([10, 11, 12, 13], dtype=np.int64)
                self.uv = np.array([[0, 0], [1, 0]], dtype=np.float32)
                self.retina = np.array([0, 1], dtype=np.int32)
                self.lamina = np.array([2], dtype=np.int32)
                self.sugar = np.array([3], dtype=np.int32)
                self.cursor = 0
                self.v = np.zeros(4, dtype=np.float32)
                self.g = np.ones(4, dtype=np.float32)
                self.drive = np.ones(4, dtype=np.float32)
                self.refractory = np.ones(4, dtype=np.int16)
                self.queue = np.ones((2, 4), dtype=np.int32)
                self.queue_count = np.ones(2, dtype=np.int32)
                self.counts = np.ones(4, dtype=np.int32)
                self.luminance = np.ones(2, dtype=np.float32)
                self.active = np.zeros(4, dtype=np.int32)
                self.active_flag = np.zeros(4, dtype=np.uint8)
                self.nactive = np.array([0], dtype=np.int32)
                self.total_spikes = 99
                self.sim_ms = 99
            def step(self, luminance, duration_ms):
                self.last_luminance = np.asarray(luminance)
                return np.array([1, 2, 0, 1], dtype=np.int32), 0.01

        brain = FakeBrain()
        stimulus = {
            'width': 2, 'height': 2,
            'luminance': [0.1, 0.9, 0.2, 0.8],
        }
        features, telemetry = activity_features(brain, stimulus, bins=8, frame_ms=20, repeats=2)
        self.assertEqual(features.shape, (8,))
        self.assertEqual(telemetry['totalSpikes'], 8)
        self.assertEqual(telemetry['simMs'], 40.0)
        np.testing.assert_allclose(brain.last_luminance, [0.1, 0.9])

    def test_decode_readout_clamps_to_slider_range(self):
        features = np.array([2.0, -1.0], dtype=np.float32)
        weights = np.array([2.0, 0.0], dtype=np.float32)
        self.assertEqual(decode_readout(features, weights, bias=0.2), 1.0)
        self.assertEqual(decode_readout(features, -weights, bias=-0.2), 0.0)


if __name__ == '__main__':
    unittest.main()
