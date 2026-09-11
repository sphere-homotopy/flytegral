import unittest
import numpy as np

from brain_runtime.training import fit_ridge, generate_problem, answer_to_slider, render_graph_raster


class TrainingTests(unittest.TestCase):
    def test_python_problem_generator_matches_browser_distribution_for_seed(self):
        problem = generate_problem(12345)
        self.assertEqual(problem['coefficients'], [2.25, -1.0, 0.0, 1.5])
        self.assertEqual(problem['interval'], [-0.5, 1.0])
        self.assertAlmostEqual(problem['target'], 3.3515625)
        self.assertEqual(problem['answerRange'], [-7.5, 7.5])
        self.assertEqual(problem['graphDomain'], [-2.5, 2.5])
        self.assertAlmostEqual(answer_to_slider(problem['target'], problem['answerRange']), 0.7234375)

    def test_python_raster_matches_browser_encoder_summary(self):
        problem = generate_problem(12345)
        raster = render_graph_raster(problem, width=64, height=40)
        self.assertEqual(raster.shape, (40, 64))
        self.assertAlmostEqual(float(raster.sum()), 2346.6, places=2)
        self.assertAlmostEqual(float(raster.min()), 0.08, places=5)
        self.assertAlmostEqual(float(raster.max()), 1.0, places=5)

    def test_ridge_fits_simple_linear_mapping(self):
        x = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ], dtype=np.float32)
        y = 0.2 + 0.3 * x[:, 0] - 0.1 * x[:, 1]
        model = fit_ridge(x, y, alpha=1e-6)
        pred = model.predict(x)
        np.testing.assert_allclose(pred, y, atol=1e-4)
        self.assertEqual(model.weights.shape, (2,))


if __name__ == '__main__':
    unittest.main()
