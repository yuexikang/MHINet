import unittest

import numpy as np

from mhinet.dataio.generate_rgb512 import (
    augment_b,
    l0_gradient_smoothing,
    normalized_dlt,
    project_points,
    sample_geometry,
)


class RGB512GenerationTests(unittest.TestCase):
    def test_normalized_dlt_exact(self):
        source = np.array([[0, 0], [639, 0], [639, 639], [0, 639]], dtype=np.float64)
        target = np.array([[0, 0], [511, 0], [511, 511], [0, 511]], dtype=np.float64)
        matrix = normalized_dlt(source, target)
        np.testing.assert_allclose(project_points(matrix, source), target, atol=1e-8)

    def test_sample_geometry_policy(self):
        geometry = sample_geometry(np.random.default_rng(7), 640, 640)
        self.assertTrue(2.0 <= geometry["area_ratio_A_to_B"] <= 4.0)
        self.assertGreaterEqual(min(geometry["overlap_fractions"]), 0.20)
        np.testing.assert_allclose(
            project_points(geometry["H_A_to_B"], geometry["corner_A"]),
            geometry["corner_B"],
            atol=1e-6,
        )

    def test_l0_is_deterministic_and_finite(self):
        image = np.random.default_rng(3).random((32, 32, 3), dtype=np.float32)
        first = l0_gradient_smoothing(image, 5e-4, beta_max=1.0)
        second = l0_gradient_smoothing(image, 5e-4, beta_max=1.0)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)

    def test_augmentation_reproducible(self):
        image = np.full((64, 64, 3), 0.5, dtype=np.float32)
        first = augment_b(image, 123)
        second = augment_b(image, 123)
        self.assertEqual(first[1:], second[1:])
        np.testing.assert_array_equal(first[0], second[0])


if __name__ == "__main__":
    unittest.main()
