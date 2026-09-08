from __future__ import annotations

import unittest

import torch

from mhinet.evaluate import _trajectory_errors


class EvaluationGeometryTests(unittest.TestCase):
    def test_identity_trajectory_has_zero_input_and_native_error(self) -> None:
        trajectory = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
        result = _trajectory_errors(
            trajectory,
            torch.eye(3).unsqueeze(0),
            native_target_hw=(611, 937),
        )
        self.assertTrue(bool(result["geometry_valid"].all()))
        self.assertLess(float(result["mace_input_px"].abs().max()), 1e-6)
        self.assertLess(float(result["mace_native_target_px"].abs().max()), 1e-6)
        self.assertLess(float(result["grid5_input_px"].abs().max()), 1e-6)
        self.assertLess(float(result["grid5_native_target_px"].abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
