from __future__ import annotations

import unittest

import torch

from mhinet.ops.geometry import image_corners
from mhinet.engine.losses import sequence_corner_l1
from mhinet.engine.metrics import homography_trajectory_metrics


class LossAndMetricTests(unittest.TestCase):
    def test_isolated_nonfinite_decoder_output_is_not_hidden_by_safe_proposal(self) -> None:
        outputs = {
            "proposal_Q_norm": torch.zeros(1, 2, 4, 2),
            "decoder_output_finite": torch.tensor([[True, False]]),
            "stage1_valid": torch.tensor([True]),
        }
        with self.assertRaises(FloatingPointError):
            sequence_corner_l1(outputs, torch.eye(3).unsqueeze(0))

    def _identity_outputs(self) -> dict[str, torch.Tensor]:
        corners = image_corners((784, 784)).reshape(1, 1, 4, 2)
        return {
            "H0_norm": torch.eye(3).reshape(1, 3, 3),
            "H_updates_norm": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 6, 1, 1),
            "proposal_Q_norm": corners.repeat(1, 6, 1, 1),
            "stage1_valid": torch.tensor([True]),
        }

    def test_default_loss_is_equal_weight_coordinate_l1(self) -> None:
        outputs = self._identity_outputs()
        outputs["proposal_Q_norm"] = outputs["proposal_Q_norm"].clone()
        outputs["proposal_Q_norm"][..., 0] += 4.0 / 784.0  # +2 target pixels in x
        result = sequence_corner_l1(outputs, torch.eye(3).reshape(1, 3, 3))
        self.assertFalse(bool(result["skip_step"]))
        self.assertEqual(result["updates"], 6)
        torch.testing.assert_close(result["loss"], torch.tensor(1.0), atol=2e-5, rtol=0)
        torch.testing.assert_close(
            result["per_update_corner_l1_px"], torch.ones(1, 6), atol=2e-5, rtol=0
        )

    def test_all_invalid_requests_skip(self) -> None:
        outputs = self._identity_outputs()
        outputs["proposal_Q_norm"] = outputs["proposal_Q_norm"].requires_grad_()
        outputs["stage1_valid"] = torch.tensor([False])
        result = sequence_corner_l1(outputs, torch.eye(3).reshape(1, 3, 3))
        self.assertTrue(bool(result["skip_step"]))
        self.assertEqual(result["valid_pairs"], 0)
        self.assertEqual(float(result["loss"].detach()), 0.0)

    def test_identity_trajectory_metrics(self) -> None:
        outputs = self._identity_outputs()
        metrics = homography_trajectory_metrics(
            outputs, torch.eye(3).reshape(1, 3, 3)
        )
        torch.testing.assert_close(metrics["trajectory_mace_px"], torch.zeros(1, 7))
        torch.testing.assert_close(
            metrics["trajectory_grid5_error_px"], torch.zeros(1, 7)
        )


if __name__ == "__main__":
    unittest.main()
