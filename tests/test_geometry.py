"""CPU unit tests for MHINet coordinate geometry and guarded DLT."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from mhinet.geometry import (
    GuardReason,
    build_four_point_system,
    four_point_dlt,
    geometry_channels,
    guarded_four_point_dlt,
    image_corners,
    normalized_homography_to_pixel,
    normalized_to_pixel,
    pixel_homography_to_normalized,
    pixel_normalization_matrix,
    pixel_to_normalized,
    safe_project_points,
)


class CoordinateConventionTests(unittest.TestCase):
    def test_align_corners_false_pixel_centers_round_trip(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 6.0], [10.0, 6.0], [3.25, 4.5]],
            dtype=torch.float64,
        )
        normalized = pixel_to_normalized(points, (7, 11))
        torch.testing.assert_close(
            normalized[0],
            torch.tensor([-1.0 + 1.0 / 11.0, -1.0 + 1.0 / 7.0], dtype=torch.float64),
            rtol=0.0,
            atol=1e-15,
        )
        torch.testing.assert_close(normalized_to_pixel(normalized, (7, 11)), points)

        homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
        via_matrix = homogeneous @ pixel_normalization_matrix(
            (7, 11), dtype=torch.float64
        ).T
        torch.testing.assert_close(via_matrix[:, :2], normalized)

    def test_control_corner_order_is_tl_tr_bl_br(self) -> None:
        expected = torch.tensor(
            [[0.0, 0.0], [6.0, 0.0], [0.0, 4.0], [6.0, 4.0]],
            dtype=torch.float64,
        )
        actual = image_corners((5, 7), normalized=False, dtype=torch.float64)
        self.assertTrue(torch.equal(actual, expected))
        torch.testing.assert_close(
            normalized_to_pixel(
                image_corners((5, 7), normalized=True, dtype=torch.float64),
                (5, 7),
            ),
            expected,
        )

    def test_normalized_pixel_homography_round_trip_and_projection(self) -> None:
        h_norm = torch.tensor(
            [[1.02, 0.03, 0.08], [-0.02, 0.97, -0.05], [0.04, -0.03, 1.0]],
            dtype=torch.float64,
        )
        source_hw = (7, 11)
        target_hw = (9, 13)
        h_pixel = normalized_homography_to_pixel(h_norm, source_hw, target_hw)
        recovered = pixel_homography_to_normalized(h_pixel, source_hw, target_hw)
        torch.testing.assert_close(recovered, h_norm, atol=2e-15, rtol=2e-15)

        points_px = torch.tensor([[0.0, 0.0], [4.25, 3.5], [10.0, 6.0]], dtype=torch.float64)
        projected_px, valid_px, _ = safe_project_points(h_pixel, points_px)
        points_norm = pixel_to_normalized(points_px, source_hw)
        projected_norm, valid_norm, _ = safe_project_points(h_norm, points_norm)
        expected_px = normalized_to_pixel(projected_norm, target_hw)
        self.assertTrue(bool(valid_px.all() and valid_norm.all()))
        torch.testing.assert_close(projected_px, expected_px, atol=2e-13, rtol=2e-13)

    def test_safe_projection_masks_before_division(self) -> None:
        h = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        points = torch.tensor([[0.2, -0.3], [float("nan"), 1.0]], dtype=torch.float64)
        projected, valid, denominator = safe_project_points(h, points)
        self.assertFalse(bool(valid.any()))
        self.assertTrue(bool(torch.isfinite(projected).all()))
        self.assertTrue(torch.equal(projected, torch.zeros_like(projected)))
        self.assertEqual(float(denominator[0].detach()), 0.0)

    def test_geometry_channels_identity_and_clipped_flow(self) -> None:
        identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        positions, flow, valid = geometry_channels(identity, (3, 5))
        self.assertEqual(positions.shape, (1, 2, 3, 5))
        self.assertEqual(flow.shape, (1, 2, 3, 5))
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(flow, torch.zeros_like(flow), atol=1e-15, rtol=0.0)

        translated = identity.clone()
        translated[:, 0, 2] = 5.0
        _, clipped, valid = geometry_channels(translated, (3, 5))
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.equal(clipped[:, 0], torch.full_like(clipped[:, 0], 2.0)))
        self.assertTrue(torch.equal(clipped[:, 1], torch.zeros_like(clipped[:, 1])))


class FourPointDLTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = image_corners((31, 47), normalized=True, dtype=torch.float64)

    def test_system_is_exactly_8_by_8_h33_one_form(self) -> None:
        target = self.source + torch.tensor(
            [[0.01, -0.02], [0.02, -0.01], [-0.01, 0.03], [0.015, 0.02]],
            dtype=torch.float64,
        )
        matrix, rhs = build_four_point_system(self.source, target)
        self.assertEqual(matrix.shape, (8, 8))
        self.assertEqual(rhs.shape, (8,))
        x, y = self.source[0]
        u, v = target[0]
        expected_u = torch.tensor([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        expected_v = torch.tensor([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        torch.testing.assert_close(matrix[0], expected_u)
        torch.testing.assert_close(matrix[1], expected_v)

    def test_known_perspective_homography_reconstruction(self) -> None:
        expected_h = torch.tensor(
            [[1.01, 0.025, 0.07], [-0.018, 0.98, -0.04], [0.035, -0.022, 1.0]],
            dtype=torch.float64,
        )
        target, valid, _ = safe_project_points(expected_h, self.source)
        self.assertTrue(bool(valid.all()))
        result = guarded_four_point_dlt(
            self.source,
            target,
            minimum_supported_queries=0,
        )
        self.assertTrue(bool(result.accepted))
        self.assertEqual(int(result.reason), int(GuardReason.ACCEPTED))
        self.assertEqual(float(result.homography[2, 2]), 1.0)
        torch.testing.assert_close(result.homography, expected_h, atol=2e-12, rtol=2e-12)
        reprojection, valid, _ = safe_project_points(result.homography, self.source)
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(reprojection, target, atol=2e-12, rtol=2e-12)

    def test_dlt_gradcheck(self) -> None:
        target = (
            self.source
            + torch.tensor(
                [[0.011, -0.017], [0.019, -0.009], [-0.013, 0.021], [0.014, 0.026]],
                dtype=torch.float64,
            )
        ).requires_grad_()

        passed = torch.autograd.gradcheck(
            lambda dst: four_point_dlt(self.source, dst).square().sum(),
            (target,),
            eps=1e-6,
            atol=2e-5,
            rtol=2e-4,
            fast_mode=True,
        )
        self.assertTrue(passed)

    def test_mixed_batch_reason_codes_and_no_illegal_solve(self) -> None:
        legal_target = self.source.clone()
        degenerate_source = torch.zeros_like(self.source)
        source = torch.stack((self.source, degenerate_source, self.source))
        nonfinite_target = self.source.clone()
        nonfinite_target[0, 0] = float("nan")
        target = torch.stack((legal_target, legal_target, nonfinite_target))
        result = guarded_four_point_dlt(
            source,
            target,
            support_count=torch.tensor([16, 16, 16]),
        )
        self.assertEqual(result.accepted.tolist(), [True, False, False])
        self.assertEqual(
            result.reason.tolist(),
            [
                int(GuardReason.ACCEPTED),
                int(GuardReason.ILL_CONDITIONED_SYSTEM),
                int(GuardReason.NONFINITE_SYSTEM),
            ],
        )
        self.assertTrue(bool(torch.isfinite(result.homography).all()))

        with mock.patch("torch.linalg.solve_ex", wraps=torch.linalg.solve_ex) as solve_ex:
            rejected = guarded_four_point_dlt(
                degenerate_source,
                legal_target,
                minimum_supported_queries=0,
            )
        self.assertFalse(bool(rejected.accepted))
        self.assertEqual(int(rejected.reason), int(GuardReason.ILL_CONDITIONED_SYSTEM))
        solve_ex.assert_not_called()

    def test_no_support_has_explicit_reason(self) -> None:
        candidate_valid = torch.zeros(1, 9, 3, 4, dtype=torch.bool)
        result = guarded_four_point_dlt(
            self.source.unsqueeze(0),
            self.source.unsqueeze(0),
            candidate_valid=candidate_valid,
        )
        self.assertFalse(bool(result.accepted[0]))
        self.assertEqual(int(result.reason[0]), int(GuardReason.NO_VALID_SUPPORT))

    def test_nine_by_nine_denominator_postcheck_and_fallback_gradient(self) -> None:
        folding_h = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 1.0]],
            dtype=torch.float64,
        )
        # The denominator 0.6*x+1 remains positive.  Increasing perspective to
        # 1.2 makes it cross zero inside the normalized source image while the
        # four control-corner projections themselves remain finite.
        folding_h[2, 0] = 1.2
        target, corner_valid, _ = safe_project_points(folding_h, self.source)
        self.assertTrue(bool(corner_valid.all()))
        fallback = torch.eye(3, dtype=torch.float64, requires_grad=True)
        result = guarded_four_point_dlt(
            self.source,
            target,
            fallback_homography=fallback,
            minimum_supported_queries=0,
        )
        self.assertFalse(bool(result.accepted))
        self.assertEqual(
            int(result.reason),
            int(GuardReason.INVALID_PROJECTION_DENOMINATOR),
        )
        torch.testing.assert_close(result.homography, fallback)
        result.homography.sum().backward()
        self.assertTrue(torch.equal(fallback.grad, torch.ones_like(fallback)))


if __name__ == "__main__":
    unittest.main()
