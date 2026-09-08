"""CPU reference and autograd tests for H-guided local correlation."""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from mhinet.correlation import (
    HGuidedLocalCorrelation,
    candidate_offsets,
    h_guided_local_correlation,
)
from mhinet.geometry import (
    normalized_homography_to_pixel,
    pixel_grid,
    pixel_to_normalized,
    safe_project_points,
)


def direct_candidate_reference(
    source: torch.Tensor,
    target: torch.Tensor,
    homography: torch.Tensor,
    radius: int,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-grid one-candidate-at-a-time reference implementation."""

    batch, _, height_a, width_a = source.shape
    _, _, height_b, width_b = target.shape
    dtype = torch.float64 if torch.float64 in (source.dtype, target.dtype, homography.dtype) else torch.float32
    source = source.to(dtype)
    target = target.to(dtype)
    homography = homography.to(dtype)
    positions = pixel_grid((height_a, width_a), dtype=dtype).reshape(1, -1, 2).expand(batch, -1, -1)
    h_pixel = normalized_homography_to_pixel(
        homography, (height_a, width_a), (height_b, width_b)
    )
    projected, projection_valid, _ = safe_project_points(h_pixel, positions)

    source_norm = torch.linalg.vector_norm(source, dim=1)
    source_valid = torch.isfinite(source).all(dim=1) & (source_norm > epsilon)
    source_unit = source / source_norm.clamp_min(epsilon)[:, None]
    maps: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for offset in candidate_offsets(radius, dtype=dtype):
        candidate = projected + offset
        in_bounds = (
            projection_valid
            & (candidate[..., 0] >= 0.0)
            & (candidate[..., 0] <= width_b - 1.0)
            & (candidate[..., 1] >= 0.0)
            & (candidate[..., 1] <= height_b - 1.0)
        )
        safe_candidate = torch.where(in_bounds[..., None], candidate, torch.zeros_like(candidate))
        grid = pixel_to_normalized(safe_candidate, (height_b, width_b)).reshape(
            batch, height_a, width_a, 2
        )
        sampled = F.grid_sample(
            target,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_norm = torch.linalg.vector_norm(sampled, dim=1)
        sampled_valid = torch.isfinite(sampled).all(dim=1) & (sampled_norm > epsilon)
        sampled_unit = sampled / sampled_norm.clamp_min(epsilon)[:, None]
        valid = in_bounds.reshape(batch, height_a, width_a) & source_valid & sampled_valid
        cosine = (source_unit * sampled_unit).sum(dim=1)
        maps.append(torch.where(valid, cosine, torch.zeros_like(cosine)))
        masks.append(valid)
    return torch.stack(maps, dim=1), torch.stack(masks, dim=1)


class CorrelationConventionTests(unittest.TestCase):
    def test_candidate_order_is_dy_outer_dx_inner(self) -> None:
        expected = torch.tensor(
            [
                [-1.0, -1.0],
                [0.0, -1.0],
                [1.0, -1.0],
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        self.assertTrue(torch.equal(candidate_offsets(1), expected))

    def test_identity_border_mask_and_zero_norm_masks(self) -> None:
        torch.manual_seed(5)
        source = torch.randn(1, 3, 3, 4)
        target = torch.randn(1, 3, 3, 4)
        source[:, :, 1, 2] = 0.0
        target[:, :, 2, 3] = 0.0
        correlation, valid = h_guided_local_correlation(
            source,
            target,
            torch.eye(3).unsqueeze(0),
            radius=1,
            query_chunk_size=2,
        )
        self.assertEqual(correlation.shape, (1, 9, 3, 4))
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(
            torch.nonzero(valid[0, :, 0, 0], as_tuple=False).flatten().tolist(),
            [4, 5, 7, 8],
        )
        self.assertFalse(bool(valid[0, :, 1, 2].any()))
        # For identity and delta=(0,0), the target's exact zero descriptor is invalid.
        self.assertFalse(bool(valid[0, 4, 2, 3]))
        self.assertTrue(torch.equal(correlation[~valid], torch.zeros_like(correlation[~valid])))

    def test_sampled_target_is_renormalized_after_bilinear_sampling(self) -> None:
        source = torch.zeros(1, 2, 1, 3, dtype=torch.float64)
        source[:, 0] = 1.0
        target = torch.zeros_like(source)
        target[0, :, 0, 0] = torch.tensor([1.0, 0.0], dtype=torch.float64)
        target[0, :, 0, 1] = torch.tensor([0.0, 2.0], dtype=torch.float64)
        target[0, :, 0, 2] = torch.tensor([1.0, 1.0], dtype=torch.float64)
        h = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        h[:, 0, 2] = 1.0 / 3.0  # +0.5 target pixels for W=3.
        correlation, valid = h_guided_local_correlation(
            source,
            target,
            h,
            radius=0,
            query_chunk_size=None,
        )
        self.assertTrue(bool(valid[0, 0, 0, 0]))
        expected = 0.5 / torch.sqrt(torch.tensor(1.25, dtype=torch.float64))
        torch.testing.assert_close(correlation[0, 0, 0, 0], expected, atol=1e-12, rtol=1e-12)
        self.assertNotAlmostEqual(float(correlation[0, 0, 0, 0]), 0.5, places=6)

    def test_vectorized_chunked_matches_direct_reference(self) -> None:
        torch.manual_seed(19)
        source = torch.randn(2, 4, 3, 4, dtype=torch.float64)
        target = torch.randn(2, 4, 4, 5, dtype=torch.float64)
        homography = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
        homography[0, 0, 2] = 0.037
        homography[0, 1, 2] = -0.041
        homography[1, 0, 1] = 0.025
        homography[1, 2, 0] = 0.02

        actual, actual_valid = h_guided_local_correlation(
            source,
            target,
            homography,
            radius=1,
            query_chunk_size=3,
        )
        expected, expected_valid = direct_candidate_reference(
            source, target, homography, radius=1
        )
        self.assertTrue(torch.equal(actual_valid, expected_valid))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    def test_two_homographies_recompute_different_correlations(self) -> None:
        torch.manual_seed(29)
        source = torch.randn(1, 3, 3, 4)
        target = torch.randn(1, 3, 3, 4)
        identity = torch.eye(3).unsqueeze(0)
        translated = identity.clone()
        translated[:, 0, 2] = 0.15
        module = HGuidedLocalCorrelation(radius=1, query_chunk_size=4)
        first, _ = module(source, target, identity)
        second, _ = module(source, target, translated)
        self.assertFalse(torch.allclose(first, second))


class CorrelationAutogradTests(unittest.TestCase):
    def test_chunk_loop_uses_one_activation_checkpoint_context(self) -> None:
        source = torch.randn(1, 3, 3, 4, requires_grad=True)
        target = torch.randn(1, 3, 4, 5, requires_grad=True)
        homography = torch.eye(3).unsqueeze(0).requires_grad_()
        with mock.patch(
            "mhinet.correlation.checkpoint", wraps=torch_checkpoint
        ) as checkpoint_call:
            correlation, _ = h_guided_local_correlation(
                source,
                target,
                homography,
                radius=1,
                query_chunk_size=3,
                activation_checkpoint=True,
            )
            correlation.square().sum().backward()
        self.assertEqual(checkpoint_call.call_count, 1)

    def test_activation_checkpoint_matches_output_and_gradient(self) -> None:
        torch.manual_seed(30)
        base_source = torch.randn(1, 3, 3, 4, dtype=torch.float64)
        base_target = torch.randn(1, 3, 4, 5, dtype=torch.float64)
        base_h = torch.tensor(
            [[[1.0, 0.01, 0.023], [-0.02, 1.0, -0.019], [0.01, 0.005, 1.0]]],
            dtype=torch.float64,
        )
        results = []
        for enabled in (False, True):
            source = base_source.clone().requires_grad_()
            target = base_target.clone().requires_grad_()
            homography = base_h.clone().requires_grad_()
            correlation, valid = h_guided_local_correlation(
                source,
                target,
                homography,
                radius=1,
                query_chunk_size=3,
                activation_checkpoint=enabled,
            )
            correlation.square().sum().backward()
            results.append(
                (
                    correlation.detach(),
                    valid,
                    source.grad.detach(),
                    target.grad.detach(),
                    homography.grad.detach(),
                )
            )
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        for plain, checkpointed in zip(results[0][::2], results[1][::2]):
            torch.testing.assert_close(plain, checkpointed, atol=3e-11, rtol=3e-11)
        torch.testing.assert_close(results[0][3], results[1][3], atol=3e-11, rtol=3e-11)

    def test_chunked_and_unchunked_outputs_and_gradients_match(self) -> None:
        torch.manual_seed(31)
        source_base = torch.randn(1, 3, 3, 4, dtype=torch.float64)
        target_base = torch.randn(1, 3, 4, 5, dtype=torch.float64)
        h_base = torch.tensor(
            [[[1.0, 0.02, 0.031], [-0.01, 1.0, -0.027], [0.015, -0.01, 1.0]]],
            dtype=torch.float64,
        )
        weight = torch.randn(1, 9, 3, 4, dtype=torch.float64)

        gradients: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        outputs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for chunk_size in (2, 1000):
            source = source_base.clone().requires_grad_()
            target = target_base.clone().requires_grad_()
            homography = h_base.clone().requires_grad_()
            correlation, valid = h_guided_local_correlation(
                source,
                target,
                homography,
                radius=1,
                query_chunk_size=chunk_size,
            )
            (correlation * weight).sum().backward()
            outputs.append(correlation.detach())
            masks.append(valid)
            gradients.append((source.grad.detach(), target.grad.detach(), homography.grad.detach()))

        self.assertTrue(torch.equal(masks[0], masks[1]))
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-12, rtol=2e-12)
        for chunked, unchunked in zip(gradients[0], gradients[1]):
            self.assertTrue(bool(torch.isfinite(chunked).all()))
            self.assertGreater(float(chunked.norm()), 0.0)
            torch.testing.assert_close(chunked, unchunked, atol=3e-11, rtol=3e-11)

    def test_feature_and_homography_gradcheck(self) -> None:
        torch.manual_seed(37)
        # Mapping a 2x2 normalized source grid into a 4x4 target lands at
        # half-pixel positions, away from grid_sample's integer kinks/borders.
        source = torch.randn(1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
        target = torch.randn(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
        homography = torch.tensor(
            [[[1.0, 0.015, 0.01], [-0.012, 1.0, -0.008], [0.01, -0.006, 1.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )

        def scalar_correlation(src: torch.Tensor, tgt: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            correlation, valid = h_guided_local_correlation(
                src,
                tgt,
                h,
                radius=0,
                query_chunk_size=2,
            )
            self.assertTrue(bool(valid.all()))
            return correlation.square().sum()

        passed = torch.autograd.gradcheck(
            scalar_correlation,
            (source, target, homography),
            eps=1e-6,
            atol=3e-5,
            rtol=3e-4,
            fast_mode=True,
        )
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
