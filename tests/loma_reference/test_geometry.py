from __future__ import annotations

import pytest
import torch

from mhinet.downstream.loma_reference.matching_utils import (
    centers_to_native,
    normalized_cell_centers,
    project_normalized,
)
from mhinet.downstream.loma_reference.overlap_masks import (
    predicted_overlap_masks,
)


def _h(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32).reshape(1, 3, 3)


def test_align_corners_false_cell_centers_and_non_square_restoration() -> None:
    grid = normalized_cell_centers(2, 4, device="cpu")
    torch.testing.assert_close(grid[0, 0], torch.tensor([-0.75, -0.5]))
    torch.testing.assert_close(grid[-1, -1], torch.tensor([0.75, 0.5]))
    native = centers_to_native(grid.reshape(-1, 2), (800, 500)).reshape(2, 4, 2)
    torch.testing.assert_close(native[0, 0], torch.tensor([99.5, 124.5]))
    torch.testing.assert_close(native[-1, -1], torch.tensor([699.5, 374.5]))


@pytest.mark.parametrize(
    "homography",
    [
        torch.eye(3),
        torch.tensor([[1.0, 0.0, 0.2], [0.0, 1.0, -0.1], [0.0, 0.0, 1.0]]),
        torch.tensor([[1.2, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]]),
        torch.tensor([[0.94, -0.25, 0.04], [0.25, 0.94, -0.03], [0.02, -0.01, 1.0]]),
    ],
)
def test_forward_backward_homography_consistency(homography: torch.Tensor) -> None:
    points = torch.tensor([[-0.6, -0.4], [0.0, 0.0], [0.55, 0.35]])
    warped, valid = project_normalized(points, homography)
    restored, inverse_valid = project_normalized(warped, torch.linalg.inv(homography))
    assert valid.all() and inverse_valid.all()
    torch.testing.assert_close(restored, points, atol=2e-6, rtol=2e-6)


def test_identity_overlap_is_full_and_translation_clips_directionally() -> None:
    identity = predicted_overlap_masks(torch.eye(3).reshape(1, 3, 3), 98)
    assert identity.a.all() and identity.b.all()
    translated = predicted_overlap_masks(
        _h([1, 0, 0.5], [0, 1, 0], [0, 0, 1]), 98
    )
    assert 0 < translated.a.sum() < identity.a.sum()
    assert translated.a.sum() == translated.b.sum()
    # Positive A->B x translation removes the right of A and the left of B.
    assert translated.a[:, -1].sum() == 0
    assert translated.b[:, 0].sum() == 0


def test_coarse_residual_corrects_pointwise_h_error() -> None:
    h = torch.tensor([[1.0, 0.0, 0.08], [0.0, 1.0, -0.04], [0.0, 0.0, 1.0]])
    coarse_a = torch.tensor([[0.0, 0.0]])
    observed_b = torch.tensor([[0.13, -0.01]])
    projected_coarse, _ = project_normalized(coarse_a, h)
    residual = observed_b - projected_coarse
    child = torch.tensor([[0.02, 0.03]])
    projected_child, _ = project_normalized(child, h)
    prior = projected_child + residual
    torch.testing.assert_close(prior, torch.tensor([[0.15, 0.02]]), atol=1e-6, rtol=0)
