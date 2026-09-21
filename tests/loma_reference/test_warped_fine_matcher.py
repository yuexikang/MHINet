from __future__ import annotations

import pytest
import torch

from mhinet.downstream.loma_reference.configs import (
    MatcherConfig,
    registered_config,
)
from mhinet.downstream.loma_reference.matching_utils import (
    flat_indices_to_centers,
    project_normalized,
)
from mhinet.downstream.loma_reference.warped_fine_matcher import (
    PATCH_POINTS,
    continuous_mask_validity,
    d8_children_lattice,
    geometry_edge_mask,
    h_warped_target_lattice,
    masked_dual_softmax,
    match_d2_h_warped_sym4,
    normalized_to_feature_uv,
    sample_bilinear,
    select_one_per_parent,
)


def test_d8_children_are_the_exact_four_by_four_d2_lattice() -> None:
    coarse = torch.tensor([20 * 98 + 10])
    child_flat, child_norm = d8_children_lattice(coarse)
    offsets = torch.arange(4)
    offset_v, offset_u = torch.meshgrid(offsets, offsets, indexing="ij")
    expected = (20 * 4 + offset_v.reshape(-1)) * 392 + (
        10 * 4 + offset_u.reshape(-1)
    )
    assert child_flat.shape == (1, PATCH_POINTS)
    assert child_flat[0].tolist() == expected.tolist()
    torch.testing.assert_close(
        child_norm[0], flat_indices_to_centers(expected, 392, 392)
    )
    torch.testing.assert_close(
        normalized_to_feature_uv(child_norm[0], 392),
        torch.stack(
            (10 * 4 + offset_u.reshape(-1), 20 * 4 + offset_v.reshape(-1)),
            dim=-1,
        ).float(),
        atol=2e-5,
        rtol=0,
    )


@pytest.mark.parametrize(
    "homography",
    [
        torch.tensor([[1.0, 0.0, 0.12], [0.0, 1.0, -0.07], [0.0, 0.0, 1.0]]),
        torch.tensor([[1.6, 0.0, 0.02], [0.0, 0.7, -0.03], [0.0, 0.0, 1.0]]),
        torch.tensor([[0.92, -0.31, 0.04], [0.27, 1.08, -0.02], [0.0, 0.0, 1.0]]),
        torch.tensor([[1.1, 0.18, 0.03], [-0.08, 0.95, -0.01], [0.04, -0.03, 1.0]]),
    ],
)
def test_every_target_lattice_point_uses_full_homography(
    homography: torch.Tensor,
) -> None:
    coarse_a = torch.tensor([42 * 98 + 37])
    coarse_b = torch.tensor([45 * 98 + 41])
    _, source = d8_children_lattice(coarse_a)
    target, valid = h_warped_target_lattice(
        coarse_a, coarse_b, source, homography
    )
    center_a = flat_indices_to_centers(coarse_a, 98, 98)
    center_b = flat_indices_to_centers(coarse_b, 98, 98)
    projected_center, center_valid = project_normalized(center_a, homography)
    projected_source, source_valid = project_normalized(source[0], homography)
    expected = projected_source + center_b - projected_center
    assert center_valid.all() and source_valid.all() and valid.all()
    torch.testing.assert_close(target[0], expected, atol=2e-6, rtol=2e-6)


def test_warped_lattice_is_not_axis_aligned_when_h_has_a_jacobian() -> None:
    coarse_a = torch.tensor([40 * 98 + 40])
    coarse_b = torch.tensor([43 * 98 + 44])
    _, source = d8_children_lattice(coarse_a)
    homography = torch.tensor(
        [[1.45, 0.25, 0.03], [-0.15, 0.8, -0.02], [0.02, -0.01, 1.0]]
    )
    target, valid = h_warped_target_lattice(
        coarse_a, coarse_b, source, homography
    )
    assert valid.all()
    source_horizontal = source[0, 1] - source[0, 0]
    target_horizontal = target[0, 1] - target[0, 0]
    assert not torch.allclose(source_horizontal, target_horizontal, atol=1e-5)
    assert abs(float(target_horizontal[1])) > 1e-5


def test_bilinear_sampling_retains_fractional_coordinates_without_rounding() -> None:
    height, width = 5, 7
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    feature = (2.0 * xx + 3.0 * yy)[None].float()
    uv = torch.tensor([[2.25, 1.5], [4.6, 2.2]])
    points = torch.stack(
        (
            2.0 * (uv[:, 0] + 0.5) / width - 1.0,
            2.0 * (uv[:, 1] + 0.5) / height - 1.0,
        ),
        dim=-1,
    )
    sampled = sample_bilinear(feature, points)[:, 0]
    torch.testing.assert_close(sampled, 2.0 * uv[:, 0] + 3.0 * uv[:, 1])


def test_continuous_mask_validity_rejects_mask_edge_and_outside() -> None:
    mask = torch.ones((392, 392), dtype=torch.bool)
    mask[100, 100] = False
    uv = torch.tensor([[20.25, 30.5], [99.75, 99.75], [-0.1, 12.0]])
    points = torch.stack(
        (
            2.0 * (uv[:, 0] + 0.5) / 392 - 1.0,
            2.0 * (uv[:, 1] + 0.5) / 392 - 1.0,
        ),
        dim=-1,
    )
    valid, in_image, mask_valid = continuous_mask_validity(
        mask, points, torch.ones(3, dtype=torch.bool)
    )
    assert valid.tolist() == [True, False, False]
    assert in_image.tolist() == [True, True, False]
    assert mask_valid.tolist() == [True, False, False]


def test_masked_dual_softmax_is_finite_and_invalid_parent_is_zero() -> None:
    logits = torch.zeros((2, PATCH_POINTS, PATCH_POINTS), dtype=torch.float16)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    valid[0, :2, :2] = True
    probability = masked_dual_softmax(logits, valid)
    assert probability.dtype == torch.float32
    assert torch.isfinite(probability).all()
    torch.testing.assert_close(
        probability[0, :2, :2], torch.full((2, 2), 0.25)
    )
    assert not probability[0, 2:].any()
    assert not probability[1].any()


@pytest.mark.parametrize(
    ("radius", "expected_edges"),
    [(0, 16), (1, 100), (2, 196), (None, 256)],
)
def test_geometry_edge_mask_uses_lattice_chebyshev_radius(
    radius: int | None, expected_edges: int
) -> None:
    mask = geometry_edge_mask(radius, device="cpu")
    assert mask.shape == (PATCH_POINTS, PATCH_POINTS)
    assert int(mask.sum()) == expected_edges
    assert torch.equal(mask, mask.T)
    assert mask.diagonal().all()
    # Flat indices 0=(0,0), 5=(1,1), 10=(2,2), and 15=(3,3).
    if radius is not None:
        assert bool(mask[0, 5]) == (radius >= 1)
        assert bool(mask[0, 10]) == (radius >= 2)
        assert not mask[0, 15]
    else:
        assert mask.all()


def test_geometry_mask_is_applied_before_dual_softmax() -> None:
    logits = torch.zeros((1, PATCH_POINTS, PATCH_POINTS))
    logits[0, 0, 15] = 100.0
    valid = geometry_edge_mask(1, device="cpu")[None]
    probability = masked_dual_softmax(logits, valid)
    assert probability[0, 0, 15] == 0
    assert torch.isfinite(probability).all()
    _, source, target, _ = select_one_per_parent(probability, valid)
    source_row, source_column = divmod(int(source[0]), 4)
    target_row, target_column = divmod(int(target[0]), 4)
    assert max(abs(source_row - target_row), abs(source_column - target_column)) <= 1


def test_one_flat_argmax_is_selected_per_valid_parent_with_stable_ties() -> None:
    confidence = torch.zeros((3, PATCH_POINTS, PATCH_POINTS))
    valid = torch.zeros_like(confidence, dtype=torch.bool)
    valid[0, 0, 0] = True
    valid[0, 2, 3] = True
    confidence[0, 2, 3] = 0.9
    valid[2, 1, 1] = True
    valid[2, 1, 2] = True
    confidence[2, 1, 1] = confidence[2, 1, 2] = 0.7
    parent, source, target, score = select_one_per_parent(confidence, valid)
    assert parent.tolist() == [0, 2]
    assert source.tolist() == [2, 1]
    assert target.tolist() == [3, 1]
    torch.testing.assert_close(score, torch.tensor([0.9, 0.7]))


def test_empty_coarse_input_is_safe_without_real_descriptors() -> None:
    descriptor = torch.empty((256, 392, 392), device="meta")
    mask = torch.ones((392, 392), dtype=torch.bool, device="meta")
    result = match_d2_h_warped_sym4(
        descriptor,
        descriptor,
        mask,
        mask,
        torch.empty(0, dtype=torch.long, device="meta"),
        torch.empty(0, dtype=torch.long, device="meta"),
        torch.empty(0, device="meta"),
        torch.eye(3, device="meta"),
        temperature=0.03,
        threshold=0.0,
        max_final_matches=6000,
    )
    assert result.parent_coarse.numel() == 0
    assert result.diagnostics["selection"] == "no_coarse_matches"
    assert result.diagnostics["global_dedup_enabled"] is False


def test_c3_p6_is_a_frozen_h_warped_symmetric_configuration() -> None:
    config = registered_config("C3-P6")
    assert config.coarse_policy == "mutual"
    assert config.run_fine
    assert config.fine_mode == "h_warped_sym4"
    assert config.fine_prior == "h_residual"
    assert config.fine_geometry_radius is None
    frozen = registered_config(
        "C3-P6", coarse_temperature=0.03, fine_temperature=0.03
    )
    assert frozen.sha256() == "eae276aafaee4a4107ff4559afad00904718c3a08791c28e263f4b0b075b7562"
    with pytest.raises(ValueError, match="reserved"):
        MatcherConfig(result_id="C3-P6", fine_mode="expanded_local")
    with pytest.raises(ValueError, match="requires the H-residual"):
        MatcherConfig(
            result_id="C3-P4",
            run_fine=True,
            fine_mode="h_warped_sym4",
            fine_prior="direct_d8",
        )


@pytest.mark.parametrize(
    ("result_id", "radius"),
    [("C3-P7-D0", 0), ("C3-P7-D1", 1), ("C3-P7-D2", 2)],
)
def test_c3_p7_result_id_freezes_geometry_radius(
    result_id: str, radius: int
) -> None:
    config = registered_config(result_id)
    assert config.coarse_policy == "mutual"
    assert config.run_fine
    assert config.fine_mode == "h_warped_sym4"
    assert config.fine_geometry_radius == radius
    with pytest.raises(ValueError, match="disagree"):
        registered_config(result_id, fine_geometry_radius=(radius + 1) % 3)
