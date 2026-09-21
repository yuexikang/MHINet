from __future__ import annotations

import torch

from mhinet.downstream.loma_reference.fine_matcher import (
    enumerate_d2_children,
    fine_prior_normalized,
    local_window_candidates,
    match_d2,
)
from mhinet.downstream.loma_reference.matching_utils import (
    flat_indices_to_centers,
)


def test_no_coarse_matches_is_explicit_and_safe() -> None:
    descriptor = torch.empty((256, 392, 392), device="meta")
    mask = torch.ones((392, 392), dtype=torch.bool, device="meta")
    result = match_d2(
        descriptor,
        descriptor,
        mask,
        mask,
        torch.empty(0, dtype=torch.long, device="meta"),
        torch.empty(0, dtype=torch.long, device="meta"),
        torch.empty(0, device="meta"),
        torch.eye(3, device="meta"),
        fine_prior="h_residual",
        window=5,
        temperature=0.05,
        threshold=0,
        selection="local_mutual",
        local_topk=2,
        max_final_matches=100,
    )
    assert result.source_flat.numel() == 0
    assert result.diagnostics["selection"] == "no_coarse_matches"


def test_global_dedup_rejects_repeated_targets() -> None:
    # The deterministic greedy behavior itself is covered without allocating a
    # 392x392x256 descriptor tensor in the fast CPU suite.
    from mhinet.downstream.loma_reference.matching_utils import (
        stable_greedy_one_to_one,
    )

    source = torch.tensor([1, 2, 3, 3])
    target = torch.tensor([7, 7, 8, 9])
    confidence = torch.tensor([0.9, 0.8, 0.7, 0.6])
    keep = stable_greedy_one_to_one(source, target, confidence, cap=10)
    assert keep.tolist() == [0, 2]


def test_direct_prior_matches_h_residual_for_identity_plus_translation() -> None:
    coarse_source = torch.tensor([20 * 98 + 10])
    coarse_target = torch.tensor([23 * 98 + 15])
    mask = torch.ones((392, 392), dtype=torch.bool)
    children, parent = enumerate_d2_children(coarse_source, mask)
    direct, direct_valid = fine_prior_normalized(
        coarse_source,
        coarse_target,
        children,
        parent,
        None,
        "direct_d8",
    )
    h_residual, h_valid = fine_prior_normalized(
        coarse_source,
        coarse_target,
        children,
        parent,
        torch.eye(3),
        "h_residual",
    )
    assert direct_valid.all() and h_valid.all()
    torch.testing.assert_close(direct, h_residual, atol=2e-7, rtol=0)


def test_direct_prior_preserves_exact_d8_to_d2_child_offsets() -> None:
    coarse_source = torch.tensor([20 * 98 + 10])
    coarse_target = torch.tensor([23 * 98 + 15])
    mask = torch.ones((392, 392), dtype=torch.bool)
    children, parent = enumerate_d2_children(coarse_source, mask)
    prior, valid = fine_prior_normalized(
        coarse_source,
        coarse_target,
        children,
        parent,
        None,
        "direct_d8",
    )
    center_uv, _, _ = local_window_candidates(prior[valid], mask, 9)
    offsets = torch.arange(4)
    offset_v, offset_u = torch.meshgrid(offsets, offsets, indexing="ij")
    expected = torch.stack(
        (
            4 * 15 + offset_u.reshape(-1),
            4 * 23 + offset_v.reshape(-1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(center_uv, expected)


def test_window_center_and_border_clipping_are_unchanged() -> None:
    mask = torch.ones((392, 392), dtype=torch.bool)
    prior = flat_indices_to_centers(torch.tensor([0]), 392, 392)
    center_uv, candidates, valid = local_window_candidates(prior, mask, 9)
    assert center_uv.tolist() == [[0, 0]]
    assert candidates.shape == valid.shape == (1, 81)
    assert candidates[0, 40].item() == 0
    assert valid[0, 40]
    assert valid.sum().item() == 25


def test_invalid_d2_children_are_filtered_before_prior_computation() -> None:
    coarse_source = torch.tensor([7 * 98 + 5])
    mask = torch.zeros((392, 392), dtype=torch.bool)
    expected = (7 * 4 + 2) * 392 + (5 * 4 + 3)
    mask.reshape(-1)[expected] = True
    children, parent = enumerate_d2_children(coarse_source, mask)
    assert children.tolist() == [expected]
    assert parent.tolist() == [0]
