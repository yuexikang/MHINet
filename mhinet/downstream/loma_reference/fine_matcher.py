"""Scale-2 local matching with switchable H-residual or Direct-D8 priors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .configs import FinePrior, FineSelection
from .matching_utils import (
    flat_indices_to_centers,
    project_normalized,
    stable_greedy_one_to_one,
)


@dataclass(frozen=True)
class FineMatchResult:
    source_flat: torch.Tensor
    target_flat: torch.Tensor
    confidence: torch.Tensor
    parent_coarse: torch.Tensor
    evaluated_source_flat: torch.Tensor
    evaluated_prior_b_norm: torch.Tensor
    evaluated_prior_b_center_uv: torch.Tensor
    diagnostics: dict[str, float | int | str]


def _group_max(values: torch.Tensor, groups: torch.Tensor, group_count: int) -> torch.Tensor:
    output = torch.full(
        (group_count,), -torch.inf, dtype=values.dtype, device=values.device
    )
    output.scatter_reduce_(0, groups, values, reduce="amax", include_self=True)
    return output


def _group_logsumexp(
    values: torch.Tensor, groups: torch.Tensor, group_count: int
) -> torch.Tensor:
    maxima = _group_max(values, groups, group_count)
    shifted = torch.exp(values - maxima[groups])
    totals = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    totals.scatter_add_(0, groups, shifted)
    return maxima + torch.log(totals.clamp_min(1e-30))


def _empty(device: torch.device, reason: str, fine_prior: FinePrior) -> FineMatchResult:
    indices = torch.empty(0, dtype=torch.long, device=device)
    return FineMatchResult(
        source_flat=indices,
        target_flat=indices.clone(),
        confidence=torch.empty(0, dtype=torch.float32, device=device),
        parent_coarse=indices.clone(),
        evaluated_source_flat=indices.clone(),
        evaluated_prior_b_norm=torch.empty((0, 2), dtype=torch.float32, device=device),
        evaluated_prior_b_center_uv=torch.empty((0, 2), dtype=torch.long, device=device),
        diagnostics={
            "selection": reason,
            "prior": fine_prior,
            "coarse_pairs": 0,
            "children_enumerated": 0,
            "children_valid": 0,
            "fine_edges_evaluated": 0,
            "raw_selected": 0,
            "thresholded_selected": 0,
            "deduplicated_selected": 0,
            "duplicate_rejections": 0,
        },
    )


def enumerate_d2_children(
    coarse_source_flat: torch.Tensor, mask_a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate valid 4x4 D2 children and their D8 parent indices."""

    if mask_a.shape != (392, 392):
        raise ValueError("D2 source mask shape mismatch")
    coarse_u = torch.remainder(coarse_source_flat, 98)
    coarse_v = torch.div(coarse_source_flat, 98, rounding_mode="floor")
    child_offsets = torch.arange(4, device=coarse_source_flat.device)
    child_v, child_u = torch.meshgrid(child_offsets, child_offsets, indexing="ij")
    child_flat = (
        (coarse_v[:, None] * 4 + child_v.reshape(1, -1)) * 392
        + coarse_u[:, None] * 4
        + child_u.reshape(1, -1)
    )
    child_parent = torch.arange(
        coarse_source_flat.numel(), device=coarse_source_flat.device
    )[:, None].expand(-1, 16)
    child_valid = mask_a.reshape(-1)[child_flat]
    return child_flat[child_valid], child_parent[child_valid]


def fine_prior_normalized(
    coarse_source_flat: torch.Tensor,
    coarse_target_flat: torch.Tensor,
    source_flat: torch.Tensor,
    parent: torch.Tensor,
    homography_a_to_b: torch.Tensor | None,
    fine_prior: FinePrior,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return D2 target priors in the shared normalized cell-center system."""

    coarse_a_norm = flat_indices_to_centers(coarse_source_flat, 98, 98)
    coarse_b_norm = flat_indices_to_centers(coarse_target_flat, 98, 98)
    source_norm = flat_indices_to_centers(source_flat, 392, 392)
    if fine_prior == "direct_d8":
        # Translation-only propagation: J=I. This branch deliberately does not
        # project either the D8 anchor or D2 child through Stage1 H.
        prior_b_norm = coarse_b_norm[parent] + (
            source_norm - coarse_a_norm[parent]
        )
        return prior_b_norm, torch.isfinite(prior_b_norm).all(dim=-1)
    if fine_prior != "h_residual":
        raise ValueError(f"Unknown fine prior: {fine_prior}")
    if homography_a_to_b is None:
        raise ValueError("h_residual fine prior requires a homography")
    projected_coarse, valid_coarse = project_normalized(
        coarse_a_norm, homography_a_to_b
    )
    residual = coarse_b_norm - projected_coarse
    projected_source, valid_source = project_normalized(
        source_norm, homography_a_to_b
    )
    prior_b_norm = projected_source + residual[parent]
    valid_source &= valid_coarse[parent] & torch.isfinite(prior_b_norm).all(dim=-1)
    return prior_b_norm, valid_source


def local_window_candidates(
    prior_b_norm: torch.Tensor, mask_b: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the unchanged clipped D2 window around rounded prior centers."""

    if mask_b.shape != (392, 392):
        raise ValueError("D2 target mask shape mismatch")
    if window not in {5, 9} or window % 2 != 1:
        raise ValueError("T0 local window must be 5x5 or 9x9")
    prior_u = torch.round((prior_b_norm[:, 0] + 1.0) * 392 / 2.0 - 0.5).long()
    prior_v = torch.round((prior_b_norm[:, 1] + 1.0) * 392 / 2.0 - 0.5).long()
    prior_center_uv = torch.stack((prior_u, prior_v), dim=-1)
    radius = window // 2
    offsets = torch.arange(-radius, radius + 1, device=prior_b_norm.device)
    offset_v, offset_u = torch.meshgrid(offsets, offsets, indexing="ij")
    candidate_u = prior_u[:, None] + offset_u.reshape(1, -1)
    candidate_v = prior_v[:, None] + offset_v.reshape(1, -1)
    candidate_in_image = (
        (candidate_u >= 0)
        & (candidate_u < 392)
        & (candidate_v >= 0)
        & (candidate_v < 392)
    )
    safe_u = candidate_u.clamp(0, 391)
    safe_v = candidate_v.clamp(0, 391)
    candidate_flat = safe_v * 392 + safe_u
    candidate_valid = candidate_in_image & mask_b.reshape(-1)[candidate_flat]
    return prior_center_uv, candidate_flat, candidate_valid


def match_d2(
    descriptor_a: torch.Tensor,
    descriptor_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    coarse_source_flat: torch.Tensor,
    coarse_target_flat: torch.Tensor,
    coarse_confidence: torch.Tensor,
    homography_a_to_b: torch.Tensor | None,
    *,
    fine_prior: FinePrior,
    window: int,
    temperature: float,
    threshold: float,
    selection: FineSelection,
    local_topk: int,
    max_final_matches: int,
    similarity_batch: int = 1024,
) -> FineMatchResult:
    """Match 4x4 D2 children in configurable local target windows."""

    device = descriptor_a.device
    if descriptor_a.shape != descriptor_b.shape or descriptor_a.shape != (256, 392, 392):
        raise ValueError(f"D2 contract is 256x392x392, got {tuple(descriptor_a.shape)}")
    if mask_a.shape != (392, 392) or mask_b.shape != (392, 392):
        raise ValueError("D2 mask shape mismatch")
    if window not in {5, 9} or window % 2 != 1:
        raise ValueError("T0 local window must be 5x5 or 9x9")
    coarse_count = int(coarse_source_flat.numel())
    if coarse_count == 0:
        return _empty(device, "no_coarse_matches", fine_prior)

    source_flat, parent = enumerate_d2_children(coarse_source_flat, mask_a)
    if not source_flat.numel():
        result = _empty(device, "empty_fine_source_support", fine_prior)
        result.diagnostics["coarse_pairs"] = coarse_count
        result.diagnostics["children_enumerated"] = coarse_count * 16
        return result

    prior_b_norm, valid_source = fine_prior_normalized(
        coarse_source_flat,
        coarse_target_flat,
        source_flat,
        parent,
        homography_a_to_b,
        fine_prior,
    )
    source_flat = source_flat[valid_source]
    parent = parent[valid_source]
    prior_b_norm = prior_b_norm[valid_source]
    if not source_flat.numel():
        return _empty(device, "invalid_fine_prior", fine_prior)

    prior_center_uv, candidate_flat, candidate_valid = local_window_candidates(
        prior_b_norm, mask_b, window
    )

    normalized_a = F.normalize(
        descriptor_a.permute(1, 2, 0).reshape(-1, 256).float(), dim=-1, eps=1e-8
    )
    normalized_b = F.normalize(
        descriptor_b.permute(1, 2, 0).reshape(-1, 256).float(), dim=-1, eps=1e-8
    )
    logits = torch.full(
        candidate_flat.shape, -torch.inf, dtype=torch.float32, device=device
    )
    for start in range(0, source_flat.numel(), similarity_batch):
        end = min(source_flat.numel(), start + similarity_batch)
        targets = normalized_b[candidate_flat[start:end]]
        similarities = torch.einsum(
            "bd,bnd->bn", normalized_a[source_flat[start:end]], targets
        )
        logits[start:end] = similarities / temperature
    logits[~candidate_valid] = -torch.inf
    edge_valid = torch.isfinite(logits)
    if not edge_valid.any():
        result = _empty(device, "empty_fine_target_windows", fine_prior)
        result.diagnostics.update(
            coarse_pairs=coarse_count,
            children_enumerated=coarse_count * 16,
            children_valid=int(source_flat.numel()),
        )
        object.__setattr__(result, "evaluated_source_flat", source_flat)
        object.__setattr__(result, "evaluated_prior_b_norm", prior_b_norm)
        object.__setattr__(result, "evaluated_prior_b_center_uv", prior_center_uv)
        return result

    edge_source_node = torch.arange(source_flat.numel(), device=device)[:, None].expand_as(logits)[edge_valid]
    edge_target_flat = candidate_flat[edge_valid]
    edge_logits = logits[edge_valid]
    edge_parent = parent[edge_source_node]
    # Reverse consistency is local to a coarse pair, not leaked across unrelated windows.
    reverse_key = edge_parent * (392 * 392) + edge_target_flat
    _, reverse_group = torch.unique(reverse_key, sorted=True, return_inverse=True)
    reverse_count = int(reverse_group.max().item()) + 1
    source_count = int(source_flat.numel())
    row_lse = _group_logsumexp(edge_logits, edge_source_node, source_count)
    col_lse = _group_logsumexp(edge_logits, reverse_group, reverse_count)
    edge_confidence = torch.exp(
        2.0 * edge_logits - row_lse[edge_source_node] - col_lse[reverse_group]
    )

    if selection == "local_mutual":
        row_max = _group_max(edge_logits, edge_source_node, source_count)
        col_max = _group_max(edge_logits, reverse_group, reverse_count)
        selected = (edge_logits == row_max[edge_source_node]) & (
            edge_logits == col_max[reverse_group]
        )
    elif selection == "local_topk":
        selected = torch.zeros_like(edge_confidence, dtype=torch.bool)
        working = edge_confidence.clone()
        for _ in range(local_topk):
            group_max = _group_max(working, edge_source_node, source_count)
            current = torch.isfinite(working) & (working == group_max[edge_source_node])
            selected |= current
            working[current] = -torch.inf
    else:
        raise ValueError(f"Unknown fine selection: {selection}")

    raw_selected = int(selected.sum().item())
    chosen_edges = torch.nonzero(selected, as_tuple=False).flatten()
    chosen_source_nodes = edge_source_node[chosen_edges]
    chosen_source_flat = source_flat[chosen_source_nodes]
    chosen_target_flat = edge_target_flat[chosen_edges]
    chosen_parent = edge_parent[chosen_edges]
    fine_confidence = edge_confidence[chosen_edges]
    combined = torch.sqrt(
        fine_confidence.clamp_min(0) * coarse_confidence[chosen_parent].float().clamp_min(0)
    )
    above_threshold = combined >= threshold
    chosen_source_flat = chosen_source_flat[above_threshold]
    chosen_target_flat = chosen_target_flat[above_threshold]
    chosen_parent = chosen_parent[above_threshold]
    combined = combined[above_threshold]
    thresholded_selected = int(combined.numel())
    unique = stable_greedy_one_to_one(
        chosen_source_flat,
        chosen_target_flat,
        combined,
        cap=max_final_matches,
    )
    return FineMatchResult(
        source_flat=chosen_source_flat[unique],
        target_flat=chosen_target_flat[unique],
        confidence=combined[unique],
        parent_coarse=chosen_parent[unique],
        evaluated_source_flat=source_flat,
        evaluated_prior_b_norm=prior_b_norm,
        evaluated_prior_b_center_uv=prior_center_uv,
        diagnostics={
            "selection": selection,
            "prior": fine_prior,
            "window": window,
            "temperature": temperature,
            "confidence_rule": "sqrt(coarse_dual_confidence * fine_local_dual_confidence)",
            "coarse_pairs": coarse_count,
            "children_enumerated": coarse_count * 16,
            "children_valid": int(source_flat.numel()),
            "fine_edges_evaluated": int(edge_logits.numel()),
            "raw_selected": raw_selected,
            "thresholded_selected": thresholded_selected,
            "deduplicated_selected": int(unique.numel()),
            "duplicate_rejections": thresholded_selected - int(unique.numel()),
            "duplicate_rejection_rate": (
                float(1.0 - unique.numel() / thresholded_selected)
                if thresholded_selected
                else 0.0
            ),
        },
    )
