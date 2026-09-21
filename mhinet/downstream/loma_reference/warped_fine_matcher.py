"""Zero-shot H-warped symmetric D2 fine matching.

Each mutual D8 parent owns one regular 4x4 A lattice.  Every A lattice point is
projected through the frozen Stage1 homography and shifted by the parent's D8
coarse residual to form a continuous B lattice.  Descriptor correlation remains
all-to-all inside the parent; geometry supplies a local coordinate system, not
an index-wise correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .matching_utils import flat_indices_to_centers, project_normalized


PATCH_SIZE = 4
PATCH_POINTS = PATCH_SIZE * PATCH_SIZE
D2_SIZE = 392


@dataclass(frozen=True)
class WarpedFineMatchResult:
    source_norm: torch.Tensor
    target_norm: torch.Tensor
    confidence: torch.Tensor
    parent_coarse: torch.Tensor
    lattice_source_norm: torch.Tensor
    lattice_target_norm: torch.Tensor
    lattice_parent_coarse: torch.Tensor
    diagnostics: dict[str, float | int | str | bool]


def d8_children_lattice(
    coarse_source_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each D8 cell's exact 4x4 D2 children and normalized centers."""

    coarse_u = torch.remainder(coarse_source_flat, 98)
    coarse_v = torch.div(coarse_source_flat, 98, rounding_mode="floor")
    offsets = torch.arange(PATCH_SIZE, device=coarse_source_flat.device)
    offset_v, offset_u = torch.meshgrid(offsets, offsets, indexing="ij")
    child_u = coarse_u[:, None] * PATCH_SIZE + offset_u.reshape(1, -1)
    child_v = coarse_v[:, None] * PATCH_SIZE + offset_v.reshape(1, -1)
    child_flat = child_v * D2_SIZE + child_u
    child_norm = flat_indices_to_centers(
        child_flat.reshape(-1), D2_SIZE, D2_SIZE
    ).reshape(-1, PATCH_POINTS, 2)
    return child_flat, child_norm


def h_warped_target_lattice(
    coarse_source_flat: torch.Tensor,
    coarse_target_flat: torch.Tensor,
    source_lattice_norm: torch.Tensor,
    homography_a_to_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``q_i = H(p_i) + [b8 - H(a8)]`` to all 16 A samples."""

    if source_lattice_norm.ndim != 3 or source_lattice_norm.shape[1:] != (
        PATCH_POINTS,
        2,
    ):
        raise ValueError("Source lattice must have shape Mx16x2")
    coarse_a = flat_indices_to_centers(coarse_source_flat, 98, 98)
    coarse_b = flat_indices_to_centers(coarse_target_flat, 98, 98)
    projected_coarse, valid_coarse = project_normalized(
        coarse_a, homography_a_to_b
    )
    residual = coarse_b - projected_coarse
    projected_source, valid_source = project_normalized(
        source_lattice_norm.reshape(-1, 2), homography_a_to_b
    )
    target = projected_source.reshape(-1, PATCH_POINTS, 2) + residual[:, None]
    valid = valid_source.reshape(-1, PATCH_POINTS) & valid_coarse[:, None]
    valid &= torch.isfinite(target).all(dim=-1)
    return target, valid


def normalized_to_feature_uv(points: torch.Tensor, size: int) -> torch.Tensor:
    """Convert align_corners=False normalized centers to continuous feature UV."""

    return (points + 1.0) * size / 2.0 - 0.5


def sample_bilinear(
    feature: torch.Tensor, points_norm: torch.Tensor
) -> torch.Tensor:
    """Sample CxHxW features at arbitrary normalized points as NxC."""

    if feature.ndim != 3:
        raise ValueError("Feature must have shape CxHxW")
    if points_norm.ndim != 2 or points_norm.shape[-1] != 2:
        raise ValueError("Sampling points must have shape Nx2")
    grid = points_norm.reshape(1, -1, 1, 2).to(feature.dtype)
    sampled = F.grid_sample(
        feature[None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0, :, :, 0].T


def continuous_mask_validity(
    mask: torch.Tensor,
    points_norm: torch.Tensor,
    geometric_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Require the continuous center to be in-image and fully inside HROI mask."""

    if mask.shape != (D2_SIZE, D2_SIZE):
        raise ValueError("D2 mask shape mismatch")
    uv = normalized_to_feature_uv(points_norm, D2_SIZE)
    in_image = (
        (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= D2_SIZE - 1.0)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= D2_SIZE - 1.0)
    )
    safe_points = torch.where(
        (geometric_valid & torch.isfinite(points_norm).all(dim=-1))[..., None],
        points_norm,
        torch.zeros_like(points_norm),
    )
    coverage = sample_bilinear(mask[None].float(), safe_points.reshape(-1, 2))[
        :, 0
    ].reshape(points_norm.shape[:-1])
    mask_valid = coverage >= 1.0 - 1e-5
    return geometric_valid & in_image & mask_valid, in_image, mask_valid


def masked_dual_softmax(
    logits: torch.Tensor, valid_edges: torch.Tensor
) -> torch.Tensor:
    """FP32 dual-softmax with zero probability for fully masked rows/columns."""

    if logits.shape != valid_edges.shape or logits.ndim != 3:
        raise ValueError("Logits and edge mask must have identical Mx16x16 shape")
    logits = logits.float()

    def masked_softmax(dim: int) -> torch.Tensor:
        masked = torch.where(valid_edges, logits, -torch.inf)
        maxima = masked.amax(dim=dim, keepdim=True)
        maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
        numerator = torch.where(
            valid_edges, torch.exp(masked - maxima), torch.zeros_like(masked)
        )
        denominator = numerator.sum(dim=dim, keepdim=True)
        return numerator / denominator.clamp_min(1e-30)

    return masked_softmax(2) * masked_softmax(1)


def geometry_edge_mask(
    radius: int | None, *, device: torch.device | str
) -> torch.Tensor:
    """Return the 16x16 Chebyshev lattice-neighborhood mask."""

    if radius not in {None, 0, 1, 2}:
        raise ValueError("Geometry radius must be None, 0, 1, or 2")
    if radius is None:
        return torch.ones((PATCH_POINTS, PATCH_POINTS), dtype=torch.bool, device=device)
    index = torch.arange(PATCH_POINTS, device=device)
    row = torch.div(index, PATCH_SIZE, rounding_mode="floor")
    column = torch.remainder(index, PATCH_SIZE)
    distance = torch.maximum(
        (row[:, None] - row[None, :]).abs(),
        (column[:, None] - column[None, :]).abs(),
    )
    return distance <= radius


def select_one_per_parent(
    confidence: torch.Tensor, valid_edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the deterministic flattened argmax for every valid parent."""

    if confidence.shape != valid_edges.shape:
        raise ValueError("Confidence and edge mask shape mismatch")
    parent_valid = valid_edges.flatten(1).any(dim=1)
    parent = torch.nonzero(parent_valid, as_tuple=False).flatten()
    flat = confidence[parent].flatten(1)
    best_confidence, best = flat.max(dim=1)
    source_index = torch.div(best, PATCH_POINTS, rounding_mode="floor")
    target_index = torch.remainder(best, PATCH_POINTS)
    return parent, source_index, target_index, best_confidence


def _empty(
    device: torch.device, reason: str, geometry_radius: int | None
) -> WarpedFineMatchResult:
    empty_points = torch.empty((0, 2), dtype=torch.float32, device=device)
    empty_indices = torch.empty(0, dtype=torch.long, device=device)
    return WarpedFineMatchResult(
        source_norm=empty_points,
        target_norm=empty_points.clone(),
        confidence=torch.empty(0, dtype=torch.float32, device=device),
        parent_coarse=empty_indices,
        lattice_source_norm=empty_points.clone(),
        lattice_target_norm=empty_points.clone(),
        lattice_parent_coarse=empty_indices.clone(),
        diagnostics={
            "mode": "h_warped_sym4",
            "selection": reason,
            "patch_size": PATCH_SIZE,
            "coarse_pairs": 0,
            "raw_selected": 0,
            "thresholded_selected": 0,
            "selected": 0,
            "global_dedup_enabled": False,
            "geometry_radius": "unrestricted" if geometry_radius is None else geometry_radius,
        },
    )


def _offset_diagnostics(
    source_index: torch.Tensor, target_index: torch.Tensor
) -> dict[str, float | int]:
    if not source_index.numel():
        return {
            "selected_offset_linf_mean": 0.0,
            "selected_offset_linf_0": 0,
            "selected_offset_linf_1": 0,
            "selected_offset_linf_2": 0,
            "selected_offset_linf_3": 0,
        }
    source_row = torch.div(source_index, PATCH_SIZE, rounding_mode="floor")
    source_column = torch.remainder(source_index, PATCH_SIZE)
    target_row = torch.div(target_index, PATCH_SIZE, rounding_mode="floor")
    target_column = torch.remainder(target_index, PATCH_SIZE)
    distance = torch.maximum(
        (source_row - target_row).abs(), (source_column - target_column).abs()
    )
    return {
        "selected_offset_linf_mean": float(distance.float().mean().item()),
        **{
            f"selected_offset_linf_{value}": int((distance == value).sum().item())
            for value in range(PATCH_SIZE)
        },
    }


def _duplicate_diagnostics(
    source_norm: torch.Tensor, target_norm: torch.Tensor
) -> dict[str, float | int]:
    count = int(source_norm.shape[0])
    if count == 0:
        return {
            "source_exact_duplicate_count": 0,
            "target_exact_duplicate_count": 0,
            "source_quantized_0p5cell_duplicate_count": 0,
            "target_quantized_0p5cell_duplicate_count": 0,
            "source_exact_duplicate_rate": 0.0,
            "target_exact_duplicate_rate": 0.0,
            "source_quantized_0p5cell_duplicate_rate": 0.0,
            "target_quantized_0p5cell_duplicate_rate": 0.0,
        }

    def counts(points: torch.Tensor) -> tuple[int, int]:
        exact_unique = int(torch.unique(points, dim=0).shape[0])
        uv = normalized_to_feature_uv(points, D2_SIZE)
        quantized_unique = int(torch.unique(torch.round(uv * 2.0).long(), dim=0).shape[0])
        return count - exact_unique, count - quantized_unique

    source_exact, source_near = counts(source_norm)
    target_exact, target_near = counts(target_norm)
    return {
        "source_exact_duplicate_count": source_exact,
        "target_exact_duplicate_count": target_exact,
        "source_quantized_0p5cell_duplicate_count": source_near,
        "target_quantized_0p5cell_duplicate_count": target_near,
        "source_exact_duplicate_rate": source_exact / count,
        "target_exact_duplicate_rate": target_exact / count,
        "source_quantized_0p5cell_duplicate_rate": source_near / count,
        "target_quantized_0p5cell_duplicate_rate": target_near / count,
    }


def match_d2_h_warped_sym4(
    descriptor_a: torch.Tensor,
    descriptor_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    coarse_source_flat: torch.Tensor,
    coarse_target_flat: torch.Tensor,
    coarse_confidence: torch.Tensor,
    homography_a_to_b: torch.Tensor,
    *,
    temperature: float,
    threshold: float,
    max_final_matches: int,
    geometry_radius: int | None = None,
) -> WarpedFineMatchResult:
    """Return at most one continuous warped-lattice fine pair per D8 parent."""

    device = descriptor_a.device
    if descriptor_a.shape != descriptor_b.shape or descriptor_a.shape != (
        256,
        D2_SIZE,
        D2_SIZE,
    ):
        raise ValueError(
            f"D2 contract is 256x392x392, got {tuple(descriptor_a.shape)}"
        )
    if mask_a.shape != (D2_SIZE, D2_SIZE) or mask_b.shape != (
        D2_SIZE,
        D2_SIZE,
    ):
        raise ValueError("D2 mask shape mismatch")
    if temperature <= 0 or threshold < 0 or max_final_matches <= 0:
        raise ValueError("Invalid fine matching scalar configuration")
    if geometry_radius not in {None, 0, 1, 2}:
        raise ValueError("Geometry radius must be None, 0, 1, or 2")
    coarse_count = int(coarse_source_flat.numel())
    if coarse_count == 0:
        return _empty(device, "no_coarse_matches", geometry_radius)

    source_flat, source_norm = d8_children_lattice(coarse_source_flat)
    target_norm, target_geometry_valid = h_warped_target_lattice(
        coarse_source_flat,
        coarse_target_flat,
        source_norm,
        homography_a_to_b,
    )
    source_valid = mask_a.reshape(-1)[source_flat]
    target_valid, target_in_image, target_mask_valid = continuous_mask_validity(
        mask_b, target_norm, target_geometry_valid
    )

    normalized_a = F.normalize(
        descriptor_a.permute(1, 2, 0).reshape(-1, 256).float(),
        dim=-1,
        eps=1e-8,
    )
    safe_target = torch.where(
        target_geometry_valid[..., None], target_norm, torch.zeros_like(target_norm)
    )
    sampled_b = sample_bilinear(descriptor_b.float(), safe_target.reshape(-1, 2))
    normalized_b = F.normalize(
        sampled_b.reshape(coarse_count, PATCH_POINTS, 256), dim=-1, eps=1e-8
    )
    source_features = normalized_a[source_flat]
    logits = torch.einsum("mic,mjc->mij", source_features, normalized_b) / float(
        temperature
    )
    hroi_valid_edges = source_valid[:, :, None] & target_valid[:, None, :]
    geometry_valid_edges = geometry_edge_mask(
        geometry_radius, device=device
    )[None]
    # The geometry constraint changes both row and column normalization, so it
    # must be applied before dual-softmax rather than masking probabilities.
    valid_edges = hroi_valid_edges & geometry_valid_edges
    fine_probability = masked_dual_softmax(logits, valid_edges)
    parent, source_index, target_index, fine_confidence = select_one_per_parent(
        fine_probability, valid_edges
    )
    selected_source_norm = source_norm[parent, source_index]
    selected_target_norm = target_norm[parent, target_index]
    combined = torch.sqrt(
        fine_confidence.clamp_min(0)
        * coarse_confidence[parent].float().clamp_min(0)
    )
    above_threshold = combined >= threshold
    parent = parent[above_threshold]
    selected_source_norm = selected_source_norm[above_threshold]
    selected_target_norm = selected_target_norm[above_threshold]
    selected_source_index = source_index[above_threshold]
    selected_target_index = target_index[above_threshold]
    combined = combined[above_threshold]
    thresholded = int(parent.numel())
    if parent.numel() > max_final_matches:
        order = torch.argsort(combined, descending=True, stable=True)[
            :max_final_matches
        ]
        parent = parent[order]
        selected_source_norm = selected_source_norm[order]
        selected_target_norm = selected_target_norm[order]
        selected_source_index = selected_source_index[order]
        selected_target_index = selected_target_index[order]
        combined = combined[order]

    lattice_parent = torch.arange(coarse_count, device=device)[:, None].expand(
        -1, PATCH_POINTS
    )
    diagnostic_source = torch.where(
        source_valid[..., None], source_norm, torch.full_like(source_norm, torch.nan)
    )
    diagnostic_target = torch.where(
        target_valid[..., None], target_norm, torch.full_like(target_norm, torch.nan)
    )
    duplicate = _duplicate_diagnostics(
        selected_source_norm, selected_target_norm
    )
    offsets = _offset_diagnostics(selected_source_index, selected_target_index)
    return WarpedFineMatchResult(
        source_norm=selected_source_norm,
        target_norm=selected_target_norm,
        confidence=combined,
        parent_coarse=parent,
        lattice_source_norm=diagnostic_source.reshape(-1, 2),
        lattice_target_norm=diagnostic_target.reshape(-1, 2),
        lattice_parent_coarse=lattice_parent.reshape(-1),
        diagnostics={
            "mode": "h_warped_sym4",
            "selection": "per_parent_flat_argmax",
            "patch_size": PATCH_SIZE,
            "geometry_metric": "lattice_index_chebyshev",
            "geometry_radius": "unrestricted" if geometry_radius is None else geometry_radius,
            "geometry_matrix_entries": int(geometry_valid_edges.sum().item()),
            "temperature": float(temperature),
            "confidence_rule": "sqrt(coarse_dual_confidence * fine_local_dual_confidence)",
            "coarse_pairs": coarse_count,
            "source_samples": coarse_count * PATCH_POINTS,
            "source_valid": int(source_valid.sum().item()),
            "target_samples": coarse_count * PATCH_POINTS,
            "target_geometry_valid": int(target_geometry_valid.sum().item()),
            "target_in_image": int((target_geometry_valid & target_in_image).sum().item()),
            "target_mask_valid": int(target_valid.sum().item()),
            "fine_edges_hroi_valid": int(hroi_valid_edges.sum().item()),
            "fine_edges_evaluated": int(valid_edges.sum().item()),
            "geometry_edges_rejected": int(
                hroi_valid_edges.sum().item() - valid_edges.sum().item()
            ),
            "dense_parent_matrix_entries": coarse_count * PATCH_POINTS * PATCH_POINTS,
            "valid_parents": int(valid_edges.flatten(1).any(dim=1).sum().item()),
            "raw_selected": int(fine_confidence.numel()),
            "thresholded_selected": thresholded,
            "selected": int(parent.numel()),
            "outputs_per_parent_max": 1,
            "global_dedup_enabled": False,
            "rounding_used": False,
            **offsets,
            **duplicate,
        },
    )
