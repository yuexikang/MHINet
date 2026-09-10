"""Differentiable homography geometry for MHINet.

The persistent homographies used by MHINet map align-corners-false normalized
coordinates from image A to image B.  This module keeps that convention
explicit, including the TL/TR/BL/BR control-point order used by the four-point
DLT updater.

The guarded DLT routine deliberately performs finite/conditioning checks and a
no-grad probe *before* invoking the autograd-enabled ``solve_ex``.  Unsafe
systems are never mixed into the differentiable solve batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence

import torch
from torch import Tensor


Size2D = tuple[int, int]


class GuardReason(IntEnum):
    """Per-sample reason code emitted by :func:`guarded_four_point_dlt`."""

    ACCEPTED = 0
    NO_VALID_SUPPORT = 1
    NONFINITE_SYSTEM = 2
    ILL_CONDITIONED_SYSTEM = 3
    SOLVE_FAILED = 4
    NONFINITE_SOLUTION = 5
    INVALID_PROJECTION_DENOMINATOR = 6


GUARD_REASON_NAMES: dict[int, str] = {
    int(reason): reason.name.lower() for reason in GuardReason
}


@dataclass(frozen=True)
class GuardedDLTResult:
    """Result of a guarded four-point homography reconstruction.

    ``homography`` contains the solved homography for accepted samples and the
    supplied fallback (identity by default) for rejected samples.  Assignment
    is differentiable, so a rejected sample retains the fallback's upstream
    gradient, while an accepted sample retains the DLT gradient.
    """

    homography: Tensor
    accepted: Tensor
    reason: Tensor
    condition_number: Tensor
    solve_info: Tensor
    solve_eligible: Tensor


def _validate_hw(hw: Sequence[int]) -> Size2D:
    if len(hw) != 2:
        raise ValueError(f"Expected (height, width), got {tuple(hw)!r}")
    height, width = int(hw[0]), int(hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}")
    return height, width


def _floating_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def pixel_normalization_matrix(
    hw: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the align-corners-false pixel-to-normalized matrix ``N(W,H)``."""

    height, width = _validate_hw(hw)
    return torch.tensor(
        [
            [2.0 / width, 0.0, 1.0 / width - 1.0],
            [0.0, 2.0 / height, 1.0 / height - 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def normalized_to_pixel_matrix(
    hw: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the analytic inverse of :func:`pixel_normalization_matrix`."""

    height, width = _validate_hw(hw)
    return torch.tensor(
        [
            [width / 2.0, 0.0, (width - 1.0) / 2.0],
            [0.0, height / 2.0, (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def pixel_to_normalized(points: Tensor, hw: Sequence[int]) -> Tensor:
    """Convert ``(..., 2)`` pixel-center coordinates to normalized coordinates."""

    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError(f"points must end in dimension 2, got {tuple(points.shape)}")
    height, width = _validate_hw(hw)
    scale = points.new_tensor((2.0 / width, 2.0 / height))
    offset = points.new_tensor((1.0 / width - 1.0, 1.0 / height - 1.0))
    return points * scale + offset


def normalized_to_pixel(
    points: Tensor,
    hw: Sequence[int] | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
) -> Tensor:
    """Convert ``(..., 2)`` normalized coordinates to pixel-center coordinates."""

    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError(f"points must end in dimension 2, got {tuple(points.shape)}")
    if hw is None:
        if height is None or width is None:
            raise ValueError("Pass hw=(height,width) or both height= and width=")
        hw = (height, width)
    elif height is not None or width is not None:
        raise ValueError("Pass either hw or height=/width=, not both")
    height, width = _validate_hw(hw)
    scale = points.new_tensor((width / 2.0, height / 2.0))
    offset = points.new_tensor(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
    return points * scale + offset


def pixel_grid(
    hw: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return an ``(H, W, 2)`` row-major pixel-center grid in ``(x, y)`` order."""

    height, width = _validate_hw(hw)
    ys = torch.arange(height, dtype=dtype, device=device)
    xs = torch.arange(width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def normalized_grid(
    hw_or_batch: Sequence[int] | int,
    height: int | None = None,
    width: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return an align-corners-false normalized source grid.

    ``normalized_grid((H,W))`` returns ``(H,W,2)``.  The compatibility form
    ``normalized_grid(B,H,W)`` returns an expanded ``(B,H,W,2)`` tensor.
    """

    if isinstance(hw_or_batch, int):
        batch = int(hw_or_batch)
        if batch <= 0 or height is None or width is None:
            raise ValueError("normalized_grid(B,H,W) requires positive B, H, and W")
        hw = _validate_hw((height, width))
        grid = pixel_to_normalized(pixel_grid(hw, dtype=dtype, device=device), hw)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)
    if height is not None or width is not None:
        raise ValueError("Use normalized_grid((H,W)) or normalized_grid(B,H,W)")
    hw = _validate_hw(hw_or_batch)
    return pixel_to_normalized(pixel_grid(hw, dtype=dtype, device=device), hw)


def image_corners(
    hw: Sequence[int],
    *,
    normalized: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return image control points in the fixed TL, TR, BL, BR order."""

    height, width = _validate_hw(hw)
    corners = torch.tensor(
        [[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]],
        dtype=dtype,
        device=device,
    )
    return pixel_to_normalized(corners, hw) if normalized else corners


def normalized_homography_to_pixel(
    homography: Tensor,
    source_hw: Sequence[int],
    target_hw: Sequence[int] | None = None,
) -> Tensor:
    """Convert a normalized A-to-B homography to feature-pixel coordinates."""

    if homography.shape[-2:] != (3, 3):
        raise ValueError(f"homography must end in (3,3), got {tuple(homography.shape)}")
    target_hw = source_hw if target_hw is None else target_hw
    dtype = _floating_dtype(homography.dtype)
    h = homography.to(dtype=dtype)
    source_n = pixel_normalization_matrix(source_hw, dtype=dtype, device=h.device)
    target_n_inv = normalized_to_pixel_matrix(target_hw, dtype=dtype, device=h.device)
    return target_n_inv @ h @ source_n


def pixel_homography_to_normalized(
    homography: Tensor,
    source_hw: Sequence[int],
    target_hw: Sequence[int] | None = None,
) -> Tensor:
    """Convert a feature-pixel A-to-B homography to normalized coordinates."""

    if homography.shape[-2:] != (3, 3):
        raise ValueError(f"homography must end in (3,3), got {tuple(homography.shape)}")
    target_hw = source_hw if target_hw is None else target_hw
    dtype = _floating_dtype(homography.dtype)
    h = homography.to(dtype=dtype)
    target_n = pixel_normalization_matrix(target_hw, dtype=dtype, device=h.device)
    source_n_inv = normalized_to_pixel_matrix(source_hw, dtype=dtype, device=h.device)
    return target_n @ h @ source_n_inv


def _broadcast_homography_and_points(homography: Tensor, points: Tensor) -> tuple[Tensor, Tensor, bool]:
    if homography.shape[-2:] != (3, 3) or homography.ndim < 2:
        raise ValueError(f"homography must have shape (...,3,3), got {tuple(homography.shape)}")
    if points.shape[-1:] != (2,) or points.ndim < 2:
        raise ValueError(f"points must have shape (...,N,2), got {tuple(points.shape)}")
    if homography.device != points.device:
        raise ValueError(f"homography and points must share a device, got {homography.device} and {points.device}")
    unbatched = homography.ndim == 2 and points.ndim == 2
    try:
        leading = torch.broadcast_shapes(homography.shape[:-2], points.shape[:-2])
    except RuntimeError as error:
        raise ValueError(
            f"Incompatible leading shapes: H={homography.shape[:-2]}, points={points.shape[:-2]}"
        ) from error
    return (
        homography.expand(*leading, 3, 3),
        points.expand(*leading, points.shape[-2], 2),
        unbatched,
    )


def safe_project_points(
    homography: Tensor,
    points: Tensor,
    *,
    denominator_epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project points without ever dividing an already-invalid denominator.

    Returns ``(projected_xy, valid, denominator)``.  Invalid projected values
    are finite zeros.  The continuous path for valid samples remains fully
    differentiable with respect to both the homography and the points.
    """

    if denominator_epsilon <= 0:
        raise ValueError("denominator_epsilon must be positive")
    h, p, unbatched = _broadcast_homography_and_points(homography, points)
    dtype = torch.promote_types(h.dtype, p.dtype)
    dtype = _floating_dtype(dtype)
    h = h.to(dtype=dtype)
    p = p.to(dtype=dtype)

    h_finite = torch.isfinite(h).all(dim=(-2, -1))
    p_finite = torch.isfinite(p).all(dim=-1)
    identity = torch.eye(3, dtype=dtype, device=h.device).expand_as(h)
    safe_h = torch.where(h_finite[..., None, None], h, identity)
    safe_p = torch.where(p_finite[..., None], p, torch.zeros_like(p))
    homogeneous_points = torch.cat((safe_p, torch.ones_like(safe_p[..., :1])), dim=-1)
    projected_h = homogeneous_points @ safe_h.transpose(-1, -2)
    denominator = projected_h[..., 2]
    finite_projection = torch.isfinite(projected_h).all(dim=-1)
    valid = (
        h_finite[..., None]
        & p_finite
        & finite_projection
        & (denominator.abs() > denominator_epsilon)
    )

    # Mask numerator and denominator before division.  In particular, this is
    # not ``where(valid, numerator / denominator, 0)``.
    safe_numerator = torch.where(valid[..., None], projected_h[..., :2], torch.zeros_like(projected_h[..., :2]))
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    projected = safe_numerator / safe_denominator[..., None]

    if unbatched:
        return projected, valid, denominator
    return projected, valid, denominator


def geometry_channels(
    homography_normalized: Tensor,
    source_hw: Sequence[int],
    *,
    flow_clip: tuple[float, float] = (-2.0, 2.0),
    denominator_epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build normalized source-position and current-H flow channels.

    Returns ``(position_xy, flow_xy, projection_valid)`` with channel-first
    tensors of shape ``(B, 2, H, W)``, ``(B, 2, H, W)`` and
    ``(B, 1, H, W)``.  Invalid flow is zero and only the CNN input flow is
    clipped; the input homography is untouched.
    """

    if flow_clip[0] > flow_clip[1]:
        raise ValueError(f"Invalid flow_clip={flow_clip}")
    h = homography_normalized.unsqueeze(0) if homography_normalized.ndim == 2 else homography_normalized
    if h.ndim != 3 or h.shape[-2:] != (3, 3):
        raise ValueError(f"homography must have shape (3,3) or (B,3,3), got {tuple(homography_normalized.shape)}")
    dtype = _floating_dtype(h.dtype)
    h = h.to(dtype=dtype)
    height, width = _validate_hw(source_hw)
    positions = normalized_grid((height, width), dtype=dtype, device=h.device).reshape(1, -1, 2)
    positions = positions.expand(h.shape[0], -1, -1)
    projected, valid, _ = safe_project_points(h, positions, denominator_epsilon=denominator_epsilon)
    flow = projected - positions
    flow = torch.where(valid[..., None], flow, torch.zeros_like(flow))
    flow = flow.clamp(min=float(flow_clip[0]), max=float(flow_clip[1]))
    position_channels = positions.reshape(h.shape[0], height, width, 2).permute(0, 3, 1, 2).contiguous()
    flow_channels = flow.reshape(h.shape[0], height, width, 2).permute(0, 3, 1, 2).contiguous()
    valid_channels = valid.reshape(h.shape[0], 1, height, width)
    return position_channels, flow_channels, valid_channels


def pixel_delta_to_normalized(delta_px: Tensor, target_hw: Sequence[int]) -> Tensor:
    """Convert ``(..., 2)`` target-input pixel displacements to normalized units."""

    if delta_px.shape[-1:] != (2,):
        raise ValueError(f"delta_px must end in dimension 2, got {tuple(delta_px.shape)}")
    height, width = _validate_hw(target_hw)
    return delta_px * delta_px.new_tensor((2.0 / width, 2.0 / height))


def corner_displacements_from_homography(
    homography_normalized: Tensor,
    source_corners_normalized: Tensor,
    *,
    denominator_epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return normalized target corner displacement and projection validity."""

    projected, valid, _ = safe_project_points(
        homography_normalized,
        source_corners_normalized,
        denominator_epsilon=denominator_epsilon,
    )
    corners = source_corners_normalized
    if projected.ndim == 3 and corners.ndim == 2:
        corners = corners.unsqueeze(0)
    displacement = projected - corners
    displacement = torch.where(valid[..., None], displacement, torch.zeros_like(displacement))
    return displacement, valid


def build_four_point_system(source_points: Tensor, target_points: Tensor) -> tuple[Tensor, Tensor]:
    """Build the h33=1, 8-by-8 four-point DLT linear system.

    Point order is not rearranged.  MHINet callers must supply TL, TR, BL, BR.
    """

    if source_points.shape != target_points.shape:
        raise ValueError(
            f"source_points and target_points must have identical shape, got "
            f"{tuple(source_points.shape)} and {tuple(target_points.shape)}"
        )
    if source_points.shape[-2:] != (4, 2) or source_points.ndim not in (2, 3):
        raise ValueError(f"Expected (4,2) or (B,4,2), got {tuple(source_points.shape)}")
    unbatched = source_points.ndim == 2
    source = source_points.unsqueeze(0) if unbatched else source_points
    target = target_points.unsqueeze(0) if unbatched else target_points
    dtype = _floating_dtype(torch.promote_types(source.dtype, target.dtype))
    source = source.to(dtype=dtype)
    target = target.to(dtype=dtype)

    x, y = source.unbind(dim=-1)
    u, v = target.unbind(dim=-1)
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    row_u = torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y), dim=-1)
    row_v = torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y), dim=-1)
    matrix = torch.stack((row_u, row_v), dim=-2).reshape(source.shape[0], 8, 8)
    right_hand_side = torch.stack((u, v), dim=-1).reshape(source.shape[0], 8)
    if unbatched:
        return matrix[0], right_hand_side[0]
    return matrix, right_hand_side


def _assemble_homography(theta: Tensor) -> Tensor:
    one = torch.ones_like(theta[..., :1])
    return torch.cat((theta, one), dim=-1).reshape(*theta.shape[:-1], 3, 3)


def count_supported_queries(candidate_valid: Tensor) -> Tensor:
    """Count source queries having at least one valid local candidate."""

    if candidate_valid.ndim != 4:
        raise ValueError(f"candidate_valid must be (B,K,H,W), got {tuple(candidate_valid.shape)}")
    return candidate_valid.to(dtype=torch.bool).any(dim=1).flatten(1).sum(dim=1)


def _validation_denominators(homography: Tensor, grid_hw: Sequence[int]) -> Tensor:
    grid_height, grid_width = _validate_hw(grid_hw)
    ys = torch.linspace(-1.0, 1.0, grid_height, dtype=homography.dtype, device=homography.device)
    xs = torch.linspace(-1.0, 1.0, grid_width, dtype=homography.dtype, device=homography.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = xx.reshape(1, -1)
    y = yy.reshape(1, -1)
    return homography[:, 2, 0:1] * x + homography[:, 2, 1:2] * y + homography[:, 2, 2:3]


def guarded_four_point_dlt(
    source_points: Tensor,
    target_points: Tensor,
    *,
    fallback_homography: Tensor | None = None,
    candidate_valid: Tensor | None = None,
    support_count: Tensor | int | None = None,
    minimum_supported_queries: int = 16,
    condition_max: float = 1e6,
    projection_denominator_min: float = 1e-4,
    validity_grid_hw: Sequence[int] = (9, 9),
) -> GuardedDLTResult:
    """Reconstruct homographies with strict pre-solve and post-projection guards.

    Finite/support/condition checks and a probe ``solve_ex`` run under
    ``torch.no_grad``.  Only the subset passing all probe checks enters the
    differentiable ``solve_ex``.  Consequently a singular/non-finite system is
    never solved on an autograd path and then hidden with ``where``.
    """

    if candidate_valid is not None and support_count is not None:
        raise ValueError("Pass candidate_valid or support_count, not both")
    if minimum_supported_queries < 0:
        raise ValueError("minimum_supported_queries must be non-negative")
    if not math.isfinite(condition_max) or condition_max <= 0:
        raise ValueError("condition_max must be finite and positive")
    if projection_denominator_min <= 0:
        raise ValueError("projection_denominator_min must be positive")

    unbatched = source_points.ndim == 2
    matrix, rhs = build_four_point_system(source_points, target_points)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
        rhs = rhs.unsqueeze(0)
    batch = matrix.shape[0]
    dtype, device = matrix.dtype, matrix.device

    if fallback_homography is None:
        fallback = torch.eye(3, dtype=dtype, device=device).expand(batch, -1, -1).clone()
    else:
        fallback = fallback_homography
        if fallback.ndim == 2:
            fallback = fallback.unsqueeze(0)
        if fallback.ndim != 3 or fallback.shape[-2:] != (3, 3) or fallback.shape[0] not in (1, batch):
            raise ValueError(
                f"fallback_homography must be (3,3), (1,3,3), or (B,3,3); got {tuple(fallback.shape)}"
            )
        fallback = fallback.to(dtype=dtype, device=device).expand(batch, -1, -1)

    if candidate_valid is not None:
        counts = count_supported_queries(candidate_valid).to(device=device)
    elif support_count is None:
        counts = torch.full((batch,), minimum_supported_queries, dtype=torch.long, device=device)
    else:
        counts = torch.as_tensor(support_count, device=device)
        if counts.ndim == 0:
            counts = counts.expand(batch)
        if counts.shape != (batch,):
            raise ValueError(f"support_count must be scalar or shape ({batch},), got {tuple(counts.shape)}")

    reason = torch.full((batch,), int(GuardReason.ACCEPTED), dtype=torch.long, device=device)
    condition = torch.full((batch,), float("inf"), dtype=dtype, device=device)
    solve_info = torch.full((batch,), -1, dtype=torch.long, device=device)
    solve_eligible = torch.zeros((batch,), dtype=torch.bool, device=device)

    with torch.no_grad():
        support_ok = counts >= minimum_supported_queries
        reason[~support_ok] = int(GuardReason.NO_VALID_SUPPORT)

        finite_system = torch.isfinite(matrix).all(dim=(-2, -1)) & torch.isfinite(rhs).all(dim=-1)
        nonfinite = support_ok & ~finite_system
        reason[nonfinite] = int(GuardReason.NONFINITE_SYSTEM)

        condition_candidates = support_ok & finite_system
        for index in torch.nonzero(condition_candidates, as_tuple=False).flatten().tolist():
            try:
                condition[index] = torch.linalg.cond(matrix[index])
            except RuntimeError:
                condition[index] = float("inf")
        condition_ok = torch.isfinite(condition) & (condition < condition_max)
        ill_conditioned = condition_candidates & ~condition_ok
        reason[ill_conditioned] = int(GuardReason.ILL_CONDITIONED_SYSTEM)

        probe_candidates = condition_candidates & condition_ok
        probe_indices = torch.nonzero(probe_candidates, as_tuple=False).flatten()
        if probe_indices.numel() > 0:
            probe_theta, probe_status = torch.linalg.solve_ex(
                matrix.index_select(0, probe_indices),
                rhs.index_select(0, probe_indices).unsqueeze(-1),
                check_errors=False,
            )
            probe_theta = probe_theta.squeeze(-1)
            solve_info[probe_indices] = probe_status.to(dtype=torch.long)
            probe_solved = probe_status == 0
            failed_indices = probe_indices[~probe_solved]
            reason[failed_indices] = int(GuardReason.SOLVE_FAILED)

            finite_theta = torch.isfinite(probe_theta).all(dim=-1)
            nonfinite_indices = probe_indices[probe_solved & ~finite_theta]
            reason[nonfinite_indices] = int(GuardReason.NONFINITE_SOLUTION)

            solution_candidates = probe_solved & finite_theta
            if solution_candidates.any():
                local_indices = torch.nonzero(solution_candidates, as_tuple=False).flatten()
                candidate_h = _assemble_homography(probe_theta.index_select(0, local_indices))
                denominators = _validation_denominators(candidate_h, validity_grid_hw)
                denominator_ok = torch.isfinite(denominators).all(dim=-1) & (
                    (denominators > projection_denominator_min).all(dim=-1)
                    | (denominators < -projection_denominator_min).all(dim=-1)
                )
                global_indices = probe_indices.index_select(0, local_indices)
                rejected_projection = global_indices[~denominator_ok]
                reason[rejected_projection] = int(GuardReason.INVALID_PROJECTION_DENOMINATOR)
                solve_eligible[global_indices[denominator_ok]] = True

    selected = fallback.clone()
    eligible_indices = torch.nonzero(solve_eligible, as_tuple=False).flatten()
    if eligible_indices.numel() > 0:
        # This is the only autograd-enabled solve, and it contains only the
        # finite, well-conditioned, successfully probed subset.
        theta, differentiable_status = torch.linalg.solve_ex(
            matrix.index_select(0, eligible_indices),
            rhs.index_select(0, eligible_indices).unsqueeze(-1),
            check_errors=False,
        )
        theta = theta.squeeze(-1)
        proposal_h = _assemble_homography(theta)
        with torch.no_grad():
            differentiable_ok = (differentiable_status == 0) & torch.isfinite(theta).all(dim=-1)
            denominators = _validation_denominators(proposal_h.detach(), validity_grid_hw)
            differentiable_ok &= torch.isfinite(denominators).all(dim=-1) & (
                (denominators > projection_denominator_min).all(dim=-1)
                | (denominators < -projection_denominator_min).all(dim=-1)
            )
            unexpected_rejects = eligible_indices[~differentiable_ok]
            if unexpected_rejects.numel() > 0:
                reason[unexpected_rejects] = int(GuardReason.SOLVE_FAILED)
                solve_eligible[unexpected_rejects] = False
            accepted_indices = eligible_indices[differentiable_ok]
        if accepted_indices.numel() > 0:
            selected[accepted_indices] = proposal_h[differentiable_ok]

    accepted = solve_eligible
    if unbatched:
        return GuardedDLTResult(
            homography=selected[0],
            accepted=accepted[0],
            reason=reason[0],
            condition_number=condition[0],
            solve_info=solve_info[0],
            solve_eligible=solve_eligible[0],
        )
    return GuardedDLTResult(
        homography=selected,
        accepted=accepted,
        reason=reason,
        condition_number=condition,
        solve_info=solve_info,
        solve_eligible=solve_eligible,
    )


def four_point_dlt(
    source_points: Tensor,
    target_points: Tensor,
    *,
    condition_max: float = 1e6,
    projection_denominator_min: float = 1e-4,
    validity_grid_hw: Sequence[int] = (9, 9),
) -> Tensor:
    """Differentiable legal-input convenience wrapper around the guarded DLT.

    Raises instead of silently returning a placeholder if any system is
    rejected.  Runtime iteration code should use :func:`guarded_four_point_dlt`
    so it can retain the previous H/T and record the reason code.
    """

    result = guarded_four_point_dlt(
        source_points,
        target_points,
        minimum_supported_queries=0,
        condition_max=condition_max,
        projection_denominator_min=projection_denominator_min,
        validity_grid_hw=validity_grid_hw,
    )
    if not bool(result.accepted.all().item()):
        codes = result.reason.detach().reshape(-1).cpu().tolist()
        names = [GUARD_REASON_NAMES[int(code)] for code in codes]
        raise ValueError(f"Four-point DLT rejected unsafe input: {names}")
    return result.homography


__all__ = [
    "GUARD_REASON_NAMES",
    "GuardReason",
    "GuardedDLTResult",
    "build_four_point_system",
    "corner_displacements_from_homography",
    "count_supported_queries",
    "four_point_dlt",
    "geometry_channels",
    "guarded_four_point_dlt",
    "image_corners",
    "normalized_grid",
    "normalized_homography_to_pixel",
    "normalized_to_pixel",
    "normalized_to_pixel_matrix",
    "pixel_delta_to_normalized",
    "pixel_grid",
    "pixel_homography_to_normalized",
    "pixel_normalization_matrix",
    "pixel_to_normalized",
    "safe_project_points",
]
