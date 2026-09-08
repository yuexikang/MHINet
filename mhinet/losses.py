"""Training-protocol v1.2 default proposal-corner supervision."""

from __future__ import annotations

from typing import Any

import torch

from .geometry import image_corners, normalized_to_pixel, safe_project_points


def sequence_corner_l1(
    outputs: dict[str, Any],
    H_gt_norm: torch.Tensor,
    *,
    target_hw: tuple[int, int] = (784, 784),
) -> dict[str, torch.Tensor | int | bool]:
    """Equal-weight L1 over all 8 proposal corners, before geometry guards."""

    proposals = outputs["proposal_Q_norm"].float()
    if proposals.ndim != 4 or proposals.shape[-2:] != (4, 2):
        raise ValueError(f"proposal_Q_norm must be BxUx4x2, got {tuple(proposals.shape)}")
    batch, updates = proposals.shape[:2]
    if H_gt_norm.shape != (batch, 3, 3):
        raise ValueError(f"H_gt_norm mismatch: {tuple(H_gt_norm.shape)}")
    decoder_output_finite = outputs.get("decoder_output_finite")
    if decoder_output_finite is not None and not bool(
        decoder_output_finite.bool().all()
    ):
        raise FloatingPointError(
            "A non-finite decoder output was isolated before DLT; refusing to "
            "treat its safe placeholder proposal as a training target"
        )
    if not torch.isfinite(proposals).all():
        raise FloatingPointError("Non-finite proposal corners are a numerical error")
    height, width = target_hw
    source_corners = image_corners(
        (height, width), normalized=True, device=H_gt_norm.device, dtype=torch.float32
    ).unsqueeze(0).expand(batch, -1, -1)
    target_corners, target_valid, _ = safe_project_points(
        H_gt_norm.float(), source_corners
    )
    if not target_valid.all() or not torch.isfinite(target_corners).all():
        raise FloatingPointError("Ground-truth homography has invalid corner projection")
    proposal_px = normalized_to_pixel(proposals, (height, width))
    target_px = normalized_to_pixel(target_corners, (height, width))
    absolute = (proposal_px - target_px[:, None]).abs()
    per_update = absolute.mean(dim=(2, 3))
    per_pair = per_update.mean(dim=1)
    valid = outputs["stage1_valid"].bool()
    if valid.shape != (batch,):
        raise ValueError(f"stage1_valid must have shape {(batch,)}, got {tuple(valid.shape)}")
    valid_count = int(valid.sum().item())
    if valid_count:
        loss = per_pair[valid].mean()
    else:
        # Trainer must skip optimizer/scheduler; retain a finite graph-valued scalar.
        loss = proposals.sum() * 0.0
    return {
        "loss": loss,
        "per_update_corner_l1_px": per_update,
        "per_pair_corner_l1_px": per_pair,
        "valid_pairs": valid_count,
        "updates": updates,
        "skip_step": valid_count == 0,
    }


def mean_average_corner_error_px(
    predicted_corners_norm: torch.Tensor,
    target_corners_norm: torch.Tensor,
    *,
    target_hw: tuple[int, int] = (784, 784),
) -> torch.Tensor:
    height, width = target_hw
    predicted = normalized_to_pixel(predicted_corners_norm, (height, width))
    target = normalized_to_pixel(target_corners_norm, (height, width))
    return torch.linalg.vector_norm(predicted - target, dim=-1).mean(dim=-1)
