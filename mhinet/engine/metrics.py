"""Per-pair H0/update/final geometry metrics without failure filtering."""

from __future__ import annotations

from typing import Any

import torch

from mhinet.ops.geometry import (
    image_corners,
    normalized_grid,
    normalized_to_pixel,
    safe_project_points,
)


def homography_trajectory_metrics(
    outputs: dict[str, Any],
    H_gt_norm: torch.Tensor,
    *,
    target_hw: tuple[int, int] = (784, 784),
    grid_hw: tuple[int, int] = (5, 5),
) -> dict[str, torch.Tensor]:
    """Return per-pair errors for H0, every accepted update, and final H."""

    batch = H_gt_norm.shape[0]
    height, width = target_hw
    corners = image_corners(
        (height, width), normalized=True, device=H_gt_norm.device, dtype=torch.float32
    ).unsqueeze(0).expand(batch, -1, -1)
    target_corners, gt_valid, _ = safe_project_points(H_gt_norm.float(), corners)
    trajectory = torch.cat(
        (outputs["H0_norm"][:, None].float(), outputs["H_updates_norm"].float()), dim=1
    )
    time = trajectory.shape[1]
    points = corners[:, None].expand(-1, time, -1, -1)
    projected_flat, predicted_valid_flat, _ = safe_project_points(
        trajectory.reshape(batch * time, 3, 3),
        points.reshape(batch * time, 4, 2),
    )
    projected = projected_flat.reshape(batch, time, 4, 2)
    predicted_valid = predicted_valid_flat.reshape(batch, time, 4)
    target_px = normalized_to_pixel(target_corners, (height, width))
    projected_px = normalized_to_pixel(projected, (height, width))
    mace = torch.linalg.vector_norm(
        projected_px - target_px[:, None], dim=-1
    ).mean(dim=-1)
    valid = predicted_valid.all(dim=-1) & gt_valid.all(dim=-1, keepdim=True)
    mace = torch.where(valid, mace, torch.full_like(mace, float("inf")))

    grid = normalized_grid(
        grid_hw, device=H_gt_norm.device, dtype=torch.float32
    ).reshape(1, -1, 2).expand(batch, -1, -1)
    gt_grid, gt_grid_valid, _ = safe_project_points(H_gt_norm.float(), grid)
    grid_points = grid[:, None].expand(-1, time, -1, -1)
    predicted_grid_flat, predicted_grid_valid_flat, _ = safe_project_points(
        trajectory.reshape(batch * time, 3, 3),
        grid_points.reshape(batch * time, -1, 2),
    )
    predicted_grid = predicted_grid_flat.reshape(batch, time, -1, 2)
    predicted_grid_valid = predicted_grid_valid_flat.reshape(batch, time, -1)
    gt_grid_px = normalized_to_pixel(gt_grid, (height, width))
    predicted_grid_px = normalized_to_pixel(predicted_grid, (height, width))
    grid_error = torch.linalg.vector_norm(
        predicted_grid_px - gt_grid_px[:, None], dim=-1
    ).mean(dim=-1)
    grid_valid = predicted_grid_valid.all(dim=-1) & gt_grid_valid.all(
        dim=-1, keepdim=True
    )
    grid_error = torch.where(
        grid_valid, grid_error, torch.full_like(grid_error, float("inf"))
    )
    return {
        "trajectory_mace_px": mace,
        "H0_mace_px": mace[:, 0],
        "H_updates_mace_px": mace[:, 1:],
        "H_final_mace_px": mace[:, -1],
        "trajectory_grid5_error_px": grid_error,
        "geometry_valid": valid,
    }
