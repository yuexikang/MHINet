"""Coordinate, homography, confidence, and deterministic deduplication helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_cell_centers(
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return HxWx2 cell centers under ``align_corners=False``."""

    ys = 2.0 * (torch.arange(height, device=device, dtype=dtype) + 0.5) / height - 1.0
    xs = 2.0 * (torch.arange(width, device=device, dtype=dtype) + 0.5) / width - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def flat_indices_to_centers(
    indices: torch.Tensor, height: int, width: int, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    v = torch.div(indices, width, rounding_mode="floor").to(dtype)
    u = torch.remainder(indices, width).to(dtype)
    return torch.stack((2.0 * (u + 0.5) / width - 1.0, 2.0 * (v + 0.5) / height - 1.0), dim=-1)


def centers_to_native(points: torch.Tensor, size_wh: tuple[int, int]) -> torch.Tensor:
    """Map normalized centers to independent native W/H pixel coordinates."""

    width, height = size_wh
    scale = points.new_tensor((width, height))
    return (points + 1.0) * scale / 2.0 - 0.5


def project_normalized(points: torch.Tensor, homography: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Nx2 normalized points and return finite, non-degenerate status."""

    if homography.ndim == 3:
        if homography.shape[0] != 1:
            raise ValueError("Only one image pair is supported per matcher call")
        homography = homography[0]
    homogeneous = torch.cat((points.float(), torch.ones_like(points[:, :1], dtype=torch.float32)), dim=-1)
    warped_h = homogeneous @ homography.float().T
    denominator = warped_h[:, 2]
    valid = denominator.abs() > 1e-8
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    warped = warped_h[:, :2] / safe_denominator[:, None]
    valid &= torch.isfinite(warped).all(dim=-1)
    return warped, valid


def inside_normalized(points: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    inside = (points >= -1.0).all(dim=-1) & (points <= 1.0).all(dim=-1)
    return inside if valid is None else inside & valid


def stable_greedy_one_to_one(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    cap: int,
) -> torch.Tensor:
    """Return indices of descending-confidence unique source/target pairs."""

    if not source.numel():
        return torch.empty(0, dtype=torch.long, device=source.device)
    # Stable sorting makes tied results reproducible in flat candidate order.
    order = torch.argsort(confidence, descending=True, stable=True)
    chosen: list[int] = []
    used_source: set[int] = set()
    used_target: set[int] = set()
    for raw_index in order.detach().cpu().tolist():
        a = int(source[raw_index].item())
        b = int(target[raw_index].item())
        if a in used_source or b in used_target:
            continue
        used_source.add(a)
        used_target.add(b)
        chosen.append(raw_index)
        if len(chosen) >= cap:
            break
    return torch.tensor(chosen, dtype=torch.long, device=source.device)


def l2_normalize_descriptors(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), p=2, dim=-1, eps=1e-8)
