"""Direct cell-center overlap support rasterization for predicted and oracle modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .matching_utils import inside_normalized, normalized_cell_centers, project_normalized


@dataclass(frozen=True)
class MaskPair:
    a: torch.Tensor
    b: torch.Tensor


def predicted_overlap_masks(homography_a_to_b: torch.Tensor, resolution: int) -> MaskPair:
    """Rasterize A intersect H^-1(B), B intersect H(A) at exact cell centers."""

    device = homography_a_to_b.device
    grid = normalized_cell_centers(resolution, resolution, device=device).reshape(-1, 2)
    projected_b, valid_a = project_normalized(grid, homography_a_to_b)
    mask_a = inside_normalized(projected_b, valid_a).reshape(resolution, resolution)
    try:
        inverse = torch.linalg.inv(homography_a_to_b.float())
    except RuntimeError:
        empty = torch.zeros((resolution, resolution), dtype=torch.bool, device=device)
        return MaskPair(empty, empty.clone())
    projected_a, valid_b = project_normalized(grid, inverse)
    mask_b = inside_normalized(projected_a, valid_b).reshape(resolution, resolution)
    return MaskPair(mask_a, mask_b)


def load_oracle_mask(path: str | Path, resolution: int, device: torch.device | str) -> torch.Tensor:
    """Sample a native overlap mask at normalized cell centers without resize ambiguity."""

    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    native = torch.from_numpy(array > 0).to(device=device, dtype=torch.float32)[None, None]
    grid = normalized_cell_centers(resolution, resolution, device=device)[None]
    sampled = F.grid_sample(native, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return sampled[0, 0] > 0.5


def oracle_overlap_masks(
    mask_a_path: str | Path,
    mask_b_path: str | Path,
    resolution: int,
    device: torch.device | str,
) -> MaskPair:
    return MaskPair(
        load_oracle_mask(mask_a_path, resolution, device),
        load_oracle_mask(mask_b_path, resolution, device),
    )


def dilate_mask(mask: torch.Tensor, cells: int) -> torch.Tensor:
    if cells == 0:
        return mask
    kernel = 2 * cells + 1
    return F.max_pool2d(mask.float()[None, None], kernel, stride=1, padding=cells)[0, 0] > 0


def scale_d8_dilation_to_d2(mask: torch.Tensor, d8_cells: int) -> torch.Tensor:
    return dilate_mask(mask, d8_cells * 4)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    neighbors = F.avg_pool2d(mask.float()[None, None], 3, stride=1, padding=1)[0, 0]
    return (neighbors > 0) & (neighbors < 1)


def mask_diagnostics(predicted: MaskPair, oracle: MaskPair) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for label, pred, gt in (("A", predicted.a, oracle.a), ("B", predicted.b, oracle.b)):
        intersection = pred & gt
        gt_count = int(gt.sum().item())
        union = pred | gt
        boundary_union = _boundary(pred) | _boundary(gt)
        output.update(
            {
                f"valid_cells_{label}": int(pred.sum().item()),
                f"valid_fraction_{label}": float(pred.float().mean().item()),
                f"gt_recall_{label}": float(intersection.sum().item() / gt_count) if gt_count else 1.0,
                f"iou_{label}": float(intersection.sum().item() / union.sum().item()) if union.any() else 1.0,
                f"boundary_disagreement_{label}": float(((pred ^ gt) & boundary_union).sum().item() / boundary_union.sum().item()) if boundary_union.any() else 0.0,
            }
        )
    return output
