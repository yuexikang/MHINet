"""V2 geometry: inherit GT supervision, add homography-free D8 child grids."""
import torch
from mhinet.downstream.supervision import (to_uv, to_norm, sample_uv, valid_points,
    legal_h, coarse_labels, fine_labels, offsets, interpolate_controls, qrru_targets,
    fine_grids as h0_residual_fine_grids)


def direct_d8_fine_grids(source_flat, target_flat, coarse_hw, fine_hw):
    """Expand each matched D8 parent pair into its exact 4x4 D2 children.

    ``b = grid[j] + a - grid[i]``. Both lattices remain cell-centered and are
    independent of a predicted H, Flow, GT, or an inferred target overlap mask.
    """
    hc, wc = coarse_hw
    hf, wf = fine_hw
    if (hf, wf) != (4 * hc, 4 * wc):
        raise ValueError('Require D8-to-D2 scale ratio four')
    if source_flat.ndim != 1 or target_flat.shape != source_flat.shape:
        raise ValueError('Matched parent indices must be equal-length vectors')
    if bool(((source_flat < 0) | (source_flat >= hc * wc) | (target_flat < 0) | (target_flat >= hc * wc)).any()):
        raise ValueError('D8 parent index out of bounds')
    axis = torch.arange(4, device=source_flat.device)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    children = torch.stack((x.flatten(), y.flatten()), -1)
    def expand(indices):
        parent = torch.stack((indices % wc, indices // wc), -1)
        return to_norm((parent[:, None] * 4 + children[None]).float(), fine_hw)
    return expand(source_flat), expand(target_flat)
