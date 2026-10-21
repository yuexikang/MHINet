"""V2 geometry: inherit GT supervision, add homography-free D8 child grids."""
import torch
from mhinet.downstream.supervision import (to_uv, to_norm, sample_uv, valid_points,
    legal_h, coarse_labels as _h_coarse_labels, fine_labels as _h_fine_labels, offsets, interpolate_controls,
    fine_grids as h0_residual_fine_grids)
from .geometry_gt import GeometryGT, qrru_targets
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers


@torch.no_grad()
def coarse_labels(geometry, mask_a, mask_b, hw):
    if not isinstance(geometry, GeometryGT):
        return _h_coarse_labels(geometry, mask_a, mask_b, hw)
    gt = geometry
    if gt.kind == 'homography' and gt.exact:
        return _h_coarse_labels(gt.H, mask_a, mask_b, hw)
    device = mask_a.device
    grid = normalized_cell_centers(*hw, device=device).reshape(-1, 2)
    target, known, visible = gt.query(grid)
    uv = to_uv(target, hw).round().long()
    h, w = hw
    good = known & visible & (uv >= 0).all(-1) & (uv < uv.new_tensor([w, h])).all(-1)
    j = uv[:, 1].clamp(0, h-1)*w+uv[:, 0].clamp(0, w-1)
    good &= valid_points(mask_b, grid[j])
    if gt.has_reverse:
        back, labeled, seen = gt.query(grid[j], reverse=True)
        good &= labeled & seen & (to_uv(back, hw).round() == to_uv(grid, hw).round()).all(-1)
    i = good.nonzero().flatten()
    return i, j[i]


@torch.no_grad()
def fine_labels(a, b, geometry, mask_a, mask_b, fine_hw, tolerance=1.5):
    if not isinstance(geometry, GeometryGT):
        return _h_fine_labels(a, b, geometry, mask_a, mask_b, fine_hw, tolerance)
    gt = geometry
    if gt.kind == 'homography' and gt.exact:
        return _h_fine_labels(a, b, gt.H, mask_a, mask_b, fine_hw, tolerance)
    truth, known, visible = gt.query(a)
    va = known & visible
    vb = valid_points(mask_b, b)
    distance = torch.cdist(to_uv(truth, fine_hw), to_uv(b, fine_hw))
    distance = distance.masked_fill(~(va[:, :, None] & vb[:, None, :]), float('inf'))
    nearest, reverse = distance.argmin(-1), distance.argmin(-2)
    ids = torch.arange(a.shape[1], device=a.device)[None].expand(a.shape[:-1])
    good = (distance.gather(-1, nearest[..., None]).squeeze(-1) <= tolerance)
    good &= reverse.gather(-1, nearest) == ids
    if gt.has_reverse:
        back, labeled, seen = gt.query(b, reverse=True)
        inverse = torch.cdist(to_uv(a, fine_hw), to_uv(back, fine_hw))
        inverse = inverse.masked_fill(~(va[:, :, None] & (vb & labeled & seen)[:, None, :]), float('inf'))
        good &= inverse.argmin(-2).gather(-1, nearest) == ids
        good &= (labeled & seen).gather(-1, nearest)
    m, k = good.nonzero(as_tuple=True)
    return m, k, nearest[m, k], va, vb


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
