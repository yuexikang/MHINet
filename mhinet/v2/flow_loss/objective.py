"""The registered old objective and a calibrated radial pixel Smooth-EPE.

Every sample keeps the old GT cell centers, visibility and target-center box.
Losses normalize each pair independently before the batch macro mean. Geometry
never uses predicted confidence as a weight. L0 executes the original loss.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from mhinet.ops.geometry import safe_project_points
from mhinet.pretraining.loss import sample
from mhinet.v2.losses import flow_head_loss


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _source_grid(batch, hw, device):
    h, w = hw
    y, x = torch.meshgrid((torch.arange(h, device=device)+.5)*2/h-1,
                          (torch.arange(w, device=device)+.5)*2/w-1, indexing="ij")
    return torch.stack((x, y), -1).reshape(1, -1, 2).expand(batch, -1, -1)


@torch.no_grad()
def flow_supervision(H_gt, mask_a, mask_b=None, hw=(32, 32)):
    """Return old-compatible ``(target_norm[B,N,2], gt_valid[B,N])``.

    The target bounds are descriptor cell centers, ±(1-1/W), not ±1. Source
    and target masks are sampled bilinearly at their true pixel centers.
    """
    if len(hw) != 2 or any(type(value) is not int or value < 1 for value in hw):
        raise ValueError("hw must contain positive integer height and width")
    if H_gt.ndim != 3 or tuple(H_gt.shape[1:]) != (3, 3) or H_gt.shape[0] < 1:
        raise ValueError("H_gt must have shape Bx3x3")
    batch = H_gt.shape[0]
    for name, mask in (("mask_a", mask_a), ("mask_b", mask_b)):
        if mask is not None and (mask.ndim != 4 or mask.shape[:2] != (batch, 1)):
            raise ValueError(f"{name} must have shape Bx1xHxW")
    if mask_a is None:
        raise ValueError("mask_a is required")
    h, w = hw
    grid = _source_grid(batch, hw, H_gt.device)
    target, legal, _ = safe_project_points(H_gt.float(), grid)
    valid = legal & (target.abs() <= target.new_tensor([1-1/w, 1-1/h])).all(-1)
    visible_a = torch.stack([sample(mask_a[b].float(), grid[b])[:, 0] for b in range(batch)])
    valid &= visible_a > .999
    if mask_b is not None:
        safe_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        visible_b = torch.stack([sample(mask_b[b].float(), safe_target[b])[:, 0] for b in range(batch)])
        valid &= visible_b > .999
    return target, valid


def flow_point_diagnostics(stage1, H_gt, mask_a, mask_b=None, *, input_size=512):
    """Point tensors for shared GT-valid query populations and tail analysis.

    Invalid residuals/EPE are zero; consumers MUST select ``gt_valid`` before
    averaging. ``predicted_valid`` follows the old D16 center-box coverage;
    ``deployed_valid`` follows the raw provider ±1/probability validity.
    """
    if type(input_size) is not int or input_size != 512:
        raise ValueError("This pilot only accepts input_size=512")
    warp = stage1["coarse_warp"].float()
    if warp.ndim != 4 or warp.shape[1] != 2 or not bool(torch.isfinite(warp).all()):
        raise FloatingPointError("Invalid Flow target coordinates")
    batch, _, h, w = warp.shape
    if H_gt.shape[0] != batch:
        raise ValueError("Flow/GT batch size mismatch")
    probability = stage1["coarse_matchability"].float()
    if probability.shape != (batch, 1, h, w) or not bool(torch.isfinite(probability).all()):
        raise FloatingPointError("Invalid Flow matchability")
    target, valid = flow_supervision(H_gt, mask_a, mask_b, (h, w))
    prediction = warp.permute(0, 2, 3, 1).reshape(batch, -1, 2)
    probability = probability[:, 0].flatten(1)
    source = _source_grid(batch, (h, w), warp.device)
    # Select before subtraction/norm so invalid GT cannot form 0*NaN gradients.
    residual_px = torch.zeros_like(prediction)
    residual_px[valid] = (prediction[valid]-target[valid])*(input_size/2)
    epe_px = torch.zeros_like(probability)
    epe_px[valid] = residual_px[valid].norm(dim=-1)
    predicted_valid = (prediction.abs() <= prediction.new_tensor([1-1/w, 1-1/h])).all(-1)
    deployed_valid = ((prediction.abs() <= 1).all(-1)
                      & (probability >= 0) & (probability <= 1))
    def boundary_distance(points):
        pixels = (points+1)*(input_size/2)-.5
        return torch.minimum(pixels, input_size-1-pixels).amin(-1)
    return dict(source_norm=source, gt_target_norm=target, pred_target_norm=prediction,
                residual_px=residual_px, epe_px=epe_px, gt_valid=valid,
                predicted_valid=predicted_valid, deployed_valid=deployed_valid,
                matchability=probability, source_boundary_distance_px=boundary_distance(source),
                gt_boundary_distance_px=boundary_distance(target),
                pred_boundary_distance_px=boundary_distance(prediction))


def _stage_slice(stage1, index):
    return {name: stage1[name][index:index+1] for name in
            ("coarse_warp", "coarse_matchability", "match_probabilities")}


def _macro(records):
    keys = {key for record in records for key in record}
    result = {}
    for key in sorted(keys):
        values = [record[key] for record in records if isinstance(record.get(key), (int, float))]
        result[key] = sum(values)/len(values) if values else None
    return result


def flow_loss_components(stage1, H_gt, mask_a, mask_b=None, *, input_size=512,
                         robust_scale=.1, smooth_eps_px=.5, mat_weight=.01, cls_weight=.0001):
    """Return live ``geo_old/geo_new/mat/cls/old_total`` plus diagnostics.

    ``old_total`` runs the original production function independently per pair.
    Exposed individual terms reproduce that function for gradient inspection;
    they retain the graph and are never converted to scalars for optimization.
    """
    _positive_finite(robust_scale, "robust_scale")
    _positive_finite(smooth_eps_px, "smooth_eps_px")
    for value, name in ((mat_weight, "mat_weight"), (cls_weight, "cls_weight")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    points = flow_point_diagnostics(stage1, H_gt, mask_a, mask_b, input_size=input_size)
    batch, count = points["gt_valid"].shape
    classes = stage1["match_probabilities"].float()
    if classes.ndim != 3 or tuple(classes.shape[:2]) != (batch, count) or not bool(torch.isfinite(classes).all()):
        raise FloatingPointError("Invalid Flow classification probabilities")
    side = math.isqrt(classes.shape[2]-1)
    if side*side != classes.shape[2]-1:
        raise ValueError("Flow classification grid mismatch")
    lists = {name: [] for name in ("geo_old", "geo_new", "mat", "cls", "old_total")}
    records = []
    for b in range(batch):
        valid = points["gt_valid"][b]
        residual_norm = points["pred_target_norm"][b][valid]-points["gt_target_norm"][b][valid]
        geo_old = torch.log1p(.5*(residual_norm/robust_scale).square()).sum()/valid.sum().clamp_min(1)
        residual_px = residual_norm*(input_size/2)
        penalties = torch.sqrt(residual_px.square().sum(-1)+smooth_eps_px**2)-smooth_eps_px
        geo_new = penalties.mean() if penalties.numel() else penalties.sum()
        probability = points["matchability"][b].clamp(1e-6, 1-1e-6)
        positive = valid.float()
        negative = 1-positive
        mat = .5*(-(positive*probability.log()).sum()/positive.sum().clamp_min(1)
                  -(negative*torch.log1p(-probability)).sum()/negative.sum().clamp_min(1))
        coords = torch.where(valid[..., None], points["gt_target_norm"][b],
                             torch.zeros_like(points["gt_target_norm"][b]))
        xy = (coords+1)*side/2-.5
        low = xy.floor()
        fraction = xy-low
        logp = classes[b].clamp_min(1e-9).log()
        positive_logp = torch.zeros_like(logp[..., 0])
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ix = (low[..., 0].long()+dx).clamp(0, side-1)
            iy = (low[..., 1].long()+dy).clamp(0, side-1)
            weight = ((fraction[..., 0] if dx else 1-fraction[..., 0])
                      *(fraction[..., 1] if dy else 1-fraction[..., 1]))
            positive_logp = positive_logp+weight*logp.gather(1, (iy*side+ix)[..., None]).squeeze(-1)
        cls = -torch.where(valid, positive_logp, logp[..., -1]).mean()
        old_total, legacy_metrics = flow_head_loss(
            _stage_slice(stage1, b), H_gt[b:b+1], mask_a[b:b+1],
            None if mask_b is None else mask_b[b:b+1], mat_weight=mat_weight,
            cls_weight=cls_weight, robust_scale=robust_scale)
        for name, value in (("geo_old", geo_old), ("geo_new", geo_new), ("mat", mat),
                            ("cls", cls), ("old_total", old_total)):
            lists[name].append(value)
        records.append({**legacy_metrics, "geo_old": float(geo_old.detach()), "geo_new": float(geo_new.detach())})
    result = {name: torch.stack(values).mean() for name, values in lists.items()}
    if not all(bool(torch.isfinite(value)) for value in result.values()):
        raise FloatingPointError("Nonfinite Flow loss component")
    result.update(metrics=_macro(records), per_pair_metrics=records, point_diagnostics=points)
    return result


def compose_flow_loss(components: Mapping, *, alpha=1., use_new=False, mat_weight=.01, cls_weight=.0001):
    _positive_finite(alpha, "alpha")
    if use_new:
        return alpha*components["geo_new"]+mat_weight*components["mat"]+cls_weight*components["cls"]
    return components["old_total"]


def flow_objective(stage1, H_gt, mask_a, mask_b=None, *, arm="L0", alpha=1.,
                   mat_weight=.01, cls_weight=.0001, robust_scale=.1, smooth_eps_px=.5, input_size=512):
    """Runner API returning ``(live loss, old-compatible scalar metrics)``."""
    if arm not in ("L0", "L1"):
        raise ValueError("arm must be L0 or L1")
    terms = flow_loss_components(stage1, H_gt, mask_a, mask_b, input_size=input_size,
        robust_scale=robust_scale, smooth_eps_px=smooth_eps_px,
        mat_weight=mat_weight, cls_weight=cls_weight)
    loss = compose_flow_loss(terms, alpha=alpha, use_new=arm == "L1",
                             mat_weight=mat_weight, cls_weight=cls_weight)
    metrics = {**terms["metrics"], "alpha": alpha, "arm": arm,
               "objective_loss": float(loss.detach()),
               "active_geo_weighted": float((alpha*terms["geo_new"] if arm == "L1" else terms["geo_old"]).detach())}
    return loss, metrics


__all__ = ["flow_supervision", "flow_point_diagnostics", "flow_loss_components",
           "flow_objective", "compose_flow_loss"]
