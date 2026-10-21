"""Flow0 losses depend on GT geometry, never on a fitted predicted H."""
import torch
import torch.nn.functional as F

from .geometry_gt import GeometryGT, as_geometry
from mhinet.pretraining.loss import sample


def flow_head_loss(stage1, H_gt=None, mask_a=None, mask_b=None, *, geometry=None, mat_weight=.01,
                   cls_weight=.0001, robust_scale=.1):
    warp = stage1["coarse_warp"].float()
    batch, channels, h, w = warp.shape
    if channels != 2 or not torch.isfinite(warp).all():
        raise FloatingPointError("Invalid Flow0 target coordinates")
    y, x = torch.meshgrid((torch.arange(h, device=warp.device)+.5)*2/h-1,
                          (torch.arange(w, device=warp.device)+.5)*2/w-1, indexing="ij")
    grid = torch.stack((x, y), -1).reshape(1, -1, 2).expand(batch, -1, -1)
    geometries = geometry if isinstance(geometry, (list, tuple)) else [geometry] if geometry is not None else [
        as_geometry(H_gt[b], mask_a[b], mask_b[b] if mask_b is not None else None) for b in range(batch)]
    if len(geometries) != batch:
        raise ValueError("One geometry per image pair is required")
    if any(gt.kind == 'sparse_points' for gt in geometries) and batch != 1:
        # Variable annotation counts are reduced per pair; dense legacy batches
        # continue to use their original global positive/negative normalization.
        values = [flow_head_loss({k: v[i:i+1] for k, v in stage1.items() if isinstance(v, torch.Tensor)},
            geometry=gt, mat_weight=mat_weight, cls_weight=cls_weight, robust_scale=robust_scale)
            for i, gt in enumerate(geometries)]
        return torch.stack([v[0] for v in values]).mean(), dict(valid_gt_points=sum(v[1]['valid_gt_points'] for v in values))
    sparse = geometries[0].kind == 'sparse_points'
    if sparse:
        grid = geometries[0].points_a[None]
    targets, labels, visibility = zip(*(gt.query(grid[b]) for b, gt in enumerate(geometries)))
    target, known, matchable = torch.stack(targets), torch.stack(labels), torch.stack(visibility)
    if not sparse:
        matchable &= (target.abs() <= target.new_tensor([1-1/w, 1-1/h])).all(-1)
    if grid.shape[1] == 0 or not bool(known.any()):
        return warp.sum()*0, dict(geo=0., mat=0., cls=0., flow_epe_px=None, valid_gt_points=0,
                                 supervised=False)
    prediction = warp.permute(0, 2, 3, 1).reshape(batch, -1, 2)
    probability = stage1["coarse_matchability"][:, 0].flatten(1).float()
    classes = stage1["match_probabilities"].float()
    if sparse:
        def at_queries(field):
            safe_grid = torch.where(known[..., None], grid, torch.zeros_like(grid))
            return F.grid_sample(field.float(), safe_grid[:, None], align_corners=False,
                                 padding_mode='border')[:, :, 0].transpose(1, 2)
        prediction = at_queries(warp)
        probability = at_queries(stage1['coarse_matchability'])[..., 0]
        classes = at_queries(classes.transpose(1, 2).reshape(batch, -1, h, w))
    residual = prediction[matchable] - target[matchable]
    if any(not gt.exact for gt in geometries):
        # Approximate geographic H is weak supervision, with an explicit dead zone.
        tolerance = torch.stack([target.new_full((target.shape[1], 2), gt.tolerance_px)
                                 / target.new_tensor([gt.hw_b[1]/2, gt.hw_b[0]/2]) for gt in geometries])
        residual = residual.sign()*(residual.abs()-tolerance[matchable]).clamp_min(0)
    geo = torch.log1p(.5*(residual/robust_scale).square()).sum()/matchable.sum().clamp_min(1)
    if not torch.isfinite(probability).all() or not torch.isfinite(classes).all():
        raise FloatingPointError("Nonfinite Flow0 probabilities")
    probability = probability.clamp(1e-6, 1-1e-6)
    positive = matchable.float(); negative = known.float()-positive
    mat = .5*(-(positive*probability.log()).sum()/positive.sum().clamp_min(1)
              -(negative*torch.log1p(-probability)).sum()/negative.sum().clamp_min(1))
    count, candidates = classes.shape[1:]
    side = round((candidates-1)**.5)
    if (not sparse and count != h*w) or side*side != candidates-1:
        raise ValueError("Flow0 classification grid mismatch")
    coords = torch.where(matchable[..., None], target, torch.zeros_like(target))
    xy = (coords+1)*side/2-.5
    low = xy.floor(); fraction = xy-low
    logp = classes.clamp_min(1e-9).log()
    positive_logp = torch.zeros_like(logp[..., 0])
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ix = (low[..., 0].long()+dx).clamp(0, side-1)
        iy = (low[..., 1].long()+dy).clamp(0, side-1)
        weight = (fraction[..., 0] if dx else 1-fraction[..., 0])*(fraction[..., 1] if dy else 1-fraction[..., 1])
        positive_logp += weight*logp.gather(2, (iy*side+ix)[..., None]).squeeze(-1)
    cls = -torch.where(matchable, positive_logp, logp[..., -1])[known].mean()
    total = geo + mat_weight*mat + cls_weight*cls
    with torch.no_grad():
        error = (prediction-target)*prediction.new_tensor([256., 256.])
        finite = torch.isfinite(prediction).all(-1)
        inside = finite & (prediction.abs() <= prediction.new_tensor([1-1/w, 1-1/h])).all(-1)
        epe = error.norm(dim=-1)[matchable]
        metrics = dict(geo=float(geo.detach()), mat=float(mat.detach()), cls=float(cls.detach()),
                       flow_epe_px=float(epe.mean()) if len(epe) else None,
                       valid_gt_points=int(matchable.sum()), predicted_valid_coverage=float(inside.float().mean()),
                       mean_matchability=float(probability.mean()))
    return total, metrics
