"""Flow0 losses depend on GT geometry, never on a fitted predicted H."""
import torch
import torch.nn.functional as F

from mhinet.ops.geometry import safe_project_points
from mhinet.pretraining.loss import sample


def flow_head_loss(stage1, H_gt, mask_a, mask_b=None, *, mat_weight=.01,
                   cls_weight=.0001, robust_scale=.1):
    warp = stage1["coarse_warp"].float()
    batch, channels, h, w = warp.shape
    if channels != 2 or not torch.isfinite(warp).all():
        raise FloatingPointError("Invalid Flow0 target coordinates")
    y, x = torch.meshgrid((torch.arange(h, device=warp.device)+.5)*2/h-1,
                          (torch.arange(w, device=warp.device)+.5)*2/w-1, indexing="ij")
    grid = torch.stack((x, y), -1).reshape(1, -1, 2).expand(batch, -1, -1)
    target, legal, _ = safe_project_points(H_gt.float(), grid)
    matchable = legal & (target.abs() <= target.new_tensor([1-1/w, 1-1/h])).all(-1)
    # D16 cell centers lie at 7.5, 23.5, ... in a 512 image. Nearest
    # downsampling reads 0,16,... and shifts visibility labels by half a cell.
    source_visible = torch.stack([sample(mask_a[b].float(), grid[b])[:, 0] for b in range(batch)])
    matchable &= source_visible > .999
    if mask_b is not None:
        safe = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        visible = torch.stack([sample(mask_b[b].float(), safe[b])[:, 0] for b in range(batch)])
        matchable &= visible > .999
    prediction = warp.permute(0, 2, 3, 1).reshape(batch, -1, 2)
    residual = prediction[matchable] - target[matchable]
    geo = torch.log1p(.5*(residual/robust_scale).square()).sum()/matchable.sum().clamp_min(1)
    probability = stage1["coarse_matchability"][:, 0].flatten(1).float()
    classes = stage1["match_probabilities"].float()
    if not torch.isfinite(probability).all() or not torch.isfinite(classes).all():
        raise FloatingPointError("Nonfinite Flow0 probabilities")
    probability = probability.clamp(1e-6, 1-1e-6)
    positive = matchable.float(); negative = 1-positive
    mat = .5*(-(positive*probability.log()).sum()/positive.sum().clamp_min(1)
              -(negative*torch.log1p(-probability)).sum()/negative.sum().clamp_min(1))
    count, candidates = classes.shape[1:]
    side = round((candidates-1)**.5)
    if count != h*w or side*side != candidates-1:
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
    cls = -torch.where(matchable, positive_logp, logp[..., -1]).mean()
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
