"""LoMa GHIM four-term supervision, with safe projection/fit isolation.

Reference: LoMa/experiments/loretta_stage1_h/losses.py.
Coordinates and robust scale are normalized; H loss is not pixel corner L1.
"""
import torch
import torch.nn.functional as F
from mhinet.ops.geometry import safe_project_points


def ghim_loss(outputs, H_gt, overlap_mask, *, mat_weight=.01,
              cls_weight=1e-4, h_weight=.05, robust_scale=.1):
    warp = outputs['coarse_warp'].float()
    b, _, h, w = warp.shape
    y = (torch.arange(h, device=warp.device).float() + .5) * 2 / h - 1
    x = (torch.arange(w, device=warp.device).float() + .5) * 2 / w - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xx, yy), -1).reshape(1, -1, 2).expand(b, -1, -1)
    target, valid, _ = safe_project_points(H_gt.float(), grid)
    target = target.reshape(b, h, w, 2)
    matchable = valid.reshape(b, h, w) & (target.abs() <= 1).all(-1)
    matchable &= F.interpolate(overlap_mask.float(), (h, w), mode='nearest')[:, 0] > .5
    if not torch.isfinite(warp).all():
        raise FloatingPointError('Non-finite GHIM coarse warp')
    # Index before penalty evaluation: invalid target projections never enter it.
    residual = (warp.permute(0, 2, 3, 1)[matchable] - target[matchable])
    geo = torch.log1p(.5 * (residual / robust_scale).square()).sum() / matchable.sum().clamp_min(1)
    probability = outputs['coarse_matchability'][:, 0].float()
    probabilities = outputs['match_probabilities'].float()
    if not torch.isfinite(probability).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError('Non-finite GHIM probabilities')
    probability = probability.clamp(1e-6, 1 - 1e-6)
    positives = matchable.float()
    negatives = 1 - positives
    mat = .5 * (-(positives * probability.log()).sum() / positives.sum().clamp_min(1)
                - (negatives * torch.log1p(-probability)).sum() / negatives.sum().clamp_min(1))
    count, candidates = probabilities.shape[1:]
    side = round((candidates - 1) ** .5)
    if side * side != candidates - 1 or count != h * w:
        raise ValueError('GHIM candidate grid mismatch')
    # Non-match targets use the no-match class, not arbitrary huge coordinates.
    coords = torch.where(matchable[..., None], target, torch.zeros_like(target)).reshape(b, count, 2)
    xy = (coords + 1) * side / 2 - .5
    low = xy.floor()
    frac = xy - low
    logp = probabilities.clamp_min(1e-9).log()
    positive_logp = torch.zeros_like(logp[..., 0])
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ix = (low[..., 0].long() + dx).clamp(0, side - 1)
        iy = (low[..., 1].long() + dy).clamp(0, side - 1)
        weight = (frac[..., 0] if dx else 1 - frac[..., 0]) * (frac[..., 1] if dy else 1 - frac[..., 1])
        positive_logp = positive_logp + weight * logp.gather(2, (iy * side + ix)[..., None]).squeeze(-1)
    cls = -torch.where(matchable.reshape(b, count), positive_logp, logp[..., -1]).mean()
    prediction = outputs['H_A_to_B_norm'].float()
    gt = H_gt.float()
    safe = (outputs['fit_succeeded'].bool() & torch.isfinite(prediction).all(dim=(1, 2))
            & torch.isfinite(gt).all(dim=(1, 2)) & (prediction[:, 2, 2].abs() > 1e-6)
            & (gt[:, 2, 2].abs() > 1e-6))
    # Select before division, including failed fitter placeholders.
    pred_safe, gt_safe = prediction[safe], gt[safe]
    difference = pred_safe / pred_safe[:, 2:3, 2:3] - gt_safe / gt_safe[:, 2:3, 2:3]
    h_loss = difference[:, :].reshape(-1, 9)[:, :8].square().sum() / (safe.sum().clamp_min(1) * 8)
    total = geo + mat_weight * mat + cls_weight * cls + h_weight * h_loss
    return {'total': total, 'geo': geo, 'mat': mat, 'cls': cls, 'H': h_loss,
            'valid_H_pairs': safe.sum()}
