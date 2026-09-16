"""Bidirectional H-supervised sampled InfoNCE in normalized pixel-centre coordinates."""
import torch
import torch.nn.functional as F
from mhinet.ops.geometry import safe_project_points


def grid(h, w, device):
    y, x = torch.meshgrid((torch.arange(h, device=device) + .5) * 2 / h - 1,
                          (torch.arange(w, device=device) + .5) * 2 / w - 1, indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2)


def sample(feature, points):
    return F.grid_sample(feature[None].float(), points[None, None],
                         align_corners=False)[0, :, 0].T


def inverse_gt(H):
    # Reject illegal labels before inversion, never conceal a singular inverse.
    with torch.no_grad():
        if not torch.isfinite(H).all() or (torch.linalg.det(H.float()).abs() < 1e-8).any():
            raise ValueError('Invalid ground-truth homography')
        return torch.linalg.inv(H.float())


def directional_loss(a, b, H, mask_a, mask_b, *, queries, temperature, generator):
    h, w = a.shape[-2:]
    xy = grid(h, w, a.device)
    target, legal, _ = safe_project_points(H[None], xy[None])
    target, legal = target[0], legal[0]
    # Conservative support: all pixels covered by a descriptor cell must be valid.
    ma = F.adaptive_avg_pool2d(mask_a[None].float(), (h, w))[0]
    mb = F.adaptive_avg_pool2d(mask_b[None].float(), b.shape[-2:])[0]
    bounds = target.new_tensor([1 - 1 / b.shape[-1], 1 - 1 / b.shape[-2]])
    good = legal & (target.abs() <= bounds).all(-1) & (ma.flatten() > .999)
    good &= sample(mb, target)[:, 0] > .999
    indices = good.nonzero().flatten()
    indices = indices[torch.randperm(len(indices), device=a.device, generator=generator)[:queries]]
    if len(indices) < 2:
        return (a.sum() + b.sum()) * 0, {'queries': 0, 'correct': 0}
    qxy, txy = xy[indices], target[indices]
    q = F.normalize(sample(a, qxy), dim=-1)
    k = F.normalize(sample(b, txy), dim=-1)
    logits = q @ k.T / temperature
    # Other sampled correspondences are negatives except neighbours within 2 feature pixels.
    dist = torch.cdist(txy * txy.new_tensor([b.shape[-1]/2, b.shape[-2]/2]),
                       txy * txy.new_tensor([b.shape[-1]/2, b.shape[-2]/2]))
    allowed = dist > 2
    # Add fixed-radius hard negatives around each GT match, never within 2 pixels.
    offsets = txy.new_tensor([(dx * r, dy * r) for r in (3, 6)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))])
    local_xy = txy[:, None] + offsets[None] * txy.new_tensor([2/b.shape[-1], 2/b.shape[-2]])
    local_valid = (local_xy.abs() <= bounds).all(-1)
    local_valid &= (sample(mb, local_xy.reshape(-1,2))[:,0].reshape(len(txy),-1) > .999)
    local_features = F.normalize(sample(b, local_xy.reshape(-1,2)), dim=-1).reshape(len(txy),16,-1)
    local_logits = (q[:,None] * local_features).sum(-1) / temperature
    local_logits = local_logits.masked_fill(~local_valid, -1e4)
    usable = allowed.any(-1) | local_valid.any(-1)
    allowed.fill_diagonal_(True)
    logits = logits.masked_fill(~allowed, -1e4)
    logits = torch.cat((logits, local_logits), dim=1)
    labels = torch.arange(len(indices), device=a.device)
    if not usable.any():
        return logits.sum() * 0, {'queries': 0, 'correct': 0}
    loss = F.cross_entropy(logits[usable], labels[usable])
    return loss, {'queries': int(usable.sum()),
                  'correct': int((logits.argmax(-1)[usable] == labels[usable]).sum())}


def descriptor_loss(pyramid, H, mask_a, mask_b, *, queries=1024, temperature=.1, seed=0):
    if queries < 2 or temperature <= 0:
        raise ValueError('Need >=2 queries and positive temperature')
    reverse = inverse_gt(H)
    generator = torch.Generator(device=H.device).manual_seed(seed)
    losses, metrics = [], {}
    for scale, pairs in pyramid.items():
        for batch in range(len(pairs)):
            for direction, src, dst, transform, ma, mb in (
                ('ab', 0, 1, H[batch], mask_a[batch], mask_b[batch]),
                ('ba', 1, 0, reverse[batch], mask_b[batch], mask_a[batch])):
                loss, info = directional_loss(pairs[batch, src], pairs[batch, dst],
                    transform, ma, mb, queries=queries, temperature=temperature, generator=generator)
                losses.append(loss)
                metrics[f'D{scale}_{direction}_b{batch}'] = dict(loss=float(loss.detach()), **info)
    if not losses:
        raise ValueError('Empty pyramid')
    return torch.stack(losses).mean(), metrics
