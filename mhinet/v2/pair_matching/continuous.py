"""Continuous D8 localization; 512 native pixels, align_corners=False."""
import torch
import torch.nn.functional as F

def sample(feature, points):
    shape = points.shape[:-1]
    grid = ((points + .5) / 256 - 1).reshape(1, 1, -1, 2)
    values = F.grid_sample(feature[None].float(), grid, align_corners=False, padding_mode='border')
    return values[0, :, 0].T.reshape(*shape, feature.shape[0])


def source_grid(device):
    y, x = torch.meshgrid(torch.arange(64, device=device), torch.arange(64, device=device), indexing='ij')
    return (torch.stack((x, y), -1).reshape(-1, 2) + .5) * 8 - .5


def decode(logits, candidates, side):
    valid = (candidates >= 0).all(-1) & (candidates <= 511).all(-1)
    logits = logits.masked_fill(~valid, -1e4)
    peak = logits.detach().argmax(-1)
    index = torch.arange(side*side, device=logits.device)
    nearby = ((index[None] % side - peak[:, None] % side).abs() <= 1)
    nearby &= ((index[None] // side - peak[:, None] // side).abs() <= 1)
    support = nearby & valid
    probabilities = logits.masked_fill(~support, -1e4).softmax(-1) * support
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
    # Residual form preserves the live candidate-center coordinate, including at edges.
    anchor = candidates[:, 0]
    point = anchor + (probabilities[..., None] * (candidates-anchor[:, None])).sum(1)
    confidence = logits.softmax(-1).amax(-1) * valid.any(-1)
    return point, confidence, logits, valid


def coarse(shared, matcher, source):
    d8 = shared['pyramid'][8][0].float()
    p0 = source + sample(shared['flow16_norm'][0], source) * 256
    m = sample(shared['flow16_matchability'][0], source).squeeze(-1).detach()
    y, x = torch.meshgrid(torch.arange(-4, 5, device=source.device), torch.arange(-4, 5, device=source.device), indexing='ij')
    delta = torch.stack((x, y), -1).reshape(-1, 2).float() * 8
    candidates = p0[:, None] + delta
    a = F.normalize(sample(d8[0], source), dim=-1)
    b = F.normalize(sample(d8[1], candidates), dim=-1)
    logits = (a[:, None] * b).sum(-1) / matcher.tau_c
    logits = logits + 3*m[:, None]*torch.exp(-delta.square().sum(-1)[None]/(2*32**2))
    p8, score, logits, valid = decode(logits, candidates, 9)
    return dict(source=source, p0=p0, p8=p8, score8=score, candidates8=candidates, logits8=logits, valid8=valid)
