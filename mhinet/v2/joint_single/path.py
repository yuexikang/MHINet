"""Piecewise differentiable local matching; all public coordinates are native pixels."""
import torch
import torch.nn.functional as F
from mhinet.downstream.supervision import offsets


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


def fine(shared, matcher, result):
    d2 = shared['pyramid'][2][0].float()
    delta = offsets(d2.device).reshape(-1, 2) * 2
    source = result['source'][:, None] + delta
    candidates = result['p8'][:, None] + delta
    a = F.normalize(sample(d2[0], source), dim=-1)
    b = F.normalize(sample(d2[1], candidates), dim=-1)
    logits = torch.einsum('nsc,ntc->nst', a, b) / matcher.tau_f
    valid_b = (candidates >= 0).all(-1) & (candidates <= 511).all(-1)
    logits = logits.masked_fill(~valid_b[:, None], -1e4)
    # Dual-softmax compatibility; normalize selected rows when decoding coordinates.
    lp = logits.log_softmax(-1) + logits.log_softmax(-2)
    expanded = candidates[:, None].expand(-1, 16, -1, -1).reshape(-1, 16, 2)
    p2, score, lp, valid = decode(lp.reshape(-1, 16), expanded, 4)
    source = source.reshape(-1, 2)
    projected = matcher.qrru.project(d2)
    update = matcher.qrru(projected[0], projected[1], (source+.5)/2-.5, (p2+.5)/2-.5,
                          projected=True, iterations=1)
    final = (update['centers'][0]+.5)*2-.5
    confidence = (score * result['score8'][:, None].expand(-1, 16).reshape(-1)).sqrt()
    return dict(**result, fine_source=source, p2=p2, final=final, confidence=confidence,
                candidates2=expanded, logits2=lp, valid2=valid)


def forward(shared, matcher, source):
    return fine(shared, matcher, coarse(shared, matcher, source))


@torch.no_grad()
def infer(shared, matcher):
    grid = source_grid(shared['flow16_norm'].device)
    parts = [coarse(shared, matcher, grid[start:start+128]) for start in range(0, len(grid), 128)]
    result = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    take = result['score8'].argsort(descending=True, stable=True)[:128]
    result = fine(shared, matcher, {key:value[take] for key,value in result.items()})
    return result
