"""Optional historical global D8 control, never the production default."""
import torch
import torch.nn.functional as F
from mhinet.downstream.loma_reference.coarse_matcher import match_d8 as reference_d8
from ..coarse_matcher import match_d8, geometry_reward
from .continuous import sample, decode, source_grid
from .local import center_indices, match_windows

@torch.no_grad()
def global_infer(shared, matcher, adapter, prior=None, check_reference=False):
    d8 = shared['pyramid'][8][0].float()
    config = matcher.config
    common = dict(temperature=float(matcher.tau_c), threshold=config.confidence_threshold,
                  max_coarse=128, policy='mutual', force_chunked=False, chunk_rows=128)
    result = match_d8(d8[0], d8[1], flow_prior=prior,
                      flow_lambda=config.flow_lambda, flow_sigma_d8=config.flow_sigma_d8,
                      **common)
    if check_reference:
        support = torch.ones((64, 64), device=d8.device, dtype=torch.bool)
        ref = reference_d8(d8[0], d8[1], support, support, **common)
        torch.testing.assert_close(result.source_flat, ref.source_flat)
        torch.testing.assert_close(result.target_flat, ref.target_flat)
        torch.testing.assert_close(result.confidence, ref.confidence, rtol=1e-4, atol=1e-7)
    grid = source_grid(d8.device)
    source = grid[result.source_flat]
    if not len(source):
        return dict(points_a=source, points_b=source.clone(), confidence=source.new_empty(0),
                    coarse_a=source, coarse_b=source.clone())
    a = F.normalize(d8[0].flatten(1).T, dim=-1)
    b = F.normalize(d8[1].flatten(1).T, dim=-1)
    logits = a[result.source_flat] @ b.T / matcher.tau_c
    if prior is not None:
        logits += geometry_reward(prior, result.source_flat, torch.arange(4096, device=d8.device),
                                  width=64, strength=config.flow_lambda, sigma=config.flow_sigma_d8)
    # Same continuous 3x3-around-peak decode in both global variants.
    target, score, _, _ = decode(logits, grid[None].expand(len(source), -1, -1), 64)
    axis = (torch.arange(8, device=d8.device)-3.5)*2
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    delta = torch.stack((x, y), -1).reshape(1, 64, 2)
    ca, cb = source[:, None]+delta, target[:, None]+delta
    d2 = shared['pyramid'][2][0].float()
    windows = dict(a=sample(d2[0], ca), b=sample(d2[1], cb),
                   valid_a=(ca >= 0).all(-1) & (ca <= 511).all(-1),
                   valid_b=(cb >= 0).all(-1) & (cb <= 511).all(-1), candidates=cb)
    fine = match_windows(adapter, windows, matcher.tau_f)
    return dict(points_a=ca[:, center_indices(d8.device)].reshape(-1, 2),
                points_b=fine['points_b'].reshape(-1, 2),
                confidence=(fine['confidence']*score[:, None]).sqrt().flatten(),
                coarse_a=source, coarse_b=target)


