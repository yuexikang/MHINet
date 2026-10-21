"""Fixed validation queries: observe all four QRRU iterations at every epoch."""
import torch
from mhinet.downstream.supervision import to_uv, to_norm
from .geometry_gt import qrru_targets


@torch.no_grad()
def observe(system, shared, gt, seed=123):
    qrru = system.matcher.qrru
    features = shared['pyramid'][2][0].float()
    hw = features.shape[-2:]
    generator = torch.Generator(device=features.device).manual_seed(seed)
    source = gt.sources(hw, features.device)
    take = torch.randperm(len(source), device=features.device, generator=generator)[:128]
    source = source[take]
    target, _, _ = gt.query(source)
    pa = to_uv(source, hw)
    noise = torch.randn(pa.shape, device=pa.device, generator=generator).mul(.8).clamp(-2.4, 2.4)
    pb = to_uv(target, hw)+noise
    # Fixed GT-query locations/noise across checkpoints isolate refinement learning.
    groups = dict(fixed_perturbation=(pa, pb))
    prediction = system.matcher.infer(shared, refine=False)[0]
    groups['actual_candidates'] = ((prediction['points_a'][:256]+.5)/2-.5,
                                   (prediction['points_b'][:256]+.5)/2-.5)
    projected = qrru.project(features)
    report = {}
    for name, (pa, pb) in groups.items():
        if not len(pa):
            report[name] = dict(points=0)
            continue
        _, _, truth, valid = qrru_targets(pa, pb, gt, hw=hw)
        if not bool(valid.any()):
            report[name] = dict(points=0)
            continue
        result = qrru(projected[0], projected[1], pa, pb, projected=True, history=True)
        trajectory = torch.cat((pb[None], result['centers']))
        errors = ((trajectory-truth[None])*2).norm(dim=-1)[:, valid]
        report[name] = dict(points=int(valid.sum()),
            epe_px=errors.mean(-1).tolist(),
            pre1=(errors <= 1).float().mean(-1).tolist(),
            pre3=(errors <= 3).float().mean(-1).tolist(),
            improved_vs_input=(errors < errors[0:1]).float().mean(-1).tolist(),
            damaged_vs_input=(errors > errors[0:1]).float().mean(-1).tolist(),
            max_control_d2=float(result['flows'].abs().max()))
    return report
