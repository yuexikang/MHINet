"""Exact-H or uncertainty-region supervision, with fixed labels and live predictions."""
import torch
import torch.nn.functional as F
from mhinet.v2.losses import flow_head_loss
from mhinet.v2.supervision import coarse_labels
from mhinet.downstream.losses import coarse_positive_logp, positive_focal
from .path import source_grid


def position_loss(prediction, truth, valid, radius):
    error = ((prediction-truth).norm(dim=-1)-radius).clamp_min(0) / 8
    return torch.log1p(error[valid].square()).mean() if valid.any() else prediction.sum()*0


def local_loss(source, prediction, candidates, logits, candidate_valid, gt):
    with torch.no_grad():
        target, known, visible = gt.query((source.detach()+.5)/256-1)
        truth = (target+1)*256-.5
        valid = known & visible
        lower = candidates.detach().masked_fill(~candidate_valid[..., None], torch.inf).amin(1)
        upper = candidates.detach().masked_fill(~candidate_valid[..., None], -torch.inf).amax(1)
        covered = valid & (truth >= lower).all(-1) & (truth <= upper).all(-1)
        distance = (candidates.detach()-truth[:, None]).square().sum(-1)
        # Fixed interpolation weights for exact targets; uncertainty-region mass for weak H.
        if gt.exact:
            delta = (candidates.detach()-truth[:, None]).abs()
            spacing = (candidates.detach()[:, 1, 0]-candidates.detach()[:, 0, 0]).abs().clamp_min(1)
            labels = (1-delta/spacing[:, None, None]).clamp_min(0).prod(-1)*candidate_valid
        else:
            spacing = (candidates.detach()[:, 1, 0]-candidates.detach()[:, 0, 0]).abs()
            labels = (distance.sqrt() <= gt.tolerance_px+spacing[:, None]/2**.5) & candidate_valid
            labels = labels.float()
        usable = covered & (labels.sum(-1)>0)
        labels = labels/labels.sum(-1,keepdim=True).clamp_min(1e-8)
    logp = logits.masked_fill(~candidate_valid, -1e4).log_softmax(-1)
    if gt.exact:
        classification = -(labels[usable]*logp[usable]).sum(-1).mean() if usable.any() else logits.sum()*0
    else:
        selected = logp[usable].masked_fill(labels[usable]<=0, -1e4)
        classification = -torch.logsumexp(selected,-1).mean() if usable.any() else logits.sum()*0
    geo = position_loss(prediction, truth, covered, 0 if gt.exact else gt.tolerance_px)
    return classification+geo, truth, covered, dict(visible=int(valid.sum()), covered=int(covered.sum()))


def flow_loss(shared, gt):
    if gt.exact:
        return flow_head_loss(shared['stage1'], geometry=gt)[0]
    # Real approximate H supplies no certified non-correspondence labels here.
    field = shared['flow16_target_norm'][0]
    y,x=torch.meshgrid(torch.arange(32,device=field.device),torch.arange(32,device=field.device),indexing='ij')
    source=(torch.stack((x,y),-1).reshape(-1,2)+.5)*16-.5
    with torch.no_grad():
        target,known,visible=gt.query((source+.5)/256-1)
        valid=known&visible
        truth=(target+1)*256-.5
        difference=(source[None]-truth[:,None]).abs()
        region=(difference<=8+gt.tolerance_px).all(-1)
        valid &= region.any(-1)
    predicted=(field.flatten(1).T+1)*256-.5
    # Retain the legacy normalized-coordinate robust scale for Flow regression.
    residual=(predicted-truth).abs().sub(gt.tolerance_px).clamp_min(0)/256
    geo=torch.log1p(.5*(residual[valid]/.1).square()).sum()/valid.sum().clamp_min(1)
    logp=shared['stage1']['match_probabilities'][0].clamp_min(1e-9).log()
    cls=-torch.logsumexp(logp[valid,:-1].masked_fill(~region[valid],-1e4),-1).mean() if valid.any() else predicted.sum()*0
    return geo+.0001*cls


def global_d8(shared, matcher, gt):
    d8=shared['pyramid'][8][0].float()
    ma=gt.mask_a if gt.mask_a is not None else d8.new_ones(1,512,512)
    mb=gt.mask_b if gt.mask_b is not None else d8.new_ones(1,512,512)
    i,j=coarse_labels(gt,ma,mb,(64,64))
    take=torch.randperm(len(i),device=d8.device)[:256]
    if not len(take):return d8.sum()*0
    a,b=[F.normalize(x.flatten(1).T,dim=-1) for x in d8]
    return positive_focal(coarse_positive_logp(a,b,matcher.tau_c,i[take],j[take],128))


def losses(shared, matcher, result, gt):
    l8,_,_,counts8=local_loss(result['source'],result['p8'],result['candidates8'],result['logits8'],result['valid8'],gt)
    l2,truth,covered,counts2=local_loss(result['fine_source'],result['p2'],result['candidates2'],result['logits2'],result['valid2'],gt)
    final=position_loss(result['final'],truth,covered,0 if gt.exact else gt.tolerance_px)
    flow=flow_loss(shared,gt)
    broad=global_d8(shared,matcher,gt)
    total=flow+broad+l8+l2+final
    return total, dict(flow=float(flow.detach()),d8_global=float(broad.detach()),d8_local=float(l8.detach()),
                       d2=float(l2.detach()),final=float(final.detach()),d8_counts=counts8,d2_counts=counts2),final
