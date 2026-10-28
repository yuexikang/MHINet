"""Exact/approximate geometry supervision for D8 and Pair-D2."""
import torch
import torch.nn.functional as F
from ..supervision import coarse_labels
from mhinet.downstream.losses import coarse_positive_logp, positive_focal

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


def global_d8(shared, matcher, gt):
    d8=shared['pyramid'][8][0].float()
    ma=gt.mask_a if gt.mask_a is not None else d8.new_ones(1,512,512)
    mb=gt.mask_b if gt.mask_b is not None else d8.new_ones(1,512,512)
    i,j=coarse_labels(gt,ma,mb,(64,64))
    take=torch.randperm(len(i),device=d8.device)[:256]
    if not len(take):return d8.sum()*0
    a,b=[F.normalize(x.flatten(1).T,dim=-1) for x in d8]
    return positive_focal(coarse_positive_logp(a,b,matcher.tau_c,i[take],j[take],128))




def exact_labels(record):
    """Four-neighbour soft targets; ignore queries outside the actual support."""
    c, truth = record['candidates'], record['truth']
    valid = record['valid_b']
    lo = c.masked_fill(~valid[...,None], torch.inf).amin(1)
    hi = c.masked_fill(~valid[...,None], -torch.inf).amax(1)
    covered = record['labeled'] & (truth>=lo[:,None]).all(-1) & (truth<=hi[:,None]).all(-1)
    delta = (c[:,None]-truth[:,:,None]).abs()
    weights = (1-delta/2).clamp_min(0).prod(-1)*valid[:,None]
    usable = covered & (weights.sum(-1)>0)
    weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    return weights, usable


def loss_fn(result, record):
    weights, usable = exact_labels(record)
    if not usable.any():
        return result['logp'].sum()*0, 0
    # All 64 candidates compete. Dual normalization also learns competition across queries.
    loss = -(weights[usable]*result['logp'][usable]).sum(-1).mean()
    return loss, int(usable.sum())
