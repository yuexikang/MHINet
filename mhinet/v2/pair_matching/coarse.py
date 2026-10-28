"""Discrete sparse D8 selection, without executing Fine matching."""
import torch
import torch.nn.functional as F
from ..coarse_matcher import flow_prior_from_shared
from .continuous import source_grid

def local_coarse(d8,prior,matcher,radius=4):
    c=matcher.config;device=d8.device
    source=prior.valid.nonzero().flatten()
    if not len(source):
        return source,source.clone(),d8.new_empty(0),dict(source_queries=0,edges=0)
    center=prior.target_uv[source].round().long()
    yy,xx=torch.meshgrid(torch.arange(-radius,radius+1,device=device),torch.arange(-radius,radius+1,device=device),indexing='ij')
    uv=center[:,None]+torch.stack((xx.flatten(),yy.flatten()),-1)[None]
    valid=(uv>=0).all(-1)&(uv<64).all(-1)
    target=uv[...,1].clamp(0,63)*64+uv[...,0].clamp(0,63)
    fa=F.normalize(d8[0].flatten(1).T,dim=-1);fb=F.normalize(d8[1].flatten(1).T,dim=-1)
    logits=(fa[source,None]*fb[target]).sum(-1)/matcher.tau_c
    distance=(uv-prior.target_uv[source,None]).square().sum(-1)
    logits+=c.flow_lambda*prior.confidence[source,None]*torch.exp(-distance/(2*c.flow_sigma_d8**2))
    logits=logits.masked_fill(~valid,-torch.inf)
    # Sparse dual softmax: column normalization uses exactly the same valid edges.
    colmax=logits.new_full((4096,),-torch.inf)
    colmax.scatter_reduce_(0,target[valid],logits[valid],reduce='amax',include_self=True)
    colsums=logits.new_zeros(4096)
    colsums.scatter_add_(0,target[valid],(logits[valid]-colmax[target[valid]]).exp())
    collse=colmax+colsums.clamp_min(1e-30).log()
    rowbest=logits.argmax(-1);rowids=torch.arange(len(source),device=device)
    besttarget=target[rowids,rowbest];bestlogit=logits[rowids,rowbest]
    # Match the global backend's deterministic smallest-source tie break.
    reverse=torch.full((4096,),4096,device=device,dtype=torch.long)
    best_edges=valid & (logits==colmax[target])
    reverse.scatter_reduce_(0,target[best_edges],source[:,None].expand_as(target)[best_edges],reduce='amin',include_self=True)
    mutual=reverse[besttarget]==source
    confidence=(2*bestlogit-torch.logsumexp(logits,-1)-collse[besttarget]).exp()
    keep=mutual&(confidence>=c.confidence_threshold)
    si,ti,score=source[keep],besttarget[keep],confidence[keep]
    order=score.argsort(descending=True,stable=True)[:c.max_coarse]
    return si[order],ti[order],score[order],dict(source_queries=len(source),edges=int(valid.sum()))


@torch.no_grad()
def select_local_parents(shared, matcher, max_parents=128):
    """Select source D8 parents without running D2 matching; native pixel coordinates."""
    prior = flow_prior_from_shared(shared, 0, (64, 64))
    si, _, _, _ = local_coarse(shared['pyramid'][8][0].float(), prior, matcher)
    return source_grid(si.device)[si[:max_parents]]
