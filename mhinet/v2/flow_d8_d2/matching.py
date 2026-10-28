"""Sparse D8 search around Flow0, followed by the existing 4x4 D2 matcher."""
import torch
import torch.nn.functional as F
from mhinet.v2.coarse_matcher import flow_prior_from_shared,match_d8
from mhinet.v2.supervision import direct_d8_fine_grids,valid_points
from mhinet.v2.joint_single.path import source_grid


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


@torch.no_grad()
def infer(shared,matcher,variant):
    """Discrete D2 baseline used by earlier Flow/D2 and QRRU comparisons.

    Call select_local_parents when only D8 parents are needed.
    """
    d8=shared['pyramid'][8][0].float();d2=shared['pyramid'][2][0].float();c=matcher.config
    threshold=.1 if variant=='local_mat01' else 0.
    prior=flow_prior_from_shared(shared,0,(64,64),min_matchability=threshold)
    if variant=='global':
        result=match_d8(d8[0],d8[1],temperature=float(matcher.tau_c),threshold=c.confidence_threshold,
            max_coarse=c.max_coarse,policy='mutual',force_chunked=False,flow_prior=prior,
            flow_lambda=c.flow_lambda,flow_sigma_d8=c.flow_sigma_d8)
        si,ti,score=result.source_flat,result.target_flat,result.confidence
        counts=dict(source_queries=4096,edges=4096**2)
    else:si,ti,score,counts=local_coarse(d8,prior,matcher)
    points=[];support=torch.ones((256,256),device=d2.device,dtype=torch.bool)
    for start in range(0,len(si),c.inference_window_chunk):
        ga,gb=direct_d8_fine_grids(si[start:start+c.inference_window_chunk],ti[start:start+c.inference_window_chunk],(64,64),(256,256))
        va,vb=valid_points(support,ga),valid_points(support,gb)
        lp=matcher.fine_scores(d2[0],d2[1],ga,gb,va,vb)
        best,reverse=lp.argmax(-1),lp.argmax(-2)
        ids=torch.arange(16,device=d2.device)[None].expand_as(best)
        good=va&vb.gather(1,best)&(reverse.gather(1,best)==ids)
        m,s=good.nonzero(as_tuple=True);t=best[m,s]
        confidence=(lp[m,s,t].exp()*score[start:start+c.inference_window_chunk][m]).sqrt()
        keep=confidence>=c.confidence_threshold
        points.append((ga[m[keep],s[keep]],gb[m[keep],t[keep]],confidence[keep],m[keep]+start))
    if points:
        a,b,confidence,parent=[torch.cat([p[k] for p in points]) for k in range(4)]
    else:
        a=b=d2.new_empty((0,2));confidence=d2.new_empty(0);parent=si.new_empty(0)
    before=len(a);order=confidence.argsort(descending=True,stable=True)[:min(c.max_matches,c.max_output_matches)]
    xy=lambda index: (torch.stack((index%64,index//64),-1).float()+.5)*8-.5
    return dict(coarse_a=xy(si),coarse_b=xy(ti),coarse_confidence=score,
        points_a=(a[order]+1)*256-.5,points_b=(b[order]+1)*256-.5,confidence=confidence[order],
        parent=parent[order],fine_before_cap=before,prior=prior,**counts)
