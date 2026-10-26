"""Diagnostic alternative: fixed feature grids, live geometric-prior coordinates."""
import torch
import torch.nn.functional as F
from mhinet.downstream.supervision import offsets
from .path import sample, decode, fine as moving_fine


def coarse(shared, matcher, source):
    d8 = shared['pyramid'][8][0].float()
    p0 = source + sample(shared['flow16_norm'][0], source) * 256
    m = sample(shared['flow16_matchability'][0], source).squeeze(-1).detach()
    anchor = ((p0.detach()+.5)/8-.5).round()*8+3.5
    y,x = torch.meshgrid(torch.arange(-4,5,device=source.device),torch.arange(-4,5,device=source.device),indexing='ij')
    delta = torch.stack((x,y),-1).reshape(-1,2)*8
    candidates = anchor[:,None]+delta
    a = F.normalize(sample(d8[0],source),dim=-1)
    b = F.normalize(sample(d8[1],candidates),dim=-1)
    logits = (a[:,None]*b).sum(-1)/matcher.tau_c
    distance = (candidates-p0[:,None]).square().sum(-1)
    logits = logits+3*m[:,None]*torch.exp(-distance/(2*32**2))
    p8,score,logits,valid = decode(logits,candidates,9)
    return dict(source=source,p0=p0,p8=p8,score8=score,candidates8=candidates,logits8=logits,valid8=valid)


def fine(shared,matcher,r,side=4):
    if side%2:raise ValueError('Even side keeps all candidates on actual D2 centers')
    d2=shared['pyramid'][2][0].float();device=d2.device
    source_delta=offsets(device).reshape(-1,2)*2
    source=r['source'][:,None]+source_delta
    anchor=((r['p8'].detach()+.5)/2).round()*2-.5
    axis=(torch.arange(side,device=device)-(side-1)/2)*2
    y,x=torch.meshgrid(axis,axis,indexing='ij');delta=torch.stack((x,y),-1).reshape(-1,2)
    candidates=anchor[:,None]+delta
    a=F.normalize(sample(d2[0],source),dim=-1);b=F.normalize(sample(d2[1],candidates),dim=-1)
    logits=torch.einsum('nsc,ntc->nst',a,b)/matcher.tau_f
    # A bounded spatial prior provides the live p8 path without moving feature samples.
    expected=r['p8'][:,None]+source_delta
    distance=(candidates[:,None]-expected[:,:,None]).square().sum(-1)
    logits=logits+torch.exp(-distance/(2*4**2))
    valid=(candidates>=0).all(-1)&(candidates<=511).all(-1)
    logits=logits.masked_fill(~valid[:,None],-1e4)
    lp=logits.log_softmax(-1)+logits.log_softmax(-2)
    expanded=candidates[:,None].expand(-1,16,-1,-1).reshape(-1,side*side,2)
    p2,score,flat_lp,flat_valid=decode(lp.reshape(-1,side*side),expanded,side)
    source=source.reshape(-1,2)
    projected=matcher.qrru.project(d2)
    qr=matcher.qrru(projected[0],projected[1],(source+.5)/2-.5,(p2+.5)/2-.5,projected=True,iterations=1)
    best=lp.argmax(-1);reverse=lp.argmax(-2)
    mutual=(reverse.gather(1,best)==torch.arange(16,device=device)[None]).flatten()
    mutual &= flat_valid.gather(1,best.flatten()[:,None]).flatten()
    return dict(**r,fine_source=source,p2=p2,final=(qr['centers'][0]+.5)*2-.5,
        confidence=(score*r['score8'][:,None].expand(-1,16).reshape(-1)).sqrt(),
        candidates2=expanded,logits2=flat_lp,valid2=flat_valid,mutual=mutual)


def forward(shared,matcher,source,coarse_only=False):
    r=coarse(shared,matcher,source)
    return moving_fine(shared,matcher,r) if coarse_only else fine(shared,matcher,r)
