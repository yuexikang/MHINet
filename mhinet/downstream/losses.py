"""Stable positive focal matching and cumulative QRRU supervision."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def dual_log_probability(logits, valid=None):
    if valid is None:
        valid = torch.ones_like(logits,dtype=torch.bool)
    # Finite mask constant prevents all-masked row/column NaN gradients.
    x = logits.float().masked_fill(~valid,-1e4)
    return (2*x-torch.logsumexp(x,-1,keepdim=True)-torch.logsumexp(x,-2,keepdim=True)).masked_fill(~valid,-1e4)


def positive_focal(logp):
    return (-(1-logp.exp()).square()*logp).mean() if logp.numel() else logp.sum()


def coarse_positive_logp(a,b,temperature,i,j,chunk=128):
    """Exact full denominators for selected positives, rematerialized on backward.

    Only positive rows/columns affect positive-only focal; each denominator still
    includes every opposing grid cell. Saves O(Npositive*Ngrid) activations.
    """
    if not len(i):
        return (a.sum()+b.sum()+temperature)*0 + a.new_empty((0,))
    def denominator(query,keys,tau):
        return torch.logsumexp((query.float()@keys.float().T)/tau.float(),-1)
    def blocks(query,keys):
        return torch.cat([checkpoint(denominator,query[s:s+chunk],keys,temperature,use_reentrant=False)
                          for s in range(0,len(query),chunk)])
    row,col = blocks(a[i],b),blocks(b[j],a)
    selected = (a[i].float()*b[j].float()).sum(-1)/temperature.float()
    return 2*selected-row-col


def qrru_loss(result,target,valid,center_target,center_valid,gamma=.8):
    def average(error,mask):
        values = torch.sqrt(error.square()+1e-6)-.001
        return values[mask].mean() if bool(mask.any()) else values.sum()*0
    control = result['flows'].sum()*0
    center = result['centers'].sum()*0
    count = len(result['flows'])
    for t in range(count):
        control = control+gamma**(count-1-t)*average(
            (result['flows'][t]-target).permute(0,2,3,1),valid)
        center = center+gamma**(count-1-t)*average(result['centers'][t]-center_target,center_valid)
    return control+center,dict(control=control.detach(),center=center.detach())
