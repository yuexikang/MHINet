"""Focused checks: identity, pair symmetry, padding masks, and trainable ranking."""
import torch
from .model import LocalPairModulation, match_windows, center_indices
from .experiment import loss_fn


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    a,b=torch.randn(3,64,256),torch.randn(3,64,256)
    va,vb=torch.ones(3,64,dtype=torch.bool),torch.ones(3,64,dtype=torch.bool)
    va[:,:4]=False;vb[:,-4:]=False
    module=LocalPairModulation('pair')
    x,y=module(a,b,va,vb)
    assert torch.equal(x,a) and torch.equal(y,b)
    with torch.no_grad():module.condition[-1].weight.normal_(0,.001)
    x,y=module(a,b,va,vb);sy,sx=module(b,a,vb,va)
    torch.testing.assert_close(x,sx);torch.testing.assert_close(y,sy)
    bad_a=a.clone();bad_a[~va]=10000
    bad_b=b.clone();bad_b[~vb]=-10000
    mx,my=module(bad_a,bad_b,va,vb)
    torch.testing.assert_close(x[va],mx[va]);torch.testing.assert_close(y[vb],my[vb])
    # Known identity correspondences, with 16 center source tokens and 64 targets.
    axis=torch.arange(8).float()*2+20;ygrid,xgrid=torch.meshgrid(axis,axis,indexing='ij')
    candidates=torch.stack((xgrid,ygrid),-1).reshape(1,64,2).expand(3,-1,-1)
    source=candidates[:,center_indices('cpu')]
    record=dict(a=a,b=b,valid_a=va,valid_b=vb,source=source,candidates=candidates,
                truth=source.clone(),labeled=torch.ones(3,16,dtype=torch.bool))
    pair=LocalPairModulation('pair');base=LocalPairModulation('none')
    baseline=match_windows(base,record,.1)
    initial=match_windows(pair,record,.1)
    torch.testing.assert_close(initial['logp'],baseline['logp'],rtol=0,atol=0)
    torch.testing.assert_close(initial['points_b'],baseline['points_b'],rtol=0,atol=0)
    opt=torch.optim.AdamW(pair.parameters(),lr=.001)
    for step in range(2):
        opt.zero_grad();result=match_windows(pair,record,.1);loss,n=loss_fn(result,record)
        assert n>0 and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pair.parameters())
        if step:assert pair.condition[0].weight.grad.abs().sum()>0
        opt.step()
    print('PASS: identity logits/coordinates, A/B symmetry, masked statistics, two-step classification backward')

if __name__=='__main__':main()
