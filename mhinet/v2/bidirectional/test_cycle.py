import torch
from mhinet.v2.feature_provider import normalized_d16_grid
from .cycle import cycle_consistency


def mapping(target):
    return dict(flow16_target_norm=target,flow16_valid=torch.isfinite(target).all(1,keepdim=True)&(target.abs()<=1).all(1,keepdim=True),flow16_matchability=torch.ones(1,1,32,32))


def test_identity_and_inverse_translation():
    g=normalized_d16_grid(1,device='cpu',dtype=torch.float32)
    c=cycle_consistency(mapping(g),mapping(g));assert c['valid'].all();assert c['error_px'].max()<1e-4
    shift=torch.tensor([.04,0.]).reshape(1,2,1,1)
    c=cycle_consistency(mapping(g+shift),mapping(g-shift))
    assert c['valid'].any() and not c['valid'].all()
    assert c['error_px'][c['valid']].max()<1e-4
    assert (c['reliability'][~c['valid']]==0).all()


def test_invalid_support_and_boundary():
    g=normalized_d16_grid(1,device='cpu',dtype=torch.float32)
    b=g.clone();b[:,:,10,10]=float('nan')
    c=cycle_consistency(mapping(g),mapping(b));assert not c['valid'][0,0,10,10]
    a=g.clone();a[:,:,0,0]=-.99
    c=cycle_consistency(mapping(a),mapping(g));assert not c['valid'][0,0,0,0]
    assert torch.isfinite(c['reliability']).all()
