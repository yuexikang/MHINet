import numpy as np
import torch
from .labels import support_labels, MATCH, NO_MATCH, UNKNOWN
from .inference import conditional_output
from .losses import existence_loss
from .nomatch_head import NoMatchHead, query_features


def test_full_overlap():
    labels = support_labels(np.eye(3))
    assert (labels['state']==MATCH).all()
    np.testing.assert_allclose(labels['source_xy'],labels['gt_target_xy'],atol=1e-5)


def test_low_overlap():
    H = np.eye(3); H[0,2] = 384
    labels = support_labels(H)
    assert (labels['state'].reshape(32,32)[:,:8]==MATCH).all()
    assert (labels['state'].reshape(32,32)[:,8:]==NO_MATCH).all()
    assert np.isnan(labels['gt_target_xy'][labels['state']==NO_MATCH]).all()


def test_texture_support():
    for reverse in (False,True):
        labels = support_labels(np.eye(3),[100,100,200,200],reverse)
        state = labels['state'].reshape(32,32)
        assert state[8,8] == NO_MATCH
        assert state[6,6] == UNKNOWN
        assert state[0,0] == MATCH
        assert (labels['negative_kind'][labels['state']==NO_MATCH]==2).all()

    H=np.eye(3);H[0,2]=384
    geometric=support_labels(H,[100,100,200,200],True)
    assert geometric["state"].reshape(32,32)[8,8]==NO_MATCH
    assert geometric["negative_kind"].reshape(32,32)[8,8]==1


def test_u0_frozen():
    from mhinet.v2.u0 import load_u0
    torch.set_num_threads(4)
    torch.manual_seed(42)
    u0 = load_u0(device='cuda:0')
    images = torch.rand(1,2,3,512,512,device='cuda:0')
    with torch.no_grad():
        before = u0(images,pyramid_scales=())
        features,_ = query_features(before,u0.stage1_head.no_match_logit)
    head = NoMatchHead().cuda(0)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.001)
    state = torch.arange(1024,device='cuda:0')[None]%3
    loss = existence_loss(head(features),state)
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in u0.parameters())
    assert head[0].weight.grad.abs().sum() > 0
    assert not set(map(id,u0.parameters())) & set(map(id,optimizer.param_groups[0]['params']))
    optimizer.step()
    with torch.no_grad():
        after = u0(images,pyramid_scales=())
    assert torch.equal(before['flow16_target_norm'],after['flow16_target_norm'])


def test_conditional_output():
    p = torch.full((2,1024),.5)
    p[0,7] = .1; p[1,9] = .1; p[0,10] = .9
    result = conditional_output(torch.zeros(2,2,32,32),p,.2,.8)
    assert result['accepted_indices'].tolist()==[7,1033]
    assert result['target_xy'].shape==(2,2)
    assert result['state'][0].flatten()[10] == NO_MATCH
    assert result['state'][1].flatten()[0] == UNKNOWN
    assert (result['state'].flatten()[result['accepted_indices']]==MATCH).all()

    tied = conditional_output(torch.zeros(1,2,32,32),torch.ones(1,1024),1.,float(np.nextafter(1.,np.inf)))
    assert (tied["state"]==MATCH).all()
