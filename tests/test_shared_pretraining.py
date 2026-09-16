import torch
import unittest
from mhinet.pretraining.model import LoRALinear
from mhinet.pretraining.loss import descriptor_loss, inverse_gt, sample
from mhinet.pretraining.evaluation import retrieval_metrics


def test_lora_alignment_and_second_step():
    torch.manual_seed(1)
    base = torch.nn.Linear(4, 12)
    layer = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(8, 4)
    assert torch.equal(base(x), layer(x))
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=.1)
    layer(x).square().mean().backward()
    assert layer.lora_A.grad.abs().sum() == 0
    assert layer.lora_B.grad.abs().sum() > 0
    assert base.weight.grad is None
    optimizer.step()
    optimizer.zero_grad()
    layer(x).square().mean().backward()
    assert layer.lora_A.grad.abs().sum() > 0


def test_identity_and_masks():
    torch.manual_seed(2)
    a = torch.randn(1, 1, 32, 12, 12)
    pair = a.repeat(1, 2, 1, 1, 1).requires_grad_()
    mask = torch.ones(1, 1, 24, 24)
    H = torch.eye(3)[None]
    good, info = descriptor_loss({2: pair}, H, mask, mask, queries=64)
    wrong_pair = torch.stack((pair[:,0],pair[:,1].flip(-1)),dim=1)
    wrong, _ = descriptor_loss({2: wrong_pair}, H, mask, mask, queries=64)
    assert good.isfinite() and wrong.isfinite()
    assert good < wrong
    assert all(v['correct'] == v['queries'] for v in info.values())
    good.backward()
    assert torch.isfinite(pair.grad).all()
    empty, info = descriptor_loss({2: pair}, H, mask * 0, mask, queries=64)
    assert empty == 0 and all(v['queries'] == 0 for v in info.values())
    empty_b, _ = descriptor_loss({2: pair}, H, mask, mask * 0, queries=64)
    assert empty_b == 0


def test_translation_coordinates():
    torch.manual_seed(3)
    a = torch.randn(1,32,12,12)
    pair = torch.stack((a,a.roll(1,dims=-1)),dim=1)
    mask = torch.ones(1,1,24,24)
    H = torch.eye(3)[None]
    H[:,0,2] = 2/12
    loss, info = descriptor_loss({2:pair},H,mask,mask,queries=64)
    wrong, _ = descriptor_loss({2:pair},torch.eye(3)[None],mask,mask,queries=64)
    assert loss < wrong
    assert all(v['correct']==v['queries'] for v in info.values())


def test_singular_label_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        inverse_gt(torch.zeros(1, 3, 3))


def test_full_gallery_identity():
    torch.manual_seed(7)
    a = torch.randn(1,1,32,12,12)
    pair = a.repeat(1,2,1,1,1)
    mask = torch.ones(1,1,24,24)
    metrics = retrieval_metrics({2:pair},torch.eye(3)[None],[mask,mask],queries=16)
    assert all(max(v['errors_input_px']) < 1e-4 for v in metrics.values())


def test_bilinear_reference_forward_backward():
    torch.manual_seed(8)
    x=torch.randn(3,9,11,requires_grad=True)
    xy=(torch.rand(75,2)*2.4-1.2).requires_grad_()
    actual=sample(x,xy)
    expected=torch.nn.functional.grid_sample(x[None],xy[None,None],align_corners=False)[0,:,0].T
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
    gx, gp=torch.autograd.grad(actual.sum(),(x,xy),retain_graph=True)
    ex, ep=torch.autograd.grad(expected.sum(),(x,xy))
    torch.testing.assert_close(gx,ex,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(gp,ep,atol=1e-5,rtol=1e-5)


def test_tiny_descriptor_overfit():
    torch.manual_seed(42)
    pair = torch.randn(1, 2, 16, 8, 8, requires_grad=True)
    mask = torch.ones(1, 1, 8, 8)
    optimizer = torch.optim.Adam([pair], lr=.1)
    history = []
    for _ in range(30):
        optimizer.zero_grad()
        loss, _ = descriptor_loss({2: pair}, torch.eye(3)[None], mask, mask, queries=64)
        history.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert history[-1] < history[0] * .1


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in list(globals().items())
                              if name.startswith('test_'))


if __name__ == '__main__':
    suite = load_tests(None,None,None)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
