import torch

from mhinet.v2.local_refinement import LocalRefinement, local_refinement_loss
from mhinet.v2.qrru import QRRUV2


def test_matches_trained_calculation_and_empty_input():
    torch.manual_seed(14)
    reference = QRRUV2(iterations=1)
    with torch.no_grad():
        reference.head[-1].weight.normal_(std=.02)
        reference.head[-1].bias.normal_(std=.02)
    refiner = LocalRefinement(chunk_size=3)
    refiner.load_state_dict(reference.state_dict(), strict=True)
    f = torch.randn(2, 256, 16, 16)
    a = torch.rand(7, 2)*25
    b = a+.1
    projected = reference.project(f)
    r = reference(projected[0], projected[1], (a+.5)/2-.5, (b+.5)/2-.5, projected=True)
    expected = b + r['flows'][0].mean((-2, -1))*2
    torch.testing.assert_close(refiner(f, a, b), expected, rtol=1e-6, atol=1e-6)
    assert refiner(f, a[:0], b[:0]).shape == (0, 2)


def test_identity_and_projection_gradient():
    torch.manual_seed(1)
    refiner = LocalRefinement()
    f = torch.randn(2, 256, 16, 16)
    a = torch.tensor([[8.2, 10.3], [14.7, 16.9]])
    b = a+.2
    torch.testing.assert_close(refiner(f,a,b),b,atol=0,rtol=0)
    optimizer = torch.optim.AdamW(refiner.parameters(),lr=1e-3)
    for update in range(2):
        optimizer.zero_grad()
        loss,_ = local_refinement_loss(b,refiner(f,a,b),b+1,torch.ones(2,dtype=torch.bool))
        loss.backward()
        if update:
            assert refiner.project.weight.grad.abs().sum()>0
        optimizer.step()


def test_harm_loss_and_reference_stopgrad():
    before = torch.tensor([[2.,0.],[2.,0.],[10.,0.]],requires_grad=True)
    after = torch.tensor([[3.,0.],[1.,0.],[20.,0.]],requires_grad=True)
    loss,detail = local_refinement_loss(before,after,torch.zeros_like(before),torch.tensor([True,True,False]))
    assert loss.item()==2.5 and detail['harm'].item()==.5
    loss.backward()
    assert before.grad is None
    torch.testing.assert_close(after.grad,torch.tensor([[1.,0.],[.5,0.],[0.,0.]]))
