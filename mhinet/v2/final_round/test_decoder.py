import unittest
import torch
from .decoder import LocalResidual


class DecoderTests(unittest.TestCase):
    def test_corner_has_no_duplicate_probability_mass(self):
        model=LocalResidual()
        probability=torch.zeros(1,1024,1025)
        probability[...,0]=.4;probability[...,1]=.3;probability[...,32]=.2;probability[...,33]=.1
        context=torch.randn(1,2,32,32,1024)
        warp,info=model(dict(stage1=dict(match_probabilities=probability),context=context))
        expected=torch.tensor([7.5+16*.4,7.5+16*.3])
        self.assertTrue(torch.allclose(info['coarse_px'][0,0],expected))
        self.assertTrue(torch.allclose(info['local_mass'],torch.ones_like(info['local_mass'])))
        self.assertEqual(float(info['delta_px'].detach().abs().max()),0)
        self.assertEqual(tuple(warp.shape),(1,2,32,32))
        self.assertTrue(torch.allclose((warp[0,:,0,0]+1)*256-.5,expected))

    def test_residual_bound_and_finite_gradients(self):
        model=LocalResidual();model.net[-1].bias.data.fill_(100)
        logits=torch.randn(1,1024,1025,requires_grad=True)
        warp,info=model(dict(stage1=dict(match_probabilities=logits.softmax(-1)),context=torch.randn(1,2,32,32,1024)))
        self.assertLessEqual(float(info['delta_px'].detach().abs().max()),8)
        warp.sum().backward();self.assertTrue(torch.isfinite(logits.grad).all())


if __name__=='__main__':unittest.main()
