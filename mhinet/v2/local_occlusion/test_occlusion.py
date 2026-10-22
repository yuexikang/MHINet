"""Targeted checks for the new augmentation/ignore/gradient contracts."""
import unittest
import torch
from mhinet.v2.geometry_gt import GeometryGT
from .occlusion import make_occlusion,label_rectangle,augmented_losses


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        self.gt=GeometryGT('homography',H=torch.eye(3),mask_a=torch.ones(1,512,512),mask_b=torch.ones(1,512,512),exact=False,tolerance_px=1.)
        self.images=torch.full((1,2,3,512,512),.3)
        self.config=dict(lambda_occ=.1,lambda_m=.1,lambda_keep=.1,lambda_keep_geo=1.,lambda_keep_match=.005)

    def test_three_states(self):
        t=label_rectangle(self.gt,(96,96,160,160))
        self.assertTrue(t['negative'][7*32+7])  # full support around (119.5,119.5)
        self.assertTrue(t['positive'][12*32+12])
        self.assertTrue(t['ignore'][6*32+6])  # support straddles rectangle edge
        self.assertTrue(t['ignore'][0])  # image boundary cannot certify full support
        self.assertTrue(((t['positive'].int()+t['negative'].int()+t['ignore'].int())==1).all())
        self.gt.mask_b[:,115:125,115:125]=0
        self.assertFalse(label_rectangle(self.gt,(96,96,160,160))['negative'][7*32+7])

    def test_reproducibility_and_pixel_values(self):
        a,t,r=make_occlusion(self.images,self.gt,'pair',1,42,fill=1.)
        b,_,s=make_occlusion(self.images,self.gt,'pair',1,42,fill=1.)
        self.assertEqual(r,s);self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a[:,0],self.images[:,0]))
        x0,y0,x1,y1=r['rect'];self.assertTrue((a[:,1,:,y0:y1,x0:x1]==1).all())
        self.assertLess(abs(r['actual_area']-r['requested_area']),.002)
        self.assertGreaterEqual(r['negative'],4)

    def test_no_visible_queries_skip(self):
        self.gt.mask_b.zero_()
        _,t,r=make_occlusion(self.images,self.gt,'none',1,42,max_attempts=2)
        self.assertTrue(r['skipped']);self.assertTrue(t['ignore'].all())

    def test_negative_and_ignore_have_no_geo_gradient(self):
        t=label_rectangle(self.gt,(96,96,160,160))
        warp=torch.zeros(1,2,32,32,requires_grad=True)
        logits=torch.randn(1,1024,1025,requires_grad=True)
        match=torch.full((1,1,32,32),.5,requires_grad=True)
        stage=dict(coarse_warp=warp,match_probabilities=logits.softmax(-1),coarse_matchability=match)
        losses=augmented_losses(stage,self.gt,t,self.config);sum(losses.values()).backward()
        grad=warp.grad[0].permute(1,2,0).reshape(-1,2)
        self.assertTrue((grad[~t['positive']]==0).all());self.assertGreater(float(grad[t['positive']].abs().sum()),0)
        self.assertTrue((logits.grad[0,t['ignore']]==0).all())
        self.assertTrue((logits.grad[0,t['negative'],-1]<0).all())
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__=='__main__':unittest.main()
