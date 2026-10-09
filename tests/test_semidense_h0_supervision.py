"""The independent GHIM loss repairs the detached semi-dense H0 gradient route."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from mhinet.downstream.training import SemidenseSystem
from mhinet.downstream.train import configure_h0_training


class TinyShared(nn.Module):
    def __init__(self,fit_succeeded=True):
        super().__init__()
        self.dino=nn.Linear(1,1,bias=False)
        self.mvt=nn.Linear(1,1,bias=False)
        self.stage1_head=nn.Linear(1,1,bias=False)
        self.fit_succeeded=fit_succeeded
        for module in (self.dino,self.mvt,self.stage1_head):
            nn.init.constant_(module.weight,.8)

    def set_training_groups(self,*,mvt,vgg,dedode,ghim_head):
        for module,enabled in ((self.dino,False),(self.mvt,mvt),(self.stage1_head,ghim_head)):
            for parameter in module.parameters():parameter.requires_grad_(enabled)

    def forward(self,images):
        size=len(images)
        feature=self.dino(images.mean((1,2,3,4))[:,None])
        context=self.mvt(feature)
        prediction=self.stage1_head(context).reshape(size,1,1,1)
        grid=prediction.new_tensor([[[-.5,.5],[-.5,.5]],[[-.5,-.5],[.5,.5]]])[None]
        warp=grid+prediction*.2
        matchability=prediction.sigmoid().expand(size,1,2,2)
        classes=prediction.new_tensor([-2.,-1.,0.,1.,2.])[None,None]
        probabilities=(prediction.reshape(size,1,1)*classes).expand(size,4,5).softmax(-1)
        delta=prediction.reshape(size)
        homography=torch.eye(3,device=images.device).repeat(size,1,1)
        homography[:,0,2]=delta*.2
        homography[:,1,2]=delta*.2
        success=torch.full((size,),self.fit_succeeded,dtype=torch.bool,device=images.device)
        return dict(context=context,H0_norm=homography,stage1_valid=success,
            stage1=dict(coarse_warp=warp,coarse_matchability=matchability,
                match_probabilities=probabilities,H_A_to_B_norm=homography,fit_succeeded=success))


class TinyMatcher(nn.Module):
    config=SimpleNamespace(input_size=512)

    def training_losses(self,shared,H_gt,mask_A_overlap,mask_B_overlap):
        # Represents a descriptor-only objective; H0 does not enter this path.
        losses=shared['context'].square().mean(-1)*.01
        return losses.mean(),[dict(lc=float(value.detach()),lf=0.,lq=0.) for value in losses]


class H0SupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def batch(self,count=1):
        return (torch.ones(count,2,3,512,512),torch.eye(3).repeat(count,1,1),
                torch.ones(count,1,512,512),torch.ones(count,1,512,512))

    def system(self,weight=1.,fit=True):
        shared=TinyShared(fit)
        system=SemidenseSystem(shared,TinyMatcher(),h0_loss_weight=weight)
        shared.set_training_groups(mvt=True,vgg=True,dedode=True,ghim_head=True)
        return system

    def test_h0_and_mvt_receive_finite_nonzero_gradients_dino_stays_frozen(self):
        system=self.system()
        loss,records=system(*self.batch())
        loss.backward()
        for module in (system.shared.mvt,system.shared.stage1_head):
            gradient=module.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()),0.)
        self.assertFalse(system.shared.dino.weight.requires_grad)
        self.assertIsNone(system.shared.dino.weight.grad)
        self.assertGreater(records[0]['lh0'],0.)
        self.assertEqual(records[0]['h0_supervised_points'],4)
        self.assertTrue(records[0]['h0_fit_valid'])
        self.assertEqual(records[0]['h0_valid_H_pairs'],1)

    def test_zero_weight_exactly_preserves_legacy_loss_and_records(self):
        system=self.system(weight=0.)
        batch=self.batch()
        expected=system.matcher.training_losses(system.shared(batch[0]),*batch[1:])
        actual=system(*batch)
        torch.testing.assert_close(actual[0],expected[0],atol=0.,rtol=0.)
        self.assertEqual(actual[1],expected[1])
        actual[0].backward()
        self.assertIsNone(system.shared.stage1_head.weight.grad)

    def test_failed_h_fit_still_trains_coarse_ghim_outputs(self):
        system=self.system(fit=False)
        loss,records=system(*self.batch())
        loss.backward()
        self.assertEqual(records[0]['h0_h'],0.)
        self.assertFalse(records[0]['h0_fit_valid'])
        self.assertEqual(records[0]['h0_valid_H_pairs'],0)
        self.assertGreater(records[0]['h0_geo'],0.)
        self.assertGreater(float(system.shared.stage1_head.weight.grad.abs().sum()),0.)

    def test_multibatch_records_and_pair_weighted_total(self):
        system=self.system(weight=.7)
        batch=list(self.batch(count=2));batch[0][1]*=.4
        loss,records=system(*batch)
        matcher_loss,_=system.matcher.training_losses(system.shared(batch[0]),*batch[1:])
        self.assertEqual(len(records),2)
        self.assertNotEqual(records[0]['lh0'],records[1]['lh0'])
        expected=matcher_loss+.7*sum(record['lh0'] for record in records)/2
        torch.testing.assert_close(loss,expected)
        self.assertTrue(all(record['h0_supervised_points']==4 for record in records))

    def test_validation_and_inference_do_not_change_prior_ownership(self):
        system=self.system()
        shared=system(self.batch()[0])
        self.assertIn('stage1',shared)
        self.assertTrue(shared['H0_norm'].requires_grad)
        self.assertFalse(hasattr(system.matcher,'h0_loss_weight'))
        self.assertFalse(hasattr(system.matcher,'stage1_head'))

    def test_invalid_weights_are_rejected(self):
        for value in (-.1,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                SemidenseSystem(TinyShared(),TinyMatcher(),h0_loss_weight=value)
        for weights in ({'H':-.1},{'mat':float('inf')},{'cls':float('nan')},
                        {'robust_scale':0.},{'unknown':1.}):
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                SemidenseSystem(TinyShared(),TinyMatcher(),h0_loss_weights=weights)

    def test_training_configuration_enables_loss_for_trainable_h0(self):
        system=self.system(weight=0.)
        policy=configure_h0_training(system,{'freeze_h0':False})
        self.assertEqual(system.h0_loss_weight,1.)
        self.assertEqual(policy,dict(weight=1.,mat=.01,cls=1e-4,H=.05,robust_scale=.1))
        with self.assertRaisesRegex(ValueError,'requires h0_loss_weight > 0'):
            configure_h0_training(system,{'freeze_h0':False,'h0_loss_weight':0.})

    def test_training_configuration_preserves_frozen_default_and_custom_weights(self):
        system=self.system()
        configure_h0_training(system,{})
        self.assertEqual(system.h0_loss_weight,0.)
        policy=configure_h0_training(system,{'freeze_h0':False,'h0_loss_weight':.25,
                                           'h0_loss_weights':{'H':.02}})
        self.assertEqual(policy['weight'],.25)
        self.assertEqual(policy['H'],.02)
        self.assertEqual(policy['mat'],.01)


if __name__=='__main__':unittest.main()
