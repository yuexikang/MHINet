import unittest
import torch
import torch.nn.functional as F
from mhinet.downstream.supervision import (to_norm,to_uv,coarse_labels,fine_grids,fine_labels,
    offsets,interpolate_controls,qrru_targets)
from mhinet.downstream.qrru import QRRU
from mhinet.downstream.losses import coarse_positive_logp,dual_log_probability,qrru_loss


class SemidenseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)

    def test_legacy_config_and_inference_chunk_leave_training_unchanged(self):
        from dataclasses import replace,asdict
        from mhinet.downstream.semidense import SemidenseMatcher,SemidenseConfig
        legacy=dict(coarse_queries=8,fine_windows=3,qrru_queries=4,window_chunk=2,iterations=2)
        config=SemidenseConfig(**legacy)
        self.assertEqual(config.inference_window_chunk,1024)
        with self.assertRaises(ValueError):replace(config,inference_window_chunk=0)
        matcher=SemidenseMatcher(config,channels=8)
        d8=torch.randn(1,2,8,8,8,requires_grad=True)
        d2=torch.randn(1,2,8,32,32,requires_grad=True)
        h=torch.eye(3)[None];mask=torch.ones(1,1,32,32)
        shared={'pyramid':{8:d8,2:d2},'H0_norm':h,'stage1_valid':torch.tensor([True])}
        params=(d8,d2,*matcher.parameters());reference=None
        for chunk in (32,1024):
            matcher.config=replace(config,inference_window_chunk=chunk)
            torch.manual_seed(17)
            loss,records=matcher.training_losses(shared,h,mask,mask)
            grads=torch.autograd.grad(loss,params)
            if reference is None:reference=(loss.detach(),records,grads)
            else:
                torch.testing.assert_close(loss,reference[0],rtol=0,atol=0)
                self.assertEqual(records,reference[1])
                for actual,expected in zip(grads,reference[2]):
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        # Execution settings do not alter checkpoint tensor names or shapes.
        restored=SemidenseMatcher(SemidenseConfig(**asdict(config)),channels=8)
        restored.load_state_dict(matcher.state_dict(),strict=True)

    def test_coordinate_roundtrip(self):
        uv=torch.tensor([[0.,0.],[12.3,8.7]])
        for hw in ((392,392),(80,140)):
            torch.testing.assert_close(to_uv(to_norm(uv,hw),hw),uv,atol=1e-5,rtol=1e-5)
        native=to_uv(to_norm(uv,(392,392)),(784,784))
        torch.testing.assert_close(native,2*uv+.5,atol=2e-5,rtol=1e-5)

    def test_next_tier_accepts_legacy_inference_config_but_checks_training(self):
        from dataclasses import asdict,replace
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from mhinet.downstream.semidense import SemidenseConfig
        from scripts.train_semidense_next_tier import load_completed_semidense
        config=SemidenseConfig();legacy=asdict(config);legacy.pop('inference_window_chunk');legacy.pop('input_size')
        payload=dict(format='mhinet.training',progress={'optimizer_step':1},model={},
            metadata=dict(task='semidense_qrru_v1',total_steps=1,matcher=legacy,
                dino_sha256='hash',source_checkpoint='/unused/shared.pt',source_sha256='hash'))
        system=Mock();runtime=SimpleNamespace(dino_checkpoint='/unused/dino.pt')
        with patch('scripts.train_semidense_next_tier.torch.load',return_value=payload), \
             patch('scripts.train_semidense_next_tier.sha256_file',return_value='hash'), \
             patch('scripts.train_semidense_next_tier.initialize_system',return_value=system):
            self.assertIs(load_completed_semidense(runtime,'unused.pt',config),system)
            system.load_state_dict.assert_called_once_with({},strict=True)
            self.assertIs(load_completed_semidense(runtime,'unused.pt',replace(config,inference_window_chunk=128)),system)
            self.assertIs(load_completed_semidense(runtime,'unused.pt',replace(config,input_size=512)),system)
            with self.assertRaisesRegex(ValueError,'Matcher configuration differs'):
                load_completed_semidense(runtime,'unused.pt',replace(config,iterations=2))

    def test_affine_control_field(self):
        control=offsets('cpu',True)
        matrix=torch.tensor([[.2,.3],[-.4,.1]])
        shift=torch.tensor([2.,-1.])
        flow=(control@matrix.T+shift).permute(2,0,1)[None]
        expected=offsets('cpu')@matrix.T+shift
        torch.testing.assert_close(interpolate_controls(flow,offsets('cpu'))[0],expected)
        torch.testing.assert_close(flow.mean((-2,-1))[0],shift)

    def test_gt_and_h_residual(self):
        H=torch.eye(3); mask=torch.ones(1,32,32)
        i,j=coarse_labels(H,mask,mask,(8,8))
        self.assertEqual(len(i),64)
        torch.testing.assert_close(i,j)
        a,b=fine_grids(i[10:13],j[10:13],H,(8,8),(32,32))
        torch.testing.assert_close(a,b)
        m,s,t,_,_=fine_labels(a,b,H,mask,mask,(32,32))
        self.assertEqual(len(m),48)
        torch.testing.assert_close(s,t)
        pa=torch.tensor([[12.,12.]])
        pb=pa+torch.tensor([.5,-.25])
        target,valid,center,cv=qrru_targets(pa,pb,H,mask,mask,(32,32))
        torch.testing.assert_close(target,torch.tensor([-.5,.25])[None,:,None,None].expand_as(target))
        self.assertTrue(valid.all() and cv.all())
        torch.testing.assert_close(center,pa)

    def test_rematerialized_probabilities_and_gradients(self):
        a=F.normalize(torch.randn(11,7),dim=-1).requires_grad_()
        b=F.normalize(torch.randn(13,7),dim=-1).requires_grad_()
        tau=torch.tensor(.1,requires_grad=True)
        i,j=torch.tensor([0,2,3,9]),torch.tensor([1,4,8,10])
        sparse=coarse_positive_logp(a,b,tau,i,j,chunk=2)
        full=dual_log_probability(a@b.T/tau)[i,j]
        torch.testing.assert_close(sparse,full)
        grad1=torch.autograd.grad(sparse.sum(),(a,b,tau),retain_graph=True)
        grad2=torch.autograd.grad(full.sum(),(a,b,tau))
        for x,y in zip(grad1,grad2):torch.testing.assert_close(x,y,atol=2e-4,rtol=1e-4)

    def test_qrru_identity_and_second_step_gradient(self):
        net=QRRU(channels=8,iterations=2)
        a=torch.randn(8,20,20,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
        pa=torch.tensor([[8.,8.],[12.,12.]])
        pb=pa+torch.tensor([.5,-.25])
        target,valid,center,cv=qrru_targets(pa,pb,torch.eye(3),torch.ones(1,20,20),torch.ones(1,20,20),(20,20))
        optimizer=torch.optim.Adam(net.parameters(),lr=.01)
        for step in range(2):
            out=net(a,b,pa,pb)
            if step==0:torch.testing.assert_close(out['centers'][-1],pb)
            loss,_=qrru_loss(out,target,valid,center,cv)
            optimizer.zero_grad();a.grad=None;b.grad=None
            loss.backward();optimizer.step()
        self.assertTrue(a.grad.isfinite().all() and a.grad.abs().sum()>0)
        self.assertTrue(b.grad.isfinite().all() and b.grad.abs().sum()>0)
        self.assertGreater(net.project.weight.grad.abs().sum(),0)

    def test_decoupled_training_and_invalid_h0(self):
        from mhinet.downstream.semidense import SemidenseMatcher,SemidenseConfig
        matcher=SemidenseMatcher(SemidenseConfig(coarse_queries=8,fine_windows=3,
            qrru_queries=4,window_chunk=2,iterations=2),channels=8)
        d8=torch.randn(2,2,8,8,8,requires_grad=True)
        d2=torch.randn(2,2,8,32,32,requires_grad=True)
        h=torch.eye(3).repeat(2,1,1);h[1]=0;h.requires_grad_()
        shared={'pyramid':{8:d8,2:d2},'H0_norm':h,'stage1_valid':torch.ones(2,dtype=torch.bool)}
        loss,records=matcher.training_losses(shared,torch.eye(3).repeat(2,1,1),
            torch.ones(2,1,32,32),torch.ones(2,1,32,32))
        loss.backward()
        # H0 selects local grids as a detached prior; enabling requires_grad on
        # the GHIM head alone does not supply an H0 supervision signal.
        self.assertIsNone(h.grad)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(d8.grad.abs().sum(),0)
        self.assertGreater(d2.grad.abs().sum(),0)
        self.assertGreater(records[0]['fine_positive'],0)
        self.assertEqual(records[1]['fine_positive'],0)
        self.assertGreater(records[1]['qrru_queries'],0)

    def test_checkpoint_optimizer_boundary_replay(self):
        import tempfile
        from pathlib import Path
        from mhinet.engine.checkpointing import save_checkpoint,load_checkpoint
        net=QRRU(channels=8,iterations=2)
        optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        a=torch.randn(8,20,20);b=torch.randn_like(a)
        pa=torch.tensor([[8.,8.]])
        def step():
            optimizer.zero_grad()
            target=pa+torch.randn_like(pa)*.2
            out=net(a,b,pa,pa)['centers'][-1]
            (out-target).square().mean().backward();optimizer.step()
        step()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'resume.pt'
            save_checkpoint(path,model=net,optimizer=optimizer,optimizer_step=1,
                data_progress={'cursor':1},metadata={'protocol':'qrru-test'})
            step();expected={k:v.clone() for k,v in net.state_dict().items()}
            load_checkpoint(path,model=net,optimizer=optimizer,map_location='cpu',expected_metadata={'protocol':'qrru-test'})
            step()
            for k,v in net.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)

    def test_empty_supervision_and_masked_softmax(self):
        from mhinet.downstream.semidense import SemidenseMatcher,SemidenseConfig
        net=SemidenseMatcher(SemidenseConfig(coarse_queries=4,fine_windows=2,qrru_queries=2),channels=8)
        d8=torch.randn(1,2,8,4,4,requires_grad=True);d2=torch.randn(1,2,8,16,16,requires_grad=True)
        h=torch.eye(3)[None]
        loss,info=net.training_losses({'pyramid':{8:d8,2:d2},'H0_norm':h,'stage1_valid':torch.tensor([True])},
            h,torch.zeros(1,1,16,16),torch.zeros(1,1,16,16))
        loss.backward();self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(torch.isfinite(d8.grad).all() and torch.isfinite(d2.grad).all())
        logits=torch.randn(2,4,4,requires_grad=True)
        lp=dual_log_probability(logits,torch.zeros_like(logits,dtype=torch.bool))
        lp.sum().backward();self.assertTrue(logits.grad.isfinite().all())

    def test_qrru_small_pair_overfit(self):
        net=QRRU(channels=8,iterations=2)
        optimizer=torch.optim.Adam(net.parameters(),lr=.003)
        a=torch.randn(8,20,20);b=torch.randn_like(a)
        pa=torch.tensor([[8.,8.],[12.,12.]])
        pb=pa+torch.tensor([.5,-.25])
        target,valid,center,cv=qrru_targets(pa,pb,torch.eye(3),torch.ones(1,20,20),torch.ones(1,20,20),(20,20))
        initial=None
        for _ in range(35):
            result=net(a,b,pa,pb)
            loss,_=qrru_loss(result,target,valid,center,cv)
            if initial is None:initial=float(loss.detach())
            optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertLess(float(loss.detach()),initial*.3)


if __name__=='__main__':unittest.main()
