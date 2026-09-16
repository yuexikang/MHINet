import copy
import unittest
import torch
from mhinet.engine.pair_afss.adapter import RoundStream,score_pair


def refresh(stream):
    c=stream.controller;n=stream.length;s=torch.linspace(0,1,n)
    c.state.update_scores(round_id=c.current_round,score=s,precision_geo=s,recall_geo=s,
        homography_auc=s,fit_succeeded=torch.ones(n,dtype=torch.bool),scored=torch.ones(n,dtype=torch.bool),config=c.config)


class AFSSTests(unittest.TestCase):
    def test_resume_and_warmup_identity(self):
        ids=[str(i) for i in range(20)]
        a=RoundStream(ids,0,4,5,'uniform_rounds','hash')
        b=RoundStream(ids,0,4,5,'afss_v2','hash')
        for _ in range(10):
            a.prepare(refresh);b.prepare(refresh)
            self.assertEqual([a.next() for i in range(4)],[b.next() for i in range(4)])
        b.prepare(refresh)
        c=RoundStream(ids,0,4,5,'afss_v2','hash',copy.deepcopy(b.state_dict()))
        self.assertGreaterEqual(len(b.controller.selected_unique_indices),9)
        while not b.done:
            b.prepare(refresh);c.prepare(refresh)
            self.assertEqual([b.next() for i in range(4)],[c.next() for i in range(4)])
        self.assertTrue(c.done)
        with self.assertRaises(ValueError):RoundStream(ids,0,4,5,'afss_v2','changed',b.state_dict())
        with self.assertRaises(ValueError):RoundStream(list(reversed(ids)),0,4,5,'afss_v2','hash',b.state_dict())

    def test_h6_not_ignored(self):
        h=4;y=(torch.arange(h)+.5)*2/h-1
        yy,xx=torch.meshgrid(y,y,indexing='ij');warp=torch.stack([xx,yy])[None]
        H=torch.eye(3)[None]
        out={'ghim_outputs':{'coarse_warp':warp,'coarse_matchability':torch.ones(1,1,h,h)},
             'H0_norm':H,'H_updates_norm':H[:,None].repeat(1,6,1,1),'stage1_valid':torch.tensor([True])}
        mask=torch.ones(1,1,16,16)
        self.assertAlmostEqual(score_pair(out,H,mask)['score'].item(),1.)
        out['H_updates_norm'][:,-1,0,2]=.2
        self.assertEqual(score_pair(out,H,mask)['score'].item(),0.)
        self.assertEqual(score_pair(out,H,mask*0)['score'].item(),0.)
