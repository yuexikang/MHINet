"""The timing wrapper must restore methods even on a failed prediction."""
import unittest
from types import SimpleNamespace
import torch
from scripts.semidense_timing import timed_inference,distribution
import mhinet.downstream.semidense as matching


class Leaf(torch.nn.Module):
    def forward(self,x):return x


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__();self.stage1_head=Leaf();self.vgg=Leaf()
    def _extract_batched_descriptors(self,x):return x
    def _contextualize(self,x):return x
    def _decode_pyramid(self,x):return x
    def forward(self,x):
        x=self._extract_batched_descriptors(x);x=self._contextualize(x)
        return self._decode_pyramid(self.vgg(self.stage1_head(x)))


class TimingTest(unittest.TestCase):
    def system(self, fail=False):
        class Matcher:
            qrru=SimpleNamespace(project=Leaf())
            def infer(self,features,sizes):
                if fail:raise RuntimeError('deliberate failure')
                return [features]
        shared=Shared()
        class System:
            matcher=Matcher()
            def __call__(self,images):return self.shared(images)
        system=System();system.shared=shared;return system
    def test_accounting_and_no_mutation(self):
        s=self.system();images=torch.ones(1,2,1);original=matching.predicted_overlap_masks
        result,t=timed_inference(s,images,None)
        self.assertTrue(torch.equal(result[0],images))
        self.assertAlmostEqual(sum(t['module_ms'].values()),t['stream_total_ms'],places=8)
        self.assertEqual(t['module_ms']['coarse'],0)
        self.assertIs(matching.predicted_overlap_masks,original)
        self.assertNotIn('_extract_batched_descriptors',s.shared.__dict__)
        self.assertNotIn('forward',s.matcher.qrru.project.__dict__)
    def test_exception_restores_wrappers(self):
        s=self.system(True);original=matching.predicted_overlap_masks
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            timed_inference(s,torch.ones(1,2,1),None)
        self.assertIs(matching.predicted_overlap_masks,original)
        self.assertNotIn('forward',s.shared.stage1_head.__dict__)
        self.assertNotIn('forward',s.matcher.qrru.project.__dict__)
    def test_distribution(self):
        d=distribution([1,2,3]);self.assertEqual(d['mean'],2);self.assertEqual(d['p50'],2)
        self.assertIsNone(distribution([]))


if __name__=='__main__':unittest.main()
