"""Resolution contracts and geometry, independent of pretrained weight availability."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from mhinet.downstream.semidense import SemidenseConfig,SemidenseMatcher
from mhinet.downstream.supervision import fine_grids,to_uv,to_norm
from mhinet.downstream.loma_reference.matching_utils import centers_to_native
from mhinet.downstream.visualize import query_maps
from mhinet.downstream.training import SemidenseSystem

class InputSizeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_old_metadata_default_and_rejection(self):
        self.assertEqual(SemidenseConfig().input_size,784)
        self.assertEqual(SemidenseConfig(input_size=512).input_size,512)
        with self.assertRaises(ValueError):SemidenseConfig(input_size=513)
    def test_fine_lattice_and_native_roundtrip_at_both_sizes(self):
        for size in (512,784):
            hc=(size//8,)*2;hf=(size//2,)*2
            ids=torch.tensor([0,hc[0]+1,hc[0]*hc[1]-1])
            a,b=fine_grids(ids,ids,torch.eye(3),hc,hf)
            torch.testing.assert_close(a,b)
            uv=to_uv(a,hf)
            torch.testing.assert_close(uv,uv.round(),atol=1e-4,rtol=0)
            self.assertTrue((uv>=-1e-4).all() and (uv<=hf[0]-1+1e-4).all())
            native=centers_to_native(a,(600,400))
            torch.testing.assert_close(native,(uv+.5)*torch.tensor([600/hf[1],400/hf[0]])-.5,atol=1e-4,rtol=1e-5)
    def test_system_rejects_mismatched_config(self):
        class Shared(torch.nn.Module):
            def set_training_groups(self,**kwargs):pass
            def forward(self,x):return x
        system=SemidenseSystem(Shared(),SemidenseMatcher(SemidenseConfig(input_size=512),channels=8))
        with self.assertRaisesRegex(ValueError,'input_size=512'):system(torch.empty(1,2,3,784,784))
        self.assertEqual(system(torch.empty(1,2,3,512,512)).shape[-1],512)
    def test_512_diagnostic_queries_in_bounds(self):
        d8=torch.randn(2,8,64,64);mask=torch.zeros(64,64,dtype=torch.bool)
        q,cos,p,best,mutual=query_maps(d8,dict(mask_a=mask,mask_b=mask),.1)
        self.assertEqual(cos.shape,(16,4096));self.assertLess(int(q.max()),4096)
    def test_infer_maps_native_coordinates_and_uses_dynamic_bounds(self):
        size=512;matcher=SemidenseMatcher(SemidenseConfig(input_size=size,max_matches=16))
        # Expanded constant channels avoid materializing a random 128MiB pair.
        d8=torch.ones(1,2,1,64,64).expand(-1,-1,256,-1,-1)
        d2=torch.ones(1,2,1,256,256).expand(-1,-1,256,-1,-1)
        shared=dict(pyramid={8:d8,2:d2},H0_norm=torch.eye(3)[None],stage1_valid=torch.tensor([True]))
        coarse=SimpleNamespace(source_flat=torch.tensor([65]),target_flat=torch.tensor([65]),confidence=torch.ones(1),diagnostics={})
        def scores(da,db,a,b,va,vb):return torch.eye(16)[None].expand(len(a),-1,-1)*10
        with patch('mhinet.downstream.semidense.match_d8',return_value=coarse),patch.object(matcher,'fine_scores',side_effect=scores):
            r=matcher.infer(shared,[((600,400),(600,400))])[0]
        self.assertEqual(r['failure_reason'],'none');self.assertEqual(len(r['points_a']),16)
        torch.testing.assert_close(r['points_a'],r['points_b'])
        self.assertEqual(r['qrru_outside_samples'],0)

if __name__=='__main__':unittest.main()
