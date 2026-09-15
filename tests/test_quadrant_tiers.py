import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from mhinet.dataio.quadrant_tiers import generate


class QuadrantTiersTests(unittest.TestCase):
    def test_geometry_and_visibility_all_tiers(self):
        image=np.random.default_rng(0).integers(0,256,(192,192,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('images','masks','metadata'): (root/name).mkdir()
            for tier in (1,2,3):
                for i,ratio in enumerate((.8,.6,.4)):
                    row=generate(image,image,root,f'{tier}_{i}',i,42+tier*10+i,None,
                        {'parent_image_A':'same','parent_image_B':'same','tier':tier,'resolution_ratio':ratio})
                    self.assertGreaterEqual(min(row['geometric_overlap_fractions']),.5)
                    self.assertGreaterEqual(min(row['visible_overlap_fractions']),.3 if tier==3 else .5)
                    H=np.array(row['H_A_to_B']);Ta=np.array(row['T_source_to_A']);Tb=np.array(row['T_source_to_B'])
                    expected=Tb@np.linalg.inv(Ta);expected/=expected[2,2]
                    np.testing.assert_allclose(H,expected,atol=1e-9)
                    for a,b,t in [('A','B',np.array(row['H_B_to_A'])),('B','A',H)]:
                        own=cv2.imread(str(root/row[f'mask_{a}_visible']),0)
                        other=cv2.imread(str(root/row[f'mask_{b}_visible']),0)
                        mask=cv2.imread(str(root/row[f'mask_{a}_overlap']),0)
                        projected=cv2.warpPerspective(other,t,(own.shape[1],own.shape[0]),flags=cv2.INTER_NEAREST)
                        np.testing.assert_array_equal(mask,((own>0)&(projected>0)).astype(np.uint8)*255)
                    self.assertEqual(bool(row['photometric'][0]),tier>=2)
                    self.assertEqual(any(row['occlusion']),tier==3)
                    for occlusion in row['occlusion']:
                        if occlusion:self.assertTrue(.1<=occlusion['area_fraction']<=.3)
