import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from mhinet.downstream.dense import HGuidedDenseDownstream
from mhinet.downstream.metrics import dense_metrics


class DenseDownstreamTests(unittest.TestCase):
    def test_reference_hashes(self):
        root=Path(__file__).resolve().parents[1]
        registry=json.loads((root/'artifacts/loma_downstream_source.json').read_text())
        for name,expected in registry['files_sha256'].items():
            self.assertEqual(hashlib.sha256((root/'mhinet/downstream/loma_reference'/name).read_bytes()).hexdigest(),expected)

    def test_invalid_h_isolated_before_features(self):
        matcher=HGuidedDenseDownstream()
        for h,reason in ((torch.zeros(1,3,3),'singular_h0'),(torch.full((1,3,3),float('nan')),'nonfinite_h0')):
            r=matcher({'H0_norm':h,'stage1_valid':torch.tensor([True])})
            self.assertEqual(r['failure_reason'],reason)
            self.assertEqual(len(r['confidence']),0)
        with self.assertRaises(ValueError):HGuidedDenseDownstream('C3-P0')

    def test_metrics_include_outside_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.fromarray(np.full((8,8),255,np.uint8)).save(root/'mask.png')
            row={'H_A_to_B':np.eye(3).tolist(),'mask_A_overlap':'mask.png','mask_B_overlap':'mask.png'}
            a=np.array([[1,1],[2,2],[20,20]],float);b=a.copy();b[1]+=[4,0]
            r=dense_metrics(row,root,a,b)
            self.assertEqual(r['thresholds']['1.0']['NCM'],1)
            self.assertEqual(r['thresholds']['1.0']['precision'],1/3)
            self.assertEqual(r['thresholds']['5.0']['NCM'],2)
            self.assertFalse(r['success_20_correct_at_5px'])
            self.assertEqual(dense_metrics(row,root,np.empty((0,2)),np.empty((0,2)))['matches'],0)
