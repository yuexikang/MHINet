import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from mhinet.downstream.visualize import query_maps,feature_gallery


class VisualTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(0)

    def test_query_probabilities_have_full_column_denominator(self):
        features=torch.randn(2,8,98,98)
        ma=torch.zeros(98,98,dtype=torch.bool);mb=ma.clone()
        ids=torch.tensor([12*98+12,36*98+36,60*98+60])
        targets=torch.tensor([1,12,50,100])
        ma.flatten()[ids]=True;mb.flatten()[targets]=True
        q,cos,p,best,mutual=query_maps(features,dict(mask_a=ma,mask_b=mb),.2,chunk=2)
        a=F.normalize(features[0].flatten(1).T,dim=-1);b=F.normalize(features[1].flatten(1).T,dim=-1)
        x=a[ids]@b[targets].T/.2
        expected=x.softmax(0)*x.softmax(1)
        for k,i in enumerate(ids):
            row=(q==i).nonzero().item();torch.testing.assert_close(p[row,targets],expected[k])
        self.assertEqual(float(p[~ma.flatten()[q]].sum()),0.)

    def test_pca_basis_and_scale_are_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cal=root/'cal'
            feature=torch.randn(2,8,12,12)
            feature_gallery(feature,root,cal,'D8')
            old=(cal/'D8.npz').read_bytes()
            feature_gallery(feature*2,root,cal,'D8')
            self.assertEqual(old,(cal/'D8.npz').read_bytes())
            self.assertTrue((root/'features_D8_summary.png').exists())


if __name__=='__main__':unittest.main()
