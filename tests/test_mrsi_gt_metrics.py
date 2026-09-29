import unittest
import numpy as np
from scripts.mrsi_gt_metrics import score_matches,aggregate,project

class GTMetricsTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.arange(10,80,10),np.arange(10,80,10));self.a=np.c_[x.ravel(),y.ravel()]
        self.H=np.array([[1.05,.03,4.],[-.02,.95,6.],[.0001,.0002,1.]])
        self.b,_=project(self.a,self.H)
    def test_true_projective_gt_not_identity_or_inverse(self):
        r=score_matches(self.a,self.b,(100,100),(100,100),self.H)
        self.assertEqual(r['precision']['1'],1.);self.assertEqual(r['matching_rmse'],0.)
        self.assertLess(r['registration_rmse'],1e-4);self.assertTrue(r['sr']['1'])
        wrong=score_matches(self.a,self.b,(100,100),(100,100),np.linalg.inv(self.H))
        self.assertGreater(wrong['matching_rmse'],5.)
    def test_offset_thresholds(self):
        r=score_matches(self.a,self.b+[2.,0],(100,100),(100,100),self.H)
        self.assertEqual(r['precision'],{'1':0.,'3':1.,'5':1.})
        self.assertAlmostEqual(r['registration_rmse'],2.,places=4)
        self.assertFalse(r['sr']['1']);self.assertTrue(r['sr']['3'])
    def test_rmse_includes_outlier(self):
        b=self.b.copy();b[0]+=[30,0]
        r=score_matches(self.a,b,(100,100),(100,100),self.H)
        self.assertAlmostEqual(r['matching_rmse'],30/np.sqrt(len(b)))
        self.assertEqual(r['ncm']['5'],len(b)-1)
    def test_empty_kept_in_success_denominator(self):
        good=score_matches(self.a,self.b,(100,100),(100,100),self.H)
        empty=score_matches([],[],(100,100),(100,100),self.H)
        rows=[dict(metrics=m,failure_reason=f,timing={'wall_total_ms':1.}) for m,f in [(good,'none'),(empty,'invalid_h0')]]
        r=aggregate(rows)
        self.assertEqual(r['precision']['1'],.5);self.assertEqual(r['sr']['3'],.5)
        self.assertEqual(r['matching_rmse_pairs'],1)
    def test_overlap_does_not_use_rescaled_identity(self):
        a=np.array([[1.,1.],[10.,10.],[20.,20.],[90.,90.]])
        r=score_matches(a,a,(100,100),(80,80),np.eye(3))
        self.assertEqual(r['precision']['1'],.75)

if __name__=='__main__':unittest.main()
