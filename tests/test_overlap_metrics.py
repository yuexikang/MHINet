import unittest
import torch
from mhinet.pretraining.overlap_metrics import overlap_projection_error, summarize_overlap


class OverlapMetricsTests(unittest.TestCase):
    def setUp(self):
        self.H=torch.eye(3)[None]
        self.mask=torch.ones(1,1,8,12)

    def test_identity(self):
        r=overlap_projection_error(self.H,self.H,self.mask,self.mask)
        self.assertEqual(r['support_pixels'],96)
        self.assertEqual(r['mean_px'],0)

    def test_prediction_does_not_select_support(self):
        H=self.H.clone();H[:,0,2]=5*2/12
        r=overlap_projection_error(H,self.H,self.mask,self.mask)
        self.assertEqual(r['support_pixels'],96)
        self.assertAlmostEqual(r['mean_px'],5,places=5)

    def test_gt_selects_support_and_both_masks(self):
        H=self.H.clone();H[:,0,2]=2/12
        a=self.mask.clone();a[:,:,:4]=0
        r=overlap_projection_error(self.H,H,a,self.mask)
        self.assertEqual(r['support_pixels'],44)
        self.assertAlmostEqual(r['mean_px'],1,places=5)
        r=overlap_projection_error(self.H,H,a,self.mask*0)
        self.assertEqual(r['support_pixels'],0)
        self.assertIsNone(r['mean_px'])

    def test_illegal_projection_and_failed_fit_stay_in_denominator(self):
        for pred,flag in [(self.H*0,True),(self.H,False)]:
            r=overlap_projection_error(pred,self.H,self.mask,self.mask,fit_valid=flag)
            s=summarize_overlap([r])
            self.assertEqual(s['invalid_projection_rate'],1)
            self.assertEqual(s['point_recall']['5'],0)
            self.assertIsNone(s['pixel_mean_px'])


if __name__=='__main__':
    unittest.main()
