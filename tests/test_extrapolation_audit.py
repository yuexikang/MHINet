import unittest
import numpy as np
from scripts.analyze_homography_extrapolation import geometry


class ExtrapolationAuditTests(unittest.TestCase):
    def row(self,H):
        return dict(pair_id='synthetic',tier=1,size_A=[784,784],size_B=[784,784],
            H_A_to_B=H.tolist(),T_source_to_A=np.eye(3).tolist(),T_source_to_B=H.tolist(),
            visible_overlap_fractions=[.5,.5])

    def test_affine_has_no_finite_horizon(self):
        r=geometry(self.row(np.eye(3)))
        self.assertIsNone(r['horizon_distance_min_px'])
        self.assertEqual(r['max_corner_local_magnification'],1.)
        self.assertEqual(r['composition_max_abs_error'],0.)

    def test_near_horizon_is_not_a_sign_change(self):
        H=np.eye(3);H[2,0]=-1/784
        r=geometry(self.row(H))
        self.assertFalse(r['denominator_sign_change'])
        self.assertAlmostEqual(r['horizon_distance_min_px'],1.)
        self.assertGreater(r['gt_corner_max_abs_px'],600000)
        self.assertGreater(r['max_corner_local_magnification'],100000)
