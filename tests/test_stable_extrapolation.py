import unittest
import numpy as np
from scripts.audit_stable_extrapolation import measure


class StableExtrapolationTests(unittest.TestCase):
    def row(self,H):
        return dict(size_A=[784,784],size_B=[784,784],H_A_to_B=H,pair_id='test')

    def test_identity(self):
        for result in measure(self.row(np.eye(3).tolist())):
            self.assertEqual(result['max_abs_corner_px'],783)
            self.assertEqual(result['max_corner_magnification'],1)
            self.assertIsNone(result['horizon_min_px'])

    def test_crossing_rejected(self):
        with self.assertRaises(ValueError):
            measure(self.row([[1,0,0],[0,1,0],[-2/783,0,1]]))

    def test_inverse_included(self):
        result=measure(self.row(np.diag([.5,.5,1]).tolist()))
        self.assertEqual(result[0]['max_corner_magnification'],.5)
        self.assertEqual(result[1]['max_corner_magnification'],2)
