import unittest
import numpy as np
from mhinet.dataio.stable_geometry import check_geometry


class StableGeometryTests(unittest.TestCase):
    def test_identity_and_scale_invariance(self):
        for factor in (1., -3., 1e-20, 1e20):
            self.assertTrue(check_geometry(np.eye(3)*factor, (784,784), (784,784))[0])

    def test_resolution_normalization(self):
        self.assertTrue(check_geometry(np.diag([.5,.5,1.]), (785,785), (393,393))[0])

    def test_near_horizon_and_crossing(self):
        for v in (-.999, -1., -2.):
            H=np.eye(3); H[2,0]=v
            self.assertFalse(check_geometry(H,(2,2),(2,2))[0])

    def test_inverse_magnification(self):
        self.assertFalse(check_geometry(np.diag([.01,.01,1.]),(2,2),(2,2))[0])

    def test_invalid(self):
        for H in (np.zeros((3,3)),np.diag([1,0,1]),np.full((3,3),np.nan)):
            self.assertFalse(check_geometry(H,(2,2),(2,2))[0])
