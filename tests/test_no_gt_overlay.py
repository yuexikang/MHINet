import json
from pathlib import Path
import tempfile
import unittest
import torch
from mhinet.visualization.visualization import write_iteration_overlays


class NoGTOverlayTests(unittest.TestCase):
    def test_no_fake_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = write_iteration_overlays(torch.zeros(2,3,16,16), None,
                [], [], Path(tmp), pair_id='raw', h0=torch.eye(3))
            self.assertIsNone(result['images'][0]['H_gt_pixel'])
            self.assertTrue((Path(tmp)/'H0_initialization.png').is_file())
