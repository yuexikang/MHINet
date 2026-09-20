import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from scripts.verify_stable_dataset import check_files


class DatasetAuditTests(unittest.TestCase):
    def test_headers_metadata_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            r={'size_A':[8,6],'size_B':[4,3],'metadata':'row.json'}
            for key in ('image_A','image_B','mask_A_overlap','mask_B_overlap','mask_A_geometry','mask_B_geometry','mask_A_visible','mask_B_visible'):
                w,h=r['size_'+('A' if '_A' in key else 'B')]
                r[key]=key+'.png'
                self.assertTrue(cv2.imwrite(str(root/r[key]),np.zeros((h,w),np.uint8)))
            (root/'row.json').write_text(json.dumps(r))
            self.assertEqual(check_files((root,r)),8)
            cv2.imwrite(str(root/r['image_A']),np.zeros((7,8),np.uint8))
            with self.assertRaises(AssertionError):check_files((root,r))
            (root/r['image_A']).unlink()
            with self.assertRaises(FileNotFoundError):check_files((root,r))
