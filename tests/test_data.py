from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from mhinet.data import HomographyPairDataset, parent_group, parse_geo_region


class DataTests(unittest.TestCase):
    def test_grouping_fields(self) -> None:
        record = {
            "source": "current/Train/34.801100126.376330.jpg",
            "parent_image_A": "same.jpg",
            "parent_image_B": "same.jpg",
        }
        self.assertEqual(parse_geo_region(record), "geo_0.01:3480:12637")
        self.assertEqual(parent_group(record), "parent:same.jpg")
        self.assertEqual(
            parent_group({"input_pair_row_index": 12}), "test_pair:12"
        )

    def test_dataset_bicubic_pair_and_identity_h(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((6, 8, 3), dtype=np.uint8)
            image[..., 0] = 127
            Image.fromarray(image).save(root / "a.png")
            Image.fromarray(image).save(root / "b.png")
            record = {
                "pair_id": "tiny",
                "source": "current/Train/34.801100126.376330.jpg",
                "parent_image_A": "mother.jpg",
                "parent_image_B": "mother.jpg",
                "image_A": "a.png",
                "image_B": "b.png",
                "size_A": [8, 6],
                "size_B": [8, 6],
                "H_A_to_B": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
            manifest = root / "pairs.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = HomographyPairDataset(manifest, image_size=16)
            sample = dataset[0]
            self.assertEqual(sample["images"].shape, (2, 3, 16, 16))
            torch.testing.assert_close(sample["H_gt_norm"], torch.eye(3), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
