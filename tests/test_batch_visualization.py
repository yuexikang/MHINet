import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from mhinet.train import TrainConfig
from mhinet.feature_provider import SharedFeatureProvider
from mhinet.visualization import write_iteration_overlays
from types import SimpleNamespace
from tests.test_feature_provider import RecordingCumulativeDecoder


class BatchAndOverlayTests(unittest.TestCase):
    def test_config_preserves_effective_batch_and_rejects_silent_budget_change(self):
        base = json.loads(Path("configs/e01_heads_v1.2.json").read_text())
        for batch, accumulation in [(1, 4), (2, 2), (4, 1)]:
            config = {**base, "batch_size": batch, "gradient_accumulation": accumulation,
                      "allow_experimental_batch": True}
            TrainConfig(Path("unused"), config, "test").validate()
        with self.assertRaises(ValueError):
            TrainConfig(Path("unused"), {**base, "batch_size": 4}, "test").validate()

    def test_larger_batch_requires_explicit_experimental_opt_in(self):
        base = json.loads(Path("configs/e01_heads_v1.2.json").read_text())
        with self.assertRaisesRegex(ValueError, "allow_experimental_batch"):
            TrainConfig(Path("unused"), {**base, "batch_size": 2,
                        "gradient_accumulation": 2}, "test").validate()

    def test_overlay_rejects_missing_iteration(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            write_iteration_overlays(torch.zeros(2, 3, 32, 32), torch.eye(3),
                                     torch.eye(3).repeat(5, 1, 1), [8,8,4,4,2,2],
                                     temp, pair_id="incomplete")

    def test_overlay_includes_h0_as_seventh_image(self):
        with tempfile.TemporaryDirectory() as temp:
            result = write_iteration_overlays(torch.zeros(2,3,32,32), torch.eye(3),
                torch.eye(3).repeat(6,1,1), [8,8,4,4,2,2], temp,
                pair_id="with_h0", h0=torch.eye(3), accepted=[True]*6)
            self.assertEqual(len(list(Path(temp).glob("*.png"))), 7)
            self.assertEqual(result["images"][0]["iteration"], 0)
            self.assertIsNone(result["images"][0]["scale"])
            self.assertIsNone(result["images"][0]["update_accepted"])
            self.assertTrue((Path(temp)/"H0_initialization.png").is_file())

    def test_cumulative_decoder_preserves_two_pairs_and_stops_at_d2(self):
        decoder = RecordingCumulativeDecoder()
        owner = SimpleNamespace(dedode=decoder, _call_counts={
            "dedode_decode_calls": 0, "dedode_steps": 0, "dedode_scale1": 0})
        output = SharedFeatureProvider._decode_pyramid(
            owner, [torch.zeros(4, 8, size, size) for size in (32, 16, 8, 4)],
            [(size, size) for size in (32, 16, 8, 4)], torch.zeros(2, 2, 2, 2, 8), (8, 4, 2))
        self.assertEqual(output[2].shape, (2, 2, 256, 16, 16))
        self.assertEqual(decoder.calls, ["16", "8", "4", "2"])

    def test_overlay_six_files_correct_pixel_translation_and_invalid_fallback(self):
        images = torch.zeros(2, 3, 32, 32)
        images[0, 0] = 1
        images[1, 2] = 1
        prediction = torch.eye(3)
        prediction[0, 2] = 2 * 4 / 32
        updates = prediction.repeat(6, 1, 1)
        updates[-1] = 0  # Singular display input must never be inverted/warped.
        with tempfile.TemporaryDirectory() as temp:
            result = write_iteration_overlays(images, torch.eye(3), updates, [8, 8, 4, 4, 2, 2],
                                             temp, pair_id="synthetic", accepted=[True]*5+[False])
            self.assertEqual(len(list(Path(temp).glob("*.png"))), 6)
            self.assertEqual(result["images"][0]["scale"], 8)
            self.assertAlmostEqual(result["images"][0]["H_pred_pixel"][0][2], 4.)
            self.assertFalse(result["images"][-1]["warp_valid"])
            rgb = np.asarray(Image.open(result["images"][0]["path"]))
            self.assertTrue(np.any(np.all(rgb == (0, 255, 0), axis=-1)))
            self.assertTrue(np.any(np.all(rgb == (255, 0, 0), axis=-1)))


if __name__ == "__main__":
    unittest.main()
