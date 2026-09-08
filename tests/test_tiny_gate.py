from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mhinet.tiny_gate import REQUIRED_EXPERIMENTS, merge_tiny_gate_artifacts


class TinyGateMergeTests(unittest.TestCase):
    def _artifact(self, name: str, scales: tuple[int, ...]) -> dict:
        return {
            "gate": "P4_TINY_S_TINY_8",
            "status": "passed",
            "protocol": {
                "training_revision": "1.2",
                "loss": "uniform proposal-corner coordinate L1",
                "FGO": False,
                "extra_losses": False,
                "planar_head": False,
                "precision": "BF16 adapter/CNN autocast, FP32 correlation/geometry",
                "tiny_weight_average_start_step": 1536,
            },
            "cache": {
                "sample_protocol": "one_pair_residuals",
                "diagnostic_sample_count": 32,
                "test_used": False,
            },
            "experiments": [
                {
                    "name": name,
                    "status": "passed",
                    "active_scales": list(scales),
                    "sample_count": 32,
                    "criterion": {"met": True, "threshold_mace_px": 0.1},
                    "final": {
                        "H_final_mace_px": {"mean": 0.05},
                        "failed_pairs": 0,
                        "rejected_updates": 0,
                    },
                }
            ],
        }

    def test_complete_consistent_merge_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, (name, scales) in enumerate(REQUIRED_EXPERIMENTS.items()):
                path = Path(directory) / f"{index}.json"
                path.write_text(json.dumps(self._artifact(name, scales)))
                paths.append(path)
            report = merge_tiny_gate_artifacts(paths)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["errors"], [])

    def test_incomplete_merge_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d8.json"
            path.write_text(json.dumps(self._artifact("TINY-S-D8", (8,))))
            report = merge_tiny_gate_artifacts([path])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("missing experiments" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
