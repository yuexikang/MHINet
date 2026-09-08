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
                "shared_once_per_pair": True,
            },
            "experiments": [
                {
                    "name": name,
                    "status": "passed",
                    "active_scales": list(scales),
                    "updates": 2 * len(scales),
                    "sample_count": 32,
                    "sample_protocol": "one_pair_residuals",
                    "residual_profile": "translation",
                    "precision": "bf16",
                    "primary_readout": "raw_parameters",
                    "weight_averaging": {
                        "enabled": True,
                        "start_optimizer_step": 1536,
                        "averaged_snapshots": 0,
                        "training_optimizer_unchanged": True,
                        "formal_training_enabled": False,
                    },
                    "optimizer": {
                        "name": "AdamW",
                        "learning_rate": 1e-3,
                        "weight_decay": 0.0,
                        "scheduler": None,
                        "batch_size": 1,
                        "gradient_accumulation": 4,
                        "effective_batch_size": 4,
                        "gradient_clip_norm": 1.0,
                        "optimizer_steps": 400,
                        "maximum_steps": 2000,
                    },
                    "controlled_H0": {
                        "profile": "translation",
                        "formal_training_injection": False,
                    },
                    "criterion": {"met": True, "threshold_mace_px": 0.1},
                    "final": {
                        "H_final_mace_px": {"mean": 0.05},
                        "failed_pairs": 0,
                        "rejected_updates": 0,
                    },
                    "gradient_finite": True,
                    "checkpoint": {
                        "sha256": "0" * 64,
                        "metadata": {
                            "formal_training_checkpoint": False,
                            "sample_count": 32,
                            "sample_protocol": "one_pair_residuals",
                            "residual_profile": "translation",
                            "precision": "bf16",
                        },
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

    def test_optimizer_drift_fails_closed(self) -> None:
        artifact = self._artifact("TINY-S-D8", (8,))
        artifact["experiments"][0]["optimizer"]["gradient_accumulation"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d8.json"
            path.write_text(json.dumps(artifact))
            report = merge_tiny_gate_artifacts([path])
        self.assertTrue(any("optimizer recipe" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
