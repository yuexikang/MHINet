from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mhinet.train import (
    DeterministicIndexStream,
    TrainConfig,
    _validate_tiny_gate,
    warmup_cosine_factor,
)


class TrainingProtocolTests(unittest.TestCase):
    def test_formal_gate_validator_checks_metrics_not_only_top_level_status(self) -> None:
        scales = {
            "TINY-S-D8": [8],
            "TINY-S-D4": [4],
            "TINY-S-D2": [2],
            "TINY-S-D1": [1],
            "TINY-8": [8, 4, 2, 1],
        }
        payload = {
            "gate": "P4_TINY_S_TINY_8",
            "status": "passed",
            "protocol": {
                "training_revision": "1.2",
                "loss": "uniform proposal-corner coordinate L1",
                "FGO": False,
                "extra_losses": False,
                "planar_head": False,
                "sample_protocol": "one_pair_residuals",
                "diagnostic_sample_count": 32,
                "test_used": False,
            },
            "errors": [],
            "experiments": [
                {
                    "name": name,
                    "status": "passed",
                    "criterion": {"met": True, "threshold_mace_px": 0.1},
                    "sample_count": 32,
                    "active_scales": active,
                    "final": {
                        "H_final_mace_px": {"mean": 0.05},
                        "failed_pairs": 0,
                        "rejected_updates": 0,
                    },
                }
                for name, active in scales.items()
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNotNone(_validate_tiny_gate(path, True))
            payload["experiments"][0]["final"]["H_final_mace_px"]["mean"] = 0.2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _validate_tiny_gate(path, True)

    def test_minimal_config_is_protocol_v12_and_explicitly_not_formal(self) -> None:
        config = TrainConfig.from_json("configs/train_minimal_smoke.json")
        self.assertEqual(config.profile, "heads")
        self.assertEqual(config.active_scales, (8,))
        self.assertFalse(config.raw["require_passed_tiny_gate"])
        self.assertEqual(config.raw["gradient_accumulation"], 4)

    def test_mainline_optimizer_protocol_drift_is_rejected(self) -> None:
        config = TrainConfig.from_json("configs/train_minimal_smoke.json")
        payload = dict(config.raw)
        payload["gradient_accumulation"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                TrainConfig.from_json(path)

    def test_formal_e01_requires_gate_and_full_eight_round_schedule(self) -> None:
        config = TrainConfig.from_json("configs/e01_heads_v1.2.json")
        self.assertTrue(config.raw["require_passed_tiny_gate"])
        self.assertEqual(config.active_scales, (8, 4, 2, 1))
        self.assertEqual(config.raw["iterations_per_scale"], 2)

    def test_deterministic_index_stream_resumes_at_exact_boundary(self) -> None:
        uninterrupted = DeterministicIndexStream(7, 13)
        prefix = [uninterrupted.next() for _ in range(11)]
        state = uninterrupted.state_dict()
        suffix = [uninterrupted.next() for _ in range(20)]
        resumed = DeterministicIndexStream(
            state["length"],
            state["seed"],
            epoch=state["epoch"],
            position=state["position"],
        )
        self.assertEqual(suffix, [resumed.next() for _ in range(20)])
        self.assertEqual(len(prefix), 11)

    def test_warmup_cosine_reaches_declared_endpoints(self) -> None:
        self.assertAlmostEqual(
            warmup_cosine_factor(
                0, total_steps=100, warmup_steps=5, minimum_ratio=0.1
            ),
            0.2,
        )
        self.assertAlmostEqual(
            warmup_cosine_factor(
                5, total_steps=100, warmup_steps=5, minimum_ratio=0.1
            ),
            1.0,
        )
        self.assertAlmostEqual(
            warmup_cosine_factor(
                100, total_steps=100, warmup_steps=5, minimum_ratio=0.1
            ),
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
