from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mhinet.config import load_architecture_config
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
            "TINY-6": [8, 4, 2],
        }
        payload = {
            "gate": "P4_TINY_S_TINY_6",
            "status": "passed",
            "data_and_resource_evidence": {
                "architecture_sha256": load_architecture_config().sha256,
            },
            "protocol": {
                "training_revision": "1.2",
                "loss": "uniform proposal-corner coordinate L1",
                "FGO": False,
                "extra_losses": False,
                "planar_head": False,
                "precision": "BF16 adapter/CNN autocast, FP32 correlation/geometry",
                "tiny_weight_average_start_step": 1536,
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
                    "updates": 2 * len(active),
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
                for name, active in scales.items()
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            def metric_errors(item: dict, name: str, *, source: str) -> list[str]:
                mean = item["final"]["H_final_mace_px"]["mean"]
                return [] if mean < 0.1 else [f"{source}: {name} metric failed"]

            with patch(
                "mhinet.train.registered_experiment_errors",
                side_effect=metric_errors,
            ):
                self.assertIsNotNone(_validate_tiny_gate(path, True))
                payload["data_and_resource_evidence"]["architecture_sha256"] = (
                    "f" * 64
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    _validate_tiny_gate(path, True)
                payload["data_and_resource_evidence"]["architecture_sha256"] = (
                    load_architecture_config().sha256
                )
                payload["experiments"][0]["final"]["H_final_mace_px"][
                    "mean"
                ] = 0.2
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

    def test_formal_e01_requires_gate_and_full_six_round_schedule(self) -> None:
        config = TrainConfig.from_json("configs/e01_heads_v1.2.json")
        self.assertTrue(config.raw["require_passed_tiny_gate"])
        self.assertEqual(config.active_scales, (8, 4, 2))
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
