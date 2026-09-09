from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from mhinet.checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from mhinet.config import load_architecture_config, sha256_file
from mhinet.tiny_gate import (
    REGISTERED_MAX_RESIDUAL_PX,
    REQUIRED_EXPERIMENTS,
    merge_tiny_gate_artifacts,
)


class TinyGateMergeTests(unittest.TestCase):
    @staticmethod
    def _metric_block(updates: int, *, h0: float, final: float) -> dict:
        rows = [
            {
                "diagnostic_sample_index": index,
                "pair_id": "fixed-train-pair",
                "trajectory_mace_px": [h0] + [final] * updates,
                "H0_mace_px": h0,
                "H_updates_mace_px": [final] * updates,
                "H_final_mace_px": final,
                "update_accepted": [True] * updates,
                "failure_reason_codes": [0] * updates,
            }
            for index in range(32)
        ]
        return {
            "H_final_mace_px": {
                "mean": final,
                "median": final,
                "p90": final,
                "max": final,
            },
            "failed_pairs": 0,
            "rejected_updates": 0,
            "per_pair": rows,
        }

    def _artifact(
        self,
        name: str,
        scales: tuple[int, ...],
        checkpoint_path: Path,
    ) -> dict:
        updates = 2 * len(scales)
        optimizer_steps = 400
        pair_ids = ["fixed-train-pair"] * 32
        architecture_sha256 = load_architecture_config().sha256
        metadata = {
            "diagnostic": name,
            "active_scales": scales,
            "known_H0_residual_injected": True,
            "formal_training_checkpoint": False,
            "sample_count": 32,
            "sample_protocol": "one_pair_residuals",
            "residual_profile": "translation",
            "precision": "bf16",
            "architecture_sha256": architecture_sha256,
            "mhir_revision": "mcnet_correlation_decoder_784_v1",
            "weight_average_start_step": 1536,
            "weight_average_count": 0,
            "checkpoint_contains_weight_average": False,
            "optimizer_resume_supported": True,
            "pair_ids": pair_ids,
            "tiny_progress_signature": {
                "resume_context": {
                    "architecture_sha256": architecture_sha256,
                }
            },
        }
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "version": CHECKPOINT_VERSION,
                "progress": {"optimizer_step": optimizer_steps},
                "metadata": metadata,
            },
            checkpoint_path,
        )
        checkpoint = {
            "path": str(checkpoint_path.resolve()),
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "bytes": checkpoint_path.stat().st_size,
            "optimizer_step": optimizer_steps,
            "metadata": json.loads(json.dumps(metadata)),
            "sha256": sha256_file(checkpoint_path),
        }
        initial = self._metric_block(updates, h0=1.0, final=1.0)
        final = self._metric_block(updates, h0=1.0, final=0.05)
        return {
            "gate": "P4_TINY_S_TINY_6",
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
                "manifest": "/data/train/pairs.jsonl",
                "manifest_sha256": "1" * 64,
                "sample_protocol": "one_pair_residuals",
                "sample_count": 1,
                "unique_image_pair_count": 1,
                "diagnostic_sample_count": 32,
                "pair_ids": ["fixed-train-pair"],
                "parent_groups": ["fixed-parent"],
                "geo_groups": ["fixed-region"],
                "test_used": False,
                "shared_once_per_pair": True,
            },
            "build": {
                "architecture_sha256": architecture_sha256,
                "provider": {
                    "loma_checkpoint": "/weights/loma.pt",
                    "dino_checkpoint": "/weights/dino.pt",
                    "selected_stage1_checkpoint": "/weights/ghim.pt",
                },
            },
            "experiments": [
                {
                    "name": name,
                    "status": "passed",
                    "active_scales": list(scales),
                    "updates": updates,
                    "seed": 0,
                    "sample_count": 32,
                    "sample_protocol": "one_pair_residuals",
                    "residual_profile": "translation",
                    "precision": "bf16",
                    "primary_readout": "raw_parameters",
                    "weight_averaging": {
                        "enabled": True,
                        "start_optimizer_step": 1536,
                        "averaged_snapshots": 0,
                        "applied_to_primary_readout": False,
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
                        "optimizer_steps": optimizer_steps,
                        "maximum_steps": 2000,
                    },
                    "controlled_H0": {
                        "profile": "translation",
                        "formal_training_injection": False,
                        "bound_fraction_of_coarsest_active_decoder": 0.5,
                        "maximum_declared_abs_residual_px": (
                            REGISTERED_MAX_RESIDUAL_PX[name]
                        ),
                        "actual_abs_residual_px": {
                            "max": REGISTERED_MAX_RESIDUAL_PX[name]
                        },
                    },
                    "criterion": {"met": True, "threshold_mace_px": 0.1},
                    "initial": initial,
                    "raw_final": deepcopy(final),
                    "final": final,
                    "history": [
                        {
                            "optimizer_step": optimizer_steps,
                            "primary_metrics": {
                                key: deepcopy(value)
                                for key, value in final.items()
                                if key != "per_pair"
                            },
                        }
                    ],
                    "gradient_finite": True,
                    "checkpoint": checkpoint,
                }
            ],
        }

    def test_complete_consistent_merge_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, (name, scales) in enumerate(REQUIRED_EXPERIMENTS.items()):
                artifact = self._artifact(name, scales, root / f"{index}.pt")
                path = root / f"{index}.json"
                path.write_text(json.dumps(artifact))
                paths.append(path)
            report = merge_tiny_gate_artifacts(paths)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["errors"], [])

    def test_incomplete_merge_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "d8.json"
            path.write_text(
                json.dumps(
                    self._artifact("TINY-S-D8", (8,), root / "d8.pt")
                )
            )
            report = merge_tiny_gate_artifacts([path])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("missing experiments" in item for item in report["errors"]))

    def test_optimizer_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact("TINY-S-D8", (8,), root / "d8.pt")
            artifact["experiments"][0]["optimizer"]["gradient_accumulation"] = 1
            path = root / "d8.json"
            path.write_text(json.dumps(artifact))
            report = merge_tiny_gate_artifacts([path])
        self.assertTrue(any("optimizer recipe" in item for item in report["errors"]))

    def test_superseded_architecture_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact("TINY-S-D8", (8,), root / "d8.pt")
            artifact["build"]["architecture_sha256"] = "2" * 64
            path = root / "d8.json"
            path.write_text(json.dumps(artifact))
            report = merge_tiny_gate_artifacts([path])
        self.assertTrue(
            any("architecture SHA256" in item for item in report["errors"])
        )

    def test_checkpoint_hash_and_metadata_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact("TINY-S-D8", (8,), root / "d8.pt")
            artifact["experiments"][0]["checkpoint"]["sha256"] = "0" * 64
            artifact["experiments"][0]["checkpoint"]["metadata"][
                "diagnostic"
            ] = "wrong"
            path = root / "d8.json"
            path.write_text(json.dumps(artifact))
            report = merge_tiny_gate_artifacts([path])
        joined = "\n".join(report["errors"])
        self.assertIn("SHA256 does not match", joined)
        self.assertIn("metadata", joined)

    def test_metric_summary_and_readout_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._artifact("TINY-S-D8", (8,), root / "d8.pt")
            experiment = artifact["experiments"][0]
            experiment["final"]["H_final_mace_px"]["mean"] = 0.01
            experiment["primary_readout"] = "equal_weight_parameter_average"
            path = root / "d8.json"
            path.write_text(json.dumps(artifact))
            report = merge_tiny_gate_artifacts([path])
        joined = "\n".join(report["errors"])
        self.assertIn("summary statistics do not match", joined)
        self.assertIn("primary readout disagrees", joined)


if __name__ == "__main__":
    unittest.main()
