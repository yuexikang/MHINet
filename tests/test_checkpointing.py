"""Round-trip tests for resumable MHINet checkpoints."""

from __future__ import annotations

import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from mhinet.engine.checkpointing import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
)


class CheckpointRoundTripTests(unittest.TestCase):
    def test_training_state_progress_and_rng_round_trip(self) -> None:
        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        scaler = torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=1)

        inputs = torch.randn(4, 3)
        loss = model(inputs).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        expected_model = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        first_parameter = next(model.parameters())
        expected_exp_avg = optimizer.state[first_parameter]["exp_avg"].detach().clone()
        expected_optimizer_step = optimizer.state[first_parameter]["step"].detach().clone()
        expected_scheduler = dict(scheduler.state_dict())
        expected_scaler = dict(scaler.state_dict())
        expected_lr = optimizer.param_groups[0]["lr"]

        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        microbatch_progress = {
            "window_index": 17,
            "valid_microbatches": 3,
            "pending_pair_ids": ["p17"],
        }
        data_progress = {
            "epoch": 2,
            "sample_offset": 41,
            "sampler": {"permutation_position": 41},
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "resume.pt"
            saved = save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                optimizer_step=23,
                microbatch_progress=microbatch_progress,
                data_progress=data_progress,
                metadata={"run_id": "tiny", "config_path": Path("configs/tiny.json")},
                auxiliary_state={
                    "averaged_snapshots": 4,
                    "averages": [torch.tensor([1.5, 2.5])],
                },
            )
            self.assertTrue(path.is_file())
            self.assertEqual(saved["optimizer_step"], 23)
            self.assertGreater(saved["bytes"], 0)
            self.assertNotIn(
                "auxiliary_state",
                saved,
                msg="large binary-only state must never be expanded into JSON reports",
            )
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.*.tmp")),
                [],
                msg="atomic-save temporary file leaked",
            )

            # Prove the complete payload can be read by PyTorch's restricted
            # unpickler; no unsafe fallback is needed for our own checkpoints.
            raw = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(raw["format"], CHECKPOINT_FORMAT)
            self.assertEqual(raw["version"], CHECKPOINT_VERSION)
            self.assertIn("torch_cpu", raw["rng"])
            self.assertIn("torch_cuda", raw["rng"])
            self.assertEqual(raw["auxiliary_state"]["averaged_snapshots"], 4)

            expected_python = [random.random() for _ in range(4)]
            expected_numpy = np.random.random(4)
            expected_torch = torch.rand(4)

            # Constructing replacement objects and consuming randomness makes
            # restoration observable rather than a no-op comparison.
            replacement = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 2))
            replacement_optimizer = torch.optim.AdamW(replacement.parameters(), lr=0.2)
            replacement_scheduler = torch.optim.lr_scheduler.StepLR(
                replacement_optimizer, step_size=3, gamma=0.1
            )
            replacement_scaler = torch.amp.GradScaler(
                "cpu", init_scale=8.0, growth_interval=9
            )
            random.random()
            np.random.random()
            torch.rand(7)

            loaded = load_checkpoint(
                path,
                model=replacement,
                optimizer=replacement_optimizer,
                scheduler=replacement_scheduler,
                scaler=replacement_scaler,
                map_location="cpu",
            )

            self.assertEqual(loaded["optimizer_step"], 23)
            self.assertEqual(loaded["microbatch_progress"], microbatch_progress)
            self.assertEqual(loaded["data_progress"], data_progress)
            self.assertEqual(loaded["metadata"]["run_id"], "tiny")
            self.assertEqual(loaded["metadata"]["config_path"], "configs/tiny.json")
            self.assertEqual(loaded["auxiliary_state"]["averaged_snapshots"], 4)
            torch.testing.assert_close(
                loaded["auxiliary_state"]["averages"][0],
                torch.tensor([1.5, 2.5]),
            )
            self.assertEqual(loaded["missing_model_keys"], [])
            self.assertEqual(loaded["unexpected_model_keys"], [])
            self.assertEqual(
                loaded["restored"],
                {"model": True, "optimizer": True, "scheduler": True, "scaler": True},
            )
            self.assertTrue(loaded["rng_restored"]["cpu"])

            for key, expected in expected_model.items():
                torch.testing.assert_close(replacement.state_dict()[key], expected)
            restored_parameter = next(replacement.parameters())
            torch.testing.assert_close(
                replacement_optimizer.state[restored_parameter]["exp_avg"],
                expected_exp_avg,
            )
            torch.testing.assert_close(
                replacement_optimizer.state[restored_parameter]["step"],
                expected_optimizer_step,
            )
            self.assertEqual(replacement_optimizer.param_groups[0]["lr"], expected_lr)
            self.assertEqual(replacement_scheduler.state_dict(), expected_scheduler)
            self.assertEqual(replacement_scaler.state_dict(), expected_scaler)

            self.assertEqual([random.random() for _ in range(4)], expected_python)
            np.testing.assert_array_equal(np.random.random(4), expected_numpy)
            torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)

    def test_none_scheduler_scaler_and_explicit_map_location(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "minimal.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                optimizer_step=0,
                microbatch_progress=0,
                data_progress={"epoch": 0},
            )
            with self.assertRaises(TypeError):
                # map_location is intentionally a required keyword argument.
                load_checkpoint(path, model=model, optimizer=optimizer)  # type: ignore[call-arg]
            result = load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                map_location=torch.device("cpu"),
            )
            self.assertEqual(result["optimizer_step"], 0)
            self.assertFalse(result["restored"]["scheduler"])
            self.assertFalse(result["restored"]["scaler"])

    def test_expected_metadata_is_checked_before_model_mutation(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "architecture-bound.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                optimizer_step=0,
                metadata={
                    "architecture_sha256": "old",
                    "mhir_revision": "residual_mlp",
                },
            )
            replacement = nn.Linear(2, 1)
            before = {
                key: value.detach().clone()
                for key, value in replacement.state_dict().items()
            }
            with self.assertRaisesRegex(RuntimeError, "architecture_sha256"):
                load_checkpoint(
                    path,
                    model=replacement,
                    map_location="cpu",
                    restore_rng=False,
                    expected_metadata={
                        "architecture_sha256": "current",
                        "mhir_revision": "mcnet_correlation_decoder_784_v1",
                    },
                )
            for key, value in replacement.state_dict().items():
                torch.testing.assert_close(value, before[key])


if __name__ == "__main__":
    unittest.main()
