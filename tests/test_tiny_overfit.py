from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from mhinet.geometry import image_corners, normalized_to_pixel, safe_project_points
from mhinet.tiny_overfit import (
    CachedTinySample,
    _ParameterAverager,
    _preflight_tiny_progress,
    controlled_h0_from_ground_truth,
    parse_experiments,
    run_one_tiny_experiment,
)


class TinyOverfitProtocolTests(unittest.TestCase):
    def test_weight_average_evaluation_accepts_preloaded_repeated_pair(self) -> None:
        class FakeIterator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapters = nn.ModuleDict(
                    {str(scale): nn.Linear(1, 1, bias=False) for scale in (8, 4, 2, 1)}
                )
                self.decoders = nn.ModuleDict(
                    {str(scale): nn.Linear(1, 1, bias=False) for scale in (8, 4, 2, 1)}
                )

            def forward(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return {}

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.iterator = FakeIterator()

            @property
            def adapters(self) -> nn.ModuleDict:
                return self.iterator.adapters

            @property
            def refinement_decoders(self) -> nn.ModuleDict:
                return self.iterator.decoders

        model = FakeModel()
        sample = CachedTinySample(
            pair_id="train-only",
            parent_group="parent",
            geo_group="geo",
            H_gt_norm=torch.eye(3),
            pyramid={8: torch.ones((1, 2, 256, 1, 1))},
        )
        cache = [sample for _ in range(32)]
        evaluation_calls: list[dict[str, object]] = []

        def fake_evaluate(
            _model: nn.Module,
            samples: list[CachedTinySample],
            h0_values: list[torch.Tensor],
            *,
            active_scales: tuple[int, ...],
            device: torch.device,
            cnn_autocast_enabled: bool,
            preloaded_pyramid: dict[int, torch.Tensor] | None,
        ) -> dict[str, object]:
            self.assertEqual(len(samples), 32)
            self.assertEqual(len(h0_values), 32)
            self.assertEqual(active_scales, (8,))
            self.assertEqual(device.type, "cpu")
            self.assertFalse(cnn_autocast_enabled)
            self.assertIsNotNone(preloaded_pyramid)
            evaluation_calls.append({"preloaded": preloaded_pyramid})
            return {
                "H_final_mace_px": {
                    "mean": 0.05,
                    "median": 0.05,
                    "p90": 0.05,
                    "max": 0.05,
                },
                "failed_pairs": 0,
                "rejected_updates": 0,
            }

        def fake_loss(*_args: object, **_kwargs: object) -> dict[str, object]:
            parameter = next(
                value for value in model.iterator.parameters() if value.requires_grad
            )
            return {"skip_step": False, "loss": parameter.square().sum()}

        with TemporaryDirectory() as directory, patch(
            "mhinet.tiny_overfit._fresh_new_modules"
        ), patch(
            "mhinet.tiny_overfit.evaluate_tiny_training_set",
            side_effect=fake_evaluate,
        ), patch(
            "mhinet.tiny_overfit.sequence_corner_l1", side_effect=fake_loss
        ), patch(
            "mhinet.tiny_overfit.save_checkpoint", return_value={}
        ), patch(
            "mhinet.tiny_overfit.sha256_file", return_value="test-sha256"
        ):
            report = run_one_tiny_experiment(
                model,  # type: ignore[arg-type]
                cache,
                SimpleNamespace(device="cpu", output_root=Path(directory)),  # type: ignore[arg-type]
                name="TINY-S-D8",
                active_scales=(8,),
                seed=0,
                max_steps=1,
                eval_interval=1,
                threshold_mace_px=0.1,
                sample_protocol="one_pair_residuals",
                residual_profile="translation",
                precision="fp32",
                weight_average_start_step=1,
                overwrite=True,
            )

        self.assertEqual(len(evaluation_calls), 3)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["primary_readout"], "equal_weight_parameter_average")

    def test_parameter_average_is_equal_weight_and_reversibly_applied(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        averager = _ParameterAverager([parameter])
        averager.update()
        with torch.no_grad():
            parameter.copy_(torch.tensor([3.0, 7.0]))
        averager.update()
        original = averager.apply()
        self.assertTrue(torch.equal(parameter, torch.tensor([2.0, 5.0])))
        averager.restore(original)
        self.assertTrue(torch.equal(parameter, torch.tensor([3.0, 7.0])))

    def test_parameter_average_state_is_strictly_resumable(self) -> None:
        first = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        averager = _ParameterAverager([first], ["adapter.weight"])
        averager.update()
        with torch.no_grad():
            first.copy_(torch.tensor([3.0, 7.0]))
        averager.update()
        state = averager.state_dict()

        replacement = torch.nn.Parameter(torch.zeros(2))
        restored = _ParameterAverager([replacement], ["adapter.weight"])
        restored.load_state_dict(state)
        self.assertEqual(restored.count, 2)
        restored.apply()
        self.assertTrue(torch.equal(replacement, torch.tensor([2.0, 5.0])))

        wrong_name = dict(state)
        wrong_name["parameter_names"] = ["decoder.weight"]
        with self.assertRaisesRegex(ValueError, "names"):
            restored.load_state_dict(wrong_name)
        nonfinite = dict(state)
        nonfinite["averages"] = [torch.tensor([float("nan"), 1.0])]
        with self.assertRaises(FloatingPointError):
            restored.load_state_dict(nonfinite)

    def test_progress_resume_matches_uninterrupted_optimizer_boundary(self) -> None:
        class FakeIterator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapters = nn.ModuleDict(
                    {str(scale): nn.Linear(1, 1, bias=False) for scale in (8, 4, 2, 1)}
                )
                self.decoders = nn.ModuleDict(
                    {str(scale): nn.Linear(1, 1, bias=False) for scale in (8, 4, 2, 1)}
                )
                with torch.no_grad():
                    for index, parameter in enumerate(self.parameters()):
                        parameter.fill_(0.4 + index * 0.01)

            def forward(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return {}

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.iterator = FakeIterator()

            @property
            def adapters(self) -> nn.ModuleDict:
                return self.iterator.adapters

            @property
            def refinement_decoders(self) -> nn.ModuleDict:
                return self.iterator.decoders

        sample = CachedTinySample(
            pair_id="train-only",
            parent_group="parent",
            geo_group="geo",
            H_gt_norm=torch.eye(3),
            pyramid={8: torch.ones((1, 2, 256, 1, 1))},
        )
        cache = [sample for _ in range(32)]

        def fake_evaluate(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "H_final_mace_px": {
                    "mean": 1.0,
                    "median": 1.0,
                    "p90": 1.0,
                    "max": 1.0,
                },
                "failed_pairs": 0,
                "rejected_updates": 0,
                "elapsed_seconds": 0.0,
            }

        def run_fake(
            model: FakeModel,
            output_root: Path,
            *,
            progress: Path | None = None,
            resume: Path | None = None,
            callback: object = None,
        ) -> dict[str, object]:
            def fake_loss(*_args: object, **_kwargs: object) -> dict[str, object]:
                loss = sum(
                    (parameter - 0.125).square().sum()
                    for parameter in model.iterator.parameters()
                    if parameter.requires_grad
                )
                return {"skip_step": False, "loss": loss}

            with patch("mhinet.tiny_overfit._fresh_new_modules"), patch(
                "mhinet.tiny_overfit.evaluate_tiny_training_set",
                side_effect=fake_evaluate,
            ), patch(
                "mhinet.tiny_overfit.sequence_corner_l1", side_effect=fake_loss
            ):
                return run_one_tiny_experiment(
                    model,  # type: ignore[arg-type]
                    cache,
                    SimpleNamespace(device="cpu", output_root=output_root),  # type: ignore[arg-type]
                    name="TINY-S-D8",
                    active_scales=(8,),
                    seed=0,
                    max_steps=3,
                    eval_interval=2,
                    threshold_mace_px=0.1,
                    sample_protocol="one_pair_residuals",
                    residual_profile="translation",
                    precision="fp32",
                    weight_average_start_step=2,
                    overwrite=True,
                    progress_checkpoint=progress,
                    resume_progress=resume,
                    heartbeat_interval=1,
                    progress_callback=callback,  # type: ignore[arg-type]
                )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            continuous_model = FakeModel()
            continuous = run_fake(continuous_model, root / "continuous")

            progress = root / "interrupted" / "progress.pt"
            interrupted_model = FakeModel()

            def interrupt_after_step_one(payload: dict[str, object]) -> None:
                if payload["optimizer_step"] == 1:
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_fake(
                    interrupted_model,
                    root / "interrupted",
                    progress=progress,
                    callback=interrupt_after_step_one,
                )
            interrupted_payload = torch.load(
                progress, map_location="cpu", weights_only=True
            )
            self.assertEqual(interrupted_payload["progress"]["optimizer_step"], 1)
            self.assertFalse(
                interrupted_payload["metadata"]["checkpoint_contains_weight_average"]
            )
            mismatched_signature = dict(
                interrupted_payload["metadata"]["tiny_progress_signature"]
            )
            mismatched_signature["seed"] = 99
            with self.assertRaisesRegex(RuntimeError, "signature mismatch"):
                _preflight_tiny_progress(progress, mismatched_signature)

            resumed_model = FakeModel()
            resumed = run_fake(
                resumed_model,
                root / "resumed",
                progress=progress,
                resume=progress,
            )

            self.assertEqual(
                [item["optimizer_step"] for item in continuous["history"]],
                [item["optimizer_step"] for item in resumed["history"]],
            )
            self.assertEqual(
                continuous["weight_averaging"]["averaged_snapshots"],  # type: ignore[index]
                resumed["weight_averaging"]["averaged_snapshots"],  # type: ignore[index]
            )
            self.assertEqual(resumed["progress"]["resume_segments"], 1)  # type: ignore[index]
            self.assertEqual(
                resumed["progress"]["checkpoint"]["optimizer_step"],  # type: ignore[index]
                3,
            )
            for key, expected in continuous_model.iterator.state_dict().items():
                torch.testing.assert_close(
                    resumed_model.iterator.state_dict()[key], expected, rtol=0, atol=0
                )

            continuous_checkpoint = torch.load(
                continuous["checkpoint"]["path"],  # type: ignore[index]
                map_location="cpu",
                weights_only=True,
            )
            resumed_checkpoint = torch.load(
                resumed["checkpoint"]["path"],  # type: ignore[index]
                map_location="cpu",
                weights_only=True,
            )
            for state_index, state in continuous_checkpoint["optimizer"]["state"].items():
                for state_name, expected in state.items():
                    actual = resumed_checkpoint["optimizer"]["state"][state_index][state_name]
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        self.assertEqual(actual, expected)

    def test_experiment_parser_has_four_single_scales_and_eight_round(self) -> None:
        parsed = parse_experiments("all")
        self.assertEqual(
            parsed,
            (
                ("TINY-S-D8", (8,)),
                ("TINY-S-D4", (4,)),
                ("TINY-S-D2", (2,)),
                ("TINY-S-D1", (1,)),
                ("TINY-8", (8, 4, 2, 1)),
            ),
        )

    def test_controlled_h0_is_deterministic_legal_and_inside_range(self) -> None:
        H_gt = torch.eye(3)
        first_h, first_residual = controlled_h0_from_ground_truth(
            H_gt,
            sample_index=11,
            max_abs_residual_px=3.0,
            seed=7,
        )
        second_h, second_residual = controlled_h0_from_ground_truth(
            H_gt,
            sample_index=11,
            max_abs_residual_px=3.0,
            seed=7,
        )
        self.assertTrue(torch.equal(first_h, second_h))
        self.assertTrue(torch.equal(first_residual, second_residual))
        self.assertGreater(float(first_residual.abs().max()), 0.1)
        self.assertLessEqual(float(first_residual.abs().max()), 3.001)
        corners = image_corners((784, 784))
        projected, valid, _ = safe_project_points(first_h, corners)
        self.assertTrue(bool(valid.all()))
        measured = normalized_to_pixel(corners, (784, 784)) - normalized_to_pixel(
            projected, (784, 784)
        )
        self.assertTrue(torch.allclose(measured, first_residual, atol=1e-3, rtol=0))


if __name__ == "__main__":
    unittest.main()
