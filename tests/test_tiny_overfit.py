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
