from __future__ import annotations

import unittest

import torch
from torch import nn

from mhinet.ops.geometry import image_corners, normalized_to_pixel, safe_project_points
from mhinet.models.iterator import IteratorReason, MultiScaleHIterator
from mhinet.models.modules import (
    SCALE_SPECS,
    ResidualGeometryDecoder,
    build_multiscale_modules,
)


class TinyAdapter(nn.Module):
    def forward(self, descriptors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = descriptors[:, :, :2]
        norm = torch.linalg.vector_norm(features, dim=2, keepdim=True)
        valid = norm > 1e-6
        return features / norm.clamp_min(1e-6), valid


class DeltaDecoder(nn.Module):
    def __init__(self, initial: float = 0.0) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.full((4, 2), initial))
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(decoder_input.shape))
        return self.delta.unsqueeze(0).expand(decoder_input.shape[0], -1, -1)


class RecordingCorrelation(nn.Module):
    def __init__(self, candidates: int, valid: bool = True) -> None:
        super().__init__()
        self.candidates = candidates
        self.valid = valid
        self.h_inputs: list[torch.Tensor] = []

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, homography: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.h_inputs.append(homography.detach().clone())
        batch, _, height, width = source.shape
        correlation = source.new_zeros(batch, self.candidates, height, width)
        mask = torch.full(
            (batch, self.candidates, height, width),
            self.valid,
            dtype=torch.bool,
            device=source.device,
        )
        return correlation, mask


def build_stub_iterator(*, delta: float = 0.0, support: bool = True) -> MultiScaleHIterator:
    adapters = nn.ModuleDict({str(scale): TinyAdapter() for scale in (8, 4, 2, 1)})
    decoders = nn.ModuleDict(
        {str(scale): DeltaDecoder(delta) for scale in (8, 4, 2, 1)}
    )
    iterator = MultiScaleHIterator(
        adapters,
        decoders,
        minimum_supported_queries=1,
    )
    iterator.correlations = nn.ModuleDict(
        {
            str(scale): RecordingCorrelation(
                SCALE_SPECS[scale].correlation_candidates, support
            )
            for scale in (8, 4, 2, 1)
        }
    )
    return iterator


def tiny_pyramid(batch: int = 1, spatial: int = 4) -> dict[int, torch.Tensor]:
    torch.manual_seed(41)
    return {
        scale: torch.randn(batch, 2, 256, spatial, spatial)
        for scale in (8, 4, 2, 1)
    }


class IteratorTests(unittest.TestCase):
    def test_zero_update_is_six_round_noop_with_complete_contract(self) -> None:
        iterator = build_stub_iterator()
        identity = torch.eye(3).unsqueeze(0)
        outputs = iterator(tiny_pyramid(), identity, torch.tensor([True]))
        self.assertEqual(outputs["H_updates_norm"].shape, (1, 6, 3, 3))
        self.assertEqual(outputs["H_scales_norm"].shape, (1, 3, 3, 3))
        self.assertEqual(outputs["proposal_Q_norm"].shape, (1, 6, 4, 2))
        self.assertEqual(outputs["decoder_output_finite"].shape, (1, 6))
        self.assertTrue(bool(outputs["decoder_output_finite"].all()))
        self.assertTrue(bool(outputs["update_accepted"].all()))
        self.assertTrue(bool(outputs["overall_valid"].all()))
        self.assertTrue(
            torch.equal(
                outputs["failure_reason_codes"], torch.zeros(1, 6, dtype=torch.long)
            )
        )
        corners = image_corners((784, 784)).unsqueeze(0)
        projected, valid, _ = safe_project_points(outputs["H_final_norm"], corners)
        self.assertTrue(bool(valid.all()))
        error_px = torch.linalg.vector_norm(
            normalized_to_pixel(projected, (784, 784))
            - normalized_to_pixel(corners, (784, 784)),
            dim=-1,
        ).max()
        self.assertLess(float(error_px.detach()), 1e-3)
        for scale in (8, 4, 2):
            self.assertEqual(len(iterator.correlations[str(scale)].h_inputs), 2)
            self.assertEqual(
                iterator.decoders[str(scale)].input_shapes,
                [
                    (1, SCALE_SPECS[scale].correlation_candidates, 4, 4),
                    (1, SCALE_SPECS[scale].correlation_candidates, 4, 4),
                ],
            )
        self.assertEqual(len(iterator.correlations["1"].h_inputs), 0)
        self.assertEqual(iterator.decoders["1"].input_shapes, [])

    def test_second_round_uses_first_round_h_and_recomputes(self) -> None:
        iterator = build_stub_iterator(delta=0.5)
        outputs = iterator(
            tiny_pyramid(), torch.eye(3).unsqueeze(0), torch.tensor([True])
        )
        recorder = iterator.correlations["8"]
        self.assertFalse(torch.equal(recorder.h_inputs[0], recorder.h_inputs[1]))
        self.assertFalse(torch.equal(outputs["H_updates_norm"][:, 0], outputs["H_updates_norm"][:, 1]))

    def test_final_geometry_gradient_reaches_every_earlier_scale(self) -> None:
        iterator = build_stub_iterator(delta=0.05)
        outputs = iterator(
            tiny_pyramid(), torch.eye(3).unsqueeze(0), torch.tensor([True])
        )
        corners = image_corners((784, 784)).unsqueeze(0)
        projected, valid, _ = safe_project_points(outputs["H_final_norm"], corners)
        self.assertTrue(bool(valid.all()))
        projected.sum().backward()
        for scale in (8, 4, 2):
            gradient = iterator.decoders[str(scale)].delta.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.norm()), 0.0)
        self.assertIsNone(iterator.decoders["1"].delta.grad)

    def test_no_support_and_stage1_failure_have_explicit_reasons(self) -> None:
        iterator = build_stub_iterator(support=False)
        outputs = iterator(
            tiny_pyramid(batch=2),
            torch.eye(3).repeat(2, 1, 1),
            torch.tensor([True, False]),
        )
        self.assertFalse(bool(outputs["update_accepted"].any()))
        self.assertTrue(
            torch.equal(
                outputs["failure_reason_codes"][0],
                torch.full((6,), int(IteratorReason.NO_VALID_SUPPORT)),
            )
        )
        self.assertTrue(
            torch.equal(
                outputs["failure_reason_codes"][1],
                torch.full((6,), int(IteratorReason.STAGE1_INVALID)),
            )
        )
        self.assertEqual(outputs["overall_valid"].tolist(), [True, False])

    def test_real_zero_initialized_decoders_are_noop_on_small_grid(self) -> None:
        adapters, _ = build_multiscale_modules()
        decoders = nn.ModuleDict(
            {
                str(scale): ResidualGeometryDecoder(
                    SCALE_SPECS[scale].decoder_input_channels,
                    1,
                    SCALE_SPECS[scale].max_delta_px,
                    input_spatial_size=4,
                )
                for scale in (8, 4, 2, 1)
            }
        )
        iterator = MultiScaleHIterator(
            adapters,
            decoders,
            query_chunk_size=8,
            minimum_supported_queries=1,
        )
        outputs = iterator(
            tiny_pyramid(spatial=4),
            torch.eye(3).unsqueeze(0),
            torch.tensor([True]),
        )
        self.assertTrue(torch.equal(outputs["delta_px"], torch.zeros_like(outputs["delta_px"])))
        corners = image_corners((784, 784)).unsqueeze(0)
        final, valid, _ = safe_project_points(outputs["H_final_norm"], corners)
        self.assertTrue(bool(valid.all()))
        max_error_px = (
            normalized_to_pixel(final, (784, 784))
            - normalized_to_pixel(corners, (784, 784))
        ).abs().max()
        self.assertLess(float(max_error_px.detach()), 1e-3)

    def test_single_scale_diagnostic_schedule_uses_only_requested_pyramid(self) -> None:
        iterator = build_stub_iterator()
        outputs = iterator(
            {2: tiny_pyramid()[2]},
            torch.eye(3).unsqueeze(0),
            torch.tensor([True]),
            active_scales=(2,),
        )
        self.assertEqual(outputs["H_updates_norm"].shape, (1, 2, 3, 3))
        self.assertEqual(outputs["H_scales_norm"].shape, (1, 1, 3, 3))
        self.assertEqual(outputs["proposal_Q_norm"].shape, (1, 2, 4, 2))
        self.assertEqual(outputs["active_scales"], (2,))
        self.assertEqual(outputs["update_scale_schedule"], (2, 2))
        self.assertEqual(len(iterator.correlations["2"].h_inputs), 2)
        for scale in (8, 4, 1):
            self.assertEqual(len(iterator.correlations[str(scale)].h_inputs), 0)

    def test_d1_is_retained_as_an_explicit_opt_in_only(self) -> None:
        iterator = build_stub_iterator()
        outputs = iterator(
            {1: tiny_pyramid()[1]},
            torch.eye(3).unsqueeze(0),
            torch.tensor([True]),
            active_scales=(1,),
        )
        self.assertEqual(outputs["active_scales"], (1,))
        self.assertEqual(outputs["update_scale_schedule"], (1, 1))
        self.assertEqual(len(iterator.correlations["1"].h_inputs), 2)
        for scale in (8, 4, 2):
            self.assertEqual(len(iterator.correlations[str(scale)].h_inputs), 0)


if __name__ == "__main__":
    unittest.main()
