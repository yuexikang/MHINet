from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from mhinet.models.feature_provider import SharedFeatureProvider


class RecordingCumulativeDecoder:
    scales = ["16", "8", "4", "2", "1"]

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        feature_map: torch.Tensor,
        *,
        scale: str,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(scale)
        batch, _channels, height, width = feature_map.shape
        delta = torch.zeros(batch, 256, height, width)
        next_context = torch.zeros(batch, 8, height, width)
        return delta, next_context


class CgmdpEarlyStopTests(unittest.TestCase):
    def test_joint_training_keeps_registered_d1_decoder_frozen_and_in_eval(self) -> None:
        descriptor = torch.nn.Module()
        descriptor.encoder = torch.nn.Module()
        descriptor.encoder.frozen_dinov3 = torch.nn.Module()
        descriptor.encoder.frozen_dinov3.model = torch.nn.Linear(2, 2)
        descriptor.encoder.frozen_dinov3.multi_view_transformer = torch.nn.Linear(2, 2)
        descriptor.encoder.vgg = torch.nn.Linear(2, 2)
        descriptor.decoder = torch.nn.Module()
        descriptor.decoder.layers = torch.nn.ModuleDict({
            scale: torch.nn.Linear(2, 2) for scale in ("16", "8", "4", "2", "1")
        })
        provider = SharedFeatureProvider(descriptor, torch.nn.Linear(2, 2))
        provider.set_training_groups(mvt=True, vgg=True, dedode=True)
        provider.eval()
        provider.train()
        inactive = list(provider.dedode.layers["1"].parameters())
        self.assertTrue(all(not p.requires_grad for p in inactive))
        self.assertFalse(provider.dedode.layers["1"].training)
        self.assertTrue(all(p.requires_grad for p in provider.dedode.layers["2"].parameters()))
        optimizer_ids = {id(p) for p in provider.parameters() if p.requires_grad}
        self.assertTrue(all(id(p) not in optimizer_ids for p in inactive))
        self.assertIn("descriptor.decoder.layers.1.weight", provider.state_dict())

    def test_mainline_cumulative_decode_stops_at_d2_without_scale1_call(self) -> None:
        decoder = RecordingCumulativeDecoder()
        owner = SimpleNamespace(
            dedode=decoder,
            _call_counts={
                "dedode_decode_calls": 0,
                "dedode_steps": 0,
                "dedode_scale1": 0,
            },
        )
        vgg_features = [
            torch.zeros(2, 8, 32, 32),
            torch.zeros(2, 8, 16, 16),
            torch.zeros(2, 8, 8, 8),
            torch.zeros(2, 8, 4, 4),
        ]
        vgg_sizes = [(32, 32), (16, 16), (8, 8), (4, 4)]
        contextualized = torch.zeros(1, 2, 2, 2, 8)

        outputs = SharedFeatureProvider._decode_pyramid(
            owner,  # type: ignore[arg-type]
            vgg_features,
            vgg_sizes,
            contextualized,
            (8, 4, 2),
        )

        self.assertEqual(tuple(outputs), (8, 4, 2))
        self.assertEqual(decoder.calls, ["16", "8", "4", "2"])
        self.assertEqual(owner._call_counts["dedode_decode_calls"], 1)
        self.assertEqual(owner._call_counts["dedode_steps"], 4)
        self.assertEqual(owner._call_counts["dedode_scale1"], 0)

    def test_d1_path_is_still_available_only_when_explicitly_requested(self) -> None:
        decoder = RecordingCumulativeDecoder()
        owner = SimpleNamespace(
            dedode=decoder,
            _call_counts={
                "dedode_decode_calls": 0,
                "dedode_steps": 0,
                "dedode_scale1": 0,
            },
        )
        vgg_features = [
            torch.zeros(2, 8, 32, 32),
            torch.zeros(2, 8, 16, 16),
            torch.zeros(2, 8, 8, 8),
            torch.zeros(2, 8, 4, 4),
        ]
        contextualized = torch.zeros(1, 2, 2, 2, 8)
        SharedFeatureProvider._decode_pyramid(
            owner,  # type: ignore[arg-type]
            vgg_features,
            [(32, 32), (16, 16), (8, 8), (4, 4)],
            contextualized,
            (8, 4, 2, 1),
        )
        self.assertEqual(decoder.calls[-1], "1")
        self.assertEqual(owner._call_counts["dedode_scale1"], 1)


if __name__ == "__main__":
    unittest.main()
