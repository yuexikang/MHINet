from __future__ import annotations

import unittest

import torch
from torch import nn

from mhinet.models.model import MHINet
from mhinet.models.modules import EXPECTED_MAINLINE_TRAINABLE_NEW_PARAMETERS


class FakeFeatureProvider(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mvt = nn.Linear(1, 1)
        self.vgg = nn.Linear(1, 1)
        self.dedode = nn.Linear(1, 1)
        self.dino = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)
        self.last_pyramid_scales: tuple[int, ...] | None = None

    def set_training_groups(self, *, mvt: bool, vgg: bool, dedode: bool) -> None:
        for module, enabled in (
            (self.mvt, mvt),
            (self.vgg, vgg),
            (self.dedode, dedode),
            (self.dino, False),
            (self.head, False),
        ):
            for parameter in module.parameters():
                parameter.requires_grad = enabled

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "mvt": list(self.mvt.parameters()),
            "vgg": list(self.vgg.parameters()),
            "dedode": list(self.dedode.parameters()),
            "dino": list(self.dino.parameters()),
            "stage1_head_parameters": list(self.head.parameters()),
        }

    def forward(
        self,
        images: torch.Tensor,
        *,
        pyramid_scales: tuple[int, ...],
    ) -> dict[str, object]:
        self.last_pyramid_scales = pyramid_scales
        return {
            "pyramid": {scale: torch.zeros(1) for scale in pyramid_scales},
            "H0_norm": torch.eye(3).unsqueeze(0),
            "stage1_valid": torch.tensor([True]),
            "stage1_valid_correspondences": torch.tensor([16]),
            "call_counts": {
                "dino": 1,
                "mvt": 1,
                "vgg": 1,
                "dedode_scale1": 0,
            },
        }


class RecordingIterator(nn.Module):
    scales = (8, 4, 2, 1)

    def __init__(self) -> None:
        super().__init__()
        self.active_scales: tuple[int, ...] | None = None

    def forward(self, pyramid: object, H0: object, valid: object, **kwargs: object) -> dict[str, object]:
        self.active_scales = tuple(kwargs["active_scales"])  # type: ignore[arg-type]
        return {"feature_valid_counts": {}}


class MainlineCutoffTests(unittest.TestCase):
    def test_d1_is_registered_but_not_in_mainline_optimizer(self) -> None:
        model = MHINet(FakeFeatureProvider())
        report = model.set_training_phase("heads")
        self.assertEqual(
            report["groups"]["new_modules"]["trainable_parameters"],
            EXPECTED_MAINLINE_TRAINABLE_NEW_PARAMETERS,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (model.adapters["1"], model.refinement_decoders["1"])
                for parameter in module.parameters()
            )
        )

    def test_default_forward_requests_cgmdp_only_through_d2(self) -> None:
        provider = FakeFeatureProvider()
        model = MHINet(provider)
        recorder = RecordingIterator()
        model.iterator = recorder
        output = model(torch.zeros(1))
        self.assertEqual(provider.last_pyramid_scales, (8, 4, 2))
        self.assertEqual(recorder.active_scales, (8, 4, 2))
        self.assertEqual(output["shared_call_counts"]["dedode_scale1"], 0)


if __name__ == "__main__":
    unittest.main()
