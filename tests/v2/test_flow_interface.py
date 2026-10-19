"""CPU contracts for the real v2 flow adapter without external checkpoints."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mhinet.models.feature_provider import SharedFeatureProvider
from mhinet.v2.feature_provider import (
    FlowStage1Adapter, V2SharedDescriptorNetwork, build_v2_shared, normalized_d16_grid,
)


class _Fitter(nn.Module):
    def __init__(self, forbidden=False):
        super().__init__()
        self.calls = 0
        self.forbidden = forbidden

    def forward(self, warp, matchability):
        self.calls += 1
        if self.forbidden:
            raise AssertionError("H fitting must never run in Flow0")
        b = warp.shape[0]
        return torch.eye(3).expand(b, -1, -1), torch.ones(b, dtype=torch.bool), torch.full((b,), 1024)


class _Decoder(nn.Module):
    def __init__(self, offset=(0., 0.), embedding_response=False):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset))
        self.embedding_response = embedding_response
        self.last_embeddings = None

    def forward(self, embeddings):
        self.last_embeddings = embeddings
        b, h, w, _ = embeddings.shape
        yy, xx = torch.meshgrid(torch.linspace(-1 + 1 / h, 1 - 1 / h, h),
                                torch.linspace(-1 + 1 / w, 1 - 1 / w, w), indexing="ij")
        warp = torch.stack((xx, yy))[None].expand(b, -1, -1, -1) + self.offset[None, :, None, None]
        if self.embedding_response:
            warp = warp + .01 * embeddings[..., :2].permute(0, 3, 1, 2)
        return warp, torch.full((b, 1, h, w), .8)


class _Head(nn.Module):
    def __init__(self, offset=(0., 0.), forbidden=True):
        super().__init__()
        self.localization_dim = 4
        self.enable_amp = False
        self.register_buffer("position_frequencies", torch.eye(2))
        self.register_buffer("position_scale", torch.tensor(1.))
        self.register_buffer("temperature", torch.tensor(.2))
        self.no_match_embedding = nn.Parameter(torch.zeros(4))
        self.no_match_logit = nn.Parameter(torch.tensor(-2.))
        self.match_embedding_decoder = _Decoder(offset)
        self.homography_fitter = _Fitter(forbidden)

    def forward(self, contextualized, *, position_dtype=None):
        result = FlowStage1Adapter.forward(self, contextualized, position_dtype=position_dtype)
        H, valid, count = self.homography_fitter(result["coarse_warp"], result["coarse_matchability"])
        return {**result, "H_A_to_B_norm": H, "fit_succeeded": valid, "valid_correspondences": count}


class _VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        return [], []


def _descriptor():
    descriptor = nn.Module()
    descriptor.encoder = nn.Module()
    descriptor.encoder.frozen_dinov3 = nn.Module()
    descriptor.encoder.frozen_dinov3.model = nn.Linear(4, 4)
    descriptor.encoder.frozen_dinov3.multi_view_transformer = nn.Linear(4, 4)
    descriptor.encoder.vgg = _VGG()
    descriptor.decoder = nn.Module()
    descriptor.decoder.scales = ["16", "8", "4", "2", "1"]
    descriptor.decoder.layers = nn.ModuleDict({s: nn.Linear(4, 4) for s in descriptor.decoder.scales})
    return descriptor


class _CheapProvider(V2SharedDescriptorNetwork):
    """Replace expensive DINO/CGMDP with shaped tensors, retaining real flow."""
    def _extract_batched_descriptors(self, images):
        return images.new_ones((images.shape[0], 2, 32, 32, 4)), (32, 32)

    def _decode_pyramid(self, features, sizes, context, scales):
        self._call_counts["dedode_decode_calls"] += 1
        return {s: context.new_zeros(()).expand(context.shape[0], 2, 256, 512 // s, 512 // s)
                for s in scales}


class FlowInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.zeros(1, 2, 3, 512, 512)

    def test_identity_and_translation_have_exact_normalized_xy_convention(self):
        for pixels in ((0., 0.), (16., -8.)):
            normalized = (pixels[0] * 2 / 512, pixels[1] * 2 / 512)
            model = _CheapProvider(_descriptor(), _Head(normalized))
            result = model(self.images, pyramid_scales=(), compute_stage1=False)
            expected = torch.tensor(normalized)[None, :, None, None].expand(1, 2, 32, 32)
            torch.testing.assert_close(result["flow16_norm"], expected, atol=1e-7, rtol=0)
            source = normalized_d16_grid(1, device=self.images.device, dtype=torch.float32)
            torch.testing.assert_close(result["flow16_target_norm"], source + expected)
            self.assertEqual(result["flow16_valid"].shape, (1, 1, 32, 32))
            self.assertEqual(model.stage1_head.homography_fitter.calls, 0)
            self.assertIsNone(result["H0_norm"])
            self.assertIsNone(result["stage1_valid"])
            self.assertIsNotNone(result["stage1"]["match_probabilities"])
            torch.testing.assert_close(result["flow16_matchability"], torch.full((1, 1, 32, 32), .8))

    def test_flow_mode_shapes_preserve_full_pyramid_and_only_one_head_registration(self):
        descriptor, head = _descriptor(), _Head()
        legacy = SharedFeatureProvider(descriptor, head)
        model = _CheapProvider(descriptor, head)
        self.assertEqual(set(model.state_dict()), set(legacy.state_dict()))
        self.assertEqual(len({id(p) for p in model.parameters()}), len(list(model.parameters())))
        result = model(self.images)
        for scale in (8, 4, 2):
            self.assertEqual(result["pyramid"][scale].shape, (1, 2, 256, 512 // scale, 512 // scale))
        self.assertEqual(result["call_counts"]["homography_fitter"], 0)

    def test_head_only_mode_has_decoder_gradients_with_frozen_backbone(self):
        model = _CheapProvider(_descriptor(), _Head())
        model.set_training_groups(mvt=False, vgg=False, dedode=False, ghim_head=True)
        result = model(self.images, pyramid_scales=())
        result["flow16_target_norm"].sum().backward()
        self.assertIsNotNone(model.stage1_head.match_embedding_decoder.offset.grad)
        self.assertTrue(torch.isfinite(model.stage1_head.match_embedding_decoder.offset.grad).all())
        self.assertTrue(all(p.grad is None for p in model.descriptor.parameters()))
        self.assertEqual(result["pyramid"], {})
        self.assertEqual(result["call_counts"]["vgg"], 0)
        self.assertEqual(result["call_counts"]["dedode_decode_calls"], 0)

    def test_none_skips_head_unless_flow_diagnostics_requested(self):
        model = _CheapProvider(_descriptor(), _Head(), geometry_mode="none")
        result = model(self.images, pyramid_scales=())
        self.assertIsNone(result["stage1"])
        self.assertIsNone(result["flow16_norm"])
        self.assertEqual(result["call_counts"]["flow_head"], 0)
        result = model(self.images, pyramid_scales=(), compute_flow=True)
        self.assertIsNotNone(result["flow16_norm"])
        self.assertIsNone(result["H0_norm"])
        self.assertEqual(model.stage1_head.homography_fitter.calls, 0)

    def test_h0_runs_legacy_head_once_and_retains_h0_values(self):
        model = _CheapProvider(_descriptor(), _Head(forbidden=False), geometry_mode="h0")
        result = model(self.images, pyramid_scales=())
        torch.testing.assert_close(result["H0_norm"], torch.eye(3)[None])
        self.assertTrue(result["stage1_valid"].all())
        self.assertEqual(model.stage1_head.homography_fitter.calls, 1)
        self.assertEqual(result["call_counts"]["homography_fitter"], 1)
        self.assertIsNotNone(result["flow16_norm"])

    def test_invalid_targets_and_probability_are_excluded_without_resigmoid(self):
        model = _CheapProvider(_descriptor(), _Head((1., 0.)))
        result = model(self.images, pyramid_scales=())
        self.assertTrue(result["flow16_valid"][..., :16].all())
        self.assertFalse(result["flow16_valid"][..., 16:].any())
        real_forward = model.stage1_head.match_embedding_decoder.forward
        def invalid(embeddings):
            target, probability = real_forward(embeddings)
            target = target.clone()
            probability = probability.clone()
            target[..., 0, 0] = float("nan")
            probability[..., 1, 1] = float("inf")
            probability[..., 2, 2] = -.1
            return target, probability
        with patch.object(model.stage1_head.match_embedding_decoder, "forward", invalid):
            result = model(self.images, pyramid_scales=())
        for y, x in ((0, 0), (1, 1), (2, 2)):
            self.assertFalse(result["flow16_valid"][0, 0, y, x])

    def test_only_512_inputs_are_accepted(self):
        model = _CheapProvider(_descriptor(), _Head())
        for shape in ((1, 2, 3, 784, 784), (1, 2, 3, 512, 511), (2, 3, 512, 512), (0, 2, 3, 512, 512)):
            with self.assertRaisesRegex(ValueError, "512"):
                model(torch.empty(shape))

    def test_adapter_position_matching_matches_independent_matrix_reference(self):
        head = _Head()
        head.match_embedding_decoder.embedding_response = True
        context = torch.randn(1, 2, 2, 2, 4, requires_grad=True)
        result = FlowStage1Adapter.forward(head, context)
        vectors_a = torch.nn.functional.normalize(context[:, 0].reshape(1, 4, 4), dim=-1)
        vectors_b = torch.nn.functional.normalize(context[:, 1].reshape(1, 4, 4), dim=-1)
        logits = vectors_a @ vectors_b.transpose(-1, -2) / .2
        logits = torch.cat((logits, torch.full((1, 4, 1), -2.)), dim=-1)
        probabilities = logits.softmax(-1)
        positions = torch.tensor([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
        candidates = torch.cat((positions.sin(), positions.cos()), dim=-1)
        candidates = torch.cat((candidates, torch.zeros(1, 4)), dim=0)
        expected_embeddings = probabilities @ candidates
        torch.testing.assert_close(result["match_probabilities"], probabilities)
        torch.testing.assert_close(head.match_embedding_decoder.last_embeddings.reshape(1, 4, 4), expected_embeddings)
        result["coarse_warp"].sum().backward()
        self.assertGreater(float(context.grad.abs().sum()), 0)
        self.assertEqual(head.homography_fitter.calls, 0)

    def test_builder_preserves_original_state_keys_and_rejects_mode_before_loading(self):
        original = SharedFeatureProvider(_descriptor(), _Head())
        with patch("mhinet.v2.feature_provider.build_feature_provider", return_value=(original, {})) as factory:
            model, report = build_v2_shared(SimpleNamespace(device="cpu"))
            self.assertEqual(set(model.state_dict()), set(original.state_dict()))
            self.assertFalse(report["homography_fitter_enabled"])
            self.assertEqual(report["flow_grid_hw"], [32, 32])
            with self.assertRaises(ValueError):
                build_v2_shared(SimpleNamespace(device="cpu"), "invalid")
            self.assertEqual(factory.call_count, 1)


if __name__ == "__main__":
    unittest.main()
