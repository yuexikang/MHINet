"""CPU acceptance for the isolated, pixel-scaled Flow loss pilot."""
import math
import unittest

import torch

from mhinet.v2.losses import flow_head_loss
from mhinet.v2.flow_loss.objective import (
    flow_loss_components, flow_objective, flow_point_diagnostics, flow_supervision,
)


def fixture(batch=1, side=4, offsets=None):
    axis = (torch.arange(side)+.5)*2/side-1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    warp = torch.stack((x, y))[None].repeat(batch, 1, 1, 1)
    if offsets is not None:
        warp = warp+torch.tensor(offsets)[:, :, None, None]
    warp.requires_grad_()
    mat_logits = torch.randn(batch, 1, side, side, requires_grad=True)
    class_logits = torch.randn(batch, side*side, side*side+1, requires_grad=True)
    stage = dict(coarse_warp=warp, coarse_matchability=mat_logits.sigmoid(),
                 match_probabilities=class_logits.softmax(-1))
    H = torch.eye(3).repeat(batch, 1, 1)
    mask = torch.ones(batch, 1, 512, 512)
    return stage, (warp, mat_logits, class_logits), H, mask


def slice_stage(stage, index):
    return {key: value[index:index+1] for key, value in stage.items()}


class ObjectiveAcceptanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261020)

    def test_l0_value_and_all_leaf_gradients_are_bitwise_the_original(self):
        stage, leaves, H, mask = fixture(offsets=[(.003, -.007)])
        mask[:, :, :, 256:] = 0
        original, original_metrics = flow_head_loss(stage, H, mask, torch.ones_like(mask))
        actual, metrics = flow_objective(stage, H, mask, torch.ones_like(mask), arm="L0")
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
        expected_gradients = torch.autograd.grad(original, leaves, retain_graph=True)
        actual_gradients = torch.autograd.grad(actual, leaves)
        for expected, measured in zip(expected_gradients, actual_gradients):
            torch.testing.assert_close(measured, expected, atol=0, rtol=0)
        for key, expected in original_metrics.items():
            self.assertEqual(metrics[key], expected)

    def test_live_old_components_recompose_original_value_and_gradients(self):
        stage, leaves, H, mask = fixture(offsets=[(.014, -.002)])
        mask[:, :, :, 256:] = 0
        terms = flow_loss_components(stage, H, mask, torch.ones_like(mask))
        reconstructed = terms["geo_old"]+.01*terms["mat"]+.0001*terms["cls"]
        torch.testing.assert_close(reconstructed, terms["old_total"], atol=1e-8, rtol=1e-6)
        original_gradients = torch.autograd.grad(terms["old_total"], leaves, retain_graph=True)
        reconstructed_gradients = torch.autograd.grad(reconstructed, leaves)
        for original, reconstructed in zip(original_gradients, reconstructed_gradients):
            torch.testing.assert_close(reconstructed, original, atol=1e-8, rtol=1e-6)
        self.assertTrue(all(terms[name].requires_grad for name in ("geo_old", "geo_new", "mat", "cls")))

    def test_new_radial_error_uses_pixel_units_and_half_pixel_epsilon(self):
        stage, leaves, H, mask = fixture(offsets=[(3/256, 4/256)])
        terms = flow_loss_components(stage, H, mask, mask)
        self.assertAlmostEqual(float(terms["geo_new"].detach()), math.sqrt(25+.25)-.5, delta=1e-6)
        diagnostics = terms["point_diagnostics"]
        torch.testing.assert_close(diagnostics["epe_px"], torch.full((1, 16), 5.), atol=0, rtol=0)
        alpha = .003
        actual, metrics = flow_objective(stage, H, mask, mask, arm="L1", alpha=alpha)
        expected = alpha*terms["geo_new"]+.01*terms["mat"]+.0001*terms["cls"]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(metrics["alpha"], alpha)
        self.assertEqual(metrics["arm"], "L1")
        actual.backward()
        self.assertTrue(all(bool(torch.isfinite(leaf.grad).all()) for leaf in leaves))

    def test_identity_new_geometry_is_exact_zero_with_finite_zero_gradient(self):
        stage, leaves, H, mask = fixture()
        terms = flow_loss_components(stage, H, mask, mask)
        self.assertEqual(float(terms["geo_new"].detach()), 0.)
        gradient, = torch.autograd.grad(terms["geo_new"], leaves[0])
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)

    def test_batch_macro_matches_separate_pairs_with_unequal_valid_counts(self):
        stage, leaves, H, masks = fixture(batch=2, offsets=[(.1, .1), (.001, .001)])
        masks[0].zero_()
        masks[0, :, :128, :128] = 1  # One valid source center, vs sixteen in pair 2.
        target_masks = torch.ones_like(masks)
        joint = flow_loss_components(stage, H, masks, target_masks)
        singles = [flow_loss_components(slice_stage(stage, b), H[b:b+1], masks[b:b+1],
                                       target_masks[b:b+1]) for b in range(2)]
        for name in ("geo_old", "geo_new", "mat", "cls", "old_total"):
            expected = torch.stack([part[name] for part in singles]).mean()
            torch.testing.assert_close(joint[name], expected, atol=0, rtol=0)
        original_global, _ = flow_head_loss(stage, H, masks, target_masks)
        self.assertGreater(float((joint["old_total"]-original_global).abs().detach()), .1)
        self.assertEqual(joint["metrics"]["valid_gt_points"], 8.5)
        expected_gradients = torch.autograd.grad((singles[0]["old_total"]+singles[1]["old_total"])/2,
                                                 leaves, retain_graph=True)
        actual_gradients = torch.autograd.grad(joint["old_total"], leaves)
        for expected, actual in zip(expected_gradients, actual_gradients):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_empty_gt_and_illegal_projection_geometry_are_finite_connected_zero(self):
        for failure in ("mask", "zero_denominator", "nonfinite_H"):
            with self.subTest(failure=failure):
                stage, leaves, H, mask = fixture(offsets=[(.02, -.02)])
                if failure == "mask":
                    mask.zero_()
                elif failure == "zero_denominator":
                    H[:, 2].zero_()
                else:
                    H.fill_(float("nan"))
                terms = flow_loss_components(stage, H, mask, torch.ones_like(mask))
                self.assertEqual(terms["metrics"]["valid_gt_points"], 0.)
                self.assertIsNone(terms["metrics"]["flow_epe_px"])
                for name in ("geo_old", "geo_new"):
                    self.assertEqual(float(terms[name].detach()), 0.)
                    gradient, = torch.autograd.grad(terms[name], leaves[0], retain_graph=True)
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                    self.assertEqual(float(gradient.abs().sum()), 0.)
                self.assertEqual(float(terms["point_diagnostics"]["residual_px"].detach().abs().sum()), 0.)
                self.assertTrue(bool(torch.isfinite(terms["point_diagnostics"]["epe_px"]).all()))
                new_loss, _ = flow_objective(stage, H, mask, torch.ones_like(mask), arm="L1", alpha=.001)
                new_loss.backward()
                self.assertTrue(all(bool(torch.isfinite(leaf.grad).all()) for leaf in leaves))

    def test_source_mask_samples_real_centers_and_target_mask_controls_same_gt_set(self):
        H = torch.eye(3)[None]
        mask = torch.zeros(1, 1, 512, 512)
        mask[..., 20:] = 1
        target, valid = flow_supervision(H, mask, torch.ones_like(mask), hw=(32, 32))
        self.assertEqual(int(valid.sum()), 992)  # x=23.5 valid; x=7.5 invalid.
        self.assertAlmostEqual(float(target[0, 1, 0]), 23.5/256+.5/256-1)
        _, invisible = flow_supervision(H, mask, torch.zeros_like(mask), hw=(32, 32))
        self.assertFalse(bool(invisible.any()))

    def test_diagnostics_separate_center_box_and_deployed_prior_validity(self):
        stage, leaves, H, mask = fixture()
        stage["coarse_warp"] = torch.full_like(stage["coarse_warp"], .99, requires_grad=True)
        diagnostics = flow_point_diagnostics(stage, H, mask, mask)
        self.assertFalse(bool(diagnostics["predicted_valid"].any()))
        self.assertTrue(bool(diagnostics["deployed_valid"].all()))
        self.assertTrue(bool(diagnostics["gt_valid"].all()))

    def test_invalid_modes_scalars_shapes_and_nonfinite_predictions_are_rejected(self):
        stage, leaves, H, mask = fixture()
        for kwargs in ({"arm": "other"}, {"alpha": 0.}, {"alpha": float("nan")},
                       {"smooth_eps_px": 0.}, {"input_size": 784}, {"mat_weight": -1.}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    flow_objective(stage, H, mask, mask, **kwargs)
        invalid = {**stage, "coarse_warp": torch.full_like(stage["coarse_warp"], float("nan"))}
        with self.assertRaises(FloatingPointError):
            flow_objective(invalid, H, mask, mask)


if __name__ == "__main__":
    unittest.main()
