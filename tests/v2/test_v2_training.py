"""CPU acceptance for v2 contracts, H-free Flow0 labels, and stage ownership."""
from dataclasses import asdict
from types import SimpleNamespace
import math
import tempfile
import unittest

import torch
from torch import nn

from mhinet.v2.config import SemidenseConfigV2, variant_config
from mhinet.v2.losses import flow_head_loss
from mhinet.v2.qrru import QRRUV2
from mhinet.v2.training import (SemidenseSystemV2, assert_frozen_gradients,
                                mixed_qrru_loss)


class _HeadOnlyStage(dict):
    """Fail if Flow0 supervision attempts to use fitted-H status or values."""
    allowed = {'coarse_warp', 'coarse_matchability', 'match_probabilities'}

    def __getitem__(self, name):
        if name not in self.allowed:
            raise AssertionError(f'Flow0 supervision read forbidden field: {name}')
        return super().__getitem__(name)

    def get(self, name, default=None):
        return self[name] if name in self else default


def _flow_inputs(side=4, batch=1, *, offset=0.):
    axis = (torch.arange(side)+.5)*2/side-1
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    warp = torch.stack((x, y))[None].repeat(batch, 1, 1, 1)
    warp = (warp+offset).requires_grad_()
    mat_logits = torch.randn(batch, 1, side, side, requires_grad=True)
    class_logits = torch.randn(batch, side*side, side*side+1, requires_grad=True)
    stage = _HeadOnlyStage(coarse_warp=warp, coarse_matchability=mat_logits.sigmoid(),
                           match_probabilities=class_logits.softmax(-1))
    return stage, (warp, mat_logits, class_logits)


class _FakeShared(nn.Module):
    """Small shared provider honoring the production training-group contract."""
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 3)
        self.descriptor = nn.Linear(3, 3)
        self.dino = nn.Linear(3, 3)
        self.frozen_bn = nn.BatchNorm1d(3)
        self.group_calls = []
        self.forward_scales = []

    def set_training_groups(self, **flags):
        self.group_calls.append(flags)
        self.requires_grad_(False)
        self.head.requires_grad_(flags['ghim_head'])

    def train(self, mode=True):
        super().train(mode)
        self.descriptor.eval()
        self.dino.eval()
        self.frozen_bn.eval()
        self.head.train(mode and any(p.requires_grad for p in self.head.parameters()))
        return self

    def forward(self, images, *, pyramid_scales=None):
        self.forward_scales.append(pyramid_scales)
        x = images.mean((1, 3, 4))
        descriptor = self.descriptor(self.frozen_bn(self.dino(x)))
        return {'head': self.head(descriptor), 'descriptors': descriptor}


class _FakeMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.qrru = nn.Linear(3, 2)
        self.coarse = nn.Linear(3, 3)


class _PredictedMatcher(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.config = SemidenseConfigV2(qrru_queries=8, window_chunk=3)
        self.qrru = QRRUV2(channels=8, use_discrepancy=True, use_dgfo=True)
        self.prediction = prediction
        self.infer_calls = 0

    def infer(self, shared, *, refine):
        if refine is not False:
            raise AssertionError('Prediction cache must precede QRRU')
        if any(key in shared for key in ('H_gt_norm', 'mask_A_overlap', 'mask_B_overlap')):
            raise AssertionError('GT leaked into inference')
        self.infer_calls += 1
        return [self.prediction]


class V2ContractTests(unittest.TestCase):
    def test_campaign_cannot_advance_before_final_validation_is_saved(self):
        from mhinet.v2.train import final_run_status
        self.assertEqual(final_run_status(10, 10, 9, True), "interrupted")
        self.assertEqual(final_run_status(10, 10, 9, False), "interrupted")
        self.assertEqual(final_run_status(10, 10, 10, True), "completed")
        self.assertEqual(final_run_status(9, 10, 9, True), "interrupted")
        self.assertEqual(final_run_status(2, 10, 2, False), "diagnostic_stop")

    def test_defaults_and_registered_variants_keep_independent_budgets(self):
        config = SemidenseConfigV2()
        self.assertEqual(config.input_size, 512)
        self.assertEqual(config.max_output_matches, 2048)
        self.assertEqual(config.max_matches, 12000)
        self.assertEqual(SemidenseConfigV2.from_dict(asdict(config)), config)
        for name in 'ABCDEF':
            variant = variant_config(name)
            self.assertEqual((variant.input_size, variant.max_output_matches, variant.max_matches),
                             (512, 2048, 12000))
        self.assertEqual(variant_config('E').use_dgfo, True)
        self.assertEqual(variant_config('F').flow_guidance_enabled, False)

    def test_illegal_types_unknown_fields_and_non512_rejected(self):
        for field, values in (
            ('input_size', [784, 512., True, '512']),
            ('max_output_matches', [0, -1, True, 2048., '2048']),
            ('max_matches', [0, True, 12000.]),
            ('iterations', [0, True, 4.]),
            ('coarse_queries', [0, True, 1.5]),
            ('flow_guidance_enabled', [0, 1, 'true']),
            ('use_discrepancy', [0, 1, 'false']),
            ('flow_lambda', [-1, True, float('nan'), float('inf'), '3']),
            ('flow_sigma_d8', [0, True, float('nan')]),
            ('flow_min_matchability', [-.1, 1.1, True]),
            ('dgfo_radius', [0, 2, True, 1.]),
            ('dgfo_dim', [16, 0, True, 32.]),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        SemidenseConfigV2.from_dict({field: value})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            SemidenseConfigV2.from_dict({'unknown_training_knob': 1})
        with self.assertRaises(TypeError):
            SemidenseConfigV2.from_dict([])
        with self.assertRaises(ValueError):
            SemidenseConfigV2(use_dgfo=True)


class V2FlowLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(52)

    def test_identity_zero_epe_and_no_h_or_fit_status_access(self):
        stage, leaves = _flow_inputs()
        # These poison fields exist solely to catch any attempt to gate labels
        # with predicted-H fit status. The loss must use its three head fields.
        stage['H0_norm'] = torch.full((1, 3, 3), float('nan'))
        stage['fit_succeeded'] = torch.tensor([False])
        loss, metrics = flow_head_loss(stage, torch.eye(3)[None], torch.ones(1, 1, 16, 16),
                                      torch.ones(1, 1, 16, 16))
        self.assertEqual(metrics['flow_epe_px'], 0.)
        self.assertEqual(metrics['geo'], 0.)
        self.assertEqual(metrics['valid_gt_points'], 16)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for value in leaves:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(leaves[1].grad.abs().sum()), 0)
        self.assertGreater(float(leaves[2].grad.abs().sum()), 0)

    def test_shifted_warp_has_finite_geometric_gradient_without_predicted_h(self):
        stage, leaves = _flow_inputs(batch=2, offset=.05)
        loss, metrics = flow_head_loss(stage, torch.eye(3).repeat(2, 1, 1),
                                      torch.ones(2, 1, 16, 16))
        self.assertAlmostEqual(metrics['flow_epe_px'], math.sqrt(2)*.05*256, places=4)
        loss.backward()
        for value in leaves:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_nonmatch_labels_train_bin_and_matchability_without_geometry(self):
        stage, leaves = _flow_inputs()
        loss, metrics = flow_head_loss(stage, torch.eye(3)[None], torch.zeros(1, 1, 16, 16))
        probability = stage['coarse_matchability'].detach()
        classes = stage['match_probabilities'].detach()
        self.assertEqual(metrics['valid_gt_points'], 0)
        self.assertEqual(metrics['geo'], 0.)
        self.assertIsNone(metrics['flow_epe_px'])
        self.assertAlmostEqual(metrics['cls'], float(-classes[..., -1].log().mean()), places=6)
        self.assertAlmostEqual(metrics['mat'], float(-.5*torch.log1p(-probability).mean()), places=6)
        loss.backward()
        self.assertEqual(float(leaves[0].grad.abs().sum()), 0)
        self.assertTrue((leaves[1].grad > 0).all())
        self.assertTrue((leaves[2].grad[..., -1] < 0).all())
        self.assertTrue((leaves[2].grad[..., :-1] > 0).all())

    def test_target_visibility_changes_labels_and_nonfinite_predictions_rejected(self):
        stage, _ = _flow_inputs()
        _, metrics = flow_head_loss(stage, torch.eye(3)[None], torch.ones(1, 1, 16, 16),
                                    torch.zeros(1, 1, 16, 16))
        self.assertEqual(metrics['valid_gt_points'], 0)
        bad = _HeadOnlyStage(stage)
        bad['coarse_warp'] = torch.full((1, 2, 4, 4), float('nan'))
        with self.assertRaises(FloatingPointError):
            flow_head_loss(bad, torch.eye(3)[None], torch.ones(1, 1, 16, 16))

    def test_source_visibility_samples_d16_centers_instead_of_nearest_resize(self):
        stage, leaves = _flow_inputs(side=32)
        source_mask = torch.zeros(1, 1, 512, 512)
        source_mask[..., 20:] = 1
        loss, metrics = flow_head_loss(stage, torch.eye(3)[None], source_mask,
                                      torch.ones(1, 1, 512, 512))
        # D16 x centers 7.5,23.5,... retain 31 columns. Nearest resize
        # incorrectly sampled 0,16,..., retaining only 30 columns (960).
        self.assertEqual(metrics['valid_gt_points'], 31*32)
        self.assertEqual(metrics['flow_epe_px'], 0.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for value in leaves:
            self.assertTrue(torch.isfinite(value.grad).all())


class V2TrainingOwnershipTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(53)

    def test_flow_head_only_parameters_optimizer_and_frozen_buffers(self):
        shared, matcher = _FakeShared(), _FakeMatcher()
        system = SemidenseSystemV2(shared, matcher, stage='flow_head')
        self.assertEqual(set(system.training_parameter_names()), {'shared.head.weight', 'shared.head.bias'})
        self.assertEqual(shared.group_calls[-1], dict(mvt=False, vgg=False, dedode=False, ghim_head=True))
        groups = system.optimizer_groups(.001)
        self.assertEqual(groups[0]['name'], 'flow_head')
        self.assertEqual({id(p) for p in groups[0]['params']}, {id(p) for p in shared.head.parameters()})
        frozen_before = shared.frozen_bn.running_mean.clone()
        result = system(torch.randn(1, 2, 3, 512, 512))
        result['head'].square().mean().backward()
        self.assertEqual(shared.forward_scales[-1], ())
        self.assertTrue(shared.head.training)
        self.assertFalse(shared.descriptor.training)
        self.assertFalse(matcher.training)
        assert_frozen_gradients(system)
        self.assertTrue(all(p.grad is not None for p in shared.head.parameters()))
        torch.testing.assert_close(shared.frozen_bn.running_mean, frozen_before, rtol=0, atol=0)
        system.eval()
        self.assertFalse(shared.head.training)
        system.train()
        self.assertTrue(shared.head.training)

    def test_qrru_only_and_512_input_shape_contract(self):
        shared, matcher = _FakeShared(), _FakeMatcher()
        system = SemidenseSystemV2(shared, matcher, stage='qrru')
        self.assertEqual(set(system.training_parameter_names()), {'matcher.qrru.weight', 'matcher.qrru.bias'})
        result = system(torch.randn(1, 2, 3, 512, 512))
        self.assertFalse(result['descriptors'].requires_grad)
        self.assertFalse(shared.training)
        self.assertTrue(matcher.training)
        self.assertEqual(shared.forward_scales[-1], None)
        matcher.qrru(result['descriptors']).square().mean().backward()
        assert_frozen_gradients(system)
        self.assertTrue(all(p.grad is not None for p in matcher.qrru.parameters()))
        groups = system.optimizer_groups(.002)
        self.assertEqual({id(p) for p in groups[0]['params']}, {id(p) for p in matcher.qrru.parameters()})
        for shape in ((1, 2, 3, 784, 784), (1, 3, 3, 512, 512), (1, 2, 3, 512)):
            with self.assertRaises(ValueError):
                system(torch.empty(shape))
        with self.assertRaises(ValueError):
            system.configure_stage('joint')


class V2MixedQRRUTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(54)

    def _fixture(self, points_a=None, points_b=None, *, visible=True):
        if points_a is None:
            points_a = torch.empty(0, 2)
            points_b = torch.empty(0, 2)
        prediction = dict(fine_all_a=(points_a+.5)*2-.5, fine_all_b=(points_b+.5)*2-.5,
                          fine_all_confidence=torch.linspace(1., .1, len(points_a)))
        matcher = _PredictedMatcher(prediction)
        system = SimpleNamespace(matcher=matcher)
        shared = {'pyramid': {2: torch.randn(1, 2, 8, 20, 20)}}
        tensors = dict(H_gt_norm=torch.eye(3)[None], mask_A_overlap=torch.full((1, 1, 20, 20), float(visible)),
                       mask_B_overlap=torch.full((1, 1, 20, 20), float(visible)))
        return system, shared, tensors

    def test_empty_predictions_fall_back_to_noise_with_finite_backward(self):
        system, shared, tensors = self._fixture()
        loss, metrics = mixed_qrru_loss(system, shared, tensors, 'empty-prediction')
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['predicted_queries'], 0)
        self.assertEqual(metrics['noise_queries'], 8)
        loss.backward()
        self.assertTrue(torch.isfinite(system.matcher.qrru.head[-1].weight.grad).all())

    def test_invalid_wrong_predictions_filtered_and_cache_reused(self):
        a = torch.tensor([[8., 8.], [10., 10.], [float('nan'), 5.], [6., 6.], [4., 4.]])
        b = torch.tensor([[8.5, 8.], [100., 100.], [5., 5.], [float('inf'), 5.], [float('nan'), 4.]])
        system, shared, tensors = self._fixture(a, b)
        with tempfile.TemporaryDirectory() as cache:
            for _ in range(2):
                loss, metrics = mixed_qrru_loss(system, shared, tensors, 'bad-prediction', cache_root=cache,
                                                source_identity='controlled-source')
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(metrics['predicted_queries'], 1)
                self.assertEqual(metrics['noise_queries'], 7)
                self.assertTrue(math.isfinite(metrics['initial_epe_d2']))
                loss.backward()
                system.matcher.qrru.zero_grad(set_to_none=True)
            self.assertEqual(system.matcher.infer_calls, 1)

    def test_empty_visible_gt_raises_explicitly(self):
        system, shared, tensors = self._fixture(visible=False)
        with self.assertRaisesRegex(ValueError, 'No usable QRRU'):
            mixed_qrru_loss(system, shared, tensors, 'empty-gt')

    def test_invalid_input_mixture_correction_range_and_unidentified_cache_rejected(self):
        system, shared, tensors = self._fixture()
        for fraction in (-.01, 1.01, True, '0.5', float('nan'), float('inf')):
            with self.subTest(predicted_fraction=fraction):
                with self.assertRaisesRegex(ValueError, 'mixture/correction'):
                    mixed_qrru_loss(system, shared, tensors, 'invalid-mixture', predicted_fraction=fraction)
        for limit in (0, -.01, 6.01, True, '6', float('nan'), float('inf')):
            with self.subTest(correction_limit_d2=limit):
                with self.assertRaisesRegex(ValueError, 'mixture/correction'):
                    mixed_qrru_loss(system, shared, tensors, 'invalid-correction', correction_limit_d2=limit)
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaisesRegex(ValueError, 'source identity'):
                mixed_qrru_loss(system, shared, tensors, 'unidentified-cache', cache_root=cache,
                                source_identity='')
        # Reject invalid runtime configuration before producing/caching any D2 predictions.
        self.assertEqual(system.matcher.infer_calls, 0)


if __name__ == '__main__':
    unittest.main()
