"""CPU numerical gates for global Flow matching and the no-H cascade."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from mhinet.v2.coarse_matcher import FlowPriorD8, geometry_reward, flow_prior_from_shared, match_d8
from mhinet.v2.supervision import direct_d8_fine_grids, to_uv, to_norm
from mhinet.downstream.supervision import fine_grids
from mhinet.downstream.loma_reference.coarse_matcher import CoarseMatchResult
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers
from mhinet.v2.semidense import SemidenseMatcherV2, cap_output_matches
from mhinet.v2.config import SemidenseConfigV2


class FlowMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.a = torch.randn(12, 5, 6)
        self.b = torch.randn_like(self.a)
        uv = to_uv(normalized_cell_centers(5, 6, device='cpu'), (5, 6)).reshape(-1, 2)
        self.prior = FlowPriorD8(uv + .3, torch.rand(30), torch.ones(30, dtype=torch.bool))
        self.options = dict(temperature=.12, threshold=0., max_coarse=30, chunk_rows=7)

    def assert_matches_equal(self, a, b, *, exact=False):
        self.assertTrue(torch.equal(a.source_flat, b.source_flat))
        self.assertTrue(torch.equal(a.target_flat, b.target_flat))
        self.assertTrue(torch.equal(a.prefilter_target_for_source, b.prefilter_target_for_source))
        torch.testing.assert_close(a.confidence, b.confidence,
            rtol=0 if exact else 2e-5, atol=0 if exact else 2e-6)

    def test_full_chunked_flow_mutual_and_topk(self):
        for policy in ('mutual', 'dual_topk'):
            full = match_d8(self.a, self.b, flow_prior=self.prior,
                force_chunked=False, policy=policy, **self.options)
            chunked = match_d8(self.a, self.b, flow_prior=self.prior,
                force_chunked=True, policy=policy, **self.options)
            self.assert_matches_equal(full, chunked)
            self.assertEqual(full.diagnostics['matrix_rows'], 30)
            self.assertEqual(full.diagnostics['matrix_cols'], 30)

    def test_lambda_zero_is_exact_appearance_path(self):
        for chunked in (False, True):
            for policy in ('mutual', 'dual_topk'):
                plain = match_d8(self.a, self.b, force_chunked=chunked, policy=policy, **self.options)
                disabled = match_d8(self.a, self.b, flow_prior=self.prior,
                    flow_lambda=0., force_chunked=chunked, policy=policy, **self.options)
                self.assert_matches_equal(plain, disabled, exact=True)

    def test_invalid_prior_is_finite_appearance_fallback(self):
        prior = FlowPriorD8(torch.full((30, 2), torch.nan),
            torch.full((30,), torch.inf), torch.ones(30, dtype=torch.bool))
        baseline = match_d8(self.a, self.b, **self.options)
        output = match_d8(self.a, self.b, flow_prior=prior, **self.options)
        self.assert_matches_equal(baseline, output, exact=True)
        self.assertTrue(torch.isfinite(output.confidence).all())

    def test_reward_is_bounded_and_invalid_queries_only_cancel_reward(self):
        prior = FlowPriorD8(self.prior.target_uv, self.prior.confidence,
            torch.arange(30) % 2 == 0)
        reward = geometry_reward(prior, torch.arange(30), torch.arange(30),
            width=6, strength=3., sigma=4.)
        self.assertTrue(((reward >= 0) & (reward <= 3)).all())
        self.assertTrue((reward[1::2] == 0).all())
        self.assertGreater(float(reward[::2].sum()), 0)

    def test_empty_support_and_stable_mutual_ties(self):
        mask = torch.zeros(5, 6, dtype=torch.bool)
        empty = match_d8(self.a, self.b, mask, mask, **self.options)
        self.assertEqual(len(empty.source_flat), 0)
        tied = torch.ones(1, 5, 6)  # Exactly representable cosine avoids GEMM roundoff ties.
        for chunked in (False, True):
            result = match_d8(tied, tied, force_chunked=chunked, **self.options)
            self.assertEqual(result.source_flat.tolist(), [0])
            self.assertEqual(result.target_flat.tolist(), [0])

    def shared_flow(self):
        grid = normalized_cell_centers(3, 3, device='cpu').permute(2, 0, 1)[None]
        return dict(flow16_target_norm=grid, flow16_norm=torch.zeros_like(grid),
            flow16_matchability=torch.ones(1, 1, 3, 3),
            flow16_valid=torch.ones(1, 1, 3, 3, dtype=torch.bool))

    def test_identity_displacement_upsampling_is_exact_at_boundaries(self):
        prior = flow_prior_from_shared(self.shared_flow(), 0, (6, 6))
        expected = to_uv(normalized_cell_centers(6, 6, device='cpu'), (6, 6)).reshape(-1, 2)
        torch.testing.assert_close(prior.target_uv, expected, rtol=0, atol=0)
        self.assertTrue(prior.valid.all())

    def test_translation_upsampling_and_invalid_flow(self):
        shared = self.shared_flow()
        shared['flow16_norm'][:, 0] = .1
        shared['flow16_target_norm'] = shared['flow16_target_norm'] + shared['flow16_norm']
        prior = flow_prior_from_shared(shared, 0, (6, 6))
        expected = to_uv(normalized_cell_centers(6, 6, device='cpu'), (6, 6)).reshape(-1, 2)
        torch.testing.assert_close(prior.target_uv-expected,
            torch.tensor([.3, 0.]).expand_as(expected), atol=3e-7, rtol=1e-5)
        shared['flow16_norm'][0, :, 1, 1] = torch.nan
        bad = flow_prior_from_shared(shared, 0, (6, 6))
        self.assertTrue(torch.isfinite(bad.target_uv).all())
        self.assertLess(int(bad.valid.sum()), 36)
        reward = geometry_reward(bad, torch.arange(36), torch.arange(36),
            width=6, strength=3., sigma=4.)
        self.assertTrue((reward[~bad.valid] == 0).all())

    def test_low_probability_and_out_of_bounds_flow_cancel_locally(self):
        shared = self.shared_flow()
        shared['flow16_matchability'][0, 0, 0, 0] = .1
        shared['flow16_target_norm'][0, :, 2, 2] = 5
        prior = flow_prior_from_shared(shared, 0, (6, 6), min_matchability=.5)
        self.assertFalse(prior.valid[0])
        self.assertFalse(prior.valid[-1])
        self.assertTrue(prior.valid[5])

    def test_direct_grid_identity_and_exact_child_indices(self):
        i, j = torch.tensor([0, 11, 63]), torch.tensor([63, 29, 0])
        a, b = direct_d8_fine_grids(i, j, (8, 8), (32, 32))
        old_a, old_b = fine_grids(i, j, torch.eye(3), (8, 8), (32, 32))
        torch.testing.assert_close(a, old_a, atol=0, rtol=0)
        torch.testing.assert_close(b, old_b, atol=1e-7, rtol=1e-6)
        uv = to_uv(a, (32, 32))
        self.assertTrue(torch.equal(uv[0, :, 0], torch.tensor([0., 1., 2., 3.]).repeat(4)))
        self.assertTrue(((to_uv(b, (32, 32)) >= 0) & (to_uv(b, (32, 32)) <= 31)).all())
        empty = direct_d8_fine_grids(i[:0], j[:0], (8, 8), (32, 32))
        self.assertEqual(empty[0].shape, (0, 16, 2))

    def test_stable_final_cap_keeps_fields_aligned(self):
        for count in (0, 2, 3, 6):
            points = torch.arange(count*2).reshape(count, 2).float()
            confidence = torch.ones(count)
            output = cap_output_matches(dict(points_a=points, points_b=points+100,
                confidence=confidence, fine_points_b=points+200,
                lineage=torch.arange(count)[:, None],
                diagnostics={'precap_fine_all_a': points}), 3)
            self.assertEqual(output['confidence'].numel(), min(3, count))
            torch.testing.assert_close(output['points_a'], points[:3])
            torch.testing.assert_close(output['points_b'], points[:3]+100)
            torch.testing.assert_close(output['fine_points_b'], points[:3]+200)
            torch.testing.assert_close(output['lineage'], torch.arange(count)[:3, None])
            self.assertEqual(len(output['diagnostics']['precap_fine_all_a']), count)
        output = cap_output_matches(dict(points_a=torch.arange(8).reshape(4, 2),
            points_b=torch.arange(8).reshape(4, 2), confidence=torch.tensor([.2, .8, .8, .4])), 2)
        self.assertEqual(output['points_a'][:, 0].tolist(), [2, 4])
        for cap in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                cap_output_matches(dict(confidence=torch.empty(0)), cap)

    def test_no_h_training_has_fine_supervision_and_gradients(self):
        config = SemidenseConfigV2(geometry_mode='none', flow_guidance_enabled=False,
            coarse_queries=4, fine_windows=2, qrru_queries=2, window_chunk=2)
        matcher = SemidenseMatcherV2(config, channels=8)
        d8 = torch.randn(1, 2, 8, 4, 4, requires_grad=True)
        d2 = torch.randn(1, 2, 8, 16, 16, requires_grad=True)
        loss, records = matcher.training_losses({'pyramid': {8: d8, 2: d2}},
            torch.eye(3)[None], torch.ones(1, 1, 16, 16), torch.ones(1, 1, 16, 16))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(records[0]['fine_positive'], 0)
        self.assertGreater(float(d8.grad.abs().sum()), 0)
        self.assertGreater(float(d2.grad.abs().sum()), 0)
        self.assertEqual(records[0]['fine_prior'], 'direct_d8')

    def test_no_h_inference_and_refine_false_skip_qrru(self):
        config = SemidenseConfigV2(geometry_mode='none', flow_guidance_enabled=False,
            max_output_matches=3, max_matches=12)
        matcher = SemidenseMatcherV2(config, channels=8)
        # Registered spatial contract, small channels for a CPU integration gate.
        d8 = torch.randn(1, 2, 8, 64, 64)
        d2 = torch.randn(1, 2, 8, 256, 256)
        coarse = CoarseMatchResult(torch.tensor([100]), torch.tensor([101]), torch.ones(1),
            torch.arange(4096), torch.arange(4096), torch.arange(4096), {})
        with patch('mhinet.v2.semidense.match_d8', return_value=coarse), \
             patch.object(matcher.qrru, 'forward', side_effect=AssertionError('QRRU must be skipped')):
            result = matcher.infer({'pyramid': {8: d8, 2: d2}}, refine=False, diagnostics=True)[0]
        self.assertLessEqual(len(result['confidence']), 3)
        self.assertGreaterEqual(len(result['fine_all_confidence']), len(result['confidence']))
        self.assertEqual(len(result['fine_all_a']), len(result['fine_all_confidence']))
        with patch('mhinet.v2.semidense.match_d8', return_value=coarse):
            refined = matcher.infer({'pyramid': {8: d8, 2: d2}}, diagnostics=True)[0]
        self.assertLessEqual(len(refined['confidence']), 3)
        self.assertEqual(len(refined['lineage']), len(refined['confidence']))
        self.assertEqual(len(refined['fine_points_b']), len(refined['confidence']))


if __name__ == '__main__':
    unittest.main()
