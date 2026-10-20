"""CPU gates for point-level pilot diagnostics and GT-free Head inference."""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mhinet.config import sha256_file
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers
from mhinet.v2.flow_loss.evaluate import (SOURCE_SHA256, assert_frozen_states_equal,
    evaluate_flow, paired_group_bootstrap, point_statistics, read_hard_cases,
    reserve_output, _boundary_distances)


class FlowLossEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fixed_gt_denominator_and_confidence_coverage(self):
        result = point_statistics([1, 10, 40, 99], [.8, .6, .1, .9],
            [True, True, True, False], [True, False, True, True])
        self.assertEqual(result['all_gt_matchable']['points'], 3)
        self.assertEqual(result['all_gt_matchable']['epe_px'], 17.)
        self.assertAlmostEqual(result['confidence_strata']['0.5']['coverage_of_gt'], 2/3)
        self.assertEqual(result['confidence_strata']['0.5']['epe_px'], 5.5)
        self.assertAlmostEqual(result['deployed']['coverage_of_gt'], 2/3)
        self.assertEqual(result['deployed_confidence_strata']['0.5']['points'], 1)
        self.assertAlmostEqual(result['confidence_weighted_tail_fraction']['32.0'], .1/1.5)
        self.assertEqual(result['all_gt_matchable']['tail_fraction']['8.0'], 2/3)

    def test_empty_gt_and_nonfinite_gt_prediction(self):
        empty = point_statistics([np.nan], [np.nan], [False])
        self.assertIsNone(empty['all_gt_matchable']['epe_px'])
        self.assertEqual(empty['all_gt_matchable']['coverage_of_gt'], 0)
        with self.assertRaises(FloatingPointError):
            point_statistics([np.nan], [.8], [True])

    def test_frozen_state_audit_only_allows_head_changes(self):
        a = {'shared.stage1_head.weight': torch.ones(2), 'shared.descriptor.weight': torch.ones(2), 'matcher.qrru.weight': torch.ones(2)}
        b = {key: value.clone() for key, value in a.items()}
        b['shared.stage1_head.weight'].add_(1)
        self.assertEqual(assert_frozen_states_equal(a, b)['checked_frozen_tensors'], 2)
        b['matcher.qrru.weight'].add_(1)
        with self.assertRaisesRegex(ValueError, 'Non-head'):
            assert_frozen_states_equal(a, b)

    def test_boundary_distances_include_label_box_and_overlap_mask(self):
        points = torch.tensor([[0., 0.], [-1+1/32, -1+1/32]])
        mask = torch.ones(1, 512, 512)
        result = _boundary_distances(points.numpy(), mask, (32, 32))
        self.assertEqual(result['gt_target_label_box_boundary_distance_px'][1], 0.)
        self.assertEqual(result['gt_target_image_boundary_distance_px'][1], 7.5)
        self.assertTrue(np.isfinite(result['gt_target_overlap_boundary_distance_px']).all())

    def test_hard_registry_and_exclusive_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root/'hard.json'
            registry.write_text(json.dumps({'pair_ids': [f'pair-{i}' for i in range(52)]}))
            self.assertEqual(len(read_hard_cases(registry)), 52)
            registry.write_text(json.dumps({'pair_ids': ['duplicate']*52}))
            with self.assertRaises(ValueError):
                read_hard_cases(registry)
            reserve_output(root/'new')
            with self.assertRaises(FileExistsError):
                reserve_output(root/'new')

    def test_group_bootstrap_uses_source_groups_and_handles_missing(self):
        rows = [dict(source_dataset='a', parent_group='shared'),
                dict(source_dataset='a', parent_group='shared'),
                dict(source_dataset='b', parent_group='other')]
        result = paired_group_bootstrap(rows, [-1., -1., -1.], iterations=20)
        self.assertEqual(result['mean_delta'], -1)
        self.assertEqual(result['source_image_group_bootstrap_95ci'], [-1, -1])
        missing = paired_group_bootstrap(rows, [None, None, None], iterations=20)
        self.assertIsNone(missing['mean_delta'])
        self.assertIsNone(missing['source_image_group_bootstrap_95ci'])

    def test_head_only_smoke_uses_images_before_post_prediction_gt(self):
        class Dataset:
            def __init__(self, path, image_size):
                self.manifest = path
            def __len__(self):
                return 5106
            def __getitem__(self, index):
                return dict(images=torch.zeros(2, 3, 512, 512),
                    H_gt_norm=torch.eye(3), mask_A_overlap=torch.ones(1, 512, 512),
                    mask_B_overlap=torch.ones(1, 512, 512), pair_id=f'pair-{index}', parent_group='fixed-parent')
        calls = []
        def head(images, *, pyramid_scales):
            self.assertEqual(pyramid_scales, ())
            self.assertEqual(images.shape, (1, 2, 3, 512, 512))
            calls.append(True)
            target = normalized_cell_centers(32, 32, device='cpu').permute(2, 0, 1)[None]
            confidence = torch.ones(1, 1, 32, 32)
            return dict(pyramid={}, H0_norm=None, call_counts={'homography_fitter': 0},
                stage1=dict(coarse_warp=target, coarse_matchability=confidence),
                flow16_target_norm=target, flow16_norm=torch.zeros_like(target),
                flow16_matchability=confidence, flow16_valid=torch.ones_like(confidence, dtype=torch.bool))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'pairs.jsonl'
            manifest.write_text(''.join(json.dumps(dict(pair_id=f'pair-{i}', source_dataset='test',
                size_A=[512,512], size_B=[512,512]))+'\n' for i in range(52)))
            registry = root/'hard.json'
            registry.write_text(json.dumps({'pair_ids': [f'pair-{i}' for i in range(52)]}))
            args = Namespace(manifest=manifest, hard_cases=registry, limit=1, workers=0, device='cpu',
                checkpoint=root/'unused.pt', expected_source_sha256=SOURCE_SHA256,
                expected_manifest_sha256=sha256_file(manifest), output=root/'evaluation', runtime=None)
            registration = dict(checkpoint='unused', checkpoint_sha256='fake',
                source_sha256=SOURCE_SHA256, matcher={}, config={}, runtime='unused')
            with patch('mhinet.v2.flow_loss.evaluate.RGB512Dataset', Dataset), \
                 patch('mhinet.v2.flow_loss.evaluate.build_system', return_value=(SimpleNamespace(shared=head),registration,{})), \
                 patch('mhinet.v2.flow_loss.evaluate._code_identity', return_value={'code_sha256':'test'}), \
                 patch('builtins.print'):
                self.assertEqual(evaluate_flow(args), 0)
            self.assertEqual(len(calls), 1)
            summary = json.loads((root/'evaluation/summary.json').read_text())
            self.assertEqual(summary['macro_flow_epe_px'], 0.)
            self.assertEqual(summary['hard_cases_evaluated'], 1)
            self.assertEqual(summary['point_level']['d8']['deployed']['coverage_of_gt'], 1.)
            with np.load(root/'evaluation/hard_cases/000.npz', allow_pickle=False) as points:
                self.assertEqual(points['d16_epe_px'].shape, (1024,))
                self.assertEqual(points['d8_epe_px'].shape, (4096,))
                self.assertIn('d8_gt_target_overlap_boundary_distance_px', points.files)


if __name__ == '__main__':
    unittest.main()
