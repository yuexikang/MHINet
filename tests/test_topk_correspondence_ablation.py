"""CPU tests of stable selection, exact Full reproduction, and metric semantics."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import evaluate_topk_correspondence_ablation as runner


class SelectionTests(unittest.TestCase):
    def test_stable_ties_original_order_and_nested_prefixes(self):
        confidence = np.full(5000, .5, dtype=np.float32)
        confidence[[8, 3, 4100]] = .9
        original = confidence.copy()
        ranked, selections = runner.stable_selections(confidence)
        np.testing.assert_array_equal(ranked[:3], [3, 8, 4100])
        np.testing.assert_array_equal(ranked[3:10], [0, 1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(selections['Full'], np.arange(5000))
        np.testing.assert_array_equal(selections['Top-1024'], selections['Top-2048'][:1024])
        np.testing.assert_array_equal(selections['Top-2048'], selections['Top-4096'][:2048])
        np.testing.assert_array_equal(confidence, original)

    def test_under_k_keeps_every_original_match_without_padding(self):
        confidence = np.linspace(1., 0., 3619, dtype=np.float32)
        _, selections = runner.stable_selections(confidence)
        self.assertEqual(len(selections['Top-4096']), 3619)
        np.testing.assert_array_equal(selections['Top-4096'], selections['Full'])
        self.assertEqual(len(selections['Top-2048']), 2048)
        self.assertEqual(len(selections['Top-1024']), 1024)

    def test_empty_confidence_is_valid_and_all_selections_empty(self):
        ranked, selections = runner.stable_selections(np.empty(0, dtype=np.float32))
        self.assertEqual(len(ranked), 0)
        self.assertTrue(all(value.dtype == np.int64 and len(value) == 0 for value in selections.values()))

    def test_invalid_confidence_is_rejected_instead_of_filtered(self):
        for confidence in (np.array([.1, np.nan]), np.array([np.inf]), np.ones((2, 1)), np.array([1, 2])):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    runner.stable_selections(confidence)

    def test_original_indices_survive_npz_without_clipping_edge_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pair.npz'
            a = np.array([[-.5, 10.], [511.49, 20.]], dtype=np.float32)
            b = a + np.array([0., 1.], dtype=np.float32)
            confidence = np.array([.2, .8], dtype=np.float32)
            np.savez_compressed(path, pair_id=np.asarray('sample'), points_a=a, points_b=b, confidence=confidence)
            loaded_a, loaded_b, loaded_confidence = runner.load_source_matches(path, 'sample')
            np.testing.assert_array_equal(loaded_a, a)
            np.testing.assert_array_equal(loaded_b, b)
            np.testing.assert_array_equal(loaded_confidence, confidence)
            with self.assertRaisesRegex(ValueError, 'pair_id'):
                runner.load_source_matches(path, 'wrong')


class MetricTests(unittest.TestCase):
    def grid(self):
        x, y = np.meshgrid(np.linspace(20., 450., 8), np.linspace(20., 450., 8))
        return np.stack((x.ravel(), y.ravel()), -1).astype(np.float32)

    def test_known_geometry_precision_counts_grid_sr_and_paper_rmse(self):
        a = self.grid()
        H = np.array([[1., 0., 7.], [0., 1., 11.], [0., 0., 1.]])
        b = a + np.array([7., 11.], dtype=np.float32)
        metrics = runner.evaluate_matches(a, b, (512, 512), (512, 512), H)
        self.assertEqual(metrics['matches'], 64)
        self.assertEqual(metrics['ncm'], {'1': 64, '3': 64, '5': 64})
        self.assertEqual(metrics['precision'], {'1': 1., '3': 1., '5': 1.})
        self.assertTrue(all(metrics['sr'].values()))
        self.assertLess(metrics['registration_rmse'], 1e-3)
        self.assertEqual(metrics['rmse_correct5_or_failure10'], 0.)

    def test_full_original_order_reproduces_all_metrics_and_detects_tampering(self):
        a = self.grid()
        b = a.copy()
        b[::9] += 13.
        confidence = np.linspace(.1, .9, len(a)).astype(np.float32)
        old = runner.evaluate_matches(a, b, (512, 512), (512, 512), np.eye(3))
        _, selections = runner.stable_selections(confidence)
        full = selections['Full']
        actual = runner.evaluate_matches(a[full], b[full], (512, 512), (512, 512), np.eye(3))
        runner.require_metrics_equal(actual, old)
        changed = dict(actual, registration_rmse=actual['registration_rmse'] + 1e-6)
        with self.assertRaisesRegex(RuntimeError, 'registration_rmse'):
            runner.require_metrics_equal(changed, old)
        changed = dict(actual, sr=dict(actual['sr'], **{'3': not actual['sr']['3']}))
        with self.assertRaisesRegex(RuntimeError, 'sr.3'):
            runner.require_metrics_equal(changed, old)

    def test_confidence_selection_has_no_gt_or_correctness_filter(self):
        a = np.tile(self.grid(), (32, 1))
        b = a.copy()
        b[1024:] += 12.
        confidence = np.r_[np.full(1024, .1), np.full(1024, .9)].astype(np.float32)
        _, selections = runner.stable_selections(confidence)
        np.testing.assert_array_equal(selections['Top-1024'], np.arange(1024, 2048))
        full = runner.evaluate_matches(a, b, (512, 512), (512, 512), np.eye(3))
        ids = selections['Top-1024']
        selected = runner.evaluate_matches(a[ids], b[ids], (512, 512), (512, 512), np.eye(3))
        self.assertEqual(full['precision']['5'], .5)
        self.assertEqual(selected['precision']['5'], 0.)
        self.assertFalse(selected['sr']['3'])
        self.assertEqual(selected['rmse_correct5_or_failure10'], 10.)

    def test_empty_output_counts_zero_and_failed_pair_rmse_ten(self):
        metrics = runner.evaluate_matches(np.empty((0, 2)), np.empty((0, 2)), (512, 512), (512, 512), np.eye(3))
        self.assertEqual(metrics['matches'], 0)
        self.assertEqual(metrics['ncm'], {'1': 0, '3': 0, '5': 0})
        self.assertEqual(metrics['precision'], {'1': 0., '3': 0., '5': 0.})
        self.assertFalse(any(metrics['sr'].values()))
        self.assertIsNone(metrics['registration_rmse'])
        self.assertEqual(metrics['rmse_correct5_or_failure10'], 10.)

    def test_summary_labels_cpu_metric_time_and_full_under_k_as_not_applicable(self):
        a = self.grid()
        metric = runner.evaluate_matches(a, a, (512, 512), (512, 512), np.eye(3))
        record = dict(metrics=metric, metric_evaluation_ms=3., source_nm=64, selected_nm=64, under_k=None)
        summary = runner.summarize([record])
        self.assertEqual(summary['mean_matches'], 64)
        self.assertEqual(summary['metric_evaluation_ms_mean'], 3.)
        self.assertFalse(any('inference_ms' in key for key in summary))
        self.assertIsNone(runner.selection_summary([record], None)['under_k_pairs'])

    def test_import_does_not_load_torch_or_model_modules(self):
        command = [sys.executable, '-c',
                   'import sys; import scripts.evaluate_topk_correspondence_ablation; '
                   'assert "torch" not in sys.modules; '
                   'assert not any(k.startswith("mhinet.models") for k in sys.modules)']
        subprocess.run(command, cwd=runner.REPO, check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
