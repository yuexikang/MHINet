"""CPU-only checks of the fixed-weight ablation audit and execution boundary."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts import evaluate_h0_position_ablation as runner
import mhinet.downstream.semidense as matching


class AuditTests(unittest.TestCase):
    def features(self):
        return dict(pyramid={8: torch.randn(1, 2, 3, 2, 2), 2: torch.randn(1, 2, 3, 8, 8)},
                    H0_norm=torch.eye(3)[None], stage1_valid=torch.ones(1, dtype=torch.bool))

    def test_state_hash_includes_buffers_dtype_and_scalar(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer('scalar', torch.tensor(3, dtype=torch.int64))
        before = runner.state_sha256(model.state_dict())
        model.scalar.add_(1)
        self.assertNotEqual(before, runner.state_sha256(model.state_dict()))
        self.assertNotEqual(runner.tensor_sha256(torch.ones(2)), runner.tensor_sha256(torch.ones(2, dtype=torch.float64)))
        self.assertEqual(runner.tensor_sha256(torch.ones(2, dtype=torch.bfloat16)),
                         runner.tensor_sha256(torch.ones(2, dtype=torch.bfloat16)))
        self.assertEqual(runner.tensor_sha256(torch.empty(0)), runner.tensor_sha256(torch.empty(0)))

    def test_feature_hash_detects_mutation(self):
        features = self.features()
        before = runner.feature_sha256(features)
        features['pyramid'][2].add_(1)
        self.assertNotEqual(before, runner.feature_sha256(features))

    def test_all_variants_receive_same_feature_object(self):
        features = self.features()
        seen = []

        class Matcher:
            def infer(self, shared, sizes, **flags):
                seen.append((id(shared), flags))
                return [dict(points_a=torch.empty(0, 2), points_b=torch.empty(0, 2), confidence=torch.empty(0))]

        measure = lambda callback, device: (callback(), dict(wall_ms=1., stream_ms=1.))
        predictions, hashes = runner.infer_variants(SimpleNamespace(matcher=Matcher()), features, None, None, measure=measure)
        self.assertEqual(list(predictions), ['A', 'B', 'C', 'D'])
        self.assertEqual([value[0] for value in seen], [id(features)] * 4)
        self.assertEqual([value[1] for value in seen], list(runner.VARIANTS.values()))
        self.assertEqual(hashes, runner.feature_sha256(features))

    def test_infer_mutation_fails_and_probe_restores_after_exception(self):
        original_masks = matching.inference_overlap_masks
        original_coarse = matching.match_d8
        features = self.features()

        class MutatingMatcher:
            def infer(self, shared, sizes, **flags):
                shared['pyramid'][8].add_(1)
                return [{}]

        measure = lambda callback, device: (callback(), dict(wall_ms=1., stream_ms=1.))
        with self.assertRaisesRegex(RuntimeError, 'changed shared feature'):
            runner.infer_variants(SimpleNamespace(matcher=MutatingMatcher()), features, None, None, measure=measure)
        self.assertIs(matching.inference_overlap_masks, original_masks)
        self.assertIs(matching.match_d8, original_coarse)
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with runner.matcher_probe():
                raise RuntimeError('intentional')
        self.assertIs(matching.inference_overlap_masks, original_masks)
        self.assertIs(matching.match_d8, original_coarse)

    def test_baseline_requires_exact_coordinates_confidence_and_order(self):
        prediction = dict(points_a=torch.tensor([[1., 2.], [3., 4.]]),
                          points_b=torch.tensor([[5., 6.], [7., 8.]]), confidence=torch.tensor([.5, .7]))
        expected = {key: value.numpy().copy() for key, value in prediction.items()}
        self.assertEqual(runner.check_array_equal(prediction, expected, label='pair'),
                         ['points_a', 'points_b', 'confidence'])
        expected['points_b'][0, 0] = np.nextafter(expected['points_b'][0, 0], np.float32(100.))
        with self.assertRaisesRegex(RuntimeError, 'A baseline changed'):
            runner.check_array_equal(prediction, expected, label='pair')
        expected = {key: value.numpy().copy() for key, value in prediction.items()}
        expected['confidence'] = expected['confidence'][::-1].copy()
        with self.assertRaisesRegex(RuntimeError, 'A baseline changed'):
            runner.check_array_equal(prediction, expected, label='pair')

    def test_idle_requires_no_compute_process(self):
        base = dict(memory_mib=100, utilization=0, compute_process=False)
        self.assertTrue(runner.idle(base))
        self.assertFalse(runner.idle(dict(base, compute_process=True)))
        self.assertFalse(runner.idle(dict(base, memory_mib=4771, utilization=79)))

    def test_legacy_extra_tensors_and_failure_reason_are_checked(self):
        old = dict(points_a=torch.tensor([[1., 2.]]), coarse_a=torch.tensor([[3., 4.]]), failure_reason='none')
        current = dict(old)
        runner.check_legacy_equal(current, old)
        current['coarse_a'] = current['coarse_a'] + 1
        with self.assertRaisesRegex(RuntimeError, 'coarse_a'):
            runner.check_legacy_equal(current, old)


if __name__ == '__main__':
    unittest.main()
