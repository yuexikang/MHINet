from __future__ import annotations

import unittest

import torch

from mhinet.real_correlation_audit import _bf16_signal_diagnostics


class CorrelationPrecisionAuditTests(unittest.TestCase):
    def test_bf16_roundtrip_reports_ties_and_argmax_loss(self) -> None:
        scores = torch.tensor(
            [
                [1.0, 1.003, -4.0],
                [0.5, -3.0, 0.51],
            ],
            dtype=torch.float32,
        )
        valid = torch.tensor(
            [
                [True, True, False],
                [True, False, True],
            ]
        )
        nearest = torch.tensor([1, 2])

        report = _bf16_signal_diagnostics(
            scores,
            valid,
            nearest,
            center_index=0,
        )

        self.assertEqual(report["nearest_center_equal_after_bf16_fraction"], 0.5)
        self.assertEqual(report["raw_positive_margin_fraction"], 1.0)
        self.assertEqual(report["positive_margin_fraction_after_bf16"], 0.5)
        self.assertEqual(report["raw_positive_margin_retained_fraction"], 0.5)
        self.assertEqual(report["argmax_candidate_preserved_fraction"], 0.5)
        self.assertEqual(
            report["nearest_candidate_rank1_after_bf16_fraction"], 1.0
        )
        self.assertGreater(
            report["valid_score_abs_quantization_error"]["max"], 0.0
        )

    def test_bf16_roundtrip_rejects_invalid_shapes_and_empty_rows(self) -> None:
        scores = torch.zeros((2, 3))
        with self.assertRaises(ValueError):
            _bf16_signal_diagnostics(
                scores,
                torch.ones((2, 2), dtype=torch.bool),
                torch.zeros(2, dtype=torch.long),
                center_index=0,
            )
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            _bf16_signal_diagnostics(
                scores,
                torch.tensor([[True, False, False], [False, False, False]]),
                torch.zeros(2, dtype=torch.long),
                center_index=0,
            )


if __name__ == "__main__":
    unittest.main()
