from __future__ import annotations

import unittest

import torch

from mhinet.evaluate import (
    _flatten_evaluation_row,
    _trajectory_conditional_validity,
    _trajectory_errors,
    _trajectory_metric_summaries,
    evaluate_model,
)


class EvaluationGeometryTests(unittest.TestCase):
    def test_identity_trajectory_has_zero_input_and_native_error(self) -> None:
        trajectory = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1)
        result = _trajectory_errors(
            trajectory,
            torch.eye(3).unsqueeze(0),
            native_target_hw=(611, 937),
        )
        self.assertTrue(bool(result["geometry_valid"].all()))
        self.assertTrue(bool(result["grid5_geometry_valid"].all()))
        self.assertLess(float(result["mace_input_px"].abs().max()), 1e-6)
        self.assertLess(float(result["mace_native_target_px"].abs().max()), 1e-6)
        self.assertLess(float(result["grid5_input_px"].abs().max()), 1e-6)
        self.assertLess(float(result["grid5_native_target_px"].abs().max()), 1e-6)

    def test_conditional_summary_does_not_treat_failed_fallback_as_valid(self) -> None:
        rows = [
            {
                "trajectory_mace_input_px": [0.0, 0.0],
                "trajectory_conditional_valid": [False, False],
            },
            {
                "trajectory_mace_input_px": [10.0, 8.0],
                "trajectory_conditional_valid": [True, True],
            },
        ]
        all_finite, conditional = _trajectory_metric_summaries(
            rows, "trajectory_mace_input_px", 2
        )
        self.assertEqual(all_finite[0]["count"], 2)
        self.assertEqual(all_finite[0]["mean"], 5.0)
        self.assertEqual(conditional[0]["mean"], 10.0)
        self.assertEqual(conditional[0]["total_pair_count"], 2)
        self.assertEqual(conditional[0]["conditional_valid_count"], 1)
        self.assertEqual(conditional[0]["conditional_failure_rate"], 0.5)

    def test_trajectory_validity_keeps_guarded_rejected_states(self) -> None:
        validity = _trajectory_conditional_validity(
            stage1_valid=True,
            overall_valid=True,
            geometry_valid=[True, True, True, True],
            update_accepted=[False, True, False],
        )
        self.assertEqual(validity, [True, True, True, True])

        with self.assertRaises(ValueError):
            _trajectory_conditional_validity(
                stage1_valid=True,
                overall_valid=True,
                geometry_valid=[True],
                update_accepted=[True],
            )

    def test_csv_flatten_retains_each_state_and_update_diagnostic(self) -> None:
        row = {
            "index": 0,
            "pair_id": "pair-0",
            "parent_group": "parent-0",
            "geo_group": "geo-0",
            "stage1_valid": True,
            "stage1_valid_correspondences": 49,
            "overall_valid": True,
            "size_A_wh": [937, 611],
            "size_B_wh": [901, 603],
            "window_recall_status": "not_computed_missing_gt_window_membership",
            "shared_call_counts": {"dino": 1, "mvt": 1},
            "latency_ms_single_pass": 12.5,
            "trajectory_mace_input_px": [3.0, 2.0, 1.0],
            "trajectory_mace_native_target_px": [3.1, 2.1, 1.1],
            "trajectory_grid5_input_px": [4.0, 3.0, 2.0],
            "trajectory_grid5_native_target_px": [4.1, 3.1, 2.1],
            "trajectory_geometry_valid": [True, True, True],
            "trajectory_conditional_valid": [True, True, True],
            "trajectory_grid5_geometry_valid": [True, True, True],
            "trajectory_grid5_conditional_valid": [True, True, True],
            "update_accepted": [True, False],
            "failure_reason_codes": [0, 3],
            "failure_reason_names": ["accepted", "ill_conditioned_system"],
            "supported_query_count": [81, 72],
            "condition_number": [10.0, 1.0e7],
            "solve_info": [0, -1],
            "delta_corner_l2_mean_px": [0.5, 0.25],
            "delta_corner_l2_max_px": [0.8, 0.4],
            "tanh_saturation_fraction": [0.0, 0.125],
        }
        flattened = _flatten_evaluation_row(row, ["H0", "H1", "H2"], [8, 8])
        self.assertEqual(flattened["H0_mace_input_px"], 3.0)
        self.assertEqual(flattened["H1_accepted"], True)
        self.assertEqual(flattened["H2_failure_reason_code"], 3)
        self.assertEqual(flattened["H2_supported_query_count"], 72)
        self.assertEqual(flattened["H2_delta_corner_l2_max_px"], 0.4)
        self.assertEqual(flattened["H_final_mace_input_px"], 1.0)
        self.assertEqual(
            flattened["window_recall_status"],
            "not_computed_missing_gt_window_membership",
        )

    def test_full_evaluation_exports_existing_update_diagnostics(self) -> None:
        class FakeModel:
            def eval(self) -> "FakeModel":
                return self

            def __call__(self, _images: torch.Tensor, **_kwargs: object) -> dict:
                identity = torch.eye(3).unsqueeze(0)
                return {
                    "H0_norm": identity,
                    "H_updates_norm": identity[:, None].repeat(1, 2, 1, 1),
                    "stage1_valid": torch.tensor([True]),
                    "overall_valid": torch.tensor([True]),
                    "update_accepted": torch.tensor([[True, False]]),
                    "failure_reason_codes": torch.tensor([[0, 3]]),
                    "failure_reason_names": {
                        0: "accepted",
                        3: "ill_conditioned_system",
                    },
                    "supported_query_count": torch.tensor([[81, 72]]),
                    "condition_number": torch.tensor([[10.0, 1.0e7]]),
                    "solve_info": torch.tensor([[0, -1]]),
                    "delta_px": torch.tensor(
                        [
                            [
                                [[3.0, 4.0]] * 4,
                                [[0.0, 0.0]] * 4,
                            ]
                        ]
                    ),
                    "tanh_saturation_fraction": torch.tensor([[0.0, 0.125]]),
                    "update_scale_schedule": (8, 8),
                    "shared_call_counts": {"dino": 1, "mvt": 1},
                    "stage1_valid_correspondences": torch.tensor([49]),
                }

        dataset = [
            {
                "images": torch.zeros(2, 3, 4, 4),
                "H_gt_norm": torch.eye(3),
                "pair_id": "pair-0",
                "parent_group": "parent-0",
                "geo_group": "geo-0",
                "size_A": torch.tensor([4.0, 4.0]),
                "size_B": torch.tensor([4.0, 4.0]),
            }
        ]
        summary, rows = evaluate_model(
            FakeModel(),  # type: ignore[arg-type]
            dataset,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            active_scales=(8,),
            iterations_per_scale=2,
        )
        self.assertEqual(summary["active_scales"], [8])
        self.assertEqual(summary["iterations_per_scale"], 2)
        self.assertEqual(summary["update_scale_schedule"], [8, 8])
        self.assertEqual(
            rows[0]["failure_reason_names"][1], "ill_conditioned_system"
        )
        self.assertEqual(rows[0]["supported_query_count"], [81, 72])
        self.assertEqual(rows[0]["delta_corner_l2_mean_px"], [5.0, 0.0])
        self.assertIsNone(rows[0]["window_recall"])
        self.assertFalse(
            summary["diagnostic_availability"]["window_recall"]["available"]
        )


if __name__ == "__main__":
    unittest.main()
