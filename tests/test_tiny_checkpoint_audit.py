from __future__ import annotations

import unittest

import torch

from mhinet.engine.checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from mhinet.diagnostics.tiny_checkpoint_audit import (
    _inspect_checkpoint_payload,
    _resolve_protocol,
    _state_dict_sha256,
    _summarize_delta_diagnostics,
)


class TinyCheckpointAuditTests(unittest.TestCase):
    @staticmethod
    def _payload(metadata: dict[str, object]) -> dict[str, object]:
        return {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "model": {"weight": torch.tensor([1.0])},
            "progress": {"optimizer_step": 7},
            "metadata": metadata,
        }

    @staticmethod
    def _signature() -> dict[str, object]:
        return {
            "diagnostic": "TINY-S-D1",
            "active_scales": (1,),
            "sample_count": 2,
            "sample_protocol": "one_pair_residuals",
            "residual_profile": "translation",
            "precision": "bf16",
            "seed": 3,
            "residual_bound_fraction": 1.0,
            "pair_ids": ("pair", "pair"),
            "target_hw": (784, 784),
            "maximum_declared_abs_residual_px": 2.0,
            "condition_index_offset": 2,
            "resume_context": {},
        }

    def test_progress_signature_resolves_complete_protocol(self) -> None:
        signature = self._signature()
        header = _inspect_checkpoint_payload(
            self._payload(
                {
                    "checkpoint_role": "tiny_progress",
                    "tiny_progress_signature": signature,
                }
            )
        )
        protocol = _resolve_protocol(header, {})
        self.assertEqual(protocol["diagnostic"], "TINY-S-D1")
        self.assertEqual(protocol["active_scales"], (1,))
        self.assertEqual(protocol["seed"], 3)
        self.assertEqual(protocol["condition_index_offset"], 2)
        self.assertEqual(protocol["maximum_declared_abs_residual_px"], 2.0)
        self.assertEqual(protocol["legacy_supplemented_fields"], [])

    def test_signature_metadata_and_cli_conflicts_fail_closed(self) -> None:
        signature = self._signature()
        payload = self._payload(
            {
                "checkpoint_role": "tiny_final_evidence",
                "tiny_progress_signature": signature,
                "precision": "fp32",
            }
        )
        header = _inspect_checkpoint_payload(payload)
        with self.assertRaisesRegex(RuntimeError, "conflict for precision"):
            _resolve_protocol(header, {})

        payload["metadata"] = {
            "checkpoint_role": "tiny_final_evidence",
            "tiny_progress_signature": signature,
        }
        header = _inspect_checkpoint_payload(payload)
        with self.assertRaisesRegex(RuntimeError, "conflict for seed"):
            _resolve_protocol(header, {"seed": 4})

    def test_legacy_checkpoint_requires_and_records_missing_seed_and_bound(self) -> None:
        metadata = {
            "known_H0_residual_injected": True,
            "formal_training_checkpoint": False,
            "diagnostic": "TINY-S-D1",
            "active_scales": (1,),
            "sample_count": 2,
            "sample_protocol": "one_pair_residuals",
            "residual_profile": "translation",
            "precision": "bf16",
            "pair_ids": ("pair", "pair"),
        }
        header = _inspect_checkpoint_payload(self._payload(metadata))
        self.assertTrue(header["legacy"])
        with self.assertRaisesRegex(ValueError, "residual_bound_fraction, seed"):
            _resolve_protocol(header, {})
        protocol = _resolve_protocol(
            header,
            {"seed": 0, "residual_bound_fraction": 1.0},
        )
        self.assertEqual(
            protocol["legacy_supplemented_fields"],
            ["residual_bound_fraction", "seed"],
        )
        self.assertEqual(protocol["target_hw"], (784, 784))
        self.assertEqual(protocol["maximum_declared_abs_residual_px"], 2.0)

    def test_formal_or_unknown_checkpoint_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported tiny checkpoint role"):
            _inspect_checkpoint_payload(
                self._payload({"checkpoint_role": "formal_training"})
            )
        with self.assertRaisesRegex(ValueError, "Formal training"):
            _inspect_checkpoint_payload(
                self._payload(
                    {
                        "known_H0_residual_injected": True,
                        "formal_training_checkpoint": True,
                    }
                )
            )

    def test_pair_count_target_and_declared_residual_are_checked(self) -> None:
        for field, value, message in (
            ("pair_ids", ("pair",), "pair_ids length"),
            ("target_hw", (640, 640), "target_hw"),
            ("maximum_declared_abs_residual_px", 1.0, "maximum tiny residual"),
        ):
            signature = self._signature()
            signature[field] = value
            header = _inspect_checkpoint_payload(
                self._payload(
                    {
                        "checkpoint_role": "tiny_progress",
                        "tiny_progress_signature": signature,
                    }
                )
            )
            with self.assertRaisesRegex((ValueError, RuntimeError), message):
                _resolve_protocol(header, {})

    def test_state_digest_binds_names_values_dtype_and_shape(self) -> None:
        base = _state_dict_sha256({"weight": torch.tensor([1.0])})
        self.assertEqual(base, _state_dict_sha256({"weight": torch.tensor([1.0])}))
        self.assertNotEqual(base, _state_dict_sha256({"weight": torch.tensor([2.0])}))
        self.assertNotEqual(base, _state_dict_sha256({"bias": torch.tensor([1.0])}))
        self.assertNotEqual(
            base, _state_dict_sha256({"weight": torch.tensor([1.0], dtype=torch.float64)})
        )
        self.assertEqual(
            _state_dict_sha256({"counter": torch.tensor(1)}),
            _state_dict_sha256({"counter": torch.tensor(1)}),
        )

    def test_delta_summary_keeps_x_and_y_failure_modes_separate(self) -> None:
        endpoint = {
            "per_pair": [
                {
                    "H0_corner_residual_px": [[2.0, -1.0]] * 4,
                    "H_updates_corner_residual_px": [
                        [[0.0, -1.0]] * 4,
                        [[0.0, -1.0]] * 4,
                    ],
                    "delta_px": [[[2.0, 0.0]] * 4, [[0.0, 0.0]] * 4],
                    "update_accepted": [True, True],
                    "update_scale_schedule": [1, 1],
                    "tanh_saturation_fraction": [0.0, 0.0],
                    "supported_query_count": [100, 100],
                    "condition_number": [2.0, 2.0],
                }
            ]
        }
        summary = _summarize_delta_diagnostics(endpoint)
        first = summary["updates"][0]
        self.assertEqual(first["coordinate_axes"]["x"]["desired_after_abs_px"]["mean"], 0.0)
        self.assertEqual(first["coordinate_axes"]["y"]["desired_after_abs_px"]["mean"], 1.0)
        self.assertEqual(
            first["coordinate_axes"]["x"]["delta_target_sign_agreement_fraction"],
            1.0,
        )
        self.assertEqual(
            first["coordinate_axes"]["y"]["delta_target_sign_agreement_fraction"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
