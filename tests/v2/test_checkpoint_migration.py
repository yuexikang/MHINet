"""Migration rejects incomplete v1 weights before changing the model."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from mhinet.engine.checkpointing import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from mhinet.v2.checkpointing import load_v1_weights, migrate_v1_state


def _system(new_branches=True):
    system = nn.Module()
    system.shared = nn.Linear(3, 4)
    system.matcher = nn.Module()
    system.matcher.qrru = nn.Module()
    system.matcher.qrru.original = nn.Linear(4, 2)
    if new_branches:
        for name in ("diff_encoder", "diff_adapter", "dgfo", "flow_adapter"):
            setattr(system.matcher.qrru, name, nn.Linear(2, 2))
    return system


class MigrationTests(unittest.TestCase):
    def test_all_legacy_tensors_load_and_new_branches_keep_their_initialization(self):
        v1, v2 = _system(False), _system(True)
        before = {k: v.clone() for k, v in v2.state_dict().items()}
        report = migrate_v1_state(v2, v1.state_dict())
        self.assertTrue(report["all_original_keys_verified"])
        self.assertEqual(report["loaded_tensors"], len(v1.state_dict()))
        for key, value in v2.state_dict().items():
            torch.testing.assert_close(value, v1.state_dict().get(key, before[key]), rtol=0, atol=0)
        self.assertEqual(len(report["initialized_v2_keys"]), 8)
        self.assertEqual(report["unexpected_keys"], [])

    def _assert_rejected_without_mutation(self, state, pattern):
        target = _system(True)
        before = {k: v.clone() for k, v in target.state_dict().items()}
        with self.assertRaisesRegex(ValueError, pattern):
            migrate_v1_state(target, state)
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_missing_shared_or_original_qrru_keys_is_fatal(self):
        for key in ("shared.weight", "matcher.qrru.original.weight"):
            state = dict(_system(False).state_dict())
            del state[key]
            self._assert_rejected_without_mutation(state, "missing original")

    def test_extra_keys_and_shape_mismatches_are_fatal(self):
        state = dict(_system(False).state_dict())
        state["extra.weight"] = torch.ones(1)
        self._assert_rejected_without_mutation(state, "unexpected")
        state = dict(_system(False).state_dict())
        state["matcher.qrru.original.weight"] = torch.ones(3, 4)
        self._assert_rejected_without_mutation(state, "shape mismatch")

    def test_new_branch_whitelist_does_not_allow_arbitrary_shared_or_qrru_modules(self):
        target = _system(True)
        target.shared.diff_encoder = nn.Linear(3, 3)
        with self.assertRaisesRegex(ValueError, "shared.diff_encoder"):
            migrate_v1_state(target, _system(False).state_dict())
        target = _system(True)
        target.matcher.qrru.unreviewed = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            migrate_v1_state(target, _system(False).state_dict())

    def test_invalid_tensor_values_fail_before_any_copy(self):
        state = dict(_system(False).state_dict())
        state["shared.bias"] = None
        self._assert_rejected_without_mutation(state, "non-tensor")

    def test_identical_topology_requires_no_missing_keys(self):
        report = migrate_v1_state(_system(False), _system(False).state_dict())
        self.assertEqual(report["missing_keys"], [])

    def test_restricted_checkpoint_load_checks_task_before_model_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.pt"
            payload = {"format": CHECKPOINT_FORMAT, "version": CHECKPOINT_VERSION,
                       "model": _system(False).state_dict(),
                       "metadata": {"task": "semidense_qrru_v1"},
                       "progress": {"optimizer_step": 23}}
            torch.save(payload, path)
            target = _system(True)
            report = load_v1_weights(target, path)
            self.assertEqual(report["source_optimizer_step"], 23)
            self.assertEqual(report["metadata"]["task"], "semidense_qrru_v1")
            payload["metadata"]["task"] = "other-task"
            torch.save(payload, path)
            before = {k: v.clone() for k, v in target.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "task"):
                load_v1_weights(target, path)
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
