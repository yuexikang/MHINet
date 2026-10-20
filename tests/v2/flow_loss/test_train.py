"""CPU tests for exact source cloning, extension schedules and restart guards."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from mhinet.engine.checkpointing import capture_rng_state, load_checkpoint, save_checkpoint
from mhinet.v2.flow_loss.common import EXTRA_STEPS, SOURCE_STEP, digest_tree
from mhinet.v2.flow_loss.train import (FixedExtensionScheduler, assert_tree_equal,
    clone_source_state, completion_status, progress_valid, read_pilot_config)


class FlowPilotRunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        random.seed(3); np.random.seed(3); torch.manual_seed(3)

    def model(self):
        return nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Dropout(.2), nn.Linear(8, 2))

    def step(self, model, optimizer, scheduler=None):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(5, 4)+random.random()+float(np.random.rand())
        loss = model(x).square().mean()
        loss.backward(); optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def source(self):
        model = self.model(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        optimizer.param_groups[0]['initial_lr'] = 1e-5
        for _ in range(3):
            self.step(model, optimizer)
        return copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng_state()))

    def clone(self, payload):
        model = self.model(); optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
        proof = clone_source_state(model, optimizer, payload)
        scheduler = FixedExtensionScheduler(optimizer)
        return model, optimizer, scheduler, proof

    def test_source_moments_and_rng_are_cloned_before_lr_reset(self):
        payload = self.source()
        model, optimizer, scheduler, proof = self.clone(payload)
        self.assertTrue(proof['every_source_tensor_exact'])
        self.assertEqual(proof['optimizer_tensors_verified'], 12)
        self.assertEqual(proof['rng_sha256'], digest_tree(capture_rng_state()))
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-6)
        self.assertEqual(optimizer.param_groups[0]['initial_lr'], 1e-6)
        self.assertEqual(optimizer.param_groups[0]['betas'], payload['optimizer']['param_groups'][0]['betas'])
        self.assertEqual(scheduler.state_dict(), dict(kind='fixed_lr_extension_v1', learning_rate=1e-6, completed=0))
        for a, b in zip(optimizer.state_dict()['state'].values(), payload['optimizer']['state'].values()):
            assert_tree_equal(a, b)
        assert_tree_equal(model.state_dict(), payload['model'])

    def test_continuous_two_updates_equal_one_update_then_resume(self):
        payload = self.source()
        model, optimizer, scheduler, _ = self.clone(payload)
        for _ in range(2):
            self.step(model, optimizer, scheduler)
        expected_model = copy.deepcopy(model.state_dict()); expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_rng = capture_rng_state()
        model, optimizer, scheduler, _ = self.clone(payload)
        self.step(model, optimizer, scheduler)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'latest.pt'; metadata = dict(arm='L0', immutable='identity')
            save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                optimizer_step=SOURCE_STEP+1, data_progress=dict(epoch=5, cursor=8), metadata=metadata)
            model, optimizer, scheduler, _ = self.clone(payload)
            restored = load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                map_location='cpu', expected_metadata=metadata)
            progress_valid(restored['optimizer_step'], restored['data_progress']['cursor'], 5, scheduler.completed)
            self.step(model, optimizer, scheduler)
            assert_tree_equal(model.state_dict(), expected_model)
            assert_tree_equal(optimizer.state_dict(), expected_optimizer)
            assert_tree_equal(capture_rng_state(), expected_rng)
            self.assertEqual(scheduler.completed, 2)

    def test_wrong_arm_or_changed_identity_resume_rejected(self):
        model, optimizer, scheduler, _ = self.clone(self.source())
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'latest.pt'
            save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                optimizer_step=SOURCE_STEP, metadata=dict(arm='L0', code='hash-A', config='hash-B'))
            for mismatch in (dict(arm='L1'), dict(code='different'), dict(config='different')):
                with self.assertRaisesRegex(RuntimeError, 'metadata mismatch'):
                    load_checkpoint(path, model=model, map_location='cpu', expected_metadata=mismatch)

    def test_progress_and_terminal_full_validation_are_required(self):
        progress_valid(SOURCE_STEP, 0, 5, 0)
        progress_valid(SOURCE_STEP+1, 8, 5, 1)
        progress_valid(SOURCE_STEP+EXTRA_STEPS, 40849, 5, EXTRA_STEPS)
        for args in ((SOURCE_STEP+1, 7, 5, 1), (SOURCE_STEP+1, 8, 4, 1), (SOURCE_STEP+1, 8, 5, 0)):
            with self.assertRaises(ValueError):
                progress_valid(*args)
        terminal = SOURCE_STEP+EXTRA_STEPS
        self.assertEqual(completion_status(terminal, -1, False), 'interrupted')
        self.assertEqual(completion_status(terminal, terminal, False), 'completed')
        self.assertEqual(completion_status(SOURCE_STEP+1, -1, False), 'diagnostic_stop')
        self.assertEqual(completion_status(SOURCE_STEP+1, -1, True), 'interrupted')

    def test_scheduler_rejects_old_cosine_and_tampered_lr(self):
        _, optimizer, scheduler, _ = self.clone(self.source())
        with self.assertRaises(ValueError):
            scheduler.load_state_dict(dict(kind='old_cosine', learning_rate=1e-6, completed=0))
        optimizer.param_groups[0]['lr'] = .01
        with self.assertRaises(RuntimeError):
            scheduler.step()

    def test_config_registration_rejects_budget_changes_and_unknown_knobs(self):
        base = dict(schema='mhi.flow-loss-pilot.v1', arm='L0', base_config='base.json',
            runtime='runtime.json', source_checkpoint='source.pt', calibration_report='pending.json')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'pilot.json'; path.write_text(json.dumps(base))
            config = read_pilot_config(path)
            self.assertEqual(config['additional_steps'], EXTRA_STEPS)
            self.assertEqual(config['order_seed'], 47)
            for field, value in (('additional_steps', 1), ('workers', 8), ('seed', 43), ('order_seed', 48),
                                 ('learning_rate', 2e-6), ('accumulation', 4), ('alpha', .1), ('epsilon_px', True)):
                path.write_text(json.dumps({**base, field:value}))
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        read_pilot_config(path)


if __name__ == '__main__':
    unittest.main()
