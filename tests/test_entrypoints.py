"""Package layout and shell entry-point contracts; never start a GPU job."""
import importlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from mhinet.cli import COMMAND_MODULES

ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    def test_e00_shell_uses_pretrained_h0_and_full_test_split(self):
        env = dict(os.environ, DRY_RUN='1')
        env.pop('GPU_ID', None)
        result = subprocess.run(['bash', str(ROOT/'scripts/test_e00.sh')],
            cwd='/tmp', env=env, text=True, capture_output=True, check=True)
        for value in ('CUDA_VISIBLE_DEVICES=1', 'mhinet.cli evaluate', '--h0-only',
                      '--split test', 'outputs/E00_h0_seed0'):
            self.assertIn(value, result.stdout)
        self.assertNotIn('--checkpoint', result.stdout)
        self.assertNotIn('--max-pairs', result.stdout)
        rejected = subprocess.run(['bash', str(ROOT/'scripts/test_e00.sh'),
                                   '--checkpoint=/tmp/model.pt'], env=env, capture_output=True)
        self.assertEqual(rejected.returncode, 2)

    def test_accuracy_shell_requires_checkpoint_and_runs_evaluate_not_unittest(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)/'model with spaces.pt'
            checkpoint.touch()
            result = subprocess.run(['bash', str(ROOT/'scripts/test.sh'), str(checkpoint),
                                     '--max-pairs', '16'], cwd='/tmp',
                env=dict(os.environ, DRY_RUN='1', GPU_ID='0'), text=True, capture_output=True, check=True)
            self.assertIn('mhinet.cli evaluate', result.stdout)
            self.assertIn('--split test', result.stdout)
            self.assertIn('--checkpoint', result.stdout)
            self.assertNotIn('unittest', result.stdout)
        failed = subprocess.run(['bash', str(ROOT/'scripts/test.sh'), '/missing/mhinet.pt'],
                                capture_output=True)
        self.assertEqual(failed.returncode, 2)

    def test_all_cli_targets_resolve_after_reorganization(self):
        for command, module in COMMAND_MODULES.items():
            with self.subTest(command=command):
                self.assertTrue(callable(importlib.import_module(module).main))

    def test_train_shell_is_cwd_independent_and_defaults_to_gpu1(self):
        env = dict(os.environ, DRY_RUN="1")
        env.pop("GPU_ID", None)
        result = subprocess.run(["bash", str(ROOT/"scripts/train_e01.sh")],
            cwd="/tmp", env=env, text=True, capture_output=True, check=True)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", result.stdout)
        self.assertIn("--config", result.stdout)
        self.assertIn("e01_heads_v1.2", result.stdout)
        self.assertIn("--tiny-gate-artifact", result.stdout)

    def test_train_shell_forwards_resume_and_gpu_override(self):
        result = subprocess.run(["bash", str(ROOT/"scripts/train_e01.sh"),
            "--resume", "/tmp/checkpoint with spaces.pt"], cwd="/tmp",
            env=dict(os.environ, DRY_RUN="1", GPU_ID="2"),
            text=True, capture_output=True, check=True)
        self.assertIn("CUDA_VISIBLE_DEVICES=2", result.stdout)
        self.assertIn("--resume", result.stdout)
        self.assertIn(r"/tmp/checkpoint\ with\ spaces.pt", result.stdout)

    def test_train_shell_rejects_multiple_gpus(self):
        result = subprocess.run(["bash", str(ROOT/"scripts/train_e01.sh")],
            env=dict(os.environ, DRY_RUN="1", GPU_ID="1,2"), capture_output=True)
        self.assertEqual(result.returncode, 2)
