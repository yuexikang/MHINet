import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from mhinet.dataio.check_temporal_ready import check_ready
from mhinet.engine.train import TrainConfig


class TemporalReadyTests(unittest.TestCase):
    def test_script_defaults(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('MHINET_RUNTIME', 'MHINET_CONFIG', 'MHINET_OUTPUT_DIR', 'GPU_ID'):
            env.pop(key, None)
        result = subprocess.run(['bash', 'scripts/train_frozen_dino_mvt.sh'],
                                env=env, capture_output=True, text=True, check=True)
        for value in ('CUDA_VISIBLE_DEVICES=1', 'runtime_paths.temporal4.server.json',
                      'train_frozen_dino_mvt_temporal4.json', 'outputs/GHIM_joint_frozen_dino_mvt_temporal4_seed0'):
            self.assertIn(value, result.stdout)
        self.assertEqual(TrainConfig.from_json('configs/train_frozen_dino_mvt_temporal4.json').profile,
                         'frozen_dino_mvt')

    def test_incomplete_and_smoke_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for status, smoke in [('planned', False), ('completed', True)]:
                (root/'dataset_summary.json').write_text(json.dumps({'status': status, 'smoke_only': smoke}))
                with self.assertRaises(ValueError):
                    check_ready(root)

    def test_complete_dataset_and_missing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {'pairs': 4, 'generated_pairs': 4, 'parent_groups': 1}
            (root/'dataset_summary.json').write_text(json.dumps({
                'status': 'completed', 'smoke_only': False, 'version': 'temporal_four_pairs_v1',
                'planned': {'train': expected, 'val': expected}}))
            for split, lat in [('train', '37.534000'), ('val', '37.634000')]:
                (root/split).mkdir()
                (root/split/'image.png').touch()
                rows = []
                for kind in ('same_past', 'same_current', 'cross_past_current', 'cross_current_past'):
                    rows.append(dict(pair_id=kind, parent_temporal_group=lat, pair_kind=kind,
                        source=f'{lat}126.9115.jpg', parent_image_A=lat+'A', parent_image_B=lat+'B',
                        image_A='image.png', image_B='image.png', mask_A_overlap='image.png', mask_B_overlap='image.png'))
                (root/split/'pairs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (root/'test').mkdir()
            (root/'test/pairs.jsonl').touch()
            self.assertEqual(check_ready(root), {'train': 4, 'val': 4})
            (root/'val/image.png').unlink()
            with self.assertRaises(FileNotFoundError):
                check_ready(root)
