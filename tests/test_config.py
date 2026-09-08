from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mhinet.config import RuntimePaths, load_architecture_config, load_training_profiles


class ConfigTests(unittest.TestCase):
    def test_design_config_is_protocol_v12(self) -> None:
        config = load_architecture_config()
        self.assertEqual(config.scales, (8, 4, 2, 1))
        self.assertEqual(config.radii, (4, 4, 3, 2))
        self.assertEqual(config.expected_new_parameters, 2_544_160)

    def test_training_profiles(self) -> None:
        profiles = load_training_profiles()
        self.assertEqual(profiles["heads"], ("adapters", "refinement_decoders"))
        self.assertIn("mvt", profiles["joint"])
        self.assertNotIn("dino", profiles["joint"])

    def test_null_runtime_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps({"repo_root": None}), encoding="utf-8")
            with self.assertRaises(ValueError):
                RuntimePaths.from_json(path, require_files=False)


if __name__ == "__main__":
    unittest.main()
