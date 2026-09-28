import unittest
from dataclasses import replace
from unittest.mock import Mock,patch
from mhinet.downstream.semidense import SemidenseConfig
from scripts.train_semidense_chunk_experiment import load_chunk_variant


class ChunkExperimentTests(unittest.TestCase):
    def test_only_window_batching_changes_after_strict_source_load(self):
        config=SemidenseConfig(window_chunk=128)
        system=Mock()
        with patch('scripts.train_semidense_chunk_experiment.load_completed_semidense',return_value=system) as load:
            self.assertIs(load_chunk_variant('runtime','source',config),system)
            load.assert_called_once_with('runtime','source',replace(config,window_chunk=32))
            self.assertEqual(system.matcher.config,config)
            with self.assertRaises(ValueError):load_chunk_variant('runtime','source',replace(config,window_chunk=64))


if __name__=='__main__':unittest.main()
