import unittest
from types import SimpleNamespace
from mhinet.dataio.generate_temporal import recipes, split_groups


class TemporalGenerationTests(unittest.TestCase):
    def test_four_recipes(self):
        rows = recipes('example', SimpleNamespace(derive_seed=lambda *args: 12), 42)
        self.assertEqual(len(rows), 4)
        self.assertEqual([(a, b) for _, a, b, _ in rows],
                         [('past', 'past'), ('current', 'current'),
                          ('past', 'current'), ('current', 'past')])
        self.assertEqual(sorted(r[-1] for r in rows), [0, 0, 1, 1])

    def test_grouped_ratio_and_reproducibility(self):
        groups = [{'group_id': str(i), 'geo_group': str(i//2)} for i in range(100)]
        result = split_groups(groups, 42)
        self.assertEqual(len(result['train']), 90)
        self.assertEqual(len(result['val']), 10)
        self.assertFalse({r['geo_group'] for r in result['train']} &
                         {r['geo_group'] for r in result['val']})
        self.assertEqual(result, split_groups(list(reversed(groups)), 42))

    def test_unattainable_ratio_keeps_cells_whole(self):
        groups = [{'group_id': str(i), 'geo_group': str(i//3)} for i in range(30)]
        result = split_groups(groups)
        self.assertEqual(len(result['val']), 3)
        with self.assertRaises(ValueError):
            split_groups([{'group_id': str(i), 'geo_group': 'one'} for i in range(10)])
