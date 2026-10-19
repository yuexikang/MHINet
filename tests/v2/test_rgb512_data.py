"""Index-only RGB512 manifest tests; no images, legacy geo parsing, or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mhinet.v2.data import RGB512Dataset


class RGB512DataTests(unittest.TestCase):
    def _row(self, pair='pair-0', split='train', dataset='mock-rgb', source='tiles/original.png'):
        # Deliberately lacks source/parent_image_A, which the legacy geographic
        # parser requires. The real RGB512 source-identity schema works alone.
        return dict(pair_id=pair, split=split, source_dataset=dataset,
                    source_relative_path=source, size_A=[512, 512], size_B=[512, 512])

    def _manifest(self, root, rows, split='train', *, blanks=False):
        path = Path(root)/split/'pairs.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        if blanks:
            path.write_text('\n\n'+'\n\n'.join(lines)+'\n', encoding='utf-8')
        else:
            path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        return path

    def test_index_bypasses_legacy_geographic_parsing_and_does_not_load_images(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [self._row(source='tiles/原图.png'), self._row(pair='pair-1', dataset='second-rgb')]
            path = self._manifest(root, rows, blanks=True)
            with patch('mhinet.dataio.data.parse_geo_region', side_effect=AssertionError('legacy geo parser')), \
                 patch('mhinet.dataio.data.parent_group', side_effect=AssertionError('legacy parent parser')), \
                 patch('mhinet.dataio.data.load_rgb_bicubic', side_effect=AssertionError('image loading')):
                data = RGB512Dataset(path)
                self.assertEqual(len(data), 2)
                self.assertEqual(data.image_size, 512)
                self.assertEqual(data.source_counts, {'mock-rgb': 1, 'second-rgb': 1})
                self.assertEqual(data.index[0].parent_group, 'source:mock-rgb:tiles/原图.png')
                self.assertEqual(data.index[0].geo_group, data.index[0].parent_group)
                # Byte offsets remain correct across blank lines and UTF-8 paths.
                for entry, row in zip(data.index, rows):
                    self.assertEqual(data._read_record(entry.offset), row)
                limited = RGB512Dataset(path, max_pairs=1)
                self.assertEqual(len(limited), 1)

    def test_only_512_loader_and_stored_shape_are_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._manifest(root, [self._row()])
            for size in (784, 256, 0, None, '512'):
                with self.subTest(image_size=size):
                    with self.assertRaisesRegex(ValueError, 'only accepts 512'):
                        RGB512Dataset(path, image_size=size)
            for side in ('size_A', 'size_B'):
                row = self._row()
                row[side] = [784, 512]
                path = self._manifest(root, [row])
                with self.subTest(side=side):
                    with self.assertRaisesRegex(ValueError, 'stored 512x512'):
                        RGB512Dataset(path)

    def test_mismatched_split_and_duplicate_pair_id_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            mismatch = self._manifest(root, [self._row(split='val')])
            with self.assertRaisesRegex(ValueError, 'split disagrees'):
                RGB512Dataset(mismatch)
            duplicate = self._manifest(root, [self._row(), self._row(source='tiles/another.png')])
            with self.assertRaisesRegex(ValueError, 'Duplicate RGB512 pair ID'):
                RGB512Dataset(duplicate)

    def test_source_identity_fields_required_and_source_path_fallback_supported(self):
        with tempfile.TemporaryDirectory() as root:
            for missing in ('source_dataset', 'source_relative_path'):
                row = self._row()
                row.pop(missing)
                path = self._manifest(root, [row])
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(ValueError, 'original source identity'):
                        RGB512Dataset(path)
            row = self._row()
            row['source_relative_path'] = ''
            row['source_path'] = '/source/archive/original.png'
            data = RGB512Dataset(self._manifest(root, [row]))
            self.assertEqual(data.index[0].parent_group, 'source:mock-rgb:/source/archive/original.png')

    def test_cross_split_source_overlap_is_detectable_independent_of_pair_ids(self):
        with tempfile.TemporaryDirectory() as root:
            train = RGB512Dataset(self._manifest(root, [
                self._row(pair='train-shared', source='tiles/shared.png'),
                self._row(pair='train-only', source='tiles/train-only.png'),
            ]))
            val = RGB512Dataset(self._manifest(root, [
                self._row(pair='val-shared', split='val', source='tiles/shared.png'),
                self._row(pair='val-other-dataset', split='val', dataset='other-rgb', source='tiles/train-only.png'),
            ], split='val'))
            train_sources = {entry.parent_group for entry in train.index}
            val_sources = {entry.parent_group for entry in val.index}
            self.assertEqual(train_sources & val_sources, {'source:mock-rgb:tiles/shared.png'})
            self.assertFalse({entry.pair_id for entry in train.index} & {entry.pair_id for entry in val.index})
            self.assertIn('source-image isolation', train.grouping_policy)


if __name__ == '__main__':
    unittest.main()
