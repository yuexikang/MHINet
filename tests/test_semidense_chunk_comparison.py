import unittest
from scripts.evaluate_semidense_chunk_accuracy import compare_points


class ComparisonTest(unittest.TestCase):
    def test_permutation_aligns_coordinates(self):
        r=compare_points([[1,2],[3,4]],[[5,6],[7,8]],[[3,4],[1,2]],[[7,8],[5,6]])
        self.assertTrue(r['source_sets_equal']);self.assertFalse(r['source_order_equal']);self.assertEqual(r['common_target_max_delta_px'],0)
    def test_missing_and_added_sources(self):
        r=compare_points([[1,2],[3,4]],[[5,6],[7,8]],[[1,2],[9,9]],[[8,10],[0,0]])
        self.assertEqual(r['dropped_sources'],1);self.assertEqual(r['added_sources'],1);self.assertEqual(r['common_target_mean_delta_px'],5)
    def test_empty_and_ambiguous(self):
        r=compare_points([],[],[],[]);self.assertTrue(r['source_sets_equal']);self.assertIsNone(r['common_target_mean_delta_px'])
        r=compare_points([[1,2],[1,2]],[[3,4],[4,5]],[[1,2]],[[3,4]])
        self.assertTrue(r['duplicate_sources']);self.assertEqual(r['compared_targets'],0)


if __name__=='__main__':unittest.main()
