import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.gcp_annotator.core import Catalog, Store, write_json
from tools.gcp_annotator.evaluate import aggregate, evaluate, fit_matches, ground_metrics, selections, match_mapping
from tools.gcp_annotator.tests.helpers import dataset, point, prediction


class MetricTests(unittest.TestCase):
    def test_ground_formulas_and_threshold_boundaries(self):
        a=np.array([[1.,1.],[2.,2.],[3.,3.]])
        b=a+np.array([[3.,4.],[0.,0.],[0.,0.]])
        m=ground_metrics(np.eye(3),a,b)
        self.assertAlmostEqual(m['ground_rmse'],np.sqrt(25/3))
        self.assertEqual(m['ground_median_error'],0.)
        self.assertAlmostEqual(m['ground_p90_error'],4.)
        self.assertEqual(m['ground_sr'],{'1':False,'3':True,'5':True})
        self.assertTrue(ground_metrics(np.eye(3),a,a+[3.,0.])['ground_sr']['3'])
        with self.assertRaises(ValueError):ground_metrics(np.zeros((3,3)),a,b)

    def test_stable_topk_full_order_and_short_inputs(self):
        confidence=np.array([.4,.9,.9,.2])
        result=selections(confidence)
        np.testing.assert_array_equal(result['Full'],[0,1,2,3])
        np.testing.assert_array_equal(result['Top-1024'],[1,2,0,3])
        ranked=selections(np.ones(5000))
        for k in (1024,2048,4096):np.testing.assert_array_equal(ranked['Top-'+str(k)],np.arange(k))

    def test_ransac_identical_to_repository_and_no_gcp_argument(self):
        from scripts.mrsi_gt_metrics import score_matches
        rng=np.random.default_rng(8);a=rng.uniform(4,60,(80,2));b=a+[2.,-1.];b[:12]=rng.uniform(0,60,(12,2))
        H,n,f=fit_matches(a,b)
        old=score_matches(a,b,[64,64],[64,64],np.array([[1.,0,2],[0,1,-1],[0,0,1.]]))
        self.assertIsNone(f);self.assertEqual(n,old['ransac_inliers'])
        np.testing.assert_array_equal(H,np.array(old['H_estimated']))
        self.assertIsNotNone(fit_matches(a[:3],b[:3])[2])

    def test_failures_remain_in_sr_denominator(self):
        good=dict(ground_rmse=2.,ground_median_error=2.,ground_p90_error=2.,errors=[2.],ground_sr={'1':False,'3':True,'5':True},fit_failure=None)
        bad=dict(ground_sr={'1':False,'3':False,'5':False},fit_failure='missing_saved_matches')
        summary=aggregate([good,bad]);self.assertEqual(summary['ground_sr']['3'],.5)
        self.assertEqual(summary['valid_pairs'],1);self.assertEqual(summary['failed_pairs'],1)

    def test_end_to_end_evaluation_provenance_missing_pair_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=dataset(root/'data')
            store=Store(Catalog(root/'data'),root/'labels',4)
            ids=[r['pair_id'] for r in rows[:2]]
            for pair_id in ids:
                doc=store.load(pair_id);doc.update(points=[point(a=(20.,20.),b=(22.,23.)),point(2,a=(30.,35.),b=(32.,38.))],status='completed')
                store.save(pair_id,doc)
            entry,a,b=prediction(root/'matches',ids[0])
            write_json(root/'matches/match_index.json',[entry])
            report=evaluate(store.annotations,root/'matches',root/'eval',root/'data')
            self.assertEqual(report['evaluation'],'manual_ground_gcp')
            self.assertAlmostEqual(report['overall']['Full']['ground_rmse_pair_mean'],0,places=5)
            self.assertEqual(report['overall']['Full']['ground_sr']['1'],.5)
            detailed=[json.loads(l) for l in (root/'eval/pairs.jsonl').read_text().splitlines()]
            self.assertEqual(len(detailed),8)
            H1=detailed[0]['H_estimated']
            doc=store.load(ids[0]);doc['points'][0]['b'][0]+=4;store.save(ids[0],doc)
            evaluate(store.annotations,root/'matches',root/'eval2',root/'data')
            other=json.loads((root/'eval2/pairs.jsonl').read_text().splitlines()[0])
            self.assertEqual(H1,other['H_estimated'])  # Changing human truth cannot change fitted H.
            self.assertGreater(other['ground_rmse'],1.)
            with self.assertRaises(ValueError):evaluate(store.annotations,root/'matches',root/'eval',root/'data')
            with self.assertRaises(ValueError):evaluate(store.annotations,root/'matches',root/'matches/new',root/'data')
            with self.assertRaises(ValueError):evaluate(store.annotations,root/'matches',root/'eval3',root/'data',reviewed_only=True)

    def test_index_uses_explicit_pair_id_not_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entry,_,_=prediction(root,'identity','unrelated-file-name.npz')
            write_json(root/'match_index.json',[entry]);mapping,_=match_mapping(root)
            self.assertEqual(mapping['identity'].name,'unrelated-file-name.npz')
            write_json(root/'match_index.json',[entry,entry])
            with self.assertRaises(ValueError):match_mapping(root)

    def test_coordinate_size_gt_type_and_image_integrity_guards(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=dataset(root/'data')
            store=Store(Catalog(root/'data'),root/'labels',4)
            pair=rows[0]['pair_id'];doc=store.load(pair)
            doc.update(points=[point()],status='completed');saved=store.save(pair,doc)
            entry,_,_=prediction(root/'matches',pair)
            write_json(root/'matches/match_index.json',[entry])
            write_json(root/'plan.json',dict(input_size=784))
            with self.assertRaisesRegex(ValueError,'coordinate size'):
                evaluate(store.annotations,root/'matches',root/'wrong_size',root/'data')
            (root/'plan.json').unlink()
            saved['ground_truth_kind']='synthetic_homography'
            write_json(store.annotations/(pair+'.json'),saved)
            with self.assertRaisesRegex(ValueError,'synthetic H GT is forbidden'):
                evaluate(store.annotations,root/'matches',root/'wrong_type',root/'data')
            saved['ground_truth_kind']='manual_ground_gcp'
            write_json(store.annotations/(pair+'.json'),saved)
            Image.new('RGB',(64,64),(255,0,0)).save(root/'data/val'/rows[0]['image_A'])
            with self.assertRaisesRegex(ValueError,'image changed'):
                evaluate(store.annotations,root/'matches',root/'wrong_image',root/'data')


if __name__=='__main__':unittest.main()
