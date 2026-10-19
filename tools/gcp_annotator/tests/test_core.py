import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.gcp_annotator.core import Catalog, Conflict, Store, atomic_write, coordinate, project
from tools.gcp_annotator.tests.helpers import dataset, point


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        dataset(self.root/'data')
        self.catalog=Catalog(self.root/'data')
        self.store=Store(self.catalog,self.root/'annotations',4,42)
        self.pair=self.catalog.rows[0]['pair_id']

    def test_native_test_roundtrip_subpixel(self):
        row=self.catalog.row(self.pair)
        rng=np.random.default_rng(42)
        for side in ['A','B']:
            H=np.asarray(row['T_source_to_'+side])
            for p in rng.uniform(10,110,(100,2)):
                np.testing.assert_allclose(project(project(p,H),np.linalg.inv(H)),p,rtol=0,atol=1e-11)

    def test_pixel_center_edges_and_partial_point_rejection(self):
        doc=self.store.load(self.pair)
        doc['points']=[point(a=(0.,0.),b=(63.,63.))]
        saved=self.store.save(self.pair,doc)
        self.assertEqual(saved['points'][0]['native_a'],[0.,0.])
        saved['points'][0].pop('b')
        with self.assertRaises(ValueError):self.store.save(self.pair,saved)
        saved=self.store.load(self.pair);saved['draft']={'a':[2.,3.],'b':[4.,5.]}
        with self.assertRaises(ValueError):self.store.save(self.pair,saved)

    def test_schema_and_numeric_guards(self):
        for value in [[-1,0],[0,64],[float('nan'),3],[float('inf'),2],[True,2],['3',2],[2]]:
            with self.subTest(value=value),self.assertRaises(ValueError):coordinate(value,[64,64])
        for points in [[point(),point()], [dict(id='GCP-001',a=[2,3])], [point(b=[999,2])]]:
            doc=self.store.load(self.pair);doc['points']=points
            with self.assertRaises(ValueError):self.store.save(self.pair,doc)
        doc=self.store.load(self.pair);doc['status']='skipped'
        with self.assertRaises(ValueError):self.store.save(self.pair,doc)

    def test_draft_save_restart_export_and_conflict(self):
        doc=self.store.load(self.pair);doc['draft']={'a':[12.125,13.875]};doc['status']='in_progress'
        saved=self.store.save(self.pair,doc)
        restarted=Store(Catalog(self.root/'data'),self.root/'annotations',6,7)
        self.assertEqual(restarted.load(self.pair),saved)
        self.assertEqual(restarted.selection,self.store.selection)
        self.assertEqual(restarted.export()['pairs'],0)
        self.assertEqual((restarted.output/'exports/ground_gcps.jsonl').read_text(),'')
        with self.assertRaises(Conflict):self.store.save(self.pair,doc)
        saved['draft']=None;saved['points']=[point()];saved['status']='completed'
        final=self.store.save(self.pair,saved)
        self.assertEqual(final['points'][0]['a'],[20.125,30.25])
        self.assertEqual(self.store.export()['points'],1)
        self.assertEqual(len(list((self.store.output/'history'/self.pair).glob('*.json'))),2)
        backup=json.loads((self.store.annotations/(self.pair+'.json.bak')).read_text())
        self.assertEqual(backup['revision'],1)
        exported=json.loads((self.store.output/'exports/ground_gcps.jsonl').read_text())
        self.assertEqual(exported['points'],final['points'])

    def test_atomic_failure_leaves_previous_file_intact(self):
        path=self.root/'atomic.json';atomic_write(path,'original')
        with patch('tools.gcp_annotator.core.os.replace',side_effect=OSError('disk error')):
            with self.assertRaises(OSError):atomic_write(path,'new')
        self.assertEqual(path.read_text(),'original')
        self.assertEqual(list(self.root.glob('.atomic.json.*.tmp')),[])

    def test_sampling_balanced_reproducible_prefix_and_manifest_guard(self):
        other=Store(self.catalog,self.root/'other',4,42)
        self.assertEqual(other.selection['pair_ids'],self.store.selection['pair_ids'])
        self.assertEqual(set(other.selection['strata'].values()),{2})
        old=self.store.selection['pair_ids'][:]
        self.store.extend(7)
        self.assertEqual(self.store.selection['pair_ids'][:4],old)
        with self.assertRaises(ValueError):self.store.extend(3)
        manifest=self.root/'data/val/pairs.jsonl';manifest.write_text(manifest.read_text()+'\n')
        with self.assertRaises(ValueError):Store(Catalog(self.root/'data'),self.root/'annotations',4)

    def test_completion_review_and_invalidation(self):
        doc=self.store.load(self.pair);doc['points']=[point()];doc['points'][0]['needs_review']=True;doc['status']='completed'
        with self.assertRaises(ValueError):self.store.save(self.pair,doc)
        doc['points'][0]['needs_review']=False
        saved=self.store.save(self.pair,doc)
        saved['review']={'status':'approved','reviewer':'reviewer','checked_ids':[]}
        with self.assertRaises(ValueError):self.store.save(self.pair,saved)
        saved['review']['checked_ids']=['GCP-001']
        approved=self.store.save(self.pair,saved)
        self.assertEqual(approved['review']['status'],'approved')
        approved['points'][0]['a'][0]+=.25
        revised=self.store.save(self.pair,approved)
        self.assertEqual(revised['review']['status'],'unreviewed')
        self.assertEqual(revised['review']['checked_ids'],[])


if __name__=='__main__':unittest.main()
