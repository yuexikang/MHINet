from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from tools.gcp_annotator.app import create_app
from tools.gcp_annotator.tests.helpers import dataset, point


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        dataset(self.root/'data')
        self.args=dict(root=self.root/'data',output=self.root/'out',sample_size=4)
        self.client=TestClient(create_app(**self.args))
        self.pair=self.client.get('/api/catalog').json()['selection']['pair_ids'][0]
        self.url='/api/pairs/'+self.pair

    def test_complete_save_restart_export_flow(self):
        self.assertEqual(self.client.get('/').status_code,200)
        doc=self.client.get(self.url).json()['annotation']
        for side in ['A','B']:
            for mode in ['native','test']:
                r=self.client.get(self.url+'/image/'+side+'/'+mode)
                self.assertEqual(r.status_code,200);self.assertTrue(r.headers['content-type'].startswith('image/'))
        doc.update(points=[point()],status='completed')
        saved=self.client.put(self.url,json=doc)
        self.assertEqual(saved.status_code,200,saved.text)
        self.assertEqual(self.client.put(self.url,json=doc).status_code,409)
        restarted=TestClient(create_app(**self.args))
        self.assertEqual(restarted.get(self.url).json()['annotation'],saved.json())
        exported=restarted.post('/api/export').json();self.assertEqual(exported['pairs'],1)
        self.assertEqual(restarted.get(exported['csv']).status_code,200)
        self.assertEqual(restarted.get(exported['jsonl']).status_code,200)
        self.assertEqual(restarted.get('/api/catalog').json()['counts']['completed'],1)

    def test_two_writers_cannot_overwrite(self):
        doc=self.client.get(self.url).json()['annotation'];doc['points']=[point()]
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes=list(pool.map(lambda _:self.client.put(self.url,json=doc).status_code,range(2)))
        self.assertEqual(sorted(codes),[200,409])

    def test_invalid_side_pair_and_origin(self):
        self.assertEqual(self.client.get(self.url+'/image/C/native').status_code,422)
        self.assertEqual(self.client.get('/api/pairs/unknown').status_code,404)
        doc=self.client.get(self.url).json()['annotation']
        self.assertEqual(self.client.put(self.url,json=doc,headers={'Origin':'https://unrelated.test'}).status_code,403)
        self.assertEqual(self.client.get('/api/exports/other').status_code,404)


if __name__=='__main__':unittest.main()
