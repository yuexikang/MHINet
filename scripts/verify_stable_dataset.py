"""Full manifest/geometry/file-header audit, plus stratified decoded mask audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from mhinet.dataio.stable_geometry import check_geometry


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(item):
    base,r=item
    assert json.loads((base/r['metadata']).read_text())==r
    for key in ('image_A','image_B','mask_A_overlap','mask_B_overlap','mask_A_geometry','mask_B_geometry','mask_A_visible','mask_B_visible'):
        f=base/r[key]
        with f.open('rb') as stream:header=stream.read(24)
        assert header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR'
        assert list(struct.unpack('>II',header[16:24]))==r['size_'+('A' if '_A' in key else 'B')]
    return 8


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--old-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.monotonic(); root=args.root
    summary=json.loads((root/'dataset_summary.json').read_text())
    assert summary['status']=='completed' and not summary['smoke_only']
    parents=json.loads((root/'parent_split_manifest.json').read_text())
    assert parents==json.loads((args.old_root/'parent_split_manifest.json').read_text())
    report={'status':'passed','root':str(root),'splits':{},'limitations':'All PNG headers checked; full pixel decoding and exact mask reconstruction only on stratified sample. No model accuracy evaluation.'}
    sets={}
    for split in ('train','val'):
        ids=set();geo=set();mother=set(); counts=Counter(); recipes=Counter(); sample={}; extrema=[]
        expected={g['group_id']:g for g in parents[split]}
        path=root/split/'pairs.jsonl'; max_error=0.; checked=0; file_batch=[]
        pool=ThreadPoolExecutor(max_workers=128)
        for line in path.open():
            r=json.loads(line); pid=r['pair_id']; assert pid not in ids;ids.add(pid)
            assert r['split']==split and r['parent_image_A']==r['parent_image_B']
            group=expected[r['parent_temporal_group']]
            assert r['geo_group']==group['geo_group'] and r['parent_image_A'] in (group['past'],group['current'])
            geo.add(r['geo_group']);mother.add(r['parent_image_A']);tier=r['tier'];counts[tier]+=1
            recipes[(r['parent_image_A'],tier)]+=1
            H=np.array(r['H_A_to_B']); inverse=np.array(r['H_B_to_A'])
            composed=np.array(r['T_source_to_B'])@np.linalg.inv(np.array(r['T_source_to_A']));composed/=composed[2,2]
            error=float(np.abs(composed-H).max());max_error=max(error,max_error)
            assert np.allclose(composed,H,rtol=1e-9,atol=1e-8)
            assert np.allclose(H@inverse,np.eye(3),rtol=1e-8,atol=1e-8)
            ok,detail=check_geometry(H,r['size_A'],r['size_B']);assert ok,pid
            extrema.extend(detail['directions'])
            assert min(r['geometric_overlap_fractions'])>=.5
            assert min(r['visible_overlap_fractions'])>=(.3 if tier==3 else .5)
            assert -30<=r['rotation_B_degrees']<=30
            assert bool(r['photometric'][0])==(tier>=2) and bool(any(r['occlusion']))==(tier==3)
            for occ in r['occlusion']:
                if occ:assert .1<=occ['area_fraction']<=.3
            file_batch.append((root/split,r))
            if len(file_batch)>=1000:
                checked+=sum(pool.map(check_files,file_batch));file_batch=[]
            key=(tier,r['resolution_ratio']);sample.setdefault(key,[])
            if len(sample[key])<10:sample[key].append(r)
            if len(ids)%10000==0:print(split,len(ids),flush=True)
        checked+=sum(pool.map(check_files,file_batch));pool.shutdown()
        assert len(ids)==summary['planned'][split]['generated_pairs']==14*len(expected)
        assert all(n=={1:3,2:2,3:2}[t] for (_,t),n in recipes.items())
        assert len(recipes)==len(expected)*2*3
        decoded=0
        for rows in sample.values():
            for r in rows:
                for side,other,matrix in [('A','B',r['H_B_to_A']),('B','A',r['H_A_to_B'])]:
                    image=cv2.imread(str(root/split/r['image_'+side]));assert image is not None
                    own=cv2.imread(str(root/split/r[f'mask_{side}_visible']),0)
                    target=cv2.imread(str(root/split/r[f'mask_{other}_visible']),0)
                    overlap=cv2.imread(str(root/split/r[f'mask_{side}_overlap']),0)
                    geometry=cv2.imread(str(root/split/r[f'mask_{side}_geometry']),0)
                    assert all(v is not None for v in (own,target,overlap,geometry))
                    assert set(np.unique(own))<=set((0,255))
                    size=tuple(r['size_'+side]);M=np.array(matrix)
                    expected_mask=(own>0)&(cv2.warpPerspective(target,M,size,flags=cv2.INTER_NEAREST)>0)
                    assert np.array_equal(overlap,expected_mask.astype(np.uint8)*255)
                    expected_geo=cv2.warpPerspective(np.ones_like(target),M,size,flags=cv2.INTER_NEAREST)
                    assert np.array_equal(geometry,expected_geo*255)
                    index=0 if side=='A' else 1
                    assert abs(expected_mask.mean()-r['visible_overlap_fractions'][index])<1e-10
                decoded+=1
        sets[split]=(geo,mother,ids)
        report['splits'][split]=dict(pairs=len(ids),tiers=dict(counts),png_headers_checked=checked,
            decoded_pairs=decoded,manifest_sha256=digest(path),max_composition_error=max_error,
            max_jacobian_bound=max(x['jacobian_upper_bound'] for x in extrema),
            max_centered_extent=max(x['centered_extent'] for x in extrema),
            min_horizon_distance=min(x['horizon_distance'] for x in extrema if x['horizon_distance'] is not None))
    assert all(not (a&b) for a,b in zip(sets['train'],sets['val']))
    assert (root/'test/pairs.jsonl').read_bytes()==(args.old_root/'test/pairs.jsonl').read_bytes()
    report.update(parent_split_exact_match=True,train_val_overlap=0,test_manifest_unchanged=True,elapsed_seconds=time.monotonic()-start)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
