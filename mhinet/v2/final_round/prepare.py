"""Fix source/spatially separated lists before any new model evaluation."""
import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT=Path('outputs/d16_final_round')
HZ=Path('/home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4')
RGB=Path('/home/disk1/Data/datasets/MHINet_RGB_512')
CACHE=Path('/home/u2ssd2_TQL/mhi_v2_cache/MHINet_RGB_512')


def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def polygon(row):
    return [np.asarray(row[k],dtype=np.float64)[:-1] for k in ('footprint_A','footprint_B')]


def bounds(row):
    p=np.concatenate(polygon(row))
    return np.r_[p.min(0),p.max(0)]


def overlaps(row,others):
    a=bounds(row)
    for b,poly in others:
        if a[0]>=b[2] or a[2]<=b[0] or a[1]>=b[3] or a[3]<=b[1]:continue
        origin=a[:2]
        for p in polygon(row):
            for q in poly:
                area,_=cv2.intersectConvexConvex((p-origin).astype('float32'),(q-origin).astype('float32'))
                if area>1.:return True
    return False


def footprint_index(rows):return [(bounds(r),polygon(r)) for r in rows]


def metadata_audit():
    old=json.loads(Path('outputs/hangzhou_cleaned_restart/selection.json').read_text())
    history={r['pair_id']:r for rows in old.values() for r in rows}
    paths=list(Path('outputs/mhi_v2').glob('hangzhou*/selection.json'))+list(Path('outputs/hangzhou_frozen_locator').glob('*/selection.json'))+list(Path('outputs/hangzhou_v1_control').glob('*/selection.json'))
    for p in paths:
        value=json.loads(p.read_text())
        if isinstance(value,dict):
            for split in ('train','val'):
                for r in value.get(split,[]):
                    if isinstance(r,dict) and 'footprint_A' in r:history[r['pair_id']]=r
    history=list(history.values());blocks={r['anchor_block_id'] for r in history};bases={r['base_pair_id'] for r in history}
    index=footprint_index(history)
    result={}
    for split in ('train','val','test_controlled'):
        rows=read(HZ/'manifests'/f'{split}.jsonl')
        independent=[r for r in rows if r['anchor_block_id'] not in blocks and r['base_pair_id'] not in bases]
        separated=[r for r in independent if not overlaps(r,index)]
        result['unused_train' if split=='train' else split]=separated
        print(split,'all',len(rows),'unseen block/base',len(independent),'no footprint intersection',len(separated),
              'blocks',len({r['anchor_block_id'] for r in separated}),'groups',dict(Counter(r['G'] for r in separated)),flush=True)
    (ROOT/'candidate_pool.json').write_text(json.dumps(dict(history=history,**result),ensure_ascii=False))
    print('history pairs',len(history),'blocks',len(blocks),'history paths',len(paths),flush=True)





def balanced(rows, count, seed, key='G'):
    rng=random.Random(seed);groups=sorted({r[key] for r in rows});chosen=[]
    for i,g in enumerate(groups):
        subset=[r for r in rows if r[key]==g];rng.shuffle(subset)
        by_block={}
        for r in subset:by_block.setdefault(r.get('anchor_block_id',r['pair_id']),[]).append(r)
        names=list(by_block);rng.shuffle(names);need=count//len(groups)+(i<count%len(groups))
        n=0
        while n<need:
            progress=False
            for b in names:
                if by_block[b] and n<need:
                    chosen.append(by_block[b].pop());n+=1;progress=True
            if not progress:raise ValueError(f'Insufficient {g}: {n}/{need}')
    return chosen


def finalize():
    if (ROOT/'selection.json').exists():raise FileExistsError('Frozen selection already exists; do not redraw it after model evaluation.')
    pool=json.loads((ROOT/'candidate_pool.json').read_text());old=json.loads(Path('outputs/hangzhou_cleaned_restart/selection.json').read_text())
    available=pool['val']+pool['unused_train']
    blocks=sorted({r['anchor_block_id'] for r in available});random.Random(20261023).shuffle(blocks)
    first=set(blocks[:len(blocks)//2])
    cal=balanced([r for r in available if r['anchor_block_id'] in first],128,20261023)
    index=footprint_index(cal);bases={r['base_pair_id'] for r in cal}
    operational=balanced([r for r in available if r['anchor_block_id'] not in first and r['base_pair_id'] not in bases and not overlaps(r,index)],128,20261024)
    reserved=cal+operational;index=footprint_index(reserved);blocks={r['anchor_block_id'] for r in reserved};bases={r['base_pair_id'] for r in reserved}
    test=balanced([r for r in pool['test_controlled'] if r['anchor_block_id'] not in blocks and r['base_pair_id'] not in bases and not overlaps(r,index)],256,20261025)
    selected=dict(hz_train=old['train'],hz_dev=old['val'],hz_calibration=cal,hz_operating=operational,hz_test=test)
    sources={};rgb_audit={}
    for split,n,role in [('train',1024,'rgb_train'),('val',256,'rgb_dev'),('test',256,'rgb_test')]:
        rows=read(RGB/split/'pairs.jsonl');identities={(r['source_dataset'],r['source_relative_path']) for r in rows}
        for prev,ids in sources.items():assert not identities & ids,(split,prev)
        sources[split]=identities
        subset=balanced(rows,n,20261023,key='source_dataset');errors=[];inverse=[]
        for r in subset:
            h=np.array(r['H_A_to_B']);expected=np.array(r['T_source_to_B'])@np.linalg.inv(np.array(r['T_source_to_A']));expected/=expected[2,2]
            errors.append(float(np.max(np.abs(h/h[2,2]-expected))))
            back=np.array(r['H_B_to_A'])@h;back/=back[2,2];inverse.append(float(np.abs(back-np.eye(3)).max()))
            for k in ['image_A','image_B','mask_A_overlap','mask_B_overlap']:r[k]=split+'/'+r[k]
            r['G']=r['source_dataset']
        assert max(errors)<1e-6 and max(inverse)<1e-6
        selected[role]=subset;rgb_audit[split]=dict(manifest_rows=len(rows),unique_sources=len(identities),selected=len(subset),
            max_H_composition_difference=max(errors),max_inverse_difference=max(inverse))
    paths=dict(hz=str(HZ),rgb_train=str(CACHE),rgb_dev=str(CACHE),rgb_test=str(RGB))
    report=dict(seed=20261023,counts={k:len(v) for k,v in selected.items()},rgb_geometry=rgb_audit,
        calibration_original_splits=dict(Counter(r['split'] for r in cal)),operating_original_splits=dict(Counter(r['split'] for r in operational)),
        calibration_blocks=len({r['anchor_block_id'] for r in cal}),operating_blocks=len({r['anchor_block_id'] for r in operational}),
        test_blocks=len({r['anchor_block_id'] for r in test}),
        change_from_plan='Old experiments touched 415 HZ pairs in 168 blocks. Only75 val pairs remained spatially separate; calibration/operating reserve previously unused original train regions as well. They are excluded from all new training.',
        exposure='RGB original test split has 5106 source-disjoint rows. Existing v2 configs documented train/val-only use. DINO external pretraining exposure cannot be certified; RGB source-disjoint does not imply geographic-neighbour independence.',
        spatial_rule='Different anchor blocks and base_pair_id; no >1m2 convex footprint intersection (A or B) across historic use/calibration/operating/test. Within-split spatial dependence retained and block-bootstrap used.')
    for role,rows in selected.items():
        root=Path(paths['hz'] if role.startswith('hz_') else paths[role])
        assert all((root/r['image_A']).is_file() and (root/r['image_B']).is_file() for r in rows),role
    (ROOT/'selection.json').write_text(json.dumps(dict(roots=paths,**selected),ensure_ascii=False))
    (ROOT/'data_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');args=p.parse_args()
    finalize() if args.select else metadata_audit()
