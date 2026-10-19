import json
from pathlib import Path

import numpy as np
from PIL import Image


def dataset(root, count=8):
    root = Path(root)
    (root / 'val/images').mkdir(parents=True)
    native = root / 'native'
    native.mkdir()
    rows = []
    for i in range(count):
        pair_id = f'googleearth_eval__{i:03d}'
        row = dict(pair_id=pair_id, source_dataset='googleearth_evaluation_cross_temporal',
                   size_A=[64, 64], size_B=[64, 64], target_image_size=[128, 128], source_image_size=[128, 128],
                   T_source_to_A=[[63/127,0,0],[0,63/127,0],[0,0,1]],
                   T_source_to_B=[[.7,.04,-10],[.02,.7,-8],[.0002,.0001,1]],
                   area_ratio_A_to_B=2.5 if i%2 else 3.5)
        for side, key in [('A','target_path'),('B','source_path')]:
            path = native / f'{pair_id}_{side}.png'
            Image.new('RGB', (128,128), (45+i,70,80)).save(path)
            row[key] = str(path)
            name = f'images/{pair_id}_{side}.png'
            row['image_'+side] = name
            Image.new('RGB',(64,64),(30,60+i,90)).save(root/'val'/name)
        rows.append(row)
    (root/'val/pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return rows


def point(i=1, a=(20.125,30.25), b=(24.5,32.75)):
    return dict(id=f'GCP-{i:03d}', a=list(a), b=list(b), type='ground', needs_review=False)


def prediction(root, pair_id, filename='00000.npz', shift=(2.,3.), n=48):
    root = Path(root)
    (root/'matches').mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(4)
    a = rng.uniform(10,45,(n,2)).astype(np.float64)
    b = a+np.asarray(shift)
    path=root/'matches'/filename
    np.savez(path,pair_id=pair_id,points_a=a,points_b=b,confidence=np.ones(n,dtype=np.float64))
    return dict(pair_id=pair_id,npz='matches/'+filename),a,b
