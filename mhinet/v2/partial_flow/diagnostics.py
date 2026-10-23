"""Frozen-U0 feature extraction and per-query diagnostics, no model training."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from mhinet.v2.u0 import load_u0
from mhinet.v2.data import GeometryPairDataset
from .dataset import load_config
from .labels import support_labels, UNKNOWN
from .nomatch_head import query_features


def extract(shard, shards, device):
    torch.set_num_threads(4)
    config = load_config(); out = Path(config['output'])
    rows = [json.loads(s) for s in (out/'pairs.jsonl').read_text().splitlines()]
    rows = [r for r in rows if not (r['split']=='train' and r['kind']=='full')]
    model = load_u0(device=device)
    cache = out/'cache'; cache.mkdir(exist_ok=True)
    with gzip.open(out/f'queries_{shard}.csv.gz','wt') as stream:
        writer = None
        for index,row in enumerate(rows[shard::shards]):
            images = torch.stack([torch.from_numpy(np.array(Image.open(row['image_'+s]).convert('RGB'))).permute(2,0,1).float()/255 for s in ('a','b')])[None].to(device)
            directions = ('ab',) if row['split']=='hz' else ('ab','ba')
            for direction in directions:
                with torch.no_grad():
                    result = model(images if direction=='ab' else images.flip(1),pyramid_scales=())
                    features,signals = query_features(result,model.stage1_head.no_match_logit)
                features = features[0].cpu()
                signals = {k:v[0].cpu().numpy() for k,v in signals.items()}
                pred = ((result['flow16_target_norm'][0].permute(1,2,0).reshape(-1,2)+1)*256-.5).cpu().numpy()
                if row['split']=='hz':
                    from mhinet.v2.d16_experiment.supervision import grid
                    sample = GeometryPairDataset(config['hangzhou_root'],[row['reference_row']])[0]
                    source = grid(torch.device('cpu'))
                    reference,known,visible = sample['geometry'].query((source+.5)/256-1)
                    reference = ((reference+1)*256-.5).numpy()
                    reference[~(known&visible).numpy()] = np.nan
                    labels = dict(source_xy=source.numpy(),gt_target_xy=np.full((1024,2),np.nan,np.float32),state=np.full(1024,UNKNOWN,np.int8),negative_kind=np.zeros(1024,np.int8))
                else:
                    labels = support_labels(np.array(row['H_A_to_B']),row['rectangle'],direction=='ba')
                    reference = np.full((1024,2),np.nan,np.float32)
                epe = np.linalg.norm(pred-labels['gt_target_xy'],axis=-1)
                ref_epe = np.linalg.norm(pred-reference,axis=-1)
                record = dict(row=row,direction=direction,features=features,**labels,**signals,
                    pred_target_xy=pred,epe_px=epe,reference_target_xy=reference,reference_epe_px=ref_epe)
                torch.save(record,cache/f'{row["pair_id"]}_{direction}.pt')
                for i in range(1024):
                    item = dict(pair_id=row['pair_id'],direction=direction,split=row['split'],kind=row['kind'],mother_id=row['mother_id'],
                        query_index=i,source_x=float(labels['source_xy'][i,0]),source_y=float(labels['source_xy'][i,1]),
                        gt_state=('MATCH','NO_MATCH','UNKNOWN')[labels['state'][i]],negative_kind=int(labels['negative_kind'][i]))
                    for name,values in [('gt_target',labels['gt_target_xy']),('pred_target',pred),('reference_target',reference)]:
                        for dim,suffix in enumerate(('x','y')):
                            item[name+'_'+suffix] = float(values[i,dim]) if np.isfinite(values[i,dim]) else ''
                    for name,values in dict(signals,epe_px=epe,reference_epe_px=ref_epe).items():
                        item[name] = float(values[i]) if np.isfinite(values[i]) else ''
                    if writer is None:
                        writer = csv.DictWriter(stream,fieldnames=list(item)); writer.writeheader()
                    writer.writerow(item)
            if index%10==0: print(f'shard{shard}: {index+1}/{len(rows[shard::shards])} {row["pair_id"]}',flush=True)
    (out/f'cache_done_{shard}.json').write_text(json.dumps(dict(shard=shard,pairs=len(rows[shard::shards]),u0_parameters_frozen=all(not p.requires_grad for p in model.parameters()),pyramid_scales=[])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();extract(a.shard,a.shards,a.device)
