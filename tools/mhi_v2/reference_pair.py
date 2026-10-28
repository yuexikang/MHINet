"""Capture the unmodified research implementation in a separate interpreter.

Run from the research checkout, passing its path explicitly. Never trains.
"""
import argparse
import json
from pathlib import Path
import sys
import os
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', default='cuda:2')
p.add_argument('--deterministic', action='store_true')
a=p.parse_args()
a.source_root=a.source_root.resolve();a.output=a.output.resolve()
os.chdir(a.source_root)
sys.path.insert(0,str(a.source_root))
import torch
from mhinet.v2.qrru_corners.experiment import load_models
from mhinet.v2.d2_modulation.model import load_adapter, extract_windows, match_windows, center_indices
from mhinet.v2.d2_modulation.full_rgb import datasets, restore
from mhinet.v2.frozen_locator import load_locator_weights
from mhinet.v2.flow_d8_d2.matching import infer as legacy_infer, select_local_parents
from mhinet.v2.joint_single.path import coarse, source_grid
from mhinet.v2.flow_d8_d2.run import measure

torch.set_num_threads(2);torch.manual_seed(42)
if a.deterministic: torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
a.output.mkdir(parents=True,exist_ok=True)
c=json.loads((a.source_root/'configs/mhi_v2/full_rgb_d2_10epoch.json').read_text())
ds,_=datasets(c)
model,matcher=load_models(a.device)
load_locator_weights(model,c['frozen_flow_checkpoint'])
adapter=load_adapter(c['initial_adapter'],a.device)
s=torch.load(a.source_root/'outputs/full_rgb_d2_10epoch_20261028/d2/best.pt',map_location=a.device,weights_only=False)
restore(s,model,matcher,adapter,'d2');del s
rows=[]
with torch.inference_mode():
 for split in ('test','hangzhou_test_real'):
  for i in range(16):
   item=ds[split][i];shared=model(item['images'].to(a.device),pyramid_scales=(8,4,2))
   w=extract_windows(shared,matcher);r=match_windows(adapter,w,matcher.tau_f)
   parents=w['source'].mean(1)
   old=legacy_infer(shared,matcher,'local')['coarse_a'][:128]
   if a.deterministic:
    torch.testing.assert_close(parents,old,rtol=0,atol=0)
   else:
    # Compare sets: separate sparse reductions can reorder near-tied parents.
    torch.testing.assert_close(parents.sort(dim=0).values,old.sort(dim=0).values,rtol=0,atol=0)
   centers=coarse(shared,matcher,parents)['p8']
   ids=((parents+.5)/8-.5).round().long();ids=ids[:,1]*64+ids[:,0]
   result=dict(points_a=w['source'].reshape(-1,2),points_b=r['points_b'].reshape(-1,2),confidence=(r['confidence']*w['score8'][:,None]).sqrt().flatten())
   counts,_,_=measure(result['points_a'],result['points_b'],item['geometry'].to(a.device))
   tensors={**result,'flow16_norm':shared['flow16_norm'],'flow16_target_norm':shared['flow16_target_norm'],'parents':parents,'parent_ids':ids,'centers':centers,**{f'window_{k}':w[k] for k in ('a','b','valid_a','valid_b','source','candidates','score8')},'candidate_logp':r['logp'],'candidate_valid':r['valid']}
   filename=f'{split}_{i:03d}.pt';torch.save({k:v.cpu() for k,v in tensors.items()},a.output/filename)
   rows.append(dict(split=split,index=i,pair_id=item['pair_id'][0],file=filename,counts=counts))
   print(split,i+1,counts['correct3'],flush=True)
(a.output/'index.json').write_text(json.dumps(dict(device=a.device,deterministic=a.deterministic,source_root=str(a.source_root),rows=rows),indent=2))
print('REFERENCE COMPLETE: 32 pairs, legacy parent selection identical',flush=True)
