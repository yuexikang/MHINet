"""Locate scalar EPE differences on identical cached features; no training."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from mhinet.v2.pipeline import load_mhi_v2
from mhinet.v2.pair_matching.evaluate import dataset
from mhinet.v2.pair_matching.metrics import measure
from mhinet.v2.pair_matching.local import infer

parser=argparse.ArgumentParser(description=__doc__)
for name in ('assets-root','runtime','rgb-root','hangzhou-root'):
 parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--device',default='cuda:0')
parser.add_argument('--index',type=int)
parser.add_argument('--output',type=Path)
a=parser.parse_args();root=a.assets_root
runtime=json.loads(a.runtime.read_text())
torch.set_num_threads(2)
model=load_mhi_v2(device=a.device,runtime=runtime,assets_root=root)
rgb=a.rgb_root
hz=a.hangzhou_root
ds=dict(test=dataset(rgb/'test',rgb/'test/pairs.jsonl'),hangzhou_test_real=dataset(hz,hz/'manifests/test_real.jsonl'))
report=[]
with torch.inference_mode():
 for split,data in ds.items():
  changes=json.loads((root/f'outputs/mhi_v2_integration/{split}_pair_differences.json').read_text())
  old=json.loads((root/f'outputs/full_rgb_d2_10epoch_20261028/d2_best_test_{split}.json').read_text())['pairs']
  for row in changes:
   if a.index is not None and row['index'] != a.index:continue
   item=data[row['index']];shared=model.shared(item['images'].to(a.device))
   gt=item['geometry'].to(a.device);runs=[];base=None;order0=None;reorders=0
   for iteration in range(32):
    result=model.match_shared(shared) if iteration%2==0 else infer(shared,model.matcher,model.adapter);counts,error,good=measure(result['points_a'],result['points_b'],gt)
    order=(result['points_a'][:,1]*512+result['points_a'][:,0]).argsort()
    aligned=torch.cat((result['points_a'],result['points_b']),1)[order]
    if base is None:base=aligned.clone();order0=order.clone()
    torch.testing.assert_close(aligned,base,rtol=0,atol=0)
    reorders+=int(not torch.equal(order,order0))
    runs.append(dict(error_sum=counts['error_sum'],canonical_fp64_sum=float(error[order][good[order]].double().sum())))
   totals=sorted({r['error_sum'] for r in runs});canonical=sorted({r['canonical_fp64_sum'] for r in runs})
   previous=old[row['index']]['local_flow']['error_sum']
   entry=dict(paths='alternating formal output and original infer kernel',split=split,pair_id=row['pair_id'],index=row['index'],runs=32,coordinates_equal_after_alignment=True,
              reorders=reorders,fp32_error_sums=totals,canonical_fp64_sums=canonical,
              original_error_sum=previous,original_sum_observed=previous in totals)
   report.append(entry);print(json.dumps(entry),flush=True)
(a.output or root/'outputs/mhi_v2_integration/reduction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
