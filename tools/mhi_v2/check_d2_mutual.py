"""Replay archived diagnostic pairs through the released optional D2 filter."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mhinet.v2.pipeline import load_mhi_v2
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.pair_matching.local import infer
from mhinet.v2.pair_matching.metrics import measure,totals


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:2')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    model=load_mhi_v2(device=args.device,runtime=json.loads(args.runtime.read_text()),assets_root=args.assets_root)
    report=dict(release=model.release,tests={})
    for split,root,manifest in (
        ('rgb',Path('/home/disk1/Data/datasets/MHINet_RGB_512/test'),'pairs.jsonl'),
        ('hangzhou',Path('/home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4'),'manifests/test_real.jsonl')):
        archived=[json.loads(s) for s in (args.reference/f'{split}_pairs.jsonl').read_text().splitlines()]
        selected=archived[::max(1,len(archived)//16)][:16]
        records=[json.loads(s) for s in (root/manifest).read_text().splitlines() if s.strip()]
        data=GeometryPairDataset(root,[records[r['index']] for r in selected])
        rows=[]
        for i in range(len(data)):
            item=data[i]
            assert item['pair_id'][0]==selected[i]['pair_id']
            shared=model.shared(item['images'].to(args.device),pyramid_scales=(8,4,2))
            base=model.match_shared(shared,diagnostics=True)
            filtered=model.match_shared(shared,mutual=True,diagnostics=True)
            trace=base['diagnostics'];lp=trace['candidate_logp'];best=lp.argmax(-1)
            # Independent original diagnostic rule, retained as the test oracle.
            expected=lp.argmax(-2).gather(1,best)==torch.arange(16,device=lp.device)[None]
            expected &= trace['candidate_valid'].gather(2,best[...,None]).squeeze(-1)
            expected=expected.flatten() & base['valid']
            for xy in (base['points_a'],base['points_b']):
                expected &= torch.isfinite(xy).all(-1)&(xy>=0).all(-1)&(xy<=511).all(-1)
            torch.testing.assert_close(filtered['diagnostics']['output_mask'],expected,rtol=0,atol=0)
            for key in ('points_a','points_b','confidence','valid','parent','parent_ids'):
                torch.testing.assert_close(filtered[key],base[key][expected],rtol=0,atol=0)
            other=infer(shared,model.matcher,model.adapter,mutual=True)
            for key in other:torch.testing.assert_close(other[key],filtered[key],rtol=0,atol=0)
            counts={}
            for name,result in [('d2',base),('mutual',filtered)]:
                counts[name]=measure(result['points_a'],result['points_b'],item['geometry'].to(args.device))[0]
                for k in ('n','visible','correct1','correct3','correct5','correct10'):
                    assert counts[name][k]==selected[i][name][k],(split,i,name,k)
            rows.append(dict(pair_id=selected[i]['pair_id'],**counts))
        report['tests'][split]=dict(pairs=rows,base=totals([r['d2'] for r in rows]),mutual=totals([r['mutual'] for r in rows]))
        print(f'{split}: 16 pairs passed; archived counts and exact subset/lineage agree',flush=True)
    report['peak_gib']=torch.cuda.max_memory_allocated(args.device)/2**30
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
