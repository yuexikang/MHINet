"""No-training, same-source window and soft-coordinate diagnostics."""
import json
from pathlib import Path
import torch
from mhinet.v2.qrru_corners.experiment import selections,load_models
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.flow_d8_d2.run import measure
from .path import infer,fine


@torch.no_grad()
def main():
    torch.set_num_threads(2);device='cuda:2';model,matcher=load_models(device)
    model.eval();matcher.eval()
    output={}
    for domain,spec in selections().items():
        dataset=GeometryPairDataset(spec['root'],spec['val'][::4]);rows=[]
        for batch in dataset:
            gt=batch['geometry'].to(device);shared=model(batch['images'].to(device));r=infer(shared,matcher)
            a=r['fine_source'];target,known,visible=gt.query((a+.5)/256-1);truth=(target+1)*256-.5
            def stats(result):
                b=result['p2'];c=result['candidates2'];v=result['valid2']
                lo=c.masked_fill(~v[...,None],torch.inf).amin(1);hi=c.masked_fill(~v[...,None],-torch.inf).amax(1)
                inside=(truth>=lo).all(-1)&(truth<=hi).all(-1)&known&visible
                nearest=(c-truth[:,None]).norm(dim=-1).masked_fill(~v,torch.inf).amin(-1)
                hard=c[torch.arange(len(a),device=device),result['logits2'].argmax(-1)]
                lp=result['logits2'].reshape(-1,16,16)
                best=lp.argmax(-1);reverse=lp.argmax(-2);ids=torch.arange(16,device=device)[None]
                mutual=(reverse.gather(1,best)==ids).flatten() & v.gather(1,best.flatten()[:,None]).flatten()
                return dict(soft=measure(a,b,gt)[0],hard=measure(a,hard,gt)[0],mutual=measure(a[mutual],b[mutual],gt)[0],n=len(a),visible=int(visible.sum()),
                    covered=int(inside.sum()),nearest_p3=int(((nearest<=3)&known&visible).sum()))
            normal=stats(r)
            center,_,_=gt.query((r['source']+.5)/256-1)
            c={k:r[k] for k in ('source','p0','p8','score8','candidates8','logits8','valid8')}
            oracle=fine(shared,matcher,{**c,'p8':(center+1)*256-.5})
            rows.append(dict(pair_id=batch['pair_id'][0],normal=normal,oracle_parent=stats(oracle)))
        summary={}
        for variant in ('normal','oracle_parent'):
            x=[r[variant] for r in rows];n=sum(v['n'] for v in x);vis=sum(v['visible'] for v in x)
            summary[variant]=dict(coverage=sum(v['covered'] for v in x)/max(1,vis),
                nearest_p3=sum(v['nearest_p3'] for v in x)/n,
                soft_p3=sum(v['soft']['correct3'] for v in x)/n,
                hard_p3=sum(v['hard']['correct3'] for v in x)/n,
                mutual_p3=sum(v['mutual']['correct3'] for v in x)/max(1,sum(v['mutual']['n'] for v in x)),
                mutual_points_per_pair=sum(v['mutual']['n'] for v in x)/len(x))
        output[domain]=dict(summary=summary,pairs=rows)
        print(domain,summary,flush=True)
    Path('outputs/joint_diagnosis/windows.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
