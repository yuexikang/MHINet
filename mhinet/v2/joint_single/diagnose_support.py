"""Same source points and parent predictions; vary only fixed D2 support size."""
import json
from pathlib import Path
import torch
from mhinet.v2.qrru_corners.experiment import selections,load_models
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.flow_d8_d2.run import measure
from .path import infer
from .fixed_support import fine


@torch.no_grad()
def main():
    torch.set_num_threads(2);device='cuda:2';model,matcher=load_models(device);model.eval();matcher.eval()
    output={}
    for domain,spec in selections().items():
        dataset=GeometryPairDataset(spec['root'],spec['val'][::4]);rows=[]
        for batch in dataset:
            gt=batch['geometry'].to(device);shared=model(batch['images'].to(device));r=infer(shared,matcher)
            coarse={k:r[k] for k in ('source','p0','p8','score8','candidates8','logits8','valid8')}
            record=dict(pair_id=batch['pair_id'][0],variants={})
            target,known,visible=gt.query((r['fine_source']+.5)/256-1);truth=(target+1)*256-.5
            for side in (4,8,12):
                result=fine(shared,matcher,coarse,side=side);a=result['fine_source'];b=result['p2'];v=result['valid2'];c=result['candidates2']
                lower=c.masked_fill(~v[...,None],torch.inf).amin(1);upper=c.masked_fill(~v[...,None],-torch.inf).amax(1)
                covered=known&visible&(truth>=lower).all(-1)&(truth<=upper).all(-1)
                keep=result['mutual']
                record['variants'][str(side)]=dict(coverage=int(covered.sum()),visible=int(visible.sum()),
                    all=measure(a,b,gt)[0],mutual=measure(a[keep],b[keep],gt)[0])
            rows.append(record)
        summary={}
        for side in ('4','8','12'):
            x=[r['variants'][side] for r in rows]
            item=dict(coverage=sum(v['coverage'] for v in x)/max(1,sum(v['visible'] for v in x)))
            for kind in ('all','mutual'):
                total=lambda key:sum(v[kind][key] for v in x)
                item[kind]=dict(p3=total('correct3')/max(1,total('n')),points=total('n')/len(x),correct3=total('correct3')/len(x),epe=total('error_sum')/max(1,total('visible')))
            summary[side]=item
        output[domain]=dict(summary=summary,pairs=rows);print(domain,summary,flush=True)
    Path('outputs/joint_diagnosis/support_sizes.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
