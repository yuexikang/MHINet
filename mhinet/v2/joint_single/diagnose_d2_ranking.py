"""Frozen same-point D2 ranking versus decoding audit; no QRRU or training."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from mhinet.downstream.supervision import offsets
from mhinet.v2.qrru_corners.experiment import selections, load_models
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.flow_d8_d2.matching import infer as discrete
from mhinet.v2.flow_d8_d2.run import measure
from . import path
from .diagnose_attribution import summarize


def oracle_peak_decode(logits,candidates,valid,peak):
    index=torch.arange(64,device=logits.device)
    support=((index[None]%8-peak[:,None]%8).abs()<=1)&((index[None]//8-peak[:,None]//8).abs()<=1)&valid
    weights=logits.masked_fill(~support,-1e4).softmax(-1)*support
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    return (weights[...,None]*candidates).sum(1)


def summary(rows):
    result=dict(metrics=summarize(rows))
    result['counts']={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['synthetic','hangzhou'],required=True);p.add_argument('--device',required=True);args=p.parse_args()
    torch.set_num_threads(2);model,matcher=load_models(args.device);model.eval();matcher.eval()
    spec=selections()[args.domain];ds=GeometryPairDataset(spec['root'],spec['val'])
    output=Path('outputs/joint_diagnosis/d2_ranking');output.mkdir(parents=True,exist_ok=True);rows=[]
    for batch in ds:
        gt=batch['geometry'].to(args.device);shared=model(batch['images'].to(args.device))
        old=discrete(shared,matcher,'local');r=path.coarse(shared,matcher,old['coarse_a'][:128])
        a=(r['source'][:,None]+offsets(args.device).reshape(1,16,2)*2).reshape(-1,2)
        axis=(torch.arange(8,device=args.device)-3.5)*2;y,x=torch.meshgrid(axis,axis,indexing='ij')
        c=r['p8'][:,None]+torch.stack((x,y),-1).reshape(1,64,2)
        d2=shared['pyramid'][2][0].float()
        fa=F.normalize(path.sample(d2[0],a.reshape(-1,16,2)),dim=-1)
        fb=F.normalize(path.sample(d2[1],c),dim=-1)
        raw=fa@fb.transpose(-1,-2)/matcher.tau_f
        valid=(c>=0).all(-1)&(c<=511).all(-1)
        raw=raw.masked_fill(~valid[:,None],-1e4)
        dual=raw.log_softmax(-1)+raw.log_softmax(-2)
        c=c[:,None].expand(-1,16,-1,-1).reshape(-1,64,2)
        valid=valid[:,None].expand(-1,16,-1).reshape(-1,64)
        truth,known,visible=gt.query((a+.5)/256-1);truth=(truth+1)*256-.5;visible=known&visible
        distance=(c-truth[:,None]).norm(dim=-1).masked_fill(~valid,torch.inf)
        nearest=distance.argmin(-1);covered=visible&(distance.amin(-1)<=3)
        ids=torch.arange(len(a),device=args.device)
        flow=a+path.sample(shared['flow16_norm'][0],a)*256
        metrics={};counts=dict(n=len(a),visible=int(visible.sum()),covered3=int(covered.sum()))
        def add(name,b):
            stats,error,vis=measure(a,b,gt);metrics[name]=stats
            return vis&(error<=3)
        flowgood=add('flow',flow);add('nearest_candidate_oracle',c[ids,nearest])
        for name,logits in [('raw',raw.reshape(-1,64)),('dual',dual.reshape(-1,64))]:
            peak=logits.argmax(-1);hard=c[ids,peak]
            soft,_,_,_=path.decode(logits,c,8)
            hg=add(name+'_hard',hard);sg=add(name+'_soft',soft)
            add(name+'_oracle_peak_soft',oracle_peak_decode(logits,c,valid,nearest))
            # Rank the nearest GT candidate, and the best candidate within 3 px separately.
            rank=(logits>logits.gather(1,nearest[:,None])).sum(-1)+1
            positive=valid&(distance<=3)
            best_positive=logits.masked_fill(~positive,-torch.inf).amax(-1)
            positive_rank=(logits>best_positive[:,None]).sum(-1)+1
            for k in (1,5,10):
                counts[f'{name}_nearest_rank{k}']=int(((rank<=k)&covered).sum())
                counts[f'{name}_positive_rank{k}']=int(((positive_rank<=k)&covered).sum())
            damaged=flowgood&~sg
            counts.update({name+'_flow_good':int(flowgood.sum()),name+'_damaged':int(damaged.sum()),
                name+'_rescued':int((~flowgood&sg).sum()),name+'_damage_no_candidate':int((damaged&~covered).sum()),
                name+'_damage_wrong_peak':int((damaged&covered&~hg).sum()),
                name+'_damage_soft_only':int((damaged&covered&hg).sum()),
                name+'_hard_good_soft_bad':int((hg&~sg).sum()),name+'_hard_bad_soft_good':int((~hg&sg).sum())})
        rows.append(dict(pair_id=batch['pair_id'][0],metrics=metrics,counts=counts))
        (output/(args.domain+'.json')).write_text(json.dumps(dict(summary=summary(rows),pairs=rows)))
        print(args.domain,len(rows),flush=True)
    print(json.dumps(summary(rows),indent=2),flush=True)

if __name__=='__main__':main()
